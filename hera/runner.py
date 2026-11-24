"""Training loop, distributed synchronization and resumable checkpoints."""
import contextlib
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .masking import stage1_loss
from .training import Stage2Model, make_optimizer


def distributed_context(device_name="auto"):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device must be auto, cuda or cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    return device, rank, world


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


class Objective(nn.Module):
    """One DDP forward contains all domain forwards, including routed FFNs."""
    def __init__(self, network, stage, mask_id=None, stage1_options=None):
        super().__init__()
        self.network = network
        self.stage = stage
        self.mask_id = mask_id
        self.stage1_options = stage1_options or {}

    def forward(self, batches):
        outputs = []
        for batch in batches:
            if self.stage == 1:
                out = stage1_loss(self.network, batch["images"], batch["input_ids"],
                                  batch["attention_mask"], batch["special_tokens_mask"],
                                  batch["visual_token_targets"], batch["modality"],
                                  self.mask_id, **self.stage1_options)
                im_weight = self.stage1_options.get("image_loss_weight", 1.0)
                text_weight = self.stage1_options.get("text_loss_weight", 1.0)
                if not ((im_weight > 0 and out["image_mask"].any()) or
                        (text_weight > 0 and out["text_mask"].any())):
                    raise ValueError("stage 1 batch has no enabled masked targets")
            else:
                out = self.network(batch)
            outputs.append(out)
        keys = ("loss", "image_loss", "text_loss") if self.stage == 1 else (
            "loss", "contrastive_loss", "distillation_loss")
        return {key: torch.stack([out[key] for out in outputs]).mean() for key in keys}


class CyclingDomains:
    """Deterministic, per-domain distributed sampler with resumable cursors."""
    def __init__(self, loaders, cursors=None):
        self.loaders = loaders
        self.cursors = {domain: {"cycle": 0, "offset": 0} for domain in loaders}
        if cursors is not None:
            if set(cursors) != set(loaders):
                raise ValueError("checkpoint domains differ from current data")
            self.cursors = {domain: dict(cursor) for domain, cursor in cursors.items()}
        self.iterators = {}
        for domain, loader in self.loaders.items():
            if len(loader) == 0:
                raise ValueError("domain {} has no full batch".format(domain))
            cursor = self.cursors[domain]
            if not (0 <= cursor["offset"] <= len(loader) and cursor["cycle"] >= 0):
                raise ValueError("invalid sampler cursor")
            loader.sampler.set_epoch(cursor["cycle"])
            iterator = iter(loader)
            for _ in range(cursor["offset"]):
                next(iterator)
            self.iterators[domain] = iterator

    def next(self):
        batches = []
        for domain, loader in self.loaders.items():
            cursor = self.cursors[domain]
            if cursor["offset"] == len(loader):
                cursor["cycle"] += 1
                cursor["offset"] = 0
                loader.sampler.set_epoch(cursor["cycle"])
                self.iterators[domain] = iter(loader)
            batches.append(next(self.iterators[domain]))
            cursor["offset"] += 1
        return batches


def _rng_state():
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save(path, network, optimizer, scaler, step, cursors, config, rank, world):
    state = _rng_state()
    if world > 1:
        states = [None] * world
        dist.all_gather_object(states, state)
    else:
        states = [state]
    if rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save({"model": network.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "cursors": cursors,
                    "config": config, "rng_states": states, "world_size": world}, temporary)
        os.replace(str(temporary), str(path))
    if world > 1:
        dist.barrier()


def _schedule(optimizer, step, total, warmup, lr, min_lr, wd_start, wd_end):
    if step < warmup:
        current_lr = lr * (step + 1) / warmup
    else:
        progress = (step - warmup) / max(1, total - warmup - 1)
        current_lr = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
    progress = step / max(1, total - 1)
    current_wd = wd_end + 0.5 * (wd_start - wd_end) * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = current_lr
        if group.get("scheduled_decay", False):
            group["weight_decay"] = current_wd
    return current_lr, current_wd


def train(network, loaders, config, device, rank=0, world=1, teacher_pool=None):
    """Run stages using full domain batches; all ranks must call in the same order.

    Checkpoints are written at optimizer-step boundaries. Data replay assumes
    deterministic dataset transforms; randomness in third-party adapters must
    be managed by the adapter itself.
    """
    stage = int(config["stage"])
    train_cfg = config["training"]
    total = int(train_cfg["steps"])
    accum = int(train_cfg.get("gradient_accumulation", 1))
    warmup = int(train_cfg.get("warmup_steps", 0))
    if total <= 0 or accum <= 0 or not 0 <= warmup < total:
        raise ValueError("steps and accumulation must be positive, warmup in [0, steps)")
    lr = float(train_cfg.get("lr", 1e-4))
    min_lr = float(train_cfg.get("min_lr", 0.0))
    wd_start, wd_end = float(train_cfg.get("weight_decay", 0.04)), float(train_cfg.get("weight_decay_end", 0.4))
    if not (0 <= min_lr <= lr and 0 <= wd_start and 0 <= wd_end):
        raise ValueError("invalid optimizer schedule")
    network.to(device)
    objective = Objective(network, stage, config.get("mask_token_id"), config.get("stage1", {})).to(device)
    optimizer = make_optimizer(network, lr=lr, weight_decay=wd_start)
    amp = bool(train_cfg.get("amp", False))
    if amp and device.type != "cuda":
        raise ValueError("AMP currently requires CUDA")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start, cursors, resume_rng = 0, None, None
    resume = train_cfg.get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["world_size"] != world:
            raise ValueError("resume requires the same world size")
        saved = checkpoint["config"]
        for key in ("stage", "model", "data", "tokenizer", "distillation", "stage1", "stage2", "mask_token_id"):
            if saved.get(key) != config.get(key):
                raise ValueError("resume config differs: {}".format(key))
        for key in ("steps", "batch_size", "gradient_accumulation", "lr", "min_lr", "warmup_steps", "weight_decay", "weight_decay_end", "grad_clip", "amp", "seed"):
            if saved["training"].get(key) != train_cfg.get(key):
                raise ValueError("resume training config differs: {}".format(key))
        network.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, cursors = checkpoint["step"], checkpoint["cursors"]
        resume_rng = checkpoint["rng_states"][rank]
    domains = CyclingDomains(loaders, cursors)
    if resume_rng is not None:
        _set_rng_state(resume_rng)
    wrapped = (DistributedDataParallel(objective, device_ids=[device.index] if device.type == "cuda" else None,
                                       find_unused_parameters=True) if world > 1 else objective)
    objective.train()
    save_every = int(train_cfg.get("save_every", total))
    log_every = int(train_cfg.get("log_every", 10))
    grad_clip = float(train_cfg.get("grad_clip", 3.0))
    if save_every <= 0 or log_every <= 0 or not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("save_every, log_every and grad_clip must be positive")
    output_dir = Path(train_cfg.get("output_dir", "runs/hera"))
    for step in range(start, total):
        current_lr, current_wd = _schedule(optimizer, step, total, warmup, lr, min_lr, wd_start, wd_end)
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for micro in range(accum):
            batches = [to_device(batch, device) for batch in domains.next()]
            if stage == 2 and teacher_pool is not None:
                for batch in batches:
                    online = (teacher_pool.extract(batch["modality"], batch["teacher_images"])
                              if batch["modality"] in teacher_pool.teachers else {})
                    cached = batch.get("teacher_features", {})
                    if set(online) & set(cached):
                        raise ValueError("teacher appears in online and cached features")
                    batch["teacher_features"] = dict(cached, **online)
            sync = wrapped.no_sync() if world > 1 and micro < accum - 1 else contextlib.nullcontext()
            with sync:
                with torch.cuda.amp.autocast(enabled=amp):
                    output = wrapped(batches)
                    loss = output["loss"] / accum
                if not torch.isfinite(loss.detach()).all():
                    raise ValueError("non-finite loss")
                scaler.scale(loss).backward()
            for key, value in output.items():
                metrics[key] = metrics.get(key, 0.0) + value.detach().float().item() / accum
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(network.parameters(), grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if isinstance(network, Stage2Model):
            with torch.no_grad():
                network.logit_scale.clamp_(max=math.log(100.0))
        if rank == 0 and ((step + 1) % log_every == 0 or step == 0):
            print("step={} lr={:.6g} wd={:.5g} {}".format(
                step + 1, current_lr, current_wd, " ".join("{}={:.5f}".format(k, v) for k, v in metrics.items())), flush=True)
        if (step + 1) % save_every == 0 or step + 1 == total:
            _save(output_dir / "last.pt", network, optimizer, scaler, step + 1, domains.cursors, config, rank, world)
            if train_cfg.get("keep_step_checkpoints", False):
                _save(output_dir / ("step-{:08d}.pt".format(step + 1)), network, optimizer,
                      scaler, step + 1, domains.cursors, config, rank, world)
    return network
