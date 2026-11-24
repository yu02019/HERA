

import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from .masking import stage1_loss
from .objectives import symmetric_contrastive_loss


def make_optimizer(module, lr=1e-4, weight_decay=0.04, betas=(0.9, 0.999)):
    """Create AdamW over all trainable parameters, excluding bias/norm decay.

    Pass the complete Stage2Model for stage 2 so its projection and teacher
    adaptor heads, as well as logit_scale, are included in the optimizer.
    """
    if not math.isfinite(lr) or not math.isfinite(weight_decay) or lr <= 0 or weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay nonnegative")
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        leaf = name.rsplit(".", 1)[-1]
        exempt = leaf in {"image_cls", "image_mask_token", "image_pos", "text_pos", "logit_scale"}
        (no_decay if param.ndim <= 1 or name.endswith(".bias") or exempt else decay).append(param)
    if not decay and not no_decay:
        raise ValueError("module has no trainable parameters")
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay, "scheduled_decay": True})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "scheduled_decay": False})
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def set_cosine_weight_decay(optimizer, step, total_steps, start=0.04, end=0.4):
    """Update decay groups for zero-based step; endpoints include last step."""
    if total_steps < 2 or not 0 <= step < total_steps:
        raise ValueError("total_steps must be >= 2 and step in [0, total_steps)")
    if not all(math.isfinite(x) and x >= 0 for x in (start, end)):
        raise ValueError("weight decay endpoints must be finite and nonnegative")
    value = end + (start - end) * (1 + math.cos(math.pi * step / (total_steps - 1))) / 2
    for group in optimizer.param_groups:
        if group.get("scheduled_decay", False):
            group["weight_decay"] = value
    return value


def _batches(batches):
    if isinstance(batches, Mapping):
        batches = [batches]
    else:
        batches = list(batches)
    if not batches or not all(isinstance(batch, Mapping) for batch in batches):
        raise ValueError("batches must be a nonempty mapping or sequence of mappings")
    if len(batches) > 1:
        domains = [batch.get("modality") for batch in batches]
        if any(not isinstance(domain, str) or domain not in ("radiology", "pathology")
               for domain in domains) or len(set(domains)) != len(domains):
            raise ValueError("multiple batches must each have a distinct domain string")
    return batches


def _update(loss, module, optimizer, max_grad_norm):
    if not torch.isfinite(loss.detach()).all():
        raise ValueError("non-finite loss; optimizer step skipped")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        nn.utils.clip_grad_norm_(module.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()


def stage1_step(model, optimizer, batches, mask_token_id, max_grad_norm=3.0,
                image_mask_ratio=0.4, text_mask_ratio=0.15,
                image_loss_weight=1.0, text_loss_weight=1.0):
    """Run one stage-1 step and return detached scalar losses.

    Every mapping requires images, input_ids, attention_mask,
    special_tokens_mask, visual_token_targets, and modality. Images and texts
    are encoded independently and need not be paired, although each combined
    batch must contain equal numbers of images and text sequences.
    """
    model.train()
    outputs = [stage1_loss(
        model, batch["images"], batch["input_ids"], batch["attention_mask"],
        batch["special_tokens_mask"], batch["visual_token_targets"], batch["modality"],
        mask_token_id, image_mask_ratio, text_mask_ratio, image_loss_weight, text_loss_weight,
    ) for batch in _batches(batches)]
    for output in outputs:
        if not ((image_loss_weight > 0 and output["image_mask"].any()) or
                (text_loss_weight > 0 and output["text_mask"].any())):
            raise ValueError("stage-1 batch has no enabled masked targets; optimizer step skipped")
    loss = torch.stack([o["loss"] for o in outputs]).mean()
    _update(loss, model, optimizer, max_grad_norm)
    return {key: torch.stack([o[key].detach() for o in outputs]).mean()
            for key in ("loss", "image_loss", "text_loss")}


class Stage2Model(nn.Module):
    """HERA with learned contrastive projections and per-domain distillation.

    ``distillers`` maps each domain to its own AMRadioSummaryDistillation
    module. Teacher features are produced externally from the same images;
    no teacher executes inside this class. Text position 0 must be a valid
    summary/CLS token. Features and supplied IDs must be on the same device.
    """
    def __init__(self, model, distillers, projection_dim=None, initial_temperature=0.07,
                 distillation_weight=1.0, distributed_contrastive=False):
        super().__init__()
        if not isinstance(distillers, Mapping) or not distillers:
            raise ValueError("distillers must map domains to modules")
        if not set(distillers).issubset({"radiology", "pathology"}):
            raise ValueError("unknown domain in distillers")
        if any(not isinstance(x, nn.Module) for x in distillers.values()):
            raise TypeError("each distiller must be a torch module")
        if not math.isfinite(initial_temperature) or initial_temperature <= 0:
            raise ValueError("initial_temperature must be finite and positive")
        if not math.isfinite(distillation_weight) or distillation_weight < 0:
            raise ValueError("distillation_weight must be finite and nonnegative")
        width = model.config.hidden_size
        projection_dim = width if projection_dim is None else projection_dim
        if not isinstance(projection_dim, int) or projection_dim <= 0:
            raise ValueError("projection_dim must be a positive integer")
        self.model = model
        self.distillers = nn.ModuleDict(distillers)
        self.image_projection = nn.Linear(width, projection_dim, bias=False)
        self.text_projection = nn.Linear(width, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / initial_temperature)))
        self.distillation_weight = float(distillation_weight)
        self.distributed_contrastive = bool(distributed_contrastive)

    def forward(self, batch):
        domain = batch["modality"]
        if not isinstance(domain, str) or domain not in self.distillers:
            raise ValueError("each stage-2 batch must use a configured single domain")
        attention_mask = batch["attention_mask"]
        if attention_mask.ndim != 2 or attention_mask.shape[1] == 0 or not attention_mask[:, 0].bool().all():
            raise ValueError("text position 0 must be a valid summary token")
        image = self.model.encode_image(batch["images"], domain)[:, 0]
        text = self.model.encode_text(batch["input_ids"], batch["attention_mask"])[:, 0]
        contrastive = symmetric_contrastive_loss(
            self.image_projection(image), self.text_projection(text),
            self.logit_scale.clamp(max=math.log(100.0)).exp(),
            image_positive_ids=batch.get("image_positive_ids"),
            text_positive_ids=batch.get("text_positive_ids"),
            image_group_ids=batch.get("image_group_ids"),
            text_group_ids=batch.get("text_group_ids"),
            distributed=self.distributed_contrastive,
        )
        distillation = (self.distillers[domain](image, batch["teacher_features"])
                        if self.distillation_weight else image.sum() * 0.0)
        return {"loss": contrastive + self.distillation_weight * distillation,
                "contrastive_loss": contrastive, "distillation_loss": distillation}


def stage2_step(model, optimizer, batches, max_grad_norm=3.0):
    """Run a single-process step, averaging per-domain objectives equally."""
    model.train()
    outputs = [model(batch) for batch in _batches(batches)]
    loss = torch.stack([o["loss"] for o in outputs]).mean()
    _update(loss, model, optimizer, max_grad_norm)
    # Keep the learned log-scale in the differentiable region after AdamW.
    with torch.no_grad():
        model.logit_scale.clamp_(max=math.log(100.0))
    return {key: torch.stack([o[key].detach() for o in outputs]).mean()
            for key in ("loss", "contrastive_loss", "distillation_loss")}


def save_checkpoint(path, model, optimizer, step, metadata=None):
    """Save model and optimizer state with the next training step and metadata.

    Metadata should contain only tensors and built-in primitive containers so it
    can be loaded by the restricted weights-only unpickler.
    """
    if not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "metadata": metadata}, Path(path))


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """Restore a trusted local checkpoint; return next step and metadata.

    Instantiate the same model/optimizer configuration before loading. Data
    loader, RNG, and LR scheduler states are not captured by this helper.
    """
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"], checkpoint.get("metadata")
