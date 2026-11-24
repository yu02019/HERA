"""Run HERA training: python -m hera --config path/to/config.yaml."""
import argparse
import importlib
import random

import torch
import torch.distributed as dist
import yaml
from training.hera_model.factory import create_model
from training.hera_model.tokenizer import create_tokenizer

from .data import ImageTextDataset, make_domain_loaders
from .model import HERAConfig
from .runner import distributed_context, train
from .teachers import build_teacher_pool


def _factory(spec):
    module, name = spec["factory"].split(":", 1)
    return getattr(importlib.import_module(module), name)(**spec.get("kwargs", {}))


def main():
    parser = argparse.ArgumentParser(description="HERA two-stage pretraining")
    parser.add_argument("--config", required=True, help="YAML config; paths are relative to the working directory")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("stage") not in (1, 2):
        raise ValueError("config must specify stage: 1 or 2")
    device, rank, world = distributed_context(config.get("device", "auto"))
    try:
        seed = int(config["training"].get("seed", 0))
        random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + rank)
        tokenizer = create_tokenizer(config["tokenizer"])
        if config["tokenizer"].get("kind") == "synthetic" and rank == 0:
            print("WARNING: synthetic tokenizer is for tests only", flush=True)
        model_cfg = HERAConfig(**config.get("model", {}))
        if tokenizer.vocab_size != model_cfg.vocab_size:
            raise ValueError("tokenizer vocabulary size does not match model.vocab_size")
        if tokenizer.max_length > model_cfg.max_text_length:
            raise ValueError("tokenizer max_length exceeds model.max_text_length")
        config["mask_token_id"] = tokenizer.mask_id
        data_cfg = dict(config["data"])
        manifest = data_cfg.pop("manifest")
        visual_spec = data_cfg.pop("visual_tokenizer", None)
        visual_tokenizer = _factory(visual_spec) if visual_spec else None
        dataset = ImageTextDataset(manifest, tokenizer, image_size=model_cfg.image_size,
                                   require_visual_tokens=config["stage"] == 1,
                                   visual_tokenizer=visual_tokenizer, **data_cfg)
        training = config["training"]
        loaders = make_domain_loaders(dataset, int(training["batch_size"]),
                                      num_workers=int(training.get("num_workers", 0)),
                                      shuffle=True, drop_last=True, seed=seed, rank=rank, world_size=world)
        network = create_model(config, domains=tuple(loaders), distributed_contrastive=world > 1)
        init = config.get("initialize_from_stage1")
        if init:
            if config["stage"] != 2 or training.get("resume"):
                raise ValueError("initialize_from_stage1 is for a fresh stage 2 only")
            checkpoint = torch.load(init, map_location="cpu", weights_only=True)
            if checkpoint.get("config", {}).get("stage") != 1 or checkpoint["config"].get("model", {}) != config.get("model", {}):
                raise ValueError("stage 1 checkpoint model configuration does not match")
            network.model.load_state_dict(checkpoint["model"], strict=True)
        teacher_pool = None
        if config["stage"] == 2:
            distill = config.get("distillation", {})
            weight = float(distill.get("weight", 1.0))
            specs = distill.get("teachers", {})
            online = {}
            for domain in loaders:
                domain_specs = specs.get(domain, {})
                if weight:
                    online_domain = {name: options["online"] for name, options in domain_specs.items() if "online" in options}
                    if online_domain:
                        online[domain] = online_domain
            teacher_pool = build_teacher_pool(online, device=device)
        train(network, loaders, config, device, rank, world, teacher_pool)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
