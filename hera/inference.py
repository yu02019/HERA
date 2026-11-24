"""Compatibility imports; inference implementation lives in the inference package."""

from inference.encoder import HERAInference, load_stage2_encoder

__all__ = ["HERAInference", "load_stage2_encoder"]
