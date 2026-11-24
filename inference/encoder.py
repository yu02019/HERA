"""Load a trained stage-2 checkpoint for image/text embedding extraction."""
import torch
from torch import nn
from torch.nn import functional as F

from hera.model import HERAConfig, HERAModel


class HERAInference(nn.Module):
    """Provides backbone representations and normalized contrastive embeddings."""
    def __init__(self, config, projection_dim):
        super().__init__()
        self.model = HERAModel(config)
        self.image_projection = nn.Linear(config.hidden_size, projection_dim, bias=False)
        self.text_projection = nn.Linear(config.hidden_size, projection_dim, bias=False)

    @torch.no_grad()
    def encode_image(self, images, modality, projected=True):
        feature = self.model.encode_image(images, modality)[:, 0]
        return F.normalize(self.image_projection(feature), dim=-1) if projected else feature

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask, projected=True):
        if attention_mask.ndim != 2 or not attention_mask[:, 0].bool().all():
            raise ValueError("text position zero must be a valid summary token")
        feature = self.model.encode_text(input_ids, attention_mask)[:, 0]
        return F.normalize(self.text_projection(feature), dim=-1) if projected else feature


def load_stage2_encoder(checkpoint_path, device="cpu"):
    """Load only the backbone and contrastive projection heads from a runner checkpoint.

    The caller must apply the exact training tokenizer and image preprocessing.
    This loader requires a trusted checkpoint created by hera.runner.train.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("stage") != 2:
        raise ValueError("a stage 2 checkpoint is required")
    model_config = HERAConfig(**config.get("model", {}))
    projection_dim = config.get("stage2", {}).get("projection_dim") or model_config.hidden_size
    encoder = HERAInference(model_config, projection_dim)
    state = checkpoint["model"]
    for prefix, module in (("model.", encoder.model), ("image_projection.", encoder.image_projection),
                           ("text_projection.", encoder.text_projection)):
        module.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                if key.startswith(prefix)}, strict=True)
    return encoder.to(device).eval()
