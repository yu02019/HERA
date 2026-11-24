"""Independent masked image modelling and text MLM for HERA stage 1."""
import math
from typing import Dict, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import HERAModel


def _choose_mask(eligible: Tensor, ratio: float) -> Tensor:
    """Sample eligible positions independently, ensuring one per nonempty row."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError("mask ratio must be in (0, 1]")
    random_values = torch.rand(eligible.shape, device=eligible.device)
    chosen = (random_values < ratio) & eligible
    missing = eligible.any(dim=1) & ~chosen.any(dim=1)
    if missing.any():
        # Select the lowest random draw among eligible positions in each empty row.
        indices = random_values.masked_fill(~eligible, float("inf")).argmin(dim=1)
        rows = missing.nonzero(as_tuple=True)[0]
        chosen[rows, indices[rows]] = True
    return chosen


def stage1_loss(model: HERAModel, images: Tensor, input_ids: Tensor, attention_mask: Tensor,
                special_tokens_mask: Tensor, visual_token_targets: Tensor,
                modality: Union[str, Tensor], mask_token_id: int,
                image_mask_ratio: float = 0.4, text_mask_ratio: float = 0.15,
                image_loss_weight: float = 1.0, text_loss_weight: float = 1.0) -> Dict[str, Tensor]:
    """Return stage-1 losses from independent image and text masked encoding.

    visual_token_targets is [B, N] of integer visual-tokenizer IDs, aligned
    with the student's post-transform patch grid; -100 denotes unavailable
    targets. special_tokens_mask is [B, L], true for every special token.
    attention_mask uses 1/True for valid tokens and 0/False for padding.
    The external tokenizer and geometric alignment are the caller's responsibility.
    No paired text is visible to image modelling, or vice versa.
    """
    b, n = images.shape[0], model.num_patches
    if visual_token_targets.shape != (b, n) or visual_token_targets.dtype != torch.long:
        raise ValueError("visual_token_targets must be long [B, num_patches]")
    if input_ids.ndim != 2 or input_ids.shape[0] != b or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be long [B, L]")
    if attention_mask.shape != input_ids.shape or special_tokens_mask.shape != input_ids.shape:
        raise ValueError("attention_mask and special_tokens_mask must match input_ids")
    if not 0 <= mask_token_id < model.config.vocab_size:
        raise ValueError("mask_token_id outside text vocabulary")
    if not all(math.isfinite(weight) and weight >= 0 for weight in (image_loss_weight, text_loss_weight)):
        raise ValueError("loss weights must be finite and nonnegative")
    if any(tensor.device != images.device for tensor in
           (input_ids, attention_mask, special_tokens_mask, visual_token_targets)):
        raise ValueError("all stage-1 inputs must be on the same device")
    if torch.any((special_tokens_mask != 0) & (special_tokens_mask != 1)):
        raise ValueError("special_tokens_mask values must be 0 or 1")
    valid_visual = visual_token_targets != -100
    if torch.any(valid_visual & ((visual_token_targets < 0) | (visual_token_targets >= model.config.visual_vocab_size))):
        raise ValueError("visual target ID outside visual vocabulary")
    image_mask = _choose_mask(valid_visual, image_mask_ratio)
    text_mask = _choose_mask(attention_mask.bool() & ~special_tokens_mask.bool(), text_mask_ratio)
    masked_ids = input_ids.masked_fill(text_mask, mask_token_id)
    image_tokens = model.encode_image(images, modality, image_mask=image_mask)
    text_tokens = model.encode_text(masked_ids, attention_mask)
    # Project only selected tokens: a [B, L, vocabulary] tensor can be enormous.
    # Empty eligible sets can occur (e.g. text containing only special tokens).
    # Keep both the representation and head in the autograd graph in that case.
    if image_mask.any():
        image_loss = F.cross_entropy(model.visual_head(image_tokens[:, 1:][image_mask]),
                                     visual_token_targets[image_mask])
    else:
        image_loss = (image_tokens.float().sum() + model.visual_head.weight.float().sum()
                      + model.visual_head.bias.float().sum()) * 0.0
    if text_mask.any():
        text_loss = F.cross_entropy(model.text_head(text_tokens[text_mask]), input_ids[text_mask])
    else:
        text_loss = (text_tokens.float().sum() + model.text_head.weight.float().sum()
                     + model.text_head.bias.float().sum()) * 0.0
    total = image_loss_weight * image_loss + text_loss_weight * text_loss
    return {"loss": total, "image_loss": image_loss, "text_loss": text_loss,
            "image_mask": image_mask, "text_mask": text_mask}
