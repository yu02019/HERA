"""HERA backbone, with shared attention and routed FFNs.


"""
from dataclasses import dataclass
import math
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass
class HERAConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    vocab_size: int = 64010
    visual_vocab_size: int = 8192
    max_text_length: int = 256
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    intermediate_size: int = 4096
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02
    gradient_checkpointing: bool = False
    rescale_residual: bool = True

    def __post_init__(self):
        integer_fields = ("image_size", "patch_size", "in_channels", "vocab_size",
                          "visual_vocab_size", "max_text_length", "hidden_size", "depth",
                          "num_heads", "intermediate_size")
        if any(not isinstance(getattr(self, field), int) or isinstance(getattr(self, field), bool)
               or getattr(self, field) <= 0 for field in integer_fields):
            raise ValueError("model sizes must be positive integers")
        if self.image_size <= 0 or self.patch_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("image_size must be a positive multiple of patch_size")
        if self.hidden_size <= 0 or self.num_heads <= 0 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if min(self.depth, self.intermediate_size, self.vocab_size, self.visual_vocab_size, self.max_text_length) <= 0:
            raise ValueError("depth, widths, vocabulary sizes and max_text_length must be positive")
        for field in ("dropout", "attention_dropout", "drop_path_rate"):
            value = getattr(self, field)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError("{} must be finite and in [0, 1)".format(field))
        if not math.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be finite and positive")
        if not math.isfinite(self.init_std) or self.init_std <= 0:
            raise ValueError("init_std must be finite and positive")
        if not isinstance(self.gradient_checkpointing, bool) or not isinstance(self.rescale_residual, bool):
            raise ValueError("gradient_checkpointing and rescale_residual must be bool")


class DropPath(nn.Module):
    """Sample one residual branch mask per example, not per token."""
    def __init__(self, probability: float):
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0:
            return x
        keep = 1.0 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class MultiwayBlock(nn.Module):
    """Shared self-attention followed by per-token modality-specific FFNs."""
    def __init__(self, config: HERAConfig, drop_path: float = 0.0):
        super().__init__()
        d = config.hidden_size
        self.norm1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.attn = nn.MultiheadAttention(d, config.num_heads, dropout=config.attention_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(d, config.intermediate_size), nn.GELU(),
                          nn.Dropout(config.dropout), nn.Linear(config.intermediate_size, d),
                          nn.Dropout(config.dropout))
            for _ in range(3)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, routes: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        y = self.norm1(x)
        y = self.attn(y, y, y, key_padding_mask=padding_mask, need_weights=False)[0]
        x = x + self.drop_path(self.dropout(y))
        y = self.norm2(x)
        routed = torch.zeros_like(y)
        for index, ffn in enumerate(self.ffns):
            select = routes == index
            if select.any():
                routed[select] = ffn(y[select])
        return x + self.drop_path(routed)


class HERAModel(nn.Module):
    """Image routes 0/1 correspond to radiology/pathology; route 2 is text.

    Image encoding returns [B, 1+N, D], including an image CLS token;
    text encoding returns [B, L, D] in the input token order. The caller is
    responsible for including any desired text CLS token in input_ids.
    """
    def __init__(self, config: HERAConfig):
        super().__init__()
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.patch_embed = nn.Conv2d(config.in_channels, config.hidden_size,
                                     kernel_size=config.patch_size, stride=config.patch_size)
        self.image_cls = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.image_mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.image_pos = nn.Parameter(torch.zeros(1, 1 + self.num_patches, config.hidden_size))
        self.text_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_pos = nn.Parameter(torch.zeros(1, config.max_text_length, config.hidden_size))
        self.blocks = nn.ModuleList([
            MultiwayBlock(config, config.drop_path_rate * i / max(config.depth - 1, 1))
            for i in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_head = nn.Linear(config.hidden_size, config.visual_vocab_size)
        self.text_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.apply(self._init_weights)
        for param in (self.image_cls, self.image_mask_token, self.image_pos, self.text_pos):
            nn.init.trunc_normal_(param, std=config.init_std)
        # Depth-dependent residual projection scaling, following the ViT/BEiT
        # initialization pattern exposed by MUSK. This is configurable because
        # it changes training from scratch and is not checkpoint conversion.
        if config.rescale_residual:
            with torch.no_grad():
                for index, block in enumerate(self.blocks, 1):
                    scale = math.sqrt(2.0 * index)
                    block.attn.out_proj.weight.div_(scale)
                    for ffn in block.ffns:
                        ffn[3].weight.div_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=self.config.init_std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            # in_proj_weight is a direct Parameter, rather than an nn.Linear.
            nn.init.trunc_normal_(module.in_proj_weight, std=self.config.init_std)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _image_inputs(self, images: Tensor, modality: Union[str, Tensor],
                      image_mask: Optional[Tensor] = None):
        if images.ndim != 4 or tuple(images.shape[1:]) != (self.config.in_channels, self.config.image_size, self.config.image_size):
            raise ValueError("images must be [B, in_channels, image_size, image_size]")
        if images.shape[0] == 0 or not images.is_floating_point():
            raise ValueError("images must be nonempty floating point tensors")
        b = images.shape[0]
        if isinstance(modality, str):
            if modality not in ("radiology", "pathology"):
                raise ValueError("modality must be radiology or pathology")
            ids = torch.full((b,), 0 if modality == "radiology" else 1, device=images.device, dtype=torch.long)
        elif isinstance(modality, Tensor):
            if modality.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("modality tensor must contain integer route IDs")
            ids = modality.to(device=images.device, dtype=torch.long)
            if ids.shape != (b,) or not torch.all((ids == 0) | (ids == 1)):
                raise ValueError("modality tensor must be [B] with 0=radiology or 1=pathology")
        else:
            raise ValueError("modality must be a string or integer tensor")
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        if image_mask is not None:
            if image_mask.shape != (b, self.num_patches) or image_mask.dtype != torch.bool or image_mask.device != images.device:
                raise ValueError("image_mask must be a boolean [B, num_patches] tensor")
            x = torch.where(image_mask.unsqueeze(-1), self.image_mask_token.expand(b, self.num_patches, -1), x)
        x = torch.cat((self.image_cls.expand(b, -1, -1), x), dim=1) + self.image_pos
        return x, ids[:, None].expand(-1, x.shape[1])

    def _text_inputs(self, input_ids: Tensor, attention_mask: Tensor):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both be [B, L]")
        if input_ids.dtype != torch.long or input_ids.shape[0] == 0 or input_ids.device != attention_mask.device:
            raise ValueError("input_ids must be a nonempty long tensor on the mask device")
        if input_ids.shape[1] == 0 or input_ids.shape[1] > self.config.max_text_length:
            raise ValueError("text length must be between 1 and max_text_length")
        if torch.any((attention_mask != 0) & (attention_mask != 1)):
            raise ValueError("attention_mask values must be 0 or 1")
        if torch.any((input_ids < 0) | (input_ids >= self.config.vocab_size)):
            raise ValueError("text token ID outside the vocabulary")
        valid = attention_mask.bool()
        if not torch.all(valid.any(dim=1)):
            raise ValueError("every text example must contain a valid token")
        x = self.text_embed(input_ids) + self.text_pos[:, :input_ids.shape[1]]
        routes = torch.full(input_ids.shape, 2, dtype=torch.long, device=input_ids.device)
        return x, routes, ~valid

    def _run(self, x: Tensor, routes: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, routes, padding_mask, use_reentrant=False)
            else:
                x = block(x, routes, padding_mask)
        return self.norm(x)

    def encode_image(self, images: Tensor, modality: Union[str, Tensor], image_mask: Optional[Tensor] = None) -> Tensor:
        x, routes = self._image_inputs(images, modality, image_mask)
        return self._run(x, routes)

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        x, routes, padding = self._text_inputs(input_ids, attention_mask)
        return self._run(x, routes, padding)

    def forward_joint(self, images: Tensor, input_ids: Tensor, attention_mask: Tensor,
                      modality: Union[str, Tensor], image_mask: Optional[Tensor] = None):
        """Joint self-attention across image and text, returning separate tokens."""
        image, image_routes = self._image_inputs(images, modality, image_mask)
        text, text_routes, text_padding = self._text_inputs(input_ids, attention_mask)
        if image.shape[0] != text.shape[0]:
            raise ValueError("image and text batch sizes must match")
        if image.device != text.device:
            raise ValueError("image and text must be on the same device")
        image_padding = torch.zeros(image.shape[:2], dtype=torch.bool, device=image.device)
        joined = self._run(torch.cat((image, text), dim=1), torch.cat((image_routes, text_routes), dim=1),
                           torch.cat((image_padding, text_padding), dim=1))
        return joined[:, :image.shape[1]], joined[:, image.shape[1]:]
