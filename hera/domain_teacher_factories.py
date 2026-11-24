
from pathlib import Path

import torch
from torch import nn


def _checkpoint_path(checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError("teacher checkpoint is not a local file: {}".format(path))
    return str(path)


def _load_state(model, checkpoint_path):
    # Restricted loading protects against arbitrary pickle code. Checkpoints
    # from the official model releases are plain state dictionaries.
    state = torch.load(_checkpoint_path(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def uni_v1(checkpoint_path):
    """UNI ViT-L/16; emits a 1024-dimensional image embedding [B,1024].

    Requires timm and the gated UNI v1 pytorch_model.bin. Match upstream 224
    input/ImageNet normalization. This is not UNI2-h.
    """
    path = _checkpoint_path(checkpoint_path)
    try:
        import timm
    except ImportError as exc:
        raise ImportError("UNI v1 requires timm; follow the official UNI installation") from exc
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16,
                              init_values=1e-5, num_classes=0, dynamic_img_size=True)
    return _load_state(model, path)


def gpfm(checkpoint_path):
    """GPFM ViT-L/14, whose official model card specifies [B,1024] outputs."""
    path = _checkpoint_path(checkpoint_path)
    try:
        import timm
    except ImportError as exc:
        raise ImportError("GPFM requires timm; follow the official GPFM installation") from exc
    model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=False,
                              img_size=224, init_values=1e-5)
    return _load_state(model, path)


class _ConchImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.encode_image(images, proj_contrast=False, normalize=False)


def conch_v1(checkpoint_path):
    """CONCH v1 unprojected, unnormalized global image embedding.

    Requires the official CONCH package and gated local pytorch_model.bin.
    Its loader returns its official preprocess separately; check its settings
    against TeacherAdapter and the exact image view used to make student input.
    """
    path = _checkpoint_path(checkpoint_path)
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as exc:
        raise ImportError("CONCH v1 requires the official CONCH package") from exc
    model, _preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=path)
    return _ConchImageEncoder(model).eval()
