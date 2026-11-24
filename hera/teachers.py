
import importlib
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class TeacherAdapter(nn.Module):
    """Wrap a vision encoder with preprocessing and a summary-token selector.

    ``method`` names an encoder method (e.g. ``forward_features``); ``None``
    calls the module. ``output_key`` selects one dict key or tuple index when
    necessary. The selected tensor is [B,D] for identity, or [B,N,D] for cls
    and mean. CLS takes token 0; mean averages all tokens. Models requiring a
    different convention should implement it in their own factory wrapper.
    """
    def __init__(self, model, image_size, mean, std, output="identity",
                 output_key=None, method=None, interpolation="bicubic"):
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("teacher factory must return torch.nn.Module")
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
                or any(not isinstance(x, int) or x <= 0 for x in image_size)):
            raise ValueError("image_size must be a positive integer or [height, width]")
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must have three channel values")
        if not all(math.isfinite(float(x)) for x in list(mean) + list(std)) or any(float(x) <= 0 for x in std):
            raise ValueError("mean/std must be finite and std positive")
        if output not in ("identity", "cls", "mean"):
            raise ValueError("output must be identity, cls, or mean")
        if interpolation not in ("bilinear", "bicubic"):
            raise ValueError("interpolation must be bilinear or bicubic")
        if method is not None and (not isinstance(method, str) or not callable(getattr(model, method, None))):
            raise ValueError("method must name a callable teacher method")
        self.model = model.requires_grad_(False).eval()
        self.image_size = tuple(image_size)
        self.output = output
        self.output_key = output_key
        self.method = method
        self.interpolation = interpolation
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        super().train(False)

    def train(self, mode=True):
        # Parent calls to train() must not re-enable teacher dropout or BN.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("teacher images must be RGB [B,3,H,W]")
        if not images.is_floating_point() or images.shape[0] == 0:
            raise ValueError("teacher images must be nonempty floating-point tensors")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("teacher images must be finite and in [0,1]")
        self.model.eval()
        x = F.interpolate(images.float(), size=self.image_size, mode=self.interpolation,
                          align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        result = self.model(x) if self.method is None else getattr(self.model, self.method)(x)
        if self.output_key is not None:
            try:
                result = result[self.output_key]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("teacher output_key does not select an output") from exc
        if not isinstance(result, torch.Tensor):
            raise TypeError("teacher must return a tensor after output selection")
        if self.output == "cls":
            if result.ndim != 3 or result.shape[1] == 0:
                raise ValueError("cls expects teacher output [B,N,D] with N > 0")
            result = result[:, 0]
        elif self.output == "mean":
            if result.ndim != 3 or result.shape[1] == 0:
                raise ValueError("mean expects teacher output [B,N,D] with N > 0")
            result = result.mean(dim=1)
        if result.ndim != 2 or result.shape[0] != images.shape[0] or not result.is_floating_point():
            raise ValueError("teacher summary must be a floating [B,D] tensor")
        if result.shape[1] == 0 or not torch.isfinite(result).all():
            raise ValueError("teacher features must be nonempty and finite")
        return result.detach()


class TeacherPool(nn.Module):
    """Domain -> name -> frozen TeacherAdapter; images are shared pre-normalization views."""
    def __init__(self, teachers):
        super().__init__()
        if not isinstance(teachers, Mapping) or not teachers:
            raise ValueError("teachers must be a nonempty domain mapping")
        if not set(teachers).issubset({"radiology", "pathology"}):
            raise ValueError("teacher domain must be radiology or pathology")
        for domain, models in teachers.items():
            if not isinstance(models, Mapping) or not models:
                raise ValueError("each domain must contain at least one teacher")
            if any(not isinstance(name, str) or not name or "." in name for name in models):
                raise ValueError("teacher names must be nonempty strings without periods")
            if any(not isinstance(model, TeacherAdapter) for model in models.values()):
                raise TypeError("teacher pool entries must be TeacherAdapter instances")
        self.teachers = nn.ModuleDict({domain: nn.ModuleDict(models) for domain, models in teachers.items()})
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def extract(self, domain, images):
        if domain not in self.teachers:
            raise ValueError("no teachers configured for domain {}".format(domain))
        return {name: teacher(images) for name, teacher in self.teachers[domain].items()}

    def forward(self, domain, images):
        return self.extract(domain, images)


def build_teacher_pool(config, device="cpu"):
    """Create a pool from trusted factory specs; return None for an empty config.

    Config maps domain -> teacher -> spec. A spec has a trusted ``factory``
    import path ``package.module:callable``, optional ``kwargs`` passed to the
    factory, and TeacherAdapter arguments: image_size, mean, std, output,
    output_key, method, interpolation. Checkpoint loading belongs in the factory.
    Executing a factory runs local Python code; never use untrusted configs.
    """
    if not config:
        return None
    teachers = {}
    for domain, specs in config.items():
        teachers[domain] = {}
        for name, settings in specs.items():
            options = dict(settings)
            path = options.pop("factory")
            kwargs = options.pop("kwargs", {})
            if not isinstance(path, str) or ":" not in path:
                raise ValueError("teacher factory must use module:callable syntax")
            module_name, factory_name = path.split(":", 1)
            factory = getattr(importlib.import_module(module_name), factory_name)
            model = factory(**kwargs)
            teachers[domain][name] = TeacherAdapter(model, **options)
    return TeacherPool(teachers).to(device)
