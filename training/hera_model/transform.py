"""Student image preprocessing consistent with the training dataset."""

import numpy as np
from PIL import Image
import torch


class ImageTransform:
    """Convert a PIL image into the normalized CHW student input.

    The callable is defined at module scope so it can be serialized for
    multi-process data loading. The deterministic square resize matches
    ``hera.data.ImageTextDataset`` and does not preserve aspect ratio.
    """

    def __init__(self, image_size, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        try:
            if len(mean) != 3 or len(std) != 3 or any(not np.isfinite(x) for x in (*mean, *std)) or any(x <= 0 for x in std):
                raise ValueError("mean and std must contain three finite values, with std > 0")
        except TypeError as exc:
            raise ValueError("mean and std must contain three finite values, with std > 0") from exc
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")
        resampling = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        rgb = image.convert("RGB").resize((self.image_size, self.image_size), resampling)
        pixels = torch.from_numpy(np.array(rgb, dtype=np.float32, copy=True)).permute(2, 0, 1) / 255.0
        return (pixels - self.mean) / self.std


def create_image_transform(image_size, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Build the deterministic PIL-to-tensor student preprocessing callable."""
    return ImageTransform(image_size, mean, std)
