
import math
from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _feature_matrix(name, value):
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError("{} must be a tensor of shape [batch, feature_dim]".format(name))
    if not value.is_floating_point() or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("{} must be nonempty and floating point".format(name))


def _ids(name, value, length, device):
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] != length:
        raise ValueError("{} must be a one-dimensional tensor of length {}".format(name, length))
    if value.device != device:
        raise ValueError("{} must be on the same device as the features".format(name))
    if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
        raise ValueError("{} must contain integer IDs".format(name))
    return value


def symmetric_contrastive_loss(
    image_features,
    text_features,
    logit_scale,
    image_positive_ids=None,
    text_positive_ids=None,
    image_group_ids=None,
    text_group_ids=None,
    distributed=False,
):
    """Symmetric CLIP loss, with optional explicit positives and group masking.

    Features are L2-normalized internally. ``logit_scale`` multiplies cosine
    similarities directly; a CLIP log-scale parameter must be exponentiated by
    the caller. By default image i and text i are the sole positive pair.
    Supplying both positive-ID tensors makes equal IDs explicit positive pairs.
    Supplying both group-ID tensors excludes equal-group non-positive pairs from
    the denominator, as in KEEP. All supplied IDs must be valid (no missing-ID
    sentinel), and every image and text must have at least one positive.
    Multi-positive loss is the negative log probability of the positive set.
    With distributed=True, all ranks compute the same global loss using
    differentiable all-gather. This assumes DDP's default gradient averaging,
    identical batch shapes and optional-ID presence on all ranks, and identical
    call ordering. It creates a global-batch-squared logits matrix per rank.
    """
    _feature_matrix("image_features", image_features)
    _feature_matrix("text_features", text_features)
    if image_features.device != text_features.device:
        raise ValueError("image and text features must be on the same device")
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError("image and text feature dimensions must match")
    device = image_features.device
    n_image, n_text = image_features.shape[0], text_features.shape[0]
    if distributed:
        (image_features, text_features, image_positive_ids, text_positive_ids,
         image_group_ids, text_group_ids) = _gather_contrastive_inputs(
            image_features, text_features, image_positive_ids, text_positive_ids,
            image_group_ids, text_group_ids)
        n_image, n_text = image_features.shape[0], text_features.shape[0]
    if isinstance(logit_scale, torch.Tensor):
        if logit_scale.numel() != 1 or logit_scale.device != device:
            raise ValueError("logit_scale must be scalar and on the feature device")
        if not logit_scale.is_floating_point():
            raise ValueError("logit_scale must be floating point")
        scale = logit_scale.float().reshape(())
        scale_value = float(scale.detach().item())
    else:
        scale_value = float(logit_scale)
        scale = scale_value
    if not math.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("logit_scale must be finite and positive")

    if (image_positive_ids is None) != (text_positive_ids is None):
        raise ValueError("provide both image_positive_ids and text_positive_ids")
    if image_positive_ids is None:
        if n_image != n_text:
            raise ValueError("paired contrastive learning requires equal batch sizes")
        positive = torch.eye(n_image, dtype=torch.bool, device=device)
    else:
        image_positive_ids = _ids("image_positive_ids", image_positive_ids, n_image, device)
        text_positive_ids = _ids("text_positive_ids", text_positive_ids, n_text, device)
        positive = image_positive_ids[:, None] == text_positive_ids[None, :]
        if not positive.any(dim=1).all() or not positive.any(dim=0).all():
            raise ValueError("each image and text must have at least one positive")

    if (image_group_ids is None) != (text_group_ids is None):
        raise ValueError("provide both image_group_ids and text_group_ids")
    allowed = torch.ones_like(positive)
    if image_group_ids is not None:
        image_group_ids = _ids("image_group_ids", image_group_ids, n_image, device)
        text_group_ids = _ids("text_group_ids", text_group_ids, n_text, device)
        same_group = image_group_ids[:, None] == text_group_ids[None, :]
        allowed = ~same_group | positive

    images = F.normalize(image_features.float(), dim=-1)
    texts = F.normalize(text_features.float(), dim=-1)
    logits = scale * images @ texts.T
    positive_logits = logits.masked_fill(~positive, float("-inf"))
    allowed_logits = logits.masked_fill(~allowed, float("-inf"))
    image_loss = torch.logsumexp(allowed_logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    text_loss = torch.logsumexp(allowed_logits, dim=0) - torch.logsumexp(positive_logits, dim=0)
    return (image_loss.mean() + text_loss.mean()) / 2


def _gather_contrastive_inputs(images, texts, image_positive_ids, text_positive_ids,
                               image_group_ids, text_group_ids):
    """Gather features with gradients and optional integer metadata without them."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed contrastive loss requires an initialized process group")
    world = dist.get_world_size()
    if world == 1:
        return (images, texts, image_positive_ids, text_positive_ids,
                image_group_ids, text_group_ids)
    ids = (image_positive_ids, text_positive_ids, image_group_ids, text_group_ids)
    # Validate collective layout across ranks before entering further collectives.
    feature_dtype_codes = {torch.float16: 1, torch.bfloat16: 2, torch.float32: 3, torch.float64: 4}
    local = torch.tensor([images.shape[0], texts.shape[0], images.shape[1], texts.shape[1],
                          feature_dtype_codes.get(images.dtype, -1), feature_dtype_codes.get(texts.dtype, -1)]
                         + [int(value is not None) for value in ids],
                         dtype=torch.long, device=images.device)
    layouts = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(layouts, local)
    if any(not torch.equal(local, layout) for layout in layouts) or local[4] < 0 or local[5] < 0:
        raise ValueError("all ranks must have identical feature shapes and ID availability")
    if (ids[0] is None) != (ids[1] is None) or (ids[2] is None) != (ids[3] is None):
        raise ValueError("positive and group IDs must each be supplied in image/text pairs")
    # Ensure IDs have the same dtype on every rank before all_gather.
    dtype_codes = {torch.uint8: 1, torch.int8: 2, torch.int16: 3, torch.int32: 4, torch.int64: 5}
    local_codes = []
    for name, value, length in zip(("image_positive_ids", "text_positive_ids", "image_group_ids", "text_group_ids"),
                                   ids, (images.shape[0], texts.shape[0], images.shape[0], texts.shape[0])):
        if value is None:
            local_codes.append(0)
        elif (isinstance(value, torch.Tensor) and value.ndim == 1 and value.shape[0] == length
              and value.device == images.device and value.dtype in dtype_codes):
            local_codes.append(dtype_codes[value.dtype])
        else:
            local_codes.append(-1)
    codes = torch.tensor(local_codes, dtype=torch.long, device=images.device)
    rank_codes = [torch.empty_like(codes) for _ in range(world)]
    dist.all_gather(rank_codes, codes)
    if any(not torch.equal(codes, other) for other in rank_codes) or (codes < 0).any():
        raise ValueError("ID shape, device and dtype must be valid and match across ranks")
    from torch.distributed.nn.functional import all_gather as gather_with_grad
    images = torch.cat(gather_with_grad(images), dim=0)
    texts = torch.cat(gather_with_grad(texts), dim=0)
    all_ids = []
    for value in ids:
        if value is None:
            all_ids.append(None)
        else:
            parts = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(parts, value.contiguous())
            all_ids.append(torch.cat(parts, dim=0))
    return (images, texts, *all_ids)


class AMRadioSummaryDistillation(nn.Module):
    """Match frozen teacher summary vectors through separate trainable heads.

    ``teacher_dims`` maps teacher names to feature dimensions. Each head is
    Linear -> LayerNorm -> GELU -> Linear, with the largest teacher dimension
    as the hidden size. The loss is the weighted sum of cosine distances, as
    in AM-RADIO; no spatial-feature matching is implied. Targets are detached.
    Teacher outputs must be [batch, teacher_dim] and correspond to the same
    images and ordering as the student features.
    """

    def __init__(self, student_dim, teacher_dims, teacher_weights=None):
        super().__init__()
        if not isinstance(student_dim, int) or isinstance(student_dim, bool) or student_dim <= 0:
            raise ValueError("student_dim must be a positive integer")
        if not isinstance(teacher_dims, Mapping) or not teacher_dims:
            raise ValueError("teacher_dims must be a nonempty mapping")
        for name, dim in teacher_dims.items():
            if not isinstance(name, str) or not name or "." in name:
                raise ValueError("teacher names must be nonempty strings without periods")
            if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                raise ValueError("teacher dimensions must be positive integers")
        if teacher_weights is None:
            teacher_weights = {name: 1.0 for name in teacher_dims}
        if not isinstance(teacher_weights, Mapping) or set(teacher_weights) != set(teacher_dims):
            raise ValueError("teacher_weights must have exactly the teacher_dims keys")
        self.weights = {}
        for name, weight in teacher_weights.items():
            weight = float(weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("teacher weights must be finite and nonnegative")
            self.weights[name] = weight
        if not any(self.weights.values()):
            raise ValueError("at least one teacher weight must be positive")
        self.student_dim = student_dim
        self.teacher_dims = dict(teacher_dims)
        hidden_dim = max(teacher_dims.values())
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for name, dim in teacher_dims.items()
        })

    def forward(self, student_features, teacher_features):
        _feature_matrix("student_features", student_features)
        if student_features.shape[1] != self.student_dim:
            raise ValueError("student feature dimension does not match student_dim")
        if not isinstance(teacher_features, Mapping) or set(teacher_features) != set(self.teacher_dims):
            raise ValueError("teacher_features must have exactly the teacher_dims keys")
        losses = []
        for name, dim in self.teacher_dims.items():
            target = teacher_features[name]
            _feature_matrix("teacher_features[{}]".format(name), target)
            if target.shape != (student_features.shape[0], dim):
                raise ValueError("teacher {} has incompatible batch or feature dimensions".format(name))
            if target.device != student_features.device:
                raise ValueError("teacher and student features must be on the same device")
            prediction = self.heads[name](student_features)
            loss = 1.0 - F.cosine_similarity(prediction.float(), target.detach().float(), dim=-1)
            losses.append(self.weights[name] * loss.mean())
        return torch.stack(losses).sum()


def freeze_teacher(teacher):
    """Freeze a teacher module and switch it to evaluation mode."""
    if not isinstance(teacher, nn.Module):
        raise TypeError("teacher must be a torch.nn.Module")
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def frozen_teacher_forward(teacher, *args, **kwargs):
    """Run a previously frozen teacher in eval mode, without autograd."""
    if not isinstance(teacher, nn.Module):
        raise TypeError("teacher must be a torch.nn.Module")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("freeze the teacher with freeze_teacher before forwarding")
    teacher.eval()
    with torch.no_grad():
        return teacher(*args, **kwargs)
