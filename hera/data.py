
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


class SentencePieceTokenizer:
    """Explicit SentencePiece adapter. CLS, PAD and MASK must exist in its model.

    The summary token is placed at position zero. Unknown, control and unused
    pieces are excluded from masking, as are the explicitly configured IDs.
    User-defined pieces are not inherently control tokens; list any that should
    be excluded with extra_special_ids. No vocabulary entries are added.
    """
    def __init__(self, model_path, max_length, cls_id, pad_id, mask_id, eos_id=None,
                 extra_special_ids=()):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError("Install sentencepiece to use SentencePieceTokenizer") from exc
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))
        self.vocab_size = self.processor.GetPieceSize()
        if not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.max_length = max_length
        self.cls_id, self.pad_id, self.mask_id, self.eos_id = cls_id, pad_id, mask_id, eos_id
        values = [cls_id, pad_id, mask_id] + ([] if eos_id is None else [eos_id])
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 or x >= self.vocab_size for x in values):
            raise ValueError("special IDs must be integer IDs in the SentencePiece vocabulary")
        if len(set(values)) != len(values):
            raise ValueError("CLS, PAD, MASK and EOS IDs must be distinct")
        extra_special_ids = tuple(extra_special_ids)
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 or x >= self.vocab_size
               for x in extra_special_ids):
            raise ValueError("extra_special_ids must contain valid integer IDs")
        self.special_ids = set(values) | set(extra_special_ids)
        self.special_ids.update(i for i in range(self.vocab_size)
                                if self.processor.IsUnknown(i) or self.processor.IsControl(i)
                                or self.processor.IsUnused(i))

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        remaining = self.max_length - 1 - (self.eos_id is not None)
        pieces = self.processor.encode(text, out_type=int)[:remaining]
        ids = [self.cls_id] + pieces + ([] if self.eos_id is None else [self.eos_id])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool),
            "special_tokens_mask": torch.tensor([x in self.special_ids for x in ids], dtype=torch.bool),
        }


class SyntheticTokenizer:
    """Deterministic byte tokenizer for demos/tests only, not medical training."""
    vocab_size = 259
    pad_id = 0
    cls_id = 1
    mask_id = 2

    def __init__(self, max_length=64):
        if not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.max_length = max_length

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        ids = [self.cls_id] + [value + 3 for value in text.encode("utf-8")[:self.max_length - 1]]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.bool),
                "special_tokens_mask": torch.tensor([True] + [False] * (len(ids) - 1), dtype=torch.bool)}


def _resolve(base, name, field):
    if not isinstance(name, str) or not name:
        raise ValueError("{} must be a nonempty path".format(field))
    path = Path(name)
    return path if path.is_absolute() else base / path


def _load_array(path):
    # No pickle support: cache files contain numeric arrays only.
    value = np.load(str(path), allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ValueError("expected a single .npy array at {}".format(path))
    return value


class ImageTextDataset(Dataset):
    """Load one image-text pair per JSONL line.

    Required fields: image (path), text (str), modality (radiology/pathology).
    Optional: visual_tokens (.npy path), teacher_features ({name: .npy path}),
    positive_id (int), group_id (int). Relative paths are relative to the JSONL.
    Offline caches MUST have been made from the same resize geometry and image.
    An optional visual_tokenizer callable receives the resized RGB PIL image and
    returns a 1-D integer array on the student patch grid. It cannot be combined
    with precomputed visual_tokens in a sample.
    """
    def __init__(self, manifest_path, tokenizer, image_size=224,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 require_visual_tokens=False, require_teacher_features=False,
                 visual_tokenizer=None):
        self.manifest_path = Path(manifest_path)
        self.base = self.manifest_path.resolve().parent
        self.tokenizer = tokenizer
        if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "pad_id"):
            raise TypeError("tokenizer must provide encode(text) and pad_id")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        self.image_size = image_size
        if len(mean) != 3 or len(std) != 3 or any(not np.isfinite(x) for x in (*mean, *std)) or any(x <= 0 for x in std):
            raise ValueError("mean and std must contain three finite values, with std > 0")
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.visual_tokenizer = visual_tokenizer
        self.records = []
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("row must be an object")
                    _resolve(self.base, row.get("image"), "image")
                    if not isinstance(row.get("text"), str) or not row["text"].strip():
                        raise ValueError("text must be nonempty")
                    if row.get("modality") not in ("radiology", "pathology"):
                        raise ValueError("modality must be radiology or pathology")
                    if "visual_tokens" in row:
                        _resolve(self.base, row["visual_tokens"], "visual_tokens")
                        if visual_tokenizer is not None:
                            raise ValueError("choose cached visual tokens OR visual_tokenizer")
                    if require_visual_tokens and "visual_tokens" not in row and visual_tokenizer is None:
                        raise ValueError("visual tokens are required")
                    features = row.get("teacher_features")
                    if features is not None:
                        if not isinstance(features, dict) or not features:
                            raise ValueError("teacher_features must be a nonempty object")
                        for name, path in features.items():
                            if not isinstance(name, str) or not name:
                                raise ValueError("teacher name must be nonempty")
                            _resolve(self.base, path, "teacher_features")
                    if require_teacher_features and not features:
                        raise ValueError("teacher features are required")
                    for key in ("positive_id", "group_id"):
                        if key in row and (not isinstance(row[key], int) or isinstance(row[key], bool)):
                            raise ValueError("{} must be an integer".format(key))
                    self.records.append(row)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError("{}:{}: {}".format(self.manifest_path, line_no, exc)) from exc
        if not self.records:
            raise ValueError("manifest contains no samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image_path = _resolve(self.base, row["image"], "image")
        # PIL's BILINEAR resize is the explicit, reproducible view contract.
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
        pixels = torch.from_numpy(np.array(rgb, dtype=np.float32, copy=True)).permute(2, 0, 1) / 255.0
        sample = {"images": (pixels - self.mean) / self.std, "teacher_images": pixels,
                  "modality": row["modality"]}
        tokens = self.tokenizer.encode(row["text"])
        if not isinstance(tokens, Mapping) or not {"input_ids", "attention_mask", "special_tokens_mask"}.issubset(tokens):
            raise ValueError("tokenizer.encode must return input_ids, attention_mask, special_tokens_mask")
        sample.update(tokens)
        if "visual_tokens" in row:
            visual = _load_array(_resolve(self.base, row["visual_tokens"], "visual_tokens"))
        elif self.visual_tokenizer is not None:
            visual = np.asarray(self.visual_tokenizer(rgb))
        else:
            visual = None
        if visual is not None:
            if visual.ndim != 1 or not np.issubdtype(visual.dtype, np.integer):
                raise ValueError("visual tokens must be a 1-D integer array")
            sample["visual_token_targets"] = torch.as_tensor(visual.astype(np.int64, copy=True))
        if "teacher_features" in row:
            sample["teacher_features"] = {}
            for name, path in row["teacher_features"].items():
                feature = _load_array(_resolve(self.base, path, "teacher_features"))
                if feature.ndim != 1 or not np.issubdtype(feature.dtype, np.floating) or not np.isfinite(feature).all():
                    raise ValueError("teacher {} must be a finite 1-D floating array".format(name))
                sample["teacher_features"][name] = torch.as_tensor(feature.astype(np.float32, copy=True))
        for key in ("positive_id", "group_id"):
            if key in row:
                sample[key] = row[key]
        return sample


class ImageTextCollator:
    """Pad token sequences; collate only one domain and consistent cache keys."""
    def __init__(self, pad_id):
        if not isinstance(pad_id, int) or pad_id < 0:
            raise ValueError("pad_id must be a nonnegative integer")
        self.pad_id = pad_id

    def __call__(self, samples):
        if not samples or len({s["modality"] for s in samples}) != 1:
            raise ValueError("a batch must contain exactly one domain")
        result = {"images": torch.stack([s["images"] for s in samples]),
                  "teacher_images": torch.stack([s["teacher_images"] for s in samples]),
                  "modality": samples[0]["modality"]}
        lengths = []
        for sample in samples:
            ids, valid, special = (sample[k] for k in ("input_ids", "attention_mask", "special_tokens_mask"))
            if not all(isinstance(x, torch.Tensor) and x.ndim == 1 for x in (ids, valid, special)) or ids.shape != valid.shape or ids.shape != special.shape or ids.numel() == 0:
                raise ValueError("tokenizer outputs must be nonempty 1-D tensors of equal length")
            if ids.dtype != torch.long:
                raise ValueError("input_ids must have dtype torch.long")
            lengths.append(ids.numel())
        n, length = len(samples), max(lengths)
        ids = torch.full((n, length), self.pad_id, dtype=torch.long)
        attention = torch.zeros((n, length), dtype=torch.bool)
        special = torch.ones((n, length), dtype=torch.bool)
        for i, sample in enumerate(samples):
            size = lengths[i]
            ids[i, :size] = sample["input_ids"]
            attention[i, :size] = sample["attention_mask"].bool()
            special[i, :size] = sample["special_tokens_mask"].bool()
        result.update(input_ids=ids, attention_mask=attention, special_tokens_mask=special)
        for key in ("visual_token_targets", "teacher_features", "positive_id", "group_id"):
            present = [key in sample for sample in samples]
            if any(present) and not all(present):
                raise ValueError("{} must be provided for all samples in a batch or none".format(key))
            if not all(present):
                continue
            if key == "visual_token_targets":
                result[key] = torch.stack([s[key] for s in samples])
            elif key == "teacher_features":
                keys = set(samples[0][key])
                if any(set(s[key]) != keys for s in samples):
                    raise ValueError("teacher feature names must agree within a batch")
                result[key] = {name: torch.stack([s[key][name] for s in samples]) for name in sorted(keys)}
            else:
                values = torch.tensor([s[key] for s in samples], dtype=torch.long)
                prefix = "positive" if key == "positive_id" else "group"
                result["image_{}_ids".format(prefix)] = values
                result["text_{}_ids".format(prefix)] = values.clone()
        return result


def make_domain_loaders(dataset, batch_size, num_workers=0, shuffle=True, drop_last=True, seed=0,
                        rank=0, world_size=1):
    """Return domain loaders; call loader.sampler.set_epoch(cycle) before iterating.

    Each rank has an equal number of samples. With drop_last=False the distributed
    sampler may duplicate samples to make partitions evenly divisible; with True
    it drops the remainder. Caller cycles the domains for balanced sampling.
    """
    if not isinstance(batch_size, int) or batch_size <= 0 or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if not isinstance(world_size, int) or world_size <= 0 or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size), world_size positive")
    loaders = {}
    for domain_index, domain in enumerate(("radiology", "pathology")):
        indices = [i for i, row in enumerate(dataset.records) if row["modality"] == domain]
        if not indices:
            continue
        subset = Subset(dataset, indices)
        sampler = DistributedSampler(subset, num_replicas=world_size, rank=rank, shuffle=shuffle,
                                     seed=seed + domain_index, drop_last=drop_last)
        if sampler.num_samples == 0 or (drop_last and sampler.num_samples < batch_size):
            raise ValueError("{} has fewer per-rank samples than required for a batch".format(domain))
        generator = torch.Generator().manual_seed(seed + domain_index + rank * 1009)
        loaders[domain] = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                                     drop_last=drop_last, num_workers=num_workers,
                                     collate_fn=ImageTextCollator(dataset.tokenizer.pad_id), generator=generator)
    return loaders
