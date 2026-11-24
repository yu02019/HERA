# HERA Model API


| File | Purpose |
| --- | --- |
| `factory.py` | Creates a stage 1 or stage 2 model from the project configuration; loads stage 2 inference checkpoints. |
| `transform.py` | Creates an RGB conversion, fixed-size resize and normalization transform consistent with the training data pipeline. |
| `tokenizer.py` | Creates a SentencePiece tokenizer or a synthetic tokenizer for testing from the configuration. |
| `loss.py` | Exports the stage 1 masking loss, stage 2 contrastive loss and teacher distillation loss. |

From the project root, use an existing configuration to create the model and input processors:

```python
import yaml
from training.hera_model import create_model, create_image_transform, create_tokenizer

with open("configs/stage1.yaml", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

model = create_model(config)
tokenizer = create_tokenizer(config["tokenizer"])
data = config["data"]
transform = create_image_transform(
    model.config.image_size,
    mean=data.get("mean", [0.5, 0.5, 0.5]),
    std=data.get("std", [0.5, 0.5, 0.5]),
)
```

For stage 1, `create_model` returns a `HERAModel`. For stage 2, provide the domains present in the data, for example `create_model(config, domains=("radiology", "pathology"))`; the function returns a `Stage2Model`. When stage 2 distillation is enabled, configure teachers for each domain in `distillation.teachers`. Set `distillation.weight: 0` to train without teachers. The factory creates a newly initialized model; it does not load a stage 1 checkpoint, resume training or run online teachers. The training entry point handles those operations.

`transform` accepts a PIL image and returns a normalized `[3, H, W]` tensor. `tokenizer.encode(text)` returns token IDs, an attention mask and a special-token mask; the `synthetic` tokenizer is for testing only. Fixed-size image resizing changes the aspect ratio, so cached targets and inference preprocessing must match training.

If you have a stage 2 checkpoint from this project, use `load_stage2_encoder(checkpoint_path, device="cpu")` to load the inference model; see the feature-extraction instructions in the project root README. To train, continue to use `python -m hera --config configs/stage1.yaml` or the stage 2 configuration.
