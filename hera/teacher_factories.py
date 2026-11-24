
from pathlib import Path

import torch


def timm_encoder(model_name, checkpoint_path=None, pretrained=False,
                 model_kwargs=None, state_dict_key=None, strip_prefix="", strict=True):
    """Build a timm model and optionally load a local state dict.

    ``model_kwargs`` defaults to {num_classes: 0}. A nested checkpoint can be
    selected by ``state_dict_key``; ``strip_prefix`` removes a uniform prefix
    such as ``module.``. The user must choose a model implementation and
    forward/output behavior compatible with their exact checkpoint.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError("the timm teacher factory requires timm") from exc
    kwargs = {"num_classes": 0} if model_kwargs is None else dict(model_kwargs)
    model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
    if checkpoint_path is not None:
        state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        if state_dict_key is not None:
            state = state[state_dict_key]
        if not isinstance(state, dict):
            raise ValueError("checkpoint must contain a state dictionary")
        if strip_prefix:
            if not all(key.startswith(strip_prefix) for key in state):
                raise ValueError("not all checkpoint keys have strip_prefix")
            state = {key[len(strip_prefix):]: value for key, value in state.items()}
        model.load_state_dict(state, strict=strict)
    return model


def torchscript_encoder(checkpoint_path):
    """Load a trusted TorchScript teacher; outputs must match adapter settings."""
    return torch.jit.load(str(Path(checkpoint_path)), map_location="cpu")
