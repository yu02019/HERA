

from collections.abc import Mapping

from torch import nn

from hera.model import HERAConfig, HERAModel
from hera.objectives import AMRadioSummaryDistillation
from hera.training import Stage2Model
from inference.encoder import HERAInference, load_stage2_encoder


def create_model(config, domains=None, distributed_contrastive=False):
    """Build a stage-1 backbone or a stage-2 training model.

    Args:
        config: Full configuration mapping, including ``stage`` and optionally
            ``model``, ``stage2`` and ``distillation``. It uses the same keys
            and defaults as the training CLI.
        domains: For stage 2, the nonempty iterable of domains present in the
            data (``radiology`` and/or ``pathology``). Each domain requires a
            teacher specification when distillation has nonzero weight.
        distributed_contrastive: Enable cross-rank contrastive gathering in
            stage 2. A distributed process group must already be initialized
            when the resulting model is used.

    Returns:
        A ``HERAModel`` for stage 1 or ``Stage2Model`` for stage 2. Models are
        newly initialized on CPU; move them to the intended device as needed.
        For trained inference checkpoints use ``load_stage2_encoder`` instead.

    The caller is responsible for applying ``initialize_from_stage1``, resume
    state, and building/running any online teachers. A model built here does not
    load a checkpoint or acquire pretrained weights automatically.
    """
    if not isinstance(config, Mapping) or config.get("stage") not in (1, 2):
        raise ValueError("config must be a mapping with stage: 1 or 2")
    model_config = HERAConfig(**config.get("model", {}))
    if config["stage"] == 1:
        return HERAModel(model_config)

    if domains is None or isinstance(domains, (str, bytes)):
        raise ValueError("stage 2 requires an iterable of domains present in the data")
    domains = list(domains)
    if not domains or len(set(domains)) != len(domains) or not set(domains).issubset({"radiology", "pathology"}):
        raise ValueError("domains must contain radiology and/or pathology without duplicates")
    distill = config.get("distillation", {})
    weight = float(distill.get("weight", 1.0))
    specs = distill.get("teachers", {})
    domain_options = {}
    for domain in domains:
        if weight:
            domain_specs = specs.get(domain, {})
            if not domain_specs:
                raise ValueError("teacher dimensions required for {}".format(domain))
            dims = {name: int(options["dim"]) for name, options in domain_specs.items()}
            weights = {name: float(options.get("weight", 1.0)) for name, options in domain_specs.items()}
            domain_options[domain] = (dims, weights)
    backbone = HERAModel(model_config)
    distillers = {
        domain: AMRadioSummaryDistillation(model_config.hidden_size, *domain_options[domain])
        if weight else nn.Identity()
        for domain in domains
    }
    return Stage2Model(backbone, distillers, distillation_weight=weight,
                       distributed_contrastive=distributed_contrastive,
                       **dict(config.get("stage2", {})))


__all__ = ["create_model", "HERAInference", "load_stage2_encoder"]
