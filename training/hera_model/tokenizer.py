"""Construct the project's existing text tokenizers from a configuration."""

from collections.abc import Mapping

from hera.data import SentencePieceTokenizer, SyntheticTokenizer


def create_tokenizer(config):
    """Create a tokenizer from a mapping such as a YAML ``tokenizer`` section.

    ``synthetic`` is for demonstrations and tests only, not real training.
    ``sentencepiece`` requires a real model and explicitly configured IDs.
    """
    if not isinstance(config, Mapping):
        raise TypeError("tokenizer config must be a mapping")
    options = dict(config)
    kind = options.pop("kind", None)
    if kind == "sentencepiece":
        return SentencePieceTokenizer(**options)
    if kind == "synthetic":
        return SyntheticTokenizer(**options)
    raise ValueError("tokenizer.kind must be sentencepiece or synthetic")
