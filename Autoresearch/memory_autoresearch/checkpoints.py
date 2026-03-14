from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mlx.core as mx
import torch
from mlx.utils import tree_flatten
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

from .modeling import BertLikeConfig


def checkpoint_config(checkpoint: str, num_labels: int = 8) -> BertLikeConfig:
    config = AutoConfig.from_pretrained(checkpoint)
    return BertLikeConfig(
        vocab_size=int(config.vocab_size),
        hidden_size=int(getattr(config, "hidden_size", 384)),
        intermediate_size=int(getattr(config, "intermediate_size", 1536)),
        num_layers=int(getattr(config, "num_hidden_layers", 6)),
        num_heads=int(getattr(config, "num_attention_heads", 6)),
        max_position_embeddings=int(getattr(config, "max_position_embeddings", 512)),
        type_vocab_size=int(getattr(config, "type_vocab_size", 2)),
        num_labels=num_labels,
    )


def save_mlx_weights(model, output_path: Path, metadata: dict[str, str] | None = None) -> None:
    flat = tree_flatten(model.parameters())
    arrays = {path: np.array(value) for path, value in flat}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    if metadata:
        meta_path = output_path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _find_tensor(state_dict: dict[str, torch.Tensor], suffixes: list[str]) -> torch.Tensor | None:
    for suffix in suffixes:
        for key, value in state_dict.items():
            if key.endswith(suffix):
                return value
    return None


def _assign_array(target, attr: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        return
    setattr(target, attr, mx.array(tensor.detach().cpu().numpy()))


def load_pretrained_weights(model, component: str, checkpoint: str) -> None:
    if component == "reranker":
        source_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    else:
        source_model = AutoModel.from_pretrained(checkpoint)
    state_dict = source_model.state_dict()

    embeddings = model.backbone.embeddings
    _assign_array(
        embeddings.word_embeddings,
        "weight",
        _find_tensor(state_dict, ["embeddings.word_embeddings.weight", "word_embeddings.weight"]),
    )
    _assign_array(
        embeddings.position_embeddings,
        "weight",
        _find_tensor(state_dict, ["embeddings.position_embeddings.weight", "position_embeddings.weight"]),
    )
    _assign_array(
        embeddings.token_type_embeddings,
        "weight",
        _find_tensor(state_dict, ["embeddings.token_type_embeddings.weight", "token_type_embeddings.weight"]),
    )
    _assign_array(
        embeddings.layer_norm,
        "weight",
        _find_tensor(state_dict, ["embeddings.LayerNorm.weight", "layer_norm.weight"]),
    )
    _assign_array(
        embeddings.layer_norm,
        "bias",
        _find_tensor(state_dict, ["embeddings.LayerNorm.bias", "layer_norm.bias"]),
    )

    for index, layer in enumerate(model.backbone.layers):
        prefix = f"encoder.layer.{index}"
        _assign_array(layer.attention.query, "weight", _find_tensor(state_dict, [f"{prefix}.attention.self.query.weight"]))
        _assign_array(layer.attention.query, "bias", _find_tensor(state_dict, [f"{prefix}.attention.self.query.bias"]))
        _assign_array(layer.attention.key, "weight", _find_tensor(state_dict, [f"{prefix}.attention.self.key.weight"]))
        _assign_array(layer.attention.key, "bias", _find_tensor(state_dict, [f"{prefix}.attention.self.key.bias"]))
        _assign_array(layer.attention.value, "weight", _find_tensor(state_dict, [f"{prefix}.attention.self.value.weight"]))
        _assign_array(layer.attention.value, "bias", _find_tensor(state_dict, [f"{prefix}.attention.self.value.bias"]))
        _assign_array(layer.attention.output, "weight", _find_tensor(state_dict, [f"{prefix}.attention.output.dense.weight"]))
        _assign_array(layer.attention.output, "bias", _find_tensor(state_dict, [f"{prefix}.attention.output.dense.bias"]))
        _assign_array(layer.attention_norm, "weight", _find_tensor(state_dict, [f"{prefix}.attention.output.LayerNorm.weight"]))
        _assign_array(layer.attention_norm, "bias", _find_tensor(state_dict, [f"{prefix}.attention.output.LayerNorm.bias"]))
        _assign_array(layer.intermediate, "weight", _find_tensor(state_dict, [f"{prefix}.intermediate.dense.weight"]))
        _assign_array(layer.intermediate, "bias", _find_tensor(state_dict, [f"{prefix}.intermediate.dense.bias"]))
        _assign_array(layer.output, "weight", _find_tensor(state_dict, [f"{prefix}.output.dense.weight"]))
        _assign_array(layer.output, "bias", _find_tensor(state_dict, [f"{prefix}.output.dense.bias"]))
        _assign_array(layer.output_norm, "weight", _find_tensor(state_dict, [f"{prefix}.output.LayerNorm.weight"]))
        _assign_array(layer.output_norm, "bias", _find_tensor(state_dict, [f"{prefix}.output.LayerNorm.bias"]))

    if component == "reranker":
        head_weight = _find_tensor(state_dict, ["classifier.weight", "score.weight"])
        head_bias = _find_tensor(state_dict, ["classifier.bias", "score.bias"])
        if head_weight is not None:
            model.classifier.weight = mx.array(head_weight.detach().cpu().numpy())
        if head_bias is not None:
            model.classifier.bias = mx.array(head_bias.detach().cpu().numpy())
