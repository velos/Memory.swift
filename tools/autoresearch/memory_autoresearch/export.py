from __future__ import annotations

import math
import shutil
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto
import mlx.core as mx
import numpy as np
import torch
import torch.nn as torch_nn

from .cache import candidate_artifact_path
from .config import DEFAULT_MAX_SEQUENCE_LENGTH, EXPORT_QUANTIZATION, MODEL_SPECS
from .modeling import BertLikeConfig, EmbeddingModel, RerankerModel, TypingModel


def _to_numpy(value):
    return np.array(value)


class TorchBertEmbeddings(torch_nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.word_embeddings = torch_nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = torch_nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = torch_nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = torch_nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, token_type_ids):
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        embedded = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.layer_norm(embedded)


class TorchSelfAttention(torch_nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.query = torch_nn.Linear(config.hidden_size, config.hidden_size)
        self.key = torch_nn.Linear(config.hidden_size, config.hidden_size)
        self.value = torch_nn.Linear(config.hidden_size, config.hidden_size)
        self.output = torch_nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_size = hidden_states.shape
        query = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = attention_mask[:, None, None, :].float()
        scores = scores.masked_fill(mask == 0, -1e9)
        probs = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probs, value).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.output(attended)


class TorchEncoderLayer(torch_nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.attention = TorchSelfAttention(config)
        self.attention_norm = torch_nn.LayerNorm(config.hidden_size)
        self.intermediate = torch_nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = torch_nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_norm = torch_nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask):
        attended = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + attended)
        ff = self.output(torch_nn.functional.gelu(self.intermediate(hidden_states)))
        return self.output_norm(hidden_states + ff)


class TorchBertBackbone(torch_nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.config = config
        self.embeddings = TorchBertEmbeddings(config)
        self.layers = torch_nn.ModuleList([TorchEncoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden_states = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

    def cls(self, input_ids, attention_mask, token_type_ids):
        return self.forward(input_ids, attention_mask, token_type_ids)[:, 0, :]

    def mean_pool(self, input_ids, attention_mask, token_type_ids):
        hidden_states = self.forward(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        return torch_nn.functional.normalize(pooled, p=2, dim=-1)


class TorchTypingModel(torch_nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.backbone = TorchBertBackbone(config)
        self.classifier = torch_nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.classifier(self.backbone.cls(input_ids, attention_mask, token_type_ids))


class TorchEmbeddingModel(torch_nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.backbone = TorchBertBackbone(config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.backbone.mean_pool(input_ids, attention_mask, token_type_ids)


class TorchRerankerModel(torch_nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.backbone = TorchBertBackbone(config)
        self.classifier = torch_nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.classifier(self.backbone.cls(input_ids, attention_mask, token_type_ids))


def _copy_linear_weights(torch_linear, mlx_linear):
    torch_linear.weight.data.copy_(torch.from_numpy(_to_numpy(mlx_linear.weight)))
    if torch_linear.bias is not None and getattr(mlx_linear, "bias", None) is not None:
        torch_linear.bias.data.copy_(torch.from_numpy(_to_numpy(mlx_linear.bias)))


def _copy_layer_norm(torch_ln, mlx_ln):
    torch_ln.weight.data.copy_(torch.from_numpy(_to_numpy(mlx_ln.weight)))
    torch_ln.bias.data.copy_(torch.from_numpy(_to_numpy(mlx_ln.bias)))


def _copy_embeddings(torch_embeddings, mlx_embeddings):
    torch_embeddings.word_embeddings.weight.data.copy_(
        torch.from_numpy(_to_numpy(mlx_embeddings.word_embeddings.weight))
    )
    torch_embeddings.position_embeddings.weight.data.copy_(
        torch.from_numpy(_to_numpy(mlx_embeddings.position_embeddings.weight))
    )
    torch_embeddings.token_type_embeddings.weight.data.copy_(
        torch.from_numpy(_to_numpy(mlx_embeddings.token_type_embeddings.weight))
    )
    _copy_layer_norm(torch_embeddings.layer_norm, mlx_embeddings.layer_norm)


def _copy_backbone(torch_backbone, mlx_backbone):
    _copy_embeddings(torch_backbone.embeddings, mlx_backbone.embeddings)
    for torch_layer, mlx_layer in zip(torch_backbone.layers, mlx_backbone.layers):
        _copy_linear_weights(torch_layer.attention.query, mlx_layer.attention.query)
        _copy_linear_weights(torch_layer.attention.key, mlx_layer.attention.key)
        _copy_linear_weights(torch_layer.attention.value, mlx_layer.attention.value)
        _copy_linear_weights(torch_layer.attention.output, mlx_layer.attention.output)
        _copy_layer_norm(torch_layer.attention_norm, mlx_layer.attention_norm)
        _copy_linear_weights(torch_layer.intermediate, mlx_layer.intermediate)
        _copy_linear_weights(torch_layer.output, mlx_layer.output)
        _copy_layer_norm(torch_layer.output_norm, mlx_layer.output_norm)


def _mirror_model(component: str, mlx_model, config: BertLikeConfig):
    if component == "typing":
        torch_model = TorchTypingModel(config)
        _copy_backbone(torch_model.backbone, mlx_model.backbone)
        _copy_linear_weights(torch_model.classifier, mlx_model.classifier)
        output_name = "type_logits"
    elif component == "embedding":
        torch_model = TorchEmbeddingModel(config)
        _copy_backbone(torch_model.backbone, mlx_model.backbone)
        output_name = "embedding"
    elif component == "reranker":
        torch_model = TorchRerankerModel(config)
        _copy_backbone(torch_model.backbone, mlx_model.backbone)
        _copy_linear_weights(torch_model.classifier, mlx_model.classifier)
        output_name = "relevance_score"
    else:
        raise ValueError(f"Unsupported component: {component}")
    torch_model.eval()
    return torch_model, output_name


def export_coreml_model(component: str, mlx_model, config: BertLikeConfig, output_path: Path | None = None) -> Path:
    torch_model, output_name = _mirror_model(component, mlx_model, config)
    output_path = output_path or candidate_artifact_path(component)
    if output_path.exists():
        shutil.rmtree(output_path)
    dummy_ids = torch.zeros((1, config.max_position_embeddings), dtype=torch.int32)
    dummy_mask = torch.ones((1, config.max_position_embeddings), dtype=torch.int32)
    dummy_types = torch.zeros((1, config.max_position_embeddings), dtype=torch.int32)
    traced = torch.jit.trace(torch_model, (dummy_ids, dummy_mask, dummy_types))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=dummy_ids.shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=dummy_mask.shape, dtype=np.int32),
            ct.TensorType(name="token_type_ids", shape=dummy_types.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name=output_name, dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    quantized_dtype = EXPORT_QUANTIZATION.get(component)
    if quantized_dtype:
        compression_config = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=quantized_dtype,
            )
        )
        mlmodel = cto.coreml.linear_quantize_weights(mlmodel, compression_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return output_path


def artifact_size_mb(path: Path) -> float:
    total_bytes = 0
    if path.is_file():
        total_bytes = path.stat().st_size
    elif path.exists():
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_bytes += file_path.stat().st_size
    return total_bytes / (1024 * 1024)
