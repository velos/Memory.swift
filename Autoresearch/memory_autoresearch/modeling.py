from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


def gelu(x):
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))


@dataclass
class BertLikeConfig:
    vocab_size: int
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_layers: int = 6
    num_heads: int = 6
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    num_labels: int = 8


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, input_ids, token_type_ids):
        batch_size, seq_len = input_ids.shape
        positions = mx.broadcast_to(mx.arange(seq_len)[None, :], (batch_size, seq_len))
        embedded = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.layer_norm(embedded)


class SelfAttention(nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_size = hidden_states.shape
        query = self.query(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        mask = attention_mask[:, None, None, :].astype(mx.float32)
        additive = mx.where(mask > 0, mx.array(0.0), mx.array(float("-inf")))
        scores = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=1.0 / math.sqrt(self.head_dim),
            mask=additive,
        )
        scores = scores.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
        return self.output(scores)


class EncoderLayer(nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.attention = SelfAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, hidden_states, attention_mask):
        attended = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + attended)
        feed_forward = self.output(gelu(self.intermediate(hidden_states)))
        hidden_states = self.output_norm(hidden_states + feed_forward)
        return hidden_states


class BertBackbone(nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.layers = [EncoderLayer(config) for _ in range(config.num_layers)]

    def __call__(self, input_ids, attention_mask, token_type_ids):
        hidden_states = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

    def cls(self, input_ids, attention_mask, token_type_ids):
        return self(input_ids, attention_mask, token_type_ids)[:, 0, :]

    def mean_pool(self, input_ids, attention_mask, token_type_ids):
        hidden_states = self(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.astype(mx.float32)[..., None]
        summed = mx.sum(hidden_states * mask, axis=1)
        counts = mx.maximum(mx.sum(mask, axis=1), 1.0)
        pooled = summed / counts
        norms = mx.sqrt(mx.maximum(mx.sum(pooled * pooled, axis=-1, keepdims=True), 1e-12))
        return pooled / norms


class TypingModel(nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.backbone = BertBackbone(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, input_ids, attention_mask, token_type_ids):
        return self.classifier(self.backbone.cls(input_ids, attention_mask, token_type_ids))


class EmbeddingModel(nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.backbone = BertBackbone(config)

    def __call__(self, input_ids, attention_mask, token_type_ids):
        return self.backbone.mean_pool(input_ids, attention_mask, token_type_ids)


class RerankerModel(nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.backbone = BertBackbone(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def __call__(self, input_ids, attention_mask, token_type_ids):
        return self.classifier(self.backbone.cls(input_ids, attention_mask, token_type_ids)).squeeze(-1)
