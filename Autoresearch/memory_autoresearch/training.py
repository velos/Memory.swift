from __future__ import annotations

from collections import defaultdict
import random
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

from .config import DEFAULT_TIME_BUDGET_SECONDS, MODEL_SPECS, RANDOM_SEED
from .data import RetrievalExample, TypingExample, class_weights
from .hardware import HardwareProfile
from .modeling import EmbeddingModel, RerankerModel, TypingModel
from .tokenization import BertTokenizerAdapter


@dataclass
class TrainingResult:
    model: object
    training_seconds: float
    steps: int
    average_loss: float


class AdamW:
    def __init__(
        self, model, lr: float, weight_decay: float = 0.01, betas=(0.9, 0.999)
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.state = {}
        self.paths = [path for path, _ in tree_flatten(model.parameters())]

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        beta1, beta2 = self.betas
        for path, grad in flat_grads.items():
            if path not in flat_params:
                continue
            param = flat_params[path]
            grad_f32 = grad.astype(mx.float32)
            param_f32 = param.astype(mx.float32)
            state = self.state.setdefault(
                path,
                {"m": mx.zeros_like(param_f32), "v": mx.zeros_like(param_f32), "t": 0},
            )
            state["t"] += 1
            state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad_f32
            state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad_f32 * grad_f32)
            m_hat = state["m"] / (1.0 - beta1 ** state["t"])
            v_hat = state["v"] / (1.0 - beta2 ** state["t"])
            updated = param_f32 * (1.0 - self.lr * self.weight_decay)
            updated = updated - self.lr * (m_hat / (mx.sqrt(v_hat) + 1e-8))
            self._set_path_value(model, path, updated.astype(param.dtype))


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def _shuffle_indices(length: int):
    indices = list(range(length))
    random.shuffle(indices)
    return indices


def _weighted_mean(losses, weights=None):
    if weights is None:
        return mx.mean(losses)
    return mx.sum(losses * weights) / mx.maximum(mx.sum(weights), 1.0)


def _focal_loss(logits, labels, gamma: float, weights=None):
    losses = nn.losses.cross_entropy(logits, labels, reduction="none")
    probabilities = mx.softmax(logits, axis=-1)
    target_probabilities = mx.take_along_axis(
        probabilities, labels.reshape(-1, 1), axis=1
    ).reshape((-1,))
    target_probabilities = mx.clip(target_probabilities, 1e-6, 1.0)
    focal_scale = (1.0 - target_probabilities) ** gamma
    return _weighted_mean(losses * focal_scale, weights)


def _sample_typing_examples(
    examples: list[TypingExample],
    sampling_mode: str,
    sampling_balance: float = 1.0,
) -> list[TypingExample]:
    if sampling_mode == "natural" or sampling_balance <= 0.0:
        return examples
    if sampling_mode != "balanced":
        raise ValueError(f"Unsupported typing sampling_mode: {sampling_mode}")
    buckets: dict[str, list[TypingExample]] = defaultdict(list)
    for example in examples:
        buckets[example.label].append(example)
    labels = [label for label, bucket in buckets.items() if bucket]
    if not labels:
        return examples
    sample_size = len(examples)
    balanced_count = min(
        sample_size, max(0, int(round(sample_size * sampling_balance)))
    )
    sampled_examples = list(examples[: sample_size - balanced_count])
    balanced_examples: list[TypingExample] = []
    while len(balanced_examples) < balanced_count:
        round_labels = list(labels)
        random.shuffle(round_labels)
        for label in round_labels:
            bucket = buckets[label]
            balanced_examples.append(bucket[random.randrange(len(bucket))])
            if len(balanced_examples) >= balanced_count:
                break
    sampled_examples.extend(balanced_examples)
    random.shuffle(sampled_examples)
    return sampled_examples


def train_typing(
    model: TypingModel,
    tokenizer: BertTokenizerAdapter,
    examples: list[TypingExample],
    hardware: HardwareProfile,
    time_budget_seconds: int = DEFAULT_TIME_BUDGET_SECONDS,
    learning_rate: float = 3e-4,
    class_weight_amplify: float = 1.0,
    focal_gamma: float | None = None,
    sampling_mode: str = "natural",
    sampling_balance: float = 1.0,
) -> TrainingResult:
    set_random_seed()
    batch_size = hardware.typing_batch_size
    label_weights = mx.array(
        class_weights(examples, amplify=class_weight_amplify), dtype=mx.float32
    )

    def loss_fn(model, input_ids, attention_mask, token_type_ids, labels):
        logits = model(input_ids, attention_mask, token_type_ids)
        weights = label_weights[labels]
        if focal_gamma is not None:
            return _focal_loss(logits, labels, gamma=focal_gamma, weights=weights)
        losses = nn.losses.cross_entropy(logits, labels, reduction="none")
        return _weighted_mean(losses, weights)

    grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = AdamW(model, lr=learning_rate, weight_decay=0.01)

    start = time.time()
    steps = 0
    loss_total = 0.0
    while time.time() - start < time_budget_seconds:
        epoch_examples = _sample_typing_examples(
            examples,
            sampling_mode=sampling_mode,
            sampling_balance=sampling_balance,
        )
        for offset in range(0, len(epoch_examples), batch_size):
            if time.time() - start >= time_budget_seconds:
                break
            batch = epoch_examples[offset : offset + batch_size]
            tokenized = tokenizer.encode_texts([example.text for example in batch])
            labels = np.array(
                [class_weights_label(example.label) for example in batch],
                dtype=np.int32,
            )
            loss, grads = grad_fn(
                model,
                mx.array(tokenized.input_ids),
                mx.array(tokenized.attention_mask),
                mx.array(tokenized.token_type_ids),
                mx.array(labels),
            )
            mx.eval(loss, grads)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            loss_total += float(loss.item())
            steps += 1
    return TrainingResult(
        model=model,
        training_seconds=time.time() - start,
        steps=steps,
        average_loss=loss_total / max(steps, 1),
    )


def class_weights_label(label: str) -> int:
    from .config import MEMORY_TYPE_TO_INDEX

    return MEMORY_TYPE_TO_INDEX[label]


def _contrastive_loss(query_embeddings, document_embeddings):
    logits = query_embeddings @ document_embeddings.T
    labels = mx.arange(logits.shape[0], dtype=mx.int32)
    forward_loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
    backward_loss = nn.losses.cross_entropy(logits.T, labels, reduction="mean")
    return 0.5 * (forward_loss + backward_loss)


def train_embedding(
    model: EmbeddingModel,
    tokenizer: BertTokenizerAdapter,
    examples: list[RetrievalExample],
    document_lookup: dict[str, str],
    hardware: HardwareProfile,
    time_budget_seconds: int = DEFAULT_TIME_BUDGET_SECONDS,
    learning_rate: float = 2e-4,
) -> TrainingResult:
    set_random_seed()
    batch_size = hardware.embedding_batch_size

    def loss_fn(model, q_ids, q_mask, q_types, d_ids, d_mask, d_types):
        query_embeddings = model(q_ids, q_mask, q_types)
        document_embeddings = model(d_ids, d_mask, d_types)
        return _contrastive_loss(query_embeddings, document_embeddings)

    grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = AdamW(model, lr=learning_rate, weight_decay=0.01)
    start = time.time()
    steps = 0
    loss_total = 0.0
    shuffled = list(examples)
    while time.time() - start < time_budget_seconds:
        random.shuffle(shuffled)
        for offset in range(0, len(shuffled), batch_size):
            if time.time() - start >= time_budget_seconds:
                break
            batch = shuffled[offset : offset + batch_size]
            queries = [example.query for example in batch]
            documents = [example.positive_document_text for example in batch]
            query_tokens = tokenizer.encode_texts(queries)
            document_tokens = tokenizer.encode_texts(documents)
            loss, grads = grad_fn(
                model,
                mx.array(query_tokens.input_ids),
                mx.array(query_tokens.attention_mask),
                mx.array(query_tokens.token_type_ids),
                mx.array(document_tokens.input_ids),
                mx.array(document_tokens.attention_mask),
                mx.array(document_tokens.token_type_ids),
            )
            mx.eval(loss, grads)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            loss_total += float(loss.item())
            steps += 1
    return TrainingResult(
        model=model,
        training_seconds=time.time() - start,
        steps=steps,
        average_loss=loss_total / max(steps, 1),
    )


def train_reranker(
    model: RerankerModel,
    tokenizer: BertTokenizerAdapter,
    examples: list[RetrievalExample],
    document_lookup: dict[str, str],
    hardware: HardwareProfile,
    time_budget_seconds: int = DEFAULT_TIME_BUDGET_SECONDS,
    learning_rate: float = 2e-4,
) -> TrainingResult:
    set_random_seed()
    batch_size = hardware.reranker_batch_size

    def loss_fn(model, input_ids, attention_mask, token_type_ids, labels):
        logits = model(input_ids, attention_mask, token_type_ids)
        probs = mx.sigmoid(logits)
        labels_f = labels.astype(mx.float32)
        loss = -(
            labels_f * mx.log(mx.maximum(probs, 1e-6))
            + (1.0 - labels_f) * mx.log(mx.maximum(1.0 - probs, 1e-6))
        )
        return mx.mean(loss)

    grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = AdamW(model, lr=learning_rate, weight_decay=0.01)
    start = time.time()
    steps = 0
    loss_total = 0.0

    positives = list(examples)
    while time.time() - start < time_budget_seconds:
        random.shuffle(positives)
        pair_queries: list[str] = []
        pair_docs: list[str] = []
        pair_labels: list[int] = []
        for example in positives:
            pair_queries.append(example.query)
            pair_docs.append(example.positive_document_text)
            pair_labels.append(1)
            for negative_id in example.hard_negative_ids[:1]:
                negative_doc = document_lookup.get(negative_id)
                if negative_doc is None:
                    continue
                pair_queries.append(example.query)
                pair_docs.append(negative_doc)
                pair_labels.append(0)
        for offset in range(0, len(pair_labels), batch_size):
            if time.time() - start >= time_budget_seconds:
                break
            queries = pair_queries[offset : offset + batch_size]
            documents = pair_docs[offset : offset + batch_size]
            labels = np.array(
                pair_labels[offset : offset + batch_size], dtype=np.float32
            )
            tokenized = tokenizer.encode_pairs(queries, documents)
            loss, grads = grad_fn(
                model,
                mx.array(tokenized.input_ids),
                mx.array(tokenized.attention_mask),
                mx.array(tokenized.token_type_ids),
                mx.array(labels),
            )
            mx.eval(loss, grads)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            loss_total += float(loss.item())
            steps += 1
    return TrainingResult(
        model=model,
        training_seconds=time.time() - start,
        steps=steps,
        average_loss=loss_total / max(steps, 1),
    )
