from __future__ import annotations

from dataclasses import dataclass

from .config import (
    COMPONENT_GATES,
    HYSTERESIS_MIN_IMPROVEMENT,
    HYSTERESIS_TIE_TOLERANCE,
    MEMORY_SCORE_WEIGHTS,
    RETRIEVAL_SCORE_WEIGHTS,
    STORAGE_SCORE_WEIGHTS,
)


@dataclass
class EvalMetrics:
    component: str
    storage_score: float
    recall_score: float
    memory_score: float
    model_mb: float
    latency_ms: float
    type_accuracy: float = 0.0
    macro_f1: float = 0.0
    span_coverage: float = 0.0
    hit_rate_at_10: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0


def compute_storage_score(type_accuracy: float, macro_f1: float, span_coverage: float) -> float:
    return (
        STORAGE_SCORE_WEIGHTS["type_accuracy"] * type_accuracy
        + STORAGE_SCORE_WEIGHTS["macro_f1"] * macro_f1
        + STORAGE_SCORE_WEIGHTS["span_coverage"] * span_coverage
    )


def compute_recall_score(hit_rate: float, recall: float, mrr: float, ndcg: float) -> float:
    return (
        RETRIEVAL_SCORE_WEIGHTS["mrr"] * mrr
        + RETRIEVAL_SCORE_WEIGHTS["ndcg"] * ndcg
        + RETRIEVAL_SCORE_WEIGHTS["recall"] * recall
        + RETRIEVAL_SCORE_WEIGHTS["hit_rate"] * hit_rate
    )


def build_memory_score(component: str, storage_score: float, recall_score: float) -> float:
    weights = MEMORY_SCORE_WEIGHTS[component]
    return weights["storage"] * storage_score + weights["recall"] * recall_score


def gate_passed(component: str, model_mb: float, latency_ms: float) -> bool:
    gates = COMPONENT_GATES[component]
    return model_mb <= gates["model_mb"] and latency_ms <= gates["latency_ms"]


def should_keep_candidate(current: EvalMetrics, baseline: EvalMetrics | None) -> bool:
    if not gate_passed(current.component, current.model_mb, current.latency_ms):
        return False
    if baseline is None:
        return True
    improvement = current.memory_score - baseline.memory_score
    if improvement >= HYSTERESIS_MIN_IMPROVEMENT:
        return True
    if abs(improvement) <= HYSTERESIS_TIE_TOLERANCE:
        if current.model_mb < baseline.model_mb or current.latency_ms < baseline.latency_ms:
            return True
    return False
