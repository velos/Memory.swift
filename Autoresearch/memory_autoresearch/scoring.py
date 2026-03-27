from __future__ import annotations

from dataclasses import dataclass

from .config import (
    COMPONENT_GATES,
    MODEL_SIZE_TOLERANCE_MULTIPLIER,
    HYSTERESIS_MIN_IMPROVEMENT,
    HYSTERESIS_TIE_TOLERANCE,
    MEMORY_SCORE_WEIGHTS,
    PRIMARY_LATENCY_TOLERANCE,
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


def decide_candidate_status(
    quick_metrics: EvalMetrics,
    full_metrics: EvalMetrics,
    baseline_metrics: EvalMetrics | None,
    current_report: dict | None,
    baseline_report: dict | None,
) -> tuple[str, str]:
    if not gate_passed(quick_metrics.component, quick_metrics.model_mb, quick_metrics.latency_ms):
        return "discard", (
            f"hard gate failed: model_mb={quick_metrics.model_mb:.1f}, "
            f"latency_ms={quick_metrics.latency_ms:.1f}"
        )

    if quick_metrics.component == "typing":
        if baseline_report is None:
            return "keep", "no compatible typing baseline report present; establishing new product-aligned baseline"
        if "typing_gold_v1" not in baseline_report.get("full", {}).get("corpora", {}):
            return "keep", "incompatible baseline: typing_gold_v1 not present; establishing new product-aligned baseline"

    if baseline_metrics is None:
        return "keep", "no baseline summary present; establishing first product-aligned baseline"

    size_limit = baseline_metrics.model_mb * MODEL_SIZE_TOLERANCE_MULTIPLIER
    if baseline_metrics.model_mb > 0 and quick_metrics.model_mb > size_limit:
        return "discard", (
            f"model size exceeded tolerance: {quick_metrics.model_mb:.1f}MB > {size_limit:.1f}MB"
        )

    quick_keep = should_keep_candidate(quick_metrics, baseline_metrics)
    if not quick_keep:
        return "discard", _improvement_reason(quick_metrics, baseline_metrics)

    quick_ok, quick_reason = _primary_dataset_guard(
        phase="quick",
        current_report=current_report,
        baseline_report=baseline_report,
    )
    if not quick_ok:
        return "discard", quick_reason

    full_ok, full_reason = _primary_dataset_guard(
        phase="full",
        current_report=current_report,
        baseline_report=baseline_report,
    )
    if not full_ok:
        return "discard_full", full_reason

    return "keep", full_reason


def _improvement_reason(current: EvalMetrics, baseline: EvalMetrics) -> str:
    improvement = current.memory_score - baseline.memory_score
    if improvement >= HYSTERESIS_MIN_IMPROVEMENT:
        return f"memory_score improved by {improvement:.4f}"
    if abs(improvement) <= HYSTERESIS_TIE_TOLERANCE:
        return (
            "memory_score tied within hysteresis but size/latency did not improve enough "
            f"(delta={improvement:.4f})"
        )
    return f"memory_score regressed or improved too little (delta={improvement:.4f})"


def _primary_dataset_guard(
    phase: str,
    current_report: dict | None,
    baseline_report: dict | None,
) -> tuple[bool, str]:
    if not current_report or not baseline_report:
        return True, f"{phase} pass: no corpus-level baseline report available"

    current_phase = current_report.get(phase, {})
    baseline_phase = baseline_report.get(phase, {})
    current_corpora = current_phase.get("corpora", {})
    baseline_corpora = baseline_phase.get("corpora", {})
    if not current_corpora or not baseline_corpora:
        return True, f"{phase} pass: missing corpus-level details"

    general_reason = _corpus_memory_check("general", current_corpora, baseline_corpora)
    if general_reason is not None:
        return False, f"{phase} fail: {general_reason}"

    long_reason = _corpus_memory_check("longmemeval", current_corpora, baseline_corpora, allow_tie=True)
    if long_reason is not None:
        return False, f"{phase} fail: {long_reason}"

    if current_report.get("component") == "typing":
        gold_reason = _corpus_memory_check("typing_gold_v1", current_corpora, baseline_corpora)
        if gold_reason is not None:
            return False, f"{phase} fail: {gold_reason}"

    latency_reason = _latency_guard(phase, current_corpora, baseline_corpora)
    if latency_reason is not None:
        return False, f"{phase} fail: {latency_reason}"

    general_delta = _corpus_delta("general", current_corpora, baseline_corpora, "memory_score")
    long_delta = _corpus_delta("longmemeval", current_corpora, baseline_corpora, "memory_score")
    gold_delta = _corpus_delta("typing_gold_v1", current_corpora, baseline_corpora, "memory_score")
    detail = f"{phase} pass: general_delta={general_delta:.4f}, longmemeval_delta={long_delta:.4f}"
    if current_report.get("component") == "typing" and gold_delta is not None:
        detail += f", typing_gold_delta={gold_delta:.4f}"
    return (True, detail)


def _corpus_memory_check(
    corpus: str,
    current_corpora: dict,
    baseline_corpora: dict,
    allow_tie: bool = False,
) -> str | None:
    delta = _corpus_delta(corpus, current_corpora, baseline_corpora, "memory_score")
    if delta is None:
        return None
    floor = -HYSTERESIS_TIE_TOLERANCE if allow_tie else 0.0
    if delta < floor:
        comparator = "regressed" if delta < 0 else "did not improve"
        return f"{corpus} {comparator} (delta={delta:.4f})"
    return None


def _latency_guard(phase: str, current_corpora: dict, baseline_corpora: dict) -> str | None:
    for corpus, tolerance in PRIMARY_LATENCY_TOLERANCE.items():
        current = current_corpora.get(corpus, {}).get("latency_ms")
        baseline = baseline_corpora.get(corpus, {}).get("latency_ms")
        if not current or not baseline:
            continue
        limit = baseline * (1.0 + tolerance)
        if current > limit:
            return (
                f"{corpus} latency exceeded tolerance in {phase}: "
                f"{current:.1f}ms > {limit:.1f}ms"
            )
    return None


def _corpus_delta(corpus: str, current_corpora: dict, baseline_corpora: dict, field: str) -> float | None:
    current = current_corpora.get(corpus, {}).get(field)
    baseline = baseline_corpora.get(corpus, {}).get(field)
    if current is None or baseline is None:
        return None
    return float(current) - float(baseline)
