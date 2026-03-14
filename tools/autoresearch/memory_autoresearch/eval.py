from __future__ import annotations

import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np

from .cache import metrics_path
from .config import MEMORY_TYPE_TO_INDEX
from .scoring import EvalMetrics, build_memory_score, compute_recall_score, compute_storage_score
from .tokenization import BertTokenizerAdapter
from .upstream import install_artifact_into_upstream, restore_baseline_artifacts, run_memory_eval


def _artifact_model(path: Path):
    return ct.models.MLModel(str(path))


def benchmark_artifact(path: Path, component: str, tokenizer: BertTokenizerAdapter, sample_texts: list[str]) -> float:
    if not sample_texts:
        return 0.0
    model = _artifact_model(path)
    elapsed: list[float] = []
    for text in sample_texts[:16]:
        tokenized = tokenizer.encode_texts([text])
        inputs = {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "token_type_ids": tokenized.token_type_ids,
        }
        start = time.perf_counter()
        model.predict(inputs)
        elapsed.append((time.perf_counter() - start) * 1000.0)
    if len(elapsed) == 1:
        return elapsed[0]
    return float(np.percentile(np.array(elapsed, dtype=np.float32), 95))


def _macro_f1(expected: list[int], predicted: list[int], num_labels: int) -> float:
    f1_scores = []
    for label in range(num_labels):
        tp = sum(1 for exp, pred in zip(expected, predicted) if exp == pred == label)
        fp = sum(1 for exp, pred in zip(expected, predicted) if exp != label and pred == label)
        fn = sum(1 for exp, pred in zip(expected, predicted) if exp == label and pred != label)
        if tp == fp == fn == 0:
            f1_scores.append(0.0)
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return sum(f1_scores) / max(len(f1_scores), 1)


def evaluate_typing_artifact(artifact_path: Path, dataset_root: Path, checkpoint: str) -> tuple[float, float, float, float]:
    tokenizer = BertTokenizerAdapter(checkpoint)
    model = _artifact_model(artifact_path)
    expected: list[int] = []
    predicted: list[int] = []
    span_hits = 0
    span_total = 0
    sample_texts: list[str] = []
    storage_path = dataset_root / "storage_cases.jsonl"
    if not storage_path.exists():
        return 0.0, 0.0, 0.0, 0.0
    with storage_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            sample_texts.append(case["text"])
            tokenized = tokenizer.encode_texts([case["text"]])
            outputs = model.predict(
                {
                    "input_ids": tokenized.input_ids,
                    "attention_mask": tokenized.attention_mask,
                    "token_type_ids": tokenized.token_type_ids,
                }
            )
            logits = np.asarray(outputs["type_logits"]).reshape(-1)
            predicted_label = int(np.argmax(logits))
            expected.append(_memory_type_index(case["expected_memory_type"]))
            predicted.append(predicted_label)
            for span in case.get("required_spans", []):
                span_total += 1
                if span.lower() in case["text"].lower():
                    span_hits += 1
    if not expected:
        return 0.0, 0.0, 0.0, 0.0
    type_accuracy = sum(1 for exp, pred in zip(expected, predicted) if exp == pred) / len(expected)
    macro_f1 = _macro_f1(expected, predicted, num_labels=8)
    span_coverage = span_hits / max(span_total, 1)
    latency = benchmark_artifact(artifact_path, "typing", tokenizer, sample_texts)
    return type_accuracy, macro_f1, span_coverage, latency


def _memory_type_index(name: str) -> int:
    return MEMORY_TYPE_TO_INDEX[name]


def _parse_recall_metrics(report: dict) -> tuple[float, float, float, float]:
    metrics = {entry["k"]: entry for entry in report["recall"]["metricsByK"]}
    metric = metrics.get(10) or max(metrics.values(), key=lambda entry: entry["k"])
    return (
        float(metric["hitRate"]),
        float(metric["recall"]),
        float(metric["mrr"]),
        float(metric["ndcg"]),
    )


def evaluate_candidate(
    component: str,
    artifact_path: Path | None,
    quick_dataset_root: Path,
    full_dataset_root: Path,
    checkpoint: str,
) -> tuple[EvalMetrics, EvalMetrics]:
    restore_baseline_artifacts()
    try:
        quick_report = _evaluate_recall_report(component, artifact_path, quick_dataset_root)
        full_report = _evaluate_recall_report(component, artifact_path, full_dataset_root)
        quick_typing = (0.0, 0.0, 0.0, 0.0)
        full_typing = (0.0, 0.0, 0.0, 0.0)
        if component == "typing" and artifact_path is not None:
            quick_typing = evaluate_typing_artifact(artifact_path, quick_dataset_root, checkpoint)
            full_typing = evaluate_typing_artifact(artifact_path, full_dataset_root, checkpoint)
        quick_metrics = _build_metrics(
            component=component,
            quick_report=quick_report,
            artifact_path=artifact_path,
            typing_type_accuracy=quick_typing[0],
            typing_macro_f1=quick_typing[1],
            typing_span_coverage=quick_typing[2],
            typing_latency=quick_typing[3],
        )
        full_metrics = _build_metrics(
            component=component,
            quick_report=full_report,
            artifact_path=artifact_path,
            typing_type_accuracy=full_typing[0],
            typing_macro_f1=full_typing[1],
            typing_span_coverage=full_typing[2],
            typing_latency=full_typing[3],
        )
        return quick_metrics, full_metrics
    finally:
        restore_baseline_artifacts()


def _evaluate_recall_report(component: str, artifact_path: Path | None, dataset_root: Path) -> dict:
    if component in {"embedding", "reranker"} and artifact_path is not None:
        install_artifact_into_upstream(component, artifact_path)
    output_path = metrics_path(component).with_name(f"{component}_{dataset_root.name}.json")
    return run_memory_eval(dataset_root=dataset_root, output_path=output_path)


def _build_metrics(
    component: str,
    quick_report: dict,
    artifact_path: Path | None,
    typing_type_accuracy: float,
    typing_macro_f1: float,
    typing_span_coverage: float,
    typing_latency: float,
) -> EvalMetrics:
    hit_rate, recall, mrr, ndcg = _parse_recall_metrics(quick_report)
    recall_score = compute_recall_score(hit_rate, recall, mrr, ndcg)
    storage = quick_report["storage"]
    type_accuracy = typing_type_accuracy if component == "typing" else float(storage["typeAccuracy"])
    macro_f1 = typing_macro_f1 if component == "typing" else float(storage["macroF1"])
    span_coverage = typing_span_coverage if component == "typing" else float(storage["spanCoverage"])
    storage_score = compute_storage_score(type_accuracy, macro_f1, span_coverage)
    artifact_size = 0.0
    if artifact_path is not None:
        from .export import artifact_size_mb

        artifact_size = artifact_size_mb(artifact_path)
    latency = typing_latency if component == "typing" else float(
        quick_report["recall"].get("latencyStats", {}).get("p95Ms", 0.0)
    )
    memory_score = build_memory_score(component, storage_score, recall_score)
    return EvalMetrics(
        component=component,
        storage_score=storage_score,
        recall_score=recall_score,
        memory_score=memory_score,
        model_mb=artifact_size,
        latency_ms=latency,
        type_accuracy=type_accuracy,
        macro_f1=macro_f1,
        span_coverage=span_coverage,
        hit_rate_at_10=hit_rate,
        recall_at_10=recall,
        mrr_at_10=mrr,
        ndcg_at_10=ndcg,
    )
