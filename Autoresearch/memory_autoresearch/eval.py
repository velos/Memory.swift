from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import coremltools as ct
import numpy as np

from .cache import metrics_path
from .config import COMPONENT_CORPUS_SCORE_WEIGHTS, MEMORY_TYPE_TO_INDEX
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


@dataclass
class CorpusEvalMetrics:
    corpus: str
    storage_score: float
    recall_score: float
    memory_score: float
    latency_ms: float
    recall_query_count: int = 0
    type_accuracy: float = 0.0
    macro_f1: float = 0.0
    span_coverage: float = 0.0
    hit_rate_at_10: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    stage_latency_p95: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalSummary:
    aggregate: EvalMetrics
    corpora: dict[str, CorpusEvalMetrics] = field(default_factory=dict)

    def asdict(self) -> dict:
        return {
            "aggregate": asdict(self.aggregate),
            "corpora": {name: asdict(metrics) for name, metrics in self.corpora.items()},
        }


def evaluate_typing_artifact_detailed(
    artifact_path: Path,
    dataset_root: Path,
    checkpoint: str,
) -> tuple[tuple[float, float, float, float], dict[str, tuple[float, float, float]]]:
    tokenizer = BertTokenizerAdapter(checkpoint)
    model = _artifact_model(artifact_path)
    expected: list[int] = []
    predicted: list[int] = []
    span_hits = 0
    span_total = 0
    sample_texts: list[str] = []
    per_corpus_expected: dict[str, list[int]] = {}
    per_corpus_predicted: dict[str, list[int]] = {}
    per_corpus_span_hits: dict[str, int] = {}
    per_corpus_span_total: dict[str, int] = {}
    storage_path = dataset_root / "storage_cases.jsonl"
    if not storage_path.exists():
        return (0.0, 0.0, 0.0, 0.0), {}
    with storage_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            corpus = str(case.get("corpus") or _record_corpus(case["id"]) or "unknown")
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
            expected_label = _memory_type_index(case["expected_memory_type"])
            expected.append(expected_label)
            predicted.append(predicted_label)
            per_corpus_expected.setdefault(corpus, []).append(expected_label)
            per_corpus_predicted.setdefault(corpus, []).append(predicted_label)
            for span in case.get("required_spans", []):
                span_total += 1
                per_corpus_span_total[corpus] = per_corpus_span_total.get(corpus, 0) + 1
                if span.lower() in case["text"].lower():
                    span_hits += 1
                    per_corpus_span_hits[corpus] = per_corpus_span_hits.get(corpus, 0) + 1
    if not expected:
        return (0.0, 0.0, 0.0, 0.0), {}
    latency = benchmark_artifact(artifact_path, "typing", tokenizer, sample_texts)
    overall = (
        sum(1 for exp, pred in zip(expected, predicted) if exp == pred) / len(expected),
        _macro_f1(expected, predicted, num_labels=8),
        span_hits / max(span_total, 1),
        latency,
    )
    per_corpus: dict[str, tuple[float, float, float]] = {}
    for corpus, corpus_expected in per_corpus_expected.items():
        corpus_predicted = per_corpus_predicted[corpus]
        per_corpus[corpus] = (
            sum(1 for exp, pred in zip(corpus_expected, corpus_predicted) if exp == pred) / len(corpus_expected),
            _macro_f1(corpus_expected, corpus_predicted, num_labels=8),
            per_corpus_span_hits.get(corpus, 0) / max(per_corpus_span_total.get(corpus, 0), 1),
        )
    return overall, per_corpus


def evaluate_candidate(
    component: str,
    artifact_path: Path | None,
    quick_dataset_root: Path,
    full_dataset_root: Path,
    checkpoint: str,
) -> tuple[EvalSummary, EvalSummary]:
    restore_baseline_artifacts()
    try:
        quick_report = _evaluate_recall_report(component, artifact_path, quick_dataset_root)
        full_report = _evaluate_recall_report(component, artifact_path, full_dataset_root)
        quick_typing = ((0.0, 0.0, 0.0, 0.0), {})
        full_typing = ((0.0, 0.0, 0.0, 0.0), {})
        if component == "typing" and artifact_path is not None:
            quick_typing = evaluate_typing_artifact_detailed(artifact_path, quick_dataset_root, checkpoint)
            full_typing = evaluate_typing_artifact_detailed(artifact_path, full_dataset_root, checkpoint)
        quick_summary = _build_eval_summary(
            component=component,
            report=quick_report,
            dataset_root=quick_dataset_root,
            artifact_path=artifact_path,
            typing_metrics=quick_typing[0],
            typing_per_corpus=quick_typing[1],
        )
        full_summary = _build_eval_summary(
            component=component,
            report=full_report,
            dataset_root=full_dataset_root,
            artifact_path=artifact_path,
            typing_metrics=full_typing[0],
            typing_per_corpus=full_typing[1],
        )
        return quick_summary, full_summary
    finally:
        restore_baseline_artifacts()


def _evaluate_recall_report(component: str, artifact_path: Path | None, dataset_root: Path) -> dict:
    if component in {"embedding", "reranker"} and artifact_path is not None:
        install_artifact_into_upstream(component, artifact_path)
    output_path = metrics_path(component).with_name(f"{component}_{dataset_root.name}.json")
    return run_memory_eval(dataset_root=dataset_root, output_path=output_path)


def _build_eval_summary(
    component: str,
    report: dict,
    dataset_root: Path,
    artifact_path: Path | None,
    typing_metrics: tuple[float, float, float, float],
    typing_per_corpus: dict[str, tuple[float, float, float]],
) -> EvalSummary:
    artifact_size = 0.0
    if artifact_path is not None:
        from .export import artifact_size_mb

        artifact_size = artifact_size_mb(artifact_path)

    corpus_metrics = _build_corpus_metrics(
        component=component,
        report=report,
        dataset_root=dataset_root,
        typing_per_corpus=typing_per_corpus,
    )
    aggregate = _aggregate_corpus_metrics(
        component=component,
        corpus_metrics=corpus_metrics,
        artifact_size=artifact_size,
        typing_latency=typing_metrics[3],
        fallback_report=report,
    )
    return EvalSummary(aggregate=aggregate, corpora=corpus_metrics)


def _aggregate_corpus_metrics(
    component: str,
    corpus_metrics: dict[str, CorpusEvalMetrics],
    artifact_size: float,
    typing_latency: float,
    fallback_report: dict,
) -> EvalMetrics:
    weighted_storage = 0.0
    weighted_recall = 0.0
    weighted_type_accuracy = 0.0
    weighted_macro_f1 = 0.0
    weighted_span_coverage = 0.0
    weighted_hit_rate = 0.0
    weighted_recall_at_10 = 0.0
    weighted_mrr = 0.0
    weighted_ndcg = 0.0
    total_weight = 0.0
    recall_weight = 0.0
    for corpus, metrics in corpus_metrics.items():
        weight = COMPONENT_CORPUS_SCORE_WEIGHTS.get(component, {}).get(corpus, 0.0)
        if weight <= 0:
            continue
        total_weight += weight
        weighted_storage += metrics.storage_score * weight
        weighted_type_accuracy += metrics.type_accuracy * weight
        weighted_macro_f1 += metrics.macro_f1 * weight
        weighted_span_coverage += metrics.span_coverage * weight

        include_recall = component != "typing" or metrics.recall_query_count > 0
        if include_recall:
            weighted_recall += metrics.recall_score * weight
            weighted_hit_rate += metrics.hit_rate_at_10 * weight
            weighted_recall_at_10 += metrics.recall_at_10 * weight
            weighted_mrr += metrics.mrr_at_10 * weight
            weighted_ndcg += metrics.ndcg_at_10 * weight
            recall_weight += weight

    if total_weight == 0:
        hit_rate, recall, mrr, ndcg = _parse_recall_metrics(fallback_report)
        recall_score = compute_recall_score(hit_rate, recall, mrr, ndcg)
        storage = fallback_report["storage"]
        type_accuracy = float(storage["typeAccuracy"])
        macro_f1 = float(storage["macroF1"])
        span_coverage = float(storage["spanCoverage"])
        storage_score = compute_storage_score(type_accuracy, macro_f1, span_coverage)
        latency = float(fallback_report["recall"].get("latencyStats", {}).get("p95Ms", 0.0))
        return EvalMetrics(
            component=component,
            storage_score=storage_score,
            recall_score=recall_score,
            memory_score=build_memory_score(component, storage_score, recall_score),
            model_mb=artifact_size,
            latency_ms=typing_latency if component == "typing" else latency,
            type_accuracy=type_accuracy,
            macro_f1=macro_f1,
            span_coverage=span_coverage,
            hit_rate_at_10=hit_rate,
            recall_at_10=recall,
            mrr_at_10=mrr,
            ndcg_at_10=ndcg,
        )

    storage_score = weighted_storage / total_weight
    if component == "typing":
        recall_score = (weighted_recall / recall_weight) if recall_weight > 0 else 0.0
        memory_score = storage_score if recall_weight == 0 else build_memory_score(component, storage_score, recall_score)
        recall_denom = recall_weight if recall_weight > 0 else None
    else:
        recall_score = weighted_recall / total_weight
        memory_score = build_memory_score(component, storage_score, recall_score)
        recall_denom = total_weight
    total_latency = max(
        (metrics.latency_ms for metrics in corpus_metrics.values() if metrics.latency_ms > 0),
        default=0.0,
    )
    return EvalMetrics(
        component=component,
        storage_score=storage_score,
        recall_score=recall_score,
        memory_score=memory_score,
        model_mb=artifact_size,
        latency_ms=typing_latency if component == "typing" else total_latency,
        type_accuracy=weighted_type_accuracy / total_weight,
        macro_f1=weighted_macro_f1 / total_weight,
        span_coverage=weighted_span_coverage / total_weight,
        hit_rate_at_10=(weighted_hit_rate / recall_denom) if recall_denom else 0.0,
        recall_at_10=(weighted_recall_at_10 / recall_denom) if recall_denom else 0.0,
        mrr_at_10=(weighted_mrr / recall_denom) if recall_denom else 0.0,
        ndcg_at_10=(weighted_ndcg / recall_denom) if recall_denom else 0.0,
    )


def _build_corpus_metrics(
    component: str,
    report: dict,
    dataset_root: Path,
    typing_per_corpus: dict[str, tuple[float, float, float]],
) -> dict[str, CorpusEvalMetrics]:
    storage_metadata = _load_storage_case_metadata(dataset_root)
    storage_results = report["storage"]["caseResults"]
    query_results = report["recall"]["queryResults"]
    corpus_names = sorted(
        {
            corpus
            for corpus in [_record_corpus(result["id"]) for result in storage_results + query_results]
            if corpus
        }
    )
    output: dict[str, CorpusEvalMetrics] = {}
    for corpus in corpus_names:
        filtered_query_results = [result for result in query_results if _record_corpus(result["id"]) == corpus]
        storage_score, type_accuracy, macro_f1, span_coverage = _build_corpus_storage_metrics(
            component=component,
            corpus=corpus,
            storage_results=storage_results,
            storage_metadata=storage_metadata,
            typing_per_corpus=typing_per_corpus,
        )
        hit_rate, recall, mrr, ndcg = _build_recall_metrics(filtered_query_results)
        stage_latency_p95 = _build_stage_latency_p95(filtered_query_results)
        latency = stage_latency_p95.get("totalMs", _build_latency_p95(filtered_query_results))
        recall_score = compute_recall_score(hit_rate, recall, mrr, ndcg)
        memory_score = (
            storage_score
            if component == "typing" and not filtered_query_results
            else build_memory_score(component, storage_score, recall_score)
        )
        output[corpus] = CorpusEvalMetrics(
            corpus=corpus,
            storage_score=storage_score,
            recall_score=recall_score,
            memory_score=memory_score,
            latency_ms=latency,
            recall_query_count=len(filtered_query_results),
            type_accuracy=type_accuracy,
            macro_f1=macro_f1,
            span_coverage=span_coverage,
            hit_rate_at_10=hit_rate,
            recall_at_10=recall,
            mrr_at_10=mrr,
            ndcg_at_10=ndcg,
            stage_latency_p95=stage_latency_p95,
        )
    return output


def _build_corpus_storage_metrics(
    component: str,
    corpus: str,
    storage_results: list[dict],
    storage_metadata: dict[str, dict],
    typing_per_corpus: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float, float]:
    if component == "typing" and corpus in typing_per_corpus:
        type_accuracy, macro_f1, span_coverage = typing_per_corpus[corpus]
        return (
            compute_storage_score(type_accuracy, macro_f1, span_coverage),
            type_accuracy,
            macro_f1,
            span_coverage,
        )

    filtered = [result for result in storage_results if _record_corpus(result["id"]) == corpus]
    if not filtered:
        return 0.0, 0.0, 0.0, 0.0
    expected = [result["expectedType"] for result in filtered]
    predicted = [result["predictedType"] for result in filtered]
    type_accuracy = sum(1 for exp, pred in zip(expected, predicted) if exp == pred) / len(filtered)
    labels = sorted({*expected, *predicted})
    macro_f1 = _macro_f1(
        [_memory_type_index(label) for label in expected],
        [_memory_type_index(label) for label in predicted],
        num_labels=max(len(MEMORY_TYPE_TO_INDEX), len(labels)),
    )
    span_hits = 0
    span_total = 0
    for result in filtered:
        metadata = storage_metadata.get(result["id"], {})
        required_spans = int(metadata.get("required_spans", 0))
        span_total += required_spans
        span_hits += max(required_spans - len(result.get("missingSpans", [])), 0)
    span_coverage = span_hits / max(span_total, 1)
    return (
        compute_storage_score(type_accuracy, macro_f1, span_coverage),
        type_accuracy,
        macro_f1,
        span_coverage,
    )


def _build_recall_metrics(query_results: list[dict]) -> tuple[float, float, float, float]:
    if not query_results:
        return 0.0, 0.0, 0.0, 0.0
    hit_total = 0.0
    recall_total = 0.0
    mrr_total = 0.0
    ndcg_total = 0.0
    for result in query_results:
        hit_total += 1.0 if result.get("hitByK", {}).get("10") or result.get("hitByK", {}).get(10) else 0.0
        recall_total += float(result.get("recallByK", {}).get("10") or result.get("recallByK", {}).get(10) or 0.0)
        mrr_total += float(result.get("mrrByK", {}).get("10") or result.get("mrrByK", {}).get(10) or 0.0)
        ndcg_total += float(result.get("ndcgByK", {}).get("10") or result.get("ndcgByK", {}).get(10) or 0.0)
    count = len(query_results)
    return (
        hit_total / count,
        recall_total / count,
        mrr_total / count,
        ndcg_total / count,
    )


def _build_latency_p95(query_results: list[dict]) -> float:
    latencies = [float(result["latencyMs"]) for result in query_results if result.get("latencyMs") is not None]
    return _p95(latencies)


def _build_stage_latency_p95(query_results: list[dict]) -> dict[str, float]:
    stages = (
        "analysisMs",
        "expansionMs",
        "queryEmbeddingMs",
        "semanticSearchMs",
        "lexicalSearchMs",
        "fusionMs",
        "rerankMs",
        "totalMs",
    )
    output: dict[str, float] = {}
    for stage in stages:
        values = [
            float(result["stageTimings"][stage])
            for result in query_results
            if isinstance(result.get("stageTimings"), dict)
            and result["stageTimings"].get(stage) is not None
        ]
        if values:
            output[stage] = _p95(values)
    return output


def _load_storage_case_metadata(dataset_root: Path) -> dict[str, dict]:
    storage_path = dataset_root / "storage_cases.jsonl"
    if not storage_path.exists():
        return {}
    metadata: dict[str, dict] = {}
    with storage_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata[record["id"]] = {
                "required_spans": len(record.get("required_spans", [])),
                "corpus": record.get("corpus") or _record_corpus(record["id"]),
            }
    return metadata


def _record_corpus(record_id: str) -> str | None:
    if ":" not in record_id:
        return None
    return record_id.split(":", 1)[0]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return float(np.percentile(np.array(values, dtype=np.float32), 95))
