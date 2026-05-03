#!/usr/bin/env python3
"""Classify retrieval-diagnostics misses and optional pre/post regressions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if isinstance(row_id, str):
                rows[row_id] = row
    return rows


def by_k(mapping: dict[str, Any] | None, k: int, default: Any = None) -> Any:
    if not mapping:
        return default
    return mapping.get(str(k), mapping.get(k, default))


def metric_for_k(report: dict[str, Any], k: int | None = None) -> tuple[int, dict[str, Any]]:
    metrics = report.get("metricsByK") or []
    if not metrics:
        return k or 10, {}
    if k is None:
        selected = max(metrics, key=lambda item: item.get("k", 0))
    else:
        selected = next((item for item in metrics if item.get("k") == k), metrics[-1])
    return int(selected.get("k", k or 10)), selected


def pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def compact(text: str | None, limit: int = 160) -> str:
    value = " ".join((text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def first_text_line(document: dict[str, Any] | None) -> str:
    if not document:
        return ""
    for key in ("text", "content", "body"):
        text = document.get(key)
        if not isinstance(text, str):
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
    return ""


def document_hint(document: dict[str, Any] | None) -> str:
    if not document:
        return ""
    path = document.get("relative_path") or document.get("path") or ""
    first = first_text_line(document)
    if path and first:
        return f"{path}: {compact(first, 130)}"
    return str(path) or compact(first, 130)


TEMPORAL_TERMS = [
    "when",
    "what time",
    "how long",
    "days ago",
    "weeks ago",
    "months ago",
    "years ago",
    "past ",
    "last ",
    "next ",
    "since",
    "before",
    "after",
    "earliest",
    "latest",
    "recently",
    "current",
    "today",
    "yesterday",
    "tomorrow",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "monday",
]


MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)


def classify_query_shape(query: str, relevant_count: int, query_row: dict[str, Any]) -> list[str]:
    lower = query.lower()
    labels: list[str] = []

    if "how many" in lower or "number of" in lower or "count" in lower:
        labels.append("count-query")
    if "order of" in lower or "earliest" in lower or "latest" in lower or "first to last" in lower:
        labels.append("order-query")
    if "recommend" in lower or "suggestion" in lower or "suggestions" in lower or "tips" in lower:
        labels.append("recommendation-context")
    if lower.startswith(("which ", "who ", "whose ", "where ", "at which ")) or "which " in lower:
        labels.append("direct-lookup")
    if any(term in lower for term in TEMPORAL_TERMS) or MONTH_RE.search(query):
        labels.append("temporal/session")
    if relevant_count > 1 or "all " in lower or "different " in lower or " and " in lower or "total" in lower:
        labels.append("multi-evidence")
    if any(term in lower for term in [" it ", " that ", " this ", " they ", " them ", " those "]):
        labels.append("contextual-ellipsis")

    memory_types = query_row.get("memory_types") or []
    for memory_type in memory_types:
        if memory_type in {"temporal", "episodic", "preference", "profile", "commitment", "decision"}:
            typed = f"type:{memory_type}"
            if typed not in labels:
                labels.append(typed)

    if not labels:
        labels.append("lexical-semantic")
    return labels


def classify_surface(row: dict[str, Any], max_k: int) -> str:
    supplied = row.get("diagnosticSurface")
    if isinstance(supplied, str) and supplied:
        return supplied

    retrieved = row.get("retrievedDocumentIds") or []
    candidate_recall = float(row.get("candidateRecall") or 0)
    hit = bool(by_k(row.get("hitByK"), max_k, False))
    recall = float(by_k(row.get("recallByK"), max_k, 0) or 0)
    relevant_count = len(row.get("relevantDocumentIds") or [])
    first_rank = row.get("firstRelevantRank")
    first_candidate_rank = row.get("firstCandidateRelevantRank")

    if not retrieved:
        return "empty-retrieval"
    if not hit and candidate_recall <= 0 and first_candidate_rank is None:
        return "candidate_generation_miss"
    if not hit and (candidate_recall > 0 or first_candidate_rank is not None):
        return "ranking_or_packing_miss"
    if hit and relevant_count > 1 and recall < 1:
        return "partial_multi_evidence"
    if hit and isinstance(first_rank, int) and first_rank > 1:
        return "rank_headroom"
    return "covered"


def result_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row.get("id"): row for row in report.get("queryResults", []) if isinstance(row.get("id"), str)}


def summarize_rows(rows: list[dict[str, Any]], max_k: int, queries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in rows:
        query_id = row.get("id")
        query_row = queries.get(query_id, {}) if isinstance(query_id, str) else {}
        relevant = row.get("relevantDocumentIds") or []
        hit = bool(by_k(row.get("hitByK"), max_k, False))
        recall = float(by_k(row.get("recallByK"), max_k, 0) or 0)
        summaries.append(
            {
                "id": query_id,
                "query": row.get("query") or query_row.get("query") or "",
                "hit": hit,
                "recall": recall,
                "mrr": float(by_k(row.get("mrrByK"), max_k, 0) or 0),
                "surface": classify_surface(row, max_k),
                "taxonomy": row.get("queryShape")
                or classify_query_shape(row.get("query") or query_row.get("query") or "", len(relevant), query_row),
                "firstRelevantRank": row.get("firstRelevantRank"),
                "firstCandidateRelevantRank": row.get("firstCandidateRelevantRank"),
                "candidateRecall": float(row.get("candidateRecall") or 0),
                "relevantDocumentIds": relevant,
                "retrievedDocumentIds": row.get("retrievedDocumentIds") or [],
                "candidateDocumentIds": row.get("candidateDocumentIds") or [],
                "latencyMs": row.get("latencyMs"),
            }
        )
    return summaries


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return lines


def top_counter_rows(counter: Counter[str]) -> list[list[Any]]:
    return [[key, value] for key, value in counter.most_common()]


def write_reports(
    run_path: Path,
    report: dict[str, Any],
    summaries: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    baseline_path: Path | None,
    baseline_report: dict[str, Any] | None,
    baseline_summaries: dict[str, dict[str, Any]],
    output: Path,
    output_json: Path,
    max_k: int,
    detail_limit: int,
) -> None:
    max_k, metric = metric_for_k(report, max_k)
    misses = [row for row in summaries if not row["hit"]]
    candidate_only = [row for row in misses if row["surface"] in {"ranking_or_packing_miss", "candidate-only-miss"}]
    candidate_generation = [row for row in misses if row["surface"] in {"candidate_generation_miss", "candidate-generation-miss"}]
    partials = [row for row in summaries if row["surface"] in {"partial_multi_evidence", "partial-multi-evidence"}]
    rank_headroom = [row for row in summaries if row["surface"] in {"rank_headroom", "rank-headroom"}]

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    if baseline_report is not None:
        for row in summaries:
            old = baseline_summaries.get(str(row["id"]))
            if not old:
                continue
            if old["hit"] and not row["hit"]:
                regressions.append({**row, "baselineSurface": old["surface"], "baselineFirstRelevantRank": old["firstRelevantRank"]})
            if row["hit"] and not old["hit"]:
                improvements.append({**row, "baselineSurface": old["surface"], "baselineFirstRelevantRank": old["firstRelevantRank"]})

    surface_counts = Counter(row["surface"] for row in summaries)
    miss_taxonomy = Counter(label for row in misses for label in row["taxonomy"])
    candidate_only_taxonomy = Counter(label for row in candidate_only for label in row["taxonomy"])
    regression_taxonomy = Counter(label for row in regressions for label in row["taxonomy"])

    latency_by_surface: dict[str, list[float]] = defaultdict(list)
    for row in summaries:
        latency = row.get("latencyMs")
        if isinstance(latency, (int, float)):
            latency_by_surface[row["surface"]].append(float(latency))

    lines: list[str] = []
    lines.append("# Retrieval Diagnostics Miss Report")
    lines.append("")
    lines.append(f"- Candidate run: `{run_path}`")
    if baseline_path:
        lines.append(f"- Baseline run: `{baseline_path}`")
    lines.append(f"- Dataset root: `{report.get('datasetRoot', 'unknown')}`")
    lines.append(f"- Profile: `{report.get('profile', 'unknown')}`")
    lines.append(f"- Max K: `{max_k}`")
    lines.append(
        "- Metrics: "
        f"Hit@{max_k} {pct(metric.get('hitRate'))}, "
        f"Recall@{max_k} {pct(metric.get('recall'))}, "
        f"MRR@{max_k} {fmt(metric.get('mrr'))}, "
        f"nDCG@{max_k} {fmt(metric.get('ndcg'))}"
    )
    lines.append(
        "- Candidate pool: "
        f"Hit {pct(report.get('candidatePoolHitRate'))}, "
        f"Recall {pct(report.get('candidatePoolRecall'))}, "
        f"candidate-generation miss {pct(report.get('candidateGenerationMissRate'))}, "
        f"candidate-only miss {pct(report.get('candidateOnlyMissRate'))}"
    )
    lines.append("")

    summary_rows = [
        ["misses", len(misses)],
        ["candidate-generation misses", len(candidate_generation)],
        ["candidate-only misses", len(candidate_only)],
        ["partial multi-evidence hits", len(partials)],
        ["rank-headroom hits", len(rank_headroom)],
    ]
    if baseline_report is not None:
        summary_rows.extend([["hit regressions", len(regressions)], ["hit improvements", len(improvements)]])
    lines.extend(table(["Bucket", "Count"], summary_rows))
    lines.append("")

    lines.append("## Failure Surfaces")
    surface_rows = []
    for surface, count in surface_counts.most_common():
        latencies = latency_by_surface.get(surface, [])
        surface_rows.append([surface, count, f"{mean(latencies):.0f}ms" if latencies else "n/a"])
    lines.extend(table(["Surface", "Count", "Avg latency"], surface_rows))
    lines.append("")

    if miss_taxonomy:
        lines.append("## Miss Taxonomy")
        lines.extend(table(["Taxonomy", "Count"], top_counter_rows(miss_taxonomy)))
        lines.append("")

    if candidate_only_taxonomy:
        lines.append("## Candidate-Only Miss Taxonomy")
        lines.extend(table(["Taxonomy", "Count"], top_counter_rows(candidate_only_taxonomy)))
        lines.append("")

    if regression_taxonomy:
        lines.append("## Regression Taxonomy")
        lines.extend(table(["Taxonomy", "Count"], top_counter_rows(regression_taxonomy)))
        lines.append("")

    def detail_section(title: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        lines.append(f"## {title}")
        for row in rows[:detail_limit]:
            relevant = []
            for doc_id in row["relevantDocumentIds"]:
                hint = document_hint(documents.get(doc_id))
                relevant.append(f"`{doc_id}`" + (f" - {hint}" if hint else ""))
            lines.append("")
            lines.append(f"### {row['id']}")
            lines.append(f"- Query: {row['query']}")
            lines.append(f"- Surface: {row['surface']}")
            lines.append(f"- Taxonomy: {', '.join(row['taxonomy'])}")
            if "baselineSurface" in row:
                lines.append(
                    f"- Baseline: {row['baselineSurface']}, firstRelevantRank={row.get('baselineFirstRelevantRank')}"
                )
            lines.append(
                f"- Candidate ranks: firstRelevantRank={row.get('firstRelevantRank')}, "
                f"firstCandidateRelevantRank={row.get('firstCandidateRelevantRank')}, "
                f"candidateRecall={row.get('candidateRecall'):.3f}"
            )
            lines.append(f"- Relevant: {'; '.join(relevant) if relevant else 'n/a'}")
            lines.append(
                f"- Retrieved top-{max_k}: "
                f"{', '.join('`' + item + '`' for item in row['retrievedDocumentIds']) or 'none'}"
            )
        if len(rows) > detail_limit:
            lines.append("")
            lines.append(f"_Omitted {len(rows) - detail_limit} additional rows; rerun with `--limit {len(rows)}` to include all._")
        lines.append("")

    detail_section("Hit Regressions", regressions)
    detail_section("Candidate-Only Misses", candidate_only)
    detail_section("Candidate-Generation Misses", candidate_generation)
    detail_section("Partial Multi-Evidence Hits", partials)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    payload = {
        "run": str(run_path),
        "baseline": str(baseline_path) if baseline_path else None,
        "maxK": max_k,
        "metrics": metric,
        "candidatePoolHitRate": report.get("candidatePoolHitRate"),
        "candidatePoolRecall": report.get("candidatePoolRecall"),
        "candidateGenerationMissRate": report.get("candidateGenerationMissRate"),
        "candidateOnlyMissRate": report.get("candidateOnlyMissRate"),
        "counts": {
            "misses": len(misses),
            "candidateGenerationMisses": len(candidate_generation),
            "candidateOnlyMisses": len(candidate_only),
            "partialMultiEvidenceHits": len(partials),
            "rankHeadroomHits": len(rank_headroom),
            "hitRegressions": len(regressions),
            "hitImprovements": len(improvements),
        },
        "surfaceCounts": dict(surface_counts),
        "missTaxonomy": dict(miss_taxonomy),
        "candidateOnlyTaxonomy": dict(candidate_only_taxonomy),
        "regressionTaxonomy": dict(regression_taxonomy),
        "hitRegressions": regressions,
        "candidateOnlyMisses": candidate_only,
        "candidateGenerationMisses": candidate_generation,
        "partialMultiEvidenceHits": partials,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json", type=Path, help="Retrieval diagnostics JSON to analyze.")
    parser.add_argument("--baseline-json", type=Path, help="Optional earlier retrieval diagnostics JSON.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Dataset root containing recall_documents.jsonl and recall_queries.jsonl. Defaults to report datasetRoot.",
    )
    parser.add_argument("--k", type=int, help="K to analyze. Defaults to largest K in the run.")
    parser.add_argument("--output", type=Path, help="Markdown output path. Defaults beside run JSON.")
    parser.add_argument("--output-json", type=Path, help="JSON output path. Defaults beside run JSON.")
    parser.add_argument("--limit", type=int, default=40, help="Maximum rows per detail section.")
    args = parser.parse_args()

    report = load_json(args.run_json)
    max_k, _ = metric_for_k(report, args.k)
    dataset_root = args.dataset_root or Path(str(report.get("datasetRoot") or args.run_json.parent.parent))
    documents = load_jsonl(dataset_root / "recall_documents.jsonl")
    queries = load_jsonl(dataset_root / "recall_queries.jsonl")
    summaries = summarize_rows(report.get("queryResults") or [], max_k, queries)

    baseline_report = load_json(args.baseline_json) if args.baseline_json else None
    baseline_summaries: dict[str, dict[str, Any]] = {}
    if baseline_report is not None:
        baseline_dataset_root = Path(str(baseline_report.get("datasetRoot") or dataset_root))
        baseline_queries = load_jsonl(baseline_dataset_root / "recall_queries.jsonl") or queries
        baseline_k, _ = metric_for_k(baseline_report, max_k)
        baseline_summaries = {
            str(row["id"]): row
            for row in summarize_rows(baseline_report.get("queryResults") or [], baseline_k, baseline_queries)
        }

    output = args.output or args.run_json.with_name(args.run_json.stem + ".miss-report.md")
    output_json = args.output_json or args.run_json.with_name(args.run_json.stem + ".miss-report.json")
    write_reports(
        args.run_json,
        report,
        summaries,
        documents,
        args.baseline_json,
        baseline_report,
        baseline_summaries,
        output,
        output_json,
        max_k,
        max(1, args.limit),
    )
    print(f"Markdown report: {output}")
    print(f"JSON report: {output_json}")


if __name__ == "__main__":
    main()
