#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def by_k(mapping, k, default=None):
    if not mapping:
        return default
    return mapping.get(str(k), mapping.get(k, default))


def percent(value):
    return f"{value * 100:.2f}%"


def fmt_ms(value):
    if value is None:
        return "n/a"
    return f"{value:.0f}ms"


def compact_text(text, limit=180):
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def document_hint(document):
    if not document:
        return ""
    path = document.get("relative_path") or document.get("path") or ""
    text = document.get("text") or ""
    first_content = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            first_content = stripped
            break
    if path and first_content:
        return f"{path}: {compact_text(first_content, 140)}"
    return path or compact_text(first_content, 140)


TEMPORAL_PATTERNS = [
    r"\bhow many\b",
    r"\bwhat time\b",
    r"\bwhen\b",
    r"\bwhich day\b",
    r"\bdate\b",
    r"\bdays?\b",
    r"\bweeks?\b",
    r"\bmonths?\b",
    r"\byears?\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bsince\b",
    r"\bduring\b",
    r"\bpast\b",
    r"\blast\b",
    r"\bthis year\b",
    r"\btypical week\b",
    r"\bJanuary|February|March|April|May|June|July|August|September|October|November|December\b",
]

MULTI_EVIDENCE_PATTERNS = [
    r"\bacross\b",
    r"\ball the\b",
    r"\bbetween\b",
    r"\bfrom\b.+\bto\b",
    r"\btotal\b",
    r"\bcombined\b",
    r"\bcompare\b",
    r"\band\b.+\band\b",
]

CONTEXTUAL_PATTERNS = [
    r"\bit\b",
    r"\bthat\b",
    r"\bthis\b",
    r"\bthere\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\bthose\b",
    r"\bthe one\b",
]


def classify_taxonomy(query, relevant_count, memory_types):
    lower = query.lower()
    labels = []

    if "temporal" in memory_types or any(re.search(pattern, query, re.IGNORECASE) for pattern in TEMPORAL_PATTERNS):
        labels.append("temporal/count")

    if relevant_count > 1 or any(re.search(pattern, lower) for pattern in MULTI_EVIDENCE_PATTERNS):
        labels.append("multi-evidence")

    if any(re.search(pattern, lower) for pattern in CONTEXTUAL_PATTERNS):
        labels.append("contextual-ellipsis")

    quoted = re.search(r"'[^']+'|\"[^\"]+\"", query) is not None
    proper_tokens = [
        token
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", query)
        if token not in {"How", "What", "When", "Where", "Which", "Who", "Why", "Did", "Can", "The"}
    ]
    if quoted or proper_tokens or "entity" in memory_types:
        labels.append("entity/alias")

    if "episodic" in memory_types and "temporal/count" not in labels:
        labels.append("episodic/contextual")

    if not labels:
        labels.append("lexical/semantic-mismatch")

    return labels


def classify_surface(result, max_k):
    retrieved = result.get("retrievedDocumentIds") or []
    relevant = set(result.get("relevantDocumentIds") or [])
    hit = by_k(result.get("hitByK"), max_k, False)
    recall = by_k(result.get("recallByK"), max_k, 0) or 0
    hit_at_1 = by_k(result.get("hitByK"), 1, False)

    if not retrieved:
        return "empty-retrieval"
    if not hit:
        return "no-relevant-in-top-k"
    if recall < 1 and len(relevant) > 1:
        return "partial-multi-evidence-recall"
    if hit and not hit_at_1:
        return "rank-headroom"
    return "covered"


def average(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def metric_at_max_k(report):
    metrics = report.get("recall", {}).get("metricsByK") or []
    if not metrics:
        return 10, {}
    best = max(metrics, key=lambda item: item.get("k", 0))
    return best.get("k", 10), best


def render_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    return lines


def build_report(run_path, dataset_root, limit):
    report = load_json(run_path)
    documents = load_jsonl(dataset_root / "recall_documents.jsonl")
    queries = load_jsonl(dataset_root / "recall_queries.jsonl")

    max_k, max_metric = metric_at_max_k(report)
    results = report.get("recall", {}).get("queryResults") or []

    enriched = []
    for result in results:
        query_row = queries.get(result.get("id"), {})
        memory_types = query_row.get("memory_types") or []
        relevant = result.get("relevantDocumentIds") or []
        retrieved = result.get("retrievedDocumentIds") or []
        labels = classify_taxonomy(result.get("query", ""), len(relevant), memory_types)
        surface = classify_surface(result, max_k)
        enriched.append(
            {
                "result": result,
                "query_row": query_row,
                "memory_types": memory_types,
                "taxonomy": labels,
                "surface": surface,
                "relevant": relevant,
                "retrieved": retrieved,
                "hit": by_k(result.get("hitByK"), max_k, False),
                "recall": by_k(result.get("recallByK"), max_k, 0) or 0,
                "mrr": by_k(result.get("mrrByK"), max_k, 0) or 0,
            }
        )

    misses = [row for row in enriched if not row["hit"]]
    partials = [row for row in enriched if row["hit"] and row["recall"] < 1]
    rank_headroom = [row for row in enriched if row["surface"] == "rank-headroom"]

    surface_counts = Counter(row["surface"] for row in enriched)
    miss_taxonomy_counts = Counter(label for row in misses for label in row["taxonomy"])
    partial_taxonomy_counts = Counter(label for row in partials for label in row["taxonomy"])
    difficulty_counts = Counter((row["result"].get("difficulty") or "unknown") for row in misses)
    type_counts = Counter(memory_type for row in misses for memory_type in row["memory_types"])

    latency_by_surface = defaultdict(list)
    fusion_by_surface = defaultdict(list)
    for row in enriched:
        result = row["result"]
        latency_by_surface[row["surface"]].append(result.get("latencyMs"))
        timings = result.get("stageTimings") or {}
        fusion_by_surface[row["surface"]].append(timings.get("fusionMs"))

    lines = []
    lines.append("# LongMemEval Miss Analysis")
    lines.append("")
    lines.append(f"- Run: `{run_path}`")
    lines.append(f"- Dataset root: `{dataset_root}`")
    lines.append(f"- Profile: `{report.get('profile', 'unknown')}`")
    lines.append(f"- Max K: `{max_k}`")
    lines.append(
        "- Metrics: "
        f"Hit@{max_k} {percent(max_metric.get('hitRate', 0))}, "
        f"Recall@{max_k} {percent(max_metric.get('recall', 0))}, "
        f"MRR@{max_k} {max_metric.get('mrr', 0):.4f}, "
        f"nDCG@{max_k} {max_metric.get('ndcg', 0):.4f}"
    )
    lines.append(
        f"- Misses: {len(misses)}/{len(enriched)}; "
        f"partial multi-evidence hits: {len(partials)}; rank-headroom hits: {len(rank_headroom)}"
    )
    lines.append("")

    lines.append("## Failure Surfaces")
    surface_rows = []
    for surface, count in surface_counts.most_common():
        surface_rows.append(
            [
                surface,
                count,
                fmt_ms(average(latency_by_surface[surface])),
                fmt_ms(average(fusion_by_surface[surface])),
            ]
        )
    lines.extend(render_table(["Surface", "Count", "Avg latency", "Avg fusion"], surface_rows))
    lines.append("")

    lines.append("## Miss Taxonomy")
    lines.extend(render_table(["Taxonomy", "Miss count"], miss_taxonomy_counts.most_common()))
    lines.append("")

    if partials:
        lines.append("## Partial Recall Taxonomy")
        lines.extend(render_table(["Taxonomy", "Partial count"], partial_taxonomy_counts.most_common()))
        lines.append("")

    lines.append("## Misses By Difficulty")
    lines.extend(render_table(["Difficulty", "Miss count"], difficulty_counts.most_common()))
    lines.append("")

    if type_counts:
        lines.append("## Misses By Memory Type")
        lines.extend(render_table(["Memory type", "Miss count"], type_counts.most_common()))
        lines.append("")

    lines.append(f"## Hit@{max_k} Miss Details")
    for row in misses[:limit]:
        result = row["result"]
        relevant_hints = []
        for doc_id in row["relevant"]:
            hint = document_hint(documents.get(doc_id))
            relevant_hints.append(f"`{doc_id}`" + (f" - {hint}" if hint else ""))

        lines.append("")
        lines.append(f"### {result.get('id')}")
        lines.append(f"- Query: {result.get('query')}")
        lines.append(f"- Difficulty: {result.get('difficulty', 'unknown')}")
        lines.append(f"- Memory types: {', '.join(row['memory_types']) if row['memory_types'] else 'n/a'}")
        lines.append(f"- Surface: {row['surface']}")
        lines.append(f"- Taxonomy: {', '.join(row['taxonomy'])}")
        lines.append(f"- Latency: {fmt_ms(result.get('latencyMs'))}")
        lines.append(f"- Relevant: {'; '.join(relevant_hints)}")
        lines.append(f"- Retrieved top-{max_k}: {', '.join('`' + item + '`' for item in row['retrieved']) or 'none'}")

    if len(misses) > limit:
        lines.append("")
        lines.append(f"_Omitted {len(misses) - limit} additional misses; rerun with `--limit {len(misses)}` to include all._")

    if partials:
        lines.append("")
        lines.append(f"## Partial Recall Details")
        for row in partials[: min(limit, 20)]:
            result = row["result"]
            lines.append("")
            lines.append(f"### {result.get('id')}")
            lines.append(f"- Query: {result.get('query')}")
            lines.append(f"- Recall@{max_k}: {row['recall']:.3f}")
            lines.append(f"- Taxonomy: {', '.join(row['taxonomy'])}")
            lines.append(f"- Relevant: {', '.join('`' + item + '`' for item in row['relevant'])}")
            lines.append(f"- Retrieved top-{max_k}: {', '.join('`' + item + '`' for item in row['retrieved'])}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize LongMemEval misses from a memory_eval run JSON.")
    parser.add_argument("run_json", type=Path, help="Path to a longmemeval_v2 run JSON.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Evals/longmemeval_v2"),
        help="Dataset root containing recall_documents.jsonl and recall_queries.jsonl.",
    )
    parser.add_argument("--output", type=Path, help="Markdown report path. Defaults beside the run JSON.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum hit@K miss details to include.")
    args = parser.parse_args()

    run_path = args.run_json
    output = args.output or run_path.with_name(run_path.stem + "-miss-analysis.md")
    markdown = build_report(run_path, args.dataset_root, max(1, args.limit))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
