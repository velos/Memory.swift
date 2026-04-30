#!/usr/bin/env python3
"""Analyze AMB LifeBench retrieval diagnostics and build focused eval slices."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from eval_data_codex_support import normalize_spaces, write_jsonl_atomic, write_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOCUS_CATEGORIES = {"multi-hop", "temporal-updating", "nondeclarative"}


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def relative_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return parsed


def unique(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for raw in values:
        value = str(raw)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def category(result: Dict[str, Any]) -> str:
    meta = result.get("meta") or {}
    return str(meta.get("category") or "unknown")


def hit_at(result: Dict[str, Any], k: int) -> bool:
    return bool((result.get("hit_at") or {}).get(str(k)))


def recall_at(result: Dict[str, Any], k: int) -> float:
    value = (result.get("gold_recall_at") or {}).get(str(k), 0)
    return float(value or 0)


def normalized_retrieved_ids(result: Dict[str, Any]) -> List[str]:
    values = result.get("retrieved_ids") or []
    if values:
        return [str(value) for value in values]
    retrieved = result.get("retrieved") or []
    return [str(item.get("normalized_id") or item.get("id")) for item in retrieved if isinstance(item, dict)]


def month_day_count(query: str) -> int:
    month = (
        r"january|february|march|april|may|june|july|august|"
        r"september|october|november|december"
    )
    matches = re.findall(rf"\b(?:{month})\s+\d{{1,2}}(?:st|nd|rd|th)?\b", query, flags=re.I)
    return len(matches)


def taxonomy_tags(result: Dict[str, Any], k: int) -> List[str]:
    query = str(result.get("query") or "")
    lower = query.lower()
    gold_count = len(result.get("gold_ids") or [])
    tags: Set[str] = set()

    cat = category(result)
    if cat in DEFAULT_FOCUS_CATEGORIES:
        tags.add(cat)
    if gold_count >= 3:
        tags.add("multi-evidence")
    if month_day_count(query) >= 2 or re.search(r"\bfrom\s+\w+\s+\d{1,2}(?:st|nd|rd|th)?\s+to\s+\d{1,2}", lower):
        tags.add("multi-date")
    if re.search(r"\b(19|20)\d{2}\b", lower) or month_day_count(query) > 0:
        tags.add("time-anchored")
    if any(phrase in lower for phrase in [
        "how many", "how long", "in total", "total", "count", "number of",
    ]):
        tags.add("count")
    if any(phrase in lower for phrase in [
        "list all", "please list", "all activities", "which occasions",
        "specific occasions", "what preparations", "what adjustments",
        "what methods", "which instances", "in which occasions",
        "provide a brief description", "key progress", "related preparations",
    ]):
        tags.add("list-or-summary")
    if re.search(r"\bA:\s+.*\bB:\s+", query):
        tags.add("multiple-choice")
    if (result.get("gold_ids") or []) and not hit_at(result, k):
        tags.add("miss")
    elif (result.get("gold_ids") or []) and recall_at(result, k) < 1:
        tags.add("partial")
    if category(result) == "unanswerable" or not result.get("gold_ids"):
        tags.add("no-gold")

    return sorted(tags)


def compare_runs(memory_report: Dict[str, Any], baseline_report: Dict[str, Any] | None, k: int) -> Dict[str, Any]:
    memory_results = {str(result["query_id"]): result for result in memory_report.get("results") or []}
    baseline_results = {
        str(result["query_id"]): result
        for result in (baseline_report or {}).get("results") or []
    }

    overlap = Counter()
    by_category: Dict[str, Counter] = defaultdict(Counter)
    taxonomy = Counter()
    recall_delta_by_category: Dict[str, float] = defaultdict(float)
    target_cases: List[Dict[str, Any]] = []

    for query_id, result in sorted(memory_results.items()):
        gold_ids = [str(value) for value in result.get("gold_ids") or []]
        if not gold_ids:
            continue

        base = baseline_results.get(query_id)
        memory_hit = hit_at(result, k)
        baseline_hit = hit_at(base, k) if base else False
        if base:
            if memory_hit and baseline_hit:
                bucket = "both_hit"
            elif memory_hit:
                bucket = "memory_only"
            elif baseline_hit:
                bucket = "baseline_only"
            else:
                bucket = "both_miss"
            overlap[bucket] += 1
            by_category[category(result)][bucket] += 1
            recall_delta_by_category[category(result)] += recall_at(result, k) - recall_at(base, k)

        tags = taxonomy_tags(result, k)
        taxonomy.update(tags)
        is_target_category = category(result) in DEFAULT_FOCUS_CATEGORIES
        is_target = (
            len(gold_ids) >= 3
            and is_target_category
            and (not memory_hit or recall_at(result, k) < 0.999)
        )
        if is_target:
            target_cases.append({
                "query_id": query_id,
                "category": category(result),
                "taxonomy": tags,
                "query": result.get("query"),
                "gold_ids": gold_ids,
                "gold_count": len(gold_ids),
                "memory_hit_at_k": memory_hit,
                "memory_recall_at_k": recall_at(result, k),
                "memory_first_gold_rank": result.get("first_gold_rank"),
                "baseline_hit_at_k": baseline_hit if base else None,
                "baseline_recall_at_k": recall_at(base, k) if base else None,
                "baseline_first_gold_rank": base.get("first_gold_rank") if base else None,
                "memory_retrieved_ids": normalized_retrieved_ids(result),
                "baseline_retrieved_ids": normalized_retrieved_ids(base) if base else [],
            })

    target_cases.sort(key=lambda item: (
        item["memory_hit_at_k"],
        item["memory_recall_at_k"],
        -item["gold_count"],
        item["category"],
        item["query_id"],
    ))

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": memory_report.get("dataset"),
        "split": memory_report.get("split"),
        "memory_run": memory_report.get("run_name"),
        "baseline_run": (baseline_report or {}).get("run_name"),
        "k": k,
        "metrics": memory_report.get("metrics") or {},
        "baseline_metrics": (baseline_report or {}).get("metrics") or {},
        "overlap": dict(sorted(overlap.items())),
        "category_overlap": {key: dict(value) for key, value in sorted(by_category.items())},
        "recall_delta_by_category": {
            key: round(value, 6) for key, value in sorted(recall_delta_by_category.items())
        },
        "taxonomy_counts": dict(sorted(taxonomy.items())),
        "target_cases": target_cases,
    }


def markdown_summary(analysis: Dict[str, Any], memory_path: Path, baseline_path: Path | None) -> str:
    metrics = analysis.get("metrics") or {}
    baseline_metrics = analysis.get("baseline_metrics") or {}
    k = analysis["k"]

    def metric(name: str, source: Dict[str, Any]) -> str:
        value = source.get(name)
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value if value is not None else "-")

    lines = [
        f"# LifeBench Retrieval Analysis",
        "",
        f"- Memory run: `{relative_label(memory_path)}`",
        f"- Baseline run: `{relative_label(baseline_path)}`" if baseline_path else "- Baseline run: none",
        f"- k: `{k}`",
        "",
        "## Metrics",
        "",
        "| Metric | Memory | Baseline |",
        "| --- | ---: | ---: |",
    ]
    for name in [
        f"hit_at_{k}",
        f"with_gold_hit_at_{k}",
        f"gold_recall_at_{k}",
        f"with_gold_gold_recall_at_{k}",
        f"mrr_at_{k}",
        "avg_context_tokens",
    ]:
        lines.append(f"| `{name}` | {metric(name, metrics)} | {metric(name, baseline_metrics)} |")

    lines.extend(["", "## Overlap", "", "| Bucket | Count |", "| --- | ---: |"])
    for key, value in analysis.get("overlap", {}).items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Target Cases", "", "| Query | Category | Gold | Recall | Tags | Question |", "| --- | --- | ---: | ---: | --- | --- |"])
    for case in analysis.get("target_cases", [])[:40]:
        query = normalize_spaces(str(case.get("query") or ""))
        if len(query) > 140:
            query = query[:137] + "..."
        lines.append(
            f"| `{case['query_id']}` | {case['category']} | {case['gold_count']} | "
            f"{case['memory_recall_at_k']:.3f} | {', '.join(case['taxonomy'])} | {query} |"
        )

    return "\n".join(lines) + "\n"


def session_keys(conversation: Dict[str, Any]) -> List[str]:
    return sorted(
        (
            key for key, value in conversation.items()
            if re.match(r"^session_\d+$", key) and isinstance(value, list)
        ),
        key=lambda key: int(key.split("_", 1)[1]),
    )


def date_stripped_id(dia_id: str) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}_", "", dia_id)


def build_lifebench_records(data_path: Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Set[str]]]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    docs_by_id: Dict[str, Dict[str, Any]] = {}
    queries_by_id: Dict[str, Dict[str, Any]] = {}
    docs_by_user: Dict[str, Set[str]] = defaultdict(set)

    category_names = {
        "0": "information-extraction",
        "1": "multi-hop",
        "2": "temporal-updating",
        "3": "nondeclarative",
        "4": "unanswerable",
    }

    for item in data:
        sample_id = str(item["sample_id"])
        conversation = item["conversation"]
        keys = session_keys(conversation)
        dia_to_session: Dict[str, str] = {}
        for key in keys:
            for turn in conversation.get(key, []):
                if isinstance(turn, dict) and "dia_id" in turn:
                    dia_id = str(turn["dia_id"])
                    dia_to_session[dia_id] = key
                    dia_to_session[date_stripped_id(dia_id)] = key

            doc_id = f"{sample_id}_{key}"
            docs_by_user[sample_id].add(doc_id)
            docs_by_id[doc_id] = {
                "id": doc_id,
                "relative_path": f"lifebench/{sample_id}/{key}.json",
                "kind": "json",
                "text": json.dumps(conversation.get(key, []), ensure_ascii=False),
                "memory_type": "episodic",
            }

        for index, qa in enumerate(item.get("qa", [])):
            evidence = [str(value) for value in qa.get("evidence") or []]
            gold_session_keys = unique(
                dia_to_session[evidence_id]
                for evidence_id in evidence
                if evidence_id in dia_to_session
            )
            gold_ids = [f"{sample_id}_{key}" for key in gold_session_keys]
            query_id = f"{sample_id}_q{index}"
            queries_by_id[query_id] = {
                "id": query_id,
                "query": str(qa.get("question") or ""),
                "relevant_document_ids": gold_ids,
                "source_category": category_names.get(str(qa.get("category", "")), str(qa.get("category", ""))),
                "source_answer": qa.get("answer"),
                "source_evidence_ids": evidence,
            }

    return docs_by_id, queries_by_id, docs_by_user


def write_eval_slice(
    analysis: Dict[str, Any],
    *,
    lifebench_data: Path,
    output_root: Path,
    max_cases: int,
    document_scope: str,
    slice_user: str | None,
) -> None:
    docs_by_id, queries_by_id, docs_by_user = build_lifebench_records(lifebench_data)
    target_cases = list(analysis.get("target_cases", []))
    if slice_user:
        target_cases = [
            case for case in target_cases
            if str(case.get("query_id") or "").rsplit("_q", 1)[0] == slice_user
        ]
    target_cases = target_cases[:max_cases]
    if not target_cases:
        raise RuntimeError("No target cases available for slice output.")

    output_queries: List[Dict[str, Any]] = []
    selected_doc_ids: List[str] = []
    selected_users: Set[str] = set()

    for case in target_cases:
        query_id = str(case["query_id"])
        source = dict(queries_by_id.get(query_id) or {})
        if not source:
            continue
        source["source_taxonomy"] = case.get("taxonomy") or []
        source["source_memory_recall_at_k"] = case.get("memory_recall_at_k")
        source["source_memory_first_gold_rank"] = case.get("memory_first_gold_rank")
        output_queries.append(source)
        selected_doc_ids.extend(source.get("relevant_document_ids") or [])
        sample_id = query_id.rsplit("_q", 1)[0]
        selected_users.add(sample_id)

    if document_scope == "selected-users":
        for user in sorted(selected_users):
            selected_doc_ids.extend(sorted(docs_by_user.get(user, [])))
    else:
        for case in target_cases:
            selected_doc_ids.extend(case.get("memory_retrieved_ids") or [])
            selected_doc_ids.extend(case.get("baseline_retrieved_ids") or [])

    selected_doc_ids = unique(selected_doc_ids)
    output_documents = [docs_by_id[doc_id] for doc_id in selected_doc_ids if doc_id in docs_by_id]
    missing = sorted({doc_id for query in output_queries for doc_id in query.get("relevant_document_ids", []) if doc_id not in docs_by_id})
    if missing:
        raise RuntimeError(f"Focused slice references unknown documents: {', '.join(missing[:10])}")

    slice_taxonomy = Counter(
        tag
        for query in output_queries
        for tag in query.get("source_taxonomy", [])
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": output_root.name,
        "source": "lifebench",
        "source_run": analysis.get("memory_run"),
        "case_count": len(output_queries),
        "document_count": len(output_documents),
        "document_scope": document_scope,
        "max_cases": max_cases,
        "slice_user": slice_user,
        "taxonomy_counts": dict(sorted(slice_taxonomy.items())),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_root / "recall_queries.jsonl", output_queries)
    write_jsonl_atomic(output_root / "recall_documents.jsonl", output_documents)
    write_manifest(output_root / "manifest.json", manifest)
    (output_root / "README.md").write_text(
        "# LifeBench Multi-Evidence Slice\n\n"
        "Focused local slice generated from AMB LifeBench retrieval diagnostics. "
        "Use this as an exploratory target, not a release gate.\n\n"
        "```sh\n"
        f"swift run memory_eval retrieval-diagnostics --profile coreml_default --dataset-root ./{relative_label(output_root)} --candidate-pool-depth 40 --context-token-budget 4096 --per-document-token-budget 384 --no-cache --no-index-cache\n"
        "```\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-run", required=True, help="AMB Memory.swift retrieval JSON.")
    parser.add_argument("--baseline-run", help="Optional AMB baseline retrieval JSON, usually BM25.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-prefix", default="Explorations/lifebench/retrieval-analysis")
    parser.add_argument("--lifebench-data", help="Path to LifeBench our_en.json for optional focused slice output.")
    parser.add_argument("--output-eval-root", help="Optional focused eval output root.")
    parser.add_argument("--max-slice-cases", type=int, default=32)
    parser.add_argument("--document-scope", choices=["selected-users", "selected-gold-and-confusers"], default="selected-users")
    parser.add_argument("--slice-user", help="Optional LifeBench sample_id/user to make a user-scoped focused slice.")
    args = parser.parse_args()

    memory_path = resolve_path(args.memory_run)
    baseline_path = resolve_path(args.baseline_run) if args.baseline_run else None
    output_prefix = resolve_path(args.output_prefix)

    memory_report = load_json(memory_path)
    baseline_report = load_json(baseline_path) if baseline_path else None
    analysis = compare_runs(memory_report, baseline_report, args.k)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown_summary(analysis, memory_path, baseline_path), encoding="utf-8")

    print(f"Analysis JSON: {relative_label(json_path)}")
    print(f"Analysis Markdown: {relative_label(md_path)}")
    print(f"Target cases: {len(analysis['target_cases'])}")

    if args.output_eval_root:
        if not args.lifebench_data:
            raise RuntimeError("--output-eval-root requires --lifebench-data")
        output_root = resolve_path(args.output_eval_root)
        write_eval_slice(
            analysis,
            lifebench_data=resolve_path(args.lifebench_data),
            output_root=output_root,
            max_cases=args.max_slice_cases,
            document_scope=args.document_scope,
            slice_user=args.slice_user,
        )
        print(f"Focused eval root: {relative_label(output_root)}")


if __name__ == "__main__":
    main()
