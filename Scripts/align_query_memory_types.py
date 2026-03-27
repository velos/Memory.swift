#!/usr/bin/env python3
"""Align query memory types with their relevant document labels."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval_data_codex_support import load_jsonl, log, write_jsonl_atomic
from tag_eval_data_codex import MEMORY_TYPES


def ordered_unique(items: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def choose_query_types(current_types: List[str], doc_types: List[str]) -> List[str]:
    filtered_current = [item for item in current_types if item in doc_types]
    if filtered_current:
        return filtered_current
    return doc_types[:2]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align query memory_types with relevant document labels.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing recall JSONL files.")
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional output report path (default: <dataset-root>/query_type_alignment_report.json).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze and report without writing changes.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    docs_path = dataset_root / "recall_documents.jsonl"
    queries_path = dataset_root / "recall_queries.jsonl"
    report_path = Path(args.report_path).resolve() if args.report_path else dataset_root / "query_type_alignment_report.json"

    documents = load_jsonl(docs_path)
    queries = load_jsonl(queries_path)
    docs_by_id = {str(document.get("id", "")): document for document in documents}

    updated_queries = 0
    unresolved_queries: List[str] = []
    changed_examples: List[Dict[str, Any]] = []
    pattern_counts: Counter[str] = Counter()

    for query in queries:
        query_id = str(query.get("id", "")).strip()
        relevant_doc_ids = [str(value).strip() for value in query.get("relevant_document_ids", []) if str(value).strip()]
        doc_types = ordered_unique(
            str(docs_by_id[doc_id].get("memory_type", "")).strip().lower()
            for doc_id in relevant_doc_ids
            if doc_id in docs_by_id and str(docs_by_id[doc_id].get("memory_type", "")).strip().lower() in MEMORY_TYPES
        )
        if not doc_types:
            unresolved_queries.append(query_id)
            continue

        current_types = [
            str(value).strip().lower()
            for value in query.get("memory_types", [])
            if str(value).strip().lower() in MEMORY_TYPES
        ]
        desired_types = choose_query_types(current_types, doc_types)
        if current_types == desired_types:
            continue

        pattern_counts[f"{tuple(current_types)} -> {tuple(desired_types)}"] += 1
        updated_queries += 1
        if len(changed_examples) < 20:
            changed_examples.append(
                {
                    "id": query_id,
                    "before": current_types,
                    "after": desired_types,
                    "relevant_document_ids": relevant_doc_ids,
                    "document_types": doc_types,
                    "query": query.get("query", ""),
                }
            )
        query["memory_types"] = desired_types

    if not args.dry_run:
        write_jsonl_atomic(queries_path, queries)

    report = {
        "dataset": dataset_root.name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": bool(args.dry_run),
        "counts": {
            "total_queries": len(queries),
            "updated_queries": updated_queries,
            "unresolved_queries": len(unresolved_queries),
        },
        "pattern_counts": dict(pattern_counts.most_common()),
        "unresolved_query_ids": unresolved_queries,
        "changed_examples": changed_examples,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(
        f"[align-query-types] {dataset_root.name}: "
        f"updated {updated_queries}/{len(queries)} queries, unresolved {len(unresolved_queries)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
