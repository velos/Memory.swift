#!/usr/bin/env python3
"""Build a focused LongMemEval recall slice from branch diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from eval_data_codex_support import load_jsonl, log, normalize_spaces, write_jsonl_atomic, write_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = "Evals/longmemeval_rescue_v1"
DEFAULT_CLASSIFICATIONS = ["candidate_generation"]


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


def unique_preserving_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for raw in values:
        value = normalize_spaces(str(raw)).strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def branch_rank_summary(case: Dict[str, Any]) -> Dict[str, Optional[int]]:
    summary: Dict[str, Optional[int]] = {}
    for branch in case.get("branchResults") or []:
        if not isinstance(branch, dict):
            continue
        name = str(branch.get("name") or "").strip()
        if not name:
            continue
        raw_rank = branch.get("bestRelevantRank")
        summary[name] = int(raw_rank) if isinstance(raw_rank, int) else None
    return summary


def case_document_ids(case: Dict[str, Any], max_candidates_per_case: int, per_branch_limit: int) -> List[str]:
    selected: List[str] = []

    def add_many(values: Iterable[Any]) -> None:
        nonlocal selected
        selected = unique_preserving_order(selected + [str(value) for value in values])

    relevant_ids = [str(value) for value in case.get("relevantDocumentIds") or []]
    add_many(relevant_ids)
    add_many(case.get("originalRetrievedDocumentIds") or [])

    branch_priority = {
        "current_top_k": 0,
        "current_wide": 1,
        "no_expansion_wide": 2,
        "lexical_wide": 3,
        "semantic_wide": 4,
    }
    branches = [
        branch for branch in case.get("branchResults") or []
        if isinstance(branch, dict)
    ]
    branches.sort(key=lambda branch: branch_priority.get(str(branch.get("name") or ""), 99))
    for branch in branches:
        add_many((branch.get("retrievedDocumentIds") or [])[:per_branch_limit])

    if len(selected) <= max_candidates_per_case:
        return selected

    relevant_set = set(relevant_ids)
    kept: List[str] = []
    for document_id in selected:
        if document_id in relevant_set:
            kept.append(document_id)
    for document_id in selected:
        if document_id in relevant_set:
            continue
        kept.append(document_id)
        if len(kept) >= max_candidates_per_case:
            break
    return unique_preserving_order(kept)


def selected_diagnostic_cases(
    report: Dict[str, Any],
    classifications: Sequence[str],
    taxonomy: Sequence[str],
    max_cases: int,
) -> List[Dict[str, Any]]:
    classification_filter = {value.strip() for value in classifications if value.strip()}
    taxonomy_filter = {value.strip() for value in taxonomy if value.strip()}

    cases: List[Dict[str, Any]] = []
    for case in report.get("caseResults") or []:
        if not isinstance(case, dict):
            continue
        classification = str(case.get("classification") or "").strip()
        case_taxonomy = {str(value) for value in case.get("taxonomy") or []}
        if classification_filter and classification not in classification_filter:
            continue
        if taxonomy_filter and case_taxonomy.isdisjoint(taxonomy_filter):
            continue
        cases.append(case)

    cases.sort(
        key=lambda case: (
            str(case.get("classification") or ""),
            -len(case.get("relevantDocumentIds") or []),
            str(case.get("id") or ""),
        )
    )
    return cases[:max_cases]


def recall_query_record(case: Dict[str, Any], source_queries_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    query_id = str(case.get("id") or "")
    source = dict(source_queries_by_id.get(query_id) or {})
    if not source:
        source = {
            "id": query_id,
            "query": str(case.get("query") or ""),
            "relevant_document_ids": [str(value) for value in case.get("relevantDocumentIds") or []],
        }

    source["source_classification"] = str(case.get("classification") or "")
    source["source_taxonomy"] = [str(value) for value in case.get("taxonomy") or []]
    source["source_branch_ranks"] = branch_rank_summary(case)
    return source


def readme_text(manifest: Dict[str, Any]) -> str:
    classification_lines = "\n".join(
        f"- `{key}`: {value}" for key, value in sorted(manifest["classification_counts"].items())
    ) or "- none"
    taxonomy_lines = "\n".join(
        f"- `{key}`: {value}" for key, value in sorted(manifest["taxonomy_counts"].items())
    ) or "- none"

    dataset = manifest["dataset"]

    return f"""# {dataset}

Focused LongMemEval recall slice mined from branch diagnostics.

This is a small diagnostic benchmark, not a replacement for the full LongMemEval gate. It keeps the selected source queries, their relevant documents, and the top confuser documents seen across diagnostic retrieval branches so targeted recall changes can be tested quickly.

Source diagnostic: `{manifest["source_diagnostic"]}`
Source run: `{manifest["source_run"]}`

Classification counts:

{classification_lines}

Taxonomy counts:

{taxonomy_lines}

Commands:

```sh
python3 Scripts/build_longmemeval_rescue.py --diagnostic <branch-diagnostics.json> --overwrite
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/{dataset} --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/<focused-baseline>.json ./Evals/{dataset}/runs/<run>.json
```
"""


def build(args: argparse.Namespace) -> None:
    dataset_root = resolve_path(args.dataset_root)
    diagnostic_path = resolve_path(args.diagnostic)
    output_root = resolve_path(args.output_root)

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(f"{relative_label(output_root)} already exists. Pass --overwrite to replace generated files.")

    documents = load_jsonl(dataset_root / "recall_documents.jsonl")
    queries = load_jsonl(dataset_root / "recall_queries.jsonl")
    docs_by_id = {str(document["id"]): document for document in documents}
    queries_by_id = {str(query["id"]): query for query in queries}
    report = load_json(diagnostic_path)

    selected_cases = selected_diagnostic_cases(
        report=report,
        classifications=args.classification or DEFAULT_CLASSIFICATIONS,
        taxonomy=args.taxonomy or [],
        max_cases=args.max_cases,
    )
    if not selected_cases:
        raise RuntimeError("No diagnostic cases matched the requested classification/taxonomy filters.")

    output_queries: List[Dict[str, Any]] = []
    selected_doc_ids: List[str] = []
    missing_relevant: List[str] = []
    for case in selected_cases:
        output_queries.append(recall_query_record(case, queries_by_id))
        case_doc_ids = case_document_ids(
            case,
            max_candidates_per_case=args.max_candidates_per_case,
            per_branch_limit=args.per_branch_limit,
        )
        selected_doc_ids = unique_preserving_order(selected_doc_ids + case_doc_ids)
        for document_id in case.get("relevantDocumentIds") or []:
            if str(document_id) not in docs_by_id:
                missing_relevant.append(str(document_id))

    if missing_relevant:
        raise RuntimeError(f"Missing relevant documents in source dataset: {', '.join(sorted(set(missing_relevant)))}")

    output_documents = [
        docs_by_id[document_id]
        for document_id in selected_doc_ids
        if document_id in docs_by_id
    ]

    classification_counts = Counter(str(case.get("classification") or "") for case in selected_cases)
    taxonomy_counts = Counter(
        str(tag)
        for case in selected_cases
        for tag in case.get("taxonomy") or []
    )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": output_root.name,
        "source_dataset": relative_label(dataset_root),
        "source_diagnostic": relative_label(diagnostic_path),
        "source_run": relative_label(resolve_path(str(report.get("sourceRun") or ""))) if report.get("sourceRun") else "",
        "case_count": len(output_queries),
        "document_count": len(output_documents),
        "max_cases": args.max_cases,
        "max_candidates_per_case": args.max_candidates_per_case,
        "per_branch_limit": args.per_branch_limit,
        "classification_filter": args.classification or DEFAULT_CLASSIFICATIONS,
        "taxonomy_filter": args.taxonomy or [],
        "classification_counts": dict(sorted(classification_counts.items())),
        "taxonomy_counts": dict(sorted(taxonomy_counts.items())),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_root / "recall_queries.jsonl", output_queries)
    write_jsonl_atomic(output_root / "recall_documents.jsonl", output_documents)
    write_manifest(output_root / "manifest.json", manifest)
    (output_root / "README.md").write_text(readme_text(manifest), encoding="utf-8")

    log(
        f"Wrote {len(output_queries)} cases and {len(output_documents)} documents "
        f"to {relative_label(output_root)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="Evals/longmemeval_v2")
    parser.add_argument("--diagnostic", required=True, help="LongMemEval branch diagnostic JSON.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--classification", action="append", help="Diagnostic classification to include. Repeatable.")
    parser.add_argument("--taxonomy", action="append", help="Require at least one taxonomy tag. Repeatable.")
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--max-candidates-per-case", type=int, default=80)
    parser.add_argument("--per-branch-limit", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    if args.max_candidates_per_case <= 0:
        raise ValueError("--max-candidates-per-case must be positive")
    if args.per_branch_limit <= 0:
        raise ValueError("--per-branch-limit must be positive")
    build(args)


if __name__ == "__main__":
    main()
