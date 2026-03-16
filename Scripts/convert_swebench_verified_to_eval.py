#!/usr/bin/env python3
"""Convert a simplified SWE-bench Verified slice into Memory.swift eval format."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval_data_codex_support import log, normalize_spaces, write_jsonl_atomic, write_manifest

DATASET_ID = "princeton-nlp/SWE-bench_Verified"
DEFAULT_DIFFICULTIES = ("<15 min fix", "15 min - 1 hour")


def configure_hf_cache(cache_dir: Path) -> None:
    cache_dir = cache_dir.resolve()
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")


def parse_json_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [normalize_spaces(str(item)) for item in raw if normalize_spaces(str(item))]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [normalize_spaces(str(item)) for item in payload if normalize_spaces(str(item))]


def extract_issue_title(problem_statement: str, instance_id: str) -> str:
    for line in problem_statement.splitlines():
        candidate = normalize_spaces(line)
        if candidate:
            return candidate
    return instance_id


def map_query_difficulty(raw_value: str) -> str:
    normalized = normalize_spaces(raw_value).lower()
    if normalized == "<15 min fix":
        return "easy"
    if normalized == "15 min - 1 hour":
        return "medium"
    return "hard"


def render_document_text(row: Dict[str, Any], *, max_passing_tests: int) -> str:
    repo = normalize_spaces(str(row.get("repo", "")))
    instance_id = normalize_spaces(str(row.get("instance_id", "")))
    difficulty = normalize_spaces(str(row.get("difficulty", "")))
    version = normalize_spaces(str(row.get("version", "")))
    problem_statement = normalize_spaces(str(row.get("problem_statement", "")))
    hints_text = normalize_spaces(str(row.get("hints_text", "")))
    failing_tests = parse_json_list(row.get("FAIL_TO_PASS"))
    passing_tests = parse_json_list(row.get("PASS_TO_PASS"))[:max_passing_tests]

    lines: List[str] = []
    title_bits = [bit for bit in (repo, instance_id) if bit]
    if title_bits:
        lines.append(f"# {' / '.join(title_bits)}")
    if difficulty or version:
        metadata = []
        if difficulty:
            metadata.append(f"Difficulty: {difficulty}")
        if version:
            metadata.append(f"Version: {version}")
        lines.extend(["", " | ".join(metadata)])
    if problem_statement:
        lines.extend(["", "## Problem", "", problem_statement])
    if hints_text:
        lines.extend(["", "## Hints", "", hints_text])
    if failing_tests:
        lines.extend(["", "## Failing Tests", ""])
        lines.extend(f"- {name}" for name in failing_tests)
    if passing_tests:
        lines.extend(["", "## Stable Tests", ""])
        lines.extend(f"- {name}" for name in passing_tests)
    return "\n".join(lines).strip()


def build_manifest(split: str, difficulties: Sequence[str], max_instances: Optional[int]) -> Dict[str, Any]:
    return {
        "dataset": "swebench_verified",
        "provenance": "external_swebench_verified",
        "synthetic_status": "external",
        "primary_use": "software_engineering_retrieval",
        "license_scope": "license_review_required",
        "source_datasets": [
            {
                "name": DATASET_ID,
                "split": split,
                "difficulty_filter": list(difficulties),
                **({"max_instances": max_instances} if max_instances is not None else {}),
            }
        ],
        "known_limits": [
            "This is a simplified issue-centric conversion, not the full SWE-bench task with repo snapshots and executable patches.",
            "Queries are derived from issue titles and should be augmented with harder paraphrases before promotion.",
            "Hugging Face metadata did not expose a clear license in local lookup, so review licensing before treating this corpus as default-commercial-safe.",
        ],
        "recommended_weight": "secondary",
        "promotion_status": "draft",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a simplified SWE-bench Verified slice into Memory.swift eval format.")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--split", default="test", help="Dataset split to load (default: test).")
    parser.add_argument("--cache-dir", default="/tmp/swebench_verified_cache", help="Hugging Face cache directory.")
    parser.add_argument(
        "--difficulties",
        default=",".join(DEFAULT_DIFFICULTIES),
        help="Comma-separated SWE-bench difficulty buckets to include.",
    )
    parser.add_argument("--max-instances", type=int, default=250, help="Optional cap on issue instances after filtering.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic shuffle seed.")
    parser.add_argument("--max-passing-tests", type=int, default=10, help="Maximum stable tests to retain per document.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    difficulties = [normalize_spaces(item) for item in args.difficulties.split(",") if normalize_spaces(item)]
    if not difficulties:
        raise RuntimeError("At least one difficulty bucket is required.")

    configure_hf_cache(cache_dir)
    from datasets import load_dataset

    frame = load_dataset(DATASET_ID, split=args.split, cache_dir=str(cache_dir / "datasets"))

    rows = [dict(row) for row in frame if normalize_spaces(str(row.get("difficulty", ""))) in set(difficulties)]
    random.Random(args.seed).shuffle(rows)
    if args.max_instances is not None:
        rows = rows[: args.max_instances]

    documents: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        repo = normalize_spaces(str(row.get("repo", "")))
        instance_id = normalize_spaces(str(row.get("instance_id", ""))) or f"instance-{index:04d}"
        problem_statement = normalize_spaces(str(row.get("problem_statement", "")))
        title = extract_issue_title(problem_statement, instance_id)
        query = f"{title} in {repo}" if repo and repo.lower() not in title.lower() else title
        document_id = f"doc-{index:04d}"

        documents.append(
            {
                "id": document_id,
                "relative_path": f"issues/{repo.replace('/', '__')}/{instance_id}.md" if repo else f"issues/{instance_id}.md",
                "kind": "markdown",
                "text": render_document_text(row, max_passing_tests=args.max_passing_tests),
            }
        )
        queries.append(
            {
                "id": f"q-{index:04d}",
                "query": query,
                "relevant_document_ids": [document_id],
                "difficulty": map_query_difficulty(str(row.get("difficulty", ""))),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "storage_cases.jsonl", [])
    write_jsonl_atomic(output_dir / "recall_documents.jsonl", documents)
    write_jsonl_atomic(output_dir / "recall_queries.jsonl", queries)
    write_manifest(output_dir / "manifest.json", build_manifest(args.split, difficulties, args.max_instances))

    log(f"Wrote {len(documents)} documents and {len(queries)} queries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
