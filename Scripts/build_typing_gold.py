#!/usr/bin/env python3
"""Build a small audited typing gold set from curated eval records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_data_codex_support import load_jsonl, write_jsonl_atomic, write_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_ROOT = REPO_ROOT / "Evals"
DEFAULT_OUTPUT_ROOT = EVALS_ROOT / "typing_gold_v1"

SELECTIONS = [
    {
        "id": "factual-001",
        "source_corpus": "tech_v2",
        "source_kind": "storage_case",
        "source_id": "storage-0001",
    },
    {
        "id": "factual-002",
        "source_corpus": "general_v2",
        "source_kind": "storage_case",
        "source_id": "storage-0003",
    },
    {
        "id": "procedural-001",
        "source_corpus": "tech_v2",
        "source_kind": "storage_case",
        "source_id": "storage-0112",
    },
    {
        "id": "procedural-002",
        "source_corpus": "general_v2",
        "source_kind": "storage_case",
        "source_id": "storage-0013",
    },
    {
        "id": "episodic-001",
        "source_corpus": "general_v2",
        "source_kind": "storage_case",
        "source_id": "storage-0009",
    },
    {
        "id": "episodic-002",
        "source_corpus": "longmemeval_v2",
        "source_kind": "storage_case",
        "source_id": "storage-answer_39900a0a_1",
    },
    {
        "id": "semantic-001",
        "source_corpus": "general_v2",
        "source_kind": "storage_case",
        "source_id": "storage-0081",
    },
    {
        "id": "semantic-002",
        "source_corpus": "longmemeval_v2",
        "source_kind": "storage_case",
        "source_id": "storage-answer_555dfb94",
    },
    {
        "id": "contextual-001",
        "source_corpus": "general_v2",
        "source_kind": "storage_case",
        "source_id": "storage-0152",
    },
    {
        "id": "contextual-002",
        "source_corpus": "longmemeval_v2",
        "source_kind": "storage_case",
        "source_id": "storage-answer_a7b44747_1",
    },
    {
        "id": "temporal-001",
        "source_corpus": "longmemeval_v2",
        "source_kind": "storage_case",
        "source_id": "storage-answer_e936197f_1",
    },
    {
        "id": "temporal-002",
        "source_corpus": "longmemeval_v2",
        "source_kind": "storage_case",
        "source_id": "storage-answer_6ea1541e_2",
    },
    {
        "id": "emotional-001",
        "source_corpus": "general_v2",
        "source_kind": "recall_document",
        "source_id": "doc-0463",
        "expected_memory_type": "emotional",
        "required_spans": ["Veteran suicide prevention", "You re not alone", "Veterans Crisis Line"],
    },
    {
        "id": "emotional-002",
        "source_corpus": "general_v2",
        "source_kind": "recall_document",
        "source_id": "doc-0672",
        "expected_memory_type": "emotional",
        "required_spans": ["battle with anxiety", "Mind Matters", "Real People, Real Stories"],
    },
    {
        "id": "social-001",
        "source_corpus": "general_v2",
        "source_kind": "recall_document",
        "source_id": "doc-0679",
        "expected_memory_type": "social",
        "required_spans": ["Multicultural Summer Fiesta", "The Harmony Project", "crosscultural empathy and unity"],
    },
    {
        "id": "social-002",
        "source_corpus": "general_v2",
        "source_kind": "recall_document",
        "source_id": "doc-0798",
        "expected_memory_type": "social",
        "required_spans": ["Support Your Local Amateur Sports Teams", "our collective identity", "Roseville Ravens"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Target dataset directory. Defaults to Evals/typing_gold_v1.",
    )
    return parser.parse_args()


def load_source_records(corpus: str, source_kind: str) -> dict[str, dict]:
    filename = "storage_cases.jsonl" if source_kind == "storage_case" else "recall_documents.jsonl"
    records = load_jsonl(EVALS_ROOT / corpus / filename)
    return {record["id"]: record for record in records}


def materialize_case(selection: dict, source_record: dict) -> dict:
    if selection["source_kind"] == "storage_case":
        case = {
            "id": selection["id"],
            "kind": source_record.get("kind", "markdown"),
            "text": source_record["text"],
            "expected_memory_type": source_record["expected_memory_type"],
            "required_spans": list(source_record.get("required_spans", [])),
        }
    else:
        case = {
            "id": selection["id"],
            "kind": source_record.get("kind", "markdown"),
            "text": source_record["text"],
            "expected_memory_type": selection["expected_memory_type"],
            "required_spans": list(selection["required_spans"]),
        }

    case["source_corpus"] = selection["source_corpus"]
    case["source_kind"] = selection["source_kind"]
    case["source_id"] = selection["source_id"]
    return case


def validate_case(case: dict) -> None:
    if case["expected_memory_type"] == "":
        raise ValueError(f"{case['id']}: expected_memory_type is empty")
    if not case["required_spans"]:
        raise ValueError(f"{case['id']}: required_spans must not be empty")

    lowered_text = case["text"].lower()
    missing = [span for span in case["required_spans"] if span.lower() not in lowered_text]
    if missing:
        raise ValueError(f"{case['id']}: required spans missing from text: {missing}")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()

    source_cache: dict[tuple[str, str], dict[str, dict]] = {}
    cases: list[dict] = []
    for selection in SELECTIONS:
        key = (selection["source_corpus"], selection["source_kind"])
        if key not in source_cache:
            source_cache[key] = load_source_records(*key)
        source_record = source_cache[key].get(selection["source_id"])
        if source_record is None:
            raise KeyError(f"Missing source record {selection['source_corpus']}:{selection['source_id']}")
        case = materialize_case(selection, source_record)
        validate_case(case)
        cases.append(case)

    type_counts: dict[str, int] = {}
    for case in cases:
        key = case["expected_memory_type"]
        type_counts[key] = type_counts.get(key, 0) + 1

    write_jsonl_atomic(output_root / "storage_cases.jsonl", cases)
    write_jsonl_atomic(output_root / "recall_documents.jsonl", [])
    write_jsonl_atomic(output_root / "recall_queries.jsonl", [])
    (output_root / "README.md").write_text(
        "# typing_gold_v1\n\n"
        "Curated storage-only typing benchmark built from repaired staged corpora.\n",
        encoding="utf-8",
    )
    (output_root / "selection.json").write_text(
        json.dumps(SELECTIONS, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        output_root / "manifest.json",
        {
            "provenance": "curated-from-staged-corpora",
            "synthetic_status": "mixed",
            "primary_use": "typing-eval",
            "typing_coverage": type_counts,
            "known_limits": [
                "small storage-only benchmark",
                "social and emotional classes are sourced from recall documents because repaired staged storage cases do not cover them",
            ],
            "recommended_weight": "high",
            "promotion_status": "audited",
            "source_datasets": ["general_v2", "tech_v2", "longmemeval_v2"],
            "license_scope": "inherits-source-datasets",
        },
    )


if __name__ == "__main__":
    main()
