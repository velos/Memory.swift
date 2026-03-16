#!/usr/bin/env python3
"""Merge multiple eval dataset roots into one staged corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval_data_codex_support import load_jsonl, load_manifest, normalize_spaces, write_jsonl_atomic, write_manifest


def parse_source_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value).resolve()
        return path.name, path
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path).resolve()


def dedupe_key_for_document(row: Dict[str, Any]) -> str:
    return normalize_spaces(str(row.get("text", ""))).lower()


def dedupe_key_for_query(row: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    relevant = tuple(sorted(str(item) for item in row.get("relevant_document_ids", [])))
    return normalize_spaces(str(row.get("query", ""))).lower(), relevant


def dedupe_key_for_storage(row: Dict[str, Any]) -> str:
    return normalize_spaces(str(row.get("text", ""))).lower()


def merge_sources(
    sources: Sequence[Tuple[str, Path]],
    *,
    max_documents: Optional[int],
    max_queries: Optional[int],
    max_storage_cases: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    merged_documents: List[Dict[str, Any]] = []
    merged_queries: List[Dict[str, Any]] = []
    merged_storage: List[Dict[str, Any]] = []
    source_manifests: List[Dict[str, Any]] = []

    doc_dedupe: Dict[str, str] = {}
    query_dedupe = set()
    storage_dedupe = set()
    remapped_doc_ids: Dict[Tuple[str, str], str] = {}
    next_doc_id = 1
    next_query_id = 1
    next_storage_id = 1

    for source_name, source_path in sources:
        manifest = load_manifest(source_path / "manifest.json")
        if manifest:
            source_manifests.append(manifest)

        documents = load_jsonl(source_path / "recall_documents.jsonl")
        for document in documents:
            key = dedupe_key_for_document(document)
            if not key:
                continue
            mapped = doc_dedupe.get(key)
            if mapped is None:
                if max_documents is not None and len(merged_documents) >= max_documents:
                    continue
                mapped = f"doc-{next_doc_id:04d}"
                next_doc_id += 1
                doc_dedupe[key] = mapped
                relative_path = str(document.get("relative_path", "")).strip()
                merged_documents.append(
                    {
                        "id": mapped,
                        "relative_path": f"{source_name}/{relative_path}" if relative_path else f"{source_name}/{mapped}.md",
                        "kind": str(document.get("kind", "markdown")).strip() or "markdown",
                        "text": str(document.get("text", "")),
                        **({"memory_type": str(document.get("memory_type", "")).strip()} if str(document.get("memory_type", "")).strip() else {}),
                    }
                )
            remapped_doc_ids[(source_name, str(document.get("id", "")))] = mapped

        queries = load_jsonl(source_path / "recall_queries.jsonl")
        for query in queries:
            relevant = []
            for raw_id in query.get("relevant_document_ids", []):
                mapped = remapped_doc_ids.get((source_name, str(raw_id)))
                if mapped and mapped not in relevant:
                    relevant.append(mapped)
            if not relevant:
                continue
            candidate = {
                "query": str(query.get("query", "")),
                "relevant_document_ids": relevant[:3],
            }
            if "memory_types" in query:
                candidate["memory_types"] = list(query.get("memory_types", []))
            if "difficulty" in query:
                candidate["difficulty"] = str(query.get("difficulty", ""))
            dedupe_key = dedupe_key_for_query(candidate)
            if dedupe_key in query_dedupe:
                continue
            if max_queries is not None and len(merged_queries) >= max_queries:
                continue
            query_dedupe.add(dedupe_key)
            merged_queries.append(
                {
                    "id": f"q-{next_query_id:04d}",
                    **candidate,
                }
            )
            next_query_id += 1

        storage_cases = load_jsonl(source_path / "storage_cases.jsonl")
        for storage_case in storage_cases:
            key = dedupe_key_for_storage(storage_case)
            if not key or key in storage_dedupe:
                continue
            if max_storage_cases is not None and len(merged_storage) >= max_storage_cases:
                continue
            storage_dedupe.add(key)
            merged_storage.append(
                {
                    "id": f"storage-{next_storage_id:04d}",
                    "kind": str(storage_case.get("kind", "markdown")).strip() or "markdown",
                    "text": str(storage_case.get("text", "")),
                    "expected_memory_type": str(storage_case.get("expected_memory_type", "")),
                    "required_spans": list(storage_case.get("required_spans", [])),
                    **({"source_document_id": str(storage_case.get("source_document_id", ""))} if str(storage_case.get("source_document_id", "")).strip() else {}),
                }
            )
            next_storage_id += 1

    return merged_documents, merged_queries, merged_storage, source_manifests


def build_manifest(
    dataset_name: str,
    source_names: Sequence[str],
    source_manifests: Sequence[Dict[str, Any]],
    *,
    primary_use: str,
    recommended_weight: str,
) -> Dict[str, Any]:
    source_datasets: List[Any] = []
    source_license_scopes: List[str] = []
    inherited_limits: List[str] = []
    for manifest in source_manifests:
        sources = manifest.get("source_datasets")
        if isinstance(sources, list):
            source_datasets.extend(sources)
        license_scope = str(manifest.get("license_scope", "")).strip()
        if license_scope:
            source_license_scopes.append(license_scope)
        known_limits = manifest.get("known_limits")
        if isinstance(known_limits, list):
            inherited_limits.extend(str(item) for item in known_limits if str(item).strip())
    if not source_datasets:
        source_datasets = list(source_names)

    normalized_license_scopes = [scope for scope in source_license_scopes if scope]
    if not normalized_license_scopes:
        license_scope = "license_review_required"
    elif all(scope == "commercial_safe" for scope in normalized_license_scopes):
        license_scope = "commercial_safe"
    elif len(set(normalized_license_scopes)) == 1:
        license_scope = normalized_license_scopes[0]
    else:
        license_scope = "mixed_or_review_required"

    known_limits = list(dict.fromkeys(
        inherited_limits
        + [
            "Merged from multiple public corpora with document-level deduplication.",
            "Merged queries preserve relevant-document mapping but not source-specific identifiers.",
            "Use tag_eval_data_minimax.py or tag_eval_data_codex.py to add or refresh memory tags and adversarial coverage.",
        ]
    ))

    return {
        "dataset": dataset_name,
        "provenance": "merged_public_corpora",
        "synthetic_status": "mixed_external",
        "primary_use": primary_use,
        "license_scope": license_scope,
        "source_datasets": source_datasets,
        "known_limits": known_limits,
        "recommended_weight": recommended_weight,
        "promotion_status": "draft",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple Memory.swift eval dataset roots.")
    parser.add_argument("--output-dir", required=True, help="Output merged dataset directory.")
    parser.add_argument("--dataset-name", required=True, help="Dataset name written to manifest.json.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source dataset root. Use name=/path to override the source prefix.",
    )
    parser.add_argument("--primary-use", default="staging_candidate", help="Manifest primary_use value.")
    parser.add_argument("--recommended-weight", default="primary", help="Manifest recommended_weight value.")
    parser.add_argument("--max-documents", type=int, default=None, help="Optional cap on merged documents.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap on merged queries.")
    parser.add_argument("--max-storage-cases", type=int, default=None, help="Optional cap on merged storage cases.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    sources = [parse_source_arg(value) for value in args.source]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    documents, queries, storage_cases, source_manifests = merge_sources(
        sources,
        max_documents=args.max_documents,
        max_queries=args.max_queries,
        max_storage_cases=args.max_storage_cases,
    )

    write_jsonl_atomic(output_dir / "recall_documents.jsonl", documents)
    write_jsonl_atomic(output_dir / "recall_queries.jsonl", queries)
    write_jsonl_atomic(output_dir / "storage_cases.jsonl", storage_cases)
    write_manifest(
        output_dir / "manifest.json",
        build_manifest(
            args.dataset_name,
            [name for name, _ in sources],
            source_manifests,
            primary_use=args.primary_use,
            recommended_weight=args.recommended_weight,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
