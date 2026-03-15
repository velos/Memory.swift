#!/usr/bin/env python3
"""Build deterministic audit packets for staged Memory.swift eval datasets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from eval_data_audit_support import (
    highlight_spans,
    parse_numeric_tail,
    stratified_sample,
    truncate_text,
)
from eval_data_codex_support import load_jsonl, write_jsonl_atomic


def packet_id(dataset: str, entry_type: str, source_id: str) -> str:
    return f"{dataset}:{entry_type}:{source_id}"


def render_doc_excerpt(doc: Dict[str, Any], limit: int = 1800) -> Dict[str, Any]:
    return {
        "id": str(doc.get("id", "")),
        "relative_path": str(doc.get("relative_path", "")),
        "memory_type": doc.get("memory_type"),
        "text": truncate_text(str(doc.get("text", "")), limit),
    }


def build_query_item(
    dataset: str,
    query: Dict[str, Any],
    docs_by_id: Dict[str, Dict[str, Any]],
    *,
    sample_reason: str,
    review_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_id = str(query.get("id", ""))
    relevant_documents = []
    for doc_id in query.get("relevant_document_ids", []):
        doc = docs_by_id.get(str(doc_id))
        if doc:
            relevant_documents.append(render_doc_excerpt(doc))
    difficulty = str(query.get("difficulty", "unknown")).strip() or "unknown"
    return {
        "packet_id": packet_id(dataset, "query", source_id),
        "dataset": dataset,
        "entry_type": "query",
        "sample_reason": sample_reason,
        "source_id": source_id,
        "query": str(query.get("query", "")),
        "current_memory_types": list(query.get("memory_types", [])),
        "current_difficulty": difficulty,
        "relevant_document_ids": list(query.get("relevant_document_ids", [])),
        "relevant_documents": relevant_documents,
        "review_context": review_entry,
        "is_hard_query": difficulty == "hard",
        "numeric_tail": parse_numeric_tail(source_id),
    }


def build_document_item(
    dataset: str,
    document: Dict[str, Any],
    *,
    sample_reason: str,
    review_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_id = str(document.get("id", ""))
    return {
        "packet_id": packet_id(dataset, "document", source_id),
        "dataset": dataset,
        "entry_type": "document",
        "sample_reason": sample_reason,
        "source_id": source_id,
        "relative_path": str(document.get("relative_path", "")),
        "current_memory_type": document.get("memory_type"),
        "document": render_doc_excerpt(document, limit=2400),
        "review_context": review_entry,
    }


def build_storage_item(
    dataset: str,
    storage_case: Dict[str, Any],
    *,
    sample_reason: str,
    review_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_id = str(storage_case.get("id", ""))
    spans = list(storage_case.get("required_spans", []))
    text = str(storage_case.get("text", ""))
    return {
        "packet_id": packet_id(dataset, "storage", source_id),
        "dataset": dataset,
        "entry_type": "storage",
        "sample_reason": sample_reason,
        "source_id": source_id,
        "source_document_id": storage_case.get("source_document_id"),
        "current_memory_type": storage_case.get("expected_memory_type"),
        "required_spans": spans,
        "text": truncate_text(text, 2600),
        "highlighted_text": truncate_text(highlight_spans(text, spans), 2600),
        "review_context": review_entry,
    }


def sample_query_rows(
    dataset_name: str,
    queries: Sequence[Dict[str, Any]],
    *,
    target: int,
    seed: int,
    exclude_ids: Set[str],
    include_hard: bool,
) -> List[Dict[str, Any]]:
    candidates = []
    for query in queries:
        query_id = str(query.get("id", ""))
        if query_id in exclude_ids:
            continue
        difficulty = str(query.get("difficulty", "unknown")).strip().lower()
        if not include_hard and difficulty == "hard":
            continue
        candidates.append(query)

    return stratified_sample(
        candidates,
        target,
        key_fn=lambda row: (
            str(row.get("difficulty", "unknown")).strip().lower() or "unknown",
            str((row.get("memory_types") or ["unknown"])[0]),
        ),
        id_fn=lambda row: str(row.get("id", "")),
        seed=seed + len(dataset_name),
    )


def sample_document_rows(
    dataset_name: str,
    documents: Sequence[Dict[str, Any]],
    *,
    target: int,
    seed: int,
    exclude_ids: Set[str],
) -> List[Dict[str, Any]]:
    candidates = [row for row in documents if str(row.get("id", "")) not in exclude_ids]
    return stratified_sample(
        candidates,
        target,
        key_fn=lambda row: (str(row.get("memory_type", "unknown")).strip().lower() or "unknown",),
        id_fn=lambda row: str(row.get("id", "")),
        seed=seed + len(dataset_name) * 3,
    )


def sample_storage_rows(
    dataset_name: str,
    storage_cases: Sequence[Dict[str, Any]],
    *,
    target: int,
    seed: int,
    exclude_ids: Set[str],
) -> List[Dict[str, Any]]:
    candidates = [row for row in storage_cases if str(row.get("id", "")) not in exclude_ids]
    return stratified_sample(
        candidates,
        target,
        key_fn=lambda row: (str(row.get("expected_memory_type", "unknown")).strip().lower() or "unknown",),
        id_fn=lambda row: str(row.get("id", "")),
        seed=seed + len(dataset_name) * 5,
    )


def build_review_items(
    dataset_name: str,
    review_rows: Sequence[Dict[str, Any]],
    docs_by_id: Dict[str, Dict[str, Any]],
    queries_by_id: Dict[str, Dict[str, Any]],
    storage_by_id: Dict[str, Dict[str, Any]],
    storage_by_source_doc_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for review_row in review_rows:
        mode = str(review_row.get("mode", "")).strip()
        record_id = str(review_row.get("record_id", "")).strip()
        if not mode or not record_id:
            continue
        if mode == "query-tags":
            query = queries_by_id.get(record_id)
            if query:
                items.append(build_query_item(dataset_name, query, docs_by_id, sample_reason="review_queue", review_entry=review_row))
        elif mode == "document-tags":
            document = docs_by_id.get(record_id)
            if document:
                items.append(build_document_item(dataset_name, document, sample_reason="review_queue", review_entry=review_row))
        elif mode == "storage-cases":
            storage_case = storage_by_id.get(record_id)
            if storage_case:
                items.append(build_storage_item(dataset_name, storage_case, sample_reason="review_queue", review_entry=review_row))
                continue
            document = docs_by_id.get(record_id)
            if document:
                fallback_case = storage_by_source_doc_id.get(record_id, {
                    "id": record_id,
                    "source_document_id": record_id,
                    "expected_memory_type": document.get("memory_type"),
                    "required_spans": [],
                    "text": document.get("text", ""),
                })
                items.append(build_storage_item(dataset_name, fallback_case, sample_reason="review_queue", review_entry=review_row))
    return items


def render_markdown(items: Sequence[Dict[str, Any]], dataset_name: str, generated_at: str) -> str:
    lines = [
        f"# Audit Packet: {dataset_name}",
        "",
        f"- Generated: {generated_at}",
        f"- Total packet items: {len(items)}",
        "",
    ]
    for item in items:
        entry_type = str(item.get("entry_type", "unknown"))
        lines.append(f"## {item['packet_id']}")
        lines.append("")
        lines.append(f"- Type: `{entry_type}`")
        lines.append(f"- Sample reason: `{item.get('sample_reason', 'unknown')}`")
        if item.get("review_context"):
            lines.append(f"- Review flag: `{item['review_context'].get('reason', 'unknown')}`")
        if entry_type == "query":
            lines.append(f"- Query ID: `{item['source_id']}`")
            lines.append(f"- Difficulty: `{item.get('current_difficulty', 'unknown')}`")
            lines.append(f"- Memory types: `{item.get('current_memory_types', [])}`")
            lines.append("")
            lines.append(item.get("query", ""))
            lines.append("")
            lines.append("Relevant documents:")
            lines.append("")
            for doc in item.get("relevant_documents", []):
                lines.append(f"### `{doc.get('id')}` `{doc.get('memory_type')}`")
                lines.append("")
                lines.append(f"`{doc.get('relative_path', '')}`")
                lines.append("")
                lines.append(doc.get("text", ""))
                lines.append("")
        elif entry_type == "document":
            lines.append(f"- Document ID: `{item['source_id']}`")
            lines.append(f"- Memory type: `{item.get('current_memory_type', 'unknown')}`")
            lines.append("")
            lines.append(f"`{item.get('relative_path', '')}`")
            lines.append("")
            lines.append(item.get("document", {}).get("text", ""))
            lines.append("")
        else:
            lines.append(f"- Storage ID: `{item['source_id']}`")
            lines.append(f"- Memory type: `{item.get('current_memory_type', 'unknown')}`")
            lines.append(f"- Required spans: `{item.get('required_spans', [])}`")
            if item.get("source_document_id"):
                lines.append(f"- Source document ID: `{item.get('source_document_id')}`")
            lines.append("")
            lines.append(item.get("highlighted_text", ""))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic audit packets for eval datasets.")
    parser.add_argument("--dataset-root", action="append", required=True, help="Dataset root to audit (repeatable).")
    parser.add_argument("--output-dir", default="./Evals/_audit", help="Directory for generated audit packets.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed for sample selection.")
    parser.add_argument("--query-sample", type=int, default=50, help="Standard query sample size.")
    parser.add_argument("--document-sample", type=int, default=25, help="Document sample size.")
    parser.add_argument("--storage-sample", type=int, default=50, help="Storage-case sample size.")
    parser.add_argument("--adversarial-sample", type=int, default=10, help="Hard-query sample size.")
    parser.add_argument("--longmemeval-query-sample", type=int, default=75, help="Query sample size for longmemeval-like datasets.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    index_lines = ["# Audit Packets", ""]

    for raw_root in args.dataset_root:
        dataset_root = Path(raw_root).resolve()
        dataset_name = dataset_root.name
        packet_dir = output_dir / dataset_name
        packet_dir.mkdir(parents=True, exist_ok=True)

        documents = load_jsonl(dataset_root / "recall_documents.jsonl")
        queries = load_jsonl(dataset_root / "recall_queries.jsonl")
        storage_cases = load_jsonl(dataset_root / "storage_cases.jsonl")
        review_rows = load_jsonl(dataset_root / "review_queue.jsonl")
        docs_by_id = {str(row.get("id", "")): row for row in documents}
        queries_by_id = {str(row.get("id", "")): row for row in queries}
        storage_by_id = {str(row.get("id", "")): row for row in storage_cases}
        storage_by_source_doc_id = {
            str(row.get("source_document_id", "")): row
            for row in storage_cases
            if str(row.get("source_document_id", "")).strip()
        }

        packet_items = build_review_items(
            dataset_name,
            review_rows,
            docs_by_id,
            queries_by_id,
            storage_by_id,
            storage_by_source_doc_id,
        )

        selected_query_ids = {item["source_id"] for item in packet_items if item["entry_type"] == "query"}
        selected_doc_ids = {item["source_id"] for item in packet_items if item["entry_type"] == "document"}
        selected_storage_ids = {item["source_id"] for item in packet_items if item["entry_type"] == "storage"}

        is_longmemeval = "longmemeval" in dataset_name.lower()
        query_target = args.longmemeval_query_sample if is_longmemeval else args.query_sample
        sampled_queries = sample_query_rows(
            dataset_name,
            queries,
            target=query_target,
            seed=args.seed,
            exclude_ids=selected_query_ids,
            include_hard=is_longmemeval,
        )
        for row in sampled_queries:
            packet_items.append(build_query_item(dataset_name, row, docs_by_id, sample_reason="query_sample"))
            selected_query_ids.add(str(row.get("id", "")))

        if not is_longmemeval:
            hard_queries = [row for row in queries if str(row.get("difficulty", "")).strip().lower() == "hard"]
            sampled_hard = sample_query_rows(
                dataset_name,
                hard_queries,
                target=args.adversarial_sample,
                seed=args.seed + 101,
                exclude_ids=selected_query_ids,
                include_hard=True,
            )
            for row in sampled_hard:
                packet_items.append(build_query_item(dataset_name, row, docs_by_id, sample_reason="adversarial_sample"))
                selected_query_ids.add(str(row.get("id", "")))

            sampled_documents = sample_document_rows(
                dataset_name,
                documents,
                target=args.document_sample,
                seed=args.seed,
                exclude_ids=selected_doc_ids,
            )
            for row in sampled_documents:
                packet_items.append(build_document_item(dataset_name, row, sample_reason="document_sample"))
                selected_doc_ids.add(str(row.get("id", "")))

            sampled_storage = sample_storage_rows(
                dataset_name,
                storage_cases,
                target=args.storage_sample,
                seed=args.seed,
                exclude_ids=selected_storage_ids,
            )
            for row in sampled_storage:
                packet_items.append(build_storage_item(dataset_name, row, sample_reason="storage_sample"))
                selected_storage_ids.add(str(row.get("id", "")))

        packet_items.sort(key=lambda item: (str(item.get("entry_type", "")), str(item.get("source_id", ""))))

        packet_jsonl_path = packet_dir / "packet.jsonl"
        packet_md_path = packet_dir / "packet.md"
        manifest_path = packet_dir / "packet_manifest.json"
        write_jsonl_atomic(packet_jsonl_path, packet_items)
        packet_md_path.write_text(render_markdown(packet_items, dataset_name, generated_at), encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "generated_at": generated_at,
                    "seed": args.seed,
                    "counts": {
                        "packet_items": len(packet_items),
                        "review_queue_items": len(review_rows),
                        "query_samples": query_target,
                        "document_samples": 0 if is_longmemeval else args.document_sample,
                        "storage_samples": 0 if is_longmemeval else args.storage_sample,
                        "adversarial_samples": 0 if is_longmemeval else args.adversarial_sample,
                    },
                    "packet_files": {
                        "jsonl": str(packet_jsonl_path),
                        "markdown": str(packet_md_path),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        index_lines.append(f"- `{dataset_name}`: `{packet_jsonl_path}`")

    (output_dir / "README.md").write_text("\n".join(index_lines).rstrip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
