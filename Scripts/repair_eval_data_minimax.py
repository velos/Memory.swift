#!/usr/bin/env python3
"""Repair staged eval datasets using MiniMax."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval_data_codex_support import (
    ensure_json_object_with_list,
    extract_json_payload,
    load_jsonl,
    log,
    normalize_spaces,
    truncate_for_log,
    write_jsonl_atomic,
)
from generate_eval_data_minimax import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MiniMaxAnthropicClient,
    load_env_file,
    resolve_api_key,
)
from tag_eval_data_codex import (
    DIFFICULTY_LEVELS,
    MEMORY_TYPES,
    build_storage_record,
    chunked,
    clear_review_entry,
    load_review_queue,
    next_id_number,
    sanitize_storage_proposal,
    write_review_queue,
)

QUERY_REPAIR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "action", "notes"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "action": {"type": "string", "enum": ["keep", "drop"]},
                    "query": {"type": ["string", "null"]},
                    "memory_types": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "difficulty": {
                        "type": ["string", "null"],
                        "enum": DIFFICULTY_LEVELS + [None],
                    },
                    "relevant_document_ids": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "notes": {"type": "string"},
                },
            },
        }
    },
}

STORAGE_REPAIR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "action", "notes"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "action": {"type": "string", "enum": ["keep", "drop"]},
                    "expected_memory_type": {
                        "type": ["string", "null"],
                        "enum": MEMORY_TYPES + [None],
                    },
                    "required_spans": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "notes": {"type": "string"},
                },
            },
        }
    },
}


def make_client(
    *,
    env_file: Path,
    base_url: Optional[str],
    model: Optional[str],
    timeout_seconds: int,
    max_retries_per_request: int,
) -> MiniMaxAnthropicClient:
    load_env_file(env_file)
    api_key = resolve_api_key()
    resolved_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL
    resolved_model = model or os.environ.get("MINIMAX_MODEL") or DEFAULT_MODEL
    return MiniMaxAnthropicClient(
        api_key=api_key,
        base_url=resolved_base_url,
        model=resolved_model,
        max_retries_per_request=max_retries_per_request,
        timeout_seconds=timeout_seconds,
    )


def load_id_filter(path: Optional[Path]) -> Optional[set[str]]:
    if path is None or not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return set()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {line.strip() for line in raw.splitlines() if line.strip()}
    if isinstance(parsed, list):
        return {str(item).strip() for item in parsed if str(item).strip()}
    return {str(parsed).strip()} if str(parsed).strip() else set()


def minimax_items(
    client: MiniMaxAnthropicClient,
    *,
    system_prompt: str,
    user_prompt: str,
    output_schema: Dict[str, Any],
    progress_label: str,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    raw = client.create_message(
        system_prompt=(
            "You are repairing evaluation data.\n"
            "Return JSON only.\n"
            "Do not use markdown fences.\n\n"
            f"{system_prompt}\n\n"
            "Expected output JSON schema:\n"
            f"{output_schema}"
        ),
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=max_tokens,
        progress_label=progress_label,
    )
    payload = extract_json_payload(raw)
    return ensure_json_object_with_list(payload, "items")


def trim_text(text: str, limit: int) -> str:
    cleaned = normalize_spaces(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def sanitize_memory_types(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        candidate = str(item).strip().lower()
        if candidate in MEMORY_TYPES and candidate not in result:
            result.append(candidate)
    return result


def sanitize_difficulty(value: Any) -> Optional[str]:
    candidate = str(value).strip().lower()
    if candidate in DIFFICULTY_LEVELS:
        return candidate
    return None


def sanitize_query_repair(
    raw: Dict[str, Any],
    *,
    query_ids: set[str],
    allowed_doc_ids_by_query: Dict[str, set[str]],
    rewrite_query_text: bool,
) -> Optional[Dict[str, Any]]:
    record_id = str(raw.get("id", "")).strip()
    action = str(raw.get("action", "")).strip().lower()
    if record_id not in query_ids or action not in {"keep", "drop"}:
        return None

    result: Dict[str, Any] = {
        "id": record_id,
        "action": action,
        "notes": str(raw.get("notes", "")).strip(),
    }
    if action == "drop":
        return result

    rewritten_query = normalize_spaces(str(raw.get("query", "")))
    memory_types = sanitize_memory_types(raw.get("memory_types"))
    difficulty = sanitize_difficulty(raw.get("difficulty"))
    relevant_raw = raw.get("relevant_document_ids")
    if not memory_types or not difficulty or not isinstance(relevant_raw, list):
        return None
    if rewrite_query_text:
        word_count = len(rewritten_query.split())
        if word_count < 3 or word_count > 32:
            return None

    allowed_doc_ids = allowed_doc_ids_by_query.get(record_id, set())
    relevant_document_ids: List[str] = []
    for value in relevant_raw:
        doc_id = str(value).strip()
        if doc_id in allowed_doc_ids and doc_id not in relevant_document_ids:
            relevant_document_ids.append(doc_id)
    if not relevant_document_ids:
        return None

    result["memory_types"] = memory_types[:3]
    result["difficulty"] = difficulty
    result["relevant_document_ids"] = relevant_document_ids
    if rewrite_query_text:
        result["query"] = rewritten_query
    return result


def sanitize_storage_repair(
    raw: Dict[str, Any],
    *,
    doc_ids: set[str],
    text_by_doc_id: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    record_id = str(raw.get("id", "")).strip()
    action = str(raw.get("action", "")).strip().lower()
    if record_id not in doc_ids or action not in {"keep", "drop"}:
        return None
    if action == "drop":
        return {
            "id": record_id,
            "action": "drop",
            "notes": str(raw.get("notes", "")).strip(),
        }

    proposal = sanitize_storage_proposal(
        {
            "id": record_id,
            "expected_memory_type": raw.get("expected_memory_type"),
            "required_spans": raw.get("required_spans"),
        },
        text_by_doc_id.get(record_id, ""),
    )
    if not proposal:
        return None
    proposal["action"] = "keep"
    proposal["notes"] = str(raw.get("notes", "")).strip()
    return proposal


def build_query_batch(batch: Sequence[Dict[str, Any]], docs_by_id: Dict[str, Dict[str, Any]]) -> str:
    rows: List[str] = []
    for query in batch:
        docs = []
        for doc_id in query.get("relevant_document_ids", []):
            doc = docs_by_id.get(str(doc_id))
            if not doc:
                continue
            docs.append(
                {
                    "id": str(doc["id"]),
                    "relative_path": str(doc.get("relative_path", "")),
                    "memory_type": doc.get("memory_type"),
                    "text_snippet": trim_text(str(doc.get("text", "")), 600),
                }
            )
        rows.append(
            json.dumps(
                {
                    "id": str(query["id"]),
                    "query": str(query.get("query", "")),
                    "current_memory_types": list(query.get("memory_types", [])),
                    "current_difficulty": query.get("difficulty"),
                    "relevant_document_ids": list(query.get("relevant_document_ids", [])),
                    "relevant_documents": docs,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(rows)


def build_storage_batch(batch: Sequence[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for document in batch:
        rows.append(
            json.dumps(
                {
                    "id": str(document["id"]),
                    "relative_path": str(document.get("relative_path", "")),
                    "current_memory_type": document.get("memory_type"),
                    "text": str(document.get("text", "")),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(rows)


def repair_queries(
    client: MiniMaxAnthropicClient,
    *,
    dataset_root: Path,
    batch_size: int,
    max_records: Optional[int],
    max_tokens: int,
    id_filter: Optional[set[str]],
    rewrite_query_text: bool,
) -> Dict[str, Any]:
    docs_path = dataset_root / "recall_documents.jsonl"
    queries_path = dataset_root / "recall_queries.jsonl"
    documents = load_jsonl(docs_path)
    queries = load_jsonl(queries_path)
    docs_by_id = {str(doc["id"]): doc for doc in documents}
    dataset_name = dataset_root.name

    pending = list(queries)
    if id_filter is not None:
        pending = [query for query in pending if str(query["id"]) in id_filter]
    if max_records is not None:
        pending = pending[:max_records]

    allowed_doc_ids_by_query = {
        str(query["id"]): {str(doc_id) for doc_id in query.get("relevant_document_ids", [])}
        for query in pending
    }
    updates: Dict[str, Dict[str, Any]] = {}
    dropped_ids: set[str] = set()
    unresolved_ids: List[str] = []

    system_prompt = (
        "Repair retrieval benchmark queries for Memory.swift.\n"
        "Keep a query only when at least one provided relevant document directly supports answering it.\n"
        "If you keep a query, return corrected memory_types, difficulty, and a non-empty subset of the provided relevant_document_ids.\n"
        "Drop vague, contextless, off-topic, or unanswerable queries.\n"
        "Never invent new document ids and never refer to documents not provided in the item."
    )
    if rewrite_query_text:
        system_prompt += (
            "\nRewrite kept queries into short search-style queries that an engineer would realistically type."
            "\nKeep them concise, specific, and natural."
            "\nDo not paste issue bodies, stack traces, code blocks, or long descriptions into the rewritten query."
            "\nTarget roughly 6-18 words, but prioritize a clean realistic search query over exact word count."
        )

    for batch_index, batch in enumerate(chunked(pending, batch_size), start=1):
        user_prompt = (
            "Return one item per query id.\n"
            f"Allowed memory types: {', '.join(MEMORY_TYPES)}.\n"
            f"Allowed difficulty values: {', '.join(DIFFICULTY_LEVELS)}.\n"
            "Use action=drop when the query is broken or none of the provided documents support it.\n\n"
            "Queries (JSON lines):\n"
            f"{build_query_batch(batch, docs_by_id)}"
        )
        if rewrite_query_text:
            user_prompt = (
                "Return one item per query id.\n"
                f"Allowed memory types: {', '.join(MEMORY_TYPES)}.\n"
                f"Allowed difficulty values: {', '.join(DIFFICULTY_LEVELS)}.\n"
                "Use action=drop when the query is broken or none of the provided documents support it.\n"
                "For kept items, rewrite the query into a concise search query.\n"
                "Good rewrites are short, issue-focused, and realistic for code or bug search.\n"
                "Avoid filler words, long quotations, stack traces, environment dumps, and exact issue-body restatements.\n\n"
                "Queries (JSON lines):\n"
                f"{build_query_batch(batch, docs_by_id)}"
            )
        try:
            rows = minimax_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=QUERY_REPAIR_SCHEMA,
                progress_label=f"[repair:queries] {dataset_name} batch {batch_index}/{(len(pending) + batch_size - 1) // batch_size}",
                max_tokens=max_tokens,
            )
        except Exception as exc:
            log(f"[repair:queries] {dataset_name} batch {batch_index} failed: {truncate_for_log(str(exc))}")
            unresolved_ids.extend(str(query["id"]) for query in batch)
            continue

        seen_ids: set[str] = set()
        batch_query_ids = {str(query["id"]) for query in batch}
        batch_allowed_doc_ids_by_query = {
            query_id: allowed_doc_ids_by_query[query_id]
            for query_id in batch_query_ids
        }
        for raw in rows:
            repair = sanitize_query_repair(
                raw,
                query_ids=batch_query_ids,
                allowed_doc_ids_by_query=batch_allowed_doc_ids_by_query,
                rewrite_query_text=rewrite_query_text,
            )
            if not repair:
                continue
            query_id = repair["id"]
            seen_ids.add(query_id)
            if repair["action"] == "drop":
                dropped_ids.add(query_id)
                continue
            updates[query_id] = repair

        for query in batch:
            query_id = str(query["id"])
            if query_id not in seen_ids:
                unresolved_ids.append(query_id)

    kept_queries: List[Dict[str, Any]] = []
    updated_count = 0
    rewritten_count = 0
    for query in queries:
        query_id = str(query["id"])
        if query_id in dropped_ids:
            continue
        repair = updates.get(query_id)
        if repair:
            previous_query = str(query.get("query", ""))
            query["memory_types"] = repair["memory_types"]
            query["difficulty"] = repair["difficulty"]
            query["relevant_document_ids"] = repair["relevant_document_ids"]
            if rewrite_query_text and repair.get("query"):
                query["query"] = repair["query"]
                if normalize_spaces(previous_query) != normalize_spaces(repair["query"]):
                    rewritten_count += 1
            updated_count += 1
        kept_queries.append(query)

    write_jsonl_atomic(queries_path, kept_queries)
    report = {
        "dataset": dataset_name,
        "mode": "queries",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "input_queries": len(queries),
            "processed_queries": len(pending),
            "kept_queries": len(kept_queries),
            "dropped_queries": len(dropped_ids),
            "updated_queries": updated_count,
            "rewritten_queries": rewritten_count,
            "unresolved_queries": len(sorted(set(unresolved_ids))),
        },
        "dropped_query_ids": sorted(dropped_ids),
        "unresolved_query_ids": sorted(set(unresolved_ids)),
    }
    (dataset_root / "repair_report.queries.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def repair_storage_review_queue(
    client: MiniMaxAnthropicClient,
    *,
    dataset_root: Path,
    batch_size: int,
    max_records: Optional[int],
    max_tokens: int,
    id_filter: Optional[set[str]],
) -> Dict[str, Any]:
    docs_path = dataset_root / "recall_documents.jsonl"
    storage_path = dataset_root / "storage_cases.jsonl"
    review_path = dataset_root / "review_queue.jsonl"

    documents = load_jsonl(docs_path)
    storage_cases = load_jsonl(storage_path)
    review_map = load_review_queue(review_path)

    docs_by_id = {str(doc["id"]): doc for doc in documents}
    existing_by_source = {
        str(row.get("source_document_id", "")): row
        for row in storage_cases
        if str(row.get("source_document_id", ""))
    }
    next_storage_id = next_id_number(storage_cases, "storage")

    pending_docs: List[Dict[str, Any]] = []
    for key, review_entry in sorted(review_map.items()):
        mode, record_id = key
        if mode != "storage-cases":
            continue
        if id_filter is not None and record_id not in id_filter:
            continue
        document = docs_by_id.get(record_id)
        if document:
            pending_docs.append(document)

    if max_records is not None:
        pending_docs = pending_docs[:max_records]

    doc_ids = {str(doc["id"]) for doc in pending_docs}
    text_by_doc_id = {str(doc["id"]): str(doc.get("text", "")) for doc in pending_docs}
    resolved_ids: List[str] = []
    unresolved_ids: List[str] = []

    system_prompt = (
        "Repair storage evaluation cases for Memory.swift.\n"
        "For each document, either produce a valid storage case or drop it.\n"
        "A valid storage case needs one dominant memory type and 2-4 short verbatim required_spans copied exactly from the document text.\n"
        "Prefer precise spans over long sentences."
    )

    for batch_index, batch in enumerate(chunked(pending_docs, batch_size), start=1):
        user_prompt = (
            "Return one item per document id.\n"
            f"Allowed memory types: {', '.join(MEMORY_TYPES)}.\n"
            "Use action=drop only when the document cannot support a reliable storage case.\n\n"
            "Documents (JSON lines):\n"
            f"{build_storage_batch(batch)}"
        )
        try:
            rows = minimax_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=STORAGE_REPAIR_SCHEMA,
                progress_label=f"[repair:storage] {dataset_root.name} batch {batch_index}/{(len(pending_docs) + batch_size - 1) // batch_size}",
                max_tokens=max_tokens,
            )
        except Exception as exc:
            log(f"[repair:storage] {dataset_root.name} batch {batch_index} failed: {truncate_for_log(str(exc))}")
            unresolved_ids.extend(str(document["id"]) for document in batch)
            continue

        seen_ids: set[str] = set()
        batch_doc_ids = {str(document["id"]) for document in batch}
        batch_text_by_doc_id = {
            str(document["id"]): text_by_doc_id[str(document["id"])]
            for document in batch
        }
        for raw in rows:
            repair = sanitize_storage_repair(raw, doc_ids=batch_doc_ids, text_by_doc_id=batch_text_by_doc_id)
            if not repair:
                continue
            doc_id = repair["id"]
            seen_ids.add(doc_id)
            if repair["action"] == "drop":
                unresolved_ids.append(doc_id)
                continue

            document = docs_by_id[doc_id]
            existing_row = existing_by_source.get(doc_id)
            new_row = build_storage_record(document, repair, existing_row, next_storage_id=next_storage_id)
            if existing_row:
                index = storage_cases.index(existing_row)
                storage_cases[index] = new_row
            else:
                storage_cases.append(new_row)
                next_storage_id += 1
            existing_by_source[doc_id] = new_row
            clear_review_entry(review_map, "storage-cases", doc_id)
            resolved_ids.append(doc_id)

        for document in batch:
            doc_id = str(document["id"])
            if doc_id not in seen_ids and doc_id not in unresolved_ids:
                unresolved_ids.append(doc_id)

    write_jsonl_atomic(storage_path, storage_cases)
    write_review_queue(review_path, review_map)
    report = {
        "dataset": dataset_root.name,
        "mode": "storage-review",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "pending_review_docs": len(pending_docs),
            "resolved_docs": len(resolved_ids),
            "unresolved_docs": len(sorted(set(unresolved_ids))),
            "storage_cases_total": len(storage_cases),
            "review_queue_total": len(review_map),
        },
        "resolved_doc_ids": sorted(resolved_ids),
        "unresolved_doc_ids": sorted(set(unresolved_ids)),
    }
    (dataset_root / "repair_report.storage_review.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair staged eval datasets using MiniMax.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing eval JSONL files.")
    parser.add_argument("--mode", required=True, choices=["queries", "storage-review"])
    parser.add_argument("--env-file", default=".env", help="Path to .env file.")
    parser.add_argument("--base-url", default=None, help="Anthropic-compatible base URL.")
    parser.add_argument("--model", default=None, help="MiniMax model name.")
    parser.add_argument("--batch-size", type=int, default=8, help="Items per API batch.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on processed records.")
    parser.add_argument("--request-timeout-seconds", type=int, default=180, help="Per-request HTTP timeout in seconds.")
    parser.add_argument("--max-retries-per-request", type=int, default=2, help="HTTP retries per API request.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per API response.")
    parser.add_argument("--id-file", default=None, help="Optional file with record ids to process (one per line or JSON array).")
    parser.add_argument("--rewrite-query-text", action="store_true", help="Rewrite kept queries into concise search-style queries.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    id_filter = load_id_filter(Path(args.id_file).resolve()) if args.id_file else None
    client = make_client(
        env_file=Path(args.env_file).resolve(),
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.request_timeout_seconds,
        max_retries_per_request=args.max_retries_per_request,
    )

    if args.mode == "queries":
        report = repair_queries(
            client,
            dataset_root=dataset_root,
            batch_size=args.batch_size,
            max_records=args.max_records,
            max_tokens=args.max_tokens,
            id_filter=id_filter,
            rewrite_query_text=args.rewrite_query_text,
        )
    else:
        report = repair_storage_review_queue(
            client,
            dataset_root=dataset_root,
            batch_size=args.batch_size,
            max_records=args.max_records,
            max_tokens=args.max_tokens,
            id_filter=id_filter,
        )

    log(
        f"[repair] {dataset_root.name} {args.mode}: "
        f"{json.dumps(report['counts'], ensure_ascii=False, sort_keys=True)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
