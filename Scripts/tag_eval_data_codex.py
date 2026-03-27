#!/usr/bin/env python3
"""Tag and augment Memory.swift eval datasets via Codex."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from eval_data_codex_support import (
    CodexClient,
    DEFAULT_MODEL,
    ensure_json_object_with_list,
    extract_json_payload,
    load_jsonl,
    load_manifest,
    log,
    normalize_spaces,
    truncate_for_log,
    write_jsonl_atomic,
    write_manifest,
)

MEMORY_TYPES: List[str] = [
    "factual",
    "procedural",
    "episodic",
    "semantic",
    "emotional",
    "social",
    "contextual",
    "temporal",
]
DIFFICULTY_LEVELS: List[str] = ["easy", "medium", "hard"]
PROMPT_VERSION = "2026-03-15-codex-tagging-v1"
REVIEW_QUEUE_FILENAME = "review_queue.jsonl"

QUERY_TAG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "memory_types", "difficulty", "confidence"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "memory_types": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string", "enum": MEMORY_TYPES},
                    },
                    "difficulty": {"type": "string", "enum": DIFFICULTY_LEVELS},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        }
    },
}

DOCUMENT_TAG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "memory_type", "confidence"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "memory_type": {"type": "string", "enum": MEMORY_TYPES},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        }
    },
}

STORAGE_CASE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "expected_memory_type", "required_spans", "confidence"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "expected_memory_type": {"type": "string", "enum": MEMORY_TYPES},
                    "required_spans": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 4,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        }
    },
}

ADVERSARIAL_QUERY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query", "relevant_document_ids", "memory_types", "difficulty"],
                "properties": {
                    "query": {"type": "string", "minLength": 8},
                    "relevant_document_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "memory_types": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string", "enum": MEMORY_TYPES},
                    },
                    "difficulty": {"type": "string", "enum": ["hard"]},
                },
            },
        }
    },
}


def chunked(items: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def trim_text(text: str, limit: int) -> str:
    cleaned = normalize_spaces(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "item"


def parse_prefixed_id(value: str, prefix: str) -> Optional[int]:
    match = re.fullmatch(rf"{re.escape(prefix)}-(\d+)", value)
    if not match:
        return None
    return int(match.group(1))


def next_id_number(records: Sequence[Dict[str, Any]], prefix: str) -> int:
    max_seen = 0
    for record in records:
        raw_id = str(record.get("id", "")).strip()
        parsed = parse_prefixed_id(raw_id, prefix)
        if parsed is not None and parsed > max_seen:
            max_seen = parsed
    return max_seen + 1


def load_review_queue(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    review_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in load_jsonl(path):
        mode = str(row.get("mode", "")).strip()
        record_id = str(row.get("record_id", "")).strip()
        if not mode or not record_id:
            continue
        review_map[(mode, record_id)] = row
    return review_map


def write_review_queue(path: Path, review_map: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    rows = sorted(review_map.values(), key=lambda row: (str(row.get("mode", "")), str(row.get("record_id", ""))))
    write_jsonl_atomic(path, rows)


def add_review_entry(
    review_map: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    mode: str,
    record_id: str,
    reason: str,
    source: Dict[str, Any],
    proposal_a: Optional[Dict[str, Any]] = None,
    proposal_b: Optional[Dict[str, Any]] = None,
) -> None:
    review_map[(mode, record_id)] = {
        "mode": mode,
        "record_id": record_id,
        "reason": reason,
        "source": source,
        "proposal_a": proposal_a,
        "proposal_b": proposal_b,
        "prompt_version": PROMPT_VERSION,
        "status": "needs_review",
    }


def clear_review_entry(review_map: Dict[Tuple[str, str], Dict[str, Any]], mode: str, record_id: str) -> None:
    review_map.pop((mode, record_id), None)


def merge_manifest_defaults(manifest: Dict[str, Any], *, model: str, backend_mode: str) -> Dict[str, Any]:
    merged = dict(manifest)
    merged.setdefault("license_scope", "commercial_safe")
    merged.setdefault("promotion_status", "draft")
    merged.setdefault("source_datasets", [])
    merged.setdefault("audit", {})
    if not isinstance(merged["audit"], dict):
        merged["audit"] = {}
    merged["audit"].setdefault(
        "sample_sizes",
        {
            "query_tags": 50,
            "document_tags": 25,
            "storage_cases": 50,
        },
    )
    merged["tagging"] = {
        "backend": "codex_cli",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "auto_accept_policy": backend_mode,
    }
    return merged


def sanitize_memory_types(value: Any) -> Optional[List[str]]:
    if not isinstance(value, list):
        return None
    filtered: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        candidate = item.strip().lower()
        if candidate in MEMORY_TYPES and candidate not in filtered:
            filtered.append(candidate)
    return filtered or None


def sanitize_difficulty(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if candidate in DIFFICULTY_LEVELS:
        return candidate
    return None


def sanitize_query_tag(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    record_id = str(raw.get("id", "")).strip()
    memory_types = sanitize_memory_types(raw.get("memory_types"))
    difficulty = sanitize_difficulty(raw.get("difficulty"))
    confidence = str(raw.get("confidence", "")).strip().lower()
    if not record_id or not memory_types or not difficulty:
        return None
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "id": record_id,
        "memory_types": memory_types,
        "difficulty": difficulty,
        "confidence": confidence,
    }


def sanitize_document_tag(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    record_id = str(raw.get("id", "")).strip()
    memory_type = str(raw.get("memory_type", "")).strip().lower()
    confidence = str(raw.get("confidence", "")).strip().lower()
    if not record_id or memory_type not in MEMORY_TYPES:
        return None
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "id": record_id,
        "memory_type": memory_type,
        "confidence": confidence,
    }


def sanitize_storage_proposal(raw: Dict[str, Any], source_text: str) -> Optional[Dict[str, Any]]:
    record_id = str(raw.get("id", "")).strip()
    memory_type = str(raw.get("expected_memory_type", "")).strip().lower()
    confidence = str(raw.get("confidence", "")).strip().lower()
    spans_raw = raw.get("required_spans")
    if not record_id or memory_type not in MEMORY_TYPES or not isinstance(spans_raw, list):
        return None
    spans: List[str] = []
    for span in spans_raw:
        if not isinstance(span, str):
            continue
        cleaned = span.strip()
        if cleaned and cleaned in source_text and cleaned not in spans:
            spans.append(cleaned)
    if len(spans) < 2:
        return None
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "id": record_id,
        "expected_memory_type": memory_type,
        "required_spans": spans[:4],
        "confidence": confidence,
    }


def sanitize_adversarial_query(raw: Dict[str, Any], *, valid_doc_ids: set[str]) -> Optional[Dict[str, Any]]:
    query = normalize_spaces(str(raw.get("query", "")))
    if len(query) < 8:
        return None
    relevant_raw = raw.get("relevant_document_ids")
    memory_types = sanitize_memory_types(raw.get("memory_types"))
    difficulty = sanitize_difficulty(raw.get("difficulty"))
    if not isinstance(relevant_raw, list) or not memory_types or difficulty != "hard":
        return None
    relevant: List[str] = []
    for item in relevant_raw:
        if not isinstance(item, str):
            continue
        doc_id = item.strip()
        if doc_id in valid_doc_ids and doc_id not in relevant:
            relevant.append(doc_id)
    if not relevant:
        return None
    return {
        "query": query,
        "relevant_document_ids": relevant[:3],
        "memory_types": memory_types[:3],
        "difficulty": "hard",
    }


def build_query_tag_batch(batch: Sequence[Dict[str, Any]], docs_by_id: Dict[str, Dict[str, Any]]) -> str:
    rows: List[str] = []
    for query in batch:
        docs = []
        for doc_id in query.get("relevant_document_ids", []):
            doc = docs_by_id.get(str(doc_id))
            if not doc:
                continue
            docs.append(
                {
                    "id": doc["id"],
                    "relative_path": doc.get("relative_path", ""),
                    "text_snippet": trim_text(str(doc.get("text", "")), 900),
                }
            )
        rows.append(
            json.dumps(
                {
                    "id": query["id"],
                    "query": query["query"],
                    "relevant_documents": docs,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(rows)


def build_document_tag_batch(batch: Sequence[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for document in batch:
        rows.append(
            json.dumps(
                {
                    "id": document["id"],
                    "relative_path": document.get("relative_path", ""),
                    "text": trim_text(str(document.get("text", "")), 1300),
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
                    "id": document["id"],
                    "relative_path": document.get("relative_path", ""),
                    "text": str(document.get("text", "")),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(rows)


def build_adversarial_batch(
    candidate_docs: Sequence[Dict[str, Any]],
    existing_queries: Sequence[Dict[str, Any]],
    batch_size: int,
) -> str:
    doc_rows: List[str] = []
    for doc in candidate_docs:
        doc_rows.append(
            json.dumps(
                {
                    "id": doc["id"],
                    "relative_path": doc.get("relative_path", ""),
                    "memory_type": doc.get("memory_type"),
                    "summary": trim_text(str(doc.get("text", "")), 500),
                },
                ensure_ascii=False,
            )
        )

    query_rows: List[str] = []
    for query in existing_queries[:20]:
        query_rows.append(
            json.dumps(
                {
                    "query": query.get("query", ""),
                    "relevant_document_ids": query.get("relevant_document_ids", []),
                },
                ensure_ascii=False,
            )
        )

    return (
        f"Generate EXACTLY {batch_size} hard adversarial retrieval queries.\n\n"
        "Candidate documents (JSON lines):\n"
        f"{chr(10).join(doc_rows)}\n\n"
        "Example existing queries to avoid duplicating (JSON lines):\n"
        f"{chr(10).join(query_rows)}"
    )


def codex_items(
    client: CodexClient,
    *,
    system_prompt: str,
    user_prompt: str,
    output_schema: Dict[str, Any],
    progress_label: str,
) -> List[Dict[str, Any]]:
    raw = client.create_message(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_schema=output_schema,
        progress_label=progress_label,
    )
    payload = extract_json_payload(raw)
    return ensure_json_object_with_list(payload, "items")


def tag_queries(
    client: CodexClient,
    dataset_root: Path,
    batch_size: int,
    review_path: Path,
    max_records: Optional[int],
    retag: bool,
) -> None:
    queries_path = dataset_root / "recall_queries.jsonl"
    docs_path = dataset_root / "recall_documents.jsonl"
    if not queries_path.exists() or not docs_path.exists():
        raise RuntimeError("query-tags mode requires recall_queries.jsonl and recall_documents.jsonl.")

    queries = load_jsonl(queries_path)
    docs = load_jsonl(docs_path)
    docs_by_id = {str(doc["id"]): doc for doc in docs}
    review_map = load_review_queue(review_path)

    pending: List[Dict[str, Any]] = []
    for query in queries:
        has_types = bool(sanitize_memory_types(query.get("memory_types")))
        has_difficulty = sanitize_difficulty(query.get("difficulty")) is not None
        if retag or not (has_types and has_difficulty):
            pending.append(query)

    if max_records is not None:
        pending = pending[:max_records]

    if not pending:
        log("No queries require tagging.")
        return

    system_prompt = (
        "You tag retrieval benchmark queries for Memory.swift. "
        "Infer one to three memory types that best describe the retrieval intent. "
        "Difficulty should reflect retrieval challenge, not answer complexity."
    )

    for batch_index, batch in enumerate(chunked(pending, batch_size), start=1):
        user_payload = build_query_tag_batch(batch, docs_by_id)
        user_prompt = (
            "Return one output item per input query id.\n"
            "Use only these memory types: factual, procedural, episodic, semantic, emotional, social, contextual, temporal.\n"
            "Difficulty must be easy, medium, or hard.\n"
            "Favor the smallest correct set of memory_types.\n\n"
            "Input queries (JSON lines):\n"
            f"{user_payload}"
        )

        try:
            pass_a = codex_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=QUERY_TAG_SCHEMA,
                progress_label=f"[query-tags] batch {batch_index} pass 1",
            )
            pass_b = codex_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=QUERY_TAG_SCHEMA,
                progress_label=f"[query-tags] batch {batch_index} pass 2",
            )
        except Exception as exc:
            for query in batch:
                add_review_entry(
                    review_map,
                    mode="query-tags",
                    record_id=str(query["id"]),
                    reason="request_failure",
                    source={"query": query.get("query", "")},
                )
            write_review_queue(review_path, review_map)
            log(f"[query-tags] batch {batch_index} failed: {truncate_for_log(str(exc))}")
            continue

        proposals_a = {item["id"]: item for item in (sanitize_query_tag(row) for row in pass_a) if item}
        proposals_b = {item["id"]: item for item in (sanitize_query_tag(row) for row in pass_b) if item}

        updated = 0
        for query in batch:
            query_id = str(query["id"])
            proposal_a = proposals_a.get(query_id)
            proposal_b = proposals_b.get(query_id)
            if not proposal_a or not proposal_b:
                add_review_entry(
                    review_map,
                    mode="query-tags",
                    record_id=query_id,
                    reason="invalid_or_missing_proposal",
                    source={"query": query.get("query", "")},
                    proposal_a=proposal_a,
                    proposal_b=proposal_b,
                )
                continue

            if (
                proposal_a["difficulty"] == proposal_b["difficulty"]
                and proposal_a["memory_types"] == proposal_b["memory_types"]
            ):
                query["memory_types"] = proposal_a["memory_types"]
                query["difficulty"] = proposal_a["difficulty"]
                clear_review_entry(review_map, "query-tags", query_id)
                updated += 1
            else:
                add_review_entry(
                    review_map,
                    mode="query-tags",
                    record_id=query_id,
                    reason="pass_disagreement",
                    source={"query": query.get("query", "")},
                    proposal_a=proposal_a,
                    proposal_b=proposal_b,
                )

        write_jsonl_atomic(queries_path, queries)
        write_review_queue(review_path, review_map)
        log(f"[query-tags] batch {batch_index}: updated {updated}/{len(batch)}")


def tag_documents(
    client: CodexClient,
    dataset_root: Path,
    batch_size: int,
    review_path: Path,
    max_records: Optional[int],
    retag: bool,
) -> None:
    docs_path = dataset_root / "recall_documents.jsonl"
    if not docs_path.exists():
        raise RuntimeError("document-tags mode requires recall_documents.jsonl.")

    documents = load_jsonl(docs_path)
    review_map = load_review_queue(review_path)
    pending: List[Dict[str, Any]] = []
    for document in documents:
        current = str(document.get("memory_type", "")).strip().lower()
        if retag or current not in MEMORY_TYPES:
            pending.append(document)

    if max_records is not None:
        pending = pending[:max_records]

    if not pending:
        log("No documents require tagging.")
        return

    system_prompt = (
        "You tag retrieval corpus documents for Memory.swift. "
        "Choose the single dominant memory type a query would most likely filter on."
    )

    for batch_index, batch in enumerate(chunked(pending, batch_size), start=1):
        user_prompt = (
            "Return one output item per input document id.\n"
            "Choose the single dominant memory type from: factual, procedural, episodic, semantic, emotional, social, contextual, temporal.\n\n"
            "Input documents (JSON lines):\n"
            f"{build_document_tag_batch(batch)}"
        )

        try:
            rows = codex_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=DOCUMENT_TAG_SCHEMA,
                progress_label=f"[document-tags] batch {batch_index}",
            )
        except Exception as exc:
            for document in batch:
                add_review_entry(
                    review_map,
                    mode="document-tags",
                    record_id=str(document["id"]),
                    reason="request_failure",
                    source={"relative_path": document.get("relative_path", "")},
                )
            write_review_queue(review_path, review_map)
            log(f"[document-tags] batch {batch_index} failed: {truncate_for_log(str(exc))}")
            continue

        proposals = {item["id"]: item for item in (sanitize_document_tag(row) for row in rows) if item}
        updated = 0
        for document in batch:
            proposal = proposals.get(str(document["id"]))
            if not proposal:
                add_review_entry(
                    review_map,
                    mode="document-tags",
                    record_id=str(document["id"]),
                    reason="invalid_or_missing_proposal",
                    source={"relative_path": document.get("relative_path", "")},
                )
                continue

            document["memory_type"] = proposal["memory_type"]
            clear_review_entry(review_map, "document-tags", str(document["id"]))
            updated += 1

        write_jsonl_atomic(docs_path, documents)
        write_review_queue(review_path, review_map)
        log(f"[document-tags] batch {batch_index}: updated {updated}/{len(batch)}")


def prepare_storage_targets(
    documents: Sequence[Dict[str, Any]],
    storage_cases: Sequence[Dict[str, Any]],
    *,
    seed: int,
    max_storage_cases: int,
    retag: bool,
) -> List[Dict[str, Any]]:
    existing_by_source = {
        str(row.get("source_document_id", "")).strip(): row
        for row in storage_cases
        if str(row.get("source_document_id", "")).strip()
    }
    candidates = list(documents)
    rng = random.Random(seed)
    rng.shuffle(candidates)

    pending: List[Dict[str, Any]] = []
    for document in candidates:
        doc_id = str(document.get("id", "")).strip()
        if not doc_id:
            continue
        if not retag and doc_id in existing_by_source:
            continue
        pending.append(document)
        if len(pending) >= max_storage_cases:
            break
    return pending


def coalesce_storage_spans(spans_a: Sequence[str], spans_b: Sequence[str], source_text: str) -> List[str]:
    shared = [span for span in spans_a if span in spans_b and span in source_text]
    if len(shared) >= 2:
        return shared[:4]

    combined: List[str] = []
    for span in list(spans_a) + list(spans_b):
        if span in source_text and span not in combined:
            combined.append(span)
    return combined[:4]


def build_storage_record(
    document: Dict[str, Any],
    proposal: Dict[str, Any],
    existing_row: Optional[Dict[str, Any]],
    *,
    next_storage_id: int,
) -> Dict[str, Any]:
    row_id = str(existing_row.get("id", "")).strip() if existing_row else ""
    if not row_id:
        row_id = f"storage-{next_storage_id:04d}"
    return {
        "id": row_id,
        "kind": "markdown",
        "text": str(document.get("text", "")),
        "expected_memory_type": proposal["expected_memory_type"],
        "required_spans": proposal["required_spans"],
        "source_document_id": str(document.get("id", "")),
    }


def derive_storage_cases(
    client: CodexClient,
    dataset_root: Path,
    batch_size: int,
    review_path: Path,
    max_records: Optional[int],
    retag: bool,
    max_storage_cases: int,
    seed: int,
) -> None:
    docs_path = dataset_root / "recall_documents.jsonl"
    storage_path = dataset_root / "storage_cases.jsonl"
    if not docs_path.exists():
        raise RuntimeError("storage-cases mode requires recall_documents.jsonl.")

    documents = load_jsonl(docs_path)
    storage_cases = load_jsonl(storage_path)
    existing_by_source = {
        str(row.get("source_document_id", "")).strip(): row
        for row in storage_cases
        if str(row.get("source_document_id", "")).strip()
    }
    review_map = load_review_queue(review_path)
    next_storage_id = next_id_number(storage_cases, "storage")

    pending = prepare_storage_targets(
        documents,
        storage_cases,
        seed=seed,
        max_storage_cases=max_storage_cases,
        retag=retag,
    )
    if max_records is not None:
        pending = pending[:max_records]

    if not pending:
        log("No documents require storage-case derivation.")
        return

    system_prompt = (
        "You derive storage evaluation cases for Memory.swift from existing documents. "
        "Choose the dominant memory type and 2-4 verbatim required spans that a good storage pipeline must preserve."
    )

    for batch_index, batch in enumerate(chunked(pending, batch_size), start=1):
        user_prompt = (
            "Return one output item per input document id.\n"
            "required_spans must appear verbatim in the input text, with exact casing.\n"
            "Choose the dominant memory type from: factual, procedural, episodic, semantic, emotional, social, contextual, temporal.\n\n"
            "Input documents (JSON lines):\n"
            f"{build_storage_batch(batch)}"
        )

        try:
            pass_a = codex_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=STORAGE_CASE_SCHEMA,
                progress_label=f"[storage-cases] batch {batch_index} pass 1",
            )
            pass_b = codex_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=STORAGE_CASE_SCHEMA,
                progress_label=f"[storage-cases] batch {batch_index} pass 2",
            )
        except Exception as exc:
            for document in batch:
                add_review_entry(
                    review_map,
                    mode="storage-cases",
                    record_id=str(document["id"]),
                    reason="request_failure",
                    source={"relative_path": document.get("relative_path", "")},
                )
            write_review_queue(review_path, review_map)
            log(f"[storage-cases] batch {batch_index} failed: {truncate_for_log(str(exc))}")
            continue

        batch_text_by_id = {str(document["id"]): str(document.get("text", "")) for document in batch}
        proposals_a = {
            item["id"]: item
            for item in (
                sanitize_storage_proposal(row, batch_text_by_id.get(str(row.get("id", "")), ""))
                for row in pass_a
            )
            if item
        }
        proposals_b = {
            item["id"]: item
            for item in (
                sanitize_storage_proposal(row, batch_text_by_id.get(str(row.get("id", "")), ""))
                for row in pass_b
            )
            if item
        }

        updated = 0
        for document in batch:
            doc_id = str(document["id"])
            proposal_a = proposals_a.get(doc_id)
            proposal_b = proposals_b.get(doc_id)
            if not proposal_a or not proposal_b:
                add_review_entry(
                    review_map,
                    mode="storage-cases",
                    record_id=doc_id,
                    reason="invalid_or_missing_proposal",
                    source={"relative_path": document.get("relative_path", "")},
                    proposal_a=proposal_a,
                    proposal_b=proposal_b,
                )
                continue

            if proposal_a["expected_memory_type"] != proposal_b["expected_memory_type"]:
                add_review_entry(
                    review_map,
                    mode="storage-cases",
                    record_id=doc_id,
                    reason="pass_disagreement",
                    source={"relative_path": document.get("relative_path", "")},
                    proposal_a=proposal_a,
                    proposal_b=proposal_b,
                )
                continue

            shared_spans = coalesce_storage_spans(
                proposal_a["required_spans"],
                proposal_b["required_spans"],
                str(document.get("text", "")),
            )
            if len(shared_spans) < 2:
                add_review_entry(
                    review_map,
                    mode="storage-cases",
                    record_id=doc_id,
                    reason="insufficient_shared_spans",
                    source={"relative_path": document.get("relative_path", "")},
                    proposal_a=proposal_a,
                    proposal_b=proposal_b,
                )
                continue

            accepted = {
                "expected_memory_type": proposal_a["expected_memory_type"],
                "required_spans": shared_spans,
            }
            existing_row = existing_by_source.get(doc_id)
            new_row = build_storage_record(document, accepted, existing_row, next_storage_id=next_storage_id)
            if not existing_row:
                next_storage_id += 1
                storage_cases.append(new_row)
            else:
                index = storage_cases.index(existing_row)
                storage_cases[index] = new_row
            existing_by_source[doc_id] = new_row
            clear_review_entry(review_map, "storage-cases", doc_id)
            updated += 1

        write_jsonl_atomic(storage_path, storage_cases)
        write_review_queue(review_path, review_map)
        log(f"[storage-cases] batch {batch_index}: updated {updated}/{len(batch)}")


def augment_adversarial_queries(
    client: CodexClient,
    dataset_root: Path,
    batch_size: int,
    review_path: Path,
    augment_count: int,
    seed: int,
) -> None:
    docs_path = dataset_root / "recall_documents.jsonl"
    queries_path = dataset_root / "recall_queries.jsonl"
    if not docs_path.exists() or not queries_path.exists():
        raise RuntimeError("adversarial-augment mode requires recall_documents.jsonl and recall_queries.jsonl.")

    documents = load_jsonl(docs_path)
    queries = load_jsonl(queries_path)
    valid_doc_ids = {str(doc["id"]) for doc in documents}
    existing_queries = {normalize_spaces(str(query.get("query", ""))).lower() for query in queries}
    review_map = load_review_queue(review_path)
    next_query_id = next_id_number(queries, "q")
    rng = random.Random(seed)

    system_prompt = (
        "You create hard adversarial retrieval queries for Memory.swift. "
        "Each query should still have a clear relevant document set, but should be paraphrased, indirect, or distractor-prone."
    )

    remaining = augment_count
    batch_index = 0
    while remaining > 0:
        batch_index += 1
        current_batch = min(batch_size, remaining)
        candidate_docs = rng.sample(documents, k=min(max(current_batch * 4, 12), len(documents)))
        recent_queries = rng.sample(queries, k=min(len(queries), 20)) if queries else []
        user_prompt = (
            build_adversarial_batch(candidate_docs, recent_queries, current_batch)
            + "\n\nReturn only hard queries. Use 1-3 relevant document ids from the candidate set."
        )

        try:
            rows = codex_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=ADVERSARIAL_QUERY_SCHEMA,
                progress_label=f"[adversarial-augment] batch {batch_index}",
            )
        except Exception as exc:
            add_review_entry(
                review_map,
                mode="adversarial-augment",
                record_id=f"batch-{batch_index}",
                reason="request_failure",
                source={"batch_size": current_batch},
            )
            write_review_queue(review_path, review_map)
            log(f"[adversarial-augment] batch {batch_index} failed: {truncate_for_log(str(exc))}")
            break

        added = 0
        for row in rows:
            sanitized = sanitize_adversarial_query(row, valid_doc_ids=valid_doc_ids)
            if not sanitized:
                continue
            normalized = sanitized["query"].lower()
            if normalized in existing_queries:
                continue
            query_row = {
                "id": f"q-{next_query_id:04d}",
                "query": sanitized["query"],
                "relevant_document_ids": sanitized["relevant_document_ids"],
                "memory_types": sanitized["memory_types"],
                "difficulty": "hard",
            }
            queries.append(query_row)
            existing_queries.add(normalized)
            next_query_id += 1
            added += 1
            remaining -= 1
            if remaining <= 0:
                break

        write_jsonl_atomic(queries_path, queries)
        write_review_queue(review_path, review_map)
        log(f"[adversarial-augment] batch {batch_index}: added {added}/{current_batch}")
        if added == 0:
            break


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag and augment Memory.swift eval datasets via codex exec.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing eval JSONL files.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("query-tags", "document-tags", "storage-cases", "adversarial-augment"),
        help="Tagging or augmentation mode to run.",
    )
    parser.add_argument("--workspace", default=".", help="Workspace passed to codex exec -C.")
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI binary name or path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--batch-size", type=int, default=6, help="Number of records per Codex batch.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on processed records.")
    parser.add_argument("--max-storage-cases", type=int, default=300, help="Maximum storage cases to derive in one run.")
    parser.add_argument("--augment-count", type=int, default=50, help="Number of adversarial queries to append.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic sampling.")
    parser.add_argument("--retag", action="store_true", help="Re-run tagging even when tags already exist.")
    parser.add_argument("--review-queue", default=None, help=f"Optional review queue path (default: <dataset-root>/{REVIEW_QUEUE_FILENAME}).")
    parser.add_argument("--request-timeout-seconds", type=int, default=240, help="Timeout for each codex exec call.")
    parser.add_argument("--max-retries-per-request", type=int, default=3, help="Retries per codex exec call.")
    parser.add_argument(
        "--codex-reasoning-effort",
        default="low",
        help="Sets codex model_reasoning_effort config override (default: low).",
    )
    parser.add_argument(
        "--disable-xcode-mcp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable xcode MCP server for codex exec calls (default: true).",
    )
    parser.add_argument(
        "--codex-config",
        action="append",
        default=[],
        help="Additional `-c key=value` override passed to codex exec (repeatable).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    review_path = Path(args.review_queue).resolve() if args.review_queue else dataset_root / REVIEW_QUEUE_FILENAME

    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root}")

    codex_config_overrides: List[str] = []
    if args.codex_reasoning_effort:
        codex_config_overrides.append(f'model_reasoning_effort="{args.codex_reasoning_effort}"')
    if args.disable_xcode_mcp:
        codex_config_overrides.append("mcp_servers.xcode.enabled=false")
    codex_config_overrides.extend(args.codex_config)

    client = CodexClient(
        codex_bin=args.codex_bin,
        model=args.model,
        workspace=workspace,
        timeout_seconds=args.request_timeout_seconds,
        max_retries_per_request=args.max_retries_per_request,
        config_overrides=codex_config_overrides,
    )
    client.ensure_login()

    manifest_path = dataset_root / "manifest.json"
    manifest = merge_manifest_defaults(
        load_manifest(manifest_path),
        model=args.model,
        backend_mode="single_pass_document,double_pass_query_and_storage",
    )
    write_manifest(manifest_path, manifest)

    log(f"Using codex binary: {client.codex_bin}")
    log(f"Using model: {args.model}")
    log(f"Workspace: {workspace}")
    log(f"Dataset root: {dataset_root}")
    log(f"Mode: {args.mode}")

    if args.mode == "query-tags":
        tag_queries(client, dataset_root, args.batch_size, review_path, args.max_records, args.retag)
    elif args.mode == "document-tags":
        tag_documents(client, dataset_root, args.batch_size, review_path, args.max_records, args.retag)
    elif args.mode == "storage-cases":
        derive_storage_cases(
            client,
            dataset_root,
            args.batch_size,
            review_path,
            args.max_records,
            args.retag,
            args.max_storage_cases,
            args.seed,
        )
    else:
        augment_adversarial_queries(
            client,
            dataset_root,
            args.batch_size,
            review_path,
            args.augment_count,
            args.seed,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
