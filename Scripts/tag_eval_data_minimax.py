#!/usr/bin/env python3
"""Tag and augment Memory.swift eval datasets via MiniMax's Anthropic-compatible API."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval_data_codex_support import (
    ensure_json_object_with_list,
    extract_json_payload,
    load_jsonl,
    load_manifest,
    log,
    truncate_for_log,
    write_jsonl_atomic,
    write_manifest,
)
from generate_eval_data_minimax import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MiniMaxAnthropicClient,
    load_env_file,
    resolve_api_key,
)
from tag_eval_data_codex import (
    ADVERSARIAL_QUERY_SCHEMA,
    DIFFICULTY_LEVELS,
    DOCUMENT_TAG_SCHEMA,
    MEMORY_TYPES,
    QUERY_TAG_SCHEMA,
    REVIEW_QUEUE_FILENAME,
    STORAGE_CASE_SCHEMA,
    add_review_entry,
    build_adversarial_batch,
    build_document_tag_batch,
    build_query_tag_batch,
    build_storage_batch,
    build_storage_record,
    chunked,
    clear_review_entry,
    coalesce_storage_spans,
    load_review_queue,
    next_id_number,
    prepare_storage_targets,
    sanitize_adversarial_query,
    sanitize_difficulty,
    sanitize_document_tag,
    sanitize_query_tag,
    sanitize_storage_proposal,
    write_review_queue,
)

PROMPT_VERSION = "2026-03-15-minimax-tagging-v1"


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
        "backend": "minimax_anthropic_api",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "auto_accept_policy": backend_mode,
    }
    return merged


def minimax_items(
    client: MiniMaxAnthropicClient,
    *,
    system_prompt: str,
    user_prompt: str,
    output_schema: Dict[str, Any],
    progress_label: str,
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    raw = client.create_message(
        system_prompt=(
            "You are generating or tagging evaluation data.\n"
            "Return JSON only.\n"
            "Do not use markdown fences.\n"
            f"{system_prompt}\n\n"
            "Expected output JSON schema:\n"
            f"{output_schema}"
        ),
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        progress_label=progress_label,
    )
    payload = extract_json_payload(raw)
    return ensure_json_object_with_list(payload, "items")


def query_tags_conservatively_agree(
    proposal_a: Dict[str, Any],
    proposal_b: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if proposal_a["difficulty"] != proposal_b["difficulty"]:
        return None
    if proposal_a["memory_types"] == proposal_b["memory_types"]:
        return {
            "memory_types": proposal_a["memory_types"],
            "difficulty": proposal_a["difficulty"],
        }

    types_a = proposal_a["memory_types"]
    types_b = proposal_b["memory_types"]
    shared = [memory_type for memory_type in types_a if memory_type in types_b]
    if not shared:
        return None

    set_a = set(types_a)
    set_b = set(types_b)
    if set_a.issubset(set_b) or set_b.issubset(set_a):
        return {
            "memory_types": shared,
            "difficulty": proposal_a["difficulty"],
        }
    return None


def tag_queries(
    client: MiniMaxAnthropicClient,
    dataset_root: Path,
    batch_size: int,
    review_path: Path,
    max_records: Optional[int],
    retag: bool,
    agreement_passes: int,
    *,
    temperature: float,
    max_tokens: int,
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
        has_types = bool(query.get("memory_types"))
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
            "Prefer exactly one dominant memory type.\n"
            "Only return two memory_types when both are indispensable to the retrieval intent.\n"
            "Favor the smallest correct set of memory_types.\n\n"
            "Input queries (JSON lines):\n"
            f"{user_payload}"
        )

        try:
            pass_a = minimax_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=QUERY_TAG_SCHEMA,
                progress_label=f"[query-tags] batch {batch_index} pass 1",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            pass_b = []
            if agreement_passes >= 2:
                pass_b = minimax_items(
                    client,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=QUERY_TAG_SCHEMA,
                    progress_label=f"[query-tags] batch {batch_index} pass 2",
                    temperature=min(0.6, temperature + 0.05),
                    max_tokens=max_tokens,
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
            if agreement_passes == 1 and proposal_a:
                query["memory_types"] = proposal_a["memory_types"]
                query["difficulty"] = proposal_a["difficulty"]
                clear_review_entry(review_map, "query-tags", query_id)
                updated += 1
                continue

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

            agreed = query_tags_conservatively_agree(proposal_a, proposal_b)
            if agreed:
                query["memory_types"] = agreed["memory_types"]
                query["difficulty"] = agreed["difficulty"]
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
    client: MiniMaxAnthropicClient,
    dataset_root: Path,
    batch_size: int,
    review_path: Path,
    max_records: Optional[int],
    retag: bool,
    *,
    temperature: float,
    max_tokens: int,
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
            rows = minimax_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=DOCUMENT_TAG_SCHEMA,
                progress_label=f"[document-tags] batch {batch_index}",
                temperature=max(0.1, temperature - 0.05),
                max_tokens=max_tokens,
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


def derive_storage_cases(
    client: MiniMaxAnthropicClient,
    dataset_root: Path,
    batch_size: int,
    review_path: Path,
    max_records: Optional[int],
    retag: bool,
    max_storage_cases: int,
    seed: int,
    agreement_passes: int,
    *,
    temperature: float,
    max_tokens: int,
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
            pass_a = minimax_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=STORAGE_CASE_SCHEMA,
                progress_label=f"[storage-cases] batch {batch_index} pass 1",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            pass_b = []
            if agreement_passes >= 2:
                pass_b = minimax_items(
                    client,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=STORAGE_CASE_SCHEMA,
                    progress_label=f"[storage-cases] batch {batch_index} pass 2",
                    temperature=min(0.6, temperature + 0.05),
                    max_tokens=max_tokens,
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
            if agreement_passes == 1 and proposal_a:
                accepted = {
                    "expected_memory_type": proposal_a["expected_memory_type"],
                    "required_spans": proposal_a["required_spans"],
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
                continue

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
    client: MiniMaxAnthropicClient,
    dataset_root: Path,
    batch_size: int,
    review_path: Path,
    augment_count: int,
    seed: int,
    *,
    temperature: float,
    max_tokens: int,
) -> None:
    docs_path = dataset_root / "recall_documents.jsonl"
    queries_path = dataset_root / "recall_queries.jsonl"
    if not docs_path.exists() or not queries_path.exists():
        raise RuntimeError("adversarial-augment mode requires recall_documents.jsonl and recall_queries.jsonl.")

    documents = load_jsonl(docs_path)
    queries = load_jsonl(queries_path)
    valid_doc_ids = {str(doc["id"]) for doc in documents}
    existing_queries = {str(query.get("query", "")).strip().lower() for query in queries}
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
            rows = minimax_items(
                client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=ADVERSARIAL_QUERY_SCHEMA,
                progress_label=f"[adversarial-augment] batch {batch_index}",
                temperature=max(0.35, temperature + 0.1),
                max_tokens=max_tokens,
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
    parser = argparse.ArgumentParser(description="Tag and augment Memory.swift eval datasets via MiniMax API.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing eval JSONL files.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("query-tags", "document-tags", "storage-cases", "adversarial-augment"),
        help="Tagging or augmentation mode to run.",
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env).")
    parser.add_argument("--base-url", default=None, help="Anthropic-compatible base URL.")
    parser.add_argument("--model", default=None, help=f"Model name (default: {DEFAULT_MODEL} or MINIMAX_MODEL env).")
    parser.add_argument("--batch-size", type=int, default=6, help="Number of records per API batch.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on processed records.")
    parser.add_argument("--max-storage-cases", type=int, default=300, help="Maximum storage cases to derive in one run.")
    parser.add_argument("--augment-count", type=int, default=50, help="Number of adversarial queries to append.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic sampling.")
    parser.add_argument("--retag", action="store_true", help="Re-run tagging even when tags already exist.")
    parser.add_argument("--review-queue", default=None, help=f"Optional review queue path (default: <dataset-root>/{REVIEW_QUEUE_FILENAME}).")
    parser.add_argument("--request-timeout-seconds", type=int, default=240, help="Per-request HTTP timeout in seconds.")
    parser.add_argument("--max-retries-per-request", type=int, default=4, help="HTTP retries per API request.")
    parser.add_argument("--agreement-passes", type=int, choices=(1, 2), default=2, help="Acceptance passes for query and storage tagging.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per API response.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    review_path = Path(args.review_queue).resolve() if args.review_queue else dataset_root / REVIEW_QUEUE_FILENAME
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root}")

    env_path = Path(args.env_file)
    load_env_file(env_path)
    api_key = resolve_api_key()
    base_url = args.base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL
    model = args.model or os.environ.get("MINIMAX_MODEL") or DEFAULT_MODEL

    client = MiniMaxAnthropicClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_retries_per_request=args.max_retries_per_request,
        timeout_seconds=args.request_timeout_seconds,
    )

    manifest_path = dataset_root / "manifest.json"
    query_storage_policy = "single_pass_query_and_storage" if args.agreement_passes == 1 else "double_pass_query_and_storage"
    manifest = merge_manifest_defaults(
        load_manifest(manifest_path),
        model=model,
        backend_mode=f"single_pass_document,{query_storage_policy}",
    )
    write_manifest(manifest_path, manifest)

    log(f"Using base URL: {base_url}")
    log(f"Using model: {model}")
    log(f"Dataset root: {dataset_root}")
    log(f"Mode: {args.mode}")

    if args.mode == "query-tags":
        tag_queries(
            client,
            dataset_root,
            args.batch_size,
            review_path,
            args.max_records,
            args.retag,
            args.agreement_passes,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    elif args.mode == "document-tags":
        tag_documents(
            client,
            dataset_root,
            args.batch_size,
            review_path,
            args.max_records,
            args.retag,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
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
            args.agreement_passes,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        augment_adversarial_queries(
            client,
            dataset_root,
            args.batch_size,
            review_path,
            args.augment_count,
            args.seed,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
