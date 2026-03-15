#!/usr/bin/env python3
"""Run model-assisted audits over eval audit packets."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval_data_audit_support import (
    DEFAULT_OPENCODE_MODEL,
    OpenCodeClient,
    split_batches_by_kind,
)
from eval_data_codex_support import (
    extract_json_payload,
    load_jsonl,
    log,
    truncate_for_log,
    write_jsonl_atomic,
)
from generate_eval_data_minimax import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL as DEFAULT_MINIMAX_MODEL,
    MiniMaxAnthropicClient,
    load_env_file,
    resolve_api_key,
)
from tag_eval_data_codex import DIFFICULTY_LEVELS, MEMORY_TYPES

AUDIT_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["packet_id", "verdict", "confidence", "issues", "notes"],
                "properties": {
                    "packet_id": {"type": "string", "minLength": 1},
                    "verdict": {"type": "string", "enum": ["accept", "needs_edit", "reject"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                    "suggested_memory_type": {"type": ["string", "null"]},
                    "suggested_memory_types": {"type": ["array", "null"], "items": {"type": "string"}},
                    "suggested_difficulty": {"type": ["string", "null"]},
                    "suggested_required_spans": {"type": ["array", "null"], "items": {"type": "string"}},
                    "suggested_relevant_document_ids": {"type": ["array", "null"], "items": {"type": "string"}},
                },
            },
        }
    },
}


class MiniMaxAuditClient:
    def __init__(
        self,
        *,
        env_file: Path,
        base_url: Optional[str],
        model: Optional[str],
        timeout_seconds: int,
        max_retries_per_request: int,
    ) -> None:
        load_env_file(env_file)
        api_key = resolve_api_key()
        resolved_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL
        resolved_model = model or os.environ.get("MINIMAX_MODEL") or DEFAULT_MINIMAX_MODEL
        self.model = resolved_model
        self.client = MiniMaxAnthropicClient(
            api_key=api_key,
            base_url=resolved_base_url,
            model=resolved_model,
            max_retries_per_request=max_retries_per_request,
            timeout_seconds=timeout_seconds,
        )

    def create_message(self, *, system_prompt: str, user_prompt: str, progress_label: str) -> str:
        return self.client.create_message(
            system_prompt=(
                "You are auditing evaluation data.\n"
                "Return JSON only.\n"
                "Do not use markdown fences.\n\n"
                f"{system_prompt}\n\n"
                "Expected output JSON schema:\n"
                f"{AUDIT_RESULT_SCHEMA}"
            ),
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=4096,
            progress_label=progress_label,
        )


def heuristic_single_item_result(raw: str, *, fallback_packet_id: str) -> List[Dict[str, Any]]:
    lowered = raw.strip().lower()
    verdict = ""
    if "needs_edit" in lowered or "needs edit" in lowered:
        verdict = "needs_edit"
    elif "reject" in lowered:
        verdict = "reject"
    elif "accept" in lowered or "approved" in lowered:
        verdict = "accept"
    if not verdict:
        raise ValueError("Could not infer verdict from non-JSON audit response.")

    confidence = "low"
    if "high" in lowered:
        confidence = "high"
    elif "medium" in lowered:
        confidence = "medium"
    notes = raw.strip()
    if notes.lower() in {"accept", "needs_edit", "needs edit", "reject"}:
        notes = ""
    return [{
        "packet_id": fallback_packet_id,
        "verdict": verdict,
        "confidence": confidence,
        "issues": [],
        "notes": notes,
    }]


def parse_audit_items(raw: str, *, fallback_packet_id: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        payload = extract_json_payload(raw)
    except Exception:
        if fallback_packet_id:
            return heuristic_single_item_result(raw, fallback_packet_id=fallback_packet_id)
        raise
    if isinstance(payload, dict) and "items" in payload:
        items = payload.get("items")
    elif isinstance(payload, dict) and "packet_id" in payload:
        items = [payload]
    elif isinstance(payload, dict) and fallback_packet_id:
        single = dict(payload)
        single.setdefault("packet_id", fallback_packet_id)
        if "verdict" not in single:
            for key in ("judgment", "decision", "assessment", "label"):
                if key in single:
                    single["verdict"] = single.get(key)
                    break
        if "notes" not in single and "explanation" in single:
            single["notes"] = single.get("explanation")
        items = [single]
    elif isinstance(payload, dict):
        items = payload.get("items")
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("Expected an items list in audit response.")
    return [item for item in items if isinstance(item, dict)]


def sanitize_result(raw: Dict[str, Any], packet_ids: set[str]) -> Optional[Dict[str, Any]]:
    packet_id = str(raw.get("packet_id", "")).strip()
    verdict = str(raw.get("verdict", raw.get("judgment", ""))).strip().lower()
    confidence = str(raw.get("confidence", "")).strip().lower()
    if packet_id not in packet_ids or verdict not in {"accept", "needs_edit", "reject"}:
        return None
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    issues_raw = raw.get("issues", [])
    issues = []
    if isinstance(issues_raw, list):
        for issue in issues_raw:
            if isinstance(issue, str):
                cleaned = issue.strip()
                if cleaned:
                    issues.append(cleaned)

    result: Dict[str, Any] = {
        "packet_id": packet_id,
        "verdict": verdict,
        "confidence": confidence,
        "issues": issues,
        "notes": str(raw.get("notes", raw.get("explanation", ""))).strip(),
    }

    if isinstance(raw.get("suggested_memory_type"), str):
        memory_type = str(raw.get("suggested_memory_type")).strip().lower()
        if memory_type in MEMORY_TYPES:
            result["suggested_memory_type"] = memory_type
    if isinstance(raw.get("suggested_memory_types"), list):
        memory_types = [
            str(value).strip().lower()
            for value in raw.get("suggested_memory_types", [])
            if str(value).strip().lower() in MEMORY_TYPES
        ]
        if memory_types:
            result["suggested_memory_types"] = memory_types
    if isinstance(raw.get("suggested_difficulty"), str):
        difficulty = str(raw.get("suggested_difficulty")).strip().lower()
        if difficulty in DIFFICULTY_LEVELS:
            result["suggested_difficulty"] = difficulty
    if isinstance(raw.get("suggested_required_spans"), list):
        result["suggested_required_spans"] = [str(value).strip() for value in raw.get("suggested_required_spans", []) if str(value).strip()]
    if isinstance(raw.get("suggested_relevant_document_ids"), list):
        result["suggested_relevant_document_ids"] = [str(value).strip() for value in raw.get("suggested_relevant_document_ids", []) if str(value).strip()]
    return result


def query_prompt(batch: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for item in batch:
        lines.append(
            json.dumps(
                {
                    "packet_id": item["packet_id"],
                    "query": item["query"],
                    "current_memory_types": item.get("current_memory_types", []),
                    "current_difficulty": item.get("current_difficulty"),
                    "relevant_document_ids": item.get("relevant_document_ids", []),
                    "relevant_documents": item.get("relevant_documents", []),
                    "review_context": item.get("review_context"),
                    "sample_reason": item.get("sample_reason"),
                },
                ensure_ascii=False,
            )
        )
    return (
        "Audit these retrieval queries.\n"
        "Judge whether the current memory_types, difficulty, and relevant_document_ids are appropriate.\n"
        "Use `accept` when the current labels look correct.\n"
        "Use `needs_edit` when the item is salvageable but labels or relevant docs should change.\n"
        "Use `reject` when the item is off-topic, low-quality, or clearly broken.\n"
        f"Allowed memory_types: {', '.join(MEMORY_TYPES)}.\n"
        f"Allowed difficulty values: {', '.join(DIFFICULTY_LEVELS)}.\n"
        "Only suggest fields that need to change.\n\n"
        "Items (JSON lines):\n"
        f"{chr(10).join(lines)}"
    )


def document_prompt(batch: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for item in batch:
        lines.append(
            json.dumps(
                {
                    "packet_id": item["packet_id"],
                    "relative_path": item.get("relative_path"),
                    "current_memory_type": item.get("current_memory_type"),
                    "document": item.get("document"),
                    "review_context": item.get("review_context"),
                    "sample_reason": item.get("sample_reason"),
                },
                ensure_ascii=False,
            )
        )
    return (
        "Audit these retrieval documents.\n"
        "Judge whether the current dominant memory type is appropriate and whether the document fits the dataset.\n"
        "Use `accept`, `needs_edit`, or `reject` as above.\n"
        f"Allowed memory types: {', '.join(MEMORY_TYPES)}.\n"
        "Only suggest a memory type when the current one looks wrong.\n\n"
        "Items (JSON lines):\n"
        f"{chr(10).join(lines)}"
    )


def storage_prompt(batch: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for item in batch:
        lines.append(
            json.dumps(
                {
                    "packet_id": item["packet_id"],
                    "current_memory_type": item.get("current_memory_type"),
                    "required_spans": item.get("required_spans", []),
                    "highlighted_text": item.get("highlighted_text"),
                    "review_context": item.get("review_context"),
                    "sample_reason": item.get("sample_reason"),
                },
                ensure_ascii=False,
            )
        )
    return (
        "Audit these storage cases.\n"
        "Judge whether the expected_memory_type and required_spans are correct.\n"
        "required_spans should be verbatim, high-signal, and sufficient for a storage eval.\n"
        "Use `accept`, `needs_edit`, or `reject` as above.\n"
        f"Allowed memory types: {', '.join(MEMORY_TYPES)}.\n"
        "Only suggest spans that are present in the highlighted_text.\n\n"
        "Items (JSON lines):\n"
        f"{chr(10).join(lines)}"
    )


def build_user_prompt(batch: Sequence[Dict[str, Any]]) -> str:
    entry_type = str(batch[0].get("entry_type", ""))
    if entry_type == "query":
        return query_prompt(batch)
    if entry_type == "document":
        return document_prompt(batch)
    return storage_prompt(batch)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-assisted audits over eval packets.")
    parser.add_argument("--packet", action="append", required=True, help="Path to packet.jsonl (repeatable).")
    parser.add_argument("--backend", choices=("opencode", "minimax"), default="opencode", help="Audit backend.")
    parser.add_argument("--model", default=None, help="Model override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Items per audit request. Defaults to 1 for opencode, 4 for MiniMax.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on packet items.")
    parser.add_argument("--workspace", default=".", help="Workspace root for opencode runs.")
    parser.add_argument("--opencode-bin", default="opencode", help="opencode binary name or path.")
    parser.add_argument("--env-file", default=".env", help="Env file for MiniMax.")
    parser.add_argument("--request-timeout-seconds", type=int, default=240, help="Per-request timeout.")
    parser.add_argument("--max-retries-per-request", type=int, default=4, help="MiniMax HTTP retries.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing audit results.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace).resolve()
    if args.backend == "opencode":
        model = args.model or DEFAULT_OPENCODE_MODEL
        backend = OpenCodeClient(
            opencode_bin=args.opencode_bin,
            model=model,
            workspace=workspace,
            timeout_seconds=args.request_timeout_seconds,
        )
    else:
        model = args.model or os.environ.get("MINIMAX_MODEL") or DEFAULT_MINIMAX_MODEL
        backend = MiniMaxAuditClient(
            env_file=Path(args.env_file),
            base_url=None,
            model=model,
            timeout_seconds=args.request_timeout_seconds,
            max_retries_per_request=args.max_retries_per_request,
        )

    system_prompt = (
        "You are auditing evaluation data for Memory.swift.\n"
        "Be conservative.\n"
        "Do not invent facts not supported by the provided records.\n"
        "Prefer `needs_edit` over `reject` when the example can be fixed with a small metadata change."
    )
    batch_size = args.batch_size or (1 if args.backend == "opencode" else 4)

    for raw_packet in args.packet:
        packet_path = Path(raw_packet).resolve()
        packet_rows = load_jsonl(packet_path)
        if args.max_items is not None:
            packet_rows = packet_rows[: args.max_items]
        output_path = packet_path.parent / f"audit_results.{args.backend}.jsonl"
        existing_rows = [] if args.overwrite else load_jsonl(output_path)
        completed_ids = {str(row.get("packet_id", "")) for row in existing_rows}
        pending = [row for row in packet_rows if str(row.get("packet_id", "")) not in completed_ids]
        if not pending:
            log(f"No pending packet items in {packet_path.name}")
            continue

        all_results = list(existing_rows)
        batches = split_batches_by_kind(pending, batch_size)
        for batch_index, batch in enumerate(batches, start=1):
            progress_label = f"[audit:{args.backend}] {packet_path.parent.name} batch {batch_index}/{len(batches)}"
            try:
                raw = backend.create_message(
                    system_prompt=system_prompt,
                    user_prompt=build_user_prompt(batch),
                    progress_label=progress_label,
                )
                parsed_rows = parse_audit_items(
                    raw,
                    fallback_packet_id=str(batch[0].get("packet_id", "")) if len(batch) == 1 else None,
                )
            except Exception as exc:
                log(f"{progress_label} failed: {truncate_for_log(str(exc))}")
                continue

            valid_packet_ids = {str(item.get("packet_id", "")) for item in batch}
            batch_results = [sanitize_result(row, valid_packet_ids) for row in parsed_rows]
            accepted = [row for row in batch_results if row]
            if not accepted:
                log(f"{progress_label} produced no valid results")
                continue

            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for row in accepted:
                row["backend"] = args.backend
                row["model"] = model
                row["audited_at"] = timestamp
            all_results.extend(accepted)
            write_jsonl_atomic(output_path, all_results)
            log(f"{progress_label}: wrote {len(accepted)} results")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
