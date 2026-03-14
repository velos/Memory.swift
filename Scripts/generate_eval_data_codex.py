#!/usr/bin/env python3
"""
Generate eval datasets for Memory.swift by repeatedly calling codex exec.

This path is designed for authenticated Codex/ChatGPT workflows and uses
small, atomic generation batches instead of large one-shot calls.

Outputs:
  - Evals/storage_cases.jsonl
  - Evals/recall_documents.jsonl
  - Evals/recall_queries.jsonl

Key behavior:
  - Incremental persistence: writes JSONL as new records are accepted.
  - Resume support: can continue from existing JSONL files.

Usage:
  python3 scripts/generate_eval_data_codex.py --dataset-root Evals --resume
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

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

VALID_KINDS = {"markdown", "code", "plainText"}
DEFAULT_MODEL = "gpt-5.2"


STORAGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["cases"],
    "properties": {
        "cases": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "required_spans"],
                "properties": {
                    "text": {"type": "string", "minLength": 50},
                    "required_spans": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        }
    },
}

DOCUMENTS_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["documents"],
    "properties": {
        "documents": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text"],
                "properties": {
                    "text": {"type": "string", "minLength": 60},
                },
            },
        }
    },
}

# OpenAI JSON schema validation requires every property to be present in `required`.
# `memory_types` is represented as null when omitted.
QUERIES_OUTPUT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["queries"],
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query", "relevant_document_ids", "memory_types"],
                "properties": {
                    "query": {"type": "string", "minLength": 8},
                    "relevant_document_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "memory_types": {
                        "anyOf": [
                            {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string", "enum": MEMORY_TYPES},
                            },
                            {"type": "null"},
                        ]
                    },
                },
            },
        }
    },
}


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_for_log(value: str, limit: int = 220) -> str:
    compact = normalize_spaces(value)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def distribute(total: int, buckets: int) -> List[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def render_jsonl(records: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    result: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object in {path}:{line_number}")
        result.append(parsed)
    return result


def write_jsonl_atomic(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.tmp"
    temp_path.write_text(render_jsonl(records), encoding="utf-8")
    temp_path.replace(path)


def extract_json_payload(raw: str) -> Any:
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if object_match:
        candidate = object_match.group(0).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if array_match:
        candidate = array_match.group(0).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("Model output did not contain valid JSON.")


def ensure_json_object_with_list(payload: Any, key: str) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get(key)
    else:
        items = payload

    if not isinstance(items, list):
        raise ValueError(f"Expected list field '{key}' in model output.")

    records: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            records.append(item)
    return records


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


def normalize_kind(value: Any) -> str:
    if isinstance(value, str) and value in VALID_KINDS:
        return value
    return "markdown"


def sanitize_storage_case(raw: Dict[str, Any], expected_type: str) -> Optional[Dict[str, Any]]:
    text = normalize_spaces(str(raw.get("text", "")))
    if not text:
        return None

    wc = word_count(text)
    if wc < 50 or wc > 220:
        return None

    spans_raw = raw.get("required_spans")
    if not isinstance(spans_raw, list):
        return None

    spans: List[str] = []
    for span in spans_raw:
        if not isinstance(span, str):
            continue
        cleaned = span.strip()
        if cleaned and cleaned in text and cleaned not in spans:
            spans.append(cleaned)

    if len(spans) < 2:
        return None

    return {
        "kind": "markdown",
        "text": text,
        "expected_memory_type": expected_type,
        "required_spans": spans[:4],
    }


def sanitize_recall_document(raw: Dict[str, Any], expected_type: str) -> Optional[Dict[str, Any]]:
    text = normalize_spaces(str(raw.get("text", "")))
    if not text:
        return None

    wc = word_count(text)
    if wc < 60 or wc > 260:
        return None

    return {
        "kind": "markdown",
        "text": text,
        "memory_type": expected_type,
    }


def sanitize_recall_query(raw: Dict[str, Any], *, valid_doc_ids: set[str]) -> Optional[Dict[str, Any]]:
    query = normalize_spaces(str(raw.get("query", "")))
    if len(query) < 8:
        return None

    relevant_raw = raw.get("relevant_document_ids")
    if not isinstance(relevant_raw, list):
        return None

    relevant: List[str] = []
    for item in relevant_raw:
        if not isinstance(item, str):
            continue
        doc_id = item.strip()
        if doc_id in valid_doc_ids and doc_id not in relevant:
            relevant.append(doc_id)

    if not relevant or len(relevant) > 3:
        return None

    memory_types_raw = raw.get("memory_types")
    memory_types: Optional[List[str]] = None
    if isinstance(memory_types_raw, list):
        filtered: List[str] = []
        for item in memory_types_raw:
            if isinstance(item, str):
                memory_type = item.strip().lower()
                if memory_type in MEMORY_TYPES and memory_type not in filtered:
                    filtered.append(memory_type)
        if filtered:
            memory_types = filtered

    result: Dict[str, Any] = {
        "query": query,
        "relevant_document_ids": relevant,
    }
    if memory_types:
        result["memory_types"] = memory_types
    return result


def normalize_existing_storage_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw in records:
        raw_id = str(raw.get("id", "")).strip()
        if not raw_id:
            raise ValueError("Storage record is missing id.")
        if raw_id in seen_ids:
            raise ValueError(f"Duplicate storage id '{raw_id}'.")
        seen_ids.add(raw_id)

        expected_type = str(raw.get("expected_memory_type", "")).strip().lower()
        if expected_type not in MEMORY_TYPES:
            raise ValueError(f"Invalid expected_memory_type '{expected_type}' in storage record {raw_id}.")

        text = normalize_spaces(str(raw.get("text", "")))
        if not text:
            raise ValueError(f"Invalid storage record payload for {raw_id}: missing text.")

        spans_raw = raw.get("required_spans")
        spans: List[str] = []
        if isinstance(spans_raw, list):
            for span in spans_raw:
                if isinstance(span, str):
                    cleaned = span.strip()
                    if cleaned and cleaned not in spans:
                        spans.append(cleaned)

        # Resume should accept legacy/template rows that may not fully match strict generation rules.
        if not spans:
            words = re.findall(r"\S+", text)
            fallback = " ".join(words[:2]).strip() if words else ""
            if fallback:
                spans = [fallback]
            else:
                raise ValueError(f"Invalid storage record payload for {raw_id}: missing required_spans.")

        normalized.append(
            {
                "id": raw_id,
                "kind": normalize_kind(raw.get("kind")),
                "text": text,
                "expected_memory_type": expected_type,
                "required_spans": spans[:4],
            }
        )

    return sorted(normalized, key=lambda item: parse_prefixed_id(str(item["id"]), "storage") or 0)


def normalize_existing_document_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw in records:
        raw_id = str(raw.get("id", "")).strip()
        if not raw_id:
            raise ValueError("Document record is missing id.")
        if raw_id in seen_ids:
            raise ValueError(f"Duplicate document id '{raw_id}'.")
        seen_ids.add(raw_id)

        memory_type = str(raw.get("memory_type", "")).strip().lower()
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Invalid memory_type '{memory_type}' in document {raw_id}.")

        text = normalize_spaces(str(raw.get("text", "")))
        if not text:
            raise ValueError(f"Invalid document payload for {raw_id}: missing text.")

        relative_path = str(raw.get("relative_path", "")).strip() or f"{memory_type}/{raw_id}.md"

        normalized.append(
            {
                "id": raw_id,
                "relative_path": relative_path,
                "kind": normalize_kind(raw.get("kind")),
                "text": text,
                "memory_type": memory_type,
            }
        )

    return sorted(normalized, key=lambda item: parse_prefixed_id(str(item["id"]), "doc") or 0)


def normalize_existing_query_records(
    records: Sequence[Dict[str, Any]],
    *,
    valid_doc_ids: set[str],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw in records:
        raw_id = str(raw.get("id", "")).strip()
        if not raw_id:
            raise ValueError("Query record is missing id.")
        if raw_id in seen_ids:
            raise ValueError(f"Duplicate query id '{raw_id}'.")
        seen_ids.add(raw_id)

        sanitized = sanitize_recall_query(raw, valid_doc_ids=valid_doc_ids)
        if not sanitized:
            raise ValueError(f"Invalid query payload for {raw_id}.")

        row: Dict[str, Any] = {
            "id": raw_id,
            "query": sanitized["query"],
            "relevant_document_ids": sanitized["relevant_document_ids"],
        }
        if "memory_types" in sanitized:
            row["memory_types"] = sanitized["memory_types"]
        normalized.append(row)

    return sorted(normalized, key=lambda item: parse_prefixed_id(str(item["id"]), "q") or 0)


def summarize_codex_failure(stdout: str, stderr: str, return_code: int) -> str:
    stderr_clean = stderr.strip()
    stdout_clean = stdout.strip()
    lines = [line.strip() for line in (stderr_clean + "\n" + stdout_clean).splitlines() if line.strip()]

    for line in reversed(lines):
        if line.startswith("ERROR:"):
            return line
    for line in reversed(lines):
        if "invalid_json_schema" in line:
            return line
    for line in reversed(lines):
        if "mcp startup" in line.lower():
            return line

    if lines:
        return lines[-1]
    return f"codex exited with code {return_code}"


def build_document_catalog_snippet(
    documents: Sequence[Dict[str, Any]],
    *,
    limit: int,
    summary_chars: int,
) -> str:
    lines: List[str] = []
    for doc in documents[:limit]:
        snippet = normalize_spaces(str(doc["text"]))[:summary_chars]
        lines.append(
            json.dumps(
                {
                    "id": doc["id"],
                    "memory_type": doc["memory_type"],
                    "summary": snippet,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def assert_storage_counts_within_targets(records: Sequence[Dict[str, Any]], total_target: int) -> None:
    per_type_targets = dict(zip(MEMORY_TYPES, distribute(total_target, len(MEMORY_TYPES))))
    counts = {memory_type: 0 for memory_type in MEMORY_TYPES}

    for record in records:
        memory_type = str(record.get("expected_memory_type", "")).strip().lower()
        if memory_type in counts:
            counts[memory_type] += 1

    for memory_type in MEMORY_TYPES:
        if counts[memory_type] > per_type_targets[memory_type]:
            raise ValueError(
                f"Existing storage records exceed target for '{memory_type}': "
                f"have {counts[memory_type]}, target {per_type_targets[memory_type]}."
            )


def assert_document_counts_within_targets(records: Sequence[Dict[str, Any]], total_target: int) -> None:
    per_type_targets = dict(zip(MEMORY_TYPES, distribute(total_target, len(MEMORY_TYPES))))
    counts = {memory_type: 0 for memory_type in MEMORY_TYPES}

    for record in records:
        memory_type = str(record.get("memory_type", "")).strip().lower()
        if memory_type in counts:
            counts[memory_type] += 1

    for memory_type in MEMORY_TYPES:
        if counts[memory_type] > per_type_targets[memory_type]:
            raise ValueError(
                f"Existing recall documents exceed target for '{memory_type}': "
                f"have {counts[memory_type]}, target {per_type_targets[memory_type]}."
            )


@dataclass
class GenerationConfig:
    storage_cases: int
    recall_documents: int
    recall_queries: int
    storage_batch_size: int
    documents_batch_size: int
    queries_batch_size: int
    max_attempts_per_bucket: int
    max_retries_per_request: int
    request_timeout_seconds: int
    query_candidate_multiplier: int
    query_candidate_min: int
    query_summary_chars: int
    seed: int


class CodexClient:
    def __init__(
        self,
        *,
        codex_bin: str,
        model: str,
        workspace: Path,
        timeout_seconds: int,
        max_retries_per_request: int,
        config_overrides: Sequence[str],
    ) -> None:
        resolved = shutil.which(codex_bin)
        if not resolved:
            raise RuntimeError(f"Could not find codex binary '{codex_bin}' on PATH.")

        self.codex_bin = resolved
        self.model = model
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.max_retries_per_request = max_retries_per_request
        self.config_overrides = list(config_overrides)

    def ensure_login(self) -> None:
        command = [self.codex_bin, "login", "status"]
        for override in self.config_overrides:
            command.extend(["-c", override])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            error = truncate_for_log(result.stderr.strip() or result.stdout.strip() or "login status failed")
            raise RuntimeError(
                "Codex login check failed. Run `codex login` and retry. "
                f"Details: {error}"
            )

    def create_message(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: Dict[str, Any],
        progress_label: Optional[str] = None,
    ) -> str:
        final_prompt = (
            "You are generating synthetic evaluation data.\n"
            "Follow the output schema and return JSON only.\n"
            "Do not run shell commands.\n\n"
            f"SYSTEM:\n{system_prompt}\n\n"
            f"TASK:\n{user_prompt}\n"
        )

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries_per_request + 1):
            label = progress_label or "codex request"
            started_at = time.time()
            log(f"{label}: request {attempt}/{self.max_retries_per_request}")

            with tempfile.TemporaryDirectory(prefix="codex_eval_") as tmpdir:
                output_path = Path(tmpdir) / "last_message.txt"
                schema_path = Path(tmpdir) / "output_schema.json"
                schema_path.write_text(json.dumps(output_schema), encoding="utf-8")

                command = [
                    self.codex_bin,
                    "exec",
                    "-C",
                    str(self.workspace),
                    "--sandbox",
                    "read-only",
                    "-m",
                    self.model,
                ]

                for override in self.config_overrides:
                    command.extend(["-c", override])

                command.extend(
                    [
                        "--output-schema",
                        str(schema_path),
                        "--output-last-message",
                        str(output_path),
                        final_prompt,
                    ]
                )

                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    last_error = f"timeout after {self.timeout_seconds}s"
                else:
                    if result.returncode != 0:
                        last_error = summarize_codex_failure(result.stdout, result.stderr, result.returncode)
                    else:
                        try:
                            response = output_path.read_text(encoding="utf-8").strip()
                        except Exception as exc:
                            last_error = f"failed reading codex output: {exc}"
                        else:
                            if response:
                                elapsed = time.time() - started_at
                                log(f"{label}: response in {elapsed:.1f}s")
                                return response
                            last_error = "codex produced empty output"

            if attempt < self.max_retries_per_request:
                sleep_seconds = min(12.0, 1.8 ** attempt)
                log(
                    f"{label}: failed ({truncate_for_log(last_error or 'unknown error')}); "
                    f"retrying in {sleep_seconds:.1f}s"
                )
                time.sleep(sleep_seconds)
            else:
                log(f"{label}: failed ({truncate_for_log(last_error or 'unknown error')})")

        raise RuntimeError(f"Codex request failed after retries: {last_error}")


def generate_storage_cases(
    client: CodexClient,
    config: GenerationConfig,
    existing_cases: Sequence[Dict[str, Any]],
    persist: Callable[[Sequence[Dict[str, Any]]], None],
) -> List[Dict[str, Any]]:
    cases = list(existing_cases)
    seen_texts = {normalize_spaces(str(case["text"])).lower() for case in cases}
    per_type_targets = dict(zip(MEMORY_TYPES, distribute(config.storage_cases, len(MEMORY_TYPES))))

    counts_by_type = {memory_type: 0 for memory_type in MEMORY_TYPES}
    for case in cases:
        counts_by_type[str(case["expected_memory_type"])] += 1

    next_id = next_id_number(cases, "storage")

    system_prompt = (
        "You create high-quality evaluation datasets for memory systems. "
        "Return strict JSON matching the schema."
    )

    if cases:
        log(f"[storage] resuming with {len(cases)} existing cases")

    for memory_type in MEMORY_TYPES:
        target = per_type_targets[memory_type]
        current = counts_by_type[memory_type]
        attempts = 0

        log(f"[storage:{memory_type}] target={target}, current={current}")

        while current < target and attempts < config.max_attempts_per_bucket:
            attempts += 1
            needed = target - current
            batch_size = min(config.storage_batch_size, needed)
            before_count = current

            user_prompt = f"""
Generate EXACTLY {batch_size} objects in `cases`.

Rules:
- Every case MUST be strongly `{memory_type}` memory type.
- Use realistic markdown-like project notes.
- Keep text between 70 and 180 words.
- Include exactly 2 to 4 required_spans.
- Every required span MUST appear verbatim in the text (case-sensitive).
- Avoid ambiguous examples that fit multiple memory types equally.
"""

            try:
                raw = client.create_message(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=STORAGE_OUTPUT_SCHEMA,
                    progress_label=(
                        f"[storage:{memory_type}] batch {attempts}/{config.max_attempts_per_bucket} "
                        f"need={needed}"
                    ),
                )
            except Exception as exc:
                log(f"[storage:{memory_type}] request failed: {truncate_for_log(str(exc))}")
                continue

            try:
                payload = extract_json_payload(raw)
                raw_cases = ensure_json_object_with_list(payload, "cases")
            except Exception as exc:
                log(f"[storage:{memory_type}] invalid model JSON: {truncate_for_log(str(exc))}")
                continue

            added = 0
            for raw_case in raw_cases:
                sanitized = sanitize_storage_case(raw_case, memory_type)
                if not sanitized:
                    continue
                normalized_text = normalize_spaces(sanitized["text"]).lower()
                if normalized_text in seen_texts:
                    continue

                seen_texts.add(normalized_text)
                cases.append(
                    {
                        "id": f"storage-{next_id:04d}",
                        "kind": sanitized["kind"],
                        "text": sanitized["text"],
                        "expected_memory_type": sanitized["expected_memory_type"],
                        "required_spans": sanitized["required_spans"],
                    }
                )
                next_id += 1
                counts_by_type[memory_type] += 1
                current += 1
                added += 1
                if current >= target:
                    break

            if added > 0:
                persist(cases)

            log(
                f"[storage:{memory_type}] progress {current}/{target} "
                f"(+{current - before_count}, attempt {attempts}/{config.max_attempts_per_bucket})"
            )

        if current < target:
            raise RuntimeError(
                f"Could not generate enough storage cases for type '{memory_type}'. "
                f"Expected {target}, got {current} after {attempts} attempts."
            )

        log(f"[storage:{memory_type}] complete {current}/{target}")

    return cases


def generate_recall_documents(
    client: CodexClient,
    config: GenerationConfig,
    existing_documents: Sequence[Dict[str, Any]],
    persist: Callable[[Sequence[Dict[str, Any]]], None],
) -> List[Dict[str, Any]]:
    documents = list(existing_documents)
    seen_texts = {normalize_spaces(str(doc["text"])).lower() for doc in documents}
    per_type_targets = dict(zip(MEMORY_TYPES, distribute(config.recall_documents, len(MEMORY_TYPES))))

    counts_by_type = {memory_type: 0 for memory_type in MEMORY_TYPES}
    for doc in documents:
        counts_by_type[str(doc["memory_type"])] += 1

    next_id = next_id_number(documents, "doc")

    system_prompt = (
        "You create high-quality retrieval evaluation corpora. "
        "Return strict JSON matching the schema."
    )

    overlap_hints = [
        "deployment",
        "timeline",
        "incident",
        "release",
        "stakeholder",
        "checklist",
        "planning",
        "context",
    ]

    if documents:
        log(f"[documents] resuming with {len(documents)} existing documents")

    for memory_type in MEMORY_TYPES:
        target = per_type_targets[memory_type]
        current = counts_by_type[memory_type]
        attempts = 0

        log(f"[documents:{memory_type}] target={target}, current={current}")

        while current < target and attempts < config.max_attempts_per_bucket:
            attempts += 1
            needed = target - current
            batch_size = min(config.documents_batch_size, needed)
            before_count = current
            hint = overlap_hints[(attempts - 1) % len(overlap_hints)]

            user_prompt = f"""
Generate EXACTLY {batch_size} objects in `documents`.

Rules:
- Every document MUST primarily represent `{memory_type}` memory.
- Keep each text between 90 and 220 words.
- Use realistic internal notes/work docs.
- Include at least one term related to `{hint}` so lexical overlap is challenging.
- Do not include IDs or file paths.
"""

            try:
                raw = client.create_message(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=DOCUMENTS_OUTPUT_SCHEMA,
                    progress_label=(
                        f"[documents:{memory_type}] batch {attempts}/{config.max_attempts_per_bucket} "
                        f"need={needed}"
                    ),
                )
            except Exception as exc:
                log(f"[documents:{memory_type}] request failed: {truncate_for_log(str(exc))}")
                continue

            try:
                payload = extract_json_payload(raw)
                raw_documents = ensure_json_object_with_list(payload, "documents")
            except Exception as exc:
                log(f"[documents:{memory_type}] invalid model JSON: {truncate_for_log(str(exc))}")
                continue

            added = 0
            for raw_document in raw_documents:
                sanitized = sanitize_recall_document(raw_document, memory_type)
                if not sanitized:
                    continue
                normalized_text = normalize_spaces(sanitized["text"]).lower()
                if normalized_text in seen_texts:
                    continue

                doc_id = f"doc-{next_id:04d}"
                seen_texts.add(normalized_text)
                documents.append(
                    {
                        "id": doc_id,
                        "relative_path": f"{memory_type}/{doc_id}.md",
                        "kind": sanitized["kind"],
                        "text": sanitized["text"],
                        "memory_type": sanitized["memory_type"],
                    }
                )
                next_id += 1
                counts_by_type[memory_type] += 1
                current += 1
                added += 1
                if current >= target:
                    break

            if added > 0:
                persist(documents)

            log(
                f"[documents:{memory_type}] progress {current}/{target} "
                f"(+{current - before_count}, attempt {attempts}/{config.max_attempts_per_bucket})"
            )

        if current < target:
            raise RuntimeError(
                f"Could not generate enough recall documents for type '{memory_type}'. "
                f"Expected {target}, got {current} after {attempts} attempts."
            )

        log(f"[documents:{memory_type}] complete {current}/{target}")

    return documents


def generate_recall_queries(
    client: CodexClient,
    config: GenerationConfig,
    rng: random.Random,
    documents: Sequence[Dict[str, Any]],
    existing_queries: Sequence[Dict[str, Any]],
    persist: Callable[[Sequence[Dict[str, Any]]], None],
) -> List[Dict[str, Any]]:
    queries = list(existing_queries)
    seen_queries = {str(query["query"]).lower() for query in queries}
    valid_doc_ids = {str(doc["id"]) for doc in documents}

    next_id = next_id_number(queries, "q")

    system_prompt = (
        "You create retrieval benchmark queries. "
        "Return strict JSON matching the schema."
    )

    max_attempts = max(30, config.max_attempts_per_bucket * 3)
    attempts = 0

    if queries:
        log(f"[queries] resuming with {len(queries)} existing queries")

    log(f"[queries] target={config.recall_queries}, current={len(queries)}")

    while len(queries) < config.recall_queries and attempts < max_attempts:
        attempts += 1
        needed = config.recall_queries - len(queries)
        batch_size = min(config.queries_batch_size, needed)
        before_count = len(queries)

        candidate_count = min(
            max(batch_size * config.query_candidate_multiplier, config.query_candidate_min),
            len(documents),
        )
        candidate_docs = rng.sample(list(documents), k=candidate_count)
        catalog_snippet = build_document_catalog_snippet(
            candidate_docs,
            limit=candidate_count,
            summary_chars=config.query_summary_chars,
        )

        user_prompt = f"""
Use this document catalog and generate EXACTLY {batch_size} objects in `queries`.

Candidate document catalog (JSON lines):
{catalog_snippet}

Rules:
- Each query must map to 1 to 3 relevant_document_ids.
- relevant_document_ids MUST come from the provided candidate IDs only.
- Query phrasing should be realistic and not copied directly from summary text.
- Mix query styles: direct, paraphrased, and mildly ambiguous.
- Include memory_types for roughly 30-50% of queries. Use null when absent.
- memory_types values must be from: {", ".join(MEMORY_TYPES)}.
"""

        try:
            raw = client.create_message(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=QUERIES_OUTPUT_SCHEMA,
                progress_label=f"[queries] batch {attempts}/{max_attempts} need={needed}",
            )
        except Exception as exc:
            log(f"[queries] request failed: {truncate_for_log(str(exc))}")
            continue

        try:
            payload = extract_json_payload(raw)
            raw_queries = ensure_json_object_with_list(payload, "queries")
        except Exception as exc:
            log(f"[queries] invalid model JSON: {truncate_for_log(str(exc))}")
            continue

        added = 0
        for raw_query in raw_queries:
            sanitized = sanitize_recall_query(raw_query, valid_doc_ids=valid_doc_ids)
            if not sanitized:
                continue

            normalized_query = sanitized["query"].lower()
            if normalized_query in seen_queries:
                continue

            row: Dict[str, Any] = {
                "id": f"q-{next_id:04d}",
                "query": sanitized["query"],
                "relevant_document_ids": sanitized["relevant_document_ids"],
            }
            if "memory_types" in sanitized:
                row["memory_types"] = sanitized["memory_types"]

            queries.append(row)
            next_id += 1
            seen_queries.add(normalized_query)
            added += 1

            if len(queries) >= config.recall_queries:
                break

        if added > 0:
            persist(queries)

        log(
            f"[queries] progress {len(queries)}/{config.recall_queries} "
            f"(+{len(queries) - before_count}, attempt {attempts}/{max_attempts})"
        )

    if len(queries) < config.recall_queries:
        raise RuntimeError(
            f"Could not generate enough recall queries. Expected {config.recall_queries}, "
            f"got {len(queries)} after {attempts} attempts."
        )

    return queries


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Memory.swift eval data via codex exec (atomic batches, resumable)."
    )
    parser.add_argument("--dataset-root", default="Evals", help="Dataset root folder (default: Evals).")
    parser.add_argument("--workspace", default=".", help="Workspace passed to codex exec -C (default: current dir).")
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI binary name or path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")

    parser.add_argument("--storage-cases", type=int, default=240, help="Total storage cases (default: 240).")
    parser.add_argument("--recall-documents", type=int, default=400, help="Total recall documents (default: 400).")
    parser.add_argument("--recall-queries", type=int, default=240, help="Total recall queries (default: 240).")

    parser.add_argument("--storage-batch-size", type=int, default=6, help="Storage generation batch size (default: 6).")
    parser.add_argument("--documents-batch-size", type=int, default=8, help="Recall document generation batch size (default: 8).")
    parser.add_argument("--queries-batch-size", type=int, default=10, help="Recall query generation batch size (default: 10).")

    parser.add_argument("--max-attempts-per-bucket", type=int, default=24, help="Generation retries per type bucket.")
    parser.add_argument("--max-retries-per-request", type=int, default=3, help="Retries per codex exec call.")
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=240,
        help="Timeout for each codex exec call in seconds (default: 240).",
    )

    parser.add_argument(
        "--query-candidate-multiplier",
        type=int,
        default=4,
        help="Candidates per query batch item for query generation (default: 4).",
    )
    parser.add_argument(
        "--query-candidate-min",
        type=int,
        default=24,
        help="Minimum candidate docs in each query-generation prompt (default: 24).",
    )
    parser.add_argument(
        "--query-summary-chars",
        type=int,
        default=120,
        help="Summary snippet length per candidate doc in query prompts (default: 120).",
    )

    parser.add_argument(
        "--codex-reasoning-effort",
        default="low",
        help="Sets codex `model_reasoning_effort` config override (default: low).",
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

    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Start fresh and overwrite existing files.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output files when present (default: true).",
    )

    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.storage_cases <= 0 or args.recall_documents <= 0 or args.recall_queries <= 0:
        raise ValueError("Counts must all be > 0.")
    if args.storage_batch_size <= 0 or args.documents_batch_size <= 0 or args.queries_batch_size <= 0:
        raise ValueError("Batch sizes must be > 0.")
    if args.max_attempts_per_bucket <= 0 or args.max_retries_per_request <= 0:
        raise ValueError("Attempt/retry values must be > 0.")
    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be > 0.")
    if args.query_candidate_multiplier <= 0 or args.query_candidate_min <= 0 or args.query_summary_chars <= 0:
        raise ValueError("Query candidate and summary settings must be > 0.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    dataset_root = Path(args.dataset_root)
    workspace = Path(args.workspace).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    storage_path = dataset_root / "storage_cases.jsonl"
    documents_path = dataset_root / "recall_documents.jsonl"
    queries_path = dataset_root / "recall_queries.jsonl"

    existing_storage: List[Dict[str, Any]] = []
    existing_documents: List[Dict[str, Any]] = []
    existing_queries: List[Dict[str, Any]] = []

    if args.overwrite:
        for path in (storage_path, documents_path, queries_path):
            if path.exists():
                path.unlink()
        log("Overwrite enabled: starting from empty output files.")
    elif args.resume:
        if storage_path.exists():
            existing_storage = normalize_existing_storage_records(load_jsonl(storage_path))
        if documents_path.exists():
            existing_documents = normalize_existing_document_records(load_jsonl(documents_path))
        if queries_path.exists():
            if not documents_path.exists():
                raise RuntimeError("Cannot resume queries without existing recall_documents.jsonl.")
            existing_queries = normalize_existing_query_records(
                load_jsonl(queries_path),
                valid_doc_ids={str(doc["id"]) for doc in existing_documents},
            )

        if existing_storage or existing_documents or existing_queries:
            log(
                "Resume enabled: loaded existing records "
                f"(storage={len(existing_storage)}, docs={len(existing_documents)}, queries={len(existing_queries)})."
            )
    else:
        existing_paths = [str(path) for path in (storage_path, documents_path, queries_path) if path.exists()]
        if existing_paths:
            raise RuntimeError(
                f"Refusing to overwrite existing files: {', '.join(existing_paths)}. "
                "Use --overwrite or --resume."
            )

    assert_storage_counts_within_targets(existing_storage, args.storage_cases)
    assert_document_counts_within_targets(existing_documents, args.recall_documents)
    if len(existing_queries) > args.recall_queries:
        raise ValueError(
            f"Existing recall queries exceed target: have {len(existing_queries)}, target {args.recall_queries}."
        )

    cfg = GenerationConfig(
        storage_cases=args.storage_cases,
        recall_documents=args.recall_documents,
        recall_queries=args.recall_queries,
        storage_batch_size=args.storage_batch_size,
        documents_batch_size=args.documents_batch_size,
        queries_batch_size=args.queries_batch_size,
        max_attempts_per_bucket=args.max_attempts_per_bucket,
        max_retries_per_request=args.max_retries_per_request,
        request_timeout_seconds=args.request_timeout_seconds,
        query_candidate_multiplier=args.query_candidate_multiplier,
        query_candidate_min=args.query_candidate_min,
        query_summary_chars=args.query_summary_chars,
        seed=args.seed,
    )

    codex_config_overrides: List[str] = []
    if args.codex_reasoning_effort:
        codex_config_overrides.append(f'model_reasoning_effort="{args.codex_reasoning_effort}"')
    if args.disable_xcode_mcp:
        codex_config_overrides.append("mcp_servers.xcode.enabled=false")
    codex_config_overrides.extend(args.codex_config)

    rng = random.Random(cfg.seed)
    client = CodexClient(
        codex_bin=args.codex_bin,
        model=args.model,
        workspace=workspace,
        timeout_seconds=cfg.request_timeout_seconds,
        max_retries_per_request=cfg.max_retries_per_request,
        config_overrides=codex_config_overrides,
    )
    client.ensure_login()

    def persist_storage(records: Sequence[Dict[str, Any]]) -> None:
        write_jsonl_atomic(storage_path, records)

    def persist_documents(records: Sequence[Dict[str, Any]]) -> None:
        write_jsonl_atomic(documents_path, records)

    def persist_queries(records: Sequence[Dict[str, Any]]) -> None:
        write_jsonl_atomic(queries_path, records)

    log(f"Using codex binary: {client.codex_bin}")
    log(f"Using model: {args.model}")
    log(f"Workspace: {workspace}")
    log(f"Dataset root: {dataset_root}")
    log(f"Request timeout (seconds): {cfg.request_timeout_seconds}")
    if codex_config_overrides:
        log(f"Codex config overrides: {', '.join(codex_config_overrides)}")

    log("Generating storage cases...")
    storage_cases = generate_storage_cases(client, cfg, existing_storage, persist_storage)
    log(f"Generated {len(storage_cases)} storage cases.")

    log("Generating recall documents...")
    recall_documents = generate_recall_documents(client, cfg, existing_documents, persist_documents)
    log(f"Generated {len(recall_documents)} recall documents.")

    valid_doc_ids = {str(doc["id"]) for doc in recall_documents}
    existing_queries = normalize_existing_query_records(existing_queries, valid_doc_ids=valid_doc_ids)

    log("Generating recall queries...")
    recall_queries = generate_recall_queries(
        client,
        cfg,
        rng,
        recall_documents,
        existing_queries,
        persist_queries,
    )
    log(f"Generated {len(recall_queries)} recall queries.")

    # Final durability pass.
    persist_storage(storage_cases)
    persist_documents(recall_documents)
    persist_queries(recall_queries)

    log("Done.")
    log(f"Wrote {storage_path}")
    log(f"Wrote {documents_path}")
    log(f"Wrote {queries_path}")
    log("Next:")
    log(f"  swift run memory_eval run --profile baseline --dataset-root {dataset_root}")
    log(f"  swift run memory_eval run --profile full_apple --dataset-root {dataset_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
