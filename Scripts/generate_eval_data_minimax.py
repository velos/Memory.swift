#!/usr/bin/env python3
"""
Generate eval datasets for Memory.swift using MiniMax's Anthropic-compatible API.

Outputs:
  - Evals/storage_cases.jsonl
  - Evals/recall_documents.jsonl
  - Evals/recall_queries.jsonl

Environment variables:
  - ANTHROPIC_BASE_URL (default: https://api.minimax.io/anthropic)
  - ANTHROPIC_API_KEY or MINIMAX_API_KEY or ANTHROPIC_AUTH_TOKEN (required)
  - MINIMAX_MODEL (default: MiniMax-M2.5)

Usage:
  python3 Scripts/generate_eval_data_minimax.py --dataset-root Evals/general --dataset-mode general --env-file .env
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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

QUERY_STYLE_ROTATION: List[Dict[str, str]] = [
    {
        "label": "direct",
        "guidance": (
            "Generate direct, straightforward queries that use similar vocabulary "
            "to the source documents. These should be 'easy' difficulty."
        ),
        "default_difficulty": "easy",
    },
    {
        "label": "paraphrased",
        "guidance": (
            "Generate paraphrased queries that express the same information need "
            "but use DIFFERENT vocabulary from the source documents. Avoid copying "
            "keywords from the document summaries. These should be 'medium' difficulty."
        ),
        "default_difficulty": "medium",
    },
    {
        "label": "adversarial",
        "guidance": (
            "Generate adversarial queries: near-miss paraphrases that could easily "
            "match wrong documents, multi-hop queries that require combining information "
            "from the document, or queries using abstract/conceptual language with ZERO "
            "shared surface keywords with the target documents. "
            "These should be 'hard' difficulty."
        ),
        "default_difficulty": "hard",
    },
]

DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"
DEFAULT_MODEL = "MiniMax-M2.5"
DEFAULT_PROMPT_VERSION = "2026-03-15"

DATASET_MODES = ("general", "tech", "longmemeval-typed-queries", "adversarial-augment")

SOFTWARE_ENGINEERING_KEYWORDS = {
    "api", "apis", "backend", "build", "ci", "client", "code", "commit", "container",
    "database", "db", "debug", "deploy", "deployment", "endpoint", "incident", "infra",
    "integration", "latency", "log", "migration", "monitoring", "pipeline", "postmortem",
    "pr", "prod", "release", "repository", "rollback", "runbook", "schema", "service",
    "slo", "sql", "staging", "test", "tests", "token", "typescript", "swift",
    "kubernetes", "docker", "cache", "observability", "auth", "frontend", "ios",
}

NON_TECH_KEYWORDS = {
    "birthday", "calories", "cat", "cook", "cooking", "diet", "dog", "flight", "garden",
    "gym", "hotel", "marathon", "movie", "music", "nutrition", "parenting", "pet",
    "photosynthesis", "recipe", "restaurant", "sauce", "school", "tomato", "vacation",
    "wedding", "workout",
}


@dataclass
class DomainProfile:
    name: str
    prose_style: str
    overlap_hints: List[str]
    storage_style: str


DOMAIN_PROFILES: Dict[str, DomainProfile] = {
    "general": DomainProfile(
        name="general",
        prose_style=(
            "realistic personal knowledge base entries -- notes, decisions, observations, "
            "and records that a person or AI agent would store across ALL areas of life and work. "
            "Cover diverse domains: cooking/recipes, travel, health/fitness, personal finance, "
            "home maintenance, hobbies/crafts, parenting/family, books/movies/media, "
            "coding/technical projects, work projects/management, learning/courses, "
            "relationships/social, creative writing/art, legal/contracts, pets/animals, "
            "gardening, music, sports, and more. "
            "Each batch should span at LEAST 3 different life domains. "
            "Do NOT focus exclusively on software engineering or workplace topics."
        ),
        overlap_hints=[
            "schedule", "appointment", "recipe", "travel", "budget",
            "health", "meeting", "project", "family", "maintenance",
            "learning", "hobby", "deadline", "purchase", "conversation",
            "recommendation", "workout", "garden", "pet", "book",
            "movie", "music", "recipe", "repair", "investment",
            "course", "birthday", "medication", "contract", "flight",
            "restaurant", "insurance", "subscription", "goal", "habit",
        ],
        storage_style=(
            "Use natural, realistic personal notes across diverse life domains "
            "(cooking, travel, health, finance, hobbies, family, work, learning, etc.) "
            "in markdown-like prose. Do NOT focus exclusively on software engineering."
        ),
    ),
    "technical": DomainProfile(
        name="technical",
        prose_style=(
            "realistic software-engineering project prose -- incident notes, migration plans, "
            "debugging logs, runbooks, architecture decisions, API behavior summaries, "
            "release notes, and engineering planning artifacts. Do NOT produce cooking, travel, "
            "health, finance, personal-life, or generic workplace-only content."
        ),
        overlap_hints=[
            "deployment", "rollback", "incident", "release",
            "migration", "api", "runbook", "latency",
        ],
        storage_style=(
            "Use natural, realistic software-engineering notes and docs in markdown-like prose. "
            "Stay inside code, infrastructure, incidents, architecture, debugging, releases, "
            "testing, and operational coordination."
        ),
    ),
}


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def truncate_for_log(value: str, limit: int = 220) -> str:
    compact = normalize_spaces(value)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


@dataclass
class GenerationConfig:
    dataset_mode: str
    storage_cases: int
    recall_documents: int
    recall_queries: int
    storage_batch_size: int
    documents_batch_size: int
    queries_batch_size: int
    temperature: float
    max_tokens: int
    max_attempts_per_bucket: int
    max_retries_per_request: int
    request_timeout_seconds: int
    seed: int
    domain_profile: DomainProfile = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.domain_profile is None:
            self.domain_profile = DOMAIN_PROFILES["general"]


class MiniMaxAnthropicClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_retries_per_request: int,
        timeout_seconds: int = 120,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries_per_request = max_retries_per_request
        self.timeout_seconds = timeout_seconds

    def create_message(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        progress_label: Optional[str] = None,
    ) -> str:
        url = f"{self.base_url}/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }
            ],
        }

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries_per_request + 1):
            started_at = time.time()
            label = progress_label or "API request"
            log(f"{label}: request {attempt}/{self.max_retries_per_request}")
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "content-type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                content = parsed.get("content", [])
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(str(block.get("text", "")))
                elapsed = time.time() - started_at
                log(f"{label}: response in {elapsed:.1f}s")
                if texts:
                    return "\n".join(texts).strip()
                return body.strip()
            except urllib.error.HTTPError as exc:
                error_payload = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {error_payload}"
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                http.client.IncompleteRead,
                json.JSONDecodeError,
            ) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < self.max_retries_per_request:
                sleep_seconds = min(12.0, 1.8 ** attempt)
                log(
                    f"{label}: failed ({truncate_for_log(last_error or 'unknown error')}); "
                    f"retrying in {sleep_seconds:.1f}s"
                )
                time.sleep(sleep_seconds)
            else:
                log(f"{label}: failed ({truncate_for_log(last_error or 'unknown error')})")

        raise RuntimeError(f"MiniMax request failed after retries: {last_error}")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(raw_line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def distribute(total: int, buckets: int) -> List[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def tokenize_keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_+#.-]+", text.lower()))


def looks_software_engineering_focused(text: str) -> bool:
    tokens = tokenize_keywords(text)
    tech_hits = len(tokens.intersection(SOFTWARE_ENGINEERING_KEYWORDS))
    non_tech_hits = len(tokens.intersection(NON_TECH_KEYWORDS))
    return tech_hits >= 2 and tech_hits > non_tech_hits


def dataset_mode_requires_software_focus(dataset_mode: str) -> bool:
    return dataset_mode == "tech"


def extract_json_payload(raw: str) -> Any:
    raw = raw.strip()

    # Direct parse first.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences.
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    # Try first JSON object.
    object_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if object_match:
        candidate = object_match.group(0).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try first JSON array.
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


def sanitize_storage_case(raw: Dict[str, Any], expected_type: str, dataset_mode: str) -> Optional[Dict[str, Any]]:
    text = normalize_spaces(str(raw.get("text", "")))
    if not text:
        return None

    wc = word_count(text)
    if wc < 50 or wc > 220:
        return None
    if dataset_mode_requires_software_focus(dataset_mode) and not looks_software_engineering_focused(text):
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


def sanitize_recall_document(raw: Dict[str, Any], expected_type: str, dataset_mode: str) -> Optional[Dict[str, Any]]:
    text = normalize_spaces(str(raw.get("text", "")))
    if not text:
        return None

    wc = word_count(text)
    if wc < 60 or wc > 260:
        return None
    if dataset_mode_requires_software_focus(dataset_mode) and not looks_software_engineering_focused(text):
        return None

    return {
        "kind": "markdown",
        "text": text,
        "memory_type": expected_type,
    }


def sanitize_recall_query(
    raw: Dict[str, Any],
    *,
    valid_doc_ids: set[str],
    default_difficulty: str = "medium",
) -> Optional[Dict[str, Any]]:
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
        filtered = []
        for item in memory_types_raw:
            if isinstance(item, str):
                memory_type = item.strip().lower()
                if memory_type in MEMORY_TYPES and memory_type not in filtered:
                    filtered.append(memory_type)
        if filtered:
            memory_types = filtered

    difficulty_raw = str(raw.get("difficulty", default_difficulty)).strip().lower()
    difficulty = difficulty_raw if difficulty_raw in DIFFICULTY_LEVELS else default_difficulty

    result: Dict[str, Any] = {
        "query": query,
        "relevant_document_ids": relevant,
        "difficulty": difficulty,
    }
    if memory_types:
        result["memory_types"] = memory_types
    return result


def render_jsonl(records: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"


def generate_storage_cases(
    client: MiniMaxAnthropicClient,
    config: GenerationConfig,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    seen_texts: set[str] = set()
    per_type_targets = dict(zip(MEMORY_TYPES, distribute(config.storage_cases, len(MEMORY_TYPES))))

    system_prompt = (
        "You create high-quality evaluation datasets for memory systems.\n"
        "Return only strict JSON. No markdown. No commentary."
    )

    for memory_type in MEMORY_TYPES:
        target = per_type_targets[memory_type]
        collected: List[Dict[str, Any]] = []
        attempts = 0
        log(f"[storage:{memory_type}] target={target}")
        while len(collected) < target and attempts < config.max_attempts_per_bucket:
            attempts += 1
            needed = target - len(collected)
            batch_size = min(config.storage_batch_size, needed)
            before_count = len(collected)

            user_prompt = f"""
Generate EXACTLY {batch_size} JSON objects for storage-memory classification eval.

Required output shape:
{{
  "cases": [
    {{
      "text": "string",
      "required_spans": ["string", "string", "..."]
    }}
  ]
}}

Rules:
- Every case MUST be strongly {memory_type} memory type.
- {config.domain_profile.storage_style}
- Keep text between 70 and 180 words.
- Include 2 to 4 required_spans.
- Each required span MUST appear verbatim in text (case-sensitive).
- Avoid ambiguous multi-label examples.
- Do not include IDs; IDs are assigned by caller.
"""
            raw = client.create_message(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                progress_label=(
                    f"[storage:{memory_type}] batch {attempts}/{config.max_attempts_per_bucket} "
                    f"need={needed}"
                ),
            )

            try:
                payload = extract_json_payload(raw)
                raw_cases = ensure_json_object_with_list(payload, "cases")
            except Exception:
                continue

            for raw_case in raw_cases:
                sanitized = sanitize_storage_case(raw_case, memory_type, config.dataset_mode)
                if not sanitized:
                    continue
                normalized_text = normalize_spaces(sanitized["text"]).lower()
                if normalized_text in seen_texts:
                    continue
                seen_texts.add(normalized_text)
                collected.append(sanitized)
                if len(collected) >= target:
                    break

            added = len(collected) - before_count
            log(
                f"[storage:{memory_type}] progress {len(collected)}/{target} "
                f"(+{added}, attempt {attempts}/{config.max_attempts_per_bucket})"
            )

        if len(collected) < target:
            raise RuntimeError(
                f"Could not generate enough storage cases for type '{memory_type}'. "
                f"Expected {target}, got {len(collected)} after {attempts} attempts."
            )

        cases.extend(collected)
        log(f"[storage:{memory_type}] complete {len(collected)}/{target}")

    rng.shuffle(cases)
    final_cases: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        final_cases.append(
            {
                "id": f"storage-{idx:04d}",
                "kind": case["kind"],
                "text": case["text"],
                "expected_memory_type": case["expected_memory_type"],
                "required_spans": case["required_spans"],
            }
        )
    return final_cases


def generate_recall_documents(
    client: MiniMaxAnthropicClient,
    config: GenerationConfig,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    seen_texts: set[str] = set()
    per_type_targets = dict(zip(MEMORY_TYPES, distribute(config.recall_documents, len(MEMORY_TYPES))))

    system_prompt = (
        "You create high-quality retrieval evaluation corpora.\n"
        "Return only strict JSON. No markdown. No commentary."
    )

    overlap_hints = config.domain_profile.overlap_hints

    for memory_type in MEMORY_TYPES:
        target = per_type_targets[memory_type]
        collected: List[Dict[str, Any]] = []
        attempts = 0
        log(f"[documents:{memory_type}] target={target}")
        while len(collected) < target and attempts < config.max_attempts_per_bucket:
            attempts += 1
            needed = target - len(collected)
            batch_size = min(config.documents_batch_size, needed)
            hint = overlap_hints[(attempts - 1) % len(overlap_hints)]
            before_count = len(collected)

            user_prompt = f"""
Generate EXACTLY {batch_size} JSON objects for retrieval corpus documents.

Required output shape:
{{
  "documents": [
    {{
      "text": "string"
    }}
  ]
}}

Rules:
- Every document MUST primarily represent {memory_type} memory.
- Keep each text between 90 and 220 words.
- Use {config.domain_profile.prose_style}
- Include at least one lexical term related to "{hint}" to create challenging near-overlap retrieval conditions.
- Do not include IDs, paths, or memory_type fields.
"""
            raw = client.create_message(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                progress_label=(
                    f"[documents:{memory_type}] batch {attempts}/{config.max_attempts_per_bucket} "
                    f"need={needed}"
                ),
            )

            try:
                payload = extract_json_payload(raw)
                raw_documents = ensure_json_object_with_list(payload, "documents")
            except Exception:
                continue

            for raw_document in raw_documents:
                sanitized = sanitize_recall_document(raw_document, memory_type, config.dataset_mode)
                if not sanitized:
                    continue
                normalized_text = normalize_spaces(sanitized["text"]).lower()
                if normalized_text in seen_texts:
                    continue
                seen_texts.add(normalized_text)
                collected.append(sanitized)
                if len(collected) >= target:
                    break

            added = len(collected) - before_count
            log(
                f"[documents:{memory_type}] progress {len(collected)}/{target} "
                f"(+{added}, attempt {attempts}/{config.max_attempts_per_bucket})"
            )

        if len(collected) < target:
            raise RuntimeError(
                f"Could not generate enough recall documents for type '{memory_type}'. "
                f"Expected {target}, got {len(collected)} after {attempts} attempts."
            )

        documents.extend(collected)
        log(f"[documents:{memory_type}] complete {len(collected)}/{target}")

    rng.shuffle(documents)
    final_documents: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents, start=1):
        doc_id = f"doc-{idx:04d}"
        relative_path = f"{doc['memory_type']}/{doc_id}.md"
        final_documents.append(
            {
                "id": doc_id,
                "relative_path": relative_path,
                "kind": doc["kind"],
                "text": doc["text"],
                "memory_type": doc["memory_type"],
            }
        )
    return final_documents


def build_document_catalog_snippet(documents: Sequence[Dict[str, Any]], limit: int) -> str:
    lines: List[str] = []
    for doc in documents[:limit]:
        snippet = normalize_spaces(doc["text"])[:180]
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


def generate_recall_queries(
    client: MiniMaxAnthropicClient,
    config: GenerationConfig,
    rng: random.Random,
    documents: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    seen_queries: set[str] = set()
    valid_doc_ids = {doc["id"] for doc in documents}

    system_prompt = (
        "You create retrieval benchmark queries.\n"
        "Return only strict JSON. No markdown. No commentary."
    )

    attempts = 0
    max_attempts = max(30, config.max_attempts_per_bucket * 3)
    log(f"[queries] target={config.recall_queries}")
    while len(queries) < config.recall_queries and attempts < max_attempts:
        attempts += 1
        needed = config.recall_queries - len(queries)
        batch_size = min(config.queries_batch_size, needed)
        before_count = len(queries)

        style = QUERY_STYLE_ROTATION[(attempts - 1) % len(QUERY_STYLE_ROTATION)]
        style_guidance = style["guidance"]
        default_difficulty = style["default_difficulty"]

        candidate_count = min(max(batch_size * 6, 40), len(documents))
        candidate_docs = rng.sample(list(documents), k=candidate_count)
        catalog_snippet = build_document_catalog_snippet(candidate_docs, limit=candidate_count)

        user_prompt = f"""
Use the following candidate documents and generate EXACTLY {batch_size} retrieval queries.

Candidate document catalog (JSON lines):
{catalog_snippet}

Required output shape:
{{
  "queries": [
    {{
      "query": "string",
      "relevant_document_ids": ["doc-0001"],
      "memory_types": ["procedural"],
      "difficulty": "medium"
    }}
  ]
}}

Style guidance for this batch:
{style_guidance}

Rules:
- Each query must have 1 to 3 relevant_document_ids.
- relevant_document_ids MUST be chosen only from the provided candidate IDs.
- Query should be realistic user wording, not copied from summaries.
- Include memory_types for roughly 30-50% of queries (optional otherwise).
- memory_types values must be from: {", ".join(MEMORY_TYPES)}.
- difficulty must be one of: easy, medium, hard.
- "easy" queries share vocabulary with their target documents.
- "medium" queries paraphrase the information need with different words.
- "hard" queries use abstract/conceptual language, near-miss phrasing, or require combining information.
"""
        raw = client.create_message(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            progress_label=(
                f"[queries:{style['label']}] batch {attempts}/{max_attempts} need={needed}"
            ),
        )

        try:
            payload = extract_json_payload(raw)
            raw_queries = ensure_json_object_with_list(payload, "queries")
        except Exception:
            continue

        for raw_query in raw_queries:
            sanitized = sanitize_recall_query(
                raw_query,
                valid_doc_ids=valid_doc_ids,
                default_difficulty=default_difficulty,
            )
            if not sanitized:
                continue
            normalized_query = sanitized["query"].lower()
            if normalized_query in seen_queries:
                continue
            seen_queries.add(normalized_query)
            queries.append(sanitized)
            if len(queries) >= config.recall_queries:
                break

        added = len(queries) - before_count
        log(
            f"[queries:{style['label']}] progress {len(queries)}/{config.recall_queries} "
            f"(+{added}, attempt {attempts}/{max_attempts})"
        )

    if len(queries) < config.recall_queries:
        raise RuntimeError(
            f"Could not generate enough recall queries. Expected {config.recall_queries}, "
            f"got {len(queries)} after {attempts} attempts."
        )

    final_queries: List[Dict[str, Any]] = []
    for idx, query in enumerate(queries[: config.recall_queries], start=1):
        row: Dict[str, Any] = {
            "id": f"q-{idx:04d}",
            "query": query["query"],
            "relevant_document_ids": query["relevant_document_ids"],
            "difficulty": query.get("difficulty", "medium"),
        }
        if "memory_types" in query:
            row["memory_types"] = query["memory_types"]
        final_queries.append(row)
    return final_queries


def validate_query_memory_type_consistency(
    queries: List[Dict[str, Any]],
    documents: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove or fix queries whose memory_types filter excludes their own relevant documents."""
    doc_type_map: Dict[str, str] = {}
    for doc in documents:
        doc_type_map[doc["id"]] = doc.get("memory_type", "")

    fixed = 0
    for query in queries:
        filter_types = query.get("memory_types")
        if not filter_types:
            continue

        relevant_ids = query.get("relevant_document_ids", [])
        relevant_types = {doc_type_map.get(did, "") for did in relevant_ids} - {""}

        if not relevant_types:
            continue

        if not relevant_types.intersection(set(filter_types)):
            corrected = sorted(relevant_types)
            log(
                f"[validate] {query.get('id', '?')}: memory_types={filter_types} "
                f"excluded relevant types {sorted(relevant_types)}; "
                f"corrected to {corrected}"
            )
            query["memory_types"] = corrected
            fixed += 1

    if fixed:
        log(f"[validate] Fixed {fixed} queries with contradictory memory_types filters.")
    else:
        log("[validate] All query memory_types filters are consistent.")
    return queries


def log_difficulty_distribution(queries: List[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    for q in queries:
        d = q.get("difficulty", "medium")
        counts[d] = counts.get(d, 0) + 1
    parts = [f"{level}={counts.get(level, 0)}" for level in DIFFICULTY_LEVELS]
    log(f"[queries] difficulty distribution: {', '.join(parts)}")


def write_dataset_manifest(
    dataset_root: Path,
    *,
    dataset_mode: str,
    model: str,
    config: GenerationConfig,
    notes: List[str],
) -> None:
    storage_path = dataset_root / "storage_cases.jsonl"
    queries_path = dataset_root / "recall_queries.jsonl"
    storage_records = _read_jsonl(storage_path) if storage_path.exists() else []
    query_records = _read_jsonl(queries_path) if queries_path.exists() else []
    typed_queries = sum(1 for record in query_records if record.get("memory_types"))

    if dataset_mode == "tech":
        primary_use = "secondary"
        provenance = "synthetic_minimax"
        synthetic_status = "synthetic"
    elif dataset_mode == "general":
        primary_use = "primary"
        provenance = "synthetic_minimax"
        synthetic_status = "synthetic"
    elif dataset_mode == "longmemeval-typed-queries":
        primary_use = "primary"
        provenance = "external_converted_plus_synthetic_query_metadata"
        synthetic_status = "hybrid"
    else:
        primary_use = "primary"
        provenance = "synthetic_minimax_augmentation"
        synthetic_status = "synthetic"

    manifest = {
        "dataset": dataset_root.name,
        "provenance": provenance,
        "synthetic_status": synthetic_status,
        "primary_use": primary_use,
        "typing_coverage": {
            "storage_cases": {
                "typed": len(storage_records),
                "total": len(storage_records),
            },
            "recall_queries": {
                "typed": typed_queries,
                "total": len(query_records),
            },
        },
        "known_limits": notes,
        "recommended_weight": primary_use,
        "generation": {
            "backend": "minimax_anthropic_api",
            "model": model,
            "prompt_version": DEFAULT_PROMPT_VERSION,
            "dataset_mode": dataset_mode,
            "seed": config.seed,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    (dataset_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _summarize_document(text: str, limit: int = 220) -> str:
    return truncate_for_log(text, limit=limit)


def augment_longmemeval_typed_queries(
    client: MiniMaxAnthropicClient,
    config: GenerationConfig,
    dataset_root: Path,
) -> List[Dict[str, Any]]:
    documents = _read_jsonl(dataset_root / "recall_documents.jsonl")
    queries = _read_jsonl(dataset_root / "recall_queries.jsonl")
    document_map = {record["id"]: record for record in documents}
    query_map = {record["id"]: dict(record) for record in queries}
    pending = [record for record in queries if not record.get("memory_types")]
    if not pending:
        return list(query_map.values())

    system_prompt = (
        "You label retrieval benchmark queries for memory systems.\n"
        "Return only strict JSON. No markdown. No commentary."
    )

    batch_size = max(1, min(config.queries_batch_size, 20))
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        lines: List[str] = []
        for query in batch:
            relevant_docs = []
            for doc_id in query["relevant_document_ids"]:
                document = document_map.get(doc_id)
                if not document:
                    continue
                relevant_docs.append(
                    {
                        "id": doc_id,
                        "memory_type": document.get("memory_type"),
                        "summary": _summarize_document(document["text"]),
                    }
                )
            lines.append(
                json.dumps(
                    {
                        "id": query["id"],
                        "query": query["query"],
                        "relevant_docs": relevant_docs,
                    },
                    ensure_ascii=False,
                )
            )

        user_prompt = f"""
Label the following queries with memory types and difficulty.

Input queries (JSON lines):
{chr(10).join(lines)}

Required output shape:
{{
  "queries": [
    {{
      "id": "q-0001",
      "memory_types": ["semantic"],
      "difficulty": "medium"
    }}
  ]
}}

Rules:
- Choose 1 or 2 memory_types from: {", ".join(MEMORY_TYPES)}.
- Use the query wording and relevant doc summaries together.
- difficulty must be one of: easy, medium, hard.
- Prefer the most specific memory type that best matches the retrieval intent.
"""

        raw = client.create_message(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=max(0.2, config.temperature - 0.1),
            max_tokens=config.max_tokens,
            progress_label=f"[longmemeval-typed-queries] batch {batch_start // batch_size + 1}",
        )

        payload = extract_json_payload(raw)
        labeled = ensure_json_object_with_list(payload, "queries")
        for item in labeled:
            query_id = str(item.get("id", "")).strip()
            if query_id not in query_map:
                continue
            memory_types = [
                memory_type
                for memory_type in [str(value).strip().lower() for value in item.get("memory_types", [])]
                if memory_type in MEMORY_TYPES
            ]
            difficulty = str(item.get("difficulty", "medium")).strip().lower()
            if difficulty not in DIFFICULTY_LEVELS:
                difficulty = "medium"

            if not memory_types:
                relevant_doc_types = []
                for doc_id in query_map[query_id]["relevant_document_ids"]:
                    memory_type = str(document_map.get(doc_id, {}).get("memory_type", "")).strip().lower()
                    if memory_type in MEMORY_TYPES and memory_type not in relevant_doc_types:
                        relevant_doc_types.append(memory_type)
                memory_types = relevant_doc_types[:2]

            if memory_types:
                query_map[query_id]["memory_types"] = memory_types
            query_map[query_id]["difficulty"] = difficulty

    return [query_map[record["id"]] for record in queries]


def augment_adversarial_queries(
    client: MiniMaxAnthropicClient,
    config: GenerationConfig,
    rng: random.Random,
    dataset_root: Path,
) -> List[Dict[str, Any]]:
    documents = _read_jsonl(dataset_root / "recall_documents.jsonl")
    existing_queries = _read_jsonl(dataset_root / "recall_queries.jsonl")
    valid_doc_ids = {record["id"] for record in documents}
    seen_queries = {normalize_spaces(record["query"]).lower() for record in existing_queries}
    generated: List[Dict[str, Any]] = []
    target = config.recall_queries
    attempts = 0
    max_attempts = max(20, config.max_attempts_per_bucket * 4)

    system_prompt = (
        "You create hard retrieval benchmark queries.\n"
        "Return only strict JSON. No markdown. No commentary."
    )

    while len(generated) < target and attempts < max_attempts:
        attempts += 1
        needed = target - len(generated)
        batch_size = min(config.queries_batch_size, needed)
        candidate_docs = rng.sample(list(documents), k=min(len(documents), max(batch_size * 6, 40)))
        catalog_snippet = build_document_catalog_snippet(candidate_docs, limit=len(candidate_docs))

        user_prompt = f"""
Use the following candidate documents and generate EXACTLY {batch_size} adversarial retrieval queries.

Candidate document catalog (JSON lines):
{catalog_snippet}

Required output shape:
{{
  "queries": [
    {{
      "query": "string",
      "relevant_document_ids": ["doc-0001"],
      "memory_types": ["semantic"],
      "difficulty": "hard"
    }}
  ]
}}

Rules:
- Every query must be hard.
- Use near-miss wording, abstract language, or multi-hop phrasing that could match the wrong documents.
- Each query must have 1 to 3 relevant_document_ids.
- relevant_document_ids MUST come from the provided catalog.
- Include memory_types when they materially constrain retrieval intent.
"""
        raw = client.create_message(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=min(0.8, config.temperature + 0.1),
            max_tokens=config.max_tokens,
            progress_label=f"[adversarial-augment] batch {attempts}/{max_attempts}",
        )

        try:
            payload = extract_json_payload(raw)
            raw_queries = ensure_json_object_with_list(payload, "queries")
        except Exception:
            continue

        for raw_query in raw_queries:
            sanitized = sanitize_recall_query(
                raw_query,
                valid_doc_ids=valid_doc_ids,
                default_difficulty="hard",
            )
            if not sanitized:
                continue
            normalized = sanitized["query"].lower()
            if normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            generated.append(sanitized)
            if len(generated) >= target:
                break

    if len(generated) < target:
        raise RuntimeError(
            f"Could not generate enough adversarial queries. Expected {target}, got {len(generated)}."
        )

    merged = list(existing_queries)
    for index, query in enumerate(generated, start=1):
        row = {
            "id": f"adv-q-{index:04d}",
            "query": query["query"],
            "relevant_document_ids": query["relevant_document_ids"],
            "difficulty": "hard",
        }
        if "memory_types" in query:
            row["memory_types"] = query["memory_types"]
        merged.append(row)
    return merged


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Memory.swift eval data via MiniMax Anthropic-compatible API.")
    parser.add_argument("--dataset-root", default="Evals", help="Dataset root folder (default: Evals).")
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env).")
    parser.add_argument("--base-url", default=None, help="Anthropic-compatible base URL.")
    parser.add_argument("--model", default=None, help="Model name (default: MiniMax-M2.5 or MINIMAX_MODEL env).")
    parser.add_argument(
        "--dataset-mode",
        choices=DATASET_MODES,
        default="general",
        help=(
            "Generation mode: 'general', 'tech', 'longmemeval-typed-queries', "
            "or 'adversarial-augment'. Default: general."
        ),
    )
    parser.add_argument(
        "--domain-profile",
        choices=list(DOMAIN_PROFILES.keys()),
        default=None,
        help="Deprecated compatibility flag. Prefer --dataset-mode.",
    )

    parser.add_argument("--storage-cases", type=int, default=240, help="Total storage cases (default: 240).")
    parser.add_argument("--recall-documents", type=int, default=400, help="Total recall documents (default: 400).")
    parser.add_argument("--recall-queries", type=int, default=240, help="Total recall queries (default: 240).")

    parser.add_argument("--storage-batch-size", type=int, default=12, help="Storage generation batch size.")
    parser.add_argument("--documents-batch-size", type=int, default=16, help="Recall document generation batch size.")
    parser.add_argument("--queries-batch-size", type=int, default=20, help="Recall query generation batch size.")

    parser.add_argument("--temperature", type=float, default=0.55, help="Sampling temperature (0.0, 1.0].")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per generation call.")
    parser.add_argument("--max-attempts-per-bucket", type=int, default=10, help="Generation retries per type bucket.")
    parser.add_argument("--max-retries-per-request", type=int, default=4, help="HTTP retries per API request.")
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=240,
        help="Per-request HTTP timeout in seconds (default: 240).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL files.")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.storage_cases <= 0 or args.recall_documents <= 0 or args.recall_queries <= 0:
        raise ValueError("Counts must all be > 0.")
    if not (0.0 < args.temperature <= 1.0):
        raise ValueError("Temperature must be in (0.0, 1.0].")
    if args.storage_batch_size <= 0 or args.documents_batch_size <= 0 or args.queries_batch_size <= 0:
        raise ValueError("Batch sizes must be > 0.")
    if args.max_tokens < 512:
        raise ValueError("--max-tokens should be at least 512.")
    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be > 0.")


def resolve_api_key() -> str:
    key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("MINIMAX_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    if key:
        return key
    raise RuntimeError(
        "Missing API key. Set ANTHROPIC_API_KEY, MINIMAX_API_KEY, or ANTHROPIC_AUTH_TOKEN (can be in .env)."
    )


def maybe_fail_if_exists(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(existing)
        raise RuntimeError(f"Refusing to overwrite existing files: {joined}. Use --overwrite.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    env_path = Path(args.env_file)
    load_env_file(env_path)

    api_key = resolve_api_key()
    base_url = args.base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL
    model = args.model or os.environ.get("MINIMAX_MODEL") or DEFAULT_MODEL

    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    storage_path = dataset_root / "storage_cases.jsonl"
    documents_path = dataset_root / "recall_documents.jsonl"
    queries_path = dataset_root / "recall_queries.jsonl"
    dataset_mode = args.dataset_mode
    if args.domain_profile == "technical" and dataset_mode == "general":
        dataset_mode = "tech"

    domain_profile = DOMAIN_PROFILES["technical" if dataset_mode == "tech" else "general"]
    cfg = GenerationConfig(
        dataset_mode=dataset_mode,
        storage_cases=args.storage_cases,
        recall_documents=args.recall_documents,
        recall_queries=args.recall_queries,
        storage_batch_size=args.storage_batch_size,
        documents_batch_size=args.documents_batch_size,
        queries_batch_size=args.queries_batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_attempts_per_bucket=args.max_attempts_per_bucket,
        max_retries_per_request=args.max_retries_per_request,
        request_timeout_seconds=args.request_timeout_seconds,
        seed=args.seed,
        domain_profile=domain_profile,
    )

    rng = random.Random(cfg.seed)
    client = MiniMaxAnthropicClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_retries_per_request=cfg.max_retries_per_request,
        timeout_seconds=cfg.request_timeout_seconds,
    )

    log(f"Using base URL: {base_url}")
    log(f"Using model: {model}")
    log(f"Dataset mode: {dataset_mode}")
    log(f"Domain profile: {domain_profile.name}")
    log(f"Dataset root: {dataset_root}")
    log(f"Request timeout (seconds): {cfg.request_timeout_seconds}")

    if dataset_mode in {"general", "tech"}:
        maybe_fail_if_exists(
            [storage_path, documents_path, queries_path, dataset_root / "manifest.json"],
            overwrite=args.overwrite,
        )

        log("Generating storage cases...")
        storage_cases = generate_storage_cases(client, cfg, rng)
        log(f"Generated {len(storage_cases)} storage cases.")

        log("Generating recall documents...")
        recall_documents = generate_recall_documents(client, cfg, rng)
        log(f"Generated {len(recall_documents)} recall documents.")

        log("Generating recall queries...")
        recall_queries = generate_recall_queries(client, cfg, rng, recall_documents)
        log(f"Generated {len(recall_queries)} recall queries.")

        log("Validating query memory_types consistency...")
        recall_queries = validate_query_memory_type_consistency(recall_queries, recall_documents)
        log_difficulty_distribution(recall_queries)

        storage_path.write_text(render_jsonl(storage_cases), encoding="utf-8")
        documents_path.write_text(render_jsonl(recall_documents), encoding="utf-8")
        queries_path.write_text(render_jsonl(recall_queries), encoding="utf-8")
        write_dataset_manifest(
            dataset_root,
            dataset_mode=dataset_mode,
            model=model,
            config=cfg,
            notes=[
                "Synthetic dataset generated through MiniMax Anthropic-compatible API.",
                "general is mixed-domain and is intended as the primary doc-style benchmark." if dataset_mode == "general"
                else "tech is synthetic and enforces software-engineering topical validation.",
            ],
        )

        log("Done.")
        log(f"Wrote {storage_path}")
        log(f"Wrote {documents_path}")
        log(f"Wrote {queries_path}")
    elif dataset_mode == "longmemeval-typed-queries":
        if not queries_path.exists() or not documents_path.exists():
            raise RuntimeError("longmemeval-typed-queries mode requires existing recall_documents.jsonl and recall_queries.jsonl.")
        if not args.overwrite:
            raise RuntimeError("longmemeval-typed-queries mode updates recall_queries.jsonl in place. Re-run with --overwrite.")

        log("Augmenting longmemeval queries with memory_types and difficulty...")
        augmented_queries = augment_longmemeval_typed_queries(client, cfg, dataset_root)
        log_difficulty_distribution(augmented_queries)
        queries_path.write_text(render_jsonl(augmented_queries), encoding="utf-8")
        write_dataset_manifest(
            dataset_root,
            dataset_mode=dataset_mode,
            model=model,
            config=cfg,
            notes=[
                "Source documents remain from converted LongMemEval.",
                "memory_types and difficulty fields were added through MiniMax query labeling.",
            ],
        )
        log(f"Wrote {queries_path}")
    else:
        if not queries_path.exists() or not documents_path.exists():
            raise RuntimeError("adversarial-augment mode requires existing recall_documents.jsonl and recall_queries.jsonl.")
        if not args.overwrite:
            raise RuntimeError("adversarial-augment mode updates recall_queries.jsonl in place. Re-run with --overwrite.")

        log("Generating adversarial query augmentation...")
        augmented_queries = augment_adversarial_queries(client, cfg, rng, dataset_root)
        log_difficulty_distribution(augmented_queries)
        queries_path.write_text(render_jsonl(augmented_queries), encoding="utf-8")
        write_dataset_manifest(
            dataset_root,
            dataset_mode=dataset_mode,
            model=model,
            config=cfg,
            notes=[
                "Adds hard retrieval queries against the existing document corpus.",
                "This mode augments recall_queries.jsonl only; it does not regenerate source documents.",
            ],
        )
        log(f"Wrote {queries_path}")

    log("Next:")
    log(f"  swift run memory_eval run --profile baseline --dataset-root {dataset_root}")
    log(f"  swift run memory_eval run --profile coreml_rerank --dataset-root {dataset_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
