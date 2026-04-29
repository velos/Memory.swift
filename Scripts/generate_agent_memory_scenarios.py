#!/usr/bin/env python3
"""Generate agent-memory eval scenarios.

The default path is deterministic and template-based so gate labels stay
reviewable. An optional MiniMax/Anthropic-compatible backend can draft extra
candidate scenarios through the same client used by the other eval generators.

Outputs:
  - scenarios.seed.jsonl
  - scenarios.generated.jsonl
  - scenarios.pressure.jsonl
  - scenarios.model_drafts.jsonl, when --backend minimax is used
  - scenarios.jsonl, unless --no-write-combined is set
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from eval_data_codex_support import extract_json_payload, load_jsonl, log, write_jsonl_atomic
from generate_eval_data_minimax import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MiniMaxAnthropicClient,
    load_env_file,
    resolve_api_key,
)


Scenario = Dict[str, Any]

ALLOWED_ROLES = {"system", "user", "assistant"}
ALLOWED_KINDS = {"fact", "profile", "decision", "commitment", "procedure", "handoff", "episode"}
ALLOWED_STATUSES = {"active", "superseded", "resolved", "archived"}
ALLOWED_FACETS = {
    "preference",
    "person",
    "relationship",
    "project",
    "goal",
    "task",
    "decision_topic",
    "tool",
    "location",
    "time_sensitive",
    "constraint",
    "habit",
    "fact_about_user",
    "fact_about_world",
    "lesson",
    "emotion",
    "identity_signal",
}
ALLOWED_UPDATE_BEHAVIORS = {"append", "dedupe", "replace_active", "merge_status", "supersede", "none"}


SEED_SCENARIOS: List[Scenario] = [
    {
        "id": "negative-question-no-write",
        "source_family": "no_write",
        "difficulty": "easy",
        "generation_method": "seed",
        "messages": [
            {
                "role": "user",
                "content": "Thanks, can you explain how vector indexes work before we decide anything?",
            }
        ],
        "expected_write_count": 0,
        "expected_memories": [],
        "recall_queries": [],
    },
    {
        "id": "profile-editor-replaces-active",
        "source_family": "profile_update",
        "difficulty": "medium",
        "generation_method": "seed",
        "setup_memories": [
            {
                "text": "Preferred editor is Vim.",
                "kind": "profile",
                "status": "active",
                "canonical_key": "profile:editor",
                "facet_tags": ["preference"],
                "entity_values": ["Vim"],
                "topics": ["preferred editor"],
            }
        ],
        "messages": [{"role": "user", "content": "Preferred editor is Zed."}],
        "expected_write_count": 1,
        "expected_update_behavior": "replace_active",
        "expected_memories": [
            {
                "kind": "profile",
                "status": "active",
                "canonical_key": "profile:editor",
                "text_contains": ["Zed"],
                "facets": ["preference"],
                "entities": ["zed"],
            }
        ],
        "recall_queries": [
            {
                "query": "What editor is preferred?",
                "expected_text_contains": ["Zed"],
                "expected_kinds": ["profile"],
                "expected_statuses": ["active"],
            }
        ],
    },
    {
        "id": "commitment-resolution-merges-status",
        "source_family": "commitment_resolution",
        "difficulty": "medium",
        "generation_method": "seed",
        "setup_memories": [
            {
                "text": "TODO: add migration tests.",
                "kind": "commitment",
                "status": "active",
                "canonical_key": "commitment:migration-tests",
                "facet_tags": ["task"],
                "topics": ["migration tests"],
            }
        ],
        "messages": [{"role": "user", "content": "TODO: add migration tests is done."}],
        "expected_write_count": 1,
        "expected_update_behavior": "merge_status",
        "expected_memories": [
            {
                "kind": "commitment",
                "status": "resolved",
                "canonical_key": "commitment:migration-tests",
                "text_contains": ["migration tests"],
            }
        ],
        "recall_queries": [
            {
                "query": "What happened to migration tests?",
                "expected_text_contains": ["migration tests"],
                "expected_kinds": ["commitment"],
                "expected_statuses": ["resolved"],
            }
        ],
    },
    {
        "id": "decision-and-recall",
        "source_family": "decision_write",
        "difficulty": "easy",
        "generation_method": "seed",
        "messages": [
            {"role": "user", "content": "We switched the prototype to SQLite for the storage layer."},
            {"role": "assistant", "content": "I will remember that storage decision."},
        ],
        "expected_write_count": 1,
        "expected_memories": [
            {
                "kind": "decision",
                "status": "active",
                "text_contains": ["SQLite", "storage layer"],
                "facets": ["decision_topic"],
            }
        ],
        "recall_queries": [
            {
                "query": "What storage layer did we choose for the prototype?",
                "expected_text_contains": ["SQLite"],
                "expected_kinds": ["decision"],
                "expected_statuses": ["active"],
            }
        ],
    },
    {
        "id": "episode-recall",
        "source_family": "episode_append",
        "difficulty": "easy",
        "generation_method": "seed",
        "messages": [{"role": "user", "content": "Yesterday we shipped the migration scaffolding after the review."}],
        "expected_write_count": 1,
        "expected_memories": [
            {"kind": "episode", "status": "active", "text_contains": ["migration scaffolding"]}
        ],
        "recall_queries": [
            {
                "query": "What did we ship after the review?",
                "expected_text_contains": ["migration scaffolding"],
                "expected_kinds": ["episode"],
                "expected_statuses": ["active"],
            }
        ],
    },
]


NO_WRITE_TEXTS = [
    "Thanks, that makes sense for now.",
    "Can you compare SQLite and Postgres before we decide?",
    "Could you explain how hybrid retrieval works?",
    "Okay, let's pause on memory changes for the moment.",
    "Please summarize the options without saving anything yet.",
    "Would you walk me through why reranking matters?",
]

PROFILE_NEW = [
    ("profile-timezone-pacific", "My timezone is Pacific time.", "profile:timezone", ["Pacific time"], "What timezone should you remember?", ["Pacific"]),
    ("profile-role-maintainer", "My role is API maintainer for Memory.swift.", "profile:role", ["API maintainer"], "What is my role?", ["API maintainer"]),
    ("profile-editor-cursor", "Preferred editor is Cursor.", "profile:editor", ["Cursor"], "What editor is preferred?", ["Cursor"]),
    ("profile-name-casey", "My name is Casey Morgan.", "profile:name", ["Casey Morgan"], "What name should you remember?", ["Casey"]),
]

PROFILE_UPDATES = [
    (
        "profile-update-editor-cursor",
        {"text": "Preferred editor is Vim.", "kind": "profile", "status": "active", "canonical_key": "profile:editor"},
        "Preferred editor is Cursor.",
        "profile:editor",
        ["Cursor"],
        "Which editor is current?",
        ["Cursor"],
    ),
    (
        "profile-update-timezone-eastern",
        {"text": "My timezone is Pacific time.", "kind": "profile", "status": "active", "canonical_key": "profile:timezone"},
        "My timezone is Eastern time.",
        "profile:timezone",
        ["Eastern"],
        "What timezone is current?",
        ["Eastern"],
    ),
    (
        "profile-update-role-backend",
        {"text": "My role is iOS maintainer.", "kind": "profile", "status": "active", "canonical_key": "profile:role"},
        "My role is backend maintainer.",
        "profile:role",
        ["backend maintainer"],
        "What role is current?",
        ["backend maintainer"],
    ),
]

COMMITMENT_NEW = [
    ("commitment-export-tests", "Action item: add export tests.", ["export tests"], "What action item is open?", ["export tests"]),
    ("commitment-release-notes", "Next step: write release notes for the SDK.", ["release notes"], "What is the next step?", ["release notes"]),
    ("commitment-audit-recall", "We should audit recall misses tomorrow.", ["audit recall misses"], "What should we audit?", ["recall misses"]),
    ("commitment-verify-alerts", "TODO: verify production alerts.", ["verify production alerts"], "What TODO is open?", ["alerts"]),
]

COMMITMENT_RESOLUTIONS = [
    (
        "commitment-resolve-verify-alerts",
        {"text": "Action item: verify alerts.", "kind": "commitment", "status": "active", "canonical_key": "commitment:verify-alerts"},
        "Done: verify alerts.",
        "commitment:verify-alerts",
        ["verify alerts"],
        "What happened to alert verification?",
        ["alerts"],
    ),
    (
        "commitment-resolve-migration-tests",
        {"text": "TODO: add migration tests.", "kind": "commitment", "status": "active", "canonical_key": "commitment:migration-tests"},
        "Done: add migration tests.",
        "commitment:migration-tests",
        ["migration tests"],
        "What happened to migration tests?",
        ["migration tests"],
    ),
]

DECISION_NEW = [
    ("decision-sqlite-storage", "We switched the prototype to SQLite for the storage layer.", ["SQLite"], "What storage layer did we choose?", ["SQLite"]),
    ("decision-hybrid-recall", "We agreed to use hybrid retrieval for agent recall.", ["hybrid retrieval"], "What recall strategy did we agree on?", ["hybrid retrieval"]),
    ("decision-coreml-default", "Decision: use CoreML embeddings for default search.", ["CoreML"], "What embedding path is default?", ["CoreML"]),
]

EPISODES = [
    ("episode-reranker-latency", "Yesterday we debugged reranker latency after the demo.", ["reranker latency"], "What did we debug yesterday?", ["reranker latency"]),
    ("episode-profile-patch", "Today we shipped the profile canonicalization patch.", ["profile canonicalization"], "What did we ship today?", ["profile canonicalization"]),
    ("episode-release-planning", "Last week we met Priya about release planning.", ["release planning"], "Who did we meet about release planning?", ["Priya"]),
]

HANDOFFS = [
    (
        "handoff-update-query-expansion",
        {"text": "Current status: schema migration is in progress.", "kind": "handoff", "status": "active", "canonical_key": "handoff:primary"},
        "Current status: query expansion audit is in progress.",
        ["query expansion audit"],
        "What is the current status?",
        ["query expansion audit"],
    )
]


def scenario(
    *,
    scenario_id: str,
    family: str,
    difficulty: str,
    messages: List[Dict[str, str]],
    expected_write_count: int,
    expected_memories: Optional[List[Dict[str, Any]]] = None,
    recall_queries: Optional[List[Dict[str, Any]]] = None,
    setup_memories: Optional[List[Dict[str, Any]]] = None,
    expected_update_behavior: Optional[str] = None,
    generation_method: str = "template",
) -> Scenario:
    result: Scenario = {
        "id": scenario_id,
        "source_family": family,
        "difficulty": difficulty,
        "generation_method": generation_method,
        "messages": messages,
        "expected_write_count": expected_write_count,
        "expected_memories": expected_memories or [],
        "recall_queries": recall_queries or [],
    }
    if setup_memories:
        result["setup_memories"] = setup_memories
    if expected_update_behavior:
        result["expected_update_behavior"] = expected_update_behavior
    return result


PRESSURE_SCENARIOS: List[Scenario] = [
    scenario(
        scenario_id="pressure-no-write-hypothetical-identity-phrase",
        family="no_write",
        difficulty="hard",
        generation_method="pressure_template",
        messages=[{"role": "user", "content": "I am asking a hypothetical question about vector indexes."}],
        expected_write_count=0,
    ),
    scenario(
        scenario_id="pressure-no-write-advice-with-commitment-cues",
        family="no_write",
        difficulty="hard",
        generation_method="pressure_template",
        messages=[{"role": "user", "content": "What should I do next if the migration fails?"}],
        expected_write_count=0,
    ),
    scenario(
        scenario_id="pressure-decision-storage-supersedes-topic",
        family="decision_supersede",
        difficulty="hard",
        generation_method="pressure_template",
        setup_memories=[
            {
                "text": "Decision: storage engine is SQLite.",
                "kind": "decision",
                "status": "active",
                "canonical_key": "decision:storage-engine",
            }
        ],
        messages=[{"role": "user", "content": "Decision: storage engine is Postgres."}],
        expected_write_count=1,
        expected_update_behavior="supersede",
        expected_memories=[
            {
                "kind": "decision",
                "status": "active",
                "canonical_key": "decision:storage-engine",
                "text_contains": ["Postgres"],
            }
        ],
        recall_queries=[
            {
                "query": "What storage engine is current?",
                "expected_text_contains": ["Postgres"],
                "expected_kinds": ["decision"],
                "expected_statuses": ["active"],
            }
        ],
    ),
    scenario(
        scenario_id="pressure-commitment-resolution-paraphrase",
        family="commitment_resolution",
        difficulty="hard",
        generation_method="pressure_template",
        setup_memories=[
            {
                "text": "Action item: update API docs.",
                "kind": "commitment",
                "status": "active",
                "canonical_key": "commitment:update-api-docs",
            }
        ],
        messages=[{"role": "user", "content": "The API documentation update is finished."}],
        expected_write_count=1,
        expected_update_behavior="merge_status",
        expected_memories=[
            {
                "kind": "commitment",
                "status": "resolved",
                "canonical_key": "commitment:update-api-docs",
                "text_contains": ["API docs"],
            }
        ],
        recall_queries=[
            {
                "query": "What happened to the API docs task?",
                "expected_text_contains": ["API docs"],
                "expected_kinds": ["commitment"],
                "expected_statuses": ["resolved"],
            }
        ],
    ),
    scenario(
        scenario_id="pressure-commitment-resolution-caching",
        family="commitment_resolution",
        difficulty="medium",
        generation_method="pressure_from_model_draft",
        setup_memories=[
            {
                "text": "TODO: implement caching layer before launch.",
                "kind": "commitment",
                "status": "active",
                "canonical_key": "commitment:caching-layer",
                "facet_tags": ["task"],
                "topics": ["caching layer", "launch"],
            }
        ],
        messages=[{"role": "user", "content": "Done: implement caching layer before launch."}],
        expected_write_count=1,
        expected_update_behavior="merge_status",
        expected_memories=[
            {
                "kind": "commitment",
                "status": "resolved",
                "canonical_key": "commitment:caching-layer",
                "text_contains": ["caching layer", "launch"],
                "facets": ["task"],
            }
        ],
        recall_queries=[
            {
                "query": "What happened to the caching layer task?",
                "expected_text_contains": ["caching layer"],
                "expected_kinds": ["commitment"],
                "expected_statuses": ["resolved"],
            }
        ],
    ),
    scenario(
        scenario_id="pressure-recall-only-storage-decision",
        family="recall_only",
        difficulty="easy",
        generation_method="pressure_from_model_draft",
        setup_memories=[
            {
                "text": "Decision: use CoreML embeddings for default search.",
                "kind": "decision",
                "status": "active",
                "canonical_key": "decision:coreml-default-search",
                "facet_tags": ["decision_topic", "tool"],
                "entity_values": ["CoreML"],
                "topics": ["default search", "embeddings"],
            }
        ],
        messages=[{"role": "user", "content": "What embedding path did we choose for default search?"}],
        expected_write_count=0,
        recall_queries=[
            {
                "query": "What embedding path did we choose for default search?",
                "expected_text_contains": ["CoreML"],
                "expected_kinds": ["decision"],
                "expected_statuses": ["active"],
            }
        ],
    ),
    scenario(
        scenario_id="pressure-hypothetical-profile-no-write",
        family="no_write",
        difficulty="hard",
        generation_method="pressure_from_model_draft",
        messages=[{"role": "user", "content": "If I were the release owner, what responsibilities would I usually have?"}],
        expected_write_count=0,
    ),
]


def slug(value: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return compact[:80] or "scenario"


def build_template_scenarios(seed: int, max_count: Optional[int]) -> List[Scenario]:
    rng = random.Random(seed)
    scenarios: List[Scenario] = []

    for text in NO_WRITE_TEXTS:
        scenarios.append(
            scenario(
                scenario_id=f"generated-no-write-{slug(text)}",
                family="no_write",
                difficulty="easy",
                messages=[{"role": "user", "content": text}],
                expected_write_count=0,
            )
        )

    scenarios.append(
        scenario(
            scenario_id="generated-assistant-ack-no-write",
            family="no_write",
            difficulty="medium",
            messages=[{"role": "assistant", "content": "I will remember that preference for next time."}],
            expected_write_count=0,
        )
    )

    for scenario_id, text, canonical_key, contains, query, recall_contains in PROFILE_NEW:
        scenarios.append(
            scenario(
                scenario_id=f"generated-{scenario_id}",
                family="profile_write",
                difficulty="easy",
                messages=[{"role": "user", "content": text}],
                expected_write_count=1,
                expected_memories=[
                    {
                        "kind": "profile",
                        "status": "active",
                        "canonical_key": canonical_key,
                        "text_contains": contains,
                    }
                ],
                recall_queries=[
                    {
                        "query": query,
                        "expected_text_contains": recall_contains,
                        "expected_kinds": ["profile"],
                        "expected_statuses": ["active"],
                    }
                ],
            )
        )

    for scenario_id, setup, text, canonical_key, contains, query, recall_contains in PROFILE_UPDATES:
        scenarios.append(
            scenario(
                scenario_id=f"generated-{scenario_id}",
                family="profile_update",
                difficulty="medium",
                setup_memories=[setup],
                messages=[{"role": "user", "content": text}],
                expected_write_count=1,
                expected_update_behavior="replace_active",
                expected_memories=[
                    {
                        "kind": "profile",
                        "status": "active",
                        "canonical_key": canonical_key,
                        "text_contains": contains,
                    }
                ],
                recall_queries=[
                    {
                        "query": query,
                        "expected_text_contains": recall_contains,
                        "expected_kinds": ["profile"],
                        "expected_statuses": ["active"],
                    }
                ],
            )
        )

    for scenario_id, text, contains, query, recall_contains in COMMITMENT_NEW:
        scenarios.append(
            scenario(
                scenario_id=f"generated-{scenario_id}",
                family="commitment_write",
                difficulty="easy",
                messages=[{"role": "user", "content": text}],
                expected_write_count=1,
                expected_memories=[
                    {"kind": "commitment", "status": "active", "text_contains": contains}
                ],
                recall_queries=[
                    {
                        "query": query,
                        "expected_text_contains": recall_contains,
                        "expected_kinds": ["commitment"],
                        "expected_statuses": ["active"],
                    }
                ],
            )
        )

    for scenario_id, setup, text, canonical_key, contains, query, recall_contains in COMMITMENT_RESOLUTIONS:
        scenarios.append(
            scenario(
                scenario_id=f"generated-{scenario_id}",
                family="commitment_resolution",
                difficulty="medium",
                setup_memories=[setup],
                messages=[{"role": "user", "content": text}],
                expected_write_count=1,
                expected_update_behavior="merge_status",
                expected_memories=[
                    {
                        "kind": "commitment",
                        "status": "resolved",
                        "canonical_key": canonical_key,
                        "text_contains": contains,
                    }
                ],
                recall_queries=[
                    {
                        "query": query,
                        "expected_text_contains": recall_contains,
                        "expected_kinds": ["commitment"],
                        "expected_statuses": ["resolved"],
                    }
                ],
            )
        )

    for scenario_id, text, contains, query, recall_contains in DECISION_NEW:
        scenarios.append(
            scenario(
                scenario_id=f"generated-{scenario_id}",
                family="decision_write",
                difficulty="easy",
                messages=[{"role": "user", "content": text}],
                expected_write_count=1,
                expected_memories=[
                    {"kind": "decision", "status": "active", "text_contains": contains}
                ],
                recall_queries=[
                    {
                        "query": query,
                        "expected_text_contains": recall_contains,
                        "expected_kinds": ["decision"],
                        "expected_statuses": ["active"],
                    }
                ],
            )
        )

    for scenario_id, text, contains, query, recall_contains in EPISODES:
        scenarios.append(
            scenario(
                scenario_id=f"generated-{scenario_id}",
                family="episode_append",
                difficulty="easy",
                messages=[{"role": "user", "content": text}],
                expected_write_count=1,
                expected_memories=[
                    {"kind": "episode", "status": "active", "text_contains": contains}
                ],
                recall_queries=[
                    {
                        "query": query,
                        "expected_text_contains": recall_contains,
                        "expected_kinds": ["episode"],
                        "expected_statuses": ["active"],
                    }
                ],
            )
        )

    for scenario_id, setup, text, contains, query, recall_contains in HANDOFFS:
        scenarios.append(
            scenario(
                scenario_id=f"generated-{scenario_id}",
                family="handoff_update",
                difficulty="medium",
                setup_memories=[setup],
                messages=[{"role": "user", "content": text}],
                expected_write_count=1,
                expected_update_behavior="replace_active",
                expected_memories=[
                    {
                        "kind": "handoff",
                        "status": "active",
                        "canonical_key": "handoff:primary",
                        "text_contains": contains,
                    }
                ],
                recall_queries=[
                    {
                        "query": query,
                        "expected_text_contains": recall_contains,
                        "expected_kinds": ["handoff"],
                        "expected_statuses": ["active"],
                    }
                ],
            )
        )

    rng.shuffle(scenarios)
    if max_count is not None:
        scenarios = scenarios[:max_count]
    return sorted(scenarios, key=lambda item: item["id"])


def build_minimax_prompt(count: int, seed_records: Sequence[Scenario]) -> str:
    examples = json.dumps(seed_records[:3], indent=2)
    return f"""
Generate {count} additional JSON objects for a Swift agent-memory eval suite.

Each object must match this schema:
- id: stable snake/kebab string
- source_family: one of no_write, profile_write, profile_update, commitment_write,
  commitment_resolution, decision_write, decision_supersede, episode_append,
  handoff_update, temporal_recall
- difficulty: easy, medium, hard
- generation_method: "minimax"
- setup_memories: optional list of memories with text, kind, status, canonical_key,
  facet_tags, entity_values, topics
- messages: list of role/content messages using role user or assistant
- expected_write_count: integer
- expected_update_behavior: optional append, dedupe, replace_active, merge_status,
  supersede, or none
- expected_memories: list of expected memories with kind, status, canonical_key
  where known, text_contains, facets, entities, topics
- recall_queries: list with query, expected_text_contains, expected_kinds,
  expected_statuses

Prefer compact conversations that test one behavior each. Include no-write cases,
profile replacements, commitment resolutions, temporal recall, and distractors.
Use only valid memory kinds: fact, profile, decision, commitment, procedure,
handoff, episode. Use only valid statuses: active, superseded, resolved, archived.

Return ONLY JSON in this shape:
{{"scenarios":[...]}}

Examples:
{examples}
""".strip()


def generate_minimax_scenarios(args: argparse.Namespace, seed_records: Sequence[Scenario]) -> List[Scenario]:
    load_env_file(Path(args.env_file))
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
    log(f"Requesting {args.model_count} model-drafted scenarios from configured Anthropic-compatible endpoint.")
    raw = client.create_message(
        system_prompt="You generate precise JSONL-style eval data. Return valid JSON only.",
        user_prompt=build_minimax_prompt(args.model_count, seed_records),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        progress_label="agent-memory-scenarios",
    )
    parsed = extract_json_payload(raw)
    if isinstance(parsed, list):
        scenario_items = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("scenarios"), list):
        scenario_items = parsed["scenarios"]
    else:
        raise ValueError("Model response did not contain a scenarios list.")
    records = []
    for index, item in enumerate(scenario_items, start=1):
        if not isinstance(item, dict):
            continue
        item = dict(item)
        item["generation_method"] = "minimax"
        item.setdefault("source_family", "model_draft")
        item.setdefault("difficulty", "hard")
        item.setdefault("id", f"minimax-{index:03d}-{slug(item.get('source_family', 'scenario'))}")
        records.append(item)
    return records


def normalize_scenario(record: Scenario) -> Scenario:
    result = dict(record)
    result["id"] = slug(str(result["id"]))
    result.setdefault("source_family", "unspecified")
    result.setdefault("difficulty", "medium")
    result.setdefault("generation_method", "unknown")
    result.setdefault("messages", [])
    result.setdefault("expected_write_count", len(result.get("expected_memories", [])))
    result.setdefault("expected_memories", [])
    result.setdefault("recall_queries", [])
    if result.get("expected_update_behavior") == "none":
        result.pop("expected_update_behavior", None)
    if result.get("expected_write_count") == 0 and not result.get("setup_memories"):
        result["recall_queries"] = []
    for memory in result.get("setup_memories", []) or []:
        memory["facet_tags"] = filter_allowed_facets(coerce_string_list(memory.get("facet_tags")))
        memory["entity_values"] = coerce_string_list(memory.get("entity_values"))
        memory["topics"] = coerce_string_list(memory.get("topics"))
    for memory in result.get("expected_memories", []) or []:
        memory["text_contains"] = coerce_string_list(memory.get("text_contains"))
        memory["facets"] = filter_allowed_facets(coerce_string_list(memory.get("facets")))
        memory["entities"] = coerce_string_list(memory.get("entities"))
        memory["topics"] = coerce_string_list(memory.get("topics"))
    for query in result.get("recall_queries", []) or []:
        query["expected_text_contains"] = coerce_string_list(query.get("expected_text_contains"))
        query["expected_kinds"] = [value for value in coerce_string_list(query.get("expected_kinds")) if value in ALLOWED_KINDS]
        query["expected_statuses"] = [
            value for value in coerce_string_list(query.get("expected_statuses")) if value in ALLOWED_STATUSES
        ]
    return result


def coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(item) for item in value.values() if str(item).strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def filter_allowed_facets(values: Sequence[str]) -> List[str]:
    return [value for value in values if value in ALLOWED_FACETS]


def validate_scenario(record: Scenario) -> None:
    scenario_id = str(record.get("id", "<missing>"))
    if not scenario_id or scenario_id == "<missing>":
        raise ValueError("Scenario missing id.")
    if not isinstance(record.get("messages"), list) or not record["messages"]:
        raise ValueError(f"{scenario_id}: messages must be a non-empty list.")
    for message in record["messages"]:
        if message.get("role") not in ALLOWED_ROLES:
            raise ValueError(f"{scenario_id}: invalid message role {message.get('role')!r}.")
        if not str(message.get("content", "")).strip():
            raise ValueError(f"{scenario_id}: message content must be non-empty.")
    expected_write_count = record.get("expected_write_count")
    if not isinstance(expected_write_count, int) or expected_write_count < 0:
        raise ValueError(f"{scenario_id}: expected_write_count must be a non-negative integer.")
    if record.get("expected_update_behavior") not in (None, *ALLOWED_UPDATE_BEHAVIORS):
        raise ValueError(f"{scenario_id}: invalid expected_update_behavior.")
    for section_name in ("setup_memories", "expected_memories"):
        for memory in record.get(section_name, []) or []:
            kind = memory.get("kind")
            status = memory.get("status")
            if kind is not None and kind not in ALLOWED_KINDS:
                raise ValueError(f"{scenario_id}: invalid {section_name} kind {kind!r}.")
            if status is not None and status not in ALLOWED_STATUSES:
                raise ValueError(f"{scenario_id}: invalid {section_name} status {status!r}.")
            for facet in memory.get("facet_tags", []) or []:
                if facet not in ALLOWED_FACETS:
                    raise ValueError(f"{scenario_id}: invalid setup facet {facet!r}.")
            for facet in memory.get("facets", []) or []:
                if facet not in ALLOWED_FACETS:
                    raise ValueError(f"{scenario_id}: invalid expected facet {facet!r}.")
    for query in record.get("recall_queries", []) or []:
        if not str(query.get("query", "")).strip():
            raise ValueError(f"{scenario_id}: recall query must include query text.")
        for kind in query.get("expected_kinds", []) or []:
            if kind not in ALLOWED_KINDS:
                raise ValueError(f"{scenario_id}: invalid recall expected kind {kind!r}.")
        for status in query.get("expected_statuses", []) or []:
            if status not in ALLOWED_STATUSES:
                raise ValueError(f"{scenario_id}: invalid recall expected status {status!r}.")


def dedupe_scenarios(records: Iterable[Scenario]) -> List[Scenario]:
    by_id: Dict[str, Scenario] = {}
    for raw in records:
        record = normalize_scenario(raw)
        validate_scenario(record)
        base_id = record["id"]
        scenario_id = base_id
        suffix = 2
        while scenario_id in by_id:
            scenario_id = f"{base_id}-{suffix}"
            suffix += 1
        record["id"] = scenario_id
        by_id[scenario_id] = record
    return [by_id[key] for key in sorted(by_id)]


def write_readme(dataset_root: Path) -> None:
    readme = """# agent_memory_gold_v1

Scenario benchmark for the default agent-memory loop:

- extract memories from multi-turn messages
- avoid writes for questions and conversational filler
- consolidate profile, decision, commitment, and handoff updates
- recall useful active or explicitly requested resolved memories

Files:

- `scenarios.seed.jsonl`: hand-curated smoke cases
- `scenarios.generated.jsonl`: deterministic template expansion
- `scenarios.pressure.jsonl`: harder known-failure or near-failure cases, not included in the default gate unless requested
- `scenarios.model_drafts.jsonl`: optional model-drafted candidates for review
- `scenarios.jsonl`: default gate set, normally seed plus deterministic generated scenarios

Generate or refresh scenarios with:

```bash
python3 Scripts/generate_agent_memory_scenarios.py --dataset-root ./Evals/agent_memory_gold_v1 --overwrite
```

Include pressure cases in the gate set with:

```bash
python3 Scripts/generate_agent_memory_scenarios.py --dataset-root ./Evals/agent_memory_gold_v1 --include-pressure --overwrite
```

Optional model-drafted candidates use the Anthropic-compatible endpoint from `.env`:

```bash
python3 Scripts/generate_agent_memory_scenarios.py --dataset-root ./Evals/agent_memory_gold_v1 --backend minimax --model-count 12 --no-write-combined --overwrite
```

Run with:

```bash
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/agent_memory_gold_v1 --no-cache --no-index-cache
```
"""
    (dataset_root / "README.md").write_text(readme, encoding="utf-8")


def write_pressure_readme(pressure_root: Path) -> None:
    readme = """# agent_memory_pressure_v1

Pressure benchmark for agent-memory behavior that should improve over time.

This dataset is generated from `Evals/agent_memory_gold_v1/scenarios.pressure.jsonl`.
It is intentionally separate from the stable release gate so hard cases can be
tracked before they are promoted into `agent_memory_gold_v1/scenarios.jsonl`.

Run with:

```bash
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/agent_memory_pressure_v1 --no-cache --no-index-cache
```

Gate with:

```bash
swift run memory_eval gate --baseline ./Evals/baselines/pressure.json <pressure-run-json>
```
"""
    pressure_root.mkdir(parents=True, exist_ok=True)
    (pressure_root / "README.md").write_text(readme, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="./Evals/agent_memory_gold_v1")
    parser.add_argument("--pressure-root", default="./Evals/agent_memory_pressure_v1")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--backend", choices=["templates", "minimax"], default="templates")
    parser.add_argument("--generated-count", type=int, default=None, help="Limit deterministic generated scenarios.")
    parser.add_argument("--model-count", type=int, default=12)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--max-retries-per-request", type=int, default=3)
    parser.add_argument("--request-timeout-seconds", type=int, default=240)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-write-combined", action="store_true")
    parser.add_argument("--include-pressure", action="store_true", help="Include pressure scenarios in scenarios.jsonl.")
    parser.add_argument("--no-write-pressure-root", action="store_true")
    parser.add_argument("--preserve-seed-file", action="store_true")
    return parser.parse_args(argv)


def fail_if_exists(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise RuntimeError(f"Refusing to overwrite existing files: {', '.join(existing)}. Use --overwrite.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    seed_path = dataset_root / "scenarios.seed.jsonl"
    generated_path = dataset_root / "scenarios.generated.jsonl"
    pressure_path = dataset_root / "scenarios.pressure.jsonl"
    model_path = dataset_root / "scenarios.model_drafts.jsonl"
    combined_path = dataset_root / "scenarios.jsonl"

    output_paths = [generated_path, pressure_path]
    if not args.preserve_seed_file:
        output_paths.append(seed_path)
    if args.backend == "minimax":
        output_paths.append(model_path)
    if not args.no_write_combined:
        output_paths.append(combined_path)
    fail_if_exists(output_paths, args.overwrite)

    seed_records = load_jsonl(seed_path) if args.preserve_seed_file and seed_path.exists() else SEED_SCENARIOS
    seed_records = dedupe_scenarios(seed_records)
    generated_records = dedupe_scenarios(build_template_scenarios(args.seed, args.generated_count))
    pressure_records = dedupe_scenarios(PRESSURE_SCENARIOS)
    model_records: List[Scenario] = []
    if args.backend == "minimax":
        model_records = dedupe_scenarios(generate_minimax_scenarios(args, seed_records))

    if not args.preserve_seed_file:
        write_jsonl_atomic(seed_path, seed_records)
    write_jsonl_atomic(generated_path, generated_records)
    write_jsonl_atomic(pressure_path, pressure_records)
    if not args.no_write_pressure_root:
        pressure_root = Path(args.pressure_root)
        write_jsonl_atomic(pressure_root / "scenarios.jsonl", pressure_records)
        write_pressure_readme(pressure_root)
    if args.backend == "minimax":
        write_jsonl_atomic(model_path, model_records)

    if not args.no_write_combined:
        combined_inputs = [*seed_records, *generated_records, *model_records]
        if args.include_pressure:
            combined_inputs.extend(pressure_records)
        combined = dedupe_scenarios(combined_inputs)
        write_jsonl_atomic(combined_path, combined)
        log(f"Wrote {len(combined)} combined scenarios to {combined_path}.")
    log(f"Wrote {len(seed_records)} seed scenarios to {seed_path}.")
    log(f"Wrote {len(generated_records)} deterministic generated scenarios to {generated_path}.")
    log(f"Wrote {len(pressure_records)} pressure scenarios to {pressure_path}.")
    if not args.no_write_pressure_root:
        log(f"Wrote {len(pressure_records)} pressure gate scenarios to {Path(args.pressure_root) / 'scenarios.jsonl'}.")
    if model_records:
        log(f"Wrote {len(model_records)} model-drafted scenarios to {model_path}.")
    write_readme(dataset_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
