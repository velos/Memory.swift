#!/usr/bin/env python3
"""Convert LongMemEval-cleaned data into Memory.swift eval format.

Usage examples:
  python3 Scripts/convert_longmemeval_to_eval.py --output-dir ./Evals/longmemeval
  python3 Scripts/convert_longmemeval_to_eval.py --split s_cleaned --output-dir ./Evals/longmemeval_s
  python3 Scripts/convert_longmemeval_to_eval.py --source-file ./longmemeval_oracle.json --output-dir ./Evals/longmemeval
"""

import argparse
import json
import os
import re
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any, Optional


DEFAULT_SOURCE_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
SPLIT_TO_FILENAME = {
    "oracle": "longmemeval_oracle.json",
    "s_cleaned": "longmemeval_s_cleaned.json",
    "m_cleaned": "longmemeval_m_cleaned.json",
}

# LongMemEval question types are mapped into Memory.swift memory taxonomy.
QUESTION_TYPE_TO_MEMORY_TYPE = {
    "temporal-reasoning": "temporal",
    "knowledge-update": "contextual",
    "multi-session": "episodic",
    "single-session-user": "episodic",
    "single-session-assistant": "semantic",
    "single-session-preference": "semantic",
}


def download_source(source_url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    parsed = urllib.parse.urlparse(source_url)
    filename = os.path.basename(parsed.path) or "longmemeval.json"
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        print(f"Using cached source: {cache_path}")
        return cache_path

    print(f"Downloading {source_url} ...")
    urllib.request.urlretrieve(source_url, cache_path)
    return cache_path


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def render_session_markdown(session_id: str, session_date: Optional[str], turns: list[dict[str, Any]]) -> str:
    lines: list[str] = [f"# Session {session_id}"]
    if session_date:
        lines.append("")
        lines.append(f"Date: {session_date}")

    lines.append("")
    for index, turn in enumerate(turns, start=1):
        role = str(turn.get("role", "unknown")).strip() or "unknown"
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"## Turn {index} ({role})")
        lines.append("")
        lines.append(content)
        lines.append("")

    rendered = "\n".join(lines).strip()
    return rendered if rendered else f"# Session {session_id}"


def trim_to_word_boundary(text: str, max_len: int = 140) -> str:
    if len(text) <= max_len:
        return text
    trimmed = text[:max_len]
    split = trimmed.rfind(" ")
    if split >= max_len // 2:
        return trimmed[:split]
    return trimmed


def extract_required_spans(text: str, spans_per_case: int) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(span: str) -> None:
        value = span.strip()
        if not value:
            return
        value = trim_to_word_boundary(value)
        if len(value) < 8:
            return
        if value not in text:
            return
        key = normalize_whitespace(value.lower())
        if key in seen:
            return
        seen.add(key)
        candidates.append(value)

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if lines:
        first = lines[0].lstrip("#").strip()
        add(first)

    sentence_pool: list[str] = []
    for line in lines:
        plain = line.lstrip("#").strip()
        for sentence in re.split(r"(?<=[.!?])\s+", plain):
            sentence = sentence.strip()
            if len(sentence.split()) < 4:
                continue
            sentence_pool.append(sentence)

    for sentence in sentence_pool:
        if not any(ch.isdigit() for ch in sentence):
            continue
        if len(sentence) < 30:
            continue
        add(sentence)
        if len(candidates) >= spans_per_case:
            break

    for sentence in sentence_pool:
        if len(sentence) < 24:
            continue
        add(sentence)
        if len(candidates) >= spans_per_case:
            break

    if len(candidates) < spans_per_case:
        fallback_lines = [line.lstrip("#").strip() for line in lines if line.strip()]
        fallback_lines.sort(key=len, reverse=True)
        for line in fallback_lines:
            add(line)
            if len(candidates) >= spans_per_case:
                break

    if len(candidates) < 2:
        raise ValueError("Unable to extract at least two required spans from source text.")

    return candidates[:spans_per_case]


def map_question_type_to_memory_type(question_type: str) -> str:
    return QUESTION_TYPE_TO_MEMORY_TYPE.get(question_type, "contextual")


def load_instances(source_path: str) -> list[dict[str, Any]]:
    with open(source_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("LongMemEval source must be a JSON array.")

    normalized: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "question_id" not in item or "question" not in item:
            continue
        normalized.append(item)

    normalized.sort(key=lambda row: str(row.get("question_id", "")))
    return normalized


def convert(
    instances: list[dict[str, Any]],
    output_dir: str,
    storage_max_cases: int,
    storage_spans_per_case: int,
    max_queries: Optional[int],
) -> None:
    if max_queries is not None:
        instances = instances[:max_queries]

    os.makedirs(output_dir, exist_ok=True)

    session_turns: dict[str, list[dict[str, Any]]] = {}
    session_dates: dict[str, str] = {}
    session_type_votes: dict[str, Counter[str]] = {}
    date_conflict_count = 0

    question_type_counts: Counter[str] = Counter()
    for instance in instances:
        question_type = str(instance.get("question_type", "unknown"))
        question_type_counts[question_type] += 1
        memory_type = map_question_type_to_memory_type(question_type)

        haystack_session_ids = instance.get("haystack_session_ids") or []
        haystack_sessions = instance.get("haystack_sessions") or []
        haystack_dates = instance.get("haystack_dates") or []

        if len(haystack_session_ids) != len(haystack_sessions):
            raise ValueError(
                f"question_id={instance.get('question_id')} has mismatched haystack_session_ids ({len(haystack_session_ids)}) "
                f"and haystack_sessions ({len(haystack_sessions)})."
            )

        for index, session_id_raw in enumerate(haystack_session_ids):
            session_id = str(session_id_raw)
            turns = haystack_sessions[index] or []
            if not isinstance(turns, list):
                raise ValueError(f"question_id={instance.get('question_id')} has non-list session turns for {session_id}.")

            existing_turns = session_turns.get(session_id)
            if existing_turns is None:
                session_turns[session_id] = turns
            elif existing_turns != turns:
                raise ValueError(f"Conflicting transcript content detected for session_id={session_id}.")

            date_value = ""
            if index < len(haystack_dates):
                date_value = str(haystack_dates[index] or "")
            existing_date = session_dates.get(session_id)
            if existing_date is None:
                session_dates[session_id] = date_value
            elif date_value and existing_date and existing_date != date_value:
                # Some cleaned rows reuse session ids with different date metadata.
                # Keep the first non-empty value to stay deterministic.
                date_conflict_count += 1
            elif date_value and not existing_date:
                session_dates[session_id] = date_value

            session_type_votes.setdefault(session_id, Counter())[memory_type] += 1

    documents: list[dict[str, Any]] = []
    text_by_doc_id: dict[str, str] = {}
    memory_type_by_doc_id: dict[str, str] = {}

    for session_id in sorted(session_turns):
        doc_id = f"doc-{sanitize_id(session_id)}"
        session_date = session_dates.get(session_id, "")
        text = render_session_markdown(session_id=session_id, session_date=session_date, turns=session_turns[session_id])

        votes = session_type_votes.get(session_id, Counter())
        memory_type = votes.most_common(1)[0][0] if votes else "contextual"
        memory_type_by_doc_id[doc_id] = memory_type
        text_by_doc_id[doc_id] = text

        documents.append(
            {
                "id": doc_id,
                "relative_path": f"sessions/{sanitize_id(session_id)}.md",
                "kind": "markdown",
                "text": text,
                "memory_type": memory_type,
            }
        )

    queries: list[dict[str, Any]] = []
    relevant_doc_order: list[str] = []
    seen_relevant_docs: set[str] = set()

    for instance in instances:
        question_id = str(instance.get("question_id", "")).strip()
        query_text = str(instance.get("question", "")).strip()
        if not question_id or not query_text:
            continue

        answer_session_ids = [str(value) for value in (instance.get("answer_session_ids") or [])]
        relevant_doc_ids: list[str] = []
        seen_for_query: set[str] = set()
        for session_id in answer_session_ids:
            doc_id = f"doc-{sanitize_id(session_id)}"
            if doc_id not in text_by_doc_id:
                continue
            if doc_id in seen_for_query:
                continue
            seen_for_query.add(doc_id)
            relevant_doc_ids.append(doc_id)
            if doc_id not in seen_relevant_docs:
                seen_relevant_docs.add(doc_id)
                relevant_doc_order.append(doc_id)

        if not relevant_doc_ids:
            continue

        queries.append(
            {
                "id": f"q-{sanitize_id(question_id)}",
                "query": query_text,
                "relevant_document_ids": relevant_doc_ids,
                "difficulty": "unknown",
            }
        )

    selected_storage_doc_ids = relevant_doc_order[:storage_max_cases]
    storage_records: list[dict[str, Any]] = []
    for doc_id in selected_storage_doc_ids:
        text = text_by_doc_id[doc_id]
        required_spans = extract_required_spans(text, storage_spans_per_case)
        storage_records.append(
            {
                "id": f"storage-{doc_id.removeprefix('doc-')}",
                "kind": "markdown",
                "text": text,
                "expected_memory_type": memory_type_by_doc_id.get(doc_id, "contextual"),
                "required_spans": required_spans,
            }
        )

    docs_path = os.path.join(output_dir, "recall_documents.jsonl")
    queries_path = os.path.join(output_dir, "recall_queries.jsonl")
    storage_path = os.path.join(output_dir, "storage_cases.jsonl")

    with open(docs_path, "w", encoding="utf-8") as handle:
        for record in documents:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(queries_path, "w", encoding="utf-8") as handle:
        for record in queries:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(storage_path, "w", encoding="utf-8") as handle:
        for record in storage_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    memory_type_counts = Counter(record["memory_type"] for record in documents)
    print(f"\nConverted LongMemEval instances to {output_dir}")
    print(f"  Input instances: {len(instances)}")
    print(f"  Recall documents: {len(documents)}")
    print(f"  Recall queries: {len(queries)}")
    print(f"  Storage cases: {len(storage_records)}")
    print(f"  Question types: {dict(question_type_counts)}")
    print(f"  Document memory types: {dict(memory_type_counts)}")
    if date_conflict_count:
        print(f"  Date conflicts ignored: {date_conflict_count}")
    print(f"  Files: {docs_path}")
    print(f"         {queries_path}")
    print(f"         {storage_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LongMemEval-cleaned dataset to Memory.swift eval format.")
    parser.add_argument("--output-dir", required=True, help="Output directory for eval files.")
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_TO_FILENAME.keys()),
        default="oracle",
        help="LongMemEval cleaned split to download (default: oracle).",
    )
    parser.add_argument(
        "--source-url",
        help="Optional source URL override. If omitted, built from --split using the public Hugging Face dataset.",
    )
    parser.add_argument("--source-file", help="Optional local JSON file path; overrides URL download if set.")
    parser.add_argument("--cache-dir", default="/tmp/longmemeval_datasets", help="Download cache directory.")
    parser.add_argument(
        "--storage-max-cases",
        type=int,
        default=300,
        help="Maximum auto-generated storage cases (default: 300).",
    )
    parser.add_argument(
        "--storage-spans-per-case",
        type=int,
        default=3,
        help="Required spans per storage case (2-4, default: 3).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap on number of queries from source (after sorting).",
    )
    args = parser.parse_args()

    if args.storage_max_cases <= 0:
        parser.error("--storage-max-cases must be > 0.")
    if args.storage_spans_per_case < 2 or args.storage_spans_per_case > 4:
        parser.error("--storage-spans-per-case must be between 2 and 4.")
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be > 0 when provided.")
    if args.source_file and args.source_url:
        parser.error("Provide either --source-file or --source-url, not both.")

    if args.source_file:
        source_path = args.source_file
    else:
        source_url = args.source_url or f"{DEFAULT_SOURCE_BASE}/{SPLIT_TO_FILENAME[args.split]}"
        source_path = download_source(source_url=source_url, cache_dir=args.cache_dir)

    instances = load_instances(source_path)
    convert(
        instances=instances,
        output_dir=args.output_dir,
        storage_max_cases=args.storage_max_cases,
        storage_spans_per_case=args.storage_spans_per_case,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()
