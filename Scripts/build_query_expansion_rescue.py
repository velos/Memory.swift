#!/usr/bin/env python3
"""Mine query-expansion rescue cases from existing recall run artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from eval_data_codex_support import load_jsonl, log, normalize_spaces, write_jsonl_atomic, write_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = [
    (
        "Evals/general_v2",
        "Evals/general_v2/runs/2026-03-27T15-42-53Z-coreml_default.json",
    ),
    (
        "Evals/longmemeval_v2",
        "Evals/longmemeval_v2/runs/2026-03-16T01-23-59Z-coreml_leaf_ir.json",
    ),
]

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*|\d+(?:/\d+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CAP_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9'_-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'_-]+|[A-Z]{2,}))*\b")

STOP_WORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "between",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "for",
    "from",
    "get",
    "go",
    "going",
    "had",
    "has",
    "have",
    "having",
    "he",
    "hello",
    "help",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "know",
    "last",
    "let",
    "like",
    "me",
    "more",
    "my",
    "need",
    "new",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "or",
    "our",
    "please",
    "recently",
    "should",
    "so",
    "some",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "up",
    "us",
    "use",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

TEMPORAL_TERMS = {
    "after",
    "before",
    "between",
    "day",
    "days",
    "month",
    "months",
    "week",
    "weeks",
    "year",
    "years",
    "yesterday",
    "today",
    "tomorrow",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "january",
    "february",
}

COUNT_TERMS = {
    "count",
    "counts",
    "different",
    "how many",
    "most",
    "total",
    "times",
}

ELLIPSIS_TERMS = {"it", "this", "that", "them", "they", "those"}

ENTITY_STOP_WORDS = STOP_WORDS.union(
    {
        "another",
        "because",
        "learn",
        "life",
        "local",
        "many",
        "should",
        "temporary",
        "the",
        "this",
        "top",
        "what",
        "when",
        "where",
        "you",
    }
)


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "source"


def stable_id(source_slug: str, original_id: str) -> str:
    return f"{source_slug}__{original_id}"


def tokens(text: str) -> List[str]:
    result: List[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("'_-")
        if len(token) < 3 or token in STOP_WORDS:
            continue
        result.append(token)
    return result


def unique_preserving_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for raw in values:
        value = normalize_spaces(str(raw)).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def first_relevant_rank(retrieved: Sequence[str], relevant: Set[str], max_k: int) -> Optional[int]:
    for index, document_id in enumerate(retrieved[:max_k], start=1):
        if document_id in relevant:
            return index
    return None


def source_from_arg(raw: str) -> Tuple[str, str]:
    if ":" not in raw:
        raise ValueError(f"Source must be DATASET_ROOT:RUN_JSON, got {raw!r}")
    dataset, run = raw.split(":", 1)
    return dataset, run


def load_json(path: Path) -> Dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return parsed


def doc_text(document: Dict[str, Any]) -> str:
    return normalize_spaces(str(document.get("text") or ""))


def best_sentence(text: str, query_tokens: Set[str], limit: int = 120) -> str:
    best = ""
    best_score = -1
    for sentence in SENTENCE_RE.split(normalize_spaces(text)):
        candidate = sentence.strip()
        if not candidate:
            continue
        candidate_tokens = set(tokens(candidate))
        score = len(query_tokens.intersection(candidate_tokens))
        if score > best_score:
            best = candidate
            best_score = score
    if not best:
        best = normalize_spaces(text)
    if len(best) <= limit:
        return best
    return best[:limit].rsplit(" ", 1)[0]


def phrase_candidates(text: str, max_count: int = 6) -> List[str]:
    stream = tokens(text)
    phrases: List[str] = []
    for size in (3, 2):
        for index in range(0, max(0, len(stream) - size + 1)):
            window = stream[index : index + size]
            if len(set(window)) != len(window):
                continue
            phrases.append(" ".join(window))
            if len(unique_preserving_order(phrases)) >= max_count:
                return unique_preserving_order(phrases)[:max_count]
    return unique_preserving_order(phrases)[:max_count]


def top_doc_terms(relevant_text: str, query_token_set: Set[str], max_count: int = 4) -> List[str]:
    counts = Counter(tokens(relevant_text))
    ranked = sorted(
        counts.items(),
        key=lambda item: (
            item[0] in query_token_set,
            item[1],
            -len(item[0]),
            item[0],
        ),
        reverse=True,
    )
    return [token for token, _ in ranked if token not in query_token_set][:max_count]


def extract_entities(query: str, relevant_text: str, max_count: int = 5) -> List[str]:
    raw_entities: List[str] = []
    for text in (query,):
        for match in CAP_ENTITY_RE.finditer(text):
            value = match.group(0).strip(" .,:;!?")
            words = [word for word in re.split(r"\s+", value) if word]
            normalized_words = [word.lower().strip("'_-") for word in words]
            if len(value) <= 2 and not value.isupper():
                continue
            if not value.isupper() and normalized_words[0] in ENTITY_STOP_WORDS:
                continue
            if all(word in ENTITY_STOP_WORDS for word in normalized_words):
                continue
            raw_entities.append(value.lower() if value.isupper() else value)
    if raw_entities:
        return unique_preserving_order(raw_entities)[:max_count]

    for match in re.finditer(r"\b[A-Z]{2,}\b", relevant_text[:1200]):
        value = match.group(0)
        if value.lower() not in ENTITY_STOP_WORDS:
            raw_entities.append(value.lower())
    return unique_preserving_order(raw_entities)[:max_count]


def infer_expected_facets(query: str) -> List[str]:
    lowered = query.lower()
    query_tokens = set(tokens(query))
    facets: List[str] = []
    if query_tokens.intersection(TEMPORAL_TERMS) or any(term in lowered for term in COUNT_TERMS):
        facets.append("time_sensitive")
    if any(term in lowered for term in ("where", "location", "city", "state", "county")):
        facets.append("location")
    return facets[:2]


def expected_fields(query: str, relevant_docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    relevant_text = normalize_spaces(" ".join(doc_text(document) for document in relevant_docs))
    query_token_set = set(tokens(query))
    query_phrases = phrase_candidates(query, max_count=5)
    doc_terms = top_doc_terms(relevant_text, query_token_set, max_count=4)

    lexical_terms = unique_preserving_order(query_phrases[:3] + doc_terms[:3])[:6]
    if not lexical_terms:
        lexical_terms = unique_preserving_order(tokens(query))[:4]

    sentence = best_sentence(relevant_text, query_token_set)
    semantic_phrases = unique_preserving_order(
        [
            f"retrieve memories about {' '.join(lexical_terms[:3])}",
            sentence,
        ]
    )[:2]
    hyde_anchors = unique_preserving_order(lexical_terms[:3] + phrase_candidates(sentence, max_count=2))[:5]
    topics = unique_preserving_order(query_phrases[:3] + phrase_candidates(sentence, max_count=3))[:5]

    return {
        "expected_lexical_terms": lexical_terms,
        "expected_semantic_phrases": semantic_phrases,
        "expected_hyde_anchors": hyde_anchors,
        "expected_facets": infer_expected_facets(query),
        "expected_entities": extract_entities(query, relevant_text),
        "expected_topics": topics,
    }


def lexical_similarity(query_tokens: Set[str], document_tokens: Set[str], relevant_tokens: Set[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0

    query_overlap = len(query_tokens.intersection(document_tokens))
    relevant_overlap = len(relevant_tokens.intersection(document_tokens))
    density = math.sqrt(max(1, len(document_tokens)))
    return (3.0 * query_overlap + 0.25 * relevant_overlap) / density


def build_candidate_ids(
    *,
    result: Dict[str, Any],
    docs_by_id: Dict[str, Dict[str, Any]],
    doc_tokens_by_id: Dict[str, Set[str]],
    query: str,
    relevant_ids: Sequence[str],
    max_candidates: int,
) -> List[str]:
    selected: List[str] = []
    query_token_set = set(tokens(query))
    relevant_tokens: Set[str] = set()
    for document_id in relevant_ids:
        relevant_tokens.update(doc_tokens_by_id.get(document_id, set()))

    def add(document_id: str) -> None:
        if document_id in docs_by_id and document_id not in selected:
            selected.append(document_id)

    for document_id in relevant_ids:
        add(document_id)
    for document_id in result.get("retrievedDocumentIds") or []:
        add(str(document_id))

    if len(selected) < max_candidates:
        scored = [
            (lexical_similarity(query_token_set, doc_tokens_by_id.get(document_id, set()), relevant_tokens), document_id)
            for document_id in docs_by_id
            if document_id not in selected
        ]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for score, document_id in scored:
            if score <= 0 and len(selected) >= max(12, len(relevant_ids) + 6):
                break
            add(document_id)
            if len(selected) >= max_candidates:
                break

    return selected[: max(max_candidates, len(relevant_ids))]


def classify_failure(
    *,
    query: str,
    relevant_ids: Sequence[str],
    retrieved_ids: Sequence[str],
    docs_by_id: Dict[str, Dict[str, Any]],
    doc_tokens_by_id: Dict[str, Set[str]],
    rank: Optional[int],
) -> List[str]:
    query_token_set = set(tokens(query))
    relevant_tokens: Set[str] = set()
    for document_id in relevant_ids:
        relevant_tokens.update(doc_tokens_by_id.get(document_id, set()))
    overlap = len(query_token_set.intersection(relevant_tokens)) / max(1, len(query_token_set))
    lowered = query.lower()

    taxonomy: List[str] = []
    if rank is None:
        taxonomy.append("retrieval_miss")
    else:
        taxonomy.append("low_rank_relevant")
    if not retrieved_ids:
        taxonomy.append("empty_retrieval")
    if overlap < 0.25:
        taxonomy.append("lexical_mismatch")
    if query_token_set.intersection(TEMPORAL_TERMS):
        taxonomy.append("temporal_reasoning")
    if any(term in lowered for term in COUNT_TERMS):
        taxonomy.append("count_aggregation")
    if len(relevant_ids) > 1:
        taxonomy.append("multi_evidence")
    if query_token_set.intersection(ELLIPSIS_TERMS) or len(query_token_set) <= 4:
        taxonomy.append("contextual_ellipsis")
    return taxonomy


def rescue_reason(rank: Optional[int], max_k: int, taxonomy: Sequence[str]) -> str:
    if rank is None:
        reason = f"source recall did not retrieve a relevant document in the top {max_k}"
    else:
        reason = f"source recall ranked the first relevant document at {rank}, below the top 3"
    if taxonomy:
        reason += f" ({', '.join(taxonomy)})"
    return reason


def selected_source_cases(
    *,
    dataset_root: Path,
    run_path: Path,
    max_cases: int,
    max_candidates: int,
    min_rank: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    documents = load_jsonl(dataset_root / "recall_documents.jsonl")
    queries = load_jsonl(dataset_root / "recall_queries.jsonl")
    docs_by_id = {str(document["id"]): document for document in documents}
    doc_tokens_by_id = {document_id: set(tokens(doc_text(document))) for document_id, document in docs_by_id.items()}
    queries_by_id = {str(query["id"]): query for query in queries}
    report = load_json(run_path)
    recall_report = report.get("recall") or {}
    k_values = [int(value) for value in recall_report.get("kValues") or [10]]
    max_k = max(k_values) if k_values else 10
    profile = str(report.get("profile") or run_path.stem.split("-")[-1])
    source_slug = safe_slug(dataset_root.name)

    candidates: List[Tuple[Tuple[int, int, int, str], Dict[str, Any]]] = []
    for result in recall_report.get("queryResults") or []:
        query_id = str(result.get("id") or "")
        query_case = queries_by_id.get(query_id, {})
        query_text = normalize_spaces(str(result.get("query") or query_case.get("query") or ""))
        if not query_text:
            continue
        relevant_ids = [str(value) for value in (result.get("relevantDocumentIds") or query_case.get("relevant_document_ids") or [])]
        relevant_ids = [document_id for document_id in unique_preserving_order(relevant_ids) if document_id in docs_by_id]
        if not relevant_ids:
            continue

        retrieved_ids = [str(value) for value in (result.get("retrievedDocumentIds") or [])]
        rank = first_relevant_rank(retrieved_ids, set(relevant_ids), max_k)
        if rank is not None and rank < min_rank:
            continue

        taxonomy = classify_failure(
            query=query_text,
            relevant_ids=relevant_ids,
            retrieved_ids=retrieved_ids,
            docs_by_id=docs_by_id,
            doc_tokens_by_id=doc_tokens_by_id,
            rank=rank,
        )

        severity = 10_000 if rank is None else rank
        lexical_mismatch = 1 if "lexical_mismatch" in taxonomy else 0
        sort_key = (
            1 if rank is None else 0,
            severity,
            lexical_mismatch,
            query_id,
        )

        candidates.append((sort_key, {
            "query_id": query_id,
            "query": query_text,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "rank": rank,
            "failure_taxonomy": taxonomy,
            "result": result,
        }))

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected_raw: List[Dict[str, Any]] = []
    per_relevant: Dict[str, int] = defaultdict(int)
    seen_queries: Set[str] = set()
    for _, candidate in candidates:
        primary_relevant = candidate["relevant_ids"][0]
        query_key = normalize_spaces(candidate["query"].lower())
        if query_key in seen_queries:
            continue
        if per_relevant[primary_relevant] >= 2:
            continue
        selected_raw.append(candidate)
        seen_queries.add(query_key)
        per_relevant[primary_relevant] += 1
        if len(selected_raw) >= max_cases:
            break

    if len(selected_raw) < max_cases:
        for _, candidate in candidates:
            query_key = normalize_spaces(candidate["query"].lower())
            if query_key in seen_queries:
                continue
            selected_raw.append(candidate)
            seen_queries.add(query_key)
            if len(selected_raw) >= max_cases:
                break

    selected: List[Dict[str, Any]] = []
    for candidate in selected_raw:
        query_id = candidate["query_id"]
        query_text = candidate["query"]
        relevant_ids = candidate["relevant_ids"]
        rank = candidate["rank"]
        taxonomy = candidate["failure_taxonomy"]
        candidate_ids = build_candidate_ids(
            result=candidate["result"],
            docs_by_id=docs_by_id,
            doc_tokens_by_id=doc_tokens_by_id,
            query=query_text,
            relevant_ids=relevant_ids,
            max_candidates=max_candidates,
        )
        if not set(relevant_ids).issubset(candidate_ids):
            continue
        case = {
            "id": f"qer-{source_slug}-{query_id}",
            "query": query_text,
            **expected_fields(query_text, [docs_by_id[document_id] for document_id in relevant_ids]),
            "relevant_document_ids": [stable_id(source_slug, document_id) for document_id in relevant_ids],
            "candidate_document_ids": [stable_id(source_slug, document_id) for document_id in candidate_ids],
            "source_dataset": dataset_root.name,
            "source_query_id": query_id,
            "source_profile": profile,
            "failure_taxonomy": taxonomy,
            "rescue_reason": rescue_reason(rank, max_k, taxonomy),
        }
        if rank is not None:
            case["source_rank_at_k"] = rank
        selected.append(case)

    needed_ids = {
        document_id.removeprefix(f"{source_slug}__")
        for case in selected
        for document_id in case["candidate_document_ids"]
    }
    selected_documents: List[Dict[str, Any]] = []
    for original_id in sorted(needed_ids):
        if original_id not in docs_by_id:
            continue
        source_doc = dict(docs_by_id[original_id])
        source_doc["id"] = stable_id(source_slug, original_id)
        relative_path = str(source_doc.get("relative_path") or f"{original_id}.md").lstrip("/")
        source_doc["relative_path"] = f"{source_slug}/{relative_path}"
        selected_documents.append(source_doc)

    metadata = {
        "dataset_root": str(dataset_root.relative_to(ROOT) if dataset_root.is_relative_to(ROOT) else dataset_root),
        "run_path": str(run_path.relative_to(ROOT) if run_path.is_relative_to(ROOT) else run_path),
        "profile": profile,
        "source_query_count": len(recall_report.get("queryResults") or []),
        "candidate_case_count": len(candidates),
        "selected_case_count": len(selected),
        "selected_document_count": len(selected_documents),
        "max_k": max_k,
    }
    return selected, selected_documents, metadata


def readme_text(manifest: Dict[str, Any]) -> str:
    sources = "\n".join(
        f"- `{source['dataset_root']}` from `{source['run_path']}` ({source['selected_case_count']} cases)"
        for source in manifest["sources"]
    )
    taxonomy = "\n".join(f"- `{key}`: {value}" for key, value in sorted(manifest["failure_taxonomy_counts"].items()))
    return f"""# query_expansion_rescue_v1

Regression benchmark for query-expansion rescue behavior.

This dataset is mined from real recall misses and low-rank cases, not model-invented prompts. Each case keeps the source query, the known relevant document IDs, a per-case candidate slice, and failure-taxonomy metadata so retrieval changes can be diagnosed by failure mode.

Sources:

{sources}

Failure taxonomy counts:

{taxonomy}

Generated files:

- `cases.jsonl`: query-expansion cases with expected lexical/topic/entity hints and source failure metadata.
- `recall_documents.jsonl`: the union of candidate documents needed by the cases.
- `manifest.json`: source run, selection, and taxonomy metadata.

Commands:

```sh
python3 Scripts/build_query_expansion_rescue.py --overwrite
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/query_expansion_rescue_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/query_expansion_rescue.json <candidate-run.json>
```

Primary metrics to watch are retrieval expanded Hit@K, expanded MRR@K, and MRR delta. Hit rate may saturate on small slices, so MRR delta is the sharper signal for whether expansion moves relevant memories closer to the top.
"""


def build(args: argparse.Namespace) -> None:
    output_root = resolve_path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(f"{output_root} already exists. Pass --overwrite to replace generated files.")

    source_args = [source_from_arg(raw) for raw in (args.source or [])] or DEFAULT_SOURCES
    all_cases: List[Dict[str, Any]] = []
    documents_by_id: Dict[str, Dict[str, Any]] = {}
    source_metadata: List[Dict[str, Any]] = []

    for dataset_raw, run_raw in source_args:
        dataset_root = resolve_path(dataset_raw)
        run_path = resolve_path(run_raw)
        log(f"Mining {dataset_root.relative_to(ROOT)} from {run_path.relative_to(ROOT)}")
        cases, documents, metadata = selected_source_cases(
            dataset_root=dataset_root,
            run_path=run_path,
            max_cases=args.max_cases_per_source,
            max_candidates=args.max_candidates,
            min_rank=args.min_rank,
        )
        source_metadata.append(metadata)
        all_cases.extend(cases)
        for document in documents:
            documents_by_id[str(document["id"])] = document

    all_cases = all_cases[: args.max_cases]
    needed_document_ids = {
        document_id
        for case in all_cases
        for document_id in case.get("candidate_document_ids", [])
    }
    all_documents = [documents_by_id[document_id] for document_id in sorted(needed_document_ids) if document_id in documents_by_id]
    taxonomy_counts = Counter(
        tag
        for case in all_cases
        for tag in case.get("failure_taxonomy", [])
    )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": "query_expansion_rescue_v1",
        "case_count": len(all_cases),
        "document_count": len(all_documents),
        "max_cases": args.max_cases,
        "max_cases_per_source": args.max_cases_per_source,
        "max_candidates": args.max_candidates,
        "min_rank": args.min_rank,
        "sources": source_metadata,
        "failure_taxonomy_counts": dict(sorted(taxonomy_counts.items())),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_root / "cases.jsonl", all_cases)
    write_jsonl_atomic(output_root / "recall_documents.jsonl", all_documents)
    write_manifest(output_root / "manifest.json", manifest)
    (output_root / "README.md").write_text(readme_text(manifest), encoding="utf-8")

    log(f"Wrote {len(all_cases)} cases and {len(all_documents)} documents to {output_root.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="Evals/query_expansion_rescue_v1")
    parser.add_argument(
        "--source",
        action="append",
        help="Source pair as DATASET_ROOT:RUN_JSON. Repeat to mine multiple reports.",
    )
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--max-cases-per-source", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--min-rank", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    if args.max_cases_per_source <= 0:
        raise ValueError("--max-cases-per-source must be positive")
    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    if args.min_rank <= 1:
        raise ValueError("--min-rank must be greater than 1")
    build(args)


if __name__ == "__main__":
    main()
