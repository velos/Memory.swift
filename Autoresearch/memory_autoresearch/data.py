from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .cache import datasets_root
from .config import (
    FULL_EVAL_CORPORA,
    MEMORY_TYPE_TO_INDEX,
    QUICK_EVAL_CORPORA,
    RANDOM_SEED,
    TYPING_GOLD_CORPORA,
    TRAINING_CORPORA,
)


def _scoped_id(corpus: str, record_id: str) -> str:
    return f"{corpus}:{record_id}"


def _scoped_relative_path(corpus: str, relative_path: str) -> str:
    path = Path(relative_path)
    return str(Path(corpus) / path)


@dataclass
class StorageCase:
    id: str
    kind: str
    text: str
    expected_memory_type: str
    required_spans: list[str]
    corpus: str


@dataclass
class RecallDocument:
    id: str
    relative_path: str
    kind: str
    text: str
    memory_type: str | None
    corpus: str


@dataclass
class RecallQuery:
    id: str
    query: str
    relevant_document_ids: list[str]
    memory_types: list[str]
    difficulty: str | None
    corpus: str


@dataclass
class TypingExample:
    id: str
    text: str
    label: str
    kind: str
    source: str


@dataclass
class RetrievalExample:
    query_id: str
    query: str
    positive_document_id: str
    positive_document_text: str
    positive_memory_types: list[str]
    hard_negative_ids: list[str]
    corpus: str


@dataclass
class BM25Index:
    documents: list[RecallDocument]
    tokenized_docs: list[list[str]]
    doc_freqs: Counter
    avg_doc_len: float


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_storage_cases(root: Path, corpus: str) -> list[StorageCase]:
    records = _read_jsonl(root / corpus / "storage_cases.jsonl")
    return [
        StorageCase(
            id=_scoped_id(corpus, record["id"]),
            kind=record.get("kind", "markdown"),
            text=record["text"],
            expected_memory_type=record["expected_memory_type"],
            required_spans=record.get("required_spans", []),
            corpus=corpus,
        )
        for record in records
    ]


def load_recall_documents(root: Path, corpus: str) -> list[RecallDocument]:
    records = _read_jsonl(root / corpus / "recall_documents.jsonl")
    return [
        RecallDocument(
            id=_scoped_id(corpus, record["id"]),
            relative_path=_scoped_relative_path(
                corpus,
                record.get("relative_path", f"{record['id']}.md"),
            ),
            kind=record.get("kind", "markdown"),
            text=record["text"],
            memory_type=record.get("memory_type"),
            corpus=corpus,
        )
        for record in records
    ]


def load_recall_queries(root: Path, corpus: str) -> list[RecallQuery]:
    records = _read_jsonl(root / corpus / "recall_queries.jsonl")
    return [
        RecallQuery(
            id=_scoped_id(corpus, record["id"]),
            query=record["query"],
            relevant_document_ids=[
                _scoped_id(corpus, document_id)
                for document_id in record["relevant_document_ids"]
            ],
            memory_types=record.get("memory_types", []),
            difficulty=record.get("difficulty"),
            corpus=corpus,
        )
        for record in records
    ]


def build_typing_examples(root: Path) -> list[TypingExample]:
    examples: list[TypingExample] = []
    for corpus in TRAINING_CORPORA:
        for case in load_storage_cases(root, corpus):
            examples.append(
                TypingExample(
                    id=case.id,
                    text=case.text,
                    label=case.expected_memory_type,
                    kind=case.kind,
                    source=f"{corpus}:storage",
                )
            )
        for document in load_recall_documents(root, corpus):
            if not document.memory_type:
                continue
            examples.append(
                TypingExample(
                    id=document.id,
                    text=document.text,
                    label=document.memory_type,
                    kind=document.kind,
                    source=f"{corpus}:recall_document",
                )
            )
    return examples


def _tokenize_for_bm25(text: str) -> list[str]:
    return [
        token
        for token in "".join(char.lower() if char.isalnum() else " " for char in text).split()
        if len(token) >= 2
    ]


def _build_bm25_index(documents: Iterable[RecallDocument]) -> BM25Index:
    docs = list(documents)
    tokenized_docs = [_tokenize_for_bm25(document.text) for document in docs]
    doc_freqs = Counter()
    for tokens in tokenized_docs:
        doc_freqs.update(set(tokens))
    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(len(tokenized_docs), 1)
    return BM25Index(
        documents=docs,
        tokenized_docs=tokenized_docs,
        doc_freqs=doc_freqs,
        avg_doc_len=avg_doc_len,
    )


def _bm25_rank(query: str, index: BM25Index, top_k: int = 5) -> list[str]:
    docs = index.documents
    if not docs:
        return []
    query_tokens = _tokenize_for_bm25(query)
    scored: list[tuple[float, str]] = []
    for document, tokens in zip(docs, index.tokenized_docs):
        tf = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            if token not in index.doc_freqs:
                continue
            idf = max(0.0, (len(docs) - index.doc_freqs[token] + 0.5) / (index.doc_freqs[token] + 0.5))
            denom = tf[token] + 1.5 * (
                1 - 0.75 + 0.75 * (len(tokens) / max(index.avg_doc_len, 1.0))
            )
            if denom == 0:
                continue
            score += idf * ((tf[token] * 2.5) / denom)
        scored.append((score, document.id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [document_id for score, document_id in scored if score > 0][:top_k]


def build_retrieval_examples(root: Path) -> list[RetrievalExample]:
    examples: list[RetrievalExample] = []
    for corpus in TRAINING_CORPORA:
        documents = load_recall_documents(root, corpus)
        queries = load_recall_queries(root, corpus)
        bm25_index = _build_bm25_index(documents)
        doc_map = {document.id: document for document in documents}
        for query in queries:
            hard_candidates = [
                doc_id
                for doc_id in _bm25_rank(query.query, bm25_index, top_k=8)
                if doc_id not in query.relevant_document_ids
            ]
            for positive_document_id in query.relevant_document_ids:
                positive_document = doc_map.get(positive_document_id)
                if positive_document is None:
                    continue
                examples.append(
                    RetrievalExample(
                        query_id=query.id,
                        query=query.query,
                        positive_document_id=positive_document_id,
                        positive_document_text=positive_document.text,
                        positive_memory_types=query.memory_types,
                        hard_negative_ids=hard_candidates,
                        corpus=corpus,
                    )
                )
    return examples


def _stratified_split(records: list, key_fn, ratio: float, seed: int) -> tuple[list, list]:
    grouped: dict[str, list] = defaultdict(list)
    for record in records:
        grouped[key_fn(record)].append(record)
    train: list = []
    held_out: list = []
    rng = random.Random(seed)
    for group in grouped.values():
        group = list(group)
        rng.shuffle(group)
        boundary = max(1, int(len(group) * ratio))
        held_out.extend(group[:boundary])
        train.extend(group[boundary:])
    return train, held_out


def build_eval_splits(root: Path) -> tuple[dict[str, list], dict[str, list], dict[str, list]]:
    quick_storage: list[StorageCase] = []
    quick_documents: list[RecallDocument] = []
    quick_queries: list[RecallQuery] = []
    full_storage: list[StorageCase] = []
    full_documents: list[RecallDocument] = []
    full_queries: list[RecallQuery] = []

    for corpus in QUICK_EVAL_CORPORA:
        storage_cases = load_storage_cases(root, corpus)
        recall_documents = load_recall_documents(root, corpus)
        recall_queries = load_recall_queries(root, corpus)
        if corpus in TYPING_GOLD_CORPORA:
            quick_storage.extend(storage_cases)
            full_storage.extend(storage_cases)
            quick_documents.extend(recall_documents)
            quick_queries.extend(recall_queries)
            full_documents.extend(recall_documents)
            full_queries.extend(recall_queries)
            continue
        _, held_storage = _stratified_split(
            storage_cases,
            key_fn=lambda item: item.expected_memory_type,
            ratio=0.2,
            seed=RANDOM_SEED,
        )
        _, held_queries = _stratified_split(
            recall_queries,
            key_fn=lambda item: item.difficulty or "unknown",
            ratio=0.2,
            seed=RANDOM_SEED,
        )
        held_query_ids = {query.id for query in held_queries}
        full_queries_for_corpus = [query for query in recall_queries if query.id not in held_query_ids]
        held_doc_ids = {doc_id for query in held_queries for doc_id in query.relevant_document_ids}
        full_doc_ids = {
            doc_id
            for query in full_queries_for_corpus
            for doc_id in query.relevant_document_ids
        }
        held_documents = [document for document in recall_documents if document.id in held_doc_ids]
        full_documents_for_corpus = [document for document in recall_documents if document.id in full_doc_ids]
        quick_storage.extend(held_storage)
        quick_documents.extend(held_documents)
        quick_queries.extend(held_queries)
        full_storage.extend([case for case in storage_cases if case not in held_storage])
        full_documents.extend(full_documents_for_corpus)
        full_queries.extend(full_queries_for_corpus)

    for corpus in FULL_EVAL_CORPORA:
        if corpus in QUICK_EVAL_CORPORA:
            continue
        full_storage.extend(load_storage_cases(root, corpus))
        full_documents.extend(load_recall_documents(root, corpus))
        full_queries.extend(load_recall_queries(root, corpus))

    quick_split = {
        "storage_cases": quick_storage,
        "recall_documents": quick_documents,
        "recall_queries": quick_queries,
    }
    full_split = {
        "storage_cases": full_storage,
        "recall_documents": full_documents,
        "recall_queries": full_queries,
    }
    training_split = {
        "typing_examples": build_typing_examples(root),
        "retrieval_examples": build_retrieval_examples(root),
    }
    return training_split, quick_split, full_split


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def _materialize_eval_dataset(target_root: Path, split: dict[str, list]) -> None:
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    readme = target_root / "README.md"
    readme.write_text(
        "# Automemory eval dataset\n\nGenerated by prepare.py.\n",
        encoding="utf-8",
    )
    _write_jsonl(
        target_root / "storage_cases.jsonl",
        [asdict(record) for record in split["storage_cases"]],
    )
    _write_jsonl(
        target_root / "recall_documents.jsonl",
        [asdict(record) for record in split["recall_documents"]],
    )
    _write_jsonl(
        target_root / "recall_queries.jsonl",
        [asdict(record) for record in split["recall_queries"]],
    )


def materialize_dataset_cache(upstream_evals_root: Path) -> dict[str, Path]:
    training_split, quick_split, full_split = build_eval_splits(upstream_evals_root)
    root = datasets_root()
    typing_path = root / "typing_train.jsonl"
    retrieval_path = root / "retrieval_train.jsonl"
    _write_jsonl(
        typing_path,
        [asdict(example) for example in training_split["typing_examples"]],
    )
    _write_jsonl(
        retrieval_path,
        [asdict(example) for example in training_split["retrieval_examples"]],
    )
    quick_root = root / "quick_eval"
    full_root = root / "full_eval"
    _materialize_eval_dataset(quick_root, quick_split)
    _materialize_eval_dataset(full_root, full_split)
    return {
        "typing_train": typing_path,
        "retrieval_train": retrieval_path,
        "quick_eval": quick_root,
        "full_eval": full_root,
    }


def load_typing_examples(path: Path) -> list[TypingExample]:
    return [TypingExample(**record) for record in _read_jsonl(path)]


def load_retrieval_examples(path: Path) -> list[RetrievalExample]:
    return [RetrievalExample(**record) for record in _read_jsonl(path)]


def class_weights(examples: list[TypingExample]) -> list[float]:
    counts = Counter(example.label for example in examples)
    weights = []
    for label in MEMORY_TYPE_TO_INDEX:
        count = counts.get(label, 1)
        weights.append(sum(counts.values()) / (len(counts) * count))
    return weights
