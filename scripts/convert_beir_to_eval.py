#!/usr/bin/env python3
"""Convert a BEIR dataset into Memory.swift eval format.

Usage:
    python3 scripts/convert_beir_to_eval.py --dataset scifact --output-dir ./Evals/scifact
    python3 scripts/convert_beir_to_eval.py --dataset scifact --output-dir ./Evals/scifact --storage-mode auto-factual
    python3 scripts/convert_beir_to_eval.py --dataset nfcorpus --output-dir ./Evals/nfcorpus
"""

import argparse
import json
import os
import re
import urllib.request
import zipfile

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def download_and_extract(dataset: str, cache_dir: str = "/tmp/beir_datasets") -> str:
    dataset_dir = os.path.join(cache_dir, dataset)
    if os.path.exists(dataset_dir):
        print(f"Using cached {dataset_dir}")
        return dataset_dir

    zip_url = f"{BEIR_BASE_URL}/{dataset}.zip"
    zip_path = os.path.join(cache_dir, f"{dataset}.zip")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading {zip_url}...")
    urllib.request.urlretrieve(zip_url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache_dir)
    os.remove(zip_path)
    return dataset_dir


def load_corpus(dataset_dir: str) -> dict:
    corpus = {}
    with open(os.path.join(dataset_dir, "corpus.jsonl")) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc
    return corpus


def load_queries(dataset_dir: str) -> dict:
    queries = {}
    with open(os.path.join(dataset_dir, "queries.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q
    return queries


def load_qrels(dataset_dir: str, split: str = "test") -> dict:
    qrels = {}
    qrels_path = os.path.join(dataset_dir, "qrels", f"{split}.tsv")
    with open(qrels_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(qid, []).append(did)
    return qrels


def compose_document_body(doc: dict) -> str:
    title = doc.get("title", "").strip()
    text = doc.get("text", "").strip()
    return f"# {title}\n\n{text}" if title else text


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
        key = " ".join(value.lower().split())
        if key in seen:
            return
        seen.add(key)
        candidates.append(value)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        first = lines[0]
        if first.startswith("#"):
            first = first.lstrip("#").strip()
        add(first)

    sentence_pool: list[str] = []
    for line in lines:
        plain = line
        if plain.startswith("#"):
            plain = plain.lstrip("#").strip()
        for sentence in re.split(r"(?<=[.!?])\s+", plain):
            value = sentence.strip()
            if len(value.split()) < 4:
                continue
            sentence_pool.append(value)

    for sentence in sentence_pool:
        if not any(ch.isdigit() for ch in sentence):
            continue
        if len(sentence) < 30:
            continue
        add(sentence)
        if len(candidates) >= spans_per_case:
            break

    for sentence in sentence_pool:
        if len(sentence) < 30:
            continue
        add(sentence)
        if len(candidates) >= spans_per_case:
            break

    if len(candidates) < spans_per_case:
        fallback_lines = [line.lstrip("#").strip() for line in lines if line.strip()]
        fallback_lines.sort(key=len, reverse=True)
        for line in fallback_lines:
            if not line:
                continue
            add(line)
            if len(candidates) >= spans_per_case:
                break

    if len(candidates) < 2:
        raise ValueError("Unable to extract at least two required spans from source text.")

    return candidates[:spans_per_case]


def build_storage_cases(
    storage_mode: str,
    doc_bodies: dict[str, str],
    qrels: dict[str, list[str]],
    storage_max_cases: int,
    storage_spans_per_case: int,
) -> list[dict]:
    if storage_mode == "stub":
        return [
            {
                "id": "stub-0001",
                "kind": "markdown",
                "text": "This is a placeholder for storage evaluation.",
                "expected_memory_type": "factual",
                "required_spans": ["placeholder"],
            }
        ]

    candidate_doc_ids = sorted(
        {
            did
            for relevant in qrels.values()
            for did in relevant
            if did in doc_bodies
        }
    )
    selected_doc_ids = candidate_doc_ids[:storage_max_cases]
    if not selected_doc_ids:
        raise ValueError("No qrels-linked corpus documents found for storage auto-generation.")

    records: list[dict] = []
    for doc_id in selected_doc_ids:
        text = doc_bodies[doc_id]
        required_spans = extract_required_spans(text, storage_spans_per_case)
        record = {
            "id": f"storage-{doc_id}",
            "kind": "markdown",
            "text": text,
            "expected_memory_type": "factual",
            "required_spans": required_spans,
        }
        for span in required_spans:
            if span not in text:
                raise ValueError(f"required_span not found in source text for storage-{doc_id}: {span!r}")
        records.append(record)

    return records


def convert(
    dataset: str,
    output_dir: str,
    split: str = "test",
    storage_mode: str = "auto-factual",
    storage_max_cases: int = 300,
    storage_spans_per_case: int = 3,
):
    dataset_dir = download_and_extract(dataset)
    corpus = load_corpus(dataset_dir)
    queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir, split)

    os.makedirs(output_dir, exist_ok=True)

    # Convert corpus to recall_documents.jsonl
    docs_path = os.path.join(output_dir, "recall_documents.jsonl")
    doc_count = 0
    doc_bodies: dict[str, str] = {}
    with open(docs_path, "w") as f:
        for doc_id in sorted(corpus):
            body = compose_document_body(corpus[doc_id])
            doc_bodies[doc_id] = body

            record = {
                "id": f"doc-{doc_id}",
                "relative_path": f"docs/{doc_id}.md",
                "kind": "document",
                "text": body,
                "memory_type": "factual",
            }
            f.write(json.dumps(record) + "\n")
            doc_count += 1

    # Convert queries + qrels to recall_queries.jsonl
    queries_path = os.path.join(output_dir, "recall_queries.jsonl")
    query_count = 0
    with open(queries_path, "w") as f:
        for qid in sorted(qrels):
            relevant_doc_ids = qrels[qid]
            query_obj = queries.get(qid)
            if not query_obj:
                continue

            valid_relevant = sorted({f"doc-{did}" for did in relevant_doc_ids if did in corpus})
            if not valid_relevant:
                continue

            record = {
                "id": f"q-{qid}",
                "query": query_obj["text"],
                "relevant_document_ids": valid_relevant,
                "difficulty": "unknown",
            }
            f.write(json.dumps(record) + "\n")
            query_count += 1

    # Create storage_cases.jsonl
    storage_path = os.path.join(output_dir, "storage_cases.jsonl")
    storage_records = build_storage_cases(
        storage_mode=storage_mode,
        doc_bodies=doc_bodies,
        qrels=qrels,
        storage_max_cases=storage_max_cases,
        storage_spans_per_case=storage_spans_per_case,
    )
    with open(storage_path, "w") as f:
        for record in storage_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nConverted {dataset} ({split} split) to {output_dir}")
    print(f"  Documents: {doc_count}")
    print(f"  Queries: {query_count}")
    print(f"  Storage cases: {len(storage_records)} ({storage_mode})")
    print(f"  Relevant pairs: {sum(len(v) for v in qrels.values())}")
    print(f"  Files: {docs_path}")
    print(f"          {queries_path}")
    print(f"          {storage_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert BEIR dataset to Memory.swift eval format")
    parser.add_argument("--dataset", required=True, help="BEIR dataset name (e.g., scifact, nfcorpus)")
    parser.add_argument("--output-dir", required=True, help="Output directory for eval files")
    parser.add_argument("--split", default="test", help="Qrels split to use (default: test)")
    parser.add_argument(
        "--storage-mode",
        choices=["auto-factual", "stub"],
        default="auto-factual",
        help="How to generate storage_cases.jsonl (default: auto-factual).",
    )
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
        help="Required spans per auto-generated storage case (2-4, default: 3).",
    )
    args = parser.parse_args()
    if args.storage_max_cases <= 0:
        parser.error("--storage-max-cases must be > 0.")
    if args.storage_spans_per_case < 2 or args.storage_spans_per_case > 4:
        parser.error("--storage-spans-per-case must be between 2 and 4.")

    convert(
        args.dataset,
        args.output_dir,
        args.split,
        storage_mode=args.storage_mode,
        storage_max_cases=args.storage_max_cases,
        storage_spans_per_case=args.storage_spans_per_case,
    )


if __name__ == "__main__":
    main()
