#!/usr/bin/env python3
"""Convert a BEIR dataset into Memory.swift eval format.

Usage:
    python3 scripts/convert_beir_to_eval.py --dataset scifact --output-dir ./Evals/scifact
    python3 scripts/convert_beir_to_eval.py --dataset nfcorpus --output-dir ./Evals/nfcorpus
"""

import argparse
import json
import os
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


def convert(dataset: str, output_dir: str, split: str = "test"):
    dataset_dir = download_and_extract(dataset)
    corpus = load_corpus(dataset_dir)
    queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir, split)

    os.makedirs(output_dir, exist_ok=True)

    # Convert corpus to recall_documents.jsonl
    docs_path = os.path.join(output_dir, "recall_documents.jsonl")
    doc_count = 0
    with open(docs_path, "w") as f:
        for doc_id, doc in corpus.items():
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            body = f"# {title}\n\n{text}" if title else text

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
        for qid, relevant_doc_ids in qrels.items():
            query_obj = queries.get(qid)
            if not query_obj:
                continue

            valid_relevant = [
                f"doc-{did}" for did in relevant_doc_ids if did in corpus
            ]
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

    # Create a minimal storage_cases.jsonl (required by eval harness but
    # BEIR datasets don't have storage eval data -- generate a stub)
    storage_path = os.path.join(output_dir, "storage_cases.jsonl")
    with open(storage_path, "w") as f:
        stub = {
            "id": "stub-0001",
            "text": "This is a placeholder for storage evaluation.",
            "expectedMemoryType": "factual",
            "requiredSpans": ["placeholder"],
        }
        f.write(json.dumps(stub) + "\n")

    print(f"\nConverted {dataset} ({split} split) to {output_dir}")
    print(f"  Documents: {doc_count}")
    print(f"  Queries: {query_count}")
    print(f"  Relevant pairs: {sum(len(v) for v in qrels.values())}")
    print(f"  Files: {docs_path}")
    print(f"          {queries_path}")
    print(f"          {storage_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert BEIR dataset to Memory.swift eval format")
    parser.add_argument("--dataset", required=True, help="BEIR dataset name (e.g., scifact, nfcorpus)")
    parser.add_argument("--output-dir", required=True, help="Output directory for eval files")
    parser.add_argument("--split", default="test", help="Qrels split to use (default: test)")
    args = parser.parse_args()
    convert(args.dataset, args.output_dir, args.split)


if __name__ == "__main__":
    main()
