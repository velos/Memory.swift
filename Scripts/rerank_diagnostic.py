#!/usr/bin/env python3
"""
Rerank diagnostic: tests whether cross-encoder rerankers improve retrieval
quality when given a wider candidate pool (top-N instead of top-10).

Uses sentence-transformers to load LEAF-IR directly, embed all documents,
retrieve top-N candidates per query, then rerank with cross-encoders.

Usage:
    pip install sentence-transformers
    python3 scripts/rerank_diagnostic.py                        # SciFact, top-40
    python3 scripts/rerank_diagnostic.py --dataset nfcorpus     # NFCorpus
    python3 scripts/rerank_diagnostic.py --top-k 100            # custom pool
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

EMBED_MODEL = "MongoDB/mdbr-leaf-ir"
RERANK_MODELS = [
    "cross-encoder/ms-marco-TinyBERT-L-2-v2",      #  4.4M params, 2-layer
    "cross-encoder/ms-marco-MiniLM-L-6-v2",         # 22.7M params, 6-layer (base of 4-bit NF4 quant)
    "mixedbread-ai/mxbai-rerank-xsmall-v1",         # 70.8M params, DeBERTa-v2
    "cross-encoder/ms-marco-MiniLM-L-12-v2",        # 33.4M params, 12-layer
]


def load_eval_dataset(dataset_dir: Path):
    """Load documents and queries from our eval format (convert_beir_to_eval.py output)."""
    docs = {}
    with open(dataset_dir / "recall_documents.jsonl") as f:
        for line in f:
            d = json.loads(line)
            docs[d["id"]] = d["text"]

    queries = []
    with open(dataset_dir / "recall_queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries.append({
                "id": q["id"],
                "query": q["query"],
                "relevant": q.get("relevant_document_ids", []),
            })
    return docs, queries


def retrieve_top_k(doc_embeddings, query_emb, doc_ids, k):
    scores = np.dot(doc_embeddings, query_emb.T).flatten()
    top_indices = np.argsort(scores)[::-1][:k]
    return [(doc_ids[i], float(scores[i])) for i in top_indices]


def dcg(relevances, k):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances, k):
    idcg = dcg(sorted(relevances, reverse=True), k)
    return dcg(relevances, k) / idcg if idcg > 0 else 0.0


def compute_metrics(queries, k=10):
    hits = mrr_sum = ndcg_sum = recall_sum = 0
    for q in queries:
        relevant = set(q["relevant"])
        ranked = q["ranked"][:k]
        rels = [1.0 if d in relevant else 0.0 for d in ranked]
        if any(r > 0 for r in rels):
            hits += 1
        for i, d in enumerate(ranked):
            if d in relevant:
                mrr_sum += 1.0 / (i + 1)
                break
        ndcg_sum += ndcg_at_k(rels, k)
        if relevant:
            recall_sum += sum(1 for d in ranked if d in relevant) / len(relevant)
    n = len(queries)
    return {"Hit@10": hits/n, "MRR@10": mrr_sum/n, "nDCG@10": ndcg_sum/n, "Recall@10": recall_sum/n}


def rerank_with_model(model_name, queries_data, docs):
    from sentence_transformers import CrossEncoder
    print(f"\n  Loading {model_name}...")
    model = CrossEncoder(model_name)
    reranked = []
    total = len(queries_data)
    for i, q in enumerate(queries_data):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Reranking {i+1}/{total}...")
        doc_ids = q["ranked"]
        doc_texts = [docs.get(did, "") for did in doc_ids]
        if not doc_texts:
            reranked.append(q)
            continue
        results = model.rank(q["query"], doc_texts, return_documents=False)
        new_ranking = [doc_ids[r["corpus_id"]] for r in results]
        reranked.append({"query": q["query"], "relevant": q["relevant"], "ranked": new_ranking})
    return reranked


def print_section(label, metrics, baseline=None):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    for k, v in metrics.items():
        if baseline:
            diff = v - baseline[k]
            arrow = "↑" if diff > 0.0005 else ("↓" if diff < -0.0005 else "=")
            print(f"  {k}: {v:.4f}  ({arrow} {abs(diff):.4f})")
        else:
            print(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact", help="Dataset name (subdir of Evals/)")
    parser.add_argument("--top-k", type=int, default=40, help="Candidate pool size for retrieval")
    parser.add_argument("--eval-k", type=int, default=10, help="K for metric computation")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    dataset_dir = root / "Evals" / args.dataset

    if not dataset_dir.exists():
        print(f"Error: {dataset_dir} not found. Run convert_beir_to_eval.py first.")
        return

    print(f"Dataset: {args.dataset}, retrieval top-{args.top_k}, eval @{args.eval_k}")
    docs, queries = load_eval_dataset(dataset_dir)
    print(f"  {len(docs)} documents, {len(queries)} queries")

    from sentence_transformers import SentenceTransformer
    print(f"\nLoading embedding model: {EMBED_MODEL}...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    doc_ids = list(docs.keys())
    doc_texts = [docs[did] for did in doc_ids]

    print(f"Embedding {len(doc_ids)} documents...")
    t0 = time.time()
    doc_embeddings = embed_model.encode(
        doc_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nRetrieving top-{args.top_k} candidates per query...")
    t0 = time.time()
    queries_data = []
    for q in queries:
        query_emb = embed_model.encode([q["query"]], normalize_embeddings=True)
        results = retrieve_top_k(doc_embeddings, query_emb, doc_ids, args.top_k)
        queries_data.append({
            "query": q["query"],
            "relevant": q["relevant"],
            "ranked": [doc_id for doc_id, _ in results],
        })
    print(f"  Done in {time.time()-t0:.1f}s")

    has_rel_10 = sum(1 for q in queries_data if any(d in q["ranked"][:10] for d in q["relevant"]))
    has_rel_k = sum(1 for q in queries_data if any(d in q["ranked"] for d in q["relevant"]))
    print(f"  Relevant in top-10: {has_rel_10}/{len(queries)} ({100*has_rel_10/len(queries):.1f}%)")
    print(f"  Relevant in top-{args.top_k}: {has_rel_k}/{len(queries)} ({100*has_rel_k/len(queries):.1f}%)")

    baseline = compute_metrics(queries_data, args.eval_k)
    print_section(f"BASELINE: LEAF-IR top-{args.top_k} → eval @{args.eval_k}", baseline)

    for model_name in RERANK_MODELS:
        reranked = rerank_with_model(model_name, queries_data, docs)
        metrics = compute_metrics(reranked, args.eval_k)
        print_section(f"RERANKED: {model_name} (pool={args.top_k})", metrics, baseline)

        improved = degraded = unchanged = 0
        for orig, new in zip(queries_data, reranked):
            o = compute_metrics([orig], args.eval_k)["nDCG@10"]
            n = compute_metrics([new], args.eval_k)["nDCG@10"]
            if n > o + 0.001: improved += 1
            elif n < o - 0.001: degraded += 1
            else: unchanged += 1
        print(f"  Per-query: {improved} improved, {degraded} degraded, {unchanged} unchanged")

    oracle = []
    for q in queries_data:
        relevant = set(q["relevant"])
        ranked = list(q["ranked"])
        oracle.append({
            "query": q["query"], "relevant": q["relevant"],
            "ranked": sorted(ranked, key=lambda d: (d not in relevant, ranked.index(d)))
        })
    oracle_metrics = compute_metrics(oracle, args.eval_k)
    print_section(f"ORACLE CEILING (perfect reranking of top-{args.top_k})", oracle_metrics, baseline)


if __name__ == "__main__":
    main()
