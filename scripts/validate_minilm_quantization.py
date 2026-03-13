#!/usr/bin/env python3
"""
Validate MiniLM-L-6-v2 CoreML quantization quality.

Compares fp16, int8, int4, and pal4 CoreML variants against each other
using SciFact query-document pairs. Uses fp16 as the reference baseline.

Usage:
    python3 scripts/validate_minilm_quantization.py
"""

import json
import os
import sys
import time

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SEQ_LEN = 512
ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "Models")

VARIANTS = [
    ("fp16", "minilm-l6-reranker-fp16.mlpackage"),
    ("int8", "minilm-l6-reranker-int8.mlpackage"),
    ("int4", "minilm-l6-reranker-int4.mlpackage"),
    ("pal4", "minilm-l6-reranker-pal4.mlpackage"),
]


def model_size_mb(path):
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path) for f in fns
    ) / 1024 / 1024


def score_pairs(cm_model, tokenizer, pairs):
    scores = []
    for query, doc in pairs:
        inputs = tokenizer(
            query, doc, return_tensors="np",
            padding="max_length", max_length=MAX_SEQ_LEN, truncation=True,
        )
        pred = cm_model.predict({
            "input_ids": inputs["input_ids"].astype(np.int32),
            "attention_mask": inputs["attention_mask"].astype(np.int32),
            "token_type_ids": inputs["token_type_ids"].astype(np.int32),
        })
        scores.append(float(np.squeeze(pred["relevance_score"])))
    return scores


def load_scifact_pairs():
    """Load SciFact query-document pairs: relevant + non-relevant for contrast."""
    docs = {}
    with open(os.path.join(ROOT, "Evals/scifact/recall_documents.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            docs[d["id"]] = d["text"]

    pairs = []
    queries = []
    with open(os.path.join(ROOT, "Evals/scifact/recall_queries.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            queries.append(q)

    all_doc_ids = list(docs.keys())
    for i, q in enumerate(queries[:80]):
        for did in q.get("relevant_document_ids", [])[:1]:
            if did in docs:
                pairs.append((q["query"], docs[did][:800]))
        neg_id = all_doc_ids[(i * 13 + 7) % len(all_doc_ids)]
        pairs.append((q["query"], docs[neg_id][:800]))

    return pairs


def kendall_tau(ref_scores, test_scores):
    concordant = discordant = 0
    n = len(ref_scores)
    for i in range(n):
        for j in range(i + 1, n):
            ref_cmp = (ref_scores[i] > ref_scores[j]) - (ref_scores[i] < ref_scores[j])
            test_cmp = (test_scores[i] > test_scores[j]) - (test_scores[i] < test_scores[j])
            if ref_cmp * test_cmp > 0:
                concordant += 1
            elif ref_cmp * test_cmp < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 1.0


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading SciFact pairs...")
    pairs = load_scifact_pairs()
    print(f"  {len(pairs)} query-document pairs\n")

    # Score with all variants
    all_scores = {}
    all_latencies = {}
    all_sizes = {}

    for name, filename in VARIANTS:
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"  [{name}] SKIPPED - {path} not found")
            continue

        size = model_size_mb(path)
        all_sizes[name] = size
        print(f"  [{name}] Loading ({size:.1f} MB)...", end=" ", flush=True)

        cm = ct.models.MLModel(path)

        t0 = time.perf_counter()
        scores = score_pairs(cm, tokenizer, pairs)
        elapsed = time.perf_counter() - t0
        latency = elapsed / len(pairs) * 1000

        all_scores[name] = scores
        all_latencies[name] = latency
        print(f"scored {len(pairs)} pairs in {elapsed:.1f}s ({latency:.1f}ms/pair)")

        # Quick sanity: show a few scores
        print(f"         Sample scores: {scores[0]:.4f}, {scores[1]:.4f}, {scores[2]:.4f}")

    if "fp16" not in all_scores:
        print("ERROR: fp16 baseline not found")
        return

    ref = all_scores["fp16"]

    print("\n" + "=" * 70)
    print("QUALITY COMPARISON (vs fp16 reference)")
    print("=" * 70)

    for name in all_scores:
        if name == "fp16":
            continue
        test = all_scores[name]

        diffs = [abs(r - t) for r, t in zip(ref, test)]
        mean_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        pct_diffs = [abs(r - t) / max(abs(r), 0.01) * 100 for r, t in zip(ref, test)]
        mean_pct = sum(pct_diffs) / len(pct_diffs)

        tau = kendall_tau(ref, test)

        ref_ranking = sorted(range(len(ref)), key=lambda i: ref[i], reverse=True)
        test_ranking = sorted(range(len(test)), key=lambda i: test[i], reverse=True)
        ref_top10 = set(ref_ranking[:10])
        test_top10 = set(test_ranking[:10])
        top10_overlap = len(ref_top10 & test_top10)
        ref_top20 = set(ref_ranking[:20])
        test_top20 = set(test_ranking[:20])
        top20_overlap = len(ref_top20 & test_top20)

        # Count rank swaps (adjacent docs that swap order)
        swaps = 0
        for i in range(len(ref)):
            for j in range(i + 1, min(i + 5, len(ref))):
                if (ref[i] > ref[j]) != (test[i] > test[j]):
                    swaps += 1

        print(f"\n  {name} vs fp16:")
        print(f"    Size:          {all_sizes[name]:.1f} MB (vs {all_sizes['fp16']:.1f} MB fp16)")
        print(f"    Mean |Δ|:      {mean_diff:.5f} (mean {mean_pct:.2f}% relative)")
        print(f"    Max |Δ|:       {max_diff:.5f}")
        print(f"    Kendall τ:     {tau:.4f}")
        print(f"    Top-10 overlap: {top10_overlap}/10")
        print(f"    Top-20 overlap: {top20_overlap}/20")
        print(f"    Local swaps:   {swaps}")
        print(f"    Latency:       {all_latencies[name]:.1f}ms/pair (vs {all_latencies['fp16']:.1f}ms fp16)")

    # Show score distributions
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTIONS")
    print("=" * 70)
    for name, scores in all_scores.items():
        arr = np.array(scores)
        print(f"  {name}: min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f} std={arr.std():.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<8} {'Size':>7} {'Mean |Δ|':>10} {'τ':>8} {'Top-10':>8} {'Top-20':>8} {'ms/pair':>8}")
    print("-" * 60)
    print(f"{'fp16':<8} {all_sizes['fp16']:>6.1f}M {'(ref)':>10} {'1.0000':>8} {'10/10':>8} {'20/20':>8} {all_latencies['fp16']:>7.1f}")
    for name in all_scores:
        if name == "fp16":
            continue
        test = all_scores[name]
        diffs = [abs(r - t) for r, t in zip(ref, test)]
        tau = kendall_tau(ref, test)
        ref_ranking = sorted(range(len(ref)), key=lambda i: ref[i], reverse=True)
        test_ranking = sorted(range(len(test)), key=lambda i: test[i], reverse=True)
        t10 = len(set(ref_ranking[:10]) & set(test_ranking[:10]))
        t20 = len(set(ref_ranking[:20]) & set(test_ranking[:20]))
        print(f"{name:<8} {all_sizes[name]:>6.1f}M {sum(diffs)/len(diffs):>10.5f} {tau:>8.4f} {f'{t10}/10':>8} {f'{t20}/20':>8} {all_latencies[name]:>7.1f}")


if __name__ == "__main__":
    main()
