#!/usr/bin/env python3
"""
Convert MiniLM-L-6-v2 cross-encoder to CoreML with multiple quantization levels,
then validate each against PyTorch reference on SciFact queries.

Produces:
  Models/minilm-l6-reranker-fp16.mlpackage   (~45 MB)
  Models/minilm-l6-reranker-int8.mlpackage   (~23 MB) -- same as existing
  Models/minilm-l6-reranker-int4.mlpackage   (~12 MB)

Usage:
    python3 scripts/convert_minilm_l6_variants.py
"""

import json
import os
import time

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SEQ_LEN = 512
ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(ROOT, "Models")


class RerankerWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs[0]
        return logits.squeeze(-1)


def convert_to_coreml(wrapper):
    print("Tracing model...")
    dummy_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int32)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.int32)
    dummy_types = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask, dummy_types))
        test_out = traced(dummy_ids, dummy_mask, dummy_types)
        print(f"  Traced output: {test_out.item():.4f}")

    print("Converting to CoreML (fp16)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, MAX_SEQ_LEN), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, MAX_SEQ_LEN), dtype=np.int32),
            ct.TensorType(name="token_type_ids", shape=(1, MAX_SEQ_LEN), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="relevance_score", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    return mlmodel


def model_size_mb(path):
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(path)
        for f in fns
    )
    return total / 1024 / 1024


def pytorch_scores(pt_model, tokenizer, pairs):
    scores = []
    for query, doc in pairs:
        inputs = tokenizer(
            query, doc, return_tensors="pt",
            padding="max_length", max_length=MAX_SEQ_LEN, truncation=True,
        )
        with torch.no_grad():
            out = pt_model(**inputs)
            logits = out.logits if hasattr(out, "logits") else out[0]
        scores.append(logits.squeeze().item())
    return scores


def coreml_scores(cm_model, tokenizer, pairs):
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


def load_scifact_pairs(n=50):
    """Load first n SciFact query-document pairs for validation."""
    docs = {}
    docs_path = os.path.join(ROOT, "Evals/scifact/recall_documents.jsonl")
    with open(docs_path) as f:
        for line in f:
            d = json.loads(line)
            docs[d["id"]] = d["text"][:500]  # truncate for speed

    pairs = []
    queries_path = os.path.join(ROOT, "Evals/scifact/recall_queries.jsonl")
    with open(queries_path) as f:
        for line in f:
            q = json.loads(line)
            for did in q.get("relevant_document_ids", [])[:1]:
                if did in docs:
                    pairs.append((q["query"], docs[did]))
            if len(pairs) >= n:
                break

    # Add some non-relevant pairs for contrast
    all_doc_ids = list(docs.keys())
    for i in range(min(n // 2, len(pairs))):
        neg_id = all_doc_ids[(i * 7) % len(all_doc_ids)]
        pairs.append((pairs[i][0], docs[neg_id]))

    return pairs[:n]


def validate_variant(name, cm_model, tokenizer, pt_scores, pairs):
    cm = coreml_scores(cm_model, tokenizer, pairs)

    diffs = [abs(p - c) for p, c in zip(pt_scores, cm)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)

    pt_ranking = sorted(range(len(pt_scores)), key=lambda i: pt_scores[i], reverse=True)
    cm_ranking = sorted(range(len(cm)), key=lambda i: cm[i], reverse=True)

    # Kendall tau-b rank correlation (simple version)
    concordant = discordant = 0
    n = len(pt_ranking)
    for i in range(n):
        for j in range(i + 1, n):
            pt_cmp = (pt_scores[i] > pt_scores[j]) - (pt_scores[i] < pt_scores[j])
            cm_cmp = (cm[i] > cm[j]) - (cm[i] < cm[j])
            if pt_cmp * cm_cmp > 0:
                concordant += 1
            elif pt_cmp * cm_cmp < 0:
                discordant += 1
    tau = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 1.0

    # Top-10 overlap
    pt_top10 = set(pt_ranking[:10])
    cm_top10 = set(cm_ranking[:10])
    top10_overlap = len(pt_top10 & cm_top10) / 10

    # Latency
    t0 = time.perf_counter()
    for _ in range(20):
        coreml_scores(cm_model, tokenizer, [pairs[0]])
    latency = (time.perf_counter() - t0) / 20 * 1000

    print(f"  {name}:")
    print(f"    Mean |Δ|: {mean_diff:.5f}  Max |Δ|: {max_diff:.5f}")
    print(f"    Kendall τ: {tau:.4f}  Top-10 overlap: {top10_overlap:.0%}")
    print(f"    Latency: {latency:.1f}ms/pair")

    return {"mean_diff": mean_diff, "max_diff": max_diff, "tau": tau, "top10": top10_overlap, "latency": latency}


def main():
    print(f"Loading PyTorch model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pt_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torchscript=True)
    pt_model.eval()

    wrapper = RerankerWrapper(pt_model)
    wrapper.eval()
    fp16_model = convert_to_coreml(wrapper)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save fp16 (default CoreML precision for weights)
    fp16_path = os.path.join(OUTPUT_DIR, "minilm-l6-reranker-fp16.mlpackage")
    fp16_model.save(fp16_path)
    print(f"\nfp16 saved: {model_size_mb(fp16_path):.1f} MB")

    # Int8 quantization
    print("Applying int8 quantization...")
    int8_model = ct.compression_utils.affine_quantize_weights(fp16_model, mode="linear", dtype=np.int8)
    int8_path = os.path.join(OUTPUT_DIR, "minilm-l6-reranker-int8.mlpackage")
    int8_model.save(int8_path)
    print(f"int8 saved: {model_size_mb(int8_path):.1f} MB")

    # Int4 quantization
    print("Applying int4 quantization...")
    int4_model = ct.compression_utils.affine_quantize_weights(fp16_model, mode="linear", dtype="int4")
    int4_path = os.path.join(OUTPUT_DIR, "minilm-l6-reranker-int4.mlpackage")
    int4_model.save(int4_path)
    print(f"int4 saved: {model_size_mb(int4_path):.1f} MB")

    # Palettization (4-bit, alternative quantization approach)
    print("Applying 4-bit palettization...")
    try:
        pal4_model = ct.compression_utils.palettize_weights(fp16_model, nbits=4)
        pal4_path = os.path.join(OUTPUT_DIR, "minilm-l6-reranker-pal4.mlpackage")
        pal4_model.save(pal4_path)
        print(f"pal4 saved: {model_size_mb(pal4_path):.1f} MB")
        has_pal4 = True
    except Exception as e:
        print(f"  Palettization failed: {e}")
        has_pal4 = False

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION: Comparing CoreML variants against PyTorch reference")
    print("=" * 70)

    print("\nLoading SciFact validation pairs...")
    pairs = load_scifact_pairs(n=60)
    print(f"  {len(pairs)} query-document pairs")

    # Reload models for prediction (coremltools models from ct.convert may not predict)
    pt_ref = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    pt_ref.eval()

    print("\nComputing PyTorch reference scores...")
    pt_scores = pytorch_scores(pt_ref, tokenizer, pairs)

    print("\nLoading and validating CoreML variants...\n")

    variants = [
        ("fp16", fp16_path),
        ("int8", int8_path),
        ("int4", int4_path),
    ]
    if has_pal4:
        variants.append(("pal4", pal4_path))

    results = {}
    for name, path in variants:
        cm = ct.models.MLModel(path)
        size = model_size_mb(path)
        print(f"  [{name}] size: {size:.1f} MB")
        results[name] = validate_variant(name, cm, tokenizer, pt_scores, pairs)
        results[name]["size_mb"] = size

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<10} {'Size':>8} {'Mean |Δ|':>10} {'Kendall τ':>10} {'Top-10':>8} {'Latency':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<10} {r['size_mb']:>7.1f}M {r['mean_diff']:>10.5f} {r['tau']:>10.4f} {r['top10']:>7.0%} {r['latency']:>9.1f}ms")


if __name__ == "__main__":
    main()
