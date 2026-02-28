#!/usr/bin/env python3
"""Verify converted TinyBERT reranker CoreML model against PyTorch original."""

import os
import time
import numpy as np
import torch
import coremltools as ct
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COREML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "Models", "minilm-reranker.mlpackage"
)
MAX_SEQ_LEN = 512

TEST_PAIRS = [
    ("What is Python?", "Python is a popular programming language created by Guido van Rossum."),
    ("What is Python?", "The python is a large nonvenomous snake found in Asia and Africa."),
    ("What is Python?", "The weather in Paris is mild in spring."),
    ("best recipe for chocolate cake", "Preheat oven to 350F. Mix cocoa powder, flour, sugar, and eggs. Bake for 30 minutes."),
    ("best recipe for chocolate cake", "The history of chocolate dates back to ancient Mesoamerica."),
    ("how to fix a leaking faucet", "Turn off the water supply, remove the handle, replace the washer and O-ring, reassemble."),
    ("how to fix a leaking faucet", "Quantum mechanics describes the behavior of particles at atomic scales."),
]


def pytorch_score(model, tokenizer, query, document):
    inputs = tokenizer(
        query, document,
        return_tensors="pt", padding="max_length",
        max_length=MAX_SEQ_LEN, truncation=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    return logits.squeeze().item()


def coreml_score(coreml_model, tokenizer, query, document):
    inputs = tokenizer(
        query, document,
        return_tensors="np", padding="max_length",
        max_length=MAX_SEQ_LEN, truncation=True,
    )
    prediction = coreml_model.predict({
        "input_ids": inputs["input_ids"].astype(np.int32),
        "attention_mask": inputs["attention_mask"].astype(np.int32),
        "token_type_ids": inputs["token_type_ids"].astype(np.int32),
    })
    score = prediction["relevance_score"]
    return float(np.squeeze(score))


def main():
    print(f"Loading PyTorch model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pt_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    pt_model.eval()

    print(f"Loading CoreML model: {COREML_PATH}")
    cm_model = ct.models.MLModel(COREML_PATH)

    print("\n--- Score Comparison ---")
    print(f"{'Query':<35} {'Document':<55} {'PyTorch':>8} {'CoreML':>8} {'Δ':>8}")
    print("-" * 120)

    for query, doc in TEST_PAIRS:
        pt = pytorch_score(pt_model, tokenizer, query, doc)
        cm = coreml_score(cm_model, tokenizer, query, doc)
        delta = abs(pt - cm)
        q_short = query[:33] + ".." if len(query) > 35 else query
        d_short = doc[:53] + ".." if len(doc) > 55 else doc
        print(f"{q_short:<35} {d_short:<55} {pt:>8.4f} {cm:>8.4f} {delta:>8.4f}")

    print("\n--- Ranking Preservation Test ---")
    query = "What is Python?"
    docs = [
        "Python is a popular programming language created by Guido van Rossum.",
        "The python is a large nonvenomous snake found in Asia and Africa.",
        "The weather in Paris is mild in spring.",
    ]

    pt_scores = [(d, pytorch_score(pt_model, tokenizer, query, d)) for d in docs]
    cm_scores = [(d, coreml_score(cm_model, tokenizer, query, d)) for d in docs]

    pt_ranked = sorted(pt_scores, key=lambda x: x[1], reverse=True)
    cm_ranked = sorted(cm_scores, key=lambda x: x[1], reverse=True)

    print(f"\nQuery: '{query}'")
    print("\nPyTorch ranking:")
    for i, (d, s) in enumerate(pt_ranked):
        print(f"  {i+1}. [{s:>8.4f}] {d[:80]}")
    print("\nCoreML ranking:")
    for i, (d, s) in enumerate(cm_ranked):
        print(f"  {i+1}. [{s:>8.4f}] {d[:80]}")

    pt_order = [d for d, _ in pt_ranked]
    cm_order = [d for d, _ in cm_ranked]
    print(f"\nRanking preserved: {'YES' if pt_order == cm_order else 'NO'}")

    print("\n--- Latency Benchmark ---")
    n = 20
    start = time.perf_counter()
    for _ in range(n):
        coreml_score(cm_model, tokenizer, "test query", "test document content")
    elapsed = time.perf_counter() - start
    print(f"{n} scores in {elapsed:.3f}s = {elapsed/n*1000:.1f}ms per pair")


if __name__ == "__main__":
    main()
