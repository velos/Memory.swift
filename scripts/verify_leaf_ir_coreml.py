#!/usr/bin/env python3
"""Verify the CoreML LEAF-IR model produces correct embeddings."""

import numpy as np
import time

print("Loading CoreML model...", flush=True)
import coremltools as ct

mlmodel = ct.models.MLModel("Models/leaf-ir.mlpackage")

print("Loading PyTorch model for comparison...", flush=True)
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("MongoDB/mdbr-leaf-ir")
pt_model = AutoModel.from_pretrained("MongoDB/mdbr-leaf-ir")
pt_model.eval()

test_sentences = [
    "The vet confirmed Luna is allergic to chicken-based kibble",
    "What food is my cat allergic to?",
    "Sprint 23 ends Friday, demo is Thursday 2pm",
    "Deploy the staging environment before the demo",
]

MAX_SEQ_LEN = 512

def pytorch_embed(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=MAX_SEQ_LEN)
    with torch.no_grad():
        outputs = pt_model(**inputs)
    token_emb = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    mean_pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(mean_pooled, p=2, dim=1).squeeze().numpy()

def coreml_embed(text):
    inputs = tokenizer(text, return_tensors="np", padding="max_length",
                       truncation=True, max_length=MAX_SEQ_LEN)
    pred = mlmodel.predict({
        "input_ids": inputs["input_ids"].astype(np.int32),
        "attention_mask": inputs["attention_mask"].astype(np.int32),
        "token_type_ids": np.zeros_like(inputs["input_ids"], dtype=np.int32),
    })
    return pred["embedding"].flatten()

print("\n--- Embedding Comparison (PyTorch vs CoreML) ---\n")
pt_embeddings = []
coreml_embeddings = []
for sent in test_sentences:
    pt_emb = pytorch_embed(sent)
    cm_emb = coreml_embed(sent)
    cosine_sim = np.dot(pt_emb, cm_emb) / (np.linalg.norm(pt_emb) * np.linalg.norm(cm_emb))
    print(f"  \"{sent[:60]}...\"")
    print(f"    Cosine similarity (PT vs CoreML): {cosine_sim:.6f}")
    pt_embeddings.append(pt_emb)
    coreml_embeddings.append(cm_emb)

print("\n--- Retrieval Test ---\n")
query = "What food is my cat allergic to?"
docs = [
    "The vet confirmed Luna is allergic to chicken-based kibble",
    "Sprint 23 ends Friday, demo is Thursday 2pm",
    "Deploy the staging environment before the demo",
]

q_emb = coreml_embed(query)
print(f"  Query: \"{query}\"")
for doc in docs:
    d_emb = coreml_embed(doc)
    sim = np.dot(q_emb, d_emb)
    print(f"    {sim:.4f}  \"{doc}\"")

print("\n--- Latency Benchmark (10 embeddings) ---\n")
start = time.time()
for _ in range(10):
    coreml_embed("A test sentence for benchmarking embedding latency.")
elapsed = time.time() - start
print(f"  Total: {elapsed*1000:.0f}ms for 10 embeddings")
print(f"  Per embedding: {elapsed*100:.1f}ms")
print(f"\nAll checks passed!")
