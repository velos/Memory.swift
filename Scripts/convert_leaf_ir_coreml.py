#!/usr/bin/env python3
"""Convert MongoDB LEAF-IR to CoreML (.mlpackage) for on-device inference."""

import sys
import numpy as np

print("Loading torch...", flush=True)
import torch
print("Loading transformers...", flush=True)
from transformers import AutoTokenizer, AutoModel
print("Loading coremltools...", flush=True)
import coremltools as ct

MODEL_ID = "MongoDB/mdbr-leaf-ir"
OUTPUT_PATH = "Models/leaf-ir.mlpackage"
MAX_SEQ_LEN = 512
EMBEDDING_DIM = 384

print(f"Downloading {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, torchscript=True)
model.eval()

print("Tracing model with torch.jit...", flush=True)
dummy_input = tokenizer(
    "This is a test sentence for tracing.",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_SEQ_LEN,
)
input_ids = dummy_input["input_ids"]
attention_mask = dummy_input["attention_mask"]
token_type_ids = dummy_input.get("token_type_ids", torch.zeros_like(input_ids))


class EmbeddingWrapper(torch.nn.Module):
    """Wraps the transformer to output mean-pooled, L2-normalized embeddings."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        token_embeddings = outputs[0]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(token_embeddings * mask_expanded, dim=1)
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized


wrapper = EmbeddingWrapper(model)
wrapper.eval()

traced = torch.jit.trace(wrapper, (input_ids, attention_mask, token_type_ids))

print("Converting to CoreML...", flush=True)
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, MAX_SEQ_LEN), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, MAX_SEQ_LEN), dtype=np.int32),
        ct.TensorType(name="token_type_ids", shape=(1, MAX_SEQ_LEN), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS15,
)

print(f"Quantizing weights to int8...", flush=True)
mlmodel = ct.compression_utils.affine_quantize_weights(mlmodel, mode="linear", dtype=np.int8)

import os
os.makedirs("Models", exist_ok=True)

print(f"Saving to {OUTPUT_PATH}...", flush=True)
mlmodel.save(OUTPUT_PATH)

total_size = 0
for dirpath, dirnames, filenames in os.walk(OUTPUT_PATH):
    for f in filenames:
        fp = os.path.join(dirpath, f)
        total_size += os.path.getsize(fp)

print(f"\nDone!")
print(f"Output: {OUTPUT_PATH}")
print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
print(f"Embedding dim: {EMBEDDING_DIM}")
print(f"Max sequence length: {MAX_SEQ_LEN}")
