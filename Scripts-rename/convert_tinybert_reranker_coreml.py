#!/usr/bin/env python3
"""Convert a cross-encoder reranker to CoreML with int8 quantization."""

import os
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SEQ_LEN = 512
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "minilm-reranker.mlpackage")


class RerankerWrapper(nn.Module):
    """Wraps the cross-encoder to output a single relevance score."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs[0]  # tuple output with torchscript=True
        return logits.squeeze(-1)  # [batch, 1] -> [batch]


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, torchscript=True
    )
    base_model.eval()

    wrapper = RerankerWrapper(base_model)
    wrapper.eval()

    print("Tracing model...")
    dummy_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int32)
    dummy_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.int32)
    dummy_types = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask, dummy_types))

    # Verify trace output
    with torch.no_grad():
        test_out = traced(dummy_ids, dummy_mask, dummy_types)
        print(f"Traced output shape: {test_out.shape}, value: {test_out.item():.4f}")

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, MAX_SEQ_LEN), dtype=np.int32),
            ct.TensorType(
                name="attention_mask", shape=(1, MAX_SEQ_LEN), dtype=np.int32
            ),
            ct.TensorType(
                name="token_type_ids", shape=(1, MAX_SEQ_LEN), dtype=np.int32
            ),
        ],
        outputs=[ct.TensorType(name="relevance_score", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )

    print("Applying int8 quantization...")
    mlmodel = ct.compression_utils.affine_quantize_weights(
        mlmodel, mode="linear", dtype=np.int8
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mlmodel.save(OUTPUT_PATH)

    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(OUTPUT_PATH)
        for f in fns
    )
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Num labels: 1 (relevance score)")
    print(f"Max sequence length: {MAX_SEQ_LEN}")

    # Save tokenizer for verification script
    tok_path = os.path.join(OUTPUT_DIR, "minilm-reranker-tokenizer")
    tokenizer.save_pretrained(tok_path)
    print(f"Tokenizer saved to: {tok_path}")


if __name__ == "__main__":
    main()
