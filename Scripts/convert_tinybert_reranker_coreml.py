#!/usr/bin/env python3
"""Convert the default cross-encoder reranker to CoreML."""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTORESEARCH_ROOT = REPO_ROOT / "Autoresearch"
if str(AUTORESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTORESEARCH_ROOT))

from memory_autoresearch.checkpoints import checkpoint_config, load_pretrained_weights
from memory_autoresearch.config import MODEL_SPECS
from memory_autoresearch.export import artifact_size_mb, export_coreml_model
from memory_autoresearch.modeling import RerankerModel
from memory_autoresearch.tokenization import BertTokenizerAdapter

OUTPUT_DIR = REPO_ROOT / "Models"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "reranker-v1.mlpackage"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MiniLM cross-encoder reranker to CoreML.")
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Destination .mlpackage path. Defaults to Models/reranker-v1.mlpackage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path).resolve()
    spec = MODEL_SPECS["reranker"]
    print(f"Loading {spec.checkpoint}...")
    tokenizer = BertTokenizerAdapter(spec.checkpoint, max_sequence_length=spec.max_sequence_length)
    config = checkpoint_config(spec.checkpoint, num_labels=8)
    config.vocab_size = tokenizer.vocab_size
    model = RerankerModel(config)
    load_pretrained_weights(model, "reranker", spec.checkpoint)

    print("Converting to CoreML...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(output_path)
    export_coreml_model("reranker", model, config, output_path)

    print(f"\nSaved to: {output_path}")
    print(f"Total size: {artifact_size_mb(output_path):.1f} MB")
    print(f"Num labels: 1 (relevance score)")
    print(f"Max sequence length: {spec.max_sequence_length}")

    # Save tokenizer for verification script
    tok_path = output_path.parent / f"{output_path.stem}-tokenizer"
    tokenizer.tokenizer.save_pretrained(tok_path)
    print(f"Tokenizer saved to: {tok_path}")


if __name__ == "__main__":
    main()
