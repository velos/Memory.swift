---
name: memory-evals
description: Run and compare Memory.swift evaluation harness (`memory_eval`) profiles with correct cache behavior, repeatable commands, and metric summaries. Use when asked to rerun evals, benchmark retrieval/storage changes, compare profiles, diagnose inconsistent scores, or produce report paths from `Evals/runs`.
---

# Memory Evals

Run evals from the repository root with deterministic defaults for code-change validation.

## Quick Commands

- Single profile (deterministic):
```bash
swift run memory_eval run --profile <profile> --dataset-root ./Evals/general --no-cache --no-index-cache
```

- All profiles (deterministic):
```bash
swift run memory_eval run --dataset-root ./Evals/general --no-cache --no-index-cache
```

- Compare runs:
```bash
swift run memory_eval compare ./Evals/general/runs/<run-a>.json ./Evals/general/runs/<run-b>.json
```

- Compare with regression guard:
```bash
swift run memory_eval compare --baseline ./Evals/general/runs/<baseline>.json ./Evals/general/runs/<candidate>.json
```

## Datasets

There are multiple dataset roots:

- `./Evals` — original technical/software-engineering-focused dataset (500 recall queries)
- `./Evals/general` — diverse general-purpose dataset spanning cooking, travel, health, finance, hobbies, family, work, learning, etc. (250 recall queries)
- `./Evals/longmemeval` — conversational long-memory benchmark converted from LongMemEval-cleaned (typically 500 recall queries)

Use `./Evals/general` by default unless specifically asked for the technical dataset.

Each dataset root contains:
`storage_cases.jsonl`, `recall_documents.jsonl`, `recall_queries.jsonl`.

To regenerate the general dataset:
```bash
python3 scripts/generate_eval_data_minimax.py --domain-profile general --output-dir ./Evals/general
```

To generate LongMemEval dataset files:
```bash
python3 scripts/convert_longmemeval_to_eval.py --split oracle --output-dir ./Evals/longmemeval
```

## Workflow

1. Validate dataset files exist in the chosen dataset root.

2. Choose cache mode:
- Use `--no-cache --no-index-cache` for correctness when evaluating code changes.
- Use `--index-cache` only for fast reruns when index schema/behavior has not changed.
- If results look suspiciously better/worse than expected, rerun with `--no-index-cache`.

3. Run profile(s) with `swift run memory_eval run ...`.

4. Report these metrics at max `k` (normally `k=10`):
- `Storage type accuracy`
- `Storage macro F1`
- `Storage span coverage`
- `Recall Hit@k`
- `Recall MRR@k`
- `Recall nDCG@k`
- `Search latency` (p50, p95, mean)

5. Always include artifact paths:
- JSON report in `<dataset-root>/runs/*.json`
- Markdown summary in `<dataset-root>/runs/*.md`

## Profile Names

Core profiles:
- `baseline` — NLContextualEmbedding with mean pooling, lemmatized FTS, default config
- `oracle_ceiling` — upper-bound measurement for the current retrieval pipeline
- `coreml_leaf_ir` — CoreML LEAF-IR embedding model (21.8 MB, 384-dim, int8 quantized). Requires `Models/leaf-ir.mlpackage` at repo root.

NLContextualEmbedding variants:
- `pooling_mean` — mean pooling (current default)
- `pooling_weighted_mean` — weighted mean pooling
- `wide_candidates` — 1000 semantic + lexical candidates (vs default 500)
- `chunker_900` — larger chunk size (900 tokens)
- `normalized_bm25` — normalized BM25 scoring

Apple Intelligence profiles (require macOS 26+ / iOS 26+):
- `apple_tags` — Apple content tagging
- `apple_storage` — Apple memory type classifier
- `apple_recall` — Apple recall capabilities (expansion + reranking)
- `expansion_only` — query expansion only
- `expansion_rerank` — expansion + reranking
- `expansion_rerank_tag` — expansion + reranking + tagging
- `full_apple` — all Apple Intelligence features combined

## CoreML LEAF-IR Notes

The `coreml_leaf_ir` profile uses the MongoDB LEAF-IR model converted to CoreML:
- Model path: `Models/leaf-ir.mlpackage` (tracked via Git LFS)
- Conversion script: `scripts/convert_leaf_ir_coreml.py`
- Verification script: `scripts/verify_leaf_ir_coreml.py`
- The model is compiled from `.mlpackage` at runtime (first run may be slower)
- Embedding dimension: 384, max sequence length: 512

## Helper Script

Use `scripts/run_eval.sh` for repeatable commands:

```bash
# Deterministic single profile (general dataset)
DATASET_ROOT=./Evals/general ./.agents/skills/memory-evals/scripts/run_eval.sh profile baseline

# Deterministic single profile (technical dataset)
./.agents/skills/memory-evals/scripts/run_eval.sh profile baseline

# Deterministic all profiles
DATASET_ROOT=./Evals/general ./.agents/skills/memory-evals/scripts/run_eval.sh all

# Fast rerun with index cache
DATASET_ROOT=./Evals/general ./.agents/skills/memory-evals/scripts/run_eval.sh profile baseline --index-cache
```
