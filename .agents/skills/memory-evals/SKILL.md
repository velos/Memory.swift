---
name: memory-evals
description: Run and compare Memory.swift evaluation harness (`memory_eval`) profiles with correct cache behavior, repeatable commands, and metric summaries. Use when asked to rerun evals, benchmark retrieval/storage changes, compare profiles, diagnose inconsistent scores, or produce report paths from `Evals/runs`.
---

# Memory Evals

Run evals from the repository root with deterministic defaults for code-change validation.

## Quick Commands

- Single profile (deterministic):
```bash
swift run memory_eval run --profile <profile> --dataset-root ./Evals/general_v2 --no-cache --no-index-cache
```

- All profiles (deterministic):
```bash
swift run memory_eval run --dataset-root ./Evals/general_v2 --no-cache --no-index-cache
```

- Compare runs:
```bash
swift run memory_eval compare ./Evals/general_v2/runs/<run-a>.json ./Evals/general_v2/runs/<run-b>.json
```

- Compare with regression guard:
```bash
swift run memory_eval compare --baseline ./Evals/general_v2/runs/<baseline>.json ./Evals/general_v2/runs/<candidate>.json
```

## Datasets

There are multiple dataset roots:

- `./Evals/general_v2` — broad retrieval gate for the shipped hybrid path
- `./Evals/longmemeval_v2` — long-horizon conversational recall benchmark
- `./Evals/memory_schema_gold_v2` — canonical write-path benchmark
- `./Evals/query_expansion_gold_v1` — expansion pressure benchmark

Use `./Evals/general_v2` by default unless the user asks for a different benchmark.

Each dataset root contains:
`storage_cases.jsonl`, `recall_documents.jsonl`, `recall_queries.jsonl`.

To regenerate an exploratory general dataset locally:
```bash
python3 Scripts/generate_eval_data_minimax.py --dataset-root ./Explorations/Evals/general --dataset-mode general
```

To generate exploratory LongMemEval dataset files locally:
```bash
python3 Scripts/convert_longmemeval_to_eval.py --split oracle --output-dir ./Explorations/Evals/longmemeval_v2
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
- `nl_baseline`
- `coreml_default`
- `oracle_ceiling`
- `apple_augmented` (experimental, requires Apple Intelligence availability)

## CoreML LEAF-IR Notes

The `coreml_leaf_ir` profile uses the MongoDB LEAF-IR model converted to CoreML:
- Model path: `Models/leaf-ir.mlpackage` (tracked via Git LFS)
- Conversion script: `Scripts/convert_leaf_ir_coreml.py`
- Verification script: `Scripts/verify_leaf_ir_coreml.py`
- The model is compiled from `.mlpackage` at runtime (first run may be slower)
- Embedding dimension: 384, max sequence length: 512

## Helper Script

Use `scripts/run_eval.sh` for repeatable commands:

```bash
# Deterministic single profile (broad retrieval gate)
DATASET_ROOT=./Evals/general_v2 ./.agents/skills/memory-evals/scripts/run_eval.sh profile nl_baseline

# Deterministic single profile (schema benchmark)
DATASET_ROOT=./Evals/memory_schema_gold_v2 ./.agents/skills/memory-evals/scripts/run_eval.sh profile coreml_default

# Deterministic all profiles
DATASET_ROOT=./Evals/general_v2 ./.agents/skills/memory-evals/scripts/run_eval.sh all

# Fast rerun with index cache
DATASET_ROOT=./Evals/general_v2 ./.agents/skills/memory-evals/scripts/run_eval.sh profile nl_baseline --index-cache
```
