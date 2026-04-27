---
name: memory-evals
description: Run and compare Memory.swift evaluation harness (`memory_eval`) profiles with correct cache behavior, repeatable commands, and metric summaries. Use when asked to rerun evals, benchmark retrieval/storage changes, compare profiles, diagnose inconsistent scores, or produce report paths from `Evals/runs`.
---

# Memory Evals

Run evals from the repository root with deterministic defaults for code-change validation. Prefer `coreml_default` for shipped-path accuracy work unless the user asks for another profile.

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

- Gate required release artifacts:
```bash
swift run memory_eval gate --baseline ./Evals/baselines/current.json <memory-schema-run.json> <agent-memory-run.json> <general-run.json> <longmemeval-run.json> <query-expansion-run.json>
```

- Diagnose LongMemEval misses across retrieval branches:
```bash
swift run memory_eval diagnose-longmemeval \
  --profile coreml_default \
  --dataset-root ./Evals/longmemeval_v2 \
  --source-run ./Evals/longmemeval_v2/runs/<run>.json \
  --scope misses \
  --wide-limit 100
```

## Datasets

There are multiple dataset roots:

- `./Evals/general_v2` — broad retrieval gate for the shipped hybrid path
- `./Evals/longmemeval_v2` — long-horizon conversational recall benchmark
- `./Evals/memory_schema_gold_v2` — canonical write-path benchmark
- `./Evals/storage_heldout_v1` — exploratory held-out storage robustness suite with unfamiliar projects/tools/people plus no-write/lifecycle scenarios
- `./Evals/agent_memory_gold_v1` — agent memory behavior benchmark: no-write, extraction, update/supersede/resolve, active-state, and recall checks
- `./Evals/query_expansion_gold_v1` — expansion pressure benchmark
- `./Evals/longmemeval_rescue_v1` — focused LongMemEval candidate-generation rescue slice mined from branch diagnostics
- `./Evals/longmemeval_ranking_v1` — focused LongMemEval ranking/pool-depth slice mined from branch diagnostics
- `./Evals/longmemeval_multievidence_v1` — focused LongMemEval multi-evidence preservation slice mined from branch diagnostics

Use `./Evals/general_v2` by default unless the user asks for a different benchmark.

Dataset roots may contain one or more suite files:
- Storage: `storage_cases.jsonl`
- Recall: `recall_documents.jsonl`, `recall_queries.jsonl`
- Query expansion: `query_expansion_cases.jsonl` or `cases.jsonl`, usually plus `recall_documents.jsonl`
- Agent memory: `scenarios.jsonl`

To regenerate an exploratory general dataset locally:
```bash
python3 Scripts/generate_eval_data_minimax.py --dataset-root ./Explorations/Evals/general --dataset-mode general
```

To generate exploratory LongMemEval dataset files locally:
```bash
python3 Scripts/convert_longmemeval_to_eval.py --split oracle --output-dir ./Explorations/Evals/longmemeval_v2
```

To regenerate agent-memory scenarios:
```bash
python3 Scripts/generate_agent_memory_scenarios.py --dataset-root ./Evals/agent_memory_gold_v1 --overwrite
```

Use `--backend minimax` only when the `.env` Anthropic-compatible endpoint should draft extra candidate scenarios; review generated scenarios before treating them as gate labels.

To mine query-expansion rescue candidates from existing recall runs:
```bash
python3 Scripts/build_query_expansion_rescue.py --source ./Evals/longmemeval_v2:./Evals/longmemeval_v2/runs/<run>.json --overwrite
```

To build the focused LongMemEval rescue slice from branch diagnostics:
```bash
python3 Scripts/build_longmemeval_rescue.py --diagnostic ./Evals/longmemeval_v2/runs/<run>.wide200.branch-diagnostics.json --overwrite
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/longmemeval_rescue_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/longmemeval_rescue.json ./Evals/longmemeval_rescue_v1/runs/<run>.json
```

To build focused LongMemEval slices for ranking/pool-depth and multi-evidence preservation:
```bash
python3 Scripts/build_longmemeval_rescue.py \
  --diagnostic ./Evals/longmemeval_v2/runs/<run>.wide200.branch-diagnostics.json \
  --output-root ./Evals/longmemeval_ranking_v1 \
  --classification ranking_or_pool_depth \
  --max-cases 64 \
  --max-candidates-per-case 140 \
  --per-branch-limit 80 \
  --overwrite

python3 Scripts/build_longmemeval_rescue.py \
  --diagnostic ./Evals/longmemeval_v2/runs/<run>.wide200.branch-diagnostics.json \
  --output-root ./Evals/longmemeval_multievidence_v1 \
  --classification multi_evidence_preservation \
  --max-cases 64 \
  --max-candidates-per-case 140 \
  --per-branch-limit 80 \
  --overwrite

swift run memory_eval run --profile coreml_default --dataset-root ./Evals/longmemeval_ranking_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/longmemeval_ranking.json ./Evals/longmemeval_ranking_v1/runs/<run>.json

swift run memory_eval run --profile coreml_default --dataset-root ./Evals/longmemeval_multievidence_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/longmemeval_multievidence.json ./Evals/longmemeval_multievidence_v1/runs/<run>.json
```

## Workflow

1. Validate dataset files exist in the chosen dataset root.

2. Choose cache mode:
- Use `--no-cache --no-index-cache` for correctness when evaluating code changes.
- Use `--index-cache` only for fast reruns when index schema/behavior has not changed.
- If results look suspiciously better/worse than expected, rerun with `--no-index-cache`.
- CoreML runs may print `E5RT encountered an STL exception... ANECompiler`; treat it as a warning if the eval completes and writes reports.

3. Run profile(s) with `swift run memory_eval run ...`.

4. For LongMemEval regressions or miss analysis:
```bash
python3 Scripts/analyze_longmemeval_misses.py ./Evals/longmemeval_v2/runs/<run>.json --dataset-root ./Evals/longmemeval_v2
swift run memory_eval diagnose-longmemeval --profile coreml_default --dataset-root ./Evals/longmemeval_v2 --source-run ./Evals/longmemeval_v2/runs/<run>.json --scope misses --wide-limit 100
```

The diagnostic command writes JSON and Markdown beside the source run by default. It labels cases as `fixed_by_current_code`, `ranking_or_pool_depth`, `candidate_generation`, `expansion_regression`, `fusion_filtering`, `multi_evidence_preservation`, or `partial_multi_evidence`, and includes branch ranks for `current_wide`, `no_expansion_wide`, `lexical_wide`, and `semantic_wide`.

Focused slice gates are useful before full LongMemEval reruns:
- `longmemeval_ranking.json` locks the current safe ranking/pool-depth wins, including q-2ce6a0f2 and q-gpt4_ab202e7f.
- `longmemeval_multievidence.json` locks multi-evidence Hit@10 at 100% and support-document Recall@10 at or above 55%.
- Treat dense unrepresented-group promotion as experimental until it improves focused ranking without regressing full LongMemEval Recall@10.

5. Report these metrics at max `k` (normally `k=10`):
- `Storage type accuracy`
- `Storage macro F1`
- `Storage span coverage`
- `Recall Hit@k`
- `Recall Recall@k`
- `Recall MRR@k`
- `Recall nDCG@k`
- `Search latency` (p50, p95, mean)
- For query expansion: lexical/semantic coverage recall, HyDE anchor recall, retrieval Hit@k, retrieval lift, expanded MRR, and MRR delta.
- For agent memory: false-write rate, expected-write recall, active-state accuracy, update-behavior accuracy, recall hit rate, and recall MRR.

6. Always include artifact paths:
- JSON report in `<dataset-root>/runs/*.json`
- Markdown summary in `<dataset-root>/runs/*.md`

7. For release gating, run the five required `coreml_default` suites and pass the fresh JSON artifacts to `memory_eval gate`:
```bash
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/memory_schema_gold_v2 --no-cache --no-index-cache
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/agent_memory_gold_v1 --no-cache --no-index-cache
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/general_v2 --no-cache --no-index-cache
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/longmemeval_v2 --no-cache --no-index-cache
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/query_expansion_gold_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/current.json <five-json-reports>
```

## Profile Names

Core profiles:
- `nl_baseline`
- `coreml_default`
- `oracle_ceiling`
- `apple_augmented` (experimental, requires Apple Intelligence availability)

Additional experimental profile names accepted by the CLI:
- `coreml_leaf_ir`
- `coreml_rerank`

## CoreML default path

The `coreml_default` eval profile exercises the shipped CoreML embedding stack. The CLI resolves `Models/embedding-v1.mlpackage` by default (same layout as `swift run memory` / `memory_eval`).

The repo also contains `Models/leaf-ir.mlpackage` and conversion helpers for experiments:
- Conversion script: `Scripts/convert_leaf_ir_coreml.py`
- Verification script: `Scripts/verify_leaf_ir_coreml.py`
- Models compile from `.mlpackage` at runtime (first run may be slower)

## Helper Script

Use `.agents/skills/memory-evals/scripts/run_eval.sh` for repeatable commands:

```bash
# Deterministic single profile (broad retrieval gate)
DATASET_ROOT=./Evals/general_v2 ./.agents/skills/memory-evals/scripts/run_eval.sh profile coreml_default

# Deterministic single profile (schema benchmark)
DATASET_ROOT=./Evals/memory_schema_gold_v2 ./.agents/skills/memory-evals/scripts/run_eval.sh profile coreml_default

# Deterministic single profile (held-out storage robustness)
DATASET_ROOT=./Evals/storage_heldout_v1 ./.agents/skills/memory-evals/scripts/run_eval.sh profile coreml_default

# Deterministic all profiles
DATASET_ROOT=./Evals/general_v2 ./.agents/skills/memory-evals/scripts/run_eval.sh all

# Fast rerun with index cache
DATASET_ROOT=./Evals/general_v2 ./.agents/skills/memory-evals/scripts/run_eval.sh profile nl_baseline --index-cache

# Gate existing artifacts
./.agents/skills/memory-evals/scripts/run_eval.sh gate ./Evals/baselines/current.json <five-json-reports>

# Diagnose LongMemEval misses
./.agents/skills/memory-evals/scripts/run_eval.sh diagnose-longmemeval ./Evals/longmemeval_v2/runs/<run>.json --scope misses --wide-limit 100
```
