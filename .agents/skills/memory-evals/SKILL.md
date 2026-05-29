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

- Retrieval-only diagnostics with context budget accounting:
```bash
swift run memory_eval retrieval-diagnostics --profile coreml_default --dataset-root ./Evals/longmemeval_v2 --candidate-pool-depth 40 --context-token-budget 4096 --per-document-token-budget 384 --context-packing-order rank --no-cache --no-index-cache
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

- Validate dataset and baseline hygiene without running evals:
```bash
swift run memory_eval validate-datasets
swift run memory_eval validate-datasets --strict ./Evals/general_v2 ./Evals/longmemeval_v2
```

- Check runtime/test code for benchmark-derived answer leakage:
```bash
python3 Scripts/check_benchmark_leakage.py
```

- Classify retrieval-diagnostics misses and optional pre/post regressions:
```bash
python3 Scripts/analyze_retrieval_diagnostics.py \
  ./Evals/longmemeval_v2/runs/<candidate>.retrieval-diagnostics.json \
  --baseline-json ./Evals/longmemeval_v2/runs/<baseline>.retrieval-diagnostics.json \
  --dataset-root ./Evals/longmemeval_v2
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

For broad cleanup or before promoting a new suite, run:
```bash
swift run memory_eval validate-datasets
```
Use `--strict` when sidecar warnings such as `.DS_Store`, `review_queue.jsonl`,
`repair_report.*`, or model-draft scenario files should fail the check instead
of only being reported. Keep bulky external benchmark artifacts and local
provider wrappers in `references/agent-memory-benchmark` out of the Memory.swift
release gate; preserve only benchmark summaries or generic diagnostics in this
repo.

2. Choose cache mode:
- Use `--no-cache --no-index-cache` for correctness when evaluating code changes.
- Use `--index-cache` only for fast reruns when index schema/behavior has not changed.
- If results look suspiciously better/worse than expected, rerun with `--no-index-cache`.
- CoreML runs may print `E5RT encountered an STL exception... ANECompiler`; treat it as a warning if the eval completes and writes reports.

3. Run profile(s) with `swift run memory_eval run ...`.

Use `memory_eval retrieval-diagnostics` when isolating retrieval quality from
answer-LLM behavior. It writes AMB-style retrieval-only JSON and Markdown with
Hit/Recall/MRR/nDCG by K, score-sorted packed sidecar metrics, candidate-pool
Hit/Recall, candidate-only miss rate, candidate-generation miss rate, average
packed context tokens, empty retrieval rate, query latency, stage timings,
candidate counts, and per-query retrieved IDs.
Use `--per-document-token-budget` to test whether shorter packed snippets make
more relevant candidates survive fixed 4k/8k context budgets.
An explicit per-document budget is honored. When a total context budget is set
and per-document budget is `0`, evidence-dense temporal or multi-evidence
queries get an adaptive per-document cap so more support documents can fit.
Use `--context-packing-order score` only as an explicit experiment; `rank` is
the default and should remain the release-gate setting unless score-order wins
on both recall and rank quality.
The `memory serve` benchmark bridge accepts equivalent JSON search params:
`contextTokenBudget`, `perDocumentTokenBudget`, and `contextPackingOrder`.

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

## Benchmark Quality Guardrails

Focused benchmarks are debugging tools, not product specs. Runtime changes should
generalize across memory-like workloads before they are kept:

- Prefer query-shape, score, metadata, date, entity/topic, and context-budget
  signals over dataset names, benchmark IDs, exact answers, or named people from
  a benchmark.
- Do not add domain aliases from benchmark misses to runtime query expansion or
  ranking unless there is a product-owned vocabulary source outside the
  benchmark. Miss-derived aliases should stay in diagnostics, not shipped
  retrieval behavior.
- Do not add proper nouns, exact answer phrases, or benchmark-specific session
  facts to `MemoryIndex` ranking logic.
- A focused-slice win must be checked against at least one broader suite before
  treating it as an improvement. For retrieval changes, run the relevant focused
  slice plus `general_v2` or `longmemeval_v2`; for external AMB discoveries,
  rerun a broader AMB sample when quota/runtime permits.
- If a change improves Hit@10 by reducing Recall@10, nDCG, or average context
  diversity on broader suites, treat it as suspect until there is a clear agent
  use-case reason.
- Run `python3 Scripts/check_benchmark_leakage.py` before PR review for
  retrieval changes touched by external benchmark analysis. The guard scans
  runtime/library tests for exact answer phrases, benchmark IDs, and rescue
  wording; extend the denylist when a new benchmark exploration exposes a risky
  answer surface.
- Use `Scripts/analyze_retrieval_diagnostics.py` after focused cleanup or
  benchmark-derived changes. Candidate-generation misses point to query analysis
  or indexing gaps; candidate-only misses point to ranking, fusion, or context
  packing gaps. Prefer fixes that improve those generic surfaces without adding
  named answer phrases.

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
