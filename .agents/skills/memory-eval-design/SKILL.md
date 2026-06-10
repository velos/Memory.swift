---
name: memory-eval-design
description: Design and implement Memory.swift evaluations as durable quality gates. Use when adding or changing eval datasets, metrics, scenario rows, baseline gates, or MemoryEvalCLI scoring for storage, recall, query expansion, and agent-memory behavior.
---

# Memory Eval Design

Use this skill when the task is to add, reshape, or promote Memory.swift
evaluations. Pair it with `memory-evals` when you need to run existing suites.

This skill is based on Apple Evaluations guidance: treat evaluations as a
repeatable specification, define measurable criteria before tuning behavior,
use datasets with golden, edge, adversarial, and known-failure coverage, prefer
code-based metrics for objective checks, reserve model-as-judge for subjective
quality, inspect both aggregate and per-sample results, and protect against
overfitting with fresh/held-out data.

## Start Here

Read these before editing:

- `AGENTS.md`
- `Evals/README.md`
- the target dataset `manifest.json`
- the relevant baseline in `Evals/baselines/`
- `Sources/MemoryEvalCLI/MemoryEvalCLI.swift` when adding fields, metrics, or report output

Keep generated or bulky exploratory data under `Explorations/Evals/` unless the
dataset README or baseline manifest makes it part of the repo contract.

## Turn Behavior Into Metrics

Before adding rows or code, write down the behavior as measurable criteria:

- **Subject under test**: storage extraction, recall/ranking, query expansion,
  agent capture, prepared context, maintenance, or tool/bridge behavior.
- **Expected output**: exact kind/status/key, relevant document IDs, required
  text/facets/entities/topics, packed context contents, or proposal count.
- **Must-not behavior**: false writes, assistant text contamination, benchmark
  answer leakage, stale/superseded memories, irrelevant context, or over-budget
  packing.
- **Metric type**: binary pass rate for objective checks, numeric score for rank
  or latency, model-as-judge only when subjective quality is unavoidable.
- **Aggregation**: mean/pass rate for binary metrics, MRR/nDCG/recall for ranked
  retrieval, median or p95 for latency-sensitive dimensions.

If the criterion cannot be expressed as a stable metric, do not promote it into
a release gate yet. Add it as exploratory diagnostics or a review queue item.

## Choose The Dataset Surface

- `Evals/memory_schema_gold_v2`: canonical write-path kind/status/facet/entity/topic/update behavior.
- `Evals/agent_memory_gold_v1`: public agent workflow behavior: no-write, capture/extract, update lifecycle, recall, context prep, and maintenance.
- `Evals/general_v2`: broad retrieval regression gate.
- `Evals/longmemeval_v2`: long-horizon conversational recall benchmark.
- `Evals/query_expansion_gold_v1`: query-expansion coverage and no-harm checks.
- focused LongMemEval slices: use only as targeted regression/debug gates and
  pair wins with a broader suite before promoting runtime changes.

For new coverage, prefer extending the smallest relevant existing suite. Create
a new committed dataset only when the behavior has a distinct purpose, manifest,
and gate plan.

## Dataset Design Checklist

For each new or edited dataset, ensure:

- clear `manifest.json` purpose, provenance, synthetic status, and review status
- unique IDs and one dominant behavior per row
- categories that cover golden path, edge cases, adversarial/refusal cases, and
  known regressions
- variation in input length, phrasing, ambiguity, entity/date density, and
  user/profile style
- at least one hard or negative row for any behavior likely to false-positive
- no benchmark-specific rescue phrases, answer strings, IDs, or named facts in
  production runtime logic
- expected labels are auditable from the input and not hidden in the prompt
- holdout or pressure rows remain separate until they are stable enough for a
  release gate

## Implementation Pattern

1. Add or update JSONL rows with `apply_patch`.
2. If new fields are needed, extend the Decodable structs in
   `MemoryEvalCLI.swift`.
3. Score the field in the suite runner and include per-case evidence in
   `caseResults`.
4. Export aggregate metrics through `reducedMetrics(from:)` if a baseline may
   gate the value.
5. Add the metric to console and Markdown summaries so failures are visible
   without reading raw JSON.
6. Update `Evals/README.md` only for durable schema/workflow changes.
7. Update `Evals/baselines/*.json` only after fresh no-cache reports support
   the new threshold.

Do not hard-code baseline thresholds outside the manifests.

## Validation

For dataset or scoring changes, run:

```bash
swift run memory_eval validate-datasets --strict
python3 Scripts/check_benchmark_leakage.py
git diff --check
```

For code changes, also run:

```bash
swift test
```

For release-gate changes, run the affected suite with deterministic settings:

```bash
swift run --traits CoreMLEmbedding memory_eval run \
  --profile coreml_default \
  --dataset-root ./Evals/<dataset> \
  --no-cache \
  --no-index-cache
```

Then gate fresh reports. If using `current.json`, provide all required run JSONs;
for a focused check, create a temporary baseline containing only the changed
requirements and state clearly that it is not the full release gate.

## Report Back

Summarize:

- changed dataset roots and metric fields
- fresh report JSON/Markdown paths
- old vs new headline metrics when relevant
- whether baseline manifests changed
- validation commands and any suites not rerun
