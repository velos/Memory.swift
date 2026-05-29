# Agent Memory Benchmark Notes

These notes preserve the useful findings from local experiments in
`references/agent-memory-benchmark` without requiring those local benchmark
repo modifications to be checked into `Memory.swift`.

## What To Keep Local

- The `memory-swift` provider for the external benchmark.
- Gemini, Minimax, and local proxy LLM wrappers.
- External benchmark output stores under `references/agent-memory-benchmark/outputs/`.
- Leaderboard submission scripts and run artifacts.

Those are integration scaffolding for a separate benchmark project, not part of
the shipped Memory.swift eval harness.

## What Belongs In `memory_eval`

The useful ideas to keep in this repo are provider-agnostic:

- Retrieval-only diagnostics with no answer LLM in the loop.
- Query-level latency percentiles, not only aggregate accuracy.
- Stage timings and candidate counts for search internals.
- Dataset hygiene checks before a benchmark is treated as a gate.
- Focused regression slices mined from broad benchmark misses.

The native `memory_eval` CLI now covers these through:

- `memory_eval run`
- `memory_eval gate`
- `memory_eval validate-datasets`
- `memory_eval retrieval-diagnostics`
- `memory_eval diagnose-longmemeval`

Native retrieval-only smoke run:

```bash
swift run memory_eval retrieval-diagnostics \
  --profile coreml_default \
  --dataset-root ./Evals/longmemeval_v2 \
  --candidate-pool-depth 40 \
  --context-token-budget 4096 \
  --per-document-token-budget 384 \
  --context-packing-order rank \
  --query-limit 50 \
  --no-cache \
  --no-index-cache
```

The native report includes actual packed-context quality, score-sorted packed
sidecar metrics, and candidate-pool coverage. When `Candidate Hit@pool` is high
but Hit@10 is lower, focus on ranking, context packing, and source preservation.
When candidate-pool coverage is low, focus on query expansion and candidate
generation.

Corpus-grounded pseudo-relevance feedback now lives in the library search path
behind `MemoryConfiguration.groundedQueryExpansion`. The conservative default is
enabled for normal `coreml_default` evals: it mines terms from bounded top
retrieved local documents, requires weak lexical coverage, a semantic feedback
cluster, enough feedback evidence, and no strong rank-one result, then runs one
weak lexical follow-up branch by default without extra query embeddings.
Evidence-dense temporal/count/ordering queries skip grounded PRF because broad
diagnostics showed added latency without ranking movement there.

Validate the runtime path with normal retrieval diagnostics, without the
eval-only `--grounded-expansion` flag:

```bash
swift run memory_eval retrieval-diagnostics \
  --profile coreml_default \
  --dataset-root ./Evals/longmemeval_v2 \
  --candidate-pool-depth 40 \
  --context-token-budget 4096 \
  --no-cache \
  --no-index-cache
```

The diagnostic `--grounded-expansion` flag is still useful for eval-only
what-if experiments, but product decisions should use the normal library path
above so grounded PRF is not applied twice.

Latest broad runtime grounded PRF diagnostics:

- `general_v2`: Hit@10 and Recall@10 unchanged at `90.85%` and `90.67%`;
  MRR@10 improved from `0.7440` to `0.7441`; nDCG@10 improved from `0.7843`
  to `0.7844`; 1 rank improvement and 0 rank regressions.
- `longmemeval_v2`: Hit@10 and Recall@10 unchanged at `89.50%` and `81.12%`;
  MRR@10 and nDCG@10 unchanged at `0.6105` and `0.6195`; 0 rank changes.

The `all` term mode failed broad no-harm on `general_v2` by reducing Hit@10 and
Recall@10, and the initial `0.35` runtime lexical branch weight regressed one
top-ranked `general_v2` case. Keep the default `phrase-entity` mode, one weak
branch, and `0.20` branch weight unless fresh broad evidence says otherwise.

The persistent `memory serve` bridge also accepts optional search params named
`contextTokenBudget`, `perDocumentTokenBudget`, and `contextPackingOrder`.
Benchmark adapters can pass `4096` plus `256` or `384` respectively to use the
same capped context packing policy that performed best in native diagnostics.
Keep `contextPackingOrder` at `rank` unless an experiment shows score-first
packing improves both recall and rank quality.

## External LongMemEval Retrieval Run

Command shape used locally:

```bash
HF_HOME=/tmp/amb-hf-cache \
MEMORY_SWIFT_SKIP_BUILD=1 \
MEMORY_SWIFT_USE_BRIDGE=1 \
MEMORY_SWIFT_CONTEXT_PROFILE=balanced \
MEMORY_SWIFT_QUERY_ENRICHMENT=0 \
uv run omb retrieval-diagnostics \
  --dataset longmemeval \
  --split s \
  --memory memory-swift \
  --skip-ingestion \
  --name memory-swift-balanced-library-temporal-protected
```

Latest useful local artifact:

```text
references/agent-memory-benchmark/outputs/longmemeval/memory-swift-balanced-library-temporal-protected/retrieval/s.json
```

Latest measured result after scoped temporary FTS:

- Runtime: `3:21` for 500 queries.
- Mean retrieval latency: `401.2 ms`.
- p50 retrieval latency: `378.8 ms`.
- p95 retrieval latency: `564.7 ms`.
- p99 retrieval latency: `982.8 ms`.
- Hit@10: `92.2%`.
- MRR@10: `0.8346`.
- Gold Recall@10: `89.1%`.
- With-gold Hit@10: `96.2%`.
- With-gold Recall@10: `93.0%`.

The prior slow run used the same broad retrieval shape but spent most time in
scoped lexical search:

- Runtime: `1:18:39`.
- Mean retrieval latency: `9436.9 ms`.
- p50 retrieval latency: `8141.1 ms`.
- p95 retrieval latency: `21234.6 ms`.

## Library Lessons From AMB

- Keep benchmark retrieval isolated from the answer LLM when tuning memory
  quality.
- Use a persistent bridge for repeated benchmark queries; process startup and
  CoreML model loading otherwise dominate.
- Scoped collections must not query global FTS or global vector indexes and then
  filter afterward.
- Balanced context around 4k tokens is a useful external-benchmark profile, but
  this should remain configurable for on-device 8k-context agents.
- External benchmark adapters should remain local or live in the benchmark repo;
  Memory.swift should only absorb generic eval infrastructure and runtime fixes.

## ChatGPT-Signed Codex Smoke Runs

For local ballpark end-to-end runs with ChatGPT subscription access, this
checkout has used a local `codex-chatgpt` LLM provider in the ignored AMB
reference tree. It shells out to `codex exec`, uses the active Codex ChatGPT
login, and asks Codex for structured JSON output. This is intentionally separate
from the API-key `openai` provider: it is useful for local signal, not for
public apples-to-apples AMB claims.

Useful environment variables:

```bash
OMB_ANSWER_LLM=codex-chatgpt
OMB_ANSWER_MODEL=gpt-5.5
OMB_JUDGE_LLM=codex-chatgpt
OMB_JUDGE_MODEL=gpt-5.4-mini
OMB_CODEX_REASONING_EFFORT=low
OMB_CODEX_VERBOSITY=low
```

Before running, verify the local auth state:

```bash
codex login status
```

Use a small `--query-limit` first. Each answer and each LLM-judge call starts a
separate Codex session, so full 500-query LongMemEval runs consume far more
ChatGPT/Codex message allowance than Gemini/API-key runs.

Local 2026-05-09 LongMemEval `s` run:

- Run path:
  `references/agent-memory-benchmark/outputs/longmemeval/memory-swift-codex-chatgpt-full/rag/s.json`
- Answer model: `codex-chatgpt:gpt-5.5:low`
- Judge model: `codex-chatgpt:gpt-5.4-mini:low`
- Result: 443/500 correct, 88.6% accuracy

Failure analysis pointed mostly at fixed-context evidence survival and answer
synthesis, not raw candidate generation. The corresponding retrieval-only run
had Hit@10 93.2%, with-gold Hit@10 97.3%, and with-gold support-document
Recall@10 93.4%.

For AMB adapter work, prefer passing `memory serve` context-packaging params
through the bridge instead of repacking full chunks only in Python:
`contextTokenBudget`, `perDocumentTokenBudget`, and `contextPackingOrder`.
Focused diagnostics on three multi-evidence failures showed document-level
support Recall@10 improvements from 50.0% to 75.0%, 60.0% to 100.0%, and 66.7%
to 83.3%. Those did not all convert to judged answer wins because some AMB gold
matches are document-level while Memory.swift returns chunk-level context; a
retrieved gold document can still expose the wrong chunk for a counting answer.
