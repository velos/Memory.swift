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
- `memory_eval diagnose-longmemeval`

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
