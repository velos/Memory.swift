# Agent Guide

This repository is a Swift package for on-device memory storage, indexing, and
retrieval. Keep this file as a navigation layer; prefer updating the referenced
docs/manifests instead of duplicating detailed policy here.

## Start Here

- Public API and package overview: `README.md`
- Targets, products, and dependencies: `Package.swift`
- Eval dataset format and workflows: `Evals/README.md`
- Release and focused eval gates: `Evals/baselines/*.json`
- External Agent Memory Benchmark notes: `Docs/agent-memory-benchmark.md`
- Autoresearch loop: `Autoresearch/README.md` and `Autoresearch/program.md`

## Architecture Map

- `Sources/Memory/`: public types, `MemoryIndex`, recall/search planning, agent
  memory APIs, and provider protocols.
- `Sources/MemoryStorage/`: internal SQLite, FTS, and sqlite-vec persistence.
  External integrations should use the `Memory` product, not this target.
- `Sources/CSQLiteVec/`: vendored sqlite-vec C shim.
- `Sources/MemoryNaturalLanguage/`: Apple NaturalLanguage query analysis and
  fallback embedding helpers.
- `Sources/MemoryCoreMLEmbedding/`: Core ML embedding/reranker providers plus
  tokenizer resources. `Models/embedding-v1.mlpackage` is the default local
  Core ML embedding artifact used by eval/CLI profiles.
- `Sources/MemoryAppleIntelligence/`: optional FoundationModels-backed
  expansion, reranking, and tagging providers.
- `Sources/MemoryCLI/`: `memory` CLI and persistent JSON-lines bridge.
- `Sources/MemoryEvalCLI/`: `memory_eval` harness, reports, diagnostics, gates,
  and eval profile definitions.

## Baselines And Gates

The source of truth is the manifest content under `Evals/baselines/`.

- `current.json`: release gate for the shipped `coreml_default` path across
  storage, agent memory, broad recall, LongMemEval recall, and query-expansion
  no-harm behavior.
- `pressure.json`: hard agent-memory pressure cases before promotion into the
  stable release gate.
- `longmemeval_rescue.json`, `longmemeval_ranking.json`,
  `longmemeval_multievidence.json`: focused LongMemEval regression slices.
- `query_expansion_rescue.json`: focused query-expansion rescue/no-harm gate.

Use the manifest notes and `required_runs` entries for exact datasets, metrics,
thresholds, and freshness requirements. Do not hard-code those values elsewhere.

Useful commands:

```bash
swift run memory_eval validate-datasets --strict
swift run memory_eval gate --baseline Evals/baselines/current.json <run-json>
python3 Scripts/check_benchmark_leakage.py
```

## Evaluation Workflow

- General eval command and report schema live in `Sources/MemoryEvalCLI/`.
- Dataset shapes and promotion/audit process live in `Evals/README.md`.
- Run artifacts are written under each dataset's `runs/` directory.
- Retrieval-only diagnostics are the first tool for recall work:

```bash
swift run memory_eval retrieval-diagnostics \
  --profile coreml_default \
  --dataset-root ./Evals/longmemeval_v2 \
  --candidate-pool-depth 40 \
  --context-token-budget 4096 \
  --no-cache \
  --no-index-cache
```

Use `Scripts/analyze_retrieval_diagnostics.py` on diagnostics JSON to separate
candidate-generation misses from ranking/packing misses. The optional
`--grounded-expansion` flag is eval-only grounded pseudo-relevance feedback;
keep production defaults unchanged unless reports show broad no-harm gains.

## External Benchmarks

The useful Memory.swift-side lessons and command shapes are summarized in
`Docs/agent-memory-benchmark.md`.

For cost control, prefer AMB `retrieval-diagnostics` runs over answer-generation
or LLM-judge runs when tuning retrieval.

## Development Checks

Default local check:

```bash
swift test
```

Before changing eval datasets or gates:

```bash
swift run memory_eval validate-datasets --strict
python3 Scripts/check_benchmark_leakage.py
```

For release-gate work, run the exact datasets/profiles listed in
`Evals/baselines/current.json`, then gate the produced run JSON against that
manifest.

## Working Rules

- Preserve the clean `coreml_default` baseline unless a gate update is
  intentional and supported by fresh reports.
- Keep benchmark-specific rescue phrases out of production retrieval logic.
- Prefer generic diagnostics, corpus-auditable terms, and dataset manifests over
  benchmark-shaped heuristics.
- Keep bulky/generated artifacts local unless the dataset README or baseline
  manifest explicitly makes them part of the repo contract.
