# Memory.swift Reranker Autoresearch

Focused autonomous loop for training an opt-in CoreML reranker for
`Memory.swift`.

This setup has one purpose:

- produce a candidate `reranker-v1.mlpackage`
- evaluate it only through `coreml_rerank`
- keep `coreml_default` unchanged unless a later promotion explicitly asks for
  it

## Quick Start

Run from `Autoresearch/`:

```bash
uv sync
uv run reranker/prepare.py
uv run reranker/train.py
uv run reranker/train.py > reranker/run.log 2>&1
```

`prepare.py` builds the local `memory_eval` binary, materializes the shared eval
cache for this setup, seeds the embedding baseline, and seeds a reranker
baseline. If `Models/reranker-v1.mlpackage` is absent, it builds a cache-local
baseline with `Scripts/convert_tinybert_reranker_coreml.py` and marks it as
generated so eval cleanup removes any temporary repo-model install afterward.

## Files That Matter

- `reranker/prepare.py` - fixed bootstrapper
- `reranker/train.py` - mutable experiment surface
- `reranker/program.md` - autonomous experiment protocol
- `reranker/results.tsv` - local append-only experiment ledger
- `memory_autoresearch/` - shared fixed support package

## Current Metric

Every run prints:

```text
component
profile
memory_score
storage_score
recall_score
model_mb
latency_ms
training_seconds
num_steps
average_loss
status
decision_reason
```

Corpus-level output includes stage p95 timings, including `rerankMs` when the
profile exercised the reranker.

## Keep Policy

Keep only candidates that improve ranking quality without broad recall harm:

- hard reranker gate passes: `model_mb <= 25`, `latency_ms <= 150`
- focused ranking or multi-evidence slices improve or tie safely
- `general` and `longmemeval` do not regress in quick or full eval
- total latency stays within configured primary-dataset tolerances

Generated artifacts, local corpora, logs, and result ledgers stay untracked.
