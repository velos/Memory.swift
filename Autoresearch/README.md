# Memory.swift Autoresearch

Apple-silicon autonomous optimization loop for the CoreML retrieval stack used by this `Memory.swift` checkout.

This tool keeps the original autoresearch process and changes the purpose:

- one fixed bootstrapper: `prepare.py`
- one mutable experiment surface: `train.py`
- one fixed train budget: 5 minutes
- one aggregate keep/revert score: `memory_score`
- per-dataset reporting for `general`, `longmemeval`, and `tech`
- local experiment logging in gitignored `results.tsv`

The intended outer agent is OpenCode or another coding agent that can read `program.md`, modify only `train.py`, run the experiment, and keep or revert based on the printed summary.

## What This Repo Optimizes

The loop trains and evaluates three separate CoreML artifacts:

- `memory-type-classifier.mlpackage`
- `memory-embedder.mlpackage`
- `memory-reranker.mlpackage`

These map onto `Memory.swift`'s current extension points:

- `MemoryTypeClassifier`
- `EmbeddingProvider`
- `Reranker`

## Quick Start

Requirements:

- Apple Silicon Mac
- Python 3.10+
- Xcode command line tools
- Swift toolchain capable of building `Memory.swift`
- [uv](https://docs.astral.sh/uv/)

```bash
cd Autoresearch

# install dependencies
uv sync

# bootstrap this Memory.swift checkout, eval datasets, baselines, and hardware profile
uv run prepare.py

# run one fixed-budget experiment
uv run train.py
```

## Files That Matter

- `prepare.py` - fixed bootstrapper for the local `Memory.swift` checkout, dataset cache, baselines, and hardware profile
- `train.py` - the only experiment file the outer agent edits
- `program.md` - the autonomous experiment protocol for OpenCode
- `results.tsv` - local append-only experiment ledger
- `memory_autoresearch/` - fixed support package for MLX training, export, scoring, and upstream integration

## Current Metric

Every run prints:

```text
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

`train.py` always trains only one component per run. The other components stay frozen at their current baseline artifacts while the candidate is evaluated.

## Current Defaults

- fixed train budget: 300 seconds
- training stack: MLX-first
- export path: PyTorch mirror -> `coremltools`
- eval source of truth: the current `Memory.swift` checkout built locally
- datasets:
  - tracked defaults: `general_v2` and `longmemeval_v2`
  - optional local extras: `tech`, `scifact`, `nfcorpus`, or other corpora generated under the gitignored `Explorations/` tree
  - quick eval: held-out slice from the tracked defaults
  - full eval: remaining held-out tracked defaults plus any optional local extras you have prepared

## Notes

- `prepare.py` is intentionally fixed. The experiment loop should not edit it.
- `train.py` is intentionally small and mutable. Change hyperparameters, model configuration, and training logic there.
- If `prepare.py` has not been run, `train.py` will fail with a clear message.
- Cache data is written under `~/.cache/memory-swift-autoresearch/`.
- Keep bulky training artifacts, logs, and CreateML project files local and untracked.

## License

MIT. See `LICENSE`.
