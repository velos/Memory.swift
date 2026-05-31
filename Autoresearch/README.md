# Memory.swift Autoresearch

Shared workspace for fixed-budget autonomous experiments against this
`Memory.swift` checkout.

The top level contains the shared Python environment and support package:

- `pyproject.toml` and `uv.lock`
- `memory_autoresearch/`

Each experiment setup owns its own `prepare.py`, `train.py`, `program.md`,
`results.tsv`, and `run.log`.

## Setups

- `retrieval/` - broad CoreML memory-stack loop for typing, embedding, and
  retrieval-oriented reranker experiments.
- `reranker/` - focused opt-in neural reranker loop that evaluates through
  `coreml_rerank` and leaves `coreml_default` unchanged.

## Commands

Run commands from this directory:

```bash
uv sync

uv run retrieval/prepare.py
uv run retrieval/train.py
uv run retrieval/train.py > retrieval/run.log 2>&1

uv run reranker/prepare.py
uv run reranker/train.py
uv run reranker/train.py > reranker/run.log 2>&1
```

## Cache Layout

Shared cache root:

```text
~/.cache/memory-swift-autoresearch/
```

Shared hardware and tokenizer caches stay directly under that root. Setup-local
datasets, candidate artifacts, baselines, and reports live under:

```text
~/.cache/memory-swift-autoresearch/retrieval/
~/.cache/memory-swift-autoresearch/reranker/
```

## Working Rules

- During normal iteration, edit only the setup's `train.py`.
- Keep `memory_autoresearch/` fixed unless changing the experiment
  infrastructure itself.
- Keep generated model artifacts, run logs, and local corpora untracked.
- Stage only the setup paths or shared support files you intentionally changed.
