---
name: memory-autoresearch
description: Explain and operate the `Memory.swift` autoresearch loop in `tools/autoresearch`. Use when asked what the autoresearch tooling is, how the optimization loop works, how to bootstrap it, how to kick off a run, or how to inspect its outputs and status.
---

# Memory Autoresearch

Understand the in-repo autoresearch tool before running anything. Start by reading:

- `tools/autoresearch/README.md`
- `tools/autoresearch/program.md`
- `tools/autoresearch/train.py`

Read `tools/autoresearch/prepare.py` only if you need to confirm bootstrap behavior or cache outputs.

## What It Is

Explain the tool as:

- an OpenCode-driven optimization loop for `Memory.swift`
- focused on three CoreML artifacts: typing, embedding, reranker
- fixed around `prepare.py`, the support package, scoring, and a 5-minute MLX train budget
- mutable only through `train.py` during normal experiment iteration
- evaluated with the real local `memory_eval` binary from the current `Memory.swift` checkout

Mention these paths when useful:

- tool root: `tools/autoresearch`
- cache root: `~/.cache/memory-swift-autoresearch/`
- experiment ledger: `tools/autoresearch/results.tsv`

## Kick Off

When the user wants to start the loop, run from `tools/autoresearch`:

```bash
uv sync
uv run prepare.py
uv run train.py
```

If the user wants a captured log:

```bash
uv run train.py > run.log 2>&1
tail -n 80 run.log
```

Before running `train.py`, check `ACTIVE_COMPONENT` and any mutable knobs in `tools/autoresearch/train.py`.

## What To Report

After a run, report:

- `component`
- `memory_score`
- `storage_score`
- `recall_score`
- `model_mb`
- `latency_ms`
- `training_seconds`
- `num_steps`
- `average_loss`
- `status`

Also mention whether a row was appended to `tools/autoresearch/results.tsv`.

## Workflow Notes

- `prepare.py` bootstraps datasets, baseline artifacts, and the hardware profile against the local repo.
- `train.py` trains one component at a time.
- Typing may use a smaller checkpoint override in `train.py`; confirm the current setting before describing defaults.
- Embedding and reranker evaluation swap candidate CoreML artifacts into `Models/` during eval, then restore baselines afterward.
- Full eval is much slower than quick eval; use the log and process state to distinguish “still running” from “stuck”.

## Guardrails

- Do not present the tool as a separate external repo; it now lives inside `Memory.swift`.
- Do not say the repo itself is the agent. The outer operator is OpenCode or another coding agent.
- Do not edit the fixed support package or scoring contract unless the user explicitly asks for infrastructure changes.
- If the user only asks for an explanation, answer from the docs and code first instead of kicking off a long run.
