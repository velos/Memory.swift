---
name: memory-autoresearch
description: Explain and operate the `AgentMemory` autoresearch loop in `Autoresearch`. Use when asked what the autoresearch tooling is, how the optimization loop works, how to bootstrap it, how to kick off a run, or how to inspect its outputs and status.
---

# Memory Autoresearch

Understand the in-repo autoresearch tool before running anything. Start by reading:

- `Autoresearch/README.md`
- the setup-specific `program.md`
- the setup-specific `train.py`

Read the setup-specific `prepare.py` only if you need to confirm bootstrap behavior or cache outputs.

## What It Is

Explain the tool as:

- an OpenCode-driven optimization loop for `AgentMemory`
- split into setup folders under `Autoresearch/`
- `retrieval` is the broad loop for typing, embedding, and retrieval-oriented model work
- `reranker` is the focused neural reranker loop evaluated through `coreml_rerank`
- fixed around `prepare.py`, the support package, scoring, and a 5-minute MLX train budget
- mutable only through the setup's `train.py` during normal experiment iteration
- evaluated with the real local `memory_eval` binary from the current `AgentMemory` checkout

Mention these paths when useful:

- tool root: `Autoresearch`
- retrieval setup: `Autoresearch/retrieval`
- reranker setup: `Autoresearch/reranker`
- cache root: `~/.cache/memory-swift-autoresearch/`
- retrieval ledger: `Autoresearch/retrieval/results.tsv`
- reranker ledger: `Autoresearch/reranker/results.tsv`

## Choose Setup

Route the work before running or editing anything:

| User goal | Setup | Eval profile | Mutable file |
| --- | --- | --- | --- |
| broad retrieval, embedding, typing, current shipped path | `retrieval` | usually `coreml_default` | `Autoresearch/retrieval/train.py` |
| opt-in neural reranker, `reranker-v1.mlpackage`, ranking experiments | `reranker` | `coreml_rerank` | `Autoresearch/reranker/train.py` |
| new independent research goal | new `Autoresearch/<setup>/` folder | explicit in setup docs | setup-local `train.py` |

For a new setup, first define the goal contract. Do not duplicate the shared
environment or support package unless the user explicitly asks for isolation.

## Goal Contract

Before creating or changing a setup, make the outcome explicit:

- `SETUP_NAME` and setup directory
- artifact or behavior being optimized
- mutable file and fixed support files
- eval profile and datasets
- keep/discard criteria
- broad no-harm checks before promotion
- cache roots and generated artifacts that must remain local

For reranker work, the default contract is: train/export a candidate
`reranker-v1.mlpackage`, evaluate through `coreml_rerank`, and leave
`coreml_default` unchanged unless the user explicitly asks for promotion.

## Kick Off

When the user wants to start the loop, run from `Autoresearch`:

```bash
uv sync
uv run retrieval/prepare.py
uv run retrieval/train.py
```

For the focused reranker setup, run from `Autoresearch`:

```bash
uv sync
uv run reranker/prepare.py
uv run reranker/train.py
```

If the user wants a captured log:

```bash
uv run retrieval/train.py > retrieval/run.log 2>&1
tail -n 80 retrieval/run.log
```

For reranker logs:

```bash
uv run reranker/train.py > reranker/run.log 2>&1
tail -n 80 reranker/run.log
```

Before running a setup `train.py`, check `ACTIVE_COMPONENT` and any mutable knobs in that file.

## Status Report

After a run or status check, report:

- `setup`
- `component`
- `profile`
- `memory_score`
- `storage_score`
- `recall_score`
- `model_mb`
- `latency_ms`
- `training_seconds`
- `num_steps`
- `average_loss`
- `status`
- latest report path when available
- whether any temporary repo model in `Models/` was restored or removed

Also mention whether a row was appended to the setup's `results.tsv`.

## Workflow Notes

- Setup `prepare.py` bootstraps datasets, baseline artifacts, and the hardware profile against the local repo.
- Setup `train.py` trains one component at a time.
- Typing may use a smaller checkpoint override in `retrieval/train.py`; confirm the current setting before describing defaults.
- Reranker setup uses the opt-in `coreml_rerank` profile and should not be described as changing `coreml_default`.
- Reranker setup may create a cache-local baseline if `Models/reranker-v1.mlpackage` is absent; temporary repo installs are restored or removed after eval.
- Embedding and reranker evaluation swap candidate CoreML artifacts into `Models/` during eval, then restore baselines afterward.
- Full eval is much slower than quick eval; use the log and process state to distinguish “still running” from “stuck”.

## Guardrails

- Do not present the tool as a separate external repo; it now lives inside `AgentMemory`.
- Do not say the repo itself is the agent. The outer operator is OpenCode or another coding agent.
- Do not edit the fixed support package or scoring contract unless the user explicitly asks for infrastructure changes.
- If the user only asks for an explanation, answer from the docs and code first instead of kicking off a long run.
