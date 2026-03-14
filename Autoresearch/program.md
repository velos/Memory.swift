# Memory.swift Autoresearch

This repo is a fixed-budget autonomous experiment loop for improving the MLX-trained CoreML components used by `Memory.swift`.

The outer agent for this workflow is OpenCode. The repo itself is not the agent. The repo defines the protocol that OpenCode follows.

## Monorepo note

This project lives inside the `Memory.swift` repo. Always stage only `Autoresearch/` paths unless you are explicitly changing `Memory.swift` runtime code as part of the experiment system. Never use blind `git add -A`.

## Setup

To start a new run:

1. Work from a dedicated feature branch in `Memory.swift`.
2. Read these files before you touch anything:
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `program.md`
3. Run `uv run prepare.py` once if the cache is missing.
4. Confirm that:
   - you are inside `Autoresearch/`
   - the local `memory_eval` binary was built from the parent `Memory.swift` checkout
   - the parent repo contains `Evals/` and `Models/`
   - `typing_train.jsonl`, `retrieval_train.jsonl`, `quick_eval/`, and `full_eval/` exist under `~/.cache/memory-swift-autoresearch/datasets/`
   - the hardware profile JSON exists under `~/.cache/memory-swift-autoresearch/hardware/`
5. Create or reset `results.tsv` with the current schema if needed.
6. Establish a hardware-local baseline by running `uv run train.py` without edits.

## Rules

You may edit only `train.py`.

Do not edit:

- `prepare.py`
- `memory_autoresearch/`
- the scoring contract
- the parent `Memory.swift` runtime unless that is the explicit goal of the run

Do not add ad-hoc dependencies during the loop. All required dependencies belong in `pyproject.toml` and are treated as fixed once the run starts.

## Goal

Maximize `memory_score` while staying under the hard keep gates for model size and latency.

Every run trains only one component:

- `typing`
- `embedding`
- `reranker`

The current component is selected by `ACTIVE_COMPONENT` in `train.py`.

## Output

At the end of every run, `train.py` prints:

```text
---
component:         typing
memory_score:      0.612345
storage_score:     0.701234
recall_score:      0.405678
model_mb:          8.2
latency_ms:        18.4
training_seconds:  300.0
num_steps:         91
average_loss:      0.412345
status:            keep
```

Read the summary directly from `run.log`:

```bash
grep "memory_score:\|storage_score:\|recall_score:\|model_mb:\|latency_ms:\|status:" run.log
```

## Logging

Append each experiment to `results.tsv` as tab-separated data with this schema:

```text
commit	component	memory_score	storage_score	recall_score	model_mb	latency_ms	status	description
```

Statuses:

- `keep`
- `discard`
- `discard_full`
- `crash`

## Experiment Loop

Loop forever:

1. Inspect the current git state.
2. Edit only `train.py`.
3. `git add Autoresearch/train.py && git commit -m "experiment: <description>"`
4. Run `uv run train.py > run.log 2>&1`
5. If the run crashes, inspect the stack trace with `tail -n 80 run.log`, fix the issue in `train.py`, and retry.
6. If the run succeeds, append the result to `results.tsv`.
7. If status is `keep`, stage `results.tsv` and amend the commit to include the log entry.
8. If status is not `keep`, revert to the previous kept commit.

## Keep/Revert Policy

Keep only if all of the following are true:

- `memory_score` improves by at least `0.003`, or ties within `0.001` while improving model size or latency
- the hard component gate passes
- the quick eval passes
- the full eval does not regress

If a quick eval wins but the full eval regresses, log `discard_full` and revert.

## Timeout

The fixed training budget is 5 minutes. Export and eval happen after that.

Treat any run that exceeds 45 minutes total wall clock as a failure and discard it.

## Autonomy

Do not stop to ask whether you should continue once the loop begins. Keep iterating until manually interrupted.
