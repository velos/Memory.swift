# AgentMemory Retrieval Autoresearch

This repo is a fixed-budget autonomous experiment loop for improving the MLX-trained CoreML components used by `AgentMemory`.

The outer agent for this workflow is OpenCode. The repo itself is not the agent. The repo defines the protocol that OpenCode follows.

## Monorepo note

This project lives inside the `AgentMemory` repo. Always stage only `Autoresearch/` paths unless you are explicitly changing `AgentMemory` runtime code as part of the experiment system. Never use blind `git add -A`.

## Setup

To start a new run:

1. Work from a dedicated feature branch in `AgentMemory`.
2. Read these files before you touch anything:
   - `Autoresearch/README.md`
   - `Autoresearch/retrieval/README.md`
   - `Autoresearch/retrieval/prepare.py`
   - `Autoresearch/retrieval/train.py`
   - `Autoresearch/retrieval/program.md`
3. From `Autoresearch/`, run `uv run retrieval/prepare.py` once if the cache is missing.
4. Confirm that:
   - you are inside `Autoresearch/`
   - the local `memory_eval` binary was built from the parent `AgentMemory` checkout
   - the parent repo contains `Evals/` and `Models/`
   - `typing_train.jsonl`, `retrieval_train.jsonl`, `quick_eval/`, and `full_eval/` exist under `~/.cache/memory-swift-autoresearch/retrieval/datasets/`
   - the hardware profile JSON exists under `~/.cache/memory-swift-autoresearch/hardware/`
5. Create or reset the local gitignored `retrieval/results.tsv` with the current schema if needed.
6. Establish a hardware-local baseline by running `uv run retrieval/train.py` without edits.

## Rules

You may edit only `Autoresearch/retrieval/train.py`.

Do not edit during normal experiment iteration:

- `Autoresearch/retrieval/prepare.py`
- `memory_autoresearch/`
- the scoring contract
- the parent `AgentMemory` runtime unless that is the explicit goal of the run

Do not add ad-hoc dependencies during the loop. All required dependencies belong in `pyproject.toml` and are treated as fixed once the run starts.

## Goal

Maximize aggregate `memory_score` while keeping `general` and `longmemeval` healthy and staying under the component gates for model size and latency.

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
decision_reason:   quick pass: general_delta=0.0123, longmemeval_delta=0.0011
```

Read the summary directly from `retrieval/run.log`:

```bash
grep "memory_score:\|storage_score:\|recall_score:\|model_mb:\|latency_ms:\|status:" retrieval/run.log
```

## Logging

Append each experiment to `retrieval/results.tsv` as tab-separated data with this schema:

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
3. `git add Autoresearch/retrieval/train.py && git commit -m "experiment: <description>"`
4. Run `uv run retrieval/train.py > retrieval/run.log 2>&1`
5. If the run crashes, inspect the stack trace with `tail -n 80 retrieval/run.log`, fix the issue in `retrieval/train.py`, and retry.
6. If the run succeeds, append the result to local `retrieval/results.tsv`.
7. Do not stage `retrieval/results.tsv` or `retrieval/run.log`; they are local experiment artifacts.
8. If status is not `keep`, revert to the previous kept commit.

## Keep/Revert Policy

Keep only if all of the following are true:

- `memory_score` improves by at least `0.003`, or ties within `0.001` while improving model size or latency
- the hard component gate passes
- `general` does not regress in quick or full eval
- `longmemeval` does not materially regress in quick or full eval
- quick/full latency stays within the configured primary-dataset tolerances

If a quick eval wins but the full eval regresses, log `discard_full` and revert.

## Timeout

The fixed training budget is 5 minutes. Export and eval happen after that.

Treat any run that exceeds 45 minutes total wall clock as a failure and discard it.

## Autonomy

Do not stop to ask whether you should continue once the loop begins. Keep iterating until manually interrupted.
