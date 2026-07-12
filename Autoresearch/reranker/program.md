# AgentMemory Reranker Autoresearch

This setup is a fixed-budget autonomous experiment loop for improving the
opt-in CoreML neural reranker used by `AgentMemory`.

The outer agent for this workflow is OpenCode or another coding agent. The repo
itself is not the agent.

## Monorepo Note

This project lives inside the `AgentMemory` repo. Always stage only the
intended `Autoresearch/reranker/` paths unless you are explicitly changing
shared infrastructure or runtime wiring. Never use blind `git add -A`.

## Setup

To start a new run:

1. Work from a dedicated feature branch in `AgentMemory`.
2. Read these files before you touch anything:
   - `Autoresearch/README.md`
   - `Autoresearch/reranker/README.md`
   - `Autoresearch/reranker/prepare.py`
   - `Autoresearch/reranker/train.py`
   - `Autoresearch/reranker/program.md`
3. From `Autoresearch/`, run `uv run reranker/prepare.py` once if the cache is missing.
4. Confirm that:
   - the local `memory_eval` binary was built from the parent `AgentMemory` checkout
   - `retrieval_train.jsonl`, `quick_eval/`, and `full_eval/` exist under `~/.cache/memory-swift-autoresearch/reranker/datasets/`
   - the hardware profile JSON exists under `~/.cache/memory-swift-autoresearch/hardware/`
   - the reranker baseline exists under `~/.cache/memory-swift-autoresearch/reranker/artifacts/baselines/reranker/current.mlpackage`
5. Create or reset the local gitignored `reranker/results.tsv` with the current schema if needed.
6. Establish a hardware-local baseline by running `uv run reranker/train.py` without edits.

## Rules

You may edit only `Autoresearch/reranker/train.py` during normal iteration.

Do not edit during normal experiment iteration:

- `Autoresearch/reranker/prepare.py`
- `Autoresearch/memory_autoresearch/`
- the scoring contract
- the parent `AgentMemory` runtime unless that is the explicit goal of the run

Do not add ad-hoc dependencies during the loop. All required dependencies
belong in `Autoresearch/pyproject.toml` and are treated as fixed once the run
starts.

## Goal

Maximize aggregate `memory_score` while improving ranking quality through the
`coreml_rerank` profile and keeping `coreml_default` untouched.

This setup always trains only:

- `reranker`

## Output

At the end of every run, `train.py` prints:

```text
---
component:         reranker
profile:           coreml_rerank
memory_score:      0.612345
storage_score:     0.701234
recall_score:      0.405678
model_mb:          18.2
latency_ms:        88.4
training_seconds:  300.0
num_steps:         91
average_loss:      0.412345
status:            keep
decision_reason:   full pass: general_delta=0.0123, longmemeval_delta=0.0011
```

Read the summary directly from `reranker/run.log`:

```bash
grep "component:\|profile:\|memory_score:\|storage_score:\|recall_score:\|model_mb:\|latency_ms:\|status:" reranker/run.log
```

## Logging

Append each experiment to `reranker/results.tsv` as tab-separated data with this
schema:

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
2. Edit only `Autoresearch/reranker/train.py`.
3. `git add Autoresearch/reranker/train.py && git commit -m "experiment: <description>"`
4. Run `uv run reranker/train.py > reranker/run.log 2>&1`
5. If the run crashes, inspect the stack trace with `tail -n 80 reranker/run.log`, fix the issue in `reranker/train.py`, and retry.
6. If the run succeeds, append the result to local `reranker/results.tsv`.
7. Do not stage `reranker/results.tsv` or `reranker/run.log`; they are local experiment artifacts.
8. If status is not `keep`, revert to the previous kept commit.

## Keep/Revert Policy

Keep only if all of the following are true:

- `memory_score` improves by at least `0.003`, or ties within `0.001` while improving model size or latency
- the hard reranker gate passes
- `general` does not regress in quick or full eval
- `longmemeval` does not materially regress in quick or full eval
- quick/full latency stays within the configured primary-dataset tolerances

If a quick eval wins but the full eval regresses, log `discard_full` and revert.

## Timeout

The fixed training budget is 5 minutes. Export and eval happen after that.

Treat any run that exceeds 45 minutes total wall clock as a failure and discard
it.

## Autonomy

Do not stop to ask whether you should continue once the loop begins. Keep
iterating until manually interrupted.
