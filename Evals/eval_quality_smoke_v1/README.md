# eval_quality_smoke_v1

Small curated synthetic smoke suite for the eval harness itself:

- validates required row metadata and accepted metadata enums
- exercises grouped storage and agent-memory metrics
- checks a small set of objective write, no-write, context, maintenance, and update behaviors

This is a focused gate, not part of `Evals/baselines/current.json`.

Run:

```bash
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/eval_quality_smoke_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/eval_quality_smoke.json <run-json>
```
