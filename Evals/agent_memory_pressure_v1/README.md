# agent_memory_pressure_v1

Pressure benchmark for agent-memory behavior that should improve over time.

This dataset is generated from `Evals/agent_memory_gold_v1/scenarios.pressure.jsonl`.
It is intentionally separate from the stable release gate so hard cases can be
tracked before they are promoted into `agent_memory_gold_v1/scenarios.jsonl`.

Run with:

```bash
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/agent_memory_pressure_v1 --no-cache --no-index-cache
```

Gate with:

```bash
swift run memory_eval gate --baseline ./Evals/baselines/pressure.json <pressure-run-json>
```
