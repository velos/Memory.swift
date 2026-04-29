# agent_memory_gold_v1

Scenario benchmark for the default agent-memory loop:

- extract memories from multi-turn messages
- avoid writes for questions and conversational filler
- consolidate profile, decision, commitment, and handoff updates
- recall useful active or explicitly requested resolved memories

Files:

- `scenarios.seed.jsonl`: hand-curated smoke cases
- `scenarios.generated.jsonl`: deterministic template expansion
- `scenarios.pressure.jsonl`: harder known-failure or near-failure cases, not included in the default gate unless requested
- `scenarios.model_drafts.jsonl`: optional model-drafted candidates for review
- `scenarios.jsonl`: default gate set, normally seed plus deterministic generated scenarios

Generate or refresh scenarios with:

```bash
python3 Scripts/generate_agent_memory_scenarios.py --dataset-root ./Evals/agent_memory_gold_v1 --overwrite
```

Include pressure cases in the gate set with:

```bash
python3 Scripts/generate_agent_memory_scenarios.py --dataset-root ./Evals/agent_memory_gold_v1 --include-pressure --overwrite
```

Optional model-drafted candidates use the Anthropic-compatible endpoint from `.env`:

```bash
python3 Scripts/generate_agent_memory_scenarios.py --dataset-root ./Evals/agent_memory_gold_v1 --backend minimax --model-count 12 --no-write-combined --overwrite
```

Run with:

```bash
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/agent_memory_gold_v1 --no-cache --no-index-cache
```
