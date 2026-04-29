# longmemeval_rescue_v1

Focused LongMemEval recall rescue slice mined from branch diagnostics.

This is a small diagnostic benchmark, not a replacement for the full LongMemEval gate. It keeps the selected source queries, their relevant documents, and the top confuser documents seen across diagnostic retrieval branches so candidate-generation changes can be tested quickly.

Source diagnostic: `Evals/longmemeval_v2/runs/2026-04-26T20-58-14Z-coreml_default.wide200.branch-diagnostics.json`
Source run: `Evals/longmemeval_v2/runs/2026-04-26T20-58-14Z-coreml_default.json`

Classification counts:

- `candidate_generation`: 4

Taxonomy counts:

- `entity/alias`: 4
- `multi-evidence`: 2
- `temporal/count`: 4

Commands:

```sh
python3 Scripts/build_longmemeval_rescue.py --diagnostic <branch-diagnostics.json> --overwrite
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/longmemeval_rescue_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/longmemeval_rescue.json ./Evals/longmemeval_rescue_v1/runs/<run>.json
```
