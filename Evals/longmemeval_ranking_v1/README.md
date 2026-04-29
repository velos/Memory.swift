# longmemeval_ranking_v1

Focused LongMemEval recall slice mined from branch diagnostics.

This is a small diagnostic benchmark, not a replacement for the full LongMemEval gate. It keeps the selected source queries, their relevant documents, and the top confuser documents seen across diagnostic retrieval branches so targeted recall changes can be tested quickly.

Source diagnostic: `Evals/longmemeval_v2/runs/2026-04-27T03-33-40Z-coreml_default.wide200.branch-diagnostics.json`
Source run: `Evals/longmemeval_v2/runs/2026-04-27T03-33-40Z-coreml_default.json`

Classification counts:

- `ranking_or_pool_depth`: 19

Taxonomy counts:

- `contextual-ellipsis`: 1
- `entity/alias`: 17
- `episodic/contextual`: 2
- `multi-evidence`: 10
- `temporal/count`: 15

Commands:

```sh
python3 Scripts/build_longmemeval_rescue.py --diagnostic <branch-diagnostics.json> --overwrite
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/longmemeval_ranking_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/<focused-baseline>.json ./Evals/longmemeval_ranking_v1/runs/<run>.json
```
