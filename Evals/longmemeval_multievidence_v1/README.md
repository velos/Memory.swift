# longmemeval_multievidence_v1

Focused LongMemEval recall slice mined from branch diagnostics.

This is a small diagnostic benchmark, not a replacement for the full LongMemEval gate. It keeps the selected source queries, their relevant documents, and the top confuser documents seen across diagnostic retrieval branches so targeted recall changes can be tested quickly.

Source diagnostic: `Evals/longmemeval_v2/runs/2026-04-27T03-33-40Z-coreml_default.wide200.branch-diagnostics.json`
Source run: `Evals/longmemeval_v2/runs/2026-04-27T03-33-40Z-coreml_default.json`

Classification counts:

- `multi_evidence_preservation`: 32

Taxonomy counts:

- `contextual-ellipsis`: 2
- `entity/alias`: 32
- `episodic/contextual`: 1
- `multi-evidence`: 32
- `temporal/count`: 27

Commands:

```sh
python3 Scripts/build_longmemeval_rescue.py --diagnostic <branch-diagnostics.json> --overwrite
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/longmemeval_multievidence_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/<focused-baseline>.json ./Evals/longmemeval_multievidence_v1/runs/<run>.json
```
