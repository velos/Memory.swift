# query_expansion_rescue_v1

Regression benchmark for query-expansion rescue behavior.

This dataset is mined from real recall misses and low-rank cases, not model-invented prompts. Each case keeps the source query, the known relevant document IDs, a per-case candidate slice, and failure-taxonomy metadata so retrieval changes can be diagnosed by failure mode.

Sources:

- `Evals/general_v2` from `Evals/general_v2/runs/2026-03-27T15-42-53Z-coreml_default.json` (16 cases)
- `Evals/longmemeval_v2` from `Evals/longmemeval_v2/runs/2026-03-16T01-23-59Z-coreml_leaf_ir.json` (16 cases)

Failure taxonomy counts:

- `contextual_ellipsis`: 2
- `count_aggregation`: 8
- `empty_retrieval`: 16
- `lexical_mismatch`: 5
- `multi_evidence`: 14
- `retrieval_miss`: 32
- `temporal_reasoning`: 14

Generated files:

- `cases.jsonl`: query-expansion cases with expected lexical/topic/entity hints and source failure metadata.
- `recall_documents.jsonl`: the union of candidate documents needed by the cases.
- `manifest.json`: source run, selection, and taxonomy metadata.

Commands:

```sh
python3 Scripts/build_query_expansion_rescue.py --overwrite
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/query_expansion_rescue_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/query_expansion_rescue.json <candidate-run.json>
```

Primary metrics to watch are retrieval expanded Hit@K, expanded MRR@K, and MRR delta. Hit rate may saturate on small slices, so MRR delta is the sharper signal for whether expansion moves relevant memories closer to the top.
