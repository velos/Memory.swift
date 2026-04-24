# query_expansion_gold_v1

Curated rescue-style gold cases for the structured on-device expander stack.

This revision is mined from real recall misses and recoveries in:

- `Evals/general_v2`
- `Evals/longmemeval_v2`

The general-query slice was refreshed again from the 2026-03-27
`coreml_default` miss report so that the baseline-vs-expanded retrieval
check does not only exercise already-recovered examples.

This dataset is intentionally adversarial:

- each query is a real or lightly normalized user query from a larger corpus
- each case keeps the real target document and a case-specific slice built from the source run that motivated the case
- the long-memory half is now sourced from `coreml_default` misses and near-misses on `longmemeval_v2`, not from the weaker baseline profile
- long-memory cases now emphasize temporal, schedule, count, and multi-event retrieval prompts that still pressure the stronger shipped path
- per-case candidate slices are intentionally larger than the shared pooled corpus so retrieval impact is measured against the original confusion set, not an over-distilled toy pool

Each row in `cases.jsonl` includes:

- `query`
- `expected_lexical_terms`
- `expected_semantic_phrases`
- `expected_hyde_anchors`
- `expected_facets`
- `expected_entities`
- `expected_topics`
- `relevant_document_ids`
- `candidate_document_ids` (optional per-case slice for retrieval impact checks)

Design goals:

- baseline search should miss a meaningful slice of cases taken from real corpora
- heuristic expansion should recover some of those misses
- semantic coverage, facet hints, entity extraction, and topic extraction should all be measurable on the same set

This corpus is intended for deterministic provider checks and retrieval-impact ablations across:

- no expansion
- heuristic structured expansion
- Apple structured expansion
- CoreML structured expansion
