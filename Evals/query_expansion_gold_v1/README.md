# query_expansion_gold_v1

Curated rescue-style gold cases for the structured on-device expander stack.

This revision is mined from real recall misses and recoveries in:

- `Evals/general_v2`
- `Evals/longmemeval_v2`

This dataset is intentionally adversarial:

- each query is a real or lightly normalized user query from a larger corpus
- each case has one relevant document plus two distractors
- each case keeps the real target document and the top-20 false-positive documents from the source baseline run
- target documents are distilled from the source evidence but phrased so expansion still has to do real work

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
