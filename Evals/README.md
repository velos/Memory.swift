# Memory Eval Datasets

This folder drives `memory_eval`, the CLI harness for measuring:
- Storage quality (classification + chunk content coverage)
- Recall quality (ranking relevance metrics)

## Files

### `storage_cases.jsonl`
One JSON object per line:

```json
{
  "id": "storage-1",
  "kind": "markdown",
  "text": "I felt frustrated during yesterday's outage review.",
  "expected_memory_type": "emotional",
  "required_spans": ["frustrated", "outage review"]
}
```

Fields:
- `id` (required): unique case ID.
- `kind` (optional): `markdown`, `code`, or `plainText`.
- `text` (required): source content to ingest.
- `expected_memory_type` (required): one of `factual`, `procedural`, `episodic`, `semantic`, `emotional`, `social`, `contextual`, `temporal`.
- `required_spans` (required): key substrings that should exist in stored chunks.

### `recall_documents.jsonl`
One document per line:

```json
{
  "id": "doc-a",
  "relative_path": "project/roadmap.md",
  "kind": "markdown",
  "text": "Q3 roadmap includes API stability work and a September launch milestone.",
  "memory_type": "temporal"
}
```

Fields:
- `id` (required): unique document ID.
- `relative_path` (optional): path under eval corpus root.
- `kind` (optional): `markdown`, `code`, or `plainText`.
- `text` (required): document body.
- `memory_type` (optional): if set and document is markdown, eval runner injects frontmatter manual override.

### `recall_queries.jsonl`
One query case per line:

```json
{
  "id": "q1",
  "query": "when is the launch milestone",
  "relevant_document_ids": ["doc-a"],
  "memory_types": ["temporal"]
}
```

Fields:
- `id` (required): unique query ID.
- `query` (required): retrieval query text.
- `relevant_document_ids` (required): IDs from `recall_documents.jsonl`.
- `memory_types` (optional): memory-type filter for typed retrieval eval.

## Running Evals

Generate datasets with MiniMax M2.5 via Anthropic-compatible API:

```bash
# .env
# ANTHROPIC_API_KEY=...
# ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic
# MINIMAX_MODEL=MiniMax-M2.5

python3 scripts/generate_eval_data_minimax.py --dataset-root ./Evals --env-file .env --overwrite
```

Generate datasets with Codex (ChatGPT-authenticated) using small atomic batches:

```bash
python3 scripts/generate_eval_data_codex.py \
  --dataset-root ./Evals \
  --model gpt-5.2 \
  --storage-batch-size 6 \
  --documents-batch-size 8 \
  --queries-batch-size 10 \
  --resume
```

Notes:
- Codex generation is now incremental: files are written as records are accepted.
- If the process fails, rerun with the same command and `--resume` to continue.
- Use `--overwrite` only when you want to restart from scratch.

Initialize example files:

```bash
swift run memory_eval init --dataset-root ./Evals
```

Run baseline:

```bash
swift run memory_eval run --profile baseline --dataset-root ./Evals
```

Run Apple profile (requires Apple Intelligence availability):

```bash
swift run memory_eval run --profile full_apple --dataset-root ./Evals
```

Compare runs:

```bash
swift run memory_eval compare ./Evals/runs/*.json
```

Convert LongMemEval-cleaned into this format:

```bash
python3 scripts/convert_longmemeval_to_eval.py \
  --split oracle \
  --output-dir ./Evals/longmemeval
```

## How To Fill Better Data

- Keep each `storage_cases` item focused on one dominant memory type.
- Make `required_spans` concrete and high-signal (facts, commitments, dates).
- Add hard negatives in `recall_documents` (similar wording but irrelevant).
- Use 1-3 relevant docs per query unless intentionally testing broad recall.
- Include queries that require `memory_types` filters when testing typed recall.
