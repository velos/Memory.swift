# Memory.swift

`Memory.swift` is a pure Swift retrieval library inspired by [qmd](https://github.com/tobi/qmd), with its own Swift-native architecture and APIs.

## Inspiration

This project is explicitly inspired by [`tobi/qmd`](https://github.com/tobi/qmd). Credit goes to that project for the original ideas and workflow inspiration.

## Development Approach

Development of `Memory.swift` was fully built using the Codex agent harness and GPT-5.3-Codex on Extra High. Each feature / change was initiated by an interactively built plan, executed by the model after the plan was finalized.

## Features

- Hybrid retrieval: semantic + BM25 + recency
- Persistent SQLite index via GRDB
- Persistent contexts for reusable chunk sets
- Typed memory classification (`factual`, `procedural`, `episodic`, `semantic`, `emotional`, `social`, `contextual`, `temporal`)
- Default embedding backend with `NLContextualEmbedding`
- Optional Apple Intelligence query expansion and reranking on supported OS versions

## Targets

- `Memory` (core APIs and retrieval engine)
- `MemoryStorage` (GRDB schema + storage)
- `MemoryNaturalLanguage` (default embedding provider)
- `MemoryAppleIntelligence` (optional Apple Intelligence expansion + reranking providers)

## Quick Start (Natural Language backend)

```swift
import Foundation
import Memory
import MemoryNaturalLanguage

let dbURL = URL(fileURLWithPath: "/tmp/memory.sqlite")
let config = MemoryConfiguration.naturalLanguageDefault(databaseURL: dbURL)
let index = try MemoryIndex(configuration: config)

try await index.rebuildIndex(from: [URL(fileURLWithPath: "/path/to/docs")])
let results = try await index.search(SearchQuery(text: "swift concurrency actors"))
```

## Optional Apple Intelligence Query Expansion + Reranking

```swift
import MemoryAppleIntelligence

if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isAvailable {
    config.queryExpander = AppleIntelligenceQueryExpander()
    config.reranker = AppleIntelligenceReranker()
    config.memoryTyping.classifier = FallbackMemoryTypeClassifier(
        primary: AppleIntelligenceMemoryTypeClassifier(),
        fallback: HeuristicMemoryTypeClassifier()
    )
} else {
    config.memoryTyping.classifier = HeuristicMemoryTypeClassifier()
}
```

## Notes

- `Memory.swift` is inspired by qmd, but intentionally not qmd-CLI or qmd-data compatible.
- v1 uses exact cosine search over in-memory vectors; ANN can be added behind a future `VectorIndex` protocol.

Memory type can be manually set in markdown frontmatter:

```markdown
---
memory_type: episodic
---
```

## CLI (`memory`)

Build and run:

```bash
swift run memory --help
```

[qmd cli-style workflow](https://github.com/tobi/qmd#quick-start):

```bash
swift run memory collection add ~/notes --name notes
swift run memory collection add ~/Documents/meetings --name meetings
swift run memory collection add ~/work/docs --name docs

swift run memory context add memory://notes "Personal notes and ideas"
swift run memory context add memory://meetings "Meeting transcripts and notes"
swift run memory context add memory://docs "Work documentation"

swift run memory embed

swift run memory search "project timeline"
swift run memory search "project timeline" --memory-type temporal
swift run memory vsearch "how to deploy"
swift run memory query "quarterly planning process"

swift run memory get "meetings/2024-01-15.md"
swift run memory get "#1a2b"
swift run memory multi-get "journals/2025-05*.md"

swift run memory search "API" -c notes
swift run memory search "API" --all --files --min-score 0.3
```

## Eval Harness (`memory_eval`)

Generate datasets with MiniMax (Anthropic-compatible API):

```bash
python3 scripts/generate_eval_data_minimax.py --dataset-root ./Evals --env-file .env --overwrite
```

Generate datasets with Codex (`gpt-5.2`) in atomic batches:

```bash
python3 scripts/generate_eval_data_codex.py \
  --dataset-root ./Evals \
  --model gpt-5.2 \
  --storage-batch-size 6 \
  --documents-batch-size 8 \
  --queries-batch-size 10 \
  --resume
```

Initialize dataset templates:

```bash
swift run memory_eval init --dataset-root ./Evals
```

Run baseline eval:

```bash
swift run memory_eval run --profile baseline --dataset-root ./Evals
```

Run Apple-powered eval (classification + expansion/reranking):

```bash
swift run memory_eval run --profile full_apple --dataset-root ./Evals
```

Compare run outputs:

```bash
swift run memory_eval compare ./Evals/runs/*.json
```
