# Memory.swift

`Memory.swift` is a pure Swift retrieval library inspired by [qmd](https://github.com/tobi/qmd), with its own Swift-native architecture and APIs.

## Inspiration

This project is explicitly inspired by [`tobi/qmd`](https://github.com/tobi/qmd). Credit goes to that project for the original ideas and workflow inspiration.

## Features

- Hybrid retrieval: semantic + BM25 + recency
- Persistent SQLite index via GRDB
- Persistent contexts for reusable chunk sets
- Default embedding backend with `NLContextualEmbedding`
- Optional MLX target for custom local embedding stacks

## Targets

- `Memory` (core APIs and retrieval engine)
- `MemoryStorage` (GRDB schema + storage)
- `MemoryNaturalLanguage` (default embedding provider)
- `MemoryMLX` (optional MLX embedding provider wrapper)

## Quick Start (Natural Language backend)

```swift
import Foundation
import Memory
import MemoryNaturalLanguage

let dbURL = URL(fileURLWithPath: "/tmp/memory.sqlite")
let config = QMDConfiguration.naturalLanguageDefault(databaseURL: dbURL)
let index = try QMDIndex(configuration: config)

try await index.rebuildIndex(from: [URL(fileURLWithPath: "/path/to/docs")])
let results = try await index.search(SearchQuery(text: "swift concurrency actors"))
```

## Notes

- `Memory.swift` is inspired by qmd, but intentionally not qmd-CLI or qmd-data compatible.
- v1 uses exact cosine search over in-memory vectors; ANN can be added behind a future `VectorIndex` protocol.

## CLI (`memory`)

Build and run:

```bash
swift run memory --help
```

memory-style workflow:

```bash
memory collection add ~/notes --name notes
memory collection add ~/Documents/meetings --name meetings
memory collection add ~/work/docs --name docs

memory context add memory://notes "Personal notes and ideas"
memory context add memory://meetings "Meeting transcripts and notes"
memory context add memory://docs "Work documentation"

memory embed

memory search "project timeline"
memory vsearch "how to deploy"
memory query "quarterly planning process"

memory get "meetings/2024-01-15.md"
memory get "#1a2b"
memory multi-get "journals/2025-05*.md"

memory search "API" -c notes
memory search "API" --all --files --min-score 0.3
```
