# QMDKit

`QMDKit` is a pure Swift, qmd-inspired retrieval library.

## Features

- Hybrid retrieval: semantic + BM25 + recency
- Persistent SQLite index via GRDB
- Persistent contexts for reusable chunk sets
- Default embedding backend with `NLContextualEmbedding`
- Optional MLX target for custom local embedding stacks

## Targets

- `QMDKit` (core APIs and retrieval engine)
- `QMDKitStorage` (GRDB schema + storage)
- `QMDKitNaturalLanguage` (default embedding provider)
- `QMDKitMLX` (optional MLX embedding provider wrapper)

## Quick Start (Natural Language backend)

```swift
import Foundation
import QMDKit
import QMDKitNaturalLanguage

let dbURL = URL(fileURLWithPath: "/tmp/qmdkit.sqlite")
let config = QMDConfiguration.naturalLanguageDefault(databaseURL: dbURL)
let index = try QMDIndex(configuration: config)

try await index.rebuildIndex(from: [URL(fileURLWithPath: "/path/to/docs")])
let results = try await index.search(SearchQuery(text: "swift concurrency actors"))
```

## Notes

- This package is qmd-inspired but intentionally not qmd-CLI or qmd-data compatible.
- v1 uses exact cosine search over in-memory vectors; ANN can be added behind a future `VectorIndex` protocol.

## CLI (`qmd.swift`)

Build and run:

```bash
swift run qmd.swift --help
```

qmd-style workflow:

```bash
qmd.swift collection add ~/notes --name notes
qmd.swift collection add ~/Documents/meetings --name meetings
qmd.swift collection add ~/work/docs --name docs

qmd.swift context add qmd://notes "Personal notes and ideas"
qmd.swift context add qmd://meetings "Meeting transcripts and notes"
qmd.swift context add qmd://docs "Work documentation"

qmd.swift embed

qmd.swift search "project timeline"
qmd.swift vsearch "how to deploy"
qmd.swift query "quarterly planning process"

qmd.swift get "meetings/2024-01-15.md"
qmd.swift get "#1a2b"
qmd.swift multi-get "journals/2025-05*.md"

qmd.swift search "API" -c notes
qmd.swift search "API" --all --files --min-score 0.3
```
