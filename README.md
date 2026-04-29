# Memory.swift

`Memory.swift` is a pure Swift retrieval and agent-memory library for Apple platforms. It combines a SQLite-backed canonical memory store, hybrid retrieval, and optional NaturalLanguage, CoreML, and Apple Intelligence providers behind a single Swift-native API.

## Inspiration

This project is explicitly inspired by [`tobi/qmd`](https://github.com/tobi/qmd). Credit goes to that project for the original ideas and workflow inspiration.

## Requirements

- iOS 18+
- macOS 15+
- Xcode 16.0+ / Swift 6.2+

## Features

- Hybrid retrieval: semantic + BM25 + recency
- Persistent SQLite index via SQLite and `sqlite-vec`
- Canonical `memories` table with deterministic update/supersede semantics for agent memories
- Persistent contexts for reusable chunk sets
- Default embedding backend with `NLContextualEmbedding`
- CoreML-first on-device path with bundled embedding model support
- Agent memory model: `profile`, `fact`, `decision`, `commitment`, `episode`, `procedure`, `handoff`
- Fixed facet tags plus open `entities` and `topics` for retrieval
- Optional Apple Intelligence augmentation on supported OS versions

## Package Products

- `Memory`: core indexing, retrieval, and agent-facing APIs
- `MemoryNaturalLanguage`: NaturalLanguage-based embedding defaults, tokenizers, and query analysis
- `MemoryCoreMLEmbedding`: CoreML embedding and reranker providers
- `MemoryAppleIntelligence`: optional FoundationModels-based query expansion, reranking, and content tagging
- `memory`: local CLI for indexing, querying, and benchmark bridge experiments
- `memory_eval`: eval harness for storage, recall, query expansion, agent-memory behavior, and regression gates

`MemoryStorage` is intentionally kept as an internal implementation target. External integrations should depend on the `Memory` product and, optionally, one provider product.

## Installation

Until tagged releases are available, depend on `main`:

```swift
dependencies: [
    .package(url: "https://github.com/velos/Memory.swift.git", branch: "main")
]
```

Most integrations need `Memory` plus one provider product:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "Memory", package: "Memory.swift"),
        .product(name: "MemoryNaturalLanguage", package: "Memory.swift"),
    ]
)
```

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

## Quick Start (CoreML default)

Provide your own compiled model URLs from the app bundle or local filesystem. The model files in this repository are reference assets, not package resources exposed by the library product.

```swift
import Foundation
import Memory
import MemoryCoreMLEmbedding

let dbURL = URL(fileURLWithPath: "/tmp/memory.sqlite")
let embeddingModel = URL(fileURLWithPath: "Models/embedding-v1.mlpackage")
let config = try MemoryConfiguration.coreMLDefault(
    databaseURL: dbURL,
    models: CoreMLDefaultModels(
        embedding: embeddingModel
    )
)
let index = try MemoryIndex(configuration: config)
```

`coreMLDefault` is the shipped on-device path: CoreML embeddings, hybrid retrieval, heuristic structured expansion, NaturalLanguage query analysis, and no neural reranker in the default hot path. The CLI and eval harness resolve `Models/embedding-v1.mlpackage` by default when run from this repository.

## Recommended Public API Surface

Most integrations only need:

- `MemoryIndex` for indexing and retrieval
- `MemoryConfiguration` plus one embedding provider product
- `rebuildIndex`, `syncDocuments`, and `removeDocuments` for document lifecycle
- `save`, `extract`, `ingest`, and `recall` for agent memory workflows
- `memorySearch` and `memoryGet` for tool-style retrieval
- customization protocols (`EmbeddingProvider`, `Reranker`, `StructuredQueryExpander`, `MemoryExtractor`, `RecallPlanner`) only when you are swapping in your own providers

## Tool-Oriented API

`MemoryIndex` now exposes high-level methods for external tool integrations:

```swift
let saved = try await index.save(
    text: "Switched to SQLite for the prototype phase.",
    kind: .decision,
    importance: 0.9
)

let extracted = try await index.extract(
    from: [
        ConversationMessage(role: .user, content: "Action item: add migration tests."),
    ]
)
let ingestResult = try await index.ingest(extracted)

let recall = try await index.recall(
    mode: .hybrid(query: "What do we know about migration tests?"),
    features: .hybridDefault
)
```

Supported recall modes:
- `.hybrid(query:)`
- `.recent`
- `.important`
- `.kind(_:)`

`RecallFeatures` is an `OptionSet` for hybrid-stage toggles (`semantic`, `lexical`, `tags`, `expansion`, `rerank`, `planner`).

## Agent Integration API (`memory_search` + `memory_get`)

For OpenClaw/qmd-style agent loops, `MemoryIndex` now exposes direct reference retrieval and document fetch APIs:

```swift
let refs = try await index.memorySearch(
    query: "What budget did the user ask for on apartment hunting?",
    limit: 10,
    features: .hybridDefault,
    dedupeDocuments: true,
    includeLineRanges: true
)

// Feed `refs` to the LLM, then fetch exact supporting text for selected paths.
if let first = refs.first {
    let full = try await index.memoryGet(path: first.documentPath)
    let focused = try await index.memoryGet(reference: first)
}
```

`memorySearch` returns lightweight `MemorySearchReference` values:
- `documentPath`, `title`, `snippet`
- optional 1-based `lineRange` when inferable
- ranking score breakdown

`memoryGet` resolves absolute, exact indexed, and suffix paths (for example `profile.md`) and returns:
- full document by default
- or a line-sliced `MemoryGetResponse` when `lineRange` is provided
- automatic fallback to indexed chunk reconstruction for in-memory `memory://...` entries

## Optional Apple Intelligence Query Expansion + Reranking

```swift
import Foundation
import Memory
import MemoryAppleIntelligence
import MemoryNaturalLanguage

let dbURL = URL(fileURLWithPath: "/tmp/memory.sqlite")
var config = MemoryConfiguration.naturalLanguageDefault(databaseURL: dbURL)

if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isAvailable {
    config.structuredQueryExpander = AppleIntelligenceStructuredQueryExpander()
    config.reranker = AppleIntelligenceReranker()
}

if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isContentTaggingAvailable {
    config.contentTagger = AppleIntelligenceContentTagger()
}
```

## Notes

- `Memory.swift` is inspired by qmd, but intentionally not qmd-CLI or qmd-data compatible.
- Semantic search runs via embedded `sqlite-vec` (`vec0`) in SQLite.
- Canonical agent memory is modeled with `MemoryKind`, `FacetTag`, `entities`, and `topics`.

## CLI (`memory`)

Build and run:

```bash
swift run memory --help
```

The CLI includes `memory serve`, a persistent JSON-lines bridge used for local benchmark adapters. It avoids repeated process startup and CoreML model loading during high-volume retrieval diagnostics.

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

The eval harness is the release gate for Memory.swift behavior. Prefer `coreml_default` for shipped-path validation.

```bash
swift run memory_eval validate-datasets
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/general_v2 --no-cache --no-index-cache
swift run memory_eval compare ./Evals/general_v2/runs/<baseline>.json ./Evals/general_v2/runs/<candidate>.json
```

Release gate:

```bash
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/memory_schema_gold_v2 --no-cache --no-index-cache
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/agent_memory_gold_v1 --no-cache --no-index-cache
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/general_v2 --no-cache --no-index-cache
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/longmemeval_v2 --no-cache --no-index-cache
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/query_expansion_gold_v1 --no-cache --no-index-cache
swift run memory_eval gate --baseline ./Evals/baselines/current.json <five-json-reports>
```

Useful diagnostic commands:

```bash
swift run memory_eval diagnose-longmemeval \
  --profile coreml_default \
  --dataset-root ./Evals/longmemeval_v2 \
  --source-run ./Evals/longmemeval_v2/runs/<run>.json \
  --scope misses \
  --wide-limit 100
```

Tracked eval suites:

- `Evals/memory_schema_gold_v2`: canonical write-path benchmark for kind/status/facet/entity/topic/update behavior
- `Evals/agent_memory_gold_v1`: no-write, extraction, update/supersede/resolve, active-state, and recall behavior
- `Evals/general_v2`: broad retrieval gate for the shipped hybrid path
- `Evals/longmemeval_v2`: long-horizon conversational recall benchmark
- `Evals/query_expansion_gold_v1`: structured query-expansion benchmark
- Focused gates: `longmemeval_rescue_v1`, `longmemeval_ranking_v1`, `longmemeval_multievidence_v1`, `query_expansion_rescue_v1`, `agent_memory_pressure_v1`
- Exploratory storage robustness: `storage_heldout_v1`

Eval caching defaults:
- Provider responses: `<dataset-root>/cache/provider/eval_provider_cache.sqlite` (disable with `--no-cache`)
- Built suite indexes: `<dataset-root>/cache/index/...` (disable with `--no-index-cache`)

See `Evals/README.md` and `.agents/skills/memory-evals/SKILL.md` for dataset generation, audit, focused-slice, and baseline maintenance workflows. See `Docs/agent-memory-benchmark.md` for local external Agent Memory Benchmark notes.

## Autoresearch

The in-repo autonomous optimization loop for CoreML retrieval models lives in `./Autoresearch`.

```bash
cd Autoresearch
uv sync
uv run prepare.py
uv run train.py
```

See `./Autoresearch/README.md` and `./Autoresearch/program.md` for the workflow and guardrails.
