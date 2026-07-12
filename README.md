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

## Package Product and Traits

- `Memory`: indexing, retrieval, agent-facing APIs, provider APIs, and SwiftUI debug views
- `memory`: local CLI for indexing, querying, and benchmark bridge experiments
- `memory_eval`: eval harness for storage, recall, query expansion, agent-memory behavior, and regression gates

Provider families are controlled with SwiftPM traits:

- `MemoryNaturalLanguage`: NaturalLanguage-backed embedding defaults, tokenizers, and query analysis. This is the only default trait.
- `CoreMLEmbedding`: CoreML embedding, tokenizer, reranker, and bundled default model resources.
- `MemoryAppleIntelligence`: optional FoundationModels-based query expansion, reranking, and content tagging.

`MemoryStorage` is intentionally kept as an internal implementation target. External integrations should depend on the `Memory` product and enable only the traits they need.

## Installation

Until tagged releases are available, depend on `main`:

```swift
dependencies: [
    .package(url: "https://github.com/velos/Memory.swift.git", branch: "main")
]
```

Most integrations use the default `MemoryNaturalLanguage` trait:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "Memory", package: "Memory.swift"),
    ]
)
```

Opt into CoreML or Apple Intelligence traits from the package dependency declaration when needed:

```swift
.package(
    url: "https://github.com/velos/Memory.swift.git",
    branch: "main",
    traits: ["MemoryNaturalLanguage", "CoreMLEmbedding"]
)
```

## Quick Start (Natural Language backend)

```swift
import Foundation
import Memory

let dbURL = URL(fileURLWithPath: "/tmp/memory.sqlite")
let config = MemoryConfiguration.naturalLanguageDefault(databaseURL: dbURL)
let index = try MemoryIndex(configuration: config)

try await index.rebuildIndex(from: [URL(fileURLWithPath: "/path/to/docs")])
let results = try await index.search(SearchQuery(text: "swift concurrency actors"))
```

## Quick Start (CoreML default)

Enable the `CoreMLEmbedding` trait to use the bundled default embedding model and tokenizer resources:

```swift
import Foundation
import Memory

let dbURL = URL(fileURLWithPath: "/tmp/memory.sqlite")
let config = try MemoryConfiguration.coreMLDefault(databaseURL: dbURL)
let index = try MemoryIndex(configuration: config)
```

`coreMLDefault` is the shipped on-device path: CoreML embeddings, hybrid retrieval, heuristic structured expansion, NaturalLanguage query analysis when that trait is also enabled, and no neural reranker in the default hot path. Advanced integrations can still pass explicit `.mlmodelc`, `.mlpackage`, or `.mlmodel` URLs through `CoreMLDefaultModels`.

## Recommended Public API Surface

Most integrations only need:

- `MemoryIndex` for indexing and retrieval
- `MemoryConfiguration` plus a trait-enabled or custom embedding provider
- `rebuildIndex`, `syncDocuments`, and `removeDocuments` for document lifecycle
- `save`, `extract`, `ingest`, and `recall` for agent memory workflows
- `capture`, `prepareContext`, `recordSignal`, and `runMaintenance` for higher-level agent memory workflows
- `memorySearch` and `memoryGet` for tool-style retrieval
- customization protocols (`EmbeddingProvider`, `Reranker`, `StructuredQueryExpander`, `MemoryExtractor`, `RecallPlanner`) only when you are swapping in your own providers

## Agent Memory Workflows

Memory.swift exposes three generic workflows for host apps that want durable agent memory without adopting a host-specific schema.

Capture extracts durable user-focused memories from conversation turns and can run in preview or ingest mode:

```swift
let capture = try await index.capture(
    MemoryCaptureRequest(
        messages: [
            ConversationMessage(role: .user, content: "I live in sf, what should I do tonight?"),
        ],
        mode: .ingest,
        policy: .agentDefault,
        sourceID: "session-123"
    )
)
```

Captured `MemoryCandidate` and stored `MemoryRecord` values can include a `subject` and original-message `evidence`. The default agent policy focuses on user-authored durable facts, rejects assistant capability/refusal text, and keeps embedded declarations separate from raw questions.

Context preparation retrieves bounded memory context for the next model turn:

```swift
let context = try await index.prepareContext(
    MemoryContextRequest(
        messages: recentMessages,
        budget: MemoryContextBudget(maxReferences: 8, maxTokens: 1_024)
    )
)
```

`context.contextBlock` is wrapped in `<memory_context>...</memory_context>` delimiters and explicitly framed as untrusted retrieved context with compact source references. Incoming messages that still contain a previously injected block are sanitized before retrieval, so host apps can prepend the block to the next turn without polluting future queries. Path-scoped `MemoryContextHint` values can be managed with `setContextHint`, `listContextHints`, and `removeContextHint`; matching hints are surfaced through `memorySearch`, `memoryGet`, and prepared context responses.

Maintenance records recall, capture, compaction, explicit, and maintenance signals, then conservatively proposes consolidations:

```swift
try await index.recordSignal(
    MemorySignal(kind: .recall, memoryID: memoryID, query: "dinner ideas")
)

let preview = try await index.runMaintenance(
    MemoryMaintenanceRequest(mode: .preview)
)
```

Apply mode ingests only candidates that pass the request thresholds. Compaction summaries can be passed as `MemoryCompactionObservation` inputs, but durable promotions still require original message evidence.

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

## SwiftUI Debug View

SwiftUI apps can mount a read-mostly memory inspector wherever debug UI belongs:

```swift
import Memory
import SwiftUI

MemoryDebugView(index: index)
```

`MemoryDebugView` lists canonical memories with paging, lexical search, kind/status filters, sort controls, metadata detail views, and an archive action. The underlying `MemoryIndex.debugMemories(_:)` API is side-effect-free and does not increment memory access counts.

See `Examples/DebugViewApp` for a minimal iOS app that saves demo memories and mounts the debug view.

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

The CLI includes `memory serve`, a persistent JSON-lines bridge used for local benchmark adapters. It avoids repeated process startup and CoreML model loading during high-volume retrieval diagnostics, and search requests can pass `contextTokenBudget`, `perDocumentTokenBudget`, and `contextPackingOrder` to exercise capped context packing.

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
swift run memory_eval retrieval-diagnostics \
  --profile coreml_default \
  --dataset-root ./Evals/longmemeval_v2 \
  --candidate-pool-depth 40 \
  --context-token-budget 4096 \
  --per-document-token-budget 384 \
  --context-packing-order rank \
  --no-cache \
  --no-index-cache

# Reports include packed-context Hit/Recall/MRR/nDCG, score-sorted packed
# sidecar metrics, candidate-pool Hit/Recall, candidate-only miss rate, and
# candidate-generation miss rate.
# Use --context-packing-order score only as an opt-in experiment; rank is the
# recall-preserving default.

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

The in-repo autonomous optimization loops for CoreML memory models live in
`./Autoresearch`.

```bash
cd Autoresearch
uv sync
uv run retrieval/prepare.py
uv run retrieval/train.py
uv run reranker/prepare.py
uv run reranker/train.py
```

See `./Autoresearch/README.md`, `./Autoresearch/retrieval/program.md`, and
`./Autoresearch/reranker/program.md` for the workflow and guardrails.
