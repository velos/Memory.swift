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
- CoreML embedding with LEAF-IR (384-dim, 23M params) and TinyBERT cross-encoder reranking (4.3 MB)
- Wider rerank candidate pool (40 minimum) for effective reranking
- Optional Apple Intelligence query expansion and reranking on supported OS versions

## Targets

- `Memory` (core APIs and retrieval engine)
- `MemoryStorage` (GRDB schema + storage)
- `MemoryNaturalLanguage` (default embedding provider)
- `MemoryCoreMLEmbedding` (CoreML LEAF-IR embeddings + TinyBERT reranker)
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

## Quick Start (CoreML LEAF-IR + TinyBERT reranker)

```swift
import Foundation
import Memory
import MemoryCoreMLEmbedding

let dbURL = URL(fileURLWithPath: "/tmp/memory.sqlite")
let embeddingModel = URL(fileURLWithPath: "Models/leaf-ir.mlpackage")
let rerankerModel = URL(fileURLWithPath: "Models/tinybert-reranker.mlpackage")

let embedder = try CoreMLEmbeddingProvider(modelURL: embeddingModel)
let reranker = try CoreMLReranker(modelURL: rerankerModel)

let config = MemoryConfiguration(
    databaseURL: dbURL,
    embeddingProvider: embedder,
    reranker: reranker
)
let index = try MemoryIndex(configuration: config)
```

## Tool-Oriented API (for agent harnesses)

`MemoryIndex` now exposes high-level methods for external tool integrations:

```swift
let saved = try await index.save(
    text: "Switched to SQLite for the prototype phase.",
    category: .decision,
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
- `.typed(category:)`

`RecallFeatures` is an `OptionSet` for hybrid-stage toggles (`semantic`, `lexical`, `tags`, `expansion`, `rerank`, `planner`).

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
- Semantic search runs via embedded `sqlite-vec` (`vec0`) in SQLite.

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

Run full profile matrix (all profiles, back-to-back):

```bash
swift run memory_eval run --dataset-root ./Evals
```

Run Apple tag-only eval (content tags as soft ranking signals):

```bash
swift run memory_eval run --profile apple_tags --dataset-root ./Evals
```

Run oracle ceiling eval (offline ranking upper bound from retrieved candidates):

```bash
swift run memory_eval run --profile oracle_ceiling --dataset-root ./Evals
```

Run Apple-powered eval (classification + expansion/reranking):

```bash
swift run memory_eval run --profile full_apple --dataset-root ./Evals
```

Compare run outputs:

```bash
swift run memory_eval compare ./Evals/runs/*.json
```

Convert LongMemEval-cleaned into eval format:

```bash
python3 scripts/convert_longmemeval_to_eval.py \
  --split oracle \
  --output-dir ./Evals/longmemeval
```

Run evals on LongMemEval:

```bash
swift run memory_eval run --profile baseline --dataset-root ./Evals/longmemeval
```

Eval caching defaults:
- Provider responses: `./Evals/cache/provider/eval_provider_cache.sqlite` (disable with `--no-cache`)
- Built suite indexes: `./Evals/cache/index/...` (disable with `--no-index-cache`)
