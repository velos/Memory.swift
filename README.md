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
- CoreML-first on-device path with LEAF-IR embeddings
- Agent memory model: `profile`, `fact`, `decision`, `commitment`, `episode`, `procedure`, `handoff`
- Fixed facet tags plus open `entities` and `topics` for retrieval
- Optional Apple Intelligence augmentation on supported OS versions

## Package Products

- `Memory`: core indexing, retrieval, and agent-facing APIs
- `MemoryNaturalLanguage`: NaturalLanguage-based embedding defaults, tokenizers, and query analysis
- `MemoryCoreMLEmbedding`: CoreML embedding and reranker providers
- `MemoryAppleIntelligence`: optional FoundationModels-based query expansion, reranking, and content tagging

`MemoryStorage` is intentionally kept as an internal implementation target. External integrations should depend on the `Memory` product and, optionally, one provider product.

## Installation

Until tagged releases are available, depend on `main`:

```swift
dependencies: [
    .package(url: "https://github.com/zac/Memory.swift.git", branch: "main")
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

`coreMLDefault` is the shipped on-device path: LEAF-IR embeddings, hybrid retrieval, and no neural reranker in the default hot path.

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
import MemoryAppleIntelligence

if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isAvailable {
    config.queryExpander = AppleIntelligenceQueryExpander()
    config.reranker = AppleIntelligenceReranker()
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
python3 Scripts/generate_eval_data_minimax.py --dataset-root ./Evals --env-file .env --overwrite
```

Generate datasets with Codex (`gpt-5.2`) in atomic batches:

```bash
python3 Scripts/generate_eval_data_codex.py \
  --dataset-root ./Evals \
  --model gpt-5.2 \
  --storage-batch-size 6 \
  --documents-batch-size 8 \
  --queries-batch-size 10 \
  --resume
```

Tag or augment existing eval datasets with Codex:

```bash
python3 Scripts/tag_eval_data_codex.py \
  --dataset-root ./Evals/longmemeval_v2 \
  --mode query-tags \
  --model gpt-5-codex
```

Tag or augment existing eval datasets with MiniMax:

```bash
python3 Scripts/tag_eval_data_minimax.py \
  --dataset-root ./Evals/longmemeval_v2 \
  --mode query-tags \
  --env-file .env
```

Build audit packets and run model-assisted review:

```bash
python3 Scripts/build_audit_packet.py \
  --dataset-root ./Evals/general_v2 \
  --dataset-root ./Evals/longmemeval_v2

python3 Scripts/audit_eval_data.py \
  --packet ./Evals/_audit/general_v2/packet.jsonl \
  --backend opencode \
  --model opencode/nemotron-3-super-free

python3 Scripts/merge_audit_results.py \
  --packet ./Evals/_audit/general_v2/packet.jsonl
```

Build additional exploratory corpora locally:

```bash
python3 Scripts/convert_multidoc2dial_to_eval.py --output-dir ./Explorations/Evals/raw_multidoc2dial
python3 Scripts/convert_repliqa_to_eval.py --output-dir ./Explorations/Evals/raw_repliqa --splits repliqa_0
python3 Scripts/convert_qasper_to_eval.py --output-dir ./Explorations/Evals/raw_qasper --splits train,dev
python3 Scripts/convert_swebench_verified_to_eval.py --output-dir ./Explorations/Evals/raw_swebench_verified --max-instances 250
```

Merge local exploratory corpora into a staged dataset:

```bash
python3 Scripts/merge_eval_corpora.py \
  --dataset-name general_v2 \
  --output-dir ./Explorations/Evals/general_v2 \
  --source multidoc2dial=./Explorations/Evals/raw_multidoc2dial \
  --source repliqa=./Explorations/Evals/raw_repliqa
```

Initialize dataset templates:

```bash
swift run memory_eval init --dataset-root ./Evals
```

Run baseline eval:

```bash
swift run memory_eval run --profile nl_baseline --dataset-root ./Evals
```

Recommended benchmark roles:
- `Evals/memory_schema_gold_v2`: canonical write-path benchmark for `MemoryKind`, `MemoryStatus`, `FacetTag`, `entities`, `topics`, and update behavior
- `Evals/general_v2`: broad retrieval gate for the shipped hybrid search path
- `Evals/longmemeval_v2`: long-horizon conversational recall benchmark; treat it as recall-first rather than a canonical write-path benchmark
- `Evals/query_expansion_gold_v1`: targeted structured-expansion pressure test for regressions and rescue cases

Everything else should be treated as optional exploration material. Keep bulky temporary corpora, audit packets, and experimental model assets under the gitignored `Explorations/` tree rather than tracking them in the main repo history.

Run the supported profile set:

```bash
swift run memory_eval run --dataset-root ./Evals
```

Run CoreML-first eval:

```bash
swift run memory_eval run --profile coreml_default --dataset-root ./Evals
```

Run oracle ceiling eval (offline ranking upper bound from retrieved candidates):

```bash
swift run memory_eval run --profile oracle_ceiling --dataset-root ./Evals
```

Run Apple-augmented eval:

```bash
swift run memory_eval run --profile apple_augmented --dataset-root ./Evals
```

Compare run outputs:

```bash
swift run memory_eval compare ./Evals/runs/*.json
```

Convert LongMemEval-cleaned into eval format:

```bash
python3 Scripts/convert_longmemeval_to_eval.py \
  --split oracle \
  --output-dir ./Explorations/Evals/longmemeval_v2
```

Run evals on LongMemEval:

```bash
swift run memory_eval run --profile coreml_default --dataset-root ./Evals/longmemeval_v2
```

Eval caching defaults:
- Provider responses: `./Evals/cache/provider/eval_provider_cache.sqlite` (disable with `--no-cache`)
- Built suite indexes: `./Evals/cache/index/...` (disable with `--no-index-cache`)

## Autoresearch

The in-repo autonomous optimization loop for CoreML retrieval models lives in `./Autoresearch`.

```bash
cd Autoresearch
uv sync
uv run prepare.py
uv run train.py
```

See `./Autoresearch/README.md` and `./Autoresearch/program.md` for the workflow and guardrails.
