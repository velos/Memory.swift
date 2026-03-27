import Foundation
import Testing
@testable import Memory

struct MemoryIndexTests {
    @Test
    func searchReturnsRelevantDocument() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("swift.md"),
            "# Swift\nActors isolate mutable state and Swift concurrency uses async await."
        )
        try writeFile(
            docs.appendingPathComponent("garden.md"),
            "# Garden\nTomatoes need sunlight and healthy soil."
        )

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: MockEmbeddingProvider(),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 120, overlapTokenCount: 20)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let results = try await index.search(SearchQuery(text: "swift actor isolation", limit: 3))

        #expect(results.isEmpty == false)
        #expect(results.first?.documentPath.contains("swift.md") == true)
    }

    @Test
    func contextCrudRoundTrip() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(docs.appendingPathComponent("a.md"), "alpha beta gamma delta")
        try writeFile(docs.appendingPathComponent("b.md"), "orange kiwi apple pear")

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: MockEmbeddingProvider(),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 100, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let initial = try await index.search(SearchQuery(text: "alpha", limit: 5))
        #expect(initial.isEmpty == false)

        let contextID = try await index.createContext(name: "focus")
        try await index.addToContext(contextID, chunkIDs: [initial[0].chunkID])

        let contextRows = try await index.listContextChunks(contextID)
        #expect(contextRows.count == 1)

        let scoped = try await index.search(SearchQuery(text: "alpha", limit: 5, contextID: contextID))
        #expect(scoped.count == 1)
        #expect(scoped.first?.chunkID == initial.first?.chunkID)

        try await index.clearContext(contextID)
        let cleared = try await index.listContextChunks(contextID)
        #expect(cleared.isEmpty)
    }

    @Test
    func recencyBoostBreaksTies() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        let now = Date()
        let old = now.addingTimeInterval(-90 * 86_400)

        try writeFile(
            docs.appendingPathComponent("new.md"),
            "alpha beta gamma",
            modifiedAt: now
        )
        try writeFile(
            docs.appendingPathComponent("old.md"),
            "alpha beta gamma",
            modifiedAt: old
        )

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 100, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let results = try await index.search(SearchQuery(text: "alpha", limit: 2))
        #expect(results.count == 2)
        #expect(results.first?.documentPath.contains("new.md") == true)
    }

    @Test
    func queryExpansionCanLiftRelevantDocument() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("deploy.md"),
            "deployment rollout checklist and service cutover notes"
        )
        try writeFile(
            docs.appendingPathComponent("other.md"),
            "gardening tomatoes soil watering routine"
        )

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            structuredQueryExpander: StaticStructuredQueryExpander(lexicalQueries: ["deployment rollout"]),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 120, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let query = SearchQuery(
            text: "release process",
            limit: 2,
            rerankLimit: 0,
            expansionLimit: 2
        )
        let results = try await index.search(query)

        #expect(results.isEmpty == false)
        #expect(results.first?.documentPath.contains("deploy.md") == true)
    }

    @Test
    func strongLexicalSignalSkipsQueryExpansion() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("runbook.md"),
            "incident response runbook. The incident response runbook covers incident response runbook procedures. Follow the incident response runbook for incident response runbook compliance. incident response runbook steps are critical."
        )
        try writeFile(
            docs.appendingPathComponent("notes.md"),
            "general observations and feedback collected during the quarterly planning session"
        )
        for i in 0..<6 {
            try writeFile(
                docs.appendingPathComponent("filler\(i).md"),
                "unrelated content about topic number \(i) covering various subjects"
            )
        }

        let expander = RecordingStructuredQueryExpander(lexicalQueries: ["deployment rollout checklist"])
        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            structuredQueryExpander: expander,
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 120, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        _ = try await index.search(
            SearchQuery(
                text: "incident response runbook",
                limit: 3,
                rerankLimit: 0,
                expansionLimit: 2
            )
        )

        let calls = await expander.calls()
        #expect(calls == 0)
    }

    @Test
    func semanticQueryEmbeddingsAreBatchedAcrossExpansions() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("deploy.md"),
            "deployment rollout checklist and service cutover notes"
        )
        try writeFile(
            docs.appendingPathComponent("ops.md"),
            "incident response timeline and postmortem follow-up items"
        )

        let embeddingProvider = CountingEmbeddingProvider()
        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: embeddingProvider,
            structuredQueryExpander: StaticStructuredQueryExpander(
                lexicalQueries: ["deployment rollout", "service cutover"]
            ),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 120, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])
        await embeddingProvider.resetStats()

        _ = try await index.search(
            SearchQuery(
                text: "release process",
                limit: 3,
                semanticCandidateLimit: 30,
                lexicalCandidateLimit: 0,
                rerankLimit: 0,
                expansionLimit: 2
            )
        )

        let stats = await embeddingProvider.stats()
        #expect(stats.singleCalls == 0)
        #expect(stats.batchSizes == [3])
    }

    @Test
    func heuristicStructuredExpanderProducesDeterministicStructuredOutput() async throws {
        let expander = HeuristicStructuredQueryExpander()
        let query = SearchQuery(text: "How do we ship Memory.swift on time with sqlite-vec?")
        let analysis = QueryAnalysis(
            entities: [
                MemoryEntity(
                    label: .project,
                    value: "Memory.swift",
                    normalizedValue: "memory.swift",
                    confidence: 0.9
                ),
                MemoryEntity(
                    label: .tool,
                    value: "sqlite-vec",
                    normalizedValue: "sqlite-vec",
                    confidence: 0.88
                ),
            ],
            keyTerms: ["ship", "memory.swift", "sqlite-vec", "time"],
            facetHints: [
                FacetHint(tag: .project, confidence: 0.92, isExplicit: true),
                FacetHint(tag: .timeSensitive, confidence: 0.91, isExplicit: true),
            ],
            topics: ["release timeline", "sqlite vec rollout"],
            isHowToQuery: true
        )

        let expansion = try await expander.expand(query: query, analysis: analysis, limit: 5)

        #expect(expansion.lexicalQueries.isEmpty == false)
        #expect(expansion.semanticQueries.isEmpty == false)
        #expect(expansion.hypotheticalDocuments.count == 1)
        #expect(expansion.facetHints.contains(where: { $0.tag == .project && $0.isExplicit }))
        #expect(expansion.entities.contains(where: { $0.normalizedValue == "memory.swift" }))
        #expect(expansion.topics.contains(where: { $0.contains("release") || $0.contains("sqlite") }))
    }

    @Test
    func positionAwareBlendingUsesRerankSignal() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(docs.appendingPathComponent("a.md"), "alpha planning roadmap")
        try writeFile(docs.appendingPathComponent("b.md"), "alpha planning roadmap")

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            reranker: ClosureReranker(scoreForCandidate: { result in
                result.documentPath.contains("b.md") ? 1.0 : 0.1
            }),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 100, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let results = try await index.search(
            SearchQuery(
                text: "alpha planning",
                limit: 2,
                rerankLimit: 2,
                expansionLimit: 0
            )
        )

        #expect(results.count == 2)
        #expect(results.first?.documentPath.contains("b.md") == true)
        #expect(results.first?.score.rerank ?? 0 > 0.9)
        #expect(results.first?.score.blended ?? 0 > results.first?.score.fused ?? 0)
    }

    @Test
    func contentTagBonusCanLiftTaggedDocument() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(docs.appendingPathComponent("a-garden.md"), "tomatoes and compost planning notes")
        try writeFile(docs.appendingPathComponent("z-deploy.md"), "deployment runbook and service cutover checklist")

        let tagger = StaticContentTagger(
            tagsByNeedle: [
                "ship safely": [ContentTag(name: "deployment", confidence: 0.95)],
                "deployment runbook": [ContentTag(name: "deployment", confidence: 0.90)],
                "tomatoes": [ContentTag(name: "gardening", confidence: 0.90)],
            ]
        )

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            contentTagger: tagger,
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 100, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let results = try await index.search(
            SearchQuery(
                text: "ship safely",
                limit: 2,
                rerankLimit: 0,
                expansionLimit: 0
            )
        )

        #expect(results.count == 2)
        #expect(results.first?.documentPath.contains("z-deploy.md") == true)
        #expect(results.first?.score.tag ?? 0 > 0)
    }

    @Test
    func structuredMetadataBranchesCanLiftTaggedDocument() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(docs.appendingPathComponent("a-garden.md"), "tomatoes and compost planning notes")
        try writeFile(docs.appendingPathComponent("z-memory.md"), "release notes and rollout status")

        let tagger = StaticContentTagger(
            tagsByNeedle: [
                "release notes": [
                    ContentTag(name: "facet:project", confidence: 0.95),
                    ContentTag(name: "entity:memory.swift", confidence: 0.95),
                    ContentTag(name: "topic:release timeline", confidence: 0.90),
                ],
                "tomatoes": [
                    ContentTag(name: "facet:fact_about_world", confidence: 0.8),
                    ContentTag(name: "topic:gardening", confidence: 0.9),
                ],
            ]
        )

        let expander = StaticStructuredQueryExpander(
            facetHints: [FacetHint(tag: .project, confidence: 0.9, isExplicit: false)],
            entities: [
                MemoryEntity(
                    label: .project,
                    value: "Memory.swift",
                    normalizedValue: "memory.swift",
                    confidence: 0.92
                )
            ],
            topics: ["release timeline"]
        )

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            structuredQueryExpander: expander,
            contentTagger: tagger,
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 100, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let results = try await index.search(
            SearchQuery(
                text: "what is the rollout status",
                limit: 2,
                rerankLimit: 0,
                expansionLimit: 5
            )
        )

        #expect(results.count == 2)
        #expect(results.first?.documentPath.contains("z-memory.md") == true)
        #expect(results.first?.score.tag ?? 0 > 0)
    }

    @Test
    func rerankerCanLiftRelevantResultBeyondInitialTopWindow() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        for index in 0..<70 {
            try writeFile(
                docs.appendingPathComponent(String(format: "doc-%03d.md", index)),
                "alpha planning roadmap notes"
            )
        }
        try writeFile(
            docs.appendingPathComponent("zzz-target.md"),
            "alpha planning roadmap notes"
        )

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            reranker: ClosureReranker(scoreForCandidate: { result in
                result.documentPath.contains("zzz-target.md") ? 1.0 : 0.1
            }),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 100, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let results = try await index.search(
            SearchQuery(
                text: "alpha planning",
                limit: 5,
                rerankLimit: 80,
                expansionLimit: 0
            )
        )

        #expect(results.isEmpty == false)
        #expect(results.first?.documentPath.contains("zzz-target.md") == true)
    }
}
