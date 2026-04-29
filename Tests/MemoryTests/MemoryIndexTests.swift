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
    func searchCanConstrainCandidatesToDocumentPathPrefix() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let userA = docs.appendingPathComponent("user-a")
        let userB = docs.appendingPathComponent("user-b")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            userA.appendingPathComponent("memory.md"),
            "shared alpha anchor for user A notes"
        )
        try writeFile(
            userB.appendingPathComponent("memory.md"),
            "shared alpha anchor for user B notes"
        )

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 100, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let scoped = try await index.search(
            SearchQuery(
                text: "shared alpha anchor",
                limit: 10,
                semanticCandidateLimit: 0,
                lexicalCandidateLimit: 10,
                rerankLimit: 0,
                expansionLimit: 0,
                documentPathPrefix: userA.path
            )
        )

        #expect(scoped.isEmpty == false)
        #expect(scoped.allSatisfy { $0.documentPath.hasPrefix(userA.path + "/") })

        let semanticScoped = try await index.search(
            SearchQuery(
                text: "shared alpha anchor",
                limit: 10,
                semanticCandidateLimit: 10,
                lexicalCandidateLimit: 0,
                rerankLimit: 0,
                expansionLimit: 0,
                documentPathPrefix: userA.path
            )
        )

        #expect(semanticScoped.isEmpty == false)
        #expect(semanticScoped.allSatisfy { $0.documentPath.hasPrefix(userA.path + "/") })
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
    func rankOneLexicalExpansionHitSurvivesFusion() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("false-certification.md"),
            "False Certification Discharge. A student loan can be discharged when a school falsely certified eligibility for a loan."
        )
        for i in 0..<8 {
            try writeFile(
                docs.appendingPathComponent("distractor-\(i).md"),
                "school attended lender eligible student loan qualified debt account repayment option \(i). school attended lender eligible student loan debt."
            )
        }

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            structuredQueryExpander: StaticStructuredQueryExpander(
                lexicalQueries: ["false certification discharge"]
            ),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 120, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let queryText = "The school I attended told the lender that I was eligible for a student loan but I was not qualified, can I get rid of this debt?"
        let baselineResults = try await index.search(
            SearchQuery(
                text: queryText,
                limit: 3,
                semanticCandidateLimit: 0,
                lexicalCandidateLimit: 50,
                rerankLimit: 0,
                expansionLimit: 0
            )
        )
        #expect(baselineResults.first?.documentPath.contains("false-certification.md") != true)

        let results = try await index.search(
            SearchQuery(
                text: queryText,
                limit: 3,
                semanticCandidateLimit: 0,
                lexicalCandidateLimit: 50,
                rerankLimit: 0,
                expansionLimit: 3
            )
        )

        #expect(results.first?.documentPath.contains("false-certification.md") == true)

        let documentListSurfaceResults = try await index.search(
            SearchQuery(
                text: queryText,
                limit: 18,
                semanticCandidateLimit: 0,
                lexicalCandidateLimit: 200,
                rerankLimit: 0,
                expansionLimit: 5,
                includeTagScoring: false
            )
        )
        #expect(documentListSurfaceResults.first?.documentPath.contains("false-certification.md") == true)

        let references = try await index.memorySearch(
            query: queryText,
            limit: 3,
            features: [.lexical, .expansion],
            dedupeDocuments: true,
            includeLineRanges: false
        )
        #expect(references.first?.documentPath.contains("false-certification.md") == true)
    }

    @Test
    func primaryBranchProtectionKeepsBaselineDocumentRepresented() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("primary.md"),
            "alpha baseline source answer with original query anchors"
        )
        for index in 0..<8 {
            try writeFile(
                docs.appendingPathComponent("expansion-\(index).md"),
                "gamma expansion distraction exact lure \(index). gamma expansion distraction exact lure."
            )
        }

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            structuredQueryExpander: StaticStructuredQueryExpander(
                lexicalQueries: ["gamma expansion distraction exact lure"]
            ),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 100, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let queryText = "alpha baseline source answer original branch details for quarterly memory planning review follow up"
        let unprotected = try await index.search(
            SearchQuery(
                text: queryText,
                limit: 1,
                semanticCandidateLimit: 0,
                lexicalCandidateLimit: 50,
                rerankLimit: 0,
                expansionLimit: 1,
                originalQueryWeight: 0.1,
                expansionQueryWeight: 50,
                primaryBranchProtectionLimit: 0,
                includeTagScoring: false
            )
        )
        #expect(unprotected.first?.documentPath.contains("primary.md") != true)

        let protected = try await index.search(
            SearchQuery(
                text: queryText,
                limit: 1,
                semanticCandidateLimit: 0,
                lexicalCandidateLimit: 50,
                rerankLimit: 0,
                expansionLimit: 1,
                originalQueryWeight: 0.1,
                expansionQueryWeight: 50,
                primaryBranchProtectionLimit: 1,
                includeTagScoring: false
            )
        )

        #expect(protected.first?.documentPath.contains("primary.md") == true)
    }

    @Test
    func distributedAnchorMatchesCanSurfaceThroughLexicalSearch() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("greenwood-reforestation.md"),
            """
            Greenwood neighborhood planning notes.
            The urban shade corridor proposal was approved.
            Reforestation volunteer schedules were drafted.
            Tree nursery purchases are ready.
            Canopy baseline mapping starts next week.
            """
        )

        let repeatedAnchors = ["Greenwood", "urban", "reforestation", "tree", "canopy"]
        for i in 0..<40 {
            let anchor = repeatedAnchors[i % repeatedAnchors.count]
            let repeated = Array(repeating: anchor, count: 28).joined(separator: " ")
            try writeFile(
                docs.appendingPathComponent("distractor-\(i).md"),
                "\(repeated) unrelated parking agenda \(i)"
            )
        }

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: ConstantEmbeddingProvider(),
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 6, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let results = try await index.search(
            SearchQuery(
                text: "What details explain the Greenwood urban reforestation tree canopy work?",
                limit: 10,
                semanticCandidateLimit: 0,
                lexicalCandidateLimit: 40,
                rerankLimit: 0,
                expansionLimit: 0,
                includeTagScoring: false
            )
        )

        #expect(results.contains { $0.documentPath.contains("greenwood-reforestation.md") })
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
    func verboseLexicalQuestionsStillRunQueryExpansion() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        let verboseQuery = "incident response runbook for payments deployment rollback owner escalation timeline and customer notices"
        try writeFile(
            docs.appendingPathComponent("runbook.md"),
            "\(verboseQuery). \(verboseQuery). \(verboseQuery)."
        )
        try writeFile(
            docs.appendingPathComponent("notes.md"),
            "general observations and feedback collected during the quarterly planning session"
        )

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
                text: verboseQuery,
                limit: 3,
                rerankLimit: 0,
                expansionLimit: 2
            )
        )

        let calls = await expander.calls()
        #expect(calls == 1)
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
                semanticQueries: ["deployment rollout", "service cutover"]
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
    func lexicalExpansionQueriesDoNotRequestUnusedSemanticEmbeddings() async throws {
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
        #expect(stats.batchSizes == [1])
    }

    @Test
    func scopedLexicalCoverageCanSkipSemanticEmbedding() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let userA = docs.appendingPathComponent("user-a")
        let dbURL = root.appendingPathComponent("index.sqlite")

        for index in 0..<10 {
            try writeFile(
                userA.appendingPathComponent("memory-\(index).md"),
                "alpha scoped memory note \(index) with recurring lexical anchors"
            )
        }

        let embeddingProvider = CountingEmbeddingProvider()
        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: embeddingProvider,
            tokenizer: DefaultTokenizer(),
            chunker: DefaultChunker(targetTokenCount: 100, overlapTokenCount: 0)
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])
        await embeddingProvider.resetStats()

        let results = try await index.search(
            SearchQuery(
                text: "alpha scoped memory",
                limit: 10,
                semanticCandidateLimit: 30,
                lexicalCandidateLimit: 30,
                rerankLimit: 0,
                expansionLimit: 0,
                documentPathPrefix: userA.path
            )
        )

        let stats = await embeddingProvider.stats()
        #expect(results.count == 10)
        #expect(stats.singleCalls == 0)
        #expect(stats.batchSizes.isEmpty)
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
    func heuristicStructuredExpanderKeepsGenericFactLookupsLexicalOnly() async throws {
        let expander = HeuristicStructuredQueryExpander()
        let query = SearchQuery(text: "What time do I wake up on Tuesdays and Thursdays?")
        let analysis = QueryAnalysis(
            entities: [],
            keyTerms: ["time", "wake", "tuesdays", "thursdays"],
            facetHints: [
                FacetHint(tag: .factAboutUser, confidence: 0.84, isExplicit: true)
            ],
            topics: ["time wake tuesdays", "wake tuesdays thursdays", "tuesdays thursdays"],
            isHowToQuery: false
        )

        let expansion = try await expander.expand(query: query, analysis: analysis, limit: 5)

        #expect(expansion.lexicalQueries.isEmpty == false)
        #expect(expansion.lexicalQueries.contains(where: { $0.contains("wake") && $0.contains("thursdays") }))
        #expect(expansion.lexicalQueries.contains(where: { $0.contains("fact about user") }) == false)
        #expect(expansion.semanticQueries.isEmpty)
        #expect(expansion.hypotheticalDocuments.isEmpty)
    }

    @Test
    func heuristicStructuredExpanderCompactsConversationalFillerIntoSalientPhraseQueries() async throws {
        let expander = HeuristicStructuredQueryExpander()
        let query = SearchQuery(
            text: "Thanks for this valuable information, I have one last question, is it advisable to have the same address in my applications, documents and transactions?"
        )
        let analysis = QueryAnalysis(
            entities: [],
            keyTerms: ["advisable", "same", "address", "applications", "documents", "transactions"],
            facetHints: [
                FacetHint(tag: .factAboutUser, confidence: 0.84, isExplicit: true)
            ],
            topics: ["advisable same address", "last question advisable", "same address applications", "applications documents transactions"],
            isHowToQuery: false
        )

        let expansion = try await expander.expand(query: query, analysis: analysis, limit: 5)

        #expect(expansion.lexicalQueries.isEmpty == false)
        #expect(expansion.lexicalQueries.contains(where: { $0.contains("same address") }))
        #expect(expansion.lexicalQueries.contains(where: { $0.contains("valuable information") }) == false)
        #expect(expansion.lexicalQueries.contains(where: { $0.contains("last question") }) == false)
    }

    @Test
    func heuristicStructuredExpanderAddsFalseCertificationStudentLoanRescueTerms() async throws {
        let expander = HeuristicStructuredQueryExpander()
        let query = SearchQuery(
            text: "The school I attended told the lender that I was eligible for a student loan but I wasn't actually qualified, is there any way to get rid of this debt?"
        )
        let analysis = QueryAnalysis(
            entities: [],
            keyTerms: ["school", "attended", "lender", "eligible", "student", "loan"],
            facetHints: [],
            topics: ["school attended told", "attended told lender", "student loan debt"],
            isHowToQuery: false
        )

        let expansion = try await expander.expand(query: query, analysis: analysis, limit: 5)

        #expect(expansion.lexicalQueries.contains(where: { $0.contains("false certification discharge") }))
    }

    @Test
    func heuristicStructuredExpanderAddsLongHorizonRecallRescueTerms() async throws {
        let expander = HeuristicStructuredQueryExpander()
        let completionAnalysis = QueryAnalysis(
            entities: [],
            keyTerms: ["projects", "completed", "painting", "classes"],
            facetHints: [],
            topics: ["completed painting projects", "painting classes"],
            isHowToQuery: false
        )

        let completionExpansion = try await expander.expand(
            query: SearchQuery(text: "How many projects have I completed since starting painting classes?"),
            analysis: completionAnalysis,
            limit: 5
        )
        let completionLexical = completionExpansion.lexicalQueries.joined(separator: " | ")
        #expect(completionLexical.contains("finished project"))
        #expect(completionLexical.contains("painting project"))

        let tripExpansion = try await expander.expand(
            query: SearchQuery(text: "What is the order of the three trips I took in the past three months?"),
            analysis: QueryAnalysis(
                entities: [],
                keyTerms: ["order", "three", "trips", "past", "months"],
                facetHints: [],
                topics: ["order trips", "past three months"],
                isHowToQuery: false
            ),
            limit: 5
        )
        let tripLexical = tripExpansion.lexicalQueries.joined(separator: " | ")
        #expect(tripLexical.contains("road trip"))
        #expect(tripLexical.contains("camping trip"))

        let referenceDate = try #require(ISO8601DateFormatter().date(from: "2023-05-30T00:00:00Z"))
        let lastWeekExpansion = try await expander.expand(
            query: SearchQuery(
                text: "How many hours of jogging and yoga did I do last week?",
                referenceDate: referenceDate
            ),
            analysis: QueryAnalysis(
                entities: [],
                keyTerms: ["hours", "jogging", "yoga", "last", "week"],
                facetHints: [],
                topics: ["jogging yoga last week", "hours jogging yoga"],
                isHowToQuery: false
            ),
            limit: 5
        )
        let lastWeekLexical = lastWeekExpansion.lexicalQueries.joined(separator: " | ")
        #expect(lastWeekLexical.contains("2023-05-22") || lastWeekLexical.contains("may 22"))
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
