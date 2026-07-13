import Foundation
import Testing
@testable import AgentMemory

struct MemoryExternalAPITests {
    @Test
    func debugMemoriesPaginatesSearchesMetadataAndDoesNotRecordAccess() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let older = try await index.save(
            text: "Alpha launch notes mention SQLite fallback.",
            kind: .fact,
            importance: 0.3,
            createdAt: Date(timeIntervalSince1970: 100),
            tags: ["launch"],
            facetTags: [.tool],
            entities: [
                MemoryEntity(label: .tool, value: "SQLite", normalizedValue: "sqlite"),
            ],
            topics: ["storage"],
            metadata: ["ticket": "DBG-1"]
        )
        let middle = try await index.save(
            text: "Beta launch notes mention Core ML embeddings.",
            kind: .decision,
            importance: 0.8,
            createdAt: Date(timeIntervalSince1970: 200),
            metadata: ["ticket": "DBG-2"]
        )
        let newer = try await index.save(
            text: "Gamma launch notes mention SwiftUI diagnostics.",
            kind: .handoff,
            importance: 0.6,
            createdAt: Date(timeIntervalSince1970: 300),
            metadata: ["ticket": "DBG-3"]
        )

        let firstPage = try await index.debugMemories(
            MemoryDebugQuery(limit: 2, sort: .createdAtDescending)
        )
        #expect(firstPage.records.map(\.id) == [newer.id, middle.id])
        #expect(firstPage.totalCount == 3)
        #expect(firstPage.hasMore)

        let secondPage = try await index.debugMemories(
            MemoryDebugQuery(limit: 2, offset: 2, sort: .createdAtDescending)
        )
        #expect(secondPage.records.map(\.id) == [older.id])
        #expect(secondPage.totalCount == 3)
        #expect(secondPage.hasMore == false)

        let metadataMatch = try await index.debugMemories(
            MemoryDebugQuery(searchText: "DBG-1", limit: 10)
        )
        #expect(metadataMatch.records.map(\.id) == [older.id])
        #expect(metadataMatch.records.first?.metadata["ticket"] == "DBG-1")
        #expect(metadataMatch.records.first?.source == "memory_save")

        let tagMatch = try await index.debugMemories(
            MemoryDebugQuery(searchText: "sqlite storage", limit: 10)
        )
        #expect(tagMatch.records.map(\.id).contains(older.id))

        let beforeAccessCount = try #require(
            metadataMatch.records.first(where: { $0.id == older.id })
        ).accessCount
        _ = try await index.debugMemories(MemoryDebugQuery(searchText: "DBG-1", limit: 10))
        let afterPage = try await index.debugMemories(MemoryDebugQuery(searchText: "DBG-1", limit: 10))
        let afterAccessCount = try #require(
            afterPage.records.first(where: { $0.id == older.id })
        ).accessCount
        #expect(afterAccessCount == beforeAccessCount)
    }

    @Test
    func debugArchiveHidesDefaultListAndRefreshesDerivedMemoryStatus() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let memory = try await index.save(
            text: "Archive this debug-only memory.",
            kind: .fact
        )

        let archived = try #require(try await index.setMemoryStatus(id: memory.id, status: .archived))
        #expect(archived.status == .archived)

        let defaultPage = try await index.debugMemories(MemoryDebugQuery(limit: 10))
        #expect(defaultPage.records.contains(where: { $0.id == memory.id }) == false)

        let archivedPage = try await index.debugMemories(
            MemoryDebugQuery(limit: 10, statuses: Set([.archived]))
        )
        let archivedRecord = try #require(archivedPage.records.first(where: { $0.id == memory.id }))
        #expect(archivedRecord.status == .archived)

        let rematerializedChunk = try await index.getChunk(id: archivedRecord.chunkID)
        #expect(rematerializedChunk?.memoryStatus == .archived)
    }

    @Test
    func saveAndNonHybridRecallModesWork() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let decision = try await index.save(
            text: "Switched to SQLite for the prototype phase.",
            kind: .decision,
            importance: 0.9
        )
        _ = try await index.save(
            text: "I prefer shorter status updates in the morning.",
            kind: .profile,
            importance: 0.4
        )

        let typed = try await index.recall(
            mode: .kind(.decision),
            limit: 10
        )
        #expect(typed.records.contains(where: { $0.chunkID == decision.chunkID }))

        let important = try await index.recall(
            mode: .important,
            limit: 2
        )
        #expect(important.records.count == 2)
        #expect(important.records[0].importance >= important.records[1].importance)
    }

    @Test
    func extractIngestAndHybridRecallRoundTrip() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(
                    role: .user,
                    content: "Let's switch to SQLite for now. Action item: add migration tests."
                ),
            ],
            limit: 10
        )
        #expect(extracted.isEmpty == false)

        let ingestResult = try await index.ingest(extracted)
        #expect(ingestResult.storedCount > 0)

        let hybrid = try await index.recall(
            mode: .hybrid(query: "SQLite migration tests"),
            limit: 5
        )
        #expect(hybrid.records.isEmpty == false)
        #expect(hybrid.records.contains(where: { $0.text.lowercased().contains("sqlite") }))

        let mostAccessed = try await index.recall(
            mode: .kind(.decision),
            limit: 5,
            sort: .mostAccessed
        )
        if let first = mostAccessed.records.first {
            #expect(first.accessCount >= 1)
        }
    }

    @Test
    func semanticSearchFindsNewlySavedMemoryWithoutRebuild() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        _ = try await index.save(
            text: "alpha-only memory marker",
            kind: .fact
        )

        let firstSemantic = try await index.search(
            SearchQuery(
                text: "alpha-only marker",
                limit: 5,
                semanticCandidateLimit: 50,
                lexicalCandidateLimit: 0,
                rerankLimit: 0,
                expansionLimit: 0,
                includeTagScoring: false
            )
        )
        #expect(firstSemantic.isEmpty == false)

        _ = try await index.save(
            text: "zulu-only memory marker",
            kind: .fact
        )

        let secondSemantic = try await index.search(
            SearchQuery(
                text: "zulu-only marker",
                limit: 5,
                semanticCandidateLimit: 50,
                lexicalCandidateLimit: 0,
                rerankLimit: 0,
                expansionLimit: 0,
                includeTagScoring: false
            )
        )
        #expect(secondSemantic.contains(where: { $0.content.contains("zulu-only memory marker") }))
    }

    @Test
    func memorySearchAndMemoryGetSupportAgentStyleLookup() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")
        let profilePath = docs.appendingPathComponent("profile.md")

        try writeFile(
            profilePath,
            """
            # Profile
            Name: Casey
            Favorite tea: jasmine green tea
            Timezone: PST
            """
        )
        try writeFile(
            docs.appendingPathComponent("notes.md"),
            "General notes about travel plans and weekend activities."
        )

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        try await index.rebuildIndex(from: [docs])

        let refs = try await index.memorySearch(
            query: "What tea does Casey prefer?",
            limit: 5,
            includeLineRanges: true
        )

        #expect(refs.isEmpty == false)
        let profileRef = try #require(refs.first(where: { $0.documentPath.hasSuffix("profile.md") }))
        #expect(profileRef.lineRange != nil)

        let focused = try await index.memoryGet(reference: profileRef)
        #expect(focused.documentPath == profilePath.path)
        #expect(focused.source == .fileSystem)
        #expect(focused.lineRange == profileRef.lineRange)
        #expect(focused.content.contains("Favorite tea: jasmine green tea"))

        let sliced = try await index.memoryGet(path: "profile.md", lineRange: MemoryLineRange(start: 3, end: 3))

        #expect(sliced.documentPath == profilePath.path)
        #expect(sliced.source == .fileSystem)
        #expect(sliced.content.contains("Favorite tea: jasmine green tea"))
    }

    @Test
    func memorySearchDoesNotTreatCompletedAsStatusFilterForPlainDocuments() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("painting.md"),
            "I completed five painting class projects since starting painting classes."
        )

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        try await index.rebuildIndex(from: [docs])

        let refs = try await index.memorySearch(
            query: "How many projects have I completed since starting painting classes?",
            limit: 5,
            features: [.lexical],
            includeLineRanges: false
        )

        #expect(refs.contains(where: { $0.documentPath.hasSuffix("painting.md") }))
    }

    @Test
    func memoryGetFallsBackToIndexedContentForIngestedMemory() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let saved = try await index.save(
            text: "I prefer concise progress updates after lunch.",
            kind: .profile
        )

        let fetched = try await index.memoryGet(path: saved.documentPath)
        #expect(fetched.source == .indexed)
        #expect(fetched.content.contains("prefer concise progress updates"))
        #expect(fetched.lineRange == MemoryLineRange(start: 1, end: 1))
    }

    @Test
    func memorySearchCanDedupeToOneResultPerDocument() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        let repeated = Array(repeating: "incident response runbook playbook", count: 80).joined(separator: " ")
        try writeFile(docs.appendingPathComponent("runbook.md"), repeated)
        try writeFile(docs.appendingPathComponent("unrelated.md"), "gardening soil sunlight watering schedule")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider(),
                chunker: DefaultChunker(targetTokenCount: 20, overlapTokenCount: 0)
            )
        )
        try await index.rebuildIndex(from: [docs])

        let nonDeduped = try await index.memorySearch(
            query: "incident response runbook",
            limit: 5,
            dedupeDocuments: false,
            includeLineRanges: false
        )
        let deduped = try await index.memorySearch(
            query: "incident response runbook",
            limit: 5,
            dedupeDocuments: true,
            includeLineRanges: false
        )

        #expect(nonDeduped.count > 1)
        #expect(nonDeduped.map(\.documentPath).count > Set(nonDeduped.map(\.documentPath)).count)
        #expect(deduped.map(\.documentPath).count == Set(deduped.map(\.documentPath)).count)
        #expect(deduped.contains(where: { $0.documentPath.hasSuffix("runbook.md") }))
    }

    @Test
    func profileMemoryReplacesActiveRecordForMatchingCanonicalKey() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let first = try await index.save(
            text: "Preferred editor is Vim.",
            kind: .profile,
            canonicalKey: "profile:editor"
        )
        let second = try await index.save(
            text: "Preferred editor is Zed.",
            kind: .profile,
            canonicalKey: "profile:editor"
        )

        let active = try await index.recall(
            mode: .kind(.profile),
            limit: 10
        )
        #expect(active.records.count == 1)
        #expect(active.records.first?.id == second.id)
        #expect(active.records.first?.status == .active)

        let historical = try await index.recall(
            mode: .kind(.profile),
            limit: 10,
            statuses: [.active, .superseded]
        )
        #expect(historical.records.contains(where: { $0.id == first.id && $0.status == .superseded }))
        #expect(historical.records.contains(where: { $0.id == second.id && $0.status == .active }))
    }

    @Test
    func decisionMemoryFiltersRespectActiveAndSupersededStates() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let first = try await index.save(
            text: "We switched the prototype to SQLite.",
            kind: .decision,
            canonicalKey: "decision:storage-engine"
        )
        let second = try await index.save(
            text: "We switched the prototype to LMDB.",
            kind: .decision,
            canonicalKey: "decision:storage-engine"
        )

        let activeRefs = try await index.memorySearch(
            query: "prototype switched storage engine",
            limit: 10,
            kinds: [.decision],
            statuses: [.active]
        )
        let supersededRefs = try await index.memorySearch(
            query: "prototype switched storage engine",
            limit: 10,
            kinds: [.decision],
            statuses: [.superseded]
        )

        #expect(activeRefs.contains(where: { $0.memoryID == second.id }))
        #expect(activeRefs.contains(where: { $0.memoryID == first.id }) == false)
        #expect(supersededRefs.contains(where: { $0.memoryID == first.id }))
        #expect(supersededRefs.contains(where: { $0.memoryID == second.id }) == false)
    }

    @Test
    func commitmentResolutionUpdatesExistingRecord() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let first = try await index.save(
            text: "Add migration coverage for canonical memories.",
            kind: .commitment,
            canonicalKey: "commitment:migration-coverage"
        )
        let resolved = try await index.save(
            text: "Add migration coverage for canonical memories.",
            kind: .commitment,
            status: .resolved,
            canonicalKey: "commitment:migration-coverage"
        )

        #expect(resolved.id == first.id)
        #expect(resolved.status == .resolved)

        let active = try await index.recall(
            mode: .kind(.commitment),
            limit: 10,
            statuses: [.active]
        )
        #expect(active.records.isEmpty)

        let completed = try await index.recall(
            mode: .kind(.commitment),
            limit: 10,
            statuses: [.resolved]
        )
        #expect(completed.records.count == 1)
        #expect(completed.records.first?.id == first.id)
        #expect(completed.records.first?.status == .resolved)
    }

    @Test
    func handoffUsesSingletonCanonicalKey() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let first = try await index.save(
            text: "Current status: canonical memory migration is in progress.",
            kind: .handoff
        )
        let second = try await index.save(
            text: "Current status: consolidation tests are now in place.",
            kind: .handoff
        )

        let active = try await index.recall(
            mode: .kind(.handoff),
            limit: 10
        )
        #expect(active.records.count == 1)
        #expect(active.records.first?.id == second.id)
        #expect(active.records.first?.canonicalKey == "handoff:primary")

        let historical = try await index.recall(
            mode: .kind(.handoff),
            limit: 10,
            statuses: [.active, .superseded]
        )
        #expect(historical.records.contains(where: { $0.id == first.id && $0.status == .superseded }))
        #expect(historical.records.contains(where: { $0.id == second.id && $0.status == .active }))
    }

    @Test
    func factsDeduplicateWhileEpisodesRemainAppendOnly() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let firstFact = try await index.save(
            text: "The repo uses SQLite as the canonical storage layer.",
            kind: .fact
        )
        let secondFact = try await index.save(
            text: "The repo uses SQLite as the canonical storage layer.",
            kind: .fact
        )
        #expect(secondFact.id == firstFact.id)

        let facts = try await index.recall(
            mode: .kind(.fact),
            limit: 10
        )
        #expect(facts.records.count == 1)
        #expect(facts.records.first?.id == firstFact.id)

        let firstEpisode = try await index.save(
            text: "Yesterday we landed the migration scaffolding.",
            kind: .episode,
            eventAt: Date(timeIntervalSince1970: 1_710_000_000)
        )
        let secondEpisode = try await index.save(
            text: "Today we added explicit consolidation coverage.",
            kind: .episode,
            eventAt: Date(timeIntervalSince1970: 1_710_086_400)
        )

        let episodes = try await index.recall(
            mode: .kind(.episode),
            limit: 10
        )
        #expect(episodes.records.count == 2)
        #expect(episodes.records.contains(where: { $0.id == firstEpisode.id }))
        #expect(episodes.records.contains(where: { $0.id == secondEpisode.id }))
    }

    @Test
    func saveRoundTripPreservesThreeLayerSchemaAndSupportsFilters() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let saved = try await index.save(
            text: "Zac prefers hybrid retrieval for the Memory.swift project.",
            kind: .profile,
            facetTags: [.preference, .project, .factAboutUser],
            entities: [
                MemoryEntity(
                    label: .project,
                    value: "Memory.swift",
                    normalizedValue: "Memory.Swift",
                    confidence: 0.9
                ),
            ],
            topics: ["Hybrid Retrieval Pipeline Design", "hybrid retrieval"]
        )

        #expect(saved.facetTags.contains(.preference))
        #expect(saved.facetTags.contains(.project))
        #expect(saved.entities.first?.normalizedValue == "memory.swift")
        #expect(saved.topics.contains("hybrid retrieval pipeline design"))
        #expect(saved.topics.contains("hybrid retrieval"))

        let filtered = try await index.recall(
            mode: .kind(.profile),
            limit: 10,
            facets: [.project],
            entityValues: ["Memory.swift"],
            topics: ["hybrid retrieval"]
        )

        #expect(filtered.records.count == 1)
        #expect(filtered.records.first?.id == saved.id)

        let refs = try await index.memorySearch(
            query: "What project uses hybrid retrieval?",
            limit: 10,
            facets: [.project],
            entityValues: ["memory.swift"],
            topics: ["hybrid retrieval"]
        )

        #expect(refs.count == 1)
        #expect(refs.first?.memoryID == saved.id)
    }

    @Test
    func entityAndTopicOverlapBonusCanBreakTiesBetweenEquivalentMemories() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: ConstantEmbeddingProvider()
            )
        )

        let preferred = try await index.save(
            text: "We should ship the retrieval changes this week after the retrieval review.",
            kind: .episode,
            facetTags: [.project, .tool],
            entities: [
                MemoryEntity(label: .project, value: "Memory.swift", normalizedValue: "memory.swift"),
                MemoryEntity(label: .tool, value: "sqlite-vec", normalizedValue: "sqlite-vec"),
            ],
            topics: ["hybrid retrieval"]
        )
        let competing = try await index.save(
            text: "We should ship the retrieval changes this week after the operations review.",
            kind: .episode,
            facetTags: [.project, .tool],
            entities: [
                MemoryEntity(label: .project, value: "OtherRepo", normalizedValue: "otherrepo"),
                MemoryEntity(label: .tool, value: "faiss", normalizedValue: "faiss"),
            ],
            topics: ["deployment workflow"]
        )

        let results = try await index.recall(
            mode: .hybrid(query: "Memory.swift sqlite-vec hybrid retrieval"),
            limit: 2
        )

        #expect(results.records.count == 2)
        #expect(results.records.first?.entities.contains(where: { $0.normalizedValue == "memory.swift" }) == true)
        #expect(results.records.first?.id == preferred.id)
        #expect(results.records.last?.id == competing.id)
    }

    @Test
    func heuristicExtractSkipsQuestionsAndChitchat() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(role: .user, content: "Thanks, can you explain how vector indexes work before we decide anything?"),
                ConversationMessage(role: .assistant, content: "Sure, I can explain that."),
            ]
        )

        #expect(extracted.isEmpty)
    }

    @Test
    func heuristicExtractFocusesOnUserSelfLocationAndRejectsAssistantCapabilityRefusals() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(
                    role: .user,
                    content: "i live in sf, what's a fun thing to do tonight there?"
                ),
                ConversationMessage(
                    role: .assistant,
                    content: """
                    I don't have real-time, location-specific entertainment suggestions like "fun things to do tonight" in San Francisco. My capabilities are focused on executing code, interacting with device APIs (like calendar, photos, etc.), and running shell commands in the workspace.

                    For local recommendations, I suggest checking websites like Yelp, TripAdvisor, or local event listings for San Francisco!
                    """
                ),
            ],
            limit: 10
        )

        #expect(extracted.count == 1)
        let profile = try #require(extracted.first)
        #expect(profile.kind == .profile)
        #expect(profile.text == "The user lives in San Francisco, CA.")
        #expect(profile.subject == .user)
        #expect(profile.canonicalKey == "profile:user:location")
        #expect(profile.evidence.first?.role == .user)
        #expect(profile.facetTags.contains(.factAboutUser))
        #expect(profile.facetTags.contains(.location))
        #expect(profile.entities.contains { entity in
            entity.label == .location && entity.normalizedValue == "san francisco"
        })
    }

    @Test
    func heuristicExtractGeneralizesSelfReportedLocationBeyondKnownAliases() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(role: .user, content: "i live in chicago, what should i do tonight?"),
            ],
            limit: 10
        )

        #expect(extracted.count == 1)
        let profile = try #require(extracted.first)
        #expect(profile.kind == .profile)
        #expect(profile.text == "The user lives in Chicago.")
        #expect(profile.subject == .user)
        #expect(profile.canonicalKey == "profile:user:location")
        #expect(profile.facetTags.contains(.location))
        #expect(profile.entities.contains { entity in
            entity.label == .location && entity.normalizedValue == "chicago"
        })
    }

    @Test
    func heuristicExtractDoesNotTreatTransientStatesAsLocation() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(role: .user, content: "i'm in a meeting, can you take notes for me?"),
            ],
            limit: 10
        )

        #expect(!extracted.contains { $0.text.contains("lives in") })
    }

    @Test
    func prepareContextIgnoresPreviouslyInjectedContextBlocks() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let saved = try await index.save(
            text: "The user lives in San Francisco, CA.",
            kind: .profile,
            facetTags: [.factAboutUser, .location],
            canonicalKey: "profile:user:location",
            subject: .user
        )

        let first = try await index.prepareContext(
            MemoryContextRequest(
                messages: [
                    ConversationMessage(role: .user, content: "What should I do tonight in San Francisco?"),
                ]
            )
        )
        #expect(first.references.contains { $0.memoryID == saved.id })

        // A host app that prepends the previous context block to the next turn
        // must not lose the user's actual message to sanitization.
        let second = try await index.prepareContext(
            MemoryContextRequest(
                messages: [
                    ConversationMessage(
                        role: .user,
                        content: "\(first.contextBlock)\n\nWhat should I do tonight in San Francisco?"
                    ),
                ]
            )
        )
        #expect(second.references.contains { $0.memoryID == saved.id })
        #expect(second.contextBlock.contains("UNTRUSTED MEMORY CONTEXT"))
    }

    @Test
    func recordSignalPrunesSignalsOlderThanConfiguredRetention() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider(),
                memorySignalRetention: 60
            )
        )

        try await index.recordSignal(
            MemorySignal(kind: .recall, query: "old", createdAt: Date().addingTimeInterval(-3_600))
        )
        try await index.recordSignal(
            MemorySignal(kind: .recall, query: "fresh")
        )

        let maintenance = try await index.runMaintenance(
            MemoryMaintenanceRequest(mode: .preview, lookbackDays: 30)
        )
        #expect(maintenance.consideredSignalCount == 1)
    }

    @Test
    func ambiguousLocationDoesNotBecomeResidenceWithoutTagger() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider(),
                entityTagger: nil
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(role: .user, content: "btw i'm in lisbon, any dinner recs?"),
            ],
            limit: 10
        )

        #expect(!extracted.contains { $0.text.contains("lives in") })
    }

#if MEMORY_NATURAL_LANGUAGE
    @Test
    func nlEntityTaggerRecognizesNamedEntitiesConservatively() {
        let tagger = NLEntityTagger()

        let entities = tagger.recognizeEntities(in: "Annie retired from Google last year.")
        #expect(entities.contains { $0.label == .person && $0.normalizedValue == "annie" })
        #expect(entities.contains { $0.label == .organization && $0.normalizedValue == "google" })
        #expect(entities.allSatisfy { ($0.confidence ?? 1) < 0.9 })

        // Lowercase chat text yields nothing; heuristics remain the floor.
        #expect(tagger.recognizeEntities(in: "annie retired from google last year").isEmpty)
    }

    @Test
    func heuristicExtractDoesNotRewritePresenceAsResidence() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(role: .user, content: "btw i'm in lisbon, any dinner recs?"),
                ConversationMessage(role: .user, content: "i'm in sf now, what should we do?"),
                ConversationMessage(role: .user, content: "i am in tokyo currently for work."),
                ConversationMessage(role: .user, content: "i'm in new york atm."),
            ],
            limit: 10
        )

        #expect(extracted.isEmpty)
    }

    @Test
    func heuristicExtractRejectsTransientTravelLocation() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(role: .user, content: "i'm in boston this week, remind me to pack"),
                ConversationMessage(role: .user, content: "i'm in tokyo tonight, any dinner recs?"),
            ],
            limit: 10
        )

        #expect(!extracted.contains { $0.text.contains("lives in") })
    }

    @Test
    func heuristicExtractUpgradesEntityLabelsFromTagger() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(role: .user, content: "Priya is the maintainer of the ingest pipeline."),
            ],
            limit: 10
        )

        let profile = try #require(extracted.first)
        #expect(profile.entities.contains { entity in
            entity.label == .person && entity.normalizedValue == "priya"
        })
    }

    @Test
    func linguisticLabelsRequireContextBeforeChangingFacets() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(role: .user, content: "On Monday, Rowan reviewed the TrailMap export failure."),
                ConversationMessage(role: .user, content: "Mina prefers Linear for AtlasKit project triage."),
                ConversationMessage(role: .user, content: "After the incident, Mina captured a lesson about Redis backoff limits."),
            ],
            limit: 10
        )

        #expect(extracted.count == 3)
        #expect(extracted.allSatisfy { !$0.facetTags.contains(.location) })
        #expect(!extracted.flatMap(\.entities).contains { entity in
            entity.label == .location && ["mina", "rowan"].contains(entity.normalizedValue)
        })
        #expect(!extracted.flatMap(\.entities).contains { entity in
            entity.label == .person && entity.normalizedValue == "redis"
        })
    }
#endif

    @Test
    func customExtractorCandidatesReceiveCentralEntityEnrichment() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")
        let candidate = MemoryCandidate(
            text: "Priya is the maintainer for AtlasKit.",
            kind: .profile,
            entities: [
                MemoryEntity(
                    label: .project,
                    value: "AtlasKit",
                    normalizedValue: "atlaskit",
                    confidence: 0.98
                ),
            ]
        )

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider(),
                memoryExtractor: StaticMemoryExtractor(
                    result: MemoryExtractionResult(candidates: [candidate])
                ),
                entityTagger: StaticEntityTagger(
                    entities: [
                        MemoryEntity(
                            label: .person,
                            value: "Priya",
                            normalizedValue: "priya",
                            confidence: 0.95
                        ),
                    ]
                )
            )
        )

        let extracted = try await index.extract(from: "ignored")
        let enriched = try #require(extracted.first)
        #expect(enriched.entities.contains { $0.label == .project && $0.normalizedValue == "atlaskit" })
        #expect(enriched.entities.contains { $0.label == .person && $0.normalizedValue == "priya" })
        #expect(enriched.facetTags.isEmpty)
    }

    @Test
    func capturePreviewAndIngestUseSubjectAwareEvidence() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let preview = try await index.capture(
            MemoryCaptureRequest(
                messages: [
                    ConversationMessage(role: .user, content: "i live in sf, what's a fun thing to do tonight there?"),
                ],
                mode: .preview,
                sourceID: "session-1"
            )
        )

        #expect(preview.ingestResult == nil)
        let previewCandidate = try #require(preview.extraction.candidates.first)
        #expect(previewCandidate.text == "The user lives in San Francisco, CA.")
        #expect(previewCandidate.subject == .user)
        #expect(previewCandidate.canonicalKey == "profile:user:location")
        #expect(previewCandidate.evidence.first?.sourceID == "session-1")

        let ingested = try await index.capture(
            MemoryCaptureRequest(
                messages: [
                    ConversationMessage(role: .user, content: "i live in sf, what's a fun thing to do tonight there?"),
                ],
                mode: .ingest,
                sourceID: "session-1"
            )
        )

        let record = try #require(ingested.ingestResult?.records.first)
        #expect(record.subject == .user)
        #expect(record.canonicalKey == "profile:user:location")
        #expect(record.evidence.first?.sourceID == "session-1")
    }

    @Test
    func prepareContextFramesUntrustedMemoryAndSurfacesPathHints() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let saved = try await index.save(
            text: "The user lives in San Francisco, CA.",
            kind: .profile,
            facetTags: [.factAboutUser, .location],
            entities: [
                MemoryEntity(label: .location, value: "San Francisco", normalizedValue: "san francisco"),
            ],
            canonicalKey: "profile:user:location",
            subject: .user,
            evidence: [
                MemoryEvidence(role: .user, excerpt: "I live in sf", messageIndex: 0, sourceID: "session-1"),
            ]
        )
        try await index.setContextHint(
            MemoryContextHint(pathPrefix: "memory://", context: "Memory records are durable user-facing facts.")
        )

        let response = try await index.prepareContext(
            MemoryContextRequest(
                messages: [
                    ConversationMessage(role: .user, content: "What should I do tonight in San Francisco?"),
                ],
                budget: MemoryContextBudget(maxReferences: 4, maxTokens: 200),
                sourceID: "session-2"
            )
        )

        #expect(response.contextBlock.contains("UNTRUSTED MEMORY CONTEXT"))
        #expect(response.contextBlock.contains("The user lives in San Francisco"))
        #expect(response.references.contains { $0.memoryID == saved.id })
        #expect(response.hints.contains { $0.context.contains("durable user-facing facts") })

        let maintenance = try await index.runMaintenance(
            MemoryMaintenanceRequest(
                mode: .preview,
                minSignalCount: 1,
                minDistinctQueries: 1,
                minConfidence: 0.1
            )
        )
        #expect(maintenance.consideredSignalCount >= 1)
    }

    @Test
    func maintenancePreviewPromotesRepeatedRecallSignalsForExistingMemory() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let saved = try await index.save(
            text: "The user prefers ramen for casual dinners.",
            kind: .profile,
            facetTags: [.factAboutUser, .preference],
            canonicalKey: "profile:user:preference:ramen",
            subject: .user,
            evidence: [
                MemoryEvidence(role: .user, excerpt: "I love ramen", messageIndex: 0, sourceID: "session-1"),
            ]
        )

        try await index.recordSignal(MemorySignal(kind: .recall, memoryID: saved.id, query: "dinner ideas", snippet: saved.text, confidence: 0.9))
        try await index.recordSignal(MemorySignal(kind: .recall, memoryID: saved.id, query: "casual food", snippet: saved.text, confidence: 0.9))
        try await index.recordSignal(MemorySignal(kind: .recall, memoryID: saved.id, query: "dinner ideas", snippet: saved.text, confidence: 0.9))

        let maintenance = try await index.runMaintenance(
            MemoryMaintenanceRequest(mode: .preview)
        )

        #expect(maintenance.proposedCandidates.contains { candidate in
            candidate.canonicalKey == "profile:user:preference:ramen"
                && candidate.source == "maintenance"
                && candidate.evidence.first?.sourceID == "session-1"
        })
    }

    @Test
    func detailedExtractReportsRejectedSpansAndProposedActions() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let result = try await index.extractDetailed(
            from: [
                ConversationMessage(role: .user, content: "Thanks, can you explain the index first?"),
                ConversationMessage(role: .user, content: "My role is the maintainer for Memory.swift."),
                ConversationMessage(role: .user, content: "Action item: add migration tests."),
            ],
            limit: 10
        )

        #expect(result.candidates.count == 2)
        #expect(result.rejectedSpans.contains(where: { $0.reason == "not_memory_worthy" }))
        #expect(result.proposedActions.contains(.replaceActive))
        #expect(result.proposedActions.contains(.create))
        let extractedProfile = result.candidates.first { candidate in
            candidate.kind == .profile && candidate.canonicalKey == "profile:user:role"
        }
        let extractedCommitment = result.candidates.first { candidate in
            candidate.kind == .commitment
        }
        #expect(extractedProfile != nil)
        #expect(extractedCommitment != nil)
    }

    @Test
    func heuristicExtractCapturesCanonicalEntitiesAndTopicsForAgentMemoryKinds() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(
                    role: .user,
                    content: "Zac prefers Zed for Memory.swift hybrid retrieval planning."
                ),
                ConversationMessage(
                    role: .user,
                    content: "We decided to use LEAF-IR embeddings for Memory.swift recall."
                ),
                ConversationMessage(
                    role: .user,
                    content: "Action item: ship entity overlap tests for Memory.swift."
                ),
            ],
            limit: 10
        )

        let profile = try #require(extracted.first { $0.kind == .profile })
        let profileEntities = Set(profile.entities.map(\.normalizedValue))
        #expect(profileEntities.isSuperset(of: ["zac", "zed", "memory.swift"]))
        #expect(profile.topics.contains("hybrid retrieval"))
        #expect(profile.topics.contains("memory.swift planning"))

        let decision = try #require(extracted.first { $0.kind == .decision })
        let decisionEntities = Set(decision.entities.map(\.normalizedValue))
        #expect(decisionEntities.isSuperset(of: ["leaf-ir", "memory.swift"]))
        #expect(decision.topics.contains("memory.swift recall"))
        #expect(decision.topics.contains("embeddings for recall"))

        let commitment = try #require(extracted.first { $0.kind == .commitment })
        let commitmentEntities = Set(commitment.entities.map(\.normalizedValue))
        #expect(commitmentEntities.contains("memory.swift"))
        #expect(commitment.topics.contains("entity overlap tests"))
    }

    @Test
    func heuristicExtractGeneralizesHeldoutEntitiesTopicsAndKindCues() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(
                    role: .user,
                    content: "Theo is the release owner for the HarborOps project."
                ),
                ConversationMessage(
                    role: .user,
                    content: "The HarborOps staging office blocks large artifact uploads."
                ),
                ConversationMessage(
                    role: .user,
                    content: "We settled on gRPC for the mobile sync boundary."
                ),
                ConversationMessage(
                    role: .user,
                    content: "Runbook: rotate Redis credentials, restart BeaconCRM, and verify session cache."
                ),
                ConversationMessage(
                    role: .user,
                    content: "On Monday, Rowan reviewed the TrailMap export failure."
                ),
            ],
            limit: 10
        )

        let profile = try #require(extracted.first { $0.text.contains("release owner") })
        #expect(profile.kind == .profile)
        #expect(Set(profile.entities.map(\.normalizedValue)).isSuperset(of: ["theo", "harborops"]))
        #expect(profile.topics.contains("release owner"))
        #expect(profile.facetTags.isSuperset(of: [.factAboutUser, .identitySignal, .project]))

        let fact = try #require(extracted.first { $0.text.contains("artifact uploads") })
        #expect(fact.kind == .fact)
        #expect(Set(fact.entities.map(\.normalizedValue)).contains("harborops"))
        #expect(fact.topics.contains("artifact uploads"))

        let decision = try #require(extracted.first { $0.text.contains("gRPC") })
        #expect(decision.kind == .decision)
        #expect(Set(decision.entities.map(\.normalizedValue)).contains("grpc"))
        #expect(decision.topics.contains("mobile sync"))
        #expect(decision.facetTags.contains(.decisionTopic))

        let procedure = try #require(extracted.first { $0.text.contains("session cache") })
        #expect(procedure.kind == .procedure)
        #expect(Set(procedure.entities.map(\.normalizedValue)).isSuperset(of: ["redis", "beaconcrm"]))
        #expect(procedure.topics.contains("session cache"))

        let episode = try #require(extracted.first { $0.text.contains("export failure") })
        #expect(episode.kind == .episode)
        #expect(Set(episode.entities.map(\.normalizedValue)).isSuperset(of: ["rowan", "trailmap"]))
        #expect(episode.topics.contains("export failure"))
        #expect(!episode.facetTags.contains(.person))
    }

    @Test
    func heuristicExtractFacetsUseEntitiesAndAvoidSubstringFalsePositives() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let extracted = try await index.extract(
            from: [
                ConversationMessage(
                    role: .user,
                    content: "Zac's timezone is Pacific Time in San Francisco for Memory.swift collaboration."
                ),
                ConversationMessage(
                    role: .user,
                    content: "TODO: add migration coverage for facet tags this week."
                ),
                ConversationMessage(
                    role: .user,
                    content: "Today we landed schema v3 for canonical memories."
                ),
                ConversationMessage(
                    role: .user,
                    content: "Guide: tag projected retrieval metadata as facet, entity, and topic prefixes."
                ),
                ConversationMessage(
                    role: .user,
                    content: "TODO: verify the mobile sync boundary with gRPC."
                ),
                ConversationMessage(
                    role: .user,
                    content: "This morning Zac met Sam in San Francisco to review Memory.swift."
                ),
                ConversationMessage(
                    role: .user,
                    content: "After the demo the team celebrated the CoreML build working offline."
                ),
            ],
            limit: 10
        )

        let timezone = try #require(extracted.first { $0.text.contains("timezone") })
        #expect(timezone.facetTags.contains(.location))
        #expect(!timezone.facetTags.contains(.project))
        #expect(!timezone.facetTags.contains(.identitySignal))

        let dueThisWeek = try #require(extracted.first { $0.text.contains("this week") })
        #expect(dueThisWeek.facetTags.isSuperset(of: [.task, .timeSensitive]))

        let landedToday = try #require(extracted.first { $0.text.contains("schema v3") })
        #expect(!landedToday.facetTags.contains(.timeSensitive))

        let projectedMetadata = try #require(extracted.first { $0.text.contains("projected retrieval") })
        #expect(!projectedMetadata.facetTags.contains(.project))

        let grpcCommitment = try #require(extracted.first { $0.text.contains("gRPC") })
        #expect(!grpcCommitment.facetTags.contains(.tool))
        #expect(!grpcCommitment.facetTags.contains(.person))
        #expect(!grpcCommitment.facetTags.contains(.relationship))

        let meeting = try #require(extracted.first { $0.text.contains("met Sam") })
        #expect(meeting.facetTags.isSuperset(of: [.person, .location, .project]))

        let celebration = try #require(extracted.first { $0.text.contains("celebrated") })
        #expect(celebration.facetTags.isSuperset(of: [.emotion, .tool]))
    }

    @Test
    func inferredCanonicalFacetsDriveProfileDecisionAndCommitmentUpdates() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let profileV1 = try #require(
            try await index.extract(
                from: [
                    ConversationMessage(
                        role: .user,
                        content: "Zac prefers Vim for Memory.swift release coordination."
                    ),
                ],
                limit: 5
            ).first
        )
        let profileV2 = try #require(
            try await index.extract(
                from: [
                    ConversationMessage(
                        role: .user,
                        content: "Zac prefers Zed for Memory.swift release coordination."
                    ),
                ],
                limit: 5
            ).first
        )
        _ = try await index.ingest([profileV1])
        _ = try await index.ingest([profileV2])

        let profiles = try await index.recall(mode: .kind(.profile), limit: 10)
        #expect(profiles.records.count == 1)
        #expect(profiles.records.first?.text.contains("Zed") == true)

        let decisionV1 = try #require(
            try await index.extract(
                from: [
                    ConversationMessage(
                        role: .user,
                        content: "We decided to use LEAF-IR embeddings for Memory.swift recall."
                    ),
                ],
                limit: 5
            ).first
        )
        let decisionV2 = try #require(
            try await index.extract(
                from: [
                    ConversationMessage(
                        role: .user,
                        content: "We decided to use CoreML embeddings for Memory.swift recall."
                    ),
                ],
                limit: 5
            ).first
        )
        _ = try await index.ingest([decisionV1])
        _ = try await index.ingest([decisionV2])

        let decisions = try await index.recall(mode: .kind(.decision), limit: 10)
        #expect(decisions.records.count == 1)
        #expect(decisions.records.first?.text.contains("CoreML") == true)

        let commitmentActive = try #require(
            try await index.extract(
                from: [
                    ConversationMessage(
                        role: .user,
                        content: "Action item: ship entity overlap tests for Memory.swift."
                    ),
                ],
                limit: 5
            ).first
        )
        let commitmentResolved = try #require(
            try await index.extract(
                from: [
                    ConversationMessage(
                        role: .user,
                        content: "Done: ship entity overlap tests for Memory.swift."
                    ),
                ],
                limit: 5
            ).first
        )
        _ = try await index.ingest([commitmentActive])
        _ = try await index.ingest([commitmentResolved])

        let activeCommitments = try await index.recall(mode: .kind(.commitment), limit: 10)
        #expect(activeCommitments.records.isEmpty)

        let resolvedCommitments = try await index.recall(
            mode: .kind(.commitment),
            limit: 10,
            statuses: [.resolved]
        )
        #expect(resolvedCommitments.records.count == 1)
        #expect(resolvedCommitments.records.first?.text.contains("entity overlap tests") == true)
    }

    @Test
    func memorySearchDefaultsToActiveMemoriesWithoutHidingDocuments() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(docs.appendingPathComponent("storage.md"), "The active document mentions SQLite storage.")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )
        try await index.rebuildIndex(from: [docs])

        let first = try await index.save(
            text: "Storage decision was LMDB.",
            kind: .decision,
            canonicalKey: "decision:storage"
        )
        let second = try await index.save(
            text: "Storage decision is SQLite.",
            kind: .decision,
            canonicalKey: "decision:storage"
        )

        let defaultRefs = try await index.memorySearch(query: "storage decision sqlite", limit: 10)
        #expect(defaultRefs.contains(where: { $0.documentPath.hasSuffix("storage.md") }))
        #expect(defaultRefs.contains(where: { $0.memoryID == second.id }))
        #expect(defaultRefs.contains(where: { $0.memoryID == first.id }) == false)

        let historicalRefs = try await index.memorySearch(
            query: "storage decision lmdb",
            limit: 10,
            statuses: [.superseded]
        )
        #expect(historicalRefs.contains(where: { $0.memoryID == first.id }))
    }

    @Test
    func memorySearchDoesNotTreatOrdinaryOldObjectQueriesAsHistoricalMemoryOnly() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("keyboard.md"),
            "Maya kept the old keyboard beside the studio monitor and used it for every demo."
        )

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )
        try await index.rebuildIndex(from: [docs])

        let refs = try await index.memorySearch(
            query: "Who kept the old keyboard?",
            limit: 5,
            features: [.lexical],
            includeLineRanges: false
        )

        #expect(refs.contains(where: { $0.documentPath.hasSuffix("keyboard.md") }))
    }

    @Test
    func memorySearchDoesNotTreatPreviousConversationAsHistoricalMemoryOnly() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("cocktail.md"),
            "In a previous conversation about building a cocktail bar, the fifth recommended bottle was orange bitters."
        )

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )
        try await index.rebuildIndex(from: [docs])

        let refs = try await index.memorySearch(
            query: "In our previous conversation about building a cocktail bar, what was the fifth bottle?",
            limit: 5,
            features: [.lexical],
            includeLineRanges: false
        )

        #expect(refs.contains(where: { $0.documentPath.hasSuffix("cocktail.md") }))
    }

    @Test
    func ingestReportsWriteActionsForDedupeReplacementAndStatusMerge() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let firstFact = try await index.ingest([
            MemoryCandidate(text: "The repo uses SQLite as the canonical storage layer.", kind: .fact),
        ])
        let duplicateFact = try await index.ingest([
            MemoryCandidate(text: "The repository uses SQLite as canonical storage layer.", kind: .fact),
        ])
        let firstProfile = try await index.ingest([
            MemoryCandidate(text: "Preferred editor is Vim.", kind: .profile),
        ])
        let replacementProfile = try await index.ingest([
            MemoryCandidate(text: "Preferred editor is Zed.", kind: .profile),
        ])
        let activeCommitment = try await index.ingest([
            MemoryCandidate(
                text: "Action item: add migration tests.",
                kind: .commitment,
                canonicalKey: "commitment:migration-tests"
            ),
        ])
        let resolvedCommitment = try await index.ingest([
            MemoryCandidate(
                text: "Done: add migration tests.",
                kind: .commitment,
                status: .resolved,
                canonicalKey: "commitment:migration-tests"
            ),
        ])

        #expect(firstFact.actions == [.create])
        #expect(duplicateFact.actions == [.dedupe])
        #expect(firstProfile.actions == [.create])
        #expect(replacementProfile.actions == [.replaceActive])
        #expect(activeCommitment.actions == [.create])
        #expect(resolvedCommitment.actions == [.mergeStatus])

        let facts = try await index.recall(mode: .kind(.fact), limit: 10)
        #expect(facts.records.count == 1)
    }

    @Test
    func hybridRecallIncludesResolvedCommitmentsForCompletionQueries() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let active = try await index.save(
            text: "Action item: add migration tests.",
            kind: .commitment,
            canonicalKey: "commitment:migration-tests"
        )
        let resolved = try await index.save(
            text: "Done: add migration tests.",
            kind: .commitment,
            status: .resolved,
            canonicalKey: "commitment:migration-tests"
        )

        #expect(resolved.id == active.id)

        let activeOnly = try await index.recall(
            mode: .kind(.commitment),
            limit: 10,
            statuses: [.active]
        )
        #expect(activeOnly.records.isEmpty)

        let planned = try await index.recall(
            mode: .hybrid(query: "What happened to migration tests?"),
            limit: 10
        )
        #expect(planned.records.contains(where: { $0.id == active.id && $0.status == .resolved }))
    }

    @Test
    func inferredProfileCanonicalKeyReplacesActiveRecord() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let first = try await index.save(text: "Preferred editor is Vim.", kind: .profile)
        let second = try await index.save(text: "Preferred editor is Zed.", kind: .profile)

        let active = try await index.recall(mode: .kind(.profile), limit: 10)
        #expect(active.records.count == 1)
        #expect(active.records.first?.id == second.id)
        #expect(active.records.first?.canonicalKey == "profile:editor")

        let historical = try await index.recall(
            mode: .kind(.profile),
            limit: 10,
            statuses: [.active, .superseded]
        )
        #expect(historical.records.contains(where: { $0.id == first.id && $0.status == .superseded }))
    }

    @Test
    func extractedProfileEditorUpdateKeepsReplacementMetadata() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        let first = try await index.save(
            text: "Preferred editor is Vim.",
            kind: .profile,
            canonicalKey: "profile:editor"
        )
        let extracted = try await index.extract(
            from: [ConversationMessage(role: .user, content: "Preferred editor is Zed.")],
            limit: 20
        )
        let result = try await index.ingest(extracted)

        #expect(result.actions == [.replaceActive])
        #expect(result.records.count == 1)
        let replacement = try #require(result.records.first)
        #expect(replacement.canonicalKey == "profile:editor")
        #expect(replacement.text.localizedCaseInsensitiveContains("Zed"))
        #expect(replacement.facetTags.contains(.preference))
        #expect(replacement.entities.contains(where: { $0.normalizedValue == "zed" }))

        let active = try await index.recall(mode: .kind(.profile), limit: 10)
        #expect(active.records.count == 1)
        #expect(active.records.first?.id == replacement.id)

        let historical = try await index.recall(
            mode: .kind(.profile),
            limit: 10,
            statuses: [.active, .superseded]
        )
        #expect(historical.records.contains(where: { $0.id == first.id && $0.status == .superseded }))
    }

    @Test
    func concurrentIdempotentIngestDoesNotDuplicateExactFact() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("index.sqlite")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<6 {
                group.addTask {
                    _ = try? await index.save(
                        text: "The repo uses SQLite as the canonical storage layer.",
                        kind: .fact
                    )
                }
            }
        }

        let facts = try await index.recall(mode: .kind(.fact), limit: 10)
        #expect(facts.records.count == 1)
    }
}
