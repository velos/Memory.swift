import Foundation
import Testing
@testable import Memory

struct MemoryExternalAPITests {
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
}
