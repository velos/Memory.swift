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
            category: .decision,
            importance: 0.9
        )
        _ = try await index.save(
            text: "I prefer shorter status updates in the morning.",
            category: .preference,
            importance: 0.4
        )

        let typed = try await index.recall(
            mode: .typed(category: .decision),
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
            mode: .typed(category: .decision),
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
            category: .fact
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
            category: .fact
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
}
