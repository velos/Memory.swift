import Foundation
import MemoryStorage
import Testing

struct MemoryStorageLexicalTests {
    @Test
    func lexicalDocumentSearchAggregatesWeakMatchesAcrossChunks() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("document-lexical.sqlite")
        let storage = try MemoryStorage(databaseURL: dbURL)

        try await storage.replaceDocument(
            makeDocument(
                path: "/tmp/target.md",
                chunks: [
                    "Greenwood neighborhood planning note",
                    "urban shade corridor proposal",
                    "reforestation volunteer schedule",
                    "tree nursery purchase order",
                    "canopy baseline mapping",
                ]
            )
        )

        let repeatedAnchors = ["Greenwood", "urban", "reforestation", "tree", "canopy"]
        for i in 0..<25 {
            let anchor = repeatedAnchors[i % repeatedAnchors.count]
            let repeated = Array(repeating: anchor, count: 30).joined(separator: " ")
            try await storage.replaceDocument(
                makeDocument(
                    path: "/tmp/distractor-\(i).md",
                    chunks: ["\(repeated) unrelated parking agenda \(i)"]
                )
            )
        }

        let hits = try await storage.lexicalDocumentSearch(
            query: "Greenwood urban reforestation tree canopy",
            limit: 3
        )
        let metadata = try await storage.fetchChunkMetadata(chunkIDs: hits.map(\.chunkID))
        let paths = Set(metadata.map(\.documentPath))

        #expect(paths.contains("/tmp/target.md"))
    }

    @Test
    func scopedLexicalSearchOnlyRanksAllowedChunks() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("scoped-lexical.sqlite")
        let storage = try MemoryStorage(databaseURL: dbURL)

        try await storage.replaceDocument(
            makeDocument(
                path: "/tmp/scope/target.md",
                chunks: [
                    "Maya booked the lantern tour at Caldera Bay for June 12.",
                    "The harbor note mentions tides and ferry timing.",
                ]
            )
        )
        try await storage.replaceDocument(
            makeDocument(
                path: "/tmp/scope/distractor.md",
                chunks: ["Routine travel planning note without the requested destination."]
            )
        )
        try await storage.replaceDocument(
            makeDocument(
                path: "/tmp/outside/global-best.md",
                chunks: [
                    Array(repeating: "Maya lantern tour Caldera Bay", count: 20).joined(separator: " ")
                ]
            )
        )

        let allowedChunkIDs = Set(try await storage.fetchChunkIDs(documentPathPrefix: "/tmp/scope"))
        let hits = try await storage.lexicalSearch(
            query: "Maya lantern tour Caldera Bay",
            limit: 5,
            allowedChunkIDs: allowedChunkIDs
        )
        let metadata = try await storage.fetchChunkMetadata(chunkIDs: hits.map(\.chunkID))
        let paths = metadata.map(\.documentPath)

        #expect(paths.first == "/tmp/scope/target.md")
        #expect(!paths.contains("/tmp/outside/global-best.md"))
    }

    private func makeDocument(path: String, chunks: [String]) -> StoredDocumentInput {
        StoredDocumentInput(
            path: path,
            title: "Test Document",
            modifiedAt: Date(timeIntervalSince1970: 1_700_000_000),
            checksum: UUID().uuidString,
            chunks: chunks.enumerated().map { index, content in
                StoredChunkInput(
                    ordinal: index,
                    content: content,
                    tokenCount: content.split(separator: " ").count,
                    embedding: [1, 2, 3],
                    norm: 3.7416573868
                )
            }
        )
    }
}
