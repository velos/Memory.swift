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
