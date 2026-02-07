import CryptoKit
import Foundation
import Testing
@testable import QMDKit

private actor PerfEmbeddingProvider: EmbeddingProvider {
    let identifier = "perf-mock"
    private let tokenizer = DefaultTokenizer()

    func embed(texts: [String]) async throws -> [[Float]] {
        texts.map { text in
            var vector = Array(repeating: Float.zero, count: 64)
            for token in tokenizer.tokenize(text) {
                let digest = SHA256.hash(data: Data(token.utf8))
                let index = digest.withUnsafeBytes { raw in Int(raw[0]) % 64 }
                vector[index] += 1
            }
            return vector
        }
    }
}

struct QMDKitPerformanceTests {
    @Test
    func searchLatencyBaseline() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qmdkit-perf")
            .appendingPathComponent(UUID().uuidString)
        let docs = root.appendingPathComponent("docs")
        let db = root.appendingPathComponent("index.sqlite")

        try FileManager.default.createDirectory(at: docs, withIntermediateDirectories: true)

        for i in 0..<1_000 {
            let url = docs.appendingPathComponent("doc-\(i).md")
            let text = """
            # Doc \(i)
            Swift concurrency actors async await retrieval vector bm25 recency.
            token\(i) token\(i + 1) token\(i + 2)
            """
            try text.write(to: url, atomically: true, encoding: .utf8)
        }

        let index = try QMDIndex(
            configuration: QMDConfiguration(
                databaseURL: db,
                embeddingProvider: PerfEmbeddingProvider()
            )
        )

        try await index.rebuildIndex(from: [docs])

        let start = Date()
        _ = try await index.search(SearchQuery(text: "swift retrieval bm25", limit: 20))
        let duration = Date().timeIntervalSince(start)

        #expect(duration < 1.0)
    }
}
