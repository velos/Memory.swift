import CryptoKit
import Foundation
import Testing
@testable import Memory
import MemoryNaturalLanguage

private actor IntegrationMockEmbeddingProvider: EmbeddingProvider {
    let identifier = "integration-mock"
    private let tokenizer = DefaultTokenizer()

    func embed(texts: [String]) async throws -> [[Float]] {
        texts.map { text in
            var vector = Array(repeating: Float.zero, count: 32)
            for token in tokenizer.tokenize(text) {
                let digest = SHA256.hash(data: Data(token.utf8))
                let index = digest.withUnsafeBytes { raw in Int(raw[0]) % 32 }
                vector[index] += 1
            }
            return vector
        }
    }
}

private func makeTempDir(_ name: String = #function) throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("memory-integration")
        .appendingPathComponent(name)
        .appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func write(_ url: URL, _ text: String) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try text.write(to: url, atomically: true, encoding: .utf8)
}

struct MemoryIntegrationTests {
    @Test
    func incrementalSyncAndRemoval() async throws {
        let root = try makeTempDir()
        let docs = root.appendingPathComponent("docs")
        let db = root.appendingPathComponent("index.sqlite")

        let a = docs.appendingPathComponent("a.md")
        let b = docs.appendingPathComponent("b.md")

        try write(a, "alpha one")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: db,
                embeddingProvider: IntegrationMockEmbeddingProvider()
            )
        )

        try await index.rebuildIndex(from: [docs])
        let first = try await index.search(SearchQuery(text: "alpha", limit: 5))
        #expect(first.isEmpty == false)

        try write(a, "alpha updated content")
        try write(b, "bravo second file")
        try await index.syncDocuments([a, b])

        let second = try await index.search(SearchQuery(text: "bravo", limit: 5))
        #expect(second.first?.documentPath.contains("b.md") == true)

        try await index.removeDocuments(at: [b])
        let afterRemoval = try await index.search(SearchQuery(text: "bravo", limit: 5))
        #expect(afterRemoval.allSatisfy { !$0.documentPath.contains("b.md") })
    }

    @Test
    func persistenceAcrossReinitialization() async throws {
        let root = try makeTempDir()
        let docs = root.appendingPathComponent("docs")
        let db = root.appendingPathComponent("index.sqlite")

        try write(docs.appendingPathComponent("note.md"), "swift package manager and async actors")

        do {
            let index = try MemoryIndex(
                configuration: MemoryConfiguration(
                    databaseURL: db,
                    embeddingProvider: IntegrationMockEmbeddingProvider()
                )
            )
            try await index.rebuildIndex(from: [docs])
        }

        let reloaded = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: db,
                embeddingProvider: IntegrationMockEmbeddingProvider()
            )
        )

        let results = try await reloaded.search(SearchQuery(text: "package manager", limit: 5))
        #expect(results.isEmpty == false)
    }

    @Test
    func naturalLanguageDefaultConfigurationConstructs() {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let config = MemoryConfiguration.naturalLanguageDefault(databaseURL: root.appendingPathComponent("index.sqlite"))

        #expect(config.semanticCandidateLimit == 200)
        #expect(config.lexicalCandidateLimit == 200)
    }

    @Test
    func canonicalMemoryPersistsAcrossReinitialization() async throws {
        let root = try makeTempDir()
        let db = root.appendingPathComponent("index.sqlite")
        let firstIndex = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: db,
                embeddingProvider: IntegrationMockEmbeddingProvider()
            )
        )

        let saved = try await firstIndex.save(
            text: "Memory.swift release timeline is September 18 and needs a launch checklist.",
            kind: .commitment,
            facetTags: [.project, .timeSensitive],
            entities: [
                MemoryEntity(
                    label: .project,
                    value: "Memory.swift",
                    normalizedValue: "memory.swift"
                ),
            ],
            topics: ["release timeline"],
            canonicalKey: "memory.swift.release.timeline"
        )

        let secondIndex = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: db,
                embeddingProvider: IntegrationMockEmbeddingProvider()
            )
        )

        let recalled = try await secondIndex.recall(
            mode: .recent,
            limit: 5,
            kinds: [.commitment],
            facets: [.project, .timeSensitive],
            entityValues: ["memory.swift"],
            topics: ["release timeline"]
        )

        #expect(recalled.records.contains(where: { $0.id == saved.id }))
    }
}
