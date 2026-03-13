import Foundation
import Testing
@testable import Memory

struct MemoryTypingTests {
    @Test
    func manualFrontmatterOverrideBeatsAutoClassifier() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("journal.md"),
            """
            ---
            memory_type: episodic
            ---
            # Daily Notes
            We discussed release planning and blockers.
            """
        )

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: MockEmbeddingProvider(),
            memoryTyping: MemoryTypingConfiguration(
                mode: .automatic,
                classifier: StaticMemoryTypeClassifier(
                    assignment: MemoryTypeAssignment(
                        type: .semantic,
                        source: .automatic,
                        confidence: 0.9
                    )
                ),
                fallbackType: .factual
            )
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let results = try await index.search(SearchQuery(text: "release planning", limit: 3))
        #expect(results.isEmpty == false)
        #expect(results.first?.memoryType == .episodic)
        #expect(results.first?.memoryTypeSource == .manual)
    }

    @Test
    func autoClassifierAppliesWhenNoManualOverrideExists() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("concepts.md"),
            "This document explains architecture patterns and design concepts."
        )

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: MockEmbeddingProvider(),
            memoryTyping: MemoryTypingConfiguration(
                mode: .automatic,
                classifier: StaticMemoryTypeClassifier(
                    assignment: MemoryTypeAssignment(
                        type: .semantic,
                        source: .automatic,
                        confidence: 0.82
                    )
                ),
                fallbackType: .factual
            )
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let results = try await index.search(SearchQuery(text: "architecture concepts", limit: 3))
        #expect(results.isEmpty == false)
        #expect(results.first?.memoryType == .semantic)
        #expect(results.first?.memoryTypeSource == .automatic)
        #expect((results.first?.memoryTypeConfidence ?? 0) > 0.8)
    }

    @Test
    func fallbackTypeIsUsedWhenClassifierFails() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("note.md"),
            "Short note with no explicit memory type metadata."
        )

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: MockEmbeddingProvider(),
            memoryTyping: MemoryTypingConfiguration(
                mode: .automatic,
                classifier: ThrowingMemoryTypeClassifier(),
                fallbackType: .contextual
            )
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let results = try await index.search(SearchQuery(text: "short note", limit: 3))
        #expect(results.isEmpty == false)
        #expect(results.first?.memoryType == .contextual)
        #expect(results.first?.memoryTypeSource == .fallback)
    }

    @Test
    func chunkOverrideTakesPrecedenceOverDocumentType() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")
        let file = docs.appendingPathComponent("timeline.md")

        try writeFile(file, "Release timeline milestone planning alpha beta gamma")

        let config = MemoryConfiguration(
            databaseURL: dbURL,
            embeddingProvider: MockEmbeddingProvider(),
            memoryTyping: MemoryTypingConfiguration(
                mode: .automatic,
                classifier: StaticMemoryTypeClassifier(
                    assignment: MemoryTypeAssignment(
                        type: .semantic,
                        source: .automatic,
                        confidence: 0.75
                    )
                ),
                fallbackType: .factual
            )
        )

        let index = try MemoryIndex(configuration: config)
        try await index.rebuildIndex(from: [docs])

        let before = try await index.search(SearchQuery(text: "timeline planning", limit: 3))
        #expect(before.isEmpty == false)
        let chunkID = try #require(before.first?.chunkID)
        #expect(before.first?.memoryType == .semantic)

        try await index.setChunkMemoryTypeOverride(chunkID: chunkID, type: .temporal)
        let updated = try await index.getChunk(id: chunkID)
        #expect(updated?.memoryType == .temporal)
        #expect(updated?.memoryTypeSource == .manual)

        let temporalOnly = try await index.search(
            SearchQuery(text: "timeline planning", limit: 3, memoryTypes: [.temporal])
        )
        #expect(temporalOnly.contains(where: { $0.chunkID == chunkID }))

        let semanticOnly = try await index.search(
            SearchQuery(text: "timeline planning", limit: 3, memoryTypes: [.semantic])
        )
        #expect(semanticOnly.contains(where: { $0.chunkID == chunkID }) == false)
    }

    @Test
    func searchMemoryTypeFiltersReturnOnlyMatchingTypes() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")

        try writeFile(
            docs.appendingPathComponent("semantic.md"),
            """
            ---
            memory_type: semantic
            ---
            Shared keyword alpha appears in this conceptual document.
            """
        )
        try writeFile(
            docs.appendingPathComponent("procedural.md"),
            """
            ---
            memory_type: procedural
            ---
            Shared keyword alpha appears in this checklist and process note.
            """
        )

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider()
            )
        )
        try await index.rebuildIndex(from: [docs])

        let filtered = try await index.search(
            SearchQuery(text: "shared keyword alpha", limit: 20, memoryTypes: [.semantic])
        )

        #expect(filtered.isEmpty == false)
        #expect(filtered.allSatisfy { $0.memoryType == .semantic })
        #expect(filtered.contains(where: { $0.documentPath.hasSuffix("semantic.md") }))
        #expect(filtered.contains(where: { $0.documentPath.hasSuffix("procedural.md") }) == false)
    }

    @Test
    func setDocumentMemoryTypeUpdatesAllChunksForThatDocument() async throws {
        let root = try makeTemporaryDirectory()
        let docs = root.appendingPathComponent("docs")
        let dbURL = root.appendingPathComponent("index.sqlite")
        let file = docs.appendingPathComponent("social.md")

        try writeFile(file, "Coordination notes with team stakeholders and communication details.")

        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: dbURL,
                embeddingProvider: MockEmbeddingProvider(),
                memoryTyping: MemoryTypingConfiguration(
                    mode: .manualOnly,
                    classifier: nil,
                    fallbackType: .factual
                )
            )
        )
        try await index.rebuildIndex(from: [docs])

        let before = try await index.search(SearchQuery(text: "coordination team", limit: 5))
        #expect(before.isEmpty == false)
        #expect(before.first?.memoryType == .factual)

        try await index.setDocumentMemoryType(path: file.path, type: .social)

        let social = try await index.search(
            SearchQuery(text: "coordination team", limit: 5, memoryTypes: [.social])
        )
        #expect(social.isEmpty == false)
        #expect(social.allSatisfy { $0.memoryType == .social })
        #expect(social.allSatisfy { $0.memoryTypeSource == .manual })

        let factual = try await index.search(
            SearchQuery(text: "coordination team", limit: 5, memoryTypes: [.factual])
        )
        #expect(factual.isEmpty)
    }
}
