import CryptoKit
import Foundation
@testable import Memory

actor MockEmbeddingProvider: EmbeddingProvider {
    let identifier = "mock-embedding"
    let dimension: Int
    private let tokenizer = DefaultTokenizer()

    init(dimension: Int = 64) {
        self.dimension = dimension
    }

    func embed(texts: [String]) async throws -> [[Float]] {
        texts.map(embedding(for:))
    }

    private func embedding(for text: String) -> [Float] {
        var vector = Array(repeating: Float.zero, count: dimension)

        for token in tokenizer.tokenize(text) {
            let digest = SHA256.hash(data: Data(token.utf8))
            let index = digest.withUnsafeBytes { raw in
                Int(raw[0]) % dimension
            }
            vector[index] += 1
        }

        return vector
    }
}

actor ConstantEmbeddingProvider: EmbeddingProvider {
    let identifier = "constant-embedding"

    func embed(texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [1, 1, 1, 1] }
    }
}

actor StaticStructuredQueryExpander: StructuredQueryExpander {
    let identifier = "static-structured-query-expander"
    let expansion: StructuredQueryExpansion

    init(
        lexicalQueries: [String] = [],
        semanticQueries: [String] = [],
        hypotheticalDocuments: [String] = [],
        facetHints: [FacetHint] = [],
        entities: [MemoryEntity] = [],
        topics: [String] = []
    ) {
        self.expansion = StructuredQueryExpansion(
            lexicalQueries: lexicalQueries,
            semanticQueries: semanticQueries,
            hypotheticalDocuments: hypotheticalDocuments,
            facetHints: facetHints,
            entities: entities,
            topics: topics
        )
    }

    func expand(query: SearchQuery, analysis: QueryAnalysis, limit: Int) async throws -> StructuredQueryExpansion {
        expansion
    }
}

actor RecordingStructuredQueryExpander: StructuredQueryExpander {
    let identifier = "recording-structured-query-expander"
    let expansion: StructuredQueryExpansion
    private var invocationCount = 0

    init(
        lexicalQueries: [String] = [],
        semanticQueries: [String] = [],
        hypotheticalDocuments: [String] = [],
        facetHints: [FacetHint] = [],
        entities: [MemoryEntity] = [],
        topics: [String] = []
    ) {
        self.expansion = StructuredQueryExpansion(
            lexicalQueries: lexicalQueries,
            semanticQueries: semanticQueries,
            hypotheticalDocuments: hypotheticalDocuments,
            facetHints: facetHints,
            entities: entities,
            topics: topics
        )
    }

    func expand(query: SearchQuery, analysis: QueryAnalysis, limit: Int) async throws -> StructuredQueryExpansion {
        invocationCount += 1
        return expansion
    }

    func calls() -> Int {
        invocationCount
    }
}

actor CountingEmbeddingProvider: EmbeddingProvider {
    let identifier = "counting-embedding-provider"
    let dimension: Int
    private let tokenizer = DefaultTokenizer()
    private var singleCallCount = 0
    private var batchCallSizes: [Int] = []

    init(dimension: Int = 64) {
        self.dimension = dimension
    }

    func embed(text: String) async throws -> [Float] {
        singleCallCount += 1
        return embedding(for: text)
    }

    func embed(texts: [String]) async throws -> [[Float]] {
        batchCallSizes.append(texts.count)
        return texts.map(embedding(for:))
    }

    func resetStats() {
        singleCallCount = 0
        batchCallSizes = []
    }

    func stats() -> (singleCalls: Int, batchSizes: [Int]) {
        (singleCalls: singleCallCount, batchSizes: batchCallSizes)
    }

    private func embedding(for text: String) -> [Float] {
        var vector = Array(repeating: Float.zero, count: dimension)

        for token in tokenizer.tokenize(text) {
            let digest = SHA256.hash(data: Data(token.utf8))
            let index = digest.withUnsafeBytes { raw in
                Int(raw[0]) % dimension
            }
            vector[index] += 1
        }

        return vector
    }
}

actor ClosureReranker: Reranker {
    let identifier = "closure-reranker"
    let scoreForCandidate: @Sendable (SearchResult) -> Double

    init(scoreForCandidate: @escaping @Sendable (SearchResult) -> Double) {
        self.scoreForCandidate = scoreForCandidate
    }

    func rerank(query: SearchQuery, candidates: [SearchResult]) async throws -> [RerankAssessment] {
        candidates.map { candidate in
            RerankAssessment(
                chunkID: candidate.chunkID,
                relevance: scoreForCandidate(candidate),
                rationale: nil
            )
        }
    }
}

actor StaticContentTagger: ContentTagger {
    let identifier = "static-content-tagger"
    private let mapping: [(needle: String, tags: [ContentTag])]

    init(tagsByNeedle: [String: [ContentTag]]) {
        self.mapping = tagsByNeedle
            .map { (needle: $0.key.lowercased(), tags: $0.value) }
            .sorted { lhs, rhs in lhs.needle.count > rhs.needle.count }
    }

    func tag(text: String, kind: DocumentKind, sourceURL: URL?) async throws -> [ContentTag] {
        let haystack = text.lowercased()
        for entry in mapping where haystack.contains(entry.needle) {
            return entry.tags
        }
        return []
    }
}

func makeTemporaryDirectory(function: String = #function) throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("memory-tests")
        .appendingPathComponent(function.replacingOccurrences(of: " ", with: "_"))
        .appendingPathComponent(UUID().uuidString)

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

func writeFile(_ url: URL, _ content: String, modifiedAt: Date? = nil) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try content.write(to: url, atomically: true, encoding: .utf8)

    if let modifiedAt {
        try FileManager.default.setAttributes([.modificationDate: modifiedAt], ofItemAtPath: url.path)
    }
}
