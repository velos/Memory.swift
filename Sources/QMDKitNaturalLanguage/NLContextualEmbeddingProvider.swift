import Foundation
import NaturalLanguage
import Memory

public actor NLContextualEmbeddingProvider: EmbeddingProvider {
    public let identifier: String

    private let fixedLanguage: NLLanguage?
    private var cachedEmbeddings: [NLLanguage: NLContextualEmbedding]
    private var loadedModelIdentifiers: Set<String>

    public init(
        identifier: String = "nl-contextual-embedding",
        language: NLLanguage? = nil
    ) {
        self.identifier = identifier
        self.fixedLanguage = language
        self.cachedEmbeddings = [:]
        self.loadedModelIdentifiers = []
    }

    public func embed(texts: [String]) async throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)

        for text in texts {
            vectors.append(try embedSingle(text: text))
        }

        return vectors
    }

    private func embedSingle(text: String) throws -> [Float] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw QMDError.embedding("Cannot embed empty text")
        }

        let detectedLanguage = fixedLanguage ?? NLLanguageRecognizer.dominantLanguage(for: trimmed) ?? .english
        let language = detectedLanguage
        let embedding: NLContextualEmbedding
        if let resolved = try resolveEmbedding(for: language) {
            embedding = resolved
        } else if let fallback = try resolveEmbedding(for: .english) {
            embedding = fallback
        } else {
            throw QMDError.embedding(
                "No NLContextualEmbedding available for language \(language.rawValue), and english fallback is unavailable"
            )
        }

        try ensureLoaded(embedding)

        let result = try embedding.embeddingResult(for: trimmed, language: language)

        var pooled: [Double] = []
        var vectorCount = 0

        result.enumerateTokenVectors(in: trimmed.startIndex..<trimmed.endIndex) { vector, _ in
            if pooled.isEmpty {
                pooled = Array(repeating: 0, count: vector.count)
            }

            guard vector.count == pooled.count else {
                return true
            }

            for index in vector.indices {
                pooled[index] += vector[index]
            }

            vectorCount += 1
            return true
        }

        if vectorCount == 0,
           let (vector, _) = result.tokenVector(at: trimmed.startIndex) {
            pooled = vector
            vectorCount = 1
        }

        guard vectorCount > 0, !pooled.isEmpty else {
            throw QMDError.embedding("Contextual embedding returned no token vectors")
        }

        let divisor = Double(vectorCount)
        return pooled.map { Float($0 / divisor) }
    }

    private func resolveEmbedding(for language: NLLanguage) throws -> NLContextualEmbedding? {
        if let cached = cachedEmbeddings[language] {
            return cached
        }

        guard let embedding = NLContextualEmbedding(language: language) else {
            return nil
        }

        cachedEmbeddings[language] = embedding
        return embedding
    }

    private func ensureLoaded(_ embedding: NLContextualEmbedding) throws {
        if loadedModelIdentifiers.contains(embedding.modelIdentifier) {
            return
        }

        do {
            try embedding.load()
            loadedModelIdentifiers.insert(embedding.modelIdentifier)
        } catch {
            throw QMDError.embedding(
                "Failed to load NLContextualEmbedding model \(embedding.modelIdentifier): \(error.localizedDescription)"
            )
        }
    }
}

public extension QMDConfiguration {
    static func naturalLanguageDefault(
        databaseURL: URL,
        language: NLLanguage? = nil,
        queryExpander: (any QueryExpander)? = nil,
        reranker: (any Reranker)? = nil,
        tokenizer: any Tokenizer = DefaultTokenizer(),
        chunker: any Chunker = DefaultChunker(),
        supportedFileExtensions: Set<String> = defaultSupportedExtensions,
        semanticCandidateLimit: Int = 200,
        lexicalCandidateLimit: Int = 200,
        fusionK: Double = 60,
        positionAwareBlending: PositionAwareBlending = .default
    ) -> QMDConfiguration {
        QMDConfiguration(
            databaseURL: databaseURL,
            embeddingProvider: NLContextualEmbeddingProvider(language: language),
            queryExpander: queryExpander,
            reranker: reranker,
            tokenizer: tokenizer,
            chunker: chunker,
            supportedFileExtensions: supportedFileExtensions,
            semanticCandidateLimit: semanticCandidateLimit,
            lexicalCandidateLimit: lexicalCandidateLimit,
            fusionK: fusionK,
            positionAwareBlending: positionAwareBlending
        )
    }
}
