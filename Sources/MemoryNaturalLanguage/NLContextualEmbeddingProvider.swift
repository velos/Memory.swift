import Foundation
import NaturalLanguage
import Memory

public enum PoolingStrategy: String, Sendable {
    case mean
    case max
    case weightedMean
}

public actor NLContextualEmbeddingProvider: EmbeddingProvider {
    public let identifier: String

    private let fixedLanguage: NLLanguage?
    private let poolingStrategy: PoolingStrategy
    private var cachedEmbeddings: [NLLanguage: NLContextualEmbedding]
    private var loadedModelIdentifiers: Set<String>

    public init(
        identifier: String = "nl-contextual-embedding",
        language: NLLanguage? = nil,
        poolingStrategy: PoolingStrategy = .max
    ) {
        self.identifier = identifier
        self.fixedLanguage = language
        self.poolingStrategy = poolingStrategy
        self.cachedEmbeddings = [:]
        self.loadedModelIdentifiers = []
    }

    public func embed(texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        if texts.count <= 4 {
            var vectors: [[Float]] = []
            vectors.reserveCapacity(texts.count)
            for text in texts {
                vectors.append(try embedSingle(text: text))
            }
            return vectors
        }

        for text in texts {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let language = fixedLanguage ?? NLLanguageRecognizer.dominantLanguage(for: trimmed) ?? .english
            if let resolved = try resolveEmbedding(for: language) {
                try ensureLoaded(resolved)
            } else if let fallback = try resolveEmbedding(for: .english) {
                try ensureLoaded(fallback)
            }
            break
        }

        let indexedTexts = texts.enumerated().map { ($0.offset, $0.element) }
        let provider = self

        return try await withThrowingTaskGroup(of: (Int, [Float]).self) { group in
            let maxConcurrency = 4
            var results: [(Int, [Float])] = []
            results.reserveCapacity(texts.count)
            var nextIndex = 0

            for _ in 0..<min(maxConcurrency, indexedTexts.count) {
                let (idx, text) = indexedTexts[nextIndex]
                nextIndex += 1
                group.addTask {
                    let vector = try await provider.embedSingle(text: text)
                    return (idx, vector)
                }
            }

            for try await result in group {
                results.append(result)
                if nextIndex < indexedTexts.count {
                    let (idx, text) = indexedTexts[nextIndex]
                    nextIndex += 1
                    group.addTask {
                        let vector = try await provider.embedSingle(text: text)
                        return (idx, vector)
                    }
                }
            }

            return results.sorted { $0.0 < $1.0 }.map(\.1)
        }
    }

    private func embedSingle(text: String) throws -> [Float] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw MemoryError.embedding("Cannot embed empty text")
        }

        let detectedLanguage = fixedLanguage ?? NLLanguageRecognizer.dominantLanguage(for: trimmed) ?? .english
        let language = detectedLanguage
        let embedding: NLContextualEmbedding
        if let resolved = try resolveEmbedding(for: language) {
            embedding = resolved
        } else if let fallback = try resolveEmbedding(for: .english) {
            embedding = fallback
        } else {
            throw MemoryError.embedding(
                "No NLContextualEmbedding available for language \(language.rawValue), and english fallback is unavailable"
            )
        }

        try ensureLoaded(embedding)

        let result = try embedding.embeddingResult(for: trimmed, language: language)

        switch poolingStrategy {
        case .mean:
            return try poolMean(result: result, text: trimmed)
        case .max:
            return try poolMax(result: result, text: trimmed)
        case .weightedMean:
            return try poolWeightedMean(result: result, text: trimmed)
        }
    }

    private func poolMean(result: NLContextualEmbeddingResult, text: String) throws -> [Float] {
        var pooled: [Double] = []
        var vectorCount = 0

        result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vector, _ in
            if pooled.isEmpty {
                pooled = Array(repeating: 0, count: vector.count)
            }
            guard vector.count == pooled.count else { return true }
            for i in vector.indices { pooled[i] += vector[i] }
            vectorCount += 1
            return true
        }

        if vectorCount == 0, let (vector, _) = result.tokenVector(at: text.startIndex) {
            return vector.map { Float($0) }
        }

        guard vectorCount > 0, !pooled.isEmpty else {
            throw MemoryError.embedding("Contextual embedding returned no token vectors")
        }

        let divisor = Double(vectorCount)
        return pooled.map { Float($0 / divisor) }
    }

    private func poolMax(result: NLContextualEmbeddingResult, text: String) throws -> [Float] {
        var pooled: [Double] = []
        var vectorCount = 0

        result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vector, _ in
            if pooled.isEmpty {
                pooled = vector
            } else if vector.count == pooled.count {
                for i in vector.indices {
                    if vector[i] > pooled[i] { pooled[i] = vector[i] }
                }
            }
            vectorCount += 1
            return true
        }

        if vectorCount == 0, let (vector, _) = result.tokenVector(at: text.startIndex) {
            return vector.map { Float($0) }
        }

        guard vectorCount > 0, !pooled.isEmpty else {
            throw MemoryError.embedding("Contextual embedding returned no token vectors")
        }

        return pooled.map { Float($0) }
    }

    private func poolWeightedMean(result: NLContextualEmbeddingResult, text: String) throws -> [Float] {
        var pooled: [Double] = []
        var totalWeight = 0.0
        var vectorCount = 0

        result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vector, _ in
            if pooled.isEmpty {
                pooled = Array(repeating: 0, count: vector.count)
            }
            guard vector.count == pooled.count else { return true }
            vectorCount += 1
            let weight = Double(vectorCount)
            totalWeight += weight
            for i in vector.indices { pooled[i] += vector[i] * weight }
            return true
        }

        if vectorCount == 0, let (vector, _) = result.tokenVector(at: text.startIndex) {
            return vector.map { Float($0) }
        }

        guard vectorCount > 0, !pooled.isEmpty, totalWeight > 0 else {
            throw MemoryError.embedding("Contextual embedding returned no token vectors")
        }

        return pooled.map { Float($0 / totalWeight) }
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
            throw MemoryError.embedding(
                "Failed to load NLContextualEmbedding model \(embedding.modelIdentifier): \(error.localizedDescription)"
            )
        }
    }
}

public struct NLWordTokenizer: Tokenizer, Sendable {
    public init() {}

    public func tokenize(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        var tokens: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let token = String(text[range]).lowercased()
            if !token.isEmpty {
                tokens.append(token)
            }
            return true
        }
        return tokens
    }
}

public struct NLLemmatizingTokenizer: Tokenizer, Sendable {
    public init() {}

    public func tokenize(_ text: String) -> [String] {
        let tagger = NLTagger(tagSchemes: [.lemma])
        tagger.string = text
        var tokens: [String] = []
        tagger.enumerateTags(
            in: text.startIndex..<text.endIndex,
            unit: .word,
            scheme: .lemma
        ) { tag, range in
            let lemma = tag?.rawValue
            let raw = String(text[range])
            let token = (lemma ?? raw).lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            if !token.isEmpty {
                tokens.append(token)
            }
            return true
        }
        return tokens
    }
}

public extension MemoryConfiguration {
    static func naturalLanguageDefault(
        databaseURL: URL,
        language: NLLanguage? = nil,
        poolingStrategy: PoolingStrategy = .mean,
        queryExpander: (any QueryExpander)? = nil,
        reranker: (any Reranker)? = nil,
        contentTagger: (any ContentTagger)? = nil,
        memoryExtractor: (any MemoryExtractor)? = nil,
        recallPlanner: (any RecallPlanner)? = nil,
        queryAnalyzer: (any QueryAnalyzer)? = NLQueryAnalyzer(),
        tokenizer: any Tokenizer = NLWordTokenizer(),
        chunker: any Chunker = DefaultChunker(),
        supportedFileExtensions: Set<String> = defaultSupportedExtensions,
        semanticCandidateLimit: Int = 200,
        lexicalCandidateLimit: Int = 200,
        fusionK: Double = 60,
        positionAwareBlending: PositionAwareBlending = .default,
        ftsTokenizer: (any Tokenizer)? = NLLemmatizingTokenizer()
    ) -> MemoryConfiguration {
        MemoryConfiguration(
            databaseURL: databaseURL,
            embeddingProvider: NLContextualEmbeddingProvider(language: language, poolingStrategy: poolingStrategy),
            queryExpander: queryExpander,
            reranker: reranker,
            contentTagger: contentTagger,
            memoryExtractor: memoryExtractor,
            recallPlanner: recallPlanner,
            queryAnalyzer: queryAnalyzer,
            tokenizer: tokenizer,
            chunker: chunker,
            supportedFileExtensions: supportedFileExtensions,
            semanticCandidateLimit: semanticCandidateLimit,
            lexicalCandidateLimit: lexicalCandidateLimit,
            fusionK: fusionK,
            positionAwareBlending: positionAwareBlending,
            ftsTokenizer: ftsTokenizer
        )
    }
}
