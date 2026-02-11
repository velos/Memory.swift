import Foundation
import MLX
import Memory

public actor MLXEmbeddingProvider: EmbeddingProvider {
    public typealias EmbedBatch = @Sendable ([String]) async throws -> [[Float]]

    public let identifier: String
    private let embedBatch: EmbedBatch

    public init(
        identifier: String = "mlx-embedders",
        embedBatch: @escaping EmbedBatch
    ) {
        self.identifier = identifier
        self.embedBatch = embedBatch
    }

    public func embed(texts: [String]) async throws -> [[Float]] {
        let vectors = try await embedBatch(texts)
        guard vectors.count == texts.count else {
            throw QMDError.embedding(
                "MLXEmbeddingProvider returned \(vectors.count) vectors for \(texts.count) inputs"
            )
        }
        return vectors
    }
}

public extension QMDConfiguration {
    static func mlxDefault(
        databaseURL: URL,
        embedBatch: @escaping MLXEmbeddingProvider.EmbedBatch,
        reranker: (any Reranker)? = nil,
        tokenizer: any Tokenizer = DefaultTokenizer(),
        chunker: any Chunker = DefaultChunker(),
        supportedFileExtensions: Set<String> = defaultSupportedExtensions,
        semanticCandidateLimit: Int = 200,
        lexicalCandidateLimit: Int = 200,
        fusionK: Double = 60
    ) -> QMDConfiguration {
        QMDConfiguration(
            databaseURL: databaseURL,
            embeddingProvider: MLXEmbeddingProvider(embedBatch: embedBatch),
            reranker: reranker,
            tokenizer: tokenizer,
            chunker: chunker,
            supportedFileExtensions: supportedFileExtensions,
            semanticCandidateLimit: semanticCandidateLimit,
            lexicalCandidateLimit: lexicalCandidateLimit,
            fusionK: fusionK
        )
    }
}
