import Foundation
import Memory
import MemoryNaturalLanguage

public struct CoreMLDefaultModels: Sendable {
    public var embedding: URL
    public var reranker: URL?

    public init(embedding: URL, reranker: URL? = nil) {
        self.embedding = embedding
        self.reranker = reranker
    }
}

public extension MemoryConfiguration {
    static func coreMLDefault(
        databaseURL: URL,
        models: CoreMLDefaultModels,
        structuredQueryExpander: (any StructuredQueryExpander)? = GenericStructuredQueryExpander(),
        contentTagger: (any ContentTagger)? = nil,
        memoryExtractor: (any MemoryExtractor)? = nil,
        recallPlanner: (any RecallPlanner)? = nil,
        queryAnalyzer: (any QueryAnalyzer)? = NLQueryAnalyzer(),
        chunker: any Chunker = DefaultChunker(),
        supportedFileExtensions: Set<String> = defaultSupportedExtensions,
        semanticCandidateLimit: Int = 500,
        lexicalCandidateLimit: Int = 500,
        fusionK: Double = 60,
        positionAwareBlending: PositionAwareBlending = .default
    ) throws -> MemoryConfiguration {
        let embeddingProvider = try CoreMLEmbeddingProvider(modelURL: models.embedding)
        let rerankerProvider: (any Reranker)?
        if let rerankerURL = models.reranker {
            rerankerProvider = try CoreMLReranker(modelURL: rerankerURL)
        } else {
            rerankerProvider = nil
        }

        return MemoryConfiguration(
            databaseURL: databaseURL,
            embeddingProvider: embeddingProvider,
            structuredQueryExpander: structuredQueryExpander,
            reranker: rerankerProvider,
            contentTagger: contentTagger,
            memoryExtractor: memoryExtractor,
            recallPlanner: recallPlanner,
            queryAnalyzer: queryAnalyzer,
            tokenizer: NLWordTokenizer(),
            chunker: chunker,
            supportedFileExtensions: supportedFileExtensions,
            semanticCandidateLimit: semanticCandidateLimit,
            lexicalCandidateLimit: lexicalCandidateLimit,
            fusionK: fusionK,
            positionAwareBlending: positionAwareBlending,
            ftsTokenizer: NLLemmatizingTokenizer()
        )
    }
}
