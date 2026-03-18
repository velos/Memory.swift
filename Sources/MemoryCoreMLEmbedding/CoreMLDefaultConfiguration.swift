import Foundation
import Memory
import MemoryNaturalLanguage

public struct CoreMLDefaultModels: Sendable {
    public var embedding: URL

    public init(embedding: URL) {
        self.embedding = embedding
    }
}

public extension MemoryConfiguration {
    static func coreMLDefault(
        databaseURL: URL,
        models: CoreMLDefaultModels,
        queryExpander: (any QueryExpander)? = nil,
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

        return MemoryConfiguration(
            databaseURL: databaseURL,
            embeddingProvider: embeddingProvider,
            queryExpander: queryExpander,
            reranker: nil,
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
