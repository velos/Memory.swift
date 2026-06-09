#if MEMORY_COREML_EMBEDDING
import Foundation
import MemoryCoreMLAssets

enum CoreMLBundledResources {
    static func embeddingModelURL() throws -> URL {
        guard let url = MemoryCoreMLAssets.url(forResource: "embedding-v1", withExtension: "mlmodelc") else {
            throw MemoryError.embedding("No bundled embedding-v1.mlmodelc found. Enable the CoreMLEmbedding trait or provide an explicit model URL.")
        }
        return url
    }

    static func vocabURL() throws -> URL {
        guard let url = MemoryCoreMLAssets.url(forResource: "vocab", withExtension: "txt") else {
            throw MemoryError.embedding("No bundled vocab.txt found. Enable the CoreMLEmbedding trait or provide an explicit vocab URL.")
        }
        return url
    }

    static func tokenizerURL() throws -> URL {
        guard let url = MemoryCoreMLAssets.url(forResource: "tokenizer", withExtension: "json") else {
            throw MemoryError.embedding("No bundled tokenizer.json found. Enable the CoreMLEmbedding trait or provide an explicit tokenizer URL.")
        }
        return url
    }
}

public struct CoreMLDefaultModels: Sendable {
    public var embedding: URL
    public var reranker: URL?

    public init(embedding: URL, reranker: URL? = nil) {
        self.embedding = embedding
        self.reranker = reranker
    }

    public static func bundled() throws -> CoreMLDefaultModels {
        try CoreMLDefaultModels(embedding: CoreMLBundledResources.embeddingModelURL())
    }
}

public extension MemoryConfiguration {
    static func coreMLDefault(
        databaseURL: URL,
        models: CoreMLDefaultModels? = nil,
        structuredQueryExpander: (any StructuredQueryExpander)? = HeuristicStructuredQueryExpander(),
        contentTagger: (any ContentTagger)? = nil,
        memoryExtractor: (any MemoryExtractor)? = nil,
        recallPlanner: (any RecallPlanner)? = nil,
        queryAnalyzer: (any QueryAnalyzer)? = defaultCoreMLQueryAnalyzer(),
        tokenizer: any Tokenizer = defaultCoreMLTokenizer(),
        chunker: any Chunker = DefaultChunker(),
        supportedFileExtensions: Set<String> = defaultSupportedExtensions,
        semanticCandidateLimit: Int = 500,
        lexicalCandidateLimit: Int = 500,
        fusionK: Double = 60,
        positionAwareBlending: PositionAwareBlending = .default,
        ftsTokenizer: (any Tokenizer)? = defaultCoreMLFTSTokenizer()
    ) throws -> MemoryConfiguration {
        let resolvedModels = try models ?? CoreMLDefaultModels.bundled()
        let embeddingProvider = try CoreMLEmbeddingProvider(modelURL: resolvedModels.embedding)
        let rerankerProvider: (any Reranker)?
        if let rerankerURL = resolvedModels.reranker {
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

@usableFromInline
func defaultCoreMLQueryAnalyzer() -> (any QueryAnalyzer)? {
    #if MEMORY_NATURAL_LANGUAGE
    return NLQueryAnalyzer()
    #else
    return nil
    #endif
}

@usableFromInline
func defaultCoreMLTokenizer() -> any Tokenizer {
    #if MEMORY_NATURAL_LANGUAGE
    return NLWordTokenizer()
    #else
    return DefaultTokenizer()
    #endif
}

@usableFromInline
func defaultCoreMLFTSTokenizer() -> (any Tokenizer)? {
    #if MEMORY_NATURAL_LANGUAGE
    return NLLemmatizingTokenizer()
    #else
    return nil
    #endif
}

#endif
