import Foundation

public enum MemoryError: Error, LocalizedError, Sendable {
    case configuration(String)
    case ingestion(String)
    case embedding(String)
    case storage(String)
    case search(String)

    public var errorDescription: String? {
        switch self {
        case let .configuration(message):
            "Configuration error: \(message)"
        case let .ingestion(message):
            "Ingestion error: \(message)"
        case let .embedding(message):
            "Embedding error: \(message)"
        case let .storage(message):
            "Storage error: \(message)"
        case let .search(message):
            "Search error: \(message)"
        }
    }
}

public enum EmbeddingFormat: Sendable {
    case query
    case document(title: String?)
}

public protocol EmbeddingProvider: Sendable {
    var identifier: String { get }
    func embed(text: String) async throws -> [Float]
    func embed(texts: [String]) async throws -> [[Float]]
    func embed(text: String, format: EmbeddingFormat) async throws -> [Float]
    func embed(texts: [String], format: EmbeddingFormat) async throws -> [[Float]]
}

public extension EmbeddingProvider {
    func embed(text: String) async throws -> [Float] {
        guard let first = try await embed(texts: [text]).first else {
            throw MemoryError.embedding("Embedding provider \(identifier) returned no vectors")
        }
        return first
    }

    func embed(text: String, format: EmbeddingFormat) async throws -> [Float] {
        let formatted = Self.applyFormat(text: text, format: format)
        return try await embed(text: formatted)
    }

    func embed(texts: [String], format: EmbeddingFormat) async throws -> [[Float]] {
        let formatted = texts.map { Self.applyFormat(text: $0, format: format) }
        return try await embed(texts: formatted)
    }

    static func applyFormat(text: String, format: EmbeddingFormat) -> String {
        switch format {
        case .query:
            return "query: \(text)"
        case .document(let title):
            if let title, !title.isEmpty {
                return "title: \(title) | text: \(text)"
            }
            return "text: \(text)"
        }
    }
}

public enum ExpansionType: String, Sendable, Codable {
    case lexical
    case semantic
    case hypotheticalDocument
}

public struct ExpandedQuery: Sendable {
    public var text: String
    public var type: ExpansionType

    public init(text: String, type: ExpansionType) {
        self.text = text
        self.type = type
    }
}

public struct StructuredQueryExpansion: Sendable, Codable, Hashable {
    public var lexicalQueries: [String]
    public var semanticQueries: [String]
    public var hypotheticalDocuments: [String]
    public var facetHints: [FacetHint]
    public var entities: [MemoryEntity]
    public var topics: [String]

    public init(
        lexicalQueries: [String] = [],
        semanticQueries: [String] = [],
        hypotheticalDocuments: [String] = [],
        facetHints: [FacetHint] = [],
        entities: [MemoryEntity] = [],
        topics: [String] = []
    ) {
        self.lexicalQueries = lexicalQueries
        self.semanticQueries = semanticQueries
        self.hypotheticalDocuments = hypotheticalDocuments
        self.facetHints = facetHints
        self.entities = entities
        self.topics = topics
    }
}

public protocol StructuredQueryExpander: Sendable {
    var identifier: String { get }
    func expand(
        query: SearchQuery,
        analysis: QueryAnalysis,
        limit: Int
    ) async throws -> StructuredQueryExpansion
}

public protocol Reranker: Sendable {
    var identifier: String { get }
    func rerank(query: SearchQuery, candidates: [SearchResult]) async throws -> [RerankAssessment]
}

public protocol ContentTagger: Sendable {
    var identifier: String { get }
    func tag(text: String, kind: DocumentKind, sourceURL: URL?) async throws -> [ContentTag]
}

public protocol MemoryExtractor: Sendable {
    var identifier: String { get }
    func extract(messages: [ConversationMessage], limit: Int) async throws -> [MemoryCandidate]
}

public struct RecallPlan: Sendable {
    public var query: String

    public init(query: String) {
        self.query = query
    }
}

public protocol RecallPlanner: Sendable {
    var identifier: String { get }
    func plan(
        query: String,
        conversationContext: [ConversationMessage],
        features: RecallFeatures
    ) async throws -> RecallPlan?
}

public struct QueryAnalysis: Sendable {
    public var entities: [MemoryEntity]
    public var keyTerms: [String]
    public var facetHints: [FacetHint]
    public var topics: [String]
    public var isHowToQuery: Bool

    public init(
        entities: [MemoryEntity] = [],
        keyTerms: [String] = [],
        facetHints: [FacetHint] = [],
        topics: [String] = [],
        isHowToQuery: Bool = false
    ) {
        self.entities = entities
        self.keyTerms = keyTerms
        self.facetHints = facetHints
        self.topics = topics
        self.isHowToQuery = isHowToQuery
    }
}

public protocol QueryAnalyzer: Sendable {
    var identifier: String { get }
    func analyze(query: String) -> QueryAnalysis
}

public protocol Tokenizer: Sendable {
    func tokenize(_ text: String) -> [String]
}

public protocol Chunker: Sendable {
    func chunk(text: String, kind: DocumentKind, sourceURL: URL?) -> [Chunk]
}

public struct MemoryConfiguration: Sendable {
    public var databaseURL: URL
    public var embeddingProvider: any EmbeddingProvider
    public var structuredQueryExpander: (any StructuredQueryExpander)?
    public var reranker: (any Reranker)?
    public var contentTagger: (any ContentTagger)?
    public var memoryExtractor: (any MemoryExtractor)?
    public var recallPlanner: (any RecallPlanner)?
    public var queryAnalyzer: (any QueryAnalyzer)?
    public var tokenizer: any Tokenizer
    public var chunker: any Chunker
    public var supportedFileExtensions: Set<String>
    public var semanticCandidateLimit: Int
    public var lexicalCandidateLimit: Int
    public var fusionK: Double
    public var positionAwareBlending: PositionAwareBlending
    public var ftsTokenizer: (any Tokenizer)?

    public init(
        databaseURL: URL,
        embeddingProvider: any EmbeddingProvider,
        structuredQueryExpander: (any StructuredQueryExpander)? = nil,
        reranker: (any Reranker)? = nil,
        contentTagger: (any ContentTagger)? = nil,
        memoryExtractor: (any MemoryExtractor)? = nil,
        recallPlanner: (any RecallPlanner)? = nil,
        queryAnalyzer: (any QueryAnalyzer)? = nil,
        tokenizer: any Tokenizer = DefaultTokenizer(),
        chunker: any Chunker = DefaultChunker(),
        supportedFileExtensions: Set<String> = Self.defaultSupportedExtensions,
        semanticCandidateLimit: Int = 500,
        lexicalCandidateLimit: Int = 500,
        fusionK: Double = 60,
        positionAwareBlending: PositionAwareBlending = .default,
        ftsTokenizer: (any Tokenizer)? = nil
    ) {
        self.databaseURL = databaseURL
        self.embeddingProvider = embeddingProvider
        self.structuredQueryExpander = structuredQueryExpander
        self.reranker = reranker
        self.contentTagger = contentTagger
        self.memoryExtractor = memoryExtractor
        self.recallPlanner = recallPlanner
        self.queryAnalyzer = queryAnalyzer
        self.tokenizer = tokenizer
        self.chunker = chunker
        self.supportedFileExtensions = supportedFileExtensions
        self.semanticCandidateLimit = max(1, semanticCandidateLimit)
        self.lexicalCandidateLimit = max(1, lexicalCandidateLimit)
        self.fusionK = max(1, fusionK)
        self.positionAwareBlending = positionAwareBlending
        self.ftsTokenizer = ftsTokenizer
    }

    public static var defaultSupportedExtensions: Set<String> {
        [
            "md", "markdown", "txt", "text",
            "swift", "m", "mm", "h", "hpp", "c", "cpp",
            "js", "jsx", "ts", "tsx", "json",
            "py", "rb", "go", "rs", "java", "kt", "kts",
            "sh", "zsh", "bash", "yaml", "yml", "toml",
            "html", "css", "scss", "sql"
        ]
    }
}
