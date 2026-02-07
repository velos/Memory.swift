import Foundation

public enum QMDError: Error, LocalizedError, Sendable {
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

public protocol EmbeddingProvider: Sendable {
    var identifier: String { get }
    func embed(text: String) async throws -> [Float]
    func embed(texts: [String]) async throws -> [[Float]]
}

public extension EmbeddingProvider {
    func embed(text: String) async throws -> [Float] {
        guard let first = try await embed(texts: [text]).first else {
            throw QMDError.embedding("Embedding provider \(identifier) returned no vectors")
        }
        return first
    }
}

public protocol Reranker: Sendable {
    var identifier: String { get }
    func rerank(query: SearchQuery, candidates: [SearchResult]) async throws -> [SearchResult]
}

public protocol Tokenizer: Sendable {
    func tokenize(_ text: String) -> [String]
}

public protocol Chunker: Sendable {
    func chunk(text: String, kind: DocumentKind, sourceURL: URL?) -> [Chunk]
}

public struct QMDConfiguration: Sendable {
    public var databaseURL: URL
    public var embeddingProvider: any EmbeddingProvider
    public var reranker: (any Reranker)?
    public var tokenizer: any Tokenizer
    public var chunker: any Chunker
    public var supportedFileExtensions: Set<String>
    public var semanticCandidateLimit: Int
    public var lexicalCandidateLimit: Int
    public var fusionK: Double

    public init(
        databaseURL: URL,
        embeddingProvider: any EmbeddingProvider,
        reranker: (any Reranker)? = nil,
        tokenizer: any Tokenizer = DefaultTokenizer(),
        chunker: any Chunker = DefaultChunker(),
        supportedFileExtensions: Set<String> = Self.defaultSupportedExtensions,
        semanticCandidateLimit: Int = 200,
        lexicalCandidateLimit: Int = 200,
        fusionK: Double = 60
    ) {
        self.databaseURL = databaseURL
        self.embeddingProvider = embeddingProvider
        self.reranker = reranker
        self.tokenizer = tokenizer
        self.chunker = chunker
        self.supportedFileExtensions = supportedFileExtensions
        self.semanticCandidateLimit = max(1, semanticCandidateLimit)
        self.lexicalCandidateLimit = max(1, lexicalCandidateLimit)
        self.fusionK = max(1, fusionK)
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
