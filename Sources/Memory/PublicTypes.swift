import Foundation

public struct ContextID: Hashable, Sendable, Codable, RawRepresentable, CustomStringConvertible {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue.lowercased()
    }

    public init() {
        self.rawValue = UUID().uuidString.lowercased()
    }

    public var description: String { rawValue }
}

public struct IndexingRequest: Sendable {
    public var roots: [URL]
    public var includeHiddenFiles: Bool
    public var followSymlinks: Bool

    public init(
        roots: [URL],
        includeHiddenFiles: Bool = false,
        followSymlinks: Bool = false
    ) {
        self.roots = roots
        self.includeHiddenFiles = includeHiddenFiles
        self.followSymlinks = followSymlinks
    }
}

public struct SearchQuery: Sendable {
    public var text: String
    public var limit: Int
    public var semanticCandidateLimit: Int
    public var lexicalCandidateLimit: Int
    public var rerankLimit: Int
    public var expansionLimit: Int
    public var originalQueryWeight: Double
    public var expansionQueryWeight: Double
    public var contextID: ContextID?

    public init(
        text: String,
        limit: Int = 20,
        semanticCandidateLimit: Int = 200,
        lexicalCandidateLimit: Int = 200,
        rerankLimit: Int = 50,
        expansionLimit: Int = 2,
        originalQueryWeight: Double = 2.0,
        expansionQueryWeight: Double = 1.0,
        contextID: ContextID? = nil
    ) {
        self.text = text
        self.limit = max(1, limit)
        self.semanticCandidateLimit = max(1, semanticCandidateLimit)
        self.lexicalCandidateLimit = max(1, lexicalCandidateLimit)
        self.rerankLimit = max(0, rerankLimit)
        self.expansionLimit = max(0, expansionLimit)
        self.originalQueryWeight = max(0.1, originalQueryWeight)
        self.expansionQueryWeight = max(0.1, expansionQueryWeight)
        self.contextID = contextID
    }
}

public struct SearchScoreBreakdown: Sendable {
    public var semantic: Double
    public var lexical: Double
    public var recency: Double
    public var fused: Double
    public var rerank: Double
    public var blended: Double

    public init(
        semantic: Double,
        lexical: Double,
        recency: Double,
        fused: Double,
        rerank: Double = 0,
        blended: Double? = nil
    ) {
        self.semantic = semantic
        self.lexical = lexical
        self.recency = recency
        self.fused = fused
        self.rerank = rerank
        self.blended = blended ?? fused
    }
}

public struct SearchResult: Sendable {
    public var chunkID: Int64
    public var documentPath: String
    public var title: String?
    public var content: String
    public var snippet: String
    public var modifiedAt: Date
    public var score: SearchScoreBreakdown

    public init(
        chunkID: Int64,
        documentPath: String,
        title: String?,
        content: String,
        snippet: String,
        modifiedAt: Date,
        score: SearchScoreBreakdown
    ) {
        self.chunkID = chunkID
        self.documentPath = documentPath
        self.title = title
        self.content = content
        self.snippet = snippet
        self.modifiedAt = modifiedAt
        self.score = score
    }
}

public enum DocumentKind: String, Sendable {
    case markdown
    case code
    case plainText
}

public struct Chunk: Sendable {
    public var ordinal: Int
    public var content: String
    public var tokenCount: Int

    public init(ordinal: Int, content: String, tokenCount: Int) {
        self.ordinal = ordinal
        self.content = content
        self.tokenCount = tokenCount
    }
}

public enum IndexingEvent: Sendable {
    case started(totalDocuments: Int)
    case readingDocument(path: String, index: Int, total: Int)
    case chunked(path: String, chunks: Int)
    case embedded(path: String, chunks: Int)
    case stored(path: String)
    case completed(processedDocuments: Int, totalChunks: Int)
}

public enum SearchEvent: Sendable {
    case started(query: String)
    case expandedQueries(count: Int)
    case embeddedQuery(dimension: Int)
    case semanticCandidates(count: Int)
    case lexicalCandidates(count: Int)
    case fusedCandidates(count: Int)
    case reranked(count: Int)
    case completed(count: Int)
}

public struct RerankAssessment: Sendable {
    public var chunkID: Int64
    public var relevance: Double
    public var rationale: String?

    public init(chunkID: Int64, relevance: Double, rationale: String? = nil) {
        self.chunkID = chunkID
        self.relevance = relevance
        self.rationale = rationale
    }
}

public struct PositionAwareBlending: Sendable {
    public var topRankFusedWeight: Double
    public var midRankFusedWeight: Double
    public var tailRankFusedWeight: Double

    public init(
        topRankFusedWeight: Double = 0.75,
        midRankFusedWeight: Double = 0.60,
        tailRankFusedWeight: Double = 0.40
    ) {
        self.topRankFusedWeight = Self.clampWeight(topRankFusedWeight)
        self.midRankFusedWeight = Self.clampWeight(midRankFusedWeight)
        self.tailRankFusedWeight = Self.clampWeight(tailRankFusedWeight)
    }

    public func blend(fused: Double, rerank: Double, position: Int) -> Double {
        let fusedWeight: Double
        switch position {
        case ...3:
            fusedWeight = topRankFusedWeight
        case 4...10:
            fusedWeight = midRankFusedWeight
        default:
            fusedWeight = tailRankFusedWeight
        }

        let rerankWeight = max(0, 1 - fusedWeight)
        return (fusedWeight * fused) + (rerankWeight * rerank)
    }

    public static let `default` = PositionAwareBlending()

    private static func clampWeight(_ value: Double) -> Double {
        min(1, max(0, value))
    }
}

public typealias IndexingEventHandler = @Sendable (IndexingEvent) -> Void
public typealias SearchEventHandler = @Sendable (SearchEvent) -> Void
