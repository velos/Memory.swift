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
    public var documentMemoryTypes: Set<DocumentMemoryType>?
    public var includeTagScoring: Bool

    public init(
        text: String,
        limit: Int = 20,
        semanticCandidateLimit: Int = 200,
        lexicalCandidateLimit: Int = 200,
        rerankLimit: Int = 50,
        expansionLimit: Int = 2,
        originalQueryWeight: Double = 2.0,
        expansionQueryWeight: Double = 1.0,
        contextID: ContextID? = nil,
        documentMemoryTypes: Set<DocumentMemoryType>? = nil,
        includeTagScoring: Bool = true
    ) {
        self.text = text
        self.limit = max(1, limit)
        self.semanticCandidateLimit = max(0, semanticCandidateLimit)
        self.lexicalCandidateLimit = max(0, lexicalCandidateLimit)
        self.rerankLimit = max(0, rerankLimit)
        self.expansionLimit = max(0, expansionLimit)
        self.originalQueryWeight = max(0.1, originalQueryWeight)
        self.expansionQueryWeight = max(0.1, expansionQueryWeight)
        self.contextID = contextID
        self.documentMemoryTypes = documentMemoryTypes
        self.includeTagScoring = includeTagScoring
    }
}

public struct SearchScoreBreakdown: Sendable, Codable, Hashable {
    public var semantic: Double
    public var lexical: Double
    public var recency: Double
    public var tag: Double
    public var fused: Double
    public var rerank: Double
    public var blended: Double

    public init(
        semantic: Double,
        lexical: Double,
        recency: Double,
        tag: Double = 0,
        fused: Double,
        rerank: Double = 0,
        blended: Double? = nil
    ) {
        self.semantic = semantic
        self.lexical = lexical
        self.recency = recency
        self.tag = tag
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
    public var documentMemoryType: DocumentMemoryType
    public var documentMemoryTypeSource: MemoryTypeSource
    public var documentMemoryTypeConfidence: Double?
    public var memoryID: String?
    public var memoryKind: MemoryKind?
    public var memoryStatus: MemoryStatus?
    public var score: SearchScoreBreakdown

    public init(
        chunkID: Int64,
        documentPath: String,
        title: String?,
        content: String,
        snippet: String,
        modifiedAt: Date,
        documentMemoryType: DocumentMemoryType = .factual,
        documentMemoryTypeSource: MemoryTypeSource = .fallback,
        documentMemoryTypeConfidence: Double? = nil,
        memoryID: String? = nil,
        memoryKind: MemoryKind? = nil,
        memoryStatus: MemoryStatus? = nil,
        score: SearchScoreBreakdown
    ) {
        self.chunkID = chunkID
        self.documentPath = documentPath
        self.title = title
        self.content = content
        self.snippet = snippet
        self.modifiedAt = modifiedAt
        self.documentMemoryType = documentMemoryType
        self.documentMemoryTypeSource = documentMemoryTypeSource
        self.documentMemoryTypeConfidence = documentMemoryTypeConfidence
        self.memoryID = memoryID
        self.memoryKind = memoryKind
        self.memoryStatus = memoryStatus
        self.score = score
    }
}

public enum MemoryKind: String, CaseIterable, Codable, Sendable {
    case profile
    case fact
    case decision
    case commitment
    case episode
    case procedure
    case handoff

    public static func parse(_ raw: String) -> MemoryKind? {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return MemoryKind(rawValue: normalized)
    }
}

public enum MemoryStatus: String, CaseIterable, Codable, Sendable {
    case active
    case superseded
    case resolved
    case archived

    public static func parse(_ raw: String) -> MemoryStatus? {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return MemoryStatus(rawValue: normalized)
    }
}

public enum ConversationRole: String, Codable, Sendable {
    case system
    case user
    case assistant
}

public struct ConversationMessage: Sendable, Codable, Hashable {
    public var role: ConversationRole
    public var content: String
    public var createdAt: Date?

    public init(role: ConversationRole, content: String, createdAt: Date? = nil) {
        self.role = role
        self.content = content
        self.createdAt = createdAt
    }
}

public struct MemoryCandidate: Sendable, Codable, Hashable {
    public var text: String
    public var kind: MemoryKind
    public var status: MemoryStatus
    public var importance: Double
    public var confidence: Double?
    public var createdAt: Date?
    public var eventAt: Date?
    public var source: String
    public var tags: [String]
    public var canonicalKey: String?
    public var metadata: [String: String]

    public init(
        text: String,
        kind: MemoryKind,
        status: MemoryStatus = .active,
        importance: Double = 0.5,
        confidence: Double? = nil,
        createdAt: Date? = nil,
        eventAt: Date? = nil,
        source: String = "extract",
        tags: [String] = [],
        canonicalKey: String? = nil,
        metadata: [String: String] = [:]
    ) {
        self.text = text
        self.kind = kind
        self.status = status
        self.importance = min(1, max(0, importance))
        self.confidence = confidence.map { min(1, max(0, $0)) }
        self.createdAt = createdAt
        self.eventAt = eventAt
        self.source = source
        self.tags = tags
        self.canonicalKey = canonicalKey
        self.metadata = metadata
    }
}

public struct MemoryRecord: Sendable, Codable, Hashable {
    public var id: String
    public var chunkID: Int64
    public var documentPath: String
    public var title: String?
    public var text: String
    public var kind: MemoryKind
    public var status: MemoryStatus
    public var canonicalKey: String?
    public var importance: Double
    public var confidence: Double?
    public var accessCount: Int
    public var createdAt: Date
    public var eventAt: Date?
    public var modifiedAt: Date
    public var lastAccessedAt: Date?
    public var documentMemoryType: DocumentMemoryType
    public var documentMemoryTypeSource: MemoryTypeSource
    public var documentMemoryTypeConfidence: Double?
    public var tags: [ContentTag]
    public var score: SearchScoreBreakdown?

    public init(
        id: String,
        chunkID: Int64,
        documentPath: String,
        title: String?,
        text: String,
        kind: MemoryKind,
        status: MemoryStatus,
        canonicalKey: String?,
        importance: Double,
        confidence: Double?,
        accessCount: Int,
        createdAt: Date,
        eventAt: Date?,
        modifiedAt: Date,
        lastAccessedAt: Date?,
        documentMemoryType: DocumentMemoryType,
        documentMemoryTypeSource: MemoryTypeSource,
        documentMemoryTypeConfidence: Double?,
        tags: [ContentTag],
        score: SearchScoreBreakdown? = nil
    ) {
        self.id = id
        self.chunkID = chunkID
        self.documentPath = documentPath
        self.title = title
        self.text = text
        self.kind = kind
        self.status = status
        self.canonicalKey = canonicalKey
        self.importance = min(1, max(0, importance))
        self.confidence = confidence.map { min(1, max(0, $0)) }
        self.accessCount = max(0, accessCount)
        self.createdAt = createdAt
        self.eventAt = eventAt
        self.modifiedAt = modifiedAt
        self.lastAccessedAt = lastAccessedAt
        self.documentMemoryType = documentMemoryType
        self.documentMemoryTypeSource = documentMemoryTypeSource
        self.documentMemoryTypeConfidence = documentMemoryTypeConfidence
        self.tags = tags
        self.score = score
    }
}

public struct MemoryIngestResult: Sendable, Codable, Hashable {
    public var requestedCount: Int
    public var storedCount: Int
    public var discardedCount: Int
    public var records: [MemoryRecord]

    public init(
        requestedCount: Int,
        storedCount: Int,
        discardedCount: Int,
        records: [MemoryRecord]
    ) {
        self.requestedCount = max(0, requestedCount)
        self.storedCount = max(0, storedCount)
        self.discardedCount = max(0, discardedCount)
        self.records = records
    }
}

public enum RecallMode: Sendable {
    case hybrid(query: String)
    case recent
    case important
    case kind(MemoryKind)
}

public enum RecallSort: String, Codable, Sendable {
    case recent
    case importance
    case mostAccessed = "most_accessed"
}

public struct RecallFeatures: OptionSet, Sendable, Hashable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public static let semantic = RecallFeatures(rawValue: 1 << 0)
    public static let lexical = RecallFeatures(rawValue: 1 << 1)
    public static let tags = RecallFeatures(rawValue: 1 << 2)
    public static let expansion = RecallFeatures(rawValue: 1 << 3)
    public static let rerank = RecallFeatures(rawValue: 1 << 4)
    public static let planner = RecallFeatures(rawValue: 1 << 5)

    public static let hybridDefault: RecallFeatures = [.semantic, .lexical, .tags, .expansion]
}

public struct MemoryRecallResponse: Sendable, Codable, Hashable {
    public var records: [MemoryRecord]

    public init(records: [MemoryRecord]) {
        self.records = records
    }
}

public enum MemoryDocumentSource: String, Sendable, Codable, Hashable {
    case fileSystem = "file_system"
    case indexed
}

public struct MemoryLineRange: Sendable, Codable, Hashable {
    public let start: Int
    public let end: Int

    public init(start: Int, end: Int) {
        self.start = max(1, min(start, end))
        self.end = max(1, max(start, end))
    }

    public var closedRange: ClosedRange<Int> {
        start...end
    }
}

public struct MemorySearchReference: Sendable, Codable, Hashable {
    public let chunkID: Int64
    public let documentPath: String
    public let title: String?
    public let snippet: String
    public let lineRange: MemoryLineRange?
    public let source: MemoryDocumentSource
    public let documentMemoryType: DocumentMemoryType
    public let documentMemoryTypeSource: MemoryTypeSource
    public let documentMemoryTypeConfidence: Double?
    public let memoryID: String?
    public let memoryKind: MemoryKind?
    public let memoryStatus: MemoryStatus?
    public let score: SearchScoreBreakdown
}

public struct MemoryGetResponse: Sendable, Codable, Hashable {
    public let documentPath: String
    public let source: MemoryDocumentSource
    public let totalLineCount: Int
    public let lineRange: MemoryLineRange
    public let content: String
}

public enum DocumentKind: String, Sendable {
    case markdown
    case code
    case plainText
}

public struct ContentTag: Sendable, Codable, Hashable {
    public var name: String
    public var confidence: Double

    public init(name: String, confidence: Double) {
        self.name = name
        self.confidence = confidence
    }
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

public enum IndexingStage: String, Sendable, Codable {
    case typing
    case chunking
    case tagging
    case embedding
    case indexWrite = "index_write"
    case total
}

public enum IndexingEvent: Sendable {
    case started(totalDocuments: Int)
    case readingDocument(path: String, index: Int, total: Int)
    case chunked(path: String, chunks: Int)
    case embedded(path: String, chunks: Int)
    case stageTiming(path: String, stage: IndexingStage, durationMs: Double)
    case stored(path: String)
    case completed(processedDocuments: Int, totalChunks: Int)
}

public enum SearchStage: String, Sendable, Codable {
    case analysis
    case expansion
    case queryEmbedding = "query_embedding"
    case semanticSearch = "semantic_search"
    case lexicalSearch = "lexical_search"
    case fusion
    case rerank
    case total
}

public enum SearchEvent: Sendable {
    case started(query: String)
    case expandedQueries(count: Int)
    case embeddedQuery(dimension: Int)
    case semanticCandidates(count: Int)
    case lexicalCandidates(count: Int)
    case fusedCandidates(count: Int)
    case reranked(count: Int)
    case stageTiming(stage: SearchStage, durationMs: Double)
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
