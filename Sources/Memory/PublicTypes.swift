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
    public var primaryBranchProtectionLimit: Int?
    public var referenceDate: Date?
    public var documentPathPrefix: String?
    public var contextID: ContextID?
    public var includeTagScoring: Bool

    public init(
        text: String,
        limit: Int = 20,
        semanticCandidateLimit: Int = 200,
        lexicalCandidateLimit: Int = 200,
        rerankLimit: Int = 50,
        expansionLimit: Int = 5,
        originalQueryWeight: Double = 2.0,
        expansionQueryWeight: Double = 1.0,
        primaryBranchProtectionLimit: Int? = nil,
        referenceDate: Date? = nil,
        documentPathPrefix: String? = nil,
        contextID: ContextID? = nil,
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
        self.primaryBranchProtectionLimit = primaryBranchProtectionLimit.map { max(0, $0) }
        self.referenceDate = referenceDate
        let trimmedDocumentPathPrefix = documentPathPrefix?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.documentPathPrefix = trimmedDocumentPathPrefix?.isEmpty == false ? trimmedDocumentPathPrefix : nil
        self.contextID = contextID
        self.includeTagScoring = includeTagScoring
    }
}

public struct SearchScoreBreakdown: Sendable, Codable, Hashable {
    public var semantic: Double
    public var lexical: Double
    public var recency: Double
    public var tag: Double
    public var schema: Double
    public var temporal: Double
    public var status: Double
    public var fused: Double
    public var rerank: Double
    public var blended: Double

    public init(
        semantic: Double,
        lexical: Double,
        recency: Double,
        tag: Double = 0,
        schema: Double = 0,
        temporal: Double = 0,
        status: Double = 0,
        fused: Double,
        rerank: Double = 0,
        blended: Double? = nil
    ) {
        self.semantic = semantic
        self.lexical = lexical
        self.recency = recency
        self.tag = tag
        self.schema = schema
        self.temporal = temporal
        self.status = status
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

public enum FacetTag: String, CaseIterable, Codable, Sendable {
    case preference = "preference"
    case person = "person"
    case relationship = "relationship"
    case project = "project"
    case goal = "goal"
    case task = "task"
    case decisionTopic = "decision_topic"
    case tool = "tool"
    case location = "location"
    case timeSensitive = "time_sensitive"
    case constraint = "constraint"
    case habit = "habit"
    case factAboutUser = "fact_about_user"
    case factAboutWorld = "fact_about_world"
    case lesson = "lesson"
    case emotion = "emotion"
    case identitySignal = "identity_signal"

    public static func parse(_ raw: String) -> FacetTag? {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return FacetTag(rawValue: normalized)
    }
}

public struct FacetHint: Codable, Hashable, Sendable {
    public var tag: FacetTag
    public var confidence: Double
    public var isExplicit: Bool

    public init(
        tag: FacetTag,
        confidence: Double,
        isExplicit: Bool
    ) {
        self.tag = tag
        self.confidence = min(1, max(0, confidence))
        self.isExplicit = isExplicit
    }
}

public enum EntityLabel: String, CaseIterable, Codable, Sendable {
    case person
    case organization
    case product
    case project
    case tool
    case location
    case date
    case other

    public static func parse(_ raw: String) -> EntityLabel? {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return EntityLabel(rawValue: normalized)
    }
}

public struct MemoryEntity: Codable, Hashable, Sendable {
    public var label: EntityLabel
    public var value: String
    public var normalizedValue: String
    public var confidence: Double?

    public init(
        label: EntityLabel,
        value: String,
        normalizedValue: String,
        confidence: Double? = nil
    ) {
        self.label = label
        self.value = value
        self.normalizedValue = normalizedValue
        self.confidence = confidence.map { min(1, max(0, $0)) }
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
    public var facetTags: Set<FacetTag>
    public var entities: [MemoryEntity]
    public var topics: [String]
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
        facetTags: Set<FacetTag> = [],
        entities: [MemoryEntity] = [],
        topics: [String] = [],
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
        self.facetTags = facetTags
        self.entities = entities
        self.topics = topics
        self.canonicalKey = canonicalKey
        self.metadata = metadata
    }
}

public enum MemoryWriteAction: String, Sendable, Codable, Hashable {
    case create
    case dedupe
    case replaceActive = "replace_active"
    case mergeStatus = "merge_status"
    case supersede
    case appendEpisode = "append_episode"
    case noWrite = "no_write"
}

public struct MemoryRejectedSpan: Sendable, Codable, Hashable {
    public var text: String
    public var reason: String
    public var confidence: Double?

    public init(text: String, reason: String, confidence: Double? = nil) {
        self.text = text
        self.reason = reason
        self.confidence = confidence.map { min(1, max(0, $0)) }
    }
}

public struct MemoryExtractionResult: Sendable, Codable, Hashable {
    public var candidates: [MemoryCandidate]
    public var rejectedSpans: [MemoryRejectedSpan]
    public var proposedActions: [MemoryWriteAction]
    public var rationale: [String]

    public init(
        candidates: [MemoryCandidate] = [],
        rejectedSpans: [MemoryRejectedSpan] = [],
        proposedActions: [MemoryWriteAction] = [],
        rationale: [String] = []
    ) {
        self.candidates = candidates
        self.rejectedSpans = rejectedSpans
        self.proposedActions = proposedActions
        self.rationale = rationale
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
    public var tags: [ContentTag]
    public var facetTags: Set<FacetTag>
    public var entities: [MemoryEntity]
    public var topics: [String]
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
        tags: [ContentTag],
        facetTags: Set<FacetTag> = [],
        entities: [MemoryEntity] = [],
        topics: [String] = [],
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
        self.tags = tags
        self.facetTags = facetTags
        self.entities = entities
        self.topics = topics
        self.score = score
    }
}

public struct MemoryIngestResult: Sendable, Codable, Hashable {
    public var requestedCount: Int
    public var storedCount: Int
    public var discardedCount: Int
    public var records: [MemoryRecord]
    public var actions: [MemoryWriteAction]

    public init(
        requestedCount: Int,
        storedCount: Int,
        discardedCount: Int,
        records: [MemoryRecord],
        actions: [MemoryWriteAction] = []
    ) {
        self.requestedCount = max(0, requestedCount)
        self.storedCount = max(0, storedCount)
        self.discardedCount = max(0, discardedCount)
        self.records = records
        self.actions = actions
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

    public static let hybridDefault: RecallFeatures = [.semantic, .lexical, .tags, .expansion, .planner]
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
    case providerFailure(path: String, stage: IndexingStage, provider: String, message: String)
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
    case providerFailure(stage: SearchStage, provider: String, message: String)
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
