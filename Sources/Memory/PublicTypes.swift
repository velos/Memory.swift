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
    public var additionalLexicalQueries: [String]
    public var additionalLexicalQueryWeight: Double
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
        additionalLexicalQueries: [String] = [],
        additionalLexicalQueryWeight: Double = 0.35,
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
        self.additionalLexicalQueries = additionalLexicalQueries
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        self.additionalLexicalQueryWeight = max(0, additionalLexicalQueryWeight)
        self.referenceDate = referenceDate
        let trimmedDocumentPathPrefix = documentPathPrefix?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.documentPathPrefix = trimmedDocumentPathPrefix?.isEmpty == false ? trimmedDocumentPathPrefix : nil
        self.contextID = contextID
        self.includeTagScoring = includeTagScoring
    }
}

public enum GroundedQueryExpansionTermMode: String, Sendable, Codable, Hashable {
    case all
    case singleToken = "single_token"
    case phraseEntity = "phrase_entity"
}

public struct GroundedQueryExpansionConfiguration: Sendable, Codable, Hashable {
    public var isEnabled: Bool
    public var maxFeedbackResults: Int
    public var maxTerms: Int
    public var termsPerQuery: Int
    public var maxQueries: Int
    public var lexicalQueryWeight: Double
    public var termMode: GroundedQueryExpansionTermMode

    public init(
        isEnabled: Bool = true,
        maxFeedbackResults: Int = 8,
        maxTerms: Int = 8,
        termsPerQuery: Int = 4,
        maxQueries: Int = 1,
        lexicalQueryWeight: Double = 0.20,
        termMode: GroundedQueryExpansionTermMode = .phraseEntity
    ) {
        self.isEnabled = isEnabled
        self.maxFeedbackResults = max(1, min(maxFeedbackResults, 20))
        self.maxTerms = max(1, min(maxTerms, 12))
        self.termsPerQuery = max(1, min(termsPerQuery, 6))
        self.maxQueries = max(1, min(maxQueries, 3))
        self.lexicalQueryWeight = max(0, min(lexicalQueryWeight, 1.0))
        self.termMode = termMode
    }

    public static let disabled = GroundedQueryExpansionConfiguration(isEnabled: false)
    public static let conservativeDefault = GroundedQueryExpansionConfiguration()
}

public struct SearchScoreBreakdown: Sendable, Codable, Hashable {
    public var semantic: Double
    public var lexical: Double
    public var recency: Double
    public var tag: Double
    public var schema: Double
    public var temporal: Double
    public var status: Double
    public var type: Double
    public var fused: Double
    public var rerank: Double
    public var blended: Double

    private enum CodingKeys: String, CodingKey {
        case semantic
        case lexical
        case recency
        case tag
        case schema
        case temporal
        case status
        case type
        case fused
        case rerank
        case blended
    }

    public init(
        semantic: Double,
        lexical: Double,
        recency: Double,
        tag: Double = 0,
        schema: Double = 0,
        temporal: Double = 0,
        status: Double = 0,
        type: Double = 0,
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
        self.type = type
        self.fused = fused
        self.rerank = rerank
        self.blended = blended ?? fused
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.semantic = try container.decode(Double.self, forKey: .semantic)
        self.lexical = try container.decode(Double.self, forKey: .lexical)
        self.recency = try container.decode(Double.self, forKey: .recency)
        self.tag = try container.decodeIfPresent(Double.self, forKey: .tag) ?? 0
        self.schema = try container.decodeIfPresent(Double.self, forKey: .schema) ?? 0
        self.temporal = try container.decodeIfPresent(Double.self, forKey: .temporal) ?? 0
        self.status = try container.decodeIfPresent(Double.self, forKey: .status) ?? 0
        self.type = try container.decodeIfPresent(Double.self, forKey: .type) ?? 0
        self.fused = try container.decode(Double.self, forKey: .fused)
        self.rerank = try container.decodeIfPresent(Double.self, forKey: .rerank) ?? 0
        self.blended = try container.decodeIfPresent(Double.self, forKey: .blended) ?? fused
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(semantic, forKey: .semantic)
        try container.encode(lexical, forKey: .lexical)
        try container.encode(recency, forKey: .recency)
        try container.encode(tag, forKey: .tag)
        try container.encode(schema, forKey: .schema)
        try container.encode(temporal, forKey: .temporal)
        try container.encode(status, forKey: .status)
        try container.encode(type, forKey: .type)
        try container.encode(fused, forKey: .fused)
        try container.encode(rerank, forKey: .rerank)
        try container.encode(blended, forKey: .blended)
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
    public var memoryType: String?
    public var memoryTypeConfidence: Double?
    public var score: SearchScoreBreakdown
    public var contextHints: [MemoryContextHint]

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
        memoryType: String? = nil,
        memoryTypeConfidence: Double? = nil,
        score: SearchScoreBreakdown,
        contextHints: [MemoryContextHint] = []
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
        self.memoryType = memoryType
        self.memoryTypeConfidence = memoryTypeConfidence
        self.score = score
        self.contextHints = contextHints
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

public enum MemorySubject: String, CaseIterable, Codable, Sendable, Hashable {
    case user
    case assistant
    case workspace
    case world
    case thirdParty = "third_party"
    case unknown
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

public struct MemoryEvidence: Sendable, Codable, Hashable {
    public var role: ConversationRole
    public var excerpt: String
    public var messageIndex: Int?
    public var timestamp: Date?
    public var sourceID: String?

    public init(
        role: ConversationRole,
        excerpt: String,
        messageIndex: Int? = nil,
        timestamp: Date? = nil,
        sourceID: String? = nil
    ) {
        self.role = role
        self.excerpt = excerpt.trimmingCharacters(in: .whitespacesAndNewlines)
        self.messageIndex = messageIndex
        self.timestamp = timestamp
        self.sourceID = sourceID?.trimmingCharacters(in: .whitespacesAndNewlines)
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
    public var subject: MemorySubject?
    public var evidence: [MemoryEvidence]

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
        metadata: [String: String] = [:],
        subject: MemorySubject? = nil,
        evidence: [MemoryEvidence] = []
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
        self.subject = subject
        self.evidence = evidence
    }
}

public enum MemoryCaptureMode: String, Sendable, Codable, Hashable {
    case preview
    case ingest
}

public enum MemoryCaptureFocus: String, Sendable, Codable, Hashable {
    case user
    case assistant
    case workspace
    case all
}

public struct MemoryCapturePolicy: Sendable, Codable, Hashable {
    public var focus: MemoryCaptureFocus
    public var minimumConfidence: Double
    public var allowAssistantAuthoredWorkflowFacts: Bool

    public init(
        focus: MemoryCaptureFocus = .user,
        minimumConfidence: Double = 0.55,
        allowAssistantAuthoredWorkflowFacts: Bool = true
    ) {
        self.focus = focus
        self.minimumConfidence = min(1, max(0, minimumConfidence))
        self.allowAssistantAuthoredWorkflowFacts = allowAssistantAuthoredWorkflowFacts
    }

    public static let agentDefault = MemoryCapturePolicy()
}

public struct MemoryCompactionObservation: Sendable, Codable, Hashable {
    public var summary: String
    public var messages: [ConversationMessage]
    public var sessionID: String?
    public var createdAt: Date

    public init(
        summary: String,
        messages: [ConversationMessage] = [],
        sessionID: String? = nil,
        createdAt: Date = Date()
    ) {
        self.summary = summary
        self.messages = messages
        self.sessionID = sessionID
        self.createdAt = createdAt
    }
}

public struct MemoryCaptureRequest: Sendable, Codable, Hashable {
    public var messages: [ConversationMessage]
    public var mode: MemoryCaptureMode
    public var policy: MemoryCapturePolicy
    public var limit: Int
    public var sourceID: String?
    public var compactionObservation: MemoryCompactionObservation?

    public init(
        messages: [ConversationMessage],
        mode: MemoryCaptureMode = .preview,
        policy: MemoryCapturePolicy = .agentDefault,
        limit: Int = 50,
        sourceID: String? = nil,
        compactionObservation: MemoryCompactionObservation? = nil
    ) {
        self.messages = messages
        self.mode = mode
        self.policy = policy
        self.limit = max(1, limit)
        self.sourceID = sourceID
        self.compactionObservation = compactionObservation
    }
}

public struct MemoryCaptureResult: Sendable, Codable, Hashable {
    public var extraction: MemoryExtractionResult
    public var ingestResult: MemoryIngestResult?

    public init(extraction: MemoryExtractionResult, ingestResult: MemoryIngestResult? = nil) {
        self.extraction = extraction
        self.ingestResult = ingestResult
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
    public var source: String
    public var accessCount: Int
    public var createdAt: Date
    public var eventAt: Date?
    public var modifiedAt: Date
    public var lastAccessedAt: Date?
    public var tags: [ContentTag]
    public var facetTags: Set<FacetTag>
    public var entities: [MemoryEntity]
    public var topics: [String]
    public var metadata: [String: String]
    public var score: SearchScoreBreakdown?
    public var subject: MemorySubject?
    public var evidence: [MemoryEvidence]

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
        source: String = "",
        accessCount: Int,
        createdAt: Date,
        eventAt: Date?,
        modifiedAt: Date,
        lastAccessedAt: Date?,
        tags: [ContentTag],
        facetTags: Set<FacetTag> = [],
        entities: [MemoryEntity] = [],
        topics: [String] = [],
        metadata: [String: String] = [:],
        score: SearchScoreBreakdown? = nil,
        subject: MemorySubject? = nil,
        evidence: [MemoryEvidence] = []
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
        self.source = source
        self.accessCount = max(0, accessCount)
        self.createdAt = createdAt
        self.eventAt = eventAt
        self.modifiedAt = modifiedAt
        self.lastAccessedAt = lastAccessedAt
        self.tags = tags
        self.facetTags = facetTags
        self.entities = entities
        self.topics = topics
        self.metadata = metadata
        self.score = score
        self.subject = subject
        self.evidence = evidence
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

public enum MemoryDebugSort: String, CaseIterable, Codable, Sendable, Hashable {
    case createdAtDescending = "created_at_descending"
    case updatedAtDescending = "updated_at_descending"
    case importanceDescending = "importance_descending"
    case mostAccessed = "most_accessed"
}

public struct MemoryDebugQuery: Sendable, Codable, Hashable {
    public var searchText: String
    public var limit: Int
    public var offset: Int
    public var sort: MemoryDebugSort
    public var kinds: Set<MemoryKind>?
    public var statuses: Set<MemoryStatus>?

    public init(
        searchText: String = "",
        limit: Int = 25,
        offset: Int = 0,
        sort: MemoryDebugSort = .createdAtDescending,
        kinds: Set<MemoryKind>? = nil,
        statuses: Set<MemoryStatus>? = Set([.active, .resolved, .superseded])
    ) {
        self.searchText = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        self.limit = max(1, limit)
        self.offset = max(0, offset)
        self.sort = sort
        self.kinds = kinds
        self.statuses = statuses
    }
}

public struct MemoryDebugPage: Sendable, Codable, Hashable {
    public var records: [MemoryRecord]
    public var totalCount: Int
    public var limit: Int
    public var offset: Int
    public var hasMore: Bool

    public init(
        records: [MemoryRecord],
        totalCount: Int,
        limit: Int,
        offset: Int
    ) {
        self.records = records
        self.totalCount = max(0, totalCount)
        self.limit = max(1, limit)
        self.offset = max(0, offset)
        self.hasMore = self.offset + records.count < self.totalCount
    }
}

public struct RecallFeatures: OptionSet, Sendable, Hashable, Codable {
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

public enum MemoryContextQueryMode: String, Sendable, Codable, Hashable {
    case message
    case recent
    case full
}

public struct MemoryContextBudget: Sendable, Codable, Hashable {
    public var maxReferences: Int
    public var maxTokens: Int

    public init(maxReferences: Int = 8, maxTokens: Int = 1_024) {
        self.maxReferences = max(1, maxReferences)
        self.maxTokens = max(64, maxTokens)
    }
}

public struct MemoryContextHint: Sendable, Codable, Hashable, Identifiable {
    public var id: String
    public var pathPrefix: String
    public var context: String
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: String = UUID().uuidString.lowercased(),
        pathPrefix: String,
        context: String,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.pathPrefix = pathPrefix.trimmingCharacters(in: .whitespacesAndNewlines)
        self.context = context.trimmingCharacters(in: .whitespacesAndNewlines)
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

public struct MemoryContextRequest: Sendable, Codable, Hashable {
    public var messages: [ConversationMessage]
    public var mode: MemoryContextQueryMode
    public var budget: MemoryContextBudget
    public var features: RecallFeatures
    public var sourceID: String?

    public init(
        messages: [ConversationMessage],
        mode: MemoryContextQueryMode = .message,
        budget: MemoryContextBudget = MemoryContextBudget(),
        features: RecallFeatures = .hybridDefault,
        sourceID: String? = nil
    ) {
        self.messages = messages
        self.mode = mode
        self.budget = budget
        self.features = features
        self.sourceID = sourceID
    }
}

public struct MemoryContextResponse: Sendable, Codable, Hashable {
    public var contextBlock: String
    public var references: [MemorySearchReference]
    public var hints: [MemoryContextHint]

    public init(
        contextBlock: String,
        references: [MemorySearchReference],
        hints: [MemoryContextHint] = []
    ) {
        self.contextBlock = contextBlock
        self.references = references
        self.hints = hints
    }
}

public enum MemorySignalKind: String, Sendable, Codable, Hashable {
    case recall
    case capture
    case compaction
    case explicit
    case maintenance
}

public struct MemorySignal: Sendable, Codable, Hashable, Identifiable {
    public var id: String
    public var kind: MemorySignalKind
    public var memoryID: String?
    public var canonicalKey: String?
    public var query: String?
    public var snippet: String?
    public var confidence: Double
    public var sourceID: String?
    public var createdAt: Date

    public init(
        id: String = UUID().uuidString.lowercased(),
        kind: MemorySignalKind,
        memoryID: String? = nil,
        canonicalKey: String? = nil,
        query: String? = nil,
        snippet: String? = nil,
        confidence: Double = 1.0,
        sourceID: String? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.kind = kind
        self.memoryID = memoryID
        self.canonicalKey = canonicalKey
        self.query = query?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.snippet = snippet?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.confidence = min(1, max(0, confidence))
        self.sourceID = sourceID
        self.createdAt = createdAt
    }
}

public enum MemoryMaintenanceMode: String, Sendable, Codable, Hashable {
    case preview
    case apply
}

public struct MemoryMaintenanceRequest: Sendable, Codable, Hashable {
    public var mode: MemoryMaintenanceMode
    public var lookbackDays: Int
    public var minSignalCount: Int
    public var minDistinctQueries: Int
    public var minConfidence: Double
    public var limit: Int
    public var compactionObservations: [MemoryCompactionObservation]

    public init(
        mode: MemoryMaintenanceMode = .preview,
        lookbackDays: Int = 30,
        minSignalCount: Int = 3,
        minDistinctQueries: Int = 2,
        minConfidence: Double = 0.75,
        limit: Int = 20,
        compactionObservations: [MemoryCompactionObservation] = []
    ) {
        self.mode = mode
        self.lookbackDays = max(1, lookbackDays)
        self.minSignalCount = max(1, minSignalCount)
        self.minDistinctQueries = max(1, minDistinctQueries)
        self.minConfidence = min(1, max(0, minConfidence))
        self.limit = max(1, limit)
        self.compactionObservations = compactionObservations
    }
}

public struct MemoryMaintenanceResult: Sendable, Codable, Hashable {
    public var proposedCandidates: [MemoryCandidate]
    public var ingestResult: MemoryIngestResult?
    public var consideredSignalCount: Int

    public init(
        proposedCandidates: [MemoryCandidate] = [],
        ingestResult: MemoryIngestResult? = nil,
        consideredSignalCount: Int = 0
    ) {
        self.proposedCandidates = proposedCandidates
        self.ingestResult = ingestResult
        self.consideredSignalCount = max(0, consideredSignalCount)
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
    public let memoryType: String?
    public let memoryTypeConfidence: Double?
    public let score: SearchScoreBreakdown
    public let contextHints: [MemoryContextHint]
}

public struct MemoryGetResponse: Sendable, Codable, Hashable {
    public let documentPath: String
    public let source: MemoryDocumentSource
    public let totalLineCount: Int
    public let lineRange: MemoryLineRange
    public let content: String
    public let contextHints: [MemoryContextHint]
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
    case groundedExpansion(applied: Bool, queryCount: Int, termCount: Int, reason: String?)
    case reranked(count: Int)
    case memoryTypeIntent(label: String, confidence: Double)
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
