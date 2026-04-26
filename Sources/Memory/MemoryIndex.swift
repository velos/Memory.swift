import Accelerate
import CryptoKit
import Foundation
import MemoryStorage

private final class MemoryAsyncLock: @unchecked Sendable {
    private let lock = NSLock()
    private var locked = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if locked {
                waiters.append(continuation)
                lock.unlock()
            } else {
                locked = true
                lock.unlock()
                continuation.resume()
            }
        }
    }

    func release() {
        lock.lock()
        if waiters.isEmpty {
            locked = false
            lock.unlock()
        } else {
            let continuation = waiters.removeFirst()
            lock.unlock()
            continuation.resume()
        }
    }
}

public actor MemoryIndex {
    private let configuration: MemoryConfiguration
    private let storage: MemoryStorage
    private let fileManager: FileManager
    private let ingestLock = MemoryAsyncLock()

    private let markdownExtensions: Set<String> = ["md", "markdown", "mdx"]
    private let codeExtensions: Set<String> = [
        "swift", "m", "mm", "h", "hpp", "c", "cpp", "cc", "cxx",
        "js", "jsx", "ts", "tsx", "java", "kt", "kts",
        "go", "rs", "py", "rb", "php", "cs", "scala", "sh", "zsh", "bash"
    ]
    private let strongLexicalProbeLimit = 20
    private let strongLexicalMinScore = 0.10
    private let strongLexicalMinGap = 0.05
    private let strongLexicalMaxExpansionSkipTokenCount = 12
    private let lexicalExpansionPromotionRankLimit = 1
    private let primarySemanticPreservationRankLimit = 10
    private let primaryLexicalPreservationRankLimit = 10
    private let documentLexicalMaxBranches = 2
    private let documentLexicalSparseHitThreshold = 12
    private let documentLexicalPrimaryWeight = 0.45
    private let documentLexicalExpansionWeight = 0.60
    private let maxCandidateHydrationLimit = 1_000

    private struct WeightedQuery {
        var text: String
        var weight: Double
        var expansionType: ExpansionType?
    }

    private struct QueryMatchSignals {
        var facets: Set<FacetTag>
        var entityValues: Set<String>
        var topics: Set<String>
        var temporalIntent: RecallTemporalIntent
        var preferredStatuses: Set<MemoryStatus>
    }

    private struct StructuredSearchPlan {
        var expandedQueries: [WeightedQuery]
        var analysis: QueryAnalysis
        var entityLexicalQueries: [String]
        var facetTagNames: [String]
        var entityTagNames: [String]
        var topicTagNames: [String]
        var temporalIntent: RecallTemporalIntent
        var preferredStatuses: Set<MemoryStatus>
    }

    private struct PreparedMemoryCandidate {
        var text: String
        var kind: MemoryKind
        var status: MemoryStatus
        var importance: Double
        var confidence: Double?
        var createdAt: Date
        var eventAt: Date?
        var source: String
        var title: String?
        var tags: [String]
        var facetTags: Set<FacetTag>
        var entities: [MemoryEntity]
        var topics: [String]
        var canonicalKey: String?
        var metadata: [String: String]
        var proposedAction: MemoryWriteAction?
    }

    private struct IngestConsolidationResult {
        var primaryMemoryID: String
        var impactedMemoryIDs: Set<String>
        var action: MemoryWriteAction
    }

    private let minimumAutoWriteConfidence = 0.55

    public init(configuration: MemoryConfiguration, fileManager: FileManager = .default) throws {
        guard !configuration.databaseURL.path.isEmpty else {
            throw MemoryError.configuration("databaseURL must not be empty")
        }

        self.configuration = configuration
        self.fileManager = fileManager

        do {
            self.storage = try MemoryStorage(databaseURL: configuration.databaseURL)
        } catch {
            throw MemoryError.storage("Failed to initialize storage: \(error.localizedDescription)")
        }
    }

    public func rebuildIndex(from roots: [URL]) async throws {
        try await rebuildIndex(from: IndexingRequest(roots: roots), events: nil)
    }

    public func rebuildIndex(from request: IndexingRequest, events: IndexingEventHandler?) async throws {
        let urls = try collectDocumentURLs(from: request)
        events?(.started(totalDocuments: urls.count))

        do {
            try await storage.wipeIndexData()

            var totalChunks = 0
            for (index, url) in urls.enumerated() {
                let documentStart = DispatchTime.now().uptimeNanoseconds
                events?(.readingDocument(path: url.path, index: index + 1, total: urls.count))
                guard let payload = try await buildDocumentPayload(for: url, events: events) else { continue }

                totalChunks += payload.chunks.count
                events?(.chunked(path: url.path, chunks: payload.chunks.count))
                events?(.embedded(path: url.path, chunks: payload.chunks.count))

                let indexWriteStart = DispatchTime.now().uptimeNanoseconds
                try await storage.replaceDocument(payload)
                events?(
                    .stageTiming(
                        path: url.path,
                        stage: .indexWrite,
                        durationMs: elapsedMilliseconds(since: indexWriteStart)
                    )
                )
                events?(
                    .stageTiming(
                        path: url.path,
                        stage: .total,
                        durationMs: elapsedMilliseconds(since: documentStart)
                    )
                )
                events?(.stored(path: url.path))
            }

            try await rematerializeStoredMemories()

            events?(.completed(processedDocuments: urls.count, totalChunks: totalChunks))
        } catch {
            throw normalizeError(error)
        }
    }

    public func syncDocuments(_ urls: [URL]) async throws {
        try await syncDocuments(urls, events: nil)
    }

    public func syncDocuments(_ urls: [URL], events: IndexingEventHandler?) async throws {
        let request = IndexingRequest(roots: urls)
        let documentURLs = try collectDocumentURLs(from: request)
        events?(.started(totalDocuments: documentURLs.count))

        do {
            var totalChunks = 0
            for (index, url) in documentURLs.enumerated() {
                let documentStart = DispatchTime.now().uptimeNanoseconds
                events?(.readingDocument(path: url.path, index: index + 1, total: documentURLs.count))

                if !fileManager.fileExists(atPath: url.path) {
                    try await storage.removeDocuments(paths: [url.path])
                    continue
                }

                guard let payload = try await buildDocumentPayload(for: url, events: events) else { continue }
                totalChunks += payload.chunks.count

                events?(.chunked(path: url.path, chunks: payload.chunks.count))
                events?(.embedded(path: url.path, chunks: payload.chunks.count))
                let indexWriteStart = DispatchTime.now().uptimeNanoseconds
                try await storage.replaceDocument(payload)
                events?(
                    .stageTiming(
                        path: url.path,
                        stage: .indexWrite,
                        durationMs: elapsedMilliseconds(since: indexWriteStart)
                    )
                )
                events?(
                    .stageTiming(
                        path: url.path,
                        stage: .total,
                        durationMs: elapsedMilliseconds(since: documentStart)
                    )
                )
                events?(.stored(path: url.path))
            }

            events?(.completed(processedDocuments: documentURLs.count, totalChunks: totalChunks))
        } catch {
            throw normalizeError(error)
        }
    }

    public func removeDocuments(at urls: [URL]) async throws {
        do {
            let paths = urls.map(\.path)
            try await storage.removeDocuments(paths: paths)
        } catch {
            throw normalizeError(error)
        }
    }

    public func search(_ query: SearchQuery) async throws -> [SearchResult] {
        try await search(query, events: nil, allowedChunkIDsOverride: nil, recallPlan: nil)
    }

    public func search(_ query: SearchQuery, events: SearchEventHandler?) async throws -> [SearchResult] {
        try await search(query, events: events, allowedChunkIDsOverride: nil, recallPlan: nil)
    }

    private func search(
        _ query: SearchQuery,
        events: SearchEventHandler?,
        allowedChunkIDsOverride: Set<Int64>?,
        recallPlan: RecallPlan?
    ) async throws -> [SearchResult] {
        let normalizedText = query.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedText.isEmpty else { return [] }

        let queryStart = DispatchTime.now().uptimeNanoseconds
        events?(.started(query: normalizedText))

        let allowedChunkIDs: Set<Int64>?
        if let contextID = query.contextID {
            let contextChunkIDs = try await storage.fetchContextChunkIDs(contextID: contextID.rawValue)
            let contextSet = Set(contextChunkIDs)
            allowedChunkIDs = combineAllowedChunkIDs(contextSet, allowedChunkIDsOverride)
            if contextSet.isEmpty || (allowedChunkIDs?.isEmpty == true) {
                events?(.completed(count: 0))
                return []
            }
        } else {
            allowedChunkIDs = allowedChunkIDsOverride
        }

        if let allowedChunkIDs, allowedChunkIDs.isEmpty {
            events?(.completed(count: 0))
            return []
        }

        let allowedMemoryTypes: Set<String>? = nil

        let analysisStart = DispatchTime.now().uptimeNanoseconds
        let queryAnalysis = configuration.queryAnalyzer?.analyze(query: normalizedText) ?? heuristicQueryAnalysis(for: normalizedText)
        events?(.stageTiming(stage: .analysis, durationMs: elapsedMilliseconds(since: analysisStart)))

        let lexicalProbeStart = DispatchTime.now().uptimeNanoseconds
        let lexicalProbe = try await runLexicalProbe(
            query: query,
            normalizedText: normalizedText,
            allowedChunkIDs: allowedChunkIDs,
            allowedMemoryTypes: allowedMemoryTypes
        )
        var lexicalSearchDurationMs = elapsedMilliseconds(since: lexicalProbeStart)

        let expansionStart = DispatchTime.now().uptimeNanoseconds
        let searchPlan = try await prepareStructuredSearchPlan(
            query: query,
            normalizedText: normalizedText,
            analysis: queryAnalysis,
            recallPlan: recallPlan,
            skipExpansion: lexicalProbe.strongSignal,
            events: events
        )
        events?(.stageTiming(stage: .expansion, durationMs: elapsedMilliseconds(since: expansionStart)))
        events?(.expandedQueries(count: max(0, searchPlan.expandedQueries.count - 1)))

        let queryEmbeddingStart = DispatchTime.now().uptimeNanoseconds
        let semanticQueryVectors = try await embedExpandedQueries(
            searchPlan.expandedQueries,
            semanticCandidateLimit: query.semanticCandidateLimit,
            events: events
        )
        events?(.stageTiming(stage: .queryEmbedding, durationMs: elapsedMilliseconds(since: queryEmbeddingStart)))

        var semanticRRF: [Int64: Double] = [:]
        var lexicalRRF: [Int64: Double] = [:]
        var lexicalExpansionPromotionRRF: [Int64: Double] = [:]
        var primarySemanticPreservationRRF: [Int64: Double] = [:]
        var primaryLexicalPreservationRRF: [Int64: Double] = [:]
        var semanticCandidateCount = 0
        var lexicalCandidateCount = 0
        var semanticSearchDurationMs = 0.0
        let hasExpandedBranches = searchPlan.expandedQueries.count > 1
        var documentLexicalBranchCount = 0

        for (index, expandedQuery) in searchPlan.expandedQueries.enumerated() {
            let skipSemantic = expandedQuery.expansionType == .lexical
            let skipLexical = expandedQuery.expansionType == .semantic
                || expandedQuery.expansionType == .hypotheticalDocument

            if !skipSemantic, let semanticQueryVectors {
                let semanticSearchStart = DispatchTime.now().uptimeNanoseconds
                let semanticHits = try await semanticSearch(
                    queryVector: semanticQueryVectors[index],
                    limit: query.semanticCandidateLimit,
                    allowedChunkIDs: allowedChunkIDs,
                    allowedMemoryTypes: allowedMemoryTypes
                )
                semanticSearchDurationMs += elapsedMilliseconds(since: semanticSearchStart)
                semanticCandidateCount += semanticHits.count
                accumulateRRF(for: semanticHits, weight: expandedQuery.weight, into: &semanticRRF)
                if index == 0, hasExpandedBranches {
                    accumulatePrimaryBranchPreservation(
                        for: semanticHits,
                        rankLimit: primarySemanticPreservationRankLimit,
                        baseContribution: 0.08,
                        weight: expandedQuery.weight,
                        into: &primarySemanticPreservationRRF
                    )
                }
            }

            if !skipLexical, query.lexicalCandidateLimit > 0 {
                let lexicalHits: [LexicalHit]
                if index == 0, let seeded = lexicalProbe.seededHits {
                    lexicalHits = seeded
                } else {
                    let lexicalSearchStart = DispatchTime.now().uptimeNanoseconds
                    lexicalHits = try await storage.lexicalSearch(
                        query: ftsPreprocess(expandedQuery.text),
                        limit: query.lexicalCandidateLimit,
                        allowedChunkIDs: allowedChunkIDs,
                        allowedMemoryTypes: allowedMemoryTypes
                    )
                    lexicalSearchDurationMs += elapsedMilliseconds(since: lexicalSearchStart)
                }
                lexicalCandidateCount += lexicalHits.count
                accumulateRRF(for: lexicalHits, weight: expandedQuery.weight, into: &lexicalRRF)
                if index == 0, hasExpandedBranches {
                    accumulatePrimaryBranchPreservation(
                        for: lexicalHits,
                        rankLimit: primaryLexicalPreservationRankLimit,
                        baseContribution: 0.05,
                        weight: expandedQuery.weight,
                        into: &primaryLexicalPreservationRRF
                    )
                }
                if index > 0, expandedQuery.expansionType == .lexical {
                    accumulateLexicalExpansionPromotion(
                        for: lexicalHits,
                        queryText: expandedQuery.text,
                        primaryQueryText: normalizedText,
                        weight: expandedQuery.weight,
                        into: &lexicalExpansionPromotionRRF
                    )
                }
                if shouldRunDocumentLexicalSearch(
                    query: query,
                    queryText: expandedQuery.text,
                    branchIndex: index,
                    expansionType: expandedQuery.expansionType,
                    lexicalHitCount: lexicalHits.count,
                    lexicalProbeStrongSignal: lexicalProbe.strongSignal,
                    usedBranches: documentLexicalBranchCount
                ) {
                    documentLexicalBranchCount += 1
                    let documentLexicalSearchStart = DispatchTime.now().uptimeNanoseconds
                    let documentHits = try await storage.lexicalDocumentSearch(
                        query: ftsPreprocess(expandedQuery.text),
                        limit: documentLexicalCandidateLimit(for: query, branchIndex: index),
                        allowedChunkIDs: allowedChunkIDs,
                        allowedMemoryTypes: allowedMemoryTypes
                    )
                    lexicalSearchDurationMs += elapsedMilliseconds(since: documentLexicalSearchStart)
                    lexicalCandidateCount += documentHits.count
                    accumulateScoredRRF(
                        for: documentHits,
                        weight: expandedQuery.weight * documentLexicalWeight(branchIndex: index),
                        into: &lexicalRRF
                    )
                }
            }
        }

        if !searchPlan.entityLexicalQueries.isEmpty, query.lexicalCandidateLimit > 0 {
            for entityQuery in searchPlan.entityLexicalQueries {
                let lexicalSearchStart = DispatchTime.now().uptimeNanoseconds
                let entityHits = try await storage.lexicalSearch(
                    query: ftsPreprocess(entityQuery),
                    limit: max(1, query.lexicalCandidateLimit / 2),
                    allowedChunkIDs: allowedChunkIDs,
                    allowedMemoryTypes: allowedMemoryTypes
                )
                lexicalSearchDurationMs += elapsedMilliseconds(since: lexicalSearchStart)
                accumulateRRF(for: entityHits, weight: 0.5, into: &lexicalRRF)
                lexicalCandidateCount += entityHits.count
            }
        }

        if query.includeTagScoring, query.lexicalCandidateLimit > 0 {
            let metadataLimit = max(8, query.lexicalCandidateLimit / 2)
            if !searchPlan.entityTagNames.isEmpty {
                let entityTagHits = try await storage.contentTagSearch(
                    tagNames: searchPlan.entityTagNames,
                    limit: metadataLimit,
                    allowedChunkIDs: allowedChunkIDs,
                    allowedMemoryTypes: allowedMemoryTypes
                )
                accumulateRRF(for: entityTagHits, weight: 0.60, into: &lexicalRRF)
                lexicalCandidateCount += entityTagHits.count
            }

            if !searchPlan.topicTagNames.isEmpty {
                let topicTagHits = try await storage.contentTagSearch(
                    tagNames: searchPlan.topicTagNames,
                    limit: metadataLimit,
                    allowedChunkIDs: allowedChunkIDs,
                    allowedMemoryTypes: allowedMemoryTypes
                )
                accumulateRRF(for: topicTagHits, weight: 0.35, into: &lexicalRRF)
                lexicalCandidateCount += topicTagHits.count
            }

            if !searchPlan.facetTagNames.isEmpty {
                let facetTagHits = try await storage.contentTagSearch(
                    tagNames: searchPlan.facetTagNames,
                    limit: metadataLimit,
                    allowedChunkIDs: allowedChunkIDs,
                    allowedMemoryTypes: allowedMemoryTypes
                )
                accumulateRRF(for: facetTagHits, weight: 0.25, into: &lexicalRRF)
                lexicalCandidateCount += facetTagHits.count
            }
        }

        events?(.stageTiming(stage: .semanticSearch, durationMs: semanticSearchDurationMs))
        events?(.stageTiming(stage: .lexicalSearch, durationMs: lexicalSearchDurationMs))
        events?(.semanticCandidates(count: semanticCandidateCount))
        events?(.lexicalCandidates(count: lexicalCandidateCount))

        let fusionStart = DispatchTime.now().uptimeNanoseconds
        let querySignals = queryMatchSignals(from: searchPlan.analysis, plan: recallPlan)
        let queryTags = query.includeTagScoring
            ? await resolveQueryContentTags(queryText: normalizedText, queryAnalysis: searchPlan.analysis, events: events)
            : []
        var fused = try await fuseCandidates(
            semanticRRF: semanticRRF,
            lexicalRRF: lexicalRRF,
            lexicalExpansionPromotionRRF: lexicalExpansionPromotionRRF,
            primarySemanticPreservationRRF: primarySemanticPreservationRRF,
            primaryLexicalPreservationRRF: primaryLexicalPreservationRRF,
            query: query,
            primaryQueryText: normalizedText,
            queryTags: queryTags,
            querySignals: querySignals
        )
        events?(.stageTiming(stage: .fusion, durationMs: elapsedMilliseconds(since: fusionStart)))
        events?(.fusedCandidates(count: fused.count))

        let rerankCount = effectiveRerankCount(query: query, fusedCount: fused.count)
        if let reranker = configuration.reranker, !fused.isEmpty, rerankCount > 0 {
            do {
                let rerankStart = DispatchTime.now().uptimeNanoseconds
                fused = try await applyReranker(
                    reranker,
                    query: query,
                    fusedResults: fused,
                    rerankCount: rerankCount
                )
                events?(.stageTiming(stage: .rerank, durationMs: elapsedMilliseconds(since: rerankStart)))
                events?(.reranked(count: rerankCount))
            } catch {
                // Fall back to fused ordering if reranking fails.
                events?(
                    .providerFailure(
                        stage: .rerank,
                        provider: reranker.identifier,
                        message: error.localizedDescription
                    )
                )
                fused = fused.map {
                    var updated = $0
                    updated.score.blended = updated.score.fused
                    updated.score.rerank = 0
                    return updated
                }
            }
        } else {
            fused = fused.map {
                var updated = $0
                updated.score.blended = updated.score.fused
                updated.score.rerank = 0
                return updated
            }
        }

        let final = Array(
            fused
                .sorted(by: sortByBlendedScore(_:_:))
                .prefix(query.limit)
        )
        events?(.stageTiming(stage: .total, durationMs: elapsedMilliseconds(since: queryStart)))
        events?(.completed(count: final.count))
        return final
    }

    public func createContext(name: String) async throws -> ContextID {
        let normalizedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedName.isEmpty else {
            throw MemoryError.configuration("Context name must not be empty")
        }

        let generated = ContextID()
        do {
            let contextID = try await storage.createContext(id: generated.rawValue, name: normalizedName)
            return ContextID(rawValue: contextID)
        } catch {
            throw normalizeError(error)
        }
    }

    public func addToContext(_ contextID: ContextID, chunkIDs: [Int64]) async throws {
        guard !chunkIDs.isEmpty else { return }
        do {
            try await storage.addContextChunks(contextID: contextID.rawValue, chunkIDs: chunkIDs)
        } catch {
            throw normalizeError(error)
        }
    }

    public func clearContext(_ contextID: ContextID) async throws {
        do {
            try await storage.clearContext(contextID: contextID.rawValue)
        } catch {
            throw normalizeError(error)
        }
    }

    public func listContextChunks(_ contextID: ContextID) async throws -> [SearchResult] {
        do {
            let rows = try await storage.listContextChunks(contextID: contextID.rawValue)
            return rows.map {
                makeSearchResult(from: $0, queryText: nil, score: SearchScoreBreakdown(semantic: 0, lexical: 0, recency: 0, fused: 0))
            }
        } catch {
            throw normalizeError(error)
        }
    }

    public func getChunk(id: Int64) async throws -> SearchResult? {
        do {
            guard let row = try await storage.fetchChunkMetadata(chunkID: id) else {
                return nil
            }

            return makeSearchResult(
                from: row,
                queryText: nil,
                score: SearchScoreBreakdown(semantic: 0, lexical: 0, recency: 0, fused: 0)
            )
        } catch {
            throw normalizeError(error)
        }
    }

    public func listIndexedDocumentPaths() async throws -> [String] {
        do {
            return try await storage.listDocumentPaths()
        } catch {
            throw normalizeError(error)
        }
    }

    public func save(
        text: String,
        kind: MemoryKind,
        status: MemoryStatus = .active,
        importance: Double = 0.5,
        source: String = "memory_save",
        createdAt: Date? = nil,
        eventAt: Date? = nil,
        tags: [String] = [],
        facetTags: Set<FacetTag> = [],
        entities: [MemoryEntity] = [],
        topics: [String] = [],
        canonicalKey: String? = nil,
        confidence: Double? = 1.0,
        metadata: [String: String] = [:]
    ) async throws -> MemoryRecord {
        let result = try await ingest(
            [
                MemoryCandidate(
                    text: text,
                    kind: kind,
                    status: status,
                    importance: importance,
                    confidence: confidence,
                    createdAt: createdAt,
                    eventAt: eventAt,
                    source: source,
                    tags: tags,
                    facetTags: facetTags,
                    entities: entities,
                    topics: topics,
                    canonicalKey: canonicalKey,
                    metadata: metadata
                ),
            ]
        )

        guard let record = result.records.first else {
            throw MemoryError.ingestion("Failed to save memory from provided text.")
        }

        return record
    }

    public func extract(
        from text: String,
        limit: Int = 50
    ) async throws -> [MemoryCandidate] {
        try await extract(
            from: [
                ConversationMessage(role: .user, content: text),
            ],
            limit: limit
        )
    }

    public func extract(
        from messages: [ConversationMessage],
        limit: Int = 50
    ) async throws -> [MemoryCandidate] {
        try await extractDetailed(from: messages, limit: limit).candidates
    }

    public func extractDetailed(
        from text: String,
        limit: Int = 50
    ) async throws -> MemoryExtractionResult {
        try await extractDetailed(
            from: [
                ConversationMessage(role: .user, content: text),
            ],
            limit: limit
        )
    }

    public func extractDetailed(
        from messages: [ConversationMessage],
        limit: Int = 50
    ) async throws -> MemoryExtractionResult {
        guard limit > 0 else { return MemoryExtractionResult() }
        guard !messages.isEmpty else { return MemoryExtractionResult() }

        if let extractor = configuration.memoryExtractor {
            return try await extractor.extract(messages: messages, limit: limit)
        }

        return heuristicExtract(messages: messages, limit: limit)
    }

    public func ingest(_ memories: [MemoryCandidate]) async throws -> MemoryIngestResult {
        await ingestLock.acquire()
        do {
            let result = try await ingestUnlocked(memories)
            ingestLock.release()
            return result
        } catch {
            ingestLock.release()
            throw error
        }
    }

    private func ingestUnlocked(_ memories: [MemoryCandidate]) async throws -> MemoryIngestResult {
        guard !memories.isEmpty else {
            return MemoryIngestResult(requestedCount: 0, storedCount: 0, discardedCount: 0, records: [])
        }

        var records: [MemoryRecord] = []
        var actions: [MemoryWriteAction] = []
        var discardedCount = 0
        records.reserveCapacity(memories.count)
        actions.reserveCapacity(memories.count)

        for memory in memories {
            guard let prepared = prepareCandidateForIngest(memory) else {
                discardedCount += 1
                actions.append(.noWrite)
                continue
            }

            do {
                let consolidation = try await ingestPreparedCandidate(prepared)
                actions.append(consolidation.action)
                for impactedMemoryID in consolidation.impactedMemoryIDs {
                    try await materializeStoredMemory(id: impactedMemoryID)
                }

                if let stored = try await storage.fetchStoredMemory(id: consolidation.primaryMemoryID),
                   let record = makeMemoryRecord(from: stored, score: nil) {
                    records.append(record)
                } else {
                    discardedCount += 1
                    if actions.indices.contains(actions.count - 1) {
                        actions[actions.count - 1] = .noWrite
                    }
                }
            } catch {
                throw normalizeError(error)
            }
        }

        return MemoryIngestResult(
            requestedCount: memories.count,
            storedCount: records.count,
            discardedCount: discardedCount,
            records: records,
            actions: actions
        )
    }

    public func recall(
        mode: RecallMode,
        limit: Int = 20,
        features: RecallFeatures = .hybridDefault,
        sort: RecallSort = .recent,
        conversationContext: [ConversationMessage] = [],
        kinds: Set<MemoryKind>? = nil,
        statuses: Set<MemoryStatus>? = [.active],
        facets: Set<FacetTag>? = nil,
        entityValues: [String]? = nil,
        topics: [String]? = nil,
        events: SearchEventHandler? = nil
    ) async throws -> MemoryRecallResponse {
        let effectiveLimit = max(1, limit)
        switch mode {
        case let .hybrid(query):
            let plan = try await resolveRecallPlan(
                query: query,
                conversationContext: conversationContext,
                features: features,
                events: events
            )
            let queryText = plan.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? query : plan.query
            let effectiveKinds = intersectKinds(resolveKindsFilter(for: mode, requestedKinds: kinds), plan.kinds)
            let effectiveStatuses = plan.statuses ?? statuses
            let effectiveFacets = intersectFacets(facets, plan.facets)
            let effectiveEntityValues = mergeFilterValues(entityValues, plan.entityValues)
            let effectiveTopics = mergeFilterValues(topics, plan.topics)
            let allowedChunkIDs = try await resolveMemoryChunkFilter(
                kinds: effectiveKinds,
                statuses: effectiveStatuses,
                facets: effectiveFacets,
                entityValues: effectiveEntityValues,
                topics: effectiveTopics
            )
            if allowedChunkIDs?.isEmpty == true {
                return MemoryRecallResponse(records: [])
            }

            let semanticLimit = features.contains(.semantic)
                ? max(plan.semanticCandidateLimit ?? configuration.semanticCandidateLimit, effectiveLimit * 4)
                : 0
            let lexicalLimit = features.contains(.lexical)
                ? max(plan.lexicalCandidateLimit ?? configuration.lexicalCandidateLimit, effectiveLimit * 4)
                : 0
            let rerankLimit = features.contains(.rerank)
                ? min(80, max(plan.rerankLimit ?? 40, effectiveLimit * 2))
                : 0
            let expansionLimit = features.contains(.expansion) ? 5 : 0

            let searchResults = try await search(
                SearchQuery(
                    text: queryText,
                    limit: effectiveLimit,
                    semanticCandidateLimit: semanticLimit,
                    lexicalCandidateLimit: lexicalLimit,
                    rerankLimit: rerankLimit,
                    expansionLimit: expansionLimit,
                    includeTagScoring: features.contains(.tags)
                ),
                events: events,
                allowedChunkIDsOverride: allowedChunkIDs,
                recallPlan: plan
            )

            var records: [MemoryRecord] = []
            records.reserveCapacity(searchResults.count)
            for result in searchResults {
                guard let memoryID = result.memoryID else { continue }
                guard let stored = try await storage.fetchStoredMemory(id: memoryID) else { continue }
                guard let record = makeMemoryRecord(from: stored, score: result.score) else { continue }
                records.append(record)
            }

            do {
                try await storage.recordChunkAccesses(records.map { $0.chunkID })
            } catch {
                throw normalizeError(error)
            }

            return MemoryRecallResponse(records: records)
        case .recent, .important, .kind:
            let effectiveKinds = resolveKindsFilter(for: mode, requestedKinds: kinds)
            let sortMode: StoredMemorySort
            switch mode {
            case .recent:
                sortMode = .recent
            case .important:
                sortMode = .importance
            case .kind:
                sortMode = storageSort(for: sort)
            default:
                sortMode = .recent
            }

            let rows: [StoredMemoryRecord]
            do {
                rows = try await storage.listStoredMemories(
                    limit: effectiveLimit,
                    sort: sortMode,
                    kinds: effectiveKinds.map { Set($0.map(\.rawValue)) },
                    statuses: statuses.map { Set($0.map(\.rawValue)) }
                )
            } catch {
                throw normalizeError(error)
            }

            let filteredRows = Array(
                filterStoredMemories(
                    rows,
                    facets: facets,
                    entityValues: entityValues,
                    topics: topics
                )
                    .prefix(effectiveLimit)
            )
            let records = filteredRows.compactMap { makeMemoryRecord(from: $0, score: nil) }
            do {
                try await storage.recordChunkAccesses(records.map { $0.chunkID })
            } catch {
                throw normalizeError(error)
            }
            return MemoryRecallResponse(records: records)
        }
    }

    public func memorySearch(
        query: String,
        limit: Int = 10,
        features: RecallFeatures = .hybridDefault,
        conversationContext: [ConversationMessage] = [],
        kinds: Set<MemoryKind>? = nil,
        statuses: Set<MemoryStatus>? = nil,
        facets: Set<FacetTag>? = nil,
        entityValues: [String]? = nil,
        topics: [String]? = nil,
        dedupeDocuments: Bool = true,
        includeLineRanges: Bool = true,
        events: SearchEventHandler? = nil
    ) async throws -> [MemorySearchReference] {
        let normalizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedQuery.isEmpty else { return [] }

        let effectiveLimit = max(1, limit)
        let plan = try await resolveRecallPlan(
            query: normalizedQuery,
            conversationContext: conversationContext,
            features: features,
            events: events
        )
        let queryText = plan.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? normalizedQuery : plan.query

        let effectiveKinds = intersectKinds(kinds, plan.kinds)
        let effectiveFacets = intersectFacets(facets, plan.facets)
        let effectiveEntityValues = mergeFilterValues(entityValues, plan.entityValues)
        let effectiveTopics = mergeFilterValues(topics, plan.topics)
        let hasEntityFilter = !(effectiveEntityValues ?? []).isEmpty
        let hasTopicFilter = !(effectiveTopics ?? []).isEmpty
        let usesMemoryOnlyFilter = effectiveKinds != nil || effectiveFacets != nil || hasEntityFilter || hasTopicFilter
        let effectiveStatuses = statuses ?? plan.statuses ?? (usesMemoryOnlyFilter ? [.active] : nil)

        let allowedChunkIDs = try await resolveMemoryChunkFilter(
            kinds: effectiveKinds,
            statuses: effectiveStatuses,
            facets: effectiveFacets,
            entityValues: effectiveEntityValues,
            topics: effectiveTopics
        )
        if (effectiveKinds != nil || effectiveStatuses != nil), allowedChunkIDs?.isEmpty == true {
            return []
        }

        let semanticLimit = features.contains(.semantic)
            ? max(plan.semanticCandidateLimit ?? configuration.semanticCandidateLimit, effectiveLimit * 4)
            : 0
        let lexicalLimit = features.contains(.lexical)
            ? max(plan.lexicalCandidateLimit ?? configuration.lexicalCandidateLimit, effectiveLimit * 4)
            : 0
        let rerankLimit = features.contains(.rerank)
            ? min(80, max(plan.rerankLimit ?? 40, effectiveLimit * 2))
            : 0
        let expansionLimit = features.contains(.expansion) ? 5 : 0
        let searchLimit: Int
        if dedupeDocuments {
            let broadLimit = min(400, max(effectiveLimit * 6, effectiveLimit))
            searchLimit = effectiveLimit >= 50 ? min(320, broadLimit) : broadLimit
        } else {
            searchLimit = effectiveLimit
        }

        let searchResults = try await search(
            SearchQuery(
                text: queryText,
                limit: searchLimit,
                semanticCandidateLimit: semanticLimit,
                lexicalCandidateLimit: lexicalLimit,
                rerankLimit: rerankLimit,
                expansionLimit: expansionLimit,
                includeTagScoring: features.contains(.tags)
            ),
            events: events,
            allowedChunkIDsOverride: allowedChunkIDs,
            recallPlan: plan
        )

        var references: [MemorySearchReference] = []
        references.reserveCapacity(effectiveLimit)

        var seenDocumentKeys: Set<String> = []
        var documentTextCache: [String: String] = [:]

        for result in searchResults {
            if statuses == nil,
               plan.statuses == nil,
               let memoryStatus = result.memoryStatus,
               memoryStatus != .active {
                continue
            }

            if dedupeDocuments {
                let key = normalizedComparisonKey(for: result.documentPath)
                guard seenDocumentKeys.insert(key).inserted else { continue }
            }

            let source = resolveDocumentSource(for: result.documentPath)
            let lineRange: MemoryLineRange?
            if includeLineRanges {
                let documentText: String?
                if let cached = documentTextCache[result.documentPath] {
                    documentText = cached
                } else {
                    let loaded = await loadDocumentTextIfAvailable(for: result.documentPath)
                    documentText = loaded
                    if let loaded {
                        documentTextCache[result.documentPath] = loaded
                    }
                }

                if let documentText {
                    lineRange = inferLineRange(
                        in: documentText,
                        chunkText: result.content,
                        snippet: result.snippet
                    )
                } else {
                    lineRange = nil
                }
            } else {
                lineRange = nil
            }

            references.append(
                MemorySearchReference(
                    chunkID: result.chunkID,
                    documentPath: result.documentPath,
                    title: result.title,
                    snippet: result.snippet,
                    lineRange: lineRange,
                    source: source,
                    memoryID: result.memoryID,
                    memoryKind: result.memoryKind,
                    memoryStatus: result.memoryStatus,
                    score: result.score
                )
            )

            if references.count >= effectiveLimit {
                break
            }
        }

        if !references.isEmpty {
            do {
                try await storage.recordChunkAccesses(references.map(\.chunkID))
            } catch {
                throw normalizeError(error)
            }
        }

        return references
    }

    public func memoryGet(
        path: String,
        lineRange: MemoryLineRange? = nil
    ) async throws -> MemoryGetResponse {
        let resolvedPath = try await resolveDocumentPath(path)
        let loaded = try await loadDocumentText(for: resolvedPath)
        let lines = {
            let split = splitLines(from: loaded.content)
            return split.isEmpty ? [""] : split
        }()
        let totalLineCount = max(1, lines.count)
        let clampedRange = clampLineRange(lineRange, totalLineCount: totalLineCount)

        let lowerIndex = max(0, clampedRange.start - 1)
        let upperIndex = max(lowerIndex, clampedRange.end - 1)
        let selected = Array(lines[lowerIndex...upperIndex]).joined(separator: "\n")

        return MemoryGetResponse(
            documentPath: resolvedPath,
            source: loaded.source,
            totalLineCount: totalLineCount,
            lineRange: clampedRange,
            content: selected
        )
    }

    public func memoryGet(reference: MemorySearchReference) async throws -> MemoryGetResponse {
        try await memoryGet(path: reference.documentPath, lineRange: reference.lineRange)
    }

    private func derivedMemoryPath(for memoryID: String) -> String {
        "memory://\(memoryID)"
    }

    private func normalizeIngestTags(_ raw: [String]) -> [StoredChunkTag] {
        var seen: Set<String> = []
        var normalized: [StoredChunkTag] = []
        normalized.reserveCapacity(raw.count)

        for (index, value) in raw.enumerated() {
            let cleaned = value
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard !cleaned.isEmpty else { continue }
            guard seen.insert(cleaned).inserted else { continue }

            let confidence = max(0.2, 1.0 - (Double(index) * 0.08))
            normalized.append(StoredChunkTag(name: cleaned, confidence: confidence))
        }

        return normalized
    }

    private func storageSort(for sort: RecallSort) -> StoredMemorySort {
        switch sort {
        case .recent:
            return .recent
        case .importance:
            return .importance
        case .mostAccessed:
            return .mostAccessed
        }
    }

    private func makeMemoryRecord(
        from metadata: StoredChunkMetadata,
        score: SearchScoreBreakdown?
    ) -> MemoryRecord? {
        guard let memoryID = metadata.memoryID else { return nil }
        guard let kind = resolveMemoryKind(from: metadata) else { return nil }
        guard let status = resolveMemoryStatus(raw: metadata.memoryStatus, hasMemoryID: true) else { return nil }
        let tags = metadata.contentTags.map { ContentTag(name: $0.name, confidence: $0.confidence) }

        return MemoryRecord(
            id: memoryID,
            chunkID: metadata.chunkID,
            documentPath: metadata.documentPath,
            title: metadata.title,
            text: metadata.content,
            kind: kind,
            status: status,
            canonicalKey: metadata.memoryCanonicalKey,
            importance: metadata.importance,
            confidence: nil,
            accessCount: metadata.accessCount,
            createdAt: metadata.createdAt,
            eventAt: nil,
            modifiedAt: metadata.modifiedAt,
            lastAccessedAt: metadata.lastAccessedAt,
            tags: tags,
            facetTags: [],
            entities: [],
            topics: [],
            score: score
        )
    }

    private func makeMemoryRecord(
        from storedMemory: StoredMemoryRecord,
        score: SearchScoreBreakdown?
    ) -> MemoryRecord? {
        guard
            let chunkID = storedMemory.chunkID,
            let documentPath = storedMemory.documentPath,
            let kind = MemoryKind.parse(storedMemory.kind),
            let status = MemoryStatus.parse(storedMemory.status)
        else {
            return nil
        }

        let tags = storedMemory.contentTags.map { ContentTag(name: $0.name, confidence: $0.confidence) }
        let modifiedAt = max(storedMemory.updatedAt, storedMemory.createdAt)

        return MemoryRecord(
            id: storedMemory.id,
            chunkID: chunkID,
            documentPath: documentPath,
            title: storedMemory.title,
            text: storedMemory.text,
            kind: kind,
            status: status,
            canonicalKey: storedMemory.canonicalKey,
            importance: storedMemory.importance,
            confidence: storedMemory.confidence,
            accessCount: storedMemory.accessCount,
            createdAt: storedMemory.createdAt,
            eventAt: storedMemory.eventAt,
            modifiedAt: modifiedAt,
            lastAccessedAt: storedMemory.lastAccessedAt,
            tags: tags,
            facetTags: Set(storedMemory.facetTags.compactMap(FacetTag.parse)),
            entities: storedMemory.entities.compactMap(makeMemoryEntity(from:)),
            topics: storedMemory.topics,
            score: score
        )
    }

    private func makeSearchResult(
        from metadata: StoredChunkMetadata,
        queryText: String?,
        score: SearchScoreBreakdown
    ) -> SearchResult {
        return SearchResult(
            chunkID: metadata.chunkID,
            documentPath: metadata.documentPath,
            title: metadata.title,
            content: metadata.content,
            snippet: makeSnippet(content: metadata.content, queryText: queryText),
            modifiedAt: metadata.modifiedAt,
            memoryID: metadata.memoryID,
            memoryKind: resolveMemoryKind(from: metadata),
            memoryStatus: resolveMemoryStatus(raw: metadata.memoryStatus, hasMemoryID: metadata.memoryID != nil),
            score: score
        )
    }

    private func heuristicExtract(messages: [ConversationMessage], limit: Int) -> MemoryExtractionResult {
        var extracted: [MemoryCandidate] = []
        extracted.reserveCapacity(limit)
        var rejected: [MemoryRejectedSpan] = []
        var rationales: [String] = []

        var seen: Set<String> = []
        for message in messages {
            let normalized = message.content
                .replacingOccurrences(of: "\r\n", with: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalized.isEmpty else { continue }

            let rawSegments = normalized.split { character in
                character == "\n" || character == "." || character == "!" || character == "?"
            }

            for rawSegment in rawSegments {
                let segment = String(rawSegment).trimmingCharacters(in: .whitespacesAndNewlines)
                guard segment.count >= 18 else {
                    if !segment.isEmpty {
                        rejected.append(MemoryRejectedSpan(text: segment, reason: "too_short", confidence: 0.9))
                    }
                    continue
                }

                let key = normalizedComparisonKey(for: segment)
                guard seen.insert(key).inserted else {
                    rejected.append(MemoryRejectedSpan(text: segment, reason: "duplicate_span", confidence: 0.9))
                    continue
                }

                let kind = inferKind(forExtractedText: segment)
                guard isExtractableMemorySegment(segment, kind: kind, role: message.role) else {
                    rejected.append(MemoryRejectedSpan(text: segment, reason: "not_memory_worthy", confidence: 0.85))
                    continue
                }

                let status = inferStatus(forExtractedText: segment, kind: kind)
                let importance = inferredImportance(for: kind)
                let confidence = inferredConfidence(for: kind)
                let facetTags = inferFacetTags(forExtractedText: segment, kind: kind)
                let entities = inferEntities(forExtractedText: segment)
                let topics = inferTopics(forExtractedText: segment, seedTags: inferredTags(forExtractedText: segment))

                extracted.append(
                    MemoryCandidate(
                        text: segment,
                        kind: kind,
                        status: status,
                        importance: importance,
                        confidence: confidence,
                        createdAt: nil,
                        eventAt: kind == .episode ? Date() : nil,
                        source: "heuristic_extract",
                        tags: inferredTags(forExtractedText: segment),
                        facetTags: facetTags,
                        entities: entities,
                        topics: topics,
                        canonicalKey: resolveCanonicalKey(
                            for: kind,
                            text: segment,
                            explicitKey: nil,
                            entities: entities,
                            topics: topics
                        )
                    )
                )
                rationales.append("\(kind.rawValue):\(status.rawValue):\(segment)")

                if extracted.count >= limit {
                    return MemoryExtractionResult(
                        candidates: extracted,
                        rejectedSpans: rejected,
                        proposedActions: extracted.map(proposedWriteAction(for:)),
                        rationale: rationales
                    )
                }
            }
        }

        return MemoryExtractionResult(
            candidates: extracted,
            rejectedSpans: rejected,
            proposedActions: extracted.map(proposedWriteAction(for:)),
            rationale: rationales
        )
    }

    private func inferKind(forExtractedText text: String) -> MemoryKind {
        let lower = text.lowercased()

        if containsAny(lower, needles: ["runbook", "procedure", "playbook", "workflow", "how to", "step by step", "guide:"]) {
            return .procedure
        }
        if containsAny(lower, needles: ["decide", "decision", "chose", "choose", "switch to", "switched", "agreed", "picked", "settled on"]) {
            return .decision
        }
        if containsAny(
            lower,
            needles: [
                "todo", "to do", "done:", "follow up", "action item", "next step",
                "need to", "must", "should", "finished", "completed", " is done",
                " is complete", " is completed", " is finished"
            ]
        ) {
            return .commitment
        }
        if containsAny(
            lower,
            needles: [
                "prefer", "preference", "favorite", "likes", "usually", "works closely",
                "timezone", "my name", "i am", "i'm", "my role", "role is",
                " is the maintainer", "standing constraint"
            ]
        ) {
            return .profile
        }
        if containsAny(lower, needles: ["handoff", "current status", "blocked on", "next owner", "for the next person", "context"]) {
            return .handoff
        }
        if containsAny(lower, needles: ["today", "yesterday", "this morning", "last week", "after the demo", "incident", "meeting", "met ", "retrospective", "shipped", "celebrated"]) {
            return .episode
        }
        return .fact
    }

    private func isExtractableMemorySegment(
        _ text: String,
        kind: MemoryKind,
        role: ConversationRole? = nil
    ) -> Bool {
        let lower = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !lower.isEmpty else { return false }

        if lower.hasSuffix("?") {
            return false
        }

        if role == .assistant,
           containsAny(
               lower,
               needles: [
                   "i will remember", "i'll remember", "i can remember",
                   "i have noted", "i noted", "noted that", "i will keep",
                   "i'll keep", "sure, i can", "happy to explain"
               ]
           ) {
            return false
        }

        let durableMemoryRequest = containsAny(
            lower,
            needles: ["remember that", "please remember", "for future reference", "note that", "keep in mind"]
        )
        if isQuestionLikeNonMemorySegment(lower), !durableMemoryRequest {
            return false
        }

        let conversationalRequest = containsAny(
            lower,
            needles: [
                "thank you", "thanks", "sounds good", "got it", "okay", "ok ",
                "can you", "could you", "would you", "please explain", "explain how",
                "hypothetical question"
            ]
        )
        if conversationalRequest && !durableMemoryRequest {
            return false
        }

        switch kind {
        case .profile, .decision, .commitment, .procedure, .handoff, .episode:
            return true
        case .fact:
            return containsAny(
                " \(lower) ",
                needles: [
                    " is ", " are ", " was ", " were ", " uses ", " use ",
                    " has ", " have ", " includes ", " requires ", " supports ",
                    " stores ", " runs ", " ships ", " powers ", " keeps ",
                    " load ", " loads ", " blocks ", " works "
                ]
            )
        }
    }

    private func isQuestionLikeNonMemorySegment(_ lower: String) -> Bool {
        let questionPrefixes = [
            "what ", "when ", "where ", "which ", "who ", "why ", "how ",
            "can ", "could ", "would ", "should ", "do ", "does ", "did ",
            "is ", "are ", "am ", "was ", "were "
        ]
        if questionPrefixes.contains(where: { lower.hasPrefix($0) }) {
            return true
        }

        if lower.hasPrefix("if "),
           containsAny(lower, needles: [" what ", " would ", " could ", " should ", " responsibilities"]) {
            return true
        }

        return containsAny(
            lower,
            needles: [
                "hypothetical question",
                "i am asking a hypothetical",
                "i'm asking a hypothetical"
            ]
        )
    }

    private func inferStatus(forExtractedText text: String, kind: MemoryKind) -> MemoryStatus {
        guard kind == .commitment else { return .active }
        let lower = text.lowercased()
        if containsAny(lower, needles: ["done", "completed", "finished", "resolved", "closed", "shipped"]) {
            return .resolved
        }
        return .active
    }

    private func inferredImportance(for kind: MemoryKind) -> Double {
        switch kind {
        case .decision:
            return 0.85
        case .commitment:
            return 0.80
        case .fact:
            return 0.70
        case .episode:
            return 0.65
        case .profile:
            return 0.62
        case .handoff:
            return 0.60
        case .procedure:
            return 0.55
        }
    }

    private func inferredConfidence(for kind: MemoryKind) -> Double {
        switch kind {
        case .decision, .commitment, .profile:
            return 0.76
        case .procedure, .episode:
            return 0.70
        case .handoff:
            return 0.66
        case .fact:
            return 0.62
        }
    }

    private func inferredTags(forExtractedText text: String) -> [String] {
        let rawTokens = text.lowercased().split { character in
            !character.isLetter && !character.isNumber
        }

        var seen: Set<String> = []
        var tags: [String] = []
        for token in rawTokens {
            let value = String(token)
            guard value.count >= 4 else { continue }
            guard !queryStopWords.contains(value) else { continue }
            if seen.insert(value).inserted {
                tags.append(value)
            }
            if tags.count >= 6 {
                break
            }
        }
        return tags
    }

    private func containsAny(_ text: String, needles: [String]) -> Bool {
        needles.contains(where: text.contains)
    }

    private func containsAnyRecallStatusCue(_ text: String, cues: [String]) -> Bool {
        let normalizedText = " \(normalizedComparisonKey(for: text)) "
        return cues.contains { cue in
            let normalizedCue = normalizedComparisonKey(for: cue)
            guard !normalizedCue.isEmpty else { return false }
            return normalizedText.contains(" \(normalizedCue) ")
        }
    }

    private func prepareCandidateForIngest(_ candidate: MemoryCandidate) -> PreparedMemoryCandidate? {
        let trimmedText = candidate.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return nil }

        let explicitSave = isExplicitSaveSource(candidate.source)
        if !explicitSave {
            if let confidence = candidate.confidence, confidence < minimumAutoWriteConfidence {
                return nil
            }
            guard isCandidateWorthSaving(text: trimmedText, kind: candidate.kind) else {
                return nil
            }
        }

        let createdAt = candidate.createdAt ?? Date()
        let tags = normalizeCandidateTags(candidate, text: trimmedText)
        let facetTags = normalizeFacetTags(candidate.facetTags, text: trimmedText, kind: candidate.kind)
        let entities = normalizeEntities(candidate.entities, text: trimmedText)
        let topics = normalizeTopics(candidate.topics, text: trimmedText, seedTags: tags)
        let canonicalKey = resolveCanonicalKey(
            for: candidate.kind,
            text: trimmedText,
            explicitKey: candidate.canonicalKey,
            entities: entities,
            topics: topics
        )

        return PreparedMemoryCandidate(
            text: trimmedText,
            kind: candidate.kind,
            status: candidate.status,
            importance: candidate.importance,
            confidence: candidate.confidence,
            createdAt: createdAt,
            eventAt: candidate.eventAt,
            source: candidate.source,
            title: inferTitle(content: trimmedText, fallback: candidate.kind.rawValue.capitalized),
            tags: tags,
            facetTags: facetTags,
            entities: entities,
            topics: topics,
            canonicalKey: canonicalKey,
            metadata: candidate.metadata,
            proposedAction: proposedWriteAction(for: candidate)
        )
    }

    private func proposedWriteAction(for candidate: MemoryCandidate) -> MemoryWriteAction {
        switch candidate.kind {
        case .episode:
            return .appendEpisode
        case .commitment where candidate.status != .active:
            return .mergeStatus
        case .commitment:
            return .create
        case .profile:
            return .replaceActive
        case .decision:
            return .supersede
        case .procedure, .handoff, .fact:
            return .create
        }
    }

    private func normalizeCandidateTags(_ candidate: MemoryCandidate, text: String) -> [String] {
        let preferred = candidate.tags.isEmpty ? inferredTags(forExtractedText: text) : candidate.tags
        var seen: Set<String> = []
        var normalized: [String] = []
        normalized.reserveCapacity(preferred.count)

        for value in preferred {
            let cleaned = value
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard !cleaned.isEmpty else { continue }
            guard seen.insert(cleaned).inserted else { continue }
            normalized.append(cleaned)
        }

        return normalized
    }

    private func normalizeFacetTags(
        _ supplied: Set<FacetTag>,
        text: String,
        kind: MemoryKind
    ) -> Set<FacetTag> {
        let preferred = supplied.isEmpty ? inferFacetTags(forExtractedText: text, kind: kind) : supplied
        return Set(preferred.prefix(6))
    }

    private func normalizeEntities(_ supplied: [MemoryEntity], text: String) -> [MemoryEntity] {
        let preferred = supplied.isEmpty ? inferEntities(forExtractedText: text) : supplied
        var normalized: [MemoryEntity] = []
        var seen: Set<String> = []
        normalized.reserveCapacity(min(preferred.count, 8))

        for entity in preferred {
            let value = entity.value.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalizedValue = normalizeEntityValue(entity.normalizedValue.isEmpty ? value : entity.normalizedValue)
            guard !value.isEmpty, !normalizedValue.isEmpty else { continue }
            guard seen.insert(normalizedValue).inserted else { continue }
            normalized.append(
                MemoryEntity(
                    label: entity.label,
                    value: value,
                    normalizedValue: normalizedValue,
                    confidence: entity.confidence
                )
            )
            if normalized.count >= 8 {
                break
            }
        }

        return normalized
    }

    private func normalizeTopics(_ supplied: [String], text: String, seedTags: [String]) -> [String] {
        let preferred = supplied.isEmpty ? inferTopics(forExtractedText: text, seedTags: seedTags) : supplied
        var normalized: [String] = []
        var seen: Set<String> = []
        let maxTopics = 12
        normalized.reserveCapacity(min(preferred.count, maxTopics))

        for topic in preferred {
            let cleaned = normalizeTopicValue(topic)
            guard !cleaned.isEmpty else { continue }
            guard seen.insert(cleaned).inserted else { continue }
            normalized.append(cleaned)
            if normalized.count >= maxTopics {
                break
            }
        }

        return normalized
    }

    private func inferFacetTags(forExtractedText text: String, kind: MemoryKind) -> Set<FacetTag> {
        let lower = text.lowercased()
        var facets: Set<FacetTag> = []

        if containsAny(lower, needles: ["prefer", "preference", "favorite", "likes", "dislikes"]) {
            facets.insert(.preference)
        }
        if containsAny(lower, needles: ["project", "repo", "repository", "branch", "milestone"]) {
            facets.insert(.project)
        }
        if containsAny(lower, needles: ["goal", "objective", "aim"]) {
            facets.insert(.goal)
        }
        if containsAny(lower, needles: ["todo", "task", "action item", "follow up", "next step"]) {
            facets.insert(.task)
        }
        if containsAny(lower, needles: ["tool", "sdk", "framework", "library", "sqlite", "coreml"]) {
            facets.insert(.tool)
        }
        if containsAny(lower, needles: ["where", "location", "office", "remote", "timezone"]) {
            facets.insert(.location)
        }
        if containsAny(lower, needles: ["today", "tomorrow", "deadline", "urgent", "asap"]) {
            facets.insert(.timeSensitive)
        }
        if containsAny(lower, needles: ["constraint", "blocked", "cannot", "must not", "limit"]) {
            facets.insert(.constraint)
        }
        if containsAny(lower, needles: ["habit", "usually", "often", "routine"]) {
            facets.insert(.habit)
        }
        if containsAny(lower, needles: ["lesson", "learned", "takeaway", "retrospective"]) {
            facets.insert(.lesson)
        }
        if containsAny(lower, needles: ["feel", "frustrated", "happy", "worried", "stressed"]) {
            facets.insert(.emotion)
        }
        if containsAny(lower, needles: ["name", "role", "timezone", "i am", "i'm"]) {
            facets.insert(.identitySignal)
        }
        if containsAny(lower, needles: ["maintainer", "owner"]) {
            facets.insert(.identitySignal)
        }
        if containsAny(lower, needles: ["with ", "closely with"]) {
            facets.insert(.relationship)
            facets.insert(.person)
        }

        switch kind {
        case .profile:
            facets.insert(.factAboutUser)
        case .fact:
            facets.insert(.factAboutWorld)
        case .decision:
            facets.insert(.decisionTopic)
        case .commitment:
            facets.insert(.task)
        default:
            break
        }

        return Set(facets.prefix(6))
    }

    private func inferEntities(forExtractedText text: String) -> [MemoryEntity] {
        let tokens = text.split(separator: " ")
        var entities: [MemoryEntity] = []
        var current: [String] = []

        func flush(label: EntityLabel = .other) {
            guard !current.isEmpty else { return }
            let value = current.joined(separator: " ")
            let normalizedValue = normalizeEntityValue(value)
            guard !normalizedValue.isEmpty else {
                current.removeAll(keepingCapacity: true)
                return
            }
            entities.append(
                MemoryEntity(
                    label: label,
                    value: value,
                    normalizedValue: normalizedValue,
                    confidence: 0.6
                )
            )
            current.removeAll(keepingCapacity: true)
        }

        for token in tokens {
            let raw = token.trimmingCharacters(in: CharacterSet.punctuationCharacters)
            guard !raw.isEmpty else {
                flush()
                continue
            }

            if raw.first?.isUppercase == true || raw.contains(".") || raw.contains("/") || raw.contains("+") || raw.contains("-") {
                current.append(raw)
            } else {
                flush()
            }
        }
        flush()

        var normalized: [MemoryEntity] = []
        var seen: Set<String> = []
        for entity in entities {
            let normalizedValue = normalizeEntityValue(entity.normalizedValue)
            guard !normalizedValue.isEmpty else { continue }
            guard seen.insert(normalizedValue).inserted else { continue }
            normalized.append(
                MemoryEntity(
                    label: entity.label,
                    value: entity.value,
                    normalizedValue: normalizedValue,
                    confidence: entity.confidence
                )
            )
            if normalized.count >= 8 {
                break
            }
        }

        return normalized
    }

    private func inferTopics(forExtractedText text: String, seedTags: [String]) -> [String] {
        let maxTopics = 12
        let tokens = text
            .lowercased()
            .split { character in
                !character.isLetter && !character.isNumber && character != "+" && character != "-" && character != "." && character != "/"
            }
            .map(String.init)
            .filter { $0.count >= 3 && !queryStopWords.contains($0) }

        var topics: [String] = []
        var seen: Set<String> = []

        for width in stride(from: min(4, tokens.count), through: 2, by: -1) {
            guard tokens.count >= width else { continue }
            for start in 0...(tokens.count - width) {
                let candidate = normalizeTopicValue(tokens[start..<(start + width)].joined(separator: " "))
                guard !candidate.isEmpty else { continue }
                guard seen.insert(candidate).inserted else { continue }
                topics.append(candidate)
                if topics.count >= maxTopics {
                    return topics
                }
            }
        }

        for tag in seedTags {
            let candidate = normalizeTopicValue(tag)
            guard !candidate.isEmpty else { continue }
            guard seen.insert(candidate).inserted else { continue }
            topics.append(candidate)
            if topics.count >= maxTopics {
                break
            }
        }

        return topics
    }

    private func normalizeEntityValue(_ raw: String) -> String {
        let punctuation = CharacterSet(charactersIn: ",:;!?()[]{}\"'`")
        return raw
            .trimmingCharacters(in: .whitespacesAndNewlines.union(punctuation))
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
            .lowercased()
    }

    private func normalizeTopicValue(_ raw: String) -> String {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .split { character in character.isWhitespace }
            .map(String.init)
            .filter { !$0.isEmpty }
            .prefix(4)
            .joined(separator: " ")
        return normalized
    }

    private func makeStoredMemoryEntity(from entity: MemoryEntity) -> StoredMemoryEntity {
        StoredMemoryEntity(
            label: entity.label.rawValue,
            value: entity.value,
            normalizedValue: entity.normalizedValue,
            confidence: entity.confidence
        )
    }

    private func makeMemoryEntity(from entity: StoredMemoryEntity) -> MemoryEntity? {
        guard let label = EntityLabel.parse(entity.label) else { return nil }
        let normalizedValue = normalizeEntityValue(entity.normalizedValue.isEmpty ? entity.value : entity.normalizedValue)
        guard !normalizedValue.isEmpty else { return nil }
        return MemoryEntity(
            label: label,
            value: entity.value,
            normalizedValue: normalizedValue,
            confidence: entity.confidence
        )
    }

    private func filterStoredMemories(
        _ rows: [StoredMemoryRecord],
        facets: Set<FacetTag>?,
        entityValues: [String]?,
        topics: [String]?
    ) -> [StoredMemoryRecord] {
        let normalizedFacets = facets ?? []
        let normalizedEntities = Set((entityValues ?? []).map(normalizeEntityValue).filter { !$0.isEmpty })
        let normalizedTopics = Set((topics ?? []).map(normalizeTopicValue).filter { !$0.isEmpty })

        return rows.filter { row in
            if !normalizedFacets.isEmpty {
                let rowFacets = Set(row.facetTags.compactMap(FacetTag.parse))
                if rowFacets.isDisjoint(with: normalizedFacets) {
                    return false
                }
            }

            if !normalizedEntities.isEmpty {
                let rowEntities = Set(row.entities.map(\.normalizedValue).map(normalizeEntityValue))
                if rowEntities.isDisjoint(with: normalizedEntities) {
                    return false
                }
            }

            if !normalizedTopics.isEmpty {
                let rowTopics = Set(row.topics.map(normalizeTopicValue))
                if rowTopics.isDisjoint(with: normalizedTopics) {
                    return false
                }
            }

            return true
        }
    }

    private func projectedRetrievalTags(for stored: StoredMemoryRecord) -> [String] {
        var projected = stored.tags
        projected.append(contentsOf: stored.facetTags.map { "facet:\($0)" })
        projected.append(contentsOf: stored.entities.map { "entity:\($0.normalizedValue)" })
        projected.append(contentsOf: stored.topics.map { "topic:\($0)" })
        return projected
    }

    private func isExplicitSaveSource(_ source: String) -> Bool {
        let normalized = source.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == "memory_save" || normalized == "save"
    }

    private func isCandidateWorthSaving(text: String, kind: MemoryKind) -> Bool {
        guard text.count >= 16 else { return false }
        if text.hasSuffix("?") {
            return false
        }

        let informativeTokens = text
            .lowercased()
            .split { character in !character.isLetter && !character.isNumber }
            .map(String.init)
            .filter { $0.count >= 3 && !queryStopWords.contains($0) }

        guard informativeTokens.count >= 3 else { return false }

        switch kind {
        case .handoff:
            return text.count >= 24
        default:
            return true
        }
    }

    private func resolveCanonicalKey(
        for kind: MemoryKind,
        text: String,
        explicitKey: String?,
        entities: [MemoryEntity] = [],
        topics: [String] = []
    ) -> String? {
        if let explicit = normalizeCanonicalKey(explicitKey) {
            return explicit
        }

        let signalSeed = canonicalSignalSeed(text: text, entities: entities, topics: topics)

        switch kind {
        case .profile:
            return profileCanonicalKey(from: text, signalSeed: signalSeed)
                ?? signalSeed.map { "profile:\($0)" }
        case .decision:
            return signalSeed.map { "decision:\($0)" }
        case .commitment:
            return signalSeed.map { "commitment:\($0)" }
        case .procedure:
            return signalSeed.map { "procedure:\($0)" }
        case .handoff:
            return "handoff:primary"
        case .fact, .episode:
            return nil
        }
    }

    private func normalizeCanonicalKey(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let cleaned = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard !cleaned.isEmpty else { return nil }
        return cleaned
    }

    private func profileCanonicalKey(from text: String, signalSeed: String?) -> String? {
        let lower = text.lowercased()

        let explicitAttributes: [(needle: String, key: String)] = [
            ("timezone", "timezone"),
            ("time zone", "timezone"),
            ("editor", "editor"),
            ("favorite", "favorite"),
            ("preference", "preference"),
            ("prefer", "preference"),
            ("my name", "name"),
            ("name is", "name"),
            ("role", "role"),
            ("maintainer", "role"),
            ("owner", "role"),
            ("location", "location"),
            ("email", "email"),
            ("phone", "phone"),
            ("birthday", "birthday")
        ]

        for attribute in explicitAttributes where lower.contains(attribute.needle) {
            if attribute.key == "preference" || attribute.key == "favorite" {
                return signalSeed.map { "profile:\(attribute.key):\($0)" } ?? "profile:\(attribute.key)"
            }
            return "profile:\(attribute.key)"
        }

        return nil
    }

    private func canonicalSignalSeed(
        text: String,
        entities: [MemoryEntity],
        topics: [String],
        maxTokens: Int = 6
    ) -> String? {
        var values: [String] = []
        values.reserveCapacity(entities.count + topics.count + maxTokens)

        for entity in entities {
            let normalized = normalizeCanonicalKey(entity.normalizedValue) ?? normalizeCanonicalKey(entity.value)
            if let normalized, !isGenericCanonicalSignal(normalized) {
                values.append(normalized)
            }
        }

        for topic in topics {
            if let normalized = normalizeCanonicalKey(topic), !isGenericCanonicalSignal(normalized) {
                values.append(normalized)
            }
        }

        if values.isEmpty, let seed = candidateKeySeed(from: text, maxTokens: maxTokens) {
            values.append(seed)
        }

        var seen: Set<String> = []
        let joined = values
            .flatMap { value in
                value.split { character in !character.isLetter && !character.isNumber }
                    .map(String.init)
            }
            .filter { token in
                token.count >= 3
                    && !queryStopWords.contains(token)
                    && !canonicalStopWords.contains(token)
                    && seen.insert(token).inserted
            }
            .prefix(maxTokens)
            .joined(separator: "-")

        return joined.isEmpty ? nil : joined
    }

    private func isGenericCanonicalSignal(_ value: String) -> Bool {
        let generic: Set<String> = [
            "todo", "task", "action", "item", "decision", "profile", "fact",
            "commitment", "memory", "memories", "project", "repo", "repository"
        ]
        let tokens = value
            .split { character in !character.isLetter && !character.isNumber }
            .map(String.init)
        return !tokens.isEmpty && tokens.allSatisfy { generic.contains($0) || queryStopWords.contains($0) }
    }

    private func candidateKeySeed(from text: String, maxTokens: Int = 6) -> String? {
        let tokens = text
            .lowercased()
            .split { character in !character.isLetter && !character.isNumber }
            .map(String.init)
            .filter {
                $0.count >= 3
                    && !queryStopWords.contains($0)
                    && !canonicalStopWords.contains($0)
            }

        guard !tokens.isEmpty else { return nil }
        return tokens.prefix(maxTokens).joined(separator: "-")
    }

    private func ingestPreparedCandidate(_ candidate: PreparedMemoryCandidate) async throws -> IngestConsolidationResult {
        switch candidate.kind {
        case .fact:
            if let duplicate = try await storage.findDuplicateStoredMemory(
                kind: candidate.kind.rawValue,
                text: candidate.text
            ) {
                return IngestConsolidationResult(
                    primaryMemoryID: duplicate.id,
                    impactedMemoryIDs: [duplicate.id],
                    action: .dedupe
                )
            }
            if let duplicate = try await findEquivalentFact(candidate) {
                return IngestConsolidationResult(
                    primaryMemoryID: duplicate.id,
                    impactedMemoryIDs: [duplicate.id],
                    action: .dedupe
                )
            }
            return try await insertPreparedMemory(candidate, action: .create)
        case .episode:
            return try await insertPreparedMemory(candidate, action: .appendEpisode)
        case .profile, .decision, .handoff:
            return try await replaceActiveMemory(candidate)
        case .procedure:
            if candidate.canonicalKey == nil {
                return try await insertPreparedMemory(candidate, action: .create)
            }
            return try await replaceActiveMemory(candidate)
        case .commitment:
            return try await mergeCommitment(candidate)
        }
    }

    private func findEquivalentFact(_ candidate: PreparedMemoryCandidate) async throws -> StoredMemoryRecord? {
        guard candidate.kind == .fact else { return nil }
        let candidateKey = normalizedSemanticKey(for: candidate.text)
        guard !candidateKey.isEmpty else { return nil }

        let candidates = try await storage.listStoredMemories(
            limit: 200,
            sort: .recent,
            kinds: [MemoryKind.fact.rawValue],
            statuses: [MemoryStatus.active.rawValue]
        )
        return candidates.first { existing in
            normalizedSemanticKey(for: existing.text) == candidateKey
        }
    }

    private func findRelatedCanonicalMemory(
        _ candidate: PreparedMemoryCandidate,
        statuses: Set<MemoryStatus>
    ) async throws -> StoredMemoryRecord? {
        guard candidate.kind == .commitment || candidate.kind == .decision else { return nil }
        let candidateTokens = canonicalMatchTokens(
            text: candidate.text,
            canonicalKey: candidate.canonicalKey,
            topics: candidate.topics,
            tags: candidate.tags
        )
        guard candidateTokens.count >= 2 else { return nil }

        let candidates = try await storage.listStoredMemories(
            limit: 200,
            sort: .recent,
            kinds: [candidate.kind.rawValue],
            statuses: Set(statuses.map(\.rawValue))
        )

        var best: (record: StoredMemoryRecord, score: Double)?
        for existing in candidates {
            let existingTokens = canonicalMatchTokens(
                text: existing.text,
                canonicalKey: existing.canonicalKey,
                topics: existing.topics,
                tags: existing.tags
            )
            guard existingTokens.count >= 2 else { continue }

            let overlap = candidateTokens.intersection(existingTokens)
            guard overlap.count >= 2 else { continue }

            let coverage = Double(overlap.count) / Double(min(candidateTokens.count, existingTokens.count))
            let score = coverage + (existing.status == candidate.status.rawValue ? 0.05 : 0)
            guard coverage >= 0.50 else { continue }

            if best == nil || score > best!.score {
                best = (existing, score)
            }
        }

        return best?.record
    }

    private func canonicalMatchTokens(
        text: String,
        canonicalKey: String?,
        topics: [String],
        tags: [String]
    ) -> Set<String> {
        let raw = ([text, canonicalKey ?? ""] + topics + tags).joined(separator: " ")
        let tokens = raw
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .map(normalizedSemanticToken)
            .filter { token in
                token.count >= 3
                    && !queryStopWords.contains(token)
                    && !canonicalStopWords.contains(token)
                    && !canonicalMatchStopWords.contains(token)
            }
        return Set(tokens)
    }

    private func insertPreparedMemory(
        _ candidate: PreparedMemoryCandidate,
        supersedesID: String? = nil,
        action: MemoryWriteAction? = nil
    ) async throws -> IngestConsolidationResult {
        let memoryID = UUID().uuidString.lowercased()
        try await storage.insertStoredMemory(
            StoredMemoryInput(
                id: memoryID,
                title: candidate.title,
                kind: candidate.kind.rawValue,
                status: candidate.status.rawValue,
                canonicalKey: candidate.canonicalKey,
                text: candidate.text,
                tags: candidate.tags,
                facetTags: candidate.facetTags.map(\.rawValue).sorted(),
                entities: candidate.entities.map(makeStoredMemoryEntity(from:)),
                topics: candidate.topics,
                importance: candidate.importance,
                confidence: candidate.confidence,
                source: candidate.source,
                createdAt: candidate.createdAt,
                eventAt: candidate.eventAt,
                updatedAt: candidate.createdAt,
                supersedesID: supersedesID,
                supersededByID: nil,
                metadata: candidate.metadata
            )
        )
        return IngestConsolidationResult(
            primaryMemoryID: memoryID,
            impactedMemoryIDs: [memoryID],
            action: action ?? candidate.proposedAction ?? .create
        )
    }

    private func replaceActiveMemory(_ candidate: PreparedMemoryCandidate) async throws -> IngestConsolidationResult {
        guard let canonicalKey = candidate.canonicalKey else {
            return try await insertPreparedMemory(candidate, action: .create)
        }

        var existing = try await storage.findStoredMemory(
            kind: candidate.kind.rawValue,
            canonicalKey: canonicalKey,
            statuses: [MemoryStatus.active.rawValue]
        )
        let matchedByRelatedKey = existing == nil
        if existing == nil {
            existing = try await findRelatedCanonicalMemory(candidate, statuses: [.active])
        }
        guard let existing else {
            return try await insertPreparedMemory(candidate, action: .create)
        }

        if normalizedComparisonKey(for: existing.text) == normalizedComparisonKey(for: candidate.text),
           existing.status == candidate.status.rawValue {
            return IngestConsolidationResult(
                primaryMemoryID: existing.id,
                impactedMemoryIDs: [existing.id],
                action: .dedupe
            )
        }

        var replacement = candidate
        if matchedByRelatedKey, let existingCanonicalKey = existing.canonicalKey {
            replacement.canonicalKey = existingCanonicalKey
        }

        let action: MemoryWriteAction = candidate.kind == .decision ? .supersede : .replaceActive
        let inserted = try await insertPreparedMemory(replacement, supersedesID: existing.id, action: action)
        try await storage.updateStoredMemoryStatus(
            id: existing.id,
            status: MemoryStatus.superseded.rawValue,
            supersededByID: inserted.primaryMemoryID,
            updatedAt: candidate.createdAt
        )

        return IngestConsolidationResult(
            primaryMemoryID: inserted.primaryMemoryID,
            impactedMemoryIDs: inserted.impactedMemoryIDs.union([existing.id]),
            action: action
        )
    }

    private func mergeCommitment(_ candidate: PreparedMemoryCandidate) async throws -> IngestConsolidationResult {
        guard let canonicalKey = candidate.canonicalKey else {
            return try await insertPreparedMemory(candidate, action: .create)
        }

        var existing = try await storage.findStoredMemory(
            kind: candidate.kind.rawValue,
            canonicalKey: canonicalKey,
            statuses: [MemoryStatus.active.rawValue, MemoryStatus.resolved.rawValue]
        )
        if existing == nil {
            existing = try await findRelatedCanonicalMemory(candidate, statuses: [.active, .resolved])
        }
        guard let existing else {
            return try await insertPreparedMemory(candidate, action: .create)
        }

        if candidate.status != .active {
            if existing.status != candidate.status.rawValue {
                try await storage.updateStoredMemoryStatus(
                    id: existing.id,
                    status: candidate.status.rawValue,
                    supersededByID: existing.supersededByID,
                    updatedAt: candidate.createdAt
                )
            }
            return IngestConsolidationResult(
                primaryMemoryID: existing.id,
                impactedMemoryIDs: [existing.id],
                action: existing.status == candidate.status.rawValue ? .dedupe : .mergeStatus
            )
        }

        if normalizedComparisonKey(for: existing.text) == normalizedComparisonKey(for: candidate.text),
           existing.status == candidate.status.rawValue {
            return IngestConsolidationResult(
                primaryMemoryID: existing.id,
                impactedMemoryIDs: [existing.id],
                action: .dedupe
            )
        }

        if existing.status == MemoryStatus.active.rawValue {
            let inserted = try await insertPreparedMemory(candidate, supersedesID: existing.id, action: .supersede)
            try await storage.updateStoredMemoryStatus(
                id: existing.id,
                status: MemoryStatus.superseded.rawValue,
                supersededByID: inserted.primaryMemoryID,
                updatedAt: candidate.createdAt
            )
            return IngestConsolidationResult(
                primaryMemoryID: inserted.primaryMemoryID,
                impactedMemoryIDs: inserted.impactedMemoryIDs.union([existing.id]),
                action: .supersede
            )
        }

        return try await insertPreparedMemory(candidate, supersedesID: existing.id, action: .create)
    }

    private func materializeStoredMemory(id: String) async throws {
        guard let stored = try await storage.fetchStoredMemory(id: id) else { return }
        try await materializeStoredMemory(stored)
    }

    private func rematerializeStoredMemories() async throws {
        let storedMemories = try await storage.listStoredMemories(
            limit: Int.max,
            sort: .recent,
            kinds: nil,
            statuses: nil
        )

        for stored in storedMemories {
            try await materializeStoredMemory(stored)
        }
    }

    private func materializeStoredMemory(_ stored: StoredMemoryRecord) async throws {
        let payload = try await makeDerivedMemoryPayload(from: stored)
        try await storage.replaceDocument(payload)
    }

    private func makeDerivedMemoryPayload(from stored: StoredMemoryRecord) async throws -> StoredDocumentInput {
        guard let kind = MemoryKind.parse(stored.kind) else {
            throw MemoryError.ingestion("Unable to materialize memory with unknown kind '\(stored.kind)'")
        }
        guard let status = MemoryStatus.parse(stored.status) else {
            throw MemoryError.ingestion("Unable to materialize memory with unknown status '\(stored.status)'")
        }

        let embedding: [Float]
        do {
            embedding = try await configuration.embeddingProvider.embed(
                text: stored.text,
                format: .document(title: stored.title)
            )
        } catch {
            throw MemoryError.embedding("Failed to embed memory for ingest: \(error.localizedDescription)")
        }

        let tags = normalizeIngestTags(projectedRetrievalTags(for: stored))
        let createdAt = stored.createdAt
        let modifiedAt = max(stored.updatedAt, stored.createdAt)

        return StoredDocumentInput(
            path: derivedMemoryPath(for: stored.id),
            title: stored.title ?? inferTitle(content: stored.text, fallback: kind.rawValue.capitalized),
            modifiedAt: modifiedAt,
            checksum: checksum(stored.text),
            memoryID: stored.id,
            memoryKind: kind.rawValue,
            memoryStatus: status.rawValue,
            memoryCanonicalKey: stored.canonicalKey,
            memoryType: "memory",
            memoryTypeSource: "system",
            memoryTypeConfidence: nil,
            chunks: [
                StoredChunkInput(
                    ordinal: 0,
                    content: stored.text,
                    tokenCount: configuration.tokenizer.tokenize(stored.text).count,
                    embedding: embedding,
                    norm: l2Norm(embedding),
                    memoryTypeOverride: "memory",
                    memoryTypeOverrideSource: "system",
                    memoryTypeOverrideConfidence: nil,
                    contentTags: tags,
                    memoryKind: kind.rawValue,
                    importance: stored.importance,
                    accessCount: stored.accessCount,
                    lastAccessedAt: stored.lastAccessedAt,
                    source: stored.source,
                    createdAt: createdAt
                ),
            ]
        )
    }

    private func resolveKindsFilter(for mode: RecallMode, requestedKinds: Set<MemoryKind>?) -> Set<MemoryKind>? {
        switch mode {
        case .kind(let modeKind):
            if let requestedKinds, !requestedKinds.contains(modeKind) {
                return []
            }
            return [modeKind]
        default:
            return requestedKinds
        }
    }

    private func intersectKinds(_ lhs: Set<MemoryKind>?, _ rhs: Set<MemoryKind>?) -> Set<MemoryKind>? {
        switch (lhs, rhs) {
        case let (.some(lhs), .some(rhs)):
            return lhs.intersection(rhs)
        case let (.some(lhs), .none):
            return lhs
        case let (.none, .some(rhs)):
            return rhs
        case (.none, .none):
            return nil
        }
    }

    private func intersectFacets(_ lhs: Set<FacetTag>?, _ rhs: Set<FacetTag>?) -> Set<FacetTag>? {
        switch (lhs, rhs) {
        case let (.some(lhs), .some(rhs)):
            return lhs.intersection(rhs)
        case let (.some(lhs), .none):
            return lhs
        case let (.none, .some(rhs)):
            return rhs
        case (.none, .none):
            return nil
        }
    }

    private func mergeFilterValues(_ lhs: [String]?, _ rhs: [String]) -> [String]? {
        var seen: Set<String> = []
        var merged: [String] = []
        for value in (lhs ?? []) + rhs {
            let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalized.isEmpty else { continue }
            let key = normalizedComparisonKey(for: normalized)
            guard seen.insert(key).inserted else { continue }
            merged.append(normalized)
        }
        return merged.isEmpty ? nil : merged
    }

    private func resolveMemoryChunkFilter(
        kinds: Set<MemoryKind>?,
        statuses: Set<MemoryStatus>?,
        facets: Set<FacetTag>?,
        entityValues: [String]?,
        topics: [String]?
    ) async throws -> Set<Int64>? {
        let hasEntityFilter = !(entityValues ?? []).isEmpty
        let hasTopicFilter = !(topics ?? []).isEmpty
        guard kinds != nil || statuses != nil || facets != nil || hasEntityFilter || hasTopicFilter else {
            return nil
        }

        if facets == nil, !hasEntityFilter, !hasTopicFilter {
            let chunkIDs = try await storage.fetchMemoryChunkIDs(
                kinds: kinds.map { Set($0.map(\.rawValue)) },
                statuses: statuses.map { Set($0.map(\.rawValue)) }
            )
            return Set(chunkIDs)
        }

        let rows = try await storage.listStoredMemories(
            limit: Int.max,
            sort: .recent,
            kinds: kinds.map { Set($0.map(\.rawValue)) },
            statuses: statuses.map { Set($0.map(\.rawValue)) }
        )
        let filtered = filterStoredMemories(
            rows,
            facets: facets,
            entityValues: entityValues,
            topics: topics
        )
        return Set(filtered.compactMap(\.chunkID))
    }

    private func combineAllowedChunkIDs(_ lhs: Set<Int64>?, _ rhs: Set<Int64>?) -> Set<Int64>? {
        switch (lhs, rhs) {
        case let (.some(lhs), .some(rhs)):
            return lhs.intersection(rhs)
        case let (.some(lhs), .none):
            return lhs
        case let (.none, .some(rhs)):
            return rhs
        case (.none, .none):
            return nil
        }
    }

    private func resolveMemoryKind(from metadata: StoredChunkMetadata) -> MemoryKind? {
        if let raw = metadata.memoryKind, let parsed = MemoryKind.parse(raw) {
            return parsed
        }
        if let parsed = MemoryKind.parse(metadata.memoryKindFallback) {
            return parsed
        }
        return nil
    }

    private func resolveMemoryStatus(raw: String?, hasMemoryID: Bool) -> MemoryStatus? {
        if let raw, let parsed = MemoryStatus.parse(raw) {
            return parsed
        }
        return hasMemoryID ? .active : nil
    }

    private func collectDocumentURLs(from request: IndexingRequest) throws -> [URL] {
        var collected: Set<URL> = []

        for root in request.roots {
            let standardized = root.standardizedFileURL
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: standardized.path, isDirectory: &isDirectory) else {
                throw MemoryError.ingestion("Path does not exist: \(standardized.path)")
            }

            if isDirectory.boolValue {
                let urls = try walkDirectory(
                    at: standardized,
                    includeHiddenFiles: request.includeHiddenFiles,
                    followSymlinks: request.followSymlinks
                )
                for url in urls {
                    collected.insert(url)
                }
            } else {
                collected.insert(standardized)
            }
        }

        return collected.sorted { $0.path < $1.path }
    }

    private func walkDirectory(
        at root: URL,
        includeHiddenFiles: Bool,
        followSymlinks: Bool
    ) throws -> [URL] {
        let resourceKeys: Set<URLResourceKey> = [.isRegularFileKey, .isDirectoryKey, .isHiddenKey, .isSymbolicLinkKey]
        let options: FileManager.DirectoryEnumerationOptions = includeHiddenFiles ? [] : [.skipsHiddenFiles]
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: Array(resourceKeys),
            options: options,
            errorHandler: { _, _ in true }
        ) else {
            return []
        }

        var urls: [URL] = []
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: resourceKeys)

            if !includeHiddenFiles, values.isHidden == true {
                if values.isDirectory == true {
                    enumerator.skipDescendants()
                }
                continue
            }

            // Keep recursion enabled for normal directories, but skip symlinks unless explicitly requested.
            if !followSymlinks, values.isSymbolicLink == true {
                if values.isDirectory == true {
                    enumerator.skipDescendants()
                }
                continue
            }

            if values.isRegularFile == true, isSupportedFile(url: url) {
                urls.append(url.standardizedFileURL)
            }
        }

        return urls
    }

    private func isSupportedFile(url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        if ext.isEmpty {
            return false
        }
        return configuration.supportedFileExtensions.contains(ext)
    }

    private func buildDocumentPayload(
        for url: URL,
        events: IndexingEventHandler? = nil
    ) async throws -> StoredDocumentInput? {
        guard isSupportedFile(url: url) else { return nil }

        let content: String
        do {
            content = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw MemoryError.ingestion("Unable to read UTF-8 file at \(url.path): \(error.localizedDescription)")
        }

        let kind = inferDocumentKind(for: url)
        let chunkingStart = DispatchTime.now().uptimeNanoseconds
        let chunks = configuration.chunker.chunk(text: content, kind: kind, sourceURL: url)
        events?(.stageTiming(path: url.path, stage: .chunking, durationMs: elapsedMilliseconds(since: chunkingStart)))
        guard !chunks.isEmpty else { return nil }

        let documentTitle = inferTitle(content: content, fallback: url.deletingPathExtension().lastPathComponent)
        let embeddings: [[Float]]
        let embeddingStart = DispatchTime.now().uptimeNanoseconds
        do {
            embeddings = try await configuration.embeddingProvider.embed(
                texts: chunks.map(\.content),
                format: .document(title: documentTitle)
            )
        } catch {
            throw MemoryError.embedding("Failed to embed chunks for \(url.path): \(error.localizedDescription)")
        }
        events?(.stageTiming(path: url.path, stage: .embedding, durationMs: elapsedMilliseconds(since: embeddingStart)))

        guard embeddings.count == chunks.count else {
            throw MemoryError.embedding("Embedding provider returned \(embeddings.count) vectors for \(chunks.count) chunks")
        }

        let taggingStart = DispatchTime.now().uptimeNanoseconds
        let chunkTags = await resolveChunkContentTags(chunks: chunks, kind: kind, sourceURL: url, events: events)
        events?(.stageTiming(path: url.path, stage: .tagging, durationMs: elapsedMilliseconds(since: taggingStart)))
        let chunkInputs: [StoredChunkInput] = zip(zip(chunks, embeddings), chunkTags).map { element in
            let (pair, contentTags) = element
            let (chunk, vector) = pair
            return StoredChunkInput(
                ordinal: chunk.ordinal,
                content: chunk.content,
                tokenCount: chunk.tokenCount,
                embedding: vector,
                norm: l2Norm(vector),
                memoryTypeOverride: nil,
                memoryTypeOverrideSource: nil,
                memoryTypeOverrideConfidence: nil,
                contentTags: contentTags
            )
        }

        let metadata = try fileManager.attributesOfItem(atPath: url.path)
        let modifiedAt = (metadata[.modificationDate] as? Date) ?? Date()

        return StoredDocumentInput(
            path: url.path,
            title: documentTitle,
            modifiedAt: modifiedAt,
            checksum: checksum(content),
            memoryType: "document",
            memoryTypeSource: "system",
            memoryTypeConfidence: nil,
            chunks: chunkInputs
        )
    }

    private func inferDocumentKind(for url: URL) -> DocumentKind {
        let ext = url.pathExtension.lowercased()
        if markdownExtensions.contains(ext) {
            return .markdown
        }
        if codeExtensions.contains(ext) {
            return .code
        }
        return .plainText
    }

    private func inferTitle(content: String, fallback: String) -> String {
        for line in content.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("#") {
                return trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "# "))
            }
        }
        return fallback
    }

    private func checksum(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func resolveChunkContentTags(
        chunks: [Chunk],
        kind: DocumentKind,
        sourceURL: URL,
        events: IndexingEventHandler?
    ) async -> [[StoredChunkTag]] {
        guard let contentTagger = configuration.contentTagger else {
            return Array(repeating: [], count: chunks.count)
        }

        var collected: [[StoredChunkTag]] = []
        collected.reserveCapacity(chunks.count)

        for chunk in chunks {
            do {
                let generated = try await contentTagger.tag(
                    text: chunk.content,
                    kind: kind,
                    sourceURL: sourceURL
                )
                let normalized = normalizeContentTags(generated, maxCount: 12)
                collected.append(
                    normalized.map { tag in
                        StoredChunkTag(name: tag.name, confidence: tag.confidence)
                    }
                )
            } catch {
                events?(
                    .providerFailure(
                        path: sourceURL.path,
                        stage: .tagging,
                        provider: contentTagger.identifier,
                        message: error.localizedDescription
                    )
                )
                collected.append([])
            }
        }

        return collected
    }

    private func resolveQueryContentTags(
        queryText: String,
        queryAnalysis: QueryAnalysis,
        events: SearchEventHandler?
    ) async -> [ContentTag] {
        guard let contentTagger = configuration.contentTagger else { return [] }

        do {
            let generated = try await contentTagger.tag(
                text: queryText,
                kind: .plainText,
                sourceURL: nil
            )
            var normalized = normalizeContentTags(generated, maxCount: 8)
            let prefixTags = queryAnalysis.facetHints.map {
                ContentTag(name: "facet:\($0.tag.rawValue)", confidence: min(1, max(0, $0.confidence)))
            } + queryAnalysis.entities.map {
                ContentTag(name: "entity:\($0.normalizedValue)", confidence: $0.confidence ?? 0.8)
            } + queryAnalysis.topics.map {
                ContentTag(name: "topic:\(normalizeTopicValue($0))", confidence: 0.7)
            }
            normalized.append(contentsOf: prefixTags)
            return normalizeContentTags(normalized, maxCount: 16)
        } catch {
            events?(
                .providerFailure(
                    stage: .fusion,
                    provider: contentTagger.identifier,
                    message: error.localizedDescription
                )
            )
            return []
        }
    }

    private func queryMatchSignals(from analysis: QueryAnalysis) -> QueryMatchSignals {
        QueryMatchSignals(
            facets: Set(analysis.facetHints.map(\.tag)),
            entityValues: Set(analysis.entities.map(\.normalizedValue).map(normalizeEntityValue).filter { !$0.isEmpty }),
            topics: Set(analysis.topics.map(normalizeTopicValue).filter { !$0.isEmpty }),
            temporalIntent: .any,
            preferredStatuses: []
        )
    }

    private func queryMatchSignals(from analysis: QueryAnalysis, plan: RecallPlan?) -> QueryMatchSignals {
        var signals = queryMatchSignals(from: analysis)
        if let plan {
            signals.temporalIntent = plan.temporalIntent
            signals.preferredStatuses = plan.statuses ?? []
        }
        return signals
    }

    private func resolveRecallPlan(
        query: String,
        conversationContext: [ConversationMessage],
        features: RecallFeatures,
        events: SearchEventHandler?
    ) async throws -> RecallPlan {
        let fallback = heuristicRecallPlan(query: query)
        guard features.contains(.planner), let planner = configuration.recallPlanner else {
            return fallback
        }

        do {
            guard let planned = try await planner.plan(
                query: query,
                conversationContext: conversationContext,
                features: features
            ) else {
                return fallback
            }
            return mergeRecallPlans(primary: planned, fallback: fallback)
        } catch {
            events?(
                .providerFailure(
                    stage: .analysis,
                    provider: planner.identifier,
                    message: error.localizedDescription
                )
            )
            return fallback
        }
    }

    private func mergeRecallPlans(primary: RecallPlan, fallback: RecallPlan) -> RecallPlan {
        let query = primary.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? fallback.query
            : primary.query
        return RecallPlan(
            query: query,
            lexicalQueries: primary.lexicalQueries.isEmpty ? fallback.lexicalQueries : primary.lexicalQueries,
            semanticQueries: primary.semanticQueries.isEmpty ? fallback.semanticQueries : primary.semanticQueries,
            hypotheticalDocuments: primary.hypotheticalDocuments.isEmpty ? fallback.hypotheticalDocuments : primary.hypotheticalDocuments,
            kinds: primary.kinds ?? fallback.kinds,
            statuses: primary.statuses ?? fallback.statuses,
            facets: primary.facets ?? fallback.facets,
            entityValues: primary.entityValues.isEmpty ? fallback.entityValues : primary.entityValues,
            topics: primary.topics.isEmpty ? fallback.topics : primary.topics,
            temporalIntent: primary.temporalIntent == .any ? fallback.temporalIntent : primary.temporalIntent,
            semanticCandidateLimit: primary.semanticCandidateLimit ?? fallback.semanticCandidateLimit,
            lexicalCandidateLimit: primary.lexicalCandidateLimit ?? fallback.lexicalCandidateLimit,
            rerankLimit: primary.rerankLimit ?? fallback.rerankLimit
        )
    }

    private func heuristicRecallPlan(query: String) -> RecallPlan {
        let lower = query.lowercased()
        let analysis = configuration.queryAnalyzer?.analyze(query: query) ?? heuristicQueryAnalysis(for: query)

        let statuses: Set<MemoryStatus>?
        if containsAnyRecallStatusCue(
            lower,
            cues: ["historical", "previous", "superseded", "archived", "old memory", "old memories", "old records"]
        ) {
            statuses = Set(MemoryStatus.allCases)
        } else if containsAnyRecallStatusCue(lower, cues: ["done", "completed", "resolved", "what happened"]) {
            statuses = [.active, .resolved]
        } else {
            statuses = nil
        }

        let temporalIntent: RecallTemporalIntent
        if containsAny(lower, needles: ["how many", "total", "count"]) {
            temporalIntent = .count
        } else if containsAny(lower, needles: ["most recent", "latest", "last "]) {
            temporalIntent = .mostRecent
        } else if isTimeAnchoredQuery(query) {
            temporalIntent = .timeAnchored
        } else {
            temporalIntent = .any
        }

        return RecallPlan(
            query: query,
            lexicalQueries: analysis.keyTerms.isEmpty ? [] : [analysis.keyTerms.joined(separator: " ")],
            statuses: statuses,
            facets: nil,
            entityValues: [],
            topics: [],
            temporalIntent: temporalIntent,
            semanticCandidateLimit: temporalIntent == .count ? configuration.semanticCandidateLimit + 150 : nil,
            lexicalCandidateLimit: temporalIntent == .count ? configuration.lexicalCandidateLimit + 150 : nil,
            rerankLimit: temporalIntent == .count || temporalIntent == .timeAnchored ? 60 : nil
        )
    }

    private func heuristicQueryAnalysis(for query: String) -> QueryAnalysis {
        let tags = inferredTags(forExtractedText: query)
        return QueryAnalysis(
            entities: inferEntities(forExtractedText: query),
            keyTerms: tags,
            facetHints: makeFacetHints(
                from: inferFacetTags(forExtractedText: query, kind: .fact),
                confidence: 0.72,
                isExplicit: false
            ),
            topics: inferTopics(forExtractedText: query, seedTags: tags),
            isHowToQuery: query.lowercased().hasPrefix("how to") || query.lowercased().hasPrefix("how do")
        )
    }

    private func makeFacetHints(
        from tags: Set<FacetTag>,
        confidence: Double,
        isExplicit: Bool
    ) -> [FacetHint] {
        tags
            .sorted { $0.rawValue < $1.rawValue }
            .map {
                FacetHint(
                    tag: $0,
                    confidence: confidence,
                    isExplicit: isExplicit
                )
            }
    }

    private func normalizeFacetHints(_ hints: [FacetHint], maxCount: Int) -> [FacetHint] {
        guard maxCount > 0 else { return [] }

        var deduped: [FacetTag: FacetHint] = [:]
        for hint in hints {
            let candidate = FacetHint(
                tag: hint.tag,
                confidence: min(1, max(0, hint.confidence)),
                isExplicit: hint.isExplicit
            )
            if let existing = deduped[candidate.tag] {
                if candidate.confidence > existing.confidence
                    || (candidate.confidence == existing.confidence && candidate.isExplicit && !existing.isExplicit) {
                    deduped[candidate.tag] = candidate
                }
            } else {
                deduped[candidate.tag] = candidate
            }
        }

        return deduped.values
            .sorted { lhs, rhs in
                if lhs.confidence == rhs.confidence {
                    if lhs.isExplicit == rhs.isExplicit {
                        return lhs.tag.rawValue < rhs.tag.rawValue
                    }
                    return lhs.isExplicit && !rhs.isExplicit
                }
                return lhs.confidence > rhs.confidence
            }
            .prefix(maxCount)
            .map { $0 }
    }

    private func normalizeMemoryEntities(_ entities: [MemoryEntity], maxCount: Int) -> [MemoryEntity] {
        guard maxCount > 0 else { return [] }

        var deduped: [String: MemoryEntity] = [:]
        for entity in entities {
            let value = entity.value.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalizedValue = normalizeEntityValue(entity.normalizedValue.isEmpty ? value : entity.normalizedValue)
            guard !value.isEmpty, !normalizedValue.isEmpty else { continue }

            let candidate = MemoryEntity(
                label: entity.label,
                value: value,
                normalizedValue: normalizedValue,
                confidence: entity.confidence
            )

            if let existing = deduped[normalizedValue] {
                let existingConfidence = existing.confidence ?? 0
                let candidateConfidence = candidate.confidence ?? 0
                if candidateConfidence > existingConfidence {
                    deduped[normalizedValue] = candidate
                }
            } else {
                deduped[normalizedValue] = candidate
            }
        }

        return deduped.values
            .sorted { lhs, rhs in
                let left = lhs.confidence ?? 0
                let right = rhs.confidence ?? 0
                if left == right {
                    return lhs.normalizedValue < rhs.normalizedValue
                }
                return left > right
            }
            .prefix(maxCount)
            .map { $0 }
    }

    private func normalizeTopicValues(_ topics: [String], maxCount: Int) -> [String] {
        guard maxCount > 0 else { return [] }

        var normalized: [String] = []
        var seen: Set<String> = []
        for topic in topics {
            let candidate = normalizeTopicValue(topic)
            guard !candidate.isEmpty else { continue }
            guard seen.insert(candidate).inserted else { continue }
            normalized.append(candidate)
            if normalized.count >= maxCount {
                break
            }
        }
        return normalized
    }

    private func normalizeContentTags(_ tags: [ContentTag], maxCount: Int) -> [ContentTag] {
        guard maxCount > 0 else { return [] }

        var deduped: [String: ContentTag] = [:]
        for tag in tags {
            let normalizedName = normalizeTagName(tag.name)
            guard !normalizedName.isEmpty else { continue }

            let clamped = min(1, max(0, tag.confidence))
            guard clamped.isFinite else { continue }

            let key = normalizedComparisonKey(for: normalizedName)
            let candidate = ContentTag(name: normalizedName, confidence: clamped)
            if let existing = deduped[key], existing.confidence >= candidate.confidence {
                continue
            }
            deduped[key] = candidate
        }

        return deduped.values
            .sorted { lhs, rhs in
                if lhs.confidence == rhs.confidence {
                    return lhs.name < rhs.name
                }
                return lhs.confidence > rhs.confidence
            }
            .prefix(maxCount)
            .map { $0 }
    }

    private func normalizeTagName(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        let collapsed = trimmed.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        return collapsed.lowercased()
    }

    private func semanticSearch(
        queryVector: [Float],
        limit: Int,
        allowedChunkIDs: Set<Int64>?,
        allowedMemoryTypes: Set<String>?
    ) async throws -> [LexicalHit] {
        guard limit > 0 else { return [] }
        do {
            return try await storage.vectorSearch(
                queryVector: queryVector,
                limit: limit,
                allowedChunkIDs: allowedChunkIDs,
                allowedMemoryTypes: allowedMemoryTypes
            )
        } catch {
            throw normalizeError(error)
        }
    }

    private func l2Norm(_ vector: [Float]) -> Double {
        guard !vector.isEmpty else { return 0 }
        let sum = vDSP.sum(vDSP.multiply(vector, vector))
        return Double(sqrt(sum))
    }

    private func elapsedMilliseconds(since startNanoseconds: UInt64) -> Double {
        let delta = DispatchTime.now().uptimeNanoseconds - startNanoseconds
        return Double(delta) / 1_000_000.0
    }

    private func accumulateRRF(
        for hits: [LexicalHit],
        weight: Double,
        into scores: inout [Int64: Double]
    ) {
        guard weight > 0 else { return }
        for (index, hit) in hits.enumerated() {
            let rank = Double(index + 1)
            let base = 1.0 / (configuration.fusionK + rank)
            var contribution = weight * base
            if index == 0 {
                contribution += weight * 0.0025
            } else if index <= 2 {
                contribution += weight * 0.001
            }
            scores[hit.chunkID, default: 0] += contribution
        }
    }

    private func accumulateScoredRRF(
        for hits: [LexicalHit],
        weight: Double,
        into scores: inout [Int64: Double]
    ) {
        guard weight > 0 else { return }
        for (index, hit) in hits.enumerated() {
            let rank = Double(index + 1)
            let boundedScore = min(1, max(0, hit.score))
            let scoreScale = 0.75 + (0.5 * boundedScore)
            var contribution = weight * scoreScale / (configuration.fusionK + rank)
            if index == 0 {
                contribution += weight * 0.0015
            }
            scores[hit.chunkID, default: 0] += contribution
        }
    }

    private func accumulatePrimaryBranchPreservation(
        for hits: [LexicalHit],
        rankLimit: Int,
        baseContribution: Double,
        weight: Double,
        into scores: inout [Int64: Double]
    ) {
        guard rankLimit > 0, weight > 0, baseContribution > 0 else { return }
        for (index, hit) in hits.prefix(rankLimit).enumerated() {
            let rank = Double(index + 1)
            let contribution = weight * baseContribution / sqrt(rank)
            scores[hit.chunkID] = max(scores[hit.chunkID] ?? 0, contribution)
        }
    }

    private func accumulateLexicalExpansionPromotion(
        for hits: [LexicalHit],
        queryText: String,
        primaryQueryText: String,
        weight: Double,
        into scores: inout [Int64: Double]
    ) {
        guard weight > 0 else { return }
        let tokenCount = normalizedComparisonKey(for: queryText).split(separator: " ").count
        let isConciseLowOverlap = tokenCount <= 5
            && lexicalQueryOverlapRatio(queryText, primaryQueryText) <= 0.35
        let isTemporalOrAggregate = isTemporalOrAggregateRecallQuery(primaryQueryText)
        guard isConciseLowOverlap || isTemporalOrAggregate else { return }

        let basePromotion = isConciseLowOverlap ? 0.75 : 0.18
        for hit in hits.prefix(lexicalExpansionPromotionRankLimit) {
            let contribution = weight * basePromotion
            scores[hit.chunkID] = max(scores[hit.chunkID] ?? 0, contribution)
        }
    }

    private func lexicalQueryOverlapRatio(_ lhs: String, _ rhs: String) -> Double {
        let left = Set(normalizedComparisonKey(for: lhs).split(separator: " ").map(String.init))
        let right = Set(normalizedComparisonKey(for: rhs).split(separator: " ").map(String.init))
        guard !left.isEmpty else { return 1 }
        return Double(left.intersection(right).count) / Double(left.count)
    }

    private func isTemporalOrAggregateRecallQuery(_ queryText: String) -> Bool {
        let lower = queryText.lowercased()
        if isTimeAnchoredQuery(queryText) {
            return true
        }

        let recallIntentPhrases = [
            "as of", "count", "days passed", "first", "from earliest to latest",
            "how many", "how much", "last time", "most recently", "order of",
            "what month", "when did", "which date"
        ]
        return recallIntentPhrases.contains { lower.contains($0) }
    }

    private func shouldRunDocumentLexicalSearch(
        query: SearchQuery,
        queryText: String,
        branchIndex: Int,
        expansionType: ExpansionType?,
        lexicalHitCount: Int,
        lexicalProbeStrongSignal: Bool,
        usedBranches: Int
    ) -> Bool {
        guard !lexicalProbeStrongSignal else { return false }
        guard query.lexicalCandidateLimit >= 32 else { return false }
        guard usedBranches < documentLexicalMaxBranches else { return false }
        guard lexicalHitCount < documentLexicalSparseHitThreshold else { return false }

        if branchIndex == 0 {
            return isBroadRecallQuery(queryText)
        }
        return expansionType == .lexical
    }

    private func documentLexicalCandidateLimit(for query: SearchQuery, branchIndex: Int) -> Int {
        let scaled = branchIndex == 0 ? query.limit * 4 : query.limit * 3
        return min(query.lexicalCandidateLimit, min(96, max(24, scaled)))
    }

    private func documentLexicalWeight(branchIndex: Int) -> Double {
        branchIndex == 0 ? documentLexicalPrimaryWeight : documentLexicalExpansionWeight
    }

    private func isBroadRecallQuery(_ queryText: String) -> Bool {
        let normalized = normalizedComparisonKey(for: queryText)
        let tokens = normalized.split(separator: " ")
        guard tokens.count >= 5 else {
            let lower = queryText.lowercased()
            let shortQuestionPrefixes = ["what ", "when ", "where ", "which ", "who ", "how "]
            return shortQuestionPrefixes.contains { lower.hasPrefix($0) }
        }

        let lower = queryText.lowercased()
        if lower.contains("?") {
            return true
        }
        let recallCues = [
            "find", "look up", "recall", "remember", "search", "show me", "tell me",
            "what", "when", "where", "which", "who", "how"
        ]
        return recallCues.contains { lower.contains($0) } || tokens.count >= 8
    }

    private func fuseCandidates(
        semanticRRF: [Int64: Double],
        lexicalRRF: [Int64: Double],
        lexicalExpansionPromotionRRF: [Int64: Double],
        primarySemanticPreservationRRF: [Int64: Double],
        primaryLexicalPreservationRRF: [Int64: Double],
        query: SearchQuery,
        primaryQueryText: String,
        queryTags: [ContentTag],
        querySignals: QueryMatchSignals
    ) async throws -> [SearchResult] {
        struct FusedCandidate {
            var metadata: StoredChunkMetadata
            var score: SearchScoreBreakdown
        }

        let candidatePoolLimit = candidatePoolLimit(for: query)
        let candidateIDs = preselectCandidateIDs(
            semanticRRF: semanticRRF,
            lexicalRRF: lexicalRRF,
            lexicalExpansionPromotionRRF: lexicalExpansionPromotionRRF,
            primarySemanticPreservationRRF: primarySemanticPreservationRRF,
            primaryLexicalPreservationRRF: primaryLexicalPreservationRRF,
            query: query,
            primaryQueryText: primaryQueryText,
            candidatePoolLimit: candidatePoolLimit
        )
        guard !candidateIDs.isEmpty else { return [] }

        let metadataRows = try await storage.fetchChunkMetadata(chunkIDs: candidateIDs)
        let metadataMap = Dictionary(uniqueKeysWithValues: metadataRows.map { ($0.chunkID, $0) })

        let now = Date()
        let weights = fusionWeights(for: primaryQueryText)
        var results: [FusedCandidate] = []
        results.reserveCapacity(candidateIDs.count)

        for chunkID in candidateIDs {
            guard let metadata = metadataMap[chunkID] else { continue }

            let semantic = (semanticRRF[chunkID] ?? 0) + (primarySemanticPreservationRRF[chunkID] ?? 0)
            let lexical = (lexicalRRF[chunkID] ?? 0)
                + (lexicalExpansionPromotionRRF[chunkID] ?? 0)
                + (primaryLexicalPreservationRRF[chunkID] ?? 0)
            let ageDays = max(0, now.timeIntervalSince(metadata.modifiedAt) / 86_400)
            let recency = exp(-ageDays / 30.0)
            let anchorBonus = anchorCoverageBonus(queryText: primaryQueryText, metadata: metadata)
            let tagBonus = contentTagBonus(queryTags: queryTags, metadata: metadata)
            let schemaBonus = memorySchemaOverlapBonus(querySignals: querySignals, metadata: metadata)
            let temporalBonus = temporalFitBonus(querySignals: querySignals, metadata: metadata)
            let statusBonus = memoryStatusBonus(querySignals: querySignals, metadata: metadata)
            let fused = (weights.semantic * semantic)
                + (weights.lexical * lexical)
                + (weights.recency * recency)
                + anchorBonus
                + tagBonus
                + schemaBonus
                + temporalBonus
                + statusBonus

            results.append(
                FusedCandidate(
                    metadata: metadata,
                    score: SearchScoreBreakdown(
                        semantic: semantic,
                        lexical: lexical,
                        recency: recency,
                        tag: tagBonus,
                        schema: schemaBonus,
                        temporal: temporalBonus,
                        status: statusBonus,
                        fused: fused
                    )
                )
            )
        }

        return results
            .sorted { lhs, rhs in
                if lhs.score.fused == rhs.score.fused {
                    return lhs.metadata.chunkID < rhs.metadata.chunkID
                }
                return lhs.score.fused > rhs.score.fused
            }
            .prefix(candidatePoolLimit)
            .map { makeSearchResult(from: $0.metadata, queryText: primaryQueryText, score: $0.score) }
    }

    private func preselectCandidateIDs(
        semanticRRF: [Int64: Double],
        lexicalRRF: [Int64: Double],
        lexicalExpansionPromotionRRF: [Int64: Double],
        primarySemanticPreservationRRF: [Int64: Double],
        primaryLexicalPreservationRRF: [Int64: Double],
        query: SearchQuery,
        primaryQueryText: String,
        candidatePoolLimit: Int
    ) -> [Int64] {
        let allCandidateIDs = Set(semanticRRF.keys).union(lexicalRRF.keys)
        guard !allCandidateIDs.isEmpty else { return [] }

        let hydrationLimit = candidateHydrationLimit(for: query, candidatePoolLimit: candidatePoolLimit)
        guard allCandidateIDs.count > hydrationLimit else {
            return Array(allCandidateIDs)
        }

        let weights = fusionWeights(for: primaryQueryText)
        var preliminaryScores: [Int64: Double] = [:]
        preliminaryScores.reserveCapacity(allCandidateIDs.count)
        for chunkID in allCandidateIDs {
            let semantic = (semanticRRF[chunkID] ?? 0) + (primarySemanticPreservationRRF[chunkID] ?? 0)
            let lexical = (lexicalRRF[chunkID] ?? 0)
                + (lexicalExpansionPromotionRRF[chunkID] ?? 0)
                + (primaryLexicalPreservationRRF[chunkID] ?? 0)
            preliminaryScores[chunkID] = (weights.semantic * semantic) + (weights.lexical * lexical)
        }

        var selected: Set<Int64> = []
        selected.reserveCapacity(hydrationLimit)
        let protectedPerSignal = candidateProtectionLimit(for: query, hydrationLimit: hydrationLimit)
        protectTopCandidates(from: semanticRRF, limit: protectedPerSignal, selected: &selected, hydrationLimit: hydrationLimit)
        protectTopCandidates(from: lexicalRRF, limit: protectedPerSignal, selected: &selected, hydrationLimit: hydrationLimit)
        protectTopCandidates(from: lexicalExpansionPromotionRRF, limit: protectedPerSignal, selected: &selected, hydrationLimit: hydrationLimit)
        protectTopCandidates(from: primarySemanticPreservationRRF, limit: protectedPerSignal, selected: &selected, hydrationLimit: hydrationLimit)
        protectTopCandidates(from: primaryLexicalPreservationRRF, limit: protectedPerSignal, selected: &selected, hydrationLimit: hydrationLimit)

        for entry in preliminaryScores.sorted(by: sortCandidateScore(_:_:)) where selected.count < hydrationLimit {
            selected.insert(entry.key)
        }
        return Array(selected)
    }

    private func candidateHydrationLimit(for query: SearchQuery, candidatePoolLimit: Int) -> Int {
        if query.rerankLimit == 0 {
            return candidatePoolLimit
        }
        let requested = max(query.limit, query.rerankLimit)
        let scaled = max(candidatePoolLimit, Int((Double(requested) * 1.5).rounded(.up)), 200)
        return min(maxCandidateHydrationLimit, scaled)
    }

    private func candidateProtectionLimit(for query: SearchQuery, hydrationLimit: Int) -> Int {
        min(max(40, query.limit / 2), max(1, hydrationLimit / 3))
    }

    private func protectTopCandidates(
        from scores: [Int64: Double],
        limit: Int,
        selected: inout Set<Int64>,
        hydrationLimit: Int
    ) {
        guard limit > 0, selected.count < hydrationLimit else { return }
        for entry in scores.sorted(by: sortCandidateScore(_:_:)).prefix(limit) {
            selected.insert(entry.key)
            if selected.count >= hydrationLimit {
                return
            }
        }
    }

    private func sortCandidateScore(_ lhs: Dictionary<Int64, Double>.Element, _ rhs: Dictionary<Int64, Double>.Element) -> Bool {
        if lhs.value == rhs.value {
            return lhs.key < rhs.key
        }
        return lhs.value > rhs.value
    }

    private func prepareStructuredSearchPlan(
        query: SearchQuery,
        normalizedText: String,
        analysis: QueryAnalysis,
        recallPlan: RecallPlan?,
        skipExpansion: Bool = false,
        events: SearchEventHandler?
    ) async throws -> StructuredSearchPlan {
        var expandedQueries: [WeightedQuery] = [
            WeightedQuery(text: normalizedText, weight: query.originalQueryWeight),
        ]
        var mergedAnalysis = QueryAnalysis(
            entities: normalizeMemoryEntities(analysis.entities, maxCount: 6),
            keyTerms: Array(Set(analysis.keyTerms.map(normalizedComparisonKey(for:))).sorted()).filter { !$0.isEmpty },
            facetHints: normalizeFacetHints(analysis.facetHints, maxCount: 4),
            topics: normalizeTopicValues(analysis.topics, maxCount: 6),
            isHowToQuery: analysis.isHowToQuery
        )
        var seen: Set<String> = [normalizedComparisonKey(for: normalizedText)]
        var remainingBudget = max(0, query.expansionLimit)
        if let recallPlan {
            let planEntities = recallPlan.entityValues.map {
                MemoryEntity(label: .other, value: $0, normalizedValue: normalizeEntityValue($0), confidence: 0.7)
            }
            mergedAnalysis.entities = normalizeMemoryEntities(mergedAnalysis.entities + planEntities, maxCount: 8)
            let planFacetHints = makeFacetHints(from: recallPlan.facets ?? [], confidence: 0.78, isExplicit: true)
            mergedAnalysis.facetHints = normalizeFacetHints(mergedAnalysis.facetHints + planFacetHints, maxCount: 6)
            mergedAnalysis.topics = normalizeTopicValues(mergedAnalysis.topics + recallPlan.topics, maxCount: 10)
            appendExpandedQueries(
                texts: recallPlan.lexicalQueries,
                type: .lexical,
                weight: query.expansionQueryWeight,
                budget: &remainingBudget,
                seen: &seen,
                into: &expandedQueries
            )
            appendExpandedQueries(
                texts: recallPlan.semanticQueries,
                type: .semantic,
                weight: query.expansionQueryWeight,
                budget: &remainingBudget,
                seen: &seen,
                into: &expandedQueries
            )
            appendExpandedQueries(
                texts: recallPlan.hypotheticalDocuments,
                type: .hypotheticalDocument,
                weight: query.expansionQueryWeight * 0.85,
                budget: &remainingBudget,
                seen: &seen,
                into: &expandedQueries
            )
        }

        guard !skipExpansion,
              query.expansionLimit > 0,
              let structuredExpander = configuration.structuredQueryExpander else {
            return makeStructuredSearchPlan(
                expandedQueries: expandedQueries,
                analysis: mergedAnalysis,
                recallPlan: recallPlan
            )
        }

        var expansionQuery = query
        expansionQuery.text = normalizedText

        let expansion: StructuredQueryExpansion
        do {
            expansion = try await structuredExpander.expand(
                query: expansionQuery,
                analysis: mergedAnalysis,
                limit: query.expansionLimit
            )
        } catch {
            events?(
                .providerFailure(
                    stage: .expansion,
                    provider: structuredExpander.identifier,
                    message: error.localizedDescription
                )
            )
            return makeStructuredSearchPlan(
                expandedQueries: expandedQueries,
                analysis: mergedAnalysis,
                recallPlan: recallPlan
            )
        }

        mergedAnalysis.entities = normalizeMemoryEntities(mergedAnalysis.entities + expansion.entities, maxCount: 6)
        mergedAnalysis.facetHints = normalizeFacetHints(mergedAnalysis.facetHints + expansion.facetHints, maxCount: 4)
        mergedAnalysis.topics = normalizeTopicValues(mergedAnalysis.topics + expansion.topics, maxCount: 6)
        appendExpandedQueries(
            texts: expansion.lexicalQueries,
            type: .lexical,
            weight: query.expansionQueryWeight,
            budget: &remainingBudget,
            seen: &seen,
            into: &expandedQueries
        )
        appendExpandedQueries(
            texts: expansion.semanticQueries,
            type: .semantic,
            weight: query.expansionQueryWeight,
            budget: &remainingBudget,
            seen: &seen,
            into: &expandedQueries
        )
        appendExpandedQueries(
            texts: expansion.hypotheticalDocuments,
            type: .hypotheticalDocument,
            weight: query.expansionQueryWeight * 0.85,
            budget: &remainingBudget,
            seen: &seen,
            into: &expandedQueries
        )

        return makeStructuredSearchPlan(
            expandedQueries: expandedQueries,
            analysis: mergedAnalysis,
            recallPlan: recallPlan
        )
    }

    private func makeStructuredSearchPlan(
        expandedQueries: [WeightedQuery],
        analysis: QueryAnalysis,
        recallPlan: RecallPlan?
    ) -> StructuredSearchPlan {
        StructuredSearchPlan(
            expandedQueries: expandedQueries,
            analysis: analysis,
            entityLexicalQueries: Array(analysis.entities.prefix(4).map(\.value)),
            facetTagNames: analysis.facetHints
                .filter { $0.confidence >= 0.55 }
                .map { "facet:\($0.tag.rawValue)" },
            entityTagNames: analysis.entities
                .prefix(4)
                .map { "entity:\($0.normalizedValue)" },
            topicTagNames: analysis.topics
                .prefix(4)
                .map { "topic:\($0)" },
            temporalIntent: recallPlan?.temporalIntent ?? .any,
            preferredStatuses: recallPlan?.statuses ?? []
        )
    }

    private func appendExpandedQueries(
        texts: [String],
        type: ExpansionType,
        weight: Double,
        budget: inout Int,
        seen: inout Set<String>,
        into queries: inout [WeightedQuery]
    ) {
        guard budget > 0 else { return }
        guard weight > 0 else { return }

        for text in texts where budget > 0 {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            let key = normalizedComparisonKey(for: trimmed)
            guard !key.isEmpty, seen.insert(key).inserted else { continue }

            queries.append(
                WeightedQuery(
                    text: trimmed,
                    weight: weight,
                    expansionType: type
                )
            )
            budget -= 1
        }
    }

    private func embedExpandedQueries(
        _ queries: [WeightedQuery],
        semanticCandidateLimit: Int,
        events: SearchEventHandler?
    ) async throws -> [[Float]]? {
        guard semanticCandidateLimit > 0 else { return nil }
        guard !queries.isEmpty else { return [] }

        let texts = queries.map(\.text)
        let vectors: [[Float]]
        do {
            vectors = try await configuration.embeddingProvider.embed(texts: texts, format: .query)
        } catch {
            throw MemoryError.embedding("Failed to embed query batch: \(error.localizedDescription)")
        }

        guard vectors.count == texts.count else {
            throw MemoryError.embedding(
                "Embedding provider \(configuration.embeddingProvider.identifier) returned \(vectors.count) vectors for \(texts.count) queries"
            )
        }

        for vector in vectors {
            events?(.embeddedQuery(dimension: vector.count))
        }

        return vectors
    }

    private func runLexicalProbe(
        query: SearchQuery,
        normalizedText: String,
        allowedChunkIDs: Set<Int64>?,
        allowedMemoryTypes: Set<String>?
    ) async throws -> (seededHits: [LexicalHit]?, strongSignal: Bool) {
        guard query.lexicalCandidateLimit > 0 else {
            return (seededHits: nil, strongSignal: false)
        }

        let probeLimit = max(query.lexicalCandidateLimit, strongLexicalProbeLimit)
        let ftsQuery = ftsPreprocess(normalizedText)
        let probeHits = try await storage.lexicalSearch(
            query: ftsQuery,
            limit: probeLimit,
            allowedChunkIDs: allowedChunkIDs,
            allowedMemoryTypes: allowedMemoryTypes
        )
        let seededHits = Array(probeHits.prefix(query.lexicalCandidateLimit))
        let strongSignal = hasStrongLexicalSignal(query: query, hits: probeHits)
        return (seededHits: seededHits, strongSignal: strongSignal)
    }

    private func hasStrongLexicalSignal(query: SearchQuery, hits: [LexicalHit]) -> Bool {
        guard query.expansionLimit > 0, configuration.structuredQueryExpander != nil else {
            return false
        }
        let queryTokenCount = normalizedComparisonKey(for: query.text).split(separator: " ").count
        guard queryTokenCount <= strongLexicalMaxExpansionSkipTokenCount else {
            return false
        }
        guard let top = hits.first else { return false }

        let second = hits.dropFirst().first?.score ?? 0
        return top.score >= strongLexicalMinScore && (top.score - second) >= strongLexicalMinGap
    }

    private func ftsPreprocess(_ text: String) -> String {
        guard let ftsTokenizer = configuration.ftsTokenizer else { return text }
        let lemmas = ftsTokenizer.tokenize(text)
        guard !lemmas.isEmpty else { return text }
        return lemmas.joined(separator: " ")
    }

    private func applyReranker(
        _ reranker: any Reranker,
        query: SearchQuery,
        fusedResults: [SearchResult],
        rerankCount: Int
    ) async throws -> [SearchResult] {
        guard !fusedResults.isEmpty else { return [] }

        let effectiveRerankCount = min(max(1, rerankCount), fusedResults.count)
        let rerankable = Array(fusedResults.prefix(effectiveRerankCount))
        let remaining = Array(fusedResults.dropFirst(effectiveRerankCount))
        let maxFusedScore = max(0, fusedResults.first?.score.fused ?? 0)

        // Record original RRF rank for each candidate (1-indexed).
        var originalRankByChunkID: [Int64: Int] = [:]
        for (index, candidate) in rerankable.enumerated() {
            originalRankByChunkID[candidate.chunkID] = index + 1
        }

        let assessments = try await reranker.rerank(query: query, candidates: rerankable)
        let allowedIDs = Set(rerankable.map(\.chunkID))

        var assessmentByChunkID: [Int64: RerankAssessment] = [:]
        for assessment in assessments where allowedIDs.contains(assessment.chunkID) {
            let clamped = min(1, max(0, assessment.relevance))
            if let existing = assessmentByChunkID[assessment.chunkID], existing.relevance >= clamped {
                continue
            }
            assessmentByChunkID[assessment.chunkID] = RerankAssessment(
                chunkID: assessment.chunkID,
                relevance: clamped,
                rationale: assessment.rationale
            )
        }

        guard !assessmentByChunkID.isEmpty else {
            throw MemoryError.search("Reranker returned no usable assessments")
        }

        var reranked = rerankable.map { candidate -> SearchResult in
            var updated = candidate
            updated.score.rerank = assessmentByChunkID[candidate.chunkID]?.relevance ?? 0
            updated.score.blended = updated.score.fused
            return updated
        }

        // Normalize fused scores into a 0-1 band so reranker scores can meaningfully
        // reorder the window without discarding the original retrieval signal.
        for index in reranked.indices {
            let chunkID = reranked[index].chunkID
            let rrfRank = originalRankByChunkID[chunkID] ?? (index + 1)
            let fusedScore = normalizedFusedScore(reranked[index].score.fused, maxFusedScore: maxFusedScore)
            reranked[index].score.blended = configuration.positionAwareBlending.blend(
                fused: fusedScore,
                rerank: reranked[index].score.rerank,
                position: rrfRank
            )
        }

        let untouched = remaining.map { candidate -> SearchResult in
            var updated = candidate
            updated.score.rerank = 0
            updated.score.blended = normalizedFusedScore(updated.score.fused, maxFusedScore: maxFusedScore)
            return updated
        }

        return (reranked + untouched).sorted(by: sortByBlendedScore(_:_:))
    }

    private func candidatePoolLimit(for query: SearchQuery) -> Int {
        let requested = max(query.limit, query.rerankLimit)
        if query.rerankLimit == 0 {
            return min(maxCandidateHydrationLimit, max(100, query.limit))
        }
        let expanded = max(query.limit * 8, query.rerankLimit * 4, 200)
        return min(1_000, max(100, max(requested, expanded)))
    }

    private func effectiveRerankCount(query: SearchQuery, fusedCount: Int) -> Int {
        guard query.rerankLimit > 0, fusedCount > 0 else { return 0 }
        return min(fusedCount, query.rerankLimit)
    }

    private func normalizedFusedScore(_ fused: Double, maxFusedScore: Double) -> Double {
        guard maxFusedScore > 0 else { return min(1, max(0, fused)) }
        return min(1, max(0, fused / maxFusedScore))
    }

    private func fusionWeights(for queryText: String) -> (semantic: Double, lexical: Double, recency: Double) {
        if isTimeAnchoredQuery(queryText) {
            return (semantic: 0.64, lexical: 0.35, recency: 0.01)
        }
        return (semantic: 0.62, lexical: 0.33, recency: 0.05)
    }

    private func isTimeAnchoredQuery(_ queryText: String) -> Bool {
        let lower = queryText.lowercased()

        if temporalCueWords.contains(where: lower.contains) {
            return true
        }

        if lower.range(of: #"\b(19|20)\d{2}\b"#, options: .regularExpression) != nil {
            return true
        }
        if lower.range(of: #"\b\d{1,2}:\d{2}\b"#, options: .regularExpression) != nil {
            return true
        }

        return false
    }

    private func anchorCoverageBonus(queryText: String, metadata: StoredChunkMetadata) -> Double {
        let anchors = anchorTokens(from: queryText)
        guard !anchors.isEmpty else { return 0 }

        let searchable = ((metadata.title ?? "") + " " + String(metadata.content.prefix(2_000)))
            .lowercased()

        var matched = 0
        for anchor in anchors where searchable.contains(anchor) {
            matched += 1
        }
        guard matched > 0 else { return 0 }

        let coverage = Double(matched) / Double(anchors.count)
        return 0.003 * coverage
    }

    private func contentTagBonus(queryTags: [ContentTag], metadata: StoredChunkMetadata) -> Double {
        let overlap = contentTagOverlap(queryTags: queryTags, chunkTags: metadata.contentTags)
        return 0.01 * overlap
    }

    private func contentTagOverlap(
        queryTags: [ContentTag],
        chunkTags: [StoredChunkTag]
    ) -> Double {
        guard !queryTags.isEmpty, !chunkTags.isEmpty else { return 0 }

        var queryWeights: [String: Double] = [:]
        for tag in queryTags {
            let key = normalizedComparisonKey(for: tag.name)
            guard !key.isEmpty else { continue }
            queryWeights[key] = max(queryWeights[key] ?? 0, min(1, max(0, tag.confidence)))
        }

        let queryMass = queryWeights.values.reduce(0, +)
        guard queryMass > 0 else { return 0 }

        var chunkWeights: [String: Double] = [:]
        for tag in chunkTags {
            let key = normalizedComparisonKey(for: tag.name)
            guard !key.isEmpty else { continue }
            chunkWeights[key] = max(chunkWeights[key] ?? 0, min(1, max(0, tag.confidence)))
        }

        var matchedMass: Double = 0
        for (name, queryWeight) in queryWeights {
            let chunkWeight = chunkWeights[name] ?? 0
            matchedMass += min(queryWeight, chunkWeight)
        }

        return min(1, max(0, matchedMass / queryMass))
    }

    private func memorySchemaOverlapBonus(
        querySignals: QueryMatchSignals,
        metadata: StoredChunkMetadata
    ) -> Double {
        guard !metadata.contentTags.isEmpty else { return 0 }

        let chunkTagNames = Set(metadata.contentTags.map(\.name))
        let matchedEntities = querySignals.entityValues.reduce(into: 0) { partialResult, value in
            if chunkTagNames.contains("entity:\(value)") {
                partialResult += 1
            }
        }
        let matchedFacets = querySignals.facets.reduce(into: 0) { partialResult, value in
            if chunkTagNames.contains("facet:\(value.rawValue)") {
                partialResult += 1
            }
        }
        let matchedTopics = querySignals.topics.reduce(into: 0) { partialResult, value in
            if chunkTagNames.contains("topic:\(value)") {
                partialResult += 1
            }
        }

        let entityBonus = min(Double(matchedEntities) * 0.03, 0.09)
        let facetBonus = min(Double(matchedFacets) * 0.015, 0.06)
        let topicBonus = min(Double(matchedTopics) * 0.01, 0.04)
        return entityBonus + facetBonus + topicBonus
    }

    private func temporalFitBonus(querySignals: QueryMatchSignals, metadata: StoredChunkMetadata) -> Double {
        switch querySignals.temporalIntent {
        case .any:
            return 0
        case .recent, .mostRecent:
            let ageDays = max(0, Date().timeIntervalSince(metadata.modifiedAt) / 86_400)
            return min(0.05, 0.05 * exp(-ageDays / 14.0))
        case .historical:
            return metadata.memoryStatus == MemoryStatus.superseded.rawValue || metadata.memoryStatus == MemoryStatus.archived.rawValue ? 0.04 : 0
        case .timeAnchored, .count:
            return isTimeAnchoredText(metadata.content) ? 0.025 : 0
        }
    }

    private func memoryStatusBonus(querySignals: QueryMatchSignals, metadata: StoredChunkMetadata) -> Double {
        guard let memoryStatus = metadata.memoryStatus,
              let status = MemoryStatus.parse(memoryStatus),
              !querySignals.preferredStatuses.isEmpty else {
            return 0
        }
        return querySignals.preferredStatuses.contains(status) ? 0.035 : 0
    }

    private func isTimeAnchoredText(_ text: String) -> Bool {
        isTimeAnchoredQuery(text)
    }

    private func anchorTokens(from queryText: String) -> [String] {
        let rawTokens = queryText.split { character in
            !character.isLetter && !character.isNumber
        }

        var prioritized: [String] = []
        var fallback: [String] = []
        var seen: Set<String> = []

        for raw in rawTokens {
            let token = String(raw)
            guard token.count >= 2 else { continue }
            let lower = token.lowercased()
            guard seen.insert(lower).inserted else { continue }

            if token.contains(where: \.isNumber) || (token.first?.isUppercase == true && token.count >= 3) {
                prioritized.append(lower)
                continue
            }

            if token.count >= 5 && !queryStopWords.contains(lower) {
                fallback.append(lower)
            }
        }

        if !prioritized.isEmpty {
            return Array(prioritized.prefix(4))
        }
        return Array(fallback.prefix(3))
    }

    private func sortByBlendedScore(_ lhs: SearchResult, _ rhs: SearchResult) -> Bool {
        if lhs.score.blended == rhs.score.blended {
            if lhs.score.fused == rhs.score.fused {
                return lhs.chunkID < rhs.chunkID
            }
            return lhs.score.fused > rhs.score.fused
        }
        return lhs.score.blended > rhs.score.blended
    }

    private func normalizedComparisonKey(for text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .joined(separator: " ")
    }

    private func normalizedSemanticKey(for text: String) -> String {
        text
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .map(normalizedSemanticToken)
            .filter { !$0.isEmpty && !queryStopWords.contains($0) }
            .joined(separator: " ")
    }

    private func normalizedSemanticToken(_ token: String) -> String {
        switch token {
        case "repo", "repository":
            return "repository"
        case "db", "database":
            return "database"
        case "doc", "docs", "documentation":
            return "docs"
        case "test", "tests", "testing":
            return "test"
        case "migration", "migrations":
            return "migration"
        case "embedding", "embeddings":
            return "embedding"
        case "cache", "caching":
            return "cache"
        default:
            return token
        }
    }

    private func makeSnippet(content: String, queryText: String?) -> String {
        let normalized = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard normalized.count > 300 else { return normalized }

        guard
            let queryText,
            !queryText.isEmpty,
            let range = normalized.range(of: queryText, options: [.caseInsensitive, .diacriticInsensitive])
        else {
            return String(normalized.prefix(300))
        }

        let center = normalized.distance(from: normalized.startIndex, to: range.lowerBound)
        let startOffset = max(0, center - 120)
        let endOffset = min(normalized.count, center + 180)

        let start = normalized.index(normalized.startIndex, offsetBy: startOffset)
        let end = normalized.index(normalized.startIndex, offsetBy: endOffset)
        return String(normalized[start..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func normalizeError(_ error: Error) -> MemoryError {
        if let typed = error as? MemoryError {
            return typed
        }
        return MemoryError.storage(error.localizedDescription)
    }

    private func resolveDocumentPath(_ inputPath: String) async throws -> String {
        let trimmed = inputPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw MemoryError.configuration("Document path must not be empty")
        }

        let expanded = (trimmed as NSString).expandingTildeInPath
        let standardizedFilePath: String?
        if expanded.hasPrefix("/") {
            standardizedFilePath = URL(fileURLWithPath: expanded).standardizedFileURL.path
            if let standardizedFilePath, fileManager.fileExists(atPath: standardizedFilePath) {
                return standardizedFilePath
            }
        } else {
            standardizedFilePath = nil
        }

        let indexedPaths: [String]
        do {
            indexedPaths = try await storage.listDocumentPaths()
        } catch {
            throw normalizeError(error)
        }

        let exactCandidates = [trimmed, expanded, standardizedFilePath].compactMap { candidate in
            candidate?.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        for candidate in exactCandidates where !candidate.isEmpty {
            if indexedPaths.contains(candidate) {
                return candidate
            }
        }

        for candidate in exactCandidates where !candidate.isEmpty {
            let normalizedCandidate = normalizedComparisonKey(for: candidate)
            if let match = indexedPaths.first(where: { normalizedComparisonKey(for: $0) == normalizedCandidate }) {
                return match
            }
        }

        var suffixMatches: [String] = []
        for rawSuffix in exactCandidates where !rawSuffix.isEmpty {
            let suffix = normalizePathSuffix(rawSuffix)
            guard !suffix.isEmpty else { continue }

            let matches = indexedPaths.filter { candidate in
                let normalizedCandidate = candidate.replacingOccurrences(of: "\\", with: "/")
                return normalizedCandidate == suffix || normalizedCandidate.hasSuffix("/\(suffix)")
            }

            if !matches.isEmpty {
                suffixMatches = matches
                break
            }
        }

        let uniqueMatches = Array(Set(suffixMatches)).sorted()
        if uniqueMatches.count == 1, let match = uniqueMatches.first {
            return match
        }

        if uniqueMatches.count > 1 {
            let rendered = uniqueMatches.prefix(3).joined(separator: ", ")
            throw MemoryError.search("Ambiguous document path '\(trimmed)'; matches: \(rendered)")
        }

        throw MemoryError.search("Document not found for path '\(trimmed)'")
    }

    private func resolveDocumentSource(for path: String) -> MemoryDocumentSource {
        if path.hasPrefix("memory://") {
            return .indexed
        }
        return fileManager.fileExists(atPath: path) ? .fileSystem : .indexed
    }

    private func loadDocumentText(for path: String) async throws -> (content: String, source: MemoryDocumentSource) {
        let source = resolveDocumentSource(for: path)

        if source == .fileSystem, let content = try? String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8) {
            return (content: content, source: .fileSystem)
        }

        do {
            let chunkMetadata = try await storage.fetchChunkMetadataForDocument(path: path)
            guard !chunkMetadata.isEmpty else {
                throw MemoryError.search("Document not found at path '\(path)'")
            }
            let reconstructed = chunkMetadata
                .map(\.content)
                .joined(separator: "\n\n")
            return (content: reconstructed, source: .indexed)
        } catch {
            throw normalizeError(error)
        }
    }

    private func loadDocumentTextIfAvailable(for path: String) async -> String? {
        do {
            let loaded = try await loadDocumentText(for: path)
            return loaded.content
        } catch {
            return nil
        }
    }

    private func normalizePathSuffix(_ rawPath: String) -> String {
        var normalized = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\\", with: "/")

        if normalized.hasPrefix("./") {
            normalized.removeFirst(2)
        }

        while normalized.hasPrefix("/") {
            normalized.removeFirst()
        }

        return normalized
    }

    private func normalizeLineEndings(in text: String) -> String {
        text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
    }

    private func splitLines(from text: String) -> [String] {
        let normalized = normalizeLineEndings(in: text)
        return normalized
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
    }

    private func clampLineRange(_ requested: MemoryLineRange?, totalLineCount: Int) -> MemoryLineRange {
        let safeTotal = max(1, totalLineCount)
        guard let requested else {
            return MemoryLineRange(start: 1, end: safeTotal)
        }

        let clampedStart = min(max(1, requested.start), safeTotal)
        let clampedEnd = min(max(clampedStart, requested.end), safeTotal)
        return MemoryLineRange(start: clampedStart, end: clampedEnd)
    }

    private func inferLineRange(in documentText: String, chunkText: String, snippet: String) -> MemoryLineRange? {
        let normalizedDocument = normalizeLineEndings(in: documentText)
        guard !normalizedDocument.isEmpty else { return nil }

        var candidates: [String] = []
        candidates.reserveCapacity(4)

        let normalizedChunk = normalizeLineEndings(in: chunkText).trimmingCharacters(in: .whitespacesAndNewlines)
        if !normalizedChunk.isEmpty {
            candidates.append(normalizedChunk)
            candidates.append(String(normalizedChunk.prefix(180)))
        }

        let normalizedSnippet = normalizeLineEndings(in: snippet).trimmingCharacters(in: .whitespacesAndNewlines)
        if !normalizedSnippet.isEmpty {
            candidates.append(normalizedSnippet)
        }

        for candidate in candidates {
            guard candidate.count >= 8 else { continue }

            if let range = normalizedDocument.range(of: candidate) {
                return lineRange(in: normalizedDocument, for: range)
            }
            if let range = normalizedDocument.range(of: candidate, options: [.caseInsensitive, .diacriticInsensitive]) {
                return lineRange(in: normalizedDocument, for: range)
            }
        }

        return nil
    }

    private func lineRange(in text: String, for characterRange: Range<String.Index>) -> MemoryLineRange {
        let startLine = lineNumber(at: characterRange.lowerBound, in: text)
        let endCharacterIndex: String.Index
        if characterRange.isEmpty {
            endCharacterIndex = characterRange.lowerBound
        } else {
            endCharacterIndex = text.index(before: characterRange.upperBound)
        }
        let endLine = lineNumber(at: endCharacterIndex, in: text)
        return MemoryLineRange(start: startLine, end: max(startLine, endLine))
    }

    private func lineNumber(at index: String.Index, in text: String) -> Int {
        var line = 1
        for character in text[..<index] where character == "\n" {
            line += 1
        }
        return line
    }
}

private let temporalCueWords: [String] = [
    "when", "timeline", "chronology", "chronological", "date", "dates",
    "schedule", "scheduled", "milestone", "kickoff", "kick-off",
    "before", "after", "between", "first", "earliest", "latest",
    "order of", "most recent", "recently", "past month", "past two months",
    "valentine", "valentine's",
    "jan", "january", "feb", "february", "mar", "march", "apr", "april",
    "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
    "oct", "october", "nov", "november", "dec", "december",
    "today", "yesterday", "tomorrow"
]

private let queryStopWords: Set<String> = [
    "about", "after", "all", "also", "an", "and", "any", "are", "as", "at",
    "be", "been", "before", "but", "by", "can", "do", "does", "for", "from",
    "how", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our",
    "that", "the", "their", "them", "there", "these", "they", "this", "to",
    "up", "was", "we", "what", "when", "where", "which", "who", "why", "with", "you", "your"
]

private let canonicalStopWords: Set<String> = [
    "action", "add", "closed", "commitment", "completed", "decision", "done",
    "fact", "finished", "item", "memory", "memories", "profile", "resolved",
    "status", "switched", "switch", "task", "todo"
]

private let canonicalMatchStopWords: Set<String> = [
    "actually", "approved", "before", "choose", "chose", "complete", "continue",
    "current", "decided", "default", "finished", "implement", "implemented",
    "instead", "launch", "ready", "recently", "remember", "update", "updated",
    "using"
]
