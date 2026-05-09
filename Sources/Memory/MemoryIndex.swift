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
    private let searchAdjustments: MemorySearchAdjustmentSet

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
        var monthDayAnchors: Set<MonthDayAnchor>
        var monthAnchors: Set<Int>
        var understanding: RecallQueryUnderstanding
    }

    private struct AnchorCoverageSignals {
        var anchors: [String]
        var quotedPhrases: [String]
    }

    private struct SupportContinuationCandidate {
        var result: SearchResult
        var originalIndex: Int
        var documentRank: Int
        var supportScore: Double
        var supportGroupKey: String?
    }

    private struct MonthDayAnchor: Hashable {
        var month: Int
        var day: Int
    }

    private struct LegacyDocumentMemoryTypeClassification {
        var label: String
        var confidence: Double
    }

    private struct RetrievalMemoryTypeIntent {
        var label: String
        var confidence: Double
        var compatibleLabels: Set<String>

        var isInformative: Bool {
            confidence >= 0.55
        }
    }

    private static let monthNameToNumber: [String: Int] = [
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    ]
    private static let monthNameByNumber: [Int: String] = Dictionary(
        uniqueKeysWithValues: monthNameToNumber.map { ($0.value, $0.key) }
    )
    private static let documentProceduralTitlePhrases: [String] = [
        "apply",
        "file",
        "appeal",
        "get",
        "renew",
        "register",
        "transfer",
        "change",
        "create",
        "open",
        "pay",
        "plead",
        "provide",
        "remove",
        "exchange",
        "learn what documents",
        "checklists",
        "instructions",
        "how to",
        "loan counseling",
        "log on",
        "link to us",
        "permit",
        "restriction",
        "restrictions",
        "status",
        "designation",
        "insurance",
        "claims",
        "complaint",
        "complaints",
        "charges",
        "lower",
        "suspend",
        "treatment",
        "complete",
        "counseling",
        "agreement",
        "eligible",
        "reemployment",
        "users",
        "foreign",
        "sales tax",
        "careers",
        "career",
        "employment",
        "copy",
        "ticket",
    ]
    private static let documentStrongProceduralBodyPhrases: [String] = [
        "step 1",
        "apply online",
        "apply by mail",
        "online or by mail",
        "in person",
        "at a dmv office",
        "follow the instructions",
        "checklist",
        "you must",
        "you need to",
    ]
    private static let documentProceduralBodyPhrases: [String] = [
        "submit",
        "form",
        "application",
        "eligible",
        "requirements",
        "by mail",
        "you can apply",
        "you can file",
        "you can renew",
        "you can get",
        "you can change",
        "you can transfer",
    ]
    private static let documentEpisodicTitlePhrases: [String] = [
        "stories",
        "story",
        "tale",
        "journey",
        "incident report",
        "spotlight",
        "campaign",
        "volunteerism",
        "resilience",
    ]
    private static let documentStrongEpisodicBodyPhrases: [String] = [
        "once upon",
        "when i",
        "i found myself",
        "i ve",
        "i have walked",
        "personal account",
        "at the starting line",
    ]
    private static let documentEpisodicBodyPhrases: [String] = [
        "on september",
        "on october",
        "on november",
        "residents were",
        "city council",
        "recently",
        "launched",
        "announced",
        "marked",
        "gathered",
        "woke",
        "struck",
        "occurred",
        "happened",
    ]
    private static let documentSemanticTitlePhrases: [String] = [
        "myths",
        "mythical",
        "legendary",
        "origins",
        "importance",
        "emerging technologies",
        "methodologies",
        "supply chain",
        "internet of things",
        "mindfulness",
        "festivals",
        "revolutionizing",
        "learn about",
        "works",
        "benefits",
    ]
    private static let documentStrongSemanticBodyPhrases: [String] = [
        "ethical debates",
        "methodologies",
        "historical origins",
        "importance",
        "origin stories",
        "creation myths",
        "internet of things",
        "emerging technologies",
        "supply chain",
        "mindfulness",
        "festivals",
        "cultural heritage",
    ]
    private static let documentSemanticBodyPhrases: [String] = [
        "concept",
        "concepts",
        "understand",
        "learn how",
        "in this section",
        "exploring",
        "importance",
        "role of",
        "history of",
        "origins",
        "landscape",
        "transforming",
        "innovation",
        "cultural",
        "society",
        "traditions",
    ]
    private static let documentContextualTitlePhrases: [String] = [
        "terms of service",
        "terms",
    ]
    private static let documentContextualBodyPhrases: [String] = [
        "terms of service",
        "registered users",
        "take effect",
        "changes expected",
        "announced an ambitious",
    ]

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
        self.searchAdjustments = MemorySearchAdjustmentSet.enabledFromProcessEnvironment()

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
        recallPlan: RecallPlan?,
        queryUnderstanding: RecallQueryUnderstanding? = nil
    ) async throws -> [SearchResult] {
        let normalizedText = query.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedText.isEmpty else { return [] }

        let queryStart = DispatchTime.now().uptimeNanoseconds
        events?(.started(query: normalizedText))

        var effectiveAllowedChunkIDsOverride = allowedChunkIDsOverride
        if let documentPathPrefix = query.documentPathPrefix {
            let scopedChunkIDs = Set(try await storage.fetchChunkIDs(documentPathPrefix: documentPathPrefix))
            effectiveAllowedChunkIDsOverride = combineAllowedChunkIDs(effectiveAllowedChunkIDsOverride, scopedChunkIDs)
            if scopedChunkIDs.isEmpty || (effectiveAllowedChunkIDsOverride?.isEmpty == true) {
                events?(.completed(count: 0))
                return []
            }
        }

        let allowedChunkIDs: Set<Int64>?
        if let contextID = query.contextID {
            let contextChunkIDs = try await storage.fetchContextChunkIDs(contextID: contextID.rawValue)
            let contextSet = Set(contextChunkIDs)
            allowedChunkIDs = combineAllowedChunkIDs(contextSet, effectiveAllowedChunkIDsOverride)
            if contextSet.isEmpty || (allowedChunkIDs?.isEmpty == true) {
                events?(.completed(count: 0))
                return []
            }
        } else {
            allowedChunkIDs = effectiveAllowedChunkIDsOverride
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
        let skipSemanticSearch = shouldSkipSemanticSearchForScopedQuery(
            query: query,
            allowedChunkIDs: allowedChunkIDs,
            lexicalProbe: lexicalProbe
        )

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
            semanticCandidateLimit: skipSemanticSearch ? 0 : query.semanticCandidateLimit,
            events: events
        )
        events?(.stageTiming(stage: .queryEmbedding, durationMs: elapsedMilliseconds(since: queryEmbeddingStart)))

        var semanticRRF: [Int64: Double] = [:]
        var lexicalRRF: [Int64: Double] = [:]
        var semanticCandidateCount = 0
        var lexicalCandidateCount = 0
        var semanticSearchDurationMs = 0.0
        var documentLexicalBranchCount = 0

        for (index, expandedQuery) in searchPlan.expandedQueries.enumerated() {
            let skipSemantic = expandedQuery.expansionType == .lexical
                || skipSemanticSearch
            let skipLexical = expandedQuery.expansionType == .semantic
                || expandedQuery.expansionType == .hypotheticalDocument

            if !skipSemantic,
               let semanticQueryVectors,
               let semanticQueryVector = semanticQueryVectors[index] {
                let semanticSearchStart = DispatchTime.now().uptimeNanoseconds
                let semanticHits = try await semanticSearch(
                    queryVector: semanticQueryVector,
                    limit: query.semanticCandidateLimit,
                    allowedChunkIDs: allowedChunkIDs,
                    allowedMemoryTypes: allowedMemoryTypes
                )
                semanticSearchDurationMs += elapsedMilliseconds(since: semanticSearchStart)
                semanticCandidateCount += semanticHits.count
                accumulateRRF(for: semanticHits, weight: expandedQuery.weight, into: &semanticRRF)
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
        let querySignals = queryMatchSignals(
            from: searchPlan.analysis,
            plan: recallPlan,
            queryText: normalizedText,
            understanding: queryUnderstanding
        )
        let memoryTypeIntent = classifyRetrievalMemoryTypeIntent(querySignals.understanding)
        events?(.memoryTypeIntent(label: memoryTypeIntent.label, confidence: memoryTypeIntent.confidence))
        let queryTags = query.includeTagScoring
            ? await resolveQueryContentTags(queryText: normalizedText, queryAnalysis: searchPlan.analysis, events: events)
            : []
        var fused = try await fuseCandidates(
            semanticRRF: semanticRRF,
            lexicalRRF: lexicalRRF,
            query: query,
            primaryQueryText: normalizedText,
            queryTags: queryTags,
            querySignals: querySignals,
            memoryTypeIntent: memoryTypeIntent
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

        fused = applyPostRerankAdjustments(
            to: fused,
            querySignals: querySignals,
            memoryTypeIntent: memoryTypeIntent,
            query: query
        )

        let final = Array(fused.sorted(by: sortByBlendedScore(_:_:)).prefix(query.limit))
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
            let queryUnderstanding = RecallQueryUnderstandingAnalyzer.analyze(queryText)
            let expansionLimit = memorySearchExpansionLimit(
                features: features,
                understanding: queryUnderstanding
            )

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
                recallPlan: plan,
                queryUnderstanding: queryUnderstanding
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
        additionalLexicalQueries: [String] = [],
        additionalLexicalQueryWeight: Double = 0.35,
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
        let queryUnderstanding = RecallQueryUnderstandingAnalyzer.analyze(queryText)

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
        let expansionLimit = memorySearchExpansionLimit(
            features: features,
            understanding: queryUnderstanding
        )
        let searchLimit = memorySearchCandidateLimit(
            effectiveLimit: effectiveLimit,
            dedupeDocuments: dedupeDocuments,
            understanding: queryUnderstanding
        )

        let searchResults = try await search(
            SearchQuery(
                text: queryText,
                limit: searchLimit,
                semanticCandidateLimit: semanticLimit,
                lexicalCandidateLimit: lexicalLimit,
                rerankLimit: rerankLimit,
                expansionLimit: expansionLimit,
                additionalLexicalQueries: additionalLexicalQueries,
                additionalLexicalQueryWeight: additionalLexicalQueryWeight,
                includeTagScoring: features.contains(.tags)
            ),
            events: events,
            allowedChunkIDsOverride: allowedChunkIDs,
            recallPlan: plan,
            queryUnderstanding: queryUnderstanding
        )
        let orderedSearchResults = searchAdjustments.contains(.aggregateSupportContinuations)
            ? preserveAggregateSupportContinuations(
                in: searchResults,
                understanding: queryUnderstanding,
                effectiveLimit: effectiveLimit,
                dedupeDocuments: dedupeDocuments,
                activeOnlyByDefault: statuses == nil && plan.statuses == nil
            )
            : searchResults
        var references: [MemorySearchReference] = []
        references.reserveCapacity(effectiveLimit)

        var seenDocumentKeys: Set<String> = []
        var documentTextCache: [String: String] = [:]

        for result in orderedSearchResults {
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
                    memoryType: result.memoryType,
                    memoryTypeConfidence: result.memoryTypeConfidence,
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

    private func memorySearchCandidateLimit(
        effectiveLimit: Int,
        dedupeDocuments: Bool,
        understanding: RecallQueryUnderstanding
    ) -> Int {
        guard dedupeDocuments else { return effectiveLimit }

        let broadLimit = min(400, max(effectiveLimit * 6, effectiveLimit))
        let defaultLimit = effectiveLimit >= 50 ? min(320, broadLimit) : broadLimit
        guard understanding.isEvidenceDense else { return defaultLimit }

        let evidenceFloor = aggregateSupportScanFloor(for: understanding)
        return min(400, max(defaultLimit, evidenceFloor))
    }

    private func memorySearchExpansionLimit(
        features: RecallFeatures,
        understanding: RecallQueryUnderstanding
    ) -> Int {
        guard features.contains(.expansion) else { return 0 }
        if shouldPreserveOriginalSurfaceForSuperlativeMoneyAggregate(understanding) {
            return 0
        }
        return 5
    }

    private func shouldPreserveOriginalSurfaceForSuperlativeMoneyAggregate(
        _ understanding: RecallQueryUnderstanding
    ) -> Bool {
        guard understanding.requiresEvidenceAggregation,
              understanding.operations.contains(.sum),
              understanding.tokens.count <= 14 else {
            return false
        }

        let tokenSet = Set(understanding.tokens)
        let hasMoneyCue = !tokenSet.isDisjoint(with: [
            "money", "spend", "spent", "paid", "pay", "cost", "costs", "price", "prices", "amount",
        ])
        guard hasMoneyCue else { return false }

        let text = " \(understanding.normalizedText) "
        return text.contains(" most ")
            || text.contains(" least ")
            || text.contains(" highest ")
            || text.contains(" lowest ")
    }

    private func aggregateSupportScanFloor(for understanding: RecallQueryUnderstanding) -> Int {
        if understanding.operations.contains(.comparison) {
            return 120
        }
        if understanding.requiresEvidenceAggregation || understanding.operations.contains(.ordering) {
            return 90
        }
        return 60
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
            memoryType: normalizedRetrievalMemoryType(metadata.memoryType),
            memoryTypeConfidence: metadata.memoryTypeConfidence,
            score: score
        )
    }

    private func heuristicExtract(messages: [ConversationMessage], limit: Int) -> MemoryExtractionResult {
        MemoryExtractionHeuristics.extract(
            messages: messages,
            limit: limit,
            canonicalKey: { kind, text, explicitKey, entities, topics in
                self.resolveCanonicalKey(
                    for: kind,
                    text: text,
                    explicitKey: explicitKey,
                    entities: entities,
                    topics: topics
                )
            },
            proposedAction: { candidate in
                self.proposedWriteAction(for: candidate)
            }
        )
    }

    private func inferredTags(forExtractedText text: String) -> [String] {
        MemoryExtractionHeuristics.inferredTags(forExtractedText: text)
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
        let maxTopics = 16
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
        MemoryExtractionHeuristics.inferFacetTags(forExtractedText: text, kind: kind)
    }

    private func inferEntities(forExtractedText text: String) -> [MemoryEntity] {
        MemoryExtractionHeuristics.inferEntities(forExtractedText: text)
    }

    private func inferTopics(forExtractedText text: String, seedTags: [String]) -> [String] {
        MemoryExtractionHeuristics.inferTopics(forExtractedText: text, seedTags: seedTags)
    }

    private func normalizeEntityValue(_ raw: String) -> String {
        MemoryExtractionHeuristics.normalizeEntityValue(raw)
    }

    private func normalizeTopicValue(_ raw: String) -> String {
        MemoryExtractionHeuristics.normalizeTopicValue(raw)
    }

    private func makeStoredMemoryEntity(from entity: MemoryEntity) -> StoredMemoryEntity {
        MemoryExtractionHeuristics.makeStoredMemoryEntity(from: entity)
    }

    private func makeMemoryEntity(from entity: StoredMemoryEntity) -> MemoryEntity? {
        MemoryExtractionHeuristics.makeMemoryEntity(from: entity)
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
            .filter { $0.count >= 3 && !MemorySearchHeuristics.queryStopWords.contains($0) }

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
                    && !MemorySearchHeuristics.queryStopWords.contains(token)
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
        return !tokens.isEmpty && tokens.allSatisfy { generic.contains($0) || MemorySearchHeuristics.queryStopWords.contains($0) }
    }

    private func candidateKeySeed(from text: String, maxTokens: Int = 6) -> String? {
        let tokens = text
            .lowercased()
            .split { character in !character.isLetter && !character.isNumber }
            .map(String.init)
            .filter {
                $0.count >= 3
                    && !MemorySearchHeuristics.queryStopWords.contains($0)
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
        guard candidate.kind == .commitment || candidate.kind == .decision || candidate.kind == .profile else { return nil }
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
                    && !MemorySearchHeuristics.queryStopWords.contains(token)
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
        let memoryType = classifyLegacyDocumentMemoryType(title: documentTitle, content: content)
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
            memoryType: memoryType.label,
            memoryTypeSource: "system",
            memoryTypeConfidence: memoryType.confidence,
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

    private func classifyLegacyDocumentMemoryType(
        title: String,
        content: String
    ) -> LegacyDocumentMemoryTypeClassification {
        let normalizedTitle = normalizedClassifierText(title)
        let normalizedContent = normalizedClassifierText(String(content.prefix(16_000)))
        let contentTokens = normalizedContent.split(separator: " ").map(String.init)

        var proceduralScore = 0
        var episodicScore = 0
        var semanticScore = 0
        var contextualScore = 0

        let proceduralTitleMatches = countNormalizedPhraseMatches(
            MemoryIndex.documentProceduralTitlePhrases,
            in: normalizedTitle
        )
        proceduralScore += proceduralTitleMatches * 3
        let numberedProcessCue = hasNumberedProcessCue(tokens: contentTokens)
        let proceduralHowCue = containsAnyNormalizedPhrase(["how do", "how can", "how to"], in: normalizedContent)
        if proceduralTitleMatches > 0 {
            proceduralScore += countNormalizedPhraseMatches(
                MemoryIndex.documentStrongProceduralBodyPhrases,
                in: normalizedContent
            ) * 2
            proceduralScore += countNormalizedPhraseMatches(
                MemoryIndex.documentProceduralBodyPhrases,
                in: normalizedContent
            )
            if proceduralHowCue {
                proceduralScore += 2
            }
            if hasOrderedActionCue(tokens: contentTokens, normalizedContent: normalizedContent) {
                proceduralScore += 1
            }
        } else if numberedProcessCue {
            proceduralScore += 4
        }
        if numberedProcessCue {
            proceduralScore += 2
        }

        let episodicTitleMatches = countNormalizedPhraseMatches(
            MemoryIndex.documentEpisodicTitlePhrases,
            in: normalizedTitle
        )
        episodicScore += episodicTitleMatches * 2
        episodicScore += countNormalizedPhraseMatches(
            MemoryIndex.documentStrongEpisodicBodyPhrases,
            in: normalizedContent
        ) * 3
        if episodicTitleMatches > 0 {
            episodicScore += countNormalizedPhraseMatches(
                MemoryIndex.documentEpisodicBodyPhrases,
                in: normalizedContent
            )
        }
        if containsNormalizedPhrase("neighborhood stories", in: normalizedTitle),
           hasNeighborhoodNarrativeCue(tokens: contentTokens, normalizedContent: normalizedContent) {
            episodicScore += 3
        }
        if hasPersonalNarrativeCue(normalizedContent: normalizedContent) {
            episodicScore += 3
        }
        if episodicTitleMatches > 0, hasMonthDateCue(tokens: contentTokens) {
            episodicScore += 1
        }

        let semanticTitleMatches = countNormalizedPhraseMatches(
            MemoryIndex.documentSemanticTitlePhrases,
            in: normalizedTitle
        )
        semanticScore += semanticTitleMatches * 2
        semanticScore += countNormalizedPhraseMatches(
            MemoryIndex.documentStrongSemanticBodyPhrases,
            in: normalizedContent
        ) * 3
        if semanticTitleMatches > 0 {
            semanticScore += countNormalizedPhraseMatches(
                MemoryIndex.documentSemanticBodyPhrases,
                in: normalizedContent
            )
        }
        if containsAnyNormalizedPhrase(["folklore", "myth", "myths", "legendary", "legends"], in: normalizedTitle) {
            semanticScore += 4
        }

        contextualScore += countNormalizedPhraseMatches(
            MemoryIndex.documentContextualTitlePhrases,
            in: normalizedTitle
        ) * 2
        contextualScore += countNormalizedPhraseMatches(
            MemoryIndex.documentContextualBodyPhrases,
            in: normalizedContent
        ) * 2

        if contextualScore >= 4,
           contextualScore >= max(proceduralScore, episodicScore, semanticScore) - 1 {
            return LegacyDocumentMemoryTypeClassification(
                label: "contextual",
                confidence: confidence(for: contextualScore)
            )
        }

        let scored = [
            ("procedural", proceduralScore),
            ("episodic", episodicScore),
            ("semantic", semanticScore),
            ("contextual", contextualScore),
        ]
        let best = scored.max { lhs, rhs in
            if lhs.1 == rhs.1 {
                return legacyDocumentTypePriority(lhs.0) < legacyDocumentTypePriority(rhs.0)
            }
            return lhs.1 < rhs.1
        } ?? ("factual", 0)

        switch best.0 {
        case "procedural" where best.1 >= 5:
            return LegacyDocumentMemoryTypeClassification(label: best.0, confidence: confidence(for: best.1))
        case "episodic" where best.1 >= 4:
            return LegacyDocumentMemoryTypeClassification(label: best.0, confidence: confidence(for: best.1))
        case "semantic" where best.1 >= 4:
            return LegacyDocumentMemoryTypeClassification(label: best.0, confidence: confidence(for: best.1))
        case "contextual" where best.1 >= 4:
            return LegacyDocumentMemoryTypeClassification(label: best.0, confidence: confidence(for: best.1))
        default:
            return LegacyDocumentMemoryTypeClassification(label: "factual", confidence: 0.55)
        }
    }

    private func legacyDocumentTypePriority(_ label: String) -> Int {
        switch label {
        case "contextual":
            return 4
        case "semantic":
            return 3
        case "episodic":
            return 2
        case "procedural":
            return 1
        default:
            return 0
        }
    }

    private func confidence(for score: Int) -> Double {
        min(0.95, 0.55 + (Double(score) * 0.04))
    }

    private func normalizedClassifierText(_ text: String) -> String {
        text
            .lowercased()
            .split { character in !character.isLetter && !character.isNumber }
            .joined(separator: " ")
    }

    private func containsNormalizedPhrase(_ phrase: String, in normalizedText: String) -> Bool {
        guard !phrase.isEmpty, !normalizedText.isEmpty else { return false }
        return " \(normalizedText) ".contains(" \(phrase) ")
    }

    private func containsAnyNormalizedPhrase(_ phrases: [String], in normalizedText: String) -> Bool {
        phrases.contains { containsNormalizedPhrase($0, in: normalizedText) }
    }

    private func countNormalizedPhraseMatches(_ phrases: [String], in normalizedText: String) -> Int {
        phrases.reduce(into: 0) { count, phrase in
            if containsNormalizedPhrase(phrase, in: normalizedText) {
                count += 1
            }
        }
    }

    private func hasNumberedProcessCue(tokens: [String]) -> Bool {
        guard tokens.count >= 2 else { return false }
        for index in 0..<(tokens.count - 1) {
            guard tokens[index] == "step" || tokens[index] == "phase" else { continue }
            if Int(tokens[index + 1]) != nil {
                return true
            }
        }
        return false
    }

    private func hasOrderedActionCue(tokens: [String], normalizedContent: String) -> Bool {
        let orderingTokens: Set<String> = ["first", "second", "third", "next", "finally"]
        guard tokens.contains(where: orderingTokens.contains) else { return false }
        return containsAnyNormalizedPhrase(["apply", "submit", "complete"], in: normalizedContent)
    }

    private func hasMonthDateCue(tokens: [String]) -> Bool {
        guard tokens.count >= 2 else { return false }
        for index in 0..<(tokens.count - 1) {
            guard MemoryIndex.monthNameToNumber[tokens[index]] != nil else { continue }
            if Int(tokens[index + 1]) != nil {
                return true
            }
        }
        return false
    }

    private func hasNeighborhoodNarrativeCue(tokens: [String], normalizedContent: String) -> Bool {
        if containsAnyNormalizedPhrase(
            [
                "when i",
                "i found myself",
                "once upon",
                "gathered",
                "marked the commencement",
                "at the starting line",
            ],
            in: normalizedContent
        ) {
            return true
        }

        return tokens.contains("september")
            || tokens.contains("october")
            || tokens.contains("november")
            || hasMonthDateCue(tokens: tokens)
    }

    private func hasPersonalNarrativeCue(normalizedContent: String) -> Bool {
        guard containsAnyNormalizedPhrase(["i", "my", "we"], in: normalizedContent) else { return false }
        return containsAnyNormalizedPhrase(
            [
                "felt",
                "learned",
                "walked",
                "found myself",
                "declared",
                "went",
                "began",
            ],
            in: normalizedContent
        )
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

    private func queryMatchSignals(
        from analysis: QueryAnalysis,
        queryText: String,
        understanding: RecallQueryUnderstanding? = nil
    ) -> QueryMatchSignals {
        QueryMatchSignals(
            facets: Set(analysis.facetHints.map(\.tag)),
            entityValues: Set(analysis.entities.map(\.normalizedValue).map(normalizeEntityValue).filter { !$0.isEmpty }),
            topics: Set(analysis.topics.map(normalizeTopicValue).filter { !$0.isEmpty }),
            temporalIntent: .any,
            preferredStatuses: [],
            monthDayAnchors: [],
            monthAnchors: [],
            understanding: understanding ?? RecallQueryUnderstandingAnalyzer.analyze(queryText)
        )
    }

    private func queryMatchSignals(
        from analysis: QueryAnalysis,
        plan: RecallPlan?,
        queryText: String,
        understanding: RecallQueryUnderstanding? = nil
    ) -> QueryMatchSignals {
        var signals = queryMatchSignals(from: analysis, queryText: queryText, understanding: understanding)
        if let plan {
            signals.temporalIntent = plan.temporalIntent
            signals.preferredStatuses = plan.statuses ?? []
        }
        signals.monthDayAnchors = monthDayAnchors(from: queryText)
        signals.monthAnchors = monthAnchors(from: queryText)
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
        let understanding = RecallQueryUnderstandingAnalyzer.analyze(query)

        let statuses: Set<MemoryStatus>?
        if containsAnyRecallStatusCue(
            lower,
            cues: ["historical", "superseded", "archived", "old memory", "old memories", "old records"]
        ) {
            statuses = Set(MemoryStatus.allCases)
        } else if hasResolvedStatusRecallCue(lower) {
            statuses = [.active, .resolved]
        } else {
            statuses = nil
        }

        let temporalIntent = understanding.temporalIntent
        let lexicalQueries = analysis.keyTerms.isEmpty ? [] : [analysis.keyTerms.joined(separator: " ")]
        let shouldExpandEvidenceWindow = understanding.requiresEvidenceAggregation

        return RecallPlan(
            query: query,
            lexicalQueries: lexicalQueries,
            statuses: statuses,
            facets: nil,
            entityValues: [],
            topics: [],
            temporalIntent: temporalIntent,
            semanticCandidateLimit: shouldExpandEvidenceWindow ? configuration.semanticCandidateLimit + 150 : nil,
            lexicalCandidateLimit: shouldExpandEvidenceWindow ? configuration.lexicalCandidateLimit + 150 : nil,
            rerankLimit: shouldExpandEvidenceWindow || temporalIntent == .timeAnchored ? 60 : nil
        )
    }

    private func hasResolvedStatusRecallCue(_ lowercasedQuery: String) -> Bool {
        if lowercasedQuery.range(
            of: #"\bwhat happened\s+(to|with)\b"#,
            options: .regularExpression
        ) != nil {
            return true
        }

        let explicitStatusObjects = [
            "action item", "action items", "commitment", "commitments",
            "memory", "memories", "record", "records", "task", "tasks",
            "todo", "todos", "to do", "to dos",
        ]
        let statusWords = ["done", "completed", "resolved", "closed", "finished"]
        for statusWord in statusWords {
            for object in explicitStatusObjects {
                if lowercasedQuery.contains("\(statusWord) \(object)")
                    || lowercasedQuery.contains("\(object) \(statusWord)") {
                    return true
                }
            }
        }

        return false
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

    private func isTemporalOrAggregateRecallQuery(_ queryText: String) -> Bool {
        MemorySearchHeuristics.isTemporalOrAggregateRecallQuery(queryText)
    }

    private func isRecommendationRecallQuery(_ queryText: String) -> Bool {
        let lower = queryText.lowercased()
        return containsAny(
            lower,
            needles: [
                "recommend", "suggest", "suggestions", "what to watch",
                "watch tonight", "tips on what"
            ]
        )
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
        return branchIndex == 0 ? documentLexicalPrimaryWeight : documentLexicalExpansionWeight
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
        query: SearchQuery,
        primaryQueryText: String,
        queryTags: [ContentTag],
        querySignals: QueryMatchSignals,
        memoryTypeIntent: RetrievalMemoryTypeIntent
    ) async throws -> [SearchResult] {
        struct FusedCandidate {
            var metadata: StoredChunkMetadata
            var score: SearchScoreBreakdown
        }

        let candidatePoolLimit = candidatePoolLimit(for: query)
        let candidateIDs = preselectCandidateIDs(
            semanticRRF: semanticRRF,
            lexicalRRF: lexicalRRF,
            query: query,
            primaryQueryText: primaryQueryText,
            candidatePoolLimit: candidatePoolLimit
        )
        guard !candidateIDs.isEmpty else { return [] }

        let metadataRows = try await storage.fetchChunkMetadata(chunkIDs: candidateIDs)
        let metadataMap = Dictionary(uniqueKeysWithValues: metadataRows.map { ($0.chunkID, $0) })

        let now = Date()
        let weights = fusionWeights(for: primaryQueryText)
        let anchorSignals = anchorCoverageSignals(for: primaryQueryText)
        var results: [FusedCandidate] = []
        results.reserveCapacity(candidateIDs.count)

        for chunkID in candidateIDs {
            guard let metadata = metadataMap[chunkID] else { continue }

            let semantic = semanticRRF[chunkID] ?? 0
            let lexical = lexicalRRF[chunkID] ?? 0
            let ageDays = max(0, now.timeIntervalSince(metadata.modifiedAt) / 86_400)
            let recency = exp(-ageDays / 30.0)
            let anchorBonus = anchorCoverageBonus(signals: anchorSignals, metadata: metadata)
            let tagBonus = contentTagBonus(queryTags: queryTags, metadata: metadata)
            let schemaBonus = memorySchemaOverlapBonus(querySignals: querySignals, metadata: metadata)
                + ellipticalStructureBonus(querySignals: querySignals, metadata: metadata)
            let temporalBonus = temporalFitBonus(querySignals: querySignals, metadata: metadata)
            let statusBonus = memoryStatusBonus(querySignals: querySignals, metadata: metadata)
            let typeBonus = searchAdjustments.contains(.memoryTypeIntent)
                ? memoryTypeIntentBonus(intent: memoryTypeIntent, metadata: metadata)
                : 0
            let fused = (weights.semantic * semantic)
                + (weights.lexical * lexical)
                + (weights.recency * recency)
                + anchorBonus
                + tagBonus
                + schemaBonus
                + temporalBonus
                + statusBonus
                + typeBonus

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
                        type: typeBonus,
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
            let semantic = semanticRRF[chunkID] ?? 0
            let lexical = lexicalRRF[chunkID] ?? 0
            preliminaryScores[chunkID] = (weights.semantic * semantic) + (weights.lexical * lexical)
        }

        var selected: Set<Int64> = []
        selected.reserveCapacity(hydrationLimit)
        let protectedPerSignal = candidateProtectionLimit(for: query, hydrationLimit: hydrationLimit)
        protectTopCandidates(from: semanticRRF, limit: protectedPerSignal, selected: &selected, hydrationLimit: hydrationLimit)
        protectTopCandidates(from: lexicalRRF, limit: protectedPerSignal, selected: &selected, hydrationLimit: hydrationLimit)

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

        appendExpandedQueries(
            texts: query.additionalLexicalQueries,
            type: .lexical,
            weight: query.additionalLexicalQueryWeight,
            budget: &remainingBudget,
            seen: &seen,
            into: &expandedQueries
        )

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
    ) async throws -> [[Float]?]? {
        guard semanticCandidateLimit > 0 else { return nil }
        guard !queries.isEmpty else { return [] }

        let semanticBranches = queries.enumerated().filter { _, query in
            query.expansionType != .lexical
        }
        guard !semanticBranches.isEmpty else {
            return Array(repeating: nil, count: queries.count)
        }

        let texts = semanticBranches.map { $0.element.text }
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

        var result = Array<[Float]?>(repeating: nil, count: queries.count)
        for (offset, branch) in semanticBranches.enumerated() {
            result[branch.offset] = vectors[offset]
        }

        return result
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

    private func shouldSkipSemanticSearchForScopedQuery(
        query: SearchQuery,
        allowedChunkIDs: Set<Int64>?,
        lexicalProbe: (seededHits: [LexicalHit]?, strongSignal: Bool)
    ) -> Bool {
        guard query.documentPathPrefix != nil else { return false }
        guard query.semanticCandidateLimit > 0, query.lexicalCandidateLimit > 0 else { return false }

        let lexicalCount = lexicalProbe.seededHits?.count ?? 0
        guard lexicalCount > 0 else { return false }

        if lexicalProbe.strongSignal, lexicalCount >= min(query.limit, 8) {
            return true
        }

        let scopedChunkCount = allowedChunkIDs?.count ?? Int.max
        let requiredCoverage = min(query.limit, min(scopedChunkCount, 24))
        return lexicalCount >= max(8, requiredCoverage)
    }

    private func hasStrongLexicalSignal(query: SearchQuery, hits: [LexicalHit]) -> Bool {
        guard query.expansionLimit > 0, configuration.structuredQueryExpander != nil else {
            return false
        }
        guard !shouldRunExpansionDespiteStrongLexicalSignal(query.text) else {
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

    private func shouldRunExpansionDespiteStrongLexicalSignal(_ queryText: String) -> Bool {
        let understanding = RecallQueryUnderstandingAnalyzer.analyze(queryText)
        if understanding.isTemporalOrAggregate {
            return true
        }

        return understanding.operations.contains(.recommendation) || isRecommendationRecallQuery(queryText)
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

    private func applyPostRerankAdjustments(
        to results: [SearchResult],
        querySignals: QueryMatchSignals,
        memoryTypeIntent: RetrievalMemoryTypeIntent,
        query: SearchQuery
    ) -> [SearchResult] {
        guard !results.isEmpty,
              !searchAdjustments.isEmpty,
              query.rerankLimit == 0,
              query.limit >= 5 else {
            return results
        }

        var adjusted = results
        let understanding = querySignals.understanding
        if searchAdjustments.contains(.evidenceSupport),
           shouldAdjustEvidenceSupport(for: understanding) {
            adjusted = applyEvidenceSupportAdjustment(
                to: adjusted,
                querySignals: querySignals,
                query: query
            )
        }
        let canApplyExpansionAdjustments = query.expansionLimit > 0 && query.limit >= 10
        if searchAdjustments.contains(.semanticPreservation),
           canApplyExpansionAdjustments,
           understanding.isProcedural || understanding.operations.contains(.currentState) {
            adjusted = applyExpansionSemanticPreservationAdjustment(
                to: adjusted,
                querySignals: querySignals,
                query: query
            )
        }
        if searchAdjustments.contains(.currentStateLexicalPreservation),
           canApplyExpansionAdjustments,
           understanding.operations.contains(.currentState) {
            adjusted = applyCurrentStateLexicalPreservationAdjustment(
                to: adjusted,
                querySignals: querySignals,
                query: query
            )
        }
        if searchAdjustments.contains(.negatedQualificationRelief),
           query.limit >= 10,
           GenericQueryRewriteLexicon.hasNegatedQualificationIntent(understanding),
           hasLoanOrDebtCue(understanding) {
            adjusted = applyNegatedQualificationReliefAdjustment(
                to: adjusted,
                querySignals: querySignals,
                query: query
            )
        }
        if searchAdjustments.contains(.proceduralRetentionChoice),
           canApplyExpansionAdjustments,
           understanding.isProcedural,
           asksAboutRetentionChoice(understanding) {
            adjusted = applyProceduralRetentionChoiceAdjustment(
                to: adjusted,
                querySignals: querySignals,
                query: query
            )
        }
        if searchAdjustments.contains(.temporalLexicalPreservation),
           canApplyExpansionAdjustments,
           understanding.requiresEvidenceAggregation,
           hasExplicitDurationRecallShape(understanding) {
            adjusted = applyExpansionTemporalLexicalPreservationAdjustment(
                to: adjusted,
                querySignals: querySignals,
                query: query
            )
        }
        if searchAdjustments.contains(.recommendationSemantic),
           understanding.operations.contains(.recommendation) {
            adjusted = applyRecommendationSemanticAdjustment(
                to: adjusted,
                querySignals: querySignals,
                query: query
            )
        }
        if searchAdjustments.contains(.memoryTypeIntent),
           memoryTypeIntent.isInformative {
            adjusted = applyMemoryTypeIntentTailAdjustment(
                to: adjusted,
                intent: memoryTypeIntent,
                query: query
            )
        }
        return adjusted
    }

    private func normalizedFusedScore(_ fused: Double, maxFusedScore: Double) -> Double {
        guard maxFusedScore > 0 else { return min(1, max(0, fused)) }
        return min(1, max(0, fused / maxFusedScore))
    }

    private func preserveAggregateSupportContinuations(
        in results: [SearchResult],
        understanding: RecallQueryUnderstanding,
        effectiveLimit: Int,
        dedupeDocuments: Bool,
        activeOnlyByDefault: Bool
    ) -> [SearchResult] {
        guard dedupeDocuments,
              effectiveLimit >= 10,
              results.count > effectiveLimit else {
            return results
        }

        guard understanding.isEvidenceDense else {
            return results
        }

        let scanFloor = aggregateSupportScanFloor(for: understanding)
        let candidates = supportContinuationCandidates(
            from: results,
            activeOnlyByDefault: activeOnlyByDefault,
            reserveLimit: max(effectiveLimit, scanFloor)
        )
        let topWindow = min(10, effectiveLimit, candidates.count)
        guard topWindow >= 6, candidates.count > topWindow else {
            return results
        }

        var selected = Array(candidates.prefix(topWindow))
        var selectedDocumentKeys = Set(selected.map { normalizedComparisonKey(for: $0.result.documentPath) })
        var selectedGroupCounts = supportGroupCounts(selected)
        let eligibleGroups = Set(
            selected
                .filter { ($0.supportGroupKey).map { selectedGroupCounts[$0] == 1 } ?? false }
                .prefix(6)
                .compactMap(\.supportGroupKey)
        )

        let medianSupport = medianSupportScore(selected)
        let supportFloor = max(0.055, medianSupport * 0.45)
        let scanLimit = min(candidates.count, max(effectiveLimit, scanFloor))

        var promotedGroups: Set<String> = []
        var promotionCount = 0
        let sparseComparisonPromotion = understanding.operations.contains(.comparison)
            && !understanding.requiresEvidenceAggregation
        let promotionLimit = sparseComparisonPromotion ? 3 : 2

        if promotionCount < promotionLimit,
           understanding.requiresEvidenceAggregation,
           !eligibleGroups.isEmpty {
            let anchoredContinuationCandidates = candidates
                .prefix(scanLimit)
                .dropFirst(topWindow)
                .filter { candidate in
                    guard let groupKey = candidate.supportGroupKey,
                          eligibleGroups.contains(groupKey),
                          !selectedDocumentKeys.contains(normalizedComparisonKey(for: candidate.result.documentPath)) else {
                        return false
                    }
                    let score = candidate.result.score
                    return score.lexical >= 0.08
                        && score.temporal > 0
                        && candidate.supportScore >= supportFloor
                }
                .sorted(by: compareSupportContinuationCandidates(_:_:))

            for candidate in anchoredContinuationCandidates where promotionCount < min(promotionLimit, 1) {
                guard let groupKey = candidate.supportGroupKey,
                      promotedGroups.insert(groupKey).inserted,
                      let replacementIndex = aggregateContinuationReplacementIndex(
                        in: selected,
                        groupCounts: selectedGroupCounts,
                        protectedGroupKey: groupKey,
                        candidate: candidate,
                        allowTailReplacement: true,
                        supportRatio: 0.55,
                        blendedRatio: 0.75
                      ) else {
                    continue
                }

                let removed = selected[replacementIndex]
                selectedDocumentKeys.remove(normalizedComparisonKey(for: removed.result.documentPath))
                if let removedGroup = removed.supportGroupKey {
                    selectedGroupCounts[removedGroup] = max(0, (selectedGroupCounts[removedGroup] ?? 0) - 1)
                }

                selected[replacementIndex] = candidate
                selectedDocumentKeys.insert(normalizedComparisonKey(for: candidate.result.documentPath))
                selectedGroupCounts[groupKey] = (selectedGroupCounts[groupKey] ?? 0) + 1
                promotionCount += 1
            }
        }

        if promotionCount < promotionLimit {
            let denseSelectedGroups = Set(
                selectedGroupCounts
                    .filter { $0.value >= 2 }
                    .map(\.key)
            )
            let denseContinuationCandidates = candidates
                .prefix(scanLimit)
                .dropFirst(topWindow)
                .filter { candidate in
                    guard let groupKey = candidate.supportGroupKey,
                          denseSelectedGroups.contains(groupKey),
                          candidate.documentRank <= topWindow + 12,
                          !selectedDocumentKeys.contains(normalizedComparisonKey(for: candidate.result.documentPath)) else {
                        return false
                    }
                    return candidate.supportScore >= supportFloor
                }
                .sorted(by: compareSupportContinuationCandidates(_:_:))

            var promotedDenseGroupCounts: [String: Int] = [:]
            for candidate in denseContinuationCandidates where promotionCount < promotionLimit {
                guard let groupKey = candidate.supportGroupKey else {
                    continue
                }
                let existingGroupCount = selectedGroupCounts[groupKey] ?? 0
                let alreadyPromotedCount = promotedDenseGroupCounts[groupKey] ?? 0
                guard alreadyPromotedCount == 0 || existingGroupCount >= 3 else {
                    continue
                }
                guard let replacementIndex = aggregateContinuationReplacementIndex(
                    in: selected,
                    groupCounts: selectedGroupCounts,
                    protectedGroupKey: groupKey,
                    candidate: candidate,
                    allowTailReplacement: true,
                    supportRatio: 0.60,
                    blendedRatio: 0.80
                ) else {
                    continue
                }

                let removed = selected[replacementIndex]
                selectedDocumentKeys.remove(normalizedComparisonKey(for: removed.result.documentPath))
                if let removedGroup = removed.supportGroupKey {
                    selectedGroupCounts[removedGroup] = max(0, (selectedGroupCounts[removedGroup] ?? 0) - 1)
                }

                selected[replacementIndex] = candidate
                selectedDocumentKeys.insert(normalizedComparisonKey(for: candidate.result.documentPath))
                selectedGroupCounts[groupKey] = (selectedGroupCounts[groupKey] ?? 0) + 1
                promotedDenseGroupCounts[groupKey] = alreadyPromotedCount + 1
                promotionCount += 1
            }
        }

        var usedLateGroupPromotion = false
        if promotionCount < promotionLimit,
           !understanding.isElliptical,
           !understanding.operations.contains(.currentState) {
            let lateGroupPromotions = lateAggregateSupportGroupPromotions(
                in: candidates,
                topWindow: topWindow,
                scanLimit: scanLimit,
                selectedDocumentKeys: selectedDocumentKeys,
                selectedGroupCounts: selectedGroupCounts,
                supportFloor: supportFloor,
                allowSparseGroup: sparseComparisonPromotion,
                understanding: understanding
            )
            for (latePromotionIndex, candidate) in lateGroupPromotions.enumerated() where promotionCount < promotionLimit {
                guard let groupKey = candidate.supportGroupKey,
                      let replacementIndex = aggregateContinuationReplacementIndex(
                        in: selected,
                        groupCounts: selectedGroupCounts,
                        protectedGroupKey: groupKey,
                        candidate: candidate,
                        allowTailReplacement: latePromotionIndex > 0,
                        supportRatio: latePromotionIndex > 0 ? 0.50 : 0.70,
                        blendedRatio: latePromotionIndex > 0 ? 0.75 : 0.86
                      ) else {
                    continue
                }

                let removed = selected[replacementIndex]
                selectedDocumentKeys.remove(normalizedComparisonKey(for: removed.result.documentPath))
                if let removedGroup = removed.supportGroupKey {
                    selectedGroupCounts[removedGroup] = max(0, (selectedGroupCounts[removedGroup] ?? 0) - 1)
                }

                selected[replacementIndex] = candidate
                selectedDocumentKeys.insert(normalizedComparisonKey(for: candidate.result.documentPath))
                selectedGroupCounts[groupKey] = (selectedGroupCounts[groupKey] ?? 0) + 1
                promotionCount += 1
                usedLateGroupPromotion = true
            }
        }

        if promotionCount < promotionLimit, !usedLateGroupPromotion, !eligibleGroups.isEmpty {
            let continuationCandidates = candidates
                .prefix(scanLimit)
                .dropFirst(topWindow)
                .filter { candidate in
                    guard let groupKey = candidate.supportGroupKey,
                          eligibleGroups.contains(groupKey),
                          !selectedDocumentKeys.contains(normalizedComparisonKey(for: candidate.result.documentPath)) else {
                        return false
                    }
                    return candidate.supportScore >= supportFloor
                }
                .sorted(by: compareSupportContinuationCandidates(_:_:))

            for candidate in continuationCandidates where promotionCount < promotionLimit {
                guard let groupKey = candidate.supportGroupKey,
                      promotedGroups.insert(groupKey).inserted,
                      let replacementIndex = aggregateContinuationReplacementIndex(
                        in: selected,
                        groupCounts: selectedGroupCounts,
                        protectedGroupKey: groupKey,
                        candidate: candidate
                      ) else {
                    continue
                }

                let removed = selected[replacementIndex]
                selectedDocumentKeys.remove(normalizedComparisonKey(for: removed.result.documentPath))
                if let removedGroup = removed.supportGroupKey {
                    selectedGroupCounts[removedGroup] = max(0, (selectedGroupCounts[removedGroup] ?? 0) - 1)
                }

                selected[replacementIndex] = candidate
                selectedDocumentKeys.insert(normalizedComparisonKey(for: candidate.result.documentPath))
                selectedGroupCounts[groupKey] = (selectedGroupCounts[groupKey] ?? 0) + 1
                promotionCount += 1
            }
        }

        guard promotionCount > 0 else {
            return results
        }

        let selectedChunkIDs = Set(selected.map { $0.result.chunkID })
        var ordered = selected.map(\.result)
        ordered.reserveCapacity(results.count)
        for result in results where !selectedChunkIDs.contains(result.chunkID) {
            ordered.append(result)
        }
        return ordered
    }

    private func supportContinuationCandidates(
        from results: [SearchResult],
        activeOnlyByDefault: Bool,
        reserveLimit: Int
    ) -> [SupportContinuationCandidate] {
        var seenDocumentKeys: Set<String> = []
        var candidates: [SupportContinuationCandidate] = []
        candidates.reserveCapacity(min(results.count, reserveLimit))

        for (index, result) in results.enumerated() {
            if activeOnlyByDefault,
               let memoryStatus = result.memoryStatus,
               memoryStatus != .active {
                continue
            }

            let documentKey = normalizedComparisonKey(for: result.documentPath)
            guard seenDocumentKeys.insert(documentKey).inserted else { continue }

            let documentRank = candidates.count + 1
            candidates.append(
                SupportContinuationCandidate(
                    result: result,
                    originalIndex: index,
                    documentRank: documentRank,
                    supportScore: evidenceSupportScore(for: result, rank: documentRank),
                    supportGroupKey: supportContinuationGroupKey(for: result.documentPath)
                )
            )

            if candidates.count >= reserveLimit {
                break
            }
        }
        return candidates
    }

    private func lateAggregateSupportGroupPromotions(
        in candidates: [SupportContinuationCandidate],
        topWindow: Int,
        scanLimit: Int,
        selectedDocumentKeys: Set<String>,
        selectedGroupCounts: [String: Int],
        supportFloor: Double,
        allowSparseGroup: Bool,
        understanding: RecallQueryUnderstanding
    ) -> [SupportContinuationCandidate] {
        let lateWindow = candidates
            .prefix(scanLimit)
            .dropFirst(topWindow)

        let candidateSupportFloor = allowSparseGroup ? supportFloor * 0.65 : supportFloor
        var groups: [String: [SupportContinuationCandidate]] = [:]
        for candidate in lateWindow {
            guard let groupKey = candidate.supportGroupKey,
                  selectedGroupCounts[groupKey] == nil,
                  !selectedDocumentKeys.contains(normalizedComparisonKey(for: candidate.result.documentPath)),
                  candidate.supportScore >= candidateSupportFloor else {
                continue
            }
            groups[groupKey, default: []].append(candidate)
        }

        let maxLeadRank = topWindow + 16
        let minimumGroupCount = allowSparseGroup ? 2 : 3
        let secondSupportFloor = allowSparseGroup ? supportFloor * 0.85 : supportFloor
        var groupedPromotions = groups.values
            .map { $0.sorted(by: compareSupportContinuationCandidates(_:_:)) }
            .filter { group in
                guard group.count >= minimumGroupCount,
                      let lead = group.first else {
                    return false
                }
                if allowSparseGroup {
                    return lead.documentRank <= maxLeadRank
                }
                guard let second = group.dropFirst().first else {
                    return false
                }
                return lead.documentRank <= maxLeadRank
                    && second.supportScore >= secondSupportFloor
            }

        if allowSparseGroup {
            return groupedPromotions
                .map { group in
                    (
                        group: group,
                        coverage: supportGroupCoreTermCoverageScore(for: group, understanding: understanding)
                    )
                }
                .sorted { lhs, rhs in
                    guard let lhsLead = lhs.group.first, let rhsLead = rhs.group.first else { return false }
                    if lhs.coverage != rhs.coverage {
                        return lhs.coverage > rhs.coverage
                    }
                    if lhsLead.documentRank == rhsLead.documentRank {
                        return lhsLead.supportScore > rhsLead.supportScore
                    }
                    return lhsLead.documentRank < rhsLead.documentRank
                }
                .prefix(3)
                .compactMap { $0.group.first }
        }

        groupedPromotions.sort { lhs, rhs in
            guard let lhsLead = lhs.first, let rhsLead = rhs.first else { return false }
            if lhsLead.supportScore == rhsLead.supportScore {
                return lhsLead.documentRank < rhsLead.documentRank
            }
            return lhsLead.supportScore > rhsLead.supportScore
        }

        guard let group = groupedPromotions.first else { return [] }
        let semanticRanked = group
            .prefix(3)
            .sorted(by: compareLateAggregateSupportCandidate(_:_:))
        guard let first = semanticRanked.first else { return [] }

        var promotions = [first]
        if let second = semanticRanked.dropFirst().first,
           second.result.score.semantic >= 0.038,
           second.result.score.semantic >= first.result.score.semantic * 0.75 {
            promotions.append(second)
        }
        return promotions
    }

    private func supportGroupCoreTermCoverageScore(
        for group: [SupportContinuationCandidate],
        understanding: RecallQueryUnderstanding
    ) -> Int {
        let terms = understanding.coreTerms.filter { term in
            term.count >= 3 && !sparseComparisonCoverageStopTerms.contains(term)
        }
        guard !terms.isEmpty else { return 0 }

        let text = group
            .map { searchableAdjustmentText(for: $0.result) }
            .joined(separator: " ")
        return terms.reduce(into: 0) { score, term in
            if text.contains(term) {
                score += 1
            }
        }
    }

    private func compareLateAggregateSupportCandidate(
        _ lhs: SupportContinuationCandidate,
        _ rhs: SupportContinuationCandidate
    ) -> Bool {
        if lhs.result.score.semantic == rhs.result.score.semantic {
            return compareSupportContinuationCandidates(lhs, rhs)
        }
        return lhs.result.score.semantic > rhs.result.score.semantic
    }

    private func supportGroupCounts(_ candidates: [SupportContinuationCandidate]) -> [String: Int] {
        var counts: [String: Int] = [:]
        for candidate in candidates {
            guard let groupKey = candidate.supportGroupKey else { continue }
            counts[groupKey, default: 0] += 1
        }
        return counts
    }

    private func aggregateContinuationReplacementIndex(
        in selected: [SupportContinuationCandidate],
        groupCounts: [String: Int],
        protectedGroupKey: String,
        candidate: SupportContinuationCandidate,
        allowTailReplacement: Bool = false,
        supportRatio: Double = 0.70,
        blendedRatio: Double = 0.86
    ) -> Int? {
        selected.enumerated()
            .filter { index, existing in
                guard index >= 4, index < (allowTailReplacement ? 10 : 9) else { return false }
                if let groupKey = existing.supportGroupKey {
                    guard groupKey != protectedGroupKey else { return false }
                    guard (groupCounts[groupKey] ?? 0) <= 1 else { return false }
                }
                return candidate.supportScore >= existing.supportScore * supportRatio
                    || candidate.result.score.blended >= existing.result.score.blended * blendedRatio
            }
            .min { lhs, rhs in
                if lhs.element.supportScore == rhs.element.supportScore {
                    return lhs.offset > rhs.offset
                }
                return lhs.element.supportScore < rhs.element.supportScore
            }?
            .offset
    }

    private func compareSupportContinuationCandidates(
        _ lhs: SupportContinuationCandidate,
        _ rhs: SupportContinuationCandidate
    ) -> Bool {
        if lhs.supportScore == rhs.supportScore {
            if lhs.documentRank == rhs.documentRank {
                return lhs.originalIndex < rhs.originalIndex
            }
            return lhs.documentRank < rhs.documentRank
        }
        return lhs.supportScore > rhs.supportScore
    }

    private func medianSupportScore(_ candidates: [SupportContinuationCandidate]) -> Double {
        guard !candidates.isEmpty else { return 0 }
        let sorted = candidates.map(\.supportScore).sorted()
        return sorted[sorted.count / 2]
    }

    private func supportContinuationGroupKey(for documentPath: String) -> String? {
        let normalizedPath = documentPath
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
            .replacingOccurrences(of: "\\", with: "/")
        guard !normalizedPath.hasPrefix("memory://"),
              let fileName = normalizedPath.split(separator: "/").last else {
            return nil
        }

        let stem = String(fileName).replacingOccurrences(
            of: #"\.[a-z0-9]+$"#,
            with: "",
            options: .regularExpression
        )
        let groupedStem = stem.replacingOccurrences(
            of: #"(?:[-_](?:part|section|chunk))?[-_]\d+$"#,
            with: "",
            options: .regularExpression
        )
        guard groupedStem != stem,
              groupedStem.count >= 3 else {
            return nil
        }

        let directory = normalizedPath
            .split(separator: "/")
            .dropLast()
            .joined(separator: "/")
        return directory.isEmpty ? groupedStem : "\(directory)/\(groupedStem)"
    }

    private func applyEvidenceSupportAdjustment(
        to results: [SearchResult],
        querySignals: QueryMatchSignals,
        query: SearchQuery
    ) -> [SearchResult] {
        guard query.rerankLimit == 0,
              query.limit >= 5,
              !results.isEmpty,
              shouldAdjustEvidenceSupport(for: querySignals.understanding) else {
            return results
        }

        var adjusted = results
        let window = min(adjusted.count, max(query.limit * 3, 24))
        let leadingDateCount = min(10, adjusted.count)
        let useTemporalNeighborBonus = querySignals.understanding.operations.contains(.ordering)
            || querySignals.understanding.operations.contains(.comparison)
        let leadingDates = useTemporalNeighborBonus
            ? adjusted.prefix(leadingDateCount).compactMap { explicitDocumentDate(in: $0.content) }
            : []
        let temporalNeighborSupportThreshold = querySignals.understanding.requiresEvidenceAggregation ? 0.07 : 0.09
        let supportBonusScale = 0.12
        let supportBonusCap = 0.018
        let topAnchorScore = adjusted.first?.score.blended
        for index in adjusted.indices.prefix(window) {
            let support = evidenceSupportScore(for: adjusted[index], rank: index + 1)
            let temporalNeighborBonus = useTemporalNeighborBonus && index >= leadingDateCount && support >= temporalNeighborSupportThreshold
                ? temporalNeighborEvidenceBonus(for: adjusted[index], leadingDates: leadingDates)
                : 0
            guard support > 0 else { continue }
            let adjustedScore = adjusted[index].score.blended
                + min(supportBonusCap, supportBonusScale * support)
                + temporalNeighborBonus
            if index > 0, let topAnchorScore {
                adjusted[index].score.blended = min(adjustedScore, topAnchorScore.nextDown)
            } else {
                adjusted[index].score.blended = adjustedScore
            }
        }
        return adjusted
    }

    private func shouldAdjustEvidenceSupport(for understanding: RecallQueryUnderstanding) -> Bool {
        understanding.requiresEvidenceAggregation
            || understanding.operations.contains(.ordering)
            || understanding.operations.contains(.comparison)
    }

    private func evidenceSupportScore(for result: SearchResult, rank: Int) -> Double {
        let score = result.score
        let strongestBranch = max(score.lexical, score.semantic)
        let branchAgreement = min(score.lexical, score.semantic)
        let metadataSupport = score.temporal + score.schema + score.tag + score.status
        let rankPrior = 0.012 / sqrt(Double(max(1, rank)))
        return strongestBranch + (0.35 * branchAgreement) + metadataSupport + rankPrior
    }

    private func temporalNeighborEvidenceBonus(for result: SearchResult, leadingDates: [Date]) -> Double {
        guard !leadingDates.isEmpty,
              let candidateDate = explicitDocumentDate(in: result.content) else {
            return 0
        }

        let nearestDistance = leadingDates
            .map { abs(candidateDate.timeIntervalSince($0)) / 86_400 }
            .min() ?? .greatestFiniteMagnitude

        if nearestDistance <= 1 {
            return 0.006
        }
        if nearestDistance <= 3 {
            return 0.0045
        }
        if nearestDistance <= 7 {
            return 0.003
        }
        if nearestDistance <= 14 {
            return 0.0015
        }
        return 0
    }

    private func explicitDocumentDate(in content: String) -> Date? {
        let prefix = String(content.prefix(160))
        let pattern = #"Date:\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return nil
        }
        let range = NSRange(prefix.startIndex..<prefix.endIndex, in: prefix)
        guard let match = regex.firstMatch(in: prefix, options: [], range: range),
              match.numberOfRanges == 4,
              let yearRange = Range(match.range(at: 1), in: prefix),
              let monthRange = Range(match.range(at: 2), in: prefix),
              let dayRange = Range(match.range(at: 3), in: prefix),
              let year = Int(prefix[yearRange]),
              let month = Int(prefix[monthRange]),
              let day = Int(prefix[dayRange]) else {
            return nil
        }

        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? calendar.timeZone
        return calendar.date(from: DateComponents(year: year, month: month, day: day))
    }

    private func applyRecommendationSemanticAdjustment(
        to results: [SearchResult],
        querySignals: QueryMatchSignals,
        query: SearchQuery
    ) -> [SearchResult] {
        guard query.rerankLimit == 0,
              query.limit >= 5,
              querySignals.understanding.operations.contains(.recommendation),
              !results.isEmpty else {
            return results
        }

        var adjusted = results
        let window = min(adjusted.count, max(query.limit * 3, 24))
        for index in adjusted.indices.prefix(window) {
            let score = adjusted[index].score
            guard score.semantic >= 0.055,
                  score.lexical > 0,
                  score.semantic > score.lexical else {
                continue
            }
            let semanticLead = score.semantic - score.lexical
            adjusted[index].score.blended += min(0.004, semanticLead * 0.10)
        }
        return adjusted
    }

    private func applyMemoryTypeIntentTailAdjustment(
        to results: [SearchResult],
        intent: RetrievalMemoryTypeIntent,
        query: SearchQuery
    ) -> [SearchResult] {
        guard query.rerankLimit == 0,
              query.limit >= 5,
              !results.isEmpty else {
            return results
        }

        var adjusted = results
        let window = min(adjusted.count, max(query.limit * 3, 24))
        for index in adjusted.indices.prefix(window) {
            let fit = memoryTypeIntentFit(intent: intent, result: adjusted[index])
            guard fit > 0 else { continue }

            let branchEvidence = max(adjusted[index].score.lexical, adjusted[index].score.semantic)
            guard branchEvidence > 0 else { continue }

            let bonus = min(0.0015, adjusted[index].score.type * 0.35 + fit * 0.0008)
            guard bonus > 0 else { continue }
            adjusted[index].score.type += bonus
            adjusted[index].score.blended += bonus
        }
        return adjusted
    }

    private func memoryTypeIntentFit(intent: RetrievalMemoryTypeIntent, result: SearchResult) -> Double {
        let labels = retrievalMemoryTypeLabels(for: result)
        guard !labels.isEmpty else { return 0 }

        var bestFit = 0.0
        for label in labels {
            if label.name == intent.label {
                bestFit = max(bestFit, label.confidence)
            } else if intent.compatibleLabels.contains(label.name) {
                bestFit = max(bestFit, label.confidence * 0.20)
            }
        }
        return bestFit
    }

    private func retrievalMemoryTypeLabels(for result: SearchResult) -> [(name: String, confidence: Double)] {
        var labels: [String: Double] = [:]

        if let label = normalizedRetrievalMemoryType(result.memoryType) {
            let confidence = min(1, max(0.35, result.memoryTypeConfidence ?? 0.65))
            labels[label] = max(labels[label] ?? 0, confidence)
        }

        if let kind = result.memoryKind,
           let kindLabel = retrievalMemoryTypeLabel(for: kind) {
            labels[kindLabel] = max(labels[kindLabel] ?? 0, 0.70)
        }

        return labels
            .map { (name: $0.key, confidence: $0.value) }
            .sorted { lhs, rhs in
                if lhs.confidence == rhs.confidence {
                    return lhs.name < rhs.name
                }
                return lhs.confidence > rhs.confidence
            }
    }

    private func applyNegatedQualificationReliefAdjustment(
        to results: [SearchResult],
        querySignals: QueryMatchSignals,
        query: SearchQuery
    ) -> [SearchResult] {
        guard query.rerankLimit == 0,
              query.limit >= 10,
              !results.isEmpty,
              GenericQueryRewriteLexicon.hasNegatedQualificationIntent(querySignals.understanding),
              hasLoanOrDebtCue(querySignals.understanding) else {
            return results
        }

        var adjusted = results
        let window = min(adjusted.count, max(80, min(query.limit, 160)))
        for index in adjusted.indices.prefix(window) {
            let text = searchableAdjustmentText(for: adjusted[index])
            let hasCertificationSignal = text.contains("false")
                && (text.contains("certified") || text.contains("certification"))
            let hasDisqualificationSignal = text.contains("disqualif")
            let hasReliefSignal = text.contains("discharge")
                || text.contains("forgiveness")
                || text.contains("forgiven")
                || text.contains("cancel")
            guard (hasCertificationSignal || hasDisqualificationSignal) && hasReliefSignal else {
                continue
            }

            var bonus = 0.0
            if hasCertificationSignal {
                bonus += 0.030
            }
            if hasDisqualificationSignal {
                bonus += 0.020
            }
            if hasReliefSignal {
                bonus += 0.010
            }
            adjusted[index].score.blended += min(0.060, bonus)
        }
        return adjusted
    }

    private func applyProceduralRetentionChoiceAdjustment(
        to results: [SearchResult],
        querySignals: QueryMatchSignals,
        query: SearchQuery
    ) -> [SearchResult] {
        guard query.rerankLimit == 0,
              query.expansionLimit > 0,
              query.limit >= 10,
              querySignals.understanding.isProcedural,
              asksAboutRetentionChoice(querySignals.understanding),
              !results.isEmpty else {
            return results
        }

        var adjusted = results
        let window = min(adjusted.count, max(80, min(query.limit, 160)))
        for index in adjusted.indices.prefix(window) {
            let text = searchableAdjustmentText(for: adjusted[index])
            let hasStorageSignal = text.contains("store")
                || text.contains("stored")
                || text.contains("storage")
                || text.contains("keep")
                || text.contains("retain")
            let hasReturnSignal = text.contains("surrender")
                || text.contains("return")
                || text.contains("submit")
                || text.contains("deliver")
            guard hasStorageSignal && hasReturnSignal else {
                continue
            }

            var bonus = 0.006
            let title = (adjusted[index].title ?? "").lowercased()
            if title.contains("store") || title.contains("surrender") || title.contains("return") {
                bonus += 0.006
            }
            adjusted[index].score.blended += min(0.014, bonus)
        }
        return adjusted
    }

    private func applyExpansionTemporalLexicalPreservationAdjustment(
        to results: [SearchResult],
        querySignals: QueryMatchSignals,
        query: SearchQuery
    ) -> [SearchResult] {
        guard query.rerankLimit == 0,
              query.expansionLimit > 0,
              query.limit >= 10,
              querySignals.understanding.requiresEvidenceAggregation,
              hasExplicitDurationRecallShape(querySignals.understanding),
              !results.isEmpty else {
            return results
        }

        var adjusted = results
        let window = min(adjusted.count, max(80, min(query.limit, 160)))
        for index in adjusted.indices.prefix(window) {
            let score = adjusted[index].score
            guard score.lexical >= 0.055,
                  score.temporal > 0 else {
                continue
            }

            let bonus = min(0.018, (score.lexical * 0.08) + (score.temporal * 0.45))
            adjusted[index].score.blended += bonus
        }
        return adjusted
    }

    private func applyExpansionSemanticPreservationAdjustment(
        to results: [SearchResult],
        querySignals: QueryMatchSignals,
        query: SearchQuery
    ) -> [SearchResult] {
        guard query.rerankLimit == 0,
              query.expansionLimit > 0,
              query.limit >= 10,
              !results.isEmpty,
              (
                querySignals.understanding.isProcedural
                    || querySignals.understanding.operations.contains(.currentState)
              ) else {
            return results
        }

        var adjusted = results
        let preservedK = min(10, query.limit)
        let window = min(adjusted.count, max(preservedK * 4, 40))
        let semanticRanked = adjusted.indices.prefix(window)
            .filter { index in
                guard adjusted[index].score.semantic >= 0.025 else {
                    return false
                }
                return true
            }
            .sorted { lhs, rhs in
                let lhsScore = adjusted[lhs].score
                let rhsScore = adjusted[rhs].score
                if lhsScore.semantic == rhsScore.semantic {
                    return lhsScore.blended > rhsScore.blended
                }
                return lhsScore.semantic > rhsScore.semantic
            }
        let protectedCount: Int
        if querySignals.understanding.isProcedural {
            protectedCount = min(preservedK, 10, semanticRanked.count)
        } else {
            protectedCount = min(max(0, preservedK - 2), 8, semanticRanked.count)
        }
        guard protectedCount > 0 else { return results }

        let blendedCutoff = adjusted
            .prefix(window)
            .map(\.score.blended)
            .sorted(by: >)[min(preservedK - 1, window - 1)]
        for semanticRank in 0..<protectedCount {
            let index = semanticRanked[semanticRank]
            let tieBreaker = Double(protectedCount - semanticRank) * 0.000_001
            adjusted[index].score.blended = max(adjusted[index].score.blended, blendedCutoff + tieBreaker)
        }
        return adjusted
    }

    private func applyCurrentStateLexicalPreservationAdjustment(
        to results: [SearchResult],
        querySignals: QueryMatchSignals,
        query: SearchQuery
    ) -> [SearchResult] {
        guard query.rerankLimit == 0,
              query.expansionLimit > 0,
              query.limit >= 10,
              querySignals.understanding.operations.contains(.currentState),
              !results.isEmpty else {
            return results
        }

        var adjusted = results
        let preservedK = min(10, query.limit)
        let window = min(adjusted.count, max(preservedK * 4, 40))
        let leadingGroups = Set(
            adjusted.prefix(preservedK)
                .compactMap { supportContinuationGroupKey(for: $0.documentPath) }
        )
        guard !leadingGroups.isEmpty else { return results }

        let lexicalRanked = adjusted.indices.prefix(window)
            .filter { index in
                let score = adjusted[index].score
                guard score.lexical >= 0.075,
                      score.semantic > 0,
                      let groupKey = supportContinuationGroupKey(for: adjusted[index].documentPath) else {
                    return false
                }
                return index >= preservedK && leadingGroups.contains(groupKey)
            }
            .sorted { lhs, rhs in
                let lhsScore = adjusted[lhs].score
                let rhsScore = adjusted[rhs].score
                if lhsScore.lexical == rhsScore.lexical {
                    return lhsScore.blended > rhsScore.blended
                }
                return lhsScore.lexical > rhsScore.lexical
            }
        let protectedCount = min(3, lexicalRanked.count)
        guard protectedCount > 0 else { return results }

        let blendedCutoff = adjusted
            .prefix(window)
            .map(\.score.blended)
            .sorted(by: >)[min(preservedK - 1, window - 1)]
        for lexicalRank in 0..<protectedCount {
            let index = lexicalRanked[lexicalRank]
            let tieBreaker = Double(protectedCount - lexicalRank) * 0.000_001
            adjusted[index].score.blended = max(adjusted[index].score.blended, blendedCutoff + tieBreaker)
        }
        return adjusted
    }

    private func searchableAdjustmentText(for result: SearchResult) -> String {
        [
            result.title ?? "",
            result.snippet,
            String(result.content.prefix(2_000)),
        ]
        .joined(separator: " ")
        .lowercased()
    }

    private func hasLoanOrDebtCue(_ understanding: RecallQueryUnderstanding) -> Bool {
        !Set(understanding.tokens).isDisjoint(with: [
            "loan", "loans", "debt", "debts", "lender", "borrower", "borrowers",
        ])
    }

    private func asksAboutRetentionChoice(_ understanding: RecallQueryUnderstanding) -> Bool {
        let tokens = Set(understanding.tokens)
        let retentionTerms: Set<String> = ["keep", "kept", "retain", "retained", "store", "stored", "hold"]
        let returnTerms: Set<String> = ["return", "returned", "surrender", "surrendered", "deliver", "send", "submit"]
        return !tokens.isDisjoint(with: retentionTerms)
            && !tokens.isDisjoint(with: returnTerms)
    }

    private func hasExplicitDurationRecallShape(_ understanding: RecallQueryUnderstanding) -> Bool {
        let text = understanding.normalizedText
        return text.contains("days ago")
            || text.contains("how long")
            || text.contains("time passed")
            || text.contains("days passed")
            || text.contains("since ")
            || text.contains("duration")
    }

    private let sparseComparisonCoverageStopTerms: Set<String> = [
        "what", "when", "where", "which", "time", "day", "days", "before",
        "after", "between", "past", "month", "months", "week", "weeks",
        "year", "years", "last", "next", "different",
    ]

    private func classifyRetrievalMemoryTypeIntent(_ understanding: RecallQueryUnderstanding) -> RetrievalMemoryTypeIntent {
        let normalized = normalizedClassifierText(understanding.originalText)
        let tokens = Set(understanding.tokens)

        if containsAnyNormalizedPhrase([
                "current status",
                "what changed",
                "changed since",
                "right now",
                "as of",
                "terms of service",
                "policy",
                "status",
                "still active",
                "currently active",
                "terms",
            ], in: normalized) {
            return RetrievalMemoryTypeIntent(
                label: "contextual",
                confidence: 0.78,
                compatibleLabels: ["factual", "semantic"]
            )
        }

        if understanding.isProcedural
            || containsAnyNormalizedPhrase([
                "how do i",
                "how can i",
                "how to",
                "what steps",
                "steps to",
                "process to",
                "set up",
                "apply for",
                "renew",
                "register",
                "submit",
                "file an",
                "file a",
            ], in: normalized) {
            return RetrievalMemoryTypeIntent(
                label: "procedural",
                confidence: 0.80,
                compatibleLabels: ["factual", "contextual"]
            )
        }

        if understanding.operations.contains(.recency)
            || understanding.operations.contains(.ordering)
            || containsAnyNormalizedPhrase([
                "when did i",
                "when was",
                "what happened",
                "last time",
                "timeline",
                "earliest",
                "latest",
                "first to last",
            ], in: normalized) {
            return RetrievalMemoryTypeIntent(
                label: "episodic",
                confidence: 0.50,
                compatibleLabels: ["contextual", "factual"]
            )
        }

        if understanding.operations.contains(.duration)
            || understanding.operations.contains(.comparison) {
            return RetrievalMemoryTypeIntent(
                label: "episodic",
                confidence: 0.50,
                compatibleLabels: ["factual", "contextual"]
            )
        }

        if containsAnyNormalizedPhrase([
            "what is",
            "what are",
            "why",
            "explain",
            "difference between",
            "how does",
            "benefits",
            "meaning of",
            "concept",
        ], in: normalized)
            || !tokens.isDisjoint(with: ["definition", "definitions", "meaning", "concept", "explain", "benefits"]) {
            return RetrievalMemoryTypeIntent(
                label: "semantic",
                confidence: 0.70,
                compatibleLabels: ["factual", "contextual"]
            )
        }

        return RetrievalMemoryTypeIntent(
            label: "factual",
            confidence: 0.50,
            compatibleLabels: ["semantic", "contextual"]
        )
    }

    private func memoryTypeIntentBonus(
        intent: RetrievalMemoryTypeIntent,
        metadata: StoredChunkMetadata
    ) -> Double {
        guard intent.isInformative else { return 0 }
        let labels = retrievalMemoryTypeLabels(for: metadata)
        guard !labels.isEmpty else { return 0 }

        var bestFit = 0.0
        for label in labels {
            let relationship: Double
            if label.name == intent.label {
                relationship = 1.0
            } else if intent.compatibleLabels.contains(label.name) {
                relationship = 0.20
            } else {
                relationship = 0
            }
            bestFit = max(bestFit, relationship * label.confidence)
        }
        guard bestFit > 0 else { return 0 }

        let base = intent.label == "factual" ? 0.002 : 0.005
        return min(0.006, base * intent.confidence * bestFit)
    }

    private func retrievalMemoryTypeLabels(for metadata: StoredChunkMetadata) -> [(name: String, confidence: Double)] {
        var labels: [String: Double] = [:]

        if let label = normalizedRetrievalMemoryType(metadata.memoryType) {
            let confidence = min(1, max(0.35, metadata.memoryTypeConfidence ?? 0.65))
            labels[label] = max(labels[label] ?? 0, confidence)
        }

        if let kind = resolveMemoryKind(from: metadata),
           let kindLabel = retrievalMemoryTypeLabel(for: kind) {
            labels[kindLabel] = max(labels[kindLabel] ?? 0, 0.70)
        }

        return labels
            .map { (name: $0.key, confidence: $0.value) }
            .sorted { lhs, rhs in
                if lhs.confidence == rhs.confidence {
                    return lhs.name < rhs.name
                }
                return lhs.confidence > rhs.confidence
            }
    }

    private func normalizedRetrievalMemoryType(_ rawValue: String?) -> String? {
        guard let rawValue else { return nil }
        let normalized = rawValue
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        switch normalized {
        case "fact", "factual":
            return "factual"
        case "procedure", "procedural":
            return "procedural"
        case "episode", "episodic", "temporal":
            return "episodic"
        case "semantic":
            return "semantic"
        case "context", "contextual":
            return "contextual"
        default:
            return nil
        }
    }

    private func retrievalMemoryTypeLabel(for kind: MemoryKind) -> String? {
        switch kind {
        case .profile, .fact, .decision:
            return "factual"
        case .commitment, .procedure:
            return "procedural"
        case .episode:
            return "episodic"
        case .handoff:
            return "contextual"
        }
    }

    private func fusionWeights(for queryText: String) -> (semantic: Double, lexical: Double, recency: Double) {
        if isTimeAnchoredQuery(queryText) {
            return (semantic: 0.64, lexical: 0.35, recency: 0.01)
        }
        return (semantic: 0.62, lexical: 0.33, recency: 0.05)
    }

    private func isTimeAnchoredQuery(_ queryText: String) -> Bool {
        MemorySearchHeuristics.isTimeAnchoredQuery(queryText)
    }

    private func anchorCoverageSignals(for queryText: String) -> AnchorCoverageSignals {
        AnchorCoverageSignals(
            anchors: anchorTokens(from: queryText),
            quotedPhrases: quotedPhraseAnchors(from: queryText)
        )
    }

    private func anchorCoverageBonus(signals: AnchorCoverageSignals, metadata: StoredChunkMetadata) -> Double {
        let searchable = ((metadata.title ?? "") + " " + String(metadata.content.prefix(800)))
            .lowercased()
        let phraseBonus = quotedPhraseCoverageBonus(phrases: signals.quotedPhrases, searchable: searchable)
        guard !signals.anchors.isEmpty else { return phraseBonus }

        var matched = 0
        for anchor in signals.anchors where searchable.contains(anchor) {
            matched += 1
        }
        guard matched > 0 else { return phraseBonus }

        let coverage = Double(matched) / Double(signals.anchors.count)
        return phraseBonus + (0.003 * coverage)
    }

    private func quotedPhraseCoverageBonus(phrases: [String], searchable: String) -> Double {
        guard !phrases.isEmpty else { return 0 }

        var matched = 0
        for phrase in phrases where searchable.contains(phrase) {
            matched += 1
        }
        guard matched > 0 else { return 0 }

        let coverage = Double(matched) / Double(phrases.count)
        return min(0.035, (0.018 * Double(matched)) + (0.017 * coverage))
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

    private func ellipticalStructureBonus(
        querySignals: QueryMatchSignals,
        metadata: StoredChunkMetadata
    ) -> Double {
        let understanding = querySignals.understanding
        guard shouldUseEllipticalStructureBonus(for: understanding) else {
            return 0
        }

        let queryTerms = Set(understanding.coreTerms)
        guard !queryTerms.isEmpty else { return 0 }

        let titleText = [metadata.title, filenameStem(metadata.documentPath)]
            .compactMap { $0 }
            .joined(separator: " ")
        let titleTokens = Set(searchableTokens(in: titleText))
        let headingTokens = Set(searchableTokens(in: structuralLeadText(from: metadata.content)))
        let titleOverlap = overlapRatio(queryTerms: queryTerms, candidateTerms: titleTokens)
        let headingOverlap = overlapRatio(queryTerms: queryTerms, candidateTerms: headingTokens)

        var bonus = 0.0
        if titleOverlap > 0 {
            bonus += min(0.018, titleOverlap * 0.020)
        }
        if headingOverlap > 0 {
            bonus += min(0.018, headingOverlap * 0.020)
        }

        let searchable = structuralLeadText(from: metadata.content).lowercased()
        let directMatches = understanding.coreTerms.filter { searchable.contains($0) }.count
        if directMatches > 0 {
            bonus += min(0.012, 0.006 * Double(directMatches))
        }

        return min(0.035, bonus)
    }

    private func shouldUseEllipticalStructureBonus(for understanding: RecallQueryUnderstanding) -> Bool {
        guard understanding.isElliptical, understanding.tokens.count <= 7 else {
            return false
        }
        let tokenSet = Set(understanding.tokens)
        if !tokenSet.isDisjoint(with: ["it", "that", "this", "they", "them", "those", "these"]) {
            return true
        }
        let lower = understanding.originalText.lowercased()
        return lower.hasPrefix("how do i apply") || lower.hasPrefix("can i do it")
    }

    private func temporalFitBonus(querySignals: QueryMatchSignals, metadata: StoredChunkMetadata) -> Double {
        let anchorBonus = timeAnchorFitBonus(querySignals: querySignals, metadata: metadata)
        switch querySignals.temporalIntent {
        case .any:
            return anchorBonus
        case .recent, .mostRecent:
            let ageDays = max(0, Date().timeIntervalSince(metadata.modifiedAt) / 86_400)
            return anchorBonus + min(0.05, 0.05 * exp(-ageDays / 14.0))
        case .historical:
            let historicalBonus = metadata.memoryStatus == MemoryStatus.superseded.rawValue || metadata.memoryStatus == MemoryStatus.archived.rawValue ? 0.04 : 0
            return anchorBonus + historicalBonus
        case .timeAnchored, .count:
            return anchorBonus + (isTimeAnchoredText(metadata.content) ? 0.025 : 0)
        }
    }

    private func filenameStem(_ path: String) -> String {
        URL(fileURLWithPath: path)
            .deletingPathExtension()
            .lastPathComponent
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
    }

    private func structuralLeadText(from content: String) -> String {
        let lines = content
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var selected: [String] = []
        for line in lines.prefix(18) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            if trimmed.hasPrefix("#")
                || trimmed.range(of: #"^\d+\."#, options: .regularExpression) != nil
                || trimmed.count <= 120 {
                selected.append(trimmed)
            }
        }
        if selected.isEmpty {
            return String(content.prefix(700))
        }
        return selected.joined(separator: " ")
    }

    private func searchableTokens(in text: String) -> [String] {
        MemorySearchHeuristics.normalizedComparisonKey(for: text)
            .split(separator: " ")
            .map(String.init)
            .filter { token in
                token.count >= 2 && !MemorySearchHeuristics.queryStopWords.contains(token)
            }
    }

    private func overlapRatio(queryTerms: Set<String>, candidateTerms: Set<String>) -> Double {
        guard !queryTerms.isEmpty, !candidateTerms.isEmpty else { return 0 }
        let overlap = queryTerms.intersection(candidateTerms).count
        guard overlap > 0 else { return 0 }
        return Double(overlap) / Double(min(queryTerms.count, max(1, candidateTerms.count)))
    }

    private func timeAnchorFitBonus(querySignals: QueryMatchSignals, metadata: StoredChunkMetadata) -> Double {
        let searchable = ((metadata.title ?? "") + " " + String(metadata.content.prefix(800)))
            .lowercased()

        if !querySignals.monthDayAnchors.isEmpty {
            var matched = 0
            for anchor in querySignals.monthDayAnchors where textContains(anchor: anchor, text: searchable) {
                matched += 1
            }
            guard matched > 0 else { return 0 }

            let coverage = Double(matched) / Double(querySignals.monthDayAnchors.count)
            return min(0.09, (0.035 * Double(matched)) + (0.045 * coverage))
        }

        guard !querySignals.monthAnchors.isEmpty else { return 0 }
        for month in querySignals.monthAnchors where textContains(month: month, text: searchable) {
            return 0.025
        }
        return 0
    }

    private func monthDayAnchors(from text: String) -> Set<MonthDayAnchor> {
        let lower = text.lowercased()
        var anchors: Set<MonthDayAnchor> = []
        let monthAlternation = Self.monthNameToNumber.keys.sorted().joined(separator: "|")

        let rangePattern = #"\b("# + monthAlternation + #")\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|through|-)\s*(\d{1,2})(?:st|nd|rd|th)?\b"#
        for match in regexCaptureGroups(pattern: rangePattern, text: lower) {
            guard match.count >= 3,
                  let month = Self.monthNameToNumber[match[0]],
                  let startDay = Int(match[1]),
                  let endDay = Int(match[2]) else {
                continue
            }
            for day in min(startDay, endDay)...max(startDay, endDay) where (1...31).contains(day) {
                anchors.insert(MonthDayAnchor(month: month, day: day))
            }
        }

        let explicitPattern = #"\b("# + monthAlternation + #")\s+(\d{1,2})(?:st|nd|rd|th)?\b"#
        for match in regexCaptureGroups(pattern: explicitPattern, text: lower) {
            guard match.count >= 2,
                  let month = Self.monthNameToNumber[match[0]],
                  let day = Int(match[1]),
                  (1...31).contains(day) else {
                continue
            }
            anchors.insert(MonthDayAnchor(month: month, day: day))
        }

        return anchors
    }

    private func monthAnchors(from text: String) -> Set<Int> {
        let lower = text.lowercased()
        var months: Set<Int> = []
        for (name, number) in Self.monthNameToNumber where lower.range(of: #"\b\#(name)\b"#, options: .regularExpression) != nil {
            months.insert(number)
        }
        if lower.range(of: #"\bsummer\b"#, options: .regularExpression) != nil {
            months.formUnion([6, 7, 8])
        }
        if lower.range(of: #"\bspring\b"#, options: .regularExpression) != nil {
            months.formUnion([3, 4, 5])
        }
        if lower.range(of: #"\b(autumn|fall)\b"#, options: .regularExpression) != nil {
            months.formUnion([9, 10, 11])
        }
        if lower.range(of: #"\bwinter\b"#, options: .regularExpression) != nil {
            months.formUnion([12, 1, 2])
        }
        return months
    }

    private func textContains(anchor: MonthDayAnchor, text: String) -> Bool {
        let iso = String(format: "-%02d-%02d", anchor.month, anchor.day)
        if text.contains(iso) {
            return true
        }

        guard let monthName = Self.monthNameByNumber[anchor.month] else {
            return false
        }
        return text.contains("\(monthName) \(anchor.day)")
            || text.contains("\(monthName) \(anchor.day)st")
            || text.contains("\(monthName) \(anchor.day)nd")
            || text.contains("\(monthName) \(anchor.day)rd")
            || text.contains("\(monthName) \(anchor.day)th")
    }

    private func textContains(month: Int, text: String) -> Bool {
        let iso = String(format: "-%02d-", month)
        if text.contains(iso) {
            return true
        }
        guard let monthName = Self.monthNameByNumber[month] else {
            return false
        }
        return text.contains(monthName)
    }

    private func regexCaptureGroups(pattern: String, text: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return [] }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, options: [], range: range).map { match in
            (1..<match.numberOfRanges).compactMap { index -> String? in
                let range = match.range(at: index)
                guard range.location != NSNotFound,
                      let swiftRange = Range(range, in: text) else {
                    return nil
                }
                return String(text[swiftRange])
            }
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

            if token.count >= 5 && !MemorySearchHeuristics.queryStopWords.contains(lower) {
                fallback.append(lower)
            }
        }

        if !prioritized.isEmpty {
            return Array(prioritized.prefix(4))
        }
        return Array(fallback.prefix(3))
    }

    private func quotedPhraseAnchors(from text: String) -> [String] {
        var seen: Set<String> = []
        return regexCaptureGroups(pattern: #"[\"']([^\"']{3,80})[\"']"#, text: text)
            .compactMap { match in match.first.map(normalizedComparisonKey(for:)) }
            .filter { phrase in
                phrase.split(separator: " ").count >= 2 && seen.insert(phrase).inserted
            }
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
        MemorySearchHeuristics.normalizedComparisonKey(for: text)
    }

    private func normalizedSemanticKey(for text: String) -> String {
        text
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .map(normalizedSemanticToken)
            .filter { !$0.isEmpty && !MemorySearchHeuristics.queryStopWords.contains($0) }
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
