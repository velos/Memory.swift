import Accelerate
import CryptoKit
import Foundation
import MemoryStorage

public actor MemoryIndex {
    private let configuration: MemoryConfiguration
    private let storage: MemoryStorage
    private let fileManager: FileManager

    private let markdownExtensions: Set<String> = ["md", "markdown", "mdx"]
    private let codeExtensions: Set<String> = [
        "swift", "m", "mm", "h", "hpp", "c", "cpp", "cc", "cxx",
        "js", "jsx", "ts", "tsx", "java", "kt", "kts",
        "go", "rs", "py", "rb", "php", "cs", "scala", "sh", "zsh", "bash"
    ]
    private let strongLexicalProbeLimit = 20
    private let strongLexicalMinScore = 0.10
    private let strongLexicalMinGap = 0.05

    private struct WeightedQuery {
        var text: String
        var weight: Double
        var expansionType: ExpansionType?
    }

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
                events?(.readingDocument(path: url.path, index: index + 1, total: urls.count))
                guard let payload = try await buildDocumentPayload(for: url) else { continue }

                totalChunks += payload.chunks.count
                events?(.chunked(path: url.path, chunks: payload.chunks.count))
                events?(.embedded(path: url.path, chunks: payload.chunks.count))

                try await storage.replaceDocument(payload)
                events?(.stored(path: url.path))
            }

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
                events?(.readingDocument(path: url.path, index: index + 1, total: documentURLs.count))

                if !fileManager.fileExists(atPath: url.path) {
                    try await storage.removeDocuments(paths: [url.path])
                    continue
                }

                guard let payload = try await buildDocumentPayload(for: url) else { continue }
                totalChunks += payload.chunks.count

                events?(.chunked(path: url.path, chunks: payload.chunks.count))
                events?(.embedded(path: url.path, chunks: payload.chunks.count))
                try await storage.replaceDocument(payload)
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
        try await search(query, events: nil)
    }

    public func search(_ query: SearchQuery, events: SearchEventHandler?) async throws -> [SearchResult] {
        let normalizedText = query.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedText.isEmpty else { return [] }

        events?(.started(query: normalizedText))

        let allowedChunkIDs: Set<Int64>?
        if let contextID = query.contextID {
            let contextChunkIDs = try await storage.fetchContextChunkIDs(contextID: contextID.rawValue)
            allowedChunkIDs = Set(contextChunkIDs)
            if contextChunkIDs.isEmpty {
                events?(.completed(count: 0))
                return []
            }
        } else {
            allowedChunkIDs = nil
        }

        let allowedMemoryTypes: Set<String>?
        if let requestedTypes = query.memoryTypes {
            let normalizedTypes = Set(requestedTypes.map(\.rawValue))
            if normalizedTypes.isEmpty {
                events?(.completed(count: 0))
                return []
            }
            allowedMemoryTypes = normalizedTypes
        } else {
            allowedMemoryTypes = nil
        }

        let queryAnalysis = configuration.queryAnalyzer?.analyze(query: normalizedText)

        let lexicalProbe = try await runLexicalProbe(
            query: query,
            normalizedText: normalizedText,
            allowedChunkIDs: allowedChunkIDs,
            allowedMemoryTypes: allowedMemoryTypes
        )
        let expandedQueries = try await buildExpandedQueries(
            query: query,
            normalizedText: normalizedText,
            skipExpansion: lexicalProbe.strongSignal
        )
        events?(.expandedQueries(count: max(0, expandedQueries.count - 1)))

        let semanticQueryVectors = try await embedExpandedQueries(
            expandedQueries,
            semanticCandidateLimit: query.semanticCandidateLimit,
            events: events
        )

        var semanticRRF: [Int64: Double] = [:]
        var lexicalRRF: [Int64: Double] = [:]
        var semanticCandidateCount = 0
        var lexicalCandidateCount = 0

        for (index, expandedQuery) in expandedQueries.enumerated() {
            let skipSemantic = expandedQuery.expansionType == .lexical
            let skipLexical = expandedQuery.expansionType == .semantic
                || expandedQuery.expansionType == .hypotheticalDocument

            if !skipSemantic, let semanticQueryVectors {
                let semanticHits = try await semanticSearch(
                    queryVector: semanticQueryVectors[index],
                    limit: query.semanticCandidateLimit,
                    allowedChunkIDs: allowedChunkIDs,
                    allowedMemoryTypes: allowedMemoryTypes
                )
                semanticCandidateCount += semanticHits.count
                accumulateRRF(for: semanticHits, weight: expandedQuery.weight, into: &semanticRRF)
            }

            if !skipLexical, query.lexicalCandidateLimit > 0 {
                let lexicalHits: [LexicalHit]
                if index == 0, let seeded = lexicalProbe.seededHits {
                    lexicalHits = seeded
                } else {
                    lexicalHits = try await storage.lexicalSearch(
                        query: ftsPreprocess(expandedQuery.text),
                        limit: query.lexicalCandidateLimit,
                        allowedChunkIDs: allowedChunkIDs,
                        allowedMemoryTypes: allowedMemoryTypes
                    )
                }
                lexicalCandidateCount += lexicalHits.count
                accumulateRRF(for: lexicalHits, weight: expandedQuery.weight, into: &lexicalRRF)
            }
        }

        if let analysis = queryAnalysis, !analysis.entities.isEmpty, query.lexicalCandidateLimit > 0 {
            for entity in analysis.entities.prefix(3) {
                let entityHits = try await storage.lexicalSearch(
                    query: ftsPreprocess(entity),
                    limit: query.lexicalCandidateLimit / 2,
                    allowedChunkIDs: allowedChunkIDs,
                    allowedMemoryTypes: allowedMemoryTypes
                )
                accumulateRRF(for: entityHits, weight: 0.5, into: &lexicalRRF)
                lexicalCandidateCount += entityHits.count
            }
        }

        events?(.semanticCandidates(count: semanticCandidateCount))
        events?(.lexicalCandidates(count: lexicalCandidateCount))

        let queryTags = query.includeTagScoring ? await resolveQueryContentTags(queryText: normalizedText) : []
        var fused = try await fuseCandidates(
            semanticRRF: semanticRRF,
            lexicalRRF: lexicalRRF,
            query: query,
            primaryQueryText: normalizedText,
            queryTags: queryTags
        )
        events?(.fusedCandidates(count: fused.count))

        let rerankCount = effectiveRerankCount(query: query, fusedCount: fused.count)
        if let reranker = configuration.reranker, !fused.isEmpty, rerankCount > 0 {
            do {
                fused = try await applyReranker(
                    reranker,
                    query: query,
                    fusedResults: fused,
                    rerankCount: rerankCount
                )
                events?(.reranked(count: rerankCount))
            } catch {
                // Fall back to fused ordering if reranking fails.
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

    public func setDocumentMemoryType(path: String, type: MemoryType) async throws {
        let normalizedPath = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedPath.isEmpty else {
            throw MemoryError.configuration("Document path must not be empty")
        }

        do {
            let updated = try await storage.setDocumentMemoryType(
                path: normalizedPath,
                type: type.rawValue,
                source: MemoryTypeSource.manual.rawValue,
                confidence: nil
            )
            guard updated else {
                throw MemoryError.storage("Document not found at path \(normalizedPath)")
            }
        } catch {
            throw normalizeError(error)
        }
    }

    public func setChunkMemoryTypeOverride(chunkID: Int64, type: MemoryType?) async throws {
        do {
            let updated = try await storage.setChunkMemoryTypeOverride(
                chunkID: chunkID,
                type: type?.rawValue,
                source: type == nil ? nil : MemoryTypeSource.manual.rawValue,
                confidence: nil
            )
            guard updated else {
                throw MemoryError.storage("Chunk not found for id \(chunkID)")
            }
        } catch {
            throw normalizeError(error)
        }
    }

    public func listContextChunks(_ contextID: ContextID) async throws -> [SearchResult] {
        do {
            let rows = try await storage.listContextChunks(contextID: contextID.rawValue)
            return rows.map {
                let assignment = resolveMemoryAssignment(
                    typeRaw: $0.memoryType,
                    sourceRaw: $0.memoryTypeSource,
                    confidence: $0.memoryTypeConfidence
                )
                return SearchResult(
                    chunkID: $0.chunkID,
                    documentPath: $0.documentPath,
                    title: $0.title,
                    content: $0.content,
                    snippet: makeSnippet(content: $0.content, queryText: nil),
                    modifiedAt: $0.modifiedAt,
                    memoryType: assignment.type,
                    memoryTypeSource: assignment.source,
                    memoryTypeConfidence: assignment.confidence,
                    score: SearchScoreBreakdown(semantic: 0, lexical: 0, recency: 0, fused: 0)
                )
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

            let assignment = resolveMemoryAssignment(
                typeRaw: row.memoryType,
                sourceRaw: row.memoryTypeSource,
                confidence: row.memoryTypeConfidence
            )
            return SearchResult(
                chunkID: row.chunkID,
                documentPath: row.documentPath,
                title: row.title,
                content: row.content,
                snippet: makeSnippet(content: row.content, queryText: nil),
                modifiedAt: row.modifiedAt,
                memoryType: assignment.type,
                memoryTypeSource: assignment.source,
                memoryTypeConfidence: assignment.confidence,
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
        category: MemoryCategory,
        importance: Double = 0.5,
        source: String = "memory_save",
        createdAt: Date? = nil,
        tags: [String] = []
    ) async throws -> MemoryRecord {
        let result = try await ingest(
            [
                ExtractedMemory(
                    text: text,
                    category: category,
                    importance: importance,
                    createdAt: createdAt,
                    source: source,
                    tags: tags
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
    ) async throws -> [ExtractedMemory] {
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
    ) async throws -> [ExtractedMemory] {
        guard limit > 0 else { return [] }
        guard !messages.isEmpty else { return [] }

        if let extractor = configuration.memoryExtractor {
            return try await extractor.extract(messages: messages, limit: limit)
        }

        return heuristicExtract(messages: messages, limit: limit)
    }

    public func ingest(_ memories: [ExtractedMemory]) async throws -> MemoryIngestResult {
        guard !memories.isEmpty else {
            return MemoryIngestResult(requestedCount: 0, storedCount: 0, discardedCount: 0, records: [])
        }

        var records: [MemoryRecord] = []
        var discardedCount = 0
        records.reserveCapacity(memories.count)

        for memory in memories {
            let trimmed = memory.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                discardedCount += 1
                continue
            }

            let embedding: [Float]
            do {
                embedding = try await configuration.embeddingProvider.embed(text: trimmed, format: .document(title: nil))
            } catch {
                throw MemoryError.embedding("Failed to embed memory for ingest: \(error.localizedDescription)")
            }

            let normalizedTags = normalizeIngestTags(memory.tags)
            let now = memory.createdAt ?? Date()
            let path = makeIngestPath(text: trimmed, category: memory.category)
            let title = inferTitle(content: trimmed, fallback: "memory")
            let memoryType = memory.category.mappedMemoryType

            let payload = StoredDocumentInput(
                path: path,
                title: title,
                modifiedAt: now,
                checksum: checksum(trimmed),
                memoryType: memoryType.rawValue,
                memoryTypeSource: MemoryTypeSource.manual.rawValue,
                memoryTypeConfidence: nil,
                chunks: [
                    StoredChunkInput(
                        ordinal: 0,
                        content: trimmed,
                        tokenCount: configuration.tokenizer.tokenize(trimmed).count,
                        embedding: embedding,
                        norm: l2Norm(embedding),
                        memoryTypeOverride: memoryType.rawValue,
                        memoryTypeOverrideSource: MemoryTypeSource.manual.rawValue,
                        memoryTypeOverrideConfidence: nil,
                        contentTags: normalizedTags,
                        memoryCategory: memory.category.rawValue,
                        importance: memory.importance,
                        accessCount: 0,
                        lastAccessedAt: nil,
                        source: memory.source,
                        createdAt: now
                    ),
                ]
            )

            do {
                try await storage.replaceDocument(payload)
                let chunkRows = try await storage.fetchChunkMetadataForDocument(path: path)
                if let first = chunkRows.first {
                    records.append(makeMemoryRecord(from: first, score: nil))
                }
            } catch {
                throw normalizeError(error)
            }
        }

        return MemoryIngestResult(
            requestedCount: memories.count,
            storedCount: records.count,
            discardedCount: discardedCount,
            records: records
        )
    }

    public func recall(
        mode: RecallMode,
        limit: Int = 20,
        features: RecallFeatures = .hybridDefault,
        sort: RecallSort = .recent,
        conversationContext: [ConversationMessage] = [],
        memoryTypes: Set<MemoryType>? = nil,
        events: SearchEventHandler? = nil
    ) async throws -> MemoryRecallResponse {
        let effectiveLimit = max(1, limit)
        let memoryTypeFilter = memoryTypes?.map(\.rawValue)

        switch mode {
        case let .hybrid(query):
            var queryText = query
            var plannedMemoryTypes = memoryTypes

            if features.contains(.planner), let planner = configuration.recallPlanner {
                do {
                    if let plan = try await planner.plan(
                        query: query,
                        conversationContext: conversationContext,
                        features: features
                    ) {
                        let plannedQuery = plan.query.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !plannedQuery.isEmpty {
                            queryText = plannedQuery
                        }
                        if let candidateTypes = plan.memoryTypes {
                            plannedMemoryTypes = candidateTypes
                        }
                    }
                } catch {
                    // Planner failures should not break retrieval.
                }
            }

            let semanticLimit = features.contains(.semantic) ? max(configuration.semanticCandidateLimit, effectiveLimit * 4) : 0
            let lexicalLimit = features.contains(.lexical) ? max(configuration.lexicalCandidateLimit, effectiveLimit * 4) : 0
            let rerankLimit = features.contains(.rerank) ? max(10, min(24, effectiveLimit * 2)) : 0
            let expansionLimit = features.contains(.expansion) ? 2 : 0

            let searchResults = try await search(
                SearchQuery(
                    text: queryText,
                    limit: effectiveLimit,
                    semanticCandidateLimit: semanticLimit,
                    lexicalCandidateLimit: lexicalLimit,
                    rerankLimit: rerankLimit,
                    expansionLimit: expansionLimit,
                    memoryTypes: plannedMemoryTypes,
                    includeTagScoring: features.contains(.tags)
                ),
                events: events
            )

            let chunkIDs = searchResults.map(\.chunkID)
            let metadataRows = try await storage.fetchChunkMetadata(chunkIDs: chunkIDs)
            let metadataByID = Dictionary(uniqueKeysWithValues: metadataRows.map { ($0.chunkID, $0) })

            let records: [MemoryRecord] = searchResults.compactMap { result in
                guard let metadata = metadataByID[result.chunkID] else { return nil }
                return makeMemoryRecord(from: metadata, score: result.score)
            }

            do {
                try await storage.recordChunkAccesses(records.map { $0.chunkID })
            } catch {
                throw normalizeError(error)
            }

            return MemoryRecallResponse(records: records)
        case .recent, .important, .typed:
            let categoryFilter: String?
            switch mode {
            case .typed(let category):
                categoryFilter = category.rawValue
            default:
                categoryFilter = nil
            }

            let sortMode: StoredMemorySort
            switch mode {
            case .recent:
                sortMode = .recent
            case .important:
                sortMode = .importance
            case .typed:
                sortMode = storageSort(for: sort)
            default:
                sortMode = .recent
            }

            let rows: [StoredChunkMetadata]
            do {
                rows = try await storage.listMemoryMetadata(
                    limit: effectiveLimit,
                    sort: sortMode,
                    memoryCategory: categoryFilter,
                    allowedMemoryTypes: memoryTypeFilter.map(Set.init)
                )
            } catch {
                throw normalizeError(error)
            }

            let records = rows.map { makeMemoryRecord(from: $0, score: nil) }
            do {
                try await storage.recordChunkAccesses(records.map { $0.chunkID })
            } catch {
                throw normalizeError(error)
            }
            return MemoryRecallResponse(records: records)
        }
    }

    private func makeIngestPath(text: String, category: MemoryCategory) -> String {
        let key = "\(category.rawValue)\n\(text)"
        let digest = SHA256.hash(data: Data(key.utf8))
        let hash = digest.map { String(format: "%02x", $0) }.joined()
        return "memory://ingest/\(hash).md"
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
    ) -> MemoryRecord {
        let assignment = resolveMemoryAssignment(
            typeRaw: metadata.memoryType,
            sourceRaw: metadata.memoryTypeSource,
            confidence: metadata.memoryTypeConfidence
        )
        let category = MemoryCategory.parse(metadata.memoryCategory) ?? assignment.type.defaultCategory
        let tags = metadata.contentTags.map { ContentTag(name: $0.name, confidence: $0.confidence) }

        return MemoryRecord(
            chunkID: metadata.chunkID,
            documentPath: metadata.documentPath,
            title: metadata.title,
            text: metadata.content,
            category: category,
            importance: metadata.importance,
            accessCount: metadata.accessCount,
            createdAt: metadata.createdAt,
            modifiedAt: metadata.modifiedAt,
            lastAccessedAt: metadata.lastAccessedAt,
            memoryType: assignment.type,
            memoryTypeSource: assignment.source,
            memoryTypeConfidence: assignment.confidence,
            tags: tags,
            score: score
        )
    }

    private func heuristicExtract(messages: [ConversationMessage], limit: Int) -> [ExtractedMemory] {
        let allText = messages
            .map(\.content)
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !allText.isEmpty else { return [] }

        let normalized = allText.replacingOccurrences(of: "\r\n", with: "\n")
        let rawSegments = normalized.split { character in
            character == "\n" || character == "." || character == "!" || character == "?"
        }

        var extracted: [ExtractedMemory] = []
        extracted.reserveCapacity(min(limit, rawSegments.count))

        var seen: Set<String> = []
        for rawSegment in rawSegments {
            let segment = String(rawSegment).trimmingCharacters(in: .whitespacesAndNewlines)
            guard segment.count >= 18 else { continue }

            let key = normalizedComparisonKey(for: segment)
            guard seen.insert(key).inserted else { continue }

            let category = inferCategory(forExtractedText: segment)
            let importance = inferredImportance(for: category)

            extracted.append(
                ExtractedMemory(
                    text: segment,
                    category: category,
                    importance: importance,
                    createdAt: nil,
                    source: "heuristic_extract",
                    tags: inferredTags(forExtractedText: segment)
                )
            )

            if extracted.count >= limit {
                break
            }
        }

        return extracted
    }

    private func inferCategory(forExtractedText text: String) -> MemoryCategory {
        let lower = text.lowercased()

        if containsAny(lower, needles: ["decide", "decision", "chose", "choose", "switch", "agreed"]) {
            return .decision
        }
        if containsAny(lower, needles: ["todo", "to do", "follow up", "action item", "next step"]) {
            return .todo
        }
        if containsAny(lower, needles: ["goal", "target", "objective", "aim"]) {
            return .goal
        }
        if containsAny(lower, needles: ["prefer", "preference", "like", "dislike"]) {
            return .preference
        }
        if containsAny(lower, needles: ["i am", "i'm", "my role", "identity"]) {
            return .identity
        }
        if containsAny(lower, needles: ["today", "yesterday", "last week", "incident", "meeting"]) {
            return .event
        }
        if containsAny(lower, needles: ["must", "always", "never", "rule", "fact"]) {
            return .fact
        }

        return .observation
    }

    private func inferredImportance(for category: MemoryCategory) -> Double {
        switch category {
        case .decision:
            return 0.85
        case .todo:
            return 0.80
        case .goal:
            return 0.78
        case .fact:
            return 0.70
        case .event:
            return 0.65
        case .preference:
            return 0.62
        case .identity:
            return 0.60
        case .observation:
            return 0.55
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

    private func buildDocumentPayload(for url: URL) async throws -> StoredDocumentInput? {
        guard isSupportedFile(url: url) else { return nil }

        let content: String
        do {
            content = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw MemoryError.ingestion("Unable to read UTF-8 file at \(url.path): \(error.localizedDescription)")
        }

        let kind = inferDocumentKind(for: url)
        let memoryAssignment = try await resolveDocumentMemoryType(
            content: content,
            kind: kind,
            sourceURL: url
        )
        let chunks = configuration.chunker.chunk(text: content, kind: kind, sourceURL: url)
        guard !chunks.isEmpty else { return nil }

        let documentTitle = inferTitle(content: content, fallback: url.deletingPathExtension().lastPathComponent)
        let embeddings: [[Float]]
        do {
            embeddings = try await configuration.embeddingProvider.embed(
                texts: chunks.map(\.content),
                format: .document(title: documentTitle)
            )
        } catch {
            throw MemoryError.embedding("Failed to embed chunks for \(url.path): \(error.localizedDescription)")
        }

        guard embeddings.count == chunks.count else {
            throw MemoryError.embedding("Embedding provider returned \(embeddings.count) vectors for \(chunks.count) chunks")
        }

        let chunkTags = await resolveChunkContentTags(chunks: chunks, kind: kind, sourceURL: url)
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
            memoryType: memoryAssignment.type.rawValue,
            memoryTypeSource: memoryAssignment.source.rawValue,
            memoryTypeConfidence: memoryAssignment.confidence,
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

    private func resolveDocumentMemoryType(
        content: String,
        kind: DocumentKind,
        sourceURL: URL
    ) async throws -> MemoryTypeAssignment {
        if kind == .markdown, let manualType = parseFrontmatterMemoryType(from: content) {
            return MemoryTypeAssignment(
                type: manualType,
                source: .manual,
                confidence: nil,
                classifierID: nil
            )
        }

        if configuration.memoryTyping.mode == .automatic,
           let classifier = configuration.memoryTyping.classifier {
            do {
                if let classified = try await classifier.classify(
                    documentText: content,
                    kind: kind,
                    sourceURL: sourceURL
                ) {
                    let source: MemoryTypeSource = classified.source == .manual ? .automatic : classified.source
                    return MemoryTypeAssignment(
                        type: classified.type,
                        source: source,
                        confidence: classified.confidence,
                        classifierID: classified.classifierID ?? classifier.identifier
                    )
                }
            } catch {
                // Classifier failures degrade to static fallback type.
            }
        }

        return MemoryTypeAssignment(
            type: configuration.memoryTyping.fallbackType,
            source: .fallback,
            confidence: nil,
            classifierID: nil
        )
    }

    private func parseFrontmatterMemoryType(from content: String) -> MemoryType? {
        let normalized = content.replacingOccurrences(of: "\r\n", with: "\n")
        let lines = normalized.components(separatedBy: "\n")
        guard lines.first?.trimmingCharacters(in: .whitespacesAndNewlines) == "---" else {
            return nil
        }

        guard let closingIndex = lines.dropFirst().firstIndex(where: { line in
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed == "---" || trimmed == "..."
        }) else {
            return nil
        }

        for line in lines[1..<closingIndex] {
            guard let separator = line.firstIndex(of: ":") else { continue }
            let key = line[..<separator].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard key == "memory_type" else { continue }

            let valueStart = line.index(after: separator)
            let rawValue = line[valueStart...].trimmingCharacters(in: .whitespacesAndNewlines)
            let unquoted = rawValue.trimmingCharacters(in: CharacterSet(charactersIn: "'\""))
            if let parsed = MemoryType.parse(unquoted) {
                return parsed
            }
        }
        return nil
    }

    private func resolveChunkContentTags(
        chunks: [Chunk],
        kind: DocumentKind,
        sourceURL: URL
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
                collected.append([])
            }
        }

        return collected
    }

    private func resolveQueryContentTags(queryText: String) async -> [ContentTag] {
        guard let contentTagger = configuration.contentTagger else { return [] }

        do {
            let generated = try await contentTagger.tag(
                text: queryText,
                kind: .plainText,
                sourceURL: nil
            )
            return normalizeContentTags(generated, maxCount: 8)
        } catch {
            return []
        }
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

    private func resolveMemoryAssignment(
        typeRaw: String,
        sourceRaw: String,
        confidence: Double?
    ) -> MemoryTypeAssignment {
        let parsedType = MemoryType.parse(typeRaw) ?? configuration.memoryTyping.fallbackType
        let parsedSource = MemoryTypeSource.parse(sourceRaw) ?? .fallback
        return MemoryTypeAssignment(
            type: parsedType,
            source: parsedSource,
            confidence: confidence,
            classifierID: nil
        )
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

    private func fuseCandidates(
        semanticRRF: [Int64: Double],
        lexicalRRF: [Int64: Double],
        query: SearchQuery,
        primaryQueryText: String,
        queryTags: [ContentTag]
    ) async throws -> [SearchResult] {
        let candidateIDs = Set(semanticRRF.keys).union(lexicalRRF.keys)
        guard !candidateIDs.isEmpty else { return [] }

        let metadataRows = try await storage.fetchChunkMetadata(chunkIDs: Array(candidateIDs))
        let metadataMap = Dictionary(uniqueKeysWithValues: metadataRows.map { ($0.chunkID, $0) })

        let now = Date()
        let weights = fusionWeights(for: primaryQueryText)
        var results: [SearchResult] = []
        results.reserveCapacity(candidateIDs.count)

        for chunkID in candidateIDs {
            guard let metadata = metadataMap[chunkID] else { continue }
            let assignment = resolveMemoryAssignment(
                typeRaw: metadata.memoryType,
                sourceRaw: metadata.memoryTypeSource,
                confidence: metadata.memoryTypeConfidence
            )

            let semantic = semanticRRF[chunkID] ?? 0
            let lexical = lexicalRRF[chunkID] ?? 0
            let ageDays = max(0, now.timeIntervalSince(metadata.modifiedAt) / 86_400)
            let recency = exp(-ageDays / 30.0)
            let anchorBonus = anchorCoverageBonus(queryText: primaryQueryText, metadata: metadata)
            let tagBonus = contentTagBonus(queryTags: queryTags, metadata: metadata)
            let fused = (weights.semantic * semantic) + (weights.lexical * lexical) + (weights.recency * recency) + anchorBonus + tagBonus

            results.append(
                SearchResult(
                    chunkID: chunkID,
                    documentPath: metadata.documentPath,
                    title: metadata.title,
                    content: metadata.content,
                    snippet: makeSnippet(content: metadata.content, queryText: primaryQueryText),
                    modifiedAt: metadata.modifiedAt,
                    memoryType: assignment.type,
                    memoryTypeSource: assignment.source,
                    memoryTypeConfidence: assignment.confidence,
                    score: SearchScoreBreakdown(
                        semantic: semantic,
                        lexical: lexical,
                        recency: recency,
                        tag: tagBonus,
                        fused: fused
                    )
                )
            )
        }

        return results
            .sorted { lhs, rhs in
                if lhs.score.fused == rhs.score.fused {
                    return lhs.chunkID < rhs.chunkID
                }
                return lhs.score.fused > rhs.score.fused
            }
            .prefix(candidatePoolLimit(for: query))
            .map { $0 }
    }

    private func buildExpandedQueries(
        query: SearchQuery,
        normalizedText: String,
        skipExpansion: Bool = false
    ) async throws -> [WeightedQuery] {
        var expanded: [WeightedQuery] = [
            WeightedQuery(text: normalizedText, weight: query.originalQueryWeight),
        ]

        guard !skipExpansion else {
            return expanded
        }

        guard query.expansionLimit > 0, let queryExpander = configuration.queryExpander else {
            return expanded
        }

        var expansionQuery = query
        expansionQuery.text = normalizedText

        let typedAlternatives: [ExpandedQuery]
        do {
            typedAlternatives = try await queryExpander.expandTyped(
                query: expansionQuery,
                limit: query.expansionLimit
            )
        } catch {
            return expanded
        }

        var seen: Set<String> = [normalizedComparisonKey(for: normalizedText)]
        for alternative in typedAlternatives {
            let trimmed = alternative.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            let key = normalizedComparisonKey(for: trimmed)
            guard !seen.contains(key) else { continue }
            seen.insert(key)

            expanded.append(
                WeightedQuery(
                    text: trimmed,
                    weight: query.expansionQueryWeight,
                    expansionType: alternative.type
                )
            )

            if expanded.count >= (query.expansionLimit + 1) {
                break
            }
        }

        return expanded
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
        guard query.expansionLimit > 0, configuration.queryExpander != nil else {
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

        // Blend using inverse original rank (1/rrfRank) instead of raw fused score,
        // so retrieval position and reranker score are on comparable 0-1 scales.
        for index in reranked.indices {
            let chunkID = reranked[index].chunkID
            let rrfRank = originalRankByChunkID[chunkID] ?? (index + 1)
            let rrfScore = 1.0 / Double(rrfRank)
            reranked[index].score.blended = configuration.positionAwareBlending.blend(
                fused: rrfScore,
                rerank: reranked[index].score.rerank,
                position: rrfRank
            )
        }

        let untouched = remaining.enumerated().map { offset, candidate -> SearchResult in
            var updated = candidate
            updated.score.rerank = 0
            let tailRank = effectiveRerankCount + offset + 1
            updated.score.blended = 1.0 / Double(tailRank)
            return updated
        }

        return (reranked + untouched).sorted(by: sortByBlendedScore(_:_:))
    }

    private func candidatePoolLimit(for query: SearchQuery) -> Int {
        let requested = max(query.limit, query.rerankLimit)
        let expanded = max(query.limit * 8, query.rerankLimit * 4, 200)
        return min(1_000, max(100, max(requested, expanded)))
    }

    private func effectiveRerankCount(query: SearchQuery, fusedCount: Int) -> Int {
        guard query.rerankLimit > 0, fusedCount > 0 else { return 0 }
        return min(fusedCount, query.rerankLimit)
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
}

private let temporalCueWords: [String] = [
    "when", "timeline", "chronology", "chronological", "date", "dates",
    "schedule", "scheduled", "milestone", "kickoff", "kick-off",
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
