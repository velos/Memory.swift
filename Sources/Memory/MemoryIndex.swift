import Accelerate
import CryptoKit
import Foundation
import MemoryStorage

public actor MemoryIndex {
    private let configuration: MemoryConfiguration
    private let storage: MemoryStorage
    private let fileManager: FileManager

    private var embeddingCache: [StoredChunkEmbedding]?

    private let markdownExtensions: Set<String> = ["md", "markdown", "mdx"]
    private let codeExtensions: Set<String> = [
        "swift", "m", "mm", "h", "hpp", "c", "cpp", "cc", "cxx",
        "js", "jsx", "ts", "tsx", "java", "kt", "kts",
        "go", "rs", "py", "rb", "php", "cs", "scala", "sh", "zsh", "bash"
    ]

    private struct WeightedQuery {
        var text: String
        var weight: Double
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
            embeddingCache = nil

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

            embeddingCache = nil
            events?(.completed(processedDocuments: documentURLs.count, totalChunks: totalChunks))
        } catch {
            throw normalizeError(error)
        }
    }

    public func removeDocuments(at urls: [URL]) async throws {
        do {
            let paths = urls.map(\.path)
            try await storage.removeDocuments(paths: paths)
            embeddingCache = nil
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

        let expandedQueries = try await buildExpandedQueries(query: query, normalizedText: normalizedText)
        events?(.expandedQueries(count: max(0, expandedQueries.count - 1)))

        var semanticRRF: [Int64: Double] = [:]
        var lexicalRRF: [Int64: Double] = [:]
        var semanticCandidateCount = 0
        var lexicalCandidateCount = 0

        for expandedQuery in expandedQueries {
            let queryVector: [Float]
            do {
                queryVector = try await configuration.embeddingProvider.embed(text: expandedQuery.text)
            } catch {
                throw MemoryError.embedding("Failed to embed query: \(error.localizedDescription)")
            }

            events?(.embeddedQuery(dimension: queryVector.count))

            let semanticHits = try await semanticSearch(
                queryVector: queryVector,
                limit: query.semanticCandidateLimit,
                allowedChunkIDs: allowedChunkIDs
            )
            semanticCandidateCount += semanticHits.count
            accumulateRRF(for: semanticHits, weight: expandedQuery.weight, into: &semanticRRF)

            let lexicalHits = try await storage.lexicalSearch(
                query: expandedQuery.text,
                limit: query.lexicalCandidateLimit,
                allowedChunkIDs: allowedChunkIDs
            )
            lexicalCandidateCount += lexicalHits.count
            accumulateRRF(for: lexicalHits, weight: expandedQuery.weight, into: &lexicalRRF)
        }

        events?(.semanticCandidates(count: semanticCandidateCount))
        events?(.lexicalCandidates(count: lexicalCandidateCount))

        var fused = try await fuseCandidates(
            semanticRRF: semanticRRF,
            lexicalRRF: lexicalRRF,
            query: query,
            primaryQueryText: normalizedText
        )
        events?(.fusedCandidates(count: fused.count))

        if let reranker = configuration.reranker, !fused.isEmpty, query.rerankLimit > 0 {
            do {
                fused = try await applyReranker(
                    reranker,
                    query: query,
                    fusedResults: fused
                )
                events?(.reranked(count: min(query.rerankLimit, fused.count)))
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

    public func listContextChunks(_ contextID: ContextID) async throws -> [SearchResult] {
        do {
            let rows = try await storage.listContextChunks(contextID: contextID.rawValue)
            return rows.map {
                SearchResult(
                    chunkID: $0.chunkID,
                    documentPath: $0.documentPath,
                    title: $0.title,
                    content: $0.content,
                    snippet: makeSnippet(content: $0.content, queryText: nil),
                    modifiedAt: $0.modifiedAt,
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

            return SearchResult(
                chunkID: row.chunkID,
                documentPath: row.documentPath,
                title: row.title,
                content: row.content,
                snippet: makeSnippet(content: row.content, queryText: nil),
                modifiedAt: row.modifiedAt,
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
        let chunks = configuration.chunker.chunk(text: content, kind: kind, sourceURL: url)
        guard !chunks.isEmpty else { return nil }

        let embeddings: [[Float]]
        do {
            embeddings = try await configuration.embeddingProvider.embed(texts: chunks.map(\.content))
        } catch {
            throw MemoryError.embedding("Failed to embed chunks for \(url.path): \(error.localizedDescription)")
        }

        guard embeddings.count == chunks.count else {
            throw MemoryError.embedding("Embedding provider returned \(embeddings.count) vectors for \(chunks.count) chunks")
        }

        let chunkInputs: [StoredChunkInput] = zip(chunks, embeddings).map { chunk, vector in
            return StoredChunkInput(
                ordinal: chunk.ordinal,
                content: chunk.content,
                tokenCount: chunk.tokenCount,
                embedding: vector,
                norm: l2Norm(vector)
            )
        }

        let metadata = try fileManager.attributesOfItem(atPath: url.path)
        let modifiedAt = (metadata[.modificationDate] as? Date) ?? Date()

        return StoredDocumentInput(
            path: url.path,
            title: inferTitle(content: content, fallback: url.deletingPathExtension().lastPathComponent),
            modifiedAt: modifiedAt,
            checksum: checksum(content),
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

    private func semanticSearch(
        queryVector: [Float],
        limit: Int,
        allowedChunkIDs: Set<Int64>?
    ) async throws -> [LexicalHit] {
        if queryVector.isEmpty {
            return []
        }

        let queryNorm = l2Norm(queryVector)
        guard queryNorm > 0 else {
            throw MemoryError.search("Query embedding norm is zero")
        }

        let embeddings = try await loadEmbeddings()
        var scored: [LexicalHit] = []
        scored.reserveCapacity(embeddings.count)

        for embedding in embeddings {
            if let allowedChunkIDs, !allowedChunkIDs.contains(embedding.chunkID) {
                continue
            }

            guard embedding.vector.count == queryVector.count else {
                continue
            }
            guard embedding.norm > 0 else {
                continue
            }

            let dot = dotProduct(queryVector, embedding.vector)
            let cosine = Double(dot) / (queryNorm * embedding.norm)
            if cosine.isFinite {
                scored.append(LexicalHit(chunkID: embedding.chunkID, score: cosine))
            }
        }

        return scored
            .sorted { lhs, rhs in
                if lhs.score == rhs.score {
                    return lhs.chunkID < rhs.chunkID
                }
                return lhs.score > rhs.score
            }
            .prefix(max(1, limit))
            .map { $0 }
    }

    private func loadEmbeddings() async throws -> [StoredChunkEmbedding] {
        if let embeddingCache {
            return embeddingCache
        }

        do {
            let loaded = try await storage.fetchAllChunkEmbeddings()
            embeddingCache = loaded
            return loaded
        } catch {
            throw normalizeError(error)
        }
    }

    private func dotProduct(_ lhs: [Float], _ rhs: [Float]) -> Float {
        if lhs.count == rhs.count {
            return vDSP.dot(lhs, rhs)
        }

        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0 }

        var sum: Float = 0
        for index in 0..<count {
            sum += lhs[index] * rhs[index]
        }
        return sum
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
            scores[hit.chunkID, default: 0] += (weight * base)
        }
    }

    private func fuseCandidates(
        semanticRRF: [Int64: Double],
        lexicalRRF: [Int64: Double],
        query: SearchQuery,
        primaryQueryText: String
    ) async throws -> [SearchResult] {
        let candidateIDs = Set(semanticRRF.keys).union(lexicalRRF.keys)
        guard !candidateIDs.isEmpty else { return [] }

        let metadataRows = try await storage.fetchChunkMetadata(chunkIDs: Array(candidateIDs))
        let metadataMap = Dictionary(uniqueKeysWithValues: metadataRows.map { ($0.chunkID, $0) })

        let now = Date()
        var results: [SearchResult] = []
        results.reserveCapacity(candidateIDs.count)

        for chunkID in candidateIDs {
            guard let metadata = metadataMap[chunkID] else { continue }

            let semantic = semanticRRF[chunkID] ?? 0
            let lexical = lexicalRRF[chunkID] ?? 0
            let ageDays = max(0, now.timeIntervalSince(metadata.modifiedAt) / 86_400)
            let recency = exp(-ageDays / 30.0)
            let fused = (0.60 * semantic) + (0.30 * lexical) + (0.10 * recency)

            results.append(
                SearchResult(
                    chunkID: chunkID,
                    documentPath: metadata.documentPath,
                    title: metadata.title,
                    content: metadata.content,
                    snippet: makeSnippet(content: metadata.content, queryText: primaryQueryText),
                    modifiedAt: metadata.modifiedAt,
                    score: SearchScoreBreakdown(
                        semantic: semantic,
                        lexical: lexical,
                        recency: recency,
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
            .prefix(max(query.limit, query.rerankLimit, 100))
            .map { $0 }
    }

    private func buildExpandedQueries(
        query: SearchQuery,
        normalizedText: String
    ) async throws -> [WeightedQuery] {
        var expanded: [WeightedQuery] = [
            WeightedQuery(text: normalizedText, weight: query.originalQueryWeight),
        ]

        guard query.expansionLimit > 0, let queryExpander = configuration.queryExpander else {
            return expanded
        }

        var expansionQuery = query
        expansionQuery.text = normalizedText

        let alternatives: [String]
        do {
            alternatives = try await queryExpander.expand(
                query: expansionQuery,
                limit: query.expansionLimit
            )
        } catch {
            // Keep search functional even when expansion generation fails.
            return expanded
        }

        var seen: Set<String> = [normalizedComparisonKey(for: normalizedText)]
        for alternative in alternatives {
            let trimmed = alternative.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            let key = normalizedComparisonKey(for: trimmed)
            guard !seen.contains(key) else { continue }
            seen.insert(key)

            expanded.append(
                WeightedQuery(
                    text: trimmed,
                    weight: query.expansionQueryWeight
                )
            )

            if expanded.count >= (query.expansionLimit + 1) {
                break
            }
        }

        return expanded
    }

    private func applyReranker(
        _ reranker: any Reranker,
        query: SearchQuery,
        fusedResults: [SearchResult]
    ) async throws -> [SearchResult] {
        guard !fusedResults.isEmpty else { return [] }

        let rerankCount = min(query.rerankLimit, fusedResults.count)
        let rerankable = Array(fusedResults.prefix(rerankCount))
        let remaining = Array(fusedResults.dropFirst(rerankCount))

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

        var reranked = rerankable.map { candidate -> SearchResult in
            var updated = candidate
            updated.score.rerank = assessmentByChunkID[candidate.chunkID]?.relevance ?? 0
            updated.score.blended = updated.score.fused
            return updated
        }

        reranked.sort { lhs, rhs in
            if lhs.score.rerank == rhs.score.rerank {
                if lhs.score.fused == rhs.score.fused {
                    return lhs.chunkID < rhs.chunkID
                }
                return lhs.score.fused > rhs.score.fused
            }
            return lhs.score.rerank > rhs.score.rerank
        }

        for index in reranked.indices {
            let position = index + 1
            reranked[index].score.blended = configuration.positionAwareBlending.blend(
                fused: reranked[index].score.fused,
                rerank: reranked[index].score.rerank,
                position: position
            )
        }

        let untouched = remaining.map { candidate -> SearchResult in
            var updated = candidate
            updated.score.rerank = 0
            updated.score.blended = updated.score.fused
            return updated
        }

        return (reranked + untouched).sorted(by: sortByBlendedScore(_:_:))
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
