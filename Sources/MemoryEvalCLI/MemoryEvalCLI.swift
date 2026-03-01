import ArgumentParser
import CryptoKit
import Foundation
import GRDB
import Memory
import MemoryAppleIntelligence
import MemoryCoreMLEmbedding
import MemoryNaturalLanguage

private let datasetReadmeTemplate = """
# Memory Eval Datasets

This folder drives `memory_eval` storage and recall eval runs.

Required files:
- `storage_cases.jsonl`
- `recall_documents.jsonl`
- `recall_queries.jsonl`

Common commands:
- `swift run memory_eval init --dataset-root ./Evals`
- `swift run memory_eval run --profile baseline --dataset-root ./Evals`
- `swift run memory_eval run --dataset-root ./Evals` (runs all profiles sequentially)
- `swift run memory_eval run --profile apple_tags --dataset-root ./Evals`
- `swift run memory_eval run --profile expansion_only --dataset-root ./Evals`
- `swift run memory_eval run --profile oracle_ceiling --dataset-root ./Evals`
- `swift run memory_eval run --profile expansion_rerank --dataset-root ./Evals`
- `swift run memory_eval run --profile expansion_rerank_tag --dataset-root ./Evals`
- `swift run memory_eval run --profile full_apple --dataset-root ./Evals`
- `swift run memory_eval compare ./Evals/runs/*.json`
"""

private let storageTemplate = """
{"id":"storage-1","kind":"markdown","text":"I felt frustrated during yesterday's outage review.","expected_memory_type":"emotional","required_spans":["frustrated","outage review"]}
{"id":"storage-2","kind":"markdown","text":"Step 1: run migration. Step 2: verify alerts. Step 3: announce release.","expected_memory_type":"procedural","required_spans":["Step 1","verify alerts"]}
"""

private let rerankerStopWords: Set<String> = [
    "about", "after", "also", "and", "are", "for", "from", "into",
    "its", "our", "that", "the", "their", "there", "these", "this",
    "what", "when", "where", "which", "with", "your"
]

private let evalIndexCacheSchemaVersion = 1

private let recallDocumentsTemplate = """
{"id":"doc-a","relative_path":"project/roadmap.md","kind":"markdown","text":"Q3 roadmap includes API stability work and a September launch milestone.","memory_type":"temporal"}
{"id":"doc-b","relative_path":"operations/runbook.md","kind":"markdown","text":"Deploy checklist: build, migrate database, run smoke tests, monitor errors.","memory_type":"procedural"}
"""

private let recallQueriesTemplate = """
{"id":"q1","query":"when is the launch milestone","relevant_document_ids":["doc-a"]}
{"id":"q2","query":"how do we deploy safely","relevant_document_ids":["doc-b"],"memory_types":["procedural"]}
"""

private enum SuiteKind {
    case storage
    case recall
}

private enum EvalError: LocalizedError {
    case invalidDataset(String)

    var errorDescription: String? {
        switch self {
        case let .invalidDataset(message):
            "Invalid dataset: \(message)"
        }
    }
}

private struct DeterminateProgress {
    let label: String
    let total: Int
    let startedAt: Date
    var completed: Int
    let printEvery: Int

    init(label: String, total: Int) {
        self.label = label
        self.total = max(1, total)
        self.startedAt = Date()
        completed = 0

        if total <= 100 {
            printEvery = 1
        } else if total <= 500 {
            printEvery = 5
        } else {
            printEvery = 10
        }
    }

    mutating func advance(detail: String? = nil) {
        completed += 1
        let shouldPrint = completed == 1 || completed == total || completed % printEvery == 0
        guard shouldPrint else { return }

        let elapsedSeconds = Date().timeIntervalSince(startedAt)
        let ratio = min(1.0, Double(completed) / Double(total))
        let percentage = String(format: "%.1f%%", ratio * 100.0)

        let etaSeconds: TimeInterval
        if completed > 0, completed < total {
            let perItem = elapsedSeconds / Double(completed)
            etaSeconds = perItem * Double(total - completed)
        } else {
            etaSeconds = 0
        }

        var message = "[progress][\(label)] \(completed)/\(total) (\(percentage)) elapsed \(formatDuration(elapsedSeconds))"
        if completed < total {
            message += " eta \(formatDuration(etaSeconds))"
        }
        if let detail, !detail.isEmpty {
            message += " - \(detail)"
        }
        print(message)
    }
}

private struct StorageCase: Decodable {
    var id: String
    var kind: String?
    var text: String
    var expectedMemoryType: String
    var requiredSpans: [String]
}

private struct RecallDocumentCase: Decodable {
    var id: String
    var relativePath: String?
    var kind: String?
    var text: String
    var memoryType: String?
}

private struct RecallQueryCase: Decodable {
    var id: String
    var query: String
    var relevantDocumentIds: [String]
    var memoryTypes: [String]?
    var difficulty: String?
}

enum EvalProfile: String, CaseIterable, Codable, ExpressibleByArgument {
    case baseline
    case appleTags = "apple_tags"
    case appleStorage = "apple_storage"
    case appleRecall = "apple_recall"
    case expansionOnly = "expansion_only"
    case oracleCeiling = "oracle_ceiling"
    case expansionRerank = "expansion_rerank"
    case expansionRerankTag = "expansion_rerank_tag"
    case fullApple = "full_apple"
    case chunker900 = "chunker_900"
    case normalizedBm25 = "normalized_bm25"
    case wideCandidates = "wide_candidates"
    case poolingMean = "pooling_mean"
    case poolingWeightedMean = "pooling_weighted_mean"
    case coremlLeafIR = "coreml_leaf_ir"
    case coremlRerank = "coreml_rerank"
    case coremlColbertRerank = "coreml_colbert_rerank"
    case leafirAppleRerank = "leafir_apple_rerank"
}

private struct StorageCaseResult: Codable {
    var id: String
    var expectedType: String
    var predictedType: String
    var predictedSource: String
    var predictedConfidence: Double?
    var missingSpans: [String]
    var chunkCount: Int
}

private struct StorageSuiteReport: Codable {
    var totalCases: Int
    var typeAccuracy: Double
    var macroF1: Double
    var spanCoverage: Double
    var fallbackRate: Double
    var confusionMatrix: [String: [String: Int]]
    var caseResults: [StorageCaseResult]
}

private struct RecallPerKMetric: Codable {
    var k: Int
    var hitRate: Double
    var recall: Double
    var mrr: Double
    var ndcg: Double
}

private struct RecallQueryResult: Codable {
    var id: String
    var query: String
    var relevantDocumentIds: [String]
    var retrievedDocumentIds: [String]
    var hitByK: [Int: Bool]
    var recallByK: [Int: Double]
    var mrrByK: [Int: Double]
    var ndcgByK: [Int: Double]
    var latencyMs: Double?
    var difficulty: String?
}

private struct RecallPerTypeMetric: Codable {
    var memoryType: String
    var queryCount: Int
    var hitRate: Double
    var mrr: Double
    var ndcg: Double
}

private struct RecallPerDifficultyMetric: Codable {
    var difficulty: String
    var queryCount: Int
    var hitRate: Double
    var mrr: Double
    var ndcg: Double
}

private struct RecallLatencyStats: Codable {
    var p50Ms: Double
    var p95Ms: Double
    var meanMs: Double
    var minMs: Double
    var maxMs: Double
}

private struct RecallSuiteReport: Codable {
    var totalQueries: Int
    var kValues: [Int]
    var metricsByK: [RecallPerKMetric]
    var queryResults: [RecallQueryResult]
    var perTypeMetrics: [RecallPerTypeMetric]?
    var perDifficultyMetrics: [RecallPerDifficultyMetric]?
    var latencyStats: RecallLatencyStats?
}

private struct RecallSuiteRunOutput {
    var report: RecallSuiteReport
    var notes: [String]
}

private struct ContentTagGenerationStats {
    var chunkCount: Int
    var taggedChunkCount: Int
    var totalTagCount: Int
}

private struct GeneratedChunkTag: Decodable {
    var name: String
    var confidence: Double
}

private struct OperationTimeoutError: Error {}

private actor ContentTaggingDiagnosticsCollector {
    private var totalCalls = 0
    private var successWithTags = 0
    private var successEmpty = 0
    private var failures = 0
    private var totalTags = 0
    private var inputLengthBuckets: [String: Int] = [:]
    private var failureReasons: [String: Int] = [:]

    func recordSuccess(tagCount: Int, textLength: Int) {
        totalCalls += 1
        totalTags += max(0, tagCount)
        inputLengthBuckets[lengthBucket(for: textLength), default: 0] += 1

        if tagCount > 0 {
            successWithTags += 1
        } else {
            successEmpty += 1
        }
    }

    func recordFailure(error: Error, textLength: Int) {
        totalCalls += 1
        failures += 1
        inputLengthBuckets[lengthBucket(for: textLength), default: 0] += 1

        let reason = "\(String(describing: type(of: error))): \(error.localizedDescription)"
        failureReasons[truncate(reason, maxLength: 120), default: 0] += 1
    }

    func summaryLine(suite: SuiteKind) -> String {
        guard totalCalls > 0 else {
            return "[\(suiteLabel(suite))][tagging-diagnostics] no tagger calls recorded."
        }

        let successRate = Double(successWithTags) / Double(totalCalls)
        let emptyRate = Double(successEmpty) / Double(totalCalls)
        let failureRate = Double(failures) / Double(totalCalls)
        let avgTags = successWithTags == 0 ? 0 : Double(totalTags) / Double(successWithTags)

        return "[\(suiteLabel(suite))][tagging-diagnostics] calls=\(totalCalls), withTags=\(successWithTags) (\(percent(successRate))), empty=\(successEmpty) (\(percent(emptyRate))), failures=\(failures) (\(percent(failureRate))), avgTagsPerTaggedCall=\(format(avgTags))"
    }

    func detailLines(suite: SuiteKind) -> [String] {
        var lines: [String] = []

        let orderedBuckets: [String] = ["0-199", "200-399", "400-799", "800-1199", "1200+"]
        let bucketSummary = orderedBuckets
            .map { (bucket: String) -> String in
                let count = inputLengthBuckets[bucket, default: 0]
                return "\(bucket):\(count)"
            }
            .joined(separator: ", ")
        lines.append("[\(suiteLabel(suite))][tagging-diagnostics] input-length-buckets \(bucketSummary)")

        if failureReasons.isEmpty {
            lines.append("[\(suiteLabel(suite))][tagging-diagnostics] failure-reasons none")
        } else {
            let topReasons = failureReasons
                .sorted { lhs, rhs in
                    if lhs.value == rhs.value {
                        return lhs.key < rhs.key
                    }
                    return lhs.value > rhs.value
                }
                .prefix(3)
                .map { "\($0.value)x \($0.key)" }
                .joined(separator: " | ")
            lines.append("[\(suiteLabel(suite))][tagging-diagnostics] top-failure-reasons \(topReasons)")
        }

        return lines
    }

    private func lengthBucket(for textLength: Int) -> String {
        switch textLength {
        case ..<200:
            return "0-199"
        case ..<400:
            return "200-399"
        case ..<800:
            return "400-799"
        case ..<1200:
            return "800-1199"
        default:
            return "1200+"
        }
    }

    private func truncate(_ text: String, maxLength: Int) -> String {
        guard text.count > maxLength else { return text }
        return String(text.prefix(maxLength - 3)) + "..."
    }
}

private actor RecallDiagnosticsCollector {
    private struct StageStats {
        var calls = 0
        var successes = 0
        var failures = 0
        var timeouts = 0
        var totalRequested = 0
        var totalProduced = 0
        var failureReasons: [String: Int] = [:]
    }

    private var expansion = StageStats()
    private var rerank = StageStats()
    private let expansionProviderIdentifier: String
    private let rerankProviderIdentifier: String

    init(expansionProviderIdentifier: String?, rerankProviderIdentifier: String?) {
        self.expansionProviderIdentifier = expansionProviderIdentifier ?? "none"
        self.rerankProviderIdentifier = rerankProviderIdentifier ?? "none"
    }

    func recordExpansionSuccess(requestedLimit: Int, alternateCount: Int) {
        expansion.calls += 1
        expansion.successes += 1
        expansion.totalRequested += max(0, requestedLimit)
        expansion.totalProduced += max(0, alternateCount)
    }

    func recordExpansionFailure(error: Error, requestedLimit: Int) {
        expansion.calls += 1
        expansion.failures += 1
        expansion.totalRequested += max(0, requestedLimit)
        if isTimeoutLikeError(error) {
            expansion.timeouts += 1
        }
        expansion.failureReasons[truncate(errorReason(error), maxLength: 120), default: 0] += 1
    }

    func recordRerankSuccess(candidateCount: Int, assessmentCount: Int) {
        rerank.calls += 1
        rerank.successes += 1
        rerank.totalRequested += max(0, candidateCount)
        rerank.totalProduced += max(0, assessmentCount)
    }

    func recordRerankFailure(error: Error, candidateCount: Int) {
        rerank.calls += 1
        rerank.failures += 1
        rerank.totalRequested += max(0, candidateCount)
        if isTimeoutLikeError(error) {
            rerank.timeouts += 1
        }
        rerank.failureReasons[truncate(errorReason(error), maxLength: 120), default: 0] += 1
    }

    func summaryLines(suite: SuiteKind) -> [String] {
        [
            "[\(suiteLabel(suite))][recall-diagnostics][providers] expansion=\(describeProvider(expansionProviderIdentifier)), rerank=\(describeProvider(rerankProviderIdentifier))",
            stageSummaryLine(
                stage: "expansion",
                suite: suite,
                stats: expansion,
                avgProducedLabel: "avgAlternates"
            ),
            stageSummaryLine(
                stage: "rerank",
                suite: suite,
                stats: rerank,
                avgProducedLabel: "avgAssessments"
            ),
        ]
    }

    func detailLines(suite: SuiteKind) -> [String] {
        [
            stageFailureDetail(stage: "expansion", suite: suite, stats: expansion),
            stageFailureDetail(stage: "rerank", suite: suite, stats: rerank),
        ]
    }

    private func stageSummaryLine(
        stage: String,
        suite: SuiteKind,
        stats: StageStats,
        avgProducedLabel: String
    ) -> String {
        guard stats.calls > 0 else {
            return "[\(suiteLabel(suite))][recall-diagnostics][\(stage)] no calls recorded."
        }

        let successRate = Double(stats.successes) / Double(stats.calls)
        let failureRate = Double(stats.failures) / Double(stats.calls)
        let timeoutRate = Double(stats.timeouts) / Double(stats.calls)
        let avgRequested = Double(stats.totalRequested) / Double(stats.calls)
        let avgProduced = Double(stats.totalProduced) / Double(stats.calls)

        return "[\(suiteLabel(suite))][recall-diagnostics][\(stage)] calls=\(stats.calls), success=\(stats.successes) (\(percent(successRate))), failures=\(stats.failures) (\(percent(failureRate))), timeouts=\(stats.timeouts) (\(percent(timeoutRate))), avgRequested=\(format(avgRequested)), \(avgProducedLabel)=\(format(avgProduced))"
    }

    private func stageFailureDetail(stage: String, suite: SuiteKind, stats: StageStats) -> String {
        guard !stats.failureReasons.isEmpty else {
            return "[\(suiteLabel(suite))][recall-diagnostics][\(stage)] failure-reasons none"
        }

        let topReasons = stats.failureReasons
            .sorted { lhs, rhs in
                if lhs.value == rhs.value {
                    return lhs.key < rhs.key
                }
                return lhs.value > rhs.value
            }
            .prefix(3)
            .map { "\($0.value)x \($0.key)" }
            .joined(separator: " | ")
        return "[\(suiteLabel(suite))][recall-diagnostics][\(stage)] top-failure-reasons \(topReasons)"
    }

    private func errorReason(_ error: Error) -> String {
        "\(String(describing: type(of: error))): \(error.localizedDescription)"
    }

    private func truncate(_ text: String, maxLength: Int) -> String {
        guard text.count > maxLength else { return text }
        return String(text.prefix(maxLength - 3)) + "..."
    }

    private func describeProvider(_ identifier: String) -> String {
        "\(identifier) (\(recallProviderKind(for: identifier)))"
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
private actor DiagnosticContentTagger: ContentTagger {
    let identifier: String
    private let base: any ContentTagger
    private let diagnostics: ContentTaggingDiagnosticsCollector

    init(base: any ContentTagger, diagnostics: ContentTaggingDiagnosticsCollector) {
        self.base = base
        self.diagnostics = diagnostics
        self.identifier = base.identifier
    }

    func tag(text: String, kind: DocumentKind, sourceURL: URL?) async throws -> [ContentTag] {
        do {
            let tags = try await base.tag(text: text, kind: kind, sourceURL: sourceURL)
            await diagnostics.recordSuccess(tagCount: tags.count, textLength: text.count)
            return tags
        } catch {
            await diagnostics.recordFailure(error: error, textLength: text.count)
            throw error
        }
    }
}

private actor DiagnosticQueryExpander: QueryExpander {
    let identifier: String
    private let base: any QueryExpander
    private let diagnostics: RecallDiagnosticsCollector

    init(base: any QueryExpander, diagnostics: RecallDiagnosticsCollector) {
        self.base = base
        self.diagnostics = diagnostics
        self.identifier = base.identifier
    }

    func expand(query: SearchQuery, limit: Int) async throws -> [String] {
        do {
            let alternatives = try await base.expand(query: query, limit: limit)
            await diagnostics.recordExpansionSuccess(requestedLimit: limit, alternateCount: alternatives.count)
            return alternatives
        } catch {
            await diagnostics.recordExpansionFailure(error: error, requestedLimit: limit)
            throw error
        }
    }
}

private actor DiagnosticReranker: Reranker {
    let identifier: String
    private let base: any Reranker
    private let diagnostics: RecallDiagnosticsCollector

    init(base: any Reranker, diagnostics: RecallDiagnosticsCollector) {
        self.base = base
        self.diagnostics = diagnostics
        self.identifier = base.identifier
    }

    func rerank(query: SearchQuery, candidates: [SearchResult]) async throws -> [RerankAssessment] {
        do {
            let assessments = try await base.rerank(query: query, candidates: candidates)
            guard !assessments.isEmpty else {
                let error = MemoryError.search("Reranker returned no assessments")
                await diagnostics.recordRerankFailure(error: error, candidateCount: candidates.count)
                throw error
            }
            await diagnostics.recordRerankSuccess(
                candidateCount: candidates.count,
                assessmentCount: assessments.count
            )
            return assessments
        } catch {
            await diagnostics.recordRerankFailure(error: error, candidateCount: candidates.count)
            throw error
        }
    }
}

private actor EvalResponseCache {
    private struct CacheStats {
        var hits = 0
        var misses = 0
        var writes = 0
    }

    private let dbQueue: DatabaseQueue
    private var statsByNamespace: [String: CacheStats] = [:]
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(databaseURL: URL) throws {
        dbQueue = try DatabaseQueue(path: databaseURL.path)
        try dbQueue.write { db in
            try db.execute(
                sql: """
                CREATE TABLE IF NOT EXISTS eval_provider_cache (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    value_blob BLOB NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            try db.execute(
                sql: """
                CREATE INDEX IF NOT EXISTS eval_provider_cache_namespace_idx
                ON eval_provider_cache(namespace)
                """
            )
        }
    }

    func load<T: Decodable>(
        namespace: String,
        keyComponents: [String],
        as type: T.Type
    ) throws -> T? {
        let cacheKey = makeCacheKey(namespace: namespace, components: keyComponents)
        let payload: Data? = try dbQueue.read { db in
            let row = try Row.fetchOne(
                db,
                sql: "SELECT value_blob FROM eval_provider_cache WHERE cache_key = ?",
                arguments: [cacheKey]
            )
            return row?["value_blob"]
        }

        if let payload {
            statsByNamespace[namespace, default: CacheStats()].hits += 1
            return try decoder.decode(T.self, from: payload)
        }

        statsByNamespace[namespace, default: CacheStats()].misses += 1
        return nil
    }

    func store<T: Encodable>(
        namespace: String,
        keyComponents: [String],
        value: T
    ) throws {
        let cacheKey = makeCacheKey(namespace: namespace, components: keyComponents)
        let payload = try encoder.encode(value)
        let updatedAt = Date().timeIntervalSince1970

        try dbQueue.write { db in
            try db.execute(
                sql: """
                INSERT INTO eval_provider_cache (cache_key, namespace, value_blob, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    namespace = excluded.namespace,
                    value_blob = excluded.value_blob,
                    updated_at = excluded.updated_at
                """,
                arguments: [cacheKey, namespace, payload, updatedAt]
            )
        }

        statsByNamespace[namespace, default: CacheStats()].writes += 1
    }

    func drainSummaryLines(suite: SuiteKind) -> [String] {
        guard !statsByNamespace.isEmpty else {
            return ["[\(suiteLabel(suite))][cache] no cache traffic."]
        }

        let lines = statsByNamespace
            .keys
            .sorted()
            .map { namespace in
                let stats = statsByNamespace[namespace, default: CacheStats()]
                let requests = stats.hits + stats.misses
                let hitRate = requests == 0 ? 0 : Double(stats.hits) / Double(requests)
                return "[\(suiteLabel(suite))][cache][\(namespace)] hits=\(stats.hits), misses=\(stats.misses), writes=\(stats.writes), hitRate=\(percent(hitRate))"
            }

        statsByNamespace.removeAll()
        return lines
    }

    private func makeCacheKey(namespace: String, components: [String]) -> String {
        ([namespace] + components).joined(separator: "\u{1F}")
    }
}

private struct CachedRerankAssessment: Codable {
    var chunkID: Int64
    var relevance: Double
    var rationale: String?
}

private actor CachingContentTagger: ContentTagger {
    let identifier: String
    private let base: any ContentTagger
    private let cache: EvalResponseCache

    init(base: any ContentTagger, cache: EvalResponseCache) {
        self.base = base
        self.cache = cache
        self.identifier = base.identifier
    }

    func tag(text: String, kind: DocumentKind, sourceURL: URL?) async throws -> [ContentTag] {
        let keyComponents = [
            identifier,
            kind.rawValue,
            sourceURL?.lastPathComponent ?? "unknown",
            text,
        ]

        if let cached = try await cache.load(
            namespace: "tag",
            keyComponents: keyComponents,
            as: [ContentTag].self
        ) {
            return cached
        }

        let generated = try await base.tag(text: text, kind: kind, sourceURL: sourceURL)
        try await cache.store(namespace: "tag", keyComponents: keyComponents, value: generated)
        return generated
    }
}

private actor CachingQueryExpander: QueryExpander {
    let identifier: String
    private let base: any QueryExpander
    private let cache: EvalResponseCache

    init(base: any QueryExpander, cache: EvalResponseCache) {
        self.base = base
        self.cache = cache
        self.identifier = base.identifier
    }

    func expand(query: SearchQuery, limit: Int) async throws -> [String] {
        let keyComponents = [
            identifier,
            query.text,
            String(limit),
        ]

        if let cached = try await cache.load(
            namespace: "expand",
            keyComponents: keyComponents,
            as: [String].self
        ) {
            return cached
        }

        let alternatives = try await base.expand(query: query, limit: limit)
        try await cache.store(namespace: "expand", keyComponents: keyComponents, value: alternatives)
        return alternatives
    }
}

private actor CachingReranker: Reranker {
    let identifier: String
    private let base: any Reranker
    private let cache: EvalResponseCache

    init(base: any Reranker, cache: EvalResponseCache) {
        self.base = base
        self.cache = cache
        self.identifier = base.identifier
    }

    func rerank(query: SearchQuery, candidates: [SearchResult]) async throws -> [RerankAssessment] {
        let keyComponents = makeKeyComponents(query: query, candidates: candidates)
        if let cached = try await cache.load(
            namespace: "rerank",
            keyComponents: keyComponents,
            as: [CachedRerankAssessment].self
        ) {
            return cached.map { cached in
                RerankAssessment(
                    chunkID: cached.chunkID,
                    relevance: cached.relevance,
                    rationale: cached.rationale
                )
            }
        }

        let assessments = try await base.rerank(query: query, candidates: candidates)
        let encoded = assessments.map { assessment in
            CachedRerankAssessment(
                chunkID: assessment.chunkID,
                relevance: assessment.relevance,
                rationale: assessment.rationale
            )
        }
        try await cache.store(namespace: "rerank", keyComponents: keyComponents, value: encoded)
        return assessments
    }

    private func makeKeyComponents(query: SearchQuery, candidates: [SearchResult]) -> [String] {
        var components: [String] = [
            identifier,
            query.text,
            String(candidates.count),
        ]
        components.reserveCapacity(components.count + candidates.count)

        for candidate in candidates {
            let pathComponent = URL(fileURLWithPath: candidate.documentPath).lastPathComponent
            let normalizedSnippet = candidate.snippet
                .replacingOccurrences(of: "\n", with: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let excerpt = String(normalizedSnippet.prefix(180))
            components.append("\(candidate.chunkID)|\(pathComponent)|\(excerpt)")
        }

        return components
    }
}

private actor HeuristicOverlapReranker: Reranker {
    let identifier = "heuristic-overlap-reranker"

    func rerank(query: SearchQuery, candidates: [SearchResult]) async throws -> [RerankAssessment] {
        guard !candidates.isEmpty else { return [] }

        let queryTokens = tokenSet(from: query.text)
        guard !queryTokens.isEmpty else {
            return candidates.map { candidate in
                RerankAssessment(chunkID: candidate.chunkID, relevance: 0, rationale: nil)
            }
        }

        return candidates.map { candidate in
            let body = "\(candidate.title ?? "") \(candidate.snippet) \(String(candidate.content.prefix(240)))"
            let candidateTokens = tokenSet(from: body)
            let overlap = queryTokens.intersection(candidateTokens).count
            let lexicalScore = Double(overlap) / Double(queryTokens.count)
            let phraseBonus = body.range(
                of: query.text,
                options: [.caseInsensitive, .diacriticInsensitive]
            ) != nil ? 0.2 : 0
            let relevance = min(1, max(0, lexicalScore + phraseBonus))

            return RerankAssessment(
                chunkID: candidate.chunkID,
                relevance: relevance,
                rationale: nil
            )
        }
    }

    private func tokenSet(from text: String) -> Set<String> {
        let tokens = text
            .lowercased()
            .split { character in !character.isLetter && !character.isNumber }
            .map(String.init)
            .filter { token in
                token.count >= 3 && !rerankerStopWords.contains(token)
            }
        return Set(tokens)
    }
}

private struct EvalRunReport: Codable {
    var schemaVersion: Int
    var createdAt: Date
    var profile: EvalProfile
    var datasetRoot: String
    var storage: StorageSuiteReport
    var recall: RecallSuiteReport
    var notes: [String]
}

@main
struct MemoryEvalCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "memory_eval",
        abstract: "Evaluation harness for Memory.swift storage and recall quality.",
        subcommands: [InitCommand.self, RunCommand.self, CompareCommand.self],
        defaultSubcommand: RunCommand.self
    )
}

struct InitCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "init",
        abstract: "Create example eval datasets in a dataset root folder."
    )

    @Option(name: .long, help: "Dataset root folder.")
    var datasetRoot: String = "Evals"

    @Flag(name: .long, help: "Overwrite existing files.")
    var force = false

    mutating func run() async throws {
        let root = URL(fileURLWithPath: NSString(string: datasetRoot).expandingTildeInPath).standardizedFileURL
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        try writeIfNeeded(
            root.appendingPathComponent("README.md"),
            content: datasetReadmeTemplate,
            force: force
        )
        try writeIfNeeded(
            root.appendingPathComponent("storage_cases.jsonl"),
            content: storageTemplate,
            force: force
        )
        try writeIfNeeded(
            root.appendingPathComponent("recall_documents.jsonl"),
            content: recallDocumentsTemplate,
            force: force
        )
        try writeIfNeeded(
            root.appendingPathComponent("recall_queries.jsonl"),
            content: recallQueriesTemplate,
            force: force
        )

        print("Initialized eval dataset in \(root.path)")
    }
}

struct RunCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Run storage + recall evaluation and write JSON/Markdown reports."
    )

    @Option(name: .long, help: "Profile to run. Omit to run all profiles sequentially.")
    var profile: EvalProfile?

    @Option(name: .long, help: "Dataset root folder.")
    var datasetRoot: String = "Evals"

    @Option(name: .long, help: "Comma-separated k values, e.g. 1,3,5,10.")
    var kValues: String = "1,3,5,10"

    @Option(name: .long, help: "Output JSON file path. Defaults to <dataset-root>/runs/<timestamp>-<profile>.json.")
    var output: String?

    @Flag(
        name: .long,
        inversion: .prefixedNo,
        help: "Enable provider response cache across eval runs (disable with --no-cache)."
    )
    var cache = true

    @Flag(
        name: .long,
        inversion: .prefixedNo,
        help: "Reuse cached suite indexes across eval runs (disable with --no-index-cache)."
    )
    var indexCache = true

    @Flag(name: .long, help: "Print per-case details.")
    var verbose = false

    mutating func run() async throws {
        let datasetRootURL = URL(fileURLWithPath: NSString(string: datasetRoot).expandingTildeInPath).standardizedFileURL
        let dataset = try loadDataset(root: datasetRootURL)
        let ks = try parseKValues(kValues)
        let responseCache = try makeResponseCacheIfEnabled(enabled: cache, datasetRoot: datasetRootURL)

        let profiles = profile.map { [$0] } ?? EvalProfile.allCases
        if profiles.count > 1, output != nil {
            throw ValidationError("When --profile is omitted (run all profiles), --output is unsupported. Let memory_eval write per-profile outputs to <dataset-root>/runs/.")
        }

        var generatedOutputs: [URL] = []
        for (index, currentProfile) in profiles.enumerated() {
            if profiles.count > 1 {
                print("[run] Starting profile \(index + 1)/\(profiles.count): \(currentProfile.rawValue)")
            }

            let runRoot = FileManager.default.temporaryDirectory
                .appendingPathComponent("memory-evals", isDirectory: true)
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
            try FileManager.default.createDirectory(at: runRoot, withIntermediateDirectories: true)

            let outputURL = try await runSingleProfile(
                profile: currentProfile,
                dataset: dataset,
                kValues: ks,
                datasetRootURL: datasetRootURL,
                runRoot: runRoot,
                responseCache: responseCache,
                outputOverride: profiles.count == 1 ? output : nil
            )
            generatedOutputs.append(outputURL)
        }

        if profiles.count > 1 {
            print("[run] Completed \(profiles.count) profiles.")
            print("Generated JSON reports:")
            for path in generatedOutputs.map(\.path) {
                print("- \(path)")
            }

            let compareArgs = generatedOutputs.map { "\"\($0.path)\"" }.joined(separator: " ")
            print("Compare command:")
            print("swift run memory_eval compare \(compareArgs)")
        }
    }

    private func runSingleProfile(
        profile: EvalProfile,
        dataset: DatasetBundle,
        kValues: [Int],
        datasetRootURL: URL,
        runRoot: URL,
        responseCache: EvalResponseCache?,
        outputOverride: String?
    ) async throws -> URL {
        let storageReport = try await runStorageSuite(
            profile: profile,
            dataset: dataset.storageCases,
            datasetRoot: datasetRootURL,
            root: runRoot,
            indexCacheEnabled: indexCache,
            verbose: verbose,
            responseCache: responseCache
        )
        let recallOutput = try await runRecallSuite(
            profile: profile,
            documents: dataset.recallDocuments,
            queries: dataset.recallQueries,
            kValues: kValues,
            datasetRoot: datasetRootURL,
            root: runRoot,
            indexCacheEnabled: indexCache,
            verbose: verbose,
            responseCache: responseCache
        )

        let report = EvalRunReport(
            schemaVersion: 1,
            createdAt: Date(),
            profile: profile,
            datasetRoot: datasetRootURL.path,
            storage: storageReport,
            recall: recallOutput.report,
            notes: [
                "Storage eval uses direct database inspection with production-like chunking for classification/span metrics.",
                "Recall eval uses document-level metrics (deduped by document path).",
                "Recall documents are indexed without injected memory_type frontmatter to stress automatic classification.",
            ] + recallOutput.notes
        )

        let outputURL = try resolvedOutputURL(baseRoot: datasetRootURL, output: outputOverride, profile: profile)
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(report).write(to: outputURL, options: .atomic)

        let markdown = makeMarkdownSummary(report)
        let markdownURL = outputURL.deletingPathExtension().appendingPathExtension("md")
        try markdown.write(to: markdownURL, atomically: true, encoding: .utf8)

        print("Profile: \(profile.rawValue)")
        print("Storage type accuracy: \(percent(report.storage.typeAccuracy))")
        print("Storage macro F1: \(percent(report.storage.macroF1))")
        print("Storage span coverage: \(percent(report.storage.spanCoverage))")
        if let maxKMetric = report.recall.metricsByK.max(by: { $0.k < $1.k }) {
            print("Recall Hit@\(maxKMetric.k): \(percent(maxKMetric.hitRate))")
            print("Recall Recall@\(maxKMetric.k): \(percent(maxKMetric.recall))")
            print("Recall MRR@\(maxKMetric.k): \(format(maxKMetric.mrr))")
            print("Recall nDCG@\(maxKMetric.k): \(format(maxKMetric.ndcg))")
        }
        if let latencyStats = report.recall.latencyStats {
            print("Search latency: p50=\(String(format: "%.0f", latencyStats.p50Ms))ms p95=\(String(format: "%.0f", latencyStats.p95Ms))ms mean=\(String(format: "%.0f", latencyStats.meanMs))ms")
        }
        print("JSON report: \(outputURL.path)")
        print("Markdown summary: \(markdownURL.path)")

        return outputURL
    }
}

struct CompareCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "compare",
        abstract: "Compare multiple eval run JSON files."
    )

    @Argument(help: "Paths to eval run JSON files.")
    var runs: [String]

    @Option(name: .long, help: "Optional output markdown path.")
    var output: String?

    @Option(name: .long, help: "Path to a baseline run JSON. Exits with code 1 if any primary metric regresses beyond --regression-threshold.")
    var baseline: String?

    @Option(name: .long, help: "Maximum allowed regression as a fraction (e.g. 0.02 = 2%). Used with --baseline.")
    var regressionThreshold: Double = 0.02

    mutating func run() async throws {
        guard !runs.isEmpty else {
            throw ValidationError("Provide at least one run JSON path.")
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let loaded: [EvalRunReport] = try runs.map { path in
            let expanded = NSString(string: path).expandingTildeInPath
            let data = try Data(contentsOf: URL(fileURLWithPath: expanded))
            return try decoder.decode(EvalRunReport.self, from: data)
        }

        let markdown = makeComparisonMarkdown(loaded)
        print(markdown)

        if let output {
            let outputURL = URL(fileURLWithPath: NSString(string: output).expandingTildeInPath).standardizedFileURL
            try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try markdown.write(to: outputURL, atomically: true, encoding: .utf8)
            print("Wrote comparison: \(outputURL.path)")
        }

        if let baselinePath = baseline {
            let expanded = NSString(string: baselinePath).expandingTildeInPath
            let data = try Data(contentsOf: URL(fileURLWithPath: expanded))
            let baselineReport = try decoder.decode(EvalRunReport.self, from: data)
            let regressions = checkForRegressions(
                baseline: baselineReport,
                candidates: loaded,
                threshold: regressionThreshold
            )
            if !regressions.isEmpty {
                print("")
                print("REGRESSION DETECTED vs baseline (\(baselinePath)):")
                for regression in regressions {
                    print("  - \(regression)")
                }
                throw ExitCode(1)
            } else {
                print("")
                print("No regressions detected vs baseline (threshold: \(percent(regressionThreshold))).")
            }
        }
    }
}

private struct DatasetBundle {
    var storageCases: [StorageCase]
    var recallDocuments: [RecallDocumentCase]
    var recallQueries: [RecallQueryCase]
}

private func loadDataset(root: URL) throws -> DatasetBundle {
    let storageURL = root.appendingPathComponent("storage_cases.jsonl")
    let recallDocumentsURL = root.appendingPathComponent("recall_documents.jsonl")
    let recallQueriesURL = root.appendingPathComponent("recall_queries.jsonl")

    guard FileManager.default.fileExists(atPath: storageURL.path) else {
        throw ValidationError("Missing dataset file: \(storageURL.path). Run 'swift run memory_eval init --dataset-root \(root.path)'.")
    }
    guard FileManager.default.fileExists(atPath: recallDocumentsURL.path) else {
        throw ValidationError("Missing dataset file: \(recallDocumentsURL.path).")
    }
    guard FileManager.default.fileExists(atPath: recallQueriesURL.path) else {
        throw ValidationError("Missing dataset file: \(recallQueriesURL.path).")
    }

    let storageCases: [StorageCase] = try loadJSONLines(from: storageURL)
    let recallDocuments: [RecallDocumentCase] = try loadJSONLines(from: recallDocumentsURL)
    let recallQueries: [RecallQueryCase] = try loadJSONLines(from: recallQueriesURL)

    guard !storageCases.isEmpty else { throw ValidationError("storage_cases.jsonl must contain at least one case.") }
    guard !recallDocuments.isEmpty else { throw ValidationError("recall_documents.jsonl must contain at least one document.") }
    guard !recallQueries.isEmpty else { throw ValidationError("recall_queries.jsonl must contain at least one query.") }

    return DatasetBundle(
        storageCases: storageCases,
        recallDocuments: recallDocuments,
        recallQueries: recallQueries
    )
}

private func makeResponseCacheIfEnabled(
    enabled: Bool,
    datasetRoot: URL
) throws -> EvalResponseCache? {
    guard enabled else {
        print("[cache] disabled (--no-cache).")
        return nil
    }

    let cacheURL = evalCacheRootURL(datasetRoot: datasetRoot)
        .appendingPathComponent("provider", isDirectory: true)
        .appendingPathComponent("eval_provider_cache.sqlite")
    try FileManager.default.createDirectory(at: cacheURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    print("[cache] enabled: \(cacheURL.path)")
    return try EvalResponseCache(databaseURL: cacheURL)
}

private struct EvalIndexWorkspace {
    var root: URL
    var docsRoot: URL
    var databaseURL: URL
    var readyMarkerURL: URL?
    var cacheKey: String?
    var cacheEnabled: Bool
}

private func prepareIndexWorkspace(
    suite: SuiteKind,
    profile: EvalProfile,
    datasetRoot: URL,
    runRoot: URL,
    cacheEnabled: Bool,
    seed: String
) throws -> EvalIndexWorkspace {
    let suiteName = suiteLabel(suite)
    if !cacheEnabled {
        let root = runRoot.appendingPathComponent("\(suiteName)_workspace", isDirectory: true)
        let docsRoot = root.appendingPathComponent("\(suiteName)_docs", isDirectory: true)
        let databaseURL = root.appendingPathComponent("\(suiteName).sqlite")
        try FileManager.default.createDirectory(at: docsRoot, withIntermediateDirectories: true)
        return EvalIndexWorkspace(
            root: root,
            docsRoot: docsRoot,
            databaseURL: databaseURL,
            readyMarkerURL: nil,
            cacheKey: nil,
            cacheEnabled: false
        )
    }

    let digestInput = "index-cache-schema=\(evalIndexCacheSchemaVersion)\n" + seed
    let cacheKey = String(sha256Hex(digestInput).prefix(24))
    let root = evalCacheRootURL(datasetRoot: datasetRoot)
        .appendingPathComponent("index", isDirectory: true)
        .appendingPathComponent(suiteName, isDirectory: true)
        .appendingPathComponent("\(profile.rawValue)-\(cacheKey)", isDirectory: true)
    let docsRoot = root.appendingPathComponent("docs", isDirectory: true)
    let databaseURL = root.appendingPathComponent("\(suiteName).sqlite")
    let readyMarkerURL = root.appendingPathComponent(".ready")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return EvalIndexWorkspace(
        root: root,
        docsRoot: docsRoot,
        databaseURL: databaseURL,
        readyMarkerURL: readyMarkerURL,
        cacheKey: cacheKey,
        cacheEnabled: true
    )
}

private func indexCacheCanReuse(
    workspace: EvalIndexWorkspace,
    expectedDocumentPaths: [URL]
) -> Bool {
    guard workspace.cacheEnabled else { return false }
    guard let readyMarkerURL = workspace.readyMarkerURL else { return false }
    let fileManager = FileManager.default

    guard fileManager.fileExists(atPath: workspace.databaseURL.path),
          fileManager.fileExists(atPath: readyMarkerURL.path) else {
        return false
    }

    return expectedDocumentPaths.allSatisfy { fileManager.fileExists(atPath: $0.path) }
}

private func resetWorkspaceForRebuild(_ workspace: EvalIndexWorkspace) throws {
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: workspace.root.path) {
        try fileManager.removeItem(at: workspace.root)
    }
    try fileManager.createDirectory(at: workspace.docsRoot, withIntermediateDirectories: true)
}

private func markIndexCacheReady(_ workspace: EvalIndexWorkspace) throws {
    guard workspace.cacheEnabled, let readyMarkerURL = workspace.readyMarkerURL else { return }
    let metadata = [
        "cache_key=\(workspace.cacheKey ?? "unknown")",
        "updated_at=\(iso8601(Date()))",
    ].joined(separator: "\n")
    try metadata.write(to: readyMarkerURL, atomically: true, encoding: .utf8)
}

private func storageIndexCacheSeed(profile: EvalProfile, dataset: [StorageCase]) -> String {
    var parts: [String] = [
        "suite=storage",
        "profile=\(profile.rawValue)",
        "chunker=target120-overlap24",
    ]
    let sorted = dataset.sorted { $0.id < $1.id }
    parts.reserveCapacity(parts.count + (sorted.count * 6))
    for entry in sorted {
        parts.append("id=\(entry.id)")
        parts.append("kind=\(entry.kind ?? "")")
        parts.append("expected=\(entry.expectedMemoryType)")
        parts.append("required=\(entry.requiredSpans.joined(separator: "\u{1E}"))")
        parts.append("text=\(entry.text)")
    }
    return parts.joined(separator: "\n")
}

private func recallIndexCacheSeed(profile: EvalProfile, documents: [RecallDocumentCase]) -> String {
    var parts: [String] = [
        "suite=recall",
        "profile=\(profile.rawValue)",
    ]
    let sorted = documents.sorted { $0.id < $1.id }
    parts.reserveCapacity(parts.count + (sorted.count * 6))
    for document in sorted {
        parts.append("id=\(document.id)")
        parts.append("relative_path=\(document.relativePath ?? "")")
        parts.append("kind=\(document.kind ?? "")")
        parts.append("memory_type=\(document.memoryType ?? "")")
        parts.append("text=\(document.text)")
    }
    return parts.joined(separator: "\n")
}

private func evalCacheRootURL(datasetRoot: URL) -> URL {
    datasetRoot.appendingPathComponent("cache", isDirectory: true)
}

private func sha256Hex(_ value: String) -> String {
    let digest = SHA256.hash(data: Data(value.utf8))
    return digest.map { String(format: "%02x", $0) }.joined()
}

private func runStorageSuite(
    profile: EvalProfile,
    dataset: [StorageCase],
    datasetRoot: URL,
    root: URL,
    indexCacheEnabled: Bool,
    verbose: Bool,
    responseCache: EvalResponseCache?
) async throws -> StorageSuiteReport {
    let indexSeed = storageIndexCacheSeed(profile: profile, dataset: dataset)
    let workspace = try prepareIndexWorkspace(
        suite: .storage,
        profile: profile,
        datasetRoot: datasetRoot,
        runRoot: root,
        cacheEnabled: indexCacheEnabled,
        seed: indexSeed
    )
    let docsRoot = workspace.docsRoot

    var casePathByID: [String: URL] = [:]
    for entry in dataset {
        let ext = extensionForKind(entry.kind) ?? "md"
        let path = docsRoot.appendingPathComponent("\(safeFilename(entry.id)).\(ext)")
        casePathByID[entry.id] = path
    }

    let casePaths = Array(casePathByID.values)
    let canReuseIndex = indexCacheCanReuse(workspace: workspace, expectedDocumentPaths: casePaths)
    if canReuseIndex {
        print("[storage][index-cache] hit: \(workspace.root.path)")
    } else {
        if workspace.cacheEnabled {
            print("[storage][index-cache] miss: \(workspace.root.path)")
        }
        try resetWorkspaceForRebuild(workspace)
        for entry in dataset {
            guard let path = casePathByID[entry.id] else {
                throw EvalError.invalidDataset("Storage case '\(entry.id)' did not materialize to a document path.")
            }
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try entry.text.write(to: path, atomically: true, encoding: .utf8)
        }
    }

    let dbURL = workspace.databaseURL
    var config = try buildConfiguration(profile: profile, suite: .storage, databaseURL: dbURL)
    // Keep span coverage meaningful by allowing long cases to split across chunks.
    config.chunker = DefaultChunker(targetTokenCount: 120, overlapTokenCount: 24)
    try await requireFunctionalContentTaggingIfNeeded(profile: profile, configuration: config, suite: .storage)
    let contentTaggingDiagnostics = installContentTaggingDiagnosticsIfNeeded(
        profile: profile,
        configuration: &config
    )
    if let responseCache {
        installProviderResponseCachingIfNeeded(configuration: &config, responseCache: responseCache)
    }

    let index = try MemoryIndex(configuration: config)
    if canReuseIndex {
        print("[storage] Using cached index for \(dataset.count) cases.")
    } else {
        print("[storage] Building index for \(dataset.count) cases...")
        let indexStart = Date()
        try await index.rebuildIndex(from: [docsRoot])
        print("[storage] Index built in \(formatDuration(Date().timeIntervalSince(indexStart))).")
        try markIndexCacheReady(workspace)
    }
    if let contentTaggingDiagnostics {
        print(await contentTaggingDiagnostics.summaryLine(suite: .storage))
        for detail in await contentTaggingDiagnostics.detailLines(suite: .storage) {
            print(detail)
        }
    }
    try requireGeneratedContentTagsIfNeeded(profile: profile, databaseURL: dbURL, suite: .storage)
    if let responseCache {
        for line in await responseCache.drainSummaryLines(suite: .storage) {
            print(line)
        }
    }

    let dbQueue = try DatabaseQueue(path: dbURL.path)

    var results: [StorageCaseResult] = []
    var confusion: [String: [String: Int]] = [:]
    var correct = 0
    var fallbackCount = 0
    var spanFound = 0
    var spanTotal = 0
    var progress = DeterminateProgress(label: "storage", total: dataset.count)

    for entry in dataset {
        guard let documentURL = casePathByID[entry.id] else {
            throw EvalError.invalidDataset("Storage case '\(entry.id)' did not materialize to a document path.")
        }

        let expectedType = try parseMemoryType(entry.expectedMemoryType, context: "storage case \(entry.id)")
        let dbRows: [Row] = try dbQueue.read { db in
            try Row.fetchAll(
                db,
                sql: """
                SELECT
                    d.memory_type AS memory_type,
                    d.memory_type_source AS memory_type_source,
                    d.memory_type_confidence AS memory_type_confidence,
                    c.content AS content
                FROM documents d
                JOIN chunks c ON c.document_id = d.id
                WHERE d.path = ?
                ORDER BY c.ordinal ASC
                """,
                arguments: [documentURL.path]
            )
        }

        guard !dbRows.isEmpty else {
            throw EvalError.invalidDataset("No indexed chunks found for storage case '\(entry.id)' (\(documentURL.path)).")
        }

        let predictedRaw: String = dbRows[0]["memory_type"]
        let predictedSourceRaw: String = dbRows[0]["memory_type_source"]
        let predictedConfidence: Double? = dbRows[0]["memory_type_confidence"]
        let predictedType = MemoryType.parse(predictedRaw) ?? .factual
        let predictedSource = MemoryTypeSource.parse(predictedSourceRaw) ?? .fallback

        if predictedType == expectedType {
            correct += 1
        }
        if predictedSource == .fallback {
            fallbackCount += 1
        }

        confusion[expectedType.rawValue, default: [:]][predictedType.rawValue, default: 0] += 1

        let chunkContents = dbRows.map { (row: Row) -> String in row["content"] }
        let normalizedContents = chunkContents.map(normalizeForMatch)
        let spans = entry.requiredSpans.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let missing = spans.filter { span in
            let normalizedSpan = normalizeForMatch(span)
            return !normalizedContents.contains(where: { $0.contains(normalizedSpan) })
        }

        spanFound += (spans.count - missing.count)
        spanTotal += spans.count

        let caseResult = StorageCaseResult(
            id: entry.id,
            expectedType: expectedType.rawValue,
            predictedType: predictedType.rawValue,
            predictedSource: predictedSource.rawValue,
            predictedConfidence: predictedConfidence,
            missingSpans: missing,
            chunkCount: chunkContents.count
        )
        results.append(caseResult)

        if verbose {
            print("[storage] \(entry.id): expected=\(expectedType.rawValue) predicted=\(predictedType.rawValue) source=\(predictedSource.rawValue)")
        }
        progress.advance(detail: verbose ? entry.id : nil)
    }

    let macroF1 = computeMacroF1(
        expected: results.map(\.expectedType),
        predicted: results.map(\.predictedType),
        labels: MemoryType.allCases.map(\.rawValue)
    )

    let accuracy = dataset.isEmpty ? 0 : Double(correct) / Double(dataset.count)
    let spanCoverage = spanTotal == 0 ? 1 : Double(spanFound) / Double(spanTotal)
    let fallbackRate = dataset.isEmpty ? 0 : Double(fallbackCount) / Double(dataset.count)

    return StorageSuiteReport(
        totalCases: dataset.count,
        typeAccuracy: accuracy,
        macroF1: macroF1,
        spanCoverage: spanCoverage,
        fallbackRate: fallbackRate,
        confusionMatrix: confusion,
        caseResults: results.sorted { $0.id < $1.id }
    )
}

private func runRecallSuite(
    profile: EvalProfile,
    documents: [RecallDocumentCase],
    queries: [RecallQueryCase],
    kValues: [Int],
    datasetRoot: URL,
    root: URL,
    indexCacheEnabled: Bool,
    verbose: Bool,
    responseCache: EvalResponseCache?
) async throws -> RecallSuiteRunOutput {
    let indexSeed = recallIndexCacheSeed(profile: profile, documents: documents)
    let workspace = try prepareIndexWorkspace(
        suite: .recall,
        profile: profile,
        datasetRoot: datasetRoot,
        runRoot: root,
        cacheEnabled: indexCacheEnabled,
        seed: indexSeed
    )
    let docsRoot = workspace.docsRoot
    var runtimeNotes: [String] = []

    var pathByDocumentID: [String: String] = [:]
    var documentIDByPath: [String: String] = [:]
    var memoryTypeByDocumentID: [String: MemoryType] = [:]
    var contentByDocumentID: [String: String] = [:]

    for document in documents {
        if let memoryTypeRaw = document.memoryType {
            let parsed = try parseMemoryType(memoryTypeRaw, context: "recall document \(document.id)")
            memoryTypeByDocumentID[document.id] = parsed
        }

        let ext = extensionForKind(document.kind) ?? "md"
        let relativePath = document.relativePath?.trimmingCharacters(in: .whitespacesAndNewlines)
        let path: URL
        if let relativePath, !relativePath.isEmpty {
            path = docsRoot.appendingPathComponent(relativePath)
        } else {
            path = docsRoot.appendingPathComponent("\(safeFilename(document.id)).\(ext)")
        }

        let content = try materializeRecallDocument(document)
        contentByDocumentID[document.id] = content

        pathByDocumentID[document.id] = path.path
        documentIDByPath[path.path] = document.id
    }

    let expectedPaths = pathByDocumentID.values.map(URL.init(fileURLWithPath:))
    let canReuseIndex = indexCacheCanReuse(workspace: workspace, expectedDocumentPaths: expectedPaths)
    if canReuseIndex {
        print("[recall][index-cache] hit: \(workspace.root.path)")
    } else {
        if workspace.cacheEnabled {
            print("[recall][index-cache] miss: \(workspace.root.path)")
        }
        try resetWorkspaceForRebuild(workspace)
        for document in documents {
            guard let pathRaw = pathByDocumentID[document.id],
                  let content = contentByDocumentID[document.id] else {
                throw EvalError.invalidDataset("Recall document '\(document.id)' did not materialize to a document path.")
            }
            let path = URL(fileURLWithPath: pathRaw)
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try content.write(to: path, atomically: true, encoding: .utf8)
        }
    }

    let dbURL = workspace.databaseURL
    var config = try buildConfiguration(profile: profile, suite: .recall, databaseURL: dbURL)
    if let rerankerNote = try await ensureFunctionalRerankerIfNeeded(
        profile: profile,
        configuration: &config,
        suite: .recall,
        responseCache: responseCache
    ) {
        print("[recall][rerank] \(rerankerNote)")
        runtimeNotes.append(rerankerNote)
    }
    if let providerNote = recallProviderRuntimeNote(profile: profile, configuration: config, suite: .recall) {
        print("[recall][providers] \(providerNote)")
        runtimeNotes.append(providerNote)
    }
    try await requireFunctionalContentTaggingIfNeeded(profile: profile, configuration: config, suite: .recall)
    let contentTaggingDiagnostics = installContentTaggingDiagnosticsIfNeeded(
        profile: profile,
        configuration: &config
    )
    let recallDiagnostics = installRecallDiagnosticsIfNeeded(
        profile: profile,
        configuration: &config
    )
    if let responseCache {
        installProviderResponseCachingIfNeeded(configuration: &config, responseCache: responseCache)
    }
    let index = try MemoryIndex(configuration: config)
    if canReuseIndex {
        print("[recall] Using cached index for \(documents.count) documents.")
    } else {
        print("[recall] Building index for \(documents.count) documents...")
        let indexStart = Date()
        try await index.rebuildIndex(from: [docsRoot])
        print("[recall] Index built in \(formatDuration(Date().timeIntervalSince(indexStart))).")
        try markIndexCacheReady(workspace)
    }
    if let contentTaggingDiagnostics {
        print(await contentTaggingDiagnostics.summaryLine(suite: .recall))
        for detail in await contentTaggingDiagnostics.detailLines(suite: .recall) {
            print(detail)
        }
    }
    try requireGeneratedContentTagsIfNeeded(profile: profile, databaseURL: dbURL, suite: .recall)
    print("[recall] Evaluating \(queries.count) queries at k=\(kValues.map(String.init).joined(separator: ","))...")

    var queryResults: [RecallQueryResult] = []
    var perKAccumulator: [Int: (hit: Double, recall: Double, mrr: Double, ndcg: Double)] = [:]
    for k in kValues {
        perKAccumulator[k] = (0, 0, 0, 0)
    }

    let maxK = kValues.max() ?? 10
    let recallLimit = recallLimitForProfile(profile: profile, maxK: maxK)
    let dedupedDocumentLimit = dedupedDocumentLimitForProfile(profile: profile, maxK: maxK)
    var incompatibleFilterQueryCount = 0
    var oracleCandidateCoverageCount = 0
    var oracleRelevantCandidateTotal = 0
    var progress = DeterminateProgress(label: "recall", total: queries.count)
    for queryCase in queries {
        let relevant = Set(queryCase.relevantDocumentIds)
        guard !relevant.isEmpty else {
            throw EvalError.invalidDataset("Recall query '\(queryCase.id)' has empty relevant_document_ids.")
        }

        let unknownRelevant = relevant.filter { pathByDocumentID[$0] == nil }
        guard unknownRelevant.isEmpty else {
            throw EvalError.invalidDataset(
                "Recall query '\(queryCase.id)' references unknown relevant_document_ids: \(unknownRelevant.sorted().joined(separator: ", "))."
            )
        }

        let filterMemoryTypes = try parseOptionalMemoryTypes(queryCase.memoryTypes)
        var effectiveMemoryTypes = filterMemoryTypes
        if let filterMemoryTypes, !filterMemoryTypes.isEmpty {
            let relevantKnownTypes = Set(relevant.compactMap { memoryTypeByDocumentID[$0] })
            if !relevantKnownTypes.isEmpty, filterMemoryTypes.isDisjoint(with: relevantKnownTypes) {
                incompatibleFilterQueryCount += 1
                effectiveMemoryTypes = nil

                let filterRaw = filterMemoryTypes.map(\.rawValue).sorted().joined(separator: ",")
                let relevantRaw = relevantKnownTypes.map(\.rawValue).sorted().joined(separator: ",")
                print("[recall][warn] \(queryCase.id): memory_types=[\(filterRaw)] excludes relevant types [\(relevantRaw)]; ignoring filter.")
            }
        }

        let queryStartTime = Date()
        let recallResponse = try await index.recall(
            mode: .hybrid(query: queryCase.query),
            limit: recallLimit,
            features: recallFeatures(for: config),
            memoryTypes: effectiveMemoryTypes
        )
        let queryLatencyMs = Date().timeIntervalSince(queryStartTime) * 1000.0

        var rankedDocumentIDs: [String] = []
        var seen: Set<String> = []
        for record in recallResponse.records {
            guard let documentID = documentIDByPath[record.documentPath] else { continue }
            if seen.insert(documentID).inserted {
                rankedDocumentIDs.append(documentID)
            }
            if rankedDocumentIDs.count >= dedupedDocumentLimit {
                break
            }
        }

        let evaluatedDocumentIDs: [String]
        if profile == .oracleCeiling {
            let relevantCandidates = rankedDocumentIDs.filter(relevant.contains)
            oracleRelevantCandidateTotal += relevantCandidates.count
            if !relevantCandidates.isEmpty {
                oracleCandidateCoverageCount += 1
            }
            evaluatedDocumentIDs = oracleReorderCandidates(
                candidates: rankedDocumentIDs,
                relevant: relevant
            )
        } else {
            evaluatedDocumentIDs = rankedDocumentIDs
        }

        var hitByK: [Int: Bool] = [:]
        var recallByK: [Int: Double] = [:]
        var mrrByK: [Int: Double] = [:]
        var ndcgByK: [Int: Double] = [:]

        for k in kValues {
            let top = Array(evaluatedDocumentIDs.prefix(k))
            let retrievedRelevant = top.filter(relevant.contains)
            let hit = !retrievedRelevant.isEmpty
            let recall = Double(retrievedRelevant.count) / Double(relevant.count)

            let firstRelevantRank = top.firstIndex(where: { relevant.contains($0) }).map { $0 + 1 }
            let mrr = firstRelevantRank.map { 1.0 / Double($0) } ?? 0

            let ndcg = computeNDCG(ranked: top, relevant: relevant, k: k)

            hitByK[k] = hit
            recallByK[k] = recall
            mrrByK[k] = mrr
            ndcgByK[k] = ndcg

            perKAccumulator[k, default: (0, 0, 0, 0)].hit += hit ? 1 : 0
            perKAccumulator[k, default: (0, 0, 0, 0)].recall += recall
            perKAccumulator[k, default: (0, 0, 0, 0)].mrr += mrr
            perKAccumulator[k, default: (0, 0, 0, 0)].ndcg += ndcg
        }

        queryResults.append(
            RecallQueryResult(
                id: queryCase.id,
                query: queryCase.query,
                relevantDocumentIds: queryCase.relevantDocumentIds,
                retrievedDocumentIds: Array(evaluatedDocumentIDs.prefix(maxK)),
                hitByK: hitByK,
                recallByK: recallByK,
                mrrByK: mrrByK,
                ndcgByK: ndcgByK,
                latencyMs: queryLatencyMs,
                difficulty: queryCase.difficulty
            )
        )

        if verbose {
            let hitAtMax = hitByK[maxK] == true ? "hit" : "miss"
            print("[recall] \(queryCase.id): \(hitAtMax) @\(maxK)")
        }
        progress.advance(detail: verbose ? queryCase.id : nil)
    }

    if let recallDiagnostics {
        for line in await recallDiagnostics.summaryLines(suite: .recall) {
            print(line)
        }
        for line in await recallDiagnostics.detailLines(suite: .recall) {
            print(line)
        }
    }
    if let responseCache {
        for line in await responseCache.drainSummaryLines(suite: .recall) {
            print(line)
        }
    }

    let totalQueries = queries.count
    let metrics = kValues.map { k in
        let sums = perKAccumulator[k, default: (0, 0, 0, 0)]
        return RecallPerKMetric(
            k: k,
            hitRate: sums.hit / Double(totalQueries),
            recall: sums.recall / Double(totalQueries),
            mrr: sums.mrr / Double(totalQueries),
            ndcg: sums.ndcg / Double(totalQueries)
        )
    }

    let perTypeMetrics = computePerTypeMetrics(
        queryResults: queryResults,
        queries: queries,
        documents: documents,
        memoryTypeByDocumentID: memoryTypeByDocumentID,
        maxK: maxK
    )
    let perDifficultyMetrics = computePerDifficultyMetrics(
        queryResults: queryResults,
        queries: queries,
        maxK: maxK
    )
    let latencyStats = computeLatencyStats(queryResults: queryResults)

    var notes: [String] = runtimeNotes
    if incompatibleFilterQueryCount > 0 {
        notes.append(
            "Ignored contradictory recall query memory_types filters for \(incompatibleFilterQueryCount) quer\(incompatibleFilterQueryCount == 1 ? "y" : "ies") because they excluded all relevant_document_ids."
        )
    }
    if profile == .oracleCeiling {
        let coverage = totalQueries == 0 ? 0 : Double(oracleCandidateCoverageCount) / Double(totalQueries)
        let avgRelevantCandidates = totalQueries == 0 ? 0 : Double(oracleRelevantCandidateTotal) / Double(totalQueries)
        notes.append("oracle_ceiling reorders retrieved candidates using ground-truth labels to estimate ranking headroom (offline upper bound, not deployable).")
        notes.append("oracle_ceiling candidate window: recall limit \(recallLimit), deduped docs \(dedupedDocumentLimit) per query.")
        notes.append("oracle_ceiling candidate coverage (>=1 relevant doc in window): \(oracleCandidateCoverageCount)/\(totalQueries) (\(percent(coverage))).")
        notes.append("oracle_ceiling average relevant docs present in candidate window: \(format(avgRelevantCandidates)).")
    }

    return RecallSuiteRunOutput(
        report: RecallSuiteReport(
            totalQueries: totalQueries,
            kValues: kValues,
            metricsByK: metrics,
            queryResults: queryResults.sorted { $0.id < $1.id },
            perTypeMetrics: perTypeMetrics.isEmpty ? nil : perTypeMetrics,
            perDifficultyMetrics: perDifficultyMetrics.isEmpty ? nil : perDifficultyMetrics,
            latencyStats: latencyStats
        ),
        notes: notes
    )
}

private func computePerTypeMetrics(
    queryResults: [RecallQueryResult],
    queries: [RecallQueryCase],
    documents: [RecallDocumentCase],
    memoryTypeByDocumentID: [String: MemoryType],
    maxK: Int
) -> [RecallPerTypeMetric] {
    let queryById = Dictionary(uniqueKeysWithValues: queries.map { ($0.id, $0) })

    var typeAccumulators: [String: (hit: Double, mrr: Double, ndcg: Double, count: Int)] = [:]
    for result in queryResults {
        guard let queryCase = queryById[result.id] else { continue }
        let relevantTypes = Set(queryCase.relevantDocumentIds.compactMap { memoryTypeByDocumentID[$0]?.rawValue })
        guard let primaryType = relevantTypes.first else { continue }

        let hit = result.hitByK[maxK] == true ? 1.0 : 0.0
        let mrr = result.mrrByK[maxK] ?? 0
        let ndcg = result.ndcgByK[maxK] ?? 0

        var acc = typeAccumulators[primaryType, default: (0, 0, 0, 0)]
        acc.hit += hit
        acc.mrr += mrr
        acc.ndcg += ndcg
        acc.count += 1
        typeAccumulators[primaryType] = acc
    }

    return typeAccumulators.map { type, acc in
        RecallPerTypeMetric(
            memoryType: type,
            queryCount: acc.count,
            hitRate: acc.count == 0 ? 0 : acc.hit / Double(acc.count),
            mrr: acc.count == 0 ? 0 : acc.mrr / Double(acc.count),
            ndcg: acc.count == 0 ? 0 : acc.ndcg / Double(acc.count)
        )
    }.sorted { $0.memoryType < $1.memoryType }
}

private func computePerDifficultyMetrics(
    queryResults: [RecallQueryResult],
    queries: [RecallQueryCase],
    maxK: Int
) -> [RecallPerDifficultyMetric] {
    let queryById = Dictionary(uniqueKeysWithValues: queries.map { ($0.id, $0) })

    var accumulators: [String: (hit: Double, mrr: Double, ndcg: Double, count: Int)] = [:]
    for result in queryResults {
        let difficulty = queryById[result.id]?.difficulty ?? result.difficulty ?? "unknown"
        let hit = result.hitByK[maxK] == true ? 1.0 : 0.0
        let mrr = result.mrrByK[maxK] ?? 0
        let ndcg = result.ndcgByK[maxK] ?? 0

        var acc = accumulators[difficulty, default: (0, 0, 0, 0)]
        acc.hit += hit
        acc.mrr += mrr
        acc.ndcg += ndcg
        acc.count += 1
        accumulators[difficulty] = acc
    }

    let order = ["easy", "medium", "hard", "unknown"]
    return accumulators.map { difficulty, acc in
        RecallPerDifficultyMetric(
            difficulty: difficulty,
            queryCount: acc.count,
            hitRate: acc.count == 0 ? 0 : acc.hit / Double(acc.count),
            mrr: acc.count == 0 ? 0 : acc.mrr / Double(acc.count),
            ndcg: acc.count == 0 ? 0 : acc.ndcg / Double(acc.count)
        )
    }.sorted { (order.firstIndex(of: $0.difficulty) ?? 99) < (order.firstIndex(of: $1.difficulty) ?? 99) }
}

private func computeLatencyStats(queryResults: [RecallQueryResult]) -> RecallLatencyStats? {
    let latencies = queryResults.compactMap(\.latencyMs).sorted()
    guard !latencies.isEmpty else { return nil }

    let count = latencies.count
    let mean = latencies.reduce(0, +) / Double(count)
    let p50Index = min(count - 1, count / 2)
    let p95Index = min(count - 1, Int(Double(count) * 0.95))

    return RecallLatencyStats(
        p50Ms: latencies[p50Index],
        p95Ms: latencies[p95Index],
        meanMs: mean,
        minMs: latencies.first ?? 0,
        maxMs: latencies.last ?? 0
    )
}

private func buildConfiguration(
    profile: EvalProfile,
    suite: SuiteKind,
    databaseURL: URL
) throws -> MemoryConfiguration {
    var configuration = MemoryConfiguration.naturalLanguageDefault(databaseURL: databaseURL)
    configuration.memoryTyping = MemoryTypingConfiguration(
        mode: .automatic,
        classifier: NLEnhancedMemoryTypeClassifier(),
        fallbackType: .factual
    )

    switch profile {
    case .baseline:
        break
    case .appleTags:
        try enableAppleContentTagging(on: &configuration)
    case .appleStorage:
        if suite == .storage {
            configuration.memoryTyping.classifier = try makeAppleFirstMemoryClassifier()
        }
    case .appleRecall:
        if suite == .recall {
            try enableAppleRecallCapabilities(on: &configuration)
        }
    case .expansionOnly:
        if suite == .recall {
            try enableAppleExpansionCapabilities(on: &configuration)
        }
    case .oracleCeiling:
        break
    case .expansionRerank:
        if suite == .recall {
            try enableAppleRecallCapabilities(on: &configuration)
        }
    case .expansionRerankTag:
        try enableAppleContentTagging(on: &configuration)
        if suite == .recall {
            try enableAppleRecallCapabilities(on: &configuration)
        }
    case .fullApple:
        configuration.memoryTyping.classifier = try makeAppleFirstMemoryClassifier()
        try enableAppleContentTagging(on: &configuration)
        if suite == .recall {
            try enableAppleRecallCapabilities(on: &configuration)
        }
    case .chunker900:
        configuration.chunker = DefaultChunker(targetTokenCount: 900, overlapTokenCount: 135)
    case .normalizedBm25:
        break
    case .wideCandidates:
        configuration.semanticCandidateLimit = 1000
        configuration.lexicalCandidateLimit = 1000
    case .poolingMean:
        configuration = MemoryConfiguration.naturalLanguageDefault(
            databaseURL: databaseURL,
            poolingStrategy: .mean
        )
        configuration.memoryTyping = MemoryTypingConfiguration(
            mode: .automatic,
            classifier: NLEnhancedMemoryTypeClassifier(),
            fallbackType: .factual
        )
    case .poolingWeightedMean:
        configuration = MemoryConfiguration.naturalLanguageDefault(
            databaseURL: databaseURL,
            poolingStrategy: .weightedMean
        )
        configuration.memoryTyping = MemoryTypingConfiguration(
            mode: .automatic,
            classifier: NLEnhancedMemoryTypeClassifier(),
            fallbackType: .factual
        )
    case .coremlLeafIR:
        let modelURL = locateCoreMLModel(name: "leaf-ir")
        let provider = try CoreMLEmbeddingProvider(modelURL: modelURL)
        configuration = MemoryConfiguration(
            databaseURL: databaseURL,
            embeddingProvider: provider,
            memoryTyping: MemoryTypingConfiguration(
                mode: .automatic,
                classifier: NLEnhancedMemoryTypeClassifier(),
                fallbackType: .factual
            ),
            tokenizer: NLWordTokenizer(),
            ftsTokenizer: NLLemmatizingTokenizer()
        )
    case .coremlRerank:
        let embeddingModelURL = locateCoreMLModel(name: "leaf-ir")
        let rerankerModelURL = locateCoreMLModel(name: "tinybert-reranker")
        let provider = try CoreMLEmbeddingProvider(modelURL: embeddingModelURL)
        let reranker = try CoreMLReranker(modelURL: rerankerModelURL)
        configuration = MemoryConfiguration(
            databaseURL: databaseURL,
            embeddingProvider: provider,
            reranker: reranker,
            memoryTyping: MemoryTypingConfiguration(
                mode: .automatic,
                classifier: NLEnhancedMemoryTypeClassifier(),
                fallbackType: .factual
            ),
            tokenizer: NLWordTokenizer(),
            ftsTokenizer: NLLemmatizingTokenizer()
        )
    case .coremlColbertRerank:
        let embeddingModelURL = locateCoreMLModel(name: "leaf-ir")
        let rerankerModelURL = locateCoreMLModel(name: "colbert-17m")
        let provider = try CoreMLEmbeddingProvider(modelURL: embeddingModelURL)
        let reranker = try CoreMLColBERTReranker(modelURL: rerankerModelURL)
        configuration = MemoryConfiguration(
            databaseURL: databaseURL,
            embeddingProvider: provider,
            reranker: reranker,
            memoryTyping: MemoryTypingConfiguration(
                mode: .automatic,
                classifier: NLEnhancedMemoryTypeClassifier(),
                fallbackType: .factual
            ),
            tokenizer: NLWordTokenizer(),
            ftsTokenizer: NLLemmatizingTokenizer()
        )
    case .leafirAppleRerank:
        let embeddingModelURL = locateCoreMLModel(name: "leaf-ir")
        let provider = try CoreMLEmbeddingProvider(modelURL: embeddingModelURL)
        #if canImport(FoundationModels)
        if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isAvailable {
            let reranker = AppleIntelligenceReranker(
                maxCandidates: 16,
                responseTimeoutSeconds: 20
            )
            configuration = MemoryConfiguration(
                databaseURL: databaseURL,
                embeddingProvider: provider,
                reranker: reranker,
                memoryTyping: MemoryTypingConfiguration(
                    mode: .automatic,
                    classifier: NLEnhancedMemoryTypeClassifier(),
                    fallbackType: .factual
                ),
                tokenizer: NLWordTokenizer(),
                positionAwareBlending: PositionAwareBlending(
                    topRankFusedWeight: 0.58,
                    midRankFusedWeight: 0.42,
                    tailRankFusedWeight: 0.28
                ),
                ftsTokenizer: NLLemmatizingTokenizer()
            )
        } else {
            throw ValidationError("Apple Intelligence is unavailable. The leafir_apple_rerank profile requires it.")
        }
        #else
        throw ValidationError("Apple Intelligence is unavailable. The leafir_apple_rerank profile requires FoundationModels.")
        #endif
    }

    return configuration
}

private func locateCoreMLModel(name: String) -> URL {
    let filename = "\(name).mlpackage"
    let candidates = [
        "Models/\(filename)",
        "../Models/\(filename)",
    ]
    for candidate in candidates {
        let url = URL(fileURLWithPath: candidate)
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
    }
    let cwd = FileManager.default.currentDirectoryPath
    return URL(fileURLWithPath: cwd)
        .appendingPathComponent("Models")
        .appendingPathComponent(filename)
}

private func makeAppleFirstMemoryClassifier() throws -> any MemoryTypeClassifier {
    #if canImport(FoundationModels)
    if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isAvailable {
        return FallbackMemoryTypeClassifier(
            primary: AppleIntelligenceMemoryTypeClassifier(),
            fallback: HeuristicMemoryTypeClassifier()
        )
    }
    #endif
    throw ValidationError(
        "Apple Intelligence is unavailable for this runtime. Apple profiles require FoundationModels on iOS 26/macOS 26/visionOS 26 with Apple Intelligence enabled."
    )
}

private func installRecallDiagnosticsIfNeeded(
    profile: EvalProfile,
    configuration: inout MemoryConfiguration
) -> RecallDiagnosticsCollector? {
    guard profileUsesAppleRecallCapabilities(profile) else { return nil }
    guard configuration.queryExpander != nil || configuration.reranker != nil else { return nil }

    let diagnostics = RecallDiagnosticsCollector(
        expansionProviderIdentifier: configuration.queryExpander?.identifier,
        rerankProviderIdentifier: configuration.reranker?.identifier
    )
    if let queryExpander = configuration.queryExpander {
        configuration.queryExpander = DiagnosticQueryExpander(
            base: queryExpander,
            diagnostics: diagnostics
        )
    }
    if let reranker = configuration.reranker {
        configuration.reranker = DiagnosticReranker(
            base: reranker,
            diagnostics: diagnostics
        )
    }
    return diagnostics
}

private func installProviderResponseCachingIfNeeded(
    configuration: inout MemoryConfiguration,
    responseCache: EvalResponseCache
) {
    if let contentTagger = configuration.contentTagger {
        configuration.contentTagger = CachingContentTagger(
            base: contentTagger,
            cache: responseCache
        )
    }
    if let queryExpander = configuration.queryExpander {
        configuration.queryExpander = CachingQueryExpander(
            base: queryExpander,
            cache: responseCache
        )
    }
    if let reranker = configuration.reranker {
        configuration.reranker = CachingReranker(
            base: reranker,
            cache: responseCache
        )
    }
}

private func ensureFunctionalRerankerIfNeeded(
    profile: EvalProfile,
    configuration: inout MemoryConfiguration,
    suite: SuiteKind,
    responseCache: EvalResponseCache?
) async throws -> String? {
    guard suite == .recall else { return nil }
    guard profileUsesAppleRecallCapabilities(profile) else { return nil }
    guard let reranker = configuration.reranker else { return nil }
    guard reranker.identifier.contains("apple-intelligence-reranker") else { return nil }

    let now = Date()
    let probeCandidates = [
        SearchResult(
            chunkID: 1,
            documentPath: "probe/a.md",
            title: "Release Checklist",
            content: "Run migration and verify alerts before rollout.",
            snippet: "Run migration and verify alerts before rollout.",
            modifiedAt: now,
            memoryType: .procedural,
            memoryTypeSource: .automatic,
            memoryTypeConfidence: 0.9,
            score: SearchScoreBreakdown(
                semantic: 0.2,
                lexical: 0.2,
                recency: 0.8,
                fused: 0.3
            )
        ),
        SearchResult(
            chunkID: 2,
            documentPath: "probe/b.md",
            title: "Team Notes",
            content: "Discuss retrospective and backlog grooming topics.",
            snippet: "Discuss retrospective and backlog grooming topics.",
            modifiedAt: now,
            memoryType: .semantic,
            memoryTypeSource: .automatic,
            memoryTypeConfidence: 0.8,
            score: SearchScoreBreakdown(
                semantic: 0.1,
                lexical: 0.1,
                recency: 0.8,
                fused: 0.2
            )
        ),
    ]
    let probeQuery = SearchQuery(
        text: "release checklist migration alerts",
        limit: 5,
        semanticCandidateLimit: 0,
        lexicalCandidateLimit: 0,
        rerankLimit: 2,
        expansionLimit: 0
    )

    let probeReranker: any Reranker
    if let responseCache {
        probeReranker = CachingReranker(base: reranker, cache: responseCache)
    } else {
        probeReranker = reranker
    }

    let probeTimeouts: [Double] = [12, 18]
    var lastError: Error?

    for timeout in probeTimeouts {
        do {
            let assessments = try await withTimeout(seconds: timeout) {
                try await probeReranker.rerank(query: probeQuery, candidates: probeCandidates)
            }
            if assessments.isEmpty {
                throw ValidationError("Apple reranker probe returned no assessments.")
            }
            return nil
        } catch {
            lastError = error
            if !isTimeoutLikeError(error) {
                break
            }
        }
    }

    let detail = lastError?.localizedDescription ?? "unknown error"
    if let lastError, isTimeoutLikeError(lastError) {
        return "Apple reranker preflight timed out after \(probeTimeouts.count) attempts (\(detail)). Keeping Apple reranker enabled; runtime failures will fail open to fused ranking."
    }
    return "Apple reranker preflight failed (\(detail)). Keeping Apple reranker enabled; runtime failures will fail open to fused ranking."
}

private func installContentTaggingDiagnosticsIfNeeded(
    profile: EvalProfile,
    configuration: inout MemoryConfiguration
) -> ContentTaggingDiagnosticsCollector? {
    guard profile == .appleTags else { return nil }
    guard let contentTagger = configuration.contentTagger else { return nil }

    #if canImport(FoundationModels)
    if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) {
        let diagnostics = ContentTaggingDiagnosticsCollector()
        configuration.contentTagger = DiagnosticContentTagger(
            base: contentTagger,
            diagnostics: diagnostics
        )
        return diagnostics
    }
    #endif

    return nil
}

private func requireGeneratedContentTagsIfNeeded(
    profile: EvalProfile,
    databaseURL: URL,
    suite: SuiteKind
) throws {
    guard profile == .appleTags else { return }

    let stats = try loadContentTagGenerationStats(databaseURL: databaseURL)
    guard stats.chunkCount > 0 else { return }
    guard stats.totalTagCount > 0 else {
        throw ValidationError(
            "apple_tags profile produced zero chunk content tags in the \(suiteLabel(suite)) suite (\(stats.chunkCount) chunks). Content tagging is unavailable or non-functional for this run."
        )
    }

    let coverage = stats.chunkCount == 0 ? 0 : Double(stats.taggedChunkCount) / Double(stats.chunkCount)
    print(
        "[\(suiteLabel(suite))] Content tagging coverage: \(stats.taggedChunkCount)/\(stats.chunkCount) chunks (\(percent(coverage))), total tags: \(stats.totalTagCount)."
    )
}

private func requireFunctionalContentTaggingIfNeeded(
    profile: EvalProfile,
    configuration: MemoryConfiguration,
    suite: SuiteKind
) async throws {
    guard profile == .appleTags else { return }
    guard let tagger = configuration.contentTagger else {
        throw ValidationError("apple_tags profile requires a configured content tagger.")
    }

    let probeText = "Release checklist: run migration, verify alerts, and announce rollout."

    let generated: [ContentTag]
    do {
        generated = try await withTimeout(seconds: 20) {
            try await tagger.tag(text: probeText, kind: .plainText, sourceURL: nil)
        }
    } catch is OperationTimeoutError {
        throw ValidationError(
            "apple_tags profile timed out while probing content tagging in the \(suiteLabel(suite)) suite. Content tagging is unavailable or unresponsive for this runtime."
        )
    } catch {
        throw ValidationError(
            "apple_tags profile failed content tagging preflight in the \(suiteLabel(suite)) suite: \(error.localizedDescription)"
        )
    }

    let valid = generated.filter { tag in
        !tag.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && tag.confidence.isFinite
    }
    guard !valid.isEmpty else {
        throw ValidationError(
            "apple_tags profile returned zero tags in content tagging preflight for the \(suiteLabel(suite)) suite. Content tagging is unavailable or non-functional for this runtime."
        )
    }
}

private func withTimeout<T: Sendable>(
    seconds: Double,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    let timeoutNanoseconds = UInt64(max(1, Int(seconds * 1_000_000_000)))
    return try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask {
            try await operation()
        }
        group.addTask {
            try await Task.sleep(nanoseconds: timeoutNanoseconds)
            throw OperationTimeoutError()
        }

        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}

private func loadContentTagGenerationStats(databaseURL: URL) throws -> ContentTagGenerationStats {
    let dbQueue = try DatabaseQueue(path: databaseURL.path)

    return try dbQueue.read { db in
        let rows = try Row.fetchAll(db, sql: "SELECT content_tags_json FROM chunks")
        var taggedChunkCount = 0
        var totalTagCount = 0

        for row in rows {
            let raw: String? = row["content_tags_json"]
            let tags = decodeGeneratedChunkTags(raw)
            guard !tags.isEmpty else { continue }
            taggedChunkCount += 1
            totalTagCount += tags.count
        }

        return ContentTagGenerationStats(
            chunkCount: rows.count,
            taggedChunkCount: taggedChunkCount,
            totalTagCount: totalTagCount
        )
    }
}

private func decodeGeneratedChunkTags(_ raw: String?) -> [GeneratedChunkTag] {
    guard let raw else { return [] }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty, trimmed != "[]" else { return [] }

    guard let data = trimmed.data(using: .utf8),
          let decoded = try? JSONDecoder().decode([GeneratedChunkTag].self, from: data) else {
        return []
    }

    return decoded.filter { tag in
        !tag.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && tag.confidence.isFinite
    }
}

private func recallProviderRuntimeNote(
    profile: EvalProfile,
    configuration: MemoryConfiguration,
    suite: SuiteKind
) -> String? {
    guard profileUsesAppleRecallCapabilities(profile) else { return nil }

    let expanderID = configuration.queryExpander?.identifier ?? "none"
    let rerankerID = configuration.reranker?.identifier ?? "none"
    return "[\(suiteLabel(suite))] active providers: queryExpander=\(expanderID) (\(recallProviderKind(for: expanderID))), reranker=\(rerankerID) (\(recallProviderKind(for: rerankerID)))"
}

private func recallProviderKind(for identifier: String) -> String {
    if identifier == "none" {
        return "none"
    }
    if identifier.contains("apple-intelligence-query-expander") || identifier.contains("apple-intelligence-reranker") {
        return "apple"
    }
    if identifier.contains("heuristic") {
        return "heuristic"
    }
    return "other"
}

private func isTimeoutLikeError(_ error: Error) -> Bool {
    let text = "\(String(describing: type(of: error))): \(error.localizedDescription)".lowercased()
    return text.contains("timed out") || text.contains("timeout")
}

private func suiteLabel(_ suite: SuiteKind) -> String {
    switch suite {
    case .storage:
        return "storage"
    case .recall:
        return "recall"
    }
}

private func profileUsesAppleRecallCapabilities(_ profile: EvalProfile) -> Bool {
    switch profile {
    case .appleRecall, .expansionOnly, .expansionRerank, .expansionRerankTag, .fullApple,
         .leafirAppleRerank:
        return true
    case .baseline, .appleTags, .appleStorage, .oracleCeiling, .chunker900, .normalizedBm25,
         .wideCandidates, .poolingMean, .poolingWeightedMean, .coremlLeafIR, .coremlRerank,
         .coremlColbertRerank:
        return false
    }
}

private func recallLimitForProfile(profile: EvalProfile, maxK: Int) -> Int {
    switch profile {
    case .oracleCeiling:
        return max(120, maxK * 12)
    default:
        return max(50, maxK * 4)
    }
}

private func dedupedDocumentLimitForProfile(profile: EvalProfile, maxK: Int) -> Int {
    switch profile {
    case .oracleCeiling:
        return max(120, maxK * 12)
    default:
        return max(100, maxK * 4)
    }
}

private func oracleReorderCandidates(
    candidates: [String],
    relevant: Set<String>
) -> [String] {
    guard !candidates.isEmpty else { return [] }

    var relevantCandidates: [String] = []
    var nonRelevantCandidates: [String] = []
    relevantCandidates.reserveCapacity(candidates.count)
    nonRelevantCandidates.reserveCapacity(candidates.count)

    for candidate in candidates {
        if relevant.contains(candidate) {
            relevantCandidates.append(candidate)
        } else {
            nonRelevantCandidates.append(candidate)
        }
    }

    return relevantCandidates + nonRelevantCandidates
}

private func recallFeatures(for configuration: MemoryConfiguration) -> RecallFeatures {
    var features: RecallFeatures = [.semantic, .lexical]
    if configuration.contentTagger != nil {
        features.insert(.tags)
    }
    if configuration.queryExpander != nil {
        features.insert(.expansion)
    }
    if configuration.reranker != nil {
        features.insert(.rerank)
    }
    return features
}

private func enableAppleContentTagging(on configuration: inout MemoryConfiguration) throws {
    #if canImport(FoundationModels)
    if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isContentTaggingAvailable {
        configuration.contentTagger = AppleIntelligenceContentTagger()
        return
    }
    #endif
    throw ValidationError(
        "Apple content tagging is unavailable for this runtime. Apple tag profiles require FoundationModels contentTagging support on iOS 26/macOS 26/visionOS 26 with Apple Intelligence enabled."
    )
}

private func enableAppleExpansionCapabilities(on configuration: inout MemoryConfiguration) throws {
    #if canImport(FoundationModels)
    if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isAvailable {
        configuration.queryExpander = AppleIntelligenceQueryExpander(responseTimeoutSeconds: 5.0)
        return
    }
    #endif
    throw ValidationError(
        "Apple Intelligence is unavailable for this runtime. Apple expansion profiles require query expansion support."
    )
}

private func enableAppleRecallCapabilities(on configuration: inout MemoryConfiguration) throws {
    #if canImport(FoundationModels)
    if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isAvailable {
        configuration.queryExpander = AppleIntelligenceQueryExpander(responseTimeoutSeconds: 5.0)
        configuration.reranker = AppleIntelligenceReranker(maxCandidates: 16, responseTimeoutSeconds: 20)
        configuration.positionAwareBlending = PositionAwareBlending(
            topRankFusedWeight: 0.58,
            midRankFusedWeight: 0.42,
            tailRankFusedWeight: 0.28
        )
        return
    }
    #endif
    throw ValidationError(
        "Apple Intelligence is unavailable for this runtime. Apple recall profiles require query expansion/reranking support."
    )
}

private func checkForRegressions(
    baseline: EvalRunReport,
    candidates: [EvalRunReport],
    threshold: Double
) -> [String] {
    var regressions: [String] = []

    for candidate in candidates {
        let label = "\(candidate.profile.rawValue) @ \(iso8601(candidate.createdAt))"

        let accDelta = candidate.storage.typeAccuracy - baseline.storage.typeAccuracy
        if accDelta < -threshold {
            regressions.append("\(label): Storage type accuracy regressed by \(percent(-accDelta)) (baseline \(percent(baseline.storage.typeAccuracy)) -> \(percent(candidate.storage.typeAccuracy)))")
        }

        let f1Delta = candidate.storage.macroF1 - baseline.storage.macroF1
        if f1Delta < -threshold {
            regressions.append("\(label): Storage macro F1 regressed by \(percent(-f1Delta)) (baseline \(percent(baseline.storage.macroF1)) -> \(percent(candidate.storage.macroF1)))")
        }

        let baseMaxK = baseline.recall.metricsByK.max(by: { $0.k < $1.k })
        let candMaxK = candidate.recall.metricsByK.max(by: { $0.k < $1.k })
        if let bm = baseMaxK, let cm = candMaxK {
            let hitDelta = cm.hitRate - bm.hitRate
            if hitDelta < -threshold {
                regressions.append("\(label): Recall Hit@\(cm.k) regressed by \(percent(-hitDelta)) (baseline \(percent(bm.hitRate)) -> \(percent(cm.hitRate)))")
            }

            let mrrDelta = cm.mrr - bm.mrr
            if mrrDelta < -threshold {
                regressions.append("\(label): Recall MRR@\(cm.k) regressed by \(format(-mrrDelta)) (baseline \(format(bm.mrr)) -> \(format(cm.mrr)))")
            }

            let ndcgDelta = cm.ndcg - bm.ndcg
            if ndcgDelta < -threshold {
                regressions.append("\(label): Recall nDCG@\(cm.k) regressed by \(format(-ndcgDelta)) (baseline \(format(bm.ndcg)) -> \(format(cm.ndcg)))")
            }
        }
    }

    return regressions
}

private func makeComparisonMarkdown(_ reports: [EvalRunReport]) -> String {
    let sorted = reports.sorted { $0.createdAt < $1.createdAt }
    var lines: [String] = [
        "# Memory Eval Comparison",
        "",
        "| Run | Profile | Storage Acc | Macro F1 | Span Coverage | Recall Hit@K | Recall MRR@K | Recall nDCG@K |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for report in sorted {
        let maxMetric = report.recall.metricsByK.max(by: { $0.k < $1.k })
        let hit = maxMetric.map { percent($0.hitRate) } ?? "n/a"
        let mrr = maxMetric.map { format($0.mrr) } ?? "n/a"
        let ndcg = maxMetric.map { format($0.ndcg) } ?? "n/a"
        let kLabel = maxMetric.map { "@\($0.k)" } ?? ""

        lines.append(
            "| \(iso8601(report.createdAt)) | `\(report.profile.rawValue)` | \(percent(report.storage.typeAccuracy)) | \(percent(report.storage.macroF1)) | \(percent(report.storage.spanCoverage)) | \(hit)\(kLabel) | \(mrr)\(kLabel) | \(ndcg)\(kLabel) |"
        )
    }

    return lines.joined(separator: "\n")
}

private func makeMarkdownSummary(_ report: EvalRunReport) -> String {
    let maxKMetric = report.recall.metricsByK.max(by: { $0.k < $1.k })
    var lines: [String] = [
        "# Memory Eval Report",
        "",
        "- Created: \(iso8601(report.createdAt))",
        "- Profile: `\(report.profile.rawValue)`",
        "- Dataset: `\(report.datasetRoot)`",
        "",
        "## Storage",
        "",
        "- Cases: \(report.storage.totalCases)",
        "- Type accuracy: \(percent(report.storage.typeAccuracy))",
        "- Macro F1: \(percent(report.storage.macroF1))",
        "- Span coverage: \(percent(report.storage.spanCoverage))",
        "- Fallback rate: \(percent(report.storage.fallbackRate))",
        "",
        "## Recall",
        "",
        "- Queries: \(report.recall.totalQueries)",
    ]

    if let maxKMetric {
        lines.append("- Hit@\(maxKMetric.k): \(percent(maxKMetric.hitRate))")
        lines.append("- Recall@\(maxKMetric.k): \(percent(maxKMetric.recall))")
        lines.append("- MRR@\(maxKMetric.k): \(format(maxKMetric.mrr))")
        lines.append("- nDCG@\(maxKMetric.k): \(format(maxKMetric.ndcg))")
    }

    lines.append("")
    lines.append("### Recall By K")
    lines.append("")
    lines.append("| K | Hit | Recall | MRR | nDCG |")
    lines.append("|---:|---:|---:|---:|---:|")
    for metric in report.recall.metricsByK.sorted(by: { $0.k < $1.k }) {
        lines.append(
            "| \(metric.k) | \(percent(metric.hitRate)) | \(percent(metric.recall)) | \(format(metric.mrr)) | \(format(metric.ndcg)) |"
        )
    }

    if let perTypeMetrics = report.recall.perTypeMetrics, !perTypeMetrics.isEmpty {
        let maxK = maxKMetric?.k ?? (report.recall.kValues.max() ?? 10)
        lines.append("")
        lines.append("### Recall By Memory Type (at k=\(maxK))")
        lines.append("")
        lines.append("| Memory Type | Queries | Hit Rate | MRR | nDCG |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in perTypeMetrics {
            lines.append("| \(m.memoryType) | \(m.queryCount) | \(percent(m.hitRate)) | \(format(m.mrr)) | \(format(m.ndcg)) |")
        }
    }

    if let perDifficultyMetrics = report.recall.perDifficultyMetrics, !perDifficultyMetrics.isEmpty {
        let maxK = maxKMetric?.k ?? (report.recall.kValues.max() ?? 10)
        lines.append("")
        lines.append("### Recall By Difficulty (at k=\(maxK))")
        lines.append("")
        lines.append("| Difficulty | Queries | Hit Rate | MRR | nDCG |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in perDifficultyMetrics {
            lines.append("| \(m.difficulty) | \(m.queryCount) | \(percent(m.hitRate)) | \(format(m.mrr)) | \(format(m.ndcg)) |")
        }
    }

    if let latencyStats = report.recall.latencyStats {
        lines.append("")
        lines.append("### Search Latency")
        lines.append("")
        lines.append("| Stat | Value |")
        lines.append("|---|---:|")
        lines.append("| p50 | \(String(format: "%.1f", latencyStats.p50Ms)) ms |")
        lines.append("| p95 | \(String(format: "%.1f", latencyStats.p95Ms)) ms |")
        lines.append("| mean | \(String(format: "%.1f", latencyStats.meanMs)) ms |")
        lines.append("| min | \(String(format: "%.1f", latencyStats.minMs)) ms |")
        lines.append("| max | \(String(format: "%.1f", latencyStats.maxMs)) ms |")
    }

    let misses = report.recall.queryResults.filter { result in
        let maxK = maxKMetric?.k ?? (report.recall.kValues.max() ?? 10)
        return result.hitByK[maxK] == false
    }
    if !misses.isEmpty {
        lines.append("")
        lines.append("### Error Analysis (\(misses.count) misses at max K)")
        lines.append("")
        for miss in misses.prefix(30) {
            let diffLabel = miss.difficulty.map { " [\($0)]" } ?? ""
            lines.append("- `\(miss.id)`\(diffLabel): \(miss.query)")
            lines.append("  - relevant: \(miss.relevantDocumentIds.joined(separator: ", "))")
            lines.append("  - retrieved top-5: \(miss.retrievedDocumentIds.prefix(5).joined(separator: ", "))")
        }
        if misses.count > 30 {
            lines.append("- ... and \(misses.count - 30) more misses")
        }
    }

    if !report.notes.isEmpty {
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        for note in report.notes {
            lines.append("- \(note)")
        }
    }

    return lines.joined(separator: "\n")
}

private func parseKValues(_ raw: String) throws -> [Int] {
    let parsed = raw
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .compactMap(Int.init)
        .filter { $0 > 0 }
    let deduped = Array(Set(parsed)).sorted()
    guard !deduped.isEmpty else {
        throw ValidationError("Invalid --k-values '\(raw)'. Example: --k-values 1,3,5,10")
    }
    return deduped
}

private func parseMemoryType(_ raw: String, context: String) throws -> MemoryType {
    guard let type = MemoryType.parse(raw) else {
        let allowed = MemoryType.allCases.map(\.rawValue).joined(separator: ", ")
        throw EvalError.invalidDataset("Unknown memory type '\(raw)' in \(context). Allowed: \(allowed)")
    }
    return type
}

private func parseOptionalMemoryTypes(_ rawValues: [String]?) throws -> Set<MemoryType>? {
    guard let rawValues, !rawValues.isEmpty else { return nil }

    var result: Set<MemoryType> = []
    for raw in rawValues {
        let type = try parseMemoryType(raw, context: "query memory_types")
        result.insert(type)
    }
    return result.isEmpty ? nil : result
}

private func computeMacroF1(expected: [String], predicted: [String], labels: [String]) -> Double {
    guard !expected.isEmpty, expected.count == predicted.count else { return 0 }
    guard !labels.isEmpty else { return 0 }

    let f1Values = labels.map { label in
        var tp = 0
        var fp = 0
        var fn = 0
        for (actual, pred) in zip(expected, predicted) {
            if actual == label, pred == label {
                tp += 1
            } else if actual != label, pred == label {
                fp += 1
            } else if actual == label, pred != label {
                fn += 1
            }
        }

        let precision = (tp + fp) == 0 ? 0 : Double(tp) / Double(tp + fp)
        let recall = (tp + fn) == 0 ? 0 : Double(tp) / Double(tp + fn)
        if precision + recall == 0 {
            return 0.0
        }
        return 2 * precision * recall / (precision + recall)
    }

    return f1Values.reduce(0, +) / Double(f1Values.count)
}

private func computeNDCG(ranked: [String], relevant: Set<String>, k: Int) -> Double {
    guard k > 0 else { return 0 }

    var dcg = 0.0
    for (index, documentID) in ranked.prefix(k).enumerated() {
        let rel = relevant.contains(documentID) ? 1.0 : 0.0
        guard rel > 0 else { continue }
        let position = Double(index + 1)
        dcg += rel / log2(position + 1)
    }

    let idealCount = min(k, relevant.count)
    guard idealCount > 0 else { return 0 }
    var idcg = 0.0
    for i in 1...idealCount {
        idcg += 1.0 / log2(Double(i) + 1)
    }

    guard idcg > 0 else { return 0 }
    return dcg / idcg
}

private func materializeRecallDocument(_ document: RecallDocumentCase) throws -> String {
    if let memoryTypeRaw = document.memoryType {
        _ = try parseMemoryType(memoryTypeRaw, context: "recall document \(document.id)")
    }
    return document.text
}

private func parseDocumentKind(_ raw: String?) -> DocumentKind? {
    guard let raw else { return nil }
    switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "markdown", "md":
        return .markdown
    case "code":
        return .code
    case "plaintext", "plain_text", "text":
        return .plainText
    default:
        return nil
    }
}

private func extensionForKind(_ raw: String?) -> String? {
    guard let kind = parseDocumentKind(raw) else { return nil }
    switch kind {
    case .markdown:
        return "md"
    case .code:
        return "swift"
    case .plainText:
        return "txt"
    }
}

private func safeFilename(_ value: String) -> String {
    let scalars = value.unicodeScalars.map { scalar -> Character in
        if CharacterSet.alphanumerics.contains(scalar) || scalar == "-" || scalar == "_" {
            return Character(scalar)
        }
        return "_"
    }
    let raw = String(scalars)
    let collapsed = raw.replacingOccurrences(of: "__+", with: "_", options: .regularExpression)
    let trimmed = collapsed.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
    return trimmed.isEmpty ? UUID().uuidString : trimmed
}

private func normalizeForMatch(_ value: String) -> String {
    value
        .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
        .lowercased()
}

private func percent(_ value: Double) -> String {
    String(format: "%.2f%%", value * 100)
}

private func formatDuration(_ seconds: TimeInterval) -> String {
    let total = max(0, Int(seconds.rounded()))
    let hours = total / 3600
    let minutes = (total % 3600) / 60
    let secs = total % 60

    if hours > 0 {
        return String(format: "%dh %02dm %02ds", hours, minutes, secs)
    }
    if minutes > 0 {
        return String(format: "%dm %02ds", minutes, secs)
    }
    return "\(secs)s"
}

private func format(_ value: Double) -> String {
    String(format: "%.4f", value)
}

private func iso8601(_ date: Date) -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    return formatter.string(from: date)
}

private func resolvedOutputURL(baseRoot: URL, output: String?, profile: EvalProfile) throws -> URL {
    if let output {
        return URL(fileURLWithPath: NSString(string: output).expandingTildeInPath).standardizedFileURL
    }

    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    let timestamp = formatter.string(from: Date())
        .replacingOccurrences(of: ":", with: "-")
    return baseRoot
        .appendingPathComponent("runs", isDirectory: true)
        .appendingPathComponent("\(timestamp)-\(profile.rawValue).json")
}

private func writeIfNeeded(_ url: URL, content: String, force: Bool) throws {
    if FileManager.default.fileExists(atPath: url.path), !force {
        return
    }
    try content.write(to: url, atomically: true, encoding: .utf8)
}

private func loadJSONLines<T: Decodable>(from url: URL) throws -> [T] {
    let raw = try String(contentsOf: url, encoding: .utf8)
    let lines = raw.components(separatedBy: .newlines)

    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase

    var entries: [T] = []
    entries.reserveCapacity(lines.count)

    for (index, line) in lines.enumerated() {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed.hasPrefix("#") {
            continue
        }

        let data = Data(trimmed.utf8)
        do {
            entries.append(try decoder.decode(T.self, from: data))
        } catch {
            throw ValidationError("Failed to decode JSONL at \(url.path):\(index + 1): \(error.localizedDescription)")
        }
    }

    return entries
}
