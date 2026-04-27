import ArgumentParser
import CryptoKit
import Foundation
import Memory
import MemoryAppleIntelligence
import MemoryCoreMLEmbedding
import MemoryNaturalLanguage
import SQLiteSupport

private let datasetReadmeTemplate = """
# Memory Eval Datasets

This folder drives `memory_eval` storage, recall, and query-expansion eval runs.

Common files:
- `storage_cases.jsonl`
- `recall_documents.jsonl`
- `recall_queries.jsonl`
- `query_expansion_cases.jsonl`

Common commands:
- `swift run memory_eval init --dataset-root ./Evals`
- `swift run memory_eval run --profile nl_baseline --dataset-root ./Evals`
- `swift run memory_eval run --profile coreml_default --dataset-root ./Evals`
- `swift run memory_eval run --dataset-root ./Evals` (runs all profiles sequentially)
- `swift run memory_eval run --profile oracle_ceiling --dataset-root ./Evals`
- `swift run memory_eval run --profile apple_augmented --dataset-root ./Evals`
- `swift run memory_eval compare ./Evals/runs/*.json`
"""

private let storageTemplate = """
{"id":"storage-1","kind":"markdown","text":"I felt frustrated during yesterday's outage review.","expected_memory_type":"emotional","required_spans":["frustrated","outage review"]}
{"id":"storage-2","kind":"markdown","text":"Step 1: run migration. Step 2: verify alerts. Step 3: announce release.","expected_memory_type":"procedural","required_spans":["Step 1","verify alerts"]}
{"id":"profile-1","kind":"markdown","text":"Zac prefers Zed for Memory.swift work.","expected_kind":"profile","expected_status":"active","expected_facets":["preference","project","fact_about_user"],"required_entities":["zac","memory.swift","zed"],"required_topics":["memory.swift work"]}
"""

private let rerankerStopWords: Set<String> = [
    "about", "after", "also", "and", "are", "for", "from", "into",
    "its", "our", "that", "the", "their", "there", "these", "this",
    "what", "when", "where", "which", "with", "your"
]

private let evalIndexCacheSchemaVersion = 2

private enum RepoCoreMLModels {
    static let embedding = "embedding-v1"
    static let reranker = "reranker-v1"
}

private let legacyDocumentTypeLabels: [String] = [
    "factual",
    "procedural",
    "episodic",
    "semantic",
    "emotional",
    "social",
    "contextual",
    "temporal",
]

private let recallDocumentsTemplate = """
{"id":"doc-a","relative_path":"project/roadmap.md","kind":"markdown","text":"Q3 roadmap includes API stability work and a September launch milestone.","memory_type":"temporal"}
{"id":"doc-b","relative_path":"operations/runbook.md","kind":"markdown","text":"Deploy checklist: build, migrate database, run smoke tests, monitor errors.","memory_type":"procedural"}
"""

private let recallQueriesTemplate = """
{"id":"q1","query":"when is the launch milestone","relevant_document_ids":["doc-a"]}
{"id":"q2","query":"how do we deploy safely","relevant_document_ids":["doc-b"],"memory_types":["procedural"]}
"""

private let queryExpansionCasesTemplate = """
{"id":"qe-1","query":"release process for Memory.swift","expected_lexical_terms":["memory.swift","release","process"],"expected_semantic_phrases":["details about release process"],"expected_hyde_anchors":["memory.swift","release"],"expected_facets":["project"],"expected_entities":["memory.swift"],"expected_topics":["release process"],"relevant_document_ids":["doc-a"]}
{"id":"qe-2","query":"how do we ship sqlite-vec changes on time","expected_lexical_terms":["sqlite-vec","ship","time"],"expected_semantic_phrases":["procedure for"],"expected_hyde_anchors":["sqlite-vec","time"],"expected_facets":["time_sensitive","tool"],"expected_entities":["sqlite-vec"],"expected_topics":["sqlite-vec changes"],"relevant_document_ids":["doc-b"]}
"""

private enum SuiteKind {
    case storage
    case recall
    case queryExpansion
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
    var expectedMemoryType: String?
    var requiredSpans: [String]?
    var expectedKind: String?
    var expectedStatus: String?
    var expectedFacets: [String]?
    var requiredEntities: [String]?
    var requiredTopics: [String]?
    var expectedUpdateBehavior: String?
    var canonicalKey: String?
    var setupMemories: [StorageSeedMemory]?
}

private struct StorageSeedMemory: Decodable {
    var text: String
    var kind: String
    var status: String?
    var canonicalKey: String?
    var facetTags: [String]?
    var entityValues: [String]?
    var topics: [String]?
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

private struct QueryExpansionCase: Decodable {
    var id: String
    var query: String
    var expectedLexicalTerms: [String]?
    var expectedSemanticPhrases: [String]?
    var expectedHydeAnchors: [String]?
    var expectedFacets: [String]?
    var expectedEntities: [String]?
    var expectedTopics: [String]?
    var relevantDocumentIds: [String]?
    var candidateDocumentIds: [String]?
    var sourceDataset: String?
    var sourceQueryId: String?
    var sourceProfile: String?
    var sourceRankAtK: Int?
    var failureTaxonomy: [String]?
    var rescueReason: String?
}

private struct AgentMemoryScenarioCase: Decodable {
    var id: String
    var messages: [AgentMemoryScenarioMessage]
    var setupMemories: [StorageSeedMemory]?
    var expectedWriteCount: Int?
    var expectedMemories: [AgentMemoryExpectedMemory]?
    var expectedUpdateBehavior: String?
    var recallQueries: [AgentMemoryRecallExpectation]?
}

private struct AgentMemoryScenarioMessage: Decodable {
    var role: String
    var content: String
}

private struct AgentMemoryExpectedMemory: Decodable {
    var kind: String?
    var status: String?
    var canonicalKey: String?
    var textContains: [String]?
    var facets: [String]?
    var entities: [String]?
    var topics: [String]?
}

private struct AgentMemoryRecallExpectation: Decodable {
    var query: String
    var limit: Int?
    var expectedTextContains: [String]?
    var expectedKinds: [String]?
    var expectedStatuses: [String]?
}

enum EvalProfile: String, CaseIterable, Codable, ExpressibleByArgument {
    case nlBaseline = "nl_baseline"
    case coreMLDefault = "coreml_default"
    case coreMLLeafIR = "coreml_leaf_ir"
    case coreMLRerank = "coreml_rerank"
    case oracleCeiling = "oracle_ceiling"
    case appleAugmented = "apple_augmented"

    static var allCases: [EvalProfile] {
        [.nlBaseline, .coreMLDefault, .oracleCeiling, .appleAugmented]
    }
}

private struct StorageCaseResult: Codable {
    var id: String
    var expectedType: String
    var predictedType: String
    var predictedSource: String
    var predictedConfidence: Double?
    var missingSpans: [String]
    var chunkCount: Int
    var expectedKind: String?
    var predictedKind: String?
    var expectedStatus: String?
    var predictedStatus: String?
    var expectedFacets: [String]?
    var predictedFacets: [String]?
    var expectedEntities: [String]?
    var predictedEntities: [String]?
    var expectedTopics: [String]?
    var predictedTopics: [String]?
    var expectedUpdateBehavior: String?
    var observedUpdateBehavior: String?
}

private struct StorageStageLatencyStats: Codable {
    var typingMs: RecallLatencyStats?
    var chunkingMs: RecallLatencyStats?
    var taggingMs: RecallLatencyStats?
    var embeddingMs: RecallLatencyStats?
    var indexWriteMs: RecallLatencyStats?
    var totalMs: RecallLatencyStats?
}

private struct StorageSuiteReport: Codable {
    var mode: String?
    var totalCases: Int
    var typeAccuracy: Double
    var macroF1: Double
    var spanCoverage: Double
    var fallbackRate: Double
    var facetPrecision: Double?
    var facetRecall: Double?
    var facetMicroF1: Double?
    var entityPrecision: Double?
    var entityRecall: Double?
    var topicRecall: Double?
    var updateBehaviorAccuracy: Double?
    var confusionMatrix: [String: [String: Int]]
    var caseResults: [StorageCaseResult]
    var stageLatencyStats: StorageStageLatencyStats?
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
    var stageTimings: RecallQueryStageTimings?
    var candidateCounts: RecallQueryCandidateCounts?
    var difficulty: String?
}

private struct RecallBranchDiagnosticReport: Codable {
    var schemaVersion: Int
    var createdAt: Date
    var profile: EvalProfile
    var datasetRoot: String
    var sourceRun: String
    var scope: String
    var maxK: Int
    var wideLimit: Int
    var totalCases: Int
    var classificationCounts: [String: Int]
    var taxonomyCounts: [String: Int]
    var branchHitCounts: [String: Int]
    var caseResults: [RecallBranchDiagnosticCase]
}

private struct RecallBranchDiagnosticCase: Codable {
    var id: String
    var query: String
    var difficulty: String?
    var memoryTypes: [String]
    var relevantDocumentIds: [String]
    var originalRetrievedDocumentIds: [String]
    var originalHitAtK: Bool
    var originalRecallAtK: Double
    var classification: String
    var taxonomy: [String]
    var branchResults: [RecallBranchDiagnosticBranch]
}

private struct RecallBranchDiagnosticBranch: Codable {
    var name: String
    var features: [String]
    var limit: Int
    var latencyMs: Double
    var bestRelevantRank: Int?
    var relevantHitCount: Int
    var relevantHitCountAtK: Int
    var retrievedDocumentIds: [String]
    var topCandidates: [RecallBranchDiagnosticRankedCandidate]
    var relevantCandidates: [RecallBranchDiagnosticRankedCandidate]
    var stageTimings: RecallQueryStageTimings?
    var candidateCounts: RecallQueryCandidateCounts?
}

private struct RecallBranchDiagnosticRankedCandidate: Codable {
    var rank: Int
    var documentId: String
    var score: SearchScoreBreakdown
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

private struct RecallQueryStageTimings: Codable {
    var analysisMs: Double?
    var expansionMs: Double?
    var queryEmbeddingMs: Double?
    var semanticSearchMs: Double?
    var lexicalSearchMs: Double?
    var fusionMs: Double?
    var rerankMs: Double?
    var totalMs: Double?
}

private struct RecallQueryCandidateCounts: Codable {
    var expandedQueries: Int?
    var semanticCandidates: Int?
    var lexicalCandidates: Int?
    var fusedCandidates: Int?
    var rerankedCandidates: Int?
}

private struct RecallStageLatencyStats: Codable {
    var analysisMs: RecallLatencyStats?
    var expansionMs: RecallLatencyStats?
    var queryEmbeddingMs: RecallLatencyStats?
    var semanticSearchMs: RecallLatencyStats?
    var lexicalSearchMs: RecallLatencyStats?
    var fusionMs: RecallLatencyStats?
    var rerankMs: RecallLatencyStats?
    var totalMs: RecallLatencyStats?
}

private struct RecallCountStats: Codable {
    var p50: Double
    var p95: Double
    var mean: Double
    var min: Int
    var max: Int
}

private struct RecallCandidateCountStats: Codable {
    var expandedQueries: RecallCountStats?
    var semanticCandidates: RecallCountStats?
    var lexicalCandidates: RecallCountStats?
    var fusedCandidates: RecallCountStats?
    var rerankedCandidates: RecallCountStats?
}

private extension StorageStageLatencyStats {
    var hasData: Bool {
        typingMs != nil
            || chunkingMs != nil
            || taggingMs != nil
            || embeddingMs != nil
            || indexWriteMs != nil
            || totalMs != nil
    }
}

private extension RecallQueryStageTimings {
    var hasData: Bool {
        analysisMs != nil
            || expansionMs != nil
            || queryEmbeddingMs != nil
            || semanticSearchMs != nil
            || lexicalSearchMs != nil
            || fusionMs != nil
            || rerankMs != nil
            || totalMs != nil
    }
}

private extension RecallQueryCandidateCounts {
    var hasData: Bool {
        expandedQueries != nil
            || semanticCandidates != nil
            || lexicalCandidates != nil
            || fusedCandidates != nil
            || rerankedCandidates != nil
    }
}

private extension RecallStageLatencyStats {
    var hasData: Bool {
        analysisMs != nil
            || expansionMs != nil
            || queryEmbeddingMs != nil
            || semanticSearchMs != nil
            || lexicalSearchMs != nil
            || fusionMs != nil
            || rerankMs != nil
            || totalMs != nil
    }
}

private extension RecallCandidateCountStats {
    var hasData: Bool {
        expandedQueries != nil
            || semanticCandidates != nil
            || lexicalCandidates != nil
            || fusedCandidates != nil
            || rerankedCandidates != nil
    }
}

private struct RecallSuiteReport: Codable {
    var totalQueries: Int
    var kValues: [Int]
    var metricsByK: [RecallPerKMetric]
    var queryResults: [RecallQueryResult]
    var perTypeMetrics: [RecallPerTypeMetric]?
    var perDifficultyMetrics: [RecallPerDifficultyMetric]?
    var latencyStats: RecallLatencyStats?
    var stageLatencyStats: RecallStageLatencyStats?
    var candidateCountStats: RecallCandidateCountStats?
}

private struct RecallSuiteRunOutput {
    var report: RecallSuiteReport
    var notes: [String]
}

private struct QueryExpansionCaseResult: Codable {
    var id: String
    var query: String
    var sourceDataset: String?
    var sourceQueryId: String?
    var sourceProfile: String?
    var sourceRankAtK: Int?
    var failureTaxonomy: [String]?
    var rescueReason: String?
    var lexicalQueries: [String]
    var semanticQueries: [String]
    var hypotheticalDocuments: [String]
    var facetHints: [String]
    var entities: [String]
    var topics: [String]
    var lexicalCoverageRecall: Double
    var semanticCoverageRecall: Double
    var hydeAnchorRecall: Double
    var facetPrecision: Double
    var facetRecall: Double
    var entityPrecision: Double
    var entityRecall: Double
    var topicRecall: Double
    var baselineRetrievedDocumentIds: [String]?
    var expandedRetrievedDocumentIds: [String]?
    var baselineHitAtK: Bool?
    var expandedHitAtK: Bool?
    var baselineBestRankAtK: Int?
    var expandedBestRankAtK: Int?
    var baselineReciprocalRankAtK: Double?
    var expandedReciprocalRankAtK: Double?
    var reciprocalRankDeltaAtK: Double?
    var rankImprovementAtK: Int?
    var branchDiagnostics: QueryExpansionBranchDiagnostics?
    var missDiagnostics: QueryExpansionMissDiagnostics?
}

private struct QueryExpansionBranchDiagnostics: Codable {
    var classification: String
    var baselineSemanticRankAtK: Int?
    var baselineLexicalRankAtK: Int?
    var expandedSemanticRankAtK: Int?
    var expandedLexicalRankAtK: Int?
    var expansionSourceHits: [QueryExpansionSourceHit]
}

private struct QueryExpansionSourceHit: Codable {
    var kind: String
    var query: String
    var rankAtK: Int?
}

private struct QueryExpansionMissDiagnostics: Codable {
    var diagnosticLimit: Int
    var finalBestRelevantRank: Int?
    var finalBestRelevantDocumentId: String?
    var finalBestRelevantScore: SearchScoreBreakdown?
    var finalTopCandidates: [QueryExpansionRankedCandidate]
    var branchRanks: [QueryExpansionBranchRankDiagnostic]
}

private struct QueryExpansionBranchRankDiagnostic: Codable {
    var branch: String
    var query: String
    var features: [String]
    var bestRelevantRank: Int?
    var bestRelevantDocumentId: String?
    var bestRelevantScore: SearchScoreBreakdown?
    var topCandidates: [QueryExpansionRankedCandidate]
}

private struct QueryExpansionRankedCandidate: Codable {
    var rank: Int
    var documentId: String
    var isRelevant: Bool
    var score: SearchScoreBreakdown
}

private struct QueryExpansionSuiteReport: Codable {
    var totalQueries: Int
    var lexicalCoverageRecall: Double
    var semanticCoverageRecall: Double
    var hydeAnchorRecall: Double
    var facetPrecision: Double
    var facetRecall: Double
    var facetMicroF1: Double
    var entityPrecision: Double
    var entityRecall: Double
    var topicRecall: Double
    var retrievalBaselineHitRate: Double?
    var retrievalExpandedHitRate: Double?
    var retrievalLift: Double?
    var retrievalBaselineMRR: Double?
    var retrievalExpandedMRR: Double?
    var retrievalMRRDelta: Double?
    var failureTaxonomyCounts: [String: Int]?
    var taxonomyMetrics: [QueryExpansionTaxonomyMetric]?
    var branchClassificationCounts: [String: Int]?
    var latencyStats: RecallLatencyStats?
    var stageLatencyStats: RecallStageLatencyStats?
    var candidateCountStats: RecallCandidateCountStats?
    var caseResults: [QueryExpansionCaseResult]
}

private struct QueryExpansionTaxonomyMetric: Codable {
    var tag: String
    var caseCount: Int
    var baselineHitRate: Double
    var expandedHitRate: Double
    var baselineMRR: Double
    var expandedMRR: Double
    var mrrDelta: Double
    var improvedCount: Int
    var worsenedCount: Int
    var flatCount: Int
    var missCount: Int
}

private struct AgentMemoryScenarioResult: Codable {
    var id: String
    var expectedWriteCount: Int
    var extractedCount: Int
    var storedCount: Int
    var matchedExpectedWrites: Int
    var falseWriteCount: Int
    var activeStateCorrect: Bool?
    var expectedUpdateBehavior: String?
    var observedUpdateBehavior: String?
    var recallHitCount: Int
    var recallQueryCount: Int
    var reciprocalRanks: [Double]
    var latencyMs: Double
}

private struct AgentMemorySuiteReport: Codable {
    var totalScenarios: Int
    var falseWriteRate: Double
    var expectedWriteRecall: Double
    var activeStateAccuracy: Double
    var updateBehaviorAccuracy: Double
    var recallHitRate: Double
    var recallMRR: Double
    var latencyStats: RecallLatencyStats?
    var caseResults: [AgentMemoryScenarioResult]
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

private actor DiagnosticStructuredQueryExpander: StructuredQueryExpander {
    let identifier: String
    private let base: any StructuredQueryExpander
    private let diagnostics: RecallDiagnosticsCollector

    init(base: any StructuredQueryExpander, diagnostics: RecallDiagnosticsCollector) {
        self.base = base
        self.diagnostics = diagnostics
        self.identifier = base.identifier
    }

    func expand(
        query: SearchQuery,
        analysis: QueryAnalysis,
        limit: Int
    ) async throws -> StructuredQueryExpansion {
        do {
            let expansion = try await base.expand(query: query, analysis: analysis, limit: limit)
            let alternateCount = expansion.lexicalQueries.count
                + expansion.semanticQueries.count
                + expansion.hypotheticalDocuments.count
            await diagnostics.recordExpansionSuccess(requestedLimit: limit, alternateCount: alternateCount)
            return expansion
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

actor EvalResponseCache {
    private struct CacheStats {
        var hits = 0
        var misses = 0
        var writes = 0
    }

    private let database: SQLiteDatabase
    private var statsByNamespace: [String: CacheStats] = [:]
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(databaseURL: URL) throws {
        database = try SQLiteDatabase(path: databaseURL.path)
        try database.execute(
            sql: """
            CREATE TABLE IF NOT EXISTS eval_provider_cache (
                cache_key TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                value_blob BLOB NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        try database.execute(
            sql: """
            CREATE INDEX IF NOT EXISTS eval_provider_cache_namespace_idx
            ON eval_provider_cache(namespace)
            """
        )
    }

    func load<T: Decodable>(
        namespace: String,
        keyComponents: [String],
        as type: T.Type
    ) throws -> T? {
        let cacheKey = makeCacheKey(namespace: namespace, components: keyComponents)
        let row = try database.fetchOne(
            sql: "SELECT value_blob FROM eval_provider_cache WHERE cache_key = ?",
            arguments: [cacheKey]
        )
        let payload: Data? = row?["value_blob"]

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

        try database.execute(
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

        statsByNamespace[namespace, default: CacheStats()].writes += 1
    }

    fileprivate func drainSummaryLines(suite: SuiteKind) -> [String] {
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

private actor CachingStructuredQueryExpander: StructuredQueryExpander {
    let identifier: String
    private let base: any StructuredQueryExpander
    private let cache: EvalResponseCache

    init(base: any StructuredQueryExpander, cache: EvalResponseCache) {
        self.base = base
        self.cache = cache
        self.identifier = base.identifier
    }

    func expand(
        query: SearchQuery,
        analysis: QueryAnalysis,
        limit: Int
    ) async throws -> StructuredQueryExpansion {
        let keyComponents = [
            identifier,
            query.text,
            analysis.facetHints.map(\.tag.rawValue).sorted().joined(separator: ","),
            analysis.entities.map(\.normalizedValue).sorted().joined(separator: ","),
            analysis.topics.sorted().joined(separator: ","),
            String(limit),
        ]

        if let cached = try await cache.load(
            namespace: "expand",
            keyComponents: keyComponents,
            as: StructuredQueryExpansion.self
        ) {
            return cached
        }

        let expansion = try await base.expand(query: query, analysis: analysis, limit: limit)
        try await cache.store(namespace: "expand", keyComponents: keyComponents, value: expansion)
        return expansion
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
    var queryExpansion: QueryExpansionSuiteReport?
    var agentMemory: AgentMemorySuiteReport?
    var notes: [String]
}

private final class IndexingStageTimingCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var samplesByStage: [IndexingStage: [Double]] = [:]

    func record(_ event: IndexingEvent) {
        guard case let .stageTiming(_, stage, durationMs) = event else { return }
        lock.lock()
        samplesByStage[stage, default: []].append(durationMs)
        lock.unlock()
    }

    func report() -> StorageStageLatencyStats? {
        lock.lock()
        let samples = samplesByStage
        lock.unlock()

        let report = StorageStageLatencyStats(
            typingMs: computeLatencyStats(samples: samples[.typing] ?? []),
            chunkingMs: computeLatencyStats(samples: samples[.chunking] ?? []),
            taggingMs: computeLatencyStats(samples: samples[.tagging] ?? []),
            embeddingMs: computeLatencyStats(samples: samples[.embedding] ?? []),
            indexWriteMs: computeLatencyStats(samples: samples[.indexWrite] ?? []),
            totalMs: computeLatencyStats(samples: samples[.total] ?? [])
        )
        return report.hasData ? report : nil
    }
}

private final class SearchStageTimingCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var timingsByStage: [SearchStage: Double] = [:]
    private var counts = RecallQueryCandidateCounts()

    func record(_ event: SearchEvent) {
        lock.lock()
        defer { lock.unlock() }

        switch event {
        case let .expandedQueries(count):
            counts.expandedQueries = count
        case let .semanticCandidates(count):
            counts.semanticCandidates = count
        case let .lexicalCandidates(count):
            counts.lexicalCandidates = count
        case let .fusedCandidates(count):
            counts.fusedCandidates = count
        case let .reranked(count):
            counts.rerankedCandidates = count
        case let .stageTiming(stage, durationMs):
            timingsByStage[stage, default: 0] += durationMs
        default:
            break
        }
    }

    func queryTimings() -> RecallQueryStageTimings? {
        lock.lock()
        let timings = timingsByStage
        lock.unlock()

        let report = RecallQueryStageTimings(
            analysisMs: timings[.analysis],
            expansionMs: timings[.expansion],
            queryEmbeddingMs: timings[.queryEmbedding],
            semanticSearchMs: timings[.semanticSearch],
            lexicalSearchMs: timings[.lexicalSearch],
            fusionMs: timings[.fusion],
            rerankMs: timings[.rerank],
            totalMs: timings[.total]
        )
        return report.hasData ? report : nil
    }

    func queryCounts() -> RecallQueryCandidateCounts? {
        lock.lock()
        let snapshot = counts
        lock.unlock()
        return snapshot.hasData ? snapshot : nil
    }
}

@main
struct MemoryEvalCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "memory_eval",
        abstract: "Evaluation harness for Memory.swift storage and recall quality.",
        subcommands: [InitCommand.self, RunCommand.self, CompareCommand.self, GateCommand.self, DiagnoseLongMemEvalCommand.self],
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
        try writeIfNeeded(
            root.appendingPathComponent("query_expansion_cases.jsonl"),
            content: queryExpansionCasesTemplate,
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
        let storageReport: StorageSuiteReport
        var notes: [String] = []
        if dataset.storageCases.isEmpty {
            storageReport = makeEmptyStorageSuiteReport()
            notes.append("No storage cases were provided; storage suite skipped.")
        } else {
            storageReport = try await runStorageSuite(
                profile: profile,
                dataset: dataset.storageCases,
                datasetRoot: datasetRootURL,
                root: runRoot,
                indexCacheEnabled: indexCache,
                verbose: verbose,
                responseCache: responseCache
            )
        }

        let recallOutput: RecallSuiteRunOutput
        if dataset.recallDocuments.isEmpty || dataset.recallQueries.isEmpty {
            recallOutput = makeEmptyRecallSuiteRunOutput(kValues: kValues)
            notes.append("No recall documents or queries were provided; recall suite skipped.")
        } else {
            recallOutput = try await runRecallSuite(
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
        }

        let queryExpansionReport: QueryExpansionSuiteReport?
        if dataset.queryExpansionCases.isEmpty {
            queryExpansionReport = nil
        } else {
            queryExpansionReport = try await runQueryExpansionSuite(
                profile: profile,
                cases: dataset.queryExpansionCases,
                documents: dataset.recallDocuments,
                kValues: kValues,
                datasetRoot: datasetRootURL,
                root: runRoot,
                indexCacheEnabled: indexCache,
                verbose: verbose,
                responseCache: responseCache
            )
        }

        let agentMemoryReport: AgentMemorySuiteReport?
        if dataset.agentMemoryScenarios.isEmpty {
            agentMemoryReport = nil
        } else {
            agentMemoryReport = try await runAgentMemorySuite(
                profile: profile,
                scenarios: dataset.agentMemoryScenarios,
                datasetRoot: datasetRootURL,
                root: runRoot,
                verbose: verbose,
                responseCache: responseCache
            )
        }

        let report = EvalRunReport(
            schemaVersion: 5,
            createdAt: Date(),
            profile: profile,
            datasetRoot: datasetRootURL.path,
            storage: storageReport,
            recall: recallOutput.report,
            queryExpansion: queryExpansionReport,
            agentMemory: agentMemoryReport,
            notes: [
                storageReport.mode == "canonical_memory_schema"
                    ? "Storage eval uses the canonical memory extraction and ingest path and scores kind/status/facet/entity/topic/update behavior."
                    : "Storage eval uses direct database inspection with production-like chunking for classification/span metrics.",
                "Recall eval uses document-level metrics (deduped by document path).",
                "Recall documents are indexed without injected memory_type frontmatter to stress automatic classification.",
            ] + notes + recallOutput.notes + (queryExpansionReport != nil ? [
                "Query expansion eval scores lexical/semantic/HyDE coverage, facet/entity/topic extraction, and optional retrieval lift/MRR delta with expansion enabled vs disabled."
            ] : []) + (agentMemoryReport != nil ? [
                "Agent memory scenario eval scores extraction writes, false writes, active-state/update behavior, recall quality, and latency."
            ] : [])
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
        if report.storage.totalCases == 0 {
            print("Storage suite: skipped")
        } else if report.storage.mode == "canonical_memory_schema" {
            print("Storage kind accuracy: \(percent(report.storage.typeAccuracy))")
            print("Storage kind macro F1: \(percent(report.storage.macroF1))")
            if let facetMicroF1 = report.storage.facetMicroF1 {
                print("Storage facet micro F1: \(percent(facetMicroF1))")
            }
            if let entityRecall = report.storage.entityRecall {
                print("Storage entity recall: \(percent(entityRecall))")
            }
            if let topicRecall = report.storage.topicRecall {
                print("Storage topic recall: \(percent(topicRecall))")
            }
            if let updateBehaviorAccuracy = report.storage.updateBehaviorAccuracy {
                print("Storage update behavior accuracy: \(percent(updateBehaviorAccuracy))")
            }
        } else {
            print("Storage type accuracy: \(percent(report.storage.typeAccuracy))")
            print("Storage macro F1: \(percent(report.storage.macroF1))")
            print("Storage span coverage: \(percent(report.storage.spanCoverage))")
        }
        if report.recall.totalQueries == 0 {
            print("Recall suite: skipped")
        } else if let maxKMetric = report.recall.metricsByK.max(by: { $0.k < $1.k }) {
            print("Recall Hit@\(maxKMetric.k): \(percent(maxKMetric.hitRate))")
            print("Recall Recall@\(maxKMetric.k): \(percent(maxKMetric.recall))")
            print("Recall MRR@\(maxKMetric.k): \(format(maxKMetric.mrr))")
            print("Recall nDCG@\(maxKMetric.k): \(format(maxKMetric.ndcg))")
        }
        if report.recall.totalQueries > 0, let latencyStats = report.recall.latencyStats {
            print("Search latency: p50=\(String(format: "%.0f", latencyStats.p50Ms))ms p95=\(String(format: "%.0f", latencyStats.p95Ms))ms mean=\(String(format: "%.0f", latencyStats.meanMs))ms")
        }
        if let queryExpansion = report.queryExpansion {
            print("Expansion lexical coverage recall: \(percent(queryExpansion.lexicalCoverageRecall))")
            print("Expansion semantic coverage recall: \(percent(queryExpansion.semanticCoverageRecall))")
            print("Expansion HyDE anchor recall: \(percent(queryExpansion.hydeAnchorRecall))")
            print("Expansion facet micro F1: \(percent(queryExpansion.facetMicroF1))")
            if let expandedHitRate = queryExpansion.retrievalExpandedHitRate {
                print("Expansion retrieval Hit@\(kValues.max() ?? 10): \(percent(expandedHitRate))")
                if let lift = queryExpansion.retrievalLift {
                    print("Expansion retrieval lift: \(percent(lift))")
                }
            }
            if let expandedMRR = queryExpansion.retrievalExpandedMRR {
                print("Expansion retrieval MRR@\(kValues.max() ?? 10): \(format(expandedMRR))")
                if let mrrDelta = queryExpansion.retrievalMRRDelta {
                    print("Expansion retrieval MRR delta: \(format(mrrDelta))")
                }
            }
        }
        if let agentMemory = report.agentMemory {
            print("Agent memory false-write rate: \(percent(agentMemory.falseWriteRate))")
            print("Agent memory expected-write recall: \(percent(agentMemory.expectedWriteRecall))")
            print("Agent memory active-state accuracy: \(percent(agentMemory.activeStateAccuracy))")
            print("Agent memory update behavior accuracy: \(percent(agentMemory.updateBehaviorAccuracy))")
            print("Agent memory recall Hit: \(percent(agentMemory.recallHitRate))")
            print("Agent memory recall MRR: \(format(agentMemory.recallMRR))")
        }
        if let ingestTotal = report.storage.stageLatencyStats?.totalMs {
            print("Indexing total/doc: p50=\(String(format: "%.0f", ingestTotal.p50Ms))ms p95=\(String(format: "%.0f", ingestTotal.p95Ms))ms mean=\(String(format: "%.0f", ingestTotal.meanMs))ms")
        }
        if let queryStages = report.recall.stageLatencyStats {
            print("Query stage p95: analysis=\(formatStageMs(queryStages.analysisMs, keyPath: \.p95Ms)) lexical=\(formatStageMs(queryStages.lexicalSearchMs, keyPath: \.p95Ms)) semantic=\(formatStageMs(queryStages.semanticSearchMs, keyPath: \.p95Ms)) rerank=\(formatStageMs(queryStages.rerankMs, keyPath: \.p95Ms)) total=\(formatStageMs(queryStages.totalMs, keyPath: \.p95Ms))")
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

private struct EvalBaselineManifest: Codable {
    var schemaVersion: Int
    var createdAt: Date?
    var freshnessHours: Double?
    var regressionThreshold: Double?
    var latencyRegressionThreshold: Double?
    var requiredRuns: [EvalBaselineRequirement]
}

private struct EvalBaselineRequirement: Codable {
    var dataset: String
    var datasetRoot: String?
    var profile: EvalProfile
    var maxAgeHours: Double?
    var metrics: [String: Double]
    var minimumMetrics: [String: Double]?
    var maximumMetrics: [String: Double]?
    var latencyP95Ms: Double?
}

struct GateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "gate",
        abstract: "Validate eval run artifacts against a reduced baseline manifest."
    )

    @Option(name: .long, help: "Path to eval baseline manifest JSON.")
    var baseline: String = "Evals/baselines/current.json"

    @Option(name: .long, help: "Override maximum run artifact age in hours.")
    var maxAgeHours: Double?

    @Argument(help: "Candidate eval run JSON files.")
    var runs: [String]

    mutating func run() async throws {
        guard !runs.isEmpty else {
            throw ValidationError("Provide candidate run JSON paths.")
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        let baselineURL = URL(fileURLWithPath: NSString(string: baseline).expandingTildeInPath).standardizedFileURL
        let manifest = try decoder.decode(EvalBaselineManifest.self, from: Data(contentsOf: baselineURL))

        let reports = try runs.map { path in
            let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath).standardizedFileURL
            return try decoder.decode(EvalRunReport.self, from: Data(contentsOf: url))
        }

        var failures: [String] = []
        let metricThreshold = manifest.regressionThreshold ?? 0.02
        let latencyThreshold = manifest.latencyRegressionThreshold ?? 0.15
        let now = Date()

        for requirement in manifest.requiredRuns {
            guard let report = reports.first(where: { reportMatchesRequirement($0, requirement: requirement) }) else {
                failures.append("Missing run for \(requirement.dataset) / \(requirement.profile.rawValue).")
                continue
            }

            let allowedAge = maxAgeHours ?? requirement.maxAgeHours ?? manifest.freshnessHours
            if let allowedAge, allowedAge > 0 {
                let ageHours = now.timeIntervalSince(report.createdAt) / 3600.0
                if ageHours > allowedAge {
                    failures.append(
                        "\(requirement.dataset) / \(requirement.profile.rawValue) is stale: \(String(format: "%.1f", ageHours))h old > \(String(format: "%.1f", allowedAge))h."
                    )
                }
            }

            let candidateMetrics = reducedMetrics(from: report)
            for (name, minimumValue) in (requirement.minimumMetrics ?? [:]).sorted(by: { $0.key < $1.key }) {
                guard let candidateValue = candidateMetrics[name] else {
                    failures.append("\(requirement.dataset) / \(requirement.profile.rawValue) missing minimum metric \(name).")
                    continue
                }
                if candidateValue < minimumValue {
                    failures.append(
                        "\(requirement.dataset) / \(requirement.profile.rawValue) \(name) below minimum (minimum \(format(minimumValue)) -> candidate \(format(candidateValue)))."
                    )
                }
            }

            for (name, maximumValue) in (requirement.maximumMetrics ?? [:]).sorted(by: { $0.key < $1.key }) {
                guard let candidateValue = candidateMetrics[name] else {
                    failures.append("\(requirement.dataset) / \(requirement.profile.rawValue) missing maximum metric \(name).")
                    continue
                }
                if candidateValue > maximumValue {
                    failures.append(
                        "\(requirement.dataset) / \(requirement.profile.rawValue) \(name) above maximum (maximum \(format(maximumValue)) -> candidate \(format(candidateValue)))."
                    )
                }
            }

            for (name, baselineValue) in requirement.metrics.sorted(by: { $0.key < $1.key }) {
                guard let candidateValue = candidateMetrics[name] else {
                    failures.append("\(requirement.dataset) / \(requirement.profile.rawValue) missing metric \(name).")
                    continue
                }

                if metricIsLowerBetter(name) {
                    let regression = candidateValue - baselineValue
                    if regression > metricThreshold {
                        failures.append(
                            "\(requirement.dataset) / \(requirement.profile.rawValue) \(name) regressed by \(percent(regression)) (baseline \(format(baselineValue)) -> candidate \(format(candidateValue)))."
                        )
                    }
                } else {
                    let regression = baselineValue - candidateValue
                    if regression > metricThreshold {
                        failures.append(
                            "\(requirement.dataset) / \(requirement.profile.rawValue) \(name) regressed by \(percent(regression)) (baseline \(format(baselineValue)) -> candidate \(format(candidateValue)))."
                        )
                    }
                }
            }

            if let baselineLatency = requirement.latencyP95Ms,
               let candidateLatency = p95Latency(from: report),
               baselineLatency > 0 {
                let limit = baselineLatency * (1.0 + latencyThreshold)
                if candidateLatency > limit {
                    failures.append(
                        "\(requirement.dataset) / \(requirement.profile.rawValue) p95 latency regressed: \(String(format: "%.1f", candidateLatency))ms > \(String(format: "%.1f", limit))ms."
                    )
                }
            } else if requirement.latencyP95Ms != nil {
                failures.append("\(requirement.dataset) / \(requirement.profile.rawValue) missing p95 latency.")
            }
        }

        if failures.isEmpty {
            print("Eval gate passed for \(manifest.requiredRuns.count) required run(s).")
        } else {
            print("Eval gate failed:")
            for failure in failures {
                print("- \(failure)")
            }
            throw ExitCode(1)
        }
    }
}

struct DiagnoseLongMemEvalCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "diagnose-longmemeval",
        abstract: "Rerun LongMemEval recall misses across retrieval branches and write diagnostic artifacts."
    )

    @Option(name: .long, help: "Profile to run.")
    var profile: EvalProfile = .coreMLDefault

    @Option(name: .long, help: "Dataset root folder.")
    var datasetRoot: String = "Evals/longmemeval_v2"

    @Option(name: .long, help: "Source eval run JSON used to select miss/partial cases.")
    var sourceRun: String

    @Option(name: .long, help: "Output JSON path. Defaults beside --source-run.")
    var output: String?

    @Option(name: .long, help: "Case scope: misses, misses-and-partials, or all.")
    var scope: String = "misses-and-partials"

    @Option(name: .long, help: "Wide diagnostic retrieval limit.")
    var wideLimit: Int = 100

    @Flag(
        name: .long,
        inversion: .prefixedNo,
        help: "Reuse cached recall index across diagnostic runs."
    )
    var indexCache = true

    mutating func run() async throws {
        let datasetRootURL = URL(fileURLWithPath: NSString(string: datasetRoot).expandingTildeInPath).standardizedFileURL
        let sourceRunURL = URL(fileURLWithPath: NSString(string: sourceRun).expandingTildeInPath).standardizedFileURL

        let dataset = try loadDataset(root: datasetRootURL)
        guard !dataset.recallDocuments.isEmpty, !dataset.recallQueries.isEmpty else {
            throw ValidationError("diagnose-longmemeval requires recall_documents.jsonl and recall_queries.jsonl.")
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let sourceReport = try decoder.decode(EvalRunReport.self, from: Data(contentsOf: sourceRunURL))
        let maxK = sourceReport.recall.metricsByK.max(by: { $0.k < $1.k })?.k ?? 10
        let normalizedScope = scope.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let selectedIDs = try selectRecallDiagnosticIDs(
            from: sourceReport.recall.queryResults,
            maxK: maxK,
            scope: normalizedScope
        )
        let selectedIDSet = Set(selectedIDs)
        let selectedQueries = dataset.recallQueries.filter { selectedIDSet.contains($0.id) }
        guard !selectedQueries.isEmpty else {
            throw ValidationError("No LongMemEval cases matched scope '\(scope)'.")
        }

        let runRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-evals", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: runRoot, withIntermediateDirectories: true)

        let indexSeed = recallIndexCacheSeed(profile: profile, documents: dataset.recallDocuments)
        let workspace = try prepareIndexWorkspace(
            suite: .recall,
            profile: profile,
            datasetRoot: datasetRootURL,
            runRoot: runRoot,
            cacheEnabled: indexCache,
            seed: indexSeed
        )

        var pathByDocumentID: [String: String] = [:]
        var documentIDByPath: [String: String] = [:]
        for document in dataset.recallDocuments {
            let path = materializedRecallDocumentURL(document, docsRoot: workspace.docsRoot)
            pathByDocumentID[document.id] = path.path
            documentIDByPath[path.path] = document.id
        }

        let expectedPaths = pathByDocumentID.values.map(URL.init(fileURLWithPath:))
        let canReuseIndex = indexCacheCanReuse(workspace: workspace, expectedDocumentPaths: expectedPaths)
        if canReuseIndex {
            print("[diagnose-longmemeval][index-cache] hit: \(workspace.root.path)")
        } else {
            if workspace.cacheEnabled {
                print("[diagnose-longmemeval][index-cache] miss: \(workspace.root.path)")
            }
            try resetWorkspaceForRebuild(workspace)
            for document in dataset.recallDocuments {
                guard let pathRaw = pathByDocumentID[document.id] else { continue }
                let path = URL(fileURLWithPath: pathRaw)
                try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
                try materializeRecallDocument(document).write(to: path, atomically: true, encoding: .utf8)
            }
        }

        var config = try buildConfiguration(profile: profile, suite: .recall, databaseURL: workspace.databaseURL)
        let recallDiagnostics = installRecallDiagnosticsIfNeeded(profile: profile, configuration: &config)
        let index = try MemoryIndex(configuration: config)
        if canReuseIndex {
            print("[diagnose-longmemeval] Using cached index for \(dataset.recallDocuments.count) documents.")
        } else {
            print("[diagnose-longmemeval] Building index for \(dataset.recallDocuments.count) documents...")
            let start = Date()
            try await index.rebuildIndex(from: [workspace.docsRoot])
            print("[diagnose-longmemeval] Index built in \(formatDuration(Date().timeIntervalSince(start))).")
            try markIndexCacheReady(workspace)
        }

        let sourceResultsByID = Dictionary(uniqueKeysWithValues: sourceReport.recall.queryResults.map { ($0.id, $0) })
        let allFeatures = recallFeatures(for: config)
        var noExpansionFeatures = allFeatures
        noExpansionFeatures.remove(.expansion)

        let diagnosticLimit = max(wideLimit, maxK)
        let branchSpecs: [(name: String, features: RecallFeatures, limit: Int)] = [
            ("current_top_k", allFeatures, maxK),
            ("current_wide", allFeatures, diagnosticLimit),
            ("no_expansion_wide", noExpansionFeatures, diagnosticLimit),
            ("lexical_wide", [.lexical], diagnosticLimit),
            ("semantic_wide", [.semantic], diagnosticLimit),
        ]

        var caseResults: [RecallBranchDiagnosticCase] = []
        var progress = DeterminateProgress(label: "diagnose-longmemeval", total: selectedQueries.count)
        for queryCase in selectedQueries {
            guard let sourceResult = sourceResultsByID[queryCase.id] else { continue }
            let relevant = Set(queryCase.relevantDocumentIds)
            var branches: [RecallBranchDiagnosticBranch] = []
            for spec in branchSpecs {
                let branch = try await runRecallBranchDiagnostic(
                    index: index,
                    query: queryCase.query,
                    relevantDocumentIds: relevant,
                    documentIDByPath: documentIDByPath,
                    name: spec.name,
                    features: spec.features,
                    limit: spec.limit,
                    evaluationK: maxK
                )
                branches.append(branch)
            }

            let originalHit = sourceResult.hitByK[maxK] ?? false
            let originalRecall = sourceResult.recallByK[maxK] ?? 0
            let taxonomy = recallDiagnosticTaxonomy(
                query: queryCase.query,
                relevantCount: queryCase.relevantDocumentIds.count,
                memoryTypes: queryCase.memoryTypes ?? []
            )
            caseResults.append(
                RecallBranchDiagnosticCase(
                    id: queryCase.id,
                    query: queryCase.query,
                    difficulty: queryCase.difficulty,
                    memoryTypes: queryCase.memoryTypes ?? [],
                    relevantDocumentIds: queryCase.relevantDocumentIds,
                    originalRetrievedDocumentIds: sourceResult.retrievedDocumentIds,
                    originalHitAtK: originalHit,
                    originalRecallAtK: originalRecall,
                    classification: classifyRecallDiagnosticCase(
                        originalHitAtK: originalHit,
                        originalRecallAtK: originalRecall,
                        maxK: maxK,
                        relevantCount: queryCase.relevantDocumentIds.count,
                        branches: branches
                    ),
                    taxonomy: taxonomy,
                    branchResults: branches
                )
            )
            progress.advance(detail: queryCase.id)
        }

        if let recallDiagnostics {
            for line in await recallDiagnostics.summaryLines(suite: .recall) {
                print(line)
            }
            for line in await recallDiagnostics.detailLines(suite: .recall) {
                print(line)
            }
        }

        let report = RecallBranchDiagnosticReport(
            schemaVersion: 1,
            createdAt: Date(),
            profile: profile,
            datasetRoot: datasetRootURL.path,
            sourceRun: sourceRunURL.path,
            scope: normalizedScope,
            maxK: maxK,
            wideLimit: diagnosticLimit,
            totalCases: caseResults.count,
            classificationCounts: countBy(caseResults.map(\.classification)),
            taxonomyCounts: countBy(caseResults.flatMap(\.taxonomy)),
            branchHitCounts: recallDiagnosticBranchHitCounts(caseResults),
            caseResults: caseResults.sorted { $0.id < $1.id }
        )

        let outputURL = try resolveLongMemEvalDiagnosticOutput(sourceRunURL: sourceRunURL, output: output)
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(report).write(to: outputURL, options: .atomic)

        let markdown = makeRecallBranchDiagnosticMarkdown(report)
        let markdownURL = outputURL.deletingPathExtension().appendingPathExtension("md")
        try markdown.write(to: markdownURL, atomically: true, encoding: .utf8)

        print("Diagnostic JSON: \(outputURL.path)")
        print("Diagnostic Markdown: \(markdownURL.path)")
    }
}

private func selectRecallDiagnosticIDs(
    from results: [RecallQueryResult],
    maxK: Int,
    scope: String
) throws -> [String] {
    switch scope {
    case "misses":
        return results.filter { ($0.hitByK[maxK] ?? false) == false }.map(\.id)
    case "misses-and-partials":
        return results.filter { result in
            let hit = result.hitByK[maxK] ?? false
            let recall = result.recallByK[maxK] ?? 0
            return !hit || recall < 1
        }.map(\.id)
    case "all":
        return results.map(\.id)
    default:
        throw ValidationError("Unsupported scope '\(scope)'. Use misses, misses-and-partials, or all.")
    }
}

private func runRecallBranchDiagnostic(
    index: MemoryIndex,
    query: String,
    relevantDocumentIds: Set<String>,
    documentIDByPath: [String: String],
    name: String,
    features: RecallFeatures,
    limit: Int,
    evaluationK: Int
) async throws -> RecallBranchDiagnosticBranch {
    let collector = SearchStageTimingCollector()
    let start = Date()
    let references = try await index.memorySearch(
        query: query,
        limit: limit,
        features: features,
        dedupeDocuments: true,
        includeLineRanges: false,
        events: { event in
            collector.record(event)
        }
    )
    let latencyMs = Date().timeIntervalSince(start) * 1000.0
    var rankedCandidates: [RecallBranchDiagnosticRankedCandidate] = []
    rankedCandidates.reserveCapacity(references.count)
    for reference in references {
        guard let documentID = documentIDByPath[reference.documentPath] else { continue }
        rankedCandidates.append(
            RecallBranchDiagnosticRankedCandidate(
                rank: rankedCandidates.count + 1,
                documentId: documentID,
                score: reference.score
            )
        )
    }
    let retrievedDocumentIds = rankedCandidates.map(\.documentId)
    let bestRank = retrievedDocumentIds.firstIndex(where: { relevantDocumentIds.contains($0) }).map { $0 + 1 }
    let relevantHitCount = retrievedDocumentIds.reduce(into: 0) { count, documentID in
        if relevantDocumentIds.contains(documentID) {
            count += 1
        }
    }
    let relevantHitCountAtK = retrievedDocumentIds.prefix(evaluationK).reduce(into: 0) { count, documentID in
        if relevantDocumentIds.contains(documentID) {
            count += 1
        }
    }

    return RecallBranchDiagnosticBranch(
        name: name,
        features: queryExpansionFeatureLabels(features),
        limit: limit,
        latencyMs: latencyMs,
        bestRelevantRank: bestRank,
        relevantHitCount: relevantHitCount,
        relevantHitCountAtK: relevantHitCountAtK,
        retrievedDocumentIds: Array(retrievedDocumentIds.prefix(limit)),
        topCandidates: Array(rankedCandidates.prefix(evaluationK)),
        relevantCandidates: rankedCandidates.filter { relevantDocumentIds.contains($0.documentId) },
        stageTimings: collector.queryTimings(),
        candidateCounts: collector.queryCounts()
    )
}

private func classifyRecallDiagnosticCase(
    originalHitAtK: Bool,
    originalRecallAtK: Double,
    maxK: Int,
    relevantCount: Int,
    branches: [RecallBranchDiagnosticBranch]
) -> String {
    let byName = Dictionary(uniqueKeysWithValues: branches.map { ($0.name, $0) })
    let currentWide = byName["current_wide"]
    let noExpansion = byName["no_expansion_wide"]
    let lexical = byName["lexical_wide"]
    let semantic = byName["semantic_wide"]

    let currentWideTopHit = (currentWide?.bestRelevantRank).map { $0 <= maxK } ?? false
    let currentWideHasCandidate = currentWide?.bestRelevantRank != nil
    let branchHit = branches.contains { $0.bestRelevantRank != nil }
    let noExpansionTopHit = (noExpansion?.bestRelevantRank).map { $0 <= maxK } ?? false
    let lexicalHit = lexical?.bestRelevantRank != nil
    let semanticHit = semantic?.bestRelevantRank != nil

    if !originalHitAtK, currentWideTopHit {
        return "fixed_by_current_code"
    }

    if originalHitAtK, originalRecallAtK < 1, relevantCount > 1 {
        let topRelevant = currentWide?.relevantHitCountAtK ?? 0
        let wideRelevant = currentWide?.relevantHitCount ?? 0
        if wideRelevant > topRelevant {
            return "multi_evidence_preservation"
        }
        return "partial_multi_evidence"
    }

    if noExpansionTopHit, !currentWideTopHit {
        return "expansion_regression"
    }

    if currentWideHasCandidate, !currentWideTopHit {
        return "ranking_or_pool_depth"
    }

    if (lexicalHit || semanticHit), !currentWideHasCandidate {
        return "fusion_filtering"
    }

    if !branchHit {
        return "candidate_generation"
    }

    return currentWideTopHit ? "covered" : "candidate_generation"
}

private func recallDiagnosticTaxonomy(
    query: String,
    relevantCount: Int,
    memoryTypes: [String]
) -> [String] {
    let lower = query.lowercased()
    var labels: [String] = []
    let normalizedTypes = Set(memoryTypes.map { $0.lowercased() })

    let temporalNeedles = [
        "how many", "what time", "when", "which day", "date", "day", "week",
        "month", "year", "before", "after", "since", "during", "past", "last",
        "currently", "now", "typical week"
    ]
    if normalizedTypes.contains("temporal") || temporalNeedles.contains(where: lower.contains) {
        labels.append("temporal/count")
    }

    let multiNeedles = ["across", "all the", "between", "total", "combined", "from ", " and "]
    if relevantCount > 1 || multiNeedles.contains(where: lower.contains) {
        labels.append("multi-evidence")
    }

    let contextualNeedles = ["previous conversation", "previous chat", "remind me", "that ", "this ", " it ", "those "]
    if contextualNeedles.contains(where: lower.contains) {
        labels.append("contextual-ellipsis")
    }

    let hasQuotedPhrase = query.contains("'") || query.contains("\"")
    let hasProperEntity = query.range(of: #"\b[A-Z][A-Za-z0-9_-]{2,}\b"#, options: .regularExpression) != nil
    if hasQuotedPhrase || hasProperEntity || normalizedTypes.contains("entity") {
        labels.append("entity/alias")
    }

    if normalizedTypes.contains("episodic"), !labels.contains("temporal/count") {
        labels.append("episodic/contextual")
    }

    return labels.isEmpty ? ["lexical/semantic-mismatch"] : labels
}

private func recallDiagnosticBranchHitCounts(_ cases: [RecallBranchDiagnosticCase]) -> [String: Int] {
    var counts: [String: Int] = [:]
    for diagnosticCase in cases {
        for branch in diagnosticCase.branchResults where branch.bestRelevantRank != nil {
            counts[branch.name, default: 0] += 1
        }
    }
    return counts
}

private func countBy(_ values: [String]) -> [String: Int] {
    values.reduce(into: [:]) { counts, value in
        counts[value, default: 0] += 1
    }
}

private func resolveLongMemEvalDiagnosticOutput(sourceRunURL: URL, output: String?) throws -> URL {
    if let output {
        return URL(fileURLWithPath: NSString(string: output).expandingTildeInPath).standardizedFileURL
    }
    return sourceRunURL
        .deletingPathExtension()
        .appendingPathExtension("branch-diagnostics.json")
}

private func makeRecallBranchDiagnosticMarkdown(_ report: RecallBranchDiagnosticReport) -> String {
    var lines: [String] = []
    lines.append("# LongMemEval Branch Diagnostics")
    lines.append("")
    lines.append("- Profile: `\(report.profile.rawValue)`")
    lines.append("- Source run: `\(report.sourceRun)`")
    lines.append("- Scope: `\(report.scope)`")
    lines.append("- Cases: \(report.totalCases)")
    lines.append("- Max K: \(report.maxK)")
    lines.append("- Wide limit: \(report.wideLimit)")
    lines.append("")

    lines.append("## Classification Counts")
    lines.append("| Classification | Count |")
    lines.append("|---|---:|")
    for entry in report.classificationCounts.sorted(by: { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }) {
        lines.append("| `\(markdownTableCell(entry.key))` | \(entry.value) |")
    }
    lines.append("")

    lines.append("## Taxonomy Counts")
    lines.append("| Taxonomy | Count |")
    lines.append("|---|---:|")
    for entry in report.taxonomyCounts.sorted(by: { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }) {
        lines.append("| `\(markdownTableCell(entry.key))` | \(entry.value) |")
    }
    lines.append("")

    lines.append("## Branch Coverage")
    lines.append("| Branch | Cases with relevant doc |")
    lines.append("|---|---:|")
    for entry in report.branchHitCounts.sorted(by: { $0.key < $1.key }) {
        lines.append("| `\(markdownTableCell(entry.key))` | \(entry.value) |")
    }
    lines.append("")

    lines.append("## Cases")
    lines.append("| ID | Classification | Taxonomy | Original | Branch ranks |")
    lines.append("|---|---|---|---|---|")
    for result in report.caseResults {
        let original = result.originalHitAtK
            ? "hit, recall \(format(result.originalRecallAtK))"
            : "miss"
        let branchRanks = result.branchResults.map { branch -> String in
            if let rank = branch.bestRelevantRank {
                let relevantSummary = branch.relevantCandidates
                    .prefix(4)
                    .map { candidate in
                        "\(candidate.rank)[\(queryExpansionScoreSummary(candidate.score))]"
                    }
                    .joined(separator: ",")
                if relevantSummary.isEmpty {
                    return "\(branch.name): \(rank)/\(branch.limit)"
                }
                return "\(branch.name): \(rank)/\(branch.limit) {\(relevantSummary)}"
            }
            return "\(branch.name): -"
        }.joined(separator: "; ")
        lines.append(
            "| `\(markdownTableCell(result.id))` | `\(markdownTableCell(result.classification))` | \(markdownTableCell(result.taxonomy.joined(separator: ", "))) | \(markdownTableCell(original)) | \(markdownTableCell(branchRanks)) |"
        )
    }
    lines.append("")
    return lines.joined(separator: "\n")
}

private struct DatasetBundle {
    var storageCases: [StorageCase]
    var recallDocuments: [RecallDocumentCase]
    var recallQueries: [RecallQueryCase]
    var queryExpansionCases: [QueryExpansionCase]
    var agentMemoryScenarios: [AgentMemoryScenarioCase]
}

private func loadDataset(root: URL) throws -> DatasetBundle {
    let storageURL = root.appendingPathComponent("storage_cases.jsonl")
    let recallDocumentsURL = root.appendingPathComponent("recall_documents.jsonl")
    let recallQueriesURL = root.appendingPathComponent("recall_queries.jsonl")
    let agentMemoryScenariosURL = root.appendingPathComponent("scenarios.jsonl")
    let queryExpansionURLCandidates = [
        root.appendingPathComponent("query_expansion_cases.jsonl"),
        root.appendingPathComponent("cases.jsonl"),
    ]

    let fileManager = FileManager.default
    let storageExists = fileManager.fileExists(atPath: storageURL.path)
    let recallDocumentsExists = fileManager.fileExists(atPath: recallDocumentsURL.path)
    let recallQueriesExists = fileManager.fileExists(atPath: recallQueriesURL.path)
    let agentMemoryScenariosExists = fileManager.fileExists(atPath: agentMemoryScenariosURL.path)
    let queryExpansionURL = queryExpansionURLCandidates.first { fileManager.fileExists(atPath: $0.path) }

    guard storageExists || recallDocumentsExists || recallQueriesExists || queryExpansionURL != nil || agentMemoryScenariosExists else {
        throw ValidationError(
            "No dataset files found in \(root.path). Expected one or more of storage_cases.jsonl, recall_documents.jsonl, recall_queries.jsonl, query_expansion_cases.jsonl, cases.jsonl, or scenarios.jsonl."
        )
    }

    let storageCases: [StorageCase] = storageExists ? try loadJSONLines(from: storageURL) : []
    let recallDocuments: [RecallDocumentCase] = recallDocumentsExists ? try loadJSONLines(from: recallDocumentsURL) : []
    let recallQueries: [RecallQueryCase] = recallQueriesExists ? try loadJSONLines(from: recallQueriesURL) : []
    let agentMemoryScenarios: [AgentMemoryScenarioCase] = agentMemoryScenariosExists ? try loadJSONLines(from: agentMemoryScenariosURL) : []
    let queryExpansionCases: [QueryExpansionCase]
    if let queryExpansionURL {
        queryExpansionCases = try loadJSONLines(from: queryExpansionURL)
    } else {
        queryExpansionCases = []
    }

    if storageExists && storageCases.isEmpty {
        throw ValidationError("storage_cases.jsonl must contain at least one case when present.")
    }
    if recallQueriesExists && !recallDocumentsExists {
        throw ValidationError("recall_queries.jsonl requires recall_documents.jsonl.")
    }
    if recallDocumentsExists && recallQueries.isEmpty && queryExpansionCases.isEmpty {
        throw ValidationError(
            "recall_documents.jsonl requires recall_queries.jsonl unless query_expansion_cases.jsonl is present for the query-expansion suite."
        )
    }
    if queryExpansionURL != nil && queryExpansionCases.isEmpty {
        throw ValidationError("\(queryExpansionURL!.lastPathComponent) must contain at least one case when present.")
    }
    if agentMemoryScenariosExists && agentMemoryScenarios.isEmpty {
        throw ValidationError("scenarios.jsonl must contain at least one case when present.")
    }
    if !storageExists && recallDocuments.isEmpty && recallQueries.isEmpty && queryExpansionCases.isEmpty && agentMemoryScenarios.isEmpty {
        throw ValidationError("Dataset root \(root.path) does not contain any runnable eval cases.")
    }

    return DatasetBundle(
        storageCases: storageCases,
        recallDocuments: recallDocuments,
        recallQueries: recallQueries,
        queryExpansionCases: queryExpansionCases,
        agentMemoryScenarios: agentMemoryScenarios
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

private func makeEmptyStorageSuiteReport() -> StorageSuiteReport {
    StorageSuiteReport(
        mode: "skipped",
        totalCases: 0,
        typeAccuracy: 0,
        macroF1: 0,
        spanCoverage: 0,
        fallbackRate: 0,
        facetPrecision: nil,
        facetRecall: nil,
        facetMicroF1: nil,
        entityPrecision: nil,
        entityRecall: nil,
        topicRecall: nil,
        updateBehaviorAccuracy: nil,
        confusionMatrix: [:],
        caseResults: [],
        stageLatencyStats: nil
    )
}

private func makeEmptyRecallSuiteRunOutput(kValues: [Int]) -> RecallSuiteRunOutput {
    RecallSuiteRunOutput(
        report: RecallSuiteReport(
            totalQueries: 0,
            kValues: kValues,
            metricsByK: kValues.map { RecallPerKMetric(k: $0, hitRate: 0, recall: 0, mrr: 0, ndcg: 0) },
            queryResults: [],
            perTypeMetrics: nil,
            perDifficultyMetrics: nil,
            latencyStats: nil,
            stageLatencyStats: nil,
            candidateCountStats: nil
        ),
        notes: []
    )
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
        parts.append("expected_type=\(entry.expectedMemoryType ?? "")")
        parts.append("expected_kind=\(entry.expectedKind ?? "")")
        parts.append("expected_status=\(entry.expectedStatus ?? "")")
        parts.append("expected_facets=\((entry.expectedFacets ?? []).joined(separator: "\u{1E}"))")
        parts.append("required_spans=\((entry.requiredSpans ?? []).joined(separator: "\u{1E}"))")
        parts.append("required_entities=\((entry.requiredEntities ?? []).joined(separator: "\u{1E}"))")
        parts.append("required_topics=\((entry.requiredTopics ?? []).joined(separator: "\u{1E}"))")
        parts.append("update=\(entry.expectedUpdateBehavior ?? "")")
        parts.append("canonical_key=\(entry.canonicalKey ?? "")")
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

private func queryExpansionIndexCacheSeed(profile: EvalProfile, documents: [RecallDocumentCase], cases: [QueryExpansionCase]) -> String {
    var parts: [String] = [
        "suite=query_expansion",
        "profile=\(profile.rawValue)",
    ]
    parts.append(recallIndexCacheSeed(profile: profile, documents: documents))
    for entry in cases.sorted(by: { $0.id < $1.id }) {
        parts.append("id=\(entry.id)")
        parts.append("query=\(entry.query)")
        parts.append("expected_lexical=\((entry.expectedLexicalTerms ?? []).joined(separator: "\u{1E}"))")
        parts.append("expected_semantic=\((entry.expectedSemanticPhrases ?? []).joined(separator: "\u{1E}"))")
        parts.append("expected_hyde=\((entry.expectedHydeAnchors ?? []).joined(separator: "\u{1E}"))")
        parts.append("expected_facets=\((entry.expectedFacets ?? []).joined(separator: "\u{1E}"))")
        parts.append("expected_entities=\((entry.expectedEntities ?? []).joined(separator: "\u{1E}"))")
        parts.append("expected_topics=\((entry.expectedTopics ?? []).joined(separator: "\u{1E}"))")
        parts.append("relevant_document_ids=\((entry.relevantDocumentIds ?? []).joined(separator: "\u{1E}"))")
        parts.append("candidate_document_ids=\((entry.candidateDocumentIds ?? []).joined(separator: "\u{1E}"))")
    }
    return parts.joined(separator: "\n")
}

private func queryExpansionCaseIndexCacheSeed(
    profile: EvalProfile,
    entry: QueryExpansionCase,
    documents: [RecallDocumentCase]
) -> String {
    let parts: [String] = [
        "suite=query_expansion_case",
        "profile=\(profile.rawValue)",
        "id=\(entry.id)",
        "query=\(entry.query)",
        "expected_lexical=\((entry.expectedLexicalTerms ?? []).joined(separator: "\u{1E}"))",
        "expected_semantic=\((entry.expectedSemanticPhrases ?? []).joined(separator: "\u{1E}"))",
        "expected_hyde=\((entry.expectedHydeAnchors ?? []).joined(separator: "\u{1E}"))",
        "expected_facets=\((entry.expectedFacets ?? []).joined(separator: "\u{1E}"))",
        "expected_entities=\((entry.expectedEntities ?? []).joined(separator: "\u{1E}"))",
        "expected_topics=\((entry.expectedTopics ?? []).joined(separator: "\u{1E}"))",
        "relevant_document_ids=\((entry.relevantDocumentIds ?? []).joined(separator: "\u{1E}"))",
        "candidate_document_ids=\((entry.candidateDocumentIds ?? []).joined(separator: "\u{1E}"))",
        recallIndexCacheSeed(profile: profile, documents: documents),
    ]
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
    if usesCanonicalMemorySchemaStorageEval(dataset) {
        return try await runCanonicalStorageSuite(
            profile: profile,
            dataset: dataset,
            datasetRoot: datasetRoot,
            root: root,
            verbose: verbose,
            responseCache: responseCache
        )
    }

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
    let indexingStageCollector = IndexingStageTimingCollector()
    if canReuseIndex {
        print("[storage] Using cached index for \(dataset.count) cases.")
    } else {
        print("[storage] Building index for \(dataset.count) cases...")
        let indexStart = Date()
        try await index.rebuildIndex(
            from: IndexingRequest(roots: [docsRoot]),
            events: { event in
                indexingStageCollector.record(event)
            }
        )
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

    let database = try SQLiteDatabase(path: dbURL.path)

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

        let expectedType = try parseLegacyDocumentTypeLabel(
            entry.expectedMemoryType ?? "",
            context: "storage case \(entry.id)"
        )
        let dbRows = try database.fetchAll(
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

        guard !dbRows.isEmpty else {
            throw EvalError.invalidDataset("No indexed chunks found for storage case '\(entry.id)' (\(documentURL.path)).")
        }

        let predictedRaw: String = dbRows[0]["memory_type"]
        let predictedSourceRaw: String = dbRows[0]["memory_type_source"]
        let predictedConfidence: Double? = dbRows[0]["memory_type_confidence"]
        let predictedType = legacyDocumentTypeLabels.contains(predictedRaw) ? predictedRaw : "factual"
        let predictedSource = predictedSourceRaw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        if predictedType == expectedType {
            correct += 1
        }
        if predictedSource == "fallback" {
            fallbackCount += 1
        }

        confusion[expectedType, default: [:]][predictedType, default: 0] += 1

        let chunkContents = dbRows.map { (row: SQLiteRow) -> String in row["content"] }
        let normalizedContents = chunkContents.map(normalizeForMatch)
        let spans = (entry.requiredSpans ?? []).filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let missing = spans.filter { span in
            let normalizedSpan = normalizeForMatch(span)
            return !normalizedContents.contains(where: { $0.contains(normalizedSpan) })
        }

        spanFound += (spans.count - missing.count)
        spanTotal += spans.count

        let caseResult = StorageCaseResult(
            id: entry.id,
            expectedType: expectedType,
            predictedType: predictedType,
            predictedSource: predictedSource,
            predictedConfidence: predictedConfidence,
            missingSpans: missing,
            chunkCount: chunkContents.count
        )
        results.append(caseResult)

        if verbose {
            print("[storage] \(entry.id): expected=\(expectedType) predicted=\(predictedType) source=\(predictedSource)")
        }
        progress.advance(detail: verbose ? entry.id : nil)
    }

    let macroF1 = computeMacroF1(
        expected: results.map(\.expectedType),
        predicted: results.map(\.predictedType),
        labels: legacyDocumentTypeLabels
    )

    let accuracy = dataset.isEmpty ? 0 : Double(correct) / Double(dataset.count)
    let spanCoverage = spanTotal == 0 ? 1 : Double(spanFound) / Double(spanTotal)
    let fallbackRate = dataset.isEmpty ? 0 : Double(fallbackCount) / Double(dataset.count)

    return StorageSuiteReport(
        mode: "document_memory_type",
        totalCases: dataset.count,
        typeAccuracy: accuracy,
        macroF1: macroF1,
        spanCoverage: spanCoverage,
        fallbackRate: fallbackRate,
        facetPrecision: nil,
        facetRecall: nil,
        facetMicroF1: nil,
        entityPrecision: nil,
        entityRecall: nil,
        topicRecall: nil,
        updateBehaviorAccuracy: nil,
        confusionMatrix: confusion,
        caseResults: results.sorted { $0.id < $1.id },
        stageLatencyStats: indexingStageCollector.report()
    )
}

private func runQueryExpansionSuite(
    profile: EvalProfile,
    cases: [QueryExpansionCase],
    documents: [RecallDocumentCase],
    kValues: [Int],
    datasetRoot: URL,
    root: URL,
    indexCacheEnabled: Bool,
    verbose: Bool,
    responseCache: EvalResponseCache?
) async throws -> QueryExpansionSuiteReport {
    let maxK = kValues.max() ?? 10
    let documentsByID = Dictionary(uniqueKeysWithValues: documents.map { ($0.id, $0) })
    let workspace = try prepareIndexWorkspace(
        suite: .queryExpansion,
        profile: profile,
        datasetRoot: datasetRoot,
        runRoot: root,
        cacheEnabled: indexCacheEnabled && !documents.isEmpty,
        seed: queryExpansionIndexCacheSeed(profile: profile, documents: documents, cases: cases)
    )

    var pathByDocumentID: [String: String] = [:]
    var documentIDByPath: [String: String] = [:]
    for document in documents {
        let path = materializedRecallDocumentURL(document, docsRoot: workspace.docsRoot)
        pathByDocumentID[document.id] = path.path
        documentIDByPath[path.path] = document.id
    }

    let expectedPaths = pathByDocumentID.values.map(URL.init(fileURLWithPath:))
    let canReuseIndex = !documents.isEmpty && indexCacheCanReuse(workspace: workspace, expectedDocumentPaths: expectedPaths)
    if !documents.isEmpty {
        if canReuseIndex {
            print("[query-expansion][index-cache] hit: \(workspace.root.path)")
        } else {
            if workspace.cacheEnabled {
                print("[query-expansion][index-cache] miss: \(workspace.root.path)")
            }
            try resetWorkspaceForRebuild(workspace)
            for document in documents {
                guard let pathRaw = pathByDocumentID[document.id] else {
                    throw EvalError.invalidDataset("Query-expansion document '\(document.id)' did not materialize to a document path.")
                }
                let path = URL(fileURLWithPath: pathRaw)
                try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
                try materializeRecallDocument(document).write(to: path, atomically: true, encoding: .utf8)
            }
        }
    }

    var config = try buildConfiguration(profile: profile, suite: .queryExpansion, databaseURL: workspace.databaseURL)
    try await requireFunctionalContentTaggingIfNeeded(profile: profile, configuration: config, suite: .queryExpansion)
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
    let indexingStageCollector = IndexingStageTimingCollector()
    if !documents.isEmpty {
        if canReuseIndex {
            print("[query-expansion] Using cached index for \(documents.count) documents.")
        } else {
            print("[query-expansion] Building index for \(documents.count) documents...")
            let start = Date()
            try await index.rebuildIndex(
                from: IndexingRequest(roots: [workspace.docsRoot]),
                events: { event in
                    indexingStageCollector.record(event)
                }
            )
            print("[query-expansion] Index built in \(formatDuration(Date().timeIntervalSince(start))).")
            try markIndexCacheReady(workspace)
        }
    } else {
        print("[query-expansion] No recall documents provided; retrieval impact checks will be skipped.")
    }

    if let contentTaggingDiagnostics {
        print(await contentTaggingDiagnostics.summaryLine(suite: .queryExpansion))
        for detail in await contentTaggingDiagnostics.detailLines(suite: .queryExpansion) {
            print(detail)
        }
    }
    if !documents.isEmpty {
        try requireGeneratedContentTagsIfNeeded(profile: profile, databaseURL: workspace.databaseURL, suite: .queryExpansion)
    }

    let allFeatures = recallFeatures(for: config)
    var baselineFeatures = allFeatures
    baselineFeatures.remove(.expansion)
    let collectBranchDiagnostics = cases.contains { entry in
        if let tags = entry.failureTaxonomy, tags.contains(where: { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
            return true
        }
        return entry.sourceDataset != nil || entry.sourceQueryId != nil
    }

    var results: [QueryExpansionCaseResult] = []
    var expandedSearchObservations: [RecallQueryResult] = []

    var totalLexicalRecall = 0.0
    var totalSemanticRecall = 0.0
    var totalHydeRecall = 0.0

    var totalExpectedFacets = 0
    var totalPredictedFacets = 0
    var totalMatchedFacets = 0

    var totalExpectedEntities = 0
    var totalPredictedEntities = 0
    var totalMatchedEntities = 0

    var totalExpectedTopics = 0
    var totalMatchedTopics = 0

    var retrievalCaseCount = 0
    var baselineHitCount = 0
    var expandedHitCount = 0
    var baselineReciprocalRankTotal = 0.0
    var expandedReciprocalRankTotal = 0.0
    var failureTaxonomyCounts: [String: Int] = [:]

    var progress = DeterminateProgress(label: "query-expansion", total: cases.count)
    for entry in cases {
        let queryText = entry.query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !queryText.isEmpty else {
            throw EvalError.invalidDataset("Query-expansion case '\(entry.id)' has an empty query.")
        }

        let analysis = config.queryAnalyzer?.analyze(query: queryText) ?? QueryAnalysis()
        let expansion = try await config.structuredQueryExpander?.expand(
            query: SearchQuery(text: queryText, limit: maxK, rerankLimit: 0, expansionLimit: 5),
            analysis: analysis,
            limit: 5
        ) ?? StructuredQueryExpansion()

        let lexicalRecall = coverageRecall(expected: entry.expectedLexicalTerms ?? [], texts: expansion.lexicalQueries)
        let semanticRecall = coverageRecall(expected: entry.expectedSemanticPhrases ?? [], texts: expansion.semanticQueries)
        let hydeRecall = coverageRecall(expected: entry.expectedHydeAnchors ?? [], texts: expansion.hypotheticalDocuments)

        totalLexicalRecall += lexicalRecall
        totalSemanticRecall += semanticRecall
        totalHydeRecall += hydeRecall

        let expectedFacets = try parseFacetTags(entry.expectedFacets ?? [], context: "query-expansion case \(entry.id)")
        let predictedFacets = Set(expansion.facetHints.map(\.tag))
        let matchedFacets = expectedFacets.intersection(predictedFacets)
        totalExpectedFacets += expectedFacets.count
        totalPredictedFacets += predictedFacets.count
        totalMatchedFacets += matchedFacets.count

        let expectedEntities = Set((entry.expectedEntities ?? []).map(normalizeForMatch).filter { !$0.isEmpty })
        let predictedEntities = Set(expansion.entities.map(\.normalizedValue).map(normalizeForMatch).filter { !$0.isEmpty })
        let matchedEntities = expectedEntities.intersection(predictedEntities)
        totalExpectedEntities += expectedEntities.count
        totalPredictedEntities += predictedEntities.count
        totalMatchedEntities += matchedEntities.count

        let expectedTopics = Set((entry.expectedTopics ?? []).map(normalizeForMatch).filter { !$0.isEmpty })
        let predictedTopics = Set(expansion.topics.map(normalizeForMatch).filter { !$0.isEmpty })
        let matchedTopics = expectedTopics.intersection(predictedTopics)
        totalExpectedTopics += expectedTopics.count
        totalMatchedTopics += matchedTopics.count

        let facetPrecision = safeRatio(matchedFacets.count, predictedFacets.count, emptyDefault: 1)
        let facetRecall = safeRatio(matchedFacets.count, expectedFacets.count, emptyDefault: 1)
        let entityPrecision = safeRatio(matchedEntities.count, predictedEntities.count, emptyDefault: 1)
        let entityRecall = safeRatio(matchedEntities.count, expectedEntities.count, emptyDefault: 1)
        let topicRecall = safeRatio(matchedTopics.count, expectedTopics.count, emptyDefault: 1)

        var baselineDocumentIDs: [String]? = nil
        var expandedDocumentIDs: [String]? = nil
        var baselineHit: Bool? = nil
        var expandedHit: Bool? = nil
        var baselineBestRank: Int? = nil
        var expandedBestRank: Int? = nil
        var baselineReciprocalRank: Double? = nil
        var expandedReciprocalRank: Double? = nil
        var branchDiagnostics: QueryExpansionBranchDiagnostics? = nil
        var missDiagnostics: QueryExpansionMissDiagnostics? = nil

        if !documents.isEmpty, let relevantIDs = entry.relevantDocumentIds, !relevantIDs.isEmpty {
            let relevantSet = Set(relevantIDs)
            let candidateDocuments: [RecallDocumentCase]
            if let candidateIDs = entry.candidateDocumentIds, !candidateIDs.isEmpty {
                let unknownCandidates = candidateIDs.filter { documentsByID[$0] == nil }
                guard unknownCandidates.isEmpty else {
                    throw EvalError.invalidDataset(
                        "Query-expansion case '\(entry.id)' references unknown candidate_document_ids: \(unknownCandidates.sorted().joined(separator: ", "))."
                    )
                }
                candidateDocuments = candidateIDs.compactMap { documentsByID[$0] }
            } else {
                candidateDocuments = documents
            }

            let candidatePathByDocumentID: [String: String]
            let candidateDocumentIDByPath: [String: String]
            let candidateIndex: MemoryIndex
            if candidateDocuments.count == documents.count {
                candidatePathByDocumentID = pathByDocumentID
                candidateDocumentIDByPath = documentIDByPath
                candidateIndex = index
            } else {
                let candidateWorkspace = try prepareIndexWorkspace(
                    suite: .queryExpansion,
                    profile: profile,
                    datasetRoot: datasetRoot,
                    runRoot: root.appendingPathComponent("query_expansion_case_slices", isDirectory: true),
                    cacheEnabled: indexCacheEnabled && !candidateDocuments.isEmpty,
                    seed: queryExpansionCaseIndexCacheSeed(profile: profile, entry: entry, documents: candidateDocuments)
                )

                var localPathByDocumentID: [String: String] = [:]
                var localDocumentIDByPath: [String: String] = [:]
                for document in candidateDocuments {
                    let path = materializedRecallDocumentURL(document, docsRoot: candidateWorkspace.docsRoot)
                    localPathByDocumentID[document.id] = path.path
                    localDocumentIDByPath[path.path] = document.id
                }

                let expectedPaths = localPathByDocumentID.values.map(URL.init(fileURLWithPath:))
                let canReuseCandidateIndex = !candidateDocuments.isEmpty && indexCacheCanReuse(
                    workspace: candidateWorkspace,
                    expectedDocumentPaths: expectedPaths
                )
                if !canReuseCandidateIndex {
                    try resetWorkspaceForRebuild(candidateWorkspace)
                    for document in candidateDocuments {
                        guard let pathRaw = localPathByDocumentID[document.id] else {
                            throw EvalError.invalidDataset("Query-expansion document '\(document.id)' did not materialize to a candidate slice path.")
                        }
                        let path = URL(fileURLWithPath: pathRaw)
                        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
                        try materializeRecallDocument(document).write(to: path, atomically: true, encoding: .utf8)
                    }
                }

                var caseConfig = config
                caseConfig.databaseURL = candidateWorkspace.databaseURL
                let localIndex = try MemoryIndex(configuration: caseConfig)
                if !canReuseCandidateIndex {
                    try await localIndex.rebuildIndex(
                        from: IndexingRequest(roots: [candidateWorkspace.docsRoot]),
                        events: { _ in }
                    )
                    try markIndexCacheReady(candidateWorkspace)
                }

                candidatePathByDocumentID = localPathByDocumentID
                candidateDocumentIDByPath = localDocumentIDByPath
                candidateIndex = localIndex
            }

            let unknownRelevant = relevantSet.filter { candidatePathByDocumentID[$0] == nil }
            guard unknownRelevant.isEmpty else {
                throw EvalError.invalidDataset(
                    "Query-expansion case '\(entry.id)' references unknown relevant_document_ids: \(unknownRelevant.sorted().joined(separator: ", "))."
                )
            }

            let baselineCollector = SearchStageTimingCollector()
            let baselineRefs = try await candidateIndex.memorySearch(
                query: queryText,
                limit: dedupedDocumentLimitForProfile(profile: profile, maxK: maxK),
                features: baselineFeatures,
                dedupeDocuments: true,
                includeLineRanges: false,
                events: { event in
                    baselineCollector.record(event)
                }
            )
            baselineDocumentIDs = queryExpansionDocumentIDs(
                from: baselineRefs,
                documentIDByPath: candidateDocumentIDByPath
            )
            baselineHit = baselineDocumentIDs?.prefix(maxK).contains(where: relevantSet.contains) ?? false
            baselineBestRank = firstRelevantRank(in: baselineDocumentIDs ?? [], relevant: relevantSet, maxK: maxK)
            baselineReciprocalRank = reciprocalRank(for: baselineBestRank)

            let expandedCollector = SearchStageTimingCollector()
            let queryStartTime = Date()
            let expandedRefs = try await candidateIndex.memorySearch(
                query: queryText,
                limit: dedupedDocumentLimitForProfile(profile: profile, maxK: maxK),
                features: allFeatures,
                dedupeDocuments: true,
                includeLineRanges: false,
                events: { event in
                    expandedCollector.record(event)
                }
            )
            let queryLatencyMs = Date().timeIntervalSince(queryStartTime) * 1000.0
            expandedDocumentIDs = queryExpansionDocumentIDs(
                from: expandedRefs,
                documentIDByPath: candidateDocumentIDByPath
            )
            expandedHit = expandedDocumentIDs?.prefix(maxK).contains(where: relevantSet.contains) ?? false
            expandedBestRank = firstRelevantRank(in: expandedDocumentIDs ?? [], relevant: relevantSet, maxK: maxK)
            expandedReciprocalRank = reciprocalRank(for: expandedBestRank)

            if collectBranchDiagnostics {
                branchDiagnostics = try await queryExpansionBranchDiagnostics(
                    query: queryText,
                    expansion: expansion,
                    index: candidateIndex,
                    documentIDByPath: candidateDocumentIDByPath,
                    relevantDocumentIds: relevantSet,
                    baselineRank: baselineBestRank,
                    expandedRank: expandedBestRank,
                    reciprocalRankDelta: reciprocalRankDelta(
                        baseline: baselineReciprocalRank,
                        expanded: expandedReciprocalRank
                    ),
                    profile: profile,
                    maxK: maxK
                )
            }
            if collectBranchDiagnostics, expandedHit != true {
                missDiagnostics = try await queryExpansionMissDiagnostics(
                    query: queryText,
                    expansion: expansion,
                    index: candidateIndex,
                    documentIDByPath: candidateDocumentIDByPath,
                    relevantDocumentIds: relevantSet,
                    expandedReferences: expandedRefs,
                    profile: profile,
                    maxK: maxK
                )
            }

            expandedSearchObservations.append(
                RecallQueryResult(
                    id: entry.id,
                    query: queryText,
                    relevantDocumentIds: Array(relevantSet).sorted(),
                    retrievedDocumentIds: expandedDocumentIDs.map { Array($0.prefix(maxK)) } ?? [],
                    hitByK: [maxK: expandedHit == true],
                    recallByK: [:],
                    mrrByK: [:],
                    ndcgByK: [:],
                    latencyMs: queryLatencyMs,
                    stageTimings: expandedCollector.queryTimings(),
                    candidateCounts: expandedCollector.queryCounts(),
                    difficulty: nil
                )
            )

            retrievalCaseCount += 1
            if baselineHit == true {
                baselineHitCount += 1
            }
            if expandedHit == true {
                expandedHitCount += 1
            }
            baselineReciprocalRankTotal += baselineReciprocalRank ?? 0
            expandedReciprocalRankTotal += expandedReciprocalRank ?? 0
        }

        for tag in entry.failureTaxonomy ?? [] {
            let normalized = tag.trimmingCharacters(in: .whitespacesAndNewlines)
            if !normalized.isEmpty {
                failureTaxonomyCounts[normalized, default: 0] += 1
            }
        }

        results.append(
            QueryExpansionCaseResult(
                id: entry.id,
                query: queryText,
                sourceDataset: entry.sourceDataset,
                sourceQueryId: entry.sourceQueryId,
                sourceProfile: entry.sourceProfile,
                sourceRankAtK: entry.sourceRankAtK,
                failureTaxonomy: entry.failureTaxonomy,
                rescueReason: entry.rescueReason,
                lexicalQueries: expansion.lexicalQueries,
                semanticQueries: expansion.semanticQueries,
                hypotheticalDocuments: expansion.hypotheticalDocuments,
                facetHints: expansion.facetHints.map(\.tag.rawValue).sorted(),
                entities: expansion.entities.map(\.normalizedValue).sorted(),
                topics: expansion.topics.sorted(),
                lexicalCoverageRecall: lexicalRecall,
                semanticCoverageRecall: semanticRecall,
                hydeAnchorRecall: hydeRecall,
                facetPrecision: facetPrecision,
                facetRecall: facetRecall,
                entityPrecision: entityPrecision,
                entityRecall: entityRecall,
                topicRecall: topicRecall,
                baselineRetrievedDocumentIds: baselineDocumentIDs.map { Array($0.prefix(maxK)) },
                expandedRetrievedDocumentIds: expandedDocumentIDs.map { Array($0.prefix(maxK)) },
                baselineHitAtK: baselineHit,
                expandedHitAtK: expandedHit,
                baselineBestRankAtK: baselineBestRank,
                expandedBestRankAtK: expandedBestRank,
                baselineReciprocalRankAtK: baselineReciprocalRank,
                expandedReciprocalRankAtK: expandedReciprocalRank,
                reciprocalRankDeltaAtK: reciprocalRankDelta(
                    baseline: baselineReciprocalRank,
                    expanded: expandedReciprocalRank
                ),
                rankImprovementAtK: rankImprovement(
                    baselineRank: baselineBestRank,
                    expandedRank: expandedBestRank
                ),
                branchDiagnostics: branchDiagnostics,
                missDiagnostics: missDiagnostics
            )
        )

        if verbose {
            print("[query-expansion] \(entry.id): lexicalRecall=\(percent(lexicalRecall)) semanticRecall=\(percent(semanticRecall)) hydeRecall=\(percent(hydeRecall))")
        }
        progress.advance(detail: verbose ? entry.id : nil)
    }

    if let recallDiagnostics {
        for line in await recallDiagnostics.summaryLines(suite: .queryExpansion) {
            print(line)
        }
        for line in await recallDiagnostics.detailLines(suite: .queryExpansion) {
            print(line)
        }
    }
    if let responseCache {
        for line in await responseCache.drainSummaryLines(suite: .queryExpansion) {
            print(line)
        }
    }

    let facetPrecision = safeRatio(totalMatchedFacets, totalPredictedFacets, emptyDefault: 1)
    let facetRecall = safeRatio(totalMatchedFacets, totalExpectedFacets, emptyDefault: 1)
    let entityPrecision = safeRatio(totalMatchedEntities, totalPredictedEntities, emptyDefault: 1)
    let entityRecall = safeRatio(totalMatchedEntities, totalExpectedEntities, emptyDefault: 1)
    let topicRecall = safeRatio(totalMatchedTopics, totalExpectedTopics, emptyDefault: 1)
    let retrievalBaselineHitRate = retrievalCaseCount == 0 ? nil : Double(baselineHitCount) / Double(retrievalCaseCount)
    let retrievalExpandedHitRate = retrievalCaseCount == 0 ? nil : Double(expandedHitCount) / Double(retrievalCaseCount)
    let retrievalLift: Double? = {
        guard let baseline = retrievalBaselineHitRate, let expanded = retrievalExpandedHitRate else { return nil }
        return expanded - baseline
    }()
    let retrievalBaselineMRR = retrievalCaseCount == 0 ? nil : baselineReciprocalRankTotal / Double(retrievalCaseCount)
    let retrievalExpandedMRR = retrievalCaseCount == 0 ? nil : expandedReciprocalRankTotal / Double(retrievalCaseCount)
    let retrievalMRRDelta: Double? = {
        guard let baseline = retrievalBaselineMRR, let expanded = retrievalExpandedMRR else { return nil }
        return expanded - baseline
    }()
    let sortedResults = results.sorted { $0.id < $1.id }

    return QueryExpansionSuiteReport(
        totalQueries: cases.count,
        lexicalCoverageRecall: cases.isEmpty ? 0 : totalLexicalRecall / Double(cases.count),
        semanticCoverageRecall: cases.isEmpty ? 0 : totalSemanticRecall / Double(cases.count),
        hydeAnchorRecall: cases.isEmpty ? 0 : totalHydeRecall / Double(cases.count),
        facetPrecision: facetPrecision,
        facetRecall: facetRecall,
        facetMicroF1: harmonicMean(precision: facetPrecision, recall: facetRecall),
        entityPrecision: entityPrecision,
        entityRecall: entityRecall,
        topicRecall: topicRecall,
        retrievalBaselineHitRate: retrievalBaselineHitRate,
        retrievalExpandedHitRate: retrievalExpandedHitRate,
        retrievalLift: retrievalLift,
        retrievalBaselineMRR: retrievalBaselineMRR,
        retrievalExpandedMRR: retrievalExpandedMRR,
        retrievalMRRDelta: retrievalMRRDelta,
        failureTaxonomyCounts: failureTaxonomyCounts.isEmpty ? nil : failureTaxonomyCounts,
        taxonomyMetrics: queryExpansionTaxonomyMetrics(from: sortedResults),
        branchClassificationCounts: queryExpansionBranchClassificationCounts(from: sortedResults),
        latencyStats: computeLatencyStats(queryResults: expandedSearchObservations),
        stageLatencyStats: computeRecallStageLatencyStats(queryResults: expandedSearchObservations),
        candidateCountStats: computeRecallCandidateCountStats(queryResults: expandedSearchObservations),
        caseResults: sortedResults
    )
}

private func queryExpansionDocumentIDs(
    from references: [MemorySearchReference],
    documentIDByPath: [String: String]
) -> [String] {
    references.compactMap { documentIDByPath[$0.documentPath] }
}

private func queryExpansionMissDiagnostics(
    query: String,
    expansion: StructuredQueryExpansion,
    index: MemoryIndex,
    documentIDByPath: [String: String],
    relevantDocumentIds: Set<String>,
    expandedReferences: [MemorySearchReference],
    profile: EvalProfile,
    maxK: Int
) async throws -> QueryExpansionMissDiagnostics {
    let diagnosticLimit = dedupedDocumentLimitForProfile(profile: profile, maxK: maxK)
    let finalCandidates = queryExpansionRankedCandidates(
        from: expandedReferences,
        documentIDByPath: documentIDByPath,
        relevantDocumentIds: relevantDocumentIds
    )
    let finalBestRelevant = finalCandidates.first(where: \.isRelevant)

    var branchRanks: [QueryExpansionBranchRankDiagnostic] = []
    let coreBranches: [(String, String, RecallFeatures)] = [
        ("baseline_semantic", query, RecallFeatures.semantic),
        ("baseline_lexical", query, RecallFeatures.lexical),
        ("expanded_semantic", query, [.semantic, .expansion]),
        ("expanded_lexical", query, [.lexical, .expansion]),
    ]
    for branch in coreBranches {
        if let diagnostic = try await queryExpansionBranchRankDiagnostic(
            branch: branch.0,
            query: branch.1,
            features: branch.2,
            index: index,
            documentIDByPath: documentIDByPath,
            relevantDocumentIds: relevantDocumentIds,
            limit: diagnosticLimit
        ) {
            branchRanks.append(diagnostic)
        }
    }

    for (indexOffset, lexicalQuery) in uniqueQueryExpansionSources(expansion.lexicalQueries).enumerated() {
        if let diagnostic = try await queryExpansionBranchRankDiagnostic(
            branch: "source_lexical_\(indexOffset + 1)",
            query: lexicalQuery,
            features: [.lexical],
            index: index,
            documentIDByPath: documentIDByPath,
            relevantDocumentIds: relevantDocumentIds,
            limit: diagnosticLimit
        ) {
            branchRanks.append(diagnostic)
        }
    }
    for (indexOffset, semanticQuery) in uniqueQueryExpansionSources(expansion.semanticQueries).enumerated() {
        if let diagnostic = try await queryExpansionBranchRankDiagnostic(
            branch: "source_semantic_\(indexOffset + 1)",
            query: semanticQuery,
            features: [.semantic],
            index: index,
            documentIDByPath: documentIDByPath,
            relevantDocumentIds: relevantDocumentIds,
            limit: diagnosticLimit
        ) {
            branchRanks.append(diagnostic)
        }
    }
    for (indexOffset, hypotheticalDocument) in uniqueQueryExpansionSources(expansion.hypotheticalDocuments).enumerated() {
        if let diagnostic = try await queryExpansionBranchRankDiagnostic(
            branch: "source_hypothetical_document_\(indexOffset + 1)",
            query: hypotheticalDocument,
            features: [.semantic],
            index: index,
            documentIDByPath: documentIDByPath,
            relevantDocumentIds: relevantDocumentIds,
            limit: diagnosticLimit
        ) {
            branchRanks.append(diagnostic)
        }
    }

    return QueryExpansionMissDiagnostics(
        diagnosticLimit: diagnosticLimit,
        finalBestRelevantRank: finalBestRelevant?.rank,
        finalBestRelevantDocumentId: finalBestRelevant?.documentId,
        finalBestRelevantScore: finalBestRelevant?.score,
        finalTopCandidates: Array(finalCandidates.prefix(5)),
        branchRanks: branchRanks
    )
}

private func queryExpansionBranchRankDiagnostic(
    branch: String,
    query: String,
    features: RecallFeatures,
    index: MemoryIndex,
    documentIDByPath: [String: String],
    relevantDocumentIds: Set<String>,
    limit: Int
) async throws -> QueryExpansionBranchRankDiagnostic? {
    let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    let references = try await index.memorySearch(
        query: trimmed,
        limit: limit,
        features: features,
        dedupeDocuments: true,
        includeLineRanges: false
    )
    let candidates = queryExpansionRankedCandidates(
        from: references,
        documentIDByPath: documentIDByPath,
        relevantDocumentIds: relevantDocumentIds
    )
    let bestRelevant = candidates.first(where: \.isRelevant)

    return QueryExpansionBranchRankDiagnostic(
        branch: branch,
        query: trimmed,
        features: queryExpansionFeatureLabels(features),
        bestRelevantRank: bestRelevant?.rank,
        bestRelevantDocumentId: bestRelevant?.documentId,
        bestRelevantScore: bestRelevant?.score,
        topCandidates: Array(candidates.prefix(5))
    )
}

private func queryExpansionRankedCandidates(
    from references: [MemorySearchReference],
    documentIDByPath: [String: String],
    relevantDocumentIds: Set<String>
) -> [QueryExpansionRankedCandidate] {
    references.enumerated().compactMap { index, reference in
        guard let documentId = documentIDByPath[reference.documentPath] else { return nil }
        return QueryExpansionRankedCandidate(
            rank: index + 1,
            documentId: documentId,
            isRelevant: relevantDocumentIds.contains(documentId),
            score: reference.score
        )
    }
}

private func queryExpansionFeatureLabels(_ features: RecallFeatures) -> [String] {
    var labels: [String] = []
    if features.contains(.semantic) { labels.append("semantic") }
    if features.contains(.lexical) { labels.append("lexical") }
    if features.contains(.tags) { labels.append("tags") }
    if features.contains(.expansion) { labels.append("expansion") }
    if features.contains(.rerank) { labels.append("rerank") }
    if features.contains(.planner) { labels.append("planner") }
    return labels
}

private func queryExpansionBranchDiagnostics(
    query: String,
    expansion: StructuredQueryExpansion,
    index: MemoryIndex,
    documentIDByPath: [String: String],
    relevantDocumentIds: Set<String>,
    baselineRank: Int?,
    expandedRank: Int?,
    reciprocalRankDelta: Double?,
    profile: EvalProfile,
    maxK: Int
) async throws -> QueryExpansionBranchDiagnostics {
    let baselineSemanticRank = try await queryExpansionBranchRank(
        query: query,
        index: index,
        documentIDByPath: documentIDByPath,
        relevantDocumentIds: relevantDocumentIds,
        profile: profile,
        maxK: maxK,
        features: [.semantic]
    )
    let baselineLexicalRank = try await queryExpansionBranchRank(
        query: query,
        index: index,
        documentIDByPath: documentIDByPath,
        relevantDocumentIds: relevantDocumentIds,
        profile: profile,
        maxK: maxK,
        features: [.lexical]
    )
    let expandedSemanticRank = try await queryExpansionBranchRank(
        query: query,
        index: index,
        documentIDByPath: documentIDByPath,
        relevantDocumentIds: relevantDocumentIds,
        profile: profile,
        maxK: maxK,
        features: [.semantic, .expansion]
    )
    let expandedLexicalRank = try await queryExpansionBranchRank(
        query: query,
        index: index,
        documentIDByPath: documentIDByPath,
        relevantDocumentIds: relevantDocumentIds,
        profile: profile,
        maxK: maxK,
        features: [.lexical, .expansion]
    )

    var sourceHits: [QueryExpansionSourceHit] = []
    for lexicalQuery in uniqueQueryExpansionSources(expansion.lexicalQueries) {
        let rank = try await queryExpansionBranchRank(
            query: lexicalQuery,
            index: index,
            documentIDByPath: documentIDByPath,
            relevantDocumentIds: relevantDocumentIds,
            profile: profile,
            maxK: maxK,
            features: [.lexical]
        )
        sourceHits.append(QueryExpansionSourceHit(kind: "lexical", query: lexicalQuery, rankAtK: rank))
    }
    for semanticQuery in uniqueQueryExpansionSources(expansion.semanticQueries) {
        let rank = try await queryExpansionBranchRank(
            query: semanticQuery,
            index: index,
            documentIDByPath: documentIDByPath,
            relevantDocumentIds: relevantDocumentIds,
            profile: profile,
            maxK: maxK,
            features: [.semantic]
        )
        sourceHits.append(QueryExpansionSourceHit(kind: "semantic", query: semanticQuery, rankAtK: rank))
    }
    for hypotheticalDocument in uniqueQueryExpansionSources(expansion.hypotheticalDocuments) {
        let rank = try await queryExpansionBranchRank(
            query: hypotheticalDocument,
            index: index,
            documentIDByPath: documentIDByPath,
            relevantDocumentIds: relevantDocumentIds,
            profile: profile,
            maxK: maxK,
            features: [.semantic]
        )
        sourceHits.append(QueryExpansionSourceHit(kind: "hypothetical_document", query: hypotheticalDocument, rankAtK: rank))
    }

    let branchRanks = [
        baselineSemanticRank,
        baselineLexicalRank,
        expandedSemanticRank,
        expandedLexicalRank
    ]
    let classification = classifyQueryExpansionBranch(
        baselineRank: baselineRank,
        expandedRank: expandedRank,
        reciprocalRankDelta: reciprocalRankDelta,
        branchRanks: branchRanks,
        sourceHits: sourceHits
    )

    return QueryExpansionBranchDiagnostics(
        classification: classification,
        baselineSemanticRankAtK: baselineSemanticRank,
        baselineLexicalRankAtK: baselineLexicalRank,
        expandedSemanticRankAtK: expandedSemanticRank,
        expandedLexicalRankAtK: expandedLexicalRank,
        expansionSourceHits: sourceHits
    )
}

private func queryExpansionBranchRank(
    query: String,
    index: MemoryIndex,
    documentIDByPath: [String: String],
    relevantDocumentIds: Set<String>,
    profile: EvalProfile,
    maxK: Int,
    features: RecallFeatures
) async throws -> Int? {
    let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    let references = try await index.memorySearch(
        query: trimmed,
        limit: dedupedDocumentLimitForProfile(profile: profile, maxK: maxK),
        features: features,
        dedupeDocuments: true,
        includeLineRanges: false
    )
    let documentIDs = queryExpansionDocumentIDs(from: references, documentIDByPath: documentIDByPath)
    return firstRelevantRank(in: documentIDs, relevant: relevantDocumentIds, maxK: maxK)
}

private func uniqueQueryExpansionSources(_ queries: [String]) -> [String] {
    var seen: Set<String> = []
    var result: [String] = []
    for query in queries {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { continue }
        let key = normalizeForMatch(trimmed)
        guard !key.isEmpty, seen.insert(key).inserted else { continue }
        result.append(trimmed)
    }
    return result
}

private func classifyQueryExpansionBranch(
    baselineRank: Int?,
    expandedRank: Int?,
    reciprocalRankDelta: Double?,
    branchRanks: [Int?],
    sourceHits: [QueryExpansionSourceHit]
) -> String {
    let epsilon = 0.000_000_1
    if let reciprocalRankDelta, reciprocalRankDelta > epsilon {
        return "improved"
    }
    if let reciprocalRankDelta, reciprocalRankDelta < -epsilon {
        return "regression_prone"
    }
    if expandedRank == 1 {
        return "already_good"
    }
    let hasBranchCoverage = branchRanks.contains(where: { $0 != nil })
        || sourceHits.contains(where: { $0.rankAtK != nil })
        || baselineRank != nil
        || expandedRank != nil
    if !hasBranchCoverage {
        return "candidate_coverage_miss"
    }
    return "fusion_ranking_miss"
}

private func queryExpansionBranchClassificationCounts(from results: [QueryExpansionCaseResult]) -> [String: Int]? {
    var counts: [String: Int] = [:]
    for result in results {
        guard let classification = result.branchDiagnostics?.classification else { continue }
        counts[classification, default: 0] += 1
    }
    return counts.isEmpty ? nil : counts
}

private func queryExpansionTaxonomyMetrics(from results: [QueryExpansionCaseResult]) -> [QueryExpansionTaxonomyMetric]? {
    var grouped: [String: [QueryExpansionCaseResult]] = [:]
    for result in results {
        for tag in result.failureTaxonomy ?? [] {
            let normalized = tag.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalized.isEmpty else { continue }
            grouped[normalized, default: []].append(result)
        }
    }
    guard !grouped.isEmpty else { return nil }

    return grouped.map { tag, cases in
        let total = Double(cases.count)
        let baselineHits = cases.filter { $0.baselineHitAtK == true }.count
        let expandedHits = cases.filter { $0.expandedHitAtK == true }.count
        let baselineMRR = cases.reduce(0.0) { $0 + ($1.baselineReciprocalRankAtK ?? 0) } / total
        let expandedMRR = cases.reduce(0.0) { $0 + ($1.expandedReciprocalRankAtK ?? 0) } / total
        let deltas = cases.map { $0.reciprocalRankDeltaAtK ?? 0 }
        return QueryExpansionTaxonomyMetric(
            tag: tag,
            caseCount: cases.count,
            baselineHitRate: Double(baselineHits) / total,
            expandedHitRate: Double(expandedHits) / total,
            baselineMRR: baselineMRR,
            expandedMRR: expandedMRR,
            mrrDelta: expandedMRR - baselineMRR,
            improvedCount: deltas.filter { $0 > 0.000_000_1 }.count,
            worsenedCount: deltas.filter { $0 < -0.000_000_1 }.count,
            flatCount: deltas.filter { abs($0) <= 0.000_000_1 }.count,
            missCount: cases.filter { $0.expandedHitAtK != true }.count
        )
    }
    .sorted { lhs, rhs in
        if lhs.mrrDelta == rhs.mrrDelta {
            if lhs.caseCount == rhs.caseCount {
                return lhs.tag < rhs.tag
            }
            return lhs.caseCount > rhs.caseCount
        }
        return lhs.mrrDelta < rhs.mrrDelta
    }
}

private func coverageRecall(expected: [String], texts: [String]) -> Double {
    let normalizedExpected = expected.map(normalizeForMatch).filter { !$0.isEmpty }
    guard !normalizedExpected.isEmpty else { return 1 }
    let haystack = normalizeForMatch(texts.joined(separator: "\n"))
    guard !haystack.isEmpty else { return 0 }

    let matched = normalizedExpected.reduce(into: 0) { partial, value in
        if haystack.contains(value) {
            partial += 1
        }
    }
    return Double(matched) / Double(normalizedExpected.count)
}

private func usesCanonicalMemorySchemaStorageEval(_ dataset: [StorageCase]) -> Bool {
    dataset.contains { entry in
        entry.expectedKind != nil
            || entry.expectedStatus != nil
            || !(entry.expectedFacets ?? []).isEmpty
            || !(entry.requiredEntities ?? []).isEmpty
            || !(entry.requiredTopics ?? []).isEmpty
            || entry.expectedUpdateBehavior != nil
    }
}

private func runCanonicalStorageSuite(
    profile: EvalProfile,
    dataset: [StorageCase],
    datasetRoot: URL,
    root: URL,
    verbose: Bool,
    responseCache: EvalResponseCache?
) async throws -> StorageSuiteReport {
    let workspace = try prepareIndexWorkspace(
        suite: .storage,
        profile: profile,
        datasetRoot: datasetRoot,
        runRoot: root,
        cacheEnabled: false,
        seed: storageIndexCacheSeed(profile: profile, dataset: dataset)
    )
    try resetWorkspaceForRebuild(workspace)

    var results: [StorageCaseResult] = []
    var confusion: [String: [String: Int]] = [:]
    var expectedKinds: [String] = []
    var predictedKinds: [String] = []
    var correctKinds = 0

    var totalExpectedFacets = 0
    var totalPredictedFacets = 0
    var totalMatchedFacets = 0

    var totalExpectedEntities = 0
    var totalPredictedEntities = 0
    var totalMatchedEntities = 0

    var totalExpectedTopics = 0
    var totalMatchedTopics = 0

    var updateExpectedCount = 0
    var updateCorrectCount = 0

    var progress = DeterminateProgress(label: "storage", total: dataset.count)

    for entry in dataset {
        let caseDatabaseURL = workspace.root
            .appendingPathComponent("cases", isDirectory: true)
            .appendingPathComponent(safeFilename(entry.id), isDirectory: true)
            .appendingPathComponent("index.sqlite")
        try FileManager.default.createDirectory(
            at: caseDatabaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var config = try buildConfiguration(profile: profile, suite: .storage, databaseURL: caseDatabaseURL)
        if let responseCache {
            installProviderResponseCachingIfNeeded(configuration: &config, responseCache: responseCache)
        }
        let index = try MemoryIndex(configuration: config)

        var setupRecords: [MemoryRecord] = []
        for seed in entry.setupMemories ?? [] {
            let kind = try parseMemoryKind(seed.kind, context: "storage setup memory \(entry.id)")
            let status = try parseMemoryStatus(seed.status ?? MemoryStatus.active.rawValue, context: "storage setup memory \(entry.id)")
            let facetTags = try parseFacetTags(seed.facetTags ?? [], context: "storage setup memory \(entry.id)")
            let entities = seed.entityValues?.map {
                MemoryEntity(label: .other, value: $0, normalizedValue: normalizeForMatch($0))
            } ?? []
            let record = try await index.save(
                text: seed.text,
                kind: kind,
                status: status,
                facetTags: facetTags,
                entities: entities,
                topics: seed.topics ?? [],
                canonicalKey: seed.canonicalKey
            )
            setupRecords.append(record)
        }

        let extracted = try await index.extract(from: entry.text, limit: 8)
        let candidate = extracted.first.map { initial in
            var adjusted = initial
            if let canonicalKey = entry.canonicalKey {
                adjusted.canonicalKey = canonicalKey
            }
            return adjusted
        }
        let ingestResult = try await index.ingest(candidate.map { [$0] } ?? [])
        let stored = ingestResult.records.first

        let expectedKind = try entry.expectedKind.map { try parseMemoryKind($0, context: "storage case \(entry.id)") }
        let predictedKind = stored?.kind ?? candidate?.kind
        let expectedStatus = try entry.expectedStatus.map { try parseMemoryStatus($0, context: "storage case \(entry.id)") }
        let predictedStatus = stored?.status ?? candidate?.status

        let expectedFacetTags = try parseFacetTags(entry.expectedFacets ?? [], context: "storage case \(entry.id)")
        let predictedFacetTags = stored?.facetTags ?? candidate?.facetTags ?? []

        let expectedEntityValues = Set((entry.requiredEntities ?? []).map(normalizeForMatch).filter { !$0.isEmpty })
        let predictedEntityValues = Set((stored?.entities ?? candidate?.entities ?? []).map(\.normalizedValue).map(normalizeForMatch))

        let expectedTopicValues = Set((entry.requiredTopics ?? []).map(normalizeForMatch).filter { !$0.isEmpty })
        let predictedTopicValues = Set((stored?.topics ?? candidate?.topics ?? []).map(normalizeForMatch))

        if let expectedKind {
            expectedKinds.append(expectedKind.rawValue)
            predictedKinds.append(predictedKind?.rawValue ?? "none")
            if predictedKind == expectedKind {
                correctKinds += 1
            }
            confusion[expectedKind.rawValue, default: [:]][predictedKind?.rawValue ?? "none", default: 0] += 1
        }

        let matchedFacets = expectedFacetTags.intersection(predictedFacetTags)
        totalExpectedFacets += expectedFacetTags.count
        totalPredictedFacets += predictedFacetTags.count
        totalMatchedFacets += matchedFacets.count

        let matchedEntities = expectedEntityValues.intersection(predictedEntityValues)
        totalExpectedEntities += expectedEntityValues.count
        totalPredictedEntities += predictedEntityValues.count
        totalMatchedEntities += matchedEntities.count

        let matchedTopics = expectedTopicValues.intersection(predictedTopicValues)
        totalExpectedTopics += expectedTopicValues.count
        totalMatchedTopics += matchedTopics.count

        let observedUpdateBehavior: String?
        if entry.expectedUpdateBehavior != nil {
            observedUpdateBehavior = try await observeUpdateBehavior(index: index, setupRecords: setupRecords, result: stored)
        } else {
            observedUpdateBehavior = nil
        }
        if let expectedUpdateBehavior = entry.expectedUpdateBehavior {
            updateExpectedCount += 1
            if observedUpdateBehavior == expectedUpdateBehavior {
                updateCorrectCount += 1
            }
        }

        results.append(
            StorageCaseResult(
                id: entry.id,
                expectedType: expectedKind?.rawValue ?? (entry.expectedMemoryType ?? ""),
                predictedType: predictedKind?.rawValue ?? "none",
                predictedSource: candidate?.source ?? "none",
                predictedConfidence: candidate?.confidence,
                missingSpans: [],
                chunkCount: stored == nil ? 0 : 1,
                expectedKind: expectedKind?.rawValue,
                predictedKind: predictedKind?.rawValue,
                expectedStatus: expectedStatus?.rawValue,
                predictedStatus: predictedStatus?.rawValue,
                expectedFacets: expectedFacetTags.map(\.rawValue).sorted(),
                predictedFacets: predictedFacetTags.map(\.rawValue).sorted(),
                expectedEntities: Array(expectedEntityValues).sorted(),
                predictedEntities: Array(predictedEntityValues).sorted(),
                expectedTopics: Array(expectedTopicValues).sorted(),
                predictedTopics: Array(predictedTopicValues).sorted(),
                expectedUpdateBehavior: entry.expectedUpdateBehavior,
                observedUpdateBehavior: observedUpdateBehavior
            )
        )

        if verbose {
            print("[storage] \(entry.id): expectedKind=\(expectedKind?.rawValue ?? "-") predictedKind=\(predictedKind?.rawValue ?? "none") expectedStatus=\(expectedStatus?.rawValue ?? "-") predictedStatus=\(predictedStatus?.rawValue ?? "none")")
        }
        progress.advance(detail: verbose ? entry.id : nil)
    }

    let kindAccuracy = expectedKinds.isEmpty ? 0 : Double(correctKinds) / Double(expectedKinds.count)
    let kindMacroF1 = computeMacroF1(
        expected: expectedKinds,
        predicted: predictedKinds,
        labels: MemoryKind.allCases.map(\.rawValue) + ["none"]
    )

    let facetPrecision = safeRatio(totalMatchedFacets, totalPredictedFacets, emptyDefault: 1)
    let facetRecall = safeRatio(totalMatchedFacets, totalExpectedFacets, emptyDefault: 1)
    let facetMicroF1 = harmonicMean(precision: facetPrecision, recall: facetRecall)
    let entityPrecision = safeRatio(totalMatchedEntities, totalPredictedEntities, emptyDefault: 1)
    let entityRecall = safeRatio(totalMatchedEntities, totalExpectedEntities, emptyDefault: 1)
    let topicRecall = safeRatio(totalMatchedTopics, totalExpectedTopics, emptyDefault: 1)
    let updateAccuracy = updateExpectedCount == 0 ? 1 : Double(updateCorrectCount) / Double(updateExpectedCount)

    return StorageSuiteReport(
        mode: "canonical_memory_schema",
        totalCases: dataset.count,
        typeAccuracy: kindAccuracy,
        macroF1: kindMacroF1,
        spanCoverage: topicRecall,
        fallbackRate: 0,
        facetPrecision: facetPrecision,
        facetRecall: facetRecall,
        facetMicroF1: facetMicroF1,
        entityPrecision: entityPrecision,
        entityRecall: entityRecall,
        topicRecall: topicRecall,
        updateBehaviorAccuracy: updateAccuracy,
        confusionMatrix: confusion,
        caseResults: results.sorted { $0.id < $1.id },
        stageLatencyStats: nil
    )
}

private func parseMemoryKind(_ raw: String, context: String) throws -> MemoryKind {
    guard let parsed = MemoryKind.parse(raw) else {
        throw ValidationError("Invalid \(context) kind '\(raw)'.")
    }
    return parsed
}

private func parseMemoryStatus(_ raw: String, context: String) throws -> MemoryStatus {
    guard let parsed = MemoryStatus.parse(raw) else {
        throw ValidationError("Invalid \(context) status '\(raw)'.")
    }
    return parsed
}

private func parseFacetTags(_ rawValues: [String], context: String) throws -> Set<FacetTag> {
    var parsed: Set<FacetTag> = []
    for raw in rawValues {
        guard let tag = FacetTag.parse(raw) else {
            throw ValidationError("Invalid \(context) facet '\(raw)'.")
        }
        parsed.insert(tag)
    }
    return parsed
}

private func observeUpdateBehavior(
    index: MemoryIndex,
    setupRecords: [MemoryRecord],
    result: MemoryRecord?
) async throws -> String? {
    guard let result else { return setupRecords.isEmpty ? nil : "none" }
    guard let setup = setupRecords.first else { return "append" }

    let postRecords = try await index.recall(
        mode: .kind(result.kind),
        limit: 20,
        statuses: [.active, .superseded, .resolved, .archived]
    ).records

    if result.id == setup.id {
        return result.status == setup.status ? "dedupe" : "merge_status"
    }

    if let prior = postRecords.first(where: { $0.id == setup.id }), prior.status == .superseded {
        return result.kind == .decision ? "supersede" : "replace_active"
    }

    return "append"
}

private func runAgentMemorySuite(
    profile: EvalProfile,
    scenarios: [AgentMemoryScenarioCase],
    datasetRoot: URL,
    root: URL,
    verbose: Bool,
    responseCache: EvalResponseCache?
) async throws -> AgentMemorySuiteReport {
    let workspace = try prepareIndexWorkspace(
        suite: .storage,
        profile: profile,
        datasetRoot: datasetRoot,
        runRoot: root.appendingPathComponent("agent_memory", isDirectory: true),
        cacheEnabled: false,
        seed: "agent-memory-\(profile.rawValue)-\(scenarios.map(\.id).joined(separator: "|"))"
    )
    try resetWorkspaceForRebuild(workspace)

    var progress = DeterminateProgress(label: "agent-memory", total: scenarios.count)
    var results: [AgentMemoryScenarioResult] = []
    var totalExpectedWrites = 0
    var totalMatchedExpectedWrites = 0
    var noWriteScenarioCount = 0
    var falseWriteScenarioCount = 0
    var activeStateExpectedCount = 0
    var activeStateCorrectCount = 0
    var updateExpectedCount = 0
    var updateCorrectCount = 0
    var totalRecallQueries = 0
    var totalRecallHits = 0
    var reciprocalRanks: [Double] = []
    var latencies: [Double] = []

    for scenario in scenarios {
        let started = Date()
        let caseDatabaseURL = workspace.root
            .appendingPathComponent("cases", isDirectory: true)
            .appendingPathComponent(safeFilename(scenario.id), isDirectory: true)
            .appendingPathComponent("index.sqlite")
        try FileManager.default.createDirectory(
            at: caseDatabaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var config = try buildConfiguration(profile: profile, suite: .storage, databaseURL: caseDatabaseURL)
        if let responseCache {
            installProviderResponseCachingIfNeeded(configuration: &config, responseCache: responseCache)
        }
        let index = try MemoryIndex(configuration: config)

        var setupRecords: [MemoryRecord] = []
        for seed in scenario.setupMemories ?? [] {
            setupRecords.append(try await saveSeedMemory(seed, in: index, context: "agent-memory scenario \(scenario.id)"))
        }

        let messages = try parseScenarioMessages(scenario.messages, context: "agent-memory scenario \(scenario.id)")
        let extracted = try await index.extract(from: messages, limit: 20)
        let ingestResult = try await index.ingest(extracted)
        let allRecords = try await fetchAllMemoryRecords(index: index)

        let expectedMemories = scenario.expectedMemories ?? []
        let expectedWriteCount = scenario.expectedWriteCount ?? expectedMemories.count
        totalExpectedWrites += expectedWriteCount
        if expectedWriteCount == 0 {
            noWriteScenarioCount += 1
        }

        let matchedExpectedWrites = countMatchedExpectedMemories(
            expectedMemories,
            records: allRecords,
            context: "agent-memory scenario \(scenario.id)"
        )
        totalMatchedExpectedWrites += min(expectedWriteCount, matchedExpectedWrites)

        let falseWriteCount = expectedWriteCount == 0
            ? ingestResult.storedCount
            : max(0, ingestResult.storedCount - expectedWriteCount)
        if expectedWriteCount == 0, falseWriteCount > 0 {
            falseWriteScenarioCount += 1
        }

        let activeStateCorrect = try activeStateMatchesExpected(
            expectedMemories,
            records: allRecords,
            context: "agent-memory scenario \(scenario.id)"
        )
        if let activeStateCorrect {
            activeStateExpectedCount += 1
            if activeStateCorrect {
                activeStateCorrectCount += 1
            }
        }

        let observedUpdateBehavior: String?
        if scenario.expectedUpdateBehavior != nil {
            observedUpdateBehavior = try await observeUpdateBehavior(
                index: index,
                setupRecords: setupRecords,
                result: ingestResult.records.first
            )
            updateExpectedCount += 1
            if observedUpdateBehavior == scenario.expectedUpdateBehavior {
                updateCorrectCount += 1
            }
        } else {
            observedUpdateBehavior = nil
        }

        var scenarioRecallHits = 0
        var scenarioReciprocalRanks: [Double] = []
        for expectation in scenario.recallQueries ?? [] {
            totalRecallQueries += 1
            let expectedStatuses = try Set((expectation.expectedStatuses ?? []).map {
                try parseMemoryStatus($0, context: "agent-memory scenario \(scenario.id) recall expectation")
            })
            let response = try await index.recall(
                mode: .hybrid(query: expectation.query),
                limit: max(1, expectation.limit ?? 10),
                statuses: expectedStatuses.isEmpty ? [.active] : expectedStatuses
            )
            if let rank = try firstMatchingRecallRank(
                records: response.records,
                expectation: expectation,
                context: "agent-memory scenario \(scenario.id)"
            ) {
                scenarioRecallHits += 1
                totalRecallHits += 1
                let reciprocalRank = 1.0 / Double(rank)
                scenarioReciprocalRanks.append(reciprocalRank)
                reciprocalRanks.append(reciprocalRank)
            } else {
                scenarioReciprocalRanks.append(0)
                reciprocalRanks.append(0)
            }
        }

        let latencyMs = Date().timeIntervalSince(started) * 1000.0
        latencies.append(latencyMs)

        results.append(
            AgentMemoryScenarioResult(
                id: scenario.id,
                expectedWriteCount: expectedWriteCount,
                extractedCount: extracted.count,
                storedCount: ingestResult.storedCount,
                matchedExpectedWrites: matchedExpectedWrites,
                falseWriteCount: falseWriteCount,
                activeStateCorrect: activeStateCorrect,
                expectedUpdateBehavior: scenario.expectedUpdateBehavior,
                observedUpdateBehavior: observedUpdateBehavior,
                recallHitCount: scenarioRecallHits,
                recallQueryCount: scenario.recallQueries?.count ?? 0,
                reciprocalRanks: scenarioReciprocalRanks,
                latencyMs: latencyMs
            )
        )

        if verbose {
            print("[agent-memory] \(scenario.id): stored=\(ingestResult.storedCount) matched=\(matchedExpectedWrites) recallHits=\(scenarioRecallHits)")
        }
        progress.advance(detail: verbose ? scenario.id : nil)
    }

    return AgentMemorySuiteReport(
        totalScenarios: scenarios.count,
        falseWriteRate: safeRatio(falseWriteScenarioCount, noWriteScenarioCount, emptyDefault: 0),
        expectedWriteRecall: safeRatio(totalMatchedExpectedWrites, totalExpectedWrites, emptyDefault: 1),
        activeStateAccuracy: safeRatio(activeStateCorrectCount, activeStateExpectedCount, emptyDefault: 1),
        updateBehaviorAccuracy: safeRatio(updateCorrectCount, updateExpectedCount, emptyDefault: 1),
        recallHitRate: safeRatio(totalRecallHits, totalRecallQueries, emptyDefault: 1),
        recallMRR: reciprocalRanks.isEmpty ? 1 : reciprocalRanks.reduce(0, +) / Double(reciprocalRanks.count),
        latencyStats: computeLatencyStats(samples: latencies),
        caseResults: results.sorted { $0.id < $1.id }
    )
}

private func saveSeedMemory(
    _ seed: StorageSeedMemory,
    in index: MemoryIndex,
    context: String
) async throws -> MemoryRecord {
    let kind = try parseMemoryKind(seed.kind, context: "\(context) setup memory")
    let status = try parseMemoryStatus(seed.status ?? MemoryStatus.active.rawValue, context: "\(context) setup memory")
    let facetTags = try parseFacetTags(seed.facetTags ?? [], context: "\(context) setup memory")
    let entities = seed.entityValues?.map {
        MemoryEntity(label: .other, value: $0, normalizedValue: normalizeForMatch($0))
    } ?? []
    return try await index.save(
        text: seed.text,
        kind: kind,
        status: status,
        facetTags: facetTags,
        entities: entities,
        topics: seed.topics ?? [],
        canonicalKey: seed.canonicalKey
    )
}

private func parseScenarioMessages(
    _ rawMessages: [AgentMemoryScenarioMessage],
    context: String
) throws -> [ConversationMessage] {
    try rawMessages.map { raw in
        guard let role = ConversationRole(rawValue: raw.role) else {
            throw ValidationError("Invalid \(context) message role '\(raw.role)'.")
        }
        return ConversationMessage(role: role, content: raw.content)
    }
}

private func fetchAllMemoryRecords(index: MemoryIndex) async throws -> [MemoryRecord] {
    var recordsByID: [String: MemoryRecord] = [:]
    for kind in MemoryKind.allCases {
        let records = try await index.recall(
            mode: .kind(kind),
            limit: 1_000,
            statuses: [.active, .superseded, .resolved, .archived]
        ).records
        for record in records {
            recordsByID[record.id] = record
        }
    }
    return recordsByID.values.sorted { $0.id < $1.id }
}

private func countMatchedExpectedMemories(
    _ expectedMemories: [AgentMemoryExpectedMemory],
    records: [MemoryRecord],
    context: String
) -> Int {
    var usedIDs: Set<String> = []
    var count = 0
    for expected in expectedMemories {
        if let record = records.first(where: { record in
            !usedIDs.contains(record.id) && expectedMemoryMatches(expected, record: record)
        }) {
            usedIDs.insert(record.id)
            count += 1
        }
    }
    _ = context
    return count
}

private func activeStateMatchesExpected(
    _ expectedMemories: [AgentMemoryExpectedMemory],
    records: [MemoryRecord],
    context: String
) throws -> Bool? {
    let statusExpectations = expectedMemories.filter { $0.status != nil }
    guard !statusExpectations.isEmpty else { return nil }
    for expected in statusExpectations {
        guard let rawStatus = expected.status else { continue }
        let expectedStatus = try parseMemoryStatus(rawStatus, context: context)
        guard records.contains(where: { record in
            expectedMemoryMatches(expected, record: record) && record.status == expectedStatus
        }) else {
            return false
        }
    }
    return true
}

private func firstMatchingRecallRank(
    records: [MemoryRecord],
    expectation: AgentMemoryRecallExpectation,
    context: String
) throws -> Int? {
    let expectedKinds = try Set((expectation.expectedKinds ?? []).map {
        try parseMemoryKind($0, context: "\(context) recall expectation").rawValue
    })
    let expectedStatuses = try Set((expectation.expectedStatuses ?? []).map {
        try parseMemoryStatus($0, context: "\(context) recall expectation").rawValue
    })
    let expectedText = (expectation.expectedTextContains ?? []).map(normalizeForMatch)

    for (index, record) in records.enumerated() {
        if !expectedKinds.isEmpty, !expectedKinds.contains(record.kind.rawValue) {
            continue
        }
        if !expectedStatuses.isEmpty, !expectedStatuses.contains(record.status.rawValue) {
            continue
        }
        let text = normalizeForMatch(record.text)
        if !expectedText.allSatisfy({ text.contains($0) }) {
            continue
        }
        return index + 1
    }
    return nil
}

private func expectedMemoryMatches(
    _ expected: AgentMemoryExpectedMemory,
    record: MemoryRecord
) -> Bool {
    if let rawKind = expected.kind, record.kind.rawValue != normalizeForMatch(rawKind) {
        return false
    }
    if let rawStatus = expected.status, record.status.rawValue != normalizeForMatch(rawStatus) {
        return false
    }
    if let canonicalKey = expected.canonicalKey,
       normalizeForMatch(record.canonicalKey ?? "") != normalizeForMatch(canonicalKey) {
        return false
    }

    let text = normalizeForMatch(record.text)
    if !(expected.textContains ?? []).map(normalizeForMatch).allSatisfy({ text.contains($0) }) {
        return false
    }

    let recordFacets = Set(record.facetTags.map(\.rawValue).map(normalizeForMatch))
    let expectedFacets = Set((expected.facets ?? []).map(normalizeForMatch))
    if !expectedFacets.isSubset(of: recordFacets) {
        return false
    }

    let recordEntities = Set(record.entities.map(\.normalizedValue).map(normalizeForMatch))
    let expectedEntities = Set((expected.entities ?? []).map(normalizeForMatch))
    if !expectedEntities.isSubset(of: recordEntities) {
        return false
    }

    let recordTopics = Set(record.topics.map(normalizeForMatch))
    let expectedTopics = Set((expected.topics ?? []).map(normalizeForMatch))
    if !expectedTopics.isSubset(of: recordTopics) {
        return false
    }

    return true
}

private func safeRatio(_ numerator: Int, _ denominator: Int, emptyDefault: Double) -> Double {
    guard denominator > 0 else { return emptyDefault }
    return Double(numerator) / Double(denominator)
}

private func harmonicMean(precision: Double, recall: Double) -> Double {
    guard precision > 0 || recall > 0 else { return 0 }
    return (2 * precision * recall) / (precision + recall)
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
    if documents.isEmpty || queries.isEmpty {
        return RecallSuiteRunOutput(
            report: RecallSuiteReport(
                totalQueries: 0,
                kValues: kValues,
                metricsByK: kValues.map { RecallPerKMetric(k: $0, hitRate: 0, recall: 0, mrr: 0, ndcg: 0) },
                queryResults: [],
                perTypeMetrics: nil,
                perDifficultyMetrics: nil,
                latencyStats: nil,
                stageLatencyStats: nil,
                candidateCountStats: nil
            ),
            notes: ["No recall documents or queries were provided; recall suite skipped."]
        )
    }

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
    var memoryTypeByDocumentID: [String: String] = [:]
    var contentByDocumentID: [String: String] = [:]

    for document in documents {
        if let memoryTypeRaw = document.memoryType {
            let parsed = try parseLegacyDocumentTypeLabel(memoryTypeRaw, context: "recall document \(document.id)")
            memoryTypeByDocumentID[document.id] = parsed
        }

        let path = materializedRecallDocumentURL(document, docsRoot: docsRoot)

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
    let indexingStageCollector = IndexingStageTimingCollector()
    if canReuseIndex {
        print("[recall] Using cached index for \(documents.count) documents.")
    } else {
        print("[recall] Building index for \(documents.count) documents...")
        let indexStart = Date()
        try await index.rebuildIndex(
            from: IndexingRequest(roots: [docsRoot]),
            events: { event in
                indexingStageCollector.record(event)
            }
        )
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

        let queryStartTime = Date()
        let searchStageCollector = SearchStageTimingCollector()
        let references = try await index.memorySearch(
            query: queryCase.query,
            limit: dedupedDocumentLimit,
            features: recallFeatures(for: config),
            dedupeDocuments: true,
            includeLineRanges: false,
            events: { event in
                searchStageCollector.record(event)
            }
        )
        let queryLatencyMs = Date().timeIntervalSince(queryStartTime) * 1000.0

        var rankedDocumentIDs: [String] = []
        rankedDocumentIDs.reserveCapacity(references.count)
        for reference in references {
            guard let documentID = documentIDByPath[reference.documentPath] else { continue }
            rankedDocumentIDs.append(documentID)
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
                stageTimings: searchStageCollector.queryTimings(),
                candidateCounts: searchStageCollector.queryCounts(),
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
        memoryTypeByDocumentID: memoryTypeByDocumentID,
        maxK: maxK
    )
    let perDifficultyMetrics = computePerDifficultyMetrics(
        queryResults: queryResults,
        queries: queries,
        maxK: maxK
    )
    let latencyStats = computeLatencyStats(queryResults: queryResults)
    let stageLatencyStats = computeRecallStageLatencyStats(queryResults: queryResults)
    let candidateCountStats = computeRecallCandidateCountStats(queryResults: queryResults)

    var notes: [String] = runtimeNotes
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
            latencyStats: latencyStats,
            stageLatencyStats: stageLatencyStats,
            candidateCountStats: candidateCountStats
        ),
        notes: notes
    )
}

private func computePerTypeMetrics(
    queryResults: [RecallQueryResult],
    queries: [RecallQueryCase],
    memoryTypeByDocumentID: [String: String],
    maxK: Int
) -> [RecallPerTypeMetric] {
    let queryById = Dictionary(uniqueKeysWithValues: queries.map { ($0.id, $0) })

    var typeAccumulators: [String: (hit: Double, mrr: Double, ndcg: Double, count: Int)] = [:]
    for result in queryResults {
        guard let queryCase = queryById[result.id] else { continue }
        let relevantTypes = Set(queryCase.relevantDocumentIds.compactMap { memoryTypeByDocumentID[$0] })
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
    computeLatencyStats(samples: queryResults.compactMap(\.latencyMs))
}

private func computeRecallStageLatencyStats(queryResults: [RecallQueryResult]) -> RecallStageLatencyStats? {
    let report = RecallStageLatencyStats(
        analysisMs: computeLatencyStats(samples: queryResults.compactMap(\.stageTimings?.analysisMs)),
        expansionMs: computeLatencyStats(samples: queryResults.compactMap(\.stageTimings?.expansionMs)),
        queryEmbeddingMs: computeLatencyStats(samples: queryResults.compactMap(\.stageTimings?.queryEmbeddingMs)),
        semanticSearchMs: computeLatencyStats(samples: queryResults.compactMap(\.stageTimings?.semanticSearchMs)),
        lexicalSearchMs: computeLatencyStats(samples: queryResults.compactMap(\.stageTimings?.lexicalSearchMs)),
        fusionMs: computeLatencyStats(samples: queryResults.compactMap(\.stageTimings?.fusionMs)),
        rerankMs: computeLatencyStats(samples: queryResults.compactMap(\.stageTimings?.rerankMs)),
        totalMs: computeLatencyStats(samples: queryResults.compactMap(\.stageTimings?.totalMs))
    )
    return report.hasData ? report : nil
}

private func computeRecallCandidateCountStats(queryResults: [RecallQueryResult]) -> RecallCandidateCountStats? {
    let report = RecallCandidateCountStats(
        expandedQueries: computeCountStats(samples: queryResults.compactMap(\.candidateCounts?.expandedQueries)),
        semanticCandidates: computeCountStats(samples: queryResults.compactMap(\.candidateCounts?.semanticCandidates)),
        lexicalCandidates: computeCountStats(samples: queryResults.compactMap(\.candidateCounts?.lexicalCandidates)),
        fusedCandidates: computeCountStats(samples: queryResults.compactMap(\.candidateCounts?.fusedCandidates)),
        rerankedCandidates: computeCountStats(samples: queryResults.compactMap(\.candidateCounts?.rerankedCandidates))
    )
    return report.hasData ? report : nil
}

private func computeLatencyStats(samples: [Double]) -> RecallLatencyStats? {
    let sorted = samples.filter(\.isFinite).sorted()
    guard !sorted.isEmpty else { return nil }

    let count = sorted.count
    let mean = sorted.reduce(0, +) / Double(count)
    let p50Index = min(count - 1, count / 2)
    let p95Index = min(count - 1, Int(Double(count) * 0.95))

    return RecallLatencyStats(
        p50Ms: sorted[p50Index],
        p95Ms: sorted[p95Index],
        meanMs: mean,
        minMs: sorted.first ?? 0,
        maxMs: sorted.last ?? 0
    )
}

private func computeCountStats(samples: [Int]) -> RecallCountStats? {
    let sorted = samples.sorted()
    guard !sorted.isEmpty else { return nil }

    let count = sorted.count
    let sum = sorted.reduce(0, +)
    let p50Index = min(count - 1, count / 2)
    let p95Index = min(count - 1, Int(Double(count) * 0.95))

    return RecallCountStats(
        p50: Double(sorted[p50Index]),
        p95: Double(sorted[p95Index]),
        mean: Double(sum) / Double(count),
        min: sorted.first ?? 0,
        max: sorted.last ?? 0
    )
}

private func buildConfiguration(
    profile: EvalProfile,
    suite: SuiteKind,
    databaseURL: URL
) throws -> MemoryConfiguration {
    switch profile {
    case .nlBaseline:
        return MemoryConfiguration.naturalLanguageDefault(databaseURL: databaseURL)
    case .oracleCeiling, .coreMLDefault, .coreMLLeafIR, .coreMLRerank:
        return try MemoryConfiguration.coreMLDefault(
            databaseURL: databaseURL,
            models: CoreMLDefaultModels(
                embedding: locateCoreMLModel(name: RepoCoreMLModels.embedding)
            )
        )
    case .appleAugmented:
        var configuration = try MemoryConfiguration.coreMLDefault(
            databaseURL: databaseURL,
            models: CoreMLDefaultModels(
                embedding: locateCoreMLModel(name: RepoCoreMLModels.embedding)
            )
        )
        try enableAppleContentTagging(on: &configuration)
        if suite == .recall {
            try enableAppleRecallCapabilities(on: &configuration)
        } else if suite == .queryExpansion {
            try enableAppleExpansionCapabilities(on: &configuration)
        }
        return configuration
    }
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

private func installRecallDiagnosticsIfNeeded(
    profile: EvalProfile,
    configuration: inout MemoryConfiguration
) -> RecallDiagnosticsCollector? {
    guard profileUsesAppleRecallCapabilities(profile) else { return nil }
    guard configuration.structuredQueryExpander != nil || configuration.reranker != nil else { return nil }

    let diagnostics = RecallDiagnosticsCollector(
        expansionProviderIdentifier: configuration.structuredQueryExpander?.identifier,
        rerankProviderIdentifier: configuration.reranker?.identifier
    )
    if let structuredQueryExpander = configuration.structuredQueryExpander {
        configuration.structuredQueryExpander = DiagnosticStructuredQueryExpander(
            base: structuredQueryExpander,
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
    if let structuredQueryExpander = configuration.structuredQueryExpander {
        configuration.structuredQueryExpander = CachingStructuredQueryExpander(
            base: structuredQueryExpander,
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
    guard profile == .appleAugmented else { return nil }
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
    guard profile == .appleAugmented else { return }

    let stats = try loadContentTagGenerationStats(databaseURL: databaseURL)
    guard stats.chunkCount > 0 else { return }
    guard stats.totalTagCount > 0 else {
        throw ValidationError(
            "apple_augmented profile produced zero chunk content tags in the \(suiteLabel(suite)) suite (\(stats.chunkCount) chunks). Content tagging is unavailable or non-functional for this run."
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
    guard profile == .appleAugmented else { return }
    guard let tagger = configuration.contentTagger else {
        throw ValidationError("apple_augmented profile requires a configured content tagger.")
    }

    let probeText = "Release checklist: run migration, verify alerts, and announce rollout."

    let generated: [ContentTag]
    do {
        generated = try await withTimeout(seconds: 20) {
            try await tagger.tag(text: probeText, kind: .plainText, sourceURL: nil)
        }
    } catch is OperationTimeoutError {
        throw ValidationError(
            "apple_augmented profile timed out while probing content tagging in the \(suiteLabel(suite)) suite. Content tagging is unavailable or unresponsive for this runtime."
        )
    } catch {
        throw ValidationError(
            "apple_augmented profile failed content tagging preflight in the \(suiteLabel(suite)) suite: \(error.localizedDescription)"
        )
    }

    let valid = generated.filter { tag in
        !tag.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && tag.confidence.isFinite
    }
    guard !valid.isEmpty else {
        throw ValidationError(
            "apple_augmented profile returned zero tags in content tagging preflight for the \(suiteLabel(suite)) suite. Content tagging is unavailable or non-functional for this runtime."
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
    let database = try SQLiteDatabase(path: databaseURL.path)
    let rows = try database.fetchAll(sql: "SELECT content_tags_json FROM chunks")
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

    let expanderID = configuration.structuredQueryExpander?.identifier ?? "none"
    let rerankerID = configuration.reranker?.identifier ?? "none"
    return "[\(suiteLabel(suite))] active providers: structuredQueryExpander=\(expanderID) (\(recallProviderKind(for: expanderID))), reranker=\(rerankerID) (\(recallProviderKind(for: rerankerID)))"
}

private func recallProviderKind(for identifier: String) -> String {
    if identifier == "none" {
        return "none"
    }
    if identifier.contains("apple-intelligence-structured-query-expander") || identifier.contains("apple-intelligence-reranker") {
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
    case .queryExpansion:
        return "query-expansion"
    }
}

private func profileUsesAppleRecallCapabilities(_ profile: EvalProfile) -> Bool {
    switch profile {
    case .appleAugmented:
        return true
    case .nlBaseline, .coreMLDefault, .coreMLLeafIR, .coreMLRerank, .oracleCeiling:
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
    if configuration.structuredQueryExpander != nil {
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
        configuration.structuredQueryExpander = AppleIntelligenceStructuredQueryExpander(responseTimeoutSeconds: 5.0)
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
        configuration.structuredQueryExpander = AppleIntelligenceStructuredQueryExpander(responseTimeoutSeconds: 5.0)
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

        if let baselineExpansion = baseline.queryExpansion,
           let candidateExpansion = candidate.queryExpansion {
            let lexicalDelta = candidateExpansion.lexicalCoverageRecall - baselineExpansion.lexicalCoverageRecall
            if lexicalDelta < -threshold {
                regressions.append("\(label): Query-expansion lexical coverage regressed by \(percent(-lexicalDelta)) (baseline \(percent(baselineExpansion.lexicalCoverageRecall)) -> \(percent(candidateExpansion.lexicalCoverageRecall)))")
            }

            if let baselineHit = baselineExpansion.retrievalExpandedHitRate,
               let candidateHit = candidateExpansion.retrievalExpandedHitRate {
                let hitDelta = candidateHit - baselineHit
                if hitDelta < -threshold {
                    regressions.append("\(label): Query-expansion retrieval Hit@K regressed by \(percent(-hitDelta)) (baseline \(percent(baselineHit)) -> \(percent(candidateHit)))")
                }
            }
        }
    }

    return regressions
}

private func reportMatchesRequirement(_ report: EvalRunReport, requirement: EvalBaselineRequirement) -> Bool {
    guard report.profile == requirement.profile else { return false }

    let reportRoot = URL(fileURLWithPath: report.datasetRoot).standardizedFileURL
    if reportRoot.lastPathComponent == requirement.dataset {
        return true
    }

    if let datasetRoot = requirement.datasetRoot {
        let requiredRoot = URL(fileURLWithPath: datasetRoot).standardizedFileURL
        return reportRoot.path == requiredRoot.path || reportRoot.path.hasSuffix(requiredRoot.path)
    }

    return report.datasetRoot.hasSuffix(requirement.dataset)
}

private func reducedMetrics(from report: EvalRunReport) -> [String: Double] {
    var metrics: [String: Double] = [
        "storage.type_accuracy": report.storage.typeAccuracy,
        "storage.macro_f1": report.storage.macroF1,
        "storage.span_coverage": report.storage.spanCoverage,
    ]

    if let facetMicroF1 = report.storage.facetMicroF1 {
        metrics["storage.facet_micro_f1"] = facetMicroF1
    }
    if let entityRecall = report.storage.entityRecall {
        metrics["storage.entity_recall"] = entityRecall
    }
    if let topicRecall = report.storage.topicRecall {
        metrics["storage.topic_recall"] = topicRecall
    }
    if let updateBehaviorAccuracy = report.storage.updateBehaviorAccuracy {
        metrics["storage.update_behavior_accuracy"] = updateBehaviorAccuracy
    }

    if let maxKMetric = report.recall.metricsByK.max(by: { $0.k < $1.k }) {
        metrics["recall.hit_at_\(maxKMetric.k)"] = maxKMetric.hitRate
        metrics["recall.recall_at_\(maxKMetric.k)"] = maxKMetric.recall
        metrics["recall.mrr_at_\(maxKMetric.k)"] = maxKMetric.mrr
        metrics["recall.ndcg_at_\(maxKMetric.k)"] = maxKMetric.ndcg

        for result in report.recall.queryResults {
            let prefix = "recall.case.\(result.id)"
            if let hitAtK = result.hitByK[maxKMetric.k] {
                metrics["\(prefix).hit_at_k"] = hitAtK ? 1 : 0
            }
            if let recallAtK = result.recallByK[maxKMetric.k] {
                metrics["\(prefix).recall_at_k"] = recallAtK
            }
            if let reciprocalRankAtK = result.mrrByK[maxKMetric.k] {
                metrics["\(prefix).reciprocal_rank_at_k"] = reciprocalRankAtK
            }
            if let ndcgAtK = result.ndcgByK[maxKMetric.k] {
                metrics["\(prefix).ndcg_at_k"] = ndcgAtK
            }
        }
    }

    if let queryExpansion = report.queryExpansion {
        metrics["query_expansion.lexical_coverage_recall"] = queryExpansion.lexicalCoverageRecall
        metrics["query_expansion.semantic_coverage_recall"] = queryExpansion.semanticCoverageRecall
        metrics["query_expansion.hyde_anchor_recall"] = queryExpansion.hydeAnchorRecall
        metrics["query_expansion.facet_micro_f1"] = queryExpansion.facetMicroF1
        metrics["query_expansion.entity_recall"] = queryExpansion.entityRecall
        metrics["query_expansion.topic_recall"] = queryExpansion.topicRecall
        if let retrievalBaselineHitRate = queryExpansion.retrievalBaselineHitRate {
            metrics["query_expansion.retrieval_baseline_hit_rate"] = retrievalBaselineHitRate
        }
        if let retrievalExpandedHitRate = queryExpansion.retrievalExpandedHitRate {
            metrics["query_expansion.retrieval_expanded_hit_rate"] = retrievalExpandedHitRate
        }
        if let retrievalLift = queryExpansion.retrievalLift {
            metrics["query_expansion.retrieval_lift"] = retrievalLift
        }
        if let retrievalBaselineMRR = queryExpansion.retrievalBaselineMRR {
            metrics["query_expansion.retrieval_baseline_mrr"] = retrievalBaselineMRR
        }
        if let retrievalExpandedMRR = queryExpansion.retrievalExpandedMRR {
            metrics["query_expansion.retrieval_expanded_mrr"] = retrievalExpandedMRR
        }
        if let retrievalMRRDelta = queryExpansion.retrievalMRRDelta {
            metrics["query_expansion.retrieval_mrr_delta"] = retrievalMRRDelta
        }
        for result in queryExpansion.caseResults {
            let prefix = "query_expansion.case.\(result.id)"
            if let expandedHitAtK = result.expandedHitAtK {
                metrics["\(prefix).expanded_hit_at_k"] = expandedHitAtK ? 1 : 0
            }
            if let expandedReciprocalRankAtK = result.expandedReciprocalRankAtK {
                metrics["\(prefix).expanded_reciprocal_rank_at_k"] = expandedReciprocalRankAtK
            }
            if let reciprocalRankDeltaAtK = result.reciprocalRankDeltaAtK {
                metrics["\(prefix).reciprocal_rank_delta_at_k"] = reciprocalRankDeltaAtK
            }
        }
    }

    if let agentMemory = report.agentMemory {
        metrics["agent_memory.false_write_rate"] = agentMemory.falseWriteRate
        metrics["agent_memory.expected_write_recall"] = agentMemory.expectedWriteRecall
        metrics["agent_memory.active_state_accuracy"] = agentMemory.activeStateAccuracy
        metrics["agent_memory.update_behavior_accuracy"] = agentMemory.updateBehaviorAccuracy
        metrics["agent_memory.recall_hit_rate"] = agentMemory.recallHitRate
        metrics["agent_memory.recall_mrr"] = agentMemory.recallMRR
    }

    return metrics
}

private func metricIsLowerBetter(_ metricName: String) -> Bool {
    metricName == "agent_memory.false_write_rate"
}

private func p95Latency(from report: EvalRunReport) -> Double? {
    report.recall.latencyStats?.p95Ms
        ?? report.queryExpansion?.latencyStats?.p95Ms
        ?? report.agentMemory?.latencyStats?.p95Ms
}

private func makeComparisonMarkdown(_ reports: [EvalRunReport]) -> String {
    let sorted = reports.sorted { $0.createdAt < $1.createdAt }
    let includesExpansion = sorted.contains { $0.queryExpansion != nil }
    var lines: [String] = [
        "# Memory Eval Comparison",
        "",
        includesExpansion
            ? "| Run | Profile | Storage Acc | Macro F1 | Span Coverage | Recall Hit@K | Recall MRR@K | Recall nDCG@K | Expansion Lex Recall | Expansion Hit@K | Expansion MRR@K | Expansion MRR Delta |"
            : "| Run | Profile | Storage Acc | Macro F1 | Span Coverage | Recall Hit@K | Recall MRR@K | Recall nDCG@K |",
        includesExpansion
            ? "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            : "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for report in sorted {
        let maxMetric = report.recall.metricsByK.max(by: { $0.k < $1.k })
        let storageAccuracy = report.storage.totalCases == 0 ? "skipped" : percent(report.storage.typeAccuracy)
        let storageMacroF1 = report.storage.totalCases == 0 ? "skipped" : percent(report.storage.macroF1)
        let storageSpanCoverage = report.storage.totalCases == 0 ? "n/a" : percent(report.storage.spanCoverage)
        let hit = report.recall.totalQueries == 0 ? "skipped" : maxMetric.map { percent($0.hitRate) } ?? "n/a"
        let mrr = report.recall.totalQueries == 0 ? "skipped" : maxMetric.map { format($0.mrr) } ?? "n/a"
        let ndcg = report.recall.totalQueries == 0 ? "skipped" : maxMetric.map { format($0.ndcg) } ?? "n/a"
        let kLabel = report.recall.totalQueries == 0 ? "" : (maxMetric.map { "@\($0.k)" } ?? "")
        let expansionKLabel = report.queryExpansion == nil ? "" : (report.recall.kValues.max().map { "@\($0)" } ?? "")
        let expansionLex = report.queryExpansion.map { percent($0.lexicalCoverageRecall) } ?? "n/a"
        let expansionHit = report.queryExpansion?.retrievalExpandedHitRate.map(percent) ?? "n/a"
        let expansionMRR = report.queryExpansion?.retrievalExpandedMRR.map(format) ?? "n/a"
        let expansionMRRDelta = report.queryExpansion?.retrievalMRRDelta.map(format) ?? "n/a"

        if includesExpansion {
            lines.append(
                "| \(iso8601(report.createdAt)) | `\(report.profile.rawValue)` | \(storageAccuracy) | \(storageMacroF1) | \(storageSpanCoverage) | \(hit)\(kLabel) | \(mrr)\(kLabel) | \(ndcg)\(kLabel) | \(expansionLex) | \(expansionHit)\(expansionKLabel) | \(expansionMRR)\(expansionKLabel) | \(expansionMRRDelta) |"
            )
        } else {
            lines.append(
                "| \(iso8601(report.createdAt)) | `\(report.profile.rawValue)` | \(storageAccuracy) | \(storageMacroF1) | \(storageSpanCoverage) | \(hit)\(kLabel) | \(mrr)\(kLabel) | \(ndcg)\(kLabel) |"
            )
        }
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
        "",
        "## Recall",
        "",
        "- Queries: \(report.recall.totalQueries)",
    ]

    if report.storage.totalCases == 0 {
        lines.insert(contentsOf: [
            "- Suite status: skipped",
        ], at: 8)
    } else if report.storage.mode == "canonical_memory_schema" {
        lines.insert(contentsOf: [
            "- Kind accuracy: \(percent(report.storage.typeAccuracy))",
            "- Kind macro F1: \(percent(report.storage.macroF1))",
            "- Facet precision: \(percent(report.storage.facetPrecision ?? 0))",
            "- Facet recall: \(percent(report.storage.facetRecall ?? 0))",
            "- Facet micro F1: \(percent(report.storage.facetMicroF1 ?? 0))",
            "- Entity precision: \(percent(report.storage.entityPrecision ?? 0))",
            "- Entity recall: \(percent(report.storage.entityRecall ?? 0))",
            "- Topic recall: \(percent(report.storage.topicRecall ?? 0))",
            "- Update behavior accuracy: \(percent(report.storage.updateBehaviorAccuracy ?? 0))",
        ], at: 8)
    } else {
        lines.insert(contentsOf: [
            "- Type accuracy: \(percent(report.storage.typeAccuracy))",
            "- Macro F1: \(percent(report.storage.macroF1))",
            "- Span coverage: \(percent(report.storage.spanCoverage))",
            "- Fallback rate: \(percent(report.storage.fallbackRate))",
        ], at: 8)
    }

    if report.recall.totalQueries == 0 {
        lines.append("- Suite status: skipped")
    } else if let maxKMetric {
        lines.append("- Hit@\(maxKMetric.k): \(percent(maxKMetric.hitRate))")
        lines.append("- Recall@\(maxKMetric.k): \(percent(maxKMetric.recall))")
        lines.append("- MRR@\(maxKMetric.k): \(format(maxKMetric.mrr))")
        lines.append("- nDCG@\(maxKMetric.k): \(format(maxKMetric.ndcg))")
    }

    if let queryExpansion = report.queryExpansion {
        lines.append("")
        lines.append("## Query Expansion")
        lines.append("")
        lines.append("- Cases: \(queryExpansion.totalQueries)")
        lines.append("- Lexical coverage recall: \(percent(queryExpansion.lexicalCoverageRecall))")
        lines.append("- Semantic coverage recall: \(percent(queryExpansion.semanticCoverageRecall))")
        lines.append("- HyDE anchor recall: \(percent(queryExpansion.hydeAnchorRecall))")
        lines.append("- Facet precision: \(percent(queryExpansion.facetPrecision))")
        lines.append("- Facet recall: \(percent(queryExpansion.facetRecall))")
        lines.append("- Facet micro F1: \(percent(queryExpansion.facetMicroF1))")
        lines.append("- Entity precision: \(percent(queryExpansion.entityPrecision))")
        lines.append("- Entity recall: \(percent(queryExpansion.entityRecall))")
        lines.append("- Topic recall: \(percent(queryExpansion.topicRecall))")
        if let baselineHit = queryExpansion.retrievalBaselineHitRate,
           let expandedHit = queryExpansion.retrievalExpandedHitRate {
            lines.append("- Retrieval baseline Hit@K: \(percent(baselineHit))")
            lines.append("- Retrieval expanded Hit@K: \(percent(expandedHit))")
            if let lift = queryExpansion.retrievalLift {
                lines.append("- Retrieval lift: \(percent(lift))")
            }
        }
        if let baselineMRR = queryExpansion.retrievalBaselineMRR,
           let expandedMRR = queryExpansion.retrievalExpandedMRR {
            lines.append("- Retrieval baseline MRR@K: \(format(baselineMRR))")
            lines.append("- Retrieval expanded MRR@K: \(format(expandedMRR))")
            if let mrrDelta = queryExpansion.retrievalMRRDelta {
                lines.append("- Retrieval MRR delta: \(format(mrrDelta))")
            }
        }
        if let counts = queryExpansion.failureTaxonomyCounts, !counts.isEmpty {
            let summary = counts
                .sorted { lhs, rhs in lhs.value == rhs.value ? lhs.key < rhs.key : lhs.value > rhs.value }
                .map { "\($0.key)=\($0.value)" }
                .joined(separator: ", ")
            lines.append("- Failure taxonomy: \(summary)")
        }
        if let counts = queryExpansion.branchClassificationCounts, !counts.isEmpty {
            let summary = counts
                .sorted { lhs, rhs in lhs.value == rhs.value ? lhs.key < rhs.key : lhs.value > rhs.value }
                .map { "\($0.key)=\($0.value)" }
                .joined(separator: ", ")
            lines.append("- Branch classifications: \(summary)")
        }
        if let taxonomyMetrics = queryExpansion.taxonomyMetrics, !taxonomyMetrics.isEmpty {
            lines.append("")
            lines.append("### Query Expansion By Taxonomy")
            lines.append("")
            lines.append("| Taxonomy | Cases | Expanded Hit@K | Expanded MRR@K | MRR Delta | Improved | Worsened | Flat | Misses |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for metric in taxonomyMetrics {
                lines.append(
                    "| `\(metric.tag)` | \(metric.caseCount) | \(percent(metric.expandedHitRate)) | \(format(metric.expandedMRR)) | \(format(metric.mrrDelta)) | \(metric.improvedCount) | \(metric.worsenedCount) | \(metric.flatCount) | \(metric.missCount) |"
                )
            }
        }
        let branchCases = queryExpansion.caseResults.filter { $0.branchDiagnostics != nil }
        if !branchCases.isEmpty {
            lines.append("")
            lines.append("### Query Expansion Branch Diagnostics")
            lines.append("")
            lines.append("| Case | Classification | Final Ranks | Branch Ranks | Best Expansion Source | Taxonomy |")
            lines.append("|---|---|---|---|---|---|")
            for result in branchCases {
                guard let diagnostics = result.branchDiagnostics else { continue }
                let finalRanks = "base \(rankLabel(result.baselineBestRankAtK)), expanded \(rankLabel(result.expandedBestRankAtK))"
                lines.append(
                    "| `\(markdownTableCell(result.id))` | `\(markdownTableCell(diagnostics.classification))` | \(markdownTableCell(finalRanks)) | \(markdownTableCell(queryExpansionBranchRankSummary(diagnostics))) | \(markdownTableCell(queryExpansionSourceHitSummary(diagnostics.expansionSourceHits))) | \(markdownTableCell((result.failureTaxonomy ?? []).joined(separator: ", "))) |"
                )
            }
        }
        let missCases = queryExpansion.caseResults.filter { $0.missDiagnostics != nil }
        if !missCases.isEmpty {
            lines.append("")
            lines.append("### Query Expansion Remaining Miss Diagnostics")
            lines.append("")
            lines.append("| Case | Classification | Final Relevant | Top Final Candidates | Branch Best Relevant | Taxonomy |")
            lines.append("|---|---|---|---|---|---|")
            for result in missCases {
                guard let diagnostics = result.missDiagnostics else { continue }
                let classification = result.branchDiagnostics?.classification ?? "-"
                lines.append(
                    "| `\(markdownTableCell(result.id))` | `\(markdownTableCell(classification))` | \(markdownTableCell(queryExpansionMissFinalRelevantSummary(diagnostics))) | \(markdownTableCell(queryExpansionRankedCandidateSummary(diagnostics.finalTopCandidates))) | \(markdownTableCell(queryExpansionMissBranchSummary(diagnostics))) | \(markdownTableCell((result.failureTaxonomy ?? []).joined(separator: ", "))) |"
                )
            }
        }
    }

    if let agentMemory = report.agentMemory {
        lines.append("")
        lines.append("## Agent Memory")
        lines.append("")
        lines.append("- Scenarios: \(agentMemory.totalScenarios)")
        lines.append("- False-write rate: \(percent(agentMemory.falseWriteRate))")
        lines.append("- Expected-write recall: \(percent(agentMemory.expectedWriteRecall))")
        lines.append("- Active-state accuracy: \(percent(agentMemory.activeStateAccuracy))")
        lines.append("- Update behavior accuracy: \(percent(agentMemory.updateBehaviorAccuracy))")
        lines.append("- Recall Hit: \(percent(agentMemory.recallHitRate))")
        lines.append("- Recall MRR: \(format(agentMemory.recallMRR))")
        if let latency = agentMemory.latencyStats {
            lines.append("- Scenario latency p95: \(String(format: "%.1f", latency.p95Ms)) ms")
        }
    }

    lines.append("")
    if let ingestStats = report.storage.stageLatencyStats {
        lines.append("### Storage Stage Latency")
        lines.append("")
        lines.append("| Stage | p50 | p95 | mean |")
        lines.append("|---|---:|---:|---:|")
        lines.append("| typing | \(formatStageMs(ingestStats.typingMs)) | \(formatStageMs(ingestStats.typingMs, keyPath: \.p95Ms)) | \(formatStageMs(ingestStats.typingMs, keyPath: \.meanMs)) |")
        lines.append("| chunking | \(formatStageMs(ingestStats.chunkingMs)) | \(formatStageMs(ingestStats.chunkingMs, keyPath: \.p95Ms)) | \(formatStageMs(ingestStats.chunkingMs, keyPath: \.meanMs)) |")
        lines.append("| tagging | \(formatStageMs(ingestStats.taggingMs)) | \(formatStageMs(ingestStats.taggingMs, keyPath: \.p95Ms)) | \(formatStageMs(ingestStats.taggingMs, keyPath: \.meanMs)) |")
        lines.append("| embedding | \(formatStageMs(ingestStats.embeddingMs)) | \(formatStageMs(ingestStats.embeddingMs, keyPath: \.p95Ms)) | \(formatStageMs(ingestStats.embeddingMs, keyPath: \.meanMs)) |")
        lines.append("| index_write | \(formatStageMs(ingestStats.indexWriteMs)) | \(formatStageMs(ingestStats.indexWriteMs, keyPath: \.p95Ms)) | \(formatStageMs(ingestStats.indexWriteMs, keyPath: \.meanMs)) |")
        lines.append("| total | \(formatStageMs(ingestStats.totalMs)) | \(formatStageMs(ingestStats.totalMs, keyPath: \.p95Ms)) | \(formatStageMs(ingestStats.totalMs, keyPath: \.meanMs)) |")
        lines.append("")
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

    if let stageStats = report.recall.stageLatencyStats {
        lines.append("")
        lines.append("### Query Stage Latency")
        lines.append("")
        lines.append("| Stage | p50 | p95 | mean |")
        lines.append("|---|---:|---:|---:|")
        lines.append("| analysis | \(formatStageMs(stageStats.analysisMs)) | \(formatStageMs(stageStats.analysisMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.analysisMs, keyPath: \.meanMs)) |")
        lines.append("| expansion | \(formatStageMs(stageStats.expansionMs)) | \(formatStageMs(stageStats.expansionMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.expansionMs, keyPath: \.meanMs)) |")
        lines.append("| query_embedding | \(formatStageMs(stageStats.queryEmbeddingMs)) | \(formatStageMs(stageStats.queryEmbeddingMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.queryEmbeddingMs, keyPath: \.meanMs)) |")
        lines.append("| semantic_search | \(formatStageMs(stageStats.semanticSearchMs)) | \(formatStageMs(stageStats.semanticSearchMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.semanticSearchMs, keyPath: \.meanMs)) |")
        lines.append("| lexical_search | \(formatStageMs(stageStats.lexicalSearchMs)) | \(formatStageMs(stageStats.lexicalSearchMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.lexicalSearchMs, keyPath: \.meanMs)) |")
        lines.append("| fusion | \(formatStageMs(stageStats.fusionMs)) | \(formatStageMs(stageStats.fusionMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.fusionMs, keyPath: \.meanMs)) |")
        lines.append("| rerank | \(formatStageMs(stageStats.rerankMs)) | \(formatStageMs(stageStats.rerankMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.rerankMs, keyPath: \.meanMs)) |")
        lines.append("| total | \(formatStageMs(stageStats.totalMs)) | \(formatStageMs(stageStats.totalMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.totalMs, keyPath: \.meanMs)) |")
    }

    if let countStats = report.recall.candidateCountStats {
        lines.append("")
        lines.append("### Query Candidate Counts")
        lines.append("")
        lines.append("| Stage | p50 | p95 | mean | min | max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append(candidateCountRow(label: "expanded_queries", stats: countStats.expandedQueries))
        lines.append(candidateCountRow(label: "semantic_candidates", stats: countStats.semanticCandidates))
        lines.append(candidateCountRow(label: "lexical_candidates", stats: countStats.lexicalCandidates))
        lines.append(candidateCountRow(label: "fused_candidates", stats: countStats.fusedCandidates))
        lines.append(candidateCountRow(label: "reranked_candidates", stats: countStats.rerankedCandidates))
    }

    if let queryExpansion = report.queryExpansion,
       let stageStats = queryExpansion.stageLatencyStats {
        lines.append("")
        lines.append("### Query Expansion Stage Latency")
        lines.append("")
        lines.append("| Stage | p50 | p95 | mean |")
        lines.append("|---|---:|---:|---:|")
        lines.append("| analysis | \(formatStageMs(stageStats.analysisMs)) | \(formatStageMs(stageStats.analysisMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.analysisMs, keyPath: \.meanMs)) |")
        lines.append("| expansion | \(formatStageMs(stageStats.expansionMs)) | \(formatStageMs(stageStats.expansionMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.expansionMs, keyPath: \.meanMs)) |")
        lines.append("| query_embedding | \(formatStageMs(stageStats.queryEmbeddingMs)) | \(formatStageMs(stageStats.queryEmbeddingMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.queryEmbeddingMs, keyPath: \.meanMs)) |")
        lines.append("| semantic_search | \(formatStageMs(stageStats.semanticSearchMs)) | \(formatStageMs(stageStats.semanticSearchMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.semanticSearchMs, keyPath: \.meanMs)) |")
        lines.append("| lexical_search | \(formatStageMs(stageStats.lexicalSearchMs)) | \(formatStageMs(stageStats.lexicalSearchMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.lexicalSearchMs, keyPath: \.meanMs)) |")
        lines.append("| fusion | \(formatStageMs(stageStats.fusionMs)) | \(formatStageMs(stageStats.fusionMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.fusionMs, keyPath: \.meanMs)) |")
        lines.append("| total | \(formatStageMs(stageStats.totalMs)) | \(formatStageMs(stageStats.totalMs, keyPath: \.p95Ms)) | \(formatStageMs(stageStats.totalMs, keyPath: \.meanMs)) |")
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

private func rankLabel(_ rank: Int?) -> String {
    rank.map(String.init) ?? "-"
}

private func markdownTableCell(_ value: String) -> String {
    value
        .replacingOccurrences(of: "|", with: "/")
        .replacingOccurrences(of: "\n", with: " ")
}

private func queryExpansionBranchRankSummary(_ diagnostics: QueryExpansionBranchDiagnostics) -> String {
    "base semantic \(rankLabel(diagnostics.baselineSemanticRankAtK)), base lexical \(rankLabel(diagnostics.baselineLexicalRankAtK)), expanded semantic \(rankLabel(diagnostics.expandedSemanticRankAtK)), expanded lexical \(rankLabel(diagnostics.expandedLexicalRankAtK))"
}

private func queryExpansionSourceHitSummary(_ hits: [QueryExpansionSourceHit]) -> String {
    guard let best = hits
        .compactMap({ hit -> (QueryExpansionSourceHit, Int)? in
            guard let rank = hit.rankAtK else { return nil }
            return (hit, rank)
        })
        .sorted(by: { lhs, rhs in
            if lhs.1 == rhs.1 {
                return lhs.0.kind < rhs.0.kind
            }
            return lhs.1 < rhs.1
        })
        .first
    else {
        return "-"
    }
    return "\(best.0.kind) \(best.1): \(truncateForMarkdown(best.0.query, maxLength: 80))"
}

private func queryExpansionMissFinalRelevantSummary(_ diagnostics: QueryExpansionMissDiagnostics) -> String {
    guard let rank = diagnostics.finalBestRelevantRank,
          let documentId = diagnostics.finalBestRelevantDocumentId,
          let score = diagnostics.finalBestRelevantScore else {
        return "-"
    }
    return "\(rank): \(documentId) [\(queryExpansionScoreSummary(score))]"
}

private func queryExpansionRankedCandidateSummary(
    _ candidates: [QueryExpansionRankedCandidate],
    maxCount: Int = 3
) -> String {
    guard !candidates.isEmpty else { return "-" }
    return candidates.prefix(maxCount).map { candidate in
        let marker = candidate.isRelevant ? "*" : ""
        return "\(candidate.rank):\(marker)\(candidate.documentId) [\(queryExpansionScoreSummary(candidate.score))]"
    }
    .joined(separator: "; ")
}

private func queryExpansionMissBranchSummary(_ diagnostics: QueryExpansionMissDiagnostics) -> String {
    guard !diagnostics.branchRanks.isEmpty else { return "-" }
    return diagnostics.branchRanks.map { branch in
        guard let rank = branch.bestRelevantRank,
              let documentId = branch.bestRelevantDocumentId else {
            return "\(branch.branch) -"
        }
        let score = branch.bestRelevantScore.map { " [\(queryExpansionScoreSummary($0))]" } ?? ""
        return "\(branch.branch) \(rank): \(documentId)\(score)"
    }
    .joined(separator: "; ")
}

private func queryExpansionScoreSummary(_ score: SearchScoreBreakdown) -> String {
    "sem \(format(score.semantic)), lex \(format(score.lexical)), rec \(format(score.recency)), tag \(format(score.tag)), schema \(format(score.schema)), temporal \(format(score.temporal)), status \(format(score.status)), fused \(format(score.fused)), blended \(format(score.blended))"
}

private func truncateForMarkdown(_ value: String, maxLength: Int) -> String {
    guard value.count > maxLength else { return value }
    return String(value.prefix(max(0, maxLength - 3))) + "..."
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

private func parseLegacyDocumentTypeLabel(_ raw: String, context: String) throws -> String {
    let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard legacyDocumentTypeLabels.contains(normalized) else {
        let allowed = legacyDocumentTypeLabels.joined(separator: ", ")
        throw EvalError.invalidDataset("Unknown legacy memory type '\(raw)' in \(context). Allowed: \(allowed)")
    }
    return normalized
}

private func parseOptionalMemoryTypes(_ rawValues: [String]?) throws -> Set<String>? {
    guard let rawValues, !rawValues.isEmpty else { return nil }

    var result: Set<String> = []
    for raw in rawValues {
        let type = try parseLegacyDocumentTypeLabel(raw, context: "query memory_types")
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

private func firstRelevantRank(in ranked: [String], relevant: Set<String>, maxK: Int) -> Int? {
    guard maxK > 0 else { return nil }
    return ranked.prefix(maxK).firstIndex { relevant.contains($0) }.map { $0 + 1 }
}

private func reciprocalRank(for rank: Int?) -> Double {
    guard let rank, rank > 0 else { return 0 }
    return 1.0 / Double(rank)
}

private func reciprocalRankDelta(baseline: Double?, expanded: Double?) -> Double? {
    guard let baseline, let expanded else { return nil }
    return expanded - baseline
}

private func rankImprovement(baselineRank: Int?, expandedRank: Int?) -> Int? {
    guard let baselineRank, let expandedRank else { return nil }
    return baselineRank - expandedRank
}

private func materializeRecallDocument(_ document: RecallDocumentCase) throws -> String {
    if let memoryTypeRaw = document.memoryType {
        _ = try parseLegacyDocumentTypeLabel(memoryTypeRaw, context: "recall document \(document.id)")
    }
    return document.text
}

private func materializedRecallDocumentURL(_ document: RecallDocumentCase, docsRoot: URL) -> URL {
    materializedRecallDocumentURL(
        id: document.id,
        kind: document.kind,
        relativePath: document.relativePath,
        docsRoot: docsRoot
    )
}

func materializedRecallDocumentURL(id: String, kind: String?, relativePath: String?, docsRoot: URL) -> URL {
    let materializedExtension = extensionForKind(kind) ?? "md"
    let relativePath = relativePath?.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let relativePath, !relativePath.isEmpty else {
        return docsRoot.appendingPathComponent("\(safeFilename(id)).\(materializedExtension)")
    }

    let rawURL = docsRoot.appendingPathComponent(relativePath)
    let rawExtension = rawURL.pathExtension.lowercased()
    if !rawExtension.isEmpty, MemoryConfiguration.defaultSupportedExtensions.contains(rawExtension) {
        return rawURL
    }
    if rawExtension.isEmpty {
        return rawURL.appendingPathExtension(materializedExtension)
    }
    return rawURL.deletingPathExtension().appendingPathExtension(materializedExtension)
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

private func formatStageMs(
    _ stats: RecallLatencyStats?,
    keyPath: KeyPath<RecallLatencyStats, Double> = \.p50Ms
) -> String {
    guard let stats else { return "n/a" }
    return String(format: "%.1f ms", stats[keyPath: keyPath])
}

private func candidateCountRow(label: String, stats: RecallCountStats?) -> String {
    guard let stats else { return "| \(label) | n/a | n/a | n/a | n/a | n/a |" }
    return "| \(label) | \(String(format: "%.1f", stats.p50)) | \(String(format: "%.1f", stats.p95)) | \(String(format: "%.1f", stats.mean)) | \(stats.min) | \(stats.max) |"
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
