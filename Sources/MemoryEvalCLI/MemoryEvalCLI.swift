import ArgumentParser
import Foundation
import GRDB
import Memory
import MemoryAppleIntelligence
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
- `swift run memory_eval run --profile full_apple --dataset-root ./Evals`
- `swift run memory_eval compare ./Evals/runs/*.json`
"""

private let storageTemplate = """
{"id":"storage-1","kind":"markdown","text":"I felt frustrated during yesterday's outage review.","expected_memory_type":"emotional","required_spans":["frustrated","outage review"]}
{"id":"storage-2","kind":"markdown","text":"Step 1: run migration. Step 2: verify alerts. Step 3: announce release.","expected_memory_type":"procedural","required_spans":["Step 1","verify alerts"]}
"""

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
}

enum EvalProfile: String, CaseIterable, Codable, ExpressibleByArgument {
    case baseline
    case appleStorage = "apple_storage"
    case appleRecall = "apple_recall"
    case fullApple = "full_apple"
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
}

private struct RecallSuiteReport: Codable {
    var totalQueries: Int
    var kValues: [Int]
    var metricsByK: [RecallPerKMetric]
    var queryResults: [RecallQueryResult]
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

    @Option(name: .long, help: "Profile to run.")
    var profile: EvalProfile = .baseline

    @Option(name: .long, help: "Dataset root folder.")
    var datasetRoot: String = "Evals"

    @Option(name: .long, help: "Comma-separated k values, e.g. 1,3,5,10.")
    var kValues: String = "1,3,5,10"

    @Option(name: .long, help: "Output JSON file path. Defaults to <dataset-root>/runs/<timestamp>-<profile>.json.")
    var output: String?

    @Flag(name: .long, help: "Print per-case details.")
    var verbose = false

    mutating func run() async throws {
        let datasetRootURL = URL(fileURLWithPath: NSString(string: datasetRoot).expandingTildeInPath).standardizedFileURL
        let dataset = try loadDataset(root: datasetRootURL)
        let ks = try parseKValues(kValues)

        let runRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-evals", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: runRoot, withIntermediateDirectories: true)

        let storageReport = try await runStorageSuite(
            profile: profile,
            dataset: dataset.storageCases,
            root: runRoot,
            verbose: verbose
        )
        let recallReport = try await runRecallSuite(
            profile: profile,
            documents: dataset.recallDocuments,
            queries: dataset.recallQueries,
            kValues: ks,
            root: runRoot,
            verbose: verbose
        )

        let report = EvalRunReport(
            schemaVersion: 1,
            createdAt: Date(),
            profile: profile,
            datasetRoot: datasetRootURL.path,
            storage: storageReport,
            recall: recallReport,
            notes: [
                "Storage eval uses direct database inspection for classification/span metrics.",
                "Recall eval uses document-level metrics (deduped by document path).",
            ]
        )

        let outputURL = try resolvedOutputURL(baseRoot: datasetRootURL, output: output, profile: profile)
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
            print("Recall MRR@\(maxKMetric.k): \(format(maxKMetric.mrr))")
            print("Recall nDCG@\(maxKMetric.k): \(format(maxKMetric.ndcg))")
        }
        print("JSON report: \(outputURL.path)")
        print("Markdown summary: \(markdownURL.path)")
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

private func runStorageSuite(
    profile: EvalProfile,
    dataset: [StorageCase],
    root: URL,
    verbose: Bool
) async throws -> StorageSuiteReport {
    let docsRoot = root.appendingPathComponent("storage_docs", isDirectory: true)
    try FileManager.default.createDirectory(at: docsRoot, withIntermediateDirectories: true)

    var casePathByID: [String: URL] = [:]
    for entry in dataset {
        let ext = extensionForKind(entry.kind) ?? "md"
        let path = docsRoot.appendingPathComponent("\(safeFilename(entry.id)).\(ext)")
        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        try entry.text.write(to: path, atomically: true, encoding: .utf8)
        casePathByID[entry.id] = path
    }

    let dbURL = root.appendingPathComponent("storage.sqlite")
    var config = try buildConfiguration(profile: profile, suite: .storage, databaseURL: dbURL)
    config.chunker = DefaultChunker(targetTokenCount: 12_000, overlapTokenCount: 0)

    let index = try MemoryIndex(configuration: config)
    print("[storage] Building index for \(dataset.count) cases...")
    let indexStart = Date()
    try await index.rebuildIndex(from: [docsRoot])
    print("[storage] Index built in \(formatDuration(Date().timeIntervalSince(indexStart))).")

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
    root: URL,
    verbose: Bool
) async throws -> RecallSuiteReport {
    let docsRoot = root.appendingPathComponent("recall_docs", isDirectory: true)
    try FileManager.default.createDirectory(at: docsRoot, withIntermediateDirectories: true)

    var pathByDocumentID: [String: String] = [:]
    var documentIDByPath: [String: String] = [:]

    for document in documents {
        let ext = extensionForKind(document.kind) ?? "md"
        let relativePath = document.relativePath?.trimmingCharacters(in: .whitespacesAndNewlines)
        let path: URL
        if let relativePath, !relativePath.isEmpty {
            path = docsRoot.appendingPathComponent(relativePath)
        } else {
            path = docsRoot.appendingPathComponent("\(safeFilename(document.id)).\(ext)")
        }

        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        let content = try materializeRecallDocument(document)
        try content.write(to: path, atomically: true, encoding: .utf8)

        pathByDocumentID[document.id] = path.path
        documentIDByPath[path.path] = document.id
    }

    let dbURL = root.appendingPathComponent("recall.sqlite")
    let config = try buildConfiguration(profile: profile, suite: .recall, databaseURL: dbURL)
    let index = try MemoryIndex(configuration: config)
    print("[recall] Building index for \(documents.count) documents...")
    let indexStart = Date()
    try await index.rebuildIndex(from: [docsRoot])
    print("[recall] Index built in \(formatDuration(Date().timeIntervalSince(indexStart))).")
    print("[recall] Evaluating \(queries.count) queries at k=\(kValues.map(String.init).joined(separator: ","))...")

    var queryResults: [RecallQueryResult] = []
    var perKAccumulator: [Int: (hit: Double, recall: Double, mrr: Double, ndcg: Double)] = [:]
    for k in kValues {
        perKAccumulator[k] = (0, 0, 0, 0)
    }

    let maxK = kValues.max() ?? 10
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
        let searchResults = try await index.search(
            SearchQuery(
                text: queryCase.query,
                limit: max(50, maxK * 4),
                semanticCandidateLimit: 300,
                lexicalCandidateLimit: 300,
                rerankLimit: config.reranker == nil ? 0 : max(50, maxK * 3),
                expansionLimit: config.queryExpander == nil ? 0 : 2,
                memoryTypes: filterMemoryTypes
            )
        )

        var rankedDocumentIDs: [String] = []
        var seen: Set<String> = []
        for result in searchResults {
            guard let documentID = documentIDByPath[result.documentPath] else { continue }
            if seen.insert(documentID).inserted {
                rankedDocumentIDs.append(documentID)
            }
            if rankedDocumentIDs.count >= max(100, maxK * 4) {
                break
            }
        }

        var hitByK: [Int: Bool] = [:]
        var recallByK: [Int: Double] = [:]
        var mrrByK: [Int: Double] = [:]
        var ndcgByK: [Int: Double] = [:]

        for k in kValues {
            let top = Array(rankedDocumentIDs.prefix(k))
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
                retrievedDocumentIds: Array(rankedDocumentIDs.prefix(maxK)),
                hitByK: hitByK,
                recallByK: recallByK,
                mrrByK: mrrByK,
                ndcgByK: ndcgByK
            )
        )

        if verbose {
            let hitAtMax = hitByK[maxK] == true ? "hit" : "miss"
            print("[recall] \(queryCase.id): \(hitAtMax) @\(maxK)")
        }
        progress.advance(detail: verbose ? queryCase.id : nil)
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

    return RecallSuiteReport(
        totalQueries: totalQueries,
        kValues: kValues,
        metricsByK: metrics,
        queryResults: queryResults.sorted { $0.id < $1.id }
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
        classifier: HeuristicMemoryTypeClassifier(),
        fallbackType: .factual
    )

    switch profile {
    case .baseline:
        break
    case .appleStorage:
        if suite == .storage {
            configuration.memoryTyping.classifier = try makeAppleFirstMemoryClassifier()
        }
    case .appleRecall:
        if suite == .recall {
            try enableAppleRecallCapabilities(on: &configuration)
        }
    case .fullApple:
        configuration.memoryTyping.classifier = try makeAppleFirstMemoryClassifier()
        if suite == .recall {
            try enableAppleRecallCapabilities(on: &configuration)
        }
    }

    return configuration
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

private func enableAppleRecallCapabilities(on configuration: inout MemoryConfiguration) throws {
    #if canImport(FoundationModels)
    if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isAvailable {
        configuration.queryExpander = AppleIntelligenceQueryExpander()
        configuration.reranker = AppleIntelligenceReranker()
        return
    }
    #endif
    throw ValidationError(
        "Apple Intelligence is unavailable for this runtime. Apple recall profiles require query expansion/reranking support."
    )
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

    let misses = report.recall.queryResults.filter { result in
        let maxK = maxKMetric?.k ?? (report.recall.kValues.max() ?? 10)
        return result.hitByK[maxK] == false
    }
    if !misses.isEmpty {
        lines.append("")
        lines.append("### Misses At Max K")
        lines.append("")
        for miss in misses.prefix(10) {
            lines.append("- `\(miss.id)`: \(miss.query)")
            lines.append("  - relevant: \(miss.relevantDocumentIds.joined(separator: ", "))")
            lines.append("  - retrieved: \(miss.retrievedDocumentIds.joined(separator: ", "))")
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

        let kind = parseDocumentKind(document.kind) ?? .markdown
        if kind == .markdown {
            let trimmed = document.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("---\n") {
                return document.text
            }
            return """
            ---
            memory_type: \(memoryTypeRaw)
            ---
            \(document.text)
            """
        }
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
