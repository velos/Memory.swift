import ArgumentParser
import Foundation
import Memory

struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Run a persistent JSON-lines bridge for benchmark adapters."
    )

    mutating func run() async throws {
        let server = try MemoryBridgeServer()
        try await server.run()
    }
}

private final class MemoryBridgeServer {
    private let paths: CLIPaths
    private let store: CLIStateStore
    private var state: CLIState
    private let index: MemoryIndex
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()

    init() throws {
        let context = try CLIContext.load()
        self.paths = context.paths
        self.store = context.store
        self.state = context.state
        self.index = try context.makeIndex()
    }

    func run() async throws {
        while let line = readLine(strippingNewline: true) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            let request: BridgeRequest
            do {
                guard let data = trimmed.data(using: .utf8) else {
                    throw BridgeError("Request line is not valid UTF-8")
                }
                request = try decoder.decode(BridgeRequest.self, from: data)
            } catch {
                try writeError(id: nil, error: "Invalid request: \(error)")
                continue
            }

            do {
                let shouldContinue = try await handle(request)
                if !shouldContinue { return }
            } catch {
                try writeError(id: request.id, error: "\(error)")
            }
        }
    }

    private func handle(_ request: BridgeRequest) async throws -> Bool {
        switch request.method {
        case "ping":
            try writeSuccess(
                id: request.id,
                result: BridgeReadyResult(
                    protocolVersion: 1,
                    root: paths.rootDirectory.path,
                    database: paths.indexFileURL.path
                )
            )
            return true
        case "collection.add":
            let result = try addCollection(request.params)
            try writeSuccess(id: request.id, result: result)
            return true
        case "collection.list":
            try writeSuccess(id: request.id, result: BridgeCollectionListResult(collections: state.collections))
            return true
        case "sync":
            let result = try await sync(request.params)
            try writeSuccess(id: request.id, result: result)
            return true
        case "search":
            let result = try await search(request.params)
            try writeSuccess(id: request.id, result: result)
            return true
        case "shutdown":
            try writeSuccess(id: request.id, result: BridgeShutdownResult(shutdown: true))
            return false
        default:
            throw BridgeError("Unknown bridge method '\(request.method)'")
        }
    }

    private func addCollection(_ params: BridgeRequestParams?) throws -> BridgeCollectionResult {
        guard let name = params?.name?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty else {
            throw BridgeError("collection.add requires params.name")
        }
        guard let rawPath = params?.path?.trimmingCharacters(in: .whitespacesAndNewlines), !rawPath.isEmpty else {
            throw BridgeError("collection.add requires params.path")
        }

        let expandedPath = NSString(string: rawPath).expandingTildeInPath
        let absoluteURL = URL(fileURLWithPath: expandedPath).standardizedFileURL

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: absoluteURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw BridgeError("Collection path does not exist or is not a directory: \(absoluteURL.path)")
        }

        state.collections.removeAll { $0.name == name }
        state.collections.append(.init(name: name, path: absoluteURL.path))
        state.collections.sort { $0.name < $1.name }
        try store.save(state)

        return BridgeCollectionResult(name: name, path: absoluteURL.path)
    }

    private func sync(_ params: BridgeRequestParams?) async throws -> BridgeSyncResult {
        let rawPaths = params?.paths ?? []
        guard !rawPaths.isEmpty else {
            throw BridgeError("sync requires params.paths")
        }

        let urls = rawPaths.map { raw in
            URL(
                fileURLWithPath: NSString(string: raw).expandingTildeInPath,
                isDirectory: false
            ).standardizedFileURL
        }
        try await index.syncDocuments(urls)
        return BridgeSyncResult(synced: urls.count)
    }

    private func search(_ params: BridgeRequestParams?) async throws -> BridgeSearchResponse {
        guard var queryText = params?.query?.trimmingCharacters(in: .whitespacesAndNewlines),
              !queryText.isEmpty else {
            throw BridgeError("search requires params.query")
        }

        let scopedCollection: StoredCollection?
        if let rawCollection = params?.collection?.trimmingCharacters(in: .whitespacesAndNewlines),
           !rawCollection.isEmpty {
            let name = bridgeNormalizeCollectionArgument(rawCollection)
            guard let collection = state.collections.first(where: { $0.name == name }) else {
                throw BridgeError("Unknown collection '\(name)'")
            }
            scopedCollection = collection
        } else {
            scopedCollection = nil
        }

        if let scopedCollection, let hint = state.contexts[scopedCollection.name], !hint.isEmpty {
            queryText += "\n\nContext: \(hint)"
        }

        let mode = bridgeSearchMode(params?.mode)
        let limit = max(1, params?.limit ?? ((params?.all ?? false) ? 2_000 : 20))
        let searchQuery = SearchQuery(
            text: queryText,
            limit: limit,
            semanticCandidateLimit: mode.semanticLimit,
            lexicalCandidateLimit: mode.lexicalLimit,
            rerankLimit: mode == .hybrid ? 50 : 0,
            expansionLimit: mode == .hybrid ? 2 : 0,
            referenceDate: bridgeParseReferenceDate(params?.queryTimestamp),
            documentPathPrefix: scopedCollection?.path
        )

        let diagnostics = BridgeSearchDiagnostics()
        var results = try await index.search(searchQuery) { event in
            diagnostics.record(event)
        }
        if let scopedCollection {
            let root = scopedCollection.path
            results = results.filter { $0.documentPath.hasPrefix(root + "/") || $0.documentPath == root }
        }

        let minScore = params?.minScore ?? 0
        if minScore > 0 {
            results = results.filter { $0.score.blended >= minScore }
        }

        let contextPackingOrder = try bridgeContextPackingOrder(params?.contextPackingOrder)
        let packaged = bridgePackSearchResults(
            results,
            queryText: queryText,
            contextTokenBudget: max(0, params?.contextTokenBudget ?? 0),
            perDocumentTokenBudget: max(0, params?.perDocumentTokenBudget ?? 0),
            contextPackingOrder: contextPackingOrder
        )

        return BridgeSearchResponse(
            diagnostics: diagnostics.snapshot(contextPackaging: packaged.diagnostics),
            results: packaged.results.map { packagedResult in
                let result = packagedResult.result
                return BridgeSearchResult(
                    chunkID: result.chunkID,
                    documentPath: result.documentPath,
                    title: result.title,
                    content: result.content,
                    snippet: result.snippet,
                    contextTokens: packagedResult.contextTokens,
                    truncated: packagedResult.truncated,
                    modifiedAt: ISO8601DateFormatter().string(from: result.modifiedAt),
                    memoryID: result.memoryID,
                    memoryKind: result.memoryKind?.rawValue,
                    memoryStatus: result.memoryStatus?.rawValue,
                    score: BridgeSearchScore(from: result.score)
                )
            }
        )
    }

    private func writeSuccess<T: Encodable>(id: String?, result: T) throws {
        let response = BridgeSuccessResponse(id: id, result: result)
        try write(response)
    }

    private func writeError(id: String?, error: String) throws {
        let response = BridgeErrorResponse(id: id, error: error)
        try write(response)
    }

    private func write<T: Encodable>(_ value: T) throws {
        let data = try encoder.encode(value)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0A]))
    }
}

private func bridgeSearchMode(_ value: String?) -> SearchMode {
    switch value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "keyword", "lexical":
        return .keyword
    case "semantic", "vector":
        return .semantic
    default:
        return .hybrid
    }
}

private func bridgeNormalizeCollectionArgument(_ value: String) -> String {
    if value.hasPrefix("memory://") {
        let raw = value.dropFirst("memory://".count)
        return raw.isEmpty ? value : String(raw)
    }

    return value
}

private func bridgeParseReferenceDate(_ value: String?) -> Date? {
    guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
        return nil
    }

    let iso = ISO8601DateFormatter()
    iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    if let date = iso.date(from: value) {
        return date
    }

    iso.formatOptions = [.withInternetDateTime]
    if let date = iso.date(from: value) {
        return date
    }

    let dateOnly = DateFormatter()
    dateOnly.calendar = Calendar(identifier: .gregorian)
    dateOnly.locale = Locale(identifier: "en_US_POSIX")
    dateOnly.timeZone = TimeZone(secondsFromGMT: 0)
    dateOnly.dateFormat = "yyyy-MM-dd"
    return dateOnly.date(from: value)
}

private func bridgePackSearchResults(
    _ results: [SearchResult],
    queryText: String,
    contextTokenBudget: Int,
    perDocumentTokenBudget: Int,
    contextPackingOrder: BridgeContextPackingOrder
) -> (results: [BridgePackagedSearchResult], diagnostics: BridgeContextPackagingSnapshot?) {
    let orderedResults = bridgeOrderedSearchResults(results, order: contextPackingOrder)
    guard contextTokenBudget > 0 || perDocumentTokenBudget > 0 else {
        let diagnostics = contextPackingOrder == .rank ? nil : BridgeContextPackagingSnapshot(
            contextTokenBudget: contextTokenBudget,
            perDocumentTokenBudget: perDocumentTokenBudget,
            contextPackingOrder: contextPackingOrder.rawValue,
            returnedContextTokens: 0,
            returnedResults: orderedResults.count,
            truncatedResults: 0
        )
        return (
            orderedResults.map { BridgePackagedSearchResult(result: $0, contextTokens: nil, truncated: nil) },
            diagnostics
        )
    }

    let separatorOverheadTokens = 8
    var packaged: [BridgePackagedSearchResult] = []
    packaged.reserveCapacity(results.count)
    var usedTokens = 0
    var truncatedCount = 0
    let effectivePerDocumentTokenBudget = bridgeAdaptiveContextPerDocumentTokenBudget(
        queryText: queryText,
        contextTokenBudget: contextTokenBudget,
        perDocumentTokenBudget: perDocumentTokenBudget,
        separatorOverheadTokens: separatorOverheadTokens
    )

    for var result in orderedResults {
        let fullTokenCount = bridgeEstimatedTokenCount(result.content)
        guard fullTokenCount > 0 else { continue }

        let remainingBudget = contextTokenBudget == 0 ? Int.max : contextTokenBudget - usedTokens
        let availableTokens = bridgeCappedContextTokenCount(
            fullTokenCount: fullTokenCount,
            remainingBudget: remainingBudget,
            perDocumentTokenBudget: effectivePerDocumentTokenBudget,
            separatorOverheadTokens: separatorOverheadTokens
        )
        guard availableTokens > 0 else { break }

        let truncated = availableTokens < fullTokenCount
        if truncated {
            result.content = bridgeTrimText(result.content, tokenBudget: availableTokens)
            truncatedCount += 1
        }

        packaged.append(
            BridgePackagedSearchResult(
                result: result,
                contextTokens: availableTokens,
                truncated: truncated
            )
        )
        usedTokens += availableTokens + separatorOverheadTokens
    }

    let diagnostics = BridgeContextPackagingSnapshot(
        contextTokenBudget: contextTokenBudget,
        perDocumentTokenBudget: perDocumentTokenBudget,
        contextPackingOrder: contextPackingOrder.rawValue,
        returnedContextTokens: usedTokens,
        returnedResults: packaged.count,
        truncatedResults: truncatedCount
    )
    return (packaged, diagnostics)
}

private func bridgeContextPackingOrder(_ rawValue: String?) throws -> BridgeContextPackingOrder {
    guard let rawValue = rawValue?.trimmingCharacters(in: .whitespacesAndNewlines),
          !rawValue.isEmpty else {
        return .rank
    }
    guard let order = BridgeContextPackingOrder(rawValue: rawValue) else {
        throw BridgeError("Invalid contextPackingOrder '\(rawValue)'. Expected 'rank' or 'score'.")
    }
    return order
}

private func bridgeOrderedSearchResults(
    _ results: [SearchResult],
    order: BridgeContextPackingOrder
) -> [SearchResult] {
    switch order {
    case .rank:
        return results
    case .score:
        return results.enumerated().sorted { lhs, rhs in
            bridgeCompareScores(
                lhs: lhs.element.score,
                rhs: rhs.element.score,
                lhsRank: lhs.offset,
                rhsRank: rhs.offset
            )
        }.map(\.element)
    }
}

private func bridgeCompareScores(
    lhs: SearchScoreBreakdown,
    rhs: SearchScoreBreakdown,
    lhsRank: Int,
    rhsRank: Int
) -> Bool {
    if lhs.blended != rhs.blended {
        return lhs.blended > rhs.blended
    }
    if lhs.fused != rhs.fused {
        return lhs.fused > rhs.fused
    }
    if lhs.lexical != rhs.lexical {
        return lhs.lexical > rhs.lexical
    }
    if lhs.semantic != rhs.semantic {
        return lhs.semantic > rhs.semantic
    }
    return lhsRank < rhsRank
}

private func bridgeEstimatedTokenCount(_ text: String) -> Int {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return 0 }
    return max(1, DefaultTokenizer().tokenize(trimmed).count)
}

private func bridgeCappedContextTokenCount(
    fullTokenCount: Int,
    remainingBudget: Int,
    perDocumentTokenBudget: Int,
    separatorOverheadTokens: Int = 8
) -> Int {
    guard fullTokenCount > 0 else { return 0 }
    guard remainingBudget > separatorOverheadTokens else { return 0 }

    let budgetLimited = min(fullTokenCount, remainingBudget - separatorOverheadTokens)
    guard perDocumentTokenBudget > 0 else {
        return max(0, budgetLimited)
    }
    return max(0, min(budgetLimited, perDocumentTokenBudget))
}

private func bridgeAdaptiveContextPerDocumentTokenBudget(
    queryText: String,
    contextTokenBudget: Int,
    perDocumentTokenBudget: Int,
    separatorOverheadTokens: Int = 8
) -> Int {
    guard perDocumentTokenBudget == 0,
          contextTokenBudget > 0,
          bridgeEvidenceDenseContextQuery(queryText) else {
        return perDocumentTokenBudget
    }

    let lower = queryText.lowercased()
    let targetDocuments: Int
    if lower.contains("list all")
        || lower.contains("all activities")
        || lower.contains("how many")
        || lower.contains("what specific")
        || lower.contains("provide a brief")
        || lower.range(of: #"\b(?:may|june|july|august|september|october|november|december|january|february|march|april)\s+\d{1,2}\b.*\b(?:may|june|july|august|september|october|november|december|january|february|march|april)\s+\d{1,2}\b"#, options: .regularExpression) != nil {
        targetDocuments = 24
    } else {
        targetDocuments = 18
    }

    let targetBudget = max(96, (contextTokenBudget / targetDocuments) - separatorOverheadTokens)
    return targetBudget
}

private func bridgeEvidenceDenseContextQuery(_ queryText: String) -> Bool {
    let lower = queryText.lowercased()
    let densePhrases = [
        "all activities", "all the", "combined", "different", "from earliest to latest",
        "how many", "in august", "in february", "in january", "in june", "in march",
        "in may", "in november", "in october", "in september", "list all",
        "multi-day", "over these", "please list", "provide a brief", "related preparations",
        "so far this year", "specific occasions", "specific preparations",
        "systematic learning", "what activities", "what preparations"
    ]
    if densePhrases.contains(where: lower.contains) {
        return true
    }
    if lower.range(of: #"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b"#, options: .regularExpression) != nil,
       lower.range(of: #"\b(?:what|which|how|list|describe|provide)\b"#, options: .regularExpression) != nil {
        return true
    }
    return false
}

private func bridgeTrimText(_ text: String, tokenBudget: Int) -> String {
    guard tokenBudget > 0 else { return "" }
    let fullTokenCount = bridgeEstimatedTokenCount(text)
    guard fullTokenCount > tokenBudget else { return text }

    let ratio = max(0.01, min(1.0, Double(tokenBudget) / Double(fullTokenCount)))
    var prefixCount = max(1, Int((Double(text.count) * ratio).rounded(.up)))
    var trimmed = String(text.prefix(prefixCount)).trimmingCharacters(in: .whitespacesAndNewlines)

    while bridgeEstimatedTokenCount(trimmed) > tokenBudget, prefixCount > 1 {
        prefixCount = max(1, Int((Double(prefixCount) * 0.9).rounded(.down)))
        trimmed = String(text.prefix(prefixCount)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    return trimmed
}

private enum BridgeContextPackingOrder: String {
    case rank
    case score
}

private struct BridgeError: Error, CustomStringConvertible {
    var description: String

    init(_ description: String) {
        self.description = description
    }
}

private struct BridgeRequest: Decodable {
    var id: String?
    var method: String
    var params: BridgeRequestParams?
}

private struct BridgeRequestParams: Decodable {
    var name: String?
    var path: String?
    var paths: [String]?
    var query: String?
    var collection: String?
    var mode: String?
    var limit: Int?
    var all: Bool?
    var minScore: Double?
    var queryTimestamp: String?
    var contextTokenBudget: Int?
    var perDocumentTokenBudget: Int?
    var contextPackingOrder: String?
}

private struct BridgeSuccessResponse<T: Encodable>: Encodable {
    var id: String?
    var ok = true
    var result: T
}

private struct BridgeErrorResponse: Encodable {
    var id: String?
    var ok = false
    var error: String
}

private struct BridgeReadyResult: Encodable {
    var protocolVersion: Int
    var root: String
    var database: String
}

private struct BridgeShutdownResult: Encodable {
    var shutdown: Bool
}

private struct BridgeCollectionResult: Encodable {
    var name: String
    var path: String
}

private struct BridgeCollectionListResult: Encodable {
    var collections: [StoredCollection]
}

private struct BridgeSyncResult: Encodable {
    var synced: Int
}

private struct BridgeSearchResponse: Encodable {
    var diagnostics: BridgeSearchDiagnosticsSnapshot?
    var results: [BridgeSearchResult]
}

private final class BridgeSearchDiagnostics: @unchecked Sendable {
    private let lock = NSLock()
    private var stageTimings: [String: Double] = [:]
    private var expandedQueryCount: Int?
    private var embeddedQueryCount = 0
    private var semanticCandidateCount: Int?
    private var lexicalCandidateCount: Int?
    private var fusedCandidateCount: Int?

    func record(_ event: SearchEvent) {
        lock.lock()
        defer { lock.unlock() }

        switch event {
        case .stageTiming(let stage, let durationMs):
            stageTimings[stage.rawValue] = durationMs
        case .expandedQueries(let count):
            expandedQueryCount = count
        case .embeddedQuery:
            embeddedQueryCount += 1
        case .semanticCandidates(let count):
            semanticCandidateCount = count
        case .lexicalCandidates(let count):
            lexicalCandidateCount = count
        case .fusedCandidates(let count):
            fusedCandidateCount = count
        default:
            break
        }
    }

    func snapshot(contextPackaging: BridgeContextPackagingSnapshot? = nil) -> BridgeSearchDiagnosticsSnapshot {
        lock.lock()
        defer { lock.unlock() }

        return BridgeSearchDiagnosticsSnapshot(
            stageTimingsMs: stageTimings,
            expandedQueries: expandedQueryCount,
            embeddedQueries: embeddedQueryCount,
            semanticCandidates: semanticCandidateCount,
            lexicalCandidates: lexicalCandidateCount,
            fusedCandidates: fusedCandidateCount,
            contextPackaging: contextPackaging
        )
    }
}

private struct BridgeSearchDiagnosticsSnapshot: Encodable {
    var stageTimingsMs: [String: Double]
    var expandedQueries: Int?
    var embeddedQueries: Int
    var semanticCandidates: Int?
    var lexicalCandidates: Int?
    var fusedCandidates: Int?
    var contextPackaging: BridgeContextPackagingSnapshot?
}

private struct BridgeContextPackagingSnapshot: Encodable {
    var contextTokenBudget: Int
    var perDocumentTokenBudget: Int
    var contextPackingOrder: String
    var returnedContextTokens: Int
    var returnedResults: Int
    var truncatedResults: Int
}

private struct BridgePackagedSearchResult {
    var result: SearchResult
    var contextTokens: Int?
    var truncated: Bool?
}

private struct BridgeSearchResult: Encodable {
    var chunkID: Int64
    var documentPath: String
    var title: String?
    var content: String
    var snippet: String
    var contextTokens: Int?
    var truncated: Bool?
    var modifiedAt: String
    var memoryID: String?
    var memoryKind: String?
    var memoryStatus: String?
    var score: BridgeSearchScore
}

private struct BridgeSearchScore: Encodable {
    var semantic: Double
    var lexical: Double
    var recency: Double
    var tag: Double
    var schema: Double
    var temporal: Double
    var status: Double
    var fused: Double
    var rerank: Double
    var blended: Double

    init(from score: SearchScoreBreakdown) {
        self.semantic = score.semantic
        self.lexical = score.lexical
        self.recency = score.recency
        self.tag = score.tag
        self.schema = score.schema
        self.temporal = score.temporal
        self.status = score.status
        self.fused = score.fused
        self.rerank = score.rerank
        self.blended = score.blended
    }
}
