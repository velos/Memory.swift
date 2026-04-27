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
            expansionLimit: mode == .hybrid ? 2 : 0
        )

        var results = try await index.search(searchQuery)
        if let scopedCollection {
            let root = scopedCollection.path
            results = results.filter { $0.documentPath.hasPrefix(root + "/") || $0.documentPath == root }
        }

        let minScore = params?.minScore ?? 0
        if minScore > 0 {
            results = results.filter { $0.score.blended >= minScore }
        }

        return BridgeSearchResponse(
            results: results.map { result in
                BridgeSearchResult(
                    chunkID: result.chunkID,
                    documentPath: result.documentPath,
                    title: result.title,
                    content: result.content,
                    snippet: result.snippet,
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
    var results: [BridgeSearchResult]
}

private struct BridgeSearchResult: Encodable {
    var chunkID: Int64
    var documentPath: String
    var title: String?
    var content: String
    var snippet: String
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
