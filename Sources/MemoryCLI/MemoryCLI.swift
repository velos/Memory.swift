import ArgumentParser
import Foundation
import Memory
import MemoryAppleIntelligence
import MemoryCoreMLEmbedding
import MemoryNaturalLanguage

struct StoredCollection: Codable, Hashable {
    var name: String
    var path: String
}

struct CLIState: Codable {
    var collections: [StoredCollection]
    var contexts: [String: String]

    static let empty = CLIState(collections: [], contexts: [:])
}

struct CLIPaths {
    let rootDirectory: URL
    let stateFileURL: URL
    let indexFileURL: URL

    static func `default`() throws -> CLIPaths {
        let root = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".memory", isDirectory: true)
        return CLIPaths(
            rootDirectory: root,
            stateFileURL: root.appendingPathComponent("state.json"),
            indexFileURL: root.appendingPathComponent("index.sqlite")
        )
    }
}

struct CLIStateStore {
    let paths: CLIPaths
    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    init(paths: CLIPaths) {
        self.paths = paths
    }

    func load() throws -> CLIState {
        try ensureRootExists()
        guard FileManager.default.fileExists(atPath: paths.stateFileURL.path) else {
            return .empty
        }

        let data = try Data(contentsOf: paths.stateFileURL)
        return try decoder.decode(CLIState.self, from: data)
    }

    func save(_ state: CLIState) throws {
        try ensureRootExists()
        let data = try encoder.encode(state)
        try data.write(to: paths.stateFileURL, options: .atomic)
    }

    private func ensureRootExists() throws {
        try FileManager.default.createDirectory(
            at: paths.rootDirectory,
            withIntermediateDirectories: true
        )
    }
}

enum SearchMode {
    case keyword
    case semantic
    case hybrid

    var semanticLimit: Int {
        switch self {
        case .keyword: 25
        case .semantic: 400
        case .hybrid: 300
        }
    }

    var lexicalLimit: Int {
        switch self {
        case .keyword: 400
        case .semantic: 25
        case .hybrid: 300
        }
    }
}

struct CLIContext {
    let paths: CLIPaths
    let store: CLIStateStore
    let state: CLIState

    static func load() throws -> CLIContext {
        let paths = try CLIPaths.default()
        let store = CLIStateStore(paths: paths)
        let state = try store.load()
        return CLIContext(paths: paths, store: store, state: state)
    }

    func makeIndex() throws -> MemoryIndex {
        if let models = resolveDefaultCoreMLModels() {
            let configuration = try MemoryConfiguration.coreMLDefault(
                databaseURL: paths.indexFileURL,
                models: models
            )
            return try MemoryIndex(configuration: configuration)
        }

        let configuration = MemoryConfiguration.naturalLanguageDefault(databaseURL: paths.indexFileURL)
        return try MemoryIndex(configuration: configuration)
    }

    func requireCollection(named name: String) throws -> StoredCollection {
        guard let collection = state.collections.first(where: { $0.name == name }) else {
            throw ValidationError("Unknown collection '\(name)'.")
        }
        return collection
    }
}

private enum CLIDefaultCoreMLModels {
    static let embedding = "embedding-v1"
}

private func resolveDefaultCoreMLModels() -> CoreMLDefaultModels? {
    guard
        let embedding = resolveDefaultModelURL(
            environmentKey: "MEMORY_EMBEDDING_MODEL_URL",
            modelName: CLIDefaultCoreMLModels.embedding
        )
    else {
        return nil
    }

    return CoreMLDefaultModels(embedding: embedding)
}

private func resolveDefaultModelURL(environmentKey: String, modelName: String) -> URL? {
    let fileManager = FileManager.default
    let environment = ProcessInfo.processInfo.environment
    if let raw = environment[environmentKey]?.trimmingCharacters(in: .whitespacesAndNewlines),
       !raw.isEmpty,
       fileManager.fileExists(atPath: raw) {
        return URL(fileURLWithPath: raw)
    }

    let filename = "\(modelName).mlpackage"
    let cwd = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
    let candidates = [
        cwd.appendingPathComponent("Models").appendingPathComponent(filename),
        cwd.deletingLastPathComponent().appendingPathComponent("Models").appendingPathComponent(filename),
    ]
    return candidates.first(where: { fileManager.fileExists(atPath: $0.path) })
}

@main
struct MemoryCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "memory",
        abstract: "Memory CLI for testing Memory.",
        subcommands: [
            CollectionCommand.self,
            ContextCommand.self,
            EmbedCommand.self,
            SearchCommand.self,
            VSearchCommand.self,
            QueryCommand.self,
            GetCommand.self,
            MultiGetCommand.self,
        ],
        defaultSubcommand: SearchCommand.self
    )
}

struct CollectionCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "collection",
        abstract: "Manage indexed collections.",
        subcommands: [CollectionAddCommand.self, CollectionListCommand.self, CollectionRemoveCommand.self]
    )
}

struct CollectionAddCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "add")

    @Argument(help: "Path to collection root.")
    var path: String

    @Option(name: .long, help: "Collection name (used as memory://<name>).")
    var name: String

    mutating func run() async throws {
        let context = try CLIContext.load()
        var state = context.state

        let normalizedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedName.isEmpty else {
            throw ValidationError("Collection name cannot be empty.")
        }

        let expandedPath = NSString(string: path).expandingTildeInPath
        let absoluteURL = URL(fileURLWithPath: expandedPath).standardizedFileURL

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: absoluteURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw ValidationError("Collection path does not exist or is not a directory: \(absoluteURL.path)")
        }

        state.collections.removeAll { $0.name == normalizedName }
        state.collections.append(.init(name: normalizedName, path: absoluteURL.path))
        state.collections.sort { $0.name < $1.name }

        try context.store.save(state)
        print("Added collection memory://\(normalizedName) -> \(absoluteURL.path)")
    }
}

struct CollectionListCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "list")

    mutating func run() async throws {
        let context = try CLIContext.load()
        if context.state.collections.isEmpty {
            print("No collections configured.")
            return
        }

        for collection in context.state.collections {
            print("memory://\(collection.name)\t\(collection.path)")
        }
    }
}

struct CollectionRemoveCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "remove")

    @Argument(help: "Collection name or memory:// URI.")
    var collection: String

    mutating func run() async throws {
        let context = try CLIContext.load()
        var state = context.state

        let name = parseCollectionRef(collection) ?? collection
        let initialCount = state.collections.count
        state.collections.removeAll { $0.name == name }
        state.contexts.removeValue(forKey: name)

        guard state.collections.count != initialCount else {
            throw ValidationError("Collection '\(name)' not found.")
        }

        try context.store.save(state)
        print("Removed collection memory://\(name)")
    }
}

struct ContextCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "context",
        abstract: "Manage collection context hints.",
        subcommands: [ContextAddCommand.self, ContextListCommand.self, ContextRemoveCommand.self]
    )
}

struct ContextAddCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "add")

    @Argument(help: "Collection URI in memory://<name> format.")
    var collectionURI: String

    @Argument(help: "Context text for the collection.")
    var description: String

    mutating func run() async throws {
        let context = try CLIContext.load()
        var state = context.state

        guard let name = parseCollectionRef(collectionURI) else {
            throw ValidationError("Collection URI must use memory://<name> format.")
        }

        guard state.collections.contains(where: { $0.name == name }) else {
            throw ValidationError("Collection memory://\(name) does not exist.")
        }

        state.contexts[name] = description
        try context.store.save(state)

        print("Added context for memory://\(name)")
    }
}

struct ContextListCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "list")

    mutating func run() async throws {
        let context = try CLIContext.load()
        if context.state.contexts.isEmpty {
            print("No context configured.")
            return
        }

        for key in context.state.contexts.keys.sorted() {
            let value = context.state.contexts[key] ?? ""
            print("memory://\(key)\t\(value)")
        }
    }
}

struct ContextRemoveCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "remove")

    @Argument(help: "Collection URI in memory://<name> format.")
    var collectionURI: String

    mutating func run() async throws {
        let context = try CLIContext.load()
        var state = context.state

        guard let name = parseCollectionRef(collectionURI) else {
            throw ValidationError("Collection URI must use memory://<name> format.")
        }

        guard state.contexts.removeValue(forKey: name) != nil else {
            throw ValidationError("No context for memory://\(name)")
        }

        try context.store.save(state)
        print("Removed context for memory://\(name)")
    }
}

struct EmbedCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "embed")

    mutating func run() async throws {
        let context = try CLIContext.load()

        guard !context.state.collections.isEmpty else {
            throw ValidationError("No collections configured. Run 'memory collection add <path> --name <name>' first.")
        }

        let roots = context.state.collections.map { URL(fileURLWithPath: $0.path) }
        let index = try context.makeIndex()

        print("Embedding \(roots.count) collections...")
        try await index.rebuildIndex(
            from: .init(roots: roots),
            events: { event in
                switch event {
                case let .started(totalDocuments):
                    print("Found \(totalDocuments) documents")
                case let .readingDocument(path, index, total):
                    print("[\(index)/\(total)] \(path)")
                case let .completed(processedDocuments, totalChunks):
                    print("Done. Processed \(processedDocuments) documents and \(totalChunks) chunks.")
                default:
                    break
                }
            }
        )
    }
}

struct SearchCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "search")

    @Argument(help: "Search query.")
    var text: String

    @Option(name: [.short, .long], help: "Collection name to scope to.")
    var collection: String?

    @Flag(name: .long, help: "Return a large result set.")
    var all = false

    @Flag(name: .long, help: "Output only file paths.")
    var files = false

    @Option(name: .long, help: "Minimum fused score threshold.")
    var minScore: Double = 0

    mutating func run() async throws {
        try await runSearch(
            mode: .keyword,
            text: text,
            collection: collection,
            all: all,
            files: files,
            minScore: minScore
        )
    }
}

struct VSearchCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "vsearch")

    @Argument(help: "Semantic search query.")
    var text: String

    @Option(name: [.short, .long], help: "Collection name to scope to.")
    var collection: String?

    @Flag(name: .long, help: "Return a large result set.")
    var all = false

    @Flag(name: .long, help: "Output only file paths.")
    var files = false

    @Option(name: .long, help: "Minimum fused score threshold.")
    var minScore: Double = 0

    mutating func run() async throws {
        try await runSearch(
            mode: .semantic,
            text: text,
            collection: collection,
            all: all,
            files: files,
            minScore: minScore
        )
    }
}

struct QueryCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "query")

    @Argument(help: "Hybrid query.")
    var text: String

    @Option(name: [.short, .long], help: "Collection name to scope to.")
    var collection: String?

    @Flag(name: .long, help: "Return a large result set.")
    var all = false

    @Flag(name: .long, help: "Output only file paths.")
    var files = false

    @Option(name: .long, help: "Minimum fused score threshold.")
    var minScore: Double = 0

    mutating func run() async throws {
        try await runSearch(
            mode: .hybrid,
            text: text,
            collection: collection,
            all: all,
            files: files,
            minScore: minScore
        )
    }
}

struct GetCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "get")

    @Argument(help: "File path, relative collection path, or #<docid>.")
    var target: String

    mutating func run() async throws {
        let context = try CLIContext.load()

        if target.hasPrefix("#") {
            let idString = String(target.dropFirst())
            guard let chunkID = Int64(idString, radix: 16) ?? Int64(idString) else {
                throw ValidationError("Invalid docid '\(target)'. Expected #<hex-id>.")
            }

            let index = try context.makeIndex()
            guard let chunk = try await index.getChunk(id: chunkID) else {
                throw ValidationError("No chunk found for docid \(target)")
            }

            print("#\(String(chunk.chunkID, radix: 16))\t\(chunk.documentPath)")
            print(chunk.content)
            return
        }

        let resolved = try await resolveDocumentTarget(target, context: context)
        let content = try String(contentsOf: resolved, encoding: .utf8)
        print(content)
    }
}

struct MultiGetCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "multi-get")

    @Argument(help: "Glob pattern, e.g. journals/2025-05*.md")
    var pattern: String

    @Option(name: [.short, .long], help: "Collection name to scope to.")
    var collection: String?

    mutating func run() async throws {
        let context = try CLIContext.load()

        guard !context.state.collections.isEmpty else {
            throw ValidationError("No collections configured.")
        }

        let collections: [StoredCollection]
        if let collection {
            collections = [try context.requireCollection(named: normalizeCollectionArgument(collection))]
        } else {
            collections = context.state.collections
        }

        let matcher = try GlobMatcher(pattern: pattern)
        var matchedFiles: [String] = []

        for collection in collections {
            let root = URL(fileURLWithPath: collection.path)
            for file in enumerateFiles(in: root) {
                let relativePath = file.path.replacingOccurrences(of: root.path + "/", with: "")
                if matcher.matches(relativePath) {
                    matchedFiles.append(file.path)
                }
            }
        }

        matchedFiles.sort()

        if matchedFiles.isEmpty {
            print("No files matched pattern '\(pattern)'.")
            return
        }

        for path in matchedFiles {
            let content = try String(contentsOfFile: path, encoding: .utf8)
            print("--- \(path) ---")
            print(content)
        }
    }
}

private func runSearch(
    mode: SearchMode,
    text: String,
    collection: String?,
    all: Bool,
    files: Bool,
    minScore: Double
) async throws {
    let context = try CLIContext.load()

    guard !context.state.collections.isEmpty else {
        throw ValidationError("No collections configured. Run 'memory collection add' first.")
    }

    let scopedCollection: StoredCollection?
    if let collection {
        scopedCollection = try context.requireCollection(named: normalizeCollectionArgument(collection))
    } else {
        scopedCollection = nil
    }

    var queryText = text
    if let scopedCollection, let hint = context.state.contexts[scopedCollection.name], !hint.isEmpty {
        queryText += "\n\nContext: \(hint)"
    }

    let index = try context.makeIndex()
    let resultLimit = all ? 2_000 : 20
    let rerankLimit = mode == .hybrid ? 50 : 0
    let expansionLimit = mode == .hybrid ? 2 : 0
    let query = SearchQuery(
        text: queryText,
        limit: resultLimit,
        semanticCandidateLimit: mode.semanticLimit,
        lexicalCandidateLimit: mode.lexicalLimit,
        rerankLimit: rerankLimit,
        expansionLimit: expansionLimit
    )

    var results = try await index.search(query)

    if let scopedCollection {
        let root = scopedCollection.path
        results = results.filter { $0.documentPath.hasPrefix(root + "/") || $0.documentPath == root }
    }

    results = results.filter { $0.score.blended >= minScore }

    if files {
        var seen: Set<String> = []
        for result in results where seen.insert(result.documentPath).inserted {
            print(result.documentPath)
        }
        return
    }

    if results.isEmpty {
        print("No matches.")
        return
    }

    for result in results {
        let docID = "#" + String(result.chunkID, radix: 16)
        print(
            "\(docID)\t[score=\(String(format: "%.3f", result.score.blended))]\t\(result.documentPath)"
        )
        print(result.snippet)
        print("")
    }
}

private func parseCollectionRef(_ value: String) -> String? {
    if value.hasPrefix("memory://") {
        let raw = value.dropFirst("memory://".count)
        return raw.isEmpty ? nil : String(raw)
    }

    return nil
}

private func normalizeCollectionArgument(_ value: String) -> String {
    parseCollectionRef(value) ?? value
}

private func resolveDocumentTarget(_ target: String, context: CLIContext) async throws -> URL {
    let expanded = NSString(string: target).expandingTildeInPath
    let fm = FileManager.default

    let absolute = URL(fileURLWithPath: expanded)
    if fm.fileExists(atPath: absolute.path) {
        return absolute
    }

    for collection in context.state.collections {
        let root = URL(fileURLWithPath: collection.path)
        let direct = root.appendingPathComponent(target)
        if fm.fileExists(atPath: direct.path) {
            return direct
        }

        if target.hasPrefix(collection.name + "/") {
            let stripped = String(target.dropFirst(collection.name.count + 1))
            let candidate = root.appendingPathComponent(stripped)
            if fm.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
    }

    let index = try context.makeIndex()
    let indexedPaths = try await index.listIndexedDocumentPaths()
    if let match = indexedPaths.first(where: { $0.hasSuffix("/" + target) || $0 == target }) {
        return URL(fileURLWithPath: match)
    }

    throw ValidationError("Document not found: \(target)")
}

private func enumerateFiles(in root: URL) -> [URL] {
    guard let enumerator = FileManager.default.enumerator(
        at: root,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
    ) else {
        return []
    }

    var files: [URL] = []
    for case let url as URL in enumerator {
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
        if values?.isRegularFile == true {
            files.append(url)
        }
    }
    return files
}

struct GlobMatcher {
    private let regex: NSRegularExpression

    init(pattern: String) throws {
        let regexPattern = "^" + Self.makeRegexPattern(from: pattern) + "$"
        do {
            self.regex = try NSRegularExpression(pattern: regexPattern)
        } catch {
            throw ValidationError("Invalid glob pattern '\(pattern)'")
        }
    }

    func matches(_ candidate: String) -> Bool {
        let range = NSRange(location: 0, length: candidate.utf16.count)
        return regex.firstMatch(in: candidate, options: [], range: range) != nil
    }

    private static func makeRegexPattern(from glob: String) -> String {
        var result = ""
        for char in glob {
            switch char {
            case "*":
                result += ".*"
            case "?":
                result += "."
            case ".", "\\", "+", "(", ")", "[", "]", "{", "}", "^", "$", "|":
                result += "\\" + String(char)
            default:
                result += String(char)
            }
        }
        return result
    }
}
