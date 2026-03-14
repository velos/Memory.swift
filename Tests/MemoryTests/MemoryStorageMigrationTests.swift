import Foundation
import MemoryStorage
import SQLiteSupport
import Testing

struct MemoryStorageMigrationTests {
    @Test
    func freshDatabaseBootstrapsCurrentSchema() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("fresh.sqlite")

        _ = try MemoryStorage(databaseURL: dbURL)

        let database = try SQLiteDatabase(path: dbURL.path)
        let version = try database.fetchOne(
            sql: "SELECT version FROM memory_schema_metadata LIMIT 1",
            as: Int.self
        )
        let tableNames = Set(
            try database.fetchAll(
                sql: "SELECT name FROM sqlite_master WHERE type = 'table'",
                as: String.self
            )
        )

        #expect(version == 1)
        #expect(tableNames.contains("memory_schema_metadata"))
        #expect(tableNames.contains("documents"))
        #expect(tableNames.contains("chunks"))
        #expect(tableNames.contains("embeddings"))
        #expect(tableNames.contains("contexts"))
        #expect(tableNames.contains("context_chunks"))
        #expect(tableNames.contains("chunks_fts"))
        #expect(tableNames.contains("vector_index_config"))
    }

    @Test
    func legacyDatabaseIsResetOnOpen() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("legacy.sqlite")

        try createLegacyGRDBBackedDatabase(at: dbURL)

        let storage = try MemoryStorage(databaseURL: dbURL)
        let row = try await storage.fetchChunkMetadata(chunkID: 1)
        let paths = try await storage.listDocumentPaths()

        let database = try SQLiteDatabase(path: dbURL.path)
        let version = try database.fetchOne(
            sql: "SELECT version FROM memory_schema_metadata LIMIT 1",
            as: Int.self
        )
        let legacyTable = try database.fetchOne(
            sql: "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'grdb_migrations'",
            as: String.self
        )
        let documentCount = try database.fetchOne(
            sql: "SELECT COUNT(*) FROM documents",
            as: Int.self
        )

        #expect(row == nil)
        #expect(paths.isEmpty)
        #expect(version == 1)
        #expect(legacyTable == nil)
        #expect(documentCount == 0)
    }

    @Test
    func reopeningFreshDatabaseKeepsIndexedData() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("reopen.sqlite")

        let storage = try MemoryStorage(databaseURL: dbURL)
        try await storage.replaceDocument(
            makeStoredDocument(
                path: "/tmp/reopen.md",
                content: "alpha release checklist"
            )
        )

        let reopened = try MemoryStorage(databaseURL: dbURL)
        let paths = try await reopened.listDocumentPaths()
        let hits = try await reopened.lexicalSearch(query: "alpha", limit: 5)

        #expect(paths == ["/tmp/reopen.md"])
        #expect(hits.count == 1)
    }

    @Test
    func ftsTriggersTrackReplacementAndDeletion() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("fts.sqlite")

        let storage = try MemoryStorage(databaseURL: dbURL)
        try await storage.replaceDocument(
            makeStoredDocument(
                path: "/tmp/fts.md",
                content: "alpha planning note"
            )
        )

        let initialAlphaHits = try await storage.lexicalSearch(query: "alpha", limit: 5)
        #expect(initialAlphaHits.count == 1)

        try await storage.replaceDocument(
            makeStoredDocument(
                path: "/tmp/fts.md",
                content: "beta planning note"
            )
        )

        let alphaHitsAfterReplace = try await storage.lexicalSearch(query: "alpha", limit: 5)
        let betaHitsAfterReplace = try await storage.lexicalSearch(query: "beta", limit: 5)

        #expect(alphaHitsAfterReplace.isEmpty)
        #expect(betaHitsAfterReplace.count == 1)

        try await storage.removeDocuments(paths: ["/tmp/fts.md"])
        let betaHitsAfterDelete = try await storage.lexicalSearch(query: "beta", limit: 5)

        #expect(betaHitsAfterDelete.isEmpty)
    }

    private func createLegacyGRDBBackedDatabase(at url: URL) throws {
        let database = try SQLiteDatabase(path: url.path)
        let now = Date().timeIntervalSince1970

        try database.execute(sql: "PRAGMA foreign_keys = ON")
        try database.transaction {
            try database.execute(
                sql: """
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE ON CONFLICT REPLACE,
                    title TEXT,
                    modified_at REAL NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            try database.execute(
                sql: """
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            try database.execute(sql: "CREATE INDEX chunks_document_ordinal ON chunks(document_id, ordinal)")
            try database.execute(
                sql: """
                CREATE TABLE embeddings (
                    chunk_id INTEGER PRIMARY KEY ON CONFLICT REPLACE REFERENCES chunks(id) ON DELETE CASCADE,
                    dim INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    norm REAL NOT NULL
                )
                """
            )
            try database.execute(
                sql: """
                CREATE TABLE grdb_migrations (
                    identifier TEXT PRIMARY KEY
                )
                """
            )
            try database.execute(
                sql: """
                INSERT INTO documents (id, path, title, modified_at, checksum, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                arguments: [1, "/tmp/legacy.md", "Legacy", now, "abc123", now, now]
            )
            try database.execute(
                sql: """
                INSERT INTO chunks (id, document_id, ordinal, content, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                arguments: [1, 1, 0, "legacy chunk content", 3, now]
            )
            try database.execute(
                sql: """
                INSERT INTO embeddings (chunk_id, dim, vector_blob, norm)
                VALUES (?, ?, ?, ?)
                """,
                arguments: [1, 3, encode([1, 2, 3]), 3.7416573868]
            )
            try database.execute(
                sql: "INSERT INTO grdb_migrations(identifier) VALUES (?)",
                arguments: ["v1_initial"]
            )
        }
    }

    private func makeStoredDocument(path: String, content: String) -> StoredDocumentInput {
        StoredDocumentInput(
            path: path,
            title: "Test Document",
            modifiedAt: Date(timeIntervalSince1970: 1_700_000_000),
            checksum: UUID().uuidString,
            chunks: [
                StoredChunkInput(
                    ordinal: 0,
                    content: content,
                    tokenCount: content.split(separator: " ").count,
                    embedding: [1, 2, 3],
                    norm: 3.7416573868
                )
            ]
        )
    }

    private func encode(_ vector: [Float]) -> Data {
        let copy = vector
        return copy.withUnsafeBytes { Data($0) }
    }
}
