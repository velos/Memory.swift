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

        #expect(version == 2)
        #expect(tableNames.contains("memory_schema_metadata"))
        #expect(tableNames.contains("documents"))
        #expect(tableNames.contains("chunks"))
        #expect(tableNames.contains("memories"))
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
        #expect(version == 2)
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

    @Test
    func v1SyntheticMemoryDocumentsMigrateIntoCanonicalMemories() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("v1-memory.sqlite")

        try createVersion1DatabaseWithSyntheticMemory(at: dbURL)

        let storage = try MemoryStorage(databaseURL: dbURL)
        let migrated = try #require(await storage.fetchStoredMemory(id: "legacy-decision"))
        let listed = try await storage.listStoredMemories(limit: 10, sort: .recent, kinds: nil, statuses: nil)
        let lexicalHits = try await storage.lexicalSearch(query: "sqlite", limit: 5)

        let database = try SQLiteDatabase(path: dbURL.path)
        let version = try database.fetchOne(
            sql: "SELECT version FROM memory_schema_metadata LIMIT 1",
            as: Int.self
        )
        let documentRow = try #require(
            try database.fetchOne(
                sql: """
                SELECT memory_id, memory_kind, memory_status, memory_canonical_key
                FROM documents
                WHERE path = ?
                """,
                arguments: ["memory://legacy-decision"]
            )
        )

        let migratedChunkID = try #require(migrated.chunkID)
        let documentMemoryID: String = documentRow["memory_id"]
        let documentMemoryKind: String = documentRow["memory_kind"]
        let documentMemoryStatus: String = documentRow["memory_status"]
        let documentCanonicalKey: String = documentRow["memory_canonical_key"]

        #expect(version == 2)
        #expect(migrated.id == "legacy-decision")
        #expect(migrated.kind == "decision")
        #expect(migrated.status == "active")
        #expect(migrated.documentPath == "memory://legacy-decision")
        #expect(migrated.canonicalKey == documentCanonicalKey)
        #expect(listed.count == 1)
        #expect(listed.first?.id == migrated.id)
        #expect(lexicalHits.contains(where: { $0.chunkID == migratedChunkID }))
        #expect(documentMemoryID == migrated.id)
        #expect(documentMemoryKind == "decision")
        #expect(documentMemoryStatus == "active")
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

    private func createVersion1DatabaseWithSyntheticMemory(at url: URL) throws {
        let database = try SQLiteDatabase(path: url.path)
        let now = Date(timeIntervalSince1970: 1_700_000_100).timeIntervalSince1970

        try database.execute(sql: "PRAGMA foreign_keys = ON")
        try database.transaction {
            try database.execute(
                sql: """
                CREATE TABLE memory_schema_metadata (
                    version INTEGER NOT NULL
                )
                """
            )
            try database.execute(
                sql: """
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE ON CONFLICT REPLACE,
                    title TEXT,
                    modified_at REAL NOT NULL,
                    checksum TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'factual',
                    memory_type_source TEXT NOT NULL DEFAULT 'fallback',
                    memory_type_confidence REAL,
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
                    memory_type_override TEXT,
                    memory_type_override_source TEXT,
                    memory_type_override_confidence REAL,
                    memory_category TEXT,
                    importance REAL NOT NULL DEFAULT 0.5,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at REAL,
                    source TEXT NOT NULL DEFAULT 'index',
                    content_tags_json TEXT NOT NULL DEFAULT '[]',
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
                CREATE TABLE vector_index_config (
                    id INTEGER NOT NULL PRIMARY KEY,
                    embedding_dim INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            try database.execute(
                sql: """
                CREATE TABLE contexts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE ON CONFLICT ABORT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            try database.execute(
                sql: """
                CREATE TABLE context_chunks (
                    context_id TEXT NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    added_at REAL NOT NULL,
                    PRIMARY KEY (context_id, chunk_id) ON CONFLICT REPLACE
                )
                """
            )
            try database.execute(sql: "CREATE INDEX context_chunks_context_chunk ON context_chunks(context_id, chunk_id)")
            try database.execute(
                sql: """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    content,
                    content = 'chunks',
                    content_rowid = 'id'
                )
                """
            )
            try database.execute(
                sql: """
                CREATE TRIGGER chunks_fts_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content)
                    VALUES (new.id, new.content);
                END
                """
            )
            try database.execute(
                sql: """
                CREATE TRIGGER chunks_fts_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                END
                """
            )
            try database.execute(
                sql: """
                CREATE TRIGGER chunks_fts_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                    INSERT INTO chunks_fts(rowid, content)
                    VALUES (new.id, new.content);
                END
                """
            )
            try database.execute(sql: "CREATE INDEX documents_memory_type ON documents(memory_type)")
            try database.execute(sql: "CREATE INDEX chunks_memory_type_override ON chunks(memory_type_override)")
            try database.execute(sql: "CREATE INDEX chunks_importance ON chunks(importance)")
            try database.execute(sql: "CREATE INDEX chunks_access_count ON chunks(access_count)")
            try database.execute(sql: "CREATE INDEX chunks_created_at ON chunks(created_at)")

            try database.execute(
                sql: "INSERT INTO memory_schema_metadata(version) VALUES (?)",
                arguments: [1]
            )
            try database.execute(
                sql: """
                INSERT INTO vector_index_config (id, embedding_dim, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                arguments: [1, 3, now, now]
            )
            try database.execute(
                sql: """
                INSERT INTO documents (
                    id, path, title, modified_at, checksum, memory_type, memory_type_source,
                    memory_type_confidence, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                arguments: [
                    1,
                    "memory://legacy-decision",
                    "Legacy Decision",
                    now,
                    "legacy-checksum",
                    "semantic",
                    "manual",
                    0.98,
                    now,
                    now,
                ]
            )
            try database.execute(
                sql: """
                INSERT INTO chunks (
                    id, document_id, ordinal, content, token_count, memory_type_override,
                    memory_type_override_source, memory_type_override_confidence, memory_category,
                    importance, access_count, last_accessed_at, source, content_tags_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                arguments: [
                    1,
                    1,
                    0,
                    "We switched the prototype to SQLite for the on-device store.",
                    10,
                    "semantic",
                    "manual",
                    0.98,
                    "decision",
                    0.9,
                    3,
                    now,
                    "legacy_import",
                    "[\"sqlite\",\"prototype\"]",
                    now,
                ]
            )
            try database.execute(
                sql: """
                INSERT INTO embeddings (chunk_id, dim, vector_blob, norm)
                VALUES (?, ?, ?, ?)
                """,
                arguments: [1, 3, encode([1, 2, 3]), 3.7416573868]
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
