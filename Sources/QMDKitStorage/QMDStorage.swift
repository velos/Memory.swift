import Foundation
import GRDB

public struct StoredChunkInput: Sendable {
    public var ordinal: Int
    public var content: String
    public var tokenCount: Int
    public var embedding: [Float]
    public var norm: Double

    public init(
        ordinal: Int,
        content: String,
        tokenCount: Int,
        embedding: [Float],
        norm: Double
    ) {
        self.ordinal = ordinal
        self.content = content
        self.tokenCount = tokenCount
        self.embedding = embedding
        self.norm = norm
    }
}

public struct StoredDocumentInput: Sendable {
    public var path: String
    public var title: String?
    public var modifiedAt: Date
    public var checksum: String
    public var chunks: [StoredChunkInput]

    public init(
        path: String,
        title: String?,
        modifiedAt: Date,
        checksum: String,
        chunks: [StoredChunkInput]
    ) {
        self.path = path
        self.title = title
        self.modifiedAt = modifiedAt
        self.checksum = checksum
        self.chunks = chunks
    }
}

public struct StoredChunkEmbedding: Sendable {
    public var chunkID: Int64
    public var vector: [Float]
    public var norm: Double
    public var content: String
    public var documentPath: String
    public var title: String?
    public var modifiedAt: Date

    public init(
        chunkID: Int64,
        vector: [Float],
        norm: Double,
        content: String,
        documentPath: String,
        title: String?,
        modifiedAt: Date
    ) {
        self.chunkID = chunkID
        self.vector = vector
        self.norm = norm
        self.content = content
        self.documentPath = documentPath
        self.title = title
        self.modifiedAt = modifiedAt
    }
}

public struct StoredChunkMetadata: Sendable {
    public var chunkID: Int64
    public var content: String
    public var documentPath: String
    public var title: String?
    public var modifiedAt: Date

    public init(chunkID: Int64, content: String, documentPath: String, title: String?, modifiedAt: Date) {
        self.chunkID = chunkID
        self.content = content
        self.documentPath = documentPath
        self.title = title
        self.modifiedAt = modifiedAt
    }
}

public struct LexicalHit: Sendable {
    public var chunkID: Int64
    public var score: Double

    public init(chunkID: Int64, score: Double) {
        self.chunkID = chunkID
        self.score = score
    }
}

public actor QMDStorage {
    private let dbQueue: DatabaseQueue

    public init(databaseURL: URL) throws {
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var configuration = Configuration()
        configuration.prepareDatabase { db in
            try db.execute(sql: "PRAGMA foreign_keys = ON")
        }

        self.dbQueue = try DatabaseQueue(path: databaseURL.path, configuration: configuration)
        try Self.makeMigrator().migrate(dbQueue)
    }

    public func wipeIndexData() throws {
        try dbQueue.write { db in
            try db.execute(sql: "DELETE FROM context_chunks")
            try db.execute(sql: "DELETE FROM embeddings")
            try db.execute(sql: "DELETE FROM chunks")
            try db.execute(sql: "DELETE FROM documents")
        }
    }

    public func replaceDocument(_ input: StoredDocumentInput) throws {
        try dbQueue.write { db in
            try db.execute(sql: "DELETE FROM documents WHERE path = ?", arguments: [input.path])

            let now = Date().timeIntervalSince1970
            try db.execute(
                sql: """
                INSERT INTO documents (path, title, modified_at, checksum, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                arguments: [
                    input.path,
                    input.title,
                    input.modifiedAt.timeIntervalSince1970,
                    input.checksum,
                    now,
                    now,
                ]
            )

            let documentID = db.lastInsertedRowID
            for chunk in input.chunks {
                try db.execute(
                    sql: """
                    INSERT INTO chunks (document_id, ordinal, content, token_count, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    arguments: [
                        documentID,
                        chunk.ordinal,
                        chunk.content,
                        chunk.tokenCount,
                        now,
                    ]
                )

                let chunkID = db.lastInsertedRowID
                try db.execute(
                    sql: """
                    INSERT INTO embeddings (chunk_id, dim, vector_blob, norm)
                    VALUES (?, ?, ?, ?)
                    """,
                    arguments: [
                        chunkID,
                        chunk.embedding.count,
                        Self.encodeVector(chunk.embedding),
                        chunk.norm,
                    ]
                )
            }
        }
    }

    public func removeDocuments(paths: [String]) throws {
        guard !paths.isEmpty else { return }

        try dbQueue.write { db in
            let sql = "DELETE FROM documents WHERE path IN (\(Self.placeholders(count: paths.count)))"
            try db.execute(sql: sql, arguments: StatementArguments(paths))
        }
    }

    public func fetchAllChunkEmbeddings() throws -> [StoredChunkEmbedding] {
        try dbQueue.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: """
                SELECT
                    c.id AS chunk_id,
                    c.content AS content,
                    d.path AS document_path,
                    d.title AS title,
                    d.modified_at AS modified_at,
                    e.vector_blob AS vector_blob,
                    e.norm AS norm
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN embeddings e ON e.chunk_id = c.id
                """
            )

            return rows.compactMap { row in
                guard
                    let data: Data = row["vector_blob"],
                    let vector = Self.decodeVector(data)
                else {
                    return nil
                }

                return StoredChunkEmbedding(
                    chunkID: row["chunk_id"],
                    vector: vector,
                    norm: row["norm"],
                    content: row["content"],
                    documentPath: row["document_path"],
                    title: row["title"],
                    modifiedAt: Date(timeIntervalSince1970: row["modified_at"])
                )
            }
        }
    }

    public func fetchChunkMetadata(chunkIDs: [Int64]) throws -> [StoredChunkMetadata] {
        guard !chunkIDs.isEmpty else { return [] }

        return try dbQueue.read { db in
            let sql = """
            SELECT
                c.id AS chunk_id,
                c.content AS content,
                d.path AS document_path,
                d.title AS title,
                d.modified_at AS modified_at
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN (\(Self.placeholders(count: chunkIDs.count)))
            """

            let rows = try Row.fetchAll(db, sql: sql, arguments: StatementArguments(chunkIDs))
            return rows.map {
                StoredChunkMetadata(
                    chunkID: $0["chunk_id"],
                    content: $0["content"],
                    documentPath: $0["document_path"],
                    title: $0["title"],
                    modifiedAt: Date(timeIntervalSince1970: $0["modified_at"])
                )
            }
        }
    }

    public func fetchChunkMetadata(chunkID: Int64) throws -> StoredChunkMetadata? {
        try dbQueue.read { db in
            let row = try Row.fetchOne(
                db,
                sql: """
                SELECT
                    c.id AS chunk_id,
                    c.content AS content,
                    d.path AS document_path,
                    d.title AS title,
                    d.modified_at AS modified_at
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.id = ?
                """,
                arguments: [chunkID]
            )

            guard let row else { return nil }
            return StoredChunkMetadata(
                chunkID: row["chunk_id"],
                content: row["content"],
                documentPath: row["document_path"],
                title: row["title"],
                modifiedAt: Date(timeIntervalSince1970: row["modified_at"])
            )
        }
    }

    public func listDocumentPaths() throws -> [String] {
        try dbQueue.read { db in
            try String.fetchAll(
                db,
                sql: "SELECT path FROM documents ORDER BY path ASC"
            )
        }
    }

    public func lexicalSearch(
        query: String,
        limit: Int,
        allowedChunkIDs: Set<Int64>? = nil
    ) throws -> [LexicalHit] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let pattern = FTS5Pattern(matchingAnyTokenIn: trimmed) else { return [] }

        return try dbQueue.read { db in
            var arguments = StatementArguments([pattern])
            var sql = """
            SELECT rowid AS chunk_id, rank AS rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            """

            if let allowedChunkIDs, !allowedChunkIDs.isEmpty {
                let orderedAllowed = allowedChunkIDs.sorted()
                sql += " AND rowid IN (\(Self.placeholders(count: orderedAllowed.count)))"
                arguments += StatementArguments(orderedAllowed)
            }

            sql += " ORDER BY rank LIMIT ?"
            arguments += StatementArguments([max(1, limit)])

            let rows = try Row.fetchAll(db, sql: sql, arguments: arguments)
            return rows.enumerated().map { index, row in
                let chunkID: Int64 = row["chunk_id"]
                let rank: Double? = row["rank"]
                let score = -(rank ?? Double(index + 1))
                return LexicalHit(chunkID: chunkID, score: score)
            }
        }
    }

    public func createContext(id: String, name: String) throws -> String {
        try dbQueue.write { db in
            if let existingID = try String.fetchOne(db, sql: "SELECT id FROM contexts WHERE name = ?", arguments: [name]) {
                return existingID
            }

            let now = Date().timeIntervalSince1970
            try db.execute(
                sql: "INSERT INTO contexts (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                arguments: [id, name, now, now]
            )
            return id
        }
    }

    public func addContextChunks(contextID: String, chunkIDs: [Int64]) throws {
        guard !chunkIDs.isEmpty else { return }

        try dbQueue.write { db in
            let now = Date().timeIntervalSince1970
            for chunkID in chunkIDs {
                try db.execute(
                    sql: """
                    INSERT OR REPLACE INTO context_chunks (context_id, chunk_id, added_at)
                    VALUES (?, ?, ?)
                    """,
                    arguments: [contextID, chunkID, now]
                )
            }

            try db.execute(
                sql: "UPDATE contexts SET updated_at = ? WHERE id = ?",
                arguments: [now, contextID]
            )
        }
    }

    public func clearContext(contextID: String) throws {
        try dbQueue.write { db in
            try db.execute(sql: "DELETE FROM context_chunks WHERE context_id = ?", arguments: [contextID])
            try db.execute(
                sql: "UPDATE contexts SET updated_at = ? WHERE id = ?",
                arguments: [Date().timeIntervalSince1970, contextID]
            )
        }
    }

    public func fetchContextChunkIDs(contextID: String) throws -> [Int64] {
        try dbQueue.read { db in
            try Int64.fetchAll(
                db,
                sql: "SELECT chunk_id FROM context_chunks WHERE context_id = ?",
                arguments: [contextID]
            )
        }
    }

    public func listContextChunks(contextID: String) throws -> [StoredChunkMetadata] {
        try dbQueue.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: """
                SELECT
                    c.id AS chunk_id,
                    c.content AS content,
                    d.path AS document_path,
                    d.title AS title,
                    d.modified_at AS modified_at
                FROM context_chunks cc
                JOIN chunks c ON c.id = cc.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE cc.context_id = ?
                ORDER BY cc.added_at DESC
                """,
                arguments: [contextID]
            )

            return rows.map {
                StoredChunkMetadata(
                    chunkID: $0["chunk_id"],
                    content: $0["content"],
                    documentPath: $0["document_path"],
                    title: $0["title"],
                    modifiedAt: Date(timeIntervalSince1970: $0["modified_at"])
                )
            }
        }
    }

    private static func makeMigrator() -> DatabaseMigrator {
        var migrator = DatabaseMigrator()

        migrator.registerMigration("v1_initial") { db in
            try db.create(table: "documents") { table in
                table.autoIncrementedPrimaryKey("id")
                table.column("path", .text).notNull().unique(onConflict: .replace)
                table.column("title", .text)
                table.column("modified_at", .double).notNull()
                table.column("checksum", .text).notNull()
                table.column("created_at", .double).notNull()
                table.column("updated_at", .double).notNull()
            }

            try db.create(table: "chunks") { table in
                table.autoIncrementedPrimaryKey("id")
                table.column("document_id", .integer).notNull()
                    .indexed()
                    .references("documents", onDelete: .cascade)
                table.column("ordinal", .integer).notNull()
                table.column("content", .text).notNull()
                table.column("token_count", .integer).notNull()
                table.column("created_at", .double).notNull()
            }

            try db.create(index: "chunks_document_ordinal", on: "chunks", columns: ["document_id", "ordinal"])

            try db.create(table: "embeddings") { table in
                table.column("chunk_id", .integer).primaryKey(onConflict: .replace)
                    .references("chunks", onDelete: .cascade)
                table.column("dim", .integer).notNull()
                table.column("vector_blob", .blob).notNull()
                table.column("norm", .double).notNull()
            }

            try db.create(table: "contexts") { table in
                table.column("id", .text).primaryKey()
                table.column("name", .text).notNull().unique(onConflict: .abort)
                table.column("created_at", .double).notNull()
                table.column("updated_at", .double).notNull()
            }

            try db.create(table: "context_chunks") { table in
                table.column("context_id", .text).notNull()
                    .references("contexts", onDelete: .cascade)
                table.column("chunk_id", .integer).notNull()
                    .references("chunks", onDelete: .cascade)
                table.column("added_at", .double).notNull()
                table.primaryKey(["context_id", "chunk_id"], onConflict: .replace)
            }

            try db.create(index: "context_chunks_context_chunk", on: "context_chunks", columns: ["context_id", "chunk_id"])
        }

        migrator.registerMigration("v2_chunks_fts5") { db in
            try db.create(virtualTable: "chunks_fts", using: FTS5()) { table in
                table.synchronize(withTable: "chunks")
                table.column("content")
            }
        }

        return migrator
    }

    private static func placeholders(count: Int) -> String {
        String(repeating: "?,", count: max(1, count)).dropLast().description
    }

    private static func encodeVector(_ vector: [Float]) -> Data {
        let copy = vector
        return copy.withUnsafeBytes { Data($0) }
    }

    private static func decodeVector(_ data: Data) -> [Float]? {
        let scalarSize = MemoryLayout<Float>.size
        guard data.count % scalarSize == 0 else { return nil }

        let count = data.count / scalarSize
        var vector = Array(repeating: Float.zero, count: count)
        _ = vector.withUnsafeMutableBytes { target in
            data.copyBytes(to: target)
        }
        return vector
    }
}
