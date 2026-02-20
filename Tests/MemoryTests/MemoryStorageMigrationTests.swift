import Foundation
import GRDB
import MemoryStorage
import Testing

struct MemoryStorageMigrationTests {
    @Test
    func v3MigrationAddsMemoryTypeColumnsAndPreservesExistingRows() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("legacy.sqlite")

        try createLegacyV2Database(at: dbURL)

        let storage = try MemoryStorage(databaseURL: dbURL)
        let row = try await storage.fetchChunkMetadata(chunkID: 1)

        #expect(row?.chunkID == 1)
        #expect(row?.content == "legacy chunk content")
        #expect(row?.documentPath == "/tmp/legacy.md")
        #expect(row?.memoryType == "factual")
        #expect(row?.memoryTypeSource == "fallback")
        #expect(row?.memoryTypeConfidence == nil)
        #expect(row?.contentTags.isEmpty == true)
    }

    @Test
    func v4MigrationAddsContentTagsColumnAndPreservesExistingRows() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("legacy-v3.sqlite")

        try createLegacyV3Database(at: dbURL)

        let storage = try MemoryStorage(databaseURL: dbURL)
        let row = try await storage.fetchChunkMetadata(chunkID: 1)

        #expect(row?.chunkID == 1)
        #expect(row?.content == "legacy chunk content")
        #expect(row?.memoryType == "semantic")
        #expect(row?.memoryTypeSource == "manual")
        #expect(row?.memoryTypeConfidence == 0.9)
        #expect(row?.contentTags.isEmpty == true)
    }

    private func createLegacyV2Database(at url: URL) throws {
        let dbQueue = try DatabaseQueue(path: url.path)
        try dbQueue.write { db in
            try db.execute(sql: "PRAGMA foreign_keys = ON")

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

            try db.create(virtualTable: "chunks_fts", using: FTS5()) { table in
                table.synchronize(withTable: "chunks")
                table.column("content")
            }

            let now = Date().timeIntervalSince1970
            try db.execute(
                sql: """
                INSERT INTO documents (id, path, title, modified_at, checksum, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                arguments: [1, "/tmp/legacy.md", "Legacy", now, "abc123", now, now]
            )
            try db.execute(
                sql: """
                INSERT INTO chunks (id, document_id, ordinal, content, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                arguments: [1, 1, 0, "legacy chunk content", 3, now]
            )
            try db.execute(
                sql: """
                INSERT INTO embeddings (chunk_id, dim, vector_blob, norm)
                VALUES (?, ?, ?, ?)
                """,
                arguments: [1, 3, encode([1, 2, 3]), 3.7416573868]
            )

            try db.create(table: "grdb_migrations") { table in
                table.column("identifier", .text).primaryKey()
            }
            try db.execute(sql: "INSERT INTO grdb_migrations(identifier) VALUES (?)", arguments: ["v1_initial"])
            try db.execute(sql: "INSERT INTO grdb_migrations(identifier) VALUES (?)", arguments: ["v2_chunks_fts5"])
        }
    }

    private func createLegacyV3Database(at url: URL) throws {
        try createLegacyV2Database(at: url)

        let dbQueue = try DatabaseQueue(path: url.path)
        try dbQueue.write { db in
            try db.alter(table: "documents") { table in
                table.add(column: "memory_type", .text).notNull().defaults(to: "factual")
                table.add(column: "memory_type_source", .text).notNull().defaults(to: "fallback")
                table.add(column: "memory_type_confidence", .double)
            }

            try db.alter(table: "chunks") { table in
                table.add(column: "memory_type_override", .text)
                table.add(column: "memory_type_override_source", .text)
                table.add(column: "memory_type_override_confidence", .double)
            }

            try db.execute(
                sql: """
                UPDATE documents
                SET memory_type = ?, memory_type_source = ?, memory_type_confidence = ?
                WHERE id = ?
                """,
                arguments: ["semantic", "manual", 0.9, 1]
            )
            try db.execute(sql: "INSERT INTO grdb_migrations(identifier) VALUES (?)", arguments: ["v3_memory_types"])
        }
    }

    private func encode(_ vector: [Float]) -> Data {
        let copy = vector
        return copy.withUnsafeBytes { Data($0) }
    }
}
