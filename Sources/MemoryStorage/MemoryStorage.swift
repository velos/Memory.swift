import Foundation
import CSQLiteVec
import GRDB

public struct StoredChunkTag: Sendable, Codable, Hashable {
    public var name: String
    public var confidence: Double

    public init(name: String, confidence: Double) {
        self.name = name
        self.confidence = confidence
    }
}

public struct StoredChunkInput: Sendable {
    public var ordinal: Int
    public var content: String
    public var tokenCount: Int
    public var embedding: [Float]
    public var norm: Double
    public var memoryTypeOverride: String?
    public var memoryTypeOverrideSource: String?
    public var memoryTypeOverrideConfidence: Double?
    public var contentTags: [StoredChunkTag]
    public var memoryCategory: String?
    public var importance: Double
    public var accessCount: Int
    public var lastAccessedAt: Date?
    public var source: String
    public var createdAt: Date?

    public init(
        ordinal: Int,
        content: String,
        tokenCount: Int,
        embedding: [Float],
        norm: Double,
        memoryTypeOverride: String? = nil,
        memoryTypeOverrideSource: String? = nil,
        memoryTypeOverrideConfidence: Double? = nil,
        contentTags: [StoredChunkTag] = [],
        memoryCategory: String? = nil,
        importance: Double = 0.5,
        accessCount: Int = 0,
        lastAccessedAt: Date? = nil,
        source: String = "index",
        createdAt: Date? = nil
    ) {
        self.ordinal = ordinal
        self.content = content
        self.tokenCount = tokenCount
        self.embedding = embedding
        self.norm = norm
        self.memoryTypeOverride = memoryTypeOverride
        self.memoryTypeOverrideSource = memoryTypeOverrideSource
        self.memoryTypeOverrideConfidence = memoryTypeOverrideConfidence
        self.contentTags = contentTags
        self.memoryCategory = memoryCategory
        self.importance = min(1, max(0, importance))
        self.accessCount = max(0, accessCount)
        self.lastAccessedAt = lastAccessedAt
        self.source = source
        self.createdAt = createdAt
    }
}

public struct StoredDocumentInput: Sendable {
    public var path: String
    public var title: String?
    public var modifiedAt: Date
    public var checksum: String
    public var memoryType: String
    public var memoryTypeSource: String
    public var memoryTypeConfidence: Double?
    public var chunks: [StoredChunkInput]

    public init(
        path: String,
        title: String?,
        modifiedAt: Date,
        checksum: String,
        memoryType: String = "factual",
        memoryTypeSource: String = "fallback",
        memoryTypeConfidence: Double? = nil,
        chunks: [StoredChunkInput]
    ) {
        self.path = path
        self.title = title
        self.modifiedAt = modifiedAt
        self.checksum = checksum
        self.memoryType = memoryType
        self.memoryTypeSource = memoryTypeSource
        self.memoryTypeConfidence = memoryTypeConfidence
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
    public var memoryType: String
    public var memoryTypeSource: String
    public var memoryTypeConfidence: Double?
    public var contentTags: [StoredChunkTag]
    public var memoryCategory: String
    public var importance: Double
    public var accessCount: Int
    public var lastAccessedAt: Date?
    public var source: String
    public var createdAt: Date

    public init(
        chunkID: Int64,
        vector: [Float],
        norm: Double,
        content: String,
        documentPath: String,
        title: String?,
        modifiedAt: Date,
        memoryType: String,
        memoryTypeSource: String,
        memoryTypeConfidence: Double?,
        contentTags: [StoredChunkTag] = [],
        memoryCategory: String,
        importance: Double,
        accessCount: Int,
        lastAccessedAt: Date?,
        source: String,
        createdAt: Date
    ) {
        self.chunkID = chunkID
        self.vector = vector
        self.norm = norm
        self.content = content
        self.documentPath = documentPath
        self.title = title
        self.modifiedAt = modifiedAt
        self.memoryType = memoryType
        self.memoryTypeSource = memoryTypeSource
        self.memoryTypeConfidence = memoryTypeConfidence
        self.contentTags = contentTags
        self.memoryCategory = memoryCategory
        self.importance = min(1, max(0, importance))
        self.accessCount = max(0, accessCount)
        self.lastAccessedAt = lastAccessedAt
        self.source = source
        self.createdAt = createdAt
    }
}

public struct StoredChunkMetadata: Sendable {
    public var chunkID: Int64
    public var content: String
    public var documentPath: String
    public var title: String?
    public var modifiedAt: Date
    public var memoryType: String
    public var memoryTypeSource: String
    public var memoryTypeConfidence: Double?
    public var contentTags: [StoredChunkTag]
    public var memoryCategory: String
    public var importance: Double
    public var accessCount: Int
    public var lastAccessedAt: Date?
    public var source: String
    public var createdAt: Date

    public init(
        chunkID: Int64,
        content: String,
        documentPath: String,
        title: String?,
        modifiedAt: Date,
        memoryType: String,
        memoryTypeSource: String,
        memoryTypeConfidence: Double?,
        contentTags: [StoredChunkTag] = [],
        memoryCategory: String,
        importance: Double,
        accessCount: Int,
        lastAccessedAt: Date?,
        source: String,
        createdAt: Date
    ) {
        self.chunkID = chunkID
        self.content = content
        self.documentPath = documentPath
        self.title = title
        self.modifiedAt = modifiedAt
        self.memoryType = memoryType
        self.memoryTypeSource = memoryTypeSource
        self.memoryTypeConfidence = memoryTypeConfidence
        self.contentTags = contentTags
        self.memoryCategory = memoryCategory
        self.importance = min(1, max(0, importance))
        self.accessCount = max(0, accessCount)
        self.lastAccessedAt = lastAccessedAt
        self.source = source
        self.createdAt = createdAt
    }
}

public enum StoredMemorySort: Sendable {
    case recent
    case importance
    case mostAccessed
}

public struct LexicalHit: Sendable {
    public var chunkID: Int64
    public var score: Double

    public init(chunkID: Int64, score: Double) {
        self.chunkID = chunkID
        self.score = score
    }
}

public actor MemoryStorage {
    private let dbQueue: DatabaseQueue
    private static let vectorTableName = "chunk_vectors_vec"
    private static let vectorConfigTableName = "vector_index_config"

    public init(databaseURL: URL) throws {
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var configuration = Configuration()
        configuration.prepareDatabase { db in
            try db.execute(sql: "PRAGMA foreign_keys = ON")
            try Self.registerSQLiteVec(on: db)
        }

        self.dbQueue = try DatabaseQueue(path: databaseURL.path, configuration: configuration)
        try Self.makeMigrator().migrate(dbQueue)
    }

    public func wipeIndexData() throws {
        try dbQueue.write { db in
            try db.execute(sql: "DROP TABLE IF EXISTS \(Self.vectorTableName)")
            try db.execute(sql: "DELETE FROM \(Self.vectorConfigTableName)")
            try db.execute(sql: "DELETE FROM context_chunks")
            try db.execute(sql: "DELETE FROM embeddings")
            try db.execute(sql: "DELETE FROM chunks")
            try db.execute(sql: "DELETE FROM documents")
        }
    }

    public func replaceDocument(_ input: StoredDocumentInput) throws {
        try dbQueue.write { db in
            let existingChunkIDs = try Int64.fetchAll(
                db,
                sql: """
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.path = ?
                """,
                arguments: [input.path]
            )
            if !existingChunkIDs.isEmpty {
                try Self.deleteVectors(in: db, chunkIDs: existingChunkIDs)
            }

            try db.execute(sql: "DELETE FROM documents WHERE path = ?", arguments: [input.path])

            if let firstChunk = input.chunks.first {
                try Self.ensureVectorIndex(
                    in: db,
                    dimension: firstChunk.embedding.count
                )
            }

            let now = Date().timeIntervalSince1970
            try db.execute(
                sql: """
                INSERT INTO documents (
                    path,
                    title,
                    modified_at,
                    checksum,
                    memory_type,
                    memory_type_source,
                    memory_type_confidence,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                arguments: [
                    input.path,
                    input.title,
                    input.modifiedAt.timeIntervalSince1970,
                    input.checksum,
                    input.memoryType,
                    input.memoryTypeSource,
                    input.memoryTypeConfidence,
                    now,
                    now,
                ]
            )

            let documentID = db.lastInsertedRowID
            for chunk in input.chunks {
                try db.execute(
                    sql: """
                    INSERT INTO chunks (
                        document_id,
                        ordinal,
                        content,
                        token_count,
                        memory_type_override,
                        memory_type_override_source,
                        memory_type_override_confidence,
                        memory_category,
                        importance,
                        access_count,
                        last_accessed_at,
                        source,
                        content_tags_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    arguments: [
                        documentID,
                        chunk.ordinal,
                        chunk.content,
                        chunk.tokenCount,
                        chunk.memoryTypeOverride,
                        chunk.memoryTypeOverrideSource,
                        chunk.memoryTypeOverrideConfidence,
                        chunk.memoryCategory ?? "observation",
                        chunk.importance,
                        chunk.accessCount,
                        chunk.lastAccessedAt?.timeIntervalSince1970,
                        chunk.source,
                        Self.encodeContentTags(chunk.contentTags),
                        (chunk.createdAt ?? Date()).timeIntervalSince1970,
                    ]
                )

                let chunkID = db.lastInsertedRowID
                if let configuredDimension = try Self.configuredVectorDimension(in: db),
                   configuredDimension != chunk.embedding.count {
                    throw DatabaseError(
                        message: "Embedding dimension mismatch. Expected \(configuredDimension), got \(chunk.embedding.count)."
                    )
                }

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

                if try Self.vectorTableExists(in: db) {
                    try db.execute(
                        sql: """
                        INSERT OR REPLACE INTO \(Self.vectorTableName) (chunk_id, embedding)
                        VALUES (?, ?)
                        """,
                        arguments: [
                            chunkID,
                            Self.encodeVector(chunk.embedding),
                        ]
                    )
                }
            }
        }
    }

    public func removeDocuments(paths: [String]) throws {
        guard !paths.isEmpty else { return }

        try dbQueue.write { db in
            let chunkIDs = try Int64.fetchAll(
                db,
                sql: """
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.path IN (\(Self.placeholders(count: paths.count)))
                """,
                arguments: StatementArguments(paths)
            )
            if !chunkIDs.isEmpty {
                try Self.deleteVectors(in: db, chunkIDs: chunkIDs)
            }

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
                    COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
                    COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
                    COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
                    c.memory_category AS memory_category,
                    c.importance AS importance,
                    c.access_count AS access_count,
                    c.last_accessed_at AS last_accessed_at,
                    c.source AS source,
                    c.created_at AS created_at,
                    COALESCE(c.content_tags_json, '[]') AS content_tags_json,
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
                    modifiedAt: Date(timeIntervalSince1970: row["modified_at"]),
                    memoryType: row["memory_type"],
                    memoryTypeSource: row["memory_type_source"],
                    memoryTypeConfidence: row["memory_type_confidence"],
                    contentTags: Self.decodeContentTags(row["content_tags_json"]),
                    memoryCategory: row["memory_category"],
                    importance: row["importance"],
                    accessCount: row["access_count"],
                    lastAccessedAt: Self.decodeTimestamp(row["last_accessed_at"]),
                    source: row["source"],
                    createdAt: Date(timeIntervalSince1970: row["created_at"])
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
                d.modified_at AS modified_at,
                COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
                COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
                COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
                c.memory_category AS memory_category,
                c.importance AS importance,
                c.access_count AS access_count,
                c.last_accessed_at AS last_accessed_at,
                c.source AS source,
                c.created_at AS created_at,
                COALESCE(c.content_tags_json, '[]') AS content_tags_json
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
                    modifiedAt: Date(timeIntervalSince1970: $0["modified_at"]),
                    memoryType: $0["memory_type"],
                    memoryTypeSource: $0["memory_type_source"],
                    memoryTypeConfidence: $0["memory_type_confidence"],
                    contentTags: Self.decodeContentTags($0["content_tags_json"]),
                    memoryCategory: $0["memory_category"],
                    importance: $0["importance"],
                    accessCount: $0["access_count"],
                    lastAccessedAt: Self.decodeTimestamp($0["last_accessed_at"]),
                    source: $0["source"],
                    createdAt: Date(timeIntervalSince1970: $0["created_at"])
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
                    d.modified_at AS modified_at,
                    COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
                    COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
                    COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
                    c.memory_category AS memory_category,
                    c.importance AS importance,
                    c.access_count AS access_count,
                    c.last_accessed_at AS last_accessed_at,
                    c.source AS source,
                    c.created_at AS created_at,
                    COALESCE(c.content_tags_json, '[]') AS content_tags_json
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
                modifiedAt: Date(timeIntervalSince1970: row["modified_at"]),
                memoryType: row["memory_type"],
                memoryTypeSource: row["memory_type_source"],
                memoryTypeConfidence: row["memory_type_confidence"],
                contentTags: Self.decodeContentTags(row["content_tags_json"]),
                memoryCategory: row["memory_category"],
                importance: row["importance"],
                accessCount: row["access_count"],
                lastAccessedAt: Self.decodeTimestamp(row["last_accessed_at"]),
                source: row["source"],
                createdAt: Date(timeIntervalSince1970: row["created_at"])
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

    public func fetchChunkMetadataForDocument(path: String) throws -> [StoredChunkMetadata] {
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
                    COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
                    COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
                    COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
                    c.memory_category AS memory_category,
                    c.importance AS importance,
                    c.access_count AS access_count,
                    c.last_accessed_at AS last_accessed_at,
                    c.source AS source,
                    c.created_at AS created_at,
                    COALESCE(c.content_tags_json, '[]') AS content_tags_json
                FROM documents d
                JOIN chunks c ON c.document_id = d.id
                WHERE d.path = ?
                ORDER BY c.ordinal ASC
                """,
                arguments: [path]
            )

            return rows.map {
                StoredChunkMetadata(
                    chunkID: $0["chunk_id"],
                    content: $0["content"],
                    documentPath: $0["document_path"],
                    title: $0["title"],
                    modifiedAt: Date(timeIntervalSince1970: $0["modified_at"]),
                    memoryType: $0["memory_type"],
                    memoryTypeSource: $0["memory_type_source"],
                    memoryTypeConfidence: $0["memory_type_confidence"],
                    contentTags: Self.decodeContentTags($0["content_tags_json"]),
                    memoryCategory: $0["memory_category"],
                    importance: $0["importance"],
                    accessCount: $0["access_count"],
                    lastAccessedAt: Self.decodeTimestamp($0["last_accessed_at"]),
                    source: $0["source"],
                    createdAt: Date(timeIntervalSince1970: $0["created_at"])
                )
            }
        }
    }

    public func listMemoryMetadata(
        limit: Int,
        sort: StoredMemorySort,
        memoryCategory: String? = nil,
        allowedMemoryTypes: Set<String>? = nil
    ) throws -> [StoredChunkMetadata] {
        guard limit > 0 else { return [] }

        return try dbQueue.read { db in
            var arguments = StatementArguments()
            var filters: [String] = []

            if let memoryCategory {
                let trimmed = memoryCategory.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                if !trimmed.isEmpty {
                    filters.append("c.memory_category = ?")
                    arguments += StatementArguments([trimmed])
                }
            }

            if let allowedMemoryTypes, !allowedMemoryTypes.isEmpty {
                let orderedTypes = allowedMemoryTypes.sorted()
                filters.append("COALESCE(c.memory_type_override, d.memory_type) IN (\(Self.placeholders(count: orderedTypes.count)))")
                arguments += StatementArguments(orderedTypes)
            }

            let whereClause = filters.isEmpty ? "" : "WHERE " + filters.joined(separator: " AND ")

            let orderClause: String
            switch sort {
            case .recent:
                orderClause = "ORDER BY c.created_at DESC, c.id DESC"
            case .importance:
                orderClause = "ORDER BY c.importance DESC, c.created_at DESC, c.id DESC"
            case .mostAccessed:
                orderClause = "ORDER BY c.access_count DESC, COALESCE(c.last_accessed_at, 0) DESC, c.created_at DESC, c.id DESC"
            }

            let sql = """
            SELECT
                c.id AS chunk_id,
                c.content AS content,
                d.path AS document_path,
                d.title AS title,
                d.modified_at AS modified_at,
                COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
                COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
                COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
                c.memory_category AS memory_category,
                c.importance AS importance,
                c.access_count AS access_count,
                c.last_accessed_at AS last_accessed_at,
                c.source AS source,
                c.created_at AS created_at,
                COALESCE(c.content_tags_json, '[]') AS content_tags_json
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            \(whereClause)
            \(orderClause)
            LIMIT ?
            """
            arguments += StatementArguments([limit])

            let rows = try Row.fetchAll(db, sql: sql, arguments: arguments)
            return rows.map {
                StoredChunkMetadata(
                    chunkID: $0["chunk_id"],
                    content: $0["content"],
                    documentPath: $0["document_path"],
                    title: $0["title"],
                    modifiedAt: Date(timeIntervalSince1970: $0["modified_at"]),
                    memoryType: $0["memory_type"],
                    memoryTypeSource: $0["memory_type_source"],
                    memoryTypeConfidence: $0["memory_type_confidence"],
                    contentTags: Self.decodeContentTags($0["content_tags_json"]),
                    memoryCategory: $0["memory_category"],
                    importance: $0["importance"],
                    accessCount: $0["access_count"],
                    lastAccessedAt: Self.decodeTimestamp($0["last_accessed_at"]),
                    source: $0["source"],
                    createdAt: Date(timeIntervalSince1970: $0["created_at"])
                )
            }
        }
    }

    public func recordChunkAccesses(_ chunkIDs: [Int64], accessedAt: Date = Date()) throws {
        guard !chunkIDs.isEmpty else { return }

        try dbQueue.write { db in
            let sql = """
            UPDATE chunks
            SET
                access_count = access_count + 1,
                last_accessed_at = ?
            WHERE id IN (\(Self.placeholders(count: chunkIDs.count)))
            """
            var arguments = StatementArguments([accessedAt.timeIntervalSince1970])
            arguments += StatementArguments(chunkIDs)
            try db.execute(sql: sql, arguments: arguments)
        }
    }

    public func lexicalSearch(
        query: String,
        limit: Int,
        allowedChunkIDs: Set<Int64>? = nil,
        allowedMemoryTypes: Set<String>? = nil
    ) throws -> [LexicalHit] {
        guard limit > 0 else { return [] }
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let strictPattern = Self.makeStrictFTSQuery(from: trimmed)
        let relaxedPattern = Self.makeRelaxedFTSQuery(from: trimmed)
        guard strictPattern != nil || relaxedPattern != nil else { return [] }

        return try dbQueue.read { db in
            let effectiveLimit = limit
            var mergedByChunkID: [Int64: Double] = [:]
            var seenChunkIDs: Set<Int64> = []

            if let strictPattern {
                let strictHits = try Self.runLexicalSearchQuery(
                    in: db,
                    pattern: strictPattern,
                    limit: effectiveLimit,
                    allowedChunkIDs: allowedChunkIDs,
                    allowedMemoryTypes: allowedMemoryTypes,
                    excludedChunkIDs: nil
                )
                for (index, hit) in strictHits.enumerated() {
                    seenChunkIDs.insert(hit.chunkID)
                    let rank = Double(index + 1)
                    let score = 1.2 / (60 + rank)
                    mergedByChunkID[hit.chunkID, default: 0] += score
                }
            }

            if let relaxedPattern {
                let relaxedHits = try Self.runLexicalSearchQuery(
                    in: db,
                    pattern: relaxedPattern,
                    limit: effectiveLimit,
                    allowedChunkIDs: allowedChunkIDs,
                    allowedMemoryTypes: allowedMemoryTypes,
                    excludedChunkIDs: nil
                )

                for (index, hit) in relaxedHits.enumerated() {
                    seenChunkIDs.insert(hit.chunkID)
                    let rank = Double(index + 1)
                    let score = 1.0 / (60 + rank)
                    mergedByChunkID[hit.chunkID, default: 0] += score
                }
            }

            return seenChunkIDs
                .map { LexicalHit(chunkID: $0, score: mergedByChunkID[$0, default: 0]) }
                .sorted { lhs, rhs in
                    if lhs.score == rhs.score {
                        return lhs.chunkID < rhs.chunkID
                    }
                    return lhs.score > rhs.score
                }
                .prefix(effectiveLimit)
                .map { $0 }
        }
    }

    public func vectorSearch(
        queryVector: [Float],
        limit: Int,
        allowedChunkIDs: Set<Int64>? = nil,
        allowedMemoryTypes: Set<String>? = nil
    ) throws -> [LexicalHit] {
        guard limit > 0 else { return [] }
        guard !queryVector.isEmpty else { return [] }

        return try dbQueue.read { db in
            guard try Self.vectorTableExists(in: db) else {
                return []
            }

            if let configuredDimension = try Self.configuredVectorDimension(in: db),
               configuredDimension != queryVector.count {
                return []
            }

            let overfetch = min(max(limit * 6, limit), 5_000)
            let queryBlob = Self.encodeVector(queryVector)

            // sqlite-vec can hang if we join directly against the vec virtual table.
            // Query vec results first, then hydrate/filter in a second query.
            let rows = try Row.fetchAll(
                db,
                sql: """
                SELECT chunk_id, distance
                FROM \(Self.vectorTableName)
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                arguments: [queryBlob, overfetch]
            )

            var candidates: [(chunkID: Int64, distance: Double)] = rows.compactMap { row in
                let chunkID: Int64 = row["chunk_id"]
                let distance: Double = row["distance"]
                guard distance.isFinite else { return nil }
                return (chunkID, distance)
            }

            if let allowedChunkIDs {
                if allowedChunkIDs.isEmpty {
                    return []
                }
                candidates = candidates.filter { allowedChunkIDs.contains($0.chunkID) }
            }

            if let allowedMemoryTypes {
                if allowedMemoryTypes.isEmpty {
                    return []
                }

                let candidateIDs = candidates.map(\.chunkID)
                if candidateIDs.isEmpty {
                    return []
                }

                let memoryTypeRows = try Row.fetchAll(
                    db,
                    sql: """
                    SELECT
                        c.id AS chunk_id,
                        COALESCE(c.memory_type_override, d.memory_type) AS memory_type
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.id IN (\(Self.placeholders(count: candidateIDs.count)))
                    """,
                    arguments: StatementArguments(candidateIDs)
                )

                let allowedIDs = Set(
                    memoryTypeRows.compactMap { row -> Int64? in
                        let memoryType: String = row["memory_type"]
                        guard allowedMemoryTypes.contains(memoryType) else { return nil }
                        let chunkID: Int64 = row["chunk_id"]
                        return chunkID
                    }
                )

                candidates = candidates.filter { allowedIDs.contains($0.chunkID) }
            }

            return candidates
                .sorted { lhs, rhs in
                    if lhs.distance == rhs.distance {
                        return lhs.chunkID < rhs.chunkID
                    }
                    return lhs.distance < rhs.distance
                }
                .prefix(limit)
                .map { LexicalHit(chunkID: $0.chunkID, score: -$0.distance) }
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
                    d.modified_at AS modified_at,
                    COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
                    COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
                    COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
                    c.memory_category AS memory_category,
                    c.importance AS importance,
                    c.access_count AS access_count,
                    c.last_accessed_at AS last_accessed_at,
                    c.source AS source,
                    c.created_at AS created_at,
                    COALESCE(c.content_tags_json, '[]') AS content_tags_json
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
                    modifiedAt: Date(timeIntervalSince1970: $0["modified_at"]),
                    memoryType: $0["memory_type"],
                    memoryTypeSource: $0["memory_type_source"],
                    memoryTypeConfidence: $0["memory_type_confidence"],
                    contentTags: Self.decodeContentTags($0["content_tags_json"]),
                    memoryCategory: $0["memory_category"],
                    importance: $0["importance"],
                    accessCount: $0["access_count"],
                    lastAccessedAt: Self.decodeTimestamp($0["last_accessed_at"]),
                    source: $0["source"],
                    createdAt: Date(timeIntervalSince1970: $0["created_at"])
                )
            }
        }
    }

    @discardableResult
    public func setDocumentMemoryType(
        path: String,
        type: String,
        source: String,
        confidence: Double?
    ) throws -> Bool {
        try dbQueue.write { db in
            let now = Date().timeIntervalSince1970
            try db.execute(
                sql: """
                UPDATE documents
                SET memory_type = ?, memory_type_source = ?, memory_type_confidence = ?, updated_at = ?
                WHERE path = ?
                """,
                arguments: [type, source, confidence, now, path]
            )
            return db.changesCount > 0
        }
    }

    @discardableResult
    public func setChunkMemoryTypeOverride(
        chunkID: Int64,
        type: String?,
        source: String?,
        confidence: Double?
    ) throws -> Bool {
        try dbQueue.write { db in
            try db.execute(
                sql: """
                UPDATE chunks
                SET
                    memory_type_override = ?,
                    memory_type_override_source = ?,
                    memory_type_override_confidence = ?
                WHERE id = ?
                """,
                arguments: [type, source, confidence, chunkID]
            )
            return db.changesCount > 0
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

            try db.create(table: Self.vectorConfigTableName) { table in
                table.column("id", .integer).notNull().primaryKey()
                table.column("embedding_dim", .integer).notNull()
                table.column("created_at", .double).notNull()
                table.column("updated_at", .double).notNull()
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

        migrator.registerMigration("v3_memory_types") { db in
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

            try db.create(index: "documents_memory_type", on: "documents", columns: ["memory_type"])
            try db.create(index: "chunks_memory_type_override", on: "chunks", columns: ["memory_type_override"])
        }

        migrator.registerMigration("v4_chunk_content_tags") { db in
            try db.alter(table: "chunks") { table in
                table.add(column: "content_tags_json", .text).notNull().defaults(to: "[]")
            }
        }

        migrator.registerMigration("v5_chunk_memory_metadata") { db in
            try db.alter(table: "chunks") { table in
                table.add(column: "memory_category", .text).notNull().defaults(to: "observation")
                table.add(column: "importance", .double).notNull().defaults(to: 0.5)
                table.add(column: "access_count", .integer).notNull().defaults(to: 0)
                table.add(column: "last_accessed_at", .double)
                table.add(column: "source", .text).notNull().defaults(to: "index")
            }

            try db.create(index: "chunks_memory_category", on: "chunks", columns: ["memory_category"])
            try db.create(index: "chunks_importance", on: "chunks", columns: ["importance"])
            try db.create(index: "chunks_access_count", on: "chunks", columns: ["access_count"])
            try db.create(index: "chunks_created_at", on: "chunks", columns: ["created_at"])
        }

        migrator.registerMigration("v6_vector_index_config") { db in
            try db.create(table: Self.vectorConfigTableName, ifNotExists: true) { table in
                table.column("id", .integer).notNull().primaryKey()
                table.column("embedding_dim", .integer).notNull()
                table.column("created_at", .double).notNull()
                table.column("updated_at", .double).notNull()
            }
        }

        return migrator
    }

    private static func placeholders(count: Int) -> String {
        String(repeating: "?,", count: max(1, count)).dropLast().description
    }

    private static func runLexicalSearchQuery(
        in db: Database,
        pattern: String,
        limit: Int,
        allowedChunkIDs: Set<Int64>?,
        allowedMemoryTypes: Set<String>?,
        excludedChunkIDs: Set<Int64>?
    ) throws -> [LexicalHit] {
        guard limit > 0 else { return [] }

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

        if let excludedChunkIDs, !excludedChunkIDs.isEmpty {
            let orderedExcluded = excludedChunkIDs.sorted()
            sql += " AND rowid NOT IN (\(Self.placeholders(count: orderedExcluded.count)))"
            arguments += StatementArguments(orderedExcluded)
        }

        if let allowedMemoryTypes, !allowedMemoryTypes.isEmpty {
            let orderedTypes = allowedMemoryTypes.sorted()
            sql += """
             AND rowid IN (
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE COALESCE(c.memory_type_override, d.memory_type) IN (\(Self.placeholders(count: orderedTypes.count)))
             )
            """
            arguments += StatementArguments(orderedTypes)
        }

        sql += " ORDER BY rank LIMIT ?"
        arguments += StatementArguments([limit])

        let rows = try Row.fetchAll(db, sql: sql, arguments: arguments)
        return rows.enumerated().map { index, row in
            let chunkID: Int64 = row["chunk_id"]
            let rank: Double? = row["rank"]
            let score = -(rank ?? Double(index + 1))
            return LexicalHit(chunkID: chunkID, score: score)
        }
    }

    private static func makeStrictFTSQuery(from query: String) -> String? {
        let significant = lexicalTokens(from: query).filter { token in
            isImportantShortToken(token) || token.contains(where: \.isNumber) || (token.count >= 3 && !lexicalStopWords.contains(token))
        }
        guard !significant.isEmpty else { return nil }

        let limited = Array(significant.prefix(6))
        if limited.count == 1 {
            return "\(limited[0])*"
        }
        return limited.map { "\($0)*" }.joined(separator: " AND ")
    }

    private static func makeRelaxedFTSQuery(from query: String) -> String? {
        let tokens = lexicalTokens(from: query)
        guard !tokens.isEmpty else { return nil }

        let preferred = tokens.filter { token in
            if token.contains(where: \.isNumber) { return true }
            if isImportantShortToken(token) { return true }
            if token.count >= 3 && !lexicalStopWords.contains(token) { return true }
            return false
        }

        let source = preferred.isEmpty ? tokens : preferred
        let limited = Array(source.prefix(12))
        guard !limited.isEmpty else { return nil }
        return limited.map { "\($0)*" }.joined(separator: " OR ")
    }

    private static func lexicalTokens(from query: String) -> [String] {
        var seen: Set<String> = []
        var tokens: [String] = []

        let rawTokens = query.lowercased().split { character in
            !character.isLetter && !character.isNumber
        }

        for raw in rawTokens {
            let token = String(raw)
            guard token.count >= 2 || token.contains(where: \.isNumber) else {
                continue
            }
            if seen.insert(token).inserted {
                tokens.append(token)
            }
        }

        return tokens
    }

    private static func isImportantShortToken(_ token: String) -> Bool {
        importantShortTokens.contains(token)
    }

    private static let importantShortTokens: Set<String> = [
        "ai", "api", "ci", "db", "id", "qa", "sdk", "sli", "slo", "ui", "ux"
    ]

    private static let lexicalStopWords: Set<String> = [
        "about", "after", "again", "against", "all", "also", "among", "an", "and", "any", "are", "as",
        "at", "be", "been", "before", "between", "both", "but", "by", "can", "did", "do", "does",
        "during", "each", "end", "for", "from", "had", "has", "have", "how", "if", "in", "into", "is",
        "it", "its", "just", "like", "more", "most", "need", "not", "of", "on", "or", "our", "out",
        "over", "should", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these",
        "they", "this", "those", "to", "too", "under", "up", "use", "using", "was", "we", "what", "when",
        "where", "which", "who", "why", "with", "would", "you", "your"
    ]

    private static func registerSQLiteVec(on db: Database) throws {
        guard let sqliteConnection = db.sqliteConnection else {
            throw SQLiteVecInitializationError(code: SQLITE_MISUSE, message: "Missing sqlite connection handle.")
        }

        var errorMessagePointer: UnsafeMutablePointer<CChar>?
        let resultCode = sqlite3_vec_init(sqliteConnection, &errorMessagePointer, nil)
        defer {
            if let errorMessagePointer {
                sqlite3_free(errorMessagePointer)
            }
        }

        guard resultCode == SQLITE_OK else {
            let message = errorMessagePointer.map { String(cString: $0) }
            throw SQLiteVecInitializationError(code: resultCode, message: message)
        }
    }

    private static func vectorTableExists(in db: Database) throws -> Bool {
        try String.fetchOne(
            db,
            sql: """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            arguments: [Self.vectorTableName]
        ) != nil
    }

    private static func configuredVectorDimension(in db: Database) throws -> Int? {
        try Int.fetchOne(
            db,
            sql: "SELECT embedding_dim FROM \(Self.vectorConfigTableName) WHERE id = 1"
        )
    }

    private static func ensureVectorIndex(in db: Database, dimension: Int) throws {
        guard dimension > 0 else {
            throw DatabaseError(message: "Embedding dimension must be greater than zero.")
        }

        let now = Date().timeIntervalSince1970
        if let existingDimension = try configuredVectorDimension(in: db) {
            guard existingDimension == dimension else {
                throw DatabaseError(
                    message: "Embedding dimension mismatch. Expected \(existingDimension), got \(dimension)."
                )
            }

            if try !vectorTableExists(in: db) {
                try createVectorTable(in: db, dimension: existingDimension)
            }

            try db.execute(
                sql: "UPDATE \(Self.vectorConfigTableName) SET updated_at = ? WHERE id = 1",
                arguments: [now]
            )
            return
        }

        try createVectorTable(in: db, dimension: dimension)
        try db.execute(
            sql: """
            INSERT INTO \(Self.vectorConfigTableName) (id, embedding_dim, created_at, updated_at)
            VALUES (1, ?, ?, ?)
            """,
            arguments: [dimension, now, now]
        )
    }

    private static func createVectorTable(in db: Database, dimension: Int) throws {
        try db.execute(sql: "DROP TABLE IF EXISTS \(Self.vectorTableName)")
        try db.execute(
            sql: """
            CREATE VIRTUAL TABLE \(Self.vectorTableName) USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[\(dimension)] distance_metric=cosine
            )
            """
        )
    }

    private static func deleteVectors(in db: Database, chunkIDs: [Int64]) throws {
        guard !chunkIDs.isEmpty else { return }
        guard try vectorTableExists(in: db) else { return }

        try db.execute(
            sql: "DELETE FROM \(Self.vectorTableName) WHERE chunk_id IN (\(Self.placeholders(count: chunkIDs.count)))",
            arguments: StatementArguments(chunkIDs)
        )
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

    private static func encodeContentTags(_ tags: [StoredChunkTag]) -> String {
        guard !tags.isEmpty else { return "[]" }
        guard
            let data = try? JSONEncoder().encode(tags),
            let encoded = String(data: data, encoding: .utf8)
        else {
            return "[]"
        }
        return encoded
    }

    private static func decodeContentTags(_ raw: String?) -> [StoredChunkTag] {
        guard let raw else { return [] }
        guard let data = raw.data(using: .utf8) else { return [] }
        return (try? JSONDecoder().decode([StoredChunkTag].self, from: data)) ?? []
    }

    private static func decodeTimestamp(_ raw: Double?) -> Date? {
        guard let raw else { return nil }
        return Date(timeIntervalSince1970: raw)
    }
}

private struct SQLiteVecInitializationError: Error, LocalizedError {
    let code: Int32
    let message: String?

    var errorDescription: String? {
        if let message, !message.isEmpty {
            return "Failed to initialize sqlite-vec extension (result code: \(code)): \(message)"
        }
        return "Failed to initialize sqlite-vec extension (result code: \(code))."
    }
}
