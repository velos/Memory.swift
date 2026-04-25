import Foundation
import CSQLiteVec
import SQLiteSupport

public struct StoredChunkTag: Sendable, Codable, Hashable {
    public var name: String
    public var confidence: Double

    public init(name: String, confidence: Double) {
        self.name = name
        self.confidence = confidence
    }
}

public struct StoredMemoryEntity: Sendable, Codable, Hashable {
    public var label: String
    public var value: String
    public var normalizedValue: String
    public var confidence: Double?

    public init(
        label: String,
        value: String,
        normalizedValue: String,
        confidence: Double? = nil
    ) {
        self.label = label
        self.value = value
        self.normalizedValue = normalizedValue
        self.confidence = confidence.map { min(1, max(0, $0)) }
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
    public var memoryKind: String?
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
        memoryKind: String? = nil,
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
        self.memoryKind = memoryKind
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
    public var memoryID: String?
    public var memoryKind: String?
    public var memoryStatus: String?
    public var memoryCanonicalKey: String?
    public var memoryType: String
    public var memoryTypeSource: String
    public var memoryTypeConfidence: Double?
    public var chunks: [StoredChunkInput]

    public init(
        path: String,
        title: String?,
        modifiedAt: Date,
        checksum: String,
        memoryID: String? = nil,
        memoryKind: String? = nil,
        memoryStatus: String? = nil,
        memoryCanonicalKey: String? = nil,
        memoryType: String = "factual",
        memoryTypeSource: String = "fallback",
        memoryTypeConfidence: Double? = nil,
        chunks: [StoredChunkInput]
    ) {
        self.path = path
        self.title = title
        self.modifiedAt = modifiedAt
        self.checksum = checksum
        self.memoryID = memoryID
        self.memoryKind = memoryKind
        self.memoryStatus = memoryStatus
        self.memoryCanonicalKey = memoryCanonicalKey
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
    public var memoryID: String?
    public var memoryKind: String?
    public var memoryStatus: String?
    public var memoryCanonicalKey: String?
    public var memoryType: String
    public var memoryTypeSource: String
    public var memoryTypeConfidence: Double?
    public var contentTags: [StoredChunkTag]
    public var memoryKindFallback: String
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
        memoryID: String? = nil,
        memoryKind: String? = nil,
        memoryStatus: String? = nil,
        memoryCanonicalKey: String? = nil,
        memoryType: String,
        memoryTypeSource: String,
        memoryTypeConfidence: Double?,
        contentTags: [StoredChunkTag] = [],
        memoryKindFallback: String,
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
        self.memoryID = memoryID
        self.memoryKind = memoryKind
        self.memoryStatus = memoryStatus
        self.memoryCanonicalKey = memoryCanonicalKey
        self.memoryType = memoryType
        self.memoryTypeSource = memoryTypeSource
        self.memoryTypeConfidence = memoryTypeConfidence
        self.contentTags = contentTags
        self.memoryKindFallback = memoryKindFallback
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
    public var memoryID: String?
    public var memoryKind: String?
    public var memoryStatus: String?
    public var memoryCanonicalKey: String?
    public var memoryType: String
    public var memoryTypeSource: String
    public var memoryTypeConfidence: Double?
    public var contentTags: [StoredChunkTag]
    public var memoryKindFallback: String
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
        memoryID: String? = nil,
        memoryKind: String? = nil,
        memoryStatus: String? = nil,
        memoryCanonicalKey: String? = nil,
        memoryType: String,
        memoryTypeSource: String,
        memoryTypeConfidence: Double?,
        contentTags: [StoredChunkTag] = [],
        memoryKindFallback: String,
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
        self.memoryID = memoryID
        self.memoryKind = memoryKind
        self.memoryStatus = memoryStatus
        self.memoryCanonicalKey = memoryCanonicalKey
        self.memoryType = memoryType
        self.memoryTypeSource = memoryTypeSource
        self.memoryTypeConfidence = memoryTypeConfidence
        self.contentTags = contentTags
        self.memoryKindFallback = memoryKindFallback
        self.importance = min(1, max(0, importance))
        self.accessCount = max(0, accessCount)
        self.lastAccessedAt = lastAccessedAt
        self.source = source
        self.createdAt = createdAt
    }
}

public struct StoredMemoryInput: Sendable {
    public var id: String
    public var title: String?
    public var kind: String
    public var status: String
    public var canonicalKey: String?
    public var text: String
    public var tags: [String]
    public var facetTags: [String]
    public var entities: [StoredMemoryEntity]
    public var topics: [String]
    public var importance: Double
    public var confidence: Double?
    public var source: String
    public var createdAt: Date
    public var eventAt: Date?
    public var updatedAt: Date
    public var supersedesID: String?
    public var supersededByID: String?
    public var metadata: [String: String]

    public init(
        id: String,
        title: String?,
        kind: String,
        status: String,
        canonicalKey: String?,
        text: String,
        tags: [String],
        facetTags: [String],
        entities: [StoredMemoryEntity],
        topics: [String],
        importance: Double,
        confidence: Double?,
        source: String,
        createdAt: Date,
        eventAt: Date?,
        updatedAt: Date,
        supersedesID: String?,
        supersededByID: String?,
        metadata: [String: String]
    ) {
        self.id = id
        self.title = title
        self.kind = kind
        self.status = status
        self.canonicalKey = canonicalKey
        self.text = text
        self.tags = tags
        self.facetTags = facetTags
        self.entities = entities
        self.topics = topics
        self.importance = min(1, max(0, importance))
        self.confidence = confidence.map { min(1, max(0, $0)) }
        self.source = source
        self.createdAt = createdAt
        self.eventAt = eventAt
        self.updatedAt = updatedAt
        self.supersedesID = supersedesID
        self.supersededByID = supersededByID
        self.metadata = metadata
    }
}

public struct StoredMemoryRecord: Sendable, Codable, Hashable {
    public var id: String
    public var title: String?
    public var kind: String
    public var status: String
    public var canonicalKey: String?
    public var text: String
    public var tags: [String]
    public var facetTags: [String]
    public var entities: [StoredMemoryEntity]
    public var topics: [String]
    public var importance: Double
    public var confidence: Double?
    public var source: String
    public var createdAt: Date
    public var eventAt: Date?
    public var updatedAt: Date
    public var supersedesID: String?
    public var supersededByID: String?
    public var metadata: [String: String]
    public var chunkID: Int64?
    public var documentPath: String?
    public var accessCount: Int
    public var lastAccessedAt: Date?
    public var legacyDocumentType: String
    public var legacyDocumentTypeSource: String
    public var legacyDocumentTypeConfidence: Double?
    public var contentTags: [StoredChunkTag]
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
    private let database: SQLiteDatabase
    private static let vectorTableName = "chunk_vectors_vec"
    private static let vectorConfigTableName = "vector_index_config"
    private static let schemaMetadataTableName = "memory_schema_metadata"
    private static let schemaVersion = 3
    private static let legacyTableNames: Set<String> = [
        "grdb_migrations",
        "documents",
        "chunks",
        "embeddings",
        "contexts",
        "context_chunks",
        "chunks_fts",
        vectorConfigTableName,
        vectorTableName,
    ]

    public init(databaseURL: URL) throws {
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        self.database = try Self.openDatabase(at: databaseURL)
    }

    public func wipeIndexData() throws {
        try database.transaction {
            try database.execute(sql: "DROP TABLE IF EXISTS \(Self.vectorTableName)")
            try database.execute(sql: "DELETE FROM \(Self.vectorConfigTableName)")
            try database.execute(sql: "DELETE FROM context_chunks")
            try database.execute(sql: "DELETE FROM embeddings")
            try database.execute(sql: "DELETE FROM chunks")
            try database.execute(sql: "DELETE FROM documents")
        }
    }

    public func replaceDocument(_ input: StoredDocumentInput) throws {
        try database.transaction {
            let existingChunkIDs = try database.fetchAll(
                sql: """
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.path = ?
                """,
                arguments: [input.path],
                as: Int64.self
            )
            if !existingChunkIDs.isEmpty {
                try Self.deleteVectors(in: database, chunkIDs: existingChunkIDs)
            }

            try database.execute(sql: "DELETE FROM documents WHERE path = ?", arguments: [input.path])

            if let firstChunk = input.chunks.first {
                try Self.ensureVectorIndex(
                    in: database,
                    dimension: firstChunk.embedding.count
                )
            }

            let now = Date().timeIntervalSince1970
            try database.execute(
                sql: """
                INSERT INTO documents (
                    path,
                    title,
                    modified_at,
                    checksum,
                    memory_id,
                    memory_kind,
                    memory_status,
                    memory_canonical_key,
                    memory_type,
                    memory_type_source,
                    memory_type_confidence,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                arguments: [
                    input.path,
                    input.title,
                    input.modifiedAt.timeIntervalSince1970,
                    input.checksum,
                    input.memoryID,
                    input.memoryKind,
                    input.memoryStatus,
                    input.memoryCanonicalKey,
                    input.memoryType,
                    input.memoryTypeSource,
                    input.memoryTypeConfidence,
                    now,
                    now,
                ]
            )

            let documentID = database.lastInsertRowID
            for chunk in input.chunks {
                try database.execute(
                    sql: """
                    INSERT INTO chunks (
                        document_id,
                        ordinal,
                        content,
                        token_count,
                        memory_type_override,
                        memory_type_override_source,
                        memory_type_override_confidence,
                        memory_kind,
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
                        chunk.memoryKind,
                        chunk.importance,
                        chunk.accessCount,
                        chunk.lastAccessedAt?.timeIntervalSince1970,
                        chunk.source,
                        Self.encodeContentTags(chunk.contentTags),
                        (chunk.createdAt ?? Date()).timeIntervalSince1970,
                    ]
                )

                let chunkID = database.lastInsertRowID
                if let configuredDimension = try Self.configuredVectorDimension(in: database),
                   configuredDimension != chunk.embedding.count {
                    throw SQLiteError(
                        message: "Embedding dimension mismatch. Expected \(configuredDimension), got \(chunk.embedding.count)."
                    )
                }

                try database.execute(
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

                if try Self.vectorTableExists(in: database) {
                    try database.execute(
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

        try database.transaction {
            let chunkIDs = try database.fetchAll(
                sql: """
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.path IN (\(SQLiteDatabase.placeholders(count: paths.count)))
                """,
                arguments: paths,
                as: Int64.self
            )
            if !chunkIDs.isEmpty {
                try Self.deleteVectors(in: database, chunkIDs: chunkIDs)
            }

            let sql = "DELETE FROM documents WHERE path IN (\(SQLiteDatabase.placeholders(count: paths.count)))"
            try database.execute(sql: sql, arguments: paths)
        }
    }

    public func fetchAllChunkEmbeddings() throws -> [StoredChunkEmbedding] {
        let rows = try database.fetchAll(
            sql: """
            SELECT
                c.id AS chunk_id,
                c.content AS content,
                d.path AS document_path,
                d.title AS title,
                d.modified_at AS modified_at,
                d.memory_id AS memory_id,
                d.memory_kind AS memory_kind,
                d.memory_status AS memory_status,
                d.memory_canonical_key AS memory_canonical_key,
                COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
                COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
                COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
                COALESCE(c.memory_kind, d.memory_kind, '') AS memory_kind_fallback,
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

        return rows.compactMap(Self.makeChunkEmbedding(from:))
    }

    public func fetchChunkMetadata(chunkIDs: [Int64]) throws -> [StoredChunkMetadata] {
        guard !chunkIDs.isEmpty else { return [] }

        let sql = """
        SELECT
            c.id AS chunk_id,
            c.content AS content,
            d.path AS document_path,
            d.title AS title,
            d.modified_at AS modified_at,
            d.memory_id AS memory_id,
            d.memory_kind AS memory_kind,
            d.memory_status AS memory_status,
            d.memory_canonical_key AS memory_canonical_key,
            COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
            COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
            COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
            COALESCE(c.memory_kind, d.memory_kind, '') AS memory_kind_fallback,
            c.importance AS importance,
            c.access_count AS access_count,
            c.last_accessed_at AS last_accessed_at,
            c.source AS source,
            c.created_at AS created_at,
            COALESCE(c.content_tags_json, '[]') AS content_tags_json
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id IN (\(SQLiteDatabase.placeholders(count: chunkIDs.count)))
        """

        return try database.fetchAll(sql: sql, arguments: chunkIDs).map(Self.makeChunkMetadata(from:))
    }

    public func fetchChunkMetadata(chunkID: Int64) throws -> StoredChunkMetadata? {
        let row = try database.fetchOne(
            sql: """
            SELECT
                c.id AS chunk_id,
                c.content AS content,
                d.path AS document_path,
                d.title AS title,
                d.modified_at AS modified_at,
                d.memory_id AS memory_id,
                d.memory_kind AS memory_kind,
                d.memory_status AS memory_status,
                d.memory_canonical_key AS memory_canonical_key,
                COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
                COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
                COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
                COALESCE(c.memory_kind, d.memory_kind, '') AS memory_kind_fallback,
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

        return row.map(Self.makeChunkMetadata(from:))
    }

    public func listDocumentPaths() throws -> [String] {
        try database.fetchAll(
            sql: "SELECT path FROM documents ORDER BY path ASC",
            as: String.self
        )
    }

    public func fetchChunkMetadataForDocument(path: String) throws -> [StoredChunkMetadata] {
        try database.fetchAll(
            sql: """
            SELECT
                c.id AS chunk_id,
                c.content AS content,
                d.path AS document_path,
                d.title AS title,
                d.modified_at AS modified_at,
                d.memory_id AS memory_id,
                d.memory_kind AS memory_kind,
                d.memory_status AS memory_status,
                d.memory_canonical_key AS memory_canonical_key,
                COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
                COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
                COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
                COALESCE(c.memory_kind, d.memory_kind, '') AS memory_kind_fallback,
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
        ).map(Self.makeChunkMetadata(from:))
    }

    public func listMemoryMetadata(
        limit: Int,
        sort: StoredMemorySort,
        memoryKind: String? = nil,
        allowedMemoryTypes: Set<String>? = nil
    ) throws -> [StoredChunkMetadata] {
        guard limit > 0 else { return [] }

        var arguments: [Any?] = []
        var filters: [String] = []

        filters.append("d.memory_id IS NOT NULL")

        if let memoryKind {
            let trimmed = memoryKind.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if !trimmed.isEmpty {
                filters.append("d.memory_kind = ?")
                arguments.append(trimmed)
            }
        }

        if let allowedMemoryTypes, !allowedMemoryTypes.isEmpty {
            let orderedTypes = allowedMemoryTypes.sorted()
            filters.append("COALESCE(c.memory_type_override, d.memory_type) IN (\(SQLiteDatabase.placeholders(count: orderedTypes.count)))")
            arguments.append(contentsOf: orderedTypes)
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
            d.memory_id AS memory_id,
            d.memory_kind AS memory_kind,
            d.memory_status AS memory_status,
            d.memory_canonical_key AS memory_canonical_key,
            COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
            COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
            COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
            COALESCE(c.memory_kind, d.memory_kind, '') AS memory_kind_fallback,
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
        arguments.append(limit)

        return try database.fetchAll(sql: sql, arguments: arguments).map(Self.makeChunkMetadata(from:))
    }

    public func fetchMemoryChunkIDs(
        kinds: Set<String>? = nil,
        statuses: Set<String>? = nil
    ) throws -> [Int64] {
        var arguments: [Any?] = []
        var filters: [String] = ["d.memory_id IS NOT NULL"]

        if let kinds, !kinds.isEmpty {
            let orderedKinds = kinds.sorted()
            filters.append("d.memory_kind IN (\(SQLiteDatabase.placeholders(count: orderedKinds.count)))")
            arguments.append(contentsOf: orderedKinds)
        }

        if let statuses, !statuses.isEmpty {
            let orderedStatuses = statuses.sorted()
            filters.append("d.memory_status IN (\(SQLiteDatabase.placeholders(count: orderedStatuses.count)))")
            arguments.append(contentsOf: orderedStatuses)
        }

        return try database.fetchAll(
            sql: """
            SELECT c.id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE \(filters.joined(separator: " AND "))
            ORDER BY c.id ASC
            """,
            arguments: arguments,
            as: Int64.self
        )
    }

    public func fetchStoredMemory(id: String) throws -> StoredMemoryRecord? {
        let row = try database.fetchOne(
            sql: """
            SELECT
                m.id AS id,
                m.title AS title,
                m.kind AS kind,
                m.status AS status,
                m.canonical_key AS canonical_key,
                m.text AS text,
                m.tags_json AS tags_json,
                COALESCE(m.facet_tags_json, '[]') AS facet_tags_json,
                COALESCE(m.entities_json, '[]') AS entities_json,
                COALESCE(m.topics_json, '[]') AS topics_json,
                m.importance AS importance,
                m.confidence AS confidence,
                m.source AS source,
                m.created_at AS created_at,
                m.event_at AS event_at,
                m.updated_at AS updated_at,
                m.supersedes_id AS supersedes_id,
                m.superseded_by_id AS superseded_by_id,
                m.metadata_json AS metadata_json,
                c.id AS chunk_id,
                d.path AS document_path,
                COALESCE(c.access_count, 0) AS access_count,
                c.last_accessed_at AS last_accessed_at,
                COALESCE(c.memory_type_override, d.memory_type, 'document') AS legacy_document_type,
                COALESCE(c.memory_type_override_source, d.memory_type_source, 'system') AS legacy_document_type_source,
                COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS legacy_document_type_confidence,
                COALESCE(c.content_tags_json, '[]') AS content_tags_json
            FROM memories m
            LEFT JOIN documents d ON d.memory_id = m.id
            LEFT JOIN chunks c ON c.document_id = d.id AND c.ordinal = 0
            WHERE m.id = ?
            """,
            arguments: [id]
        )

        return row.map(Self.makeStoredMemoryRecord(from:))
    }

    public func listStoredMemories(
        limit: Int,
        sort: StoredMemorySort,
        kinds: Set<String>? = nil,
        statuses: Set<String>? = nil
    ) throws -> [StoredMemoryRecord] {
        guard limit > 0 else { return [] }

        var arguments: [Any?] = []
        var filters: [String] = []

        if let kinds, !kinds.isEmpty {
            let orderedKinds = kinds.sorted()
            filters.append("m.kind IN (\(SQLiteDatabase.placeholders(count: orderedKinds.count)))")
            arguments.append(contentsOf: orderedKinds)
        }

        if let statuses, !statuses.isEmpty {
            let orderedStatuses = statuses.sorted()
            filters.append("m.status IN (\(SQLiteDatabase.placeholders(count: orderedStatuses.count)))")
            arguments.append(contentsOf: orderedStatuses)
        }

        let whereClause = filters.isEmpty ? "" : "WHERE " + filters.joined(separator: " AND ")

        let orderClause: String
        switch sort {
        case .recent:
            orderClause = "ORDER BY m.created_at DESC, m.id DESC"
        case .importance:
            orderClause = "ORDER BY m.importance DESC, m.created_at DESC, m.id DESC"
        case .mostAccessed:
            orderClause = "ORDER BY COALESCE(c.access_count, 0) DESC, COALESCE(c.last_accessed_at, 0) DESC, m.created_at DESC, m.id DESC"
        }

        arguments.append(limit)

        return try database.fetchAll(
            sql: """
            SELECT
                m.id AS id,
                m.title AS title,
                m.kind AS kind,
                m.status AS status,
                m.canonical_key AS canonical_key,
                m.text AS text,
                m.tags_json AS tags_json,
                COALESCE(m.facet_tags_json, '[]') AS facet_tags_json,
                COALESCE(m.entities_json, '[]') AS entities_json,
                COALESCE(m.topics_json, '[]') AS topics_json,
                m.importance AS importance,
                m.confidence AS confidence,
                m.source AS source,
                m.created_at AS created_at,
                m.event_at AS event_at,
                m.updated_at AS updated_at,
                m.supersedes_id AS supersedes_id,
                m.superseded_by_id AS superseded_by_id,
                m.metadata_json AS metadata_json,
                c.id AS chunk_id,
                d.path AS document_path,
                COALESCE(c.access_count, 0) AS access_count,
                c.last_accessed_at AS last_accessed_at,
                COALESCE(c.memory_type_override, d.memory_type, 'document') AS legacy_document_type,
                COALESCE(c.memory_type_override_source, d.memory_type_source, 'system') AS legacy_document_type_source,
                COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS legacy_document_type_confidence,
                COALESCE(c.content_tags_json, '[]') AS content_tags_json
            FROM memories m
            LEFT JOIN documents d ON d.memory_id = m.id
            LEFT JOIN chunks c ON c.document_id = d.id AND c.ordinal = 0
            \(whereClause)
            \(orderClause)
            LIMIT ?
            """,
            arguments: arguments
        ).map(Self.makeStoredMemoryRecord(from:))
    }

    public func findStoredMemory(
        kind: String,
        canonicalKey: String,
        statuses: Set<String>
    ) throws -> StoredMemoryRecord? {
        let orderedStatuses = statuses.sorted()
        let row = try database.fetchOne(
            sql: """
            SELECT
                m.id AS id,
                m.title AS title,
                m.kind AS kind,
                m.status AS status,
                m.canonical_key AS canonical_key,
                m.text AS text,
                m.tags_json AS tags_json,
                COALESCE(m.facet_tags_json, '[]') AS facet_tags_json,
                COALESCE(m.entities_json, '[]') AS entities_json,
                COALESCE(m.topics_json, '[]') AS topics_json,
                m.importance AS importance,
                m.confidence AS confidence,
                m.source AS source,
                m.created_at AS created_at,
                m.event_at AS event_at,
                m.updated_at AS updated_at,
                m.supersedes_id AS supersedes_id,
                m.superseded_by_id AS superseded_by_id,
                m.metadata_json AS metadata_json,
                c.id AS chunk_id,
                d.path AS document_path,
                COALESCE(c.access_count, 0) AS access_count,
                c.last_accessed_at AS last_accessed_at,
                COALESCE(c.memory_type_override, d.memory_type, 'document') AS legacy_document_type,
                COALESCE(c.memory_type_override_source, d.memory_type_source, 'system') AS legacy_document_type_source,
                COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS legacy_document_type_confidence,
                COALESCE(c.content_tags_json, '[]') AS content_tags_json
            FROM memories m
            LEFT JOIN documents d ON d.memory_id = m.id
            LEFT JOIN chunks c ON c.document_id = d.id AND c.ordinal = 0
            WHERE m.kind = ? AND m.canonical_key = ?
              AND m.status IN (\(SQLiteDatabase.placeholders(count: orderedStatuses.count)))
            ORDER BY m.updated_at DESC
            LIMIT 1
            """,
            arguments: [kind, canonicalKey] + orderedStatuses
        )

        return row.map(Self.makeStoredMemoryRecord(from:))
    }

    public func findDuplicateStoredMemory(kind: String, text: String) throws -> StoredMemoryRecord? {
        let row = try database.fetchOne(
            sql: """
            SELECT
                m.id AS id,
                m.title AS title,
                m.kind AS kind,
                m.status AS status,
                m.canonical_key AS canonical_key,
                m.text AS text,
                m.tags_json AS tags_json,
                COALESCE(m.facet_tags_json, '[]') AS facet_tags_json,
                COALESCE(m.entities_json, '[]') AS entities_json,
                COALESCE(m.topics_json, '[]') AS topics_json,
                m.importance AS importance,
                m.confidence AS confidence,
                m.source AS source,
                m.created_at AS created_at,
                m.event_at AS event_at,
                m.updated_at AS updated_at,
                m.supersedes_id AS supersedes_id,
                m.superseded_by_id AS superseded_by_id,
                m.metadata_json AS metadata_json,
                c.id AS chunk_id,
                d.path AS document_path,
                COALESCE(c.access_count, 0) AS access_count,
                c.last_accessed_at AS last_accessed_at,
                COALESCE(c.memory_type_override, d.memory_type, 'document') AS legacy_document_type,
                COALESCE(c.memory_type_override_source, d.memory_type_source, 'system') AS legacy_document_type_source,
                COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS legacy_document_type_confidence,
                COALESCE(c.content_tags_json, '[]') AS content_tags_json
            FROM memories m
            LEFT JOIN documents d ON d.memory_id = m.id
            LEFT JOIN chunks c ON c.document_id = d.id AND c.ordinal = 0
            WHERE m.kind = ? AND m.text = ? AND m.status = 'active'
            ORDER BY m.updated_at DESC
            LIMIT 1
            """,
            arguments: [kind, text]
        )

        return row.map(Self.makeStoredMemoryRecord(from:))
    }

    public func insertStoredMemory(_ input: StoredMemoryInput) throws {
        try database.execute(
            sql: """
            INSERT INTO memories (
                id, kind, status, canonical_key, title, text, tags_json, facet_tags_json,
                entities_json, topics_json, importance, confidence, source, created_at,
                event_at, updated_at, supersedes_id, superseded_by_id, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            arguments: [
                input.id,
                input.kind,
                input.status,
                input.canonicalKey,
                input.title,
                input.text,
                Self.encodeStringArray(input.tags),
                Self.encodeStringArray(input.facetTags),
                Self.encodeStoredMemoryEntities(input.entities),
                Self.encodeStringArray(input.topics),
                input.importance,
                input.confidence,
                input.source,
                input.createdAt.timeIntervalSince1970,
                input.eventAt?.timeIntervalSince1970,
                input.updatedAt.timeIntervalSince1970,
                input.supersedesID,
                input.supersededByID,
                Self.encodeMetadata(input.metadata),
            ]
        )
    }

    public func updateStoredMemoryStatus(
        id: String,
        status: String,
        supersededByID: String? = nil,
        updatedAt: Date = Date()
    ) throws {
        try database.execute(
            sql: """
            UPDATE memories
            SET status = ?, superseded_by_id = ?, updated_at = ?
            WHERE id = ?
            """,
            arguments: [status, supersededByID, updatedAt.timeIntervalSince1970, id]
        )
    }

    public func recordChunkAccesses(_ chunkIDs: [Int64], accessedAt: Date = Date()) throws {
        guard !chunkIDs.isEmpty else { return }

        try database.transaction {
            let sql = """
            UPDATE chunks
            SET
                access_count = access_count + 1,
                last_accessed_at = ?
            WHERE id IN (\(SQLiteDatabase.placeholders(count: chunkIDs.count)))
            """
            var arguments: [Any?] = [accessedAt.timeIntervalSince1970]
            arguments.append(contentsOf: chunkIDs)
            try database.execute(sql: sql, arguments: arguments)
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

        let effectiveLimit = limit
        var mergedByChunkID: [Int64: Double] = [:]
        var seenChunkIDs: Set<Int64> = []

        if let strictPattern {
            let strictHits = try Self.runLexicalSearchQuery(
                in: database,
                pattern: strictPattern,
                limit: effectiveLimit,
                allowedChunkIDs: allowedChunkIDs,
                allowedMemoryTypes: allowedMemoryTypes,
                excludedChunkIDs: nil
            )
            for hit in strictHits {
                seenChunkIDs.insert(hit.chunkID)
                mergedByChunkID[hit.chunkID, default: 0] += hit.score * 1.2
            }
        }

        if let relaxedPattern {
            let relaxedHits = try Self.runLexicalSearchQuery(
                in: database,
                pattern: relaxedPattern,
                limit: effectiveLimit,
                allowedChunkIDs: allowedChunkIDs,
                allowedMemoryTypes: allowedMemoryTypes,
                excludedChunkIDs: nil
            )

            for hit in relaxedHits {
                seenChunkIDs.insert(hit.chunkID)
                mergedByChunkID[hit.chunkID, default: 0] += hit.score
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

    public func lexicalDocumentSearch(
        query: String,
        limit: Int,
        allowedChunkIDs: Set<Int64>? = nil,
        allowedMemoryTypes: Set<String>? = nil
    ) throws -> [LexicalHit] {
        guard limit > 0 else { return [] }
        if let allowedChunkIDs, allowedChunkIDs.isEmpty { return [] }
        if let allowedMemoryTypes, allowedMemoryTypes.isEmpty { return [] }

        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let strictPattern = Self.makeStrictFTSQuery(from: trimmed)
        let relaxedPattern = Self.makeRelaxedFTSQuery(from: trimmed)
        guard strictPattern != nil || relaxedPattern != nil else { return [] }

        let probeLimit = min(max(limit * 12, limit + 128), 2_048)
        var mergedByChunkID: [Int64: Double] = [:]
        var seenChunkIDs: Set<Int64> = []

        if let strictPattern {
            let strictHits = try Self.runLexicalSearchQuery(
                in: database,
                pattern: strictPattern,
                limit: probeLimit,
                allowedChunkIDs: allowedChunkIDs,
                allowedMemoryTypes: allowedMemoryTypes,
                excludedChunkIDs: nil
            )
            for hit in strictHits {
                seenChunkIDs.insert(hit.chunkID)
                mergedByChunkID[hit.chunkID, default: 0] += hit.score * 1.15
            }
        }

        if let relaxedPattern {
            let relaxedHits = try Self.runLexicalSearchQuery(
                in: database,
                pattern: relaxedPattern,
                limit: probeLimit,
                allowedChunkIDs: allowedChunkIDs,
                allowedMemoryTypes: allowedMemoryTypes,
                excludedChunkIDs: nil
            )
            for hit in relaxedHits {
                seenChunkIDs.insert(hit.chunkID)
                mergedByChunkID[hit.chunkID, default: 0] += hit.score
            }
        }

        guard !seenChunkIDs.isEmpty else { return [] }

        let metadataRows = try fetchChunkMetadata(chunkIDs: Array(seenChunkIDs))
        let anchorTokens = Self.lexicalAnchorTokens(from: trimmed, maxCount: 16)
        let anchorSet = Set(anchorTokens)

        struct DocumentAggregate {
            var representativeChunkID: Int64
            var representativeChunkScore: Double
            var representativeAnchorCount: Int
            var bestChunkScore: Double
            var scoreMass: Double
            var matchedChunks: Int
            var matchedAnchors: Set<String>

            mutating func add(
                chunkID: Int64,
                chunkScore: Double,
                matchedAnchorCount: Int,
                matchedAnchors: Set<String>
            ) {
                scoreMass += chunkScore
                matchedChunks += 1
                self.matchedAnchors.formUnion(matchedAnchors)
                bestChunkScore = max(bestChunkScore, chunkScore)

                if matchedAnchorCount > representativeAnchorCount
                    || (matchedAnchorCount == representativeAnchorCount && chunkScore > representativeChunkScore)
                    || (matchedAnchorCount == representativeAnchorCount
                        && chunkScore == representativeChunkScore
                        && chunkID < representativeChunkID) {
                    representativeChunkID = chunkID
                    representativeChunkScore = chunkScore
                    representativeAnchorCount = matchedAnchorCount
                }
            }
        }

        var aggregates: [String: DocumentAggregate] = [:]
        for metadata in metadataRows {
            guard let chunkScore = mergedByChunkID[metadata.chunkID] else { continue }
            let matchedAnchors = Self.lexicalAnchorMatches(
                anchorTokens: anchorSet,
                in: ((metadata.title ?? "") + " " + metadata.content)
            )
            let matchedAnchorCount = matchedAnchors.count

            if var aggregate = aggregates[metadata.documentPath] {
                aggregate.add(
                    chunkID: metadata.chunkID,
                    chunkScore: chunkScore,
                    matchedAnchorCount: matchedAnchorCount,
                    matchedAnchors: matchedAnchors
                )
                aggregates[metadata.documentPath] = aggregate
            } else {
                aggregates[metadata.documentPath] = DocumentAggregate(
                    representativeChunkID: metadata.chunkID,
                    representativeChunkScore: chunkScore,
                    representativeAnchorCount: matchedAnchorCount,
                    bestChunkScore: chunkScore,
                    scoreMass: chunkScore,
                    matchedChunks: 1,
                    matchedAnchors: matchedAnchors
                )
            }
        }

        return aggregates.values
            .map { aggregate in
                let coverage = anchorSet.isEmpty
                    ? 0
                    : Double(aggregate.matchedAnchors.count) / Double(anchorSet.count)
                let chunkAgreement = min(0.20, 0.035 * sqrt(Double(aggregate.matchedChunks)))
                let massBonus = min(0.12, aggregate.scoreMass * 0.035)
                let score = aggregate.bestChunkScore + (0.24 * coverage) + chunkAgreement + massBonus
                return LexicalHit(chunkID: aggregate.representativeChunkID, score: score)
            }
            .sorted { lhs, rhs in
                if lhs.score == rhs.score {
                    return lhs.chunkID < rhs.chunkID
                }
                return lhs.score > rhs.score
            }
            .prefix(limit)
            .map { $0 }
    }

    public func contentTagSearch(
        tagNames: [String],
        limit: Int,
        allowedChunkIDs: Set<Int64>? = nil,
        allowedMemoryTypes: Set<String>? = nil
    ) throws -> [LexicalHit] {
        guard limit > 0 else { return [] }

        let normalizedTags = Array(
            Set(
                tagNames
                    .map {
                        $0
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                            .lowercased()
                    }
                    .filter { !$0.isEmpty }
            )
        )
        .sorted()

        guard !normalizedTags.isEmpty else { return [] }
        if let allowedChunkIDs, allowedChunkIDs.isEmpty {
            return []
        }
        if let allowedMemoryTypes, allowedMemoryTypes.isEmpty {
            return []
        }

        var arguments: [Any?] = normalizedTags
        var predicates: [String] = [
            "json_extract(jt.value, '$.name') IN (\(SQLiteDatabase.placeholders(count: normalizedTags.count)))"
        ]

        if let allowedChunkIDs {
            let sortedChunkIDs = Array(allowedChunkIDs).sorted()
            predicates.append("c.id IN (\(SQLiteDatabase.placeholders(count: sortedChunkIDs.count)))")
            arguments.append(contentsOf: sortedChunkIDs)
        }

        if let allowedMemoryTypes {
            let sortedTypes = Array(allowedMemoryTypes).sorted()
            predicates.append("COALESCE(c.memory_type_override, d.memory_type) IN (\(SQLiteDatabase.placeholders(count: sortedTypes.count)))")
            arguments.append(contentsOf: sortedTypes)
        }

        arguments.append(limit)

        let rows = try database.fetchAll(
            sql: """
            SELECT
                c.id AS chunk_id,
                SUM(COALESCE(json_extract(jt.value, '$.confidence'), 0.6)) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN json_each(COALESCE(c.content_tags_json, '[]')) jt
            WHERE \(predicates.joined(separator: " AND "))
            GROUP BY c.id
            ORDER BY score DESC, c.id ASC
            LIMIT ?
            """,
            arguments: arguments
        )

        return rows.map { row in
            LexicalHit(
                chunkID: row["chunk_id"],
                score: row["score"]
            )
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

        guard try Self.vectorTableExists(in: database) else {
            return []
        }

        if let configuredDimension = try Self.configuredVectorDimension(in: database),
           configuredDimension != queryVector.count {
            return []
        }

        let overfetch = min(max(limit * 6, limit), 4_096)
        let queryBlob = Self.encodeVector(queryVector)

        // sqlite-vec can hang if we join directly against the vec virtual table.
        // Query vec results first, then hydrate/filter in a second query.
        let rows = try database.fetchAll(
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

            let memoryTypeRows = try database.fetchAll(
                sql: """
                SELECT
                    c.id AS chunk_id,
                    COALESCE(c.memory_type_override, d.memory_type) AS memory_type
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.id IN (\(SQLiteDatabase.placeholders(count: candidateIDs.count)))
                """,
                arguments: candidateIDs
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

    public func createContext(id: String, name: String) throws -> String {
        try database.transaction {
            if let existingID = try database.fetchOne(
                sql: "SELECT id FROM contexts WHERE name = ?",
                arguments: [name],
                as: String.self
            ) {
                return existingID
            }

            let now = Date().timeIntervalSince1970
            try database.execute(
                sql: "INSERT INTO contexts (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                arguments: [id, name, now, now]
            )
            return id
        }
    }

    public func addContextChunks(contextID: String, chunkIDs: [Int64]) throws {
        guard !chunkIDs.isEmpty else { return }

        try database.transaction {
            let now = Date().timeIntervalSince1970
            for chunkID in chunkIDs {
                try database.execute(
                    sql: """
                    INSERT OR REPLACE INTO context_chunks (context_id, chunk_id, added_at)
                    VALUES (?, ?, ?)
                    """,
                    arguments: [contextID, chunkID, now]
                )
            }

            try database.execute(
                sql: "UPDATE contexts SET updated_at = ? WHERE id = ?",
                arguments: [now, contextID]
            )
        }
    }

    public func clearContext(contextID: String) throws {
        try database.transaction {
            try database.execute(sql: "DELETE FROM context_chunks WHERE context_id = ?", arguments: [contextID])
            try database.execute(
                sql: "UPDATE contexts SET updated_at = ? WHERE id = ?",
                arguments: [Date().timeIntervalSince1970, contextID]
            )
        }
    }

    public func fetchContextChunkIDs(contextID: String) throws -> [Int64] {
        try database.fetchAll(
            sql: "SELECT chunk_id FROM context_chunks WHERE context_id = ?",
            arguments: [contextID],
            as: Int64.self
        )
    }

    public func listContextChunks(contextID: String) throws -> [StoredChunkMetadata] {
        try database.fetchAll(
            sql: """
            SELECT
                c.id AS chunk_id,
                c.content AS content,
                d.path AS document_path,
                d.title AS title,
                d.modified_at AS modified_at,
                d.memory_id AS memory_id,
                d.memory_kind AS memory_kind,
                d.memory_status AS memory_status,
                d.memory_canonical_key AS memory_canonical_key,
                COALESCE(c.memory_type_override, d.memory_type) AS memory_type,
                COALESCE(c.memory_type_override_source, d.memory_type_source) AS memory_type_source,
                COALESCE(c.memory_type_override_confidence, d.memory_type_confidence) AS memory_type_confidence,
                COALESCE(c.memory_kind, d.memory_kind, '') AS memory_kind_fallback,
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
        ).map(Self.makeChunkMetadata(from:))
    }

    private static func openDatabase(at databaseURL: URL) throws -> SQLiteDatabase {
        let database = try SQLiteDatabase(path: databaseURL.path)

        do {
            try configure(database)

            if try shouldResetLegacyDatabase(database) {
                try database.close()
                try deleteDatabaseFiles(at: databaseURL)

                let recreated = try SQLiteDatabase(path: databaseURL.path)
                do {
                    try configure(recreated)
                    try bootstrapSchemaIfNeeded(recreated)
                    return recreated
                } catch {
                    try? recreated.close()
                    throw error
                }
            }

            try bootstrapSchemaIfNeeded(database)
            return database
        } catch {
            try? database.close()
            throw error
        }
    }

    private static func configure(_ database: SQLiteDatabase) throws {
        try database.execute(sql: "PRAGMA foreign_keys = ON")
        try registerSQLiteVec(on: database)
    }

    private static func shouldResetLegacyDatabase(_ database: SQLiteDatabase) throws -> Bool {
        if try tableExists(Self.schemaMetadataTableName, in: database) {
            return false
        }

        let tableNames = Set(
            try database.fetchAll(
                sql: """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """,
                as: String.self
            )
        )

        return !tableNames.isDisjoint(with: Self.legacyTableNames)
    }

    private static func deleteDatabaseFiles(at databaseURL: URL) throws {
        let fileManager = FileManager.default
        let paths = [
            databaseURL.path,
            databaseURL.path + "-wal",
            databaseURL.path + "-shm",
        ]

        for path in paths where fileManager.fileExists(atPath: path) {
            try fileManager.removeItem(atPath: path)
        }
    }

    private static func bootstrapSchemaIfNeeded(_ database: SQLiteDatabase) throws {
        if try tableExists(Self.schemaMetadataTableName, in: database) {
            let version = try database.fetchOne(
                sql: "SELECT version FROM \(Self.schemaMetadataTableName) LIMIT 1",
                as: Int.self
            )
            guard let version else {
                throw SQLiteError(message: "Missing schema version.")
            }
            if version < Self.schemaVersion {
                try database.transaction {
                    try migrateSchema(in: database, from: version)
                }
                return
            }
            guard version == Self.schemaVersion else {
                throw SQLiteError(message: "Unsupported schema version \(version).")
            }
            return
        }

        try database.transaction {
            try createLatestSchema(in: database)
        }
    }

    private static func migrateSchema(in database: SQLiteDatabase, from version: Int) throws {
        var currentVersion = version
        while currentVersion < Self.schemaVersion {
            switch currentVersion {
            case 1:
                try migrateV1ToV2(in: database)
                currentVersion = 2
            case 2:
                try migrateV2ToV3(in: database)
                currentVersion = 3
            default:
                throw SQLiteError(message: "Unsupported schema migration path from version \(currentVersion).")
            }
        }
    }

    private static func createLatestSchema(in database: SQLiteDatabase) throws {
        try database.execute(
            sql: """
            CREATE TABLE \(Self.schemaMetadataTableName) (
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
                memory_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                memory_kind TEXT,
                memory_status TEXT,
                memory_canonical_key TEXT,
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
                memory_kind TEXT,
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
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                canonical_key TEXT,
                title TEXT,
                text TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                facet_tags_json TEXT NOT NULL DEFAULT '[]',
                entities_json TEXT NOT NULL DEFAULT '[]',
                topics_json TEXT NOT NULL DEFAULT '[]',
                importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL,
                source TEXT NOT NULL,
                created_at REAL NOT NULL,
                event_at REAL,
                updated_at REAL NOT NULL,
                supersedes_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                superseded_by_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        try database.execute(sql: "CREATE INDEX memories_kind_status ON memories(kind, status)")
        try database.execute(sql: "CREATE INDEX memories_canonical_key ON memories(kind, canonical_key)")

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
            CREATE TABLE \(Self.vectorConfigTableName) (
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
        try database.execute(sql: "CREATE INDEX documents_memory_id ON documents(memory_id)")
        try database.execute(sql: "CREATE INDEX documents_memory_kind_status ON documents(memory_kind, memory_status)")
        try database.execute(sql: "CREATE INDEX chunks_memory_type_override ON chunks(memory_type_override)")
        try database.execute(sql: "CREATE INDEX chunks_memory_kind ON chunks(memory_kind)")
        try database.execute(sql: "CREATE INDEX chunks_importance ON chunks(importance)")
        try database.execute(sql: "CREATE INDEX chunks_access_count ON chunks(access_count)")
        try database.execute(sql: "CREATE INDEX chunks_created_at ON chunks(created_at)")

        try database.execute(
            sql: "INSERT INTO \(Self.schemaMetadataTableName) (version) VALUES (?)",
            arguments: [Self.schemaVersion]
        )
    }

    private static func migrateV1ToV2(in database: SQLiteDatabase) throws {
        try database.execute(sql: "ALTER TABLE documents ADD COLUMN memory_id TEXT REFERENCES memories(id) ON DELETE SET NULL")
        try database.execute(sql: "ALTER TABLE documents ADD COLUMN memory_kind TEXT")
        try database.execute(sql: "ALTER TABLE documents ADD COLUMN memory_status TEXT")
        try database.execute(sql: "ALTER TABLE documents ADD COLUMN memory_canonical_key TEXT")

        try database.execute(sql: "ALTER TABLE chunks ADD COLUMN memory_kind TEXT")

        try database.execute(
            sql: """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                canonical_key TEXT,
                title TEXT,
                text TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '[]',
                importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL,
                source TEXT NOT NULL,
                created_at REAL NOT NULL,
                event_at REAL,
                updated_at REAL NOT NULL,
                supersedes_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                superseded_by_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        try database.execute(sql: "CREATE INDEX memories_kind_status ON memories(kind, status)")
        try database.execute(sql: "CREATE INDEX memories_canonical_key ON memories(kind, canonical_key)")
        try database.execute(sql: "CREATE INDEX documents_memory_id ON documents(memory_id)")
        try database.execute(sql: "CREATE INDEX documents_memory_kind_status ON documents(memory_kind, memory_status)")
        try database.execute(sql: "CREATE INDEX chunks_memory_kind ON chunks(memory_kind)")

        let legacyRows = try database.fetchAll(
            sql: """
            SELECT
                d.path AS path,
                d.title AS title,
                d.created_at AS created_at,
                d.updated_at AS updated_at,
                c.content AS content,
                c.memory_category AS memory_category,
                c.importance AS importance,
                c.source AS source
            FROM documents d
            JOIN chunks c ON c.document_id = d.id
            WHERE d.path LIKE 'memory://%' AND c.ordinal = 0
            """
        )

        for row in legacyRows {
            let path: String = row["path"]
            let memoryID = path.replacingOccurrences(of: "memory://", with: "")
            let legacyKind: String? = row["memory_category"]
            let kind = mapLegacyMemoryKind(legacyKind)
            let createdAt = Date(timeIntervalSince1970: row["created_at"])
            let updatedAt = Date(timeIntervalSince1970: row["updated_at"])
            let text: String = row["content"]
            let canonicalKey = makeCanonicalKey(kind: kind, text: text)
            try database.execute(
                sql: """
                INSERT OR REPLACE INTO memories (
                    id, kind, status, canonical_key, title, text, tags_json, importance,
                    confidence, source, created_at, event_at, updated_at, supersedes_id,
                    superseded_by_id, metadata_json
                )
                VALUES (?, ?, 'active', ?, ?, ?, '[]', ?, NULL, ?, ?, NULL, ?, NULL, NULL, '{}')
                """,
                arguments: [
                    memoryID,
                    kind,
                    canonicalKey,
                    row["title"] as String?,
                    text,
                    row["importance"] as Double,
                    row["source"] as String,
                    createdAt.timeIntervalSince1970,
                    updatedAt.timeIntervalSince1970,
                ]
            )
            try database.execute(
                sql: """
                UPDATE documents
                SET memory_id = ?, memory_kind = ?, memory_status = 'active', memory_canonical_key = ?
                WHERE path = ?
                """,
                arguments: [memoryID, kind, canonicalKey, path]
            )
            try database.execute(
                sql: """
                UPDATE chunks
                SET memory_kind = ?
                WHERE document_id = (SELECT id FROM documents WHERE path = ?)
                """,
                arguments: [kind, path]
            )
        }

        try database.execute(
            sql: "UPDATE \(Self.schemaMetadataTableName) SET version = ?",
            arguments: [2]
        )
    }

    private static func migrateV2ToV3(in database: SQLiteDatabase) throws {
        try database.execute(sql: "ALTER TABLE memories ADD COLUMN facet_tags_json TEXT NOT NULL DEFAULT '[]'")
        try database.execute(sql: "ALTER TABLE memories ADD COLUMN entities_json TEXT NOT NULL DEFAULT '[]'")
        try database.execute(sql: "ALTER TABLE memories ADD COLUMN topics_json TEXT NOT NULL DEFAULT '[]'")

        let rows = try database.fetchAll(
            sql: """
            SELECT id, kind, text, tags_json
            FROM memories
            """
        )

        for row in rows {
            let kind: String = row["kind"]
            let text: String = row["text"]
            let tags = Self.decodeStringArray(row["tags_json"])
            let facetTags = deriveFacetTags(kind: kind, text: text, tags: tags)
            try database.execute(
                sql: """
                UPDATE memories
                SET facet_tags_json = ?, entities_json = '[]', topics_json = '[]'
                WHERE id = ?
                """,
                arguments: [
                    Self.encodeStringArray(facetTags),
                    row["id"] as String,
                ]
            )
        }

        try database.execute(
            sql: "UPDATE \(Self.schemaMetadataTableName) SET version = ?",
            arguments: [Self.schemaVersion]
        )
    }

    private static func runLexicalSearchQuery(
        in database: SQLiteDatabase,
        pattern: String,
        limit: Int,
        allowedChunkIDs: Set<Int64>?,
        allowedMemoryTypes: Set<String>?,
        excludedChunkIDs: Set<Int64>?
    ) throws -> [LexicalHit] {
        guard limit > 0 else { return [] }

        var arguments: [Any?] = [pattern]
        var sql = """
        SELECT rowid AS chunk_id, rank AS rank
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        """

        if let allowedChunkIDs, !allowedChunkIDs.isEmpty {
            let orderedAllowed = allowedChunkIDs.sorted()
            sql += " AND rowid IN (\(SQLiteDatabase.placeholders(count: orderedAllowed.count)))"
            arguments.append(contentsOf: orderedAllowed)
        }

        if let excludedChunkIDs, !excludedChunkIDs.isEmpty {
            let orderedExcluded = excludedChunkIDs.sorted()
            sql += " AND rowid NOT IN (\(SQLiteDatabase.placeholders(count: orderedExcluded.count)))"
            arguments.append(contentsOf: orderedExcluded)
        }

        if let allowedMemoryTypes, !allowedMemoryTypes.isEmpty {
            let orderedTypes = allowedMemoryTypes.sorted()
            sql += """
             AND rowid IN (
                SELECT c.id
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE COALESCE(c.memory_type_override, d.memory_type) IN (\(SQLiteDatabase.placeholders(count: orderedTypes.count)))
             )
            """
            arguments.append(contentsOf: orderedTypes)
        }

        sql += " ORDER BY rank LIMIT ?"
        arguments.append(limit)

        let rows = try database.fetchAll(sql: sql, arguments: arguments)
        return rows.enumerated().map { index, row in
            let chunkID: Int64 = row["chunk_id"]
            let rank: Double? = row["rank"]
            let rawScore = -(rank ?? Double(index + 1))
            let normalizedScore = abs(rawScore) / (1.0 + abs(rawScore))
            return LexicalHit(chunkID: chunkID, score: normalizedScore)
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

        let preferred = lexicalAnchorTokens(from: query, maxCount: Int.max)
        let source = preferred.isEmpty ? tokens : preferred
        let limited = Array(source.prefix(12))
        guard !limited.isEmpty else { return nil }
        return limited.map { "\($0)*" }.joined(separator: " OR ")
    }

    private static func lexicalAnchorTokens(from query: String, maxCount: Int) -> [String] {
        guard maxCount > 0 else { return [] }
        return Array(
            lexicalTokens(from: query)
                .filter { token in
                    if token.contains(where: \.isNumber) { return true }
                    if isImportantShortToken(token) { return true }
                    if token.count >= 3 && !lexicalStopWords.contains(token) { return true }
                    return false
                }
                .prefix(maxCount)
        )
    }

    private static func lexicalAnchorMatches(anchorTokens: Set<String>, in text: String) -> Set<String> {
        guard !anchorTokens.isEmpty else { return [] }

        let textTokens = lexicalTokens(from: text)
        guard !textTokens.isEmpty else { return [] }

        var matches: Set<String> = []
        for textToken in textTokens {
            for anchor in anchorTokens where textToken.hasPrefix(anchor) || anchor.hasPrefix(textToken) {
                matches.insert(anchor)
            }
        }
        return matches
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

    private static func registerSQLiteVec(on database: SQLiteDatabase) throws {
        guard let sqliteConnection = database.sqliteHandle else {
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

    private static func tableExists(_ name: String, in database: SQLiteDatabase) throws -> Bool {
        try database.fetchOne(
            sql: """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            arguments: [name],
            as: String.self
        ) != nil
    }

    private static func vectorTableExists(in database: SQLiteDatabase) throws -> Bool {
        try tableExists(Self.vectorTableName, in: database)
    }

    private static func configuredVectorDimension(in database: SQLiteDatabase) throws -> Int? {
        try database.fetchOne(
            sql: "SELECT embedding_dim FROM \(Self.vectorConfigTableName) WHERE id = 1"
            ,
            as: Int.self
        )
    }

    private static func ensureVectorIndex(in database: SQLiteDatabase, dimension: Int) throws {
        guard dimension > 0 else {
            throw SQLiteError(message: "Embedding dimension must be greater than zero.")
        }

        let now = Date().timeIntervalSince1970
        if let existingDimension = try configuredVectorDimension(in: database) {
            guard existingDimension == dimension else {
                throw SQLiteError(
                    message: "Embedding dimension mismatch. Expected \(existingDimension), got \(dimension)."
                )
            }

            if try !vectorTableExists(in: database) {
                try createVectorTable(in: database, dimension: existingDimension)
            }

            try database.execute(
                sql: "UPDATE \(Self.vectorConfigTableName) SET updated_at = ? WHERE id = 1",
                arguments: [now]
            )
            return
        }

        try createVectorTable(in: database, dimension: dimension)
        try database.execute(
            sql: """
            INSERT INTO \(Self.vectorConfigTableName) (id, embedding_dim, created_at, updated_at)
            VALUES (1, ?, ?, ?)
            """,
            arguments: [dimension, now, now]
        )
    }

    private static func createVectorTable(in database: SQLiteDatabase, dimension: Int) throws {
        try database.execute(sql: "DROP TABLE IF EXISTS \(Self.vectorTableName)")
        try database.execute(
            sql: """
            CREATE VIRTUAL TABLE \(Self.vectorTableName) USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[\(dimension)] distance_metric=cosine
            )
            """
        )
    }

    private static func deleteVectors(in database: SQLiteDatabase, chunkIDs: [Int64]) throws {
        guard !chunkIDs.isEmpty else { return }
        guard try vectorTableExists(in: database) else { return }

        try database.execute(
            sql: "DELETE FROM \(Self.vectorTableName) WHERE chunk_id IN (\(SQLiteDatabase.placeholders(count: chunkIDs.count)))",
            arguments: chunkIDs
        )
    }

    private static func makeChunkEmbedding(from row: SQLiteRow) -> StoredChunkEmbedding? {
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
            memoryID: row["memory_id"],
            memoryKind: row["memory_kind"],
            memoryStatus: row["memory_status"],
            memoryCanonicalKey: row["memory_canonical_key"],
            memoryType: row["memory_type"],
            memoryTypeSource: row["memory_type_source"],
            memoryTypeConfidence: row["memory_type_confidence"],
            contentTags: Self.decodeContentTags(row["content_tags_json"]),
            memoryKindFallback: row["memory_kind_fallback"],
            importance: row["importance"],
            accessCount: row["access_count"],
            lastAccessedAt: Self.decodeTimestamp(row["last_accessed_at"]),
            source: row["source"],
            createdAt: Date(timeIntervalSince1970: row["created_at"])
        )
    }

    private static func makeChunkMetadata(from row: SQLiteRow) -> StoredChunkMetadata {
        StoredChunkMetadata(
            chunkID: row["chunk_id"],
            content: row["content"],
            documentPath: row["document_path"],
            title: row["title"],
            modifiedAt: Date(timeIntervalSince1970: row["modified_at"]),
            memoryID: row["memory_id"],
            memoryKind: row["memory_kind"],
            memoryStatus: row["memory_status"],
            memoryCanonicalKey: row["memory_canonical_key"],
            memoryType: row["memory_type"],
            memoryTypeSource: row["memory_type_source"],
            memoryTypeConfidence: row["memory_type_confidence"],
            contentTags: Self.decodeContentTags(row["content_tags_json"]),
            memoryKindFallback: row["memory_kind_fallback"],
            importance: row["importance"],
            accessCount: row["access_count"],
            lastAccessedAt: Self.decodeTimestamp(row["last_accessed_at"]),
            source: row["source"],
            createdAt: Date(timeIntervalSince1970: row["created_at"])
        )
    }

    private static func makeStoredMemoryRecord(from row: SQLiteRow) -> StoredMemoryRecord {
        StoredMemoryRecord(
            id: row["id"],
            title: row["title"],
            kind: row["kind"],
            status: row["status"],
            canonicalKey: row["canonical_key"],
            text: row["text"],
            tags: Self.decodeStringArray(row["tags_json"]),
            facetTags: Self.decodeStringArray(row["facet_tags_json"]),
            entities: Self.decodeStoredMemoryEntities(row["entities_json"]),
            topics: Self.decodeStringArray(row["topics_json"]),
            importance: row["importance"],
            confidence: row["confidence"],
            source: row["source"],
            createdAt: Date(timeIntervalSince1970: row["created_at"]),
            eventAt: Self.decodeTimestamp(row["event_at"]),
            updatedAt: Date(timeIntervalSince1970: row["updated_at"]),
            supersedesID: row["supersedes_id"],
            supersededByID: row["superseded_by_id"],
            metadata: Self.decodeMetadata(row["metadata_json"]),
            chunkID: row["chunk_id"],
            documentPath: row["document_path"],
            accessCount: row["access_count"],
            lastAccessedAt: Self.decodeTimestamp(row["last_accessed_at"]),
            legacyDocumentType: row["legacy_document_type"],
            legacyDocumentTypeSource: row["legacy_document_type_source"],
            legacyDocumentTypeConfidence: row["legacy_document_type_confidence"],
            contentTags: Self.decodeContentTags(row["content_tags_json"])
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

    private static func encodeStringArray(_ values: [String]) -> String {
        guard !values.isEmpty else { return "[]" }
        guard
            let data = try? JSONEncoder().encode(values),
            let encoded = String(data: data, encoding: .utf8)
        else {
            return "[]"
        }
        return encoded
    }

    private static func decodeStringArray(_ raw: String?) -> [String] {
        guard let raw else { return [] }
        guard let data = raw.data(using: .utf8) else { return [] }
        return (try? JSONDecoder().decode([String].self, from: data)) ?? []
    }

    private static func encodeStoredMemoryEntities(_ values: [StoredMemoryEntity]) -> String {
        guard !values.isEmpty else { return "[]" }
        guard
            let data = try? JSONEncoder().encode(values),
            let encoded = String(data: data, encoding: .utf8)
        else {
            return "[]"
        }
        return encoded
    }

    private static func decodeStoredMemoryEntities(_ raw: String?) -> [StoredMemoryEntity] {
        guard let raw else { return [] }
        guard let data = raw.data(using: .utf8) else { return [] }
        return (try? JSONDecoder().decode([StoredMemoryEntity].self, from: data)) ?? []
    }

    private static func encodeMetadata(_ metadata: [String: String]) -> String {
        guard !metadata.isEmpty else { return "{}" }
        guard
            let data = try? JSONEncoder().encode(metadata),
            let encoded = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return encoded
    }

    private static func decodeMetadata(_ raw: String?) -> [String: String] {
        guard let raw else { return [:] }
        guard let data = raw.data(using: .utf8) else { return [:] }
        return (try? JSONDecoder().decode([String: String].self, from: data)) ?? [:]
    }

    private static func decodeTimestamp(_ raw: Double?) -> Date? {
        guard let raw else { return nil }
        return Date(timeIntervalSince1970: raw)
    }

    private static func mapLegacyMemoryKind(_ raw: String?) -> String {
        switch raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "fact":
            return "fact"
        case "decision":
            return "decision"
        case "goal", "todo":
            return "commitment"
        case "event":
            return "episode"
        case "preference", "identity":
            return "profile"
        case "observation":
            return "handoff"
        default:
            return "fact"
        }
    }

    private static func deriveFacetTags(kind: String, text: String, tags: [String]) -> [String] {
        let normalizedKind = kind.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let loweredText = text.lowercased()
        let loweredTags = Set(tags.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() })

        var facets: [String] = []

        func add(_ raw: String) {
            guard !facets.contains(raw) else { return }
            facets.append(raw)
        }

        if loweredText.contains("prefer") || loweredText.contains("favorite") || loweredTags.contains("preference") {
            add("preference")
        }
        if loweredText.contains("project") || loweredText.contains("repo") || loweredText.contains("repository") {
            add("project")
        }
        if loweredText.contains("goal") || loweredText.contains("objective") {
            add("goal")
        }
        if loweredText.contains("todo") || loweredText.contains("task") || loweredText.contains("follow up") {
            add("task")
        }
        if loweredText.contains("tool") || loweredText.contains("sdk") || loweredText.contains("framework") || loweredText.contains("sqlite") {
            add("tool")
        }
        if loweredText.contains("today") || loweredText.contains("tomorrow") || loweredText.contains("deadline") || loweredText.contains("urgent") {
            add("time_sensitive")
        }
        if loweredText.contains("constraint") || loweredText.contains("blocked") || loweredText.contains("cannot") {
            add("constraint")
        }
        if loweredText.contains("lesson") || loweredText.contains("learned") || loweredText.contains("takeaway") {
            add("lesson")
        }
        if loweredText.contains("feel") || loweredText.contains("frustrated") || loweredText.contains("happy") {
            add("emotion")
        }
        if loweredText.contains("name") || loweredText.contains("role") || loweredText.contains("timezone") {
            add("identity_signal")
        }

        switch normalizedKind {
        case "profile":
            add("fact_about_user")
        case "fact":
            add("fact_about_world")
        case "decision":
            add("decision_topic")
        case "commitment":
            add("task")
        default:
            break
        }

        return Array(facets.prefix(6))
    }

    private static func makeCanonicalKey(kind: String, text: String) -> String? {
        let tokens = text
            .lowercased()
            .split { character in !character.isLetter && !character.isNumber }
            .map(String.init)
            .filter { $0.count >= 3 }
        guard !tokens.isEmpty else { return nil }
        let prefix = tokens.prefix(6).joined(separator: "-")
        return "\(kind):\(prefix)"
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
