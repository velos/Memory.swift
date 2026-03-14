import Foundation
import SQLite3

private let sqliteTransient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

package struct SQLiteError: Error, LocalizedError, Sendable {
    package let code: Int32
    package let message: String
    package let sql: String?

    package init(code: Int32, message: String, sql: String? = nil) {
        self.code = code
        self.message = message
        self.sql = sql
    }

    package init(message: String, sql: String? = nil) {
        self.init(code: SQLITE_ERROR, message: message, sql: sql)
    }

    package var errorDescription: String? {
        if let sql, !sql.isEmpty {
            return "SQLite error \(code): \(message) [SQL: \(sql)]"
        }
        return "SQLite error \(code): \(message)"
    }
}

package protocol SQLiteValueDecodable {
    static func decodeSQLiteValue(_ value: SQLiteValue) -> Self?
}

package enum SQLiteValue {
    case null
    case int64(Int64)
    case double(Double)
    case text(String)
    case blob(Data)
}

extension Int64: SQLiteValueDecodable {
    package static func decodeSQLiteValue(_ value: SQLiteValue) -> Int64? {
        guard case let .int64(int64) = value else { return nil }
        return int64
    }
}

extension Int: SQLiteValueDecodable {
    package static func decodeSQLiteValue(_ value: SQLiteValue) -> Int? {
        guard let int64 = Int64.decodeSQLiteValue(value) else { return nil }
        return Int(exactly: int64)
    }
}

extension Double: SQLiteValueDecodable {
    package static func decodeSQLiteValue(_ value: SQLiteValue) -> Double? {
        switch value {
        case let .double(double):
            return double
        case let .int64(int64):
            return Double(int64)
        default:
            return nil
        }
    }
}

extension String: SQLiteValueDecodable {
    package static func decodeSQLiteValue(_ value: SQLiteValue) -> String? {
        guard case let .text(text) = value else { return nil }
        return text
    }
}

extension Data: SQLiteValueDecodable {
    package static func decodeSQLiteValue(_ value: SQLiteValue) -> Data? {
        guard case let .blob(data) = value else { return nil }
        return data
    }
}

extension Bool: SQLiteValueDecodable {
    package static func decodeSQLiteValue(_ value: SQLiteValue) -> Bool? {
        guard let intValue = Int64.decodeSQLiteValue(value) else { return nil }
        return intValue != 0
    }
}

extension Optional: SQLiteValueDecodable where Wrapped: SQLiteValueDecodable {
    package static func decodeSQLiteValue(_ value: SQLiteValue) -> Optional<Wrapped>? {
        switch value {
        case .null:
            return .some(nil)
        default:
            guard let wrapped = Wrapped.decodeSQLiteValue(value) else { return nil }
            return .some(wrapped)
        }
    }
}

package struct SQLiteRow {
    private let indexedValues: [SQLiteValue]
    private let namedValues: [String: SQLiteValue]

    fileprivate init(statement: OpaquePointer) {
        let columnCount = Int(sqlite3_column_count(statement))
        var indexedValues: [SQLiteValue] = []
        indexedValues.reserveCapacity(columnCount)

        var namedValues: [String: SQLiteValue] = [:]
        namedValues.reserveCapacity(columnCount)

        for index in 0..<columnCount {
            let value = SQLiteRow.readValue(from: statement, at: Int32(index))
            indexedValues.append(value)

            if let namePointer = sqlite3_column_name(statement, Int32(index)) {
                namedValues[String(cString: namePointer)] = value
            }
        }

        self.indexedValues = indexedValues
        self.namedValues = namedValues
    }

    package subscript<T: SQLiteValueDecodable>(_ column: String) -> T {
        guard let value = namedValues[column], let decoded = T.decodeSQLiteValue(value) else {
            fatalError("Unable to decode SQLite column '\(column)' as \(T.self)")
        }
        return decoded
    }

    package func value<T: SQLiteValueDecodable>(at index: Int, as type: T.Type = T.self) -> T? {
        guard indexedValues.indices.contains(index) else { return nil }
        return T.decodeSQLiteValue(indexedValues[index])
    }

    package func required<T: SQLiteValueDecodable>(at index: Int, as type: T.Type = T.self) throws -> T {
        guard indexedValues.indices.contains(index) else {
            throw SQLiteError(message: "Missing SQLite column at index \(index)")
        }

        let value = indexedValues[index]
        guard let decoded = T.decodeSQLiteValue(value) else {
            throw SQLiteError(message: "Unable to decode SQLite column at index \(index) as \(T.self)")
        }
        return decoded
    }

    private static func readValue(from statement: OpaquePointer, at index: Int32) -> SQLiteValue {
        switch sqlite3_column_type(statement, index) {
        case SQLITE_INTEGER:
            return .int64(sqlite3_column_int64(statement, index))
        case SQLITE_FLOAT:
            return .double(sqlite3_column_double(statement, index))
        case SQLITE_TEXT:
            guard let pointer = sqlite3_column_text(statement, index) else {
                return .null
            }
            return .text(String(cString: pointer))
        case SQLITE_BLOB:
            let length = Int(sqlite3_column_bytes(statement, index))
            guard let bytes = sqlite3_column_blob(statement, index), length > 0 else {
                return .blob(Data())
            }
            return .blob(Data(bytes: bytes, count: length))
        default:
            return .null
        }
    }
}

package final class SQLiteDatabase {
    private var handle: OpaquePointer?

    package init(path: String) throws {
        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX
        let resultCode = sqlite3_open_v2(path, &handle, flags, nil)

        guard resultCode == SQLITE_OK, let handle else {
            let message = handle.map { String(cString: sqlite3_errmsg($0)) } ?? "Unable to open SQLite database."
            if let handle {
                sqlite3_close(handle)
            }
            throw SQLiteError(code: resultCode, message: message)
        }

        sqlite3_extended_result_codes(handle, 1)
        self.handle = handle
    }

    deinit {
        if let handle {
            sqlite3_close(handle)
        }
    }

    package var sqliteHandle: OpaquePointer? {
        handle
    }

    package var lastInsertRowID: Int64 {
        guard let handle else { return 0 }
        return sqlite3_last_insert_rowid(handle)
    }

    package var changesCount: Int {
        guard let handle else { return 0 }
        return Int(sqlite3_changes(handle))
    }

    package static func placeholders(count: Int) -> String {
        String(repeating: "?,", count: max(1, count)).dropLast().description
    }

    package func close() throws {
        guard let handle else { return }
        let resultCode = sqlite3_close(handle)
        guard resultCode == SQLITE_OK else {
            throw makeError(code: resultCode, sql: nil)
        }
        self.handle = nil
    }

    package func execute(sql: String, arguments: [Any?] = []) throws {
        let statement = try prepare(sql: sql)
        defer { sqlite3_finalize(statement) }

        try bind(arguments, to: statement, sql: sql)

        let resultCode = sqlite3_step(statement)
        guard resultCode == SQLITE_DONE else {
            throw makeError(code: resultCode, sql: sql)
        }
    }

    package func fetchAll(sql: String, arguments: [Any?] = []) throws -> [SQLiteRow] {
        let statement = try prepare(sql: sql)
        defer { sqlite3_finalize(statement) }

        try bind(arguments, to: statement, sql: sql)

        var rows: [SQLiteRow] = []
        while true {
            let resultCode = sqlite3_step(statement)
            switch resultCode {
            case SQLITE_ROW:
                rows.append(SQLiteRow(statement: statement))
            case SQLITE_DONE:
                return rows
            default:
                throw makeError(code: resultCode, sql: sql)
            }
        }
    }

    package func fetchOne(sql: String, arguments: [Any?] = []) throws -> SQLiteRow? {
        let statement = try prepare(sql: sql)
        defer { sqlite3_finalize(statement) }

        try bind(arguments, to: statement, sql: sql)

        let resultCode = sqlite3_step(statement)
        switch resultCode {
        case SQLITE_ROW:
            return SQLiteRow(statement: statement)
        case SQLITE_DONE:
            return nil
        default:
            throw makeError(code: resultCode, sql: sql)
        }
    }

    package func fetchAll<T: SQLiteValueDecodable>(sql: String, arguments: [Any?] = [], as type: T.Type) throws -> [T] {
        try fetchAll(sql: sql, arguments: arguments).map {
            try $0.required(at: 0, as: T.self)
        }
    }

    package func fetchOne<T: SQLiteValueDecodable>(sql: String, arguments: [Any?] = [], as type: T.Type) throws -> T? {
        guard let row = try fetchOne(sql: sql, arguments: arguments) else {
            return nil
        }
        return try row.required(at: 0, as: T.self)
    }

    package func transaction<T>(_ body: () throws -> T) throws -> T {
        try execute(sql: "BEGIN IMMEDIATE")
        do {
            let result = try body()
            try execute(sql: "COMMIT")
            return result
        } catch {
            try? execute(sql: "ROLLBACK")
            throw error
        }
    }

    private func prepare(sql: String) throws -> OpaquePointer {
        guard let handle else {
            throw SQLiteError(message: "SQLite connection is closed.", sql: sql)
        }

        var statement: OpaquePointer?
        let resultCode = sqlite3_prepare_v2(handle, sql, -1, &statement, nil)
        guard resultCode == SQLITE_OK, let statement else {
            throw makeError(code: resultCode, sql: sql)
        }
        return statement
    }

    private func bind(_ arguments: [Any?], to statement: OpaquePointer, sql: String) throws {
        for (index, argument) in arguments.enumerated() {
            try bind(argument, at: Int32(index + 1), to: statement, sql: sql)
        }
    }

    private func bind(_ argument: Any?, at index: Int32, to statement: OpaquePointer, sql: String) throws {
        let resultCode: Int32

        switch argument {
        case nil, is NSNull:
            resultCode = sqlite3_bind_null(statement, index)
        case let value as String:
            resultCode = sqlite3_bind_text(statement, index, value, -1, sqliteTransient)
        case let value as NSString:
            resultCode = sqlite3_bind_text(statement, index, value.utf8String, -1, sqliteTransient)
        case let value as Int:
            resultCode = sqlite3_bind_int64(statement, index, Int64(value))
        case let value as Int8:
            resultCode = sqlite3_bind_int64(statement, index, Int64(value))
        case let value as Int16:
            resultCode = sqlite3_bind_int64(statement, index, Int64(value))
        case let value as Int32:
            resultCode = sqlite3_bind_int64(statement, index, Int64(value))
        case let value as Int64:
            resultCode = sqlite3_bind_int64(statement, index, value)
        case let value as UInt:
            guard let intValue = Int64(exactly: value) else {
                throw SQLiteError(message: "Unsupported SQLite binding value \(value).", sql: sql)
            }
            resultCode = sqlite3_bind_int64(statement, index, intValue)
        case let value as UInt64:
            guard let intValue = Int64(exactly: value) else {
                throw SQLiteError(message: "Unsupported SQLite binding value \(value).", sql: sql)
            }
            resultCode = sqlite3_bind_int64(statement, index, intValue)
        case let value as Bool:
            resultCode = sqlite3_bind_int64(statement, index, value ? 1 : 0)
        case let value as Double:
            resultCode = sqlite3_bind_double(statement, index, value)
        case let value as Float:
            resultCode = sqlite3_bind_double(statement, index, Double(value))
        case let value as Data:
            resultCode = value.withUnsafeBytes { rawBuffer in
                sqlite3_bind_blob(statement, index, rawBuffer.baseAddress, Int32(rawBuffer.count), sqliteTransient)
            }
        case let value as NSData:
            resultCode = sqlite3_bind_blob(statement, index, value.bytes, Int32(value.length), sqliteTransient)
        case let value as Date:
            resultCode = sqlite3_bind_double(statement, index, value.timeIntervalSince1970)
        default:
            let typeDescription = argument.map { String(describing: Swift.type(of: $0)) } ?? "nil"
            throw SQLiteError(message: "Unsupported SQLite binding type \(typeDescription).", sql: sql)
        }

        guard resultCode == SQLITE_OK else {
            throw makeError(code: resultCode, sql: sql)
        }
    }

    private func makeError(code: Int32, sql: String?) -> SQLiteError {
        guard let handle else {
            return SQLiteError(code: code, message: "SQLite connection is closed.", sql: sql)
        }

        return SQLiteError(
            code: code,
            message: String(cString: sqlite3_errmsg(handle)),
            sql: sql
        )
    }
}
