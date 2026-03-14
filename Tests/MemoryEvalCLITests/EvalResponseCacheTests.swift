import Foundation
import Testing
@testable import memory_eval

struct EvalResponseCacheTests {
    @Test
    func responseCachePersistsAcrossReopen() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("eval-cache.sqlite")

        let first = try EvalResponseCache(databaseURL: dbURL)
        try await first.store(
            namespace: "unit",
            keyComponents: ["a", "b"],
            value: ["value": 42]
        )

        let reopened = try EvalResponseCache(databaseURL: dbURL)
        let cached = try await reopened.load(
            namespace: "unit",
            keyComponents: ["a", "b"],
            as: [String: Int].self
        )

        #expect(cached?["value"] == 42)
    }

    private func makeTemporaryDirectory(function: String = #function) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-eval-tests")
            .appendingPathComponent(function.replacingOccurrences(of: " ", with: "_"))
            .appendingPathComponent(UUID().uuidString)

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
