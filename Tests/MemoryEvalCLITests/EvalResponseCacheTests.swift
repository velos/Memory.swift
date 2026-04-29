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

    @Test
    func recallDocumentMaterializationUsesIndexableExtensionForMarkdownPdfSources() throws {
        let root = try makeTemporaryDirectory()
        let path = materializedRecallDocumentURL(
            id: "general-v2__doc-0555",
            kind: "markdown",
            relativePath: "general-v2/repliqa/pdfs/repliqa_0/xjfymplj.pdf",
            docsRoot: root
        )

        #expect(path.pathExtension == "md")
        #expect(path.path.contains("xjfymplj.md"))
    }

    @Test
    func recallDocumentMaterializationKeepsSupportedRelativeExtensions() throws {
        let root = try makeTemporaryDirectory()
        let path = materializedRecallDocumentURL(
            id: "doc-1",
            kind: "markdown",
            relativePath: "notes/already.md",
            docsRoot: root
        )

        #expect(path.pathExtension == "md")
        #expect(path.path.contains("notes/already.md"))
    }

    @Test
    func estimatedContextTokenCountIgnoresWhitespaceOnlyText() {
        #expect(estimatedContextTokenCount("   \n\t  ") == 0)
        #expect(estimatedContextTokenCount("Alpha beta, gamma.") == 3)
    }

    @Test
    func cappedContextTokenCountRespectsDocumentAndRemainingBudgets() {
        #expect(cappedContextTokenCount(fullTokenCount: 900, remainingBudget: 4096, perDocumentTokenBudget: 384) == 384)
        #expect(cappedContextTokenCount(fullTokenCount: 900, remainingBudget: 256, perDocumentTokenBudget: 384) == 248)
        #expect(cappedContextTokenCount(fullTokenCount: 120, remainingBudget: 4096, perDocumentTokenBudget: 384) == 120)
        #expect(cappedContextTokenCount(fullTokenCount: 900, remainingBudget: 7, perDocumentTokenBudget: 384) == 0)
        #expect(cappedContextTokenCount(fullTokenCount: 900, remainingBudget: 4096, perDocumentTokenBudget: 0) == 900)
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
