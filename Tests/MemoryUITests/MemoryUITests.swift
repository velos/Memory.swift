import Foundation
import Memory
import Testing

struct MemoryUITests {
    @MainActor
    @Test
    func memoryDebugViewCanBeConstructed() throws {
        let root = try makeTemporaryDirectory()
        let index = try MemoryIndex(
            configuration: MemoryConfiguration(
                databaseURL: root.appendingPathComponent("index.sqlite"),
                embeddingProvider: MemoryUITestEmbeddingProvider()
            )
        )

        _ = MemoryDebugView(index: index)
    }
}

private actor MemoryUITestEmbeddingProvider: EmbeddingProvider {
    let identifier = "memory-ui-test-embedding"

    func embed(texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [1, 0, 0, 0] }
    }
}

private func makeTemporaryDirectory() throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("MemoryUITests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}
