import CryptoKit
import Foundation
@testable import Memory

actor MockEmbeddingProvider: EmbeddingProvider {
    let identifier = "mock-embedding"
    let dimension: Int
    private let tokenizer = DefaultTokenizer()

    init(dimension: Int = 64) {
        self.dimension = dimension
    }

    func embed(texts: [String]) async throws -> [[Float]] {
        texts.map(embedding(for:))
    }

    private func embedding(for text: String) -> [Float] {
        var vector = Array(repeating: Float.zero, count: dimension)

        for token in tokenizer.tokenize(text) {
            let digest = SHA256.hash(data: Data(token.utf8))
            let index = digest.withUnsafeBytes { raw in
                Int(raw[0]) % dimension
            }
            vector[index] += 1
        }

        return vector
    }
}

actor ConstantEmbeddingProvider: EmbeddingProvider {
    let identifier = "constant-embedding"

    func embed(texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [1, 1, 1, 1] }
    }
}

func makeTemporaryDirectory(function: String = #function) throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("qmdkit-tests")
        .appendingPathComponent(function.replacingOccurrences(of: " ", with: "_"))
        .appendingPathComponent(UUID().uuidString)

    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

func writeFile(_ url: URL, _ content: String, modifiedAt: Date? = nil) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try content.write(to: url, atomically: true, encoding: .utf8)

    if let modifiedAt {
        try FileManager.default.setAttributes([.modificationDate: modifiedAt], ofItemAtPath: url.path)
    }
}
