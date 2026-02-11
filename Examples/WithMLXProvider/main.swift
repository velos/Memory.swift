import Foundation
import Memory
import MemoryMLX

@main
struct MLXProviderExample {
    static func main() async throws {
        let dbURL = URL(fileURLWithPath: "/tmp/qmdkit-mlx.sqlite")
        let docsURL = URL(fileURLWithPath: "/path/to/your/docs")

        // Provide your MLXEmbedders-backed batch embedding closure.
        let config = QMDConfiguration.mlxDefault(databaseURL: dbURL) { texts in
            // Replace with real MLXEmbedders model loading and inference.
            return texts.map { _ in Array(repeating: Float.zero, count: 768) }
        }

        let index = try QMDIndex(configuration: config)
        try await index.rebuildIndex(from: [docsURL])

        let results = try await index.search(SearchQuery(text: "custom embedding backend"))
        print("Results: \(results.count)")
    }
}
