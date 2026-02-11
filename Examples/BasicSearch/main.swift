import Foundation
import Memory
import MemoryNaturalLanguage

@main
struct BasicSearchExample {
    static func main() async throws {
        let dbURL = URL(fileURLWithPath: "/tmp/memory-basic.sqlite")
        let docsURL = URL(fileURLWithPath: "/path/to/your/docs")

        let configuration = MemoryConfiguration.naturalLanguageDefault(databaseURL: dbURL)
        let index = try MemoryIndex(configuration: configuration)

        try await index.rebuildIndex(from: [docsURL])
        let results = try await index.search(SearchQuery(text: "hybrid retrieval"))

        for result in results {
            print("\(result.score.fused): \(result.documentPath)")
            print(result.snippet)
            print("---")
        }
    }
}
