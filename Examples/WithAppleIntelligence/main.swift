import Foundation
import Memory
import MemoryAppleIntelligence
import MemoryNaturalLanguage

@main
struct AppleIntelligenceExample {
    static func main() async throws {
        let dbURL = URL(fileURLWithPath: "/tmp/memory-apple-intelligence.sqlite")
        let docsURL = URL(fileURLWithPath: "/path/to/your/docs")

        var config = QMDConfiguration.naturalLanguageDefault(databaseURL: dbURL)

        if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *), AppleIntelligenceSupport.isAvailable {
            config.queryExpander = AppleIntelligenceQueryExpander()
            config.reranker = AppleIntelligenceReranker()
        }

        let index = try QMDIndex(configuration: config)
        try await index.rebuildIndex(from: [docsURL])

        let results = try await index.search(SearchQuery(text: "how do we deploy this service?"))
        print("Results: \(results.count)")
    }
}
