import Foundation
import Memory

#if canImport(FoundationModels)
import FoundationModels
#endif

public enum AppleIntelligenceSupport {
    public static var isAvailable: Bool {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) {
            return SystemLanguageModel.default.isAvailable
        }
        #endif
        return false
    }
}

#if canImport(FoundationModels)
@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public actor AppleIntelligenceQueryExpander: QueryExpander {
    public let identifier: String

    private let model: SystemLanguageModel
    private let session: LanguageModelSession
    private let options: GenerationOptions

    public init(
        identifier: String = "apple-intelligence-query-expander",
        model: SystemLanguageModel = .default,
        options: GenerationOptions = GenerationOptions(
            sampling: .greedy,
            temperature: 0.0,
            maximumResponseTokens: 220
        )
    ) {
        self.identifier = identifier
        self.model = model
        self.options = options
        self.session = LanguageModelSession(
            model: model,
            instructions: """
            You produce alternate retrieval queries for local search.
            Preserve intent. Do not add new facts.
            Keep each alternate concise and semantically close to the original query.
            """
        )
    }

    public func expand(query: SearchQuery, limit: Int) async throws -> [String] {
        guard limit > 0 else { return [] }
        guard model.isAvailable else { return [] }

        let requested = min(8, max(1, limit))
        let prompt = """
        Original query:
        \(query.text)

        Return up to \(requested) alternate phrasings that improve retrieval recall.
        Keep the original intent exactly.
        """

        let response = try await session.respond(
            to: prompt,
            generating: QueryExpansionGeneration.self,
            options: options
        )

        var seen: Set<String> = []
        var alternates: [String] = []

        for alternate in response.content.alternates {
            let trimmed = alternate.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            let key = normalize(trimmed)
            guard !seen.contains(key) else { continue }
            seen.insert(key)

            alternates.append(trimmed)
            if alternates.count >= requested {
                break
            }
        }

        return alternates
    }

    private func normalize(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public actor AppleIntelligenceReranker: Reranker {
    public let identifier: String

    private let model: SystemLanguageModel
    private let session: LanguageModelSession
    private let options: GenerationOptions
    private let maxCandidates: Int

    public init(
        identifier: String = "apple-intelligence-reranker",
        model: SystemLanguageModel = .default,
        maxCandidates: Int = 50,
        options: GenerationOptions = GenerationOptions(
            sampling: .greedy,
            temperature: 0.0,
            maximumResponseTokens: 1_200
        )
    ) {
        self.identifier = identifier
        self.model = model
        self.maxCandidates = max(1, maxCandidates)
        self.options = options
        self.session = LanguageModelSession(
            model: model,
            instructions: """
            You are a retrieval reranker.
            Score each candidate by relevance to the user query from 0.0 to 1.0.
            1.0 means directly and completely relevant.
            0.0 means irrelevant.
            """
        )
    }

    public func rerank(query: SearchQuery, candidates: [SearchResult]) async throws -> [RerankAssessment] {
        guard !candidates.isEmpty else { return [] }
        guard model.isAvailable else { return [] }

        let capped = Array(candidates.prefix(maxCandidates))
        let prompt = makePrompt(query: query.text, candidates: capped)

        let response = try await session.respond(
            to: prompt,
            generating: RerankGeneration.self,
            options: options
        )

        let allowedIDs = Set(capped.map(\.chunkID))
        var deduped: [Int64: RerankAssessment] = [:]

        for generated in response.content.assessments {
            guard let chunkID = parseChunkID(generated.chunkID) else { continue }
            guard allowedIDs.contains(chunkID) else { continue }

            let relevance = min(1, max(0, generated.relevance))
            let candidate = RerankAssessment(
                chunkID: chunkID,
                relevance: relevance,
                rationale: generated.rationale?.trimmingCharacters(in: .whitespacesAndNewlines)
            )

            if let existing = deduped[chunkID], existing.relevance >= candidate.relevance {
                continue
            }

            deduped[chunkID] = candidate
        }

        return Array(deduped.values)
    }

    private func makePrompt(query: String, candidates: [SearchResult]) -> String {
        let body = candidates.map { result in
            let cleanSnippet = result.snippet
                .replacingOccurrences(of: "\n", with: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return """
            id: \(result.chunkID)
            path: \(result.documentPath)
            snippet: \(cleanSnippet)
            """
        }
        .joined(separator: "\n---\n")

        return """
        Query:
        \(query)

        Candidates:
        \(body)

        Score every candidate from 0.0 to 1.0.
        Higher score means more useful for answering the query.
        """
    }

    private func parseChunkID(_ raw: String) -> Int64? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let decimal = Int64(trimmed) {
            return decimal
        }

        if trimmed.hasPrefix("#") {
            let payload = String(trimmed.dropFirst())
            if let hex = Int64(payload, radix: 16) {
                return hex
            }
        }

        if let hex = Int64(trimmed, radix: 16) {
            return hex
        }

        return nil
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable(description: "Alternate query phrasings for retrieval.")
private struct QueryExpansionGeneration {
    var alternates: [String]
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable(description: "Relevance assessments for candidate retrieval chunks.")
private struct RerankGeneration {
    var assessments: [RerankAssessmentGeneration]
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable(description: "A relevance score for one candidate chunk.")
private struct RerankAssessmentGeneration {
    var chunkID: String
    var relevance: Double
    var rationale: String?
}
#endif
