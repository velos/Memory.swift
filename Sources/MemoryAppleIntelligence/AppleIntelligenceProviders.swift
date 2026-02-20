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

    public static var isContentTaggingAvailable: Bool {
        #if canImport(FoundationModels)
        if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) {
            return SystemLanguageModel(useCase: .contentTagging).isAvailable
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
    private let options: GenerationOptions
    private let responseTimeoutSeconds: Double
    private let sessionInstructions: String

    public init(
        identifier: String = "apple-intelligence-query-expander",
        model: SystemLanguageModel = .default,
        responseTimeoutSeconds: Double = 12,
        options: GenerationOptions = GenerationOptions(
            sampling: .greedy,
            temperature: 0.0,
            maximumResponseTokens: 220
        )
    ) {
        self.identifier = identifier
        self.model = model
        self.options = options
        self.responseTimeoutSeconds = max(1, responseTimeoutSeconds)
        self.sessionInstructions = """
            You produce alternate retrieval queries for local search.
            Preserve intent. Do not add new facts.
            Keep each alternate concise and semantically close to the original query.
            """
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

        let generatedAlternates = try await withGenerationTimeout(
            seconds: responseTimeoutSeconds,
            label: "\(identifier).expand"
        ) { [model, sessionInstructions, options] in
            let response = try await LanguageModelSession(model: model, instructions: sessionInstructions).respond(
                to: prompt,
                generating: QueryExpansionGeneration.self,
                options: options
            )
            return response.content.alternates
        }

        var seen: Set<String> = []
        var alternates: [String] = []

        for alternate in generatedAlternates {
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
    private let options: GenerationOptions
    private let maxCandidates: Int
    private let responseTimeoutSeconds: Double
    private let sessionInstructions: String

    public init(
        identifier: String = "apple-intelligence-reranker",
        model: SystemLanguageModel = .default,
        maxCandidates: Int = 120,
        responseTimeoutSeconds: Double = 15,
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
        self.responseTimeoutSeconds = max(1, responseTimeoutSeconds)
        self.sessionInstructions = """
            You are a retrieval reranker.
            Score each candidate by relevance to the user query from 0.0 to 1.0.
            1.0 means directly and completely relevant.
            0.0 means irrelevant.
            """
    }

    public func rerank(query: SearchQuery, candidates: [SearchResult]) async throws -> [RerankAssessment] {
        guard !candidates.isEmpty else { return [] }
        guard model.isAvailable else { return [] }

        let capped = Array(candidates.prefix(maxCandidates))
        let generatedAssessments: [RerankAssessmentGeneration]
        do {
            generatedAssessments = try await generateAssessments(
                query: query.text,
                candidates: capped
            )
        } catch {
            guard shouldRetryWithSmallerWindow(error: error, candidateCount: capped.count) else {
                throw error
            }
            let reducedCount = max(8, capped.count / 2)
            let reduced = Array(capped.prefix(reducedCount))
            generatedAssessments = try await generateAssessments(
                query: query.text,
                candidates: reduced
            )
        }

        let allowedIDs = Set(capped.map(\.chunkID))
        var deduped: [Int64: RerankAssessment] = [:]

        for generated in generatedAssessments {
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
            let excerpt = String(cleanSnippet.prefix(180))
            let sourceName = URL(fileURLWithPath: result.documentPath).lastPathComponent
            return """
            id: \(result.chunkID)
            path: \(sourceName)
            snippet: \(excerpt)
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

    private func generateAssessments(
        query: String,
        candidates: [SearchResult]
    ) async throws -> [RerankAssessmentGeneration] {
        let prompt = makePrompt(query: query, candidates: candidates)
        return try await withGenerationTimeout(
            seconds: responseTimeoutSeconds,
            label: "\(identifier).rerank"
        ) { [model, sessionInstructions, options] in
            let response = try await LanguageModelSession(model: model, instructions: sessionInstructions).respond(
                to: prompt,
                generating: RerankGeneration.self,
                options: options
            )
            return response.content.assessments
        }
    }

    private func shouldRetryWithSmallerWindow(error: Error, candidateCount: Int) -> Bool {
        guard candidateCount > 8 else { return false }
        let message = error.localizedDescription.lowercased()
        return message.contains("context window")
            || message.contains("timed out")
            || message.contains("timeout")
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
public actor AppleIntelligenceMemoryTypeClassifier: MemoryTypeClassifier {
    public let identifier: String

    private let model: SystemLanguageModel
    private let session: LanguageModelSession
    private let options: GenerationOptions
    private let maxInputCharacters: Int

    public init(
        identifier: String = "apple-intelligence-memory-type-classifier",
        model: SystemLanguageModel = .default,
        maxInputCharacters: Int = 6_000,
        options: GenerationOptions = GenerationOptions(
            sampling: .greedy,
            temperature: 0.0,
            maximumResponseTokens: 160
        )
    ) {
        self.identifier = identifier
        self.model = model
        self.options = options
        self.maxInputCharacters = max(1_000, maxInputCharacters)
        self.session = LanguageModelSession(
            model: model,
            instructions: """
            Classify memory content into exactly one of:
            factual, procedural, episodic, semantic, emotional, social, contextual, temporal.
            Return only the best matching type and confidence from 0.0 to 1.0.
            """
        )
    }

    public func classify(
        documentText: String,
        kind: DocumentKind,
        sourceURL: URL?
    ) async throws -> MemoryTypeAssignment? {
        guard model.isAvailable else { return nil }

        let trimmed = documentText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let excerpt = String(trimmed.prefix(maxInputCharacters))
        let location = sourceURL?.path ?? "unknown"
        let prompt = """
        Document kind: \(kind.rawValue)
        Source path: \(location)

        Content:
        \(excerpt)

        Output requirements:
        - memoryType must be one of: factual, procedural, episodic, semantic, emotional, social, contextual, temporal
        - confidence must be between 0.0 and 1.0
        """

        let response = try await session.respond(
            to: prompt,
            generating: MemoryTypeClassificationGeneration.self,
            options: options
        )

        guard let type = MemoryType.parse(response.content.memoryType) else {
            return nil
        }

        let confidence = min(1, max(0, response.content.confidence))
        return MemoryTypeAssignment(
            type: type,
            source: .automatic,
            confidence: confidence,
            classifierID: identifier
        )
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public actor AppleIntelligenceContentTagger: ContentTagger {
    public let identifier: String

    private let model: SystemLanguageModel
    private let options: GenerationOptions
    private let maxInputCharacters: Int
    private let maxTags: Int
    private let confidenceDecayBase: Double
    private let minimumConfidence: Double
    private let responseTimeoutSeconds: Double
    private let sessionInstructions: String

    public init(
        identifier: String = "apple-intelligence-content-tagger",
        model: SystemLanguageModel = SystemLanguageModel(useCase: .contentTagging),
        maxInputCharacters: Int = 4_000,
        maxTags: Int = 12,
        confidenceDecayBase: Double = 0.88,
        minimumConfidence: Double = 0.1,
        responseTimeoutSeconds: Double = 12,
        options: GenerationOptions = GenerationOptions(
            sampling: .greedy,
            temperature: 0.0,
            maximumResponseTokens: 260
        )
    ) {
        self.identifier = identifier
        self.model = model
        self.options = options
        self.maxInputCharacters = max(500, maxInputCharacters)
        self.maxTags = min(24, max(1, maxTags))
        self.confidenceDecayBase = min(0.99, max(0.5, confidenceDecayBase))
        self.minimumConfidence = min(1, max(0, minimumConfidence))
        self.responseTimeoutSeconds = max(1, responseTimeoutSeconds)
        self.sessionInstructions = """
        Produce concise topical content tags for retrieval.
        Tags must be short noun phrases.
        Return tags in descending order of usefulness.
        Avoid generic tags like "text" or "document".
        """
    }

    public func tag(text: String, kind: DocumentKind, sourceURL: URL?) async throws -> [ContentTag] {
        guard model.isAvailable else { return [] }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let excerpt = String(trimmed.prefix(maxInputCharacters))
        let location = sourceURL?.lastPathComponent ?? "unknown"
        let prompt = """
        Document kind: \(kind.rawValue)
        Source: \(location)
        Return up to \(maxTags) useful retrieval tags.

        Content:
        \(excerpt)
        """

        // Create a fresh session per request so transcript growth does not
        // exhaust the model context window during large indexing runs.
        let generatedTags = try await withGenerationTimeout(
            seconds: responseTimeoutSeconds,
            label: "\(identifier).tag"
        ) { [model, sessionInstructions, options] in
            let response = try await LanguageModelSession(model: model, instructions: sessionInstructions).respond(
                to: prompt,
                generating: ContentTaggingGeneration.self,
                options: options
            )
            return response.content.tags
        }

        var seen: Set<String> = []
        var ranked: [ContentTag] = []
        ranked.reserveCapacity(min(maxTags, generatedTags.count))

        for rawTag in generatedTags {
            let normalized = normalize(rawTag)
            guard !normalized.isEmpty else { continue }

            let key = normalizeKey(normalized)
            guard seen.insert(key).inserted else { continue }

            let rank = ranked.count
            let confidence = rankDecayConfidence(for: rank)
            ranked.append(ContentTag(name: normalized, confidence: confidence))
            if ranked.count >= maxTags { break }
        }

        return ranked
    }

    private func normalize(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        return trimmed.split(whereSeparator: \.isWhitespace).joined(separator: " ").lowercased()
    }

    private func normalizeKey(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
    }

    private func rankDecayConfidence(for rank: Int) -> Double {
        let clampedRank = max(0, rank)
        let decayed = pow(confidenceDecayBase, Double(clampedRank))
        return max(minimumConfidence, min(1, decayed))
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

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable(description: "Memory type classification for one document.")
private struct MemoryTypeClassificationGeneration {
    var memoryType: String
    var confidence: Double
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable(description: "Content tags for retrieval ranked by usefulness.")
private struct ContentTaggingGeneration {
    var tags: [String]
}

private struct AppleIntelligenceGenerationTimeoutError: Error, LocalizedError {
    let label: String
    let seconds: Double

    var errorDescription: String? {
        let formatted = String(format: "%.1f", seconds)
        return "Timed out after \(formatted)s while waiting for \(label)."
    }
}

private func withGenerationTimeout<T: Sendable>(
    seconds: Double,
    label: String,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    let timeoutNanoseconds = UInt64(max(1, Int(seconds * 1_000_000_000)))
    return try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask {
            try await operation()
        }
        group.addTask {
            try await Task.sleep(nanoseconds: timeoutNanoseconds)
            throw AppleIntelligenceGenerationTimeoutError(label: label, seconds: seconds)
        }

        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}
#endif
