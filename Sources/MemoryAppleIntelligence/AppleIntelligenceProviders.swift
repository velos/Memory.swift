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
            maximumResponseTokens: 220
        )
    ) {
        self.identifier = identifier
        self.model = model
        self.maxCandidates = max(1, maxCandidates)
        self.options = options
        self.responseTimeoutSeconds = max(1, responseTimeoutSeconds)
        self.sessionInstructions = """
            You are a retrieval reranker.
            Return only candidate IDs ranked from most relevant to least relevant.
            Use only IDs from the provided candidate list.
            Do not include explanations or IDs not present in the candidate list.
            """
    }

    public func rerank(query: SearchQuery, candidates: [SearchResult]) async throws -> [RerankAssessment] {
        guard !candidates.isEmpty else { return [] }
        guard model.isAvailable else { return [] }

        let capped = Array(candidates.prefix(maxCandidates))
        let rankedChunkIDs: [String]
        do {
            rankedChunkIDs = try await generateRankedChunkIDs(
                query: query.text,
                candidates: capped
            )
        } catch {
            guard shouldRetryWithSmallerWindow(error: error, candidateCount: capped.count) else {
                throw error
            }
            let reducedCount = max(8, capped.count / 2)
            let reduced = Array(capped.prefix(reducedCount))
            rankedChunkIDs = try await generateRankedChunkIDs(
                query: query.text,
                candidates: reduced
            )
        }

        let allowedIDs = Set(capped.map(\.chunkID))
        var seen: Set<Int64> = []
        var deduped: [RerankAssessment] = []
        deduped.reserveCapacity(rankedChunkIDs.count)

        for (index, rawChunkID) in rankedChunkIDs.enumerated() {
            guard let chunkID = parseChunkID(rawChunkID) else { continue }
            guard allowedIDs.contains(chunkID) else { continue }
            guard !seen.contains(chunkID) else { continue }
            seen.insert(chunkID)

            deduped.append(
                RerankAssessment(
                    chunkID: chunkID,
                    relevance: rankDecay(position: index, count: rankedChunkIDs.count),
                    rationale: nil
                )
            )
        }
        guard !deduped.isEmpty else {
            throw MemoryError.search("Apple reranker returned no usable assessments")
        }
        return deduped
    }

    private func makePrompt(query: String, candidates: [SearchResult]) -> String {
        let body = candidates.map { result in
            let cleanSnippet = result.snippet
                .replacingOccurrences(of: "\n", with: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let excerpt = String(cleanSnippet.prefix(300))
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

        Return candidate IDs ranked best-to-worst for this query.
        Output only IDs from the candidate list and include each ID at most once.
        """
    }

    private func generateRankedChunkIDs(
        query: String,
        candidates: [SearchResult]
    ) async throws -> [String] {
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
            return response.content.rankedChunkIDs
        }
    }

    private func shouldRetryWithSmallerWindow(error: Error, candidateCount: Int) -> Bool {
        guard candidateCount > 8 else { return false }
        let message = error.localizedDescription.lowercased()
        return message.contains("context window")
            || message.contains("timed out")
            || message.contains("timeout")
            || message.contains("deserialize")
            || message.contains("generable")
    }

    private func parseChunkID(_ raw: String) -> Int64? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let decimal = Int64(trimmed) {
            return decimal
        }

        if let decimalRange = trimmed.range(of: #"-?\d+"#, options: .regularExpression),
           let decimal = Int64(trimmed[decimalRange]) {
            return decimal
        }

        if trimmed.hasPrefix("#") {
            let payload = String(trimmed.dropFirst())
            if let hex = Int64(payload, radix: 16) {
                return hex
            }
        }

        if trimmed.lowercased().hasPrefix("0x") {
            let payload = String(trimmed.dropFirst(2))
            if let hex = Int64(payload, radix: 16) {
                return hex
            }
        }

        if let hex = Int64(trimmed, radix: 16) {
            return hex
        }

        return nil
    }

    private func rankDecay(position: Int, count: Int) -> Double {
        guard count > 1 else { return 1.0 }
        let numerator = Double(position)
        let denominator = Double(max(1, count - 1))
        return max(0.0, 1.0 - (numerator / denominator))
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public actor AppleIntelligenceContentTagger: ContentTagger {
    private enum RelevanceSchemaState {
        case unknown
        case supported
        case unsupported
    }

    public let identifier: String

    private let model: SystemLanguageModel
    private let options: GenerationOptions
    private let maxInputCharacters: Int
    private let maxTags: Int
    private let confidenceDecayBase: Double
    private let relevanceBlendWeight: Double
    private let relevanceNoiseTolerance: Double
    private let minimumConfidence: Double
    private let relevanceTimeoutSeconds: Double
    private let responseTimeoutSeconds: Double
    private let sessionInstructions: String
    private var relevanceSchemaState: RelevanceSchemaState = .unknown

    public init(
        identifier: String = "apple-intelligence-content-tagger",
        model: SystemLanguageModel = SystemLanguageModel(useCase: .contentTagging),
        maxInputCharacters: Int = 4_000,
        maxTags: Int = 12,
        confidenceDecayBase: Double = 0.88,
        relevanceBlendWeight: Double = 0.6,
        relevanceNoiseTolerance: Double = 0.35,
        minimumConfidence: Double = 0.1,
        relevanceTimeoutSeconds: Double = 2.5,
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
        self.relevanceBlendWeight = min(1, max(0, relevanceBlendWeight))
        self.relevanceNoiseTolerance = min(1, max(0, relevanceNoiseTolerance))
        self.minimumConfidence = min(1, max(0, minimumConfidence))
        self.responseTimeoutSeconds = max(1, responseTimeoutSeconds)
        self.relevanceTimeoutSeconds = min(
            self.responseTimeoutSeconds,
            max(0.5, relevanceTimeoutSeconds)
        )
        self.sessionInstructions = """
        Produce concise topical content tags for retrieval.
        Tags must be short noun phrases.
        Return tags in descending order of usefulness.
        When possible, include an integer relevance score per tag (1 weak to 5 strong).
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
        let generatedTags = try await generateTags(prompt: prompt)

        var seen: Set<String> = []
        var ranked: [ContentTag] = []
        var lastReliableRelevance: Double?
        ranked.reserveCapacity(min(maxTags, generatedTags.count))

        for generatedTag in generatedTags {
            let rawTag = generatedTag.name
            let normalized = normalize(rawTag)
            guard !normalized.isEmpty else { continue }

            let key = normalizeKey(normalized)
            guard seen.insert(key).inserted else { continue }

            let rank = ranked.count
            let rankConfidence = rankDecayConfidence(for: rank)
            let normalizedRelevance = normalizedRelevanceScore(
                from: generatedTag.relevance,
                previous: lastReliableRelevance
            )
            if let normalizedRelevance {
                lastReliableRelevance = normalizedRelevance
            }
            let confidence = blendConfidence(
                rankConfidence: rankConfidence,
                normalizedRelevance: normalizedRelevance
            )
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

    private func blendConfidence(rankConfidence: Double, normalizedRelevance: Double?) -> Double {
        guard let normalizedRelevance else { return rankConfidence }
        let blend = (relevanceBlendWeight * normalizedRelevance) + ((1 - relevanceBlendWeight) * rankConfidence)
        return max(minimumConfidence, min(1, blend))
    }

    private func normalizedRelevanceScore(from raw: Int?, previous: Double?) -> Double? {
        guard let raw else { return nil }
        guard (1...5).contains(raw) else { return nil }

        let normalized = Double(raw - 1) / 4.0
        guard normalized.isFinite else { return nil }

        // The model sometimes emits noisy out-of-order scores. Ignore sharp
        // upward jumps and fall back to rank decay for those tags.
        if let previous, normalized > previous + relevanceNoiseTolerance {
            return nil
        }

        return normalized
    }

    private func generateTags(prompt: String) async throws -> [GeneratedContentTag] {
        switch relevanceSchemaState {
        case .supported:
            if let tags = try await generateRelevanceAwareTags(prompt: prompt) {
                return tags
            }
            relevanceSchemaState = .unsupported
            return try await generateLegacyTags(prompt: prompt)

        case .unsupported:
            return try await generateLegacyTags(prompt: prompt)

        case .unknown:
            if let tags = try await generateRelevanceAwareTags(prompt: prompt) {
                relevanceSchemaState = .supported
                return tags
            }
            relevanceSchemaState = .unsupported
            return try await generateLegacyTags(prompt: prompt)
        }
    }

    private func generateRelevanceAwareTags(prompt: String) async throws -> [GeneratedContentTag]? {
        let relevanceAware = try? await withGenerationTimeout(
            seconds: relevanceTimeoutSeconds,
            label: "\(identifier).tag.relevance"
        ) { [model, sessionInstructions, options] in
            let response = try await LanguageModelSession(model: model, instructions: sessionInstructions).respond(
                to: prompt,
                generating: ContentTaggingGenerationWithRelevance.self,
                options: options
            )
            return response.content.tags.map { GeneratedContentTag(name: $0.name, relevance: $0.relevance) }
        }

        guard let relevanceAware, !relevanceAware.isEmpty else { return nil }
        return relevanceAware
    }

    private func generateLegacyTags(prompt: String) async throws -> [GeneratedContentTag] {
        let legacy = try await withGenerationTimeout(
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
        return legacy.map { GeneratedContentTag(name: $0, relevance: nil) }
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable(description: "Alternate query phrasings for retrieval.")
private struct QueryExpansionGeneration {
    var alternates: [String]
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable(description: "Ranked candidate chunk IDs for retrieval.")
private struct RerankGeneration {
    @Guide(
        description: "Candidate chunk IDs in descending relevance order.",
        .minimumCount(1),
        .maximumCount(64)
    )
    var rankedChunkIDs: [String]
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable(description: "Content tags for retrieval ranked by usefulness.")
private struct ContentTaggingGenerationWithRelevance {
    @Guide(
        description: "Content tags sorted by descending usefulness.",
        .minimumCount(1),
        .maximumCount(24)
    )
    var tags: [ContentTaggingTagWithRelevance]
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable(description: "A retrieval tag with optional relevance.")
private struct ContentTaggingTagWithRelevance {
    @Guide(description: "Lowercase concise retrieval tag.")
    var name: String

    @Guide(description: "Optional integer relevance from 1 (weak) to 5 (strong).")
    var relevance: Int?
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
private struct GeneratedContentTag: Sendable {
    var name: String
    var relevance: Int?
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
