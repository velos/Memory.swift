import Foundation

public struct HeuristicStructuredQueryExpander: StructuredQueryExpander {
    public let identifier: String

    private let maxLexicalQueries: Int
    private let maxSemanticQueries: Int
    private let maxHypotheticalDocuments: Int
    private let maxFacetHints: Int
    private let maxEntities: Int
    private let maxTopics: Int

    public init(
        identifier: String = "heuristic-structured-query-expander",
        maxLexicalQueries: Int = 2,
        maxSemanticQueries: Int = 2,
        maxHypotheticalDocuments: Int = 1,
        maxFacetHints: Int = 4,
        maxEntities: Int = 6,
        maxTopics: Int = 6
    ) {
        self.identifier = identifier
        self.maxLexicalQueries = min(2, max(0, maxLexicalQueries))
        self.maxSemanticQueries = min(2, max(0, maxSemanticQueries))
        self.maxHypotheticalDocuments = min(1, max(0, maxHypotheticalDocuments))
        self.maxFacetHints = min(4, max(0, maxFacetHints))
        self.maxEntities = min(6, max(0, maxEntities))
        self.maxTopics = min(6, max(0, maxTopics))
    }

    public func expand(
        query: SearchQuery,
        analysis: QueryAnalysis,
        limit: Int
    ) async throws -> StructuredQueryExpansion {
        guard limit > 0 else { return StructuredQueryExpansion() }

        let trimmed = query.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return StructuredQueryExpansion() }

        let normalizedEntities = normalizeEntities(analysis.entities, maxCount: maxEntities)
        let normalizedTopics = normalizeTopics(analysis.topics, maxCount: maxTopics)
        let normalizedFacetHints = normalizeFacetHints(
            mergedFacetHints(
                analysis.facetHints,
                heuristicallyInferredFacetHints(from: trimmed)
            ),
            maxCount: maxFacetHints
        )

        let lexicalQueries = buildLexicalQueries(
            original: trimmed,
            analysis: analysis,
            entities: normalizedEntities,
            topics: normalizedTopics,
            facets: normalizedFacetHints,
            limit: min(maxLexicalQueries, limit)
        )
        let semanticQueries = buildSemanticQueries(
            original: trimmed,
            analysis: analysis,
            entities: normalizedEntities,
            topics: normalizedTopics,
            facets: normalizedFacetHints,
            limit: min(maxSemanticQueries, limit)
        )
        let hypotheticalDocuments = buildHypotheticalDocuments(
            original: trimmed,
            analysis: analysis,
            entities: normalizedEntities,
            topics: normalizedTopics,
            facets: normalizedFacetHints,
            limit: min(maxHypotheticalDocuments, limit)
        )

        return StructuredQueryExpansion(
            lexicalQueries: lexicalQueries,
            semanticQueries: semanticQueries,
            hypotheticalDocuments: hypotheticalDocuments,
            facetHints: normalizedFacetHints,
            entities: normalizedEntities,
            topics: normalizedTopics
        )
    }

    private func buildLexicalQueries(
        original: String,
        analysis: QueryAnalysis,
        entities: [MemoryEntity],
        topics: [String],
        facets: [FacetHint],
        limit: Int
    ) -> [String] {
        guard limit > 0 else { return [] }

        var queries: [String] = []
        var seen: Set<String> = [comparisonKey(for: original)]

        let prioritizedEntities = entities.prefix(2).map(\.value)
        let prioritizedTopics = topics.prefix(2)
        let prioritizedTerms = Array(
            OrderedSet(analysis.keyTerms.map(normalizeTopic).filter { !$0.isEmpty }).prefix(4)
        )
        let prioritizedFacets = facets.prefix(2).map {
            $0.tag.rawValue.replacingOccurrences(of: "_", with: " ")
        }

        let keywordRewrite = compactJoined(
            prioritizedEntities + prioritizedTopics + prioritizedTerms
        )
        appendCandidate(keywordRewrite, to: &queries, seen: &seen, limit: limit)

        let focusedRewrite = compactJoined(
            prioritizedTopics + prioritizedFacets + prioritizedEntities
        )
        appendCandidate(focusedRewrite, to: &queries, seen: &seen, limit: limit)

        if queries.count < limit, let firstTopic = prioritizedTopics.first {
            appendCandidate(firstTopic, to: &queries, seen: &seen, limit: limit)
        }

        return queries
    }

    private func buildSemanticQueries(
        original: String,
        analysis: QueryAnalysis,
        entities: [MemoryEntity],
        topics: [String],
        facets: [FacetHint],
        limit: Int
    ) -> [String] {
        guard limit > 0 else { return [] }

        let primaryTopic = topics.first ?? compactJoined(analysis.keyTerms.prefix(3))
        let entityPhrase = compactJoined(entities.prefix(2).map(\.value))
        let facetPhrase = compactJoined(
            facets.prefix(2).map { $0.tag.rawValue.replacingOccurrences(of: "_", with: " ") }
        )

        var queries: [String] = []
        var seen: Set<String> = [comparisonKey(for: original)]

        if analysis.isHowToQuery {
            appendCandidate(
                compactJoined([
                    "procedure for",
                    primaryTopic,
                    entityPhrase,
                ]),
                to: &queries,
                seen: &seen,
                limit: limit
            )
        } else {
            appendCandidate(
                compactJoined([
                    "details about",
                    primaryTopic,
                    entityPhrase,
                ]),
                to: &queries,
                seen: &seen,
                limit: limit
            )
        }

        appendCandidate(
            compactJoined([
                "relevant memory about",
                primaryTopic,
                facetPhrase,
            ]),
            to: &queries,
            seen: &seen,
            limit: limit
        )

        return queries
    }

    private func buildHypotheticalDocuments(
        original: String,
        analysis: QueryAnalysis,
        entities: [MemoryEntity],
        topics: [String],
        facets: [FacetHint],
        limit: Int
    ) -> [String] {
        guard limit > 0 else { return [] }

        let primaryTopic = topics.first ?? normalizeTopic(original)
        let entityPhrase = compactJoined(entities.prefix(2).map(\.value))
        let facetPhrase = compactJoined(
            facets.prefix(2).map { $0.tag.rawValue.replacingOccurrences(of: "_", with: " ") }
        )

        var snippets: [String] = []
        var seen: Set<String> = []
        let sentence: String
        if analysis.isHowToQuery {
            sentence = compactJoined([
                "This memory describes the procedure for",
                primaryTopic,
                entityPhrase.isEmpty ? nil : "using \(entityPhrase)",
                facetPhrase.isEmpty ? nil : "with \(facetPhrase) context",
            ]) + "."
        } else {
            sentence = compactJoined([
                "This memory covers",
                primaryTopic,
                entityPhrase.isEmpty ? nil : "involving \(entityPhrase)",
                facetPhrase.isEmpty ? nil : "with \(facetPhrase) signals",
            ]) + "."
        }

        appendCandidate(sentence, to: &snippets, seen: &seen, limit: limit)
        return snippets
    }

    private func heuristicallyInferredFacetHints(from query: String) -> [FacetHint] {
        let lowered = query.lowercased()
        var hints: [FacetHint] = []
        for (tag, needles) in facetKeywords {
            if needles.contains(where: lowered.contains) {
                hints.append(FacetHint(tag: tag, confidence: 0.82, isExplicit: true))
            }
        }
        return hints
    }

    private func mergedFacetHints(_ lhs: [FacetHint], _ rhs: [FacetHint]) -> [FacetHint] {
        lhs + rhs
    }

    private func normalizeFacetHints(_ hints: [FacetHint], maxCount: Int) -> [FacetHint] {
        guard maxCount > 0 else { return [] }

        var deduped: [FacetTag: FacetHint] = [:]
        for hint in hints {
            let candidate = FacetHint(
                tag: hint.tag,
                confidence: min(1, max(0, hint.confidence)),
                isExplicit: hint.isExplicit
            )
            if let existing = deduped[candidate.tag] {
                if candidate.confidence > existing.confidence
                    || (candidate.confidence == existing.confidence && candidate.isExplicit && !existing.isExplicit) {
                    deduped[candidate.tag] = candidate
                }
            } else {
                deduped[candidate.tag] = candidate
            }
        }

        return deduped.values
            .sorted { lhs, rhs in
                if lhs.confidence == rhs.confidence {
                    if lhs.isExplicit == rhs.isExplicit {
                        return lhs.tag.rawValue < rhs.tag.rawValue
                    }
                    return lhs.isExplicit && !rhs.isExplicit
                }
                return lhs.confidence > rhs.confidence
            }
            .prefix(maxCount)
            .map { $0 }
    }

    private func normalizeEntities(_ entities: [MemoryEntity], maxCount: Int) -> [MemoryEntity] {
        guard maxCount > 0 else { return [] }

        var deduped: [String: MemoryEntity] = [:]
        for entity in entities {
            let value = entity.value.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalizedValue = normalizeEntityValue(entity.normalizedValue.isEmpty ? value : entity.normalizedValue)
            guard !value.isEmpty, !normalizedValue.isEmpty else { continue }

            let candidate = MemoryEntity(
                label: entity.label,
                value: value,
                normalizedValue: normalizedValue,
                confidence: entity.confidence
            )
            if let existing = deduped[normalizedValue] {
                let existingConfidence = existing.confidence ?? 0
                let candidateConfidence = candidate.confidence ?? 0
                if candidateConfidence > existingConfidence {
                    deduped[normalizedValue] = candidate
                }
            } else {
                deduped[normalizedValue] = candidate
            }
        }

        return deduped.values
            .sorted { lhs, rhs in
                let left = lhs.confidence ?? 0
                let right = rhs.confidence ?? 0
                if left == right {
                    return lhs.normalizedValue < rhs.normalizedValue
                }
                return left > right
            }
            .prefix(maxCount)
            .map { $0 }
    }

    private func normalizeTopics(_ topics: [String], maxCount: Int) -> [String] {
        guard maxCount > 0 else { return [] }

        var normalized: [String] = []
        var seen: Set<String> = []
        for topic in topics {
            let candidate = normalizeTopic(topic)
            guard !candidate.isEmpty else { continue }
            guard seen.insert(candidate).inserted else { continue }
            normalized.append(candidate)
            if normalized.count >= maxCount {
                break
            }
        }
        return normalized
    }

    private func appendCandidate(
        _ raw: String?,
        to collection: inout [String],
        seen: inout Set<String>,
        limit: Int
    ) {
        guard collection.count < limit else { return }
        guard let raw else { return }

        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let key = comparisonKey(for: trimmed)
        guard !key.isEmpty, seen.insert(key).inserted else { return }
        collection.append(trimmed)
    }

    private func compactJoined<S: Sequence>(_ parts: S) -> String where S.Element == String {
        parts
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
    }

    private func compactJoined(_ parts: [String?]) -> String {
        compactJoined(parts.compactMap { $0 })
    }

    private func normalizeEntityValue(_ raw: String) -> String {
        let punctuation = CharacterSet(charactersIn: ",:;!?()[]{}\"'`")
        return raw
            .trimmingCharacters(in: .whitespacesAndNewlines.union(punctuation))
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
            .lowercased()
    }

    private func normalizeTopic(_ raw: String) -> String {
        raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
            .prefix(4)
            .joined(separator: " ")
    }

    private func comparisonKey(for raw: String) -> String {
        raw
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
    }

    private let facetKeywords: [FacetTag: [String]] = [
        .preference: ["prefer", "favorite", "like", "dislike"],
        .person: ["who", "person", "people", "contact"],
        .relationship: ["relationship", "manager", "coworker", "teammate"],
        .project: ["project", "repo", "repository", "initiative"],
        .goal: ["goal", "objective", "aim"],
        .task: ["task", "todo", "action item", "follow up"],
        .decisionTopic: ["decision", "decided", "choose", "choice"],
        .tool: ["tool", "library", "framework", "sdk"],
        .location: ["where", "location", "place", "office"],
        .timeSensitive: ["today", "tomorrow", "deadline", "schedule", "urgent"],
        .constraint: ["constraint", "blocked", "limit", "cannot"],
        .habit: ["habit", "usually", "often", "routine"],
        .factAboutUser: ["my", "i am", "i'm", "about me"],
        .factAboutWorld: ["fact", "reference", "documentation", "spec"],
        .lesson: ["lesson", "learned", "takeaway", "retrospective"],
        .emotion: ["feel", "emotion", "mood", "frustrated", "happy"],
        .identitySignal: ["identity", "role", "background", "belief"],
    ]
}

private struct OrderedSet<Element: Hashable>: Sequence {
    private var values: [Element] = []
    private var seen: Set<Element> = []

    init<S: Sequence>(_ sequence: S) where S.Element == Element {
        for value in sequence where seen.insert(value).inserted {
            values.append(value)
        }
    }

    func makeIterator() -> Array<Element>.Iterator {
        values.makeIterator()
    }
}
