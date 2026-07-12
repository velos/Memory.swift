import Foundation

internal struct GroundedQueryExpansionTerm: Sendable, Hashable {
    var text: String
    var score: Double
    var documentFrequency: Int
    var topEvidenceRank: Int
    var kind: GroundedQueryExpansionTermKind
}

internal enum GroundedQueryExpansionTermKind: Sendable, Hashable {
    case single
    case entity
    case phrase
}

internal struct GroundedQueryExpansionDecision: Sendable, Hashable {
    var shouldApply: Bool
    var reason: String
}

internal struct GroundedQueryExpansionPlan: Sendable, Hashable {
    var terms: [GroundedQueryExpansionTerm]
    var lexicalQueries: [String]
    var decision: GroundedQueryExpansionDecision
}

internal enum RuntimeGroundedQueryExpansion {
    internal static func makePlan(
        queryText: String,
        baselineResults: [SearchResult],
        configuration: GroundedQueryExpansionConfiguration
    ) -> GroundedQueryExpansionPlan {
        guard configuration.isEnabled else {
            return GroundedQueryExpansionPlan(
                terms: [],
                lexicalQueries: [],
                decision: GroundedQueryExpansionDecision(shouldApply: false, reason: "disabled")
            )
        }
        guard !baselineResults.isEmpty else {
            return GroundedQueryExpansionPlan(
                terms: [],
                lexicalQueries: [],
                decision: GroundedQueryExpansionDecision(shouldApply: false, reason: "no_results")
            )
        }

        let documents = baselineResults
            .prefix(configuration.maxFeedbackResults)
            .enumerated()
            .map { offset, result in
                feedbackDocument(
                    rank: offset + 1,
                    title: result.title,
                    documentPath: result.documentPath,
                    snippet: result.snippet,
                    content: result.content
                )
            }

        return makePlan(
            queryText: queryText,
            baselineScores: baselineResults.map(\.score),
            feedbackDocuments: Array(documents),
            configuration: configuration
        )
    }

    internal static func makePlan(
        queryText: String,
        baselineScores: [SearchScoreBreakdown],
        feedbackDocuments: [GroundedQueryExpansionDocument],
        configuration: GroundedQueryExpansionConfiguration
    ) -> GroundedQueryExpansionPlan {
        guard configuration.isEnabled else {
            return GroundedQueryExpansionPlan(
                terms: [],
                lexicalQueries: [],
                decision: GroundedQueryExpansionDecision(shouldApply: false, reason: "disabled")
            )
        }
        guard !feedbackDocuments.isEmpty else {
            return GroundedQueryExpansionPlan(
                terms: [],
                lexicalQueries: [],
                decision: GroundedQueryExpansionDecision(shouldApply: false, reason: "no_results")
            )
        }

        let terms = expansionTerms(
            query: queryText,
            documents: feedbackDocuments,
            maxTerms: configuration.maxTerms,
            termMode: configuration.termMode
        )
        let decision = expansionDecision(
            baselineScores: baselineScores,
            terms: terms
        )
        let queries = decision.shouldApply
            ? expansionQueries(
                from: terms,
                maxQueries: configuration.maxQueries,
                termsPerQuery: configuration.termsPerQuery
            )
            : []

        return GroundedQueryExpansionPlan(terms: terms, lexicalQueries: queries, decision: decision)
    }

    internal static func scoreOnlySkipReason(baselineScores: [SearchScoreBreakdown]) -> String? {
        if hasStrongRankOneConfidence(baselineScores) {
            return "strong_rank1"
        }
        guard hasWeakLexicalCoverage(baselineScores) else {
            return "strong_lexical_coverage"
        }
        guard hasSemanticFeedbackCluster(baselineScores) else {
            return "weak_semantic_cluster"
        }
        return nil
    }

    internal static func feedbackDocument(
        rank: Int,
        title: String?,
        documentPath: String,
        snippet: String,
        content: String
    ) -> GroundedQueryExpansionDocument {
        GroundedQueryExpansionDocument(
            rank: rank,
            title: cleanedTitle(title),
            filenameStem: cleanedFilenameStem(documentPath),
            snippet: cleanedFeedbackContent(snippet),
            content: String(cleanedFeedbackContent(content).prefix(700))
        )
    }

    internal static func expansionTerms(
        query: String,
        documents: [GroundedQueryExpansionDocument],
        maxTerms: Int,
        termMode: GroundedQueryExpansionTermMode
    ) -> [GroundedQueryExpansionTerm] {
        guard maxTerms > 0, !documents.isEmpty else { return [] }
        guard !isShortAmbiguousQuery(query) else { return [] }

        let originalTerms = Set(normalizedTokens(query).map(comparisonKey(for:)))
        let documentCount = documents.count
        var scores: [String: Double] = [:]
        var documentFrequency: [String: Set<Int>] = [:]
        var topEvidenceRank: [String: Int] = [:]
        var kindByTerm: [String: GroundedQueryExpansionTermKind] = [:]
        var appearsInTopTitleOrFilename: Set<String> = []

        for document in documents {
            let rankWeight = 1.0 / log2(Double(document.rank) + 2.0)
            let sections: [(text: String, weight: Double, topTitleOrFilename: Bool)] = [
                (document.title ?? "", 1.5, document.rank == 1),
                (document.filenameStem, 1.2, document.rank == 1),
                (document.snippet, 1.0, false),
                (document.content, 0.6, false),
            ]

            for section in sections where !section.text.isEmpty {
                let candidates = candidateTerms(
                    section.text,
                    originalTerms: originalTerms,
                    termMode: termMode
                )
                guard !candidates.isEmpty else { continue }

                for (candidate, kind) in candidates {
                    scores[candidate, default: 0] += section.weight * rankWeight
                    documentFrequency[candidate, default: []].insert(document.rank)
                    topEvidenceRank[candidate] = min(topEvidenceRank[candidate] ?? document.rank, document.rank)
                    kindByTerm[candidate] = mergedTermKind(kindByTerm[candidate], kind)
                    if section.topTitleOrFilename {
                        appearsInTopTitleOrFilename.insert(candidate)
                    }
                }
            }
        }

        let maxDocumentFrequency = Double(documentCount) * 0.45
        return scores.compactMap { term, score -> GroundedQueryExpansionTerm? in
            let frequency = documentFrequency[term]?.count ?? 0
            if Double(frequency) > maxDocumentFrequency,
               !appearsInTopTitleOrFilename.contains(term) {
                return nil
            }
            return GroundedQueryExpansionTerm(
                text: term,
                score: score,
                documentFrequency: frequency,
                topEvidenceRank: topEvidenceRank[term] ?? Int.max,
                kind: kindByTerm[term] ?? .single
            )
        }
        .sorted { lhs, rhs in
            if lhs.score != rhs.score {
                return lhs.score > rhs.score
            }
            if lhs.topEvidenceRank != rhs.topEvidenceRank {
                return lhs.topEvidenceRank < rhs.topEvidenceRank
            }
            if lhs.documentFrequency != rhs.documentFrequency {
                return lhs.documentFrequency > rhs.documentFrequency
            }
            return lhs.text < rhs.text
        }
        .prefix(maxTerms)
        .map { $0 }
    }

    internal static func expansionDecision(
        baselineScores: [SearchScoreBreakdown],
        terms: [GroundedQueryExpansionTerm]
    ) -> GroundedQueryExpansionDecision {
        guard !terms.isEmpty else {
            return GroundedQueryExpansionDecision(shouldApply: false, reason: "no_terms")
        }
        if let skipReason = scoreOnlySkipReason(baselineScores: baselineScores) {
            return GroundedQueryExpansionDecision(shouldApply: false, reason: skipReason)
        }
        guard hasFeedbackEvidence(terms) else {
            return GroundedQueryExpansionDecision(shouldApply: false, reason: "insufficient_feedback_evidence")
        }
        return GroundedQueryExpansionDecision(shouldApply: true, reason: "applied_guarded")
    }

    internal static func expansionQueries(
        from terms: [GroundedQueryExpansionTerm],
        maxQueries: Int,
        termsPerQuery: Int
    ) -> [String] {
        let maxTermCount = min(terms.count, maxQueries * termsPerQuery)
        guard maxTermCount > 0 else { return [] }

        let chunks = stride(from: 0, to: maxTermCount, by: termsPerQuery).compactMap { start -> String? in
            let end = min(start + termsPerQuery, maxTermCount)
            let query = terms[start..<end].map(\.text).joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return query.isEmpty ? nil : query
        }
        return Array(chunks.prefix(maxQueries))
    }

    private static func hasStrongRankOneConfidence(_ scores: [SearchScoreBreakdown]) -> Bool {
        guard let top = scores.first else { return false }
        guard hasMultipleSignals(top) else { return false }
        if top.blended >= 0.12 {
            return true
        }
        guard scores.count > 1 else {
            return false
        }
        let second = scores[1]
        let margin = top.blended - second.blended
        let relativeMargin = margin / max(abs(top.blended), 0.0001)
        return top.blended >= 0.105
            && relativeMargin >= 0.18
    }

    private static func hasMultipleSignals(_ score: SearchScoreBreakdown) -> Bool {
        if score.semantic >= 0.02, score.lexical >= 0.02 {
            return true
        }
        return score.tag + score.schema + score.temporal + score.status >= 0.03
    }

    private static func hasWeakLexicalCoverage(_ scores: [SearchScoreBreakdown]) -> Bool {
        let topScores = Array(scores.prefix(8))
        guard !topScores.isEmpty else { return false }

        let lexicalDominantCount = topScores.filter { score in
            score.lexical >= 0.05
                && score.lexical >= max(score.semantic * 1.20, 0.02)
        }.count
        if lexicalDominantCount >= 3 {
            return false
        }

        let lexicalMass = topScores.map(\.lexical).reduce(0, +)
        let semanticMass = topScores.map(\.semantic).reduce(0, +)
        return lexicalMass < max(0.08, semanticMass * 1.35)
    }

    private static func hasSemanticFeedbackCluster(_ scores: [SearchScoreBreakdown]) -> Bool {
        let semanticScores = scores
            .prefix(8)
            .map(\.semantic)
            .filter { $0 >= 0.02 }
            .sorted(by: >)
        guard let topSemantic = semanticScores.first, topSemantic >= 0.03 else {
            return false
        }
        let clusteredCount = semanticScores.filter { $0 >= topSemantic * 0.70 }.count
        return clusteredCount >= 2
    }

    private static func hasFeedbackEvidence(_ terms: [GroundedQueryExpansionTerm]) -> Bool {
        let topEvidenceTerms = terms.filter { term in
            term.topEvidenceRank <= 5 && term.score >= 1.0
        }
        guard topEvidenceTerms.count >= 2 else { return false }
        return topEvidenceTerms.contains { term in
            term.kind != .single || term.documentFrequency >= 2 || term.topEvidenceRank <= 2
        }
    }

    private static func candidateTerms(
        _ text: String,
        originalTerms: Set<String>,
        termMode: GroundedQueryExpansionTermMode
    ) -> [String: GroundedQueryExpansionTermKind] {
        let tokens = tokenMatches(text)
        var candidates: [String: GroundedQueryExpansionTermKind] = [:]

        for token in tokens where shouldKeepToken(token.normalized, originalTerms: originalTerms) {
            let kind: GroundedQueryExpansionTermKind = isEntityLikeToken(token.raw) ? .entity : .single
            guard termModeAllows(termMode, kind: kind) else { continue }
            candidates[token.normalized] = mergedTermKind(candidates[token.normalized], kind)
        }

        for length in 2...3 where tokens.count >= length {
            for start in 0...(tokens.count - length) {
                let phraseTokens = Array(tokens[start..<(start + length)])
                guard phraseTokens.allSatisfy({ shouldKeepToken($0.normalized, originalTerms: originalTerms) }) else {
                    continue
                }
                guard termModeAllows(termMode, kind: .phrase) else { continue }
                let phrase = phraseTokens.map(\.normalized).joined(separator: " ")
                candidates[phrase] = mergedTermKind(candidates[phrase], .phrase)
            }
        }

        return candidates
    }

    private static func termModeAllows(
        _ mode: GroundedQueryExpansionTermMode,
        kind: GroundedQueryExpansionTermKind
    ) -> Bool {
        switch mode {
        case .all:
            return true
        case .singleToken:
            return kind == .single || kind == .entity
        case .phraseEntity:
            return kind == .phrase || kind == .entity
        }
    }

    private static func mergedTermKind(
        _ current: GroundedQueryExpansionTermKind?,
        _ candidate: GroundedQueryExpansionTermKind
    ) -> GroundedQueryExpansionTermKind {
        guard let current else { return candidate }
        let priority: [GroundedQueryExpansionTermKind: Int] = [
            .single: 0,
            .entity: 1,
            .phrase: 2,
        ]
        return (priority[candidate] ?? 0) > (priority[current] ?? 0) ? candidate : current
    }

    private static func normalizedTokens(_ text: String) -> [String] {
        tokenMatches(text).map(\.normalized)
    }

    private static func tokenMatches(_ text: String) -> [GroundedQueryExpansionTokenMatch] {
        let pattern = #"[A-Za-z][A-Za-z0-9'-]*|\d+"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, range: range).compactMap { match in
            guard let tokenRange = Range(match.range, in: text) else { return nil }
            let raw = String(text[tokenRange])
            let token = raw
                .lowercased()
                .trimmingCharacters(in: CharacterSet(charactersIn: "_'-"))
            return token.isEmpty ? nil : GroundedQueryExpansionTokenMatch(raw: raw, normalized: token)
        }
    }

    private static func isEntityLikeToken(_ raw: String) -> Bool {
        let scalars = Array(raw.unicodeScalars)
        guard scalars.count > 1 else { return false }
        let uppercase = CharacterSet.uppercaseLetters
        let lowercase = CharacterSet.lowercaseLetters
        let hasUppercase = scalars.contains { uppercase.contains($0) }
        guard hasUppercase else { return false }

        let first = scalars[0]
        let hasInternalUppercase = scalars.dropFirst().contains { uppercase.contains($0) }
        let hasLowercase = scalars.contains { lowercase.contains($0) }
        let allCaps = !hasLowercase && scalars.filter { uppercase.contains($0) }.count >= 2
        return hasInternalUppercase || allCaps || (uppercase.contains(first) && raw.contains("-"))
    }

    private static func shouldKeepToken(_ token: String, originalTerms: Set<String>) -> Bool {
        let key = comparisonKey(for: token)
        guard token.count > 1 else { return false }
        guard token.range(of: #"^\d+$"#, options: .regularExpression) == nil else { return false }
        guard token.range(of: #"\d"#, options: .regularExpression) == nil else { return false }
        guard token.range(of: #"^[a-f0-9]{6,}$"#, options: [.regularExpression, .caseInsensitive]) == nil else {
            return false
        }
        guard !token.contains("'") else { return false }
        guard !stopwords.contains(token) else { return false }
        guard !originalTerms.contains(key) else { return false }
        return true
    }

    private static func comparisonKey(for term: String) -> String {
        var key = term.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if key.count > 3, key.hasSuffix("s") {
            key.removeLast()
        }
        return key
    }

    private static func isShortAmbiguousQuery(_ query: String) -> Bool {
        let tokens = normalizedTokens(query)
        guard tokens.count <= 5 else { return false }
        let content = tokens.filter { !stopwords.contains($0) }
        guard !content.isEmpty, content.count <= 2 else { return false }
        let genericTerms: Set<String> = [
            "cost", "costs", "detail", "details", "info", "information", "item", "items",
            "note", "notes", "one", "option", "options", "place", "plan", "plans",
            "price", "prices", "status", "task", "tasks", "thing", "things", "time",
        ]
        return content.allSatisfy { genericTerms.contains($0) }
    }

    private static func cleanedFilenameStem(_ path: String) -> String {
        let stem = URL(fileURLWithPath: path)
            .deletingPathExtension()
            .lastPathComponent
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
        let lowered = stem.lowercased()
        if lowered.range(of: #"^(?:answer|session|doc)[-_ ].*\d"#, options: .regularExpression) != nil {
            return ""
        }
        return stem
    }

    private static func cleanedTitle(_ title: String?) -> String? {
        guard let title else { return nil }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.range(of: #"^#*\s*session\s+(?:answer|doc|session)[-_]"#, options: [.regularExpression, .caseInsensitive]) != nil {
            return nil
        }
        return trimmed
    }

    private static func cleanedFeedbackContent(_ content: String) -> String {
        content.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { line in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.range(of: #"^#*\s*session\s+(?:answer|doc|session)[-_]"#, options: [.regularExpression, .caseInsensitive]) != nil {
                    return false
                }
                if trimmed.range(of: #"^date\s*:"#, options: [.regularExpression, .caseInsensitive]) != nil {
                    return false
                }
                if trimmed.range(of: #"^##\s*turn\s+\d+\s*\((?:user|assistant)\)"#, options: [.regularExpression, .caseInsensitive]) != nil {
                    return false
                }
                return true
            }
            .joined(separator: "\n")
    }

    private static let stopwords: Set<String> = MemorySearchHeuristics.queryStopWords.union([
        "a", "above", "again", "against", "am", "because", "being", "between",
        "both", "could", "did", "doing", "down", "during", "each", "few",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers",
        "him", "his", "i", "into", "just", "me", "more", "most", "my", "no",
        "nor", "not", "now", "off", "once", "only", "other", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "then",
        "those", "through", "too", "under", "until", "very", "were", "while",
        "whom", "will", "would",
        "answer", "assistant", "consider", "date", "detail", "details", "enjoy",
        "find", "get", "give", "good", "got", "great", "abs", "im", "info",
        "information", "like", "looking", "make", "need", "new", "provide",
        "recommend", "remember", "session", "sessions", "something", "specific",
        "sure", "thing", "things", "thinking", "th", "trying", "turn", "turns",
        "user", "using", "want", "way", "wondering",
        "mon", "monday", "tue", "tues", "tuesday", "wed", "wednesday", "thu",
        "thur", "thurs", "thursday", "fri", "friday", "sat", "saturday", "sun",
        "sunday",
    ])
}

internal struct GroundedQueryExpansionDocument: Sendable, Hashable {
    var rank: Int
    var title: String?
    var filenameStem: String
    var snippet: String
    var content: String
}

private struct GroundedQueryExpansionTokenMatch {
    var raw: String
    var normalized: String
}
