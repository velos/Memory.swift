import Foundation

internal struct RecallQueryUnderstanding: Sendable, Hashable {
    internal enum Operation: String, Sendable, Hashable {
        case count
        case sum
        case duration
        case frequency
        case ordering
        case recency
        case comparison
        case currentState
        case recommendation
    }

    var originalText: String
    var normalizedText: String
    var tokens: [String]
    var coreTerms: [String]
    var operations: Set<Operation>
    var temporalIntent: RecallTemporalIntent
    var isProcedural: Bool
    var isElliptical: Bool
    var requiresEvidenceAggregation: Bool

    var isTemporalOrAggregate: Bool {
        temporalIntent != .any
            || requiresEvidenceAggregation
            || operations.contains(.ordering)
            || operations.contains(.recency)
            || operations.contains(.comparison)
            || operations.contains(.currentState)
    }

    var isEvidenceDense: Bool {
        requiresEvidenceAggregation
            || operations.contains(.ordering)
            || operations.contains(.comparison)
    }
}

internal enum RecallQueryUnderstandingAnalyzer {
    internal static func analyze(_ query: String) -> RecallQueryUnderstanding {
        let normalized = MemorySearchHeuristics.normalizedComparisonKey(for: query)
        let tokens = normalized.split(separator: " ").map(String.init)
        let lower = query.lowercased()
        let tokenSet = Set(tokens)

        var operations: Set<RecallQueryUnderstanding.Operation> = []
        if containsAny(lower, [
            "how many", "number of", "count of", "count ", "how often",
        ]) || tokenSet.contains("count") || tokenSet.contains("different") {
            operations.insert(.count)
        }
        if containsAny(lower, [
            "how much", "total", "combined", "altogether", "most money", "least money",
        ]) || !tokenSet.isDisjoint(with: ["spend", "spent", "paid", "pay", "cost", "costs", "price", "prices", "money", "amount"]) {
            operations.insert(.sum)
        }
        if containsAny(lower, [
            "how long", "days ago", "days passed", "time passed", "duration", "since ",
        ]) || !tokenSet.isDisjoint(with: ["hour", "hours", "minute", "minutes", "day", "days", "week", "weeks", "month", "months", "year", "years"]) {
            operations.insert(.duration)
        }
        if containsAny(lower, [
            "typical week", "per week", "a week", "every ", "usually", "often", "routine",
        ]) || tokens.contains(where: { $0.hasSuffix("days") && weekdayStems.contains(String($0.dropLast(1))) }) {
            operations.insert(.frequency)
        }
        if containsAny(lower, [
            "order of", "earliest", "latest", "first to last", "from first", "from earliest", "chronological", "chronology", "timeline",
        ]) {
            operations.insert(.ordering)
        }
        if containsAny(lower, [
            "most recent", "most recently", "latest", "last time", "recently",
        ]) {
            operations.insert(.recency)
        }
        if containsAny(lower, [
            "before", "after", "between", "compared", "compare", "versus", " vs ",
        ]) {
            operations.insert(.comparison)
        }
        if containsAny(lower, [
            "current", "currently", "as of", "now", "still", "these days",
        ]) {
            operations.insert(.currentState)
        }
        if containsAny(lower, [
            "recommend", "suggest", "suggestion", "suggestions", "tips", "idea", "ideas", "what to watch", "what to read",
        ]) {
            operations.insert(.recommendation)
        }

        let isProcedural = isProceduralQuery(lower: lower, tokens: tokenSet)
        let isElliptical = isEllipticalQuery(lower: lower, tokens: tokens)
        let requiresEvidenceAggregation = requiresEvidenceAggregation(lower: lower, operations: operations)
        let temporalIntent = temporalIntent(
            for: lower,
            operations: operations,
            requiresEvidenceAggregation: requiresEvidenceAggregation
        )
        let coreTerms = tokens.filter { token in
            token.count >= 2
                && !coreTermStopWords.contains(token)
                && !pronounTerms.contains(token)
        }

        return RecallQueryUnderstanding(
            originalText: query,
            normalizedText: normalized,
            tokens: tokens,
            coreTerms: Array(OrderedUniqueSequence(coreTerms).prefix(12)),
            operations: operations,
            temporalIntent: temporalIntent,
            isProcedural: isProcedural,
            isElliptical: isElliptical,
            requiresEvidenceAggregation: requiresEvidenceAggregation
        )
    }

    private static func temporalIntent(
        for lower: String,
        operations: Set<RecallQueryUnderstanding.Operation>,
        requiresEvidenceAggregation: Bool
    ) -> RecallTemporalIntent {
        if operations.contains(.recency) {
            return .mostRecent
        }
        if requiresEvidenceAggregation {
            return .count
        }
        if operations.contains(.duration)
            || operations.contains(.ordering)
            || operations.contains(.comparison)
            || MemorySearchHeuristics.isTimeAnchoredQuery(lower) {
            return .timeAnchored
        }
        return .any
    }

    private static func requiresEvidenceAggregation(
        lower: String,
        operations: Set<RecallQueryUnderstanding.Operation>
    ) -> Bool {
        if operations.contains(.count),
           containsAny(lower, [
               "how many", "number of", "count of", "count ", "different ",
           ]) {
            return true
        }
        if operations.contains(.sum),
           containsAny(lower, [
               "how much", "total", "combined", "altogether", "most money", "least money",
           ]) {
            return true
        }
        if operations.contains(.duration),
           containsAny(lower, [
               "how long", "days ago", "days passed", "time passed", "duration", "since ",
           ]) {
            return true
        }
        if operations.contains(.frequency),
           containsAny(lower, [
               "how often", "typical week", "per week", "a week", "every ", "usually", "routine",
           ]) {
            return true
        }
        return false
    }

    private static func isProceduralQuery(lower: String, tokens: Set<String>) -> Bool {
        let hasQuestionShape = lower.hasPrefix("how ")
            || lower.hasPrefix("what do ")
            || lower.hasPrefix("what should ")
            || lower.hasPrefix("can i ")
            || lower.hasPrefix("where do ")
            || lower.hasPrefix("when do ")
            || lower.hasPrefix("do i ")
        let hasProcessCue = !tokens.isDisjoint(with: proceduralCueTerms)
            || containsAny(lower, ["how to", "step by step", "what documents", "what proof", "by mail", "in person"])
        return hasProcessCue && (hasQuestionShape || lower.contains("?") || tokens.contains("apply"))
    }

    private static func isEllipticalQuery(lower: String, tokens: [String]) -> Bool {
        let tokenSet = Set(tokens)
        if tokens.count <= 7,
           lower.contains("?"),
           !tokenSet.isDisjoint(with: pronounTerms) {
            return true
        }
        if lower.hasPrefix("how do i apply") || lower.hasPrefix("can i do it") {
            return true
        }
        if tokens.count <= 5,
           lower.contains("?"),
           (lower.hasPrefix("what are the")
            || lower.hasPrefix("what is the")
            || lower.hasPrefix("what's the")
            || lower.hasPrefix("how much is the")) {
            return true
        }
        return false
    }

    private static func containsAny(_ text: String, _ needles: [String]) -> Bool {
        needles.contains { text.contains($0) }
    }

    private static let coreTermStopWords: Set<String> = MemorySearchHeuristics.queryStopWords.union([
        "a", "am", "i", "me", "my", "mine", "ours", "please", "tell", "show",
        "need", "needed", "want", "wanted", "would", "could", "should",
    ])

    private static let pronounTerms: Set<String> = [
        "it", "that", "this", "they", "them", "those", "these", "one", "ones",
    ]

    private static let proceduralCueTerms: Set<String> = [
        "apply", "application", "applications", "submit", "request", "form", "forms",
        "document", "documents", "proof", "requirement", "requirements", "required",
        "fee", "fees", "pay", "payment", "mail", "online", "office", "appointment",
        "register", "registration", "renew", "change", "update", "replace", "copy",
        "appeal", "file", "report", "contact", "send",
    ]

    private static let weekdayStems: Set<String> = [
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    ]
}

internal enum GenericQueryRewriteLexicon {
    internal static func lexicalQueries(
        query: String,
        analysis: QueryAnalysis,
        understanding: RecallQueryUnderstanding,
        maxCount: Int
    ) -> [String] {
        guard maxCount > 0 else { return [] }

        let normalizedKeyTerms = analysis.keyTerms
            .map { MemorySearchHeuristics.normalizedComparisonKey(for: $0) }
            .filter { !$0.isEmpty }
        let terms = expansionTerms(for: understanding)
        let topicTerms = analysis.topics
            .flatMap { MemorySearchHeuristics.normalizedComparisonKey(for: $0).split(separator: " ").map(String.init) }
            .filter { !$0.isEmpty }

        var candidates: [String] = []
        append(
            compactJoined(understanding.coreTerms + terms.prefix(12)),
            to: &candidates,
            maxCount: maxCount
        )
        append(
            compactJoined(normalizedKeyTerms + topicTerms.prefix(8) + terms.prefix(8)),
            to: &candidates,
            maxCount: maxCount
        )
        return candidates
    }

    internal static func expansionTerms(for understanding: RecallQueryUnderstanding) -> [String] {
        var terms: [String] = []
        for token in understanding.coreTerms {
            append(token, to: &terms)
            for synonym in tokenSynonyms[token] ?? [] {
                append(synonym, to: &terms)
            }
        }

        return terms
    }

    internal static func semanticRewrite(
        for understanding: RecallQueryUnderstanding,
        entities: [MemoryEntity],
        topics: [String]
    ) -> String? {
        let entityTerms = entities.prefix(2).map(\.value)
        let topicTerms = topics.prefix(2)
        let focus = compactJoined(entityTerms + topicTerms + understanding.coreTerms.prefix(5))
        guard !focus.isEmpty else { return nil }

        if understanding.operations.contains(.recommendation) {
            return compactJoined(["preferences prior examples", focus])
        }
        if understanding.operations.contains(.currentState) {
            return compactJoined(["current active state", focus])
        }
        return nil
    }

    private static func append(_ value: String, to values: inout [String]) {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty, !values.contains(normalized) else { return }
        values.append(normalized)
    }

    private static func append(_ value: String, to values: inout [String], maxCount: Int) {
        guard values.count < maxCount else { return }
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }
        let key = MemorySearchHeuristics.normalizedComparisonKey(for: normalized)
        guard !values.contains(where: { MemorySearchHeuristics.normalizedComparisonKey(for: $0) == key }) else { return }
        values.append(normalized)
    }

    private static func compactJoined<S: Sequence>(_ parts: S) -> String where S.Element == String {
        parts
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
    }

    private static let tokenSynonyms: [String: [String]] = [
        "apply": ["application", "submit", "request", "form"],
        "application": ["apply", "submit", "request", "form"],
        "applications": ["apply", "application", "submit", "request", "forms"],
        "submit": ["send", "file", "request", "form"],
        "request": ["apply", "submit", "ask", "form"],
        "mail": ["mailing", "postal", "send", "address"],
        "mailed": ["mail", "mailing", "postal", "sent"],
        "online": ["website", "portal", "account", "web"],
        "fee": ["fees", "cost", "payment", "pay", "amount"],
        "fees": ["fee", "cost", "payment", "pay", "amount"],
        "cost": ["price", "fee", "amount", "paid", "payment"],
        "costs": ["price", "fees", "amount", "paid", "payment"],
        "pay": ["payment", "paid", "fee", "cost"],
        "paid": ["pay", "payment", "spent", "cost"],
        "spend": ["spent", "paid", "cost", "amount", "total"],
        "spent": ["spend", "paid", "cost", "amount", "total"],
        "buy": ["bought", "purchase", "paid", "cost"],
        "bought": ["buy", "purchase", "paid", "cost"],
        "purchase": ["buy", "bought", "paid", "cost"],
        "purchased": ["buy", "bought", "paid", "cost"],
        "arrive": ["arrival", "received", "delivered"],
        "arrived": ["arrival", "received", "delivered"],
        "receive": ["received", "arrived", "delivered"],
        "received": ["receive", "arrived", "delivered"],
        "change": ["update", "modify", "correct"],
        "changed": ["update", "modified", "corrected"],
        "update": ["change", "modify", "current", "correct"],
        "copy": ["duplicate", "replacement", "record"],
        "proof": ["evidence", "document", "documents"],
        "document": ["documents", "paperwork", "record"],
        "documents": ["document", "paperwork", "records"],
        "paperwork": ["documents", "forms", "records"],
        "current": ["currently", "active", "now", "still"],
        "currently": ["current", "active", "now", "still"],
        "own": ["owned", "have", "currently"],
        "owned": ["own", "have", "currently"],
        "role": ["position", "job", "title", "current"],
        "recommend": ["suggest", "preference", "favorite", "liked", "enjoyed"],
        "suggest": ["recommend", "preference", "favorite", "idea", "tips"],
        "suggestions": ["recommendations", "ideas", "tips", "preferences"],
        "tips": ["advice", "suggestions", "recommendations"],
        "recipe": ["recipes", "meal", "cook", "cooking"],
        "recipes": ["recipe", "meal", "cook", "cooking"],
        "bake": ["baking", "recipe"],
        "baking": ["bake", "recipe"],
    ]
}

private struct OrderedUniqueSequence<Element: Hashable>: Sequence {
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

    func prefix(_ maxLength: Int) -> [Element] {
        Array(values.prefix(maxLength))
    }
}
