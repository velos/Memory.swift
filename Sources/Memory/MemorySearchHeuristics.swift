import Foundation

internal enum MemorySearchHeuristics {
    internal static let queryStopWords: Set<String> = [
        "about", "after", "all", "also", "an", "and", "any", "are", "as", "at",
        "be", "been", "before", "but", "by", "can", "do", "does", "for", "from",
        "how", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our",
        "that", "the", "their", "them", "there", "these", "they", "this", "to",
        "up", "was", "we", "what", "when", "where", "which", "who", "why", "with", "you", "your"
    ]

    private static let temporalCueWords: [String] = [
        "when", "timeline", "chronology", "chronological", "date", "dates",
        "schedule", "scheduled", "milestone", "kickoff", "kick-off",
        "before", "after", "between", "first", "earliest", "latest",
        "order of", "most recent", "recently", "past month", "past two months",
        "valentine", "valentine's",
        "jan", "january", "feb", "february", "mar", "march", "apr", "april",
        "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
        "oct", "october", "nov", "november", "dec", "december",
        "today", "yesterday", "tomorrow"
    ]

    internal static func normalizedComparisonKey(for text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .joined(separator: " ")
    }

    internal static func isTemporalOrAggregateRecallQuery(_ queryText: String) -> Bool {
        let lower = queryText.lowercased()
        if isTimeAnchoredQuery(queryText) {
            return true
        }

        let recallIntentPhrases = [
            "as of", "count", "days passed", "first", "from earliest to latest",
            "how many", "how much", "last time", "most recently", "order of",
            "what month", "when did", "which date"
        ]
        return recallIntentPhrases.contains { lower.contains($0) }
    }

    internal static func isDirectContinuationLookupQuery(_ queryText: String) -> Bool {
        let lower = queryText.lowercased()
        if containsAny(
            lower,
            needles: [
                "how many", "different", "order of", "earliest to latest",
                "first to last", " in total "
            ]
        ) {
            return false
        }

        return lower.hasPrefix("which ")
            || lower.hasPrefix("whose ")
            || lower.hasPrefix("who ")
            || lower.hasPrefix("what time ")
            || lower.hasPrefix("at which ")
            || lower.hasPrefix("how long ")
            || lower.contains("most recently")
            || lower.contains("last tuesday")
            || lower.contains("wake up")
    }

    internal static func isMultiEvidenceSupportQuery(_ queryText: String) -> Bool {
        let lower = queryText.lowercased()
        let aggregatePhrases = [
            "average", "combined", "different", "from earliest to latest",
            "from first to last", "how many", "how much", "in total",
            "including", "list all", "order of", "please list", "total",
            "typical week"
        ]
        if aggregatePhrases.contains(where: lower.contains) {
            return true
        }

        let temporalSupportPhrases = [
            "after the", "before the", "day before", "days before",
            "earliest to latest", "first to last", "last week",
            "past month", "past few months", "past two months",
            "these three days"
        ]
        if temporalSupportPhrases.contains(where: lower.contains) {
            return true
        }

        let broadSupportPhrases = [
            "all activities", "conducted or planned", "creative attempts",
            "driving factor", "in which occasions",
            "key progress", "multi-day communication", "provide a brief description",
            "related preparations", "specific activities", "specific occasions",
            "systematic learning", "what activities", "what adjustments",
            "what does this reflect", "what preparations", "which instances",
            "which occasions"
        ]
        if broadSupportPhrases.contains(where: lower.contains) {
            return true
        }

        if lower.range(of: #"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b.*\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b"#, options: .regularExpression) != nil {
            return true
        }
        if lower.range(of: #"\bfrom\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\s+to\s+\d{1,2}(?:st|nd|rd|th)?\b"#, options: .regularExpression) != nil {
            return true
        }

        if lower.range(of: #"\b(which|what)\s+(two|three|four|five|six)\b"#, options: .regularExpression) != nil {
            return true
        }
        if lower.range(of: #"\b(the|my)\s+(two|three|four|five|six)\b"#, options: .regularExpression) != nil {
            return true
        }

        return false
    }

    internal static func isTimeAnchoredQuery(_ queryText: String) -> Bool {
        let lower = queryText.lowercased()

        if temporalCueWords.contains(where: lower.contains) {
            return true
        }

        if lower.range(of: #"\b(19|20)\d{2}\b"#, options: .regularExpression) != nil {
            return true
        }
        if lower.range(of: #"\b\d{1,2}:\d{2}\b"#, options: .regularExpression) != nil {
            return true
        }

        return false
    }

    private static func containsAny(_ text: String, needles: [String]) -> Bool {
        needles.contains(where: text.contains)
    }
}
