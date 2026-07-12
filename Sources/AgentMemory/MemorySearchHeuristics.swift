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
        "jan", "january", "feb", "february", "mar", "march", "apr", "april",
        "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
        "oct", "october", "nov", "november", "dec", "december",
        "today", "yesterday", "tomorrow"
    ]
    private static let temporalCueRegex = try? NSRegularExpression(
        pattern: temporalCueWords
            .map { NSRegularExpression.escapedPattern(for: $0) }
            .joined(separator: "|")
    )
    private static let yearRegex = try? NSRegularExpression(pattern: #"\b(19|20)\d{2}\b"#)
    private static let clockTimeRegex = try? NSRegularExpression(pattern: #"\b\d{1,2}:\d{2}\b"#)

    internal static func normalizedComparisonKey(for text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .joined(separator: " ")
    }

    internal static func isTemporalOrAggregateRecallQuery(_ queryText: String) -> Bool {
        RecallQueryUnderstandingAnalyzer.analyze(queryText).isTemporalOrAggregate
    }

    internal static func isTimeAnchoredQuery(_ queryText: String) -> Bool {
        let lower = queryText.lowercased()

        let range = NSRange(lower.startIndex..<lower.endIndex, in: lower)
        if temporalCueRegex?.firstMatch(in: lower, options: [], range: range) != nil {
            return true
        }
        if yearRegex?.firstMatch(in: lower, options: [], range: range) != nil {
            return true
        }
        if clockTimeRegex?.firstMatch(in: lower, options: [], range: range) != nil {
            return true
        }

        return false
    }

}
