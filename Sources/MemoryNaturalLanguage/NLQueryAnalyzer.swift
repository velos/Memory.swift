import Foundation
import NaturalLanguage
import Memory

public struct NLQueryAnalyzer: QueryAnalyzer, Sendable {
    public let identifier = "nl-query-analyzer"

    public init() {}

    public func analyze(query: String) -> QueryAnalysis {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return QueryAnalysis() }

        let lowered = trimmed.lowercased()

        var entities: [MemoryEntity] = []
        let nerTagger = NLTagger(tagSchemes: [.nameType])
        nerTagger.string = trimmed
        nerTagger.enumerateTags(
            in: trimmed.startIndex..<trimmed.endIndex,
            unit: .word,
            scheme: .nameType,
            options: [.omitWhitespace, .omitPunctuation, .joinNames]
        ) { tag, range in
            if let tag, tag != .otherWord {
                let value = String(trimmed[range]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !value.isEmpty {
                    let normalizedValue = normalizeEntityValue(value)
                    if !normalizedValue.isEmpty {
                        entities.append(
                            MemoryEntity(
                                label: entityLabel(for: tag),
                                value: value,
                                normalizedValue: normalizedValue,
                                confidence: 0.75
                            )
                        )
                    }
                }
            }
            return true
        }

        var keyTerms: [String] = []
        let posTagger = NLTagger(tagSchemes: [.lexicalClass])
        posTagger.string = trimmed
        posTagger.enumerateTags(
            in: trimmed.startIndex..<trimmed.endIndex,
            unit: .word,
            scheme: .lexicalClass,
            options: [.omitWhitespace, .omitPunctuation]
        ) { tag, range in
            if let tag, tag == .noun || tag == .verb || tag == .adjective {
                let term = String(trimmed[range]).lowercased()
                if term.count >= 3, !stopWords.contains(term) {
                    keyTerms.append(term)
                }
            }
            return true
        }

        let isHowTo = lowered.hasPrefix("how to") || lowered.hasPrefix("how do")
            || lowered.contains("steps to") || lowered.contains("procedure for")

        var facetHints: Set<FacetTag> = []
        for (facet, needles) in facetKeywords {
            if needles.contains(where: { lowered.contains($0) }) {
                facetHints.insert(facet)
            }
        }

        let topics = extractTopics(from: trimmed, keyTerms: keyTerms)

        return QueryAnalysis(
            entities: entities,
            keyTerms: keyTerms,
            facetHints: facetHints,
            topics: topics,
            isHowToQuery: isHowTo
        )
    }

    private func extractTopics(from query: String, keyTerms: [String]) -> [String] {
        let tokens = query
            .split { character in !character.isLetter && !character.isNumber && character != "+" && character != "-" && character != "." && character != "/" }
            .map { String($0).lowercased() }
            .filter { $0.count >= 3 && !stopWords.contains($0) }

        var topics: [String] = []
        var seen: Set<String> = []

        for width in stride(from: 3, through: 2, by: -1) {
            guard tokens.count >= width else { continue }
            for start in 0...(tokens.count - width) {
                let candidate = tokens[start..<(start + width)].joined(separator: " ")
                guard seen.insert(candidate).inserted else { continue }
                topics.append(candidate)
                if topics.count >= 8 {
                    return topics
                }
            }
        }

        for term in keyTerms {
            guard seen.insert(term).inserted else { continue }
            topics.append(term)
            if topics.count >= 8 {
                break
            }
        }

        return topics
    }

    private func normalizeEntityValue(_ raw: String) -> String {
        raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
            .lowercased()
    }

    private func entityLabel(for tag: NLTag) -> EntityLabel {
        switch tag {
        case .personalName:
            return .person
        case .organizationName:
            return .organization
        case .placeName:
            return .location
        default:
            return .other
        }
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

    private let stopWords: Set<String> = [
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "from",
        "been", "some", "them", "than", "its", "over", "such", "more",
        "other", "into", "also", "did", "get", "got", "how", "what",
        "where", "when", "who", "why", "which", "this", "that", "with",
    ]
}
