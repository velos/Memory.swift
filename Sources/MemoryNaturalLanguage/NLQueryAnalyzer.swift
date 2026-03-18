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

        var entities: [String] = []
        let nerTagger = NLTagger(tagSchemes: [.nameType])
        nerTagger.string = trimmed
        nerTagger.enumerateTags(
            in: trimmed.startIndex..<trimmed.endIndex,
            unit: .word,
            scheme: .nameType,
            options: [.omitWhitespace, .omitPunctuation, .joinNames]
        ) { tag, range in
            if let tag, tag != .otherWord {
                let entity = String(trimmed[range])
                if !entity.isEmpty {
                    entities.append(entity)
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

        var suggestedTypes: Set<DocumentMemoryType> = []
        let isHowTo = lowered.hasPrefix("how to") || lowered.hasPrefix("how do")
            || lowered.contains("steps to") || lowered.contains("procedure for")
        if isHowTo {
            suggestedTypes.insert(.procedural)
        }

        let temporalIndicators = [
            "when", "deadline", "timeline", "schedule", "date", "planned",
        ]
        if temporalIndicators.contains(where: { lowered.contains($0) }) {
            suggestedTypes.insert(.temporal)
        }

        let emotionalIndicators = ["feel", "emotion", "mood", "stressed", "happy", "frustrated"]
        if emotionalIndicators.contains(where: { lowered.contains($0) }) {
            suggestedTypes.insert(.emotional)
        }

        let socialIndicators = ["who", "team", "person", "contact", "stakeholder"]
        if socialIndicators.contains(where: { lowered.contains($0) }) {
            suggestedTypes.insert(.social)
        }

        return QueryAnalysis(
            entities: entities,
            keyTerms: keyTerms,
            suggestedDocumentMemoryTypes: suggestedTypes.isEmpty ? nil : suggestedTypes,
            isHowToQuery: isHowTo
        )
    }

    private let stopWords: Set<String> = [
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "from",
        "been", "some", "them", "than", "its", "over", "such", "more",
        "other", "into", "also", "did", "get", "got", "how", "what",
        "where", "when", "who", "why", "which", "this", "that", "with",
    ]
}
