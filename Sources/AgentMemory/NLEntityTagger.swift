#if MEMORY_NATURAL_LANGUAGE
import Foundation
import NaturalLanguage

/// NaturalLanguage-backed named entity recognition for the memory write path.
///
/// `NLTagger` name recognition depends on capitalization: it stays silent on
/// all-lowercase chat text, so `recognizeEntities` degrades to a no-op there
/// and heuristic extraction remains the behavior floor. `isLikelyPlaceName`
/// validates short phrases by title-casing them inside a natural carrier
/// sentence, where place recognition is markedly more precise than tagging
/// the bare phrase.
public struct NLEntityTagger: EntityTagger {
    public let identifier = "nl-entity-tagger"

    public init() {}

    public func recognizeEntities(in text: String) -> [MemoryEntity] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var entities: [MemoryEntity] = []
        var seen: Set<String> = []
        enumerateNameTags(in: trimmed) { value, label in
            let normalizedValue = MemoryExtractionHeuristics.normalizeEntityValue(value)
            guard !normalizedValue.isEmpty, seen.insert(normalizedValue).inserted else { return }
            entities.append(
                MemoryEntity(
                    label: label,
                    value: value,
                    normalizedValue: normalizedValue,
                    confidence: 0.8
                )
            )
        }
        return entities
    }

    public func isLikelyPlaceName(_ phrase: String) -> Bool {
        let trimmed = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= 60 else { return false }

        let candidate = titleCased(trimmed)
        let carrier = "They moved to \(candidate) last month."
        var foundPlace = false
        enumerateNameTags(in: carrier) { value, label in
            guard label == .location else { return }
            if candidate.range(of: value, options: [.caseInsensitive]) != nil {
                foundPlace = true
            }
        }
        return foundPlace
    }

    private func enumerateNameTags(in text: String, handler: (String, EntityLabel) -> Void) {
        let tagger = NLTagger(tagSchemes: [.nameType])
        tagger.string = text
        tagger.enumerateTags(
            in: text.startIndex..<text.endIndex,
            unit: .word,
            scheme: .nameType,
            options: [.omitWhitespace, .omitPunctuation, .joinNames]
        ) { tag, range in
            if let tag, let label = entityLabel(for: tag) {
                let value = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !value.isEmpty {
                    handler(value, label)
                }
            }
            return true
        }
    }

    private func entityLabel(for tag: NLTag) -> EntityLabel? {
        switch tag {
        case .personalName:
            .person
        case .placeName:
            .location
        case .organizationName:
            .organization
        default:
            nil
        }
    }

    private func titleCased(_ phrase: String) -> String {
        phrase
            .split(separator: " ", omittingEmptySubsequences: false)
            .map { word -> String in
                guard let first = word.first, first.isLowercase else { return String(word) }
                return first.uppercased() + word.dropFirst()
            }
            .joined(separator: " ")
    }
}
#endif
