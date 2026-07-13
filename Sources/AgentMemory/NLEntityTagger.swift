#if MEMORY_NATURAL_LANGUAGE
import Foundation
import NaturalLanguage

/// NaturalLanguage-backed named entity recognition for the memory write path.
///
/// `NLTagger` name recognition depends on capitalization: it stays silent on
/// all-lowercase chat text, so `recognizeEntities` degrades to a no-op there
/// and heuristic extraction remains the behavior floor. NaturalLanguage does
/// not expose calibrated confidence for name tags, so annotations use a
/// deliberately modest confidence and downstream reconciliation requires
/// contextual support before specializing heuristic labels.
public struct NLEntityTagger: EntityTagger {
    public let identifier = "nl-entity-tagger"

    public init() {}

    public func recognizeEntities(in text: String) -> [MemoryEntity] {
        NLNamedEntityRecognition.recognize(in: text, confidence: 0.55)
    }
}

internal enum NLNamedEntityRecognition {
    internal static func recognize(in text: String, confidence: Double) -> [MemoryEntity] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var entities: [MemoryEntity] = []
        var seen: Set<String> = []
        let tagger = NLTagger(tagSchemes: [.nameType])
        tagger.string = trimmed
        tagger.enumerateTags(
            in: trimmed.startIndex..<trimmed.endIndex,
            unit: .word,
            scheme: .nameType,
            options: [.omitWhitespace, .omitPunctuation, .joinNames]
        ) { tag, range in
            if let tag, let label = entityLabel(for: tag) {
                let value = String(trimmed[range]).trimmingCharacters(in: .whitespacesAndNewlines)
                let normalizedValue = MemoryExtractionHeuristics.normalizeEntityValue(value)
                guard !normalizedValue.isEmpty, seen.insert(normalizedValue).inserted else { return true }
                entities.append(
                    MemoryEntity(
                        label: label,
                        value: value,
                        normalizedValue: normalizedValue,
                        confidence: confidence
                    )
                )
            }
            return true
        }
        return entities
    }

    private static func entityLabel(for tag: NLTag) -> EntityLabel? {
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
}
#endif
