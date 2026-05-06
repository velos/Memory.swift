import Foundation

internal struct MemorySearchAdjustmentSet: OptionSet, Sendable, Hashable {
    internal let rawValue: Int

    internal init(rawValue: Int) {
        self.rawValue = rawValue
    }

    internal static let evidenceSupport = MemorySearchAdjustmentSet(rawValue: 1 << 0)
    internal static let semanticPreservation = MemorySearchAdjustmentSet(rawValue: 1 << 1)
    internal static let currentStateLexicalPreservation = MemorySearchAdjustmentSet(rawValue: 1 << 2)
    internal static let negatedQualificationRelief = MemorySearchAdjustmentSet(rawValue: 1 << 3)
    internal static let proceduralRetentionChoice = MemorySearchAdjustmentSet(rawValue: 1 << 4)
    internal static let temporalLexicalPreservation = MemorySearchAdjustmentSet(rawValue: 1 << 5)
    internal static let recommendationSemantic = MemorySearchAdjustmentSet(rawValue: 1 << 6)
    internal static let aggregateSupportContinuations = MemorySearchAdjustmentSet(rawValue: 1 << 7)

    internal static let all: MemorySearchAdjustmentSet = [
        .evidenceSupport,
        .semanticPreservation,
        .currentStateLexicalPreservation,
        .negatedQualificationRelief,
        .proceduralRetentionChoice,
        .temporalLexicalPreservation,
        .recommendationSemantic,
        .aggregateSupportContinuations,
    ]

    internal static let disableEnvironmentKey = "MEMORY_RECALL_DISABLE_ADJUSTMENTS"
    internal static let onlyEnvironmentKey = "MEMORY_RECALL_ONLY_ADJUSTMENTS"

    internal static func enabledFromProcessEnvironment() -> MemorySearchAdjustmentSet {
        enabled(from: ProcessInfo.processInfo.environment)
    }

    internal static func enabled(from environment: [String: String]) -> MemorySearchAdjustmentSet {
        var enabled = MemorySearchAdjustmentSet.all
        if let only = parse(environment[onlyEnvironmentKey]), !only.isEmpty {
            enabled = only
        }
        if let disabled = parse(environment[disableEnvironmentKey]), !disabled.isEmpty {
            enabled.subtract(disabled)
        }
        return enabled
    }

    internal static func parse(_ rawValue: String?) -> MemorySearchAdjustmentSet? {
        guard let rawValue else { return nil }
        let tokens = rawValue
            .split { character in
                character == "," || character == ";" || character.isWhitespace
            }
            .map { normalizeToken(String($0)) }

        var parsed: MemorySearchAdjustmentSet = []
        for token in tokens {
            if token == "all" {
                parsed.formUnion(.all)
                continue
            }
            if let adjustment = adjustment(named: token) {
                parsed.insert(adjustment)
            }
        }
        return parsed
    }

    internal static func name(for adjustment: MemorySearchAdjustmentSet) -> String? {
        switch adjustment {
        case .evidenceSupport:
            return "evidence_support"
        case .semanticPreservation:
            return "semantic_preservation"
        case .currentStateLexicalPreservation:
            return "current_state_lexical_preservation"
        case .negatedQualificationRelief:
            return "negated_qualification_relief"
        case .proceduralRetentionChoice:
            return "procedural_retention_choice"
        case .temporalLexicalPreservation:
            return "temporal_lexical_preservation"
        case .recommendationSemantic:
            return "recommendation_semantic"
        case .aggregateSupportContinuations:
            return "aggregate_support_continuations"
        default:
            return nil
        }
    }

    private static func normalizeToken(_ token: String) -> String {
        token
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
    }

    private static func adjustment(named token: String) -> MemorySearchAdjustmentSet? {
        switch token {
        case "evidence", "evidence_support", "support":
            return .evidenceSupport
        case "semantic", "semantic_preservation", "expansion_semantic_preservation":
            return .semanticPreservation
        case "current_state", "current_state_lexical", "current_state_lexical_preservation":
            return .currentStateLexicalPreservation
        case "negated_qualification", "negated_qualification_relief", "qualification_relief":
            return .negatedQualificationRelief
        case "procedural", "procedural_retention", "procedural_retention_choice":
            return .proceduralRetentionChoice
        case "temporal_lexical", "temporal_lexical_preservation", "expansion_temporal_lexical":
            return .temporalLexicalPreservation
        case "recommendation", "recommendation_semantic":
            return .recommendationSemantic
        case "aggregate_support", "aggregate_support_continuations", "support_continuations":
            return .aggregateSupportContinuations
        default:
            return nil
        }
    }
}
