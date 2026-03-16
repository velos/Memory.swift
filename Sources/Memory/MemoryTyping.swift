import Foundation

public enum MemoryType: String, CaseIterable, Codable, Sendable {
    case factual
    case procedural
    case episodic
    case semantic
    case emotional
    case social
    case contextual
    case temporal

    public static func parse(_ raw: String) -> MemoryType? {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return MemoryType(rawValue: normalized)
    }
}

public enum MemoryTypeSource: String, Codable, Sendable {
    case manual
    case automatic
    case fallback

    public static func parse(_ raw: String) -> MemoryTypeSource? {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return MemoryTypeSource(rawValue: normalized)
    }
}

public struct MemoryTypeAssignment: Sendable, Codable {
    public var type: MemoryType
    public var source: MemoryTypeSource
    public var confidence: Double?
    public var classifierID: String?

    public init(
        type: MemoryType,
        source: MemoryTypeSource,
        confidence: Double? = nil,
        classifierID: String? = nil
    ) {
        self.type = type
        self.source = source
        self.confidence = confidence.map { min(1, max(0, $0)) }
        self.classifierID = classifierID
    }
}

public protocol MemoryTypeClassifier: Sendable {
    var identifier: String { get }
    func classify(documentText: String, kind: DocumentKind, sourceURL: URL?) async throws -> MemoryTypeAssignment?
}

public enum MemoryTypingMode: String, Sendable, Codable {
    case automatic
    case manualOnly
}

public struct MemoryTypingConfiguration: Sendable {
    public var mode: MemoryTypingMode
    public var classifier: (any MemoryTypeClassifier)?
    public var fallbackType: MemoryType
    public var minimumConfidenceForFilter: Double

    public init(
        mode: MemoryTypingMode = .automatic,
        classifier: (any MemoryTypeClassifier)? = HeuristicMemoryTypeClassifier(),
        fallbackType: MemoryType = .factual,
        minimumConfidenceForFilter: Double = 0.75
    ) {
        self.mode = mode
        self.classifier = classifier
        self.fallbackType = fallbackType
        self.minimumConfidenceForFilter = min(1, max(0, minimumConfidenceForFilter))
    }

    public static var `default`: MemoryTypingConfiguration {
        MemoryTypingConfiguration(
            mode: .automatic,
            classifier: HeuristicMemoryTypeClassifier(),
            fallbackType: .factual,
            minimumConfidenceForFilter: 0.75
        )
    }
}

public actor HeuristicMemoryTypeClassifier: MemoryTypeClassifier {
    public let identifier: String
    private let tokenizer: any Tokenizer
    private let keywordWeights: [MemoryType: [String]]
    private let phraseWeights: [MemoryType: [String]]

    public init(
        identifier: String = "heuristic-memory-type-classifier",
        tokenizer: any Tokenizer = DefaultTokenizer()
    ) {
        self.identifier = identifier
        self.tokenizer = tokenizer
        self.keywordWeights = [
            .factual: ["fact", "data", "number", "metric", "spec", "reference", "definition", "detail"],
            .procedural: ["step", "procedure", "process", "runbook", "guide", "setup", "install", "how", "workflow"],
            .episodic: ["meeting", "today", "yesterday", "happened", "incident", "retrospective", "story", "experience"],
            .semantic: ["concept", "principle", "pattern", "architecture", "theory", "model", "explanation", "understand"],
            .emotional: ["feel", "emotion", "mood", "frustrated", "excited", "worried", "happy", "angry"],
            .social: ["team", "stakeholder", "manager", "customer", "collaborate", "relationship", "communication", "people"],
            .contextual: ["context", "background", "environment", "constraint", "assumption", "scope", "situation"],
            .temporal: ["deadline", "timeline", "schedule", "milestone", "quarter", "month", "week", "roadmap"],
        ]
        self.phraseWeights = [
            .procedural: ["how to", "step by step", "best practice", "playbook"],
            .episodic: ["post mortem", "after action", "last week", "this happened"],
            .semantic: ["mental model", "design principle", "high level"],
            .contextual: ["in this context", "given this", "depends on"],
            .temporal: ["next sprint", "by friday", "release date"],
        ]
    }

    public func classify(
        documentText: String,
        kind: DocumentKind,
        sourceURL: URL?
    ) async throws -> MemoryTypeAssignment? {
        let trimmed = documentText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let normalized = trimmed.lowercased()
        let tokenCounts = Dictionary(tokenizer.tokenize(normalized).map { ($0, 1) }, uniquingKeysWith: +)

        var scores: [MemoryType: Int] = [:]
        for type in MemoryType.allCases {
            let keywords = keywordWeights[type] ?? []
            let phrases = phraseWeights[type] ?? []

            let keywordScore = keywords.reduce(into: 0) { partialResult, keyword in
                partialResult += tokenCounts[keyword, default: 0]
            }
            let phraseScore = phrases.reduce(into: 0) { partialResult, phrase in
                if normalized.contains(phrase) {
                    partialResult += 2
                }
            }

            scores[type] = keywordScore + phraseScore
        }

        if kind == .code {
            scores[.procedural, default: 0] += 1
            scores[.factual, default: 0] += 1
        }

        guard let best = scores.max(by: { lhs, rhs in
            if lhs.value == rhs.value {
                return lhs.key.rawValue > rhs.key.rawValue
            }
            return lhs.value < rhs.value
        }) else {
            return nil
        }

        guard best.value > 0 else { return nil }

        let confidence = min(0.95, 0.45 + (Double(best.value) * 0.08))
        return MemoryTypeAssignment(
            type: best.key,
            source: .automatic,
            confidence: confidence,
            classifierID: identifier
        )
    }
}

public actor FallbackMemoryTypeClassifier: MemoryTypeClassifier {
    public let identifier: String
    private let primary: any MemoryTypeClassifier
    private let fallback: any MemoryTypeClassifier

    public init(
        identifier: String = "fallback-memory-type-classifier",
        primary: any MemoryTypeClassifier,
        fallback: any MemoryTypeClassifier
    ) {
        self.identifier = identifier
        self.primary = primary
        self.fallback = fallback
    }

    public func classify(
        documentText: String,
        kind: DocumentKind,
        sourceURL: URL?
    ) async throws -> MemoryTypeAssignment? {
        do {
            if let primaryResult = try await primary.classify(
                documentText: documentText,
                kind: kind,
                sourceURL: sourceURL
            ) {
                return primaryResult
            }
        } catch {
            // Fall through to fallback classifier.
        }

        do {
            if let fallbackResult = try await fallback.classify(
                documentText: documentText,
                kind: kind,
                sourceURL: sourceURL
            ) {
                return fallbackResult
            }
        } catch {
            // If fallback also fails, caller applies static fallback type.
        }

        return nil
    }
}

public extension MemoryCategory {
    var mappedMemoryType: MemoryType {
        switch self {
        case .fact:
            return .factual
        case .preference:
            return .contextual
        case .decision:
            return .procedural
        case .identity:
            return .social
        case .event:
            return .episodic
        case .observation:
            return .semantic
        case .goal:
            return .temporal
        case .todo:
            return .procedural
        }
    }
}

public extension MemoryType {
    var defaultCategory: MemoryCategory {
        switch self {
        case .factual:
            return .fact
        case .procedural:
            return .todo
        case .episodic:
            return .event
        case .semantic:
            return .observation
        case .emotional:
            return .observation
        case .social:
            return .identity
        case .contextual:
            return .preference
        case .temporal:
            return .goal
        }
    }
}
