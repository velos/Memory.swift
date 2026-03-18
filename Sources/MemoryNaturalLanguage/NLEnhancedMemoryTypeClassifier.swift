import Foundation
import NaturalLanguage
import Memory

public actor NLEnhancedMemoryTypeClassifier: MemoryTypeClassifier {
    public let identifier: String
    private let tokenizer: any Tokenizer

    private let keywordWeights: [DocumentMemoryType: [String]]
    private let phraseWeights: [DocumentMemoryType: [String]]
    private let emotionLexicon: Set<String>

    public init(
        identifier: String = "nl-enhanced-memory-type-classifier",
        tokenizer: any Tokenizer = NLWordTokenizer()
    ) {
        self.identifier = identifier
        self.tokenizer = tokenizer
        self.keywordWeights = [
            .factual: [
                "fact", "data", "number", "metric", "spec", "reference",
                "definition", "detail", "statistic", "version", "configuration",
                "parameter", "documentation", "standard",
            ],
            .procedural: [
                "step", "procedure", "process", "runbook", "guide", "setup",
                "install", "how", "workflow", "deploy", "configure", "execute",
                "checklist", "instruction", "troubleshoot", "migration",
            ],
            .episodic: [
                "meeting", "today", "yesterday", "happened", "incident",
                "retrospective", "story", "experience", "occurred", "recall",
                "remember", "event", "encountered", "discovered",
            ],
            .semantic: [
                "concept", "principle", "pattern", "architecture", "theory",
                "model", "explanation", "understand", "abstraction", "framework",
                "paradigm", "methodology",
            ],
            .emotional: [
                "feel", "emotion", "mood", "frustrated", "excited", "worried",
                "happy", "angry", "anxious", "relieved", "disappointed",
                "overwhelmed", "motivated", "stressed", "confident",
            ],
            .social: [
                "team", "stakeholder", "manager", "customer", "collaborate",
                "relationship", "communication", "people", "colleague",
                "mentor", "report", "meeting", "discussion", "feedback",
            ],
            .contextual: [
                "context", "background", "environment", "constraint",
                "assumption", "scope", "situation", "setting", "prerequisite",
                "limitation", "dependency", "condition",
            ],
            .temporal: [
                "deadline", "timeline", "schedule", "milestone", "quarter",
                "month", "week", "roadmap", "sprint", "release", "date",
                "due", "upcoming", "planned", "overdue",
            ],
        ]
        self.phraseWeights = [
            .procedural: [
                "how to", "step by step", "best practice", "playbook",
                "make sure", "in order to", "first you", "then you",
            ],
            .episodic: [
                "post mortem", "after action", "last week", "this happened",
                "i remember", "we discovered", "it turned out",
            ],
            .semantic: [
                "mental model", "design principle", "high level",
                "in general", "the idea is", "this means",
            ],
            .contextual: [
                "in this context", "given this", "depends on",
                "assuming that", "the constraint is",
            ],
            .temporal: [
                "next sprint", "by friday", "release date",
                "target date", "end of quarter", "due date",
            ],
        ]
        self.emotionLexicon = Set([
            "afraid", "amused", "angry", "annoyed", "anxious", "ashamed",
            "bitter", "bored", "calm", "cheerful", "comfortable", "concerned",
            "confident", "confused", "content", "defeated", "delighted",
            "depressed", "disappointed", "discouraged", "disgusted",
            "eager", "elated", "embarrassed", "empathetic", "encouraged",
            "enthusiastic", "envious", "euphoric", "excited", "exhausted",
            "fearful", "frustrated", "furious", "glad", "grateful", "guilty",
            "happy", "helpless", "hopeful", "hopeless", "horrified", "hostile",
            "hurt", "impatient", "impressed", "inadequate", "indifferent",
            "insecure", "inspired", "intimidated", "irritated", "jealous",
            "joyful", "lonely", "melancholy", "miserable", "motivated",
            "nervous", "nostalgic", "offended", "optimistic", "outraged",
            "overwhelmed", "panicked", "passionate", "peaceful", "pessimistic",
            "pleased", "proud", "regretful", "relaxed", "relieved", "resentful",
            "resigned", "restless", "sad", "satisfied", "scared", "shocked",
            "stressed", "surprised", "sympathetic", "tense", "terrified",
            "thankful", "thrilled", "uncomfortable", "uneasy", "upset",
            "vulnerable", "worried",
        ])
    }

    public func classify(
        documentText: String,
        kind: DocumentKind,
        sourceURL: URL?
    ) async throws -> MemoryTypeAssignment? {
        let trimmed = documentText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let normalized = trimmed.lowercased()
        let tokens = tokenizer.tokenize(normalized)
        let tokenCounts = Dictionary(tokens.map { ($0, 1) }, uniquingKeysWith: +)

        var scores: [DocumentMemoryType: Double] = [:]
        for type in DocumentMemoryType.allCases {
            scores[type] = 0
        }

        for type in DocumentMemoryType.allCases {
            let keywords = keywordWeights[type] ?? []
            let phrases = phraseWeights[type] ?? []

            let keywordScore = keywords.reduce(into: 0.0) { result, keyword in
                result += Double(tokenCounts[keyword, default: 0])
            }
            let phraseScore = phrases.reduce(into: 0.0) { result, phrase in
                if normalized.contains(phrase) {
                    result += 2.0
                }
            }

            scores[type, default: 0] += keywordScore + phraseScore
        }

        let nlSignals = extractNLSignals(from: trimmed)

        if nlSignals.personNameCount > 0 {
            scores[.social, default: 0] += Double(min(nlSignals.personNameCount, 3)) * 2.0
        }
        if nlSignals.organizationCount > 0 {
            scores[.factual, default: 0] += Double(min(nlSignals.organizationCount, 2)) * 1.5
            scores[.social, default: 0] += Double(min(nlSignals.organizationCount, 2)) * 0.5
        }
        if nlSignals.placeNameCount > 0 {
            scores[.contextual, default: 0] += Double(min(nlSignals.placeNameCount, 2)) * 1.5
        }

        if nlSignals.hasPastTenseVerbs {
            scores[.episodic, default: 0] += 2.0
        }
        if nlSignals.hasImperativePattern {
            scores[.procedural, default: 0] += 2.5
        }

        let firstPersonPast = nlSignals.hasFirstPersonPronouns && nlSignals.hasPastTenseVerbs
        if firstPersonPast {
            scores[.episodic, default: 0] += 2.0
        }

        let emotionTokenCount = tokens.filter { emotionLexicon.contains($0) }.count
        if emotionTokenCount > 0 {
            scores[.emotional, default: 0] += Double(min(emotionTokenCount, 4)) * 2.0
        }

        if nlSignals.hasDatePattern {
            scores[.temporal, default: 0] += 2.0
        }

        if kind == .code {
            scores[.procedural, default: 0] += 1.0
            scores[.factual, default: 0] += 1.0
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

        let confidence = min(0.95, 0.40 + (best.value * 0.05))
        return MemoryTypeAssignment(
            type: best.key,
            source: .automatic,
            confidence: confidence,
            classifierID: identifier
        )
    }

    private struct NLSignals {
        var personNameCount: Int = 0
        var organizationCount: Int = 0
        var placeNameCount: Int = 0
        var hasPastTenseVerbs: Bool = false
        var hasImperativePattern: Bool = false
        var hasFirstPersonPronouns: Bool = false
        var hasDatePattern: Bool = false
    }

    private func extractNLSignals(from text: String) -> NLSignals {
        var signals = NLSignals()

        let nerTagger = NLTagger(tagSchemes: [.nameType])
        nerTagger.string = text
        nerTagger.enumerateTags(
            in: text.startIndex..<text.endIndex,
            unit: .word,
            scheme: .nameType,
            options: [.omitWhitespace, .omitPunctuation, .joinNames]
        ) { tag, _ in
            switch tag {
            case .personalName:
                signals.personNameCount += 1
            case .organizationName:
                signals.organizationCount += 1
            case .placeName:
                signals.placeNameCount += 1
            default:
                break
            }
            return true
        }

        let lowered = text.lowercased()
        let firstPersonPronouns = ["i ", "i've ", "i'd ", "we ", "we've ", "we'd ", "my ", "our "]
        signals.hasFirstPersonPronouns = firstPersonPronouns.contains { lowered.contains($0) }

        let pastTenseIndicators = [
            "happened", "occurred", "discovered", "found", "realized",
            "noticed", "experienced", "learned", "decided", "completed",
            "finished", "was ", "were ", "had ", "did ",
        ]
        signals.hasPastTenseVerbs = pastTenseIndicators.contains { lowered.contains($0) }

        let imperativeStarters = [
            "run ", "execute ", "deploy ", "install ", "configure ",
            "check ", "verify ", "ensure ", "make sure ", "set up ",
            "add ", "remove ", "update ", "create ", "delete ",
        ]
        let lines = lowered.components(separatedBy: .newlines)
        for line in lines.prefix(5) {
            let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if imperativeStarters.contains(where: { trimmedLine.hasPrefix($0) }) {
                signals.hasImperativePattern = true
                break
            }
        }

        let datePattern = try? NSRegularExpression(
            pattern: #"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}"#,
            options: .caseInsensitive
        )
        if let datePattern {
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            signals.hasDatePattern = datePattern.firstMatch(in: text, range: range) != nil
        }

        return signals
    }
}
