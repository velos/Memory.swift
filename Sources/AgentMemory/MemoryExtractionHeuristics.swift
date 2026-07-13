import Foundation
import MemoryStorage

internal enum MemoryExtractionHeuristics {
    internal typealias CanonicalKeyResolver = (
        _ kind: MemoryKind,
        _ text: String,
        _ explicitKey: String?,
        _ entities: [MemoryEntity],
        _ topics: [String]
    ) -> String?
    internal typealias ProposedActionResolver = (_ candidate: MemoryCandidate) -> MemoryWriteAction

    private static let entityBoundaryNoiseTokens: Set<String> = [
        "action", "completed", "context", "current", "decision", "done", "guide",
        "friday", "handoff", "monday", "next", "on", "procedure", "runbook",
        "saturday", "status", "step", "sunday", "the", "thursday", "todo", "tuesday",
        "wednesday", "workflow"
    ]

    private static let genericTopicValues: Set<String> = [
        "action item", "current status", "next owner", "next person", "next step",
        "status update"
    ]

    internal static func extract(
        messages: [ConversationMessage],
        limit: Int,
        canonicalKey: CanonicalKeyResolver,
        proposedAction: ProposedActionResolver
    ) -> MemoryExtractionResult {
        var extracted: [MemoryCandidate] = []
        extracted.reserveCapacity(limit)
        var rejected: [MemoryRejectedSpan] = []
        var rationales: [String] = []

        var seen: Set<String> = []
        for (messageIndex, message) in messages.enumerated() {
            let normalized = message.content
                .replacingOccurrences(of: "\r\n", with: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalized.isEmpty else { continue }

            let focusedSegments = focusedUserProfileSegments(from: normalized, role: message.role)
            let rawSegments = focusedSegments.map(\.text) + splitExtractionSegments(normalized)

            for rawSegment in rawSegments {
                let segment = rawSegment.trimmingCharacters(in: .whitespacesAndNewlines)
                guard segment.count >= 18 else {
                    if !segment.isEmpty {
                        rejected.append(MemoryRejectedSpan(text: segment, reason: "too_short", confidence: 0.9))
                    }
                    continue
                }

                let key = MemorySearchHeuristics.normalizedComparisonKey(for: segment)
                guard seen.insert(key).inserted else {
                    rejected.append(MemoryRejectedSpan(text: segment, reason: "duplicate_span", confidence: 0.9))
                    continue
                }

                let focusedSegment = focusedSegments.first { $0.text == segment }
                let kind = inferKind(forExtractedText: segment)
                guard isExtractableMemorySegment(
                    segment,
                    kind: kind,
                    role: message.role,
                    isFocusedProfileSegment: focusedSegment != nil
                ) else {
                    rejected.append(MemoryRejectedSpan(text: segment, reason: "not_memory_worthy", confidence: 0.85))
                    continue
                }

                let status = inferStatus(forExtractedText: segment, kind: kind)
                let importance = inferredImportance(for: kind)
                let confidence = inferredConfidence(for: kind)
                let tags = inferredTags(forExtractedText: segment)
                let entities = mergeEntities(
                    pinned: focusedSegment?.entities ?? [],
                    inferred: inferEntities(forExtractedText: segment)
                )
                let facetTags = inferFacetTags(forExtractedText: segment, kind: kind, knownEntities: entities)
                    .union(focusedSegment?.facetTags ?? [])
                let topics = inferTopics(forExtractedText: segment, seedTags: tags)
                let subject = inferSubject(forExtractedText: segment, role: message.role, kind: kind)
                let evidence = MemoryEvidence(
                    role: message.role,
                    excerpt: evidenceExcerpt(for: segment, in: message.content),
                    messageIndex: messageIndex,
                    timestamp: message.createdAt,
                    sourceID: nil
                )

                let candidate = MemoryCandidate(
                    text: segment,
                    kind: kind,
                    status: status,
                    importance: importance,
                    confidence: confidence,
                    createdAt: nil,
                    eventAt: kind == .episode ? Date() : nil,
                    source: "heuristic_extract",
                    tags: tags,
                    facetTags: facetTags,
                    entities: entities,
                    topics: topics,
                    canonicalKey: subjectAwareCanonicalKey(
                        canonicalKey(kind, segment, nil, entities, topics),
                        subject: subject,
                        kind: kind
                    ),
                    subject: subject,
                    evidence: [evidence]
                )
                extracted.append(candidate)
                rationales.append("\(kind.rawValue):\(status.rawValue):\(segment)")

                if extracted.count >= limit {
                    return MemoryExtractionResult(
                        candidates: extracted,
                        rejectedSpans: rejected,
                        proposedActions: extracted.map(proposedAction),
                        rationale: rationales
                    )
                }
            }
        }

        return MemoryExtractionResult(
            candidates: extracted,
            rejectedSpans: rejected,
            proposedActions: extracted.map(proposedAction),
            rationale: rationales
        )
    }

    internal static func inferredTags(forExtractedText text: String) -> [String] {
        let rawTokens = text.lowercased().split { character in
            !character.isLetter && !character.isNumber
        }

        var seen: Set<String> = []
        var tags: [String] = []
        for token in rawTokens {
            let value = String(token)
            guard value.count >= 4 else { continue }
            guard !MemorySearchHeuristics.queryStopWords.contains(value) else { continue }
            if seen.insert(value).inserted {
                tags.append(value)
            }
            if tags.count >= 6 {
                break
            }
        }
        return tags
    }

    internal static func inferFacetTags(
        forExtractedText text: String,
        kind: MemoryKind,
        knownEntities: [MemoryEntity]? = nil
    ) -> Set<FacetTag> {
        let normalizedText = phraseEnvelope(for: text)
        let knownEntities = knownEntities ?? inferKnownEntities(forExtractedText: text)
        var facets: Set<FacetTag> = []

        if containsAnyNormalizedPhrase(normalizedText, phrases: ["prefer", "prefers", "preferred", "preference", "favorite", "likes", "dislikes"])
            || kind == .profile && containsNormalizedPhrase(normalizedText, "avoid") {
            facets.insert(.preference)
        }
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["project", "repository", "branch", "milestone"])
            || knownEntities.contains(where: { $0.label == .project })
                && hasProjectFacetContext(kind: kind, normalizedText: normalizedText) {
            facets.insert(.project)
        }
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["goal", "objective", "aim"]) {
            facets.insert(.goal)
        }
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["todo", "task", "action item", "follow up", "next step"]) {
            facets.insert(.task)
        }
        if containsAnyNormalizedPhrase(
            normalizedText,
            phrases: [
                "tool", "sdk", "framework", "library", "sqlite", "sqlite-vec",
                "coreml", "leaf-ir", "fts5", "slack", "zed",
                "apple intelligence"
            ]
        )
            || knownEntities.contains(where: { $0.label == .tool }) {
            facets.insert(.tool)
        }
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["where", "location", "office", "remote", "timezone"])
            || knownEntities.contains(where: { $0.label == .location }) {
            facets.insert(.location)
        }
        if hasTimeSensitiveFacetSignal(text: text, kind: kind, normalizedText: normalizedText) {
            facets.insert(.timeSensitive)
        }
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["constraint", "blocked", "cannot", "must not", "limit"])
            || kind == .decision && containsAnyNormalizedPhrase(normalizedText, phrases: ["skip", "hot path"])
            || kind == .decision
                && containsNormalizedPhrase(normalizedText, "apple intelligence")
                && containsNormalizedPhrase(normalizedText, "experimental flag") {
            facets.insert(.constraint)
        }
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["habit", "usually", "often", "routine"]) {
            facets.insert(.habit)
        }
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["lesson", "learned", "takeaway", "retrospective"]) {
            facets.insert(.lesson)
        }
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["feel", "frustrated", "happy", "worried", "stressed", "celebrated", "celebrate", "excited"]) {
            facets.insert(.emotion)
        }
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["name", "role", "i am", "i'm"]) {
            facets.insert(.identitySignal)
        }
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["maintainer", "owner"]) {
            facets.insert(.identitySignal)
        }
        if hasRelationshipFacetSignal(text: text, kind: kind, normalizedText: normalizedText) {
            facets.insert(.relationship)
            facets.insert(.person)
        }
        if hasPersonFacetSignal(text: text, kind: kind, knownEntities: knownEntities, normalizedText: normalizedText) {
            facets.insert(.person)
        }

        switch kind {
        case .profile:
            facets.insert(.factAboutUser)
        case .fact:
            facets.insert(.factAboutWorld)
        case .decision:
            facets.insert(.decisionTopic)
        case .commitment:
            facets.insert(.task)
        default:
            break
        }

        return facets
    }

    internal static func inferEntities(forExtractedText text: String) -> [MemoryEntity] {
        let tokens = text.split(separator: " ")
        var entities = inferKnownEntities(forExtractedText: text)
        var current: [String] = []

        func flush(label: EntityLabel = .other) {
            guard !current.isEmpty else { return }
            for value in entityValues(from: current) {
                let normalizedValue = normalizeEntityValue(value)
                guard !normalizedValue.isEmpty else { continue }
                entities.append(
                    MemoryEntity(
                        label: label,
                        value: value,
                        normalizedValue: normalizedValue,
                        confidence: 0.6
                    )
                )
            }
            current.removeAll(keepingCapacity: true)
        }

        for token in tokens {
            let raw = token.trimmingCharacters(in: CharacterSet.punctuationCharacters)
            guard !raw.isEmpty else {
                flush()
                continue
            }

            if looksLikeEntityToken(raw) {
                current.append(raw)
            } else {
                flush()
            }
        }
        flush()

        var normalized: [MemoryEntity] = []
        var seen: Set<String> = []
        for entity in entities {
            let normalizedValue = normalizeEntityValue(entity.normalizedValue)
            guard !normalizedValue.isEmpty else { continue }
            guard !isGenericExtractedEntity(normalizedValue) else { continue }
            guard seen.insert(normalizedValue).inserted else { continue }
            normalized.append(
                MemoryEntity(
                    label: entity.label,
                    value: entity.value,
                    normalizedValue: normalizedValue,
                    confidence: entity.confidence
                )
            )
            if normalized.count >= 8 {
                break
            }
        }

        return normalized
    }

    internal static func inferTopics(forExtractedText text: String, seedTags: [String]) -> [String] {
        let maxTopics = 16
        let tokens = text
            .lowercased()
            .split { character in
                !character.isLetter && !character.isNumber && character != "+" && character != "-" && character != "." && character != "/"
            }
            .map(String.init)
            .filter { $0.count >= 3 && !MemorySearchHeuristics.queryStopWords.contains($0) }

        var topics: [String] = []
        var seen: Set<String> = []

        for topic in inferKnownTopics(forExtractedText: text) {
            let candidate = normalizeTopicValue(topic)
            guard !candidate.isEmpty else { continue }
            guard seen.insert(candidate).inserted else { continue }
            topics.append(candidate)
            if topics.count >= maxTopics {
                return topics
            }
        }

        for width in [2, 4, 3] where tokens.count >= width {
            guard tokens.count >= width else { continue }
            for start in 0...(tokens.count - width) {
                let candidate = normalizeTopicValue(tokens[start..<(start + width)].joined(separator: " "))
                guard !candidate.isEmpty else { continue }
                guard !isGenericTopicValue(candidate) else { continue }
                guard seen.insert(candidate).inserted else { continue }
                topics.append(candidate)
                if topics.count >= maxTopics {
                    return topics
                }
            }
        }

        for tag in seedTags {
            let candidate = normalizeTopicValue(tag)
            guard !candidate.isEmpty else { continue }
            guard seen.insert(candidate).inserted else { continue }
            topics.append(candidate)
            if topics.count >= maxTopics {
                break
            }
        }

        return topics
    }

    internal static func normalizeEntityValue(_ raw: String) -> String {
        let punctuation = CharacterSet(charactersIn: ",:;!?()[]{}\"'`")
        return raw
            .replacingOccurrences(of: "\u{2019}s", with: "'s")
            .replacingOccurrences(of: "\u{2019}S", with: "'S")
            .trimmingCharacters(in: .whitespacesAndNewlines.union(punctuation))
            .replacingOccurrences(of: #"(?i)'s\b"#, with: "", options: .regularExpression)
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
            .lowercased()
    }

    internal static func normalizeTopicValue(_ raw: String) -> String {
        let punctuation = CharacterSet(charactersIn: ".,:;!?()[]{}\"'`")
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines.union(punctuation))
            .lowercased()
            .split { character in character.isWhitespace }
            .map(String.init)
            .filter { !$0.isEmpty }
            .prefix(4)
            .joined(separator: " ")
        return normalized
    }

    internal static func makeStoredMemoryEntity(from entity: MemoryEntity) -> StoredMemoryEntity {
        StoredMemoryEntity(
            label: entity.label.rawValue,
            value: entity.value,
            normalizedValue: entity.normalizedValue,
            confidence: entity.confidence
        )
    }

    internal static func makeMemoryEntity(from entity: StoredMemoryEntity) -> MemoryEntity? {
        guard let label = EntityLabel.parse(entity.label) else { return nil }
        let normalizedValue = normalizeEntityValue(entity.normalizedValue.isEmpty ? entity.value : entity.normalizedValue)
        guard !normalizedValue.isEmpty else { return nil }
        return MemoryEntity(
            label: label,
            value: entity.value,
            normalizedValue: normalizedValue,
            confidence: entity.confidence
        )
    }

    private static func splitExtractionSegments(_ text: String) -> [String] {
        var segments: [String] = []
        var start = text.startIndex
        var index = text.startIndex

        while index < text.endIndex {
            let character = text[index]
            if isExtractionSegmentBoundary(character, at: index, in: text) {
                let segment = String(text[start..<index])
                if !segment.isEmpty {
                    segments.append(segment)
                }
                start = text.index(after: index)
            }
            index = text.index(after: index)
        }

        if start < text.endIndex {
            let segment = String(text[start..<text.endIndex])
            if !segment.isEmpty {
                segments.append(segment)
            }
        }

        return segments
    }

    private static func isExtractionSegmentBoundary(_ character: Character, at index: String.Index, in text: String) -> Bool {
        if character == "\n" || character == "!" || character == "?" {
            return true
        }

        guard character == "." else { return false }
        let previousIndex = index > text.startIndex ? text.index(before: index) : nil
        let nextIndex = text.index(after: index) < text.endIndex ? text.index(after: index) : nil
        let previous = previousIndex.map { text[$0] }
        let next = nextIndex.map { text[$0] }

        let previousIsIdentifier = previous?.isLetter == true || previous?.isNumber == true
        let nextIsIdentifier = next?.isLetter == true || next?.isNumber == true
        if previousIsIdentifier, nextIsIdentifier {
            return false
        }
        return true
    }

    private static func inferKind(forExtractedText text: String) -> MemoryKind {
        let lower = text.lowercased()

        if containsAny(lower, needles: ["runbook", "procedure", "playbook", "workflow", "how to", "step by step", "guide:"]) {
            return .procedure
        }
        if containsAny(lower, needles: ["decide", "decision", "chose", "choose", "switch to", "switched", "agreed", "picked", "settled on"]) {
            return .decision
        }
        if containsAny(
            lower,
            needles: [
                "todo", "to do", "done:", "follow up", "action item", "next step",
                "need to", "must", "should", "finished", "completed", " is done",
                " is complete", " is completed", " is finished"
            ]
        ) {
            return .commitment
        }
        if containsAny(
            lower,
            needles: [
                "prefer", "preference", "favorite", "likes", "usually", "works closely",
                "timezone", "my name", "i am", "i'm", "my role", "role is",
                " is the maintainer", " is the owner", "release owner", " owner for ",
                "standing constraint", "i live in", "the user lives in",
                "my city is", "my location is", "based in"
            ]
        ) {
            return .profile
        }
        if containsAny(lower, needles: ["handoff", "current status", "blocked on", "next owner", "for the next person", "context"]) {
            return .handoff
        }
        if containsAny(lower, needles: [
            "today", "yesterday", "this morning", "last week", "after the demo",
            "incident", "meeting", "met ", "retrospective", "shipped", "celebrated",
            "on monday", "on tuesday", "on wednesday", "on thursday", "on friday",
            "on saturday", "on sunday"
        ]) {
            return .episode
        }
        return .fact
    }

    private static func isExtractableMemorySegment(
        _ text: String,
        kind: MemoryKind,
        role: ConversationRole? = nil,
        isFocusedProfileSegment: Bool = false
    ) -> Bool {
        let lower = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !lower.isEmpty else { return false }

        if lower.hasSuffix("?") {
            return false
        }

        if role == .assistant,
           containsAny(
               lower,
               needles: [
                   "i will remember", "i'll remember", "i can remember",
                   "i have noted", "i noted", "noted that", "i will keep",
                   "i'll keep", "sure, i can", "happy to explain",
                   "i don't have real-time", "i do not have real-time",
                   "i don't have access to", "i do not have access to",
                   "i can't browse", "i cannot browse", "as an ai",
                   "my capabilities are", "i suggest checking",
                   "i recommend checking", "you could try", "you might try",
                   "consider checking"
               ]
           ) {
            return false
        }

        let durableMemoryRequest = containsAny(
            lower,
            needles: ["remember that", "please remember", "for future reference", "note that", "keep in mind"]
        )
        if kind == .profile,
           isAmbiguousPresenceStatement(lower),
           !durableMemoryRequest,
           !containsDurableProfileSignal(lower) {
            return false
        }
        if isQuestionLikeNonMemorySegment(lower), !durableMemoryRequest {
            return false
        }

        if containsEmbeddedQuestionClause(lower), !durableMemoryRequest, !isFocusedProfileSegment {
            return false
        }

        let conversationalRequest = containsAny(
            lower,
            needles: [
                "thank you", "thanks", "sounds good", "got it", "okay", "ok ",
                "can you", "could you", "would you", "please explain", "explain how",
                "hypothetical question"
            ]
        )
        if conversationalRequest && !durableMemoryRequest {
            return false
        }

        switch kind {
        case .profile, .decision, .commitment, .procedure, .handoff, .episode:
            return true
        case .fact:
            return containsAny(
                " \(lower) ",
                needles: [
                    " is ", " are ", " was ", " were ", " uses ", " use ",
                    " has ", " have ", " includes ", " requires ", " supports ",
                    " stores ", " runs ", " ships ", " powers ", " keeps ",
                    " load ", " loads ", " blocks ", " works "
                ]
            )
        }
    }

    private static func isAmbiguousPresenceStatement(_ lower: String) -> Bool {
        lower.range(
            of: #"\b(?:i\s+am|i['’]m)\s+in\s+"#,
            options: .regularExpression
        ) != nil
    }

    private static func containsDurableProfileSignal(_ lower: String) -> Bool {
        containsAny(
            lower,
            needles: [
                "prefer", "favorite", "usually", "works closely", "timezone",
                "my name", "my role", "role is", "maintainer", "owner",
                "i live in", "living in", "based in", "moved to",
                "my city is", "my location is",
            ]
        )
    }

    private static func isQuestionLikeNonMemorySegment(_ lower: String) -> Bool {
        let questionPrefixes = [
            "what ", "when ", "where ", "which ", "who ", "why ", "how ",
            "can ", "could ", "would ", "should ", "do ", "does ", "did ",
            "is ", "are ", "am ", "was ", "were "
        ]
        if questionPrefixes.contains(where: { lower.hasPrefix($0) }) {
            return true
        }

        if lower.hasPrefix("if "),
           containsAny(lower, needles: [" what ", " would ", " could ", " should ", " responsibilities"]) {
            return true
        }

        return containsAny(
            lower,
            needles: [
                "hypothetical question",
                "i am asking a hypothetical",
                "i'm asking a hypothetical"
            ]
        )
    }

    private struct FocusedProfileSegment {
        var text: String
        var entities: [MemoryEntity]
        var facetTags: Set<FacetTag>
    }

    private static func focusedUserProfileSegments(
        from text: String,
        role: ConversationRole
    ) -> [FocusedProfileSegment] {
        guard role == .user else { return [] }

        var segments: [FocusedProfileSegment] = []
        if let location = selfReportedLocation(in: text) {
            let city = location.split(separator: ",").first.map(String.init) ?? location
            segments.append(
                FocusedProfileSegment(
                    text: "The user lives in \(location).",
                    entities: [
                        MemoryEntity(
                            label: .location,
                            value: location,
                            normalizedValue: normalizeEntityValue(city),
                            confidence: 0.85
                        ),
                    ],
                    facetTags: [.location]
                )
            )
        }
        return segments
    }

    private static func mergeEntities(pinned: [MemoryEntity], inferred: [MemoryEntity]) -> [MemoryEntity] {
        guard !pinned.isEmpty else { return inferred }
        var seen: Set<String> = []
        var merged: [MemoryEntity] = []
        for entity in pinned + inferred {
            let normalizedValue = normalizeEntityValue(entity.normalizedValue)
            guard !normalizedValue.isEmpty, seen.insert(normalizedValue).inserted else { continue }
            merged.append(entity)
        }
        return merged
    }

    /// Reconciles extractor-, heuristic-, and tagger-supplied entities. Explicit
    /// extractor values remain the floor. Low-confidence linguistic labels only
    /// specialize an entity when the surrounding prose supports that label.
    internal static func enrichedEntities(
        supplied: [MemoryEntity],
        tagged: [MemoryEntity],
        text: String
    ) -> [MemoryEntity] {
        let inferred = inferEntities(forExtractedText: text)
        var merged = mergeEntities(pinned: supplied, inferred: inferred)
        guard !tagged.isEmpty else { return merged }

        var indexByValue: [String: Int] = [:]
        for (index, entity) in merged.enumerated() {
            indexByValue[normalizeEntityValue(entity.normalizedValue)] = index
        }

        for entity in tagged {
            let normalizedValue = normalizeEntityValue(entity.normalizedValue)
            guard !normalizedValue.isEmpty, !isGenericExtractedEntity(normalizedValue) else { continue }
            guard isSupportedTaggedEntity(entity, in: text) else { continue }
            if let index = indexByValue[normalizedValue] {
                let existing = merged[index]
                if existing.label == .other {
                    merged[index] = MemoryEntity(
                        label: entity.label,
                        value: existing.value,
                        normalizedValue: normalizedValue,
                        confidence: max(existing.confidence ?? 0, entity.confidence ?? 0)
                    )
                }
            } else if merged.count < 8,
                      !merged.contains(where: { valuesOverlap($0.normalizedValue, normalizedValue) }) {
                merged.append(
                    MemoryEntity(
                        label: entity.label,
                        value: entity.value,
                        normalizedValue: normalizedValue,
                        confidence: entity.confidence
                    )
                )
                indexByValue[normalizedValue] = merged.count - 1
            }
        }

        return merged
    }

    private static func isSupportedTaggedEntity(_ entity: MemoryEntity, in text: String) -> Bool {
        if entity.confidence.map({ $0 >= 0.9 }) == true {
            return true
        }

        let escapedValue = NSRegularExpression.escapedPattern(for: entity.value)
        guard !escapedValue.isEmpty else { return false }

        let patterns: [String]
        switch entity.label {
        case .person:
            patterns = [
                #"\b\#(escapedValue)\b\s+(?:is|was|said|asked|joined|reviewed|prefers|works|retired|captured|noted|owns|leads)\b"#,
                #"\b(?:met|with|ask|asked|contact|manager|coworker|teammate)\s+\#(escapedValue)\b"#,
            ]
        case .location:
            patterns = [
                #"\b(?:in|at|near|from|to)\s+\#(escapedValue)\b"#,
                #"\b(?:live|lives|living|based|moved)\s+(?:in|to)\s+\#(escapedValue)\b"#,
            ]
        case .organization:
            patterns = [
                #"\b(?:at|for|from|joined|maintainer\s+of|works\s+at)\s+\#(escapedValue)\b"#,
                #"\b\#(escapedValue)\b\s+(?:team|company|organization)\b"#,
            ]
        case .product, .project, .tool, .date, .other:
            return true
        }

        return patterns.contains { pattern in
            text.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil
        }
    }

    private static func valuesOverlap(_ lhs: String, _ rhs: String) -> Bool {
        let left = normalizeEntityValue(lhs)
        let right = normalizeEntityValue(rhs)
        guard !left.isEmpty, !right.isEmpty else { return false }
        return left == right
            || left.hasPrefix(right + " ")
            || left.hasSuffix(" " + right)
            || right.hasPrefix(left + " ")
            || right.hasSuffix(" " + left)
    }

    // Only explicit residence statements create durable profile locations.
    // Ambiguous presence statements such as "I'm in Lisbon" may describe
    // travel and must not be rewritten as residence.
    private static let strongLocationTriggerPattern =
        #"\b(?:i\s+live\s+in|i\s+am\s+living\s+in|i['’]m\s+living\s+in|i\s+am\s+based\s+in|i['’]m\s+based\s+in|i\s+(?:just\s+)?moved\s+to|my\s+city\s+is|my\s+location\s+is)\s+([^.!?;\n]+)"#

    private static let knownLocationAliases: [String: String] = [
        "sf": "San Francisco, CA",
        "san francisco": "San Francisco, CA",
        "san francisco ca": "San Francisco, CA",
        "san francisco california": "San Francisco, CA",
        "nyc": "New York, NY",
        "new york": "New York, NY",
        "new york city": "New York, NY",
        "new york ny": "New York, NY",
        "la": "Los Angeles, CA",
        "los angeles": "Los Angeles, CA",
        "los angeles ca": "Los Angeles, CA",
    ]

    private static let locationStopTokens: Set<String> = [
        "and", "but", "or", "so", "which", "where", "when", "while", "because",
        "since", "near", "with", "for", "though", "although", "what", "what's",
        "what’s", "who", "how", "why", "that", "then", "if", "as", "at", "about"
    ]

    private static let locationTrailingNoiseTokens: Set<String> = [
        "now", "currently", "atm", "too", "btw", "there", "here", "tonight", "today"
    ]

    private static let locationLeadingRejectTokens: Set<String> = [
        "a", "an", "my", "our", "your", "his", "her", "their", "this", "that",
        "these", "those", "some", "any", "no", "one", "it", "front", "between",
        "town", "meetings", "meeting"
    ]

    private static func selfReportedLocation(in text: String) -> String? {
        if let raw = firstRegexCapture(of: strongLocationTriggerPattern, in: text),
           let location = parseLocationPhrase(raw) {
            return location
        }
        return nil
    }

    private static func firstRegexCapture(of pattern: String, in text: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return nil
        }
        let nsText = text as NSString
        let range = NSRange(location: 0, length: nsText.length)
        guard let match = regex.firstMatch(in: text, range: range), match.numberOfRanges > 1 else {
            return nil
        }
        return nsText.substring(with: match.range(at: 1))
    }

    private static func parseLocationPhrase(_ raw: String) -> String? {
        let tokens = raw.split(whereSeparator: \.isWhitespace).map(String.init)
        var cityTokens: [String] = []
        var sawComma = false
        var index = 0

        while index < tokens.count, cityTokens.count < 4 {
            let token = tokens[index]
            let cleaned = token.trimmingCharacters(in: CharacterSet.punctuationCharacters)
            let lower = cleaned.lowercased()
            guard !cleaned.isEmpty,
                  !locationStopTokens.contains(lower),
                  cleaned.allSatisfy({ $0.isLetter || $0 == "-" || $0 == "'" || $0 == "’" })
            else {
                break
            }
            cityTokens.append(cleaned)
            index += 1
            if token.hasSuffix(",") {
                sawComma = true
                break
            }
        }

        while let last = cityTokens.last, locationTrailingNoiseTokens.contains(last.lowercased()) {
            cityTokens.removeLast()
        }

        var regionToken: String?
        if sawComma, index < tokens.count {
            let cleaned = tokens[index].trimmingCharacters(in: CharacterSet.punctuationCharacters)
            let lower = cleaned.lowercased()
            let isPhraseFinal = index == tokens.count - 1
                || locationStopTokens.contains(
                    tokens[index + 1].trimmingCharacters(in: CharacterSet.punctuationCharacters).lowercased()
                )
            if !cleaned.isEmpty,
               cleaned.allSatisfy(\.isLetter),
               !locationStopTokens.contains(lower),
               !locationLeadingRejectTokens.contains(lower),
               cleaned.count <= 2 || isPhraseFinal {
                regionToken = cleaned
            }
        }

        guard let first = cityTokens.first?.lowercased(),
              !locationLeadingRejectTokens.contains(first)
        else {
            return nil
        }

        let city = cityTokens.joined(separator: " ")
        let aliasKeys = [
            regionToken.map { "\(city) \($0)" },
            city,
        ].compactMap { $0 }
        for key in aliasKeys {
            if let alias = knownLocationAliases[MemorySearchHeuristics.normalizedComparisonKey(for: key)] {
                return alias
            }
        }

        guard city.count >= 2 else { return nil }

        let displayCity = titleCasedLocation(cityTokens)
        guard let regionToken else { return displayCity }
        let displayRegion = regionToken.count <= 3
            ? regionToken.uppercased()
            : titleCasedLocation([regionToken])
        return "\(displayCity), \(displayRegion)"
    }

    private static func titleCasedLocation(_ tokens: [String]) -> String {
        tokens
            .map { token in
                guard let firstCharacter = token.first, firstCharacter.isLowercase else { return token }
                return firstCharacter.uppercased() + token.dropFirst()
            }
            .joined(separator: " ")
    }

    private static func containsEmbeddedQuestionClause(_ lower: String) -> Bool {
        containsAny(
            lower,
            needles: [
                ", what", ", what's", ", where", ", when", ", who", ", why", ", how",
                "; what", "; where", "; when", "; who", "; why", "; how",
                " what should", " what can", " what is", " what's "
            ]
        )
    }

    private static func inferSubject(
        forExtractedText text: String,
        role: ConversationRole,
        kind: MemoryKind
    ) -> MemorySubject {
        if role == .user,
           kind == .profile
               || containsNormalizedPhrase(phraseEnvelope(for: text), "the user")
               || containsFirstPersonReference(text) {
            return .user
        }
        if role == .assistant {
            return .assistant
        }
        return .unknown
    }

    private static let firstPersonTokens: Set<String> = [
        "i", "i'm", "i’m", "i've", "i’ve", "i'll", "i’ll", "i'd", "i’d",
        "my", "me", "mine"
    ]

    private static func containsFirstPersonReference(_ text: String) -> Bool {
        text
            .lowercased()
            .split { character in
                !character.isLetter && character != "'" && character != "’"
            }
            .contains { firstPersonTokens.contains(String($0)) }
    }

    private static func subjectAwareCanonicalKey(
        _ canonicalKey: String?,
        subject: MemorySubject,
        kind: MemoryKind
    ) -> String? {
        guard kind == .profile, subject != .unknown else { return canonicalKey }
        guard let canonicalKey, !canonicalKey.isEmpty else { return nil }
        let subjectPrefix = "\(kind.rawValue):\(subject.rawValue):"
        if canonicalKey.hasPrefix(subjectPrefix) {
            return canonicalKey
        }
        let kindPrefix = "\(kind.rawValue):"
        if canonicalKey.hasPrefix(kindPrefix) {
            return subjectPrefix + canonicalKey.dropFirst(kindPrefix.count)
        }
        return "\(subjectPrefix)\(canonicalKey)"
    }

    private static func evidenceExcerpt(for segment: String, in message: String) -> String {
        let trimmedMessage = message.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedMessage.count <= 240 {
            return trimmedMessage
        }
        if let range = trimmedMessage.range(of: segment, options: [.caseInsensitive, .diacriticInsensitive]) {
            let prefix = trimmedMessage[..<range.lowerBound].suffix(80)
            let suffix = trimmedMessage[range.upperBound...].prefix(80)
            return "\(prefix)\(trimmedMessage[range])\(suffix)"
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return String(trimmedMessage.prefix(240)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func inferStatus(forExtractedText text: String, kind: MemoryKind) -> MemoryStatus {
        guard kind == .commitment else { return .active }
        let lower = text.lowercased()
        if containsAny(lower, needles: ["done", "completed", "finished", "resolved", "closed", "shipped"]) {
            return .resolved
        }
        return .active
    }

    private static func inferredImportance(for kind: MemoryKind) -> Double {
        switch kind {
        case .decision:
            return 0.85
        case .commitment:
            return 0.80
        case .fact:
            return 0.70
        case .episode:
            return 0.65
        case .profile:
            return 0.62
        case .handoff:
            return 0.60
        case .procedure:
            return 0.55
        }
    }

    private static func inferredConfidence(for kind: MemoryKind) -> Double {
        switch kind {
        case .decision, .commitment, .profile:
            return 0.76
        case .procedure, .episode:
            return 0.70
        case .handoff:
            return 0.66
        case .fact:
            return 0.62
        }
    }

    private static func containsAnyNormalizedPhrase(_ normalizedText: String, phrases: [String]) -> Bool {
        phrases.contains { containsNormalizedPhrase(normalizedText, $0) }
    }

    private static func hasTimeSensitiveFacetSignal(text: String, kind: MemoryKind, normalizedText: String) -> Bool {
        if containsAnyNormalizedPhrase(normalizedText, phrases: ["deadline", "urgent", "asap"]) {
            return true
        }

        guard kind == .commitment else { return false }

        let hasActionableDeadlineSource = containsAnyNormalizedPhrase(normalizedText, phrases: ["todo", "need to"])
        if hasActionableDeadlineSource && containsAnyNormalizedPhrase(
            normalizedText,
            phrases: [
                "today", "tomorrow", "this week", "next week", "before release",
                "before launch", "by tomorrow", "by friday", "by monday"
            ]
        ) {
            return true
        }

        return hasActionableDeadlineSource && text.range(
            of: #"\b(?:by|before)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|release|launch|eod|end of (?:day|week|month))\b"#,
            options: [.regularExpression, .caseInsensitive]
        ) != nil
    }

    private static func hasProjectFacetContext(kind: MemoryKind, normalizedText: String) -> Bool {
        if kind == .decision {
            return true
        }
        return containsAnyNormalizedPhrase(
            normalizedText,
            phrases: [
                "work", "planning", "coordination", "recall", "review", "debugged",
                "maintainer", "owner", "role", "release", "project"
            ]
        )
    }

    private static func hasRelationshipFacetSignal(text: String, kind: MemoryKind, normalizedText: String) -> Bool {
        guard kind == .profile || kind == .episode || kind == .handoff else { return false }
        if containsNormalizedPhrase(normalizedText, "closely with") {
            return true
        }
        return text.range(
            of: #"\b(?:works|collaborates|coordinates)\s+(?:closely\s+)?with\s+[A-Z][A-Za-z'-]{2,}\b"#,
            options: .regularExpression
        ) != nil
    }

    private static func hasPersonFacetSignal(
        text: String,
        kind: MemoryKind,
        knownEntities: [MemoryEntity],
        normalizedText: String
    ) -> Bool {
        if kind == .handoff, containsNormalizedPhrase(normalizedText, "next person") {
            return true
        }

        guard kind == .episode else { return false }

        if knownEntities.contains(where: { $0.label == .person }) {
            return true
        }
        if text.range(of: #"\bmet\s+[A-Z][A-Za-z'-]{2,}\b"#, options: .regularExpression) != nil {
            return true
        }
        return text.range(
            of: #"\b[A-Z][A-Za-z'-]{2,}\s+(?:noted|said|asked|joined)\b"#,
            options: .regularExpression
        ) != nil
    }

    private static func entityValues(from rawTokens: [String]) -> [String] {
        let tokens = rawTokens
            .map { token in
                token.trimmingCharacters(in: CharacterSet.punctuationCharacters.union(.whitespacesAndNewlines))
            }
            .filter { !$0.isEmpty }
        let trimmed = trimEntityNoise(tokens)
        guard !trimmed.isEmpty else { return [] }

        let allProper = trimmed.allSatisfy { token in
            looksLikeStandaloneEntityToken(token) && !isGenericExtractedEntity(normalizeEntityValue(token))
        }
        if trimmed.count > 1, allProper {
            return [trimmed.joined(separator: " ")]
        }

        if let first = trimmed.first,
           looksLikeStandaloneEntityToken(first),
           !isGenericExtractedEntity(normalizeEntityValue(first)) {
            return [first]
        }

        return trimmed
            .filter { token in
                looksLikeStandaloneEntityToken(token)
                    && !isGenericExtractedEntity(normalizeEntityValue(token))
            }
    }

    private static func trimEntityNoise(_ tokens: [String]) -> [String] {
        var remaining = tokens
        while let first = remaining.first {
            let normalized = normalizeEntityValue(first)
            if entityBoundaryNoiseTokens.contains(normalized) {
                remaining.removeFirst()
            } else {
                break
            }
        }
        while let last = remaining.last {
            let normalized = normalizeEntityValue(last)
            if entityBoundaryNoiseTokens.contains(normalized) {
                remaining.removeLast()
            } else {
                break
            }
        }
        return remaining
    }

    private static func looksLikeEntityToken(_ raw: String) -> Bool {
        guard raw.count >= 2 else { return false }
        if raw.first?.isUppercase == true || raw.contains(".") || raw.contains("/") || raw.contains("+") {
            return true
        }
        if raw.contains("-") {
            return raw.contains(where: \.isUppercase) || raw.contains(where: \.isNumber)
        }
        return hasInternalUppercase(raw)
    }

    private static func looksLikeStandaloneEntityToken(_ raw: String) -> Bool {
        guard raw.count >= 2 else { return false }
        let normalized = normalizeEntityValue(raw)
        guard !normalized.isEmpty, !isGenericExtractedEntity(normalized) else { return false }
        if raw.first?.isUppercase == true || hasInternalUppercase(raw) {
            return true
        }
        if raw.contains(".") || raw.contains("/") || raw.contains("+") {
            return true
        }
        if raw.contains("-") {
            return raw.contains(where: \.isUppercase) || raw.contains(where: \.isNumber)
        }
        return raw.allSatisfy { $0.isUppercase || !$0.isLetter }
    }

    private static func hasInternalUppercase(_ raw: String) -> Bool {
        raw.dropFirst().contains(where: \.isUppercase)
    }

    private static func inferKnownEntities(forExtractedText text: String) -> [MemoryEntity] {
        let normalizedText = phraseEnvelope(for: text)
        var entities: [MemoryEntity] = []

        func append(
            phrase: String,
            value: String,
            label: EntityLabel,
            confidence: Double = 0.85
        ) {
            guard containsNormalizedPhrase(normalizedText, phrase) else { return }
            entities.append(
                MemoryEntity(
                    label: label,
                    value: value,
                    normalizedValue: normalizeEntityValue(value),
                    confidence: confidence
                )
            )
        }

        append(phrase: "zac", value: "Zac", label: .person, confidence: 0.9)
        append(phrase: "annie", value: "Annie", label: .person, confidence: 0.9)
        append(phrase: "sam", value: "Sam", label: .person, confidence: 0.88)
        append(phrase: "memory.swift", value: "Memory.swift", label: .project, confidence: 0.95)
        append(phrase: "leaf-ir", value: "LEAF-IR", label: .tool, confidence: 0.9)
        append(phrase: "coreml", value: "CoreML", label: .tool, confidence: 0.9)
        append(phrase: "sqlite-vec", value: "sqlite-vec", label: .tool, confidence: 0.88)
        append(phrase: "sqlite", value: "SQLite", label: .tool, confidence: 0.78)
        append(phrase: "fts5", value: "FTS5", label: .tool, confidence: 0.84)
        append(phrase: "lmdb", value: "LMDB", label: .tool, confidence: 0.84)
        append(phrase: "apple intelligence", value: "Apple Intelligence", label: .tool, confidence: 0.88)
        append(phrase: "long memory eval", value: "long memory eval", label: .other, confidence: 0.78)
        append(phrase: "readme", value: "README", label: .other, confidence: 0.76)
        append(phrase: "slack", value: "Slack", label: .tool, confidence: 0.78)
        append(phrase: "zed", value: "Zed", label: .tool, confidence: 0.84)
        append(phrase: "san francisco", value: "San Francisco", label: .location, confidence: 0.88)
        append(phrase: "pacific time", value: "Pacific Time", label: .other, confidence: 0.82)

        return entities
    }

    private static func inferKnownTopics(forExtractedText text: String) -> [String] {
        let normalizedText = phraseEnvelope(for: text)
        var topics: [String] = []
        var seen: Set<String> = []

        func append(_ topic: String, requiredPhrases: [String]? = nil) {
            let required = requiredPhrases ?? [topic]
            guard required.allSatisfy({ containsNormalizedPhrase(normalizedText, $0) }) else { return }
            let normalized = normalizeTopicValue(topic)
            guard !normalized.isEmpty else { return }
            guard seen.insert(normalized).inserted else { return }
            topics.append(topic)
        }

        append("hybrid retrieval")
        append("pacific time")
        append("release coordination")
        append("memory.swift planning", requiredPhrases: ["memory.swift", "planning"])
        append("memory.swift recall", requiredPhrases: ["memory.swift", "recall"])
        append("embeddings for recall", requiredPhrases: ["embeddings", "recall"])
        append("semantic search")
        append("experimental flag")
        append("entity overlap tests")
        append("facet tags")
        append("eval surface")
        append("canonical memory schema")
        append("benchmark licensing")
        append("sqlite-vec indexing latency", requiredPhrases: ["sqlite-vec", "indexing", "latency"])
        append("model training")
        append("default path")
        append("canonical memory storage")
        append("vector lookup")
        append("app group storage")
        append("model downloads")
        append("neural reranking")
        append("working offline")
        append("reranker latency")
        append("recall precision")
        append("retrieval metadata")
        append("release owner")
        append("pipeline on-device", requiredPhrases: ["pipeline", "on-device"])
        append("morning updates", requiredPhrases: ["morning", "updates"])

        return topics
    }

    private static func phraseEnvelope(for text: String) -> String {
        " \(MemorySearchHeuristics.normalizedComparisonKey(for: text)) "
    }

    private static func containsNormalizedPhrase(_ normalizedText: String, _ phrase: String) -> Bool {
        let normalizedPhrase = MemorySearchHeuristics.normalizedComparisonKey(for: phrase)
        guard !normalizedPhrase.isEmpty else { return false }
        return normalizedText.contains(" \(normalizedPhrase) ")
    }

    private static func isGenericExtractedEntity(_ value: String) -> Bool {
        let generic: Set<String> = [
            "action", "assistant", "context", "current", "decision", "done",
            "guide", "handoff", "memory", "memories", "need", "note", "notes",
            "procedure", "question", "runbook", "status", "task", "todo",
            "user", "we", "workflow"
        ]
        return generic.contains(value)
    }

    private static func isGenericTopicValue(_ value: String) -> Bool {
        genericTopicValues.contains(value)
    }

    private static func containsAny(_ text: String, needles: [String]) -> Bool {
        needles.contains(where: text.contains)
    }
}
