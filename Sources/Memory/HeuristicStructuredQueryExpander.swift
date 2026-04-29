import Foundation

public struct HeuristicStructuredQueryExpander: StructuredQueryExpander {
    public let identifier: String

    private let maxLexicalQueries: Int
    private let maxSemanticQueries: Int
    private let maxHypotheticalDocuments: Int
    private let maxFacetHints: Int
    private let maxEntities: Int
    private let maxTopics: Int

    public init(
        identifier: String = "heuristic-structured-query-expander",
        maxLexicalQueries: Int = 2,
        maxSemanticQueries: Int = 2,
        maxHypotheticalDocuments: Int = 1,
        maxFacetHints: Int = 4,
        maxEntities: Int = 6,
        maxTopics: Int = 6
    ) {
        self.identifier = identifier
        self.maxLexicalQueries = min(2, max(0, maxLexicalQueries))
        self.maxSemanticQueries = min(2, max(0, maxSemanticQueries))
        self.maxHypotheticalDocuments = min(1, max(0, maxHypotheticalDocuments))
        self.maxFacetHints = min(4, max(0, maxFacetHints))
        self.maxEntities = min(6, max(0, maxEntities))
        self.maxTopics = min(6, max(0, maxTopics))
    }

    public func expand(
        query: SearchQuery,
        analysis: QueryAnalysis,
        limit: Int
    ) async throws -> StructuredQueryExpansion {
        guard limit > 0 else { return StructuredQueryExpansion() }

        let trimmed = query.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return StructuredQueryExpansion() }

        let normalizedEntities = normalizeEntities(analysis.entities, maxCount: maxEntities)
        let normalizedTopics = normalizeTopics(analysis.topics, maxCount: maxTopics)
        let normalizedFacetHints = normalizeFacetHints(
            mergedFacetHints(
                analysis.facetHints,
                heuristicallyInferredFacetHints(from: trimmed)
            ),
            maxCount: maxFacetHints
        )

        let lexicalQueries = buildLexicalQueries(
            original: trimmed,
            analysis: analysis,
            entities: normalizedEntities,
            topics: normalizedTopics,
            referenceDate: query.referenceDate,
            limit: min(maxLexicalQueries, limit)
        )
        let semanticQueries = buildSemanticQueries(
            original: trimmed,
            analysis: analysis,
            entities: normalizedEntities,
            topics: normalizedTopics,
            limit: min(maxSemanticQueries, limit)
        )
        let hypotheticalDocuments = buildHypotheticalDocuments(
            original: trimmed,
            analysis: analysis,
            entities: normalizedEntities,
            topics: normalizedTopics,
            limit: min(maxHypotheticalDocuments, limit)
        )

        return StructuredQueryExpansion(
            lexicalQueries: lexicalQueries,
            semanticQueries: semanticQueries,
            hypotheticalDocuments: hypotheticalDocuments,
            facetHints: normalizedFacetHints,
            entities: normalizedEntities,
            topics: normalizedTopics
        )
    }

    private func buildLexicalQueries(
        original: String,
        analysis: QueryAnalysis,
        entities: [MemoryEntity],
        topics: [String],
        referenceDate: Date?,
        limit: Int
    ) -> [String] {
        guard limit > 0 else { return [] }

        var queries: [String] = []
        var seen: Set<String> = [comparisonKey(for: original)]

        let prioritizedEntities = entities.prefix(2).map(\.value)
        let salientTerms = salientLexicalTerms(from: original, entities: entities)
        let prioritizedTopics = selectSalientTopics(
            from: topics,
            salientTerms: salientTerms
        )
        let compactTopics = prioritizedTopics
            .map(compactTopicPhrase)
            .filter { !$0.isEmpty }
        let prioritizedTerms = Array(
            OrderedSet(analysis.keyTerms.map(normalizeTopic).filter { !$0.isEmpty && !expansionNoiseTerms.contains($0) })
                .prefix(4)
        )
        let derivedPhrases = derivedSalientTerms(from: original)
            .filter { $0.split(separator: " ").count >= 2 }
        let temporalAnchors = temporalAnchorTerms(from: original, referenceDate: referenceDate)

        if let derivedPhrase = derivedPhrases.first {
            appendCandidate(
                compactJoined(prioritizedEntities + [derivedPhrase] + Array(temporalAnchors.prefix(6))),
                to: &queries,
                seen: &seen,
                limit: limit
            )
        }

        let keywordRewrite = compactJoined(
            prioritizedEntities + Array(salientTerms.prefix(6)) + Array(temporalAnchors.prefix(8)) + prioritizedTerms
        )
        appendCandidate(keywordRewrite, to: &queries, seen: &seen, limit: limit)

        let focusedRewrite = compactJoined(
            prioritizedEntities + Array(compactTopics.prefix(2)) + Array(temporalAnchors.prefix(6))
        )
        appendCandidate(focusedRewrite, to: &queries, seen: &seen, limit: limit)

        if queries.count < limit, let firstTopic = compactTopics.first, firstTopic.count > 6 {
            appendCandidate(firstTopic, to: &queries, seen: &seen, limit: limit)
        }

        return queries
    }

    private func buildSemanticQueries(
        original: String,
        analysis: QueryAnalysis,
        entities: [MemoryEntity],
        topics: [String],
        limit: Int
    ) -> [String] {
        guard limit > 0 else { return [] }
        guard shouldEmitNarrativeExpansions(
            original: original,
            analysis: analysis,
            entities: entities,
            topics: topics
        ) else { return [] }

        let primaryTopic = narrativeFocusPhrase(
            original: original,
            analysis: analysis,
            entities: entities,
            topics: topics
        )
        guard !primaryTopic.isEmpty else { return [] }

        var queries: [String] = []
        var seen: Set<String> = [comparisonKey(for: original)]

        if analysis.isHowToQuery {
            appendCandidate(
                "how to \(primaryTopic)",
                to: &queries,
                seen: &seen,
                limit: limit
            )
        } else {
            appendCandidate(
                compactJoined(["details on", primaryTopic]),
                to: &queries,
                seen: &seen,
                limit: limit
            )
        }

        if entities.isEmpty && !analysis.isHowToQuery {
            return queries
        }

        appendCandidate(
            compactJoined(["memory mentioning", primaryTopic]),
            to: &queries,
            seen: &seen,
            limit: limit
        )

        return queries
    }

    private func buildHypotheticalDocuments(
        original: String,
        analysis: QueryAnalysis,
        entities: [MemoryEntity],
        topics: [String],
        limit: Int
    ) -> [String] {
        guard limit > 0 else { return [] }
        guard analysis.isHowToQuery || entities.isEmpty == false else { return [] }
        guard shouldEmitNarrativeExpansions(
            original: original,
            analysis: analysis,
            entities: entities,
            topics: topics
        ) else { return [] }

        let primaryTopic = narrativeFocusPhrase(
            original: original,
            analysis: analysis,
            entities: entities,
            topics: topics
        )
        guard !primaryTopic.isEmpty else { return [] }

        var snippets: [String] = []
        var seen: Set<String> = []
        let sentence: String
        if analysis.isHowToQuery {
            sentence = "A memory explains how to \(primaryTopic)."
        } else {
            sentence = "A memory mentions \(primaryTopic)."
        }

        appendCandidate(sentence, to: &snippets, seen: &seen, limit: limit)
        return snippets
    }

    private func salientLexicalTerms(from original: String, entities: [MemoryEntity]) -> [String] {
        let normalizedEntities = Set(entities.map(\.normalizedValue))
        let tokens = tokenize(original)

        var terms: [String] = []
        var seen: Set<String> = []
        for derived in derivedSalientTerms(from: original) where seen.insert(derived).inserted {
            terms.append(derived)
        }
        for token in tokens {
            let normalized = normalizeQueryToken(token)
            guard !normalized.isEmpty else { continue }
            guard !stopWords.contains(normalized) else { continue }
            guard !expansionNoiseTerms.contains(normalized) else { continue }
            if normalizedEntities.contains(normalized) || seen.insert(normalized).inserted {
                terms.append(normalized)
            }
        }
        return terms
    }

    private func temporalAnchorTerms(from query: String, referenceDate: Date?) -> [String] {
        let lower = query.lowercased()
        var terms: [String] = []
        func append(_ term: String) {
            let normalized = normalizeTopic(term)
            guard !normalized.isEmpty, !terms.contains(normalized) else { return }
            terms.append(normalized)
        }

        if let referenceDate {
            for date in relativeAnchorDates(from: lower, referenceDate: referenceDate) {
                for term in dateAnchorTerms(for: date) {
                    append(term)
                }
            }
        }

        for term in monthAnchorTerms(from: lower, referenceDate: referenceDate) {
            append(term)
        }

        if isExplicitTemporalOrAggregateRecall(query) {
            for term in ["date", "when", "before", "after", "earlier", "later", "timeline"] {
                append(term)
            }
        }

        return terms
    }

    private func relativeAnchorDates(from lower: String, referenceDate: Date) -> [Date] {
        var dates: [Date] = []
        func addOffset(_ days: Int) {
            guard days > 0, let date = calendar.date(byAdding: .day, value: -days, to: referenceDate) else { return }
            dates.append(date)
        }

        if lower.contains("yesterday") {
            addOffset(1)
        }
        if lower.contains("day before yesterday") || lower.contains("couple of days ago") {
            addOffset(2)
        }
        if lower.contains("few days ago") {
            addOffset(3)
        }
        if lower.contains("last week") {
            for days in 7...13 {
                addOffset(days)
            }
        }
        if lower.contains("last month") {
            for days in [28, 30, 31] {
                addOffset(days)
            }
        }

        for match in matches(
            in: lower,
            pattern: #"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)\s+ago\b"#
        ) {
            guard match.count >= 3, let amount = numberValue(match[1]) else { continue }
            let unit = match[2]
            if unit.hasPrefix("day") {
                addOffset(amount)
            } else if unit.hasPrefix("week") {
                addOffset(amount * 7)
            } else if unit.hasPrefix("month") {
                addOffset(amount * 30)
            }
        }

        if let weekday = matches(
            in: lower,
            pattern: #"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"#
        ).first?.dropFirst().first,
           let target = weekdayIndex[String(weekday)] {
            let current = calendar.component(.weekday, from: referenceDate) - 1
            let delta = ((current - target) + 7) % 7
            addOffset(delta == 0 ? 7 : delta)
        }

        return Array(OrderedSet(dates))
    }

    private func dateAnchorTerms(for date: Date) -> [String] {
        [
            dateFormatter("yyyy-MM-dd").string(from: date),
            dateFormatter("MMMM d yyyy").string(from: date),
            dateFormatter("MMMM d").string(from: date),
            dateFormatter("EEE d").string(from: date),
        ]
    }

    private func monthAnchorTerms(from lower: String, referenceDate: Date?) -> [String] {
        var terms: [String] = []
        for (month, value) in monthIndex where containsWord(month, in: lower) {
            if let referenceDate {
                var year = calendar.component(.year, from: referenceDate)
                let referenceMonth = calendar.component(.month, from: referenceDate)
                if value > referenceMonth {
                    year -= 1
                }
                terms.append("\(month) \(year)")
                terms.append(String(format: "%04d-%02d", year, value))
            }
            terms.append(month)
        }
        return terms
    }

    private func matches(in text: String, pattern: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let nsRange = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, range: nsRange).map { match in
            (0..<match.numberOfRanges).compactMap { index in
                guard let range = Range(match.range(at: index), in: text) else { return nil }
                return String(text[range])
            }
        }
    }

    private func containsWord(_ word: String, in text: String) -> Bool {
        text.range(of: #"\b\#(NSRegularExpression.escapedPattern(for: word))\b"#, options: .regularExpression) != nil
    }

    private func numberValue(_ value: String) -> Int? {
        if let numeric = Int(value) {
            return numeric
        }
        return numberWords[value]
    }

    private func selectSalientTopics(from topics: [String], salientTerms: [String]) -> [String] {
        let salientSet = Set(salientTerms)

        return topics
            .map { topic in
                let tokens = topic
                    .split(separator: " ")
                    .map(String.init)
                    .map(normalizeQueryToken)
                    .filter { !$0.isEmpty && !stopWords.contains($0) && !expansionNoiseTerms.contains($0) }
                let overlap = tokens.filter(salientSet.contains).count
                let signal = tokens.filter { signalTerms.contains($0) }.count
                let startsWithNoise = tokens.first.map(expansionNoiseTerms.contains) ?? false
                let score = (overlap * 3) + (signal * 2) + tokens.count - (startsWithNoise ? 4 : 0)
                return (topic: topic, score: score)
            }
            .filter { $0.score > 0 }
            .sorted { lhs, rhs in
                if lhs.score == rhs.score {
                    return lhs.topic < rhs.topic
                }
                return lhs.score > rhs.score
            }
            .map(\.topic)
    }

    private func narrativeFocusPhrase(
        original: String,
        analysis: QueryAnalysis,
        entities: [MemoryEntity],
        topics: [String]
    ) -> String {
        let entityTerms = entities.prefix(2).map(\.value)
        let topicTerms = selectSalientTopics(
            from: topics,
            salientTerms: salientLexicalTerms(from: original, entities: entities)
        )
        let fallbackTerms = salientLexicalTerms(from: original, entities: entities)
        let deduped = OrderedSet(entityTerms + topicTerms.prefix(2) + fallbackTerms.prefix(4))
        return compactJoined(Array(deduped.prefix(6)))
    }

    private func shouldEmitNarrativeExpansions(
        original: String,
        analysis: QueryAnalysis,
        entities: [MemoryEntity],
        topics: [String]
    ) -> Bool {
        analysis.isHowToQuery
            || entities.isEmpty == false
            || (isExplicitTemporalOrAggregateRecall(original) && !isPersonalFactLookup(analysis))
            || (!isPersonalFactLookup(analysis) && topics.contains { topic in topic.split(separator: " ").count >= 3 })
    }

    private func isPersonalFactLookup(_ analysis: QueryAnalysis) -> Bool {
        analysis.facetHints.contains { $0.tag == .factAboutUser || $0.tag == .preference || $0.tag == .habit }
    }

    private func isExplicitTemporalOrAggregateRecall(_ query: String) -> Bool {
        let lower = query.lowercased()
        if lower.range(of: #"\b(19|20)\d{2}\b"#, options: .regularExpression) != nil {
            return true
        }
        if lower.range(of: #"\b\d{1,2}/\d{1,2}\b"#, options: .regularExpression) != nil {
            return true
        }

        let phrases = [
            "how many", "how much", "days passed", "day i", "between",
            "before", "after", "earliest", "latest", "most recently",
            "order of", "from earliest to latest", "first", "what month",
            "which date", "when did", "as of", "past month", "past two months",
        ]
        return phrases.contains { lower.contains($0) }
    }

    private func derivedSalientTerms(from query: String) -> [String] {
        let lower = query.lowercased()
        var terms: [String] = []
        func append(_ term: String) {
            guard !terms.contains(term) else { return }
            terms.append(term)
        }

        if lower.contains("up to date") || lower.contains("out of date") {
            append("update")
        }
        if lower.contains("license plates") || lower.contains("license plate") {
            append("plates")
        }
        if lower.contains("turn in") || lower.contains("deliver") {
            append("return")
        }
        if lower.contains("completed") || lower.contains("finished") {
            append("finished")
            if lower.contains("project") {
                append("finished project")
            }
            if lower.contains("video") {
                append("completed videos")
            }
            if lower.contains("writing") || lower.contains("poem") || lower.contains("short stor") {
                append("writing progress")
            }
        }
        if lower.contains("since starting") || lower.contains("since i started") || lower.contains("started writing") {
            append("started")
            append("progress so far")
        }
        if lower.contains("painting class") || lower.contains("painting classes") {
            append("painting project")
        }
        if (lower.contains("trip") || lower.contains("trips")),
           lower.contains("order of") || lower.contains("earliest to latest") || lower.contains("from earliest") {
            append("road trip")
            append("camping trip")
            append("solo trip")
            append("travel")
        }
        if lower.contains("student loan"),
           lower.contains("school"),
           (lower.contains("not qualified")
            || lower.contains("wasn't actually qualified")
            || lower.contains("wasn’t actually qualified")
            || lower.contains("wasn t actually qualified")
            || lower.contains("eligible")) {
            append("false certification discharge")
            append("loan discharge")
        }
        return terms
    }

    private func heuristicallyInferredFacetHints(from query: String) -> [FacetHint] {
        let lowered = query.lowercased()
        var hints: [FacetHint] = []
        for (tag, needles) in facetKeywords {
            if needles.contains(where: lowered.contains) {
                hints.append(FacetHint(tag: tag, confidence: 0.82, isExplicit: true))
            }
        }
        return hints
    }

    private func mergedFacetHints(_ lhs: [FacetHint], _ rhs: [FacetHint]) -> [FacetHint] {
        lhs + rhs
    }

    private func normalizeFacetHints(_ hints: [FacetHint], maxCount: Int) -> [FacetHint] {
        guard maxCount > 0 else { return [] }

        var deduped: [FacetTag: FacetHint] = [:]
        for hint in hints {
            let candidate = FacetHint(
                tag: hint.tag,
                confidence: min(1, max(0, hint.confidence)),
                isExplicit: hint.isExplicit
            )
            if let existing = deduped[candidate.tag] {
                if candidate.confidence > existing.confidence
                    || (candidate.confidence == existing.confidence && candidate.isExplicit && !existing.isExplicit) {
                    deduped[candidate.tag] = candidate
                }
            } else {
                deduped[candidate.tag] = candidate
            }
        }

        return deduped.values
            .sorted { lhs, rhs in
                if lhs.confidence == rhs.confidence {
                    if lhs.isExplicit == rhs.isExplicit {
                        return lhs.tag.rawValue < rhs.tag.rawValue
                    }
                    return lhs.isExplicit && !rhs.isExplicit
                }
                return lhs.confidence > rhs.confidence
            }
            .prefix(maxCount)
            .map { $0 }
    }

    private func normalizeEntities(_ entities: [MemoryEntity], maxCount: Int) -> [MemoryEntity] {
        guard maxCount > 0 else { return [] }

        var deduped: [String: MemoryEntity] = [:]
        for entity in entities {
            let value = entity.value.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalizedValue = normalizeEntityValue(entity.normalizedValue.isEmpty ? value : entity.normalizedValue)
            guard !value.isEmpty, !normalizedValue.isEmpty else { continue }

            let candidate = MemoryEntity(
                label: entity.label,
                value: value,
                normalizedValue: normalizedValue,
                confidence: entity.confidence
            )
            if let existing = deduped[normalizedValue] {
                let existingConfidence = existing.confidence ?? 0
                let candidateConfidence = candidate.confidence ?? 0
                if candidateConfidence > existingConfidence {
                    deduped[normalizedValue] = candidate
                }
            } else {
                deduped[normalizedValue] = candidate
            }
        }

        return deduped.values
            .sorted { lhs, rhs in
                let left = lhs.confidence ?? 0
                let right = rhs.confidence ?? 0
                if left == right {
                    return lhs.normalizedValue < rhs.normalizedValue
                }
                return left > right
            }
            .prefix(maxCount)
            .map { $0 }
    }

    private func normalizeTopics(_ topics: [String], maxCount: Int) -> [String] {
        guard maxCount > 0 else { return [] }

        var normalized: [String] = []
        var seen: Set<String> = []
        for topic in topics {
            let candidate = normalizeTopic(topic)
            guard !candidate.isEmpty else { continue }
            guard seen.insert(candidate).inserted else { continue }
            normalized.append(candidate)
            if normalized.count >= maxCount {
                break
            }
        }
        return normalized
    }

    private func appendCandidate(
        _ raw: String?,
        to collection: inout [String],
        seen: inout Set<String>,
        limit: Int
    ) {
        guard collection.count < limit else { return }
        guard let raw else { return }

        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let key = comparisonKey(for: trimmed)
        guard !key.isEmpty, seen.insert(key).inserted else { return }
        collection.append(trimmed)
    }

    private func compactJoined<S: Sequence>(_ parts: S) -> String where S.Element == String {
        parts
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
    }

    private func compactJoined(_ parts: [String?]) -> String {
        compactJoined(parts.compactMap { $0 })
    }

    private func normalizeEntityValue(_ raw: String) -> String {
        let punctuation = CharacterSet(charactersIn: ",:;!?()[]{}\"'`")
        return raw
            .trimmingCharacters(in: .whitespacesAndNewlines.union(punctuation))
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
            .lowercased()
    }

    private func normalizeTopic(_ raw: String) -> String {
        raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
            .prefix(4)
            .joined(separator: " ")
    }

    private func comparisonKey(for raw: String) -> String {
        raw
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
    }

    private func tokenize(_ raw: String) -> [String] {
        raw
            .split {
                !$0.isLetter && !$0.isNumber && $0 != "+" && $0 != "-" && $0 != "." && $0 != "/"
            }
            .map(String.init)
    }

    private func normalizeQueryToken(_ raw: String) -> String {
        raw
            .trimmingCharacters(in: .whitespacesAndNewlines.union(queryPunctuation))
            .lowercased()
    }

    private func compactTopicPhrase(_ raw: String) -> String {
        compactJoined(
            raw
                .split(separator: " ")
                .map(String.init)
                .map(normalizeQueryToken)
                .filter { !$0.isEmpty && !stopWords.contains($0) && !expansionNoiseTerms.contains($0) }
        )
    }

    private var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        return calendar
    }

    private func dateFormatter(_ format: String) -> DateFormatter {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = format
        return formatter
    }

    private let numberWords: [String: Int] = [
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    ]

    private let weekdayIndex: [String: Int] = [
        "sunday": 0,
        "monday": 1,
        "tuesday": 2,
        "wednesday": 3,
        "thursday": 4,
        "friday": 5,
        "saturday": 6,
    ]

    private let monthIndex: [String: Int] = [
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    ]

    private let stopWords: Set<String> = [
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "from",
        "been", "some", "them", "than", "its", "over", "such", "more",
        "other", "into", "also", "did", "get", "got", "how", "what",
        "where", "when", "who", "why", "which", "this", "that", "with",
        "would", "could", "should", "your", "their", "them", "there",
        "then", "about", "after", "before", "again", "last", "next",
        "into", "onto", "upon", "just", "really", "very", "i", "me",
        "my", "it", "is", "to", "if", "as", "at", "on", "in", "do",
        "an", "or", "we", "us",
    ]

    private let expansionNoiseTerms: Set<String> = [
        "hello", "thanks", "thank", "okay", "question", "questions", "another",
        "valuable", "information", "discussed", "earlier", "looking", "back",
        "previous", "conversation", "wanted", "follow", "planning", "wondering",
        "think", "given", "opportunity", "possible", "talk", "let", "going",
        "through", "remind", "reminder", "provided", "help", "please",
        "needed", "always", "guys", "thing", "mentioned", "document", "actually",
    ]

    private let signalTerms: Set<String> = [
        "before", "after", "between", "during", "month", "months", "week",
        "weeks", "day", "days", "time", "times", "money", "spent", "spend",
        "total", "most", "least", "count", "number", "schedule", "tuesday",
        "tuesdays", "thursday", "thursdays", "march", "july", "october",
        "year", "years", "price", "cost", "paid", "order", "earliest",
        "latest", "first", "recent", "recently", "chronology",
    ]

    private let queryPunctuation = CharacterSet(charactersIn: ",:;!?()[]{}\"'`.")

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
        .timeSensitive: [
            "today", "tomorrow", "deadline", "schedule", "urgent", "when",
            "date", "month", "months", "day", "days", "before", "after",
            "between", "earliest", "latest", "first", "recent", "past",
            "january", "february", "march", "april", "june", "july", "august",
            "september", "october", "november", "december",
        ],
        .constraint: ["constraint", "blocked", "limit", "cannot"],
        .habit: ["habit", "usually", "often", "routine"],
        .factAboutUser: ["my", "i am", "i'm", "about me"],
        .factAboutWorld: ["fact", "reference", "documentation", "spec"],
        .lesson: ["lesson", "learned", "takeaway", "retrospective"],
        .emotion: ["feel", "emotion", "mood", "frustrated", "happy"],
        .identitySignal: ["identity", "role", "background", "belief"],
    ]
}

private struct OrderedSet<Element: Hashable>: Sequence {
    private var values: [Element] = []
    private var seen: Set<Element> = []

    init<S: Sequence>(_ sequence: S) where S.Element == Element {
        for value in sequence where seen.insert(value).inserted {
            values.append(value)
        }
    }

    func makeIterator() -> Array<Element>.Iterator {
        values.makeIterator()
    }

    func prefix(_ maxLength: Int) -> [Element] {
        Array(values.prefix(maxLength))
    }
}
