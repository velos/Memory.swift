import Foundation

internal struct MultiEvidenceSupportRanker: Sendable {
    private struct SupportCandidate {
        var result: SearchResult
        var originalIndex: Int
        var documentRank: Int
        var supportScore: Double
        var supportGroupKey: String?
    }

    private struct MonthDayAnchor: Hashable {
        var month: Int
        var day: Int
    }

    private static let monthNameToNumber: [String: Int] = [
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

    internal func order(
        _ results: [SearchResult],
        queryText: String,
        effectiveLimit: Int,
        dedupeDocuments: Bool,
        activeOnlyByDefault: Bool
    ) -> [SearchResult] {
        let supportOrdered = orderMultiEvidenceSupportResults(
            results,
            queryText: queryText,
            effectiveLimit: effectiveLimit,
            dedupeDocuments: dedupeDocuments,
            activeOnlyByDefault: activeOnlyByDefault
        )
        return preserveDirectLookupContinuationResults(
            supportOrdered,
            queryText: queryText,
            effectiveLimit: effectiveLimit,
            dedupeDocuments: dedupeDocuments,
            activeOnlyByDefault: activeOnlyByDefault
        )
    }

    private func orderMultiEvidenceSupportResults(
        _ results: [SearchResult],
        queryText: String,
        effectiveLimit: Int,
        dedupeDocuments: Bool,
        activeOnlyByDefault: Bool
    ) -> [SearchResult] {
        guard dedupeDocuments,
              effectiveLimit > 1,
              MemorySearchHeuristics.isMultiEvidenceSupportQuery(queryText),
              results.count > 10 else {
            return results
        }

        let candidates = multiEvidenceSupportCandidates(
            from: results,
            activeOnlyByDefault: activeOnlyByDefault,
            reserveLimit: 80
        )

        let topWindow = min(10, effectiveLimit, candidates.count)
        guard topWindow >= 4 else { return results }

        let poolLimit = min(candidates.count, max(50, topWindow * 5))
        let pool = Array(candidates.prefix(poolLimit))
        let anchorCount = min(2, topWindow)
        let anchors = Array(pool.prefix(anchorCount))
        var selected = anchors
        var selectedKeys = Set(anchors.map { MemorySearchHeuristics.normalizedComparisonKey(for: $0.result.documentPath) })

        let originalWindow = Array(pool.prefix(topWindow))
        let medianSupport = medianSupportScore(originalWindow)
        let floor = medianSupport * 0.55
        let rankWindow = min(30, poolLimit)
        let continuationRankWindow = min(30, poolLimit)
        let continuationFloor = max(0.06, medianSupport * 0.12)
        let originalSupportGroups = Set(originalWindow.compactMap(\.supportGroupKey))
        let continuationLimit = min(3, max(1, topWindow - anchorCount))
        let temporalPromotionLimit = temporalEvidencePromotionLimit(
            queryText: queryText,
            topWindow: topWindow,
            anchorCount: anchorCount
        )

        if temporalPromotionLimit > 0 {
            let temporalScoreFloor = temporalEvidencePromotionScoreFloor(queryText: queryText)
            let selectedTemporalBuckets = Set(selected.compactMap { temporalEvidenceBucket(for: $0.result) })
            let temporalCandidates = temporalEvidencePromotionCandidates(
                from: pool,
                droppingFirst: anchorCount,
                queryText: queryText,
                scoreFloor: temporalScoreFloor,
                selectedTemporalBuckets: selectedTemporalBuckets
            )

            var promotedTemporal = 0
            var promotedTemporalBuckets = selectedTemporalBuckets
            for candidate in temporalCandidates where selected.count < topWindow && promotedTemporal < temporalPromotionLimit {
                let documentKey = MemorySearchHeuristics.normalizedComparisonKey(for: candidate.result.documentPath)
                guard selectedKeys.insert(documentKey).inserted else { continue }
                if let bucket = temporalEvidenceBucket(for: candidate.result) {
                    guard promotedTemporalBuckets.insert(bucket).inserted || candidate.result.score.temporal >= 0.055 else {
                        continue
                    }
                }
                selected.append(candidate)
                promotedTemporal += 1
            }
        }

        let continuationCandidates = pool
            .dropFirst(topWindow)
            .filter { candidate in
                guard candidate.documentRank <= continuationRankWindow,
                      let supportGroupKey = candidate.supportGroupKey,
                      originalSupportGroups.contains(supportGroupKey) else {
                    return false
                }
                return candidate.supportScore >= continuationFloor || candidate.documentRank <= rankWindow
            }
            .sorted(by: compareMultiEvidenceContinuationCandidates(_:_:))

        var promotedContinuationGroups: Set<String> = []
        for candidate in continuationCandidates where selected.count < topWindow {
            guard promotedContinuationGroups.count < continuationLimit,
                  let supportGroupKey = candidate.supportGroupKey,
                  promotedContinuationGroups.insert(supportGroupKey).inserted else {
                continue
            }
            let documentKey = MemorySearchHeuristics.normalizedComparisonKey(for: candidate.result.documentPath)
            guard selectedKeys.insert(documentKey).inserted else { continue }
            selected.append(candidate)
        }

        let supportCandidates = pool
            .dropFirst(anchorCount)
            .filter { candidate in
                candidate.documentRank <= rankWindow || candidate.supportScore >= floor
            }
            .sorted(by: compareMultiEvidenceSupportCandidates(_:_:))

        for candidate in supportCandidates where selected.count < topWindow {
            let documentKey = MemorySearchHeuristics.normalizedComparisonKey(for: candidate.result.documentPath)
            guard selectedKeys.insert(documentKey).inserted else { continue }
            selected.append(candidate)
        }

        if selected.count < topWindow {
            for candidate in pool where selected.count < topWindow {
                let documentKey = MemorySearchHeuristics.normalizedComparisonKey(for: candidate.result.documentPath)
                guard selectedKeys.insert(documentKey).inserted else { continue }
                selected.append(candidate)
            }
        }

        selected = promoteMultiEvidenceSiblingContinuations(
            selected,
            from: pool,
            topWindow: topWindow,
            medianSupport: medianSupport
        )
        selected = preserveHighScoringTopAnchorSiblingContinuations(
            selected,
            from: pool,
            medianSupport: medianSupport
        )

        let selectedChunkIDs = Set(selected.map { $0.result.chunkID })
        var ordered = selected.map(\.result)
        ordered.reserveCapacity(results.count)
        for result in results where !selectedChunkIDs.contains(result.chunkID) {
            ordered.append(result)
        }
        return ordered
    }

    private func preserveDirectLookupContinuationResults(
        _ results: [SearchResult],
        queryText: String,
        effectiveLimit: Int,
        dedupeDocuments: Bool,
        activeOnlyByDefault: Bool
    ) -> [SearchResult] {
        guard dedupeDocuments,
              effectiveLimit > 1,
              MemorySearchHeuristics.isDirectContinuationLookupQuery(queryText),
              results.count > effectiveLimit else {
            return results
        }

        let candidates = multiEvidenceSupportCandidates(
            from: results,
            activeOnlyByDefault: activeOnlyByDefault,
            reserveLimit: 80
        )
        let topWindow = min(10, effectiveLimit, candidates.count)
        guard topWindow >= 3, candidates.count > topWindow else { return results }

        let selected = Array(candidates.prefix(topWindow))
        let selectedGroups = Set(selected.compactMap(\.supportGroupKey))
        guard !selectedGroups.isEmpty else { return results }

        let scanLimit = min(candidates.count, 35)
        let continuationCandidates = candidates
            .prefix(scanLimit)
            .dropFirst(topWindow)
            .filter { candidate in
                guard let supportGroupKey = candidate.supportGroupKey,
                      selectedGroups.contains(supportGroupKey) else {
                    return false
                }
                return candidate.supportScore > 0
            }
            .sorted(by: compareMultiEvidenceContinuationCandidates(_:_:))

        guard let continuation = continuationCandidates.first,
              let replacementIndex = directLookupContinuationReplacementIndex(
                in: selected,
                candidate: continuation
              ) else {
            return results
        }

        var promoted = selected
        promoted[replacementIndex] = continuation

        let selectedChunkIDs = Set(promoted.map { $0.result.chunkID })
        var ordered = promoted.map(\.result)
        ordered.reserveCapacity(results.count)
        for result in results where !selectedChunkIDs.contains(result.chunkID) {
            ordered.append(result)
        }
        return ordered
    }

    private func temporalEvidencePromotionLimit(
        queryText: String,
        topWindow: Int,
        anchorCount: Int
    ) -> Int {
        let available = max(0, topWindow - anchorCount)
        guard available > 0 else { return 0 }

        let dayAnchorCount = monthDayAnchors(from: queryText).count
        let monthAnchorCount = monthAnchors(from: queryText).count
        if dayAnchorCount >= 2 {
            return min(5, available)
        }
        if dayAnchorCount > 0 || monthAnchorCount > 0 {
            return min(4, available)
        }
        return 0
    }

    private func temporalEvidencePromotionScoreFloor(queryText: String) -> Double {
        monthDayAnchors(from: queryText).count >= 2 ? 0.055 : 0.018
    }

    private func temporalEvidencePromotionCandidates(
        from pool: [SupportCandidate],
        droppingFirst anchorCount: Int,
        queryText: String,
        scoreFloor: Double,
        selectedTemporalBuckets: Set<String>
    ) -> [SupportCandidate] {
        let rankWindow = min(60, pool.count)
        return pool
            .dropFirst(anchorCount)
            .filter { candidate in
                guard candidate.documentRank <= rankWindow else { return false }
                return temporalEvidenceCandidate(candidate, queryText: queryText, scoreFloor: scoreFloor)
            }
            .sorted {
                compareTemporalEvidenceCandidates(
                    $0,
                    $1,
                    selectedTemporalBuckets: selectedTemporalBuckets
                )
            }
    }

    private func temporalEvidenceCandidate(
        _ candidate: SupportCandidate,
        queryText: String,
        scoreFloor: Double
    ) -> Bool {
        if candidate.result.score.temporal >= scoreFloor {
            return true
        }
        guard scoreFloor <= 0.018 else {
            return false
        }
        guard MemorySearchHeuristics.isTemporalOrAggregateRecallQuery(queryText),
              temporalEvidenceBucket(for: candidate.result) != nil else {
            return false
        }
        return candidate.supportScore >= 0.025
    }

    private func compareTemporalEvidenceCandidates(
        _ lhs: SupportCandidate,
        _ rhs: SupportCandidate,
        selectedTemporalBuckets: Set<String>
    ) -> Bool {
        let lhsBucket = temporalEvidenceBucket(for: lhs.result)
        let rhsBucket = temporalEvidenceBucket(for: rhs.result)
        let lhsIsNewBucket = lhsBucket.map { !selectedTemporalBuckets.contains($0) } ?? false
        let rhsIsNewBucket = rhsBucket.map { !selectedTemporalBuckets.contains($0) } ?? false
        if lhsIsNewBucket != rhsIsNewBucket {
            return lhsIsNewBucket
        }
        if lhs.result.score.temporal != rhs.result.score.temporal {
            return lhs.result.score.temporal > rhs.result.score.temporal
        }
        if lhs.supportScore != rhs.supportScore {
            return lhs.supportScore > rhs.supportScore
        }
        if lhs.documentRank != rhs.documentRank {
            return lhs.documentRank < rhs.documentRank
        }
        return lhs.originalIndex < rhs.originalIndex
    }

    private func multiEvidenceSupportCandidates(
        from results: [SearchResult],
        activeOnlyByDefault: Bool,
        reserveLimit: Int
    ) -> [SupportCandidate] {
        var seenDocumentKeys: Set<String> = []
        var candidates: [SupportCandidate] = []
        candidates.reserveCapacity(min(results.count, reserveLimit))

        for (index, result) in results.enumerated() {
            if activeOnlyByDefault,
               let memoryStatus = result.memoryStatus,
               memoryStatus != .active {
                continue
            }

            let documentKey = MemorySearchHeuristics.normalizedComparisonKey(for: result.documentPath)
            guard seenDocumentKeys.insert(documentKey).inserted else { continue }
            let documentRank = candidates.count + 1
            candidates.append(
                SupportCandidate(
                    result: result,
                    originalIndex: index,
                    documentRank: documentRank,
                    supportScore: multiEvidenceSupportScore(for: result, documentRank: documentRank),
                    supportGroupKey: multiEvidenceSupportGroupKey(for: result.documentPath)
                )
            )
        }

        return candidates
    }

    private func promoteMultiEvidenceSiblingContinuations(
        _ selected: [SupportCandidate],
        from pool: [SupportCandidate],
        topWindow: Int,
        medianSupport: Double
    ) -> [SupportCandidate] {
        guard selected.count >= topWindow,
              pool.count > topWindow else {
            return selected
        }

        var promoted = selected
        var selectedKeys = Set(promoted.map { MemorySearchHeuristics.normalizedComparisonKey(for: $0.result.documentPath) })
        var groupCounts = multiEvidenceSupportGroupCounts(promoted)
        let protectedSupportGroupKeys = repeatedLeadingSupportGroupKeys(in: promoted)
        let selectedGroups = Set(groupCounts.keys)
        guard !selectedGroups.isEmpty else { return selected }

        let continuationRankWindow = min(60, pool.count)
        let continuationFloor = max(0.04, medianSupport * 0.08)
        let continuationCandidates = pool
            .dropFirst(topWindow)
            .filter { candidate in
                guard candidate.documentRank <= continuationRankWindow,
                      let supportGroupKey = candidate.supportGroupKey,
                      selectedGroups.contains(supportGroupKey),
                      !selectedKeys.contains(MemorySearchHeuristics.normalizedComparisonKey(for: candidate.result.documentPath)) else {
                    return false
                }
                return candidate.supportScore >= continuationFloor
            }
            .sorted(by: compareMultiEvidenceContinuationCandidates(_:_:))

        var promotionCount = 0
        let promotionLimit = 3
        let perGroupLimit = 3

        for candidate in continuationCandidates where promotionCount < promotionLimit {
            guard let supportGroupKey = candidate.supportGroupKey,
                  (groupCounts[supportGroupKey] ?? 0) < perGroupLimit,
                  let replacementIndex = multiEvidenceReplacementIndex(
                    in: promoted,
                    groupCounts: groupCounts,
                    protectedGroupKey: supportGroupKey,
                    protectedSupportGroupKeys: protectedSupportGroupKeys
                  ) else {
                continue
            }

            let removed = promoted[replacementIndex]
            let removedKey = MemorySearchHeuristics.normalizedComparisonKey(for: removed.result.documentPath)
            selectedKeys.remove(removedKey)
            if let removedGroupKey = removed.supportGroupKey {
                groupCounts[removedGroupKey] = max(0, (groupCounts[removedGroupKey] ?? 0) - 1)
            }

            promoted[replacementIndex] = candidate
            selectedKeys.insert(MemorySearchHeuristics.normalizedComparisonKey(for: candidate.result.documentPath))
            groupCounts[supportGroupKey] = (groupCounts[supportGroupKey] ?? 0) + 1
            promotionCount += 1
        }

        return promoted
    }

    private func repeatedLeadingSupportGroupKeys(
        in selected: [SupportCandidate]
    ) -> Set<String> {
        guard selected.count >= 2,
              let first = selected[0].supportGroupKey,
              first == selected[1].supportGroupKey else {
            return []
        }
        return [first]
    }

    private func preserveHighScoringTopAnchorSiblingContinuations(
        _ selected: [SupportCandidate],
        from pool: [SupportCandidate],
        medianSupport: Double
    ) -> [SupportCandidate] {
        guard selected.count >= 4,
              pool.count > selected.count else {
            return selected
        }

        let topAnchorGroups = Set(selected.prefix(2).compactMap(\.supportGroupKey))
        guard !topAnchorGroups.isEmpty else { return selected }

        var promoted = selected
        var selectedKeys = Set(promoted.map { MemorySearchHeuristics.normalizedComparisonKey(for: $0.result.documentPath) })
        let scanLimit = min(pool.count, 30)
        let scoreFloor = 0.30
        let supportFloor = max(0.12, medianSupport * 0.50)
        let replacementMargin = 0.10

        let candidates = pool.prefix(scanLimit)
            .filter { candidate in
                guard let supportGroupKey = candidate.supportGroupKey,
                      topAnchorGroups.contains(supportGroupKey),
                      !selectedKeys.contains(MemorySearchHeuristics.normalizedComparisonKey(for: candidate.result.documentPath)) else {
                    return false
                }
                return candidate.result.score.blended >= scoreFloor
                    && candidate.supportScore >= supportFloor
            }
            .sorted {
                if $0.result.score.blended == $1.result.score.blended {
                    return compareMultiEvidenceContinuationCandidates($0, $1)
                }
                return $0.result.score.blended > $1.result.score.blended
            }

        var insertedGroups: Set<String> = []
        for candidate in candidates {
            guard let supportGroupKey = candidate.supportGroupKey,
                  insertedGroups.insert(supportGroupKey).inserted,
                  let replacementIndex = highScoringSiblingReplacementIndex(
                    in: promoted,
                    candidate: candidate,
                    replacementMargin: replacementMargin
                  ) else {
                continue
            }

            let removed = promoted[replacementIndex]
            selectedKeys.remove(MemorySearchHeuristics.normalizedComparisonKey(for: removed.result.documentPath))
            promoted[replacementIndex] = candidate
            selectedKeys.insert(MemorySearchHeuristics.normalizedComparisonKey(for: candidate.result.documentPath))
        }

        return promoted
    }

    private func highScoringSiblingReplacementIndex(
        in selected: [SupportCandidate],
        candidate: SupportCandidate,
        replacementMargin: Double
    ) -> Int? {
        let groupCounts = multiEvidenceSupportGroupCounts(selected)
        return selected.enumerated()
            .filter { index, existing in
                guard index >= 2 else { return false }
                if let supportGroupKey = existing.supportGroupKey,
                   (groupCounts[supportGroupKey] ?? 0) > 1 {
                    return false
                }
                return existing.result.score.blended + replacementMargin < candidate.result.score.blended
            }
            .min { lhs, rhs in
                if lhs.element.result.score.blended == rhs.element.result.score.blended {
                    return lhs.offset > rhs.offset
                }
                return lhs.element.result.score.blended < rhs.element.result.score.blended
            }?
            .offset
    }

    private func directLookupContinuationReplacementIndex(
        in selected: [SupportCandidate],
        candidate: SupportCandidate
    ) -> Int? {
        let candidateKey = MemorySearchHeuristics.normalizedComparisonKey(for: candidate.result.documentPath)
        guard !selected.contains(where: { MemorySearchHeuristics.normalizedComparisonKey(for: $0.result.documentPath) == candidateKey }) else {
            return nil
        }

        return selected.enumerated()
            .filter { index, existing in
                guard index >= 2 else { return false }
                if let candidateGroup = candidate.supportGroupKey,
                   existing.supportGroupKey == candidateGroup {
                    return false
                }
                return true
            }
            .min { lhs, rhs in
                if lhs.element.supportScore == rhs.element.supportScore {
                    return lhs.offset > rhs.offset
                }
                return lhs.element.supportScore < rhs.element.supportScore
            }?
            .offset
    }

    private func multiEvidenceSupportGroupCounts(
        _ candidates: [SupportCandidate]
    ) -> [String: Int] {
        var counts: [String: Int] = [:]
        for candidate in candidates {
            guard let supportGroupKey = candidate.supportGroupKey else { continue }
            counts[supportGroupKey, default: 0] += 1
        }
        return counts
    }

    private func multiEvidenceReplacementIndex(
        in selected: [SupportCandidate],
        groupCounts: [String: Int],
        protectedGroupKey: String,
        protectedSupportGroupKeys: Set<String>
    ) -> Int? {
        let candidates = selected.enumerated().filter { index, candidate in
            guard index >= 2 else { return false }
            guard let supportGroupKey = candidate.supportGroupKey else { return true }
            guard !protectedSupportGroupKeys.contains(supportGroupKey) else { return false }
            return supportGroupKey != protectedGroupKey && (groupCounts[supportGroupKey] ?? 0) > 1
        }

        return candidates.min { lhs, rhs in
            if lhs.element.supportScore == rhs.element.supportScore {
                return lhs.offset > rhs.offset
            }
            return lhs.element.supportScore < rhs.element.supportScore
        }?.offset
    }

    private func compareMultiEvidenceSupportCandidates(
        _ lhs: SupportCandidate,
        _ rhs: SupportCandidate
    ) -> Bool {
        if lhs.supportScore == rhs.supportScore {
            if lhs.documentRank == rhs.documentRank {
                return lhs.originalIndex < rhs.originalIndex
            }
            return lhs.documentRank < rhs.documentRank
        }
        return lhs.supportScore > rhs.supportScore
    }

    private func compareMultiEvidenceContinuationCandidates(
        _ lhs: SupportCandidate,
        _ rhs: SupportCandidate
    ) -> Bool {
        let lhsPriority = multiEvidenceContinuationPriority(lhs)
        let rhsPriority = multiEvidenceContinuationPriority(rhs)
        if lhsPriority == rhsPriority {
            if lhs.documentRank == rhs.documentRank {
                return lhs.originalIndex < rhs.originalIndex
            }
            return lhs.documentRank < rhs.documentRank
        }
        return lhsPriority > rhsPriority
    }

    private func multiEvidenceContinuationPriority(_ candidate: SupportCandidate) -> Double {
        candidate.supportScore + (0.08 / sqrt(Double(max(1, candidate.documentRank))))
    }

    private func medianSupportScore(_ candidates: [SupportCandidate]) -> Double {
        guard !candidates.isEmpty else { return 0 }
        let sorted = candidates.map(\.supportScore).sorted()
        return sorted[sorted.count / 2]
    }

    private func multiEvidenceSupportScore(for result: SearchResult, documentRank: Int) -> Double {
        let score = result.score
        let strongestBranch = max(score.lexical, score.semantic)
        let branchAgreement = min(score.lexical, score.semantic)
        let metadataSupport = score.temporal + score.schema + score.tag + score.status
        let rankPrior = 0.02 / sqrt(Double(max(1, documentRank)))
        return strongestBranch + (0.45 * branchAgreement) + metadataSupport + rankPrior
    }

    private func temporalEvidenceBucket(for result: SearchResult) -> String? {
        let searchable = (
            result.documentPath + " " + (result.title ?? "") + " " + String(result.content.prefix(900))
        )
        .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
        .lowercased()

        let numericPatterns = [
            #"\b((?:19|20)\d{2})[-_/](\d{1,2})[-_/](\d{1,2})\b"#,
            #"\b(\d{1,2})[-_/](\d{1,2})[-_/]((?:19|20)\d{2})\b"#,
        ]
        for pattern in numericPatterns {
            for match in regexCaptureGroups(pattern: pattern, text: searchable) {
                if match[0].count == 4,
                   let year = Int(match[0]),
                   let month = Int(match[1]),
                   let day = Int(match[2]),
                   (1...12).contains(month),
                   (1...31).contains(day) {
                    return String(format: "%04d-%02d-%02d", year, month, day)
                }
                if match[2].count == 4,
                   let month = Int(match[0]),
                   let day = Int(match[1]),
                   let year = Int(match[2]),
                   (1...12).contains(month),
                   (1...31).contains(day) {
                    return String(format: "%04d-%02d-%02d", year, month, day)
                }
            }
        }

        let monthAlternation = Self.monthNameToNumber.keys.sorted().joined(separator: "|")
        let monthDayPattern = #"\b("# + monthAlternation + #")\s+(\d{1,2})(?:st|nd|rd|th)?\b"#
        for match in regexCaptureGroups(pattern: monthDayPattern, text: searchable) {
            guard match.count >= 2,
                  let month = Self.monthNameToNumber[match[0]],
                  let day = Int(match[1]),
                  (1...31).contains(day) else {
                continue
            }
            return String(format: "month-%02d-day-%02d", month, day)
        }

        for (name, month) in Self.monthNameToNumber where searchable.range(of: #"\b\#(name)\b"#, options: .regularExpression) != nil {
            return String(format: "month-%02d", month)
        }
        return nil
    }

    private func multiEvidenceSupportGroupKey(for documentPath: String) -> String? {
        let normalizedPath = documentPath
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .lowercased()
            .replacingOccurrences(of: "\\", with: "/")
        guard !normalizedPath.hasPrefix("memory://"),
              let fileName = normalizedPath.split(separator: "/").last else {
            return nil
        }

        let stem = String(fileName).replacingOccurrences(
            of: #"\.[a-z0-9]+$"#,
            with: "",
            options: .regularExpression
        )
        let groupedStem = stem.replacingOccurrences(
            of: #"(?:_abs)?_\d+$"#,
            with: "",
            options: .regularExpression
        )
        guard groupedStem != stem,
              groupedStem.contains("_") else {
            return nil
        }

        let directory = normalizedPath
            .split(separator: "/")
            .dropLast()
            .joined(separator: "/")
        return directory.isEmpty ? groupedStem : "\(directory)/\(groupedStem)"
    }

    private func monthDayAnchors(from text: String) -> Set<MonthDayAnchor> {
        let lower = text.lowercased()
        var anchors: Set<MonthDayAnchor> = []
        let monthAlternation = Self.monthNameToNumber.keys.sorted().joined(separator: "|")

        let rangePattern = #"\b("# + monthAlternation + #")\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|through|-)\s*(\d{1,2})(?:st|nd|rd|th)?\b"#
        for match in regexCaptureGroups(pattern: rangePattern, text: lower) {
            guard match.count >= 3,
                  let month = Self.monthNameToNumber[match[0]],
                  let startDay = Int(match[1]),
                  let endDay = Int(match[2]) else {
                continue
            }
            for day in min(startDay, endDay)...max(startDay, endDay) where (1...31).contains(day) {
                anchors.insert(MonthDayAnchor(month: month, day: day))
            }
        }

        let explicitPattern = #"\b("# + monthAlternation + #")\s+(\d{1,2})(?:st|nd|rd|th)?\b"#
        for match in regexCaptureGroups(pattern: explicitPattern, text: lower) {
            guard match.count >= 2,
                  let month = Self.monthNameToNumber[match[0]],
                  let day = Int(match[1]),
                  (1...31).contains(day) else {
                continue
            }
            anchors.insert(MonthDayAnchor(month: month, day: day))
        }

        return anchors
    }

    private func monthAnchors(from text: String) -> Set<Int> {
        let lower = text.lowercased()
        var months: Set<Int> = []
        for (name, number) in Self.monthNameToNumber where lower.range(of: #"\b\#(name)\b"#, options: .regularExpression) != nil {
            months.insert(number)
        }
        if lower.range(of: #"\bsummer\b"#, options: .regularExpression) != nil {
            months.formUnion([6, 7, 8])
        }
        if lower.range(of: #"\bspring\b"#, options: .regularExpression) != nil {
            months.formUnion([3, 4, 5])
        }
        if lower.range(of: #"\b(autumn|fall)\b"#, options: .regularExpression) != nil {
            months.formUnion([9, 10, 11])
        }
        if lower.range(of: #"\bwinter\b"#, options: .regularExpression) != nil {
            months.formUnion([12, 1, 2])
        }
        return months
    }

    private func regexCaptureGroups(pattern: String, text: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return [] }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, options: [], range: range).map { match in
            (1..<match.numberOfRanges).compactMap { index -> String? in
                let range = match.range(at: index)
                guard range.location != NSNotFound,
                      let swiftRange = Range(range, in: text) else {
                    return nil
                }
                return String(text[swiftRange])
            }
        }
    }

}
