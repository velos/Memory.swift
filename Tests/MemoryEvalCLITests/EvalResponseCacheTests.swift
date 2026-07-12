import Foundation
import AgentMemory
import Testing
@testable import memory_eval

struct EvalResponseCacheTests {
    @Test
    func responseCachePersistsAcrossReopen() async throws {
        let root = try makeTemporaryDirectory()
        let dbURL = root.appendingPathComponent("eval-cache.sqlite")

        let first = try EvalResponseCache(databaseURL: dbURL)
        try await first.store(
            namespace: "unit",
            keyComponents: ["a", "b"],
            value: ["value": 42]
        )

        let reopened = try EvalResponseCache(databaseURL: dbURL)
        let cached = try await reopened.load(
            namespace: "unit",
            keyComponents: ["a", "b"],
            as: [String: Int].self
        )

        #expect(cached?["value"] == 42)
    }

    @Test
    func recallDocumentMaterializationUsesIndexableExtensionForMarkdownPdfSources() throws {
        let root = try makeTemporaryDirectory()
        let path = materializedRecallDocumentURL(
            id: "general-v2__doc-0555",
            kind: "markdown",
            relativePath: "general-v2/repliqa/pdfs/repliqa_0/xjfymplj.pdf",
            docsRoot: root
        )

        #expect(path.pathExtension == "md")
        #expect(path.path.contains("xjfymplj.md"))
    }

    @Test
    func recallDocumentMaterializationKeepsSupportedRelativeExtensions() throws {
        let root = try makeTemporaryDirectory()
        let path = materializedRecallDocumentURL(
            id: "doc-1",
            kind: "markdown",
            relativePath: "notes/already.md",
            docsRoot: root
        )

        #expect(path.pathExtension == "md")
        #expect(path.path.contains("notes/already.md"))
    }

    @Test
    func estimatedContextTokenCountIgnoresWhitespaceOnlyText() {
        #expect(estimatedContextTokenCount("   \n\t  ") == 0)
        #expect(estimatedContextTokenCount("Alpha beta, gamma.") == 3)
    }

    @Test
    func cappedContextTokenCountRespectsDocumentAndRemainingBudgets() {
        #expect(cappedContextTokenCount(fullTokenCount: 900, remainingBudget: 4096, perDocumentTokenBudget: 384) == 384)
        #expect(cappedContextTokenCount(fullTokenCount: 900, remainingBudget: 256, perDocumentTokenBudget: 384) == 248)
        #expect(cappedContextTokenCount(fullTokenCount: 120, remainingBudget: 4096, perDocumentTokenBudget: 384) == 120)
        #expect(cappedContextTokenCount(fullTokenCount: 900, remainingBudget: 7, perDocumentTokenBudget: 384) == 0)
        #expect(cappedContextTokenCount(fullTokenCount: 900, remainingBudget: 4096, perDocumentTokenBudget: 0) == 900)
    }

    @Test
    func adaptiveContextBudgetPacksMoreEvidenceForDenseQueries() {
        let dense = adaptiveContextPerDocumentTokenBudget(
            queryText: "What activities did I conduct in August? Please list all activities.",
            contextTokenBudget: 4096,
            perDocumentTokenBudget: 384
        )
        let sparse = adaptiveContextPerDocumentTokenBudget(
            queryText: "Which document mentions the alpha project?",
            contextTokenBudget: 4096,
            perDocumentTokenBudget: 384
        )
        let unlimitedDense = adaptiveContextPerDocumentTokenBudget(
            queryText: "How many training sessions did I complete in May?",
            contextTokenBudget: 4096,
            perDocumentTokenBudget: 0
        )

        #expect(dense == 384)
        #expect(sparse == 384)
        #expect(unlimitedDense > 0)
        #expect(unlimitedDense < 384)
    }

    @Test
    func groundedExpansionExcludesOriginalTermsAndStopwords() {
        let terms = groundedExpansionTerms(
            query: "archive old chats",
            documents: [
                GroundedFeedbackDocument(
                    rank: 1,
                    title: "Archive export backup",
                    filenameStem: "old_chats_backup",
                    snippet: "The backup transcript export is ready.",
                    leadingContent: "Archive the old chats into a transcript backup."
                ),
                GroundedFeedbackDocument(
                    rank: 2,
                    title: "Garden schedule",
                    filenameStem: "garden_schedule",
                    snippet: "Tomato watering notes.",
                    leadingContent: "General garden maintenance."
                ),
                GroundedFeedbackDocument(
                    rank: 3,
                    title: "Weekend plan",
                    filenameStem: "weekend_plan",
                    snippet: "Dinner reservation details.",
                    leadingContent: "General weekend notes."
                ),
            ]
        ).map(\.text)

        #expect(!terms.contains("archive"))
        #expect(!terms.contains("old"))
        #expect(!terms.contains("chats"))
        #expect(!terms.contains("the"))
        #expect(terms.contains("backup"))
        #expect(terms.contains("export"))
    }

    @Test
    func groundedExpansionSuppressesShortAmbiguousEllipsis() {
        let terms = groundedExpansionTerms(
            query: "What are the costs?",
            documents: [
                GroundedFeedbackDocument(
                    rank: 1,
                    title: "Conference Budget",
                    filenameStem: "conference_budget",
                    snippet: "Registration and hotel totals were discussed.",
                    leadingContent: "Registration, hotel, travel, and dinner costs were updated."
                ),
            ]
        )

        #expect(terms.isEmpty)
    }

    @Test
    func groundedExpansionRanksStrongSectionsAboveLowRankBodyOnlyTerms() {
        let terms = groundedExpansionTerms(
            query: "launch notes",
            documents: [
                GroundedFeedbackDocument(
                    rank: 1,
                    title: "Alpha release",
                    filenameStem: "alpha_release",
                    snippet: "",
                    leadingContent: ""
                ),
                GroundedFeedbackDocument(
                    rank: 2,
                    title: nil,
                    filenameStem: "meeting",
                    snippet: "Bravo checklist was reviewed.",
                    leadingContent: ""
                ),
                GroundedFeedbackDocument(
                    rank: 12,
                    title: nil,
                    filenameStem: "misc",
                    snippet: "",
                    leadingContent: "Zebra appendix and body-only detail."
                ),
            ]
        )
        let scoreByTerm = Dictionary(uniqueKeysWithValues: terms.map { ($0.text, $0.score) })

        #expect((scoreByTerm["alpha"] ?? 0) > (scoreByTerm["zebra"] ?? 0))
        #expect((scoreByTerm["bravo"] ?? 0) > (scoreByTerm["zebra"] ?? 0))
    }

    @Test
    func groundedExpansionGroupsTopTermsIntoAtMostTwoQueries() {
        let terms = (1...10).map {
            GroundedExpansionTerm(
                text: "term\($0)",
                score: Double(20 - $0),
                documentFrequency: 1,
                topEvidenceRank: 1
            )
        }

        let queries = groundedExpansionQueries(from: terms)

        #expect(queries == [
            "term1 term2 term3 term4",
            "term5 term6 term7 term8",
        ])
    }

    @Test
    func groundedExpansionSingleTokenModeOmitsPhrases() {
        let terms = groundedExpansionTerms(
            query: "archive chats",
            documents: [
                GroundedFeedbackDocument(
                    rank: 1,
                    title: "iCloud backup",
                    filenameStem: "",
                    snippet: "The conversation history transcript backup is ready.",
                    leadingContent: ""
                ),
            ],
            termMode: .singleToken
        ).map(\.text)

        #expect(terms.contains("backup"))
        #expect(terms.contains("icloud"))
        #expect(!terms.contains("conversation history"))
    }

    @Test
    func groundedExpansionPhraseEntityModeOmitsPlainSingleTokens() {
        let terms = groundedExpansionTerms(
            query: "archive chats",
            documents: [
                GroundedFeedbackDocument(
                    rank: 1,
                    title: "iCloud backup",
                    filenameStem: "",
                    snippet: "The conversation history transcript backup is ready.",
                    leadingContent: ""
                ),
                GroundedFeedbackDocument(
                    rank: 2,
                    title: "Garden plan",
                    filenameStem: "",
                    snippet: "Seedling notes and watering schedule.",
                    leadingContent: ""
                ),
                GroundedFeedbackDocument(
                    rank: 3,
                    title: "Dinner list",
                    filenameStem: "",
                    snippet: "Reservation notes and grocery list.",
                    leadingContent: ""
                ),
            ],
            termMode: .phraseEntity
        ).map(\.text)

        #expect(terms.contains("icloud"))
        #expect(terms.contains("conversation history"))
        #expect(!terms.contains("backup"))
    }

    @Test
    func groundedExpansionGuardSkipsStrongRankOne() {
        let terms = [
            GroundedExpansionTerm(text: "conversation history", score: 1.6, documentFrequency: 1, topEvidenceRank: 1, kind: .phrase),
            GroundedExpansionTerm(text: "transcript", score: 1.3, documentFrequency: 2, topEvidenceRank: 2),
        ]
        let decision = groundedExpansionDecision(
            baselineScores: [
                SearchScoreBreakdown(
                    semantic: 0.05,
                    lexical: 0.05,
                    recency: 0,
                    fused: 0.13,
                    blended: 0.13
                ),
                SearchScoreBreakdown(
                    semantic: 0.035,
                    lexical: 0.03,
                    recency: 0,
                    fused: 0.09,
                    blended: 0.09
                ),
            ],
            terms: terms,
            policy: .guarded
        )

        #expect(!decision.shouldApply)
        #expect(decision.reason == "strong_rank1")
    }

    @Test
    func groundedExpansionGuardSkipsHighScoreTightTopCluster() {
        let terms = [
            GroundedExpansionTerm(text: "deployment notes", score: 1.6, documentFrequency: 2, topEvidenceRank: 1, kind: .phrase),
            GroundedExpansionTerm(text: "release checklist", score: 1.3, documentFrequency: 2, topEvidenceRank: 2, kind: .phrase),
        ]
        let decision = groundedExpansionDecision(
            baselineScores: [
                SearchScoreBreakdown(
                    semantic: 0.039,
                    lexical: 0.075,
                    recency: 0,
                    temporal: 0.025,
                    fused: 0.124,
                    blended: 0.124
                ),
                SearchScoreBreakdown(
                    semantic: 0.029,
                    lexical: 0.077,
                    recency: 0,
                    temporal: 0.025,
                    fused: 0.121,
                    blended: 0.121
                ),
            ],
            terms: terms,
            policy: .guarded
        )

        #expect(!decision.shouldApply)
        #expect(decision.reason == "strong_rank1")
    }

    @Test
    func groundedExpansionGuardAppliesForLowConfidenceWithEvidence() {
        let terms = [
            GroundedExpansionTerm(text: "conversation history", score: 1.6, documentFrequency: 1, topEvidenceRank: 1, kind: .phrase),
            GroundedExpansionTerm(text: "transcript", score: 1.3, documentFrequency: 2, topEvidenceRank: 2),
        ]
        let decision = groundedExpansionDecision(
            baselineScores: [
                SearchScoreBreakdown(
                    semantic: 0.04,
                    lexical: 0.035,
                    recency: 0,
                    fused: 0.10,
                    blended: 0.10
                ),
                SearchScoreBreakdown(
                    semantic: 0.039,
                    lexical: 0.034,
                    recency: 0,
                    fused: 0.094,
                    blended: 0.094
                ),
            ],
            terms: terms,
            policy: .guarded
        )

        #expect(decision.shouldApply)
        #expect(decision.reason == "applied_guarded")
    }

    @Test
    func groundedExpansionGuardRequiresWeakLexicalCoverage() {
        let terms = [
            GroundedExpansionTerm(text: "deployment notes", score: 1.6, documentFrequency: 2, topEvidenceRank: 1, kind: .phrase),
            GroundedExpansionTerm(text: "release checklist", score: 1.3, documentFrequency: 2, topEvidenceRank: 2, kind: .phrase),
        ]
        let decision = groundedExpansionDecision(
            baselineScores: [
                SearchScoreBreakdown(semantic: 0.035, lexical: 0.070, recency: 0, fused: 0.10, blended: 0.10),
                SearchScoreBreakdown(semantic: 0.034, lexical: 0.069, recency: 0, fused: 0.098, blended: 0.098),
                SearchScoreBreakdown(semantic: 0.030, lexical: 0.066, recency: 0, fused: 0.095, blended: 0.095),
                SearchScoreBreakdown(semantic: 0.020, lexical: 0.040, recency: 0, fused: 0.070, blended: 0.070),
            ],
            terms: terms,
            policy: .guarded
        )

        #expect(!decision.shouldApply)
        #expect(decision.reason == "strong_lexical_coverage")
    }

    @Test
    func groundedExpansionGuardRequiresSemanticFeedbackCluster() {
        let terms = [
            GroundedExpansionTerm(text: "deployment notes", score: 1.6, documentFrequency: 2, topEvidenceRank: 1, kind: .phrase),
            GroundedExpansionTerm(text: "release checklist", score: 1.3, documentFrequency: 2, topEvidenceRank: 2, kind: .phrase),
        ]
        let decision = groundedExpansionDecision(
            baselineScores: [
                SearchScoreBreakdown(semantic: 0.036, lexical: 0.020, recency: 0, fused: 0.090, blended: 0.090),
                SearchScoreBreakdown(semantic: 0.014, lexical: 0.018, recency: 0, fused: 0.080, blended: 0.080),
                SearchScoreBreakdown(semantic: 0.012, lexical: 0.017, recency: 0, fused: 0.070, blended: 0.070),
            ],
            terms: terms,
            policy: .guarded
        )

        #expect(!decision.shouldApply)
        #expect(decision.reason == "weak_semantic_cluster")
    }

    private func makeTemporaryDirectory(function: String = #function) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-eval-tests")
            .appendingPathComponent(function.replacingOccurrences(of: " ", with: "_"))
            .appendingPathComponent(UUID().uuidString)

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
