import Foundation
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

    private func makeTemporaryDirectory(function: String = #function) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-eval-tests")
            .appendingPathComponent(function.replacingOccurrences(of: " ", with: "_"))
            .appendingPathComponent(UUID().uuidString)

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
