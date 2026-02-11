import Foundation
import Testing
@testable import Memory

struct TokenizerAndChunkerTests {
    @Test
    func tokenizerNormalizesDeterministically() {
        let tokenizer = DefaultTokenizer()
        let tokens = tokenizer.tokenize("Hello, HELLO! café_42")

        #expect(tokens == ["hello", "hello", "café", "42"])
    }

    @Test
    func markdownChunkerSplitsHeadingBoundaries() {
        let chunker = DefaultChunker(targetTokenCount: 200, overlapTokenCount: 0)
        let markdown = """
        # Intro
        line a
        line b

        # Next
        line c
        line d
        """

        let chunks = chunker.chunk(text: markdown, kind: .markdown, sourceURL: nil)

        #expect(chunks.isEmpty == false)
        #expect(chunks.map(\.content).joined(separator: "\n").contains("Intro"))
        #expect(chunks.map(\.content).joined(separator: "\n").contains("Next"))
    }

    @Test
    func codeChunkerSplitsOnDeclarationBoundaries() {
        let chunker = DefaultChunker(targetTokenCount: 200, overlapTokenCount: 0)
        let source = """
        struct A {
            let value: Int
        }

        func run() {
            print("x")
        }
        """

        let chunks = chunker.chunk(text: source, kind: .code, sourceURL: nil)

        #expect(chunks.isEmpty == false)
        #expect(chunks.map(\.content).joined(separator: "\n").contains("struct A"))
        #expect(chunks.map(\.content).joined(separator: "\n").contains("func run"))
    }
}
