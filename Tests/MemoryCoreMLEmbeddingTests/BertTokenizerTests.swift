import Foundation
import Testing
@testable import MemoryCoreMLEmbedding

@Suite("BertTokenizer")
struct BertTokenizerTests {
    let tokenizer: BertTokenizer

    init() throws {
        guard let vocabURL = Bundle.module.url(forResource: "vocab", withExtension: "txt") else {
            throw TestError("vocab.txt not found in bundle")
        }
        self.tokenizer = try BertTokenizer(vocabURL: vocabURL, maxSequenceLength: 32)
    }

    @Test func encodeProducesCorrectFormat() {
        let result = tokenizer.encode("hello world")
        #expect(result.inputIDs.count == 32)
        #expect(result.attentionMask.count == 32)
        #expect(result.tokenTypeIDs.count == 32)
        #expect(result.inputIDs[0] == 101) // [CLS]
        #expect(result.attentionMask[0] == 1)
        #expect(result.tokenTypeIDs[0] == 0)
    }

    @Test func encodePadsToMaxLength() {
        let result = tokenizer.encode("hi")
        let nonPad = result.attentionMask.filter { $0 == 1 }.count
        // [CLS] + "hi" tokens + [SEP]
        #expect(nonPad >= 3)
        #expect(nonPad <= 5)
        let padCount = result.attentionMask.filter { $0 == 0 }.count
        #expect(padCount == 32 - nonPad)
    }

    @Test func encodeHandlesEmptyString() {
        let result = tokenizer.encode("")
        #expect(result.inputIDs[0] == 101)
        #expect(result.inputIDs[1] == 102) // [SEP] right after [CLS]
    }

    @Test func encodeHandlesPunctuation() {
        let result = tokenizer.encode("hello, world!")
        let activeTokens = result.attentionMask.filter { $0 == 1 }.count
        #expect(activeTokens >= 5) // [CLS] hello , world ! [SEP]
    }

    @Test func encodeTruncatesLongInput() {
        let longText = String(repeating: "hello ", count: 200)
        let result = tokenizer.encode(longText)
        #expect(result.inputIDs.count == 32)
        #expect(result.inputIDs[0] == 101)
        // When fully packed: [CLS] + 30 tokens + [SEP], no room for padding
        let activeTokens = result.attentionMask.filter { $0 == 1 }.count
        #expect(activeTokens == 32)
    }

    @Test func encodeIsCase_insensitive() {
        let lower = tokenizer.encode("hello world")
        let upper = tokenizer.encode("HELLO WORLD")
        #expect(lower.inputIDs == upper.inputIDs)
    }
}

private struct TestError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { self.description = message }
}
