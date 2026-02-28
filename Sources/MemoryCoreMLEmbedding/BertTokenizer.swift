import Foundation

public struct BertTokenizer: Sendable {
    private let vocab: [String: Int32]
    private let unkTokenID: Int32
    private let clsTokenID: Int32
    private let sepTokenID: Int32
    private let padTokenID: Int32
    private let maxSequenceLength: Int

    public init(vocabURL: URL, maxSequenceLength: Int = 512) throws {
        let content = try String(contentsOf: vocabURL, encoding: .utf8)
        var vocab: [String: Int32] = [:]
        vocab.reserveCapacity(31_000)
        for (index, line) in content.components(separatedBy: "\n").enumerated() {
            let token = line.trimmingCharacters(in: .carriageReturns)
            guard !token.isEmpty else { continue }
            vocab[token] = Int32(index)
        }

        self.vocab = vocab
        self.unkTokenID = vocab["[UNK]"] ?? 100
        self.clsTokenID = vocab["[CLS]"] ?? 101
        self.sepTokenID = vocab["[SEP]"] ?? 102
        self.padTokenID = vocab["[PAD]"] ?? 0
        self.maxSequenceLength = maxSequenceLength
    }

    public struct EncodedInput: Sendable {
        public let inputIDs: [Int32]
        public let attentionMask: [Int32]
        public let tokenTypeIDs: [Int32]
    }

    public func encode(_ text: String) -> EncodedInput {
        let cleaned = cleanAndLowercase(text)
        let basicTokens = basicTokenize(cleaned)
        var wordpieceIDs: [Int32] = []
        wordpieceIDs.reserveCapacity(maxSequenceLength)

        let budget = maxSequenceLength - 2
        for token in basicTokens {
            let subIDs = wordpieceTokenize(token)
            if wordpieceIDs.count + subIDs.count > budget { break }
            wordpieceIDs.append(contentsOf: subIDs)
        }

        var inputIDs = [Int32]()
        inputIDs.reserveCapacity(maxSequenceLength)
        inputIDs.append(clsTokenID)
        inputIDs.append(contentsOf: wordpieceIDs)
        inputIDs.append(sepTokenID)

        let tokenCount = inputIDs.count
        let padCount = maxSequenceLength - tokenCount
        if padCount > 0 {
            inputIDs.append(contentsOf: repeatElement(padTokenID, count: padCount))
        }

        var attentionMask = [Int32](repeating: 1, count: tokenCount)
        if padCount > 0 {
            attentionMask.append(contentsOf: repeatElement(0, count: padCount))
        }

        let tokenTypeIDs = [Int32](repeating: 0, count: maxSequenceLength)
        return EncodedInput(inputIDs: inputIDs, attentionMask: attentionMask, tokenTypeIDs: tokenTypeIDs)
    }

    private func cleanAndLowercase(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            if scalar.value == 0 || scalar.value == 0xFFFD || CharacterSet.controlCharacters.contains(scalar) {
                if scalar == "\t" || scalar == "\n" || scalar == "\r" {
                    result.append(" ")
                }
                continue
            }
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                result.append(" ")
            } else {
                for lower in String(scalar).lowercased().unicodeScalars {
                    result.unicodeScalars.append(lower)
                }
            }
        }
        return result
    }

    private func basicTokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        var current = ""

        for scalar in text.unicodeScalars {
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
            } else if isPunctuation(scalar) || isCJKCharacter(scalar) {
                if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
                tokens.append(String(scalar))
            } else {
                current.unicodeScalars.append(scalar)
            }
        }
        if !current.isEmpty {
            tokens.append(current)
        }
        return tokens
    }

    private func wordpieceTokenize(_ token: String) -> [Int32] {
        if token.count > 200 { return [unkTokenID] }

        var ids: [Int32] = []
        var start = token.startIndex
        while start < token.endIndex {
            var end = token.endIndex
            var matched = false
            while start < end {
                let substr: String
                if start == token.startIndex {
                    substr = String(token[start..<end])
                } else {
                    substr = "##" + String(token[start..<end])
                }
                if let id = vocab[substr] {
                    ids.append(id)
                    start = end
                    matched = true
                    break
                }
                end = token.index(before: end)
            }
            if !matched {
                return [unkTokenID]
            }
        }
        return ids
    }

    private func isPunctuation(_ scalar: Unicode.Scalar) -> Bool {
        let v = scalar.value
        if (v >= 33 && v <= 47) || (v >= 58 && v <= 64) ||
           (v >= 91 && v <= 96) || (v >= 123 && v <= 126) {
            return true
        }
        return CharacterSet.punctuationCharacters.contains(scalar)
    }

    private func isCJKCharacter(_ scalar: Unicode.Scalar) -> Bool {
        let v = scalar.value
        return (v >= 0x4E00 && v <= 0x9FFF) || (v >= 0x3400 && v <= 0x4DBF) ||
               (v >= 0x20000 && v <= 0x2A6DF) || (v >= 0x2A700 && v <= 0x2B73F) ||
               (v >= 0x2B740 && v <= 0x2B81F) || (v >= 0x2B820 && v <= 0x2CEAF) ||
               (v >= 0xF900 && v <= 0xFAFF) || (v >= 0x2F800 && v <= 0x2FA1F)
    }
}

private extension CharacterSet {
    static let carriageReturns = CharacterSet(charactersIn: "\r")
}
