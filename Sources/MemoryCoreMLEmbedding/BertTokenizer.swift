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
        let wordpieceIDs = tokenIDs(for: text, budget: maxSequenceLength - 2)

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

    /// Encodes a query-document pair as `[CLS] query [SEP] document [SEP]` with
    /// token_type_ids = 0 for query segment and 1 for document segment.
    public func encodePair(query: String, document: String) -> EncodedInput {
        // Budget: maxSeqLen - 3 for [CLS], [SEP], [SEP]
        let totalBudget = maxSequenceLength - 3
        let queryIDs = tokenIDs(for: query, budget: totalBudget)
        let docIDs = tokenIDs(for: document, budget: totalBudget - queryIDs.count)

        // Build: [CLS] query_tokens [SEP] doc_tokens [SEP] [PAD...]
        var inputIDs = [Int32]()
        inputIDs.reserveCapacity(maxSequenceLength)
        inputIDs.append(clsTokenID)
        inputIDs.append(contentsOf: queryIDs)
        inputIDs.append(sepTokenID)
        let segmentALen = inputIDs.count
        inputIDs.append(contentsOf: docIDs)
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

        // token_type_ids: 0 for segment A (query), 1 for segment B (document)
        var tokenTypeIDs = [Int32](repeating: 0, count: segmentALen)
        tokenTypeIDs.append(contentsOf: repeatElement(Int32(1), count: maxSequenceLength - segmentALen))

        return EncodedInput(inputIDs: inputIDs, attentionMask: attentionMask, tokenTypeIDs: tokenTypeIDs)
    }

    private func tokenIDs(for text: String, budget: Int) -> [Int32] {
        var ids: [Int32] = []
        ids.reserveCapacity(min(maxSequenceLength, budget))
        var current = ""

        func flushCurrentToken() -> Bool {
            guard !current.isEmpty else { return true }
            let appended = appendWordpieceIDs(current, to: &ids, budget: budget)
            current.removeAll(keepingCapacity: true)
            return appended
        }

        func appendTokenizedScalar(_ scalar: Unicode.Scalar) -> Bool {
            let value = scalar.value
            if value < 128 {
                if value == 9 || value == 10 || value == 13 || value == 32 {
                    return flushCurrentToken()
                }
                if isASCIIPunctuation(value) {
                    guard flushCurrentToken() else { return false }
                    return appendWordpieceIDs(String(scalar), to: &ids, budget: budget)
                }
                current.unicodeScalars.append(scalar)
                return true
            }

            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                return flushCurrentToken()
            }
            if isPunctuation(scalar) || isCJKCharacter(scalar) {
                guard flushCurrentToken() else { return false }
                return appendWordpieceIDs(String(scalar), to: &ids, budget: budget)
            }
            current.unicodeScalars.append(scalar)
            return true
        }

        for scalar in text.unicodeScalars {
            let value = scalar.value
            if value < 128 {
                if value == 0 || (value < 32 && value != 9 && value != 10 && value != 13) || value == 127 {
                    continue
                }
                if value == 9 || value == 10 || value == 13 || value == 32 {
                    guard flushCurrentToken() else { break }
                } else if value >= 65 && value <= 90, let lower = Unicode.Scalar(value + 32) {
                    current.unicodeScalars.append(lower)
                } else if isASCIIPunctuation(value) {
                    guard flushCurrentToken() else { break }
                    guard appendWordpieceIDs(String(scalar), to: &ids, budget: budget) else { break }
                } else {
                    current.unicodeScalars.append(scalar)
                }
                continue
            }

            if value == 0xFFFD || CharacterSet.controlCharacters.contains(scalar) {
                continue
            }
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                guard flushCurrentToken() else { break }
            } else {
                var shouldContinue = true
                for lower in String(scalar).lowercased().unicodeScalars {
                    if !appendTokenizedScalar(lower) {
                        shouldContinue = false
                        break
                    }
                }
                if !shouldContinue { break }
            }
        }

        _ = flushCurrentToken()
        return ids
    }

    private func cleanLowercaseAndBasicTokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        var current = ""

        func flushCurrentToken() {
            if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
        }

        func appendTokenizedScalar(_ scalar: Unicode.Scalar) {
            let value = scalar.value
            if value < 128 {
                if value == 9 || value == 10 || value == 13 || value == 32 {
                    flushCurrentToken()
                } else if isASCIIPunctuation(value) {
                    flushCurrentToken()
                    tokens.append(String(scalar))
                } else {
                    current.unicodeScalars.append(scalar)
                }
                return
            }

            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                flushCurrentToken()
            } else if isPunctuation(scalar) || isCJKCharacter(scalar) {
                flushCurrentToken()
                tokens.append(String(scalar))
            } else {
                current.unicodeScalars.append(scalar)
            }
        }

        for scalar in text.unicodeScalars {
            let value = scalar.value
            if value < 128 {
                if value == 0 || (value < 32 && value != 9 && value != 10 && value != 13) || value == 127 {
                    continue
                }
                if value == 9 || value == 10 || value == 13 || value == 32 {
                    flushCurrentToken()
                } else if value >= 65 && value <= 90, let lower = Unicode.Scalar(value + 32) {
                    current.unicodeScalars.append(lower)
                } else if isASCIIPunctuation(value) {
                    flushCurrentToken()
                    tokens.append(String(scalar))
                } else {
                    current.unicodeScalars.append(scalar)
                }
                continue
            }

            if value == 0xFFFD || CharacterSet.controlCharacters.contains(scalar) {
                continue
            }
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                flushCurrentToken()
            } else {
                for lower in String(scalar).lowercased().unicodeScalars {
                    appendTokenizedScalar(lower)
                }
            }
        }

        flushCurrentToken()
        return tokens
    }

    private func cleanAndLowercase(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            let value = scalar.value
            if value < 128 {
                if value == 0 || (value < 32 && value != 9 && value != 10 && value != 13) || value == 127 {
                    continue
                }
                if value == 9 || value == 10 || value == 13 || value == 32 {
                    result.append(" ")
                } else if value >= 65 && value <= 90, let lower = Unicode.Scalar(value + 32) {
                    result.unicodeScalars.append(lower)
                } else {
                    result.unicodeScalars.append(scalar)
                }
                continue
            }

            if value == 0xFFFD || CharacterSet.controlCharacters.contains(scalar) {
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
            let value = scalar.value
            if value < 128 {
                if value == 9 || value == 10 || value == 13 || value == 32 {
                    if !current.isEmpty {
                        tokens.append(current)
                        current = ""
                    }
                } else if isASCIIPunctuation(value) {
                    if !current.isEmpty {
                        tokens.append(current)
                        current = ""
                    }
                    tokens.append(String(scalar))
                } else {
                    current.unicodeScalars.append(scalar)
                }
                continue
            }

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

    private func isASCIIPunctuation(_ value: UInt32) -> Bool {
        (value >= 33 && value <= 47) || (value >= 58 && value <= 64) ||
        (value >= 91 && value <= 96) || (value >= 123 && value <= 126)
    }

    private func appendWordpieceIDs(_ token: String, to ids: inout [Int32], budget: Int) -> Bool {
        let originalCount = ids.count
        if token.count > 200 {
            guard originalCount + 1 <= budget else { return false }
            ids.append(unkTokenID)
            return true
        }

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
                if ids.count > originalCount {
                    ids.removeSubrange(originalCount..<ids.count)
                }
                guard originalCount + 1 <= budget else { return false }
                ids.append(unkTokenID)
                return true
            }
        }

        if ids.count > budget {
            ids.removeSubrange(originalCount..<ids.count)
            return false
        }
        return true
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
