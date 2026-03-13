import Foundation

/// Byte-level BPE tokenizer compatible with GPT-2/GPT-NeoX/ModernBERT tokenizers.
/// Loads its configuration from a HuggingFace `tokenizer.json` file.
public struct BPETokenizer: Sendable {
    private let vocab: [String: Int32]
    private let mergeRanks: [String: Int]
    private let byteEncoder: [UInt8: Character]
    private let clsTokenID: Int32
    private let sepTokenID: Int32
    private let padTokenID: Int32
    private let unkTokenID: Int32
    private let maxSequenceLength: Int
    private let pretokenPattern: NSRegularExpression
    /// Added tokens sorted longest-first for greedy prefix matching.
    private let addedTokenEntries: [(content: String, id: Int32)]

    public init(tokenizerURL: URL, maxSequenceLength: Int = 512) throws {
        let data = try Data(contentsOf: tokenizerURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let modelDict = json["model"] as! [String: Any]

        let rawVocab = modelDict["vocab"] as! [String: Int]
        var vocab: [String: Int32] = [:]
        vocab.reserveCapacity(rawVocab.count + 128)
        for (k, v) in rawVocab {
            vocab[k] = Int32(v)
        }

        var addedEntries: [(String, Int32)] = []
        if let addedTokens = json["added_tokens"] as? [[String: Any]] {
            for token in addedTokens {
                if let content = token["content"] as? String, let id = token["id"] as? Int {
                    vocab[content] = Int32(id)
                    addedEntries.append((content, Int32(id)))
                }
            }
        }
        addedEntries.sort { $0.0.count > $1.0.count }

        let rawMerges = modelDict["merges"] as! [Any]
        var mergeRanks: [String: Int] = [:]
        mergeRanks.reserveCapacity(rawMerges.count)
        for (i, merge) in rawMerges.enumerated() {
            let key: String
            if let pair = merge as? [String] {
                key = "\(pair[0]) \(pair[1])"
            } else if let str = merge as? String {
                key = str
            } else {
                continue
            }
            mergeRanks[key] = i
        }

        self.vocab = vocab
        self.mergeRanks = mergeRanks
        self.addedTokenEntries = addedEntries
        self.clsTokenID = vocab["[CLS]"] ?? 50281
        self.sepTokenID = vocab["[SEP]"] ?? 50282
        self.padTokenID = vocab["[PAD]"] ?? 50283
        self.unkTokenID = vocab["[UNK]"] ?? 50280
        self.maxSequenceLength = maxSequenceLength
        self.byteEncoder = Self.buildByteEncoder()

        self.pretokenPattern = try NSRegularExpression(
            pattern: #"'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        )
    }

    public struct EncodedInput: Sendable {
        public let inputIDs: [Int32]
        public let attentionMask: [Int32]
    }

    /// Encode text as `[CLS] tokens [SEP] [PAD...]`
    public func encode(_ text: String, prefix: String? = nil) -> EncodedInput {
        let fullText: String
        if let prefix {
            fullText = prefix + text
        } else {
            fullText = text
        }

        let normalized = fullText.precomposedStringWithCanonicalMapping
        let tokenIDs = tokenizeWithAddedTokens(normalized)

        let budget = maxSequenceLength - 2
        let truncated = Array(tokenIDs.prefix(budget))

        var inputIDs = [Int32]()
        inputIDs.reserveCapacity(maxSequenceLength)
        inputIDs.append(clsTokenID)
        inputIDs.append(contentsOf: truncated)
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

        return EncodedInput(inputIDs: inputIDs, attentionMask: attentionMask)
    }

    // MARK: - Added Token Handling

    /// Split text on added tokens first, then BPE-tokenize the remaining segments.
    private func tokenizeWithAddedTokens(_ text: String) -> [Int32] {
        var segments: [(text: String, isAdded: Bool, id: Int32)] = [(text, false, 0)]

        for (content, id) in addedTokenEntries {
            var newSegments: [(text: String, isAdded: Bool, id: Int32)] = []
            for seg in segments {
                if seg.isAdded {
                    newSegments.append(seg)
                    continue
                }
                var remaining = seg.text
                while let range = remaining.range(of: content) {
                    let before = String(remaining[remaining.startIndex..<range.lowerBound])
                    if !before.isEmpty {
                        newSegments.append((before, false, 0))
                    }
                    newSegments.append((content, true, id))
                    remaining = String(remaining[range.upperBound...])
                }
                if !remaining.isEmpty {
                    newSegments.append((remaining, false, 0))
                }
            }
            segments = newSegments
        }

        var allIDs: [Int32] = []
        for seg in segments {
            if seg.isAdded {
                allIDs.append(seg.id)
            } else {
                allIDs.append(contentsOf: bpeTokenize(seg.text))
            }
        }
        return allIDs
    }

    // MARK: - BPE

    private func bpeTokenize(_ text: String) -> [Int32] {
        let nsText = text as NSString
        let matches = pretokenPattern.matches(in: text, range: NSRange(location: 0, length: nsText.length))

        var allIDs: [Int32] = []
        for match in matches {
            let word = nsText.substring(with: match.range)
            let byteChars = byteLevelEncode(word)
            let merged = applyBPE(byteChars)
            for token in merged {
                if let id = vocab[token] {
                    allIDs.append(id)
                } else {
                    allIDs.append(unkTokenID)
                }
            }
        }
        return allIDs
    }

    private func byteLevelEncode(_ text: String) -> [String] {
        var chars: [String] = []
        for byte in Array(text.utf8) {
            if let ch = byteEncoder[byte] {
                chars.append(String(ch))
            }
        }
        return chars
    }

    private func applyBPE(_ tokens: [String]) -> [String] {
        if tokens.count <= 1 { return tokens }

        var word = tokens
        while word.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1

            for i in 0..<(word.count - 1) {
                let pair = "\(word[i]) \(word[i + 1])"
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }

            if bestIndex < 0 { break }

            let merged = word[bestIndex] + word[bestIndex + 1]
            var newWord = Array(word[0..<bestIndex])
            newWord.append(merged)
            if bestIndex + 2 < word.count {
                newWord.append(contentsOf: word[(bestIndex + 2)...])
            }
            word = newWord
        }

        return word
    }

    // MARK: - Byte Encoder

    /// Builds the GPT-2 byte-to-unicode mapping.
    /// Printable ASCII bytes map to themselves; other bytes map to Unicode chars starting at U+0100.
    private static func buildByteEncoder() -> [UInt8: Character] {
        var encoder: [UInt8: Character] = [:]
        var nextCode: UInt32 = 256

        for byte: UInt8 in 0...255 {
            let b = UInt32(byte)
            if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255) {
                encoder[byte] = Character(Unicode.Scalar(b)!)
            } else {
                encoder[byte] = Character(Unicode.Scalar(nextCode)!)
                nextCode += 1
            }
        }
        return encoder
    }
}
