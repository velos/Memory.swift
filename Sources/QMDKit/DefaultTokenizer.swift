import Foundation

public struct DefaultTokenizer: Tokenizer, Sendable {
    public init() {}

    public func tokenize(_ text: String) -> [String] {
        var normalized = String.UnicodeScalarView()
        normalized.reserveCapacity(text.unicodeScalars.count)

        for scalar in text.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) {
                for lower in String(scalar).lowercased().unicodeScalars {
                    normalized.append(lower)
                }
            } else {
                normalized.append(" ")
            }
        }

        return String(normalized)
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
    }
}
