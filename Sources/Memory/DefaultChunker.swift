import Foundation

public struct DefaultChunker: Chunker, Sendable {
    public var targetTokenCount: Int
    public var overlapTokenCount: Int
    public var tokenizer: any Tokenizer

    public init(
        targetTokenCount: Int = 600,
        overlapTokenCount: Int = 80,
        tokenizer: any Tokenizer = DefaultTokenizer()
    ) {
        self.targetTokenCount = max(50, targetTokenCount)
        self.overlapTokenCount = max(0, overlapTokenCount)
        self.tokenizer = tokenizer
    }

    public func chunk(text: String, kind: DocumentKind, sourceURL: URL?) -> [Chunk] {
        let units = splitIntoUnits(text: text, kind: kind)
        guard !units.isEmpty else { return [] }

        let targetChars = targetTokenCount * 4
        let overlapChars = overlapTokenCount * 4

        var chunks: [Chunk] = []
        var buffer = ""

        for unit in units {
            if buffer.isEmpty {
                buffer = unit
                continue
            }

            if buffer.count + unit.count + 1 <= targetChars {
                buffer += "\n" + unit
                continue
            }

            appendChunk(from: buffer, to: &chunks)

            if overlapChars > 0 {
                let overlap = String(buffer.suffix(overlapChars))
                if overlap.isEmpty {
                    buffer = unit
                } else {
                    buffer = overlap + "\n" + unit
                }
            } else {
                buffer = unit
            }
        }

        if !buffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            appendChunk(from: buffer, to: &chunks)
        }

        return chunks
    }

    private func appendChunk(from text: String, to chunks: inout [Chunk]) {
        let content = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else { return }

        chunks.append(
            Chunk(
                ordinal: chunks.count,
                content: content,
                tokenCount: max(1, tokenizer.tokenize(content).count)
            )
        )
    }

    private func splitIntoUnits(text: String, kind: DocumentKind) -> [String] {
        switch kind {
        case .markdown:
            markdownUnits(text)
        case .code:
            codeUnits(text)
        case .plainText:
            paragraphUnits(text)
        }
    }

    private func paragraphUnits(_ text: String) -> [String] {
        text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func markdownUnits(_ text: String) -> [String] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        let lines = normalized.components(separatedBy: "\n")

        var units: [String] = []
        var current: [String] = []

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let isHeading = trimmed.hasPrefix("#")
            let isSeparator = trimmed.isEmpty

            if isHeading {
                if !current.isEmpty {
                    units.append(current.joined(separator: "\n"))
                    current.removeAll(keepingCapacity: true)
                }
                current.append(line)
            } else if isSeparator {
                if !current.isEmpty {
                    units.append(current.joined(separator: "\n"))
                    current.removeAll(keepingCapacity: true)
                }
            } else {
                current.append(line)
            }
        }

        if !current.isEmpty {
            units.append(current.joined(separator: "\n"))
        }

        return units.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }

    private func codeUnits(_ text: String) -> [String] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        let lines = normalized.components(separatedBy: "\n")

        var units: [String] = []
        var current: [String] = []

        func flushCurrent() {
            let block = current.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !block.isEmpty {
                units.append(block)
            }
            current.removeAll(keepingCapacity: true)
        }

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let startsDeclaration = isCodeBoundary(trimmed)

            if startsDeclaration && !current.isEmpty {
                flushCurrent()
            }
            current.append(line)
        }

        flushCurrent()

        if units.isEmpty {
            return paragraphUnits(text)
        }

        return units
    }

    private func isCodeBoundary(_ line: String) -> Bool {
        let prefixes = [
            "func ", "class ", "struct ", "enum ", "protocol ", "extension ",
            "actor ", "public func", "private func", "internal func",
            "let ", "var ", "import ", "def ", "interface "
        ]
        return prefixes.contains { line.hasPrefix($0) }
    }
}
