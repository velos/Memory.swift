import Foundation

public struct DefaultChunker: Chunker, Sendable {
    public var targetTokenCount: Int
    public var overlapTokenCount: Int
    public var tokenizer: any Tokenizer

    public init(
        targetTokenCount: Int = 900,
        overlapTokenCount: Int = 135,
        tokenizer: any Tokenizer = DefaultTokenizer()
    ) {
        self.targetTokenCount = max(50, targetTokenCount)
        self.overlapTokenCount = max(0, overlapTokenCount)
        self.tokenizer = tokenizer
    }

    public func chunk(text: String, kind: DocumentKind, sourceURL: URL?) -> [Chunk] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        guard !normalized.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }

        let targetChars = targetTokenCount * 4
        let overlapChars = overlapTokenCount * 4
        let windowChars = targetChars / 4

        let breakPoints = detectBreakPoints(normalized, kind: kind)
        let codeFences = findCodeFences(normalized)

        var chunks: [Chunk] = []
        var position = 0
        let length = normalized.count

        while position < length {
            let remaining = length - position
            if remaining <= targetChars + windowChars / 2 {
                let content = String(normalized.dropFirst(position))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !content.isEmpty {
                    appendChunk(from: content, to: &chunks)
                }
                break
            }

            let cutPosition = findBestBreak(
                in: breakPoints,
                codeFences: codeFences,
                after: position,
                target: position + targetChars,
                window: windowChars
            )

            let end = cutPosition ?? (position + targetChars)
            let safeEnd = min(end, length)
            let content = String(normalized.dropFirst(position).prefix(safeEnd - position))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !content.isEmpty {
                appendChunk(from: content, to: &chunks)
            }

            if overlapChars > 0, safeEnd > overlapChars {
                position = safeEnd - overlapChars
            } else {
                position = safeEnd
            }
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

    private struct BreakPoint {
        var position: Int
        var score: Int
    }

    private struct CodeFenceRegion {
        var start: Int
        var end: Int
    }

    private func detectBreakPoints(_ text: String, kind: DocumentKind) -> [BreakPoint] {
        let lines = text.components(separatedBy: "\n")
        var breakPoints: [BreakPoint] = []
        var charOffset = 0

        for (index, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            let score = breakScore(for: trimmed, kind: kind)
            if score > 0 {
                breakPoints.append(BreakPoint(position: charOffset, score: score))
            }

            charOffset += line.count
            if index < lines.count - 1 {
                charOffset += 1
            }
        }

        return breakPoints
    }

    private func breakScore(for line: String, kind: DocumentKind) -> Int {
        if line.hasPrefix("# ") { return 100 }
        if line.hasPrefix("## ") { return 90 }
        if line.hasPrefix("### ") { return 80 }
        if line.hasPrefix("#### ") { return 70 }
        if line.hasPrefix("##### ") { return 60 }
        if line.hasPrefix("###### ") { return 50 }

        if line.hasPrefix("```") { return 80 }

        if line == "---" || line == "***" || line == "___" { return 60 }

        if line.isEmpty { return 20 }

        if kind == .code && isCodeBoundary(line) { return 70 }

        if line.hasPrefix("- ") || line.hasPrefix("* ") || line.hasPrefix("+ ") { return 5 }
        if let first = line.first, first.isNumber,
           line.contains(where: { $0 == "." || $0 == ")" }),
           line.dropFirst().prefix(3).contains(where: { $0 == "." || $0 == ")" }) {
            return 5
        }

        return 0
    }

    private func findCodeFences(_ text: String) -> [CodeFenceRegion] {
        let lines = text.components(separatedBy: "\n")
        var regions: [CodeFenceRegion] = []
        var fenceStart: Int?
        var charOffset = 0

        for (index, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") {
                if let start = fenceStart {
                    regions.append(CodeFenceRegion(start: start, end: charOffset + line.count))
                    fenceStart = nil
                } else {
                    fenceStart = charOffset
                }
            }
            charOffset += line.count
            if index < lines.count - 1 {
                charOffset += 1
            }
        }

        return regions
    }

    private func findBestBreak(
        in breakPoints: [BreakPoint],
        codeFences: [CodeFenceRegion],
        after start: Int,
        target: Int,
        window: Int
    ) -> Int? {
        let windowStart = max(start + 1, target - window)
        let windowEnd = target

        var bestPosition: Int?
        var bestScore: Double = -1

        for bp in breakPoints {
            guard bp.position >= windowStart, bp.position <= windowEnd else { continue }

            if isInsideCodeFence(bp.position, fences: codeFences) { continue }

            let distance = Double(target - bp.position)
            let windowSize = Double(window)
            let distanceRatio = distance / windowSize
            let decay = 1.0 - (distanceRatio * distanceRatio * 0.7)
            let finalScore = Double(bp.score) * decay

            if finalScore > bestScore {
                bestScore = finalScore
                bestPosition = bp.position
            }
        }

        return bestPosition
    }

    private func isInsideCodeFence(_ position: Int, fences: [CodeFenceRegion]) -> Bool {
        for fence in fences {
            if position > fence.start, position < fence.end {
                return true
            }
        }
        return false
    }

    private func isCodeBoundary(_ line: String) -> Bool {
        let prefixes = [
            "func ", "class ", "struct ", "enum ", "protocol ", "extension ",
            "actor ", "public func", "private func", "internal func",
            "let ", "var ", "import ", "def ", "interface ",
        ]
        return prefixes.contains { line.hasPrefix($0) }
    }
}
