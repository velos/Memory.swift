import Accelerate
import CoreML
import Foundation
import Memory

/// ColBERT-style reranker using MaxSim scoring over per-token embeddings.
///
/// Encodes query and document independently into per-token vectors, then scores
/// using MaxSim: for each query token, find the maximum cosine similarity with
/// any document token, then sum across all query tokens.
public actor CoreMLColBERTReranker: Reranker {
    public let identifier: String

    private let model: MLModel
    private let tokenizer: BPETokenizer
    private let maxSequenceLength: Int
    private let embeddingDim: Int

    public init(
        modelURL: URL,
        tokenizerURL: URL? = nil,
        maxSequenceLength: Int = 512,
        embeddingDim: Int = 48,
        identifier: String = "coreml-colbert-reranker",
        computeUnits: MLComputeUnits = .all
    ) throws {
        self.identifier = identifier
        self.maxSequenceLength = maxSequenceLength
        self.embeddingDim = embeddingDim

        let resolvedTokenizerURL: URL
        if let tokenizerURL {
            resolvedTokenizerURL = tokenizerURL
        } else if let bundledURL = Bundle.module.url(forResource: "tokenizer", withExtension: "json") {
            resolvedTokenizerURL = bundledURL
        } else {
            throw MemoryError.embedding("No tokenizer.json found. Provide tokenizerURL or include tokenizer.json in bundle.")
        }

        self.tokenizer = try BPETokenizer(tokenizerURL: resolvedTokenizerURL, maxSequenceLength: maxSequenceLength)

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try MLModel.compileModel(at: modelURL)
        }
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
    }

    public func rerank(query: SearchQuery, candidates: [SearchResult]) async throws -> [RerankAssessment] {
        guard !candidates.isEmpty else { return [] }

        let queryEmbeddings = try encodeText(query.text, prefix: "[Q] ")

        var assessments: [RerankAssessment] = []
        assessments.reserveCapacity(candidates.count)

        for candidate in candidates {
            let docEmbeddings = try encodeText(candidate.content, prefix: "[D] ")
            let score = maxSim(query: queryEmbeddings, document: docEmbeddings)
            assessments.append(RerankAssessment(chunkID: candidate.chunkID, relevance: score))
        }

        return assessments
    }

    // MARK: - Encoding

    private struct TokenEmbeddings {
        let vectors: [Float]
        let tokenCount: Int
        let dim: Int
    }

    private func encodeText(_ text: String, prefix: String) throws -> TokenEmbeddings {
        let encoded = tokenizer.encode(text, prefix: prefix)

        let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)

        let idsPtr = inputIDs.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPtr = attentionMask.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<maxSequenceLength {
            idsPtr[i] = encoded.inputIDs[i]
            maskPtr[i] = encoded.attentionMask[i]
        }

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
        ])

        let output = try model.prediction(from: inputFeatures)

        guard let embeddingFeature = output.featureValue(for: "token_embeddings"),
              let embeddingArray = embeddingFeature.multiArrayValue else {
            throw MemoryError.embedding("CoreML ColBERT model did not return 'token_embeddings' output")
        }

        let activeTokenCount = encoded.attentionMask.prefix(while: { $0 == 1 }).count
        let totalElements = maxSequenceLength * embeddingDim

        // Model outputs fp16; convert to fp32 for MaxSim computation
        var floats = [Float](repeating: 0, count: totalElements)
        if embeddingArray.dataType == .float16 {
            let fp16Ptr = embeddingArray.dataPointer.assumingMemoryBound(to: UInt16.self)
            for i in 0..<totalElements {
                floats[i] = float16ToFloat32(fp16Ptr[i])
            }
        } else {
            let fp32Ptr = embeddingArray.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<totalElements {
                floats[i] = fp32Ptr[i]
            }
        }

        return TokenEmbeddings(vectors: floats, tokenCount: activeTokenCount, dim: embeddingDim)
    }

    private func float16ToFloat32(_ h: UInt16) -> Float {
        Float(bitPattern: {
            let sign = UInt32(h >> 15) << 31
            let exp = UInt32((h >> 10) & 0x1F)
            let frac = UInt32(h & 0x3FF)
            if exp == 0 {
                if frac == 0 { return sign }
                var f = frac
                var e: UInt32 = 0
                while f & 0x400 == 0 { f <<= 1; e += 1 }
                return sign | ((127 - 15 + 1 - e) << 23) | ((f & 0x3FF) << 13)
            } else if exp == 31 {
                return sign | 0x7F800000 | (frac << 13)
            }
            return sign | ((exp + 112) << 23) | (frac << 13)
        }())
    }

    // MARK: - MaxSim

    /// For each query token, find the max cosine similarity with any document token,
    /// then sum. Since vectors are already L2-normalized, dot product = cosine similarity.
    private func maxSim(query: TokenEmbeddings, document: TokenEmbeddings) -> Double {
        var totalScore: Float = 0

        for qi in 0..<query.tokenCount {
            let qOffset = qi * embeddingDim
            var maxDot: Float = -.greatestFiniteMagnitude

            for di in 0..<document.tokenCount {
                let dOffset = di * embeddingDim
                var dot: Float = 0
                query.vectors.withUnsafeBufferPointer { qBuf in
                    document.vectors.withUnsafeBufferPointer { dBuf in
                        vDSP_dotpr(
                            qBuf.baseAddress! + qOffset, 1,
                            dBuf.baseAddress! + dOffset, 1,
                            &dot, vDSP_Length(embeddingDim)
                        )
                    }
                }
                if dot > maxDot {
                    maxDot = dot
                }
            }

            if maxDot > -.greatestFiniteMagnitude {
                totalScore += maxDot
            }
        }

        // Normalize by query token count to get a score in [-1, 1] range
        let normalizedScore = query.tokenCount > 0 ? totalScore / Float(query.tokenCount) : 0
        // Map from [-1, 1] to [0, 1]
        return Double((normalizedScore + 1) / 2)
    }
}
