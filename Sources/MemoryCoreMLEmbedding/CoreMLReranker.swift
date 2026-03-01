import CoreML
import Foundation
import Memory

public actor CoreMLReranker: Reranker {
    public let identifier: String

    private let model: MLModel
    private let tokenizer: BertTokenizer
    private let maxSequenceLength: Int
    private let batchSize: Int
    private let documentCharacterLimit: Int

    public init(
        modelURL: URL,
        vocabURL: URL? = nil,
        maxSequenceLength: Int = 512,
        batchSize: Int = 8,
        documentCharacterLimit: Int = 4_096,
        identifier: String = "coreml-tinybert-reranker",
        computeUnits: MLComputeUnits = .all
    ) throws {
        self.identifier = identifier
        self.maxSequenceLength = maxSequenceLength
        self.batchSize = max(1, batchSize)
        self.documentCharacterLimit = max(512, documentCharacterLimit)

        let resolvedVocabURL: URL
        if let vocabURL {
            resolvedVocabURL = vocabURL
        } else if let bundledURL = Bundle.module.url(forResource: "vocab", withExtension: "txt") {
            resolvedVocabURL = bundledURL
        } else {
            throw MemoryError.embedding("No vocab.txt found. Provide vocabURL or include vocab.txt in bundle.")
        }

        self.tokenizer = try BertTokenizer(vocabURL: resolvedVocabURL, maxSequenceLength: maxSequenceLength)

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
        let normalizedQuery = query.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedQuery.isEmpty else { return [] }

        var assessments: [RerankAssessment] = []
        assessments.reserveCapacity(candidates.count)

        var start = 0
        while start < candidates.count {
            let end = min(candidates.count, start + batchSize)
            let batch = Array(candidates[start..<end])
            let documents = batch.map { normalizeDocumentForReranking($0.content) }
            let scores = try scoreBatch(query: normalizedQuery, documents: documents)
            guard scores.count == batch.count else {
                throw MemoryError.embedding(
                    "CoreML reranker returned \(scores.count) scores for \(batch.count) candidates"
                )
            }

            for (index, candidate) in batch.enumerated() {
                assessments.append(
                    RerankAssessment(chunkID: candidate.chunkID, relevance: scores[index])
                )
            }

            start = end
        }

        return assessments
    }

    private func scoreBatch(query: String, documents: [String]) throws -> [Double] {
        guard !documents.isEmpty else { return [] }

        var providers: [MLFeatureProvider] = []
        providers.reserveCapacity(documents.count)
        for document in documents {
            let encoded = tokenizer.encodePair(query: query, document: document)
            providers.append(try makeFeatureProvider(encoded: encoded))
        }

        let batchProvider = MLArrayBatchProvider(array: providers)
        let outputs = try model.predictions(fromBatch: batchProvider)

        var scores: [Double] = []
        scores.reserveCapacity(outputs.count)
        for index in 0..<outputs.count {
            let output = outputs.features(at: index)
            guard let scoreFeature = output.featureValue(for: "relevance_score"),
                  let scoreArray = scoreFeature.multiArrayValue else {
                throw MemoryError.embedding("CoreML reranker did not return 'relevance_score' output")
            }
            let rawLogit = Double(scoreArray.dataPointer.assumingMemoryBound(to: Float.self).pointee)
            scores.append(sigmoid(rawLogit))
        }

        return scores
    }

    private func makeFeatureProvider(encoded: BertTokenizer.EncodedInput) throws -> MLFeatureProvider {
        let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)
        let tokenTypeIDs = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)

        let idsPtr = inputIDs.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPtr = attentionMask.dataPointer.assumingMemoryBound(to: Int32.self)
        let typePtr = tokenTypeIDs.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<maxSequenceLength {
            idsPtr[i] = encoded.inputIDs[i]
            maskPtr[i] = encoded.attentionMask[i]
            typePtr[i] = encoded.tokenTypeIDs[i]
        }

        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "token_type_ids": MLFeatureValue(multiArray: tokenTypeIDs),
        ])
    }

    private func normalizeDocumentForReranking(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return text }
        if trimmed.count <= documentCharacterLimit {
            return trimmed
        }
        return String(trimmed.prefix(documentCharacterLimit))
    }

    private func sigmoid(_ x: Double) -> Double {
        1.0 / (1.0 + exp(-x))
    }
}
