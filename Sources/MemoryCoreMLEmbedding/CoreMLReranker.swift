import CoreML
import Foundation
import Memory

public actor CoreMLReranker: Reranker {
    public let identifier: String

    private let model: MLModel
    private let tokenizer: BertTokenizer
    private let maxSequenceLength: Int

    public init(
        modelURL: URL,
        vocabURL: URL? = nil,
        maxSequenceLength: Int = 512,
        identifier: String = "coreml-tinybert-reranker",
        computeUnits: MLComputeUnits = .all
    ) throws {
        self.identifier = identifier
        self.maxSequenceLength = maxSequenceLength

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

        var assessments: [RerankAssessment] = []
        assessments.reserveCapacity(candidates.count)

        for candidate in candidates {
            let score = try scorePair(query: query.text, document: candidate.content)
            assessments.append(RerankAssessment(chunkID: candidate.chunkID, relevance: score))
        }

        return assessments
    }

    private func scorePair(query: String, document: String) throws -> Double {
        let encoded = tokenizer.encodePair(query: query, document: document)

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

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "token_type_ids": MLFeatureValue(multiArray: tokenTypeIDs),
        ])

        let output = try model.prediction(from: inputFeatures)

        guard let scoreFeature = output.featureValue(for: "relevance_score"),
              let scoreArray = scoreFeature.multiArrayValue else {
            throw MemoryError.embedding("CoreML reranker did not return 'relevance_score' output")
        }

        let rawLogit = Double(scoreArray.dataPointer.assumingMemoryBound(to: Float.self).pointee)
        return sigmoid(rawLogit)
    }

    private func sigmoid(_ x: Double) -> Double {
        1.0 / (1.0 + exp(-x))
    }
}
