import CoreML
import Foundation
import Memory

public actor CoreMLEmbeddingProvider: EmbeddingProvider {
    public let identifier: String

    private let model: MLModel
    private let tokenizer: BertTokenizer
    private let maxSequenceLength: Int

    public init(
        modelURL: URL,
        vocabURL: URL? = nil,
        maxSequenceLength: Int = 512,
        identifier: String = "coreml-embedding-v1",
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

    public func embed(texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        var results: [[Float]] = []
        results.reserveCapacity(texts.count)
        for text in texts {
            let vector = try embedSingle(text: text)
            results.append(vector)
        }
        return results
    }

    private func embedSingle(text: String) throws -> [Float] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw MemoryError.embedding("Cannot embed empty text")
        }

        let encoded = tokenizer.encode(trimmed)

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

        guard let embeddingFeature = output.featureValue(for: "embedding"),
              let embeddingArray = embeddingFeature.multiArrayValue else {
            throw MemoryError.embedding("CoreML model did not return 'embedding' output")
        }

        let count = embeddingArray.count
        let ptr = embeddingArray.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
