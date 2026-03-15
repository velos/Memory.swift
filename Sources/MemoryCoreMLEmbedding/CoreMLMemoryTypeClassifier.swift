import CoreML
import Foundation
import Memory

public actor CoreMLMemoryTypeClassifier: MemoryTypeClassifier {
    public let identifier: String

    private let model: MLModel
    private let tokenizer: BertTokenizer
    private let maxSequenceLength: Int

    // Mirrors Autoresearch/memory_autoresearch/config.py MEMORY_TYPES.
    private static let labelOrder: [MemoryType] = [
        .factual,
        .procedural,
        .episodic,
        .semantic,
        .emotional,
        .social,
        .contextual,
        .temporal,
    ]

    public init(
        modelURL: URL,
        vocabURL: URL? = nil,
        maxSequenceLength: Int = 512,
        identifier: String = "coreml-typing-v1",
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

    public func classify(
        documentText: String,
        kind: DocumentKind,
        sourceURL: URL?
    ) async throws -> MemoryTypeAssignment? {
        _ = kind
        _ = sourceURL

        let trimmed = documentText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return try classifySingle(text: trimmed)
    }

    private func classifySingle(text: String) throws -> MemoryTypeAssignment? {
        let encoded = tokenizer.encode(text)
        let output = try model.prediction(from: makeFeatureProvider(encoded: encoded))

        guard let logitsFeature = output.featureValue(for: "type_logits"),
              let logitsArray = logitsFeature.multiArrayValue else {
            throw MemoryError.embedding("CoreML typing model did not return 'type_logits' output")
        }

        let logits = logitsArrayValues(logitsArray)
        guard let prediction = classifyLogits(logits) else { return nil }

        return MemoryTypeAssignment(
            type: prediction.type,
            source: .automatic,
            confidence: prediction.confidence,
            classifierID: identifier
        )
    }

    private func makeFeatureProvider(encoded: BertTokenizer.EncodedInput) throws -> MLFeatureProvider {
        let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)
        let tokenTypeIDs = try MLMultiArray(shape: [1, NSNumber(value: maxSequenceLength)], dataType: .int32)

        let idsPtr = inputIDs.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPtr = attentionMask.dataPointer.assumingMemoryBound(to: Int32.self)
        let typePtr = tokenTypeIDs.dataPointer.assumingMemoryBound(to: Int32.self)
        for index in 0..<maxSequenceLength {
            idsPtr[index] = encoded.inputIDs[index]
            maskPtr[index] = encoded.attentionMask[index]
            typePtr[index] = encoded.tokenTypeIDs[index]
        }

        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "token_type_ids": MLFeatureValue(multiArray: tokenTypeIDs),
        ])
    }

    private func logitsArrayValues(_ logits: MLMultiArray) -> [Float] {
        let count = logits.count
        let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func classifyLogits(_ logits: [Float]) -> (type: MemoryType, confidence: Double)? {
        guard let (bestIndex, bestLogit) = logits.enumerated().max(by: { $0.element < $1.element }) else {
            return nil
        }
        guard bestIndex < Self.labelOrder.count else { return nil }

        let normalization = logits.reduce(into: 0.0) { total, logit in
            total += exp(Double(logit - bestLogit))
        }
        let confidence = normalization > 0 ? 1.0 / normalization : 0.0
        return (Self.labelOrder[bestIndex], confidence)
    }
}
