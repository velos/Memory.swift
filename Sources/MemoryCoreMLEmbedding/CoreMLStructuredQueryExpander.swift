import CoreML
import Foundation
import Memory

public final class CoreMLStructuredQueryExpander: @unchecked Sendable, StructuredQueryExpander {
    public let identifier: String

    private let model: MLModel
    private let promptFeatureName: String
    private let outputFeatureNames: [String]

    public init(
        modelURL: URL,
        promptFeatureName: String = "prompt",
        outputFeatureNames: [String] = ["structured_output", "json", "text"],
        identifier: String = "coreml-structured-query-expander",
        computeUnits: MLComputeUnits = .all
    ) throws {
        self.identifier = identifier
        self.promptFeatureName = promptFeatureName
        self.outputFeatureNames = outputFeatureNames

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits

        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try MLModel.compileModel(at: modelURL)
        }
        self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)
    }

    public func expand(
        query: SearchQuery,
        analysis: QueryAnalysis,
        limit: Int
    ) async throws -> StructuredQueryExpansion {
        guard limit > 0 else { return StructuredQueryExpansion() }

        let prompt = """
        Produce structured retrieval expansion JSON.
        Return keys: lexicalQueries, semanticQueries, hypotheticalDocuments, facetHints, entities, topics.
        facetHints items use {tag, confidence, isExplicit}.
        entities items use {label, value, normalizedValue, confidence}.
        Query: \(query.text)
        Facets: \(analysis.facetHints.map(\.tag.rawValue).joined(separator: ", "))
        Entities: \(analysis.entities.map(\.value).joined(separator: ", "))
        Topics: \(analysis.topics.joined(separator: ", "))
        Limit: \(limit)
        """

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            promptFeatureName: MLFeatureValue(string: prompt),
        ])
        let output = try await model.prediction(from: provider)

        guard let rawJSON = firstStringOutput(from: output) else {
            throw MemoryError.search("CoreML structured expander did not return a string output")
        }

        let payload = try JSONDecoder().decode(StructuredQueryExpansionPayload.self, from: Data(rawJSON.utf8))
        return payload.materialize(maxTextQueries: limit)
    }

    private func firstStringOutput(from provider: MLFeatureProvider) -> String? {
        for name in outputFeatureNames {
            if let value = provider.featureValue(for: name)?.stringValue,
               !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return value
            }
        }
        return nil
    }
}

private struct StructuredQueryExpansionPayload: Codable {
    var lexicalQueries: [String]?
    var semanticQueries: [String]?
    var hypotheticalDocuments: [String]?
    var facetHints: [StructuredFacetHintPayload]?
    var entities: [StructuredEntityPayload]?
    var topics: [String]?

    func materialize(maxTextQueries: Int) -> StructuredQueryExpansion {
        StructuredQueryExpansion(
            lexicalQueries: normalizeTextList(lexicalQueries ?? [], limit: min(2, maxTextQueries)),
            semanticQueries: normalizeTextList(semanticQueries ?? [], limit: min(2, maxTextQueries)),
            hypotheticalDocuments: normalizeTextList(hypotheticalDocuments ?? [], limit: 1),
            facetHints: normalizeFacetHints(facetHints ?? [], limit: 4),
            entities: normalizeEntities(entities ?? [], limit: 6),
            topics: normalizeTopics(topics ?? [], limit: 6)
        )
    }

    private func normalizeTextList(_ values: [String], limit: Int) -> [String] {
        guard limit > 0 else { return [] }
        var normalized: [String] = []
        var seen: Set<String> = []
        for value in values {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let key = trimmed.lowercased()
            guard seen.insert(key).inserted else { continue }
            normalized.append(trimmed)
            if normalized.count >= limit {
                break
            }
        }
        return normalized
    }

    private func normalizeFacetHints(_ payloads: [StructuredFacetHintPayload], limit: Int) -> [FacetHint] {
        guard limit > 0 else { return [] }
        var normalized: [FacetTag: FacetHint] = [:]
        for payload in payloads {
            guard let tag = FacetTag.parse(payload.tag) else { continue }
            let candidate = FacetHint(
                tag: tag,
                confidence: payload.confidence ?? 0.75,
                isExplicit: payload.isExplicit ?? false
            )
            if let existing = normalized[tag] {
                if candidate.confidence > existing.confidence {
                    normalized[tag] = candidate
                }
            } else {
                normalized[tag] = candidate
            }
        }
        return normalized.values
            .sorted { lhs, rhs in
                if lhs.confidence == rhs.confidence {
                    return lhs.tag.rawValue < rhs.tag.rawValue
                }
                return lhs.confidence > rhs.confidence
            }
            .prefix(limit)
            .map { $0 }
    }

    private func normalizeEntities(_ payloads: [StructuredEntityPayload], limit: Int) -> [MemoryEntity] {
        guard limit > 0 else { return [] }
        var normalized: [String: MemoryEntity] = [:]
        for payload in payloads {
            guard let label = EntityLabel.parse(payload.label) else { continue }
            let value = payload.value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty else { continue }
            let normalizedValue = normalizeEntityValue(payload.normalizedValue ?? value)
            guard !normalizedValue.isEmpty else { continue }
            normalized[normalizedValue] = MemoryEntity(
                label: label,
                value: value,
                normalizedValue: normalizedValue,
                confidence: payload.confidence
            )
        }
        return normalized.values
            .sorted { lhs, rhs in lhs.normalizedValue < rhs.normalizedValue }
            .prefix(limit)
            .map { $0 }
    }

    private func normalizeTopics(_ values: [String], limit: Int) -> [String] {
        guard limit > 0 else { return [] }
        var normalized: [String] = []
        var seen: Set<String> = []
        for value in values {
            let candidate = value
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
                .split(whereSeparator: \.isWhitespace)
                .map(String.init)
                .prefix(4)
                .joined(separator: " ")
            guard !candidate.isEmpty else { continue }
            guard seen.insert(candidate).inserted else { continue }
            normalized.append(candidate)
            if normalized.count >= limit {
                break
            }
        }
        return normalized
    }

    private func normalizeEntityValue(_ raw: String) -> String {
        let punctuation = CharacterSet(charactersIn: ",:;!?()[]{}\"'`")
        return raw
            .trimmingCharacters(in: .whitespacesAndNewlines.union(punctuation))
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
            .lowercased()
    }
}

private struct StructuredFacetHintPayload: Codable {
    var tag: String
    var confidence: Double?
    var isExplicit: Bool?
}

private struct StructuredEntityPayload: Codable {
    var label: String
    var value: String
    var normalizedValue: String?
    var confidence: Double?
}
