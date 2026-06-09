#if MEMORY_COREML_EMBEDDING
import CoreML
import Foundation

enum CoreMLModelResolver {
    static func compiledModelURL(for modelURL: URL) throws -> URL {
        if modelURL.pathExtension == "mlmodelc" {
            return modelURL
        }

        let cacheURL = try cachedCompiledURL(for: modelURL)
        if FileManager.default.fileExists(atPath: cacheURL.path) {
            return cacheURL
        }

        let compiledURL = try MLModel.compileModel(at: modelURL)
        do {
            try FileManager.default.copyItem(at: compiledURL, to: cacheURL)
        } catch {
            if FileManager.default.fileExists(atPath: cacheURL.path) {
                return cacheURL
            }
            throw error
        }
        return cacheURL
    }

    private static func cachedCompiledURL(for modelURL: URL) throws -> URL {
        let fileManager = FileManager.default
        let cacheRoot = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        let directory = cacheRoot.appendingPathComponent("MemoryCoreML", isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)

        let name = modelURL.deletingPathExtension().lastPathComponent
        let key = "\(name)-\(sourceStamp(for: modelURL))"
            .map { character -> Character in
                character.isLetter || character.isNumber ? character : "-"
            }
        return directory
            .appendingPathComponent(String(key), isDirectory: true)
            .appendingPathExtension("mlmodelc")
    }

    private static func sourceStamp(for modelURL: URL) -> String {
        let resourceValues = try? modelURL.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        let modified = resourceValues?.contentModificationDate?.timeIntervalSince1970 ?? 0
        let size = resourceValues?.fileSize ?? 0
        return "\(Int(modified))-\(size)"
    }
}
#endif
