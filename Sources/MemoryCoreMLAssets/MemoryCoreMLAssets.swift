import Foundation

public enum MemoryCoreMLAssets {
    public static func url(forResource name: String, withExtension ext: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: ext)
    }
}
