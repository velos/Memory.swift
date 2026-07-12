// swift-tools-version: 6.2
import Foundation
import PackageDescription

// Some Apple Swift CLT/snapshot installs place Swift Testing in Developer
// frameworks instead of the default SwiftPM search paths. When DEVELOPER_DIR
// is set, use only that toolchain's paths: mixing another Xcode's Testing
// framework with the selected compiler's @Test macro fails to build.
let developerDir = ProcessInfo.processInfo.environment["DEVELOPER_DIR"]

let developerFrameworkPaths: [String] = {
    if let developerDir, !developerDir.isEmpty {
        return [developerDir + "/Library/Developer/Frameworks"]
    }
    return [
        "/Library/Developer/CommandLineTools/Library/Developer/Frameworks",
        "/Applications/Xcode.app/Contents/Developer/Library/Developer/Frameworks",
    ]
}()

let developerLibraryPaths: [String] = {
    if let developerDir, !developerDir.isEmpty {
        return [developerDir + "/Library/Developer/usr/lib"]
    }
    return [
        "/Library/Developer/CommandLineTools/Library/Developer/usr/lib",
        "/Applications/Xcode.app/Contents/Developer/Library/Developer/usr/lib",
    ]
}()

let developerFrameworkSwiftSettings: [SwiftSetting] = developerFrameworkPaths.compactMap { path in
    guard FileManager.default.fileExists(atPath: path) else { return nil }
    return .unsafeFlags(["-F", path], .when(platforms: [.macOS]))
}

let developerFrameworkLinkerSettings: [LinkerSetting] = developerFrameworkPaths.compactMap { path in
    guard FileManager.default.fileExists(atPath: path) else { return nil }
    return .unsafeFlags(["-F", path, "-Xlinker", "-rpath", "-Xlinker", path], .when(platforms: [.macOS]))
}

let developerLibraryLinkerSettings: [LinkerSetting] = developerLibraryPaths.compactMap { path in
    guard FileManager.default.fileExists(atPath: path) else { return nil }
    return .unsafeFlags(["-L", path, "-Xlinker", "-rpath", "-Xlinker", path], .when(platforms: [.macOS]))
}

let developerTestLinkerSettings = developerFrameworkLinkerSettings + developerLibraryLinkerSettings

private let memoryTraitSwiftSettings: [SwiftSetting] = [
    .define("MEMORY_NATURAL_LANGUAGE", .when(traits: ["default"])),
    .define("MEMORY_NATURAL_LANGUAGE", .when(traits: ["MemoryNaturalLanguage"])),
    .define("MEMORY_APPLE_INTELLIGENCE", .when(traits: ["MemoryAppleIntelligence"])),
    .define("MEMORY_COREML_EMBEDDING", .when(traits: ["CoreMLEmbedding"])),
]

private let memoryTestSwiftSettings = memoryTraitSwiftSettings + developerFrameworkSwiftSettings

let package = Package(
    name: "AgentMemory",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "AgentMemory", targets: ["AgentMemory"]),
        .executable(name: "memory", targets: ["memory_cli"]),
        .executable(name: "memory_eval", targets: ["memory_eval"]),
    ],
    traits: [
        .default(enabledTraits: ["MemoryNaturalLanguage"]),
        .trait(
            name: "MemoryNaturalLanguage",
            description: "Enable NaturalLanguage-backed embedding, tokenization, and default configuration APIs."
        ),
        .trait(
            name: "MemoryAppleIntelligence",
            description: "Enable Apple Intelligence provider APIs when FoundationModels is available."
        ),
        .trait(
            name: "CoreMLEmbedding",
            description: "Enable bundled Core ML embedding, tokenizer, and reranking APIs."
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "SQLiteSupport",
            path: "Sources/SQLiteSupport",
            linkerSettings: [
                .linkedLibrary("sqlite3"),
            ]
        ),
        .target(
            name: "AgentMemory",
            dependencies: [
                "MemoryStorage",
                .target(name: "MemoryCoreMLAssets", condition: .when(traits: ["CoreMLEmbedding"])),
            ],
            path: "Sources/AgentMemory",
            swiftSettings: memoryTraitSwiftSettings
        ),
        .target(
            name: "MemoryStorage",
            dependencies: [
                "SQLiteSupport",
                "CSQLiteVec",
            ],
            path: "Sources/MemoryStorage"
        ),
        .target(
            name: "CSQLiteVec",
            path: "Sources/CSQLiteVec",
            publicHeadersPath: "include",
            cSettings: [
                .define("SQLITE_CORE", to: "1"),
            ]
        ),
        .target(
            name: "MemoryCoreMLAssets",
            path: "Sources/MemoryCoreMLAssets",
            resources: [
                .copy("Resources/vocab.txt"),
                .copy("Resources/tokenizer.json"),
                .copy("Resources/embedding-v1.mlmodelc"),
            ]
        ),
        .executableTarget(
            name: "memory_cli",
            dependencies: [
                "AgentMemory",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/MemoryCLI",
            swiftSettings: memoryTraitSwiftSettings
        ),
        .executableTarget(
            name: "memory_eval",
            dependencies: [
                "AgentMemory",
                "SQLiteSupport",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/MemoryEvalCLI",
            swiftSettings: memoryTraitSwiftSettings
        ),
        .testTarget(
            name: "MemoryTests",
            dependencies: [
                "AgentMemory",
                "MemoryStorage",
                "SQLiteSupport",
            ],
            path: "Tests/MemoryTests",
            swiftSettings: memoryTestSwiftSettings,
            linkerSettings: developerTestLinkerSettings
        ),
        .testTarget(
            name: "MemoryIntegrationTests",
            dependencies: ["AgentMemory"],
            path: "Tests/MemoryIntegrationTests",
            swiftSettings: memoryTestSwiftSettings,
            linkerSettings: developerTestLinkerSettings
        ),
        .testTarget(
            name: "MemoryPerformanceTests",
            dependencies: ["AgentMemory"],
            path: "Tests/MemoryPerformanceTests",
            swiftSettings: memoryTestSwiftSettings,
            linkerSettings: developerTestLinkerSettings
        ),
        .testTarget(
            name: "MemoryCoreMLEmbeddingTests",
            dependencies: ["AgentMemory"],
            path: "Tests/MemoryCoreMLEmbeddingTests",
            swiftSettings: memoryTestSwiftSettings,
            linkerSettings: developerTestLinkerSettings
        ),
        .testTarget(
            name: "MemoryUITests",
            dependencies: ["AgentMemory"],
            path: "Tests/MemoryUITests",
            swiftSettings: memoryTestSwiftSettings,
            linkerSettings: developerTestLinkerSettings
        ),
        .testTarget(
            name: "MemoryEvalCLITests",
            dependencies: ["memory_eval"],
            path: "Tests/MemoryEvalCLITests",
            swiftSettings: memoryTestSwiftSettings,
            linkerSettings: developerTestLinkerSettings
        ),
    ]
)
