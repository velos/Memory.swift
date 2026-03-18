// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Memory.swift",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "Memory", targets: ["Memory"]),
        .library(name: "MemoryNaturalLanguage", targets: ["MemoryNaturalLanguage"]),
        .library(name: "MemoryAppleIntelligence", targets: ["MemoryAppleIntelligence"]),
        .library(name: "MemoryCoreMLEmbedding", targets: ["MemoryCoreMLEmbedding"]),
        .executable(name: "memory", targets: ["memory_cli"]),
        .executable(name: "memory_eval", targets: ["memory_eval"]),
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
            name: "Memory",
            dependencies: ["MemoryStorage"],
            path: "Sources/Memory"
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
            name: "MemoryNaturalLanguage",
            dependencies: ["Memory"],
            path: "Sources/MemoryNaturalLanguage"
        ),
        .target(
            name: "MemoryAppleIntelligence",
            dependencies: ["Memory"],
            path: "Sources/MemoryAppleIntelligence"
        ),
        .target(
            name: "MemoryCoreMLEmbedding",
            dependencies: ["Memory", "MemoryNaturalLanguage"],
            path: "Sources/MemoryCoreMLEmbedding",
            resources: [
                .copy("Resources/vocab.txt"),
                .copy("Resources/tokenizer.json"),
            ]
        ),
        .executableTarget(
            name: "memory_cli",
            dependencies: [
                "Memory",
                "MemoryNaturalLanguage",
                "MemoryAppleIntelligence",
                "MemoryCoreMLEmbedding",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/MemoryCLI"
        ),
        .executableTarget(
            name: "memory_eval",
            dependencies: [
                "Memory",
                "MemoryNaturalLanguage",
                "MemoryAppleIntelligence",
                "MemoryCoreMLEmbedding",
                "SQLiteSupport",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/MemoryEvalCLI"
        ),
        .testTarget(
            name: "MemoryTests",
            dependencies: [
                "Memory",
                "MemoryStorage",
                "SQLiteSupport",
            ],
            path: "Tests/MemoryTests"
        ),
        .testTarget(
            name: "MemoryIntegrationTests",
            dependencies: [
                "Memory",
                "MemoryNaturalLanguage",
            ],
            path: "Tests/MemoryIntegrationTests"
        ),
        .testTarget(
            name: "MemoryPerformanceTests",
            dependencies: ["Memory"],
            path: "Tests/MemoryPerformanceTests"
        ),
        .testTarget(
            name: "MemoryCoreMLEmbeddingTests",
            dependencies: ["MemoryCoreMLEmbedding"],
            path: "Tests/MemoryCoreMLEmbeddingTests"
        ),
        .testTarget(
            name: "MemoryEvalCLITests",
            dependencies: ["memory_eval"],
            path: "Tests/MemoryEvalCLITests"
        ),
    ]
)
