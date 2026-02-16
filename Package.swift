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
        .executable(name: "memory", targets: ["memory_cli"]),
        .executable(name: "memory_eval", targets: ["memory_eval"]),
    ],
    dependencies: [
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "7.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "Memory",
            dependencies: ["MemoryStorage"],
            path: "Sources/Memory"
        ),
        .target(
            name: "MemoryStorage",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            path: "Sources/MemoryStorage"
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
        .executableTarget(
            name: "memory_cli",
            dependencies: [
                "Memory",
                "MemoryNaturalLanguage",
                "MemoryAppleIntelligence",
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
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            path: "Sources/MemoryEvalCLI"
        ),
        .testTarget(
            name: "MemoryTests",
            dependencies: [
                "Memory",
                "MemoryStorage",
                .product(name: "GRDB", package: "GRDB.swift"),
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
    ]
)
