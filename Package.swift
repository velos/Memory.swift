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
        .library(name: "MemoryMLX", targets: ["MemoryMLX"]),
        .executable(name: "memory", targets: ["memory_cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "7.0.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.27.2"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "Memory",
            dependencies: ["MemoryStorage"],
            path: "Sources/QMDKit"
        ),
        .target(
            name: "MemoryStorage",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            path: "Sources/QMDKitStorage"
        ),
        .target(
            name: "MemoryNaturalLanguage",
            dependencies: ["Memory"],
            path: "Sources/QMDKitNaturalLanguage"
        ),
        .target(
            name: "MemoryMLX",
            dependencies: [
                "Memory",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/QMDKitMLX"
        ),
        .executableTarget(
            name: "memory_cli",
            dependencies: [
                "Memory",
                "MemoryNaturalLanguage",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/MemoryCLI"
        ),
        .testTarget(
            name: "MemoryTests",
            dependencies: ["Memory"],
            path: "Tests/QMDKitTests"
        ),
        .testTarget(
            name: "MemoryIntegrationTests",
            dependencies: [
                "Memory",
                "MemoryNaturalLanguage",
            ],
            path: "Tests/QMDKitIntegrationTests"
        ),
        .testTarget(
            name: "MemoryPerformanceTests",
            dependencies: ["Memory"],
            path: "Tests/QMDKitPerformanceTests"
        ),
    ]
)
