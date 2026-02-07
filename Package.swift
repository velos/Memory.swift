// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "QMDKit",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "QMDKit", targets: ["QMDKit"]),
        .library(name: "QMDKitNaturalLanguage", targets: ["QMDKitNaturalLanguage"]),
        .library(name: "QMDKitMLX", targets: ["QMDKitMLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "7.0.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.27.2"),
    ],
    targets: [
        .target(
            name: "QMDKit",
            dependencies: ["QMDKitStorage"]
        ),
        .target(
            name: "QMDKitStorage",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
            ]
        ),
        .target(
            name: "QMDKitNaturalLanguage",
            dependencies: ["QMDKit"]
        ),
        .target(
            name: "QMDKitMLX",
            dependencies: [
                "QMDKit",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "QMDKitTests",
            dependencies: ["QMDKit"]
        ),
        .testTarget(
            name: "QMDKitIntegrationTests",
            dependencies: [
                "QMDKit",
                "QMDKitNaturalLanguage",
            ]
        ),
        .testTarget(
            name: "QMDKitPerformanceTests",
            dependencies: ["QMDKit"]
        ),
    ]
)
