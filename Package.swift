// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MimicTense",
    platforms: [.macOS(.v13)],
    products: [
        .library(
            name: "MimicTense",
            targets: [
                "MimicTense"
            ]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "MimicTense",
            dependencies: [
                "MimicComputeEngine",
                "MimicTransferables",
                "MimicCore",
            ]),
        .target(
            name: "MimicComputeEngine",
            dependencies: [
                "MimicTransferables",
                "MimicCore",
            ]),
        .target(name: "MimicTransferables",
               dependencies: ["MimicCore"]),
        .target(name: "MimicCore"),
        .testTarget(
            name: "MimicTenseTests",
            dependencies: [
                "MimicTense",
            ]),
        .testTarget(
            name: "MimicComputeEngineTests",
            dependencies: [
                "MimicComputeEngine",
            ]),
        .testTarget(
            name: "MimicTransferablesTests",
            dependencies: [
                "MimicTransferables",
            ]),
        .testTarget(
            name: "MimicCoreTests",
            dependencies: ["MimicCore"]),
    ]
)
