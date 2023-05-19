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
    dependencies: [
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "MimicTense",
            dependencies: [
                "MimicComputeEngineService",
                "MimicTransferables",
                "MimicCore",
            ]),
        .target(
            name: "MimicComputeEngineService",
            dependencies: [
                "MimicComputeEngineModule",
                "MlComputeEngineModule",
                "MimicTransferables",
                "MimicCore",
            ]),
        .target(
            name: "MimicComputeEngineModule",
            dependencies: [
                "MimicTransferables",
                "MimicCore",
            ]),
        .target(
            name: "MlComputeEngineModule",
            dependencies: [
                "MimicComputeEngineModule",
                "MimicTransferables",
                "MimicCore",
            ]),
        .target(
            name: "MpsgComputeEngineModule",
            dependencies: [
                "MimicComputeEngineModule",
                "MimicTransferables",
                "MimicCore",
            ]),
        .target(name: "MimicTransferables",
               dependencies: ["MimicCore"]),
        .target(name: "MimicCore"),
        .target(name: "MimicTesting"),
        .testTarget(
            name: "MimicTenseTests",
            dependencies: [
                "MimicTense",
            ]),
        .testTarget(
            name: "MimicComputeEngineServiceTests",
            dependencies: [
                "MimicComputeEngineService",
                "MimicTesting",
                .product(name: "Collections", package: "swift-collections")
            ]),
        .testTarget(
            name: "MimicComputeEngineModuleTests",
            dependencies: [
                "MimicComputeEngineModule",
                "MlComputeEngineModule",
                "MpsgComputeEngineModule",
                "MimicTesting",
                .product(name: "Collections", package: "swift-collections")
            ]),
        .testTarget(
            name: "MlComputeEngineModuleTests",
            dependencies: [
                "MlComputeEngineModule",
                "MimicTesting",
                .product(name: "Collections", package: "swift-collections")
            ]),
        .testTarget(
            name: "MpsgComputeEngineModuleTests",
            dependencies: [
                "MpsgComputeEngineModule",
                .product(name: "Collections", package: "swift-collections")
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
