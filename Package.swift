// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "KomorebiKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "KomorebiCore",
            targets: ["KomorebiCore"]
        ),
        .library(
            name: "KomorebiIPC",
            targets: ["KomorebiIPC"]
        ),
    ],
    targets: [
        .target(
            name: "KomorebiIPC",
            path: "Sources/KomorebiIPC"
        ),
        .target(
            name: "KomorebiANEDirect",
            path: "Sources/KomorebiANEDirect",
            linkerSettings: [
                .linkedFramework("IOSurface"),
                .linkedFramework("Foundation"),
            ]
        ),
        .target(
            name: "KomorebiCore",
            dependencies: ["KomorebiIPC", "KomorebiANEDirect"],
            path: "Sources/KomorebiCore",
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .testTarget(
            name: "KomorebiCoreTests",
            dependencies: ["KomorebiCore", "KomorebiIPC", "KomorebiANEDirect"],
            path: "Tests/KomorebiCoreTests"
        ),
    ]
)
