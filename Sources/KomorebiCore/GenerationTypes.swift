import Foundation

public enum GemmaEngineError: LocalizedError {
    case engineNotLoaded
    case inferenceFailed(String)

    public var errorDescription: String? {
        switch self {
        case .engineNotLoaded: return "The ANE Gemma engine is not loaded."
        case .inferenceFailed(let n): return "Inference error: \(n)"
        }
    }
}

public struct GenerationStats: Sendable {
    public let backend: String
    public let prefixCachedTokens: Int
    public let prefillTokens: Int
    public let prefillSeconds: Double
    public let decodeTokens: Int
    public let decodeSeconds: Double

    public init(backend: String, prefixCachedTokens: Int, prefillTokens: Int, prefillSeconds: Double,
                decodeTokens: Int, decodeSeconds: Double) {
        self.backend = backend
        self.prefixCachedTokens = prefixCachedTokens
        self.prefillTokens = prefillTokens
        self.prefillSeconds = prefillSeconds
        self.decodeTokens = decodeTokens
        self.decodeSeconds = decodeSeconds
    }

    public var prefillTPS: Double {
        prefillSeconds > 0 ? Double(prefillTokens) / prefillSeconds : 0
    }

    public var decodeTPS: Double {
        decodeSeconds > 0 ? Double(decodeTokens) / decodeSeconds : 0
    }
}

public struct GenerationResult: Sendable {
    public let text: String
    public let tokenIDs: [Int32]
    public let stats: GenerationStats

    public init(text: String, tokenIDs: [Int32], stats: GenerationStats) {
        self.text = text
        self.tokenIDs = tokenIDs
        self.stats = stats
    }
}
