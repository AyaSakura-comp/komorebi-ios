import Accelerate
import Foundation
import OpenCC
import SherpaOnnx

/// CPU-only SenseVoice-Small (int8 ONNX) recognizer + OpenCC `s2twp` Traditional Chinese converter.
/// Faithfully ports `app/sensevoice_service.py` preprocessing:
///  - DC offset removal (mean subtraction)
///  - 100 ms head silence + 50 ms tail silence padding
///  - 15 s sliding window cap during partial streaming
///  - Strictly neural ASR + standard OpenCC `s2twp` (zero hardcoded regex rules)
public final class SenseVoiceASR: @unchecked Sendable {
    public static let sampleRate = 16_000
    public static let headPadSamples = 1_600  // 100 ms @ 16 kHz
    public static let tailPadSamples = 800    // 50 ms @ 16 kHz
    public static let maxPartialSamples = 240_000 // 15 s @ 16 kHz

    private let recognizer: SherpaOnnxOfflineRecognizer
    private let converter: ChineseConverter
    private let queue = DispatchQueue(label: "dev.ayasakura.komorebi.sensevoice", qos: .userInitiated)

    public init(modelURL: URL, tokensURL: URL, numThreads: Int = 2) throws {
        var svCfg = sherpaOnnxOfflineSenseVoiceModelConfig(
            model: modelURL.path,
            language: "auto",
            useInverseTextNormalization: true
        )
        var modelCfg = sherpaOnnxOfflineModelConfig(
            tokens: tokensURL.path,
            numThreads: numThreads,
            provider: "cpu",
            debug: 0,
            senseVoice: svCfg
        )
        let featCfg = sherpaOnnxFeatureConfig(sampleRate: Self.sampleRate, featureDim: 80)
        var recCfg = sherpaOnnxOfflineRecognizerConfig(
            featConfig: featCfg,
            modelConfig: modelCfg
        )
        _ = svCfg
        _ = modelCfg
        self.recognizer = SherpaOnnxOfflineRecognizer(config: &recCfg)
        self.converter = try Self.makeTWConverter()
    }

    public static func makeTWConverter() throws -> ChineseConverter {
        let frameworkBundle = Bundle(for: ChineseConverter.self)
        let candidates: [Bundle?] = [
            frameworkBundle.url(forResource: "SwiftyOpenCC_OpenCC", withExtension: "bundle").flatMap(Bundle.init(url:)),
            Bundle.main.url(forResource: "SwiftyOpenCC_OpenCC", withExtension: "bundle").flatMap(Bundle.init(url:)),
            frameworkBundle,
            Bundle.main,
        ]
        for b in candidates.compactMap({ $0 }) {
            if b.url(forResource: "STPhrases", withExtension: "ocd", subdirectory: "Dictionary") != nil {
                return try ChineseConverter(bundle: b, option: [.traditionalize, .TWStandard, .TWIdiom])
            }
        }
        return try ChineseConverter(option: [.traditionalize, .TWStandard, .TWIdiom])
    }

    public struct Transcript: Sendable {
        public let text: String
        public let latencyMs: Double
    }

    public func transcribe(samples: [Float], isPartial: Bool) -> Transcript {
        queue.sync {
            let t0 = CFAbsoluteTimeGetCurrent()
            let slice: ArraySlice<Float>
            if isPartial && samples.count > Self.maxPartialSamples {
                slice = samples.suffix(Self.maxPartialSamples)
            } else {
                slice = samples[...]
            }
            guard slice.count >= 1600 else {
                return Transcript(text: "", latencyMs: 0)
            }

            // DC mean removal + head/tail silence padding (matches sensevoice_service.py)
            let count = slice.count
            var mean: Float = 0
            slice.withContiguousStorageIfAvailable { buf in
                vDSP_meanv(buf.baseAddress!, 1, &mean, vDSP_Length(count))
            }
            if mean == 0 {
                let raw = Array(slice)
                vDSP_meanv(raw, 1, &mean, vDSP_Length(count))
            }

            var padded = [Float](repeating: 0, count: Self.headPadSamples + count + Self.tailPadSamples)
            var idx = Self.headPadSamples
            for s in slice {
                padded[idx] = s - mean
                idx += 1
            }

            let res = recognizer.decode(samples: padded, sampleRate: Self.sampleRate)
            let rawText = res.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let twText = rawText.isEmpty ? "" : converter.convert(rawText)
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            return Transcript(text: twText, latencyMs: ms)
        }
    }
}
