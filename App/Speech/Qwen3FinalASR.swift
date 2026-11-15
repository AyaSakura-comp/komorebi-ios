import Foundation
import KomorebiCore
import OpenCC

/// Qwen3-ASR-1.7B on the Apple Neural Engine for finished sentences, as desktop Komorebi does with TEA-ASR
/// (`PARTIAL_ASR = {"tea": "sensevoice"}`): the partials while speaking stay on SenseVoice (~50 ms per re-read), and
/// once the VAD closes a sentence this LLM recogniser reads the whole sentence again (about half SenseVoice's errors on
/// Taiwanese-accent Mandarin-English code-switching, ~1 s a sentence).
public final class Qwen3FinalASR: @unchecked Sendable {
    public let engine: Qwen3ASRANEEngine
    private let converter: ChineseConverter
    private let queue = DispatchQueue(label: "dev.ayasakura.komorebi.qwen3asr", qos: .userInitiated)
    /// "Chinese" is given, as desktop Komorebi does for TEA: same MER as letting the model name the language on the
    /// code-switching set (8.84 both), English sentences still come out in English, and it saves three tokens a sentence.
    public var language: String? = "Chinese"

    public init(ggufURL: URL, cacheDirectory: URL, progress: ((String) -> Void)? = nil) throws {
        let gguf = try GGUFFile(url: ggufURL, loadTokenizerArrays: true, mapTensorData: false)
        engine = try Qwen3ASRANEEngine(gguf: gguf, options: .init(cacheDirectory: cacheDirectory), progress: progress)
        converter = try SenseVoiceASR.makeTWConverter()
    }

    public struct Output: Sendable {
        public let text: String
        public let result: Qwen3ASRResult
    }

    /// Recognises one finished sentence (16 kHz mono). nil when the engine failed (the caller falls back to SenseVoice).
    public func transcribe(_ samples: [Float]) -> Output? {
        queue.sync {
            guard samples.count >= 1600, let r = try? engine.transcribe(samples, language: language) else { return nil }
            let text = r.text.isEmpty ? "" : converter.convert(r.text)
            return Output(text: text, result: r)
        }
    }

    public func warmUp() -> Double {
        queue.sync {
            let t0 = CFAbsoluteTimeGetCurrent()
            try? engine.warmUp()
            return CFAbsoluteTimeGetCurrent() - t0
        }
    }
}
