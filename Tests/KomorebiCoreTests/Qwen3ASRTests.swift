import Foundation
import XCTest
@testable import KomorebiCore

/// Qwen3-ASR on the ANE against the reference dump of Tools/qwen3-asr/make_reference.py.
/// QWEN3_ASR_GGUF=<converted gguf> QWEN3_ASR_REF=<reference dir> swift test -c release --filter Qwen3ASRTests
/// QWEN3_ASR_MER=<meta.json of the bake-off set> also transcribes the whole set and writes the hypotheses next to it.
final class Qwen3ASRTests: XCTestCase {
    private var ggufURL: URL? { ProcessInfo.processInfo.environment["QWEN3_ASR_GGUF"].map { URL(fileURLWithPath: $0) } }
    private var refDir: URL? { ProcessInfo.processInfo.environment["QWEN3_ASR_REF"].map { URL(fileURLWithPath: $0) } }
    private var cacheDir: URL { FileManager.default.temporaryDirectory.appendingPathComponent("komorebi-qwen3asr-cache") }

    private struct Manifest: Decodable {
        struct Utt: Decodable {
            let wav: String
            let samples: Int
            let audio_tokens: Int
            let features: [Int]
            let out_auto: [Int32]
            let out_Chinese: [Int32]
            let text_auto: String
        }
        let mel_filters: [Int]
        let utterances: [Utt]
    }

    private func manifest() throws -> Manifest {
        try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: refDir!.appendingPathComponent("manifest.json")))
    }

    private func floats(_ name: String) throws -> [Float] {
        let d = try Data(contentsOf: refDir!.appendingPathComponent(name + ".f32"))
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// 16-bit PCM WAV -> mono float (channels averaged, as soundfile + mean).
    static func readWAV(_ path: String) throws -> [Float] {
        let d = try Data(contentsOf: URL(fileURLWithPath: path))
        return try d.withUnsafeBytes { raw -> [Float] in
            let b = raw.bindMemory(to: UInt8.self)
            func u32(_ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 | Int(b[o + 2]) << 16 | Int(b[o + 3]) << 24 }
            func u16(_ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 }
            var o = 12
            var channels = 1, bits = 16
            while o + 8 <= b.count {
                let id = String(bytes: b[o..<(o + 4)], encoding: .ascii) ?? ""
                let size = u32(o + 4)
                if id == "fmt " { channels = u16(o + 10); bits = u16(o + 22) }
                if id == "data" {
                    guard bits == 16 else { throw NSError(domain: "wav", code: bits) }
                    let frames = size / (2 * channels)
                    var out = [Float](repeating: 0, count: frames)
                    for f in 0..<frames {
                        var s: Float = 0
                        for c in 0..<channels {
                            let v = Int16(bitPattern: UInt16(u16(o + 8 + (f * channels + c) * 2)))
                            s += Float(v) / 32768
                        }
                        out[f] = s / Float(channels)
                    }
                    return out
                }
                o += 8 + size + (size & 1)
            }
            throw NSError(domain: "wav", code: -1)
        }
    }

    private static func stats(_ a: [Float], _ b: [Float]) -> (maxAbs: Float, cosine: Float) {
        var m: Float = 0, dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<min(a.count, b.count) {
            m = max(m, abs(a[i] - b[i])); dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        return (m, dot / (sqrtf(na) * sqrtf(nb) + 1e-12))
    }

    func testMelMatchesReference() throws {
        guard let ggufURL, refDir != nil else { throw XCTSkip("QWEN3_ASR_GGUF / QWEN3_ASR_REF not set") }
        let gguf = try GGUFFile(url: ggufURL, loadTokenizerArrays: false)
        let cfg = try Qwen3ASRConfig.from(gguf.metadata)
        XCTAssertEqual(cfg.melFilters, try floats("mel_filters"))
        for (i, u) in try manifest().utterances.enumerated() {
            let samples = try Self.readWAV(u.wav)
            XCTAssertEqual(samples.count, u.samples)
            let (feat, frames) = Qwen3ASRMel.compute(samples, config: cfg)
            let ref = try floats("u\(i)_features")            // [mel][frames]
            XCTAssertEqual(frames, u.features[1])
            XCTAssertEqual(cfg.audioTokenCount(frames: frames), u.audio_tokens)
            var mine = [Float](repeating: 0, count: ref.count)
            for f in 0..<frames { for m in 0..<cfg.melBins { mine[m * frames + f] = feat[f * cfg.melBins + m] } }
            let s = Self.stats(mine, ref)
            print("u\(i) mel: max |diff| \(s.maxAbs), cosine \(s.cosine)")
            XCTAssertLessThan(s.maxAbs, 2e-3)
        }
    }

    func testTranscribeMatchesReference() throws {
        guard let ggufURL, refDir != nil else { throw XCTSkip("QWEN3_ASR_GGUF / QWEN3_ASR_REF not set") }
        let gguf = try GGUFFile(url: ggufURL, loadTokenizerArrays: true)
        let env = ProcessInfo.processInfo.environment
        var opts = Qwen3ASRANEEngine.Options(cacheDirectory: cacheDir)
        if let v = env["QWEN3_ASR_DEC_BITS"].flatMap(Int.init) { opts.decoderLUTBits = v }
        if let v = env["QWEN3_ASR_ENC_BITS"].flatMap(Int.init) { opts.encoderLUTBits = v }
        if let v = env["QWEN3_ASR_HEAD_BITS"].flatMap(Int.init) { opts.headLUTBits = v }
        let t0 = CFAbsoluteTimeGetCurrent()
        let engine = try Qwen3ASRANEEngine(gguf: gguf, options: opts) { print("  \($0)") }
        print(String(format: "engine: %@ built in %.1f s (%d/%d programs from the aned cache), weights %.0f MB",
                     engine.backendName, CFAbsoluteTimeGetCurrent() - t0, engine.cachedPrograms, engine.programCount,
                     Double(engine.weightBytes) / 1e6))
        try engine.warmUp()
        let m = try manifest()
        for (i, u) in m.utterances.enumerated() {
            let samples = try Self.readWAV(u.wav)
            let (feat, frames) = Qwen3ASRMel.compute(samples, config: engine.config)
            // Encoder: front end and the whole tower against the float32 reference.
            let front = try engine.frontEnd(features: feat, frames: frames).map { Float($0) }
            let refFront = try floats("u\(i)_front")
            let sf = Self.stats(Array(front.prefix(refFront.count)), refFront)
            let enc = try engine.encode(features: feat, frames: frames).map { Float($0) }
            let refEnc = try floats("u\(i)_encoder")
            let se = Self.stats(enc, refEnc)
            print(String(format: "u%d front: max %.4f cos %.6f | encoder: max %.4f cos %.6f", i, sf.maxAbs, sf.cosine, se.maxAbs, se.cosine))
            let r = try engine.transcribe(samples)
            let rz = try engine.transcribe(samples, language: "Chinese")
            print("u\(i) auto: \(r.text) [\(r.language ?? "-")] | Chinese: \(rz.text)\n    ref: \(u.text_auto)\n    \(r.summary)")
            print("    ids \(r.tokenIDs.prefix(16)) / ref \(u.out_auto.prefix(16))")
            XCTAssertGreaterThan(se.cosine, 0.99)
            XCTAssertEqual(r.tokenIDs, u.out_auto)
            XCTAssertEqual(rz.tokenIDs, u.out_Chinese)
        }
    }

    /// Where the encoder's error comes from: front end on the reference mel, and encoder layer 0 alone on the
    /// reference front-end output (one layer per program). QWEN3_ASR_ENC_BITS=16 uses dense fp16 weights.
    func testEncoderLayerPrecision() throws {
        guard let ggufURL, refDir != nil else { throw XCTSkip("QWEN3_ASR_GGUF / QWEN3_ASR_REF not set") }
        let env = ProcessInfo.processInfo.environment
        var opts = Qwen3ASRANEEngine.Options(encoderLayersPerProgram: 1, cacheDirectory: cacheDir)
        if let v = env["QWEN3_ASR_ENC_BITS"].flatMap(Int.init) { opts.encoderLUTBits = v }
        let engine = try Qwen3ASRANEEngine(gguf: try GGUFFile(url: ggufURL, loadTokenizerArrays: true), options: opts)
        let u = try manifest().utterances[0]
        let samples = try Self.readWAV(u.wav)
        let (feat, frames) = Qwen3ASRMel.compute(samples, config: engine.config)
        let refFront = try floats("u0_front")
        let front = try engine.frontEnd(features: feat, frames: frames).map { Float($0) }
        let sf = Self.stats(Array(front.prefix(refFront.count)), refFront)
        var refMax: Float = 0
        for v in refFront { refMax = max(refMax, abs(v)) }
        let l0 = try engine.encoderBlock(0, input: refFront, valid: u.audio_tokens)
        let refL0 = try floats("u0_layer0")
        let s0 = Self.stats(l0, refL0)
        var l0Max: Float = 0
        for v in refL0 { l0Max = max(l0Max, abs(v)) }
        print(String(format: "enc bits %d: front max|diff| %.4f (|ref| max %.2f) cos %.6f | layer0 max|diff| %.4f (|ref| max %.2f) cos %.7f",
                     opts.encoderLUTBits, sf.maxAbs, refMax, sf.cosine, s0.maxAbs, l0Max, s0.cosine))
        // Per-token cosine of the front end, to see whether the error sits in particular chunks.
        let D = engine.config.audioHidden
        var line = "front per-token cos:"
        for t in 0..<u.audio_tokens {
            let a = Array(front[(t * D)..<((t + 1) * D)]), b = Array(refFront[(t * D)..<((t + 1) * D)])
            line += String(format: " %.4f", Self.stats(a, b).cosine)
        }
        print(line)
    }

    /// Transcribes the bake-off set (QWEN3_ASR_MER=<meta.json>) and writes `<name>.json` with the hypotheses for scoring.
    func testBakeoffSet() throws {
        let env = ProcessInfo.processInfo.environment
        guard let ggufURL, let metaPath = env["QWEN3_ASR_MER"] else { throw XCTSkip("QWEN3_ASR_GGUF / QWEN3_ASR_MER not set") }
        struct Item: Decodable { let id: String; let wav: String; let ref: String; let dur: Double }
        let metaURL = URL(fileURLWithPath: metaPath)
        let items = try JSONDecoder().decode([Item].self, from: Data(contentsOf: metaURL))
        let root = metaURL.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var opts = Qwen3ASRANEEngine.Options(cacheDirectory: cacheDir)
        if let v = env["QWEN3_ASR_DEC_BITS"].flatMap(Int.init) { opts.decoderLUTBits = v }
        if let v = env["QWEN3_ASR_ENC_BITS"].flatMap(Int.init) { opts.encoderLUTBits = v }
        if let v = env["QWEN3_ASR_HEAD_BITS"].flatMap(Int.init) { opts.headLUTBits = v }
        let language = env["QWEN3_ASR_LANGUAGE"]
        let engine = try Qwen3ASRANEEngine(gguf: try GGUFFile(url: ggufURL, loadTokenizerArrays: true), options: opts)
        try engine.warmUp()
        var hyps: [String] = []
        var audioSeconds = 0.0, busy = 0.0, decodeTok = 0, decodeSec = 0.0
        for it in items {
            let path = it.wav.hasPrefix("/") ? it.wav : root.appendingPathComponent(it.wav).path
            let samples = try Self.readWAV(path)
            let r = try engine.transcribe(samples, language: language)
            hyps.append(r.text)
            audioSeconds += Double(samples.count) / 16000
            busy += r.totalSeconds
            decodeTok += r.tokenIDs.count
            decodeSec += r.decodeSeconds
        }
        let name = env["QWEN3_ASR_OUT"] ?? "qwen3-1.7b-ane"
        let out: [String: Any] = ["name": name, "backend": engine.backendName, "rtf": busy / audioSeconds,
                                  "decode_tok_s": Double(decodeTok) / max(decodeSec, 1e-9), "hyps": hyps]
        let outURL = URL(fileURLWithPath: env["QWEN3_ASR_OUT_DIR"] ?? NSTemporaryDirectory()).appendingPathComponent("\(name).json")
        try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted]).write(to: outURL)
        print(String(format: "%@: RTF %.3f, decode %.1f tok/s -> %@", engine.backendName, busy / audioSeconds,
                     Double(decodeTok) / max(decodeSec, 1e-9), outURL.path))
    }
}

extension Qwen3ASRTests {
    /// Relative error of the LUT palettisation itself (no ANE): quantise one weight, rebuild it, compare.
    func testLUTQuantiserError() throws {
        guard let ggufURL else { throw XCTSkip("QWEN3_ASR_GGUF not set") }
        let gguf = try GGUFFile(url: ggufURL, loadTokenizerArrays: false)
        for tensor in ["a.blk.0.attn_q.weight", "a.blk.0.ffn_up.weight", "blk.0.ffn_gate.weight"] {
            let info = try gguf.tensor(tensor)
            let K = info.dimensions[0], N = info.elementCount / K
            let w = try WeightBank.floats(gguf, tensor)
            for (bits, group, iters, comp) in [(4, 8, 300, false), (4, 8, 300, true), (6, 8, 300, false), (6, 8, 300, true), (6, 4, 300, true),
                                               (8, 8, 300, false), (8, 8, 300, true), (8, 16, 300, true)] {
                let q = try LUTQuantizer.quantize(rows: N, cols: K, bits: bits, group: group, iterations: iters, companded: comp) { r0, cnt, dst in
                    for i in 0..<(cnt * K) { dst[i] = w[r0 * K + i] }
                } validate: { true }
                let E = 1 << bits
                var err: Double = 0, ref: Double = 0
                let groupBytes = group * K * bits / 8
                for g in 0..<(N / group) {
                    var bit = 0
                    for i in 0..<(group * K) {
                        var idx = 0
                        for b in 0..<bits {
                            let byte = q.indices[g * groupBytes + (bit >> 3)]
                            if byte & UInt8(1 << (bit & 7)) != 0 { idx |= 1 << b }
                            bit += 1
                        }
                        let v = Double(Float(Float16(bitPattern: q.lut[g * E + idx])))
                        let x = Double(w[g * group * K + i])
                        err += (v - x) * (v - x); ref += x * x
                    }
                }
                print(String(format: "%@ [%d x %d] LUT%d g%d it%d %@: relative RMS error %.5f", tensor, N, K, bits, group, iters, comp ? "cube-root" : "quantile", sqrt(err / ref)))
            }
        }
    }
}

extension Qwen3ASRTests {
    /// QWEN3_ASR_WAVS=a.wav:b.wav — each clip in auto and forced-Chinese mode.
    func testFiles() throws {
        let env = ProcessInfo.processInfo.environment
        guard let ggufURL, let list = env["QWEN3_ASR_WAVS"] else { throw XCTSkip("QWEN3_ASR_GGUF / QWEN3_ASR_WAVS not set") }
        let engine = try Qwen3ASRANEEngine(gguf: try GGUFFile(url: ggufURL, loadTokenizerArrays: true),
                                           options: .init(cacheDirectory: cacheDir))
        try engine.warmUp()
        for path in list.split(separator: ":").map(String.init) {
            let samples = try Self.readWAV(path)
            let a = try engine.transcribe(samples)
            let z = try engine.transcribe(samples, language: "Chinese")
            print("\((path as NSString).lastPathComponent): auto [\(a.language ?? "-")] \(a.text) | Chinese: \(z.text)\n    \(a.summary)")
        }
    }
}
