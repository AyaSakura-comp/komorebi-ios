import Foundation
import SherpaOnnx

/// Silero VAD v6.2 turn detector matching Komorebi `Turns` parameters:
///  - 512-sample frame hop (32 ms @ 16 kHz)
///  - threshold = 0.50, minSpeech = 0.25s, minSilence = 0.50s, maxSpeech = 30.0s
///  - 550 ms rolling pre-roll for live partial speech capture every 250 ms
public final class SileroTurnsVAD: @unchecked Sendable {
    public static let windowSamples = 512
    public static let sampleRate = 16_000
    public static let preRollSamples = 8_800       // 550 ms
    public static let partialMinSamples = 6_400    // 400 ms before emitting first partial
    public static let partialIntervalSamples = 4_000 // 250 ms hop between progressive partials

    public enum Event: Sendable {
        case speechStarted
        case partialWindow([Float])
        case finalizedSegment([Float])
    }

    private let vad: SherpaOnnxVoiceActivityDetectorWrapper
    private var remainder: [Float] = []
    private var rollingPreBuffer: [Float] = []
    private var activeSpeechBuffer: [Float] = []
    private var samplesSinceLastPartial = 0
    private var wasSpeechDetected = false

    public init(modelURL: URL, numThreads: Int = 1) {
        let sileroCfg = sherpaOnnxSileroVadModelConfig(
            model: modelURL.path,
            threshold: 0.5,
            minSilenceDuration: 0.5,
            minSpeechDuration: 0.25,
            windowSize: Self.windowSamples,
            maxSpeechDuration: 30.0
        )
        var vadCfg = sherpaOnnxVadModelConfig(
            sileroVad: sileroCfg,
            sampleRate: Int32(Self.sampleRate),
            numThreads: numThreads,
            provider: "cpu",
            debug: 0
        )
        self.vad = SherpaOnnxVoiceActivityDetectorWrapper(config: &vadCfg, buffer_size_in_seconds: 35.0)
    }

    public func reset() {
        vad.clear()
        remainder.removeAll(keepingCapacity: true)
        rollingPreBuffer.removeAll(keepingCapacity: true)
        activeSpeechBuffer.removeAll(keepingCapacity: true)
        samplesSinceLastPartial = 0
        wasSpeechDetected = false
    }

    /// Feeds arbitrary-length 16 kHz mono Float32 PCM chunks and emits partial & finalized speech turn events.
    public func push(samples: [Float]) -> [Event] {
        guard !samples.isEmpty else { return [] }
        remainder.append(contentsOf: samples)

        var events: [Event] = []
        var offset = 0

        while offset + Self.windowSamples <= remainder.count {
            let frame = Array(remainder[offset..<(offset + Self.windowSamples)])
            offset += Self.windowSamples

            let prevSpeaking = wasSpeechDetected
            vad.acceptWaveform(samples: frame)
            let nowSpeaking = vad.isSpeechDetected()

            if !prevSpeaking && nowSpeaking {
                wasSpeechDetected = true
                activeSpeechBuffer = rollingPreBuffer
                activeSpeechBuffer.append(contentsOf: frame)
                samplesSinceLastPartial = 0
                events.append(.speechStarted)
            } else if nowSpeaking {
                activeSpeechBuffer.append(contentsOf: frame)
                if activeSpeechBuffer.count > Self.sampleRate * 30 {
                    activeSpeechBuffer.removeFirst(activeSpeechBuffer.count - Self.sampleRate * 30)
                }
                samplesSinceLastPartial += Self.windowSamples
                if activeSpeechBuffer.count >= Self.partialMinSamples &&
                    samplesSinceLastPartial >= Self.partialIntervalSamples {
                    samplesSinceLastPartial = 0
                    events.append(.partialWindow(activeSpeechBuffer))
                }
            }

            if !vad.isEmpty() {
                while !vad.isEmpty() {
                    let seg = vad.front()
                    let pcm = seg.samples
                    vad.pop()
                    if !pcm.isEmpty {
                        events.append(.finalizedSegment(pcm))
                    }
                }
                wasSpeechDetected = false
                activeSpeechBuffer.removeAll(keepingCapacity: true)
                samplesSinceLastPartial = 0
            } else if !nowSpeaking {
                rollingPreBuffer.append(contentsOf: frame)
                if rollingPreBuffer.count > Self.preRollSamples {
                    rollingPreBuffer.removeFirst(rollingPreBuffer.count - Self.preRollSamples)
                }
                if prevSpeaking {
                    // Hangover between speech dropping and segment emitting: still keep accumulating frame.
                    activeSpeechBuffer.append(contentsOf: frame)
                }
            }
        }

        if offset > 0 {
            remainder.removeFirst(offset)
        }
        return events
    }

    /// Flushes any trailing speech in progress when user taps Stop or Finish.
    public func flush() -> [Event] {
        var events: [Event] = []
        if !remainder.isEmpty {
            let pad = Self.windowSamples - remainder.count
            if pad > 0 {
                events.append(contentsOf: push(samples: [Float](repeating: 0, count: pad)))
            }
        }
        vad.flush()
        while !vad.isEmpty() {
            let seg = vad.front()
            let pcm = seg.samples
            vad.pop()
            if !pcm.isEmpty {
                events.append(.finalizedSegment(pcm))
            }
        }
        wasSpeechDetected = false
        activeSpeechBuffer.removeAll(keepingCapacity: true)
        rollingPreBuffer.removeAll(keepingCapacity: true)
        samplesSinceLastPartial = 0
        return events
    }
}
