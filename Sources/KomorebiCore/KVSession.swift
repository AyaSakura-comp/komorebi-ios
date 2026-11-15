import Accelerate
import Foundation

/// One conversation's KV cache for `Gemma4ANEDirectEngine`: keys and values of the own-KV layers for positions
/// `0 ..< tokens.count`. The ANE programs do not depend on it, so one engine can serve several sessions (dictation,
/// chat) without one wiping the other's cache. A new prompt only prefills what follows its longest common prefix with
/// `tokens`; `save(to:)` / `load(from:)` keep a session across app launches.
public final class KVSession: @unchecked Sendable {
    public let capacity: Int
    let headDims: [Int]
    let k: [UnsafeMutablePointer<Float>]
    let v: [UnsafeMutablePointer<Float>]
    /// Tokens whose keys/values are in the cache, in position order.
    public internal(set) var tokens: [Int32] = []

    init(capacity: Int, headDims: [Int]) {
        self.capacity = capacity
        self.headDims = headDims
        self.k = headDims.map { hd in
            let p = UnsafeMutablePointer<Float>.allocate(capacity: capacity * hd)
            p.initialize(repeating: 0, count: capacity * hd)
            return p
        }
        self.v = headDims.map { hd in
            let p = UnsafeMutablePointer<Float>.allocate(capacity: capacity * hd)
            p.initialize(repeating: 0, count: capacity * hd)
            return p
        }
    }

    deinit {
        for p in k { p.deallocate() }
        for p in v { p.deallocate() }
    }

    /// Memory held by the cache (fp32).
    public var bytes: Int { capacity * headDims.reduce(0, +) * 2 * MemoryLayout<Float>.size }

    /// Forgets the cached tokens (the next prompt is prefilled from the start).
    public func reset() { tokens = [] }

    // MARK: - Persistence

    private static let magic: [UInt8] = Array("KKV1".utf8)

    private struct Header: Codable {
        var fingerprint: String
        var headDims: [Int]
        var count: Int
    }

    /// Writes the cached tokens and their keys/values (as fp16) to `url`, replacing it atomically.
    /// `fingerprint` names the model the cache belongs to (`Gemma4ANEDirectEngine.kvFingerprint`).
    public func save(to url: URL, fingerprint: String) throws {
        let n = tokens.count
        var data = Data()
        data.append(contentsOf: Self.magic)
        let header = try JSONEncoder().encode(Header(fingerprint: fingerprint, headDims: headDims, count: n))
        var headerLen = UInt32(header.count).littleEndian
        data.append(Data(bytes: &headerLen, count: 4))
        data.append(header)
        tokens.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        for (il, hd) in headDims.enumerated() {
            for src in [k[il], v[il]] {
                var half = [UInt16](repeating: 0, count: n * hd)
                Self.toHalf(src, &half, n * hd)
                half.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
            }
        }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: [.atomic])
    }

    /// Restores a cache written by `save(to:fingerprint:)`. Returns false (and leaves the session empty) when the file
    /// is missing, belongs to another model, or does not fit.
    @discardableResult
    public func load(from url: URL, fingerprint: String) -> Bool {
        tokens = []
        guard let data = try? Data(contentsOf: url, options: .alwaysMapped), data.count > 8,
              Array(data.prefix(4)) == Self.magic else { return false }
        let headerLen = Int(data.subdata(in: 4..<8).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian)
        guard data.count >= 8 + headerLen,
              let header = try? JSONDecoder().decode(Header.self, from: data.subdata(in: 8..<(8 + headerLen))),
              header.fingerprint == fingerprint, header.headDims == headDims, header.count <= capacity else { return false }
        let n = header.count
        var offset = 8 + headerLen
        let tokenBytes = n * MemoryLayout<Int32>.size
        let kvBytes = headDims.reduce(0) { $0 + 2 * n * $1 * 2 }
        guard data.count == offset + tokenBytes + kvBytes else { return false }
        var restored = [Int32](repeating: 0, count: n)
        restored.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst.bindMemory(to: UInt8.self), from: offset..<(offset + tokenBytes))
        }
        offset += tokenBytes
        for (il, hd) in headDims.enumerated() {
            for dst in [k[il], v[il]] {
                let count = n * hd
                data.withUnsafeBytes { raw in
                    let src = raw.baseAddress!.advanced(by: offset).assumingMemoryBound(to: UInt16.self)
                    Self.fromHalf(src, dst, count)
                }
                offset += count * 2
            }
        }
        tokens = restored
        return true
    }

    private static func toHalf(_ src: UnsafePointer<Float>, _ dst: inout [UInt16], _ count: Int) {
        guard count > 0 else { return }
        dst.withUnsafeMutableBufferPointer { d in
            var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
            var h = vImage_Buffer(data: d.baseAddress!, height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
            vImageConvert_PlanarFtoPlanar16F(&s, &h, 0)
        }
    }

    private static func fromHalf(_ src: UnsafePointer<UInt16>, _ dst: UnsafeMutablePointer<Float>, _ count: Int) {
        guard count > 0 else { return }
        var h = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        var f = vImage_Buffer(data: dst, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        vImageConvert_Planar16FtoPlanarF(&h, &f, 0)
    }
}
