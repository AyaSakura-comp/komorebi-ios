import Foundation
import Darwin

public enum GGMLType: UInt32, Sendable {
    case f32 = 0
    case f16 = 1
    case q4_0 = 2
    case q4_1 = 3
    case q5_0 = 6
    case q5_1 = 7
    case q8_0 = 8
    case q8_1 = 9
    case q2_K = 10
    case q3_K = 11
    case q4_K = 12
    case q5_K = 13
    case q6_K = 14
    case bf16 = 30

    public var blockSize: Int {
        switch self {
        case .f32, .f16, .bf16: return 1
        case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1: return 32
        case .q2_K, .q3_K, .q4_K, .q5_K, .q6_K: return 256
        }
    }

    public var typeSize: Int {
        switch self {
        case .f32: return 4
        case .f16, .bf16: return 2
        case .q4_0: return 18
        case .q4_1: return 20
        case .q5_0: return 22
        case .q5_1: return 24
        case .q8_0: return 34
        case .q8_1: return 36
        case .q2_K: return 84
        case .q3_K: return 110
        case .q4_K: return 144
        case .q5_K: return 176
        case .q6_K: return 210
        }
    }

    public func rowByteSize(elements: Int) -> Int {
        let nb = elements / blockSize
        return nb * typeSize
    }
}

public struct GGUFTensorInfo: Sendable {
    public let name: String
    public let dimensions: [Int]
    public let type: GGMLType
    /// Offset relative to `tensorDataOffset` in file.
    public let offset: UInt64

    public var elementCount: Int {
        dimensions.reduce(1, *)
    }

    public var byteSize: Int {
        let cols = dimensions.first ?? 0
        let rows = dimensions.dropFirst().reduce(1, *)
        return type.rowByteSize(elements: cols) * rows
    }
}

public enum GGUFValue: Sendable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case float32(Float)
    case bool(Bool)
    case string(String)
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)
    case stringArray([String])
    case intArray([Int])
    case floatArray([Float])
    case boolArray([Bool])
    case arrayCountOnly(elementType: UInt32, count: Int)
}

public final class GGUFMetadata: @unchecked Sendable {
    public internal(set) var values: [String: GGUFValue] = [:]

    public func string(_ key: String) -> String? {
        if case let .string(s) = values[key] { return s }
        return nil
    }

    public func int(_ key: String) -> Int? {
        switch values[key] {
        case let .uint8(v): return Int(v)
        case let .int8(v): return Int(v)
        case let .uint16(v): return Int(v)
        case let .int16(v): return Int(v)
        case let .uint32(v): return Int(v)
        case let .int32(v): return Int(v)
        case let .uint64(v): return Int(v)
        case let .int64(v): return Int(v)
        default: return nil
        }
    }

    public func float(_ key: String) -> Float? {
        switch values[key] {
        case let .float32(v): return v
        case let .float64(v): return Float(v)
        default: return nil
        }
    }

    public func bool(_ key: String) -> Bool? {
        if case let .bool(v) = values[key] { return v }
        return nil
    }

    public func stringArray(_ key: String) -> [String]? {
        if case let .stringArray(arr) = values[key] { return arr }
        return nil
    }

    public func intArray(_ key: String) -> [Int]? {
        if case let .intArray(arr) = values[key] { return arr }
        return nil
    }

    public func floatArray(_ key: String) -> [Float]? {
        if case let .floatArray(arr) = values[key] { return arr }
        return nil
    }

    public func boolArray(_ key: String) -> [Bool]? {
        if case let .boolArray(arr) = values[key] { return arr }
        return nil
    }

    public func arrayCount(_ key: String) -> Int? {
        switch values[key] {
        case let .stringArray(arr): return arr.count
        case let .intArray(arr): return arr.count
        case let .floatArray(arr): return arr.count
        case let .boolArray(arr): return arr.count
        case let .arrayCountOnly(_, count): return count
        default: return nil
        }
    }
}

public enum GGUFError: LocalizedError {
    case cannotOpen(String)
    case invalidMagic
    case unsupportedVersion(UInt32)
    case truncatedFile
    case unsupportedType(UInt32)
    case missingTensor(String)

    public var errorDescription: String? {
        switch self {
        case .cannotOpen(let p): return "Cannot open GGUF file: \(p)"
        case .invalidMagic: return "Invalid GGUF magic header"
        case .unsupportedVersion(let v): return "Unsupported GGUF version: \(v)"
        case .truncatedFile: return "Unexpected EOF while reading GGUF"
        case .unsupportedType(let t): return "Unsupported GGUF value/tensor type: \(t)"
        case .missingTensor(let n): return "Missing GGUF tensor: \(n)"
        }
    }
}

/// Memory-mapped read-only GGUF v3 file loader.
/// Keeps huge embedding tables (`per_layer_token_embd`, ~1.93 GB) un-faulted in virtual address space
/// until individual token rows are requested.
public final class GGUFFile: @unchecked Sendable {
    public let url: URL
    public let fileSize: Int
    public let basePointer: UnsafeRawPointer
    public let metadata: GGUFMetadata
    public let tensors: [String: GGUFTensorInfo]
    public let tensorDataOffset: Int

    private let fd: Int32

    public init(url: URL, loadTokenizerArrays: Bool = true) throws {
        self.url = url
        let path = url.path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw GGUFError.cannotOpen("\(path) (open: \(String(cString: strerror(errno))))") }

        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size > 24 else {
            close(fd)
            throw GGUFError.truncatedFile
        }
        let size = Int(st.st_size)

        guard let mapped = mmap(nil, size, PROT_READ, MAP_PRIVATE | MAP_FILE, fd, 0),
              mapped != MAP_FAILED else {
            let reason = String(cString: strerror(errno))
            close(fd)
            throw GGUFError.cannotOpen("\(path) (mmap \(size / 1_000_000) MB: \(reason))")
        }

        // Hint sequential scan ONLY during header parse; random for lazy PLE rows.
        madvise(mapped, min(size, 24 * 1024 * 1024), MADV_SEQUENTIAL)

        self.fd = fd
        self.fileSize = size
        self.basePointer = UnsafeRawPointer(mapped)
        self.metadata = GGUFMetadata()

        var cursor = BinaryCursor(base: self.basePointer, count: size)
        let magic = try cursor.readU32()
        guard magic == 0x46554747 else { // "GGUF" little-endian
            munmap(mapped, size)
            close(fd)
            throw GGUFError.invalidMagic
        }
        let version = try cursor.readU32()
        guard version == 2 || version == 3 else {
            munmap(mapped, size)
            close(fd)
            throw GGUFError.unsupportedVersion(version)
        }

        let tensorCount = Int(try cursor.readU64())
        let kvCount = Int(try cursor.readU64())

        let tokenizerKeysNeeded: Set<String> = [
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.scores",
        ]

        for _ in 0..<kvCount {
            let key = try cursor.readString()
            let vtype = try cursor.readU32()
            let decodeLarge = loadTokenizerArrays || !tokenizerKeysNeeded.contains(key)
            let val = try cursor.readValue(type: vtype, decodeLargeArrays: decodeLarge)
            self.metadata.values[key] = val
        }

        var map = [String: GGUFTensorInfo](minimumCapacity: tensorCount)
        for _ in 0..<tensorCount {
            let name = try cursor.readString()
            let ndims = Int(try cursor.readU32())
            var dims = [Int]()
            dims.reserveCapacity(ndims)
            for _ in 0..<ndims {
                dims.append(Int(try cursor.readU64()))
            }
            let rawType = try cursor.readU32()
            guard let ggmlType = GGMLType(rawValue: rawType) else {
                munmap(mapped, size)
                close(fd)
                throw GGUFError.unsupportedType(rawType)
            }
            let offset = try cursor.readU64()
            map[name] = GGUFTensorInfo(name: name, dimensions: dims, type: ggmlType, offset: offset)
        }

        let align = self.metadata.int("general.alignment") ?? 32
        let aligned = ((cursor.offset + align - 1) / align) * align
        self.tensorDataOffset = aligned
        self.tensors = map

        // Switch PLE & weight region back to normal/random to avoid eager readahead of 1.93GB PLE.
        madvise(mapped, size, MADV_RANDOM)
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: basePointer), fileSize)
        close(fd)
    }

    public func tensor(_ name: String) throws -> GGUFTensorInfo {
        guard let info = tensors[name] else {
            throw GGUFError.missingTensor(name)
        }
        return info
    }

    public func pointer(for tensor: GGUFTensorInfo) -> UnsafeRawPointer {
        basePointer.advanced(by: tensorDataOffset + Int(tensor.offset))
    }

    /// Reads an entire F32 tensor into a Swift `[Float]` array.
    public func readFloat32Tensor(_ name: String) throws -> [Float] {
        let info = try tensor(name)
        let n = info.elementCount
        let ptr = pointer(for: info)
        var out = [Float](repeating: 0, count: n)
        switch info.type {
        case .f32:
            ptr.withMemoryRebound(to: Float.self, capacity: n) { src in
                out.withUnsafeMutableBufferPointer { dst in
                    _ = memcpy(dst.baseAddress!, src, n * MemoryLayout<Float>.stride)
                }
            }
        case .f16:
            ptr.withMemoryRebound(to: UInt16.self, capacity: n) { src in
                for i in 0..<n {
                    out[i] = Float(Float16(bitPattern: src[i]))
                }
            }
        default:
            throw GGUFError.unsupportedType(info.type.rawValue)
        }
        return out
    }

    /// Dequantizes a single row `rowIndex` of a Q6_K tensor directly from the mmap without touching other rows.
    public func dequantizeQ6KRow(tensor: GGUFTensorInfo, rowIndex: Int, into dst: UnsafeMutablePointer<Float>, count: Int) {
        let rowBytes = tensor.type.rowByteSize(elements: tensor.dimensions[0])
        let rowPtr = pointer(for: tensor).advanced(by: rowIndex * rowBytes)
        GGUFQuantization.dequantizeQ6KRow(src: rowPtr, dst: dst, elements: count)
    }
}

public enum GGUFQuantization {
    /// Dequantizes a Q6_K row (blocks of 256 elements, 210 bytes per block).
    @inline(__always)
    public static func dequantizeQ6KRow(src: UnsafeRawPointer, dst: UnsafeMutablePointer<Float>, elements: Int) {
        let numBlocks = elements / 256
        var blockPtr = src
        var outIdx = 0
        for _ in 0..<numBlocks {
            let ql = blockPtr.assumingMemoryBound(to: UInt8.self)
            let qh = blockPtr.advanced(by: 128).assumingMemoryBound(to: UInt8.self)
            let sc = blockPtr.advanced(by: 192).assumingMemoryBound(to: Int8.self)
            let dBits = blockPtr.advanced(by: 208).loadUnaligned(as: UInt16.self)
            let d = Float(Float16(bitPattern: dBits))

            for n in 0..<2 {
                let pQL = ql.advanced(by: n * 64)
                let pQH = qh.advanced(by: n * 32)
                let pSC = sc.advanced(by: n * 8)
                let baseOut = outIdx + n * 128
                for l in 0..<32 {
                    let iseg = l >> 4
                    let qhByte = Int32(pQH[l])
                    let ql0 = Int32(pQL[l])
                    let ql1 = Int32(pQL[l + 32])

                    let q1 = (ql0 & 0x0F) | (((qhByte >> 0) & 0x03) << 4) - 32
                    let q2 = (ql1 & 0x0F) | (((qhByte >> 2) & 0x03) << 4) - 32
                    let q3 = (ql0 >> 4)   | (((qhByte >> 4) & 0x03) << 4) - 32
                    let q4 = (ql1 >> 4)   | (((qhByte >> 6) & 0x03) << 4) - 32

                    dst[baseOut + l + 0]  = d * Float(pSC[iseg + 0]) * Float(q1)
                    dst[baseOut + l + 32] = d * Float(pSC[iseg + 2]) * Float(q2)
                    dst[baseOut + l + 64] = d * Float(pSC[iseg + 4]) * Float(q3)
                    dst[baseOut + l + 96] = d * Float(pSC[iseg + 6]) * Float(q4)
                }
            }
            blockPtr = blockPtr.advanced(by: 210)
            outIdx += 256
        }
    }

    /// Dequantizes a Q4_0 row (blocks of 32 elements, 18 bytes per block).
    @inline(__always)
    public static func dequantizeQ4_0Row(src: UnsafeRawPointer, dst: UnsafeMutablePointer<Float>, elements: Int) {
        let numBlocks = elements / 32
        var blockPtr = src
        var outIdx = 0
        for _ in 0..<numBlocks {
            let dBits = blockPtr.loadUnaligned(as: UInt16.self)
            let d = Float(Float16(bitPattern: dBits))
            let qs = blockPtr.advanced(by: 2).assumingMemoryBound(to: UInt8.self)
            for j in 0..<16 {
                let byte = qs[j]
                let x0 = Int32(byte & 0x0F) - 8
                let x1 = Int32(byte >> 4) - 8
                dst[outIdx + j] = Float(x0) * d
                dst[outIdx + j + 16] = Float(x1) * d
            }
            blockPtr = blockPtr.advanced(by: 18)
            outIdx += 32
        }
    }
}

private struct BinaryCursor {
    let base: UnsafeRawPointer
    let count: Int
    var offset: Int = 0

    mutating func ensure(_ bytes: Int) throws {
        if offset + bytes > count { throw GGUFError.truncatedFile }
    }

    mutating func readU8() throws -> UInt8 {
        try ensure(1)
        let v = base.loadUnaligned(fromByteOffset: offset, as: UInt8.self)
        offset += 1
        return v
    }

    mutating func readI8() throws -> Int8 {
        Int8(bitPattern: try readU8())
    }

    mutating func readU16() throws -> UInt16 {
        try ensure(2)
        let v = base.loadUnaligned(fromByteOffset: offset, as: UInt16.self)
        offset += 2
        return v
    }

    mutating func readI16() throws -> Int16 {
        Int16(bitPattern: try readU16())
    }

    mutating func readU32() throws -> UInt32 {
        try ensure(4)
        let v = base.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        offset += 4
        return v
    }

    mutating func readI32() throws -> Int32 {
        Int32(bitPattern: try readU32())
    }

    mutating func readU64() throws -> UInt64 {
        try ensure(8)
        let v = base.loadUnaligned(fromByteOffset: offset, as: UInt64.self)
        offset += 8
        return v
    }

    mutating func readI64() throws -> Int64 {
        Int64(bitPattern: try readU64())
    }

    mutating func readF32() throws -> Float {
        Float(bitPattern: try readU32())
    }

    mutating func readF64() throws -> Double {
        Double(bitPattern: try readU64())
    }

    mutating func readString() throws -> String {
        let len = Int(try readU64())
        try ensure(len)
        let ptr = base.advanced(by: offset).assumingMemoryBound(to: UInt8.self)
        let buf = UnsafeBufferPointer(start: ptr, count: len)
        let str = String(decoding: buf, as: UTF8.self)
        offset += len
        return str
    }

    mutating func skip(_ n: Int) throws {
        try ensure(n)
        offset += n
    }

    mutating func readValue(type: UInt32, decodeLargeArrays: Bool) throws -> GGUFValue {
        switch type {
        case 0: return .uint8(try readU8())
        case 1: return .int8(try readI8())
        case 2: return .uint16(try readU16())
        case 3: return .int16(try readI16())
        case 4: return .uint32(try readU32())
        case 5: return .int32(try readI32())
        case 6: return .float32(try readF32())
        case 7: return .bool(try readU8() != 0)
        case 8: return .string(try readString())
        case 9:
            let elemType = try readU32()
            let len = Int(try readU64())
            if !decodeLargeArrays && len > 128 {
                if elemType == 8 {
                    for _ in 0..<len {
                        let slen = Int(try readU64())
                        try skip(slen)
                    }
                } else {
                    let elemSize = scalarSize(elemType)
                    guard elemSize > 0 else { throw GGUFError.unsupportedType(elemType) }
                    try skip(len * elemSize)
                }
                return .arrayCountOnly(elementType: elemType, count: len)
            }
            switch elemType {
            case 8:
                var arr = [String]()
                arr.reserveCapacity(len)
                for _ in 0..<len { arr.append(try readString()) }
                return .stringArray(arr)
            case 7:
                var arr = [Bool]()
                arr.reserveCapacity(len)
                for _ in 0..<len { arr.append(try readU8() != 0) }
                return .boolArray(arr)
            case 4, 5, 10, 11, 2, 3, 0, 1:
                var arr = [Int]()
                arr.reserveCapacity(len)
                for _ in 0..<len {
                    let v: Int
                    switch elemType {
                    case 0: v = Int(try readU8())
                    case 1: v = Int(try readI8())
                    case 2: v = Int(try readU16())
                    case 3: v = Int(try readI16())
                    case 4: v = Int(try readU32())
                    case 5: v = Int(try readI32())
                    case 10: v = Int(try readU64())
                    case 11: v = Int(try readI64())
                    default: v = 0
                    }
                    arr.append(v)
                }
                return .intArray(arr)
            case 6, 12:
                var arr = [Float]()
                arr.reserveCapacity(len)
                for _ in 0..<len {
                    arr.append(elemType == 6 ? try readF32() : Float(try readF64()))
                }
                return .floatArray(arr)
            default:
                let elemSize = scalarSize(elemType)
                guard elemSize > 0 else { throw GGUFError.unsupportedType(elemType) }
                try skip(len * elemSize)
                return .arrayCountOnly(elementType: elemType, count: len)
            }
        case 10: return .uint64(try readU64())
        case 11: return .int64(try readI64())
        case 12: return .float64(try readF64())
        default:
            throw GGUFError.unsupportedType(type)
        }
    }

    private func scalarSize(_ type: UInt32) -> Int {
        switch type {
        case 0, 1, 7: return 1
        case 2, 3: return 2
        case 4, 5, 6: return 4
        case 10, 11, 12: return 8
        default: return 0
        }
    }
}
