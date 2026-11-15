import Foundation

/// Tokenizer for Gemma 4 E2B matching GGUF `tokenizer.ggml.model == "gemma4"` (`LLAMA_VOCAB_PRE_TYPE_GEMMA4`).
/// Implements greedy control/user-defined special token splitting, `" " -> "▁"` normalization,
/// newline-boundary pre-splitting, rank-ordered BPE merges, and `<0xXX>` byte fallback.
public final class Gemma4Tokenizer: @unchecked Sendable {
    public let vocab: [String]
    public let tokenTypes: [Int]
    public let bosTokenID: Int32
    public let eosTokenID: Int32
    public let turnEndTokenID: Int32

    private let tokenToID: [String: Int32]
    private let byteTokens: [Int32] // 256 entries for byte 0...255
    private let specialTokensSorted: [(piece: String, id: Int32)]
    private let specialTokenSet: Set<Int32>
    private let pairRanks: [UInt64: (rank: Int32, mergedID: Int32)]

    public init(metadata: GGUFMetadata) throws {
        guard let tokens = metadata.stringArray("tokenizer.ggml.tokens") else {
            throw GGUFError.missingTensor("tokenizer.ggml.tokens")
        }
        self.vocab = tokens
        let types = metadata.intArray("tokenizer.ggml.token_type") ?? [Int](repeating: 1, count: tokens.count)
        self.tokenTypes = types

        var t2id = [String: Int32](minimumCapacity: tokens.count)
        var bytes = [Int32](repeating: Int32(metadata.int("tokenizer.ggml.unknown_token_id") ?? 3), count: 256)
        var specials: [(String, Int32)] = []
        var specialSet = Set<Int32>()

        for i in 0..<tokens.count {
            let tok = tokens[i]
            let id = Int32(i)
            t2id[tok] = id

            let ttype = i < types.count ? types[i] : 1
            if ttype == 6 && tok.count == 6 && tok.hasPrefix("<0x") && tok.hasSuffix(">") {
                let hex = tok.dropFirst(3).dropLast(1)
                if let b = UInt8(hex, radix: 16) {
                    bytes[Int(b)] = id
                }
            } else if ttype == 3 || ttype == 4 {
                specials.append((tok, id))
                specialSet.insert(id)
            }
        }
        // Longest-first matching ensures e.g. "<|channel>" matches before partial substrings.
        specials.sort {
            if $0.0.count != $1.0.count { return $0.0.count > $1.0.count }
            return $0.1 < $1.1
        }

        self.tokenToID = t2id
        self.byteTokens = bytes
        self.specialTokensSorted = specials
        self.specialTokenSet = specialSet

        self.bosTokenID = Int32(metadata.int("tokenizer.ggml.bos_token_id") ?? 2)
        self.eosTokenID = Int32(metadata.int("tokenizer.ggml.eos_token_id") ?? 1)
        self.turnEndTokenID = t2id["<turn|>"] ?? 106

        let merges = metadata.stringArray("tokenizer.ggml.merges") ?? []
        var ranks = [UInt64: (rank: Int32, mergedID: Int32)]()
        ranks.reserveCapacity(merges.count)

        for (rankIdx, mergeStr) in merges.enumerated() {
            guard let sp = mergeStr.firstIndex(of: " ") else { continue }
            let left = String(mergeStr[..<sp])
            let right = String(mergeStr[mergeStr.index(after: sp)...])
            guard let lID = t2id[left], let rID = t2id[right],
                  let mID = t2id[left + right] else { continue }
            let key = Self.pairKey(lID, rID)
            if ranks[key] == nil {
                ranks[key] = (Int32(rankIdx), mID)
            }
        }
        self.pairRanks = ranks
    }

    @inline(__always)
    private static func pairKey(_ a: Int32, _ b: Int32) -> UInt64 {
        (UInt64(UInt32(bitPattern: a)) << 32) | UInt64(UInt32(bitPattern: b))
    }

    /// Formats messages using the canonical Gemma 4 chat template with thinking disabled.
    public func formatChatPrompt(turns: [ChatTurn], addGenerationPrompt: Bool = true) -> String {
        var out = ""
        var idx = 0
        if let first = turns.first, first.role == .system {
            out += "<|turn>system\n"
            out += first.content.trimmingCharacters(in: .whitespacesAndNewlines)
            out += "<turn|>\n"
            idx = 1
        }
        while idx < turns.count {
            let t = turns[idx]
            let roleName = t.role == .assistant ? "model" : t.role.rawValue
            out += "<|turn>\(roleName)\n"
            out += t.content.trimmingCharacters(in: .whitespacesAndNewlines)
            out += "<turn|>\n"
            idx += 1
        }
        if addGenerationPrompt {
            out += "<|turn>model\n"
        }
        return out
    }

    public func encodeChat(turns: [ChatTurn], addGenerationPrompt: Bool = true) -> [Int32] {
        encode(formatChatPrompt(turns: turns, addGenerationPrompt: addGenerationPrompt), addBOS: true)
    }

    public func encode(_ text: String, addBOS: Bool = false) -> [Int32] {
        var ids: [Int32] = []
        if addBOS {
            ids.append(bosTokenID)
        }
        guard !text.isEmpty else { return ids }

        var cursor = text.startIndex
        while cursor < text.endIndex {
            // Find earliest matching special token starting at or after `cursor`.
            var bestStart = text.endIndex
            var bestMatch: (end: String.Index, id: Int32)?

            for (spStr, spID) in specialTokensSorted {
                if let range = text.range(of: spStr, range: cursor..<text.endIndex) {
                    if range.lowerBound < bestStart ||
                        (range.lowerBound == bestStart && bestMatch != nil && range.upperBound > bestMatch!.end) {
                        bestStart = range.lowerBound
                        bestMatch = (range.upperBound, spID)
                    }
                    if bestStart == cursor { break }
                }
            }

            if bestStart > cursor {
                let plain = String(text[cursor..<bestStart])
                encodePlainSegment(plain, into: &ids)
            }

            if let match = bestMatch {
                ids.append(match.id)
                cursor = match.end
            } else {
                break
            }
        }
        return ids
    }

    private func encodePlainSegment(_ segment: String, into out: inout [Int32]) {
        // Gemma4 SPM-style BPE: convert spaces to U+2581, split on newline boundaries.
        let normalized = segment.replacingOccurrences(of: " ", with: "\u{2581}")
        var pieceStart = normalized.startIndex
        var isNewlineRun = normalized[pieceStart] == "\n"

        var i = normalized.index(after: pieceStart)
        while i < normalized.endIndex {
            let curIsNL = normalized[i] == "\n"
            if curIsNL != isNewlineRun {
                bpeEncodePiece(String(normalized[pieceStart..<i]), into: &out)
                pieceStart = i
                isNewlineRun = curIsNL
            }
            i = normalized.index(after: i)
        }
        if pieceStart < normalized.endIndex {
            bpeEncodePiece(String(normalized[pieceStart..<normalized.endIndex]), into: &out)
        }
    }

    private func bpeEncodePiece(_ piece: String, into out: inout [Int32]) {
        guard !piece.isEmpty else { return }

        // Fast exact match (e.g., "\n", common single CJK tokens, spaces).
        if let exact = tokenToID[piece] {
            out.append(exact)
            return
        }

        // Initial segmentation: each character -> vocab ID or UTF-8 `<0xXX>` byte tokens.
        var symbols: [Int32] = []
        symbols.reserveCapacity(piece.utf8.count)
        for ch in piece {
            let s = String(ch)
            if let id = tokenToID[s] {
                symbols.append(id)
            } else {
                for b in s.utf8 {
                    symbols.append(byteTokens[Int(b)])
                }
            }
        }

        guard symbols.count >= 2 else {
            out.append(contentsOf: symbols)
            return
        }

        while symbols.count >= 2 {
            var bestRank = Int32.max
            var bestIdx = -1
            var bestMerged: Int32 = 0

            for idx in 0..<(symbols.count - 1) {
                let key = Self.pairKey(symbols[idx], symbols[idx + 1])
                if let entry = pairRanks[key], entry.rank < bestRank {
                    bestRank = entry.rank
                    bestIdx = idx
                    bestMerged = entry.mergedID
                }
            }

            if bestIdx < 0 { break }
            symbols[bestIdx] = bestMerged
            symbols.remove(at: bestIdx + 1)
        }

        out.append(contentsOf: symbols)
    }

    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        var result = ""
        var pendingBytes: [UInt8] = []

        func flushBytes() {
            guard !pendingBytes.isEmpty else { return }
            result += String(decoding: pendingBytes, as: UTF8.self)
            pendingBytes.removeAll(keepingCapacity: true)
        }

        for rawID in ids {
            let id = Int(rawID)
            guard id >= 0 && id < vocab.count else { continue }
            if skipSpecialTokens && specialTokenSet.contains(rawID) { continue }

            let tok = vocab[id]
            let ttype = id < tokenTypes.count ? tokenTypes[id] : 1
            if ttype == 6 && tok.count == 6 && tok.hasPrefix("<0x") && tok.hasSuffix(">") {
                let hex = tok.dropFirst(3).dropLast(1)
                if let b = UInt8(hex, radix: 16) {
                    pendingBytes.append(b)
                    continue
                }
            }

            flushBytes()
            result += tok.replacingOccurrences(of: "\u{2581}", with: " ")
        }
        flushBytes()
        return result
    }
}
