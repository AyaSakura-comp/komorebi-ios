#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// Custom Metal Compute Kernels for Gemma 4 E2B (Q4_0 / Q6_K / F16 / F32)
// Designed strictly for Apple Silicon SIMD-groups (32 lanes/simdgroup):
//  - Decode (M=1): multi-row register reuse of activations
//  - Prefill (M>1): multi-token batched tile reading quantized weights once
//  - MQA Sliding-Window & Full Attention with online softmax (scale = 1.0)
//  - NaN-safe fast-math GeGLU & logit softcap + GPU argmax
// ============================================================================

constant int Q4_0_BLOCK_BYTES = 18;
constant int Q6_K_BLOCK_BYTES = 210;

// ----------------------------------------------------------------------------
// 1. Q4_0 Matrix-Vector Multiply (M = 1 decode): y[N] = W[N, K] * x[K]
// Each SIMD group (32 lanes) computes 4 output rows simultaneously, reading x
// once into registers per step and dotting against 4 rows.
// 2 lanes collaborate per Q4_0 block => 16 blocks (512 elems) per loop iteration.
// ----------------------------------------------------------------------------
kernel void q4_0_matvec_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float *       y       [[buffer(2)]],
    constant uint &      K       [[buffer(3)]],
    constant uint &      N       [[buffer(4)]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    // 4 SIMD groups per TG * 4 rows per SIMD = 16 rows per TG
    const uint row0 = (tgid.x * 4 + simd_id) * 4;
    if (row0 >= N) return;

    const uint num_blocks = K >> 5;
    const uint row_bytes = num_blocks * Q4_0_BLOCK_BYTES;

    const uint block_offset = simd_lane >> 1; // 0..15
    const uint half_sel = simd_lane & 1;      // 0: qs[0..7], 1: qs[8..15]
    const uint q_off = half_sel * 8;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    device const uchar * r0 = weights + row0 * row_bytes;
    device const uchar * r1 = (row0 + 1 < N) ? (r0 + row_bytes) : r0;
    device const uchar * r2 = (row0 + 2 < N) ? (r0 + 2 * row_bytes) : r0;
    device const uchar * r3 = (row0 + 3 < N) ? (r0 + 3 * row_bytes) : r0;

    for (uint b = block_offset; b < num_blocks; b += 16) {
        const uint k0 = b * 32 + q_off;
        // Load 8 + 8 floats of x ONCE across all 4 rows
        float4 x_lo0 = *(device const float4 *)(x + k0);
        float4 x_lo1 = *(device const float4 *)(x + k0 + 4);
        float4 x_hi0 = *(device const float4 *)(x + k0 + 16);
        float4 x_hi1 = *(device const float4 *)(x + k0 + 20);

        const uint b_byte = b * Q4_0_BLOCK_BYTES;

        // Helper macro for 1 row accumulate (8 packed bytes -> 16 weights)
        #define ACCUM_Q4_ROW(rptr, acc) { \
            float d = float(*(device const half *)(rptr + b_byte)); \
            device const uchar * qs = rptr + b_byte + 2 + q_off; \
            uchar q0 = qs[0], q1 = qs[1], q2 = qs[2], q3 = qs[3]; \
            uchar q4 = qs[4], q5 = qs[5], q6 = qs[6], q7 = qs[7]; \
            float4 w_lo0 = float4(int(q0 & 0xF) - 8, int(q1 & 0xF) - 8, int(q2 & 0xF) - 8, int(q3 & 0xF) - 8); \
            float4 w_lo1 = float4(int(q4 & 0xF) - 8, int(q5 & 0xF) - 8, int(q6 & 0xF) - 8, int(q7 & 0xF) - 8); \
            float4 w_hi0 = float4(int(q0 >> 4)  - 8, int(q1 >> 4)  - 8, int(q2 >> 4)  - 8, int(q3 >> 4)  - 8); \
            float4 w_hi1 = float4(int(q4 >> 4)  - 8, int(q5 >> 4)  - 8, int(q6 >> 4)  - 8, int(q7 >> 4)  - 8); \
            float s = dot(w_lo0, x_lo0) + dot(w_lo1, x_lo1) + dot(w_hi0, x_hi0) + dot(w_hi1, x_hi1); \
            acc = fma(s, d, acc); \
        }

        ACCUM_Q4_ROW(r0, acc0);
        if (row0 + 1 < N) { ACCUM_Q4_ROW(r1, acc1); }
        if (row0 + 2 < N) { ACCUM_Q4_ROW(r2, acc2); }
        if (row0 + 3 < N) { ACCUM_Q4_ROW(r3, acc3); }
        #undef ACCUM_Q4_ROW
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane == 0) {
        y[row0] = acc0;
        if (row0 + 1 < N) y[row0 + 1] = acc1;
        if (row0 + 2 < N) y[row0 + 2] = acc2;
        if (row0 + 3 < N) y[row0 + 3] = acc3;
    }
}

// ----------------------------------------------------------------------------
// 2. Batched Q4_0 Matrix Multiply (M > 1 prefill): Y[M, N] = X[M, K] * W[N, K]^T
// Threadgroup tiles `sx[8][512]` in half precision (8 KB SRAM) shared across all
// 4 SIMD groups (16 output rows x 8 tokens per TG), reading Q4_0 weights ONCE across 8 tokens!
// Grid: x = ceil(N / 16), y = ceil(M / 8). 4 SIMD groups/TG (128 threads/TG).
// ----------------------------------------------------------------------------
kernel void q4_0_matmul_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float *       y       [[buffer(2)]],
    constant uint &      K       [[buffer(3)]],
    constant uint &      N       [[buffer(4)]],
    constant uint &      M       [[buffer(5)]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    const uint row0 = (tgid.x * 4 + simd_id) * 4;
    const uint m0 = tgid.y * 8;
    if (m0 >= M) return;

    const uint num_blocks = K >> 5;
    const uint row_bytes = num_blocks * Q4_0_BLOCK_BYTES;
    const uint m_cnt = min(8u, M - m0);
    const bool valid_tg_row = (row0 < N);

    const uint block_offset = simd_lane >> 1;
    const uint q_off = (simd_lane & 1) * 8;

    float acc0[8] = {0,0,0,0,0,0,0,0};
    float acc1[8] = {0,0,0,0,0,0,0,0};
    float acc2[8] = {0,0,0,0,0,0,0,0};
    float acc3[8] = {0,0,0,0,0,0,0,0};

    device const uchar * r0 = valid_tg_row ? (weights + row0 * row_bytes) : weights;
    device const uchar * r1 = (row0 + 1 < N) ? (r0 + row_bytes) : r0;
    device const uchar * r2 = (row0 + 2 < N) ? (r0 + 2 * row_bytes) : r0;
    device const uchar * r3 = (row0 + 3 < N) ? (r0 + 3 * row_bytes) : r0;
    const bool has_r1 = (row0 + 1 < N);
    const bool has_r2 = (row0 + 2 < N);
    const bool has_r3 = (row0 + 3 < N);

    threadgroup half sx[8 * 512]; // 8 KB shared tile

    for (uint k_start = 0; k_start < K; k_start += 512) {
        const uint k_tile = min(512u, K - k_start);
        const uint k_tile4 = k_tile >> 2;
        const uint total_v4 = m_cnt * k_tile4;

        for (uint idx = tid; idx < total_v4; idx += 128) {
            uint lm = idx / k_tile4;
            uint lk4 = idx % k_tile4;
            float4 xv = *(device const float4 *)(x + (m0 + lm) * K + k_start + lk4 * 4);
            *(threadgroup half4 *)(sx + lm * 512 + lk4 * 4) = half4(xv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint b = (k_start >> 5) + block_offset;
        if (valid_tg_row && b < num_blocks) {
            const uint b_byte = b * Q4_0_BLOCK_BYTES;
            const uint k_in_tile = (block_offset << 5) + q_off;

            #define LOAD_Q4_HALF(rptr, d_out, vlo0, vlo1, vhi0, vhi1) \
                float d_out = float(*(device const half *)(rptr + b_byte)); \
                { \
                    device const ushort * qs = (device const ushort *)(rptr + b_byte + 2 + q_off); \
                    ushort u0 = qs[0], u1 = qs[1], u2 = qs[2], u3 = qs[3]; \
                    uchar q0 = uchar(u0 & 0xFF), q1 = uchar(u0 >> 8); \
                    uchar q2 = uchar(u1 & 0xFF), q3 = uchar(u1 >> 8); \
                    uchar q4 = uchar(u2 & 0xFF), q5 = uchar(u2 >> 8); \
                    uchar q6 = uchar(u3 & 0xFF), q7 = uchar(u3 >> 8); \
                    vlo0 = half4(int(q0&0xF)-8, int(q1&0xF)-8, int(q2&0xF)-8, int(q3&0xF)-8) * half(d_out); \
                    vlo1 = half4(int(q4&0xF)-8, int(q5&0xF)-8, int(q6&0xF)-8, int(q7&0xF)-8) * half(d_out); \
                    vhi0 = half4(int(q0>>4)-8,  int(q1>>4)-8,  int(q2>>4)-8,  int(q3>>4)-8)  * half(d_out); \
                    vhi1 = half4(int(q4>>4)-8,  int(q5>>4)-8,  int(q6>>4)-8,  int(q7>>4)-8)  * half(d_out); \
                }

            half4 w0_lo0, w0_lo1, w0_hi0, w0_hi1;
            LOAD_Q4_HALF(r0, d0, w0_lo0, w0_lo1, w0_hi0, w0_hi1);

            half4 w1_lo0 = 0, w1_lo1 = 0, w1_hi0 = 0, w1_hi1 = 0;
            if (has_r1) { LOAD_Q4_HALF(r1, d1, w1_lo0, w1_lo1, w1_hi0, w1_hi1); }
            half4 w2_lo0 = 0, w2_lo1 = 0, w2_hi0 = 0, w2_hi1 = 0;
            if (has_r2) { LOAD_Q4_HALF(r2, d2, w2_lo0, w2_lo1, w2_hi0, w2_hi1); }
            half4 w3_lo0 = 0, w3_lo1 = 0, w3_hi0 = 0, w3_hi1 = 0;
            if (has_r3) { LOAD_Q4_HALF(r3, d3, w3_lo0, w3_lo1, w3_hi0, w3_hi1); }
            #undef LOAD_Q4_HALF

            for (uint m = 0; m < m_cnt; ++m) {
                threadgroup const half * xm = sx + m * 512 + k_in_tile;
                half4 x_lo0 = *(threadgroup const half4 *)(xm);
                half4 x_lo1 = *(threadgroup const half4 *)(xm + 4);
                half4 x_hi0 = *(threadgroup const half4 *)(xm + 16);
                half4 x_hi1 = *(threadgroup const half4 *)(xm + 20);

                acc0[m] += float(dot(w0_lo0, x_lo0)) + float(dot(w0_lo1, x_lo1)) + float(dot(w0_hi0, x_hi0)) + float(dot(w0_hi1, x_hi1));
                if (has_r1) acc1[m] += float(dot(w1_lo0, x_lo0)) + float(dot(w1_lo1, x_lo1)) + float(dot(w1_hi0, x_hi0)) + float(dot(w1_hi1, x_hi1));
                if (has_r2) acc2[m] += float(dot(w2_lo0, x_lo0)) + float(dot(w2_lo1, x_lo1)) + float(dot(w2_hi0, x_hi0)) + float(dot(w2_hi1, x_hi1));
                if (has_r3) acc3[m] += float(dot(w3_lo0, x_lo0)) + float(dot(w3_lo1, x_lo1)) + float(dot(w3_hi0, x_hi0)) + float(dot(w3_hi1, x_hi1));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_tg_row) return;
    for (uint m = 0; m < m_cnt; ++m) {
        float s0 = simd_sum(acc0[m]);
        float s1 = has_r1 ? simd_sum(acc1[m]) : 0.0f;
        float s2 = has_r2 ? simd_sum(acc2[m]) : 0.0f;
        float s3 = has_r3 ? simd_sum(acc3[m]) : 0.0f;
        if (simd_lane == 0) {
            y[(m0 + m) * N + row0] = s0;
            if (has_r1) y[(m0 + m) * N + row0 + 1] = s1;
            if (has_r2) y[(m0 + m) * N + row0 + 2] = s2;
            if (has_r3) y[(m0 + m) * N + row0 + 3] = s3;
        }
    }
}

// ----------------------------------------------------------------------------
// 2b. Prefill GEMM on the SIMD-group matrix units (simdgroup_matrix 8x8 MMA):
//     Y[M, N] = X[M, K] * W[N, K]^T   (W = Q4_0 or F16, X/Y = F32, half MMA inputs, F32 accumulate)
// Threadgroup tile: 32 tokens x 64 weight rows, K step 32 (= one Q4_0 block per row).
// 128 threads = 4 SIMD groups in a 2x2 layout, each owning 16 tokens x 32 rows (2x4 8x8 accumulators).
// Grid: x = ceil(N / 64), y = ceil(M / 32).
// ----------------------------------------------------------------------------
constant constexpr uint MM_BM = 32; // tokens per threadgroup
constant constexpr uint MM_BN = 64; // weight rows per threadgroup
constant constexpr uint MM_BK = 32; // K step

template <int WTYPE> // 0 = Q4_0, 1 = F16
inline void mm_simd_body(
    device const uchar * weights,
    device const float * x,
    device float *       y,
    const uint K, const uint N, const uint M,
    threadgroup half *   sa,   // [MM_BN][MM_BK] dequantized weights
    threadgroup half *   sb,   // [MM_BM][MM_BK] activations
    threadgroup float *  sc,   // [4][64] edge staging
    const uint tid, const uint sg, const uint lane, const uint2 tgid)
{
    const uint n0 = tgid.x * MM_BN;
    const uint m0 = tgid.y * MM_BM;
    const uint sg_n = (sg & 1) * 32;
    const uint sg_m = (sg >> 1) * 16;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Weight loader: 2 threads per weight row, 16 values each.
    const uint lr = tid >> 1;
    const uint lh = tid & 1;
    const uint wrow = n0 + lr;
    const bool wvalid = wrow < N;
    // Activation loader: 4 threads per token row, 8 values each.
    const uint xm = tid >> 2;
    const uint xk = (tid & 3) * 8;
    const uint xrow = m0 + xm;
    const bool xvalid = xrow < M;

    const uint nb = K / MM_BK;
    device const uchar * wq = weights + (ulong)wrow * nb * Q4_0_BLOCK_BYTES;
    device const half *  wh = (device const half *)weights + (ulong)wrow * K;
    device const float * xr = x + (ulong)xrow * K + xk;
    threadgroup half * dstw = sa + lr * MM_BK;
    threadgroup half * dstx = sb + xm * MM_BK + xk;

    for (uint kb = 0; kb < nb; ++kb) {
        if (WTYPE == 0) {
            if (wvalid) {
                device const uchar * blk = wq + kb * Q4_0_BLOCK_BYTES;
                const half d = *(device const half *)blk;
                device const uchar * qs = blk + 2 + lh * 8;
                for (uint t = 0; t < 8; ++t) {
                    const uchar q = qs[t];
                    dstw[lh * 8 + t]      = half(int(q & 0xF) - 8) * d;
                    dstw[lh * 8 + t + 16] = half(int(q >> 4)  - 8) * d;
                }
            } else {
                for (uint t = 0; t < 8; ++t) { dstw[lh * 8 + t] = 0.0h; dstw[lh * 8 + t + 16] = 0.0h; }
            }
        } else {
            threadgroup half4 * d4 = (threadgroup half4 *)(dstw + lh * 16);
            if (wvalid) {
                device const half4 * s4 = (device const half4 *)(wh + kb * MM_BK + lh * 16);
                d4[0] = s4[0]; d4[1] = s4[1]; d4[2] = s4[2]; d4[3] = s4[3];
            } else {
                d4[0] = 0.0h; d4[1] = 0.0h; d4[2] = 0.0h; d4[3] = 0.0h;
            }
        }
        if (xvalid) {
            device const float4 * s4 = (device const float4 *)(xr + kb * MM_BK);
            *(threadgroup half4 *)(dstx)     = half4(s4[0]);
            *(threadgroup half4 *)(dstx + 4) = half4(s4[1]);
        } else {
            *(threadgroup half4 *)(dstx)     = 0.0h;
            *(threadgroup half4 *)(dstx + 4) = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < MM_BK; kk += 8) {
            simdgroup_half8x8 ma[2];
            simdgroup_half8x8 mb[4];
            for (uint i = 0; i < 2; ++i)
                simdgroup_load(ma[i], sb + (sg_m + i * 8) * MM_BK + kk, MM_BK);
            for (uint j = 0; j < 4; ++j)
                simdgroup_load(mb[j], sa + (sg_n + j * 8) * MM_BK + kk, MM_BK, ulong2(0, 0), true);
            for (uint i = 0; i < 2; ++i)
                for (uint j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float * scs = sc + sg * 64;
    for (uint i = 0; i < 2; ++i) {
        const uint mt = m0 + sg_m + i * 8;
        if (mt >= M) continue;
        for (uint j = 0; j < 4; ++j) {
            const uint nt = n0 + sg_n + j * 8;
            if (nt >= N) continue;
            if (mt + 8 <= M && nt + 8 <= N) {
                simdgroup_store(acc[i][j], y + (ulong)mt * N + nt, N);
            } else {
                simdgroup_store(acc[i][j], scs, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64; e += 32) {
                    const uint r = e >> 3, c = e & 7;
                    if (mt + r < M && nt + c < N) y[(ulong)(mt + r) * N + nt + c] = scs[e];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

kernel void q4_0_mm_simd_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float *       y       [[buffer(2)]],
    constant uint &      K       [[buffer(3)]],
    constant uint &      N       [[buffer(4)]],
    constant uint &      M       [[buffer(5)]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    threadgroup half  sa[MM_BN * MM_BK];
    threadgroup half  sb[MM_BM * MM_BK];
    threadgroup float sc[4 * 64];
    mm_simd_body<0>(weights, x, y, K, N, M, sa, sb, sc, tid, sg, lane, tgid.xy);
}

kernel void f16_mm_simd_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float *       y       [[buffer(2)]],
    constant uint &      K       [[buffer(3)]],
    constant uint &      N       [[buffer(4)]],
    constant uint &      M       [[buffer(5)]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint sg                      [[simdgroup_index_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    threadgroup half  sa[MM_BN * MM_BK];
    threadgroup half  sb[MM_BM * MM_BK];
    threadgroup float sc[4 * 64];
    mm_simd_body<1>(weights, x, y, K, N, M, sa, sb, sc, tid, sg, lane, tgid.xy);
}

// ----------------------------------------------------------------------------
// 3. Q6_K Tied LM Head Matvec / Matmul: y[M, N] = x[M, K] * W_q6k[N, K]^T
// Maps 32 SIMD lanes 1-to-1 to Q6_K's l = 0..31 within each 256-element block!
// 1 SIMD group computes 4 vocab rows for M=1 -> peak DRAM efficiency on 330MB LM head.
// Grid: x = ceil(N / 16) for 4 SIMD groups/TG.
// ----------------------------------------------------------------------------
kernel void q6_k_matvec_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float *       y       [[buffer(2)]],
    constant uint &      K       [[buffer(3)]],
    constant uint &      N       [[buffer(4)]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    const uint row0 = (tgid.x * 4 + simd_id) * 4;
    if (row0 >= N) return;

    const uint num_blocks = K >> 8;
    const uint row_bytes = num_blocks * Q6_K_BLOCK_BYTES;
    const uint l = simd_lane; // 0..31
    const uint iseg = l >> 4; // 0 or 1

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    device const uchar * r0 = weights + row0 * row_bytes;
    device const uchar * r1 = (row0 + 1 < N) ? (r0 + row_bytes) : r0;
    device const uchar * r2 = (row0 + 2 < N) ? (r0 + 2 * row_bytes) : r0;
    device const uchar * r3 = (row0 + 3 < N) ? (r0 + 3 * row_bytes) : r0;

    for (uint b = 0; b < num_blocks; ++b) {
        const uint k_base = b * 256;
        const uint b_byte = b * Q6_K_BLOCK_BYTES;

        // Preload the 8 activation elements for lane `l` ONCE across 4 rows!
        float x0 = x[k_base + l + 0];
        float x1 = x[k_base + l + 32];
        float x2 = x[k_base + l + 64];
        float x3 = x[k_base + l + 96];
        float x4 = x[k_base + 128 + l + 0];
        float x5 = x[k_base + 128 + l + 32];
        float x6 = x[k_base + 128 + l + 64];
        float x7 = x[k_base + 128 + l + 96];

        #define ACCUM_Q6_ROW(rptr, acc) { \
            device const uchar * blk = rptr + b_byte; \
            device const uchar * ql = blk; \
            device const uchar * qh = blk + 128; \
            device const char *  sc = (device const char *)(blk + 192); \
            float d = float(*(device const half *)(blk + 208)); \
            float s = 0.0f; \
            { \
                uchar qh_b = qh[l]; \
                uchar ql_a = ql[l]; \
                uchar ql_b = ql[l + 32]; \
                int q1 = int((ql_a & 0x0F) | (((qh_b >> 0) & 3) << 4)) - 32; \
                int q2 = int((ql_b & 0x0F) | (((qh_b >> 2) & 3) << 4)) - 32; \
                int q3 = int((ql_a >> 4)   | (((qh_b >> 4) & 3) << 4)) - 32; \
                int q4 = int((ql_b >> 4)   | (((qh_b >> 6) & 3) << 4)) - 32; \
                s += float(sc[iseg + 0]) * float(q1) * x0; \
                s += float(sc[iseg + 2]) * float(q2) * x1; \
                s += float(sc[iseg + 4]) * float(q3) * x2; \
                s += float(sc[iseg + 6]) * float(q4) * x3; \
            } \
            { \
                uchar qh_b = qh[32 + l]; \
                uchar ql_a = ql[64 + l]; \
                uchar ql_b = ql[64 + l + 32]; \
                int q1 = int((ql_a & 0x0F) | (((qh_b >> 0) & 3) << 4)) - 32; \
                int q2 = int((ql_b & 0x0F) | (((qh_b >> 2) & 3) << 4)) - 32; \
                int q3 = int((ql_a >> 4)   | (((qh_b >> 4) & 3) << 4)) - 32; \
                int q4 = int((ql_b >> 4)   | (((qh_b >> 6) & 3) << 4)) - 32; \
                s += float(sc[8 + iseg + 0]) * float(q1) * x4; \
                s += float(sc[8 + iseg + 2]) * float(q2) * x5; \
                s += float(sc[8 + iseg + 4]) * float(q3) * x6; \
                s += float(sc[8 + iseg + 6]) * float(q4) * x7; \
            } \
            acc = fma(s, d, acc); \
        }

        ACCUM_Q6_ROW(r0, acc0);
        if (row0 + 1 < N) { ACCUM_Q6_ROW(r1, acc1); }
        if (row0 + 2 < N) { ACCUM_Q6_ROW(r2, acc2); }
        if (row0 + 3 < N) { ACCUM_Q6_ROW(r3, acc3); }
        #undef ACCUM_Q6_ROW
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane == 0) {
        y[row0] = acc0;
        if (row0 + 1 < N) y[row0 + 1] = acc1;
        if (row0 + 2 < N) y[row0 + 2] = acc2;
        if (row0 + 3 < N) y[row0 + 3] = acc3;
    }
}

// ----------------------------------------------------------------------------
// 4. F16 Matrix Multiply (used for `per_layer_model_proj` [8960, 1536] F16)
// Grid: x = ceil(N / 4), y = M. 4 SIMD groups/TG (128 threads/TG, 1 row per SIMD).
// ----------------------------------------------------------------------------
kernel void f16_matmul_f32(
    device const half *  weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float *       y       [[buffer(2)]],
    constant uint &      K       [[buffer(3)]],
    constant uint &      N       [[buffer(4)]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_id                 [[simdgroup_index_in_threadgroup]],
    uint3 tgid                   [[threadgroup_position_in_grid]])
{
    const uint row = tgid.x * 4 + simd_id;
    const uint m = tgid.y;
    if (row >= N) return;

    device const half4 * w4 = (device const half4 *)(weights + row * K);
    device const float4 * x4 = (device const float4 *)(x + m * K);
    const uint k4 = K >> 2;

    float acc = 0.0f;
    for (uint i = simd_lane; i < k4; i += 32) {
        acc += dot(float4(w4[i]), x4[i]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0) {
        y[m * N + row] = acc;
    }
}

// ----------------------------------------------------------------------------
// 5. RMSNorm (supports both scaled weight and unscaled when use_weight == 0)
// Supports arbitrary rows (e.g. [M, D] or [M * nHeads, headDim]).
// Grid: x = num_rows, 128 threads per TG (4 SIMD groups).
// ----------------------------------------------------------------------------
kernel void rmsnorm_f32(
    device const float * x         [[buffer(0)]],
    device const float * w         [[buffer(1)]],
    device float *       y         [[buffer(2)]],
    constant uint &      D         [[buffer(3)]],
    constant uint &      w_stride  [[buffer(4)]], // 0 -> broadcast w[D], 1 -> per-slice, or use_weight==0 -> no weight
    constant float &     eps       [[buffer(5)]],
    constant uint &      use_weight[[buffer(6)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_id                   [[simdgroup_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]])
{
    device const float * row_x = x + tgid * D;
    device float * row_y = y + tgid * D;

    float ss = 0.0f;
    for (uint i = tid; i < D; i += 128) {
        float v = row_x[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    threadgroup float partial[4];
    if (simd_lane == 0) partial[simd_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_ss = partial[0] + partial[1] + partial[2] + partial[3];
    float inv_rms = rsqrt(total_ss / float(D) + eps);

    if (use_weight != 0) {
        for (uint i = tid; i < D; i += 128) {
            row_y[i] = row_x[i] * inv_rms * w[i];
        }
    } else {
        for (uint i = tid; i < D; i += 128) {
            row_y[i] = row_x[i] * inv_rms;
        }
    }
}

// ----------------------------------------------------------------------------
// 6. Project Per-Layer Inputs:
// Given `proj` [M, nLayers, 256] (output of f16_matmul), apply RMSNorm(proj_eps)
// across 256 with `proj_norm_w[256] * (1/sqrt(2))`, add `ple_raw_scaled` [M, nLayers, 256],
// and write transposed output `[nLayers, M, 256]` for fast per-layer indexing!
// Grid: x = nLayers (35), y = M. 128 threads/TG.
// ----------------------------------------------------------------------------
kernel void ple_project_combine_f32(
    device const float * proj           [[buffer(0)]], // [M, L, 256]
    device const float * proj_norm_w    [[buffer(1)]], // [256]
    device const float * ple_raw_scaled [[buffer(2)]], // [M, L, 256]
    device float *       ple_out        [[buffer(3)]], // [L, M, 256]
    constant uint &      M              [[buffer(4)]],
    constant uint &      nLayers        [[buffer(5)]],
    constant float &     proj_eps       [[buffer(6)]],
    constant float &     inv_sqrt2      [[buffer(7)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint3 tgid                          [[threadgroup_position_in_grid]])
{
    const uint il = tgid.x;
    const uint m  = tgid.y;
    if (il >= nLayers || m >= M) return;

    const uint D = 256;
    const uint src_off = (m * nLayers + il) * D;
    const uint dst_off = (il * M + m) * D;

    device const float * p = proj + src_off;
    device const float * r = ple_raw_scaled + src_off;
    device float *       o = ple_out + dst_off;

    float ss = 0.0f;
    for (uint i = tid; i < D; i += 128) {
        float v = p[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    threadgroup float partial[4];
    if (simd_lane == 0) partial[simd_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_ss = partial[0] + partial[1] + partial[2] + partial[3];
    float inv_rms = rsqrt(total_ss / float(D) + proj_eps) * inv_sqrt2;

    for (uint i = tid; i < D; i += 128) {
        o[i] = p[i] * inv_rms * proj_norm_w[i] + r[i];
    }
}

// ----------------------------------------------------------------------------
// 7. NEOX Split-Half RoPE (in-place on Q or K) + proportional `rope_freqs`
// Buffer shape: [M, nHeads, headDim].
// Grid: x = headDim/2 (128 or 256), y = nHeads, z = M.
// ----------------------------------------------------------------------------
kernel void rope_neox_f32(
    device float *       qk          [[buffer(0)]],
    device const float * rope_freqs  [[buffer(1)]], // valid when use_freqs != 0
    constant uint &      head_dim    [[buffer(2)]],
    constant uint &      n_heads     [[buffer(3)]],
    constant uint &      pos_offset  [[buffer(4)]],
    constant float &     freq_base   [[buffer(5)]],
    constant uint &      use_freqs   [[buffer(6)]],
    uint3 gid                        [[thread_position_in_grid]])
{
    const uint half_d = head_dim >> 1;
    const uint j = gid.x;
    const uint h = gid.y;
    const uint m = gid.z;
    if (j >= half_d || h >= n_heads) return;

    const float pos = float(pos_offset + m);
    const float ff = (use_freqs != 0) ? rope_freqs[j] : 1.0f;
    const float inv_freq = pow(freq_base, -2.0f * float(j) / float(head_dim)) / ff;
    const float angle = pos * inv_freq;
    const float c = cos(angle);
    const float s = sin(angle);

    const uint base = (m * n_heads + h) * head_dim;
    const float x0 = qk[base + j];
    const float x1 = qk[base + j + half_d];
    qk[base + j]          = x0 * c - x1 * s;
    qk[base + j + half_d] = x0 * s + x1 * c;
}

// ----------------------------------------------------------------------------
// 8. Write Normalized+RoPE'd K and Normalized V into FP16 KV Cache
// For sliding layer: ring slot = (pos_offset + m) % window_size
// For full layer:    slot      = (pos_offset + m)
// Grid: x = headDim, y = M.
// ----------------------------------------------------------------------------
kernel void write_kv_cache_f16(
    device const float * k_src     [[buffer(0)]],
    device const float * v_src     [[buffer(1)]],
    device half *        k_cache   [[buffer(2)]],
    device half *        v_cache   [[buffer(3)]],
    constant uint &      head_dim  [[buffer(4)]],
    constant uint &      pos_off   [[buffer(5)]],
    constant uint &      win_size  [[buffer(6)]], // >0 for sliding ring, 0 for linear full
    uint2 gid                      [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint m = gid.y;
    if (d >= head_dim) return;

    const uint pos = pos_off + m;
    const uint slot = (win_size > 0) ? (pos % win_size) : pos;
    const uint src_idx = m * head_dim + d;
    const uint dst_idx = slot * head_dim + d;
    k_cache[dst_idx] = half(k_src[src_idx]);
    v_cache[dst_idx] = half(v_src[src_idx]);
}

// ----------------------------------------------------------------------------
// 9. Zero-Barrier Causal Sliding-Window / Full MQA Online-Softmax Attention (scale = 1.0f)
// Q shape: [M, nHeads, headDim], K/V shape: [slots, headDim] FP16.
// Output O shape: [M, nHeads, headDim] F32.
// Grid: x = nHeads (8), y = M. Threadgroup size: 32 threads (1 SIMD group = 0 barriers!).
// ----------------------------------------------------------------------------
template <uint ELEMS_PER_LANE>
inline void mqa_attention_impl(
    device const float * q_vec,
    device const half *  k_cache,
    device const half *  v_cache,
    device float *       dst,
    const uint           head_dim,
    const uint           start_pos,
    const uint           q_pos,
    const uint           win_size,
    const uint           lane
) {
    float qr[ELEMS_PER_LANE];
    float acc[ELEMS_PER_LANE];
    for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
        qr[e] = q_vec[lane + (e << 5)];
        acc[e] = 0.0f;
    }

    float m_max = -1e30f;
    float l_sum = 0.0f;

    for (uint kv_pos = start_pos; kv_pos <= q_pos; ++kv_pos) {
        const uint slot = (win_size > 0) ? (kv_pos % win_size) : kv_pos;
        const uint kv_base = slot * head_dim;

        float partial = 0.0f;
        for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
            partial = fma(qr[e], float(k_cache[kv_base + lane + (e << 5)]), partial);
        }
        float score = simd_sum(partial);

        float new_max = max(m_max, score);
        float alpha = exp(m_max - new_max);
        float p = exp(score - new_max);

        l_sum = fma(l_sum, alpha, p);
        for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
            acc[e] = fma(acc[e], alpha, p * float(v_cache[kv_base + lane + (e << 5)]));
        }
        m_max = new_max;
    }

    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    for (uint e = 0; e < ELEMS_PER_LANE; ++e) {
        dst[lane + (e << 5)] = acc[e] * inv_l;
    }
}

kernel void mqa_attention_f32(
    device const float * q         [[buffer(0)]],
    device const half *  k_cache   [[buffer(1)]],
    device const half *  v_cache   [[buffer(2)]],
    device float *       out       [[buffer(3)]],
    constant uint &      head_dim  [[buffer(4)]], // 256 or 512
    constant uint &      n_heads   [[buffer(5)]],
    constant uint &      pos_off   [[buffer(6)]],
    constant uint &      win_size  [[buffer(7)]], // 512 for sliding, 0 for full causal
    uint lane                      [[thread_index_in_simdgroup]],
    uint3 tgid                     [[threadgroup_position_in_grid]])
{
    const uint h = tgid.x;
    const uint m = tgid.y;
    if (h >= n_heads) return;

    const uint q_pos = pos_off + m;
    const uint start_pos = (win_size > 0 && q_pos >= win_size) ? (q_pos - win_size + 1) : 0;
    device const float * q_vec = q + (m * n_heads + h) * head_dim;
    device float * dst = out + (m * n_heads + h) * head_dim;

    if (head_dim == 256) {
        mqa_attention_impl<8>(q_vec, k_cache, v_cache, dst, 256, start_pos, q_pos, win_size, lane);
    } else {
        mqa_attention_impl<16>(q_vec, k_cache, v_cache, dst, head_dim, start_pos, q_pos, win_size, lane);
    }
}

// ----------------------------------------------------------------------------
// 10. GeGLU (gelu_pytorch_tanh(gate) * up) — NaN-proof under Metal fast-math
// ----------------------------------------------------------------------------
kernel void geglu_tanh_f32(
    device float *       gate_inout [[buffer(0)]],
    device const float * up         [[buffer(1)]],
    constant uint &      count      [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float x = gate_inout[gid];
    float inner = clamp(0.7978845608f * (x + 0.044715f * x * x * x), -15.0f, 15.0f);
    float gelu = 0.5f * x * (1.0f + tanh(inner));
    gate_inout[gid] = gelu * up[gid];
}

// ----------------------------------------------------------------------------
// 11. Residual Add + Optional Layer Output Scale: out[i] = (a[i] + b[i]) * scale
// Also handles unscaled residual add when scale == 1.0f.
// ----------------------------------------------------------------------------
kernel void residual_add_scale_f32(
    device const float * a     [[buffer(0)]],
    device const float * b     [[buffer(1)]],
    device float *       out   [[buffer(2)]],
    constant float &     scale [[buffer(3)]],
    constant uint &      count [[buffer(4)]],
    uint gid                   [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = (a[gid] + b[gid]) * scale;
}

// ----------------------------------------------------------------------------
// 12. Softcap + Parallel Argmax (returns 1 greedy token ID without copying 1MB)
// ----------------------------------------------------------------------------
kernel void softcap_argmax_stage1_f32(
    device const float * logits     [[buffer(0)]],
    device float *       best_vals  [[buffer(1)]],
    device int *         best_ids   [[buffer(2)]],
    constant uint &      vocab_size [[buffer(3)]],
    constant float &     softcap    [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_count                   [[threadgroups_per_grid]])
{
    float my_val = -1e30f;
    int my_id = 0;

    const uint step = tg_count * 256;
    for (uint i = tgid * 256 + tid; i < vocab_size; i += step) {
        float v = logits[i];
        if (softcap > 0.0f) {
            v = softcap * tanh(clamp(v / softcap, -15.0f, 15.0f));
        }
        if (v > my_val) {
            my_val = v;
            my_id = int(i);
        }
    }

    threadgroup float s_vals[256];
    threadgroup int   s_ids[256];
    s_vals[tid] = my_val;
    s_ids[tid]  = my_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_ids[tid]  = s_ids[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        best_vals[tgid] = s_vals[0];
        best_ids[tgid]  = s_ids[0];
    }
}

kernel void softcap_argmax_stage2_f32(
    device const float * part_vals [[buffer(0)]],
    device const int *   part_ids  [[buffer(1)]],
    device int *         out_token [[buffer(2)]],
    constant uint &      n_parts   [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]])
{
    float my_val = -1e30f;
    int my_id = 0;
    for (uint i = tid; i < n_parts; i += 256) {
        float v = part_vals[i];
        if (v > my_val) {
            my_val = v;
            my_id = part_ids[i];
        }
    }
    threadgroup float s_vals[256];
    threadgroup int   s_ids[256];
    s_vals[tid] = my_val;
    s_ids[tid]  = my_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_ids[tid]  = s_ids[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out_token[0] = s_ids[0];
    }
}
