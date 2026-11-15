// komorebi_kernels.c — see komorebi_kernels.h
#include "komorebi_kernels.h"
#include <math.h>
#include <arm_neon.h>

void kk_quantize_i8(const float *x, int n, int8_t *q, float *scale) {
    float mx = 0.f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    float s = mx > 0.f ? mx / 127.f : 1.f;
    float inv = 1.f / s;
    for (int i = 0; i < n; i++) {
        float v = x[i] * inv;
        int r = (int)lrintf(v);
        q[i] = (int8_t)(r > 127 ? 127 : (r < -127 ? -127 : r));
    }
    *scale = s;
}

__attribute__((target("dotprod")))
void kk_dot_i8_rows(const int8_t *rows, int dim, int nRows, const int8_t *q, int32_t *out) {
    for (int r = 0; r < nRows; r++) {
        const int8_t *w = rows + (long)r * dim;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int d = 0;
        for (; d + 32 <= dim; d += 32) {
            a0 = vdotq_s32(a0, vld1q_s8(w + d), vld1q_s8(q + d));
            a1 = vdotq_s32(a1, vld1q_s8(w + d + 16), vld1q_s8(q + d + 16));
        }
        for (; d + 16 <= dim; d += 16) a0 = vdotq_s32(a0, vld1q_s8(w + d), vld1q_s8(q + d));
        out[r] = vaddvq_s32(vaddq_s32(a0, a1));
    }
}

static inline float32x4_t bf16_lo(uint16x8_t v) { return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16)); }
static inline float32x4_t bf16_hi(uint16x8_t v) { return vreinterpretq_f32_u32(vshll_high_n_u16(v, 16)); }

void kk_dot_bf16_rows(const uint16_t *table, int dim, const int32_t *rowIdx, int nRows, const float *h, float *out) {
    for (int i = 0; i < nRows; i++) {
        const uint16_t *w = table + (long)rowIdx[i] * dim;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        for (int d = 0; d < dim; d += 8) {
            uint16x8_t v = vld1q_u16(w + d);
            a0 = vfmaq_f32(a0, bf16_lo(v), vld1q_f32(h + d));
            a1 = vfmaq_f32(a1, bf16_hi(v), vld1q_f32(h + d + 4));
        }
        out[i] = vaddvq_f32(vaddq_f32(a0, a1));
    }
}

void kk_gemv_bf16(const uint16_t *W, int rows, int cols, const float *x, float *y) {
    for (int r = 0; r < rows; r++) {
        const uint16_t *w = W + (long)r * cols;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        for (int d = 0; d < cols; d += 8) {
            uint16x8_t v = vld1q_u16(w + d);
            a0 = vfmaq_f32(a0, bf16_lo(v), vld1q_f32(x + d));
            a1 = vfmaq_f32(a1, bf16_hi(v), vld1q_f32(x + d + 4));
        }
        y[r] = vaddvq_f32(vaddq_f32(a0, a1));
    }
}
