// komorebi_kernels.h — small NEON kernels used by the ANE engine's CPU side
// (FlashHead candidate scoring, MTP drafter centroid head).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Symmetric per-vector int8 quantisation: q[i] = round(x[i] / scale), scale = max|x| / 127.
void kk_quantize_i8(const float *x, int n, int8_t *q, float *scale);

/// out[r] = dot(rows[r * dim ...], q) for `nRows` consecutive int8 rows (dim % 16 == 0), int32 accumulate.
void kk_dot_i8_rows(const int8_t *rows, int dim, int nRows, const int8_t *q, int32_t *out);

/// out[i] = dot(bf16 row `rowIdx[i]` of `table` [*, dim], h) with fp32 accumulate (dim % 8 == 0).
void kk_dot_bf16_rows(const uint16_t *table, int dim, const int32_t *rowIdx, int nRows, const float *h, float *out);

/// y[r] = dot(bf16 row r of W [rows, cols], x): bf16 GEMV (cols % 8 == 0).
void kk_gemv_bf16(const uint16_t *W, int rows, int cols, const float *x, float *y);

#ifdef __cplusplus
}
#endif
