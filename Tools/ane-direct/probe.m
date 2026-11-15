// probe.m — direct-ANE capability / roofline probe for Gemma 4 E2B-shaped layers.
// Build: see Makefile. Run: ./probe
#import <Foundation/Foundation.h>
#import "ane_rt.h"
#include <mach/mach_time.h>
#include <math.h>

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}
static uint32_t rng = 12345;
static float frand(void) { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) & 0xFFFF) / 65535.0f * 2.f - 1.f; }

static NSString *HDR = @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3500.14.1\"}, {\"coremlc-version\", \"3500.32.1\"}})]\n{\n";
static NSString *BLOB(uint64_t off) { return [NSString stringWithFormat:@"BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%llu))", off]; }

typedef enum { WF_FP16 = 0, WF_LUT4 = 1, WF_BLK4 = 2, WF_LUT4_KBLK = 3, WF_LUT4_ROW = 4, WF_BLK8 = 5, WF_BLK4U = 6, WF_LUT8_ROW = 7, WF_LUT6_ROW = 8 } WFmt;
static const char *wfName[] = {"fp16", "lut4(g32)", "int4-blk32", "lut4-kblk32", "lut4-row", "int8-blk32", "uint4-blk32+off", "lut8-row", "lut6-row"};
static int gFormats[16]; static int gNumFormats = 0; static int gLutG = 32, gLutBits = 4;

/// Emits a weight [N, K] declaration named `name` into `m`, filling `deq` (optional) with the float values.
static void emitWeight(NSMutableString *m, ANEBlobBuilder *b, NSString *name, int N, int K, WFmt f, float *deq, float scaleMag) {
    if (f == WF_FP16) {
        void *p; uint64_t off = [b appendZeroed:(size_t)N * K * 2 dtype:1 payload:&p];
        _Float16 *h = p;
        for (size_t i = 0; i < (size_t)N * K; i++) { float v = frand() * scaleMag; h[i] = (_Float16)v; if (deq) deq[i] = (float)h[i]; }
        [m appendFormat:@"        tensor<fp16, [%d, %d]> %@ = const()[name = string(\"%@\"), val = tensor<fp16, [%d, %d]>(%@)];\n", N, K, name, name, N, K, BLOB(off)];
    } else if (f == WF_LUT4) {
        int G = gLutG, bits = gLutBits, E = 1 << bits;
        size_t nbytes = ((size_t)N * K * bits + 7) / 8;
        uint32_t code = bits == 4 ? 11 : (bits == 8 ? 3 : (bits == 6 ? 13 : (bits == 2 ? 10 : 12)));
        void *pi; uint64_t offI = [b appendZeroed:nbytes dtype:code payload:&pi];
        uint8_t *idx = pi;
        void *pl; uint64_t offL = [b appendZeroed:(size_t)(N / G) * E * 2 dtype:1 payload:&pl];
        _Float16 *lut = pl;
        for (int g = 0; g < N / G; g++) for (int e = 0; e < E; e++) lut[g * E + e] = (_Float16)((e - (E - 1) / 2.f) / ((E - 1) / 2.f) * scaleMag * (0.5f + 0.5f * fabsf(frand())));
        for (size_t i = 0; i < (size_t)N * K; i++) {
            rng = rng * 1664525u + 1013904223u; uint32_t q = (rng >> 12) & (E - 1);
            size_t bit = i * bits;
            for (int t = 0; t < bits; t++, bit++) if (q & (1u << t)) idx[bit >> 3] |= (uint8_t)(1u << (bit & 7));
            if (deq) deq[i] = (float)lut[(i / K / G) * E + q];
        }
        [m appendFormat:@"        tensor<fp16, [%d, %d]> %@ = constexpr_lut_to_dense(indices = tensor<uint%d, [%d, %d]>(%@), lut = tensor<fp16, [%d, 1, %d, 1]>(%@))[name = string(\"%@\")];\n",
            N, K, name, bits, N, K, BLOB(offI), N / G, E, BLOB(offL), name];
    } else if (f == WF_LUT4_KBLK || f == WF_LUT4_ROW) {
        int KB = (f == WF_LUT4_KBLK) ? 32 : K;     // LUT per (row, 32-input block) == exact Q4_0
        int nKB = K / KB;
        void *pi; uint64_t offI = [b appendZeroed:(size_t)N * K / 2 dtype:11 payload:&pi];
        uint8_t *idx = pi;
        void *pl; uint64_t offL = [b appendZeroed:(size_t)N * nKB * 16 * 2 dtype:1 payload:&pl];
        _Float16 *lut = pl;
        for (size_t g = 0; g < (size_t)N * nKB; g++) { float d = scaleMag / 8.f * (0.5f + 0.5f * fabsf(frand())); for (int e = 0; e < 16; e++) lut[g * 16 + e] = (_Float16)(d * (e - 8)); }
        for (size_t i = 0; i < (size_t)N * K; i++) {
            rng = rng * 1664525u + 1013904223u; uint8_t q = (rng >> 16) & 15;
            if (i & 1) idx[i >> 1] |= (uint8_t)(q << 4); else idx[i >> 1] |= q;
            size_t r = i / K, c = i % K;
            if (deq) deq[i] = (float)lut[(r * nKB + c / KB) * 16 + q];
        }
        [m appendFormat:@"        tensor<fp16, [%d, %d]> %@ = constexpr_lut_to_dense(indices = tensor<uint4, [%d, %d]>(%@), lut = tensor<fp16, [%d, %d, 16, 1]>(%@))[name = string(\"%@\")];\n",
            N, K, name, N, K, BLOB(offI), N, nKB, BLOB(offL), name];
    } else if (f == WF_LUT8_ROW || f == WF_LUT6_ROW) {
        int bits = (f == WF_LUT8_ROW) ? 8 : 6, E = 1 << bits;
        size_t nbytes = ((size_t)N * K * bits + 7) / 8;
        void *pi; uint64_t offI = [b appendZeroed:nbytes dtype:(bits == 8 ? 3 : 13) payload:&pi];
        uint8_t *idx = pi;
        void *pl; uint64_t offL = [b appendZeroed:(size_t)N * E * 2 dtype:1 payload:&pl];
        _Float16 *lut = pl;
        for (int r = 0; r < N; r++) { float d = scaleMag / (E / 2) * (0.5f + 0.5f * fabsf(frand())); for (int e = 0; e < E; e++) lut[r * E + e] = (_Float16)(d * (e - E / 2)); }
        for (size_t i = 0; i < (size_t)N * K; i++) {
            rng = rng * 1664525u + 1013904223u; uint32_t q = (rng >> 12) & (E - 1);
            size_t bit = i * bits;                       // LSB-first bit packing
            for (int t = 0; t < bits; t++, bit++) if (q & (1u << t)) idx[bit >> 3] |= (uint8_t)(1u << (bit & 7));
            if (deq) deq[i] = (float)lut[(i / K) * E + q];
        }
        [m appendFormat:@"        tensor<fp16, [%d, %d]> %@ = constexpr_lut_to_dense(indices = tensor<uint%d, [%d, %d]>(%@), lut = tensor<fp16, [%d, 1, %d, 1]>(%@))[name = string(\"%@\")];\n",
            N, K, name, bits, N, K, BLOB(offI), N, E, BLOB(offL), name];
    } else if (f == WF_BLK8) {
        int B = 32;
        void *pd; uint64_t offD = [b appendZeroed:(size_t)N * K dtype:4 payload:&pd];
        int8_t *dq = pd;
        void *ps; uint64_t offS = [b appendZeroed:(size_t)N * (K / B) * 2 dtype:1 payload:&ps];
        _Float16 *sc = ps;
        for (size_t i = 0; i < (size_t)N * (K / B); i++) sc[i] = (_Float16)(scaleMag / 127.f * (0.5f + 0.5f * fabsf(frand())));
        for (size_t i = 0; i < (size_t)N * K; i++) {
            rng = rng * 1664525u + 1013904223u; int q = (int)((rng >> 16) & 255) - 128; dq[i] = (int8_t)q;
            size_t r = i / K, c = i % K;
            if (deq) deq[i] = (float)sc[r * (K / B) + c / B] * (float)q;
        }
        [m appendFormat:@"        tensor<fp16, [%d, %d]> %@ = constexpr_blockwise_shift_scale(data = tensor<int8, [%d, %d]>(%@), scale = tensor<fp16, [%d, %d]>(%@))[name = string(\"%@\")];\n",
            N, K, name, N, K, BLOB(offD), N, K / B, BLOB(offS), name];
    } else if (f == WF_BLK4U) {
        int B = 32;
        void *pd; uint64_t offD = [b appendZeroed:(size_t)N * K / 2 dtype:11 payload:&pd];
        uint8_t *dq = pd;
        void *ps; uint64_t offS = [b appendZeroed:(size_t)N * (K / B) * 2 dtype:1 payload:&ps];
        _Float16 *sc = ps;
        void *po; uint64_t offO = [b appendZeroed:(size_t)N * (K / B) / 2 dtype:11 payload:&po];
        memset(po, 0x88, (size_t)N * (K / B) / 2);
        for (size_t i = 0; i < (size_t)N * (K / B); i++) sc[i] = (_Float16)(scaleMag / 8.f * (0.5f + 0.5f * fabsf(frand())));
        for (size_t i = 0; i < (size_t)N * K; i++) {
            rng = rng * 1664525u + 1013904223u; uint8_t q = (rng >> 16) & 15;
            if (i & 1) dq[i >> 1] |= (uint8_t)(q << 4); else dq[i >> 1] |= q;
            size_t r = i / K, c = i % K;
            if (deq) deq[i] = (float)sc[r * (K / B) + c / B] * (float)((int)q - 8);
        }
        [m appendFormat:@"        tensor<fp16, [%d, %d]> %@ = constexpr_blockwise_shift_scale(data = tensor<uint4, [%d, %d]>(%@), scale = tensor<fp16, [%d, %d]>(%@), offset = tensor<uint4, [%d, %d]>(%@))[name = string(\"%@\")];\n",
            N, K, name, N, K, BLOB(offD), N, K / B, BLOB(offS), N, K / B, BLOB(offO), name];
    } else {
        int B = 32;
        void *pd; uint64_t offD = [b appendZeroed:(size_t)N * K / 2 dtype:8 payload:&pd];
        uint8_t *dq = pd;
        void *ps; uint64_t offS = [b appendZeroed:(size_t)N * (K / B) * 2 dtype:1 payload:&ps];
        _Float16 *sc = ps;
        for (size_t i = 0; i < (size_t)N * (K / B); i++) sc[i] = (_Float16)(scaleMag / 8.f * (0.5f + 0.5f * fabsf(frand())));
        for (size_t i = 0; i < (size_t)N * K; i++) {
            rng = rng * 1664525u + 1013904223u; int q = (int)((rng >> 16) & 15) - 8; // int4 -8..7
            uint8_t nib = (uint8_t)(q & 15);
            if (i & 1) dq[i >> 1] |= (uint8_t)(nib << 4); else dq[i >> 1] |= nib;
            size_t r = i / K, c = i % K;
            if (deq) deq[i] = (float)sc[r * (K / B) + c / B] * (float)q;
        }
        [m appendFormat:@"        tensor<fp16, [%d, %d]> %@ = constexpr_blockwise_shift_scale(data = tensor<int4, [%d, %d]>(%@), scale = tensor<fp16, [%d, %d]>(%@))[name = string(\"%@\")];\n",
            N, K, name, N, K, BLOB(offD), N, K / B, BLOB(offS), name];
    }
}

// ---------------------------------------------------------------------------------------------
// T1: single linear correctness for each weight format and S in {1, 8}
static void testLinearCorrectness(void) {
    const int K = 1536, N = 2048;
    for (int fi = 0; fi < gNumFormats; fi++) { int f = gFormats[fi];
        for (int si = 0; si < 2; si++) {
            int S = si == 0 ? 1 : 8;
            ANEBlobBuilder *b = [ANEBlobBuilder new];
            NSMutableString *m = [NSMutableString stringWithString:HDR];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, %d]> x) {\n", S, K];
            float *W = malloc(sizeof(float) * N * K);
            emitWeight(m, b, @"W", N, K, (WFmt)f, W, 0.05f);
            [m appendFormat:@"        tensor<fp16, [1, %d, %d]> y = linear(weight = W, x = x)[name = string(\"lin\")];\n    } -> (y);\n}\n", S, N];
            NSError *err = nil;
            ANEProgram *p = [ANEProgram compileMIL:m weights:[b finish] procedures:@[@{@"in": @[@(S * K * 2)], @"out": @[@(S * N * 2)]}] error:&err];
            if (!p) { printf("T1 %-10s S=%d  COMPILE FAIL: %s\n", wfName[f], S, err.description.UTF8String); free(W); continue; }
            _Float16 *xi = [p inputPtr:0 procedure:0];
            float *xr = malloc(sizeof(float) * S * K);
            for (int i = 0; i < S * K; i++) { xi[i] = (_Float16)frand(); xr[i] = (float)xi[i]; }
            if (![p eval:0 error:&err]) { printf("T1 %-10s S=%d  EVAL FAIL: %s\n", wfName[f], S, err.description.UTF8String); free(W); free(xr); continue; }
            _Float16 *yo = [p outputPtr:0 procedure:0];
            double maxErr = 0, maxRef = 0;
            for (int s = 0; s < S; s++) for (int n = 0; n < N; n++) {
                double acc = 0; for (int k = 0; k < K; k++) acc += (double)W[(size_t)n * K + k] * xr[s * K + k];
                maxErr = fmax(maxErr, fabs(acc - (double)yo[s * N + n])); maxRef = fmax(maxRef, fabs(acc));
            }
            printf("T1 %-10s S=%d  rel max err %.4f  (compile %.2fs load %.2fs)\n", wfName[f], S, maxErr / maxRef, p.compileSeconds, p.loadSeconds);
            free(W); free(xr);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// T2: dispatch latency of a tiny program
static void testDispatch(void) {
    ANEBlobBuilder *b = [ANEBlobBuilder new];
    NSMutableString *m = [NSMutableString stringWithString:HDR];
    [m appendString:@"    func main<ios18>(tensor<fp16, [1, 1, 64]> x) {\n"];
    emitWeight(m, b, @"W", 64, 64, WF_FP16, NULL, 0.1f);
    [m appendString:@"        tensor<fp16, [1, 1, 64]> y = linear(weight = W, x = x)[name = string(\"lin\")];\n    } -> (y);\n}\n"];
    NSError *err = nil;
    ANEProgram *p = [ANEProgram compileMIL:m weights:[b finish] procedures:@[@{@"in": @[@128], @"out": @[@128]}] error:&err];
    if (!p) { printf("T2 compile fail %s\n", err.description.UTF8String); return; }
    for (int i = 0; i < 20; i++) [p eval:0 error:nil];
    int it = 500; double t0 = now_ms();
    for (int i = 0; i < it; i++) [p eval:0 error:nil];
    printf("T2 dispatch latency (tiny kernel): %.3f ms/eval\n", (now_ms() - t0) / it);
}

// ---------------------------------------------------------------------------------------------
// T3: Gemma-FFN-shaped chain: L x [gate(6144x1536), up(6144x1536), gelu*up, down(1536x6144), +res]
static void testFFNChain(int L) {
    const int H = 1536, F = 6144;
    int Ss[] = {1, 64};
    for (int fi = 0; fi < gNumFormats; fi++) { int f = gFormats[fi];
        for (int si = 0; si < 2; si++) {
            int S = Ss[si];
            ANEBlobBuilder *b = [ANEBlobBuilder new];
            NSMutableString *m = [NSMutableString stringWithString:HDR];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, %d]> x) {\n", S, H];
            [m appendString:@"        string gm = const()[name = string(\"gm\"), val = string(\"TANH_APPROXIMATION\")];\n"];
            NSString *cur = @"x";
            for (int l = 0; l < L; l++) {
                emitWeight(m, b, [NSString stringWithFormat:@"Wg%d", l], F, H, (WFmt)f, NULL, 0.03f);
                emitWeight(m, b, [NSString stringWithFormat:@"Wu%d", l], F, H, (WFmt)f, NULL, 0.03f);
                emitWeight(m, b, [NSString stringWithFormat:@"Wd%d", l], H, F, (WFmt)f, NULL, 0.03f);
                [m appendFormat:@"        tensor<fp16, [1, %d, %d]> g%d = linear(weight = Wg%d, x = %@)[name = string(\"g%d\")];\n", S, F, l, l, cur, l];
                [m appendFormat:@"        tensor<fp16, [1, %d, %d]> u%d = linear(weight = Wu%d, x = %@)[name = string(\"u%d\")];\n", S, F, l, l, cur, l];
                [m appendFormat:@"        tensor<fp16, [1, %d, %d]> a%d = gelu(mode = gm, x = g%d)[name = string(\"a%d\")];\n", S, F, l, l, l];
                [m appendFormat:@"        tensor<fp16, [1, %d, %d]> h%d = mul(x = a%d, y = u%d)[name = string(\"h%d\")];\n", S, F, l, l, l, l];
                [m appendFormat:@"        tensor<fp16, [1, %d, %d]> d%d = linear(weight = Wd%d, x = h%d)[name = string(\"d%d\")];\n", S, H, l, l, l, l];
                [m appendFormat:@"        tensor<fp16, [1, %d, %d]> r%d = add(x = %@, y = d%d)[name = string(\"r%d\")];\n", S, H, l, cur, l, l];
                cur = [NSString stringWithFormat:@"r%d", l];
            }
            [m appendFormat:@"    } -> (%@);\n}\n", cur];
            NSError *err = nil;
            NSData *w = [b finish];
            ANEProgram *p = [ANEProgram compileMIL:m weights:w procedures:@[@{@"in": @[@(S * H * 2)], @"out": @[@(S * H * 2)]}] error:&err];
            if (!p) { printf("T3 %-10s S=%-3d COMPILE FAIL: %s\n", wfName[f], S, err.localizedDescription.UTF8String); continue; }
            _Float16 *xi = [p inputPtr:0 procedure:0];
            for (int i = 0; i < S * H; i++) xi[i] = (_Float16)(frand() * 0.5f);
            for (int i = 0; i < 5; i++) [p eval:0 error:nil];
            int it = S >= 64 ? 20 : 40;
            double t0 = now_ms();
            for (int i = 0; i < it; i++) [p eval:0 error:nil];
            double ms = (now_ms() - t0) / it;
            double params = (double)L * 3 * H * F;
            double bpw = f == 0 ? 2.0 : (f == WF_LUT8_ROW ? 1.0 : (f == WF_LUT6_ROW ? 0.75 : (f == WF_LUT4 ? gLutBits / 8.0 : 0.5)));
            double gbps = params * bpw / (ms / 1000) / 1e9;
            double tflops = 2 * params * S / (ms / 1000) / 1e12;
            printf("T3 %-10s L=%d S=%-3d %7.3f ms/eval  weights %.0f MB -> %6.1f GB/s  %5.2f TFLOPS  %6.0f tok/s-equiv  (compile %.1fs)\n",
                   wfName[f], L, S, ms, params * bpw / 1e6, gbps, tflops, S / (ms / 1000), p.compileSeconds);
        }
    }
}


// T4: multi-function program (shared weights) + multi-input
static void testMultiFunctionMultiInput(void) {
    const int K = 1536, N = 2048, P = 256;
    ANEBlobBuilder *b = [ANEBlobBuilder new];
    NSMutableString *body = [NSMutableString string];
    float *W = malloc(sizeof(float) * N * K);
    emitWeight(body, b, @"W", N, K, WF_LUT4, W, 0.05f);
    NSMutableString *m = [NSMutableString stringWithString:HDR];
    int Ss[2] = {1, 64};
    for (int fi = 0; fi < 2; fi++) {
        int S = Ss[fi];
        [m appendFormat:@"    func f%d<ios18>(tensor<fp16, [1, %d, %d]> x, tensor<fp16, [1, %d, %d]> z) {\n", S, S, K, S, P];
        [m appendString:body];
        [m appendFormat:@"        tensor<fp16, [1, %d, %d]> y = linear(weight = W, x = x)[name = string(\"lin\")];\n", S, N];
        [m appendFormat:@"        tensor<fp16, [1, %d, %d]> z2 = mul(x = z, y = z)[name = string(\"zz\")];\n", S, P];
        [m appendFormat:@"    } -> (y, z2);\n"];
    }
    [m appendString:@"}\n"];
    NSError *err = nil;
    ANEProgram *p = [ANEProgram compileMIL:m weights:[b finish] procedures:@[
        @{@"in": @[@(1 * K * 2), @(1 * P * 2)], @"out": @[@(1 * N * 2), @(1 * P * 2)]},
        @{@"in": @[@(64 * K * 2), @(64 * P * 2)], @"out": @[@(64 * N * 2), @(64 * P * 2)]}] error:&err];
    if (!p) { printf("T4 multi-function compile FAIL: %s\n", err.description.UTF8String); free(W); return; }
    for (int pi = 0; pi < 2; pi++) {
        int S = Ss[pi];
        _Float16 *x = [p inputPtr:0 procedure:pi], *z = [p inputPtr:1 procedure:pi];
        float *xr = malloc(sizeof(float) * S * K);
        for (int i = 0; i < S * K; i++) { x[i] = (_Float16)frand(); xr[i] = (float)x[i]; }
        for (int i = 0; i < S * P; i++) z[i] = (_Float16)(i % 7);
        if (![p eval:pi error:&err]) { printf("T4 proc %d eval FAIL: %s\n", pi, err.description.UTF8String); free(xr); continue; }
        _Float16 *y = [p outputPtr:0 procedure:pi], *z2 = [p outputPtr:1 procedure:pi];
        double me = 0, mr = 0; int zbad = 0;
        for (int s = 0; s < S; s++) for (int n = 0; n < N; n += 7) { double a = 0; for (int k = 0; k < K; k++) a += W[(size_t)n * K + k] * xr[s * K + k]; me = fmax(me, fabs(a - (double)y[s * N + n])); mr = fmax(mr, fabs(a)); }
        for (int i = 0; i < S * P; i++) if ((float)z2[i] != (float)((i % 7) * (i % 7))) zbad++;
        int it = 200; double t0 = now_ms(); for (int i = 0; i < it; i++) [p eval:pi error:nil];
        printf("T4 proc %d (S=%d): linear rel err %.4f, second input/output mismatches %d, %.3f ms/eval\n", pi, S, me / mr, zbad, (now_ms() - t0) / it);
        free(xr);
    }
    free(W);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (!ane_rt_available()) { printf("ANE private API unavailable\n"); return 1; }
        NSString *which = argc > 1 ? @(argv[1]) : @"all";
        for (int i = 2; i < argc; i++) gFormats[gNumFormats++] = atoi(argv[i]);
        if (getenv("LUTG")) gLutG = atoi(getenv("LUTG"));
        if (getenv("LUTB")) gLutBits = atoi(getenv("LUTB"));
        if (gNumFormats == 0) { for (int i = 0; i < 7; i++) gFormats[gNumFormats++] = i; }
        if ([which isEqual:@"all"] || [which isEqual:@"t1"]) testLinearCorrectness();
        if ([which isEqual:@"all"] || [which isEqual:@"t2"]) testDispatch();
        if ([which isEqual:@"all"] || [which isEqual:@"t3"]) testFFNChain(4);
        if ([which isEqual:@"t4"]) testMultiFunctionMultiInput();
        printf("compiles: %d\n", ane_rt_compile_count());
    }
    return 0;
}
