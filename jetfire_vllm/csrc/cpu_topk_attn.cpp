// CPU-side top-K attention for the JetFire decode correction path.
//
// Replaces the GPU-side flow ``(gather → H2D K_topk + V_topk MB → GPU
// post_correction attn)`` with ``(gather + softmax(Q·K_topk^T)·V_topk on
// CPU → H2D o_new ~tens of KB)``.  Cuts H2D traffic by ~125× and lets the
// shadow live in unpinned host memory, which removes the giant pinned
// allocation that otherwise scales with ``num_blocks`` (and stalls engine
// init under ``JETFIRE_INFLATE_BLOCKS=1``).
//
// Inputs (all CPU):
//   q          (B, H_q, D)         bf16
//   k_shadow   (H_kv, num_slots, D) bf16
//   v_shadow   (H_kv, num_slots, D) bf16
//   head_map   (B, H_q, K)          int64  — q-head → kv-head map
//   ts_buf     (B, H_q, K)          int64  — top-k slot indices per (b, h_q)
//   o_new      (B, H_q, D)          bf16   — [out] computed attention result
//   scale      float                       — softmax scale (1/sqrt(D))
//
// Algorithm per (b, h_q):
//   Gather K_topk[k] = k_shadow[head_map[b,h_q,k], ts_buf[b,h_q,k], :]   (K, D)
//   Gather V_topk[k] = v_shadow[head_map[b,h_q,k], ts_buf[b,h_q,k], :]   (K, D)
//   scores[k] = Q[b,h_q,:] · K_topk[k] · scale                          (K,)
//   softmax(scores)                                                      (K,)
//   o_new[b,h_q,:] = scores · V_topk                                     (D,)
//
// Parallelization: at::parallel_for over (B*H_q) outer pairs.  Each pair
// is independent, work is uniform.  GIL released at binding so a worker
// thread runs concurrently with main-thread CUDA op issue.

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <linux/futex.h>
#include <mutex>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <omp.h>

// AVX-512-BF16 SIMD path (AMD Zen 4 / Intel Sapphire Rapids).  Compile flags
// in jetfire_vllm/kernels/cpu_topk_attn.py include -march=native, which on
// EPYC 9654 (Zen 4) enables avx512f / avx512bw / avx512_bf16 / avx512vnni.
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512BF16__)
#define JETFIRE_HAVE_AVX512_BF16 1
#include <immintrin.h>
#else
#define JETFIRE_HAVE_AVX512_BF16 0
#endif

// CUDA runtime for mapped pinned memory allocator.
#include <cuda.h>           // driver API: cuStreamWaitValue64 etc.
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

// Custom thread pool (replaces libgomp for the hot decode path).
#include "jetfire_pool.h"

static inline bool valid_shadow_slot(int64_t slot, int64_t num_slots) {
    return slot >= 0 && slot < num_slots;
}

static inline bool bf16_bits_is_nan(uint16_t x) {
    return (x & 0x7f80u) == 0x7f80u && (x & 0x007fu) != 0;
}

static inline bool bf16_bits_is_finite(uint16_t x) {
    return (x & 0x7f80u) != 0x7f80u;
}

static inline uint16_t bf16_bits(const at::BFloat16* p) {
    return *reinterpret_cast<const uint16_t*>(p);
}

static inline float bf16_bits_to_float(uint16_t x) {
    uint32_t raw = static_cast<uint32_t>(x) << 16;
    float out = 0.0f;
    std::memcpy(&out, &raw, sizeof(float));
    return out;
}

static inline bool bf16_row_has_nonfinite(const at::BFloat16* p, int64_t D) {
    for (int64_t i = 0; i < D; ++i) {
        if (!bf16_bits_is_finite(bf16_bits(p + i))) {
            return true;
        }
    }
    return false;
}

static inline void zero_bf16_row(at::BFloat16* p, int64_t D) {
    for (int64_t i = 0; i < D; ++i) {
        p[i] = static_cast<at::BFloat16>(0.0f);
    }
}

namespace jetfire_pool {
// Singleton.  Lazy-init on first dispatch; size = jetfire_cpu_attn_threads().
Pool& global_pool() {
    static Pool pool;
    return pool;
}
}  // namespace jetfire_pool


// Resolve the worker count once per process: env JETFIRE_CPU_ATTN_THREADS
// wins, else min(omp_get_num_procs(), 64).  vLLM tends to set
// OMP_NUM_THREADS=1 in its worker subprocess, which silently makes
// at::parallel_for serial — explicit num_threads() on the pragma sidesteps
// that.
//
// Cap default at 64 (not 32) because the typical decode workload at B=32
// H_q=32 produces 1024 (b, h_q) rows which spreads cleanly over 32–96
// threads on EPYC 9654 (96 physical / 192 SMT).  Going above 64 starts to
// pay diminishing returns once each thread owns <16 rows (~5 µs of work),
// at which point fork/join overhead dominates.  The env var lets callers
// dial up to 96 or 192 when measuring.
//
// Microbench escape hatch: ``set_cpu_attn_threads(n)`` overrides the env at
// runtime so a single process can sweep {16, 32, 64, 128, 192} without
// restarting.  Pass n<=0 to revert to env/default.
static std::atomic<int> g_thread_override{0};
static std::atomic<int64_t> g_cpu_sanitize_count{0};
static std::atomic<int64_t> g_cpu_sanitize_log_count{0};

static int jetfire_cpu_attn_threads() {
    int override_n = g_thread_override.load(std::memory_order_acquire);
    if (override_n > 0) return override_n;
    static int cached = -1;
    if (cached > 0) return cached;
    int n = -1;
    if (const char* e = std::getenv("JETFIRE_CPU_ATTN_THREADS")) {
        n = std::atoi(e);
    }
    if (n <= 0) {
        int procs = omp_get_num_procs();
        n = procs < 64 ? procs : 64;
    }
    cached = n;
    return cached;
}

static void set_cpu_attn_threads(int n) {
    g_thread_override.store(n > 0 ? n : 0, std::memory_order_release);
}

static bool jetfire_cpu_attn_use_pool() {
    static const bool use_pool = []() {
        const char* s = std::getenv("JETFIRE_CUSTOM_POOL");
        return !(s && s[0] == '0');
    }();
    return use_pool;
}

static bool jetfire_cpu_attn_pin_workers() {
    static const bool pin = []() {
        const char* s = std::getenv("JETFIRE_CPU_ATTN_PIN");
        return s && s[0] == '1';
    }();
    return pin;
}

static bool jetfire_cpu_attn_debug() {
    static const bool debug = []() {
        const char* s = std::getenv("JETFIRE_CPU_DEBUG");
        return s && s[0] == '1';
    }();
    return debug;
}

static bool jetfire_cpu_attn_sanitize_output() {
    static const bool sanitize = []() {
        const char* s = std::getenv("JETFIRE_CPU_ATTN_SANITIZE");
        if (s != nullptr) {
            return s[0] != '0';
        }
        const char* graph = std::getenv("JETFIRE_DECODE_GRAPH");
        return graph != nullptr && graph[0] == '1';
    }();
    return sanitize;
}

static void maybe_log_sanitize_row(
    int wid,
    int64_t b,
    int64_t h_q,
    const at::BFloat16* q_bh,
    const at::BFloat16* k_shadow,
    const at::BFloat16* v_shadow,
    int64_t num_slots,
    const int64_t* head_map_bh,
    const int64_t* ts_buf_bh,
    int64_t K,
    int64_t D,
    const at::BFloat16* o_bh)
{
    if (std::getenv("JETFIRE_DEBUG_GRAPH_SCAN") == nullptr) {
        return;
    }
    const int64_t log_idx = g_cpu_sanitize_log_count.fetch_add(
        1, std::memory_order_acq_rel);
    if (log_idx >= 64) {
        return;
    }
    float q_abs_max = 0.0f;
    float k_abs_max = 0.0f;
    float v_abs_max = 0.0f;
    int64_t out_bad_i = -1;
    uint16_t out_bad_bits = 0;
    int64_t k_abs_slot = -1;
    int64_t v_abs_slot = -1;
    const int64_t kshadow_head_stride = num_slots * D;
    for (int64_t i = 0; i < D; ++i) {
        const float qv = std::fabs(bf16_bits_to_float(bf16_bits(q_bh + i)));
        q_abs_max = std::max(q_abs_max, qv);
        const uint16_t ob = bf16_bits(o_bh + i);
        if (out_bad_i < 0 && !bf16_bits_is_finite(ob)) {
            out_bad_i = i;
            out_bad_bits = ob;
        }
    }
    for (int64_t k = 0; k < K; ++k) {
        const int64_t hkv = head_map_bh[k];
        const int64_t slot = ts_buf_bh[k];
        if (!valid_shadow_slot(slot, num_slots)) {
            continue;
        }
        const at::BFloat16* k_row =
            k_shadow + hkv * kshadow_head_stride + slot * D;
        const at::BFloat16* v_row =
            v_shadow + hkv * kshadow_head_stride + slot * D;
        for (int64_t i = 0; i < D; ++i) {
            const float kv = std::fabs(bf16_bits_to_float(bf16_bits(k_row + i)));
            const float vv = std::fabs(bf16_bits_to_float(bf16_bits(v_row + i)));
            if (kv > k_abs_max) {
                k_abs_max = kv;
                k_abs_slot = slot;
            }
            if (vv > v_abs_max) {
                v_abs_max = vv;
                v_abs_slot = slot;
            }
        }
    }
    std::fprintf(stderr,
                 "[jetfire] cpu_topk_sanitize log=%ld wid=%d b=%ld hq=%ld "
                 "out_bad_i=%ld out_bits=0x%04x q_abs_max=%g "
                 "k_abs_max=%g k_slot=%ld v_abs_max=%g v_slot=%ld K=%ld\n",
                 (long)log_idx, wid, (long)b, (long)h_q,
                 (long)out_bad_i, (unsigned)out_bad_bits,
                 (double)q_abs_max, (double)k_abs_max, (long)k_abs_slot,
                 (double)v_abs_max, (long)v_abs_slot, (long)K);
    std::fflush(stderr);
}

#if JETFIRE_HAVE_AVX512_BF16
static bool jetfire_cpu_attn_use_simd() {
    static const bool use_simd = []() {
        const char* s = std::getenv("JETFIRE_CPU_ATTN_SCALAR");
        return !(s && s[0] == '1');
    }();
    return use_simd;
}

static bool jetfire_cpu_attn_use_v2() {
    static const bool use_v2 = []() {
        const char* s = std::getenv("JETFIRE_CPU_ATTN_V1");
        return !(s && s[0] == '1');
    }();
    return use_v2;
}
#endif


#if JETFIRE_HAVE_AVX512_BF16
// AVX-512-BF16 specialized fast path for D=128 (LLaMA / Mistral standard
// head dim).  Replaces 8192 scalar bf16→fp32 + fp32 mul/add with 128 fused
// 32-bf16 dot-product instructions per (b, h_q).  Per K-row dot product:
// 4× _mm512_dpbf16_ps  (each does 32 bf16 mul + 16 fp32 add, accumulates
// into a 16-fp32 register), then a single horizontal reduction.  V mul:
// load 16 bf16 → upcast via shift-left-16 → fp32 fmadd.  Output: 8×
// _mm512_cvtneps_pbh to bf16.
//
// ── v2: software prefetch + V_topk caching ─────────────────────────────
// Each (b, h_q) row touches 256 random K rows and 256 random V rows of
// the shadow KV cache.  Two micro-optimizations over the v1 SIMD path:
//
//   (a) Software-prefetch K_topk[k+P] and V_topk[k+P] inside the K-dot
//       loop so K-row scatter loads hit L1 instead of L2/L3, and V is
//       already warm in L2 by the time the V-FMA loop runs.  P chosen
//       so that ~P × (5 × dpbf16 + reduce + scalar update) ≈ L2 latency
//       (~12-14 cycles on Zen 4).  P=8 ≈ 100 cycles of compute hides L2.
//
//   (b) Pack V_topk into a stack-local contiguous bf16 buffer during
//       the K-dot loop.  Cost: one extra 256 B store per k.  Benefit:
//       the V-FMA loop streams a contiguous 64 KB buffer instead of
//       chasing 256 random pointers — eliminates a second TLB walk
//       set + lets the L1 prefetcher (Zen 4: streaming prefetch on
//       linear access) take over completely.
//
// We do NOT pack K (it's only consumed once, by the K-dot loop, in the
// same pass where we'd write the pack), so packing K would just double
// the dcache traffic for zero compute benefit.
//
// Math is bit-identical to the v1 SIMD path: per-K-row dot-product
// (4×dpbf16 + reduce_add) and V accumulator order (k=0..K-1 × 8 fp32
// lanes) are unchanged.  Only the load source for V changes.
static inline void cpu_topk_attn_one_avx512bf16_d128_v2(
    const at::BFloat16* q_bh,
    const at::BFloat16* k_shadow,
    const at::BFloat16* v_shadow,
    int64_t num_slots,
    const int64_t* head_map_bh,
    const int64_t* ts_buf_bh,
    int64_t K,
    float scale,
    at::BFloat16* o_bh)
{
    constexpr int64_t D = 128;
    // Large top_k (up to 8192) cannot use fixed stack buffers: 8192*128*2B
    // is 2 MiB per worker just for V_pack.  Keep per-worker scratch in
    // thread-local vectors so CUDA graph host callbacks do not trip a stack
    // overflow or hidden TORCH_CHECK inside worker threads.
    thread_local std::vector<at::BFloat16> v_pack_storage;
    thread_local std::vector<float> scores_storage;
    v_pack_storage.resize((size_t)K * D);
    scores_storage.resize((size_t)K);
    at::BFloat16* V_pack = v_pack_storage.data();
    float* scores = scores_storage.data();

    const int64_t kshadow_head_stride = num_slots * D;

    // Pre-load Q as 4× __m512bh (32 bf16 each = 128 lanes).
    __m512bh q_v[4];
    for (int i = 0; i < 4; ++i) {
        __m512i raw = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(q_bh + i * 32));
        q_v[i] = reinterpret_cast<__m512bh>(raw);
    }

    // ── Prologue: prefetch the first PREFETCH_AHEAD K/V rows ──────────
    // Without this, the first PREFETCH_AHEAD iterations of the K-dot loop
    // would all miss in L1.  Each row is 256 B = 4 × 64 B cache lines.
    //
    // PREFETCH_AHEAD chosen empirically: 8 rows ahead × ~30 cycles per
    // dpbf16 dot ≈ 240 cycles ≈ 80 ns, enough to hide Zen 4's L2 latency
    // (~14 cycles) and partially hide L3 (~40 ns).  Going to 16 thrashes
    // L1d on Zen 4 (limited in-flight loads + prefetch issue contention).
    constexpr int PREFETCH_AHEAD = 8;
    auto pf_row = [](const at::BFloat16* row) {
        _mm_prefetch(reinterpret_cast<const char*>(row),        _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(row +  64),  _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(row + 128),  _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(row + 192),  _MM_HINT_T0);
    };
    int prologue = (int)std::min<int64_t>(PREFETCH_AHEAD, K);
    for (int p = 0; p < prologue; ++p) {
        const int64_t hkv  = head_map_bh[p];
        const int64_t slot = ts_buf_bh[p];
        if (!valid_shadow_slot(slot, num_slots)) {
            continue;
        }
        pf_row(k_shadow + hkv * kshadow_head_stride + slot * D);
        pf_row(v_shadow + hkv * kshadow_head_stride + slot * D);
    }

    // ── Step 1: K dot products + V-pack (one pass) ────────────────────
    // Math identical to v1: per-K-row, 4× dpbf16_ps into a single __m512
    // accumulator, then reduce_add_ps and multiply by scale.  Extra work
    // per iteration: prefetch K[k+P] / V[k+P], and copy V[k] into V_pack
    // (256 B store, partially overlaps the K-dot latency).
    float max_score = -INFINITY;
    bool any_valid = false;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t hkv  = head_map_bh[k];
        const int64_t slot = ts_buf_bh[k];
        if (!valid_shadow_slot(slot, num_slots)) {
            scores[k] = -INFINITY;
            std::memset(V_pack + k * D, 0, D * sizeof(at::BFloat16));
            continue;
        }
        any_valid = true;
        const at::BFloat16* k_row =
            k_shadow + hkv * kshadow_head_stride + slot * D;
        const at::BFloat16* v_row =
            v_shadow + hkv * kshadow_head_stride + slot * D;

        if (k + PREFETCH_AHEAD < K) {
            const int64_t hkv_pf  = head_map_bh[k + PREFETCH_AHEAD];
            const int64_t slot_pf = ts_buf_bh[k + PREFETCH_AHEAD];
            if (valid_shadow_slot(slot_pf, num_slots)) {
                pf_row(k_shadow + hkv_pf * kshadow_head_stride + slot_pf * D);
                pf_row(v_shadow + hkv_pf * kshadow_head_stride + slot_pf * D);
            }
        }

        __m512 acc = _mm512_setzero_ps();
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            __m512i k_raw = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(k_row + i * 32));
            __m512bh k_vbh = reinterpret_cast<__m512bh>(k_raw);
            acc = _mm512_dpbf16_ps(acc, q_v[i], k_vbh);

            // Copy this 64 B half of v_row into the pack buffer while the
            // dpbf16 instruction is in-flight (different µops, run in
            // parallel on Zen 4: load-store unit vs FPU).
            __m512i v_raw = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(v_row + i * 32));
            _mm512_storeu_si512(
                reinterpret_cast<__m512i*>(V_pack + k * D + i * 32), v_raw);
        }
        const float dot = _mm512_reduce_add_ps(acc);
        const float s = dot * scale;
        scores[k] = s;
        if (s > max_score) max_score = s;
    }

    // ── Step 2: softmax (scalar — K is small) ─────────────────────────
    if (!any_valid) {
        for (int64_t i = 0; i < D; ++i) {
            o_bh[i] = static_cast<at::BFloat16>(0.0f);
        }
        return;
    }
    float denom = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        scores[k] = std::exp(scores[k] - max_score);
        denom += scores[k];
    }
    if (!(denom > 0.0f) || !std::isfinite(denom)) {
        for (int64_t i = 0; i < D; ++i) {
            o_bh[i] = static_cast<at::BFloat16>(0.0f);
        }
        return;
    }
    const float inv_denom = 1.0f / denom;
    for (int64_t k = 0; k < K; ++k) scores[k] *= inv_denom;

    // ── Step 3: V mul → 8 fp32 accumulators (from packed bf16 buffer) ─
    // V_pack is contiguous in memory: 256 × 256 B = 64 KB streamed in K-
    // major order.  The L1 streaming prefetcher (Zen 4: enabled by
    // default for linear access) keeps this hot in L1 without explicit
    // _mm_prefetch.  Math identical to v1.
    __m512 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = _mm512_setzero_ps();

    for (int64_t k = 0; k < K; ++k) {
        const at::BFloat16* v_row = V_pack + k * D;
        const __m512 w_v = _mm512_set1_ps(scores[k]);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            __m256i bf = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(v_row + i * 16));
            __m512i u32 = _mm512_cvtepu16_epi32(bf);
            __m512i sh  = _mm512_slli_epi32(u32, 16);
            __m512  v_f = _mm512_castsi512_ps(sh);
            acc[i] = _mm512_fmadd_ps(w_v, v_f, acc[i]);
        }
    }

    // ── Step 4: fp32 → bf16 + store ──────────────────────────────────
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        __m256bh out_bf = _mm512_cvtneps_pbh(acc[i]);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(o_bh + i * 16),
            reinterpret_cast<__m256i>(out_bf));
    }
}


// Legacy SIMD path (no gather caching).  Retained as a fallback / for A/B
// testing via env JETFIRE_CPU_ATTN_V1=1.
static inline void cpu_topk_attn_one_avx512bf16_d128(
    const at::BFloat16* q_bh,
    const at::BFloat16* k_shadow,
    const at::BFloat16* v_shadow,
    int64_t num_slots,
    const int64_t* head_map_bh,
    const int64_t* ts_buf_bh,
    int64_t K,
    float scale,
    at::BFloat16* o_bh)
{
    constexpr int64_t D = 128;
    thread_local std::vector<float> scores_storage;
    scores_storage.resize((size_t)K);
    float* scores = scores_storage.data();
    const int64_t kshadow_head_stride = num_slots * D;

    // Pre-load Q as 4× __m512bh (32 bf16 each = 128 lanes).
    __m512bh q_v[4];
    for (int i = 0; i < 4; ++i) {
        __m512i raw = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(q_bh + i * 32));
        q_v[i] = reinterpret_cast<__m512bh>(raw);
    }

    // ── Step 1: K dot products via dpbf16_ps ───────────────────────────
    float max_score = -INFINITY;
    bool any_valid = false;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t hkv  = head_map_bh[k];
        const int64_t slot = ts_buf_bh[k];
        if (!valid_shadow_slot(slot, num_slots)) {
            scores[k] = -INFINITY;
            continue;
        }
        any_valid = true;
        const at::BFloat16* k_row =
            k_shadow + hkv * kshadow_head_stride + slot * D;

        __m512 acc = _mm512_setzero_ps();
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            __m512i k_raw = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(k_row + i * 32));
            __m512bh k_v = reinterpret_cast<__m512bh>(k_raw);
            // dpbf16_ps: acc[j] += a[2j]*b[2j] + a[2j+1]*b[2j+1]  (j=0..15)
            acc = _mm512_dpbf16_ps(acc, q_v[i], k_v);
        }
        const float dot = _mm512_reduce_add_ps(acc);
        const float s = dot * scale;
        scores[k] = s;
        if (s > max_score) max_score = s;
    }

    // ── Step 2: softmax (K is small, scalar is fine) ──────────────────
    if (!any_valid) {
        for (int i = 0; i < 8; ++i) {
            __m256i z = _mm256_setzero_si256();
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(o_bh + i * 16), z);
        }
        return;
    }
    float denom = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        scores[k] = std::exp(scores[k] - max_score);
        denom += scores[k];
    }
    if (!(denom > 0.0f) || !std::isfinite(denom)) {
        for (int i = 0; i < 8; ++i) {
            __m256i z = _mm256_setzero_si256();
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(o_bh + i * 16), z);
        }
        return;
    }
    const float inv_denom = 1.0f / denom;
    for (int64_t k = 0; k < K; ++k) scores[k] *= inv_denom;

    // ── Step 3: V mul → 8 fp32 accumulators ───────────────────────────
    __m512 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = _mm512_setzero_ps();

    for (int64_t k = 0; k < K; ++k) {
        const int64_t hkv  = head_map_bh[k];
        const int64_t slot = ts_buf_bh[k];
        if (!valid_shadow_slot(slot, num_slots)) {
            continue;
        }
        const at::BFloat16* v_row =
            v_shadow + hkv * kshadow_head_stride + slot * D;
        const __m512 w_v = _mm512_set1_ps(scores[k]);

        // 8 chunks × 16 bf16 each.  bf16→fp32: zero-extend uint16 to uint32
        // then shift left 16 (bf16 lives in the upper half of fp32 bits).
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            __m256i bf = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(v_row + i * 16));
            __m512i u32  = _mm512_cvtepu16_epi32(bf);
            __m512i sh   = _mm512_slli_epi32(u32, 16);
            __m512  v_f  = _mm512_castsi512_ps(sh);
            acc[i] = _mm512_fmadd_ps(w_v, v_f, acc[i]);
        }
    }

    // ── Step 4: fp32 → bf16 + store ──────────────────────────────────
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        __m256bh out_bf = _mm512_cvtneps_pbh(acc[i]);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(o_bh + i * 16),
            reinterpret_cast<__m256i>(out_bf));
    }
}
#endif // JETFIRE_HAVE_AVX512_BF16


// Per-(b, h_q) compute kernel.  D is templated so the inner loops vectorize.
template <int64_t D>
static inline void cpu_topk_attn_one(
    const at::BFloat16* q_bh,                 // (D,)
    const at::BFloat16* k_shadow,             // (H_kv, num_slots, D)
    const at::BFloat16* v_shadow,             // same
    int64_t num_slots,
    const int64_t* head_map_bh,               // (K,)
    const int64_t* ts_buf_bh,                 // (K,)
    int64_t K,
    float scale,
    at::BFloat16* o_bh                        // (D,) [out]
) {
    thread_local std::vector<float> scores_storage;
    scores_storage.resize((size_t)K);
    float* scores = scores_storage.data();

    const int64_t kshadow_head_stride = num_slots * D;

    // Promote Q to fp32 once (small, D elements).
    float q_fp32[D];
    for (int64_t i = 0; i < D; ++i) {
        q_fp32[i] = static_cast<float>(q_bh[i]);
    }

    // Step 1: scores[k] = Q · K_topk[k] · scale
    float max_score = -INFINITY;
    bool any_valid = false;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t hkv  = head_map_bh[k];
        const int64_t slot = ts_buf_bh[k];
        if (!valid_shadow_slot(slot, num_slots)) {
            scores[k] = -INFINITY;
            continue;
        }
        any_valid = true;
        const at::BFloat16* k_row =
            k_shadow + hkv * kshadow_head_stride + slot * D;
        float dot = 0.0f;
        for (int64_t i = 0; i < D; ++i) {
            dot += q_fp32[i] * static_cast<float>(k_row[i]);
        }
        const float s = dot * scale;
        scores[k] = s;
        if (s > max_score) max_score = s;
    }

    // Step 2: softmax(scores)
    if (!any_valid) {
        for (int64_t i = 0; i < D; ++i) {
            o_bh[i] = static_cast<at::BFloat16>(0.0f);
        }
        return;
    }
    float denom = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        scores[k] = std::exp(scores[k] - max_score);
        denom += scores[k];
    }
    if (!(denom > 0.0f) || !std::isfinite(denom)) {
        for (int64_t i = 0; i < D; ++i) {
            o_bh[i] = static_cast<at::BFloat16>(0.0f);
        }
        return;
    }
    const float inv_denom = 1.0f / denom;
    for (int64_t k = 0; k < K; ++k) {
        scores[k] *= inv_denom;
    }

    // Step 3: o_new[b,h_q,:] = sum_k scores[k] · V_topk[k]
    float out_fp32[D];
    for (int64_t i = 0; i < D; ++i) out_fp32[i] = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t hkv  = head_map_bh[k];
        const int64_t slot = ts_buf_bh[k];
        if (!valid_shadow_slot(slot, num_slots)) {
            continue;
        }
        const at::BFloat16* v_row =
            v_shadow + hkv * kshadow_head_stride + slot * D;
        const float w = scores[k];
        for (int64_t i = 0; i < D; ++i) {
            out_fp32[i] += w * static_cast<float>(v_row[i]);
        }
    }
    for (int64_t i = 0; i < D; ++i) {
        o_bh[i] = static_cast<at::BFloat16>(out_fp32[i]);
    }
}

struct CpuTopkParams {
    torch::Tensor q;
    torch::Tensor k_shadow;
    torch::Tensor v_shadow;
    torch::Tensor head_map;
    torch::Tensor ts_buf;
    torch::Tensor o_new;
    const at::BFloat16* q_p = nullptr;
    const at::BFloat16* ks_p = nullptr;
    const at::BFloat16* vs_p = nullptr;
    const int64_t* hm_p = nullptr;
    const int64_t* ts_p = nullptr;
    at::BFloat16* o_p = nullptr;
    int64_t B = 0;
    int64_t H_q = 0;
    int64_t D = 0;
    int64_t H_kv = 0;
    int64_t num_slots = 0;
    int64_t K = 0;
    int64_t total = 0;
    float scale = 0.0f;
    std::atomic<int64_t>* flag_ptr = nullptr;
};

static CpuTopkParams prepare_cpu_topk_params(
    torch::Tensor q,
    torch::Tensor k_shadow,
    torch::Tensor v_shadow,
    torch::Tensor head_map,
    torch::Tensor ts_buf,
    torch::Tensor o_new,
    double scale_d,
    c10::optional<torch::Tensor> done_flag = c10::nullopt)
{
    TORCH_CHECK(q.is_cpu() && k_shadow.is_cpu() && v_shadow.is_cpu(),
                "all tensors must be CPU");
    TORCH_CHECK(head_map.is_cpu() && ts_buf.is_cpu() && o_new.is_cpu());

    CpuTopkParams p;
    if (done_flag.has_value()) {
        TORCH_CHECK(done_flag->is_cpu() && done_flag->numel() == 1
                    && done_flag->scalar_type() == torch::kInt64,
                    "done_flag must be a 1-element int64 CPU tensor");
        p.flag_ptr = reinterpret_cast<std::atomic<int64_t>*>(
            done_flag->data_ptr<int64_t>());
    }

    TORCH_CHECK(q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_shadow.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(v_shadow.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(o_new.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(head_map.scalar_type() == torch::kInt64);
    TORCH_CHECK(ts_buf.scalar_type() == torch::kInt64);
    TORCH_CHECK(q.dim() == 3 && k_shadow.dim() == 3 && v_shadow.dim() == 3);
    TORCH_CHECK(head_map.dim() == 3 && ts_buf.dim() == 3 && o_new.dim() == 3);

    p.q        = q.contiguous();
    p.k_shadow = k_shadow.contiguous();
    p.v_shadow = v_shadow.contiguous();
    p.head_map = head_map.contiguous();
    p.ts_buf   = ts_buf.contiguous();
    p.o_new    = o_new.contiguous();

    p.B         = p.q.size(0);
    p.H_q       = p.q.size(1);
    p.D         = p.q.size(2);
    p.H_kv      = p.k_shadow.size(0);
    p.num_slots = p.k_shadow.size(1);
    p.K         = p.head_map.size(2);
    p.total     = p.B * p.H_q;
    p.scale     = static_cast<float>(scale_d);

    TORCH_CHECK(p.k_shadow.size(2) == p.D && p.v_shadow.size(2) == p.D);
    TORCH_CHECK(p.v_shadow.size(0) == p.H_kv && p.v_shadow.size(1) == p.num_slots);
    TORCH_CHECK(p.head_map.sizes() == p.ts_buf.sizes());
    TORCH_CHECK(p.head_map.size(0) == p.B && p.head_map.size(1) == p.H_q);
    TORCH_CHECK(p.o_new.size(0) == p.B && p.o_new.size(1) == p.H_q
                && p.o_new.size(2) == p.D);

    p.q_p  = p.q.data_ptr<at::BFloat16>();
    p.ks_p = p.k_shadow.data_ptr<at::BFloat16>();
    p.vs_p = p.v_shadow.data_ptr<at::BFloat16>();
    p.hm_p = p.head_map.data_ptr<int64_t>();
    p.ts_p = p.ts_buf.data_ptr<int64_t>();
    p.o_p  = p.o_new.data_ptr<at::BFloat16>();
    return p;
}

static inline void run_cpu_topk_attn_rows_d128(
    const CpuTopkParams& p,
    int wid,
    int n_workers)
{
    const int64_t start = (p.total * wid) / n_workers;
    const int64_t end   = (p.total * (wid + 1)) / n_workers;
#if JETFIRE_HAVE_AVX512_BF16
    const bool use_simd = jetfire_cpu_attn_use_simd();
    const bool use_v2   = jetfire_cpu_attn_use_v2();
    if (use_simd) {
        for (int64_t i = start; i < end; ++i) {
            const int64_t b   = i / p.H_q;
            const int64_t h_q = i % p.H_q;
            const at::BFloat16* q_bh = p.q_p + (b * p.H_q + h_q) * 128;
            const int64_t* hm_bh = p.hm_p + (b * p.H_q + h_q) * p.K;
            const int64_t* ts_bh = p.ts_p + (b * p.H_q + h_q) * p.K;
            at::BFloat16* o_bh = p.o_p + (b * p.H_q + h_q) * 128;
            if (use_v2) {
                cpu_topk_attn_one_avx512bf16_d128_v2(
                    q_bh, p.ks_p, p.vs_p, p.num_slots,
                    hm_bh, ts_bh, p.K, p.scale, o_bh
                );
            } else {
                cpu_topk_attn_one_avx512bf16_d128(
                    q_bh, p.ks_p, p.vs_p, p.num_slots,
                    hm_bh, ts_bh, p.K, p.scale, o_bh
                );
            }
            if (jetfire_cpu_attn_sanitize_output()
                    && bf16_row_has_nonfinite(o_bh, 128)) {
                g_cpu_sanitize_count.fetch_add(1, std::memory_order_acq_rel);
                maybe_log_sanitize_row(
                    wid, b, h_q, q_bh, p.ks_p, p.vs_p, p.num_slots,
                    hm_bh, ts_bh, p.K, 128, o_bh);
                // A nonfinite correction poisons all following layers during
                // CUDA-graph replay.  Recompute only the bad row through the
                // scalar reference path; healthy SIMD rows stay on the fast
                // path.
                cpu_topk_attn_one<128>(
                    q_bh, p.ks_p, p.vs_p, p.num_slots,
                    hm_bh, ts_bh, p.K, p.scale, o_bh
                );
                if (bf16_row_has_nonfinite(o_bh, 128)) {
                    zero_bf16_row(o_bh, 128);
                }
            }
        }
        return;
    }
#endif
    for (int64_t i = start; i < end; ++i) {
        const int64_t b   = i / p.H_q;
        const int64_t h_q = i % p.H_q;
        const at::BFloat16* q_bh = p.q_p + (b * p.H_q + h_q) * 128;
        const int64_t* hm_bh = p.hm_p + (b * p.H_q + h_q) * p.K;
        const int64_t* ts_bh = p.ts_p + (b * p.H_q + h_q) * p.K;
        at::BFloat16* o_bh = p.o_p + (b * p.H_q + h_q) * 128;
        cpu_topk_attn_one<128>(
            q_bh, p.ks_p, p.vs_p, p.num_slots,
            hm_bh, ts_bh, p.K, p.scale, o_bh
        );
        if (jetfire_cpu_attn_sanitize_output()
                && bf16_row_has_nonfinite(o_bh, 128)) {
            g_cpu_sanitize_count.fetch_add(1, std::memory_order_acq_rel);
            maybe_log_sanitize_row(
                wid, b, h_q, q_bh, p.ks_p, p.vs_p, p.num_slots,
                hm_bh, ts_bh, p.K, 128, o_bh);
            zero_bf16_row(o_bh, 128);
        }
    }
}


static void cpu_topk_attn(
    torch::Tensor q,           // (B, H_q, D) bf16 CPU
    torch::Tensor k_shadow,    // (H_kv, num_slots, D) bf16 CPU
    torch::Tensor v_shadow,    // (H_kv, num_slots, D) bf16 CPU
    torch::Tensor head_map,    // (B, H_q, K) int64 CPU
    torch::Tensor ts_buf,      // (B, H_q, K) int64 CPU
    torch::Tensor o_new,       // (B, H_q, D) bf16 CPU [out]
    double scale_d,
    c10::optional<torch::Tensor> done_flag = c10::nullopt)
{
    CpuTopkParams p = prepare_cpu_topk_params(
        std::move(q), std::move(k_shadow), std::move(v_shadow),
        std::move(head_map), std::move(ts_buf), std::move(o_new),
        scale_d, done_flag
    );

    // Dispatch on D == 128 (common).  Other D values go through a generic
    // path that disables D-templating; perf may be lower.
    if (p.D == 128) {
        // Explicit OMP pragma instead of at::parallel_for: at::parallel_for
        // honours the global at::get_num_threads() set by torch, but vLLM's
        // engine subprocess starts with OMP_NUM_THREADS=1 which forces it
        // to a single core regardless of grain size.  Pinning num_threads
        // here makes the work span all chosen cores from the worker thread.
        const int nthr = jetfire_cpu_attn_threads();

        // ── Custom thread pool path (env JETFIRE_CUSTOM_POOL=1, default) ──
        // Replaces libgomp's `#pragma omp parallel for` to avoid the ~50 µs
        // wake-up cost of pthread_cond_broadcast + futex_wait.  Per-worker
        // direct dispatch via atomic flags; spin then futex.
        const bool use_pool = jetfire_cpu_attn_use_pool();

#if JETFIRE_HAVE_AVX512_BF16
        const bool use_simd = jetfire_cpu_attn_use_simd();

        if (use_pool && use_simd) {
            // Lazy-init pool with `nthr` workers.  If the pool is already
            // initialized at a different size (e.g. a microbench is sweeping
            // thread counts), tear down + re-init.  In production this only
            // happens once.
            auto& pool = jetfire_pool::global_pool();
            const bool pin = jetfire_cpu_attn_pin_workers();
            if (pool.is_initialized() && pool.n() != nthr) {
                pool.shutdown();
            }
            if (!pool.is_initialized()) {
                // Pinning is opt-in: JETFIRE_CPU_ATTN_PIN=1 → bind each worker
                // to core_id = worker_id (0..nthr-1).  Useful when nthr <=
                // num_physical_cores; with SMT enabled at 192 logical cores
                // and nthr ≤ 96, this keeps every worker on its own physical
                // core and avoids two workers sharing an FPU pipe.
                pool.init(nthr, /*pin_to_cores=*/pin);
            }

            // Pack pointers + shapes for the worker (pointer-only, lightweight).
            auto worker_fn = +[](int wid, int n_workers, void* ctx_void) {
                const auto* params = static_cast<const CpuTopkParams*>(ctx_void);
                run_cpu_topk_attn_rows_d128(*params, wid, n_workers);
            };
            pool.dispatch_and_wait(worker_fn, &p);

            // Flush write-combined stores to PCIe before signaling completion
            // (matches the OMP path's _mm_sfence at the end of cpu_topk_attn).
            _mm_sfence();
            if (p.flag_ptr) {
                p.flag_ptr->store(1, std::memory_order_release);
                // flag_ptr can live in cudaHostAllocWriteCombined memory.
                // Flush the doorbell store itself before a GPU stream waits
                // on the mapped device pointer.
                _mm_sfence();
            }
            return;
        }
        // OMP fallback for SIMD when pool disabled (env JETFIRE_CUSTOM_POOL=0).
        if (use_simd) {
            #pragma omp parallel for num_threads(nthr) schedule(static)
            for (int64_t wid = 0; wid < nthr; ++wid) {
                run_cpu_topk_attn_rows_d128(p, (int)wid, nthr);
            }
            return;
        }
#endif
        #pragma omp parallel for num_threads(nthr) schedule(static)
        for (int64_t wid = 0; wid < nthr; ++wid) {
            run_cpu_topk_attn_rows_d128(p, (int)wid, nthr);
        }
    } else {
        TORCH_CHECK(false, "cpu_topk_attn currently only supports D=128");
    }

    // Flush pending CPU stores so writes to write-combined / mapped pinned
    // o_new are visible to PCIe before the worker signals completion to
    // MAIN.  Without this, a subsequent GPU kernel that reads o_new via a
    // mapped pointer could observe stale data.  ~1 cycle, harmless when
    // o_new is regular pageable / pinned memory.
    _mm_sfence();
    if (p.flag_ptr) {
        p.flag_ptr->store(1, std::memory_order_release);
        // flag_ptr may be write-combined mapped pinned memory; release-store
        // orders CPU memory but does not by itself flush WC buffers to PCIe.
        _mm_sfence();
    }
}


// ── Mapped pinned-memory allocator ─────────────────────────────────────
// cudaHostAllocMapped: CPU and GPU see the SAME physical buffer through
// different virtual addresses.  CPU writes are forwarded to PCIe; GPU
// reads reach the host buffer over PCIe.  Eliminates the explicit H2D
// copy of o_new (~64 µs/layer of cuMemcpyAsync fixed cost) at the price
// of slightly slower per-element GPU access for the o_new read inside
// the blend kernel (which is read only once).
//
// Returns (host_tensor, device_tensor) sharing the same physical pages.
// `host_tensor` is a pinned CPU tensor; `device_tensor` is a CUDA tensor
// whose data_ptr() is the CUDA-mapped device pointer of the same memory.
// Caller must keep both alive together — destroying host_tensor frees
// the underlying cudaHostAlloc, invalidating device_tensor.
static py::tuple alloc_mapped_pinned_pair(
    std::vector<int64_t> shape,
    py::object dtype_obj)
{
    // Resolve dtype + bytes.
    auto dtype = torch::python::detail::py_object_to_dtype(dtype_obj);
    int64_t numel = 1;
    for (auto d : shape) numel *= d;
    size_t elem_size = c10::elementSize(dtype);
    size_t bytes = (size_t)numel * elem_size;
    TORCH_CHECK(bytes > 0, "alloc_mapped_pinned_pair: empty shape");

    // Make sure we have a current CUDA context bound to this thread —
    // cudaHostAlloc requires it.
    int dev = -1;
    cudaError_t err = cudaGetDevice(&dev);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));

    void* host_ptr = nullptr;
    err = cudaHostAlloc(&host_ptr, bytes,
                        cudaHostAllocMapped | cudaHostAllocWriteCombined);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostAlloc(mapped+wc, ", bytes, "B) failed: ",
                cudaGetErrorString(err));

    void* dev_ptr = nullptr;
    err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    if (err != cudaSuccess) {
        cudaFreeHost(host_ptr);
        TORCH_CHECK(false, "cudaHostGetDevicePointer failed: ",
                    cudaGetErrorString(err));
    }

    // host_tensor owns the cudaHostAlloc lifetime — its deleter frees the
    // underlying buffer (which also invalidates dev_ptr).  device_tensor
    // borrows; its deleter is a no-op.
    auto host_opts = torch::TensorOptions()
        .dtype(dtype).device(torch::kCPU).pinned_memory(true);
    auto host_tensor = torch::from_blob(
        host_ptr, shape,
        [host_ptr](void*) { cudaFreeHost(host_ptr); },
        host_opts);

    auto dev_opts = torch::TensorOptions()
        .dtype(dtype).device(torch::kCUDA, dev);
    auto device_tensor = torch::from_blob(
        dev_ptr, shape,
        [](void*) {},  // no-op deleter
        dev_opts);

    return py::make_tuple(host_tensor, device_tensor);
}


// ── Async CPU correction executor ─────────────────────────────────────
// Single-layer executor for the INT4 decode correction path:
//   submit_cpu_topk_attn_async(...)
//     -> publish one in-flight job directly to N fixed worker threads
//   wait_for_flag(flag)
//     -> main thread blocks only on the completion flag
//
// This replaces the previous nesting:
//   main thread -> persistent dispatcher thread -> custom pool -> workers
// with:
//   main thread -> async executor workers
//
// The goal is not to change the algorithm.  It only removes the extra
// dispatch layer and the dispatcher-side ``dispatch_and_wait``.  That
// shrinks CPU orchestration overhead and removes the deadlock-prone
// interaction between the outer persistent worker and the inner custom
// thread pool.

namespace {

class AsyncCpuTopkExecutor {
public:
    AsyncCpuTopkExecutor() = default;

    ~AsyncCpuTopkExecutor() {
        shutdown();
    }

    void warmup(int n_workers) {
        TORCH_CHECK(n_workers > 0, "AsyncCpuTopkExecutor requires n_workers > 0");
        std::unique_lock<std::mutex> lk(mu_);
        TORCH_CHECK(!inflight_, "cannot resize async CPU executor while a job is in flight");
        if ((int)threads_.size() == n_workers) return;
        shutdown_locked(lk);
        stop_ = false;
        for (int wid = 0; wid < n_workers; ++wid) {
            threads_.emplace_back(&AsyncCpuTopkExecutor::worker_main, this, wid, n_workers);
        }
    }

    void submit(CpuTopkParams&& params) {
        TORCH_CHECK(params.flag_ptr != nullptr,
                    "async CPU executor requires a completion flag");
        TORCH_CHECK(params.D == 128,
                    "async CPU executor currently only supports D=128");
        warmup(jetfire_cpu_attn_threads());
        {
            std::lock_guard<std::mutex> lk(mu_);
            TORCH_CHECK(!inflight_,
                        "async CPU executor only supports one in-flight job");
            current_ = std::move(params);
            inflight_ = true;
            done_count_.store(0, std::memory_order_release);
            ++published_gen_;
            if (jetfire_cpu_attn_debug()) {
                std::fprintf(stderr,
                             "[jetfire][cpu_async] submit gen=%llu total=%lld K=%lld nworkers=%zu flag=%p\n",
                             (unsigned long long)published_gen_,
                             (long long)current_.total,
                             (long long)current_.K,
                             threads_.size(),
                             (void*)current_.flag_ptr);
                std::fflush(stderr);
            }
        }
        cv_.notify_all();
    }

    void shutdown() {
        std::unique_lock<std::mutex> lk(mu_);
        shutdown_locked(lk);
    }

private:
    void shutdown_locked(std::unique_lock<std::mutex>& lk) {
        if (threads_.empty()) return;
        stop_ = true;
        inflight_ = false;
        lk.unlock();
        cv_.notify_all();
        for (auto& th : threads_) {
            if (th.joinable()) th.join();
        }
        lk.lock();
        threads_.clear();
        stop_ = false;
        published_gen_ = 0;
        current_ = CpuTopkParams{};
    }

    void worker_main(int wid, int n_workers) {
        uint64_t local_gen = 0;
        while (true) {
            uint64_t run_gen = 0;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] {
                    return stop_ || published_gen_ != local_gen;
                });
                if (stop_) return;
                run_gen = published_gen_;
                local_gen = run_gen;
            }
            if (jetfire_cpu_attn_debug() && wid == 0) {
                std::fprintf(stderr,
                             "[jetfire][cpu_async] worker_start gen=%llu total=%lld\n",
                             (unsigned long long)run_gen,
                             (long long)current_.total);
                std::fflush(stderr);
            }

            run_cpu_topk_attn_rows_d128(current_, wid, n_workers);

            const int finished = done_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (finished == n_workers) {
                auto* flag_ptr = current_.flag_ptr;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    inflight_ = false;
                }
                _mm_sfence();
                flag_ptr->store(1, std::memory_order_release);
                if (jetfire_cpu_attn_debug()) {
                    std::fprintf(stderr,
                                 "[jetfire][cpu_async] complete gen=%llu finished=%d flag=%p\n",
                                 (unsigned long long)run_gen,
                                 finished,
                                 (void*)flag_ptr);
                    std::fflush(stderr);
                }
                syscall(SYS_futex, reinterpret_cast<int*>(flag_ptr),
                        FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
            }
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::thread> threads_;
    CpuTopkParams current_;
    std::atomic<int> done_count_{0};
    uint64_t published_gen_ = 0;
    bool stop_ = false;
    bool inflight_ = false;
};


AsyncCpuTopkExecutor& global_async_cpu_topk_executor() {
    static AsyncCpuTopkExecutor exec;
    return exec;
}

}  // anonymous namespace


// Submit a cpu_topk_attn job to the async executor.  Returns immediately.
// ``flag`` must be a 1-element int64 CPU tensor; the last finishing worker
// atomic-release-stores 1 after compute + sfence.
static void submit_cpu_topk_attn_async(
    torch::Tensor q,
    torch::Tensor k_shadow,
    torch::Tensor v_shadow,
    torch::Tensor head_map,
    torch::Tensor ts_buf,
    torch::Tensor o_new,
    double scale,
    torch::Tensor flag)
{
    TORCH_CHECK(flag.is_cpu() && flag.numel() == 1
                && flag.scalar_type() == torch::kInt64,
                "flag must be a 1-element int64 CPU tensor");
    CpuTopkParams params = prepare_cpu_topk_params(
        std::move(q), std::move(k_shadow), std::move(v_shadow),
        std::move(head_map), std::move(ts_buf), std::move(o_new),
        scale, flag
    );
    global_async_cpu_topk_executor().submit(std::move(params));
}


// Spin-then-futex_wait.  Caller must have submitted a job whose worker
// will store 1 to this flag with release ordering AND issue a
// futex_wake on the flag.  GIL released by the pybind ``call_guard``.
//
// The old implementation fell back to ``std::this_thread::yield`` in a
// tight loop, which on Linux is essentially a no-op when no other thread
// is runnable on the same core — the dispatcher inside the persistent
// worker was busy-spinning on its own pool, the pool workers might be
// blocked on CPU contention from a neighbour process, and main would
// poll forever holding a core.  futex_wait blocks the kernel scheduler
// off this thread until the worker actually wakes us.
static void wait_for_flag(torch::Tensor flag) {
    TORCH_CHECK(flag.is_cpu() && flag.numel() == 1
                && flag.scalar_type() == torch::kInt64,
                "flag must be a 1-element int64 CPU tensor");
    auto* fp = reinterpret_cast<std::atomic<int64_t>*>(
        flag.data_ptr<int64_t>());
    if (jetfire_cpu_attn_debug()) {
        std::fprintf(stderr, "[jetfire][cpu_async] wait_enter flag=%p\n", (void*)fp);
        std::fflush(stderr);
    }
    // Brief spin first.  Typical work is 150-1000 µs but the wait may
    // already be satisfied if main thread did a chunk of GPU launches
    // between submit and wait.
    for (int i = 0; i < 1024; ++i) {
        if (fp->load(std::memory_order_acquire) != 0) {
            if (jetfire_cpu_attn_debug()) {
                std::fprintf(stderr, "[jetfire][cpu_async] wait_exit_spin flag=%p\n", (void*)fp);
                std::fflush(stderr);
            }
            return;
        }
    }
    // Slow path: futex_wait on the low 32 bits of the int64 flag.  The
    // flag only ever transitions 0 → 1, both fit in 32 bits; on x86
    // little-endian the int64's address is the low-32-bit address.
    int* fw32 = reinterpret_cast<int*>(fp);
    while (true) {
        int cur = __atomic_load_n(fw32, __ATOMIC_ACQUIRE);
        if (cur != 0) {
            if (jetfire_cpu_attn_debug()) {
                std::fprintf(stderr, "[jetfire][cpu_async] wait_exit_futex flag=%p\n", (void*)fp);
                std::fflush(stderr);
            }
            return;
        }
        // Block until value changes from 0.  Returns on wake, EAGAIN
        // (value already != 0), or EINTR — loop re-checks the load.
        syscall(SYS_futex, fw32, FUTEX_WAIT_PRIVATE, 0,
                nullptr, nullptr, 0);
    }
}


// Eagerly create the async worker set.  Optional — submit will lazy-create it
// on first use.  Useful at engine init so the first decode doesn't pay the
// thread spawn cost.
static void persistent_worker_warmup() {
    global_async_cpu_topk_executor().warmup(jetfire_cpu_attn_threads());
}


// ── Stream-level CPU/GPU sync primitives (CUDA-graph compatible) ──────
// These wrappers let the CPU/GPU handshake be captured into a CUDA graph
// node instead of going through host-side Python/C++ calls:
//
//   stream_write_value64(stream, flag_dev, 0)
//     -> cudaStreamWriteValue64; resets the doorbell, queued on stream.
//
//   stream_launch_host_func(stream, ctx)
//     -> cudaLaunchHostFunc; the callback runs on the stream-callback
//        thread when the GPU reaches this point on replay.  Callback
//        MUST NOT issue CUDA APIs; it can dispatch CPU work to our
//        worker pool and return immediately (a few µs).
//
//   stream_wait_value64(stream, flag_dev, 1)
//     -> cudaStreamWaitValue64 with EQ; GPU stream stalls until the CPU
//        worker writes 1 to the mapped-pinned flag (release-store +
//        _mm_sfence visible to GPU over PCIe).
//
// The flag MUST be allocated via ``alloc_mapped_pinned_pair`` so the
// CPU-side store is visible to the GPU through the device-pointer
// mapping.  Passing a normal pinned tensor's host_ptr will NOT work.

// Driver-API stream value ops (cudaStreamWaitValue64 isn't exported in the
// runtime headers shipped with the wheel-side CUDA 12.x toolkit; the driver
// API equivalents are layout-compatible and capturable in CUDA graphs.).
static void stream_write_value64(
    int64_t stream_id,
    torch::Tensor flag_device,
    int64_t value)
{
    TORCH_CHECK(flag_device.is_cuda(),
                "flag_device must be a CUDA tensor (use the device half of "
                "alloc_mapped_pinned_pair).");
    TORCH_CHECK(flag_device.scalar_type() == torch::kInt64,
                "flag_device must be int64.");
    TORCH_CHECK(flag_device.numel() == 1,
                "flag_device must be 1-element.");
    CUstream stream = reinterpret_cast<CUstream>(stream_id);
    CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>(
        flag_device.data_ptr<int64_t>());
    CUresult err = cuStreamWriteValue64(
        stream, dev_ptr, (cuuint64_t)value, /*flags*/0);
    TORCH_CHECK(err == CUDA_SUCCESS,
                "cuStreamWriteValue64 failed: ", (int)err);
}


static void stream_wait_value64(
    int64_t stream_id,
    torch::Tensor flag_device,
    int64_t value)
{
    TORCH_CHECK(flag_device.is_cuda(),
                "flag_device must be a CUDA tensor (use the device half of "
                "alloc_mapped_pinned_pair).");
    TORCH_CHECK(flag_device.scalar_type() == torch::kInt64,
                "flag_device must be int64.");
    TORCH_CHECK(flag_device.numel() == 1,
                "flag_device must be 1-element.");
    CUstream stream = reinterpret_cast<CUstream>(stream_id);
    CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>(
        flag_device.data_ptr<int64_t>());
    // The CPU worker writes o_new (and then the doorbell) from host threads.
    // Some CUDA stacks expose CU_STREAM_WAIT_VALUE_FLUSH, but this driver
    // returns CUDA_ERROR_NOT_SUPPORTED for graph-captured waits with FLUSH.
    // Keep EQ as the default and make FLUSH an explicit experiment only.
    unsigned int flags = CU_STREAM_WAIT_VALUE_EQ;
#ifdef CU_STREAM_WAIT_VALUE_FLUSH
    if (std::getenv("JETFIRE_STREAM_WAIT_VALUE_FLUSH") != nullptr) {
        flags |= CU_STREAM_WAIT_VALUE_FLUSH;
    }
#endif
    CUresult err = cuStreamWaitValue64(
        stream, dev_ptr, (cuuint64_t)value, flags);
    TORCH_CHECK(err == CUDA_SUCCESS,
                "cuStreamWaitValue64 failed: ", (int)err);
}


// Test-only host_func target: takes a pointer to a host-side counter
// (must be alive for the lifetime of the callback) and increments it.
// Used by Phase-1 unit tests to verify cudaLaunchHostFunc plumbing.
struct TestHostFuncCtx {
    std::atomic<int64_t>* counter;
};

static void CUDART_CB test_host_func_callback(void* user_data) {
    auto* ctx = static_cast<TestHostFuncCtx*>(user_data);
    ctx->counter->fetch_add(1, std::memory_order_acq_rel);
}

// Storage for test contexts — caller owns the memory via a returned int handle.
// Each handle index maps to a heap-allocated TestHostFuncCtx; freed on
// process teardown.  Adequate for tests; not used by the production path.
static std::mutex g_test_ctx_mu;
static std::vector<std::unique_ptr<TestHostFuncCtx>> g_test_ctxs;

// Create a long-lived test ctx referencing the (already-pinned) counter.
// Returns an opaque int handle; pass to stream_launch_test_host_func.
static int64_t create_test_host_ctx(torch::Tensor counter_host) {
    TORCH_CHECK(counter_host.is_cpu(), "counter must be a CPU tensor");
    TORCH_CHECK(counter_host.is_pinned(), "counter must be pinned");
    TORCH_CHECK(counter_host.scalar_type() == torch::kInt64,
                "counter must be int64");
    TORCH_CHECK(counter_host.numel() == 1, "counter must be 1-element");
    auto ctx = std::make_unique<TestHostFuncCtx>();
    ctx->counter = reinterpret_cast<std::atomic<int64_t>*>(
        counter_host.data_ptr<int64_t>());
    std::lock_guard<std::mutex> lk(g_test_ctx_mu);
    g_test_ctxs.emplace_back(std::move(ctx));
    return reinterpret_cast<int64_t>(g_test_ctxs.back().get());
}

static void stream_launch_test_host_func(int64_t stream_id, int64_t ctx_handle) {
    TORCH_CHECK(ctx_handle != 0, "ctx_handle is null");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    void* ud = reinterpret_cast<void*>(ctx_handle);
    cudaError_t err = cudaLaunchHostFunc(stream, test_host_func_callback, ud);
    TORCH_CHECK(err == cudaSuccess,
                "cudaLaunchHostFunc failed: ", cudaGetErrorString(err));
}


// ── Production cpu_topk_attn submit via host_func ──────────────────────
// A long-lived ctx that holds raw pointers + dims for one decode setup.
// The ctx is created once (typically at warmup) and reused across all
// graph replays; only the buffer CONTENTS change between layers/tokens
// (the pointers stay the same because warmup_decode_buffers pre-allocates
// max-shape pinned/mapped buffers).
struct CpuTopkSubmitCtx {
    // Raw pointers (extracted at create-time; tensors below keep them alive)
    const at::BFloat16* q_p = nullptr;
    const at::BFloat16* ks_p = nullptr;
    const at::BFloat16* vs_p = nullptr;
    const int64_t* hm_p = nullptr;
    const int64_t* ts_p = nullptr;
    at::BFloat16* o_p = nullptr;
    std::atomic<int64_t>* flag_p = nullptr;  // mapped pinned host_ptr
    CUdeviceptr signal_dev = 0;              // optional device-memory doorbell
    CUstream signal_stream = nullptr;        // worker writes signal_dev here
    int signal_device_index = -1;
    cudaStream_t join_stream = nullptr;      // side-stream host join
    cudaEvent_t join_submit_event = nullptr;
    cudaEvent_t join_done_event = nullptr;

    // Dims
    int64_t B = 0;
    int64_t H_q = 0;
    int64_t K = 0;
    int64_t num_slots = 0;
    int64_t total = 0;  // B * H_q
    int D = 128;
    float scale = 0.0f;

    // Workers signal completion to ``done_count``.  The default graph path
    // observes this through a side-stream host join; device/mapped doorbells
    // are retained only for explicit A/B modes.
    alignas(64) std::atomic<int> done_count{0};
    alignas(64) std::atomic<int64_t> submit_t0_ns{0};
    alignas(64) std::atomic<int64_t> active_call{-1};
    alignas(64) std::atomic<int64_t> active_generation{0};
    alignas(64) std::atomic<int64_t> done_generation{0};
    alignas(64) std::atomic<int64_t> next_wait_generation{0};
    std::atomic<int> n_workers{0};

    // Keep tensors alive for ctx lifetime.
    torch::Tensor q_ref, ks_ref, vs_ref, hm_ref, ts_ref, o_ref, flag_ref;
    torch::Tensor signal_ref;

    ~CpuTopkSubmitCtx() {
        if (signal_stream != nullptr) {
            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(signal_stream));
            signal_stream = nullptr;
        }
        if (join_stream != nullptr) {
            cudaStreamDestroy(join_stream);
            join_stream = nullptr;
        }
        if (join_submit_event != nullptr) {
            cudaEventDestroy(join_submit_event);
            join_submit_event = nullptr;
        }
        if (join_done_event != nullptr) {
            cudaEventDestroy(join_done_event);
            join_done_event = nullptr;
        }
    }
};

struct ShadowAppendSubmitCtx {
    const at::BFloat16* k_append_p = nullptr;
    const at::BFloat16* v_append_p = nullptr;
    const int64_t* slots_p = nullptr;
    at::BFloat16* k_shadow_p = nullptr;
    at::BFloat16* v_shadow_p = nullptr;

    int64_t B = 0;
    int64_t H_kv = 0;
    int64_t D = 0;
    int64_t num_slots = 0;

    torch::Tensor k_append_ref, v_append_ref, slots_ref;
    torch::Tensor k_shadow_ref, v_shadow_ref;
};

// Owned storage for all ctxes created via create_cpu_topk_submit_ctx.
// Keyed by raw pointer (returned as int64 to Python).  Freed on shutdown
// (process teardown via static destructor); intentionally simple — typical
// use creates 1-few ctxes per process lifetime.
static std::mutex g_submit_ctx_mu;
static std::vector<std::unique_ptr<CpuTopkSubmitCtx>> g_submit_ctxs;
static std::vector<std::unique_ptr<ShadowAppendSubmitCtx>> g_shadow_append_ctxs;
static std::atomic<int64_t> g_graph_submit_count{0};
static std::atomic<int64_t> g_shadow_append_count{0};
static std::atomic<int64_t> g_shadow_append_log_count{0};
static std::atomic<int64_t> g_cpu_submit_log_count{0};
static std::atomic<int64_t> g_cpu_done_log_count{0};
static std::atomic<int64_t> g_cpu_scan_log_count{0};
static std::atomic<int64_t> g_graph_topk_jobs{0};
static std::atomic<int64_t> g_graph_topk_submit_ns{0};
static std::atomic<int64_t> g_graph_topk_done_ns{0};
static std::atomic<int64_t> g_graph_topk_submit_max_ns{0};
static std::atomic<int64_t> g_graph_topk_done_max_ns{0};
static std::atomic<int64_t> g_graph_shadow_append_ns{0};
static std::atomic<int64_t> g_graph_shadow_append_max_ns{0};
static std::atomic<int64_t> g_graph_last_submit_call{-1};
static std::atomic<int64_t> g_graph_last_submit_ctx{0};
static std::atomic<int64_t> g_graph_last_submit_ctx_done{-1};
static std::atomic<int64_t> g_graph_last_submit_pool_done{-1};
static std::atomic<int64_t> g_graph_last_submit_flag{-1};
static std::atomic<int64_t> g_graph_last_submit_workers{-1};
static std::atomic<int64_t> g_graph_last_submit_mode{0};  // 0 async, 1 sync, 2 inline
static std::atomic<int64_t> g_graph_last_done_call{-1};
static std::atomic<int64_t> g_graph_last_done_ctx{0};
static std::atomic<int64_t> g_graph_last_done_prev{-1};
static std::atomic<int64_t> g_graph_last_done_flag{-1};
static std::atomic<int64_t> g_graph_join_count{0};
static std::atomic<int64_t> g_graph_join_ns{0};
static std::atomic<int64_t> g_graph_join_max_ns{0};
static std::atomic<int64_t> g_graph_last_join_call{-1};
static std::atomic<int64_t> g_graph_last_join_ctx{0};
static std::atomic<int64_t> g_graph_last_join_ctx_done{-1};
static std::atomic<int64_t> g_graph_last_join_pool_done{-1};

static inline int64_t now_steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline void atomic_add_max(std::atomic<int64_t>& max_value,
                                  int64_t candidate) {
    int64_t prev = max_value.load(std::memory_order_relaxed);
    while (candidate > prev
           && !max_value.compare_exchange_weak(
               prev, candidate,
               std::memory_order_acq_rel,
               std::memory_order_relaxed)) {
    }
}

static inline int64_t graph_heartbeat_every() {
    const char* s = std::getenv("JETFIRE_GRAPH_HEARTBEAT_EVERY");
    if (!s) return 0;
    return std::max<int64_t>(0, std::atoll(s));
}

static inline void futex_wake_atomic_int(std::atomic<int>* addr, int n = 1) {
    syscall(SYS_futex, reinterpret_cast<int*>(addr), FUTEX_WAKE_PRIVATE, n,
            nullptr, nullptr, 0);
}

static inline void futex_wait_atomic_int(std::atomic<int>* addr,
                                         int expected) {
    syscall(SYS_futex, reinterpret_cast<int*>(addr), FUTEX_WAIT_PRIVATE,
            expected, nullptr, nullptr, 0);
}

static void wait_cpu_topk_ctx_done(CpuTopkSubmitCtx* ctx,
                                   int64_t generation,
                                   int n_workers) {
    if (n_workers <= 0) return;
    static const int WAIT_SPIN = []() {
        const char* s = std::getenv("JETFIRE_GRAPH_JOIN_SPIN");
        return s ? std::atoi(s) : 200000;
    }();
    static const int64_t WAIT_WARN_MS = []() {
        const char* s = std::getenv("JETFIRE_GRAPH_WAIT_WARN_MS");
        return s ? std::max<int64_t>(0, std::atoll(s)) : 0;
    }();
    int64_t next_warn_ns = 0;
    if (WAIT_WARN_MS > 0) {
        next_warn_ns = now_steady_ns()
            + WAIT_WARN_MS * 1000LL * 1000LL;
    }

    for (int i = 0; i < WAIT_SPIN; ++i) {
        if (ctx->done_generation.load(std::memory_order_acquire) >= generation
            && ctx->done_count.load(std::memory_order_acquire) >= n_workers) {
            return;
        }
        __builtin_ia32_pause();
    }
    while (true) {
        const int done = ctx->done_count.load(std::memory_order_acquire);
        if (ctx->done_generation.load(std::memory_order_acquire) >= generation
            && done >= n_workers) {
            return;
        }
        if (WAIT_WARN_MS > 0) {
            const int64_t now_ns = now_steady_ns();
            if (now_ns >= next_warn_ns) {
                std::fprintf(stderr,
                             "[jetfire][warn] graph_cpu_topk_wait_stalled "
                             "ctx=%p target_gen=%ld active_gen=%ld "
                             "done_gen=%ld done=%d workers=%d "
                             "active_call=%ld wait_seq=%ld\n",
                             (void*)ctx, (long)generation,
                             (long)ctx->active_generation.load(
                                 std::memory_order_acquire),
                             (long)ctx->done_generation.load(
                                 std::memory_order_acquire),
                             done, n_workers,
                             (long)ctx->active_call.load(
                                 std::memory_order_acquire),
                             (long)ctx->next_wait_generation.load(
                                 std::memory_order_acquire));
                std::fflush(stderr);
                next_warn_ns = now_ns + WAIT_WARN_MS * 1000LL * 1000LL;
            }
        }
        if (done >= n_workers) {
            __builtin_ia32_pause();
            continue;
        }
        if (WAIT_WARN_MS > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        futex_wait_atomic_int(&ctx->done_count, done);
    }
}


static void CUDART_CB submit_shadow_append_host_callback(void* user_data) {
    const int64_t t0 = now_steady_ns();
    auto* ctx = static_cast<ShadowAppendSubmitCtx*>(user_data);
    const int64_t call_idx = g_shadow_append_count.fetch_add(
        1, std::memory_order_acq_rel);
    const int64_t heartbeat_every = graph_heartbeat_every();
    if (std::getenv("JETFIRE_DEBUG_GRAPH_APPEND") != nullptr
        || (heartbeat_every > 0 && (call_idx % heartbeat_every) == 0)) {
        const int64_t log_idx = g_shadow_append_log_count.fetch_add(
            1, std::memory_order_acq_rel);
        if (log_idx < 128
            || (heartbeat_every > 0 && (call_idx % heartbeat_every) == 0)) {
            const int64_t first_slot = (ctx->B > 0) ? ctx->slots_p[0] : -999;
            const uint16_t k0 = (ctx->B > 0)
                ? *reinterpret_cast<const uint16_t*>(ctx->k_append_p)
                : 0;
            const uint16_t v0 = (ctx->B > 0)
                ? *reinterpret_cast<const uint16_t*>(ctx->v_append_p)
                : 0;
            std::fprintf(stderr,
                         "[jetfire] graph_shadow_append call=%ld ctx=%p B=%ld "
                         "first_slot=%ld num_slots=%ld k0=0x%04x v0=0x%04x "
                         "H_kv=%ld D=%ld\n",
                         (long)call_idx, (void*)ctx, (long)ctx->B,
                         (long)first_slot, (long)ctx->num_slots,
                         (unsigned)k0, (unsigned)v0,
                         (long)ctx->H_kv, (long)ctx->D);
            std::fflush(stderr);
        }
    }
    const int64_t D = ctx->D;
    const int64_t shadow_head_stride = ctx->num_slots * D;
    for (int64_t t = 0; t < ctx->B; ++t) {
        const int64_t slot = ctx->slots_p[t];
        if (!valid_shadow_slot(slot, ctx->num_slots)) {
            continue;
        }
        for (int64_t h = 0; h < ctx->H_kv; ++h) {
            const at::BFloat16* k_src =
                ctx->k_append_p + (t * ctx->H_kv + h) * D;
            const at::BFloat16* v_src =
                ctx->v_append_p + (t * ctx->H_kv + h) * D;
            at::BFloat16* k_dst =
                ctx->k_shadow_p + h * shadow_head_stride + slot * D;
            at::BFloat16* v_dst =
                ctx->v_shadow_p + h * shadow_head_stride + slot * D;
            std::memcpy(k_dst, k_src, sizeof(at::BFloat16) * D);
            std::memcpy(v_dst, v_src, sizeof(at::BFloat16) * D);
        }
    }
    _mm_sfence();
    const int64_t dt = now_steady_ns() - t0;
    g_graph_shadow_append_ns.fetch_add(dt, std::memory_order_acq_rel);
    atomic_add_max(g_graph_shadow_append_max_ns, dt);
}


static void worker_run_topk_attn_with_flag(int wid, int n_workers, void* ud) {
    auto* ctx = static_cast<CpuTopkSubmitCtx*>(ud);
    // Build CpuTopkParams from ctx (cheap struct copy).
    CpuTopkParams params;
    params.q_p = ctx->q_p;
    params.ks_p = ctx->ks_p;
    params.vs_p = ctx->vs_p;
    params.hm_p = ctx->hm_p;
    params.ts_p = ctx->ts_p;
    params.o_p  = ctx->o_p;
    params.B = ctx->B;
    params.H_q = ctx->H_q;
    params.K = ctx->K;
    params.D = ctx->D;
    params.num_slots = ctx->num_slots;
    params.total = ctx->total;
    params.scale = ctx->scale;
    params.flag_ptr = nullptr;  // not used; we handle the flag here

    run_cpu_topk_attn_rows_d128(params, wid, n_workers);

    if (wid == 0 && std::getenv("JETFIRE_DEBUG_GRAPH_SCAN") != nullptr) {
        const int64_t log_idx = g_cpu_scan_log_count.fetch_add(
            1, std::memory_order_acq_rel);
        if (log_idx < 128) {
            int64_t out_nan_i = -1;
            for (int64_t i = 0; i < params.D; ++i) {
                if (bf16_bits_is_nan(bf16_bits(ctx->o_p + i))) {
                    out_nan_i = i;
                    break;
                }
            }
            if (out_nan_i >= 0) {
                std::fprintf(stderr,
                             "[jetfire] graph_cpu_topk_scan_output log=%ld "
                             "ctx=%p row0_nan_i=%ld bits=0x%04x\n",
                             (long)log_idx, (void*)ctx, (long)out_nan_i,
                             (unsigned)bf16_bits(ctx->o_p + out_nan_i));
                std::fflush(stderr);
            }
        }
    }

    // Increment our own done counter; the last worker signals completion so
    // downstream GPU work sees a fully-flushed o_new buffer.  Prefer a
    // device-memory doorbell written from a side stream; the legacy mapped
    // host flag path is kept for A/B.
    //
    // Each worker may write a disjoint slice of o_new, potentially through
    // write-combined mapped pages.  Fencing only on the last worker does not
    // flush other cores' WC buffers, which can make larger captured decode
    // graphs observe stale/invalid data during replay.  Fence per writer
    // before publishing completion.
    _mm_sfence();
    if (std::getenv("JETFIRE_DEBUG_GRAPH_APPEND") != nullptr) {
        const int64_t log_idx = g_cpu_done_log_count.fetch_add(
            1, std::memory_order_acq_rel);
        if (log_idx < 128) {
            const uint16_t o0 =
                *reinterpret_cast<const uint16_t*>(ctx->o_p);
            std::fprintf(stderr,
                         "[jetfire] graph_cpu_topk_done worker_log=%ld "
                         "ctx=%p o0=0x%04x total=%ld\n",
                         (long)log_idx, (void*)ctx, (unsigned)o0,
                         (long)ctx->total);
            std::fflush(stderr);
        }
    }
    int prev = ctx->done_count.fetch_add(1, std::memory_order_acq_rel);
    if (prev + 1 == n_workers) {
        const int64_t call_idx =
            ctx->active_call.load(std::memory_order_acquire);
        const int64_t t0 = ctx->submit_t0_ns.load(std::memory_order_acquire);
        if (t0 > 0) {
            const int64_t dt = now_steady_ns() - t0;
            g_graph_topk_done_ns.fetch_add(dt, std::memory_order_acq_rel);
            atomic_add_max(g_graph_topk_done_max_ns, dt);
        }
        int64_t flag = -1;
        if (ctx->signal_dev != 0 && ctx->signal_stream != nullptr) {
            if (ctx->signal_device_index >= 0) {
                cudaError_t set_err = cudaSetDevice(ctx->signal_device_index);
                if (set_err != cudaSuccess) {
                    std::fprintf(stderr,
                                 "[jetfire] graph_cpu_topk_signal "
                                 "cudaSetDevice(%d) failed: %s\n",
                                 ctx->signal_device_index,
                                 cudaGetErrorString(set_err));
                    std::fflush(stderr);
                }
            }
            CUresult err = cuStreamWriteValue64(
                ctx->signal_stream, ctx->signal_dev, (cuuint64_t)1,
                /*flags*/0);
            if (err != CUDA_SUCCESS) {
                std::fprintf(stderr,
                             "[jetfire] graph_cpu_topk_signal "
                             "cuStreamWriteValue64 failed: %d\n",
                             (int)err);
                std::fflush(stderr);
            }
            flag = 1;
        } else {
            ctx->flag_p->store(1, std::memory_order_release);
            // The graph-mode doorbell can be allocated with
            // cudaHostAllocWriteCombined.  The preceding fence flushes o_new
            // writes; this fence flushes the flag store so cuStreamWaitValue64
            // cannot spin forever on a stale 0.
            _mm_sfence();
            // (We do NOT futex_wake here: the consumer is a GPU stream waiting
            // via cuStreamWaitValue64, which polls hardware-side and needs no
            // syscall wakeup.)
            flag = ctx->flag_p->load(std::memory_order_acquire);
        }
        g_graph_last_done_call.store(call_idx, std::memory_order_release);
        g_graph_last_done_ctx.store(
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(ctx)),
            std::memory_order_release);
        g_graph_last_done_prev.store(prev, std::memory_order_release);
        g_graph_last_done_flag.store(flag, std::memory_order_release);
        ctx->done_generation.store(
            ctx->active_generation.load(std::memory_order_acquire),
            std::memory_order_release);
        futex_wake_atomic_int(&ctx->done_count, 1);
        const int64_t every = graph_heartbeat_every();
        if (every > 0 && call_idx >= 0 && (call_idx % every) == 0) {
            std::fprintf(stderr,
                         "[jetfire] graph_cpu_topk_last_done call=%ld ctx=%p "
                         "prev=%d n_workers=%d flag=%ld\n",
                         (long)call_idx, (void*)ctx, prev, n_workers,
                         (long)flag);
            std::fflush(stderr);
        }
    }
}


static void CUDART_CB submit_cpu_topk_host_callback(void* user_data) {
    const int64_t cb_t0 = now_steady_ns();
    auto* ctx = static_cast<CpuTopkSubmitCtx*>(user_data);
    const int64_t call_idx = g_graph_submit_count.fetch_add(
        1, std::memory_order_acq_rel);
    ctx->active_call.store(call_idx, std::memory_order_release);
    g_graph_last_submit_call.store(call_idx, std::memory_order_release);
    g_graph_last_submit_ctx.store(
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(ctx)),
        std::memory_order_release);
    if (std::getenv("JETFIRE_DEBUG_GRAPH_SCAN") != nullptr) {
        const int64_t log_idx = g_cpu_scan_log_count.fetch_add(
            1, std::memory_order_acq_rel);
        if (log_idx < 128 && ctx->B > 0 && ctx->H_q > 0) {
            const int64_t D = ctx->D;
            const int64_t shadow_head_stride = ctx->num_slots * D;
            int64_t q_nan_i = -1;
            for (int64_t i = 0; i < D; ++i) {
                if (bf16_bits_is_nan(bf16_bits(ctx->q_p + i))) {
                    q_nan_i = i;
                    break;
                }
            }
            int64_t k_nan_k = -1, k_nan_i = -1, k_nan_slot = -1, k_nan_h = -1;
            int64_t v_nan_k = -1, v_nan_i = -1, v_nan_slot = -1, v_nan_h = -1;
            for (int64_t k = 0; k < ctx->K; ++k) {
                const int64_t hkv = ctx->hm_p[k];
                const int64_t slot = ctx->ts_p[k];
                if (!valid_shadow_slot(slot, ctx->num_slots)) {
                    continue;
                }
                const at::BFloat16* k_row =
                    ctx->ks_p + hkv * shadow_head_stride + slot * D;
                const at::BFloat16* v_row =
                    ctx->vs_p + hkv * shadow_head_stride + slot * D;
                for (int64_t i = 0; i < D; ++i) {
                    if (k_nan_k < 0 && bf16_bits_is_nan(bf16_bits(k_row + i))) {
                        k_nan_k = k;
                        k_nan_i = i;
                        k_nan_slot = slot;
                        k_nan_h = hkv;
                    }
                    if (v_nan_k < 0 && bf16_bits_is_nan(bf16_bits(v_row + i))) {
                        v_nan_k = k;
                        v_nan_i = i;
                        v_nan_slot = slot;
                        v_nan_h = hkv;
                    }
                    if (k_nan_k >= 0 && v_nan_k >= 0) {
                        break;
                    }
                }
                if (k_nan_k >= 0 && v_nan_k >= 0) {
                    break;
                }
            }
            if (q_nan_i >= 0 || k_nan_k >= 0 || v_nan_k >= 0) {
                std::fprintf(stderr,
                             "[jetfire] graph_cpu_topk_scan_input call=%ld "
                             "ctx=%p q_nan_i=%ld k_nan=(k=%ld h=%ld slot=%ld i=%ld) "
                             "v_nan=(k=%ld h=%ld slot=%ld i=%ld)\n",
                             (long)call_idx, (void*)ctx, (long)q_nan_i,
                             (long)k_nan_k, (long)k_nan_h, (long)k_nan_slot,
                             (long)k_nan_i, (long)v_nan_k, (long)v_nan_h,
                             (long)v_nan_slot, (long)v_nan_i);
                std::fflush(stderr);
            }
        }
    }
    if (std::getenv("JETFIRE_DEBUG_GRAPH_APPEND") != nullptr) {
        const int64_t log_idx = g_cpu_submit_log_count.fetch_add(
            1, std::memory_order_acq_rel);
        if (log_idx < 128) {
            int64_t min_slot = INT64_MAX;
            int64_t max_slot = INT64_MIN;
            const int64_t limit = ctx->K;
            for (int64_t i = 0; i < limit; ++i) {
                const int64_t s = ctx->ts_p[i];
                min_slot = std::min(min_slot, s);
                max_slot = std::max(max_slot, s);
            }
            const uint16_t q0 =
                *reinterpret_cast<const uint16_t*>(ctx->q_p);
            std::fprintf(stderr,
                         "[jetfire] graph_cpu_topk_submit call=%ld ctx=%p B=%ld "
                         "q0=0x%04x ts0=%ld ts_min=%ld ts_max=%ld K=%ld\n",
                         (long)call_idx, (void*)ctx, (long)ctx->B,
                         (unsigned)q0, (long)ctx->ts_p[0], (long)min_slot,
                         (long)max_slot, (long)ctx->K);
            std::fflush(stderr);
        }
    }
    // The async graph path is single-flight per ctx, but CUDA graph replay can
    // enqueue a later submit before the earlier wait host callback has run.
    // Do not mutate active_generation/done_count until any previous ctx job is
    // really complete; otherwise the previous job's last worker can publish
    // completion under the next generation and the graph wait node may observe
    // stale o_new or wait forever.
    const int64_t prev_generation =
        ctx->active_generation.load(std::memory_order_acquire);
    const int prev_workers = ctx->n_workers.load(std::memory_order_acquire);
    if (prev_generation > 0 && prev_workers > 0) {
        wait_cpu_topk_ctx_done(ctx, prev_generation, prev_workers);
    }

    // Reset our done barrier before dispatching the next job.
    g_graph_last_submit_ctx_done.store(
        ctx->done_count.load(std::memory_order_acquire),
        std::memory_order_release);
    g_graph_last_submit_flag.store(
        ctx->flag_p ? ctx->flag_p->load(std::memory_order_acquire) : -1,
        std::memory_order_release);
    ctx->done_count.store(0, std::memory_order_release);
    const int64_t generation = ctx->active_generation.fetch_add(
        1, std::memory_order_acq_rel) + 1;
    // Resolve the actual worker count once, lazily warming the pool.
    auto& pool = jetfire_pool::global_pool();
    if (!pool.is_initialized()) {
        pool.init(jetfire_cpu_attn_threads(), /*pin_to_cores=*/false);
    }
    const int n_workers = pool.n();
    ctx->n_workers.store(n_workers, std::memory_order_release);
    g_graph_last_submit_workers.store(n_workers,
                                      std::memory_order_release);
    g_graph_last_submit_pool_done.store(pool.done_count(),
                                        std::memory_order_release);
    const bool inline_topk =
        std::getenv("JETFIRE_GRAPH_INLINE_CPU_TOPK") != nullptr;
    const char* sync_env = std::getenv("JETFIRE_GRAPH_SYNC_CPU_TOPK");
    const bool sync_topk =
        (sync_env != nullptr && std::strcmp(sync_env, "0") != 0);
    const int64_t mode = inline_topk ? 2 : (sync_topk ? 1 : 0);
    g_graph_last_submit_mode.store(mode, std::memory_order_release);
    const int64_t every = graph_heartbeat_every();
    if (every > 0 && (call_idx % every) == 0) {
        std::fprintf(stderr,
                     "[jetfire] graph_cpu_topk_submit_heartbeat call=%ld "
                     "ctx=%p gen=%ld mode=%ld ctx_done_before=%ld "
                     "pool_done_before=%ld flag_before=%ld workers=%d "
                     "B=%ld H_q=%ld K=%ld total=%ld\n",
                     (long)call_idx, (void*)ctx, (long)generation,
                     (long)mode,
                     (long)g_graph_last_submit_ctx_done.load(
                         std::memory_order_acquire),
                     (long)g_graph_last_submit_pool_done.load(
                         std::memory_order_acquire),
                     (long)g_graph_last_submit_flag.load(
                         std::memory_order_acquire),
                     n_workers, (long)ctx->B, (long)ctx->H_q,
                     (long)ctx->K, (long)ctx->total);
        std::fflush(stderr);
    }
    ctx->submit_t0_ns.store(cb_t0, std::memory_order_release);
    if (inline_topk) {
        ctx->n_workers.store(1, std::memory_order_release);
        g_graph_last_submit_workers.store(1, std::memory_order_release);
        worker_run_topk_attn_with_flag(0, 1, ctx);
    } else if (sync_topk) {
        pool.dispatch_and_wait(worker_run_topk_attn_with_flag, ctx);
    } else {
        // Fire-and-return: dispatch_async wakes workers and returns immediately.
        pool.dispatch_async(worker_run_topk_attn_with_flag, ctx);
    }
    const int64_t submit_dt = now_steady_ns() - cb_t0;
    g_graph_topk_jobs.fetch_add(1, std::memory_order_acq_rel);
    g_graph_topk_submit_ns.fetch_add(submit_dt, std::memory_order_acq_rel);
    atomic_add_max(g_graph_topk_submit_max_ns, submit_dt);
}


static void CUDART_CB wait_cpu_topk_host_callback(void* user_data) {
    const int64_t t0 = now_steady_ns();
    auto* ctx = static_cast<CpuTopkSubmitCtx*>(user_data);
    const int64_t call_idx =
        ctx->active_call.load(std::memory_order_acquire);
    const int64_t wait_seq = ctx->next_wait_generation.fetch_add(
        1, std::memory_order_acq_rel) + 1;
    // Wait for the generation that the preceding submit callback actually
    // published for this ctx.  A separate wait-side generation counter can
    // drift across CUDA graph warmup/replay variants and then block forever
    // on a generation that no worker will ever complete.
    const int64_t generation =
        ctx->active_generation.load(std::memory_order_acquire);
    const int n_workers = ctx->n_workers.load(std::memory_order_acquire);
    auto& pool = jetfire_pool::global_pool();
    wait_cpu_topk_ctx_done(ctx, generation, n_workers);
    const int ctx_done = ctx->done_count.load(std::memory_order_acquire);
    const int pool_done = pool.is_initialized() ? pool.done_count() : -1;
    const int64_t join_idx = g_graph_join_count.fetch_add(
        1, std::memory_order_acq_rel);
    const int64_t dt = now_steady_ns() - t0;
    g_graph_join_ns.fetch_add(dt, std::memory_order_acq_rel);
    atomic_add_max(g_graph_join_max_ns, dt);
    g_graph_last_join_call.store(call_idx, std::memory_order_release);
    g_graph_last_join_ctx.store(
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(ctx)),
        std::memory_order_release);
    g_graph_last_join_ctx_done.store(ctx_done, std::memory_order_release);
    g_graph_last_join_pool_done.store(pool_done, std::memory_order_release);
    const int64_t every = graph_heartbeat_every();
    if (every > 0 && call_idx >= 0 && (call_idx % every) == 0) {
        std::fprintf(stderr,
                     "[jetfire] graph_cpu_topk_join_heartbeat join=%ld "
                     "call=%ld ctx=%p gen=%ld wait_seq=%ld "
                     "ctx_done=%d pool_done=%d "
                     "waited_us=%.3f\n",
                     (long)join_idx, (long)call_idx, (void*)ctx,
                     (long)generation, (long)wait_seq, ctx_done, pool_done,
                     (double)dt / 1000.0);
        std::fflush(stderr);
    }
}

static void ensure_side_join_resources(CpuTopkSubmitCtx* ctx) {
    TORCH_CHECK(ctx != nullptr, "ctx is null");
    if (ctx->join_stream != nullptr
            && ctx->join_submit_event != nullptr
            && ctx->join_done_event != nullptr) {
        return;
    }
    TORCH_CHECK(ctx->join_stream == nullptr
                && ctx->join_submit_event == nullptr
                && ctx->join_done_event == nullptr,
                "ctx has partially initialized side-join resources");
    cudaError_t join_stream_err = cudaStreamCreateWithFlags(
        &ctx->join_stream, cudaStreamNonBlocking);
    TORCH_CHECK(join_stream_err == cudaSuccess,
                "cudaStreamCreateWithFlags(join_stream) failed: ",
                cudaGetErrorString(join_stream_err));
    cudaError_t submit_event_err = cudaEventCreateWithFlags(
        &ctx->join_submit_event, cudaEventDisableTiming);
    TORCH_CHECK(submit_event_err == cudaSuccess,
                "cudaEventCreateWithFlags(join_submit_event) failed: ",
                cudaGetErrorString(submit_event_err));
    cudaError_t done_event_err = cudaEventCreateWithFlags(
        &ctx->join_done_event, cudaEventDisableTiming);
    TORCH_CHECK(done_event_err == cudaSuccess,
                "cudaEventCreateWithFlags(join_done_event) failed: ",
                cudaGetErrorString(done_event_err));
}


// Create a long-lived submit ctx referencing the pre-allocated decode
// buffers.  All tensors must be CPU; ``flag_host`` must be the host half
// of a mapped pinned pair (its host pointer is what the worker writes;
// the GPU side reads the same memory through the device pointer).
//
// Returns an opaque int handle to pass to stream_submit_cpu_topk_attn.
// The ctx (and the tensors it references) live for the process lifetime.
static int64_t create_cpu_topk_submit_ctx_impl(
    torch::Tensor q,
    torch::Tensor k_shadow,
    torch::Tensor v_shadow,
    torch::Tensor head_map,
    torch::Tensor ts_buf,
    torch::Tensor o_new,
    double scale_d,
    torch::Tensor flag_host,
    torch::Tensor signal_device,
    bool use_device_signal)
{
    TORCH_CHECK(q.is_cpu() && q.is_contiguous() && q.dim() == 3,
                "q must be a contiguous CPU bf16 (B, H_q, D)");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k_shadow.is_cpu() && k_shadow.dim() == 3, "k_shadow (H_kv, S, D) CPU bf16");
    TORCH_CHECK(v_shadow.is_cpu() && v_shadow.dim() == 3, "v_shadow (H_kv, S, D) CPU bf16");
    TORCH_CHECK(head_map.is_cpu() && head_map.scalar_type() == torch::kInt64
                && head_map.is_contiguous() && head_map.dim() == 3,
                "head_map (B, H_q, K) int64 CPU contiguous");
    TORCH_CHECK(ts_buf.is_cpu() && ts_buf.scalar_type() == torch::kInt64
                && ts_buf.is_contiguous() && ts_buf.dim() == 3,
                "ts_buf (B, H_q, K) int64 CPU contiguous");
    TORCH_CHECK(o_new.is_cpu() && o_new.is_contiguous() && o_new.dim() == 3,
                "o_new (B, H_q, D) CPU contiguous");
    TORCH_CHECK(o_new.scalar_type() == torch::kBFloat16, "o_new must be bf16");
    TORCH_CHECK(flag_host.is_cpu() && flag_host.is_pinned()
                && flag_host.scalar_type() == torch::kInt64
                && flag_host.numel() == 1,
                "flag_host must be a 1-element pinned int64 CPU tensor "
                "(use the host half of alloc_mapped_pinned_pair).");
    if (use_device_signal) {
        TORCH_CHECK(signal_device.is_cuda()
                    && signal_device.scalar_type() == torch::kInt64
                    && signal_device.numel() == 1,
                    "signal_device must be a 1-element CUDA int64 tensor.");
    }

    const int64_t B   = q.size(0);
    const int64_t H_q = q.size(1);
    const int64_t D   = q.size(2);
    TORCH_CHECK(D == 128, "this path is specialized for D=128");

    auto ctx = std::make_unique<CpuTopkSubmitCtx>();
    ctx->q_p  = q.data_ptr<at::BFloat16>();
    ctx->ks_p = k_shadow.data_ptr<at::BFloat16>();
    ctx->vs_p = v_shadow.data_ptr<at::BFloat16>();
    ctx->hm_p = head_map.data_ptr<int64_t>();
    ctx->ts_p = ts_buf.data_ptr<int64_t>();
    ctx->o_p  = o_new.data_ptr<at::BFloat16>();
    ctx->flag_p = reinterpret_cast<std::atomic<int64_t>*>(
        flag_host.data_ptr<int64_t>());
    if (use_device_signal) {
        int device_index = -1;
        cudaError_t dev_err = cudaGetDevice(&device_index);
        TORCH_CHECK(dev_err == cudaSuccess,
                    "cudaGetDevice failed: ", cudaGetErrorString(dev_err));
        cudaStream_t signal_stream = nullptr;
        cudaError_t stream_err = cudaStreamCreateWithFlags(
            &signal_stream, cudaStreamNonBlocking);
        TORCH_CHECK(stream_err == cudaSuccess,
                    "cudaStreamCreateWithFlags(signal_stream) failed: ",
                    cudaGetErrorString(stream_err));
        ctx->signal_dev = reinterpret_cast<CUdeviceptr>(
            signal_device.data_ptr<int64_t>());
        ctx->signal_stream = reinterpret_cast<CUstream>(signal_stream);
        ctx->signal_device_index = device_index;
        ctx->signal_ref = signal_device;
    }
    ctx->B = B;
    ctx->H_q = H_q;
    ctx->K = ts_buf.size(2);
    ctx->num_slots = k_shadow.size(1);
    ctx->total = B * H_q;
    ctx->D = (int)D;
    ctx->scale = (float)scale_d;
    ctx->done_count.store(0, std::memory_order_release);
    ctx->n_workers.store(0, std::memory_order_release);  // resolved lazily on first dispatch

    ctx->q_ref = q;
    ctx->ks_ref = k_shadow;
    ctx->vs_ref = v_shadow;
    ctx->hm_ref = head_map;
    ctx->ts_ref = ts_buf;
    ctx->o_ref = o_new;
    ctx->flag_ref = flag_host;

    std::lock_guard<std::mutex> lk(g_submit_ctx_mu);
    g_submit_ctxs.emplace_back(std::move(ctx));
    return reinterpret_cast<int64_t>(g_submit_ctxs.back().get());
}


static int64_t create_cpu_topk_submit_ctx(
    torch::Tensor q,
    torch::Tensor k_shadow,
    torch::Tensor v_shadow,
    torch::Tensor head_map,
    torch::Tensor ts_buf,
    torch::Tensor o_new,
    double scale_d,
    torch::Tensor flag_host)
{
    return create_cpu_topk_submit_ctx_impl(
        std::move(q), std::move(k_shadow), std::move(v_shadow),
        std::move(head_map), std::move(ts_buf), std::move(o_new),
        scale_d, std::move(flag_host), torch::Tensor(),
        /*use_device_signal=*/false);
}


static int64_t create_cpu_topk_submit_ctx_device_signal(
    torch::Tensor q,
    torch::Tensor k_shadow,
    torch::Tensor v_shadow,
    torch::Tensor head_map,
    torch::Tensor ts_buf,
    torch::Tensor o_new,
    double scale_d,
    torch::Tensor flag_host,
    torch::Tensor signal_device)
{
    return create_cpu_topk_submit_ctx_impl(
        std::move(q), std::move(k_shadow), std::move(v_shadow),
        std::move(head_map), std::move(ts_buf), std::move(o_new),
        scale_d, std::move(flag_host), std::move(signal_device),
        /*use_device_signal=*/true);
}


static int64_t create_shadow_append_submit_ctx(
    torch::Tensor k_append,
    torch::Tensor v_append,
    torch::Tensor slots,
    torch::Tensor k_shadow,
    torch::Tensor v_shadow)
{
    TORCH_CHECK(k_append.is_cpu() && k_append.is_contiguous() && k_append.dim() == 3,
                "k_append must be contiguous CPU bf16 (B, H_kv, D)");
    TORCH_CHECK(v_append.is_cpu() && v_append.is_contiguous() && v_append.dim() == 3,
                "v_append must be contiguous CPU bf16 (B, H_kv, D)");
    TORCH_CHECK(k_append.scalar_type() == torch::kBFloat16, "k_append must be bf16");
    TORCH_CHECK(v_append.scalar_type() == torch::kBFloat16, "v_append must be bf16");
    TORCH_CHECK(slots.is_cpu() && slots.is_contiguous() && slots.dim() == 1
                && slots.scalar_type() == torch::kInt64,
                "slots must be contiguous CPU int64 (B,)");
    TORCH_CHECK(k_shadow.is_cpu() && k_shadow.dim() == 3
                && k_shadow.scalar_type() == torch::kBFloat16,
                "k_shadow must be CPU bf16 (H_kv, num_slots, D)");
    TORCH_CHECK(v_shadow.is_cpu() && v_shadow.dim() == 3
                && v_shadow.scalar_type() == torch::kBFloat16,
                "v_shadow must be CPU bf16 (H_kv, num_slots, D)");
    TORCH_CHECK(k_append.size(0) == slots.size(0),
                "k_append B must match slots B");
    TORCH_CHECK(v_append.sizes() == k_append.sizes(),
                "v_append shape must match k_append");
    TORCH_CHECK(k_shadow.size(0) == k_append.size(1)
                && v_shadow.size(0) == k_append.size(1),
                "shadow H_kv must match append H_kv");
    TORCH_CHECK(k_shadow.size(2) == k_append.size(2)
                && v_shadow.size(2) == k_append.size(2),
                "shadow D must match append D");

    auto ctx = std::make_unique<ShadowAppendSubmitCtx>();
    ctx->k_append_p = k_append.data_ptr<at::BFloat16>();
    ctx->v_append_p = v_append.data_ptr<at::BFloat16>();
    ctx->slots_p = slots.data_ptr<int64_t>();
    ctx->k_shadow_p = k_shadow.data_ptr<at::BFloat16>();
    ctx->v_shadow_p = v_shadow.data_ptr<at::BFloat16>();
    ctx->B = k_append.size(0);
    ctx->H_kv = k_append.size(1);
    ctx->D = k_append.size(2);
    ctx->num_slots = k_shadow.size(1);
    ctx->k_append_ref = k_append;
    ctx->v_append_ref = v_append;
    ctx->slots_ref = slots;
    ctx->k_shadow_ref = k_shadow;
    ctx->v_shadow_ref = v_shadow;

    std::lock_guard<std::mutex> lk(g_submit_ctx_mu);
    g_shadow_append_ctxs.emplace_back(std::move(ctx));
    return reinterpret_cast<int64_t>(g_shadow_append_ctxs.back().get());
}


static void stream_submit_cpu_topk_attn(int64_t stream_id, int64_t ctx_handle) {
    TORCH_CHECK(ctx_handle != 0, "ctx_handle is null");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    void* ud = reinterpret_cast<void*>(ctx_handle);
    cudaError_t err = cudaLaunchHostFunc(
        stream, submit_cpu_topk_host_callback, ud);
    TORCH_CHECK(err == cudaSuccess,
                "cudaLaunchHostFunc(submit_cpu_topk) failed: ",
                cudaGetErrorString(err));
}


static void stream_submit_cpu_topk_attn_side_join(
    int64_t stream_id,
    int64_t ctx_handle) {
    TORCH_CHECK(ctx_handle != 0, "ctx_handle is null");
    auto* ctx = reinterpret_cast<CpuTopkSubmitCtx*>(ctx_handle);
    ensure_side_join_resources(ctx);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    void* ud = reinterpret_cast<void*>(ctx_handle);

    cudaError_t err = cudaLaunchHostFunc(
        stream, submit_cpu_topk_host_callback, ud);
    TORCH_CHECK(err == cudaSuccess,
                "cudaLaunchHostFunc(submit_cpu_topk) failed: ",
                cudaGetErrorString(err));
    err = cudaEventRecord(ctx->join_submit_event, stream);
    TORCH_CHECK(err == cudaSuccess,
                "cudaEventRecord(join_submit_event) failed: ",
                cudaGetErrorString(err));
    err = cudaStreamWaitEvent(
        ctx->join_stream, ctx->join_submit_event, /*flags*/0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaStreamWaitEvent(join_submit_event) failed: ",
                cudaGetErrorString(err));
    err = cudaLaunchHostFunc(
        ctx->join_stream, wait_cpu_topk_host_callback, ud);
    TORCH_CHECK(err == cudaSuccess,
                "cudaLaunchHostFunc(side_join_cpu_topk) failed: ",
                cudaGetErrorString(err));
    err = cudaEventRecord(ctx->join_done_event, ctx->join_stream);
    TORCH_CHECK(err == cudaSuccess,
                "cudaEventRecord(join_done_event) failed: ",
                cudaGetErrorString(err));
}


static void stream_wait_cpu_topk_attn_done_event(
    int64_t stream_id,
    int64_t ctx_handle) {
    TORCH_CHECK(ctx_handle != 0, "ctx_handle is null");
    auto* ctx = reinterpret_cast<CpuTopkSubmitCtx*>(ctx_handle);
    TORCH_CHECK(ctx->join_done_event != nullptr,
                "ctx has no side-join done event");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    cudaError_t err = cudaStreamWaitEvent(
        stream, ctx->join_done_event, /*flags*/0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaStreamWaitEvent(join_done_event) failed: ",
                cudaGetErrorString(err));
}


static void stream_wait_cpu_topk_attn(int64_t stream_id, int64_t ctx_handle) {
    TORCH_CHECK(ctx_handle != 0, "ctx_handle is null");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    void* ud = reinterpret_cast<void*>(ctx_handle);
    cudaError_t err = cudaLaunchHostFunc(
        stream, wait_cpu_topk_host_callback, ud);
    TORCH_CHECK(err == cudaSuccess,
                "cudaLaunchHostFunc(wait_cpu_topk) failed: ",
                cudaGetErrorString(err));
}


static void stream_submit_shadow_append(int64_t stream_id, int64_t ctx_handle) {
    TORCH_CHECK(ctx_handle != 0, "ctx_handle is null");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
    void* ud = reinterpret_cast<void*>(ctx_handle);
    cudaError_t err = cudaLaunchHostFunc(
        stream, submit_shadow_append_host_callback, ud);
    TORCH_CHECK(err == cudaSuccess,
                "cudaLaunchHostFunc(submit_shadow_append) failed: ",
                cudaGetErrorString(err));
}


// One-shot kick of the worker pool to spawn threads + warm up — call before
// the first graph capture so dispatch_async hits already-spinning workers.
static void graph_pool_warmup() {
    auto& pool = jetfire_pool::global_pool();
    if (!pool.is_initialized()) {
        pool.init(jetfire_cpu_attn_threads(), /*pin_to_cores=*/false);
    }
}

static int graph_pool_warmup_threads(int n_threads) {
    TORCH_CHECK(n_threads > 0, "graph_pool_warmup_threads requires n_threads > 0");
    auto& pool = jetfire_pool::global_pool();
    if (pool.is_initialized()) {
        if (pool.n() == n_threads) {
            return pool.n();
        }
        pool.shutdown();
    }
    set_cpu_attn_threads(n_threads);
    pool.init(n_threads, /*pin_to_cores=*/false);
    return pool.n();
}

static int64_t get_graph_submit_count() {
    return g_graph_submit_count.load(std::memory_order_acquire);
}

static int64_t get_shadow_append_count() {
    return g_shadow_append_count.load(std::memory_order_acquire);
}

static int64_t get_cpu_topk_sanitize_count() {
    return g_cpu_sanitize_count.load(std::memory_order_acquire);
}

static py::dict get_graph_timing_stats() {
    const int64_t jobs = g_graph_topk_jobs.load(std::memory_order_acquire);
    const int64_t topk_done_ns =
        g_graph_topk_done_ns.load(std::memory_order_acquire);
    const int64_t topk_submit_ns =
        g_graph_topk_submit_ns.load(std::memory_order_acquire);
    const int64_t shadow_count =
        g_shadow_append_count.load(std::memory_order_acquire);
    const int64_t shadow_ns =
        g_graph_shadow_append_ns.load(std::memory_order_acquire);
    py::dict out;
    out["topk_jobs"] = jobs;
    out["topk_submit_avg_us"] =
        jobs > 0 ? (double)topk_submit_ns / (double)jobs / 1000.0 : 0.0;
    out["topk_submit_max_us"] =
        (double)g_graph_topk_submit_max_ns.load(std::memory_order_acquire)
        / 1000.0;
    out["topk_done_avg_us"] =
        jobs > 0 ? (double)topk_done_ns / (double)jobs / 1000.0 : 0.0;
    out["topk_done_max_us"] =
        (double)g_graph_topk_done_max_ns.load(std::memory_order_acquire)
        / 1000.0;
    out["shadow_append_jobs"] = shadow_count;
    out["shadow_append_avg_us"] =
        shadow_count > 0 ? (double)shadow_ns / (double)shadow_count / 1000.0
                         : 0.0;
    out["shadow_append_max_us"] =
        (double)g_graph_shadow_append_max_ns.load(std::memory_order_acquire)
        / 1000.0;
    auto& pool = jetfire_pool::global_pool();
    out["pool_initialized"] = pool.is_initialized();
    out["pool_workers"] = pool.is_initialized() ? pool.n() : 0;
    out["pool_done_count"] =
        pool.is_initialized() ? pool.done_count() : -1;
    out["last_submit_call"] =
        g_graph_last_submit_call.load(std::memory_order_acquire);
    out["last_submit_ctx"] =
        g_graph_last_submit_ctx.load(std::memory_order_acquire);
    out["last_submit_ctx_done"] =
        g_graph_last_submit_ctx_done.load(std::memory_order_acquire);
    out["last_submit_pool_done"] =
        g_graph_last_submit_pool_done.load(std::memory_order_acquire);
    out["last_submit_flag"] =
        g_graph_last_submit_flag.load(std::memory_order_acquire);
    out["last_submit_workers"] =
        g_graph_last_submit_workers.load(std::memory_order_acquire);
    out["last_submit_mode"] =
        g_graph_last_submit_mode.load(std::memory_order_acquire);
    out["last_done_call"] =
        g_graph_last_done_call.load(std::memory_order_acquire);
    out["last_done_ctx"] =
        g_graph_last_done_ctx.load(std::memory_order_acquire);
    out["last_done_prev"] =
        g_graph_last_done_prev.load(std::memory_order_acquire);
    out["last_done_flag"] =
        g_graph_last_done_flag.load(std::memory_order_acquire);
    const int64_t joins = g_graph_join_count.load(std::memory_order_acquire);
    const int64_t join_ns = g_graph_join_ns.load(std::memory_order_acquire);
    out["join_jobs"] = joins;
    out["join_wait_avg_us"] =
        joins > 0 ? (double)join_ns / (double)joins / 1000.0 : 0.0;
    out["join_wait_max_us"] =
        (double)g_graph_join_max_ns.load(std::memory_order_acquire)
        / 1000.0;
    out["last_join_call"] =
        g_graph_last_join_call.load(std::memory_order_acquire);
    out["last_join_ctx"] =
        g_graph_last_join_ctx.load(std::memory_order_acquire);
    out["last_join_ctx_done"] =
        g_graph_last_join_ctx_done.load(std::memory_order_acquire);
    out["last_join_pool_done"] =
        g_graph_last_join_pool_done.load(std::memory_order_acquire);
    return out;
}

static void reset_graph_submit_count() {
    g_graph_submit_count.store(0, std::memory_order_release);
    g_shadow_append_count.store(0, std::memory_order_release);
    g_shadow_append_log_count.store(0, std::memory_order_release);
    g_cpu_submit_log_count.store(0, std::memory_order_release);
    g_cpu_done_log_count.store(0, std::memory_order_release);
    g_cpu_scan_log_count.store(0, std::memory_order_release);
    g_cpu_sanitize_count.store(0, std::memory_order_release);
    g_cpu_sanitize_log_count.store(0, std::memory_order_release);
    g_graph_topk_jobs.store(0, std::memory_order_release);
    g_graph_topk_submit_ns.store(0, std::memory_order_release);
    g_graph_topk_done_ns.store(0, std::memory_order_release);
    g_graph_topk_submit_max_ns.store(0, std::memory_order_release);
    g_graph_topk_done_max_ns.store(0, std::memory_order_release);
    g_graph_shadow_append_ns.store(0, std::memory_order_release);
    g_graph_shadow_append_max_ns.store(0, std::memory_order_release);
    g_graph_last_submit_call.store(-1, std::memory_order_release);
    g_graph_last_submit_ctx.store(0, std::memory_order_release);
    g_graph_last_submit_ctx_done.store(-1, std::memory_order_release);
    g_graph_last_submit_pool_done.store(-1, std::memory_order_release);
    g_graph_last_submit_flag.store(-1, std::memory_order_release);
    g_graph_last_submit_workers.store(-1, std::memory_order_release);
    g_graph_last_submit_mode.store(0, std::memory_order_release);
    g_graph_last_done_call.store(-1, std::memory_order_release);
    g_graph_last_done_ctx.store(0, std::memory_order_release);
    g_graph_last_done_prev.store(-1, std::memory_order_release);
    g_graph_last_done_flag.store(-1, std::memory_order_release);
    g_graph_join_count.store(0, std::memory_order_release);
    g_graph_join_ns.store(0, std::memory_order_release);
    g_graph_join_max_ns.store(0, std::memory_order_release);
    g_graph_last_join_call.store(-1, std::memory_order_release);
    g_graph_last_join_ctx.store(0, std::memory_order_release);
    g_graph_last_join_ctx_done.store(-1, std::memory_order_release);
    g_graph_last_join_pool_done.store(-1, std::memory_order_release);
}

static void reset_graph_runtime_state() {
    std::lock_guard<std::mutex> lock(g_submit_ctx_mu);
    for (auto& owned_ctx : g_submit_ctxs) {
        CpuTopkSubmitCtx* ctx = owned_ctx.get();
        if (ctx == nullptr) {
            continue;
        }
        if (ctx->signal_stream != nullptr) {
            cudaStreamSynchronize(
                reinterpret_cast<cudaStream_t>(ctx->signal_stream));
        }
        if (ctx->join_stream != nullptr) {
            cudaStreamSynchronize(ctx->join_stream);
        }
        ctx->done_count.store(0, std::memory_order_release);
        ctx->submit_t0_ns.store(0, std::memory_order_release);
        ctx->active_call.store(-1, std::memory_order_release);
        ctx->active_generation.store(0, std::memory_order_release);
        ctx->done_generation.store(0, std::memory_order_release);
        ctx->next_wait_generation.store(0, std::memory_order_release);
        ctx->n_workers.store(0, std::memory_order_release);
        if (ctx->flag_p != nullptr) {
            ctx->flag_p->store(1, std::memory_order_release);
            _mm_sfence();
        }
    }
    reset_graph_submit_count();
}

static void clear_graph_contexts() {
    std::lock_guard<std::mutex> lock(g_submit_ctx_mu);
    g_submit_ctxs.clear();
    g_shadow_append_ctxs.clear();
    reset_graph_submit_count();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_topk_attn", &cpu_topk_attn,
          py::call_guard<py::gil_scoped_release>(),
          "CPU top-K attention: gather + softmax(Q·K^T)·V → o_new. "
          "Releases the GIL.  Optional `done_flag` (1-element int64 CPU "
          "tensor): atomic-release-stored to 1 after compute + sfence so a "
          "spin-waiter can detect completion without going through cv.",
          py::arg("q"), py::arg("k_shadow"), py::arg("v_shadow"),
          py::arg("head_map"), py::arg("ts_buf"),
          py::arg("o_new"), py::arg("scale"),
          py::arg("done_flag") = c10::nullopt);

    m.def("alloc_mapped_pinned_pair", &alloc_mapped_pinned_pair,
          "Allocate cudaHostAllocMapped+WriteCombined buffer; returns "
          "(host_tensor, device_tensor) sharing the same physical memory. "
          "CPU writes via host_tensor are visible to GPU through device_tensor "
          "(via PCIe) — no explicit H2D needed.",
          py::arg("shape"), py::arg("dtype"));

    // ── Async CPU executor bindings ──────────────────────────────────
    m.def("submit_cpu_topk_attn_async", &submit_cpu_topk_attn_async,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_shadow"), py::arg("v_shadow"),
          py::arg("head_map"), py::arg("ts_buf"), py::arg("o_new"),
          py::arg("scale"), py::arg("flag"),
          "Submit a cpu_topk_attn job to the async CPU executor. "
          "Returns immediately. The last finishing worker stores 1 "
          "(release) to ``flag`` after compute + sfence retires.");
    m.def("wait_for_flag", &wait_for_flag,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("flag"),
          "Spin-then-yield until ``flag`` reads 1 (acquire).  GIL released.");
    m.def("persistent_worker_warmup", &persistent_worker_warmup,
          "Eagerly construct the async CPU executor worker set (lazy by default).");

    // Microbench / sweep escape hatch: override the resolved thread count
    // at runtime.  Pass n > 0 to set, n <= 0 to revert to env/default.
    // The next ``cpu_topk_attn`` call will tear down + re-init the custom
    // thread pool with the new size if it differs from the active size.
    m.def("set_cpu_attn_threads", &set_cpu_attn_threads, py::arg("n"),
          "Override JETFIRE_CPU_ATTN_THREADS at runtime.  Mainly for sweeps; "
          "the pool re-inits lazily on the next cpu_topk_attn dispatch.");
    m.def("get_cpu_attn_threads", &jetfire_cpu_attn_threads,
          "Return the currently resolved CPU attn thread count.");

    // ── Stream-level CPU/GPU sync primitives (CUDA-graph compatible) ──
    m.def("stream_write_value64", &stream_write_value64,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("stream_id"), py::arg("flag_device"), py::arg("value"),
          "cudaStreamWriteValue64 wrapper.  Queues a 64-bit write to the "
          "mapped-pinned flag on the given stream; CUDA-graph-capturable. "
          "``flag_device`` must be the device tensor returned by "
          "``alloc_mapped_pinned_pair``.");
    m.def("stream_wait_value64", &stream_wait_value64,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("stream_id"), py::arg("flag_device"), py::arg("value"),
          "cudaStreamWaitValue64 wrapper (EQ).  Queues a stream-side wait "
          "on the mapped-pinned flag until it equals ``value``; CUDA-graph-"
          "capturable.  Resumes the stream once the CPU writes the value.");
    m.def("create_test_host_ctx", &create_test_host_ctx,
          py::arg("counter_host"),
          "Create a long-lived host context wrapping a pinned int64 counter "
          "tensor.  Returns an opaque int handle for stream_launch_test_host_func.");
    m.def("stream_launch_test_host_func", &stream_launch_test_host_func,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("stream_id"), py::arg("ctx_handle"),
          "cudaLaunchHostFunc wrapper for tests.  The callback increments "
          "the int64 counter in the context.  CUDA-graph-capturable.");

    // ── Production cpu_topk_attn submit via host_func ────────────────
    m.def("create_cpu_topk_submit_ctx", &create_cpu_topk_submit_ctx,
          py::arg("q"), py::arg("k_shadow"), py::arg("v_shadow"),
          py::arg("head_map"), py::arg("ts_buf"), py::arg("o_new"),
          py::arg("scale"), py::arg("flag_host"),
          "Create a long-lived submit context referencing the pre-allocated "
          "decode buffers.  All tensors must be CPU; `flag_host` is the host "
          "half of an alloc_mapped_pinned_pair.  Returns an opaque int "
          "handle for stream_submit_cpu_topk_attn.  Ctx lives until process "
          "teardown.");
    m.def("create_cpu_topk_submit_ctx_device_signal",
          &create_cpu_topk_submit_ctx_device_signal,
          py::arg("q"), py::arg("k_shadow"), py::arg("v_shadow"),
          py::arg("head_map"), py::arg("ts_buf"), py::arg("o_new"),
          py::arg("scale"), py::arg("flag_host"), py::arg("signal_device"),
          "Create a submit context whose last CPU worker signals completion "
          "by writing a CUDA int64 device doorbell from a side stream.  This "
          "keeps Stage B overlap without polling mapped host memory.");
    m.def("stream_submit_cpu_topk_attn", &stream_submit_cpu_topk_attn,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("stream_id"), py::arg("ctx_handle"),
          "cudaLaunchHostFunc wrapper for the production cpu_topk_attn "
          "submission.  On replay, the callback dispatches CPU work to "
          "jetfire_pool::Pool and returns in ~5 µs.  The last finishing "
          "worker writes 1 to the mapped pinned flag for legacy GPU-wait "
          "mode; the default path uses side-stream event join.");
    m.def("stream_submit_cpu_topk_attn_side_join",
          &stream_submit_cpu_topk_attn_side_join,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("stream_id"), py::arg("ctx_handle"),
          "Submit CPU topk on the main stream and schedule a side-stream "
          "host join that can overlap with downstream GPU work.");
    m.def("stream_wait_cpu_topk_attn_done_event",
          &stream_wait_cpu_topk_attn_done_event,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("stream_id"), py::arg("ctx_handle"),
          "Make the main stream wait for the side-stream CPU-topk join event.");
    m.def("stream_wait_cpu_topk_attn", &stream_wait_cpu_topk_attn,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("stream_id"), py::arg("ctx_handle"),
          "cudaLaunchHostFunc wrapper for stream-ordered CPU-topk join. "
          "Dispatch CPU work before Stage B, then wait here after Stage B "
          "so CPU/GPU overlap is preserved without GPU doorbell polling.");
    m.def("create_shadow_append_submit_ctx", &create_shadow_append_submit_ctx,
          py::arg("k_append"), py::arg("v_append"), py::arg("slots"),
          py::arg("k_shadow"), py::arg("v_shadow"),
          "Create a long-lived graph replay context for appending decode "
          "K/V tokens into JetFire's CPU shadow cache.  The append tensors "
          "are pinned CPU staging buffers populated by graph-captured D2H "
          "copies; shadow tensors are the per-layer CPU BF16 cache.");
    m.def("stream_submit_shadow_append", &stream_submit_shadow_append,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("stream_id"), py::arg("ctx_handle"),
          "cudaLaunchHostFunc wrapper that scatters staged decode K/V rows "
          "into the CPU shadow cache in stream order.  CUDA-graph-capturable.");
    m.def("graph_pool_warmup", &graph_pool_warmup,
          "Eagerly initialize jetfire_pool::Pool so the first graph replay "
          "doesn't pay thread-spawn latency.");
    m.def("graph_pool_warmup_threads", &graph_pool_warmup_threads,
          py::arg("n_threads"),
          "Initialize or resize the CUDA-graph CPU topk worker pool to an "
          "explicit size before capture.  This keeps replay deterministic for "
          "small local decode row counts.");
    m.def("get_graph_submit_count", &get_graph_submit_count,
          "Return the number of cudaLaunchHostFunc CPU-submit callbacks that "
          "have fired in this process.");
    m.def("get_shadow_append_count", &get_shadow_append_count,
          "Return the number of decode-token shadow-append host callbacks "
          "that have fired in this process.");
    m.def("get_cpu_topk_sanitize_count", &get_cpu_topk_sanitize_count,
          "Return the number of CPU topK rows recomputed by the nonfinite "
          "SIMD-output sanitizer.");
    m.def("get_graph_timing_stats", &get_graph_timing_stats,
          "Return lightweight graph host-callback timing counters: CPU topK "
          "submit latency, submit-to-done latency, and shadow append latency.");
    m.def("reset_graph_submit_count", &reset_graph_submit_count,
          "Reset graph host callback counters.");
    m.def("reset_graph_runtime_state", &reset_graph_runtime_state,
          "Reset graph CPU-topk barrier/counter state without freeing "
          "contexts captured by CUDA graphs.  Safe only when no graph replay "
          "is in flight.");
    m.def("clear_graph_contexts", &clear_graph_contexts,
          "Drop graph submit/append contexts and reset counters.  Must only "
          "be called when no captured/replayed graph can still reference the "
          "old context handles.");
}
