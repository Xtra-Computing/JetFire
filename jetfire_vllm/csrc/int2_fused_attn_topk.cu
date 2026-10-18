// JetFire fused INT2 attention + top-K + pre-correction kernel — Pattern A v0.
//
// Replaces the current decode hot path (Stage A `int2_qk_scores_topk`
// + Stage B `int2_softmax_v_with_pre_corr`) with a SINGLE kernel that
// streams K once and V once per (b, h_q), keeping all intermediate state
// in SMEM/registers.  The (B, H_q, max_S) fp32 ``scores`` and ``weights``
// tensors that the split path materialises in HBM are GONE — that's the
// largest single bandwidth saving identified in the perf review (4 ms/token
// at ctx=32k, B=32).
//
// ── Outputs (drop-in for the split path) ────────────────────────────────
//   o_hat        (B, H_q, D)  fp32  — full INT2 softmax-attention output
//   topk_slots   (B, H_q, K)  int64 — global slot ids of the top-K positions
//   o_old_unnorm (B, H_q, D)  fp32  — Σ_{k ∈ topK} w_full[k] · dequant(V[k])
//   W            (B, H_q)     fp32  — Σ_{k ∈ topK} w_full[k]
// where ``w_full[k] = exp(score[k] − m) / ℓ`` is the full-row softmax
// weight at that position.  Blend formula (split-int2 mode in
// decode_path.py): ``o_corr = o_hat + W · o_new − o_old_unnorm``.
//
// ── Algorithm (per CTA = per (b, h_q)) ──────────────────────────────────
//   Init: m = -inf, ℓ = 0, o_hat = 0, topK_heap = {-inf} × K
//
//   Pass 1 over S in chunks of BS:
//     load K chunk (packed INT2 + scales + zp), dequant to bf16 SMEM
//     score[k] = Q · K[k] for k ∈ chunk
//     update topK heap with (score[k], slot[k]) for valid positions
//     online softmax:
//       chunk_max = max(scores in chunk)
//       m_new = max(m, chunk_max);  α = exp(m − m_new)
//       chunk_sum = Σ exp(score[k] − m_new)
//       ℓ = α·ℓ + chunk_sum;  m = m_new
//     load V chunk (packed INT2 + scales), dequant to bf16 SMEM
//     o_hat[d] = α · o_hat[d] + Σ_k exp(score[k] − m_new) · V[k][d]
//
//   Final: o_hat /= ℓ
//
//   Pass 2 over topK (K positions):
//     w_full[k] = exp(top_score[k] − m) / ℓ
//     load V[top_slot[k]], dequant
//     W += w_full[k]
//     o_old_unnorm[d] += w_full[k] · V[top_slot[k]][d]
//
//   Translate top_slot via block_table → topk_slots[b][h_q] (already global)
//
// ── v0 implementation note ──────────────────────────────────────────────
// This is the CORRECTNESS-FIRST version: scalar inner loops, block-wide
// reductions for dots, single-warp top-K min-replace.  No WMMA, no
// cp.async, no double-buffered SMEM.  It exists to (1) validate the
// algorithm bit-for-bit against the existing split path on the unit
// test, and (2) give us a known-good reference for the optimised
// follow-up that ports the same algorithm to WMMA + cp.async + a
// register-resident topK heap.
//
// Behind ``JETFIRE_USE_FUSED_ATTN_TOPK=0`` (default OFF) until both
// correctness and perf are validated end-to-end.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>


namespace {

constexpr int D_FIXED   = 128;        // LLaMA / Mistral standard head dim
constexpr int D_QUARTER = D_FIXED / 4; // 32 packed bytes per row (4 INT2 / byte)
constexpr int BS_FIXED  = 16;         // KV block size (matches paper config)
constexpr int THREADS   = 128;        // 4 warps; D == THREADS so 1 element/thread
constexpr int K_MAX     = 128;        // upper bound on top_k (heap capacity)


// ── Shared block-reduction helpers (mirrors int2_softmax_v.cu) ─────────
__device__ __forceinline__ float block_max_(float v, float* smem32) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) smem32[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 4) ? smem32[lane] : -INFINITY;
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1)
            t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, off));
        if (lane == 0) smem32[0] = t;
    }
    __syncthreads();
    return smem32[0];
}

__device__ __forceinline__ float block_sum_(float v, float* smem32) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) smem32[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 4) ? smem32[lane] : 0.0f;
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1)
            t += __shfl_xor_sync(0xffffffffu, t, off);
        if (lane == 0) smem32[0] = t;
    }
    __syncthreads();
    return smem32[0];
}


// ── INT2 dequant: same packing as int2_qk_scores.cu / fused_decode_wmma.cu ─
// Packing: 1 byte = 4 signed INT2 values, big-endian, offset-encoded
// (stored = q + 2, so q ∈ {-2, -1, 0, +1} after subtract).
//   bits[7:6] = v0, bits[5:4] = v1, bits[3:2] = v2, bits[1:0] = v3
__device__ __forceinline__ float dequant_one_int2(uint8_t p, int sub,
                                                   float scale, float zp) {
    int q = int((p >> ((3 - sub) * 2)) & 0x3) - 2;
    return float(q) * scale + zp;
}


// ── The kernel ─────────────────────────────────────────────────────────
// Grid: (B, H_q).  Block: 128 threads, D=128 elements, BS=16 positions.
//
// SMEM layout (~10 KB):
//   q_smem      [D]            fp32   — Q for this (b, h_q), once
//   o_hat_smem  [D]            fp32   — running o_hat
//   o_old_smem  [D]            fp32   — running o_old_unnorm
//   k_chunk     [BS][D]        bf16   — dequantised K for current chunk
//   v_chunk     [BS][D]        bf16   — dequantised V for current chunk
//   scores      [BS]           fp32   — Q·K for current chunk
//   topk_keys   [K_MAX]        fp32   — heap of top-K scores
//   topk_pos    [K_MAX]        int32  — heap of top-K positions (S-domain)
//   reduce      [4]            fp32   — block-reduce scratch (4 warps)
//   m, ell      [2]            fp32
//
// Top-K heap: linear array, NOT a heap structure — we do a min-replace
// by linear scan (warp 0 only).  At K=64 this is 64 ops per inserted
// candidate; the per-chunk inserter checks BS=16 candidates against the
// 64-slot heap, so at most 16 × 64 = 1024 fmaxf per chunk worst-case.
// At ctx=32k that's 32000/16 × 1024 = 2M ops — which is FINE (it's all
// SMEM, single warp, ~10 µs total).  Optimised version will move to a
// register-distributed warp-collective heap (à la topk_slots_kernel_v3).
__global__ void int2_fused_attn_topk_kernel(
    const __nv_bfloat16* __restrict__ Q,            // (B, H_q, D)
    const uint8_t*       __restrict__ K_data,       // (H_kv, num_slots, D/4)
    const __nv_bfloat16* __restrict__ K_scales,     // (H_kv, num_slots) bf16
    const __nv_bfloat16* __restrict__ K_zp,         // (H_kv, num_slots) bf16
    const uint8_t*       __restrict__ V_data,       // (H_kv, num_slots, D/4)
    const __nv_bfloat16* __restrict__ V_scales,     // (H_kv, num_slots) bf16
    const int64_t*       __restrict__ BlockTable,   // (B, max_blocks)
    const int32_t*       __restrict__ SeqLens,      // (B,)
    float*               __restrict__ O_hat,        // (B, H_q, D)        [out]
    int64_t*             __restrict__ Topk_slots,   // (B, H_q, K)        [out]
    float*               __restrict__ O_old_unnorm, // (B, H_q, D)        [out]
    float*               __restrict__ W_out,        // (B, H_q)           [out]
    int max_S, int H_q, int H_kv,
    int block_size, int block_table_stride,
    int K_data_stride_h, int K_scales_stride_h,
    int top_k,
    float attn_scale)
{
    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;
    const int num_groups = H_q / H_kv;
    const int h_kv = h_q / num_groups;
    const int S    = SeqLens[b];

    // ── Dynamic SMEM layout ────────────────────────────────────────────
    extern __shared__ __align__(16) uint8_t dyn_smem[];
    size_t off = 0;
    float* q_smem        = reinterpret_cast<float*>(&dyn_smem[off]);
    off += D_FIXED * sizeof(float);
    float* o_hat_smem    = reinterpret_cast<float*>(&dyn_smem[off]);
    off += D_FIXED * sizeof(float);
    float* o_old_smem    = reinterpret_cast<float*>(&dyn_smem[off]);
    off += D_FIXED * sizeof(float);
    auto k_chunk = reinterpret_cast<__nv_bfloat16(*)[D_FIXED]>(&dyn_smem[off]);
    off += (size_t)BS_FIXED * D_FIXED * sizeof(__nv_bfloat16);
    auto v_chunk = reinterpret_cast<__nv_bfloat16(*)[D_FIXED]>(&dyn_smem[off]);
    off += (size_t)BS_FIXED * D_FIXED * sizeof(__nv_bfloat16);
    float* scores_smem   = reinterpret_cast<float*>(&dyn_smem[off]);
    off += BS_FIXED * sizeof(float);
    float*   topk_keys   = reinterpret_cast<float*>(&dyn_smem[off]);
    off += K_MAX * sizeof(float);
    int*     topk_pos    = reinterpret_cast<int*>(&dyn_smem[off]);
    off += K_MAX * sizeof(int);
    float*   reduce_buf  = reinterpret_cast<float*>(&dyn_smem[off]);
    off += 8 * sizeof(float);
    float*   ml_smem     = reinterpret_cast<float*>(&dyn_smem[off]);
    // ml_smem[0] = m, ml_smem[1] = ℓ

    // ── Init ───────────────────────────────────────────────────────────
    // Q from HBM (D elements; THREADS == D so 1 element per thread).
    if (tid < D_FIXED) {
        size_t q_off = (size_t)b * H_q * D_FIXED + (size_t)h_q * D_FIXED + tid;
        q_smem[tid] = __bfloat162float(Q[q_off]);
        o_hat_smem[tid] = 0.0f;
        o_old_smem[tid] = 0.0f;
    }
    if (tid == 0) { ml_smem[0] = -INFINITY; ml_smem[1] = 0.0f; }
    if (tid < top_k) {
        topk_keys[tid] = -INFINITY;
        topk_pos[tid]  = -1;
    }
    __syncthreads();

    const uint8_t*       k_pack_base = K_data   + (size_t)h_kv * K_data_stride_h;
    const uint8_t*       v_pack_base = V_data   + (size_t)h_kv * K_data_stride_h;
    const __nv_bfloat16* ks_base     = K_scales + (size_t)h_kv * K_scales_stride_h;
    const __nv_bfloat16* kz_base     = K_zp     + (size_t)h_kv * K_scales_stride_h;
    const __nv_bfloat16* vs_base     = V_scales + (size_t)h_kv * K_scales_stride_h;
    const int64_t*       bt_row      = BlockTable + (size_t)b * block_table_stride;

    // ── Pass 1: stream K and V in BS-sized chunks, online softmax ──────
    const int num_chunks = (S + BS_FIXED - 1) / BS_FIXED;

    for (int c = 0; c < num_chunks; ++c) {
        const int chunk_s0   = c * BS_FIXED;
        const int chunk_end  = min(chunk_s0 + BS_FIXED, S);
        const int chunk_rows = chunk_end - chunk_s0;

        // (a) Load K chunk and dequant → k_chunk[r][d] (bf16).
        //     128 threads cooperate: BS_FIXED rows × D = 16 × 128 = 2048
        //     work items, exactly 16 per thread.
        for (int i = tid; i < BS_FIXED * D_FIXED; i += THREADS) {
            int r = i / D_FIXED;
            int d = i % D_FIXED;
            if (r >= chunk_rows) {
                k_chunk[r][d] = __float2bfloat16(0.f);
                continue;
            }
            int pos = chunk_s0 + r;
            int64_t slot = bt_row[pos / block_size] * (int64_t)block_size
                           + (pos % block_size);
            uint8_t pk = k_pack_base[slot * D_QUARTER + d / 4];
            float ks = __bfloat162float(ks_base[slot]);
            float kz = __bfloat162float(kz_base[slot]);
            float v  = dequant_one_int2(pk, d & 3, ks, kz);
            k_chunk[r][d] = __float2bfloat16(v);
        }
        __syncthreads();

        // (b) scores[r] = Q · K[r] for r ∈ [0, chunk_rows).
        //     One thread per d: each row needs a 128-way reduction.
        //     Do `chunk_rows` separate block-reductions; cheap (BS_FIXED=16).
        for (int r = 0; r < BS_FIXED; ++r) {
            float partial = 0.0f;
            if (tid < D_FIXED && r < chunk_rows) {
                partial = q_smem[tid] * __bfloat162float(k_chunk[r][tid]);
            }
            float s = block_sum_(partial, reduce_buf);
            if (tid == 0) {
                scores_smem[r] = (r < chunk_rows)
                                    ? (s * attn_scale)
                                    : -INFINITY;
            }
            __syncthreads();
        }

        // (c) Top-K min-replace by warp 0.  Thread 0 walks the K candidates
        //     against the K_MAX-slot heap; cheap at top_k ≤ 128 and BS=16.
        if (tid == 0) {
            for (int r = 0; r < chunk_rows; ++r) {
                float new_key = scores_smem[r];
                int   new_idx = chunk_s0 + r;
                int   min_lane = 0;
                float min_key  = topk_keys[0];
                for (int k = 1; k < top_k; ++k) {
                    float kk = topk_keys[k];
                    if (kk < min_key) { min_key = kk; min_lane = k; }
                }
                if (new_key > min_key) {
                    topk_keys[min_lane] = new_key;
                    topk_pos [min_lane] = new_idx;
                }
            }
        }
        __syncthreads();

        // (d) Online softmax update.
        //     chunk_max → m_new, α; chunk_sum → ℓ_new.
        float local_max = (tid < chunk_rows) ? scores_smem[tid] : -INFINITY;
        float chunk_max = block_max_(local_max, reduce_buf);
        float old_m = ml_smem[0];
        float old_l = ml_smem[1];
        float m_new = fmaxf(old_m, chunk_max);
        float alpha = expf(old_m - m_new);  // 0 when old_m = -inf and chunk_max finite

        // Write per-row exp(score - m_new) back into scores_smem[r] in-place
        // so the V-multiply step below can read them.
        if (tid < BS_FIXED) {
            float w = (tid < chunk_rows && isfinite(scores_smem[tid]))
                          ? __expf(scores_smem[tid] - m_new)
                          : 0.0f;
            scores_smem[tid] = w;
        }
        __syncthreads();

        float local_sum = (tid < BS_FIXED) ? scores_smem[tid] : 0.0f;
        float chunk_sum = block_sum_(local_sum, reduce_buf);
        float l_new = old_l * alpha + chunk_sum;

        // Rescale running o_hat by α (each thread owns 1 element).
        if (tid < D_FIXED && alpha != 1.0f) {
            o_hat_smem[tid] *= alpha;
        }

        if (tid == 0) {
            ml_smem[0] = m_new;
            ml_smem[1] = l_new;
        }

        // (e) Load V chunk and dequant → v_chunk (bf16).
        for (int i = tid; i < BS_FIXED * D_FIXED; i += THREADS) {
            int r = i / D_FIXED;
            int d = i % D_FIXED;
            if (r >= chunk_rows) {
                v_chunk[r][d] = __float2bfloat16(0.f);
                continue;
            }
            int pos = chunk_s0 + r;
            int64_t slot = bt_row[pos / block_size] * (int64_t)block_size
                           + (pos % block_size);
            uint8_t pv = v_pack_base[slot * D_QUARTER + d / 4];
            float vs = __bfloat162float(vs_base[slot]);
            float v  = dequant_one_int2(pv, d & 3, vs, 0.0f);
            v_chunk[r][d] = __float2bfloat16(v);
        }
        __syncthreads();

        // (f) o_hat[d] += Σ_r exp(score[r] - m_new) · V[r][d]
        //     Each thread owns one d ∈ [0, D); accumulates BS_FIXED multiplies.
        if (tid < D_FIXED) {
            float acc = 0.0f;
            #pragma unroll
            for (int r = 0; r < BS_FIXED; ++r) {
                if (r >= chunk_rows) break;
                acc += scores_smem[r] * __bfloat162float(v_chunk[r][tid]);
            }
            o_hat_smem[tid] += acc;
        }
        __syncthreads();
    }

    // ── Final normalization: o_hat /= ℓ ────────────────────────────────
    const float ell_final = ml_smem[1];
    const float inv_ell   = (ell_final > 0.0f) ? (1.0f / ell_final) : 0.0f;
    if (tid < D_FIXED) {
        o_hat_smem[tid] *= inv_ell;
    }
    const float m_final = ml_smem[0];

    // ── Pass 2: top-K → o_old_unnorm + W ───────────────────────────────
    // Each top-K entry: load V[slot], compute weight = exp(score-m)/ℓ,
    // accumulate.  Exactly K passes over D elements.
    float w_total = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int   pos = topk_pos[k];
        float key = topk_keys[k];
        if (pos < 0) continue;   // unfilled slot (top_k > S)
        float w_full = __expf(key - m_final) * inv_ell;
        if (tid == 0) w_total += w_full;

        int64_t slot = bt_row[pos / block_size] * (int64_t)block_size
                       + (pos % block_size);
        if (tid < D_FIXED) {
            uint8_t pv = v_pack_base[slot * D_QUARTER + tid / 4];
            float vs = __bfloat162float(vs_base[slot]);
            float v  = dequant_one_int2(pv, tid & 3, vs, 0.0f);
            o_old_smem[tid] += w_full * v;
        }
    }
    __syncthreads();

    // ── Write outputs ──────────────────────────────────────────────────
    if (tid < D_FIXED) {
        size_t o_off = (size_t)b * H_q * D_FIXED + (size_t)h_q * D_FIXED + tid;
        O_hat[o_off]        = o_hat_smem[tid];
        O_old_unnorm[o_off] = o_old_smem[tid];
    }
    if (tid == 0) {
        W_out[(size_t)b * H_q + h_q] = w_total;
    }

    // Translate top-K positions to global slot ids.
    if (tid < top_k) {
        int pos = topk_pos[tid];
        int64_t slot = (pos >= 0)
                         ? bt_row[pos / block_size] * (int64_t)block_size
                           + (pos % block_size)
                         : (int64_t)0;
        Topk_slots[((size_t)b * H_q + h_q) * top_k + tid] = slot;
    }
}


// ── Host launcher ──────────────────────────────────────────────────────
std::vector<torch::Tensor> int2_fused_attn_topk(
    torch::Tensor q,           // (B, H_q, D=128)         bf16  CUDA
    torch::Tensor k_data,      // (H_kv, num_slots, D/4)  uint8 CUDA
    torch::Tensor k_scales,    // (H_kv, num_slots[, 1])  bf16  CUDA
    torch::Tensor k_zp,        // (H_kv, num_slots[, 1])  bf16  CUDA
    torch::Tensor v_data,      // (H_kv, num_slots, D/4)  uint8 CUDA
    torch::Tensor v_scales,    // (H_kv, num_slots[, 1])  bf16  CUDA
    torch::Tensor block_table, // (B, max_blocks)         int64 CUDA
    torch::Tensor seq_lens,    // (B,)                    int32 CUDA
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t top_k,
    double  attn_scale)
{
    TORCH_CHECK(q.is_cuda() && k_data.is_cuda() && v_data.is_cuda());
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_data.scalar_type() == torch::kByte);
    TORCH_CHECK(v_data.scalar_type() == torch::kByte);
    TORCH_CHECK(k_scales.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_zp.scalar_type()     == torch::kBFloat16);
    TORCH_CHECK(v_scales.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(q.dim() == 3 && q.size(2) == D_FIXED,
                "q must be (B, H_q, 128)");
    TORCH_CHECK(block_size == BS_FIXED,
                "v0 supports block_size == 16 only; got ", block_size);
    TORCH_CHECK(top_k > 0 && top_k <= K_MAX,
                "top_k must be in [1, ", K_MAX, "]; got ", top_k);
    TORCH_CHECK(H_q % H_kv == 0);

    auto qc  = q.contiguous();
    auto kdc = k_data.contiguous();
    auto ks  = (k_scales.dim() == 3) ? k_scales.squeeze(-1).contiguous()
                                     : k_scales.contiguous();
    auto kz  = (k_zp.dim()     == 3) ? k_zp.squeeze(-1).contiguous()
                                     : k_zp.contiguous();
    auto vd  = v_data.contiguous();
    auto vs  = (v_scales.dim() == 3) ? v_scales.squeeze(-1).contiguous()
                                     : v_scales.contiguous();
    auto bt  = block_table.contiguous();
    auto sl  = seq_lens.contiguous();

    const int B          = qc.size(0);
    const int max_blocks = (int)bt.size(1);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(qc.device());
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64  ).device(qc.device());

    auto o_hat        = torch::empty({B, (int64_t)H_q, (int64_t)D_FIXED},  opts_f32);
    auto topk_slots   = torch::empty({B, (int64_t)H_q, (int64_t)top_k},    opts_i64);
    auto o_old_unnorm = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED},  opts_f32);
    auto W            = torch::zeros({B, (int64_t)H_q},                    opts_f32);

    const size_t smem_bytes =
          D_FIXED * sizeof(float)                               // q_smem
        + D_FIXED * sizeof(float)                               // o_hat_smem
        + D_FIXED * sizeof(float)                               // o_old_smem
        + (size_t)BS_FIXED * D_FIXED * sizeof(__nv_bfloat16)    // k_chunk
        + (size_t)BS_FIXED * D_FIXED * sizeof(__nv_bfloat16)    // v_chunk
        + BS_FIXED * sizeof(float)                              // scores
        + K_MAX * sizeof(float)                                 // topk_keys
        + K_MAX * sizeof(int)                                   // topk_pos
        + 8 * sizeof(float)                                     // reduce_buf
        + 2 * sizeof(float);                                    // m, ℓ

    cudaFuncSetAttribute(int2_fused_attn_topk_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);

    dim3 grid((unsigned)B, (unsigned)H_q);
    dim3 block(THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();

    int2_fused_attn_topk_kernel<<<grid, block, smem_bytes, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr<at::BFloat16>()),
        kdc.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kz.data_ptr<at::BFloat16>()),
        vd.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        sl.data_ptr<int32_t>(),
        o_hat.data_ptr<float>(),
        topk_slots.data_ptr<int64_t>(),
        o_old_unnorm.data_ptr<float>(),
        W.data_ptr<float>(),
        (int)max_S, (int)H_q, (int)H_kv,
        (int)block_size, (int)bt.stride(0),
        (int)kdc.stride(0), (int)ks.stride(0),
        (int)top_k,
        (float)attn_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {o_hat, topk_slots, o_old_unnorm, W};
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int2_fused_attn_topk", &int2_fused_attn_topk,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"), py::arg("k_zp"),
          py::arg("v_data"), py::arg("v_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("top_k"), py::arg("attn_scale"),
          "Pattern A v0: fused INT2 attention + topK + pre-correction.  "
          "Returns [o_hat, topk_slots, o_old_unnorm, W].  Replaces the split "
          "Stage A + Stage B path; no (B,H_q,max_S) scores tensor in HBM.  "
          "v0 is correctness-first (scalar inner loops); WMMA/cp.async port "
          "is the next step.");
}
