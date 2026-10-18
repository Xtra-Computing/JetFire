// JetFire fused INT2 attention + top-K + pre-correction kernel — Pattern A v1.
//
// WMMA-bf16 + cp.async port of the validated v0 algorithm in
// ``int2_fused_attn_topk.cu``.  Same outputs, same algorithm; just MUCH
// faster.  Uses the proven kernel skeleton from ``fused_decode_wmma.cu``'s
// v5 INT2 kernel (Phase A: WMMA Q·K, Phase B: online softmax, Phase C: WMMA
// P·V) and adds a per-warp register-distributed top-K heap (à la
// ``topk_slots_kernel_v3``).
//
// ── Grid / parallelism ─────────────────────────────────────────────────
// Grid: (B, H_kv).  Block: 128 threads = 4 warps.  One CTA processes
// num_groups = H_q / H_kv consecutive query rows for one (b, h_kv) pair —
// exact same shape as the existing v5 kernel with NS_EQ_1=true.  WMMA
// M-dim 16 holds (num_groups real rows + padding zeros).  Each warp w:
//   * Owns one query row r = w (when w < num_groups)
//   * Maintains its own top-K heap, register-distributed across 32 lanes
//     (K_PER_LANE = top_k / 32 entries per lane; max top_k = 128)
//   * Does Phase B online softmax for row r
//   * Contributes to Phase C P·V WMMA for row r
//
// ── Outputs (drop-in for v0) ───────────────────────────────────────────
//   o_hat        (B, H_q, D)   fp32  — full INT2 softmax-attention output
//   topk_slots   (B, H_q, K)   int64 — global slot ids of the top-K positions
//   o_old_unnorm (B, H_q, D)   fp32  — Σ_{k ∈ topK} w_full[k] · dequant(V[k])
//   W            (B, H_q)      fp32  — Σ_{k ∈ topK} w_full[k]
// where w_full[k] = exp(score[k] − m) / ℓ.
//
// ── Constraints (v1 dispatch) ──────────────────────────────────────────
//   D == 128, block_size == 16, top_k ∈ {32, 64, 96, 128}, top_k % 32 == 0
//   H_q % H_kv == 0 and num_groups ∈ {1, 2, 4} (4 = LLaMA standard)
//
// Behind ``JETFIRE_USE_FUSED_ATTN_TOPK=1`` (default OFF) and dispatched
// over v0 via the same flag — host launcher picks v1 when constraints
// match, else falls back to v0 / split path.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>


namespace {

constexpr int D            = 128;
constexpr int D_QUARTER    = D / 4;            // 32 packed bytes per row
constexpr int BLOCK_S      = 64;               // Tile width (matches v5's BLOCK_S_V4)
constexpr int THREADS      = 128;              // 4 warps × 32
constexpr int STAGES       = 2;                // double-buffered cp.async pipeline

// SMEM padding to break bank-aliased WMMA loads (same as v5).
constexpr int Q_PAD        = 8;
constexpr int Q_STRIDE     = D + Q_PAD;
constexpr int K_PAD        = 8;
constexpr int K_STRIDE     = D + K_PAD;
constexpr int V_PAD        = 8;
constexpr int V_STRIDE     = D + V_PAD;
constexpr int P_PAD        = 8;
constexpr int P_STRIDE     = BLOCK_S + P_PAD;
constexpr int SCORE_PAD    = 4;
constexpr int SCORE_STRIDE = BLOCK_S + SCORE_PAD;
constexpr int O_PAD        = 4;
constexpr int O_STRIDE     = D + O_PAD;

// Per-lane top-K heap state.  K_PER_LANE_MAX = K_MAX_TOPK / 32 = 4, so
// top_k can be 32/64/96/128.  At runtime we use only the first
// (top_k/32) entries; the rest stay -INFINITY.
constexpr int K_MAX_TOPK   = 128;
constexpr int K_PER_LANE_MAX = K_MAX_TOPK / 32;  // 4


__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned smem_int = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_int), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_group_0() {
    asm volatile("cp.async.wait_group 0;");
}
__device__ __forceinline__ void cp_async_wait_group_1() {
    asm volatile("cp.async.wait_group 1;");
}


// INT2 dequant: 1 byte = 4 signed INT2, big-endian, offset-encoded
// (stored = q + 2 ∈ {0,1,2,3}, so q ∈ {-2,-1,0,1}).
__device__ __forceinline__ void int2_unpack_bf16(
    uint8_t p, float scale, float zp,
    __nv_bfloat16& v0, __nv_bfloat16& v1,
    __nv_bfloat16& v2, __nv_bfloat16& v3)
{
    int q0 = int((p >> 6) & 0x3) - 2;
    int q1 = int((p >> 4) & 0x3) - 2;
    int q2 = int((p >> 2) & 0x3) - 2;
    int q3 = int( p       & 0x3) - 2;
    v0 = __float2bfloat16(float(q0) * scale + zp);
    v1 = __float2bfloat16(float(q1) * scale + zp);
    v2 = __float2bfloat16(float(q2) * scale + zp);
    v3 = __float2bfloat16(float(q3) * scale + zp);
}


// ── The kernel ─────────────────────────────────────────────────────────
// Templated on K_PER_LANE so the heap-update loop unrolls.
template <int K_PER_LANE>
__global__ void int2_fused_attn_topk_v1_kernel(
    const __nv_bfloat16* __restrict__ Q,             // (B, H_q, D)
    const uint8_t*       __restrict__ K_data,        // (H_kv, num_slots, D/4)
    const __nv_bfloat16* __restrict__ K_scales,      // (H_kv, num_slots) bf16
    const __nv_bfloat16* __restrict__ K_zp,          // (H_kv, num_slots) bf16
    const uint8_t*       __restrict__ V_data,        // (H_kv, num_slots, D/4)
    const __nv_bfloat16* __restrict__ V_scales,      // (H_kv, num_slots) bf16
    const int64_t*       __restrict__ BlockTable,    // (B, max_blocks)
    const int32_t*       __restrict__ SeqLens,       // (B,)
    float*               __restrict__ O_hat,         // (B, H_q, D)        [out]
    int64_t*             __restrict__ Topk_slots,    // (B, H_q, K)        [out]
    float*               __restrict__ O_old_unnorm,  // (B, H_q, D)        [out]
    float*               __restrict__ W_out,         // (B, H_q)           [out]
    int max_S, int H_q, int H_kv,
    int block_size, int block_table_stride,
    int K_data_stride_h, int K_scales_stride_h,
    int top_k,
    float attn_scale)
{
    const int b       = blockIdx.x;
    const int h_kv    = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int num_groups = H_q / H_kv;          // 1, 2 or 4
    const int S          = SeqLens[b];

    // ── Dynamic SMEM layout (mirrors v5) ───────────────────────────────
    extern __shared__ __align__(16) uint8_t dyn_smem[];
    size_t off = 0;
    auto q_smem  = reinterpret_cast<__nv_bfloat16(*)[Q_STRIDE]>(&dyn_smem[off]);
    off += 16 * Q_STRIDE * sizeof(__nv_bfloat16);
    auto k_pack  = reinterpret_cast<uint8_t(*)[BLOCK_S][D_QUARTER]>(&dyn_smem[off]);
    off += STAGES * BLOCK_S * D_QUARTER;
    auto v_pack  = reinterpret_cast<uint8_t(*)[BLOCK_S][D_QUARTER]>(&dyn_smem[off]);
    off += STAGES * BLOCK_S * D_QUARTER;
    auto k_scale = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S]>(&dyn_smem[off]);
    off += STAGES * BLOCK_S * sizeof(__nv_bfloat16);
    auto k_zp_s  = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S]>(&dyn_smem[off]);
    off += STAGES * BLOCK_S * sizeof(__nv_bfloat16);
    auto v_scale = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S]>(&dyn_smem[off]);
    off += STAGES * BLOCK_S * sizeof(__nv_bfloat16);
    auto k_smem  = reinterpret_cast<__nv_bfloat16(*)[K_STRIDE]>(&dyn_smem[off]);
    off += BLOCK_S * K_STRIDE * sizeof(__nv_bfloat16);
    auto v_smem  = reinterpret_cast<__nv_bfloat16(*)[V_STRIDE]>(&dyn_smem[off]);
    off += BLOCK_S * V_STRIDE * sizeof(__nv_bfloat16);
    auto score_smem = reinterpret_cast<float(*)[SCORE_STRIDE]>(&dyn_smem[off]);
    off += 16 * SCORE_STRIDE * sizeof(float);
    auto p_smem  = reinterpret_cast<__nv_bfloat16(*)[P_STRIDE]>(&dyn_smem[off]);
    off += 16 * P_STRIDE * sizeof(__nv_bfloat16);
    auto o_smem  = reinterpret_cast<float(*)[O_STRIDE]>(&dyn_smem[off]);
    off += 16 * O_STRIDE * sizeof(float);
    float* m_smem = reinterpret_cast<float*>(&dyn_smem[off]);
    off += 16 * sizeof(float);
    float* l_smem = reinterpret_cast<float*>(&dyn_smem[off]);
    off += 16 * sizeof(float);
    // Phase 2 broadcast staging: per-row top-K entries written here from
    // the warp-distributed register heaps before the o_old / W loop.
    auto topk_keys_sm = reinterpret_cast<float(*)[K_MAX_TOPK]>(&dyn_smem[off]);
    off += 4 * K_MAX_TOPK * sizeof(float);
    auto topk_pos_sm  = reinterpret_cast<int(*)[K_MAX_TOPK]>(&dyn_smem[off]);
    // off += 4 * K_MAX_TOPK * sizeof(int);  // last region

    // ── Init Q (load + zero-pad to 16 rows) ────────────────────────────
    for (int i = tid; i < (16 - num_groups) * D; i += THREADS) {
        int r = num_groups + i / D;
        int c = i % D;
        q_smem[r][c] = __float2bfloat16(0.f);
    }
    for (int i = tid; i < num_groups * D; i += THREADS) {
        int r = i / D;
        int c = i % D;
        int h_q = h_kv * num_groups + r;
        q_smem[r][c] = Q[(size_t)b * H_q * D + (size_t)h_q * D + c];
    }
    if (tid < 16) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.0f;
    }
    for (int i = tid; i < 16 * D; i += THREADS) {
        o_smem[i / D][i % D] = 0.0f;
    }

    // ── Per-warp register top-K heap (one per query row r=warp_id) ─────
    // K_PER_LANE entries per lane × 32 lanes per warp = top_k entries per
    // warp.  Inactive warps (warp_id >= num_groups) keep an empty heap;
    // their heap state is unused but the lanes still execute the update
    // logic (cheap, no SMEM/HBM access).
    float topk_keys[K_PER_LANE_MAX];
    int   topk_pos [K_PER_LANE_MAX];
    #pragma unroll
    for (int i = 0; i < K_PER_LANE_MAX; ++i) {
        topk_keys[i] = -INFINITY;
        topk_pos [i] = -1;
    }
    // FILL_CHUNKS = K_PER_LANE: number of 32-position chunks needed to fill
    // the heap exactly once.  After consuming this many chunks per warp,
    // switch from bulk-fill to min-replace.
    const int FILL_CHUNKS = K_PER_LANE;

    // Pointer bases.
    const uint8_t*       k_pack_base = K_data   + (size_t)h_kv * K_data_stride_h;
    const uint8_t*       v_pack_base = V_data   + (size_t)h_kv * K_data_stride_h;
    const __nv_bfloat16* ks_base     = K_scales + (size_t)h_kv * K_scales_stride_h;
    const __nv_bfloat16* kz_base     = K_zp     + (size_t)h_kv * K_scales_stride_h;
    const __nv_bfloat16* vs_base     = V_scales + (size_t)h_kv * K_scales_stride_h;
    const int64_t*       bt_row      = BlockTable + (size_t)b * block_table_stride;

    const int num_tiles = (S + BLOCK_S - 1) / BLOCK_S;
    if (num_tiles <= 0 || S <= 0) {
        // S == 0: write zeros, exit.
        for (int i = tid; i < num_groups * D; i += THREADS) {
            int r = i / D, c = i % D;
            int h_q = h_kv * num_groups + r;
            size_t o_off = (size_t)b * H_q * D + (size_t)h_q * D + c;
            O_hat[o_off]        = 0.0f;
            O_old_unnorm[o_off] = 0.0f;
        }
        if (tid < num_groups) {
            int h_q = h_kv * num_groups + tid;
            W_out[(size_t)b * H_q + h_q] = 0.0f;
        }
        if (tid < num_groups * top_k) {
            int r = tid / top_k;
            int k = tid % top_k;
            int h_q = h_kv * num_groups + r;
            Topk_slots[((size_t)b * H_q + h_q) * top_k + k] = 0;
        }
        return;
    }

    // Async load helper: stage's K_pack + V_pack + scales/zp.
    auto issue_load = [&](int stage, int tile_s0, int tile_end) {
        int tt = tid & (BLOCK_S - 1);
        bool load_v = (tid >= BLOCK_S);
        int pos = tile_s0 + tt;
        int64_t slot = (pos < tile_end)
                         ? (bt_row[pos / block_size] * (int64_t)block_size + pos % block_size)
                         : -1;

        if (!load_v) {
            if (slot >= 0) {
                const uint8_t* src = k_pack_base + slot * (int64_t)D_QUARTER;
                uint8_t*       dst = &k_pack[stage][tt][0];
                cp_async16(dst,      src);
                cp_async16(dst + 16, src + 16);
                k_scale[stage][tt] = ks_base[slot];
                k_zp_s [stage][tt] = kz_base[slot];
            } else {
                uint8_t* dst = &k_pack[stage][tt][0];
                #pragma unroll
                for (int i = 0; i < D_QUARTER; ++i) dst[i] = 0;
                k_scale[stage][tt] = __float2bfloat16(0.f);
                k_zp_s [stage][tt] = __float2bfloat16(0.f);
            }
        } else {
            if (slot >= 0) {
                const uint8_t* src = v_pack_base + slot * (int64_t)D_QUARTER;
                uint8_t*       dst = &v_pack[stage][tt][0];
                cp_async16(dst,      src);
                cp_async16(dst + 16, src + 16);
                v_scale[stage][tt] = vs_base[slot];
            } else {
                uint8_t* dst = &v_pack[stage][tt][0];
                #pragma unroll
                for (int i = 0; i < D_QUARTER; ++i) dst[i] = 0;
                v_scale[stage][tt] = __float2bfloat16(0.f);
            }
        }
        cp_async_commit();
    };

    // Unpack helper: BLOCK_S × D_QUARTER = 2048 work items / 128 threads = 16/thread.
    auto unpack_tile = [&](int stage) {
        for (int i = tid; i < BLOCK_S * D_QUARTER; i += THREADS) {
            int r = i / D_QUARTER;
            int bx = i % D_QUARTER;
            uint8_t pk = k_pack[stage][r][bx];
            float ks = __bfloat162float(k_scale[stage][r]);
            float kz = __bfloat162float(k_zp_s [stage][r]);
            __nv_bfloat16 v0, v1, v2, v3;
            int2_unpack_bf16(pk, ks, kz, v0, v1, v2, v3);
            k_smem[r][4*bx + 0] = v0;
            k_smem[r][4*bx + 1] = v1;
            k_smem[r][4*bx + 2] = v2;
            k_smem[r][4*bx + 3] = v3;
            uint8_t pv = v_pack[stage][r][bx];
            float vs = __bfloat162float(v_scale[stage][r]);
            int2_unpack_bf16(pv, vs, 0.0f, v0, v1, v2, v3);
            v_smem[r][4*bx + 0] = v0;
            v_smem[r][4*bx + 1] = v1;
            v_smem[r][4*bx + 2] = v2;
            v_smem[r][4*bx + 3] = v3;
        }
    };

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>  a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major>  b_qk_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major>  b_pv_frag;
    fragment<accumulator, 16, 16, 16, float> qk_acc;
    fragment<accumulator, 16, 16, 16, float> pv_acc;

    // Prologue: stage 0 load.
    issue_load(0, 0, min(BLOCK_S, S));
    int loaded = 1;

    // Per-warp tile counter (used for FILL vs MIN-REPLACE phase decision).
    int tiles_processed = 0;

    // ── Main tile loop ─────────────────────────────────────────────────
    for (int t = 0; t < num_tiles; ++t) {
        const int stage     = t % STAGES;
        const int tile_s0   = t * BLOCK_S;
        const int tile_end  = min(tile_s0 + BLOCK_S, S);
        const int tile_rows = tile_end - tile_s0;

        if (loaded < num_tiles) {
            int next_s0 = loaded * BLOCK_S;
            int next_end = min(next_s0 + BLOCK_S, S);
            issue_load(loaded % STAGES, next_s0, next_end);
            loaded++;
        }
        if (loaded > t + 1) cp_async_wait_group_1();
        else                cp_async_wait_group_0();
        __syncthreads();

        unpack_tile(stage);
        __syncthreads();

        // ── Phase A: Q·K^T → score_smem (16, BLOCK_S) ──────────────────
        {
            int n_tile  = warp_id;
            int n_start = n_tile * 16;
            fill_fragment(qk_acc, 0.f);
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                load_matrix_sync(a_frag, &q_smem[0][k], Q_STRIDE);
                load_matrix_sync(b_qk_frag, &k_smem[n_start][k], K_STRIDE);
                mma_sync(qk_acc, a_frag, b_qk_frag, qk_acc);
            }
            store_matrix_sync(&score_smem[0][n_start], qk_acc,
                              SCORE_STRIDE, mem_row_major);
        }
        __syncthreads();

        // ── Top-K heap update (per-warp register-distributed) ──────────
        // Each warp w (r = w) reads its row of score_smem and updates its
        // register heap.  Inactive warps (w >= num_groups) skip — saves
        // SMEM read traffic.
        if (warp_id < num_groups) {
            const int r = warp_id;
            // BLOCK_S = 64 candidates per tile, processed in 2 chunks of 32.
            #pragma unroll
            for (int chunk_off = 0; chunk_off < BLOCK_S; chunk_off += 32) {
                const int c = chunk_off + lane;
                const bool valid = (c < tile_rows);
                float new_key = valid ? (score_smem[r][c] * attn_scale) : -INFINITY;
                int   new_pos = valid ? (tile_s0 + c) : -1;

                // FILL phase: bulk-store directly while heap still has
                // empty slots.  After FILL_CHUNKS chunks per warp the heap
                // is full → switch to min-replace.
                if (tiles_processed * 2 + (chunk_off >> 5) < FILL_CHUNKS) {
                    const int slot = tiles_processed * 2 + (chunk_off >> 5);
                    topk_keys[slot] = new_key;
                    topk_pos [slot] = new_pos;
                    continue;
                }

                // MIN-REPLACE phase.  Loop until no lane qualifies.
                while (true) {
                    // Per-lane min over K_PER_LANE slots.
                    float my_min = topk_keys[0];
                    int   my_min_sub = 0;
                    #pragma unroll
                    for (int i = 1; i < K_PER_LANE; ++i) {
                        if (topk_keys[i] < my_min) {
                            my_min = topk_keys[i];
                            my_min_sub = i;
                        }
                    }
                    // Warp shuffle reduction → global min, owning lane, owning subslot.
                    float global_min      = my_min;
                    int   global_min_lane = lane;
                    int   global_min_sub  = my_min_sub;
                    #pragma unroll
                    for (int off2 = 16; off2 > 0; off2 >>= 1) {
                        float other_min  = __shfl_xor_sync(0xFFFFFFFFu, global_min, off2);
                        int   other_lane = __shfl_xor_sync(0xFFFFFFFFu, global_min_lane, off2);
                        int   other_sub  = __shfl_xor_sync(0xFFFFFFFFu, global_min_sub, off2);
                        bool replace = (other_min < global_min) ||
                                       (other_min == global_min && other_lane < global_min_lane);
                        if (replace) {
                            global_min      = other_min;
                            global_min_lane = other_lane;
                            global_min_sub  = other_sub;
                        }
                    }
                    bool qual = (new_key > global_min);
                    unsigned active = __ballot_sync(0xFFFFFFFFu, qual);
                    if (active == 0) break;
                    int pick_lane = __ffs(active) - 1;
                    float picked_key = __shfl_sync(0xFFFFFFFFu, new_key, pick_lane);
                    int   picked_pos = __shfl_sync(0xFFFFFFFFu, new_pos, pick_lane);
                    if (lane == global_min_lane) {
                        topk_keys[global_min_sub] = picked_key;
                        topk_pos [global_min_sub] = picked_pos;
                    }
                    if (lane == pick_lane) {
                        new_key = -INFINITY;
                    }
                }
            }
        }
        ++tiles_processed;

        // ── Phase B: online softmax (per-warp, same as v5) ─────────────
        for (int i = tid; i < (16 - num_groups) * BLOCK_S; i += THREADS) {
            int r = num_groups + i / BLOCK_S;
            int c = i % BLOCK_S;
            p_smem[r][c] = __float2bfloat16(0.f);
        }
        if (warp_id < num_groups) {
            int r = warp_id;
            float old_m = m_smem[r];
            float old_l = l_smem[r];
            float local_max = -INFINITY;
            for (int c = lane; c < tile_rows; c += 32) {
                float v = score_smem[r][c] * attn_scale;
                local_max = fmaxf(local_max, v);
            }
            #pragma unroll
            for (int off2 = 16; off2 > 0; off2 >>= 1) {
                local_max = fmaxf(local_max,
                                   __shfl_xor_sync(0xffffffffu, local_max, off2));
            }
            float new_m = fmaxf(old_m, local_max);
            float alpha = expf(old_m - new_m);
            float row_sum = 0.0f;
            for (int c = lane; c < BLOCK_S; c += 32) {
                float v = (c < tile_rows)
                              ? __expf(score_smem[r][c] * attn_scale - new_m)
                              : 0.0f;
                p_smem[r][c] = __float2bfloat16(v);
                row_sum += v;
            }
            #pragma unroll
            for (int off2 = 16; off2 > 0; off2 >>= 1) {
                row_sum += __shfl_xor_sync(0xffffffffu, row_sum, off2);
            }
            for (int d = lane; d < D; d += 32) {
                o_smem[r][d] *= alpha;
            }
            float new_l = old_l * alpha + row_sum;
            if (lane == 0) {
                m_smem[r] = new_m;
                l_smem[r] = new_l;
            }
        }
        __syncthreads();

        // ── Phase C: P·V WMMA (same as v5) ─────────────────────────────
        #pragma unroll
        for (int w_n = 0; w_n < 2; ++w_n) {
            int n_tile  = warp_id * 2 + w_n;
            int n_start = n_tile * 16;
            fill_fragment(pv_acc, 0.f);
            #pragma unroll
            for (int k = 0; k < BLOCK_S; k += 16) {
                load_matrix_sync(a_frag, &p_smem[0][k], P_STRIDE);
                load_matrix_sync(b_pv_frag, &v_smem[k][n_start], V_STRIDE);
                mma_sync(pv_acc, a_frag, b_pv_frag, pv_acc);
            }
            store_matrix_sync(&score_smem[0][warp_id * 16], pv_acc,
                              SCORE_STRIDE, mem_row_major);
            #pragma unroll
            for (int i = lane; i < 16 * 16; i += 32) {
                int r = i / 16;
                int c = i % 16;
                o_smem[r][n_start + c] += score_smem[r][warp_id * 16 + c];
            }
        }
        __syncthreads();
    }

    // ── Epilogue: o_hat = o_smem / l_smem ──────────────────────────────
    for (int i = tid; i < num_groups * D; i += THREADS) {
        int r = i / D, c = i % D;
        int h_q = h_kv * num_groups + r;
        float l   = l_smem[r];
        float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        size_t o_off = (size_t)b * H_q * D + (size_t)h_q * D + c;
        O_hat[o_off] = o_smem[r][c] * inv;
    }

    // ── Stage warp-distributed top-K heap to SMEM for Phase 2 ──────────
    // Each warp w writes its register heap to topk_*_sm[w][:].  Layout
    // matches the SortDescending-style packing used by v3 — but we don't
    // sort here, since blend uses (score, slot) symmetrically.
    if (warp_id < num_groups) {
        #pragma unroll
        for (int i = 0; i < K_PER_LANE; ++i) {
            int slot = lane * K_PER_LANE + i;
            topk_keys_sm[warp_id][slot] = topk_keys[i];
            topk_pos_sm [warp_id][slot] = topk_pos [i];
        }
    }
    __syncthreads();

    // ── Phase 2: o_old_unnorm + W from top-K positions ─────────────────
    // Each thread (D=128, threads=128) owns one D element.  Loops over all
    // top_k positions per row, dequantising V[slot] inline.  For W, only
    // thread 0 of each warp tracks the running sum.
    // We process all 4 (= num_groups) rows in each thread to amortise the
    // V-load cost across rows... but actually the V data depends on
    // row's top-K slots which differ — so each row needs its own V load.
    // Process row-by-row:
    if (tid < D) {
        for (int r = 0; r < num_groups; ++r) {
            float acc_o_old = 0.0f;
            float acc_W     = 0.0f;
            const float m_r = m_smem[r];
            const float l_r = l_smem[r];
            const float inv_l_r = (l_r > 0.0f) ? (1.0f / l_r) : 0.0f;
            for (int k = 0; k < top_k; ++k) {
                int   pos = topk_pos_sm [r][k];
                float key = topk_keys_sm[r][k];
                if (pos < 0) continue;
                float weight = __expf(key - m_r) * inv_l_r;
                acc_W += weight;
                int64_t slot = bt_row[pos / block_size] * (int64_t)block_size
                               + (pos % block_size);
                uint8_t pv = v_pack_base[slot * D_QUARTER + (tid >> 2)];
                float vs = __bfloat162float(vs_base[slot]);
                int sub = tid & 3;
                int q = ((pv >> ((3 - sub) * 2)) & 0x3) - 2;
                float v = float(q) * vs;
                acc_o_old += weight * v;
            }
            int h_q = h_kv * num_groups + r;
            size_t o_off = (size_t)b * H_q * D + (size_t)h_q * D + tid;
            O_old_unnorm[o_off] = acc_o_old;
            // W is scalar per row; only thread 0 (or one thread per row) writes.
            if (tid == 0) {
                W_out[(size_t)b * H_q + h_q] = acc_W;
            }
        }
    }

    // ── Translate top-K positions to global slot ids and write out ─────
    // Each warp's heap is in topk_pos_sm[warp_id][0..top_k-1] (row-major).
    // 128 threads × num_groups rows × top_k entries: total work =
    // num_groups * top_k ≤ 4 * 128 = 512 entries; well-distributed.
    for (int i = tid; i < num_groups * top_k; i += THREADS) {
        int r = i / top_k;
        int k = i % top_k;
        int pos = topk_pos_sm[r][k];
        int64_t slot = (pos >= 0)
                         ? (bt_row[pos / block_size] * (int64_t)block_size
                            + (pos % block_size))
                         : (int64_t)0;
        int h_q = h_kv * num_groups + r;
        Topk_slots[((size_t)b * H_q + h_q) * top_k + k] = slot;
    }
}


// ── Host launcher ──────────────────────────────────────────────────────
std::vector<torch::Tensor> int2_fused_attn_topk_v1(
    torch::Tensor q,
    torch::Tensor k_data,
    torch::Tensor k_scales,
    torch::Tensor k_zp,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t top_k,
    double  attn_scale)
{
    TORCH_CHECK(q.is_cuda());
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_data.scalar_type() == torch::kByte);
    TORCH_CHECK(v_data.scalar_type() == torch::kByte);
    TORCH_CHECK(k_scales.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_zp.scalar_type()     == torch::kBFloat16);
    TORCH_CHECK(v_scales.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(q.dim() == 3 && q.size(2) == D);
    TORCH_CHECK(block_size == 16,
                "v1 supports block_size==16 only; got ", block_size);
    TORCH_CHECK(top_k > 0 && top_k <= K_MAX_TOPK && (top_k & 31) == 0,
                "top_k must be a multiple of 32 in [32, 128]; got ", top_k);
    TORCH_CHECK(H_q % H_kv == 0);
    const int num_groups = (int)(H_q / H_kv);
    TORCH_CHECK(num_groups == 1 || num_groups == 2 || num_groups == 4,
                "v1 supports num_groups ∈ {1,2,4}; got ", num_groups);

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

    const int B = qc.size(0);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(qc.device());
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64  ).device(qc.device());

    auto o_hat        = torch::empty({B, (int64_t)H_q, (int64_t)D},     opts_f32);
    auto topk_slots   = torch::empty({B, (int64_t)H_q, (int64_t)top_k}, opts_i64);
    auto o_old_unnorm = torch::empty({B, (int64_t)H_q, (int64_t)D},     opts_f32);
    auto W            = torch::empty({B, (int64_t)H_q},                 opts_f32);

    const size_t smem_bytes =
          16 * Q_STRIDE * sizeof(__nv_bfloat16)
        + STAGES * BLOCK_S * D_QUARTER
        + STAGES * BLOCK_S * D_QUARTER
        + STAGES * BLOCK_S * sizeof(__nv_bfloat16)
        + STAGES * BLOCK_S * sizeof(__nv_bfloat16)
        + STAGES * BLOCK_S * sizeof(__nv_bfloat16)
        + BLOCK_S * K_STRIDE * sizeof(__nv_bfloat16)
        + BLOCK_S * V_STRIDE * sizeof(__nv_bfloat16)
        + 16 * SCORE_STRIDE * sizeof(float)
        + 16 * P_STRIDE * sizeof(__nv_bfloat16)
        + 16 * O_STRIDE * sizeof(float)
        + 16 * sizeof(float)
        + 16 * sizeof(float)
        + 4 * K_MAX_TOPK * sizeof(float)
        + 4 * K_MAX_TOPK * sizeof(int);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_kv);
    dim3 block(THREADS);

    auto launch = [&](auto KPL_const) {
        constexpr int KPL = KPL_const.value;
        cudaFuncSetAttribute(int2_fused_attn_topk_v1_kernel<KPL>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
        int2_fused_attn_topk_v1_kernel<KPL><<<grid, block, smem_bytes, stream.stream()>>>(
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
    };

    const int kpl = (int)(top_k / 32);
    if      (kpl == 1) launch(std::integral_constant<int, 1>{});
    else if (kpl == 2) launch(std::integral_constant<int, 2>{});
    else if (kpl == 3) launch(std::integral_constant<int, 3>{});
    else if (kpl == 4) launch(std::integral_constant<int, 4>{});
    else TORCH_CHECK(false, "unreachable");
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {o_hat, topk_slots, o_old_unnorm, W};
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int2_fused_attn_topk_v1", &int2_fused_attn_topk_v1,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"), py::arg("k_zp"),
          py::arg("v_data"), py::arg("v_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("top_k"), py::arg("attn_scale"),
          "Pattern A v1: WMMA + cp.async fused INT2 attn + topK + pre_corr.");
}
