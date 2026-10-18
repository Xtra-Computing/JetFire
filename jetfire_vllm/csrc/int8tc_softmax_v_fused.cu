// JetFire Stage B fused INT8 tensor-core kernel (H100 / SM 9.0).
//
// Replaces the {softmax_prep + weighted_v_split + 3 zero-fill} pipeline in
// int4_softmax_v.cu with ONE launch that:
//   1. Reads FP32 scores[b, h_q, :] (full max_S, masked by seq_lens).
//   2. Block-wide softmax (max + exp + sum + normalise) over the full S.
//   3. Per query row m, fuses the per-K-row V-scale into the activation:
//        pq[m, k] = round((p[m, k] * v_scale[k]) / pq_scale[m])
//      where pq_scale[m] = max_k(|p[m,k] * v_scale[k]|) / 127.
//   4. Unpacks V_int4 nibbles → V_int8 (signed in [-8, 7]) in SMEM.
//   5. MMA: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` per warp,
//      4 warps × 4 N-tiles each = D=128 columns.
//   6. Dequantises int32 → fp32 by pq_scale[m] and atomicAdds into o_hat.
//
// Inputs (signatures preserved):
//   scores      : FP32 (B, H_q, max_S)
//   seq_lens    : int32 (B,)
//   v_data      : uint8 packed (H_kv, num_slots, D/2)         — INT4 codes
//   v_scales    : BF16  (H_kv, num_slots, num_groups=1)
//   block_table : int64 (B, max_blocks)
// Output:
//   o_hat       : FP32 (B, H_q, D)  — atomicAdd-accumulated across N_SPLITS
//
// Tile shape:
//   M_TILE = 16 (holds num_groups query rows + padded zeros)
//   N_TILE = 8  (D=128 / 8 = 16 N-tiles → 4 warps × 4 N-tiles each)
//   K_TILE = 32 (= 2 BS=16 V-blocks per MMA call)
//
// Grid:  (B, H_kv, N_SPLITS)
// Block: 128 threads = 4 warps.
//
// Numerical notes:
//   * Softmax is computed in FP32 in-kernel over the full S — same recipe
//     as softmax_prep_kernel (mask + max + exp + sum + invZ).
//   * Per-row pq_scale ensures full int8 dynamic range for the activations.
//   * Per-tile re-quantisation: we cannot fold v_scale (per-K-row) and
//     pq_scale (per-M-row) into a single scalar K-sum, so each K-tile is
//     accumulated in int32, dequantised by pq_scale[m], and added to a
//     FP32 register accumulator.
//
// Reference: int2_fused_attn_topk_v1.cu (SMEM staging + cp.async pattern)
//            int4_softmax_v.cu          (split-S, V_int4 dequant)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")


namespace {

constexpr int D_FIXED   = 128;
constexpr int D_HALF    = D_FIXED / 2;   // 64 packed bytes per V row
constexpr int THREADS_B = 128;           // 4 warps
constexpr int N_SPLITS  = 8;  // Match the scalar INT4 Stage B split count.
                              // Low-batch decode is CTA-starved with 4 splits:
                              // B=1,H_kv=8 gives only 32 CTAs.  Eight splits
                              // doubles the independent work and keeps the
                              // per-CTA K loop closer to wvs2/wmma_v3.
constexpr int BS        = 16;            // block_size (paged V)
constexpr int M_TILE    = 16;
constexpr int N_TILE    = 8;
constexpr int K_TILE    = 32;            // 2 BS-blocks per MMA call
constexpr int K_BLOCKS_PER_TILE = K_TILE / BS;     // 2
constexpr int N_TILES_TOTAL    = D_FIXED / N_TILE; // 16
constexpr int N_TILES_PER_WARP = N_TILES_TOTAL / 4; // 4
constexpr int STAGES    = 4;             // 2-way K-tile parity → 2 K-tiles in flight


__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}


// ── mbarrier inline helpers (PTX wrappers) ──────────────────────────────────
// SM_80+ async barriers used by the producer/consumer pc kernel.  Each call
// translates to ONE PTX instruction with no overhead vs raw asm.

__device__ __forceinline__ void mbar_init(uint64_t* addr, int count) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(addr));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                 :: "r"(smem), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* addr) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(addr));
    uint64_t tok;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\n"
                 : "=l"(tok) : "r"(smem));
}

__device__ __forceinline__ void mbar_wait(uint64_t* addr, int parity) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(addr));
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "WAITLOOP: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
                 "@!p bra WAITLOOP;\n"
                 "}\n"
                 :: "r"(smem), "r"(parity));
}


// Wrapper for one m16n8k32 s8 MMA call.
__device__ __forceinline__ void mma_m16n8k32_s8s8s32(
    int32_t        d[4],
    const uint32_t a[4],
    const uint32_t b[2],
    const int32_t  c[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "r"(c[0]),  "r"(c[1]),  "r"(c[2]),  "r"(c[3])
    );
}


// ── Fused softmax(scores) · V_int4 kernel ────────────────────────────────────
//
//   Grid : (B, H_kv, N_SPLITS).
//   Block: 128 threads = 4 warps.
//   Each CTA handles num_groups = H_q / H_kv consecutive query rows for one
//   (b, h_kv) pair, restricted to S-positions [s_start, s_end).
//
//   The full-S softmax is computed inside every CTA (max + exp_sum + invZ)
//   so that each split sees correctly-normalised probabilities; softmax cost
//   is amortised against L2 hits on the scores tensor.
//
// SMEM layout (~9 KB dynamic):
//   v_pack[STAGES][BS][D_HALF]                u8       = 2 * 16 * 64  = 2048
//   v_scale_b[STAGES][BS]                     bf16     = 2 * 16 * 2   = 64
//   v_unpack[K_TILE][D_FIXED+VPAD]            i8       = 32 * 132     = 4224
//   pq_int8[M_TILE][K_TILE+APAD]              i8       = 16  * 36     = 576
//   p_row_fp32[M_TILE][BS]                    f32      = 16  * 16 * 4 = 1024
//   pq_scale[M_TILE]                          f32      = 16  * 4      = 64
//   red32   [32]                              f32      = 128
//
template <int NG_PER_HEAD_MAX = 16, bool WITH_PRE_CORR = false>
__global__ void int8tc_softmax_v_fused_kernel(
    const float*         __restrict__ scores,       // (B, H_q, max_S)
    const int32_t*       __restrict__ seq_lens,     // (B,)
    const uint8_t*       __restrict__ v_data,       // (H_kv, num_slots, D/2)
    const __nv_bfloat16* __restrict__ v_scales,     // (H_kv, num_slots) — NG=1
    const int64_t*       __restrict__ block_table,  // (B, max_blocks)
    float*               __restrict__ o_hat,        // (B, H_q, D) [out, pre-zero]
    // ── pre_corr extra args (used only when WITH_PRE_CORR==true) ─────────────
    const int64_t*       __restrict__ topk_idx,     // (B, H_q, K) | nullptr
    float*               __restrict__ o_old_unnorm, // (B, H_q, D) | nullptr
    float*               __restrict__ W_out,        // (B, H_q)    | nullptr
    int top_k,
    int max_S,
    int max_blocks,
    int H_q,
    int H_kv,
    int v_data_stride_h,
    int v_scales_stride_h)
{
    const int b       = blockIdx.x;
    const int h_kv    = blockIdx.y;
    const int s_split = blockIdx.z;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int num_groups = H_q / H_kv;        // 4 for LLaMA-3.1-8B
    const int S          = seq_lens[b];

    const int chunk_per_split = (max_S + N_SPLITS - 1) / N_SPLITS;
    const int s_start = s_split * chunk_per_split;
    const int s_end   = min(s_start + chunk_per_split, S);

    // Pad the per-split S range up to a multiple of BS for clean K-tile loops.
    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    // ── Dynamic SMEM ─────────────────────────────────────────────────────────
    constexpr int VPAD = 4;          // pad cols to break 32-banked conflicts on s8 SMEM reads
    constexpr int V_STRIDE = D_FIXED + VPAD;
    constexpr int APAD = 4;          // pad K cols of pq_int8
    constexpr int A_STRIDE = K_TILE + APAD;

    extern __shared__ __align__(16) uint8_t smem_raw[];
    size_t off = 0;
    auto v_pack    = reinterpret_cast<uint8_t(*)[BS][D_HALF]>(&smem_raw[off]);
    off += STAGES * BS * D_HALF;
    auto v_scale_b = reinterpret_cast<__nv_bfloat16(*)[BS]>(&smem_raw[off]);
    off += STAGES * BS * sizeof(__nv_bfloat16);
    // Align v_unpack to 16 B.
    off = (off + 15) & ~size_t(15);
    auto v_unpack  = reinterpret_cast<int8_t(*)[V_STRIDE]>(&smem_raw[off]);
    off += K_TILE * V_STRIDE * sizeof(int8_t);
    off = (off + 15) & ~size_t(15);
    auto pq_int8   = reinterpret_cast<int8_t(*)[A_STRIDE]>(&smem_raw[off]);
    off += M_TILE * A_STRIDE * sizeof(int8_t);
    off = (off + 15) & ~size_t(15);
    float* pq_scale = reinterpret_cast<float*>(&smem_raw[off]);
    off += M_TILE * sizeof(float);
    off = (off + 15) & ~size_t(15);
    float* red32    = reinterpret_cast<float*>(&smem_raw[off]);
    // off += 32 * sizeof(float);  // last region
    (void)red32;

    // ── Pointer bases ────────────────────────────────────────────────────────
    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    // Per-warp FP32 register accumulators: 4 N-tiles × 4 fp32 (per-lane fragment).
    float c_acc[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int n = 0; n < N_TILES_PER_WARP; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) c_acc[n][i] = 0.0f;
    }

    // ── Pass 1: full-S softmax (per query row m) ─────────────────────────────
    // Each warp handles one query row m (when warp_id < num_groups).  Total
    // work per row = max_S fp32 reads + exp + sum reduction.  We use the full
    // 32 lanes of the owning warp for the row, not the whole block — this
    // saves the cross-warp shuffle and keeps the reduction lightweight.

    // We need: max[m], inv_Z[m] for each m in [0, num_groups).
    // Store in p_row[m][0] reused as m_val, p_row[m][1] as inv_Z; (BS=16 cols
    // unused for this purpose — we'll repurpose p_row for K-tile activations.)
    __shared__ float m_smem[NG_PER_HEAD_MAX];
    __shared__ float invZ_smem[NG_PER_HEAD_MAX];

    if (warp_id < num_groups) {
        const int m = warp_id;
        const int h_q = h_kv * num_groups + m;
        const float* row =
            scores + ((size_t)b * H_q + (size_t)h_q) * max_S;

        // 2-pass softmax (max then sum).  K-tile build reads scores again
        // (L2-hot), trading a bit of compute for SMEM occupancy headroom.
        float local_max = -INFINITY;
        for (int i = lane; i < S; i += 32) {
            local_max = fmaxf(local_max, row[i]);
        }
        float mx = warp_max(local_max);
        if (!isfinite(mx)) mx = 0.0f;

        float local_sum = 0.0f;
        for (int i = lane; i < S; i += 32) {
            local_sum += __expf(row[i] - mx);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, o);
        }
        float invZ = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;

        if (lane == 0) {
            m_smem[m] = mx;
            invZ_smem[m] = invZ;
        }
    }
    // Padded rows: 0.
    if (warp_id == 0 && lane < NG_PER_HEAD_MAX) {
        if (lane >= num_groups) {
            m_smem[lane]    = 0.0f;
            invZ_smem[lane] = 0.0f;
        }
    }
    __syncthreads();

    // Early-exit: split has no work.
    if (s_start >= s_end) return;

    // ── Async load helper for one V_int4 block into SMEM stage ───────────────
    auto load_block_async = [&](int blk, int stage) {
        if (blk < blk_end) {
            const int64_t base_slot = bt_row[blk] * (int64_t)BS;
            // V data: BS=16 × D_HALF=64 = 1024 bytes total.
            // 64 threads × 16B = 1024 B → use 16-byte cp.async on lanes < 64.
            if (tid < 64) {
                const int pos_in_blk = tid >> 2;        // 0..15
                const int byte_off   = (tid & 3) << 4;  // 0,16,32,48
                const void* g = v_pack_base
                              + (size_t)(base_slot + pos_in_blk) * D_HALF
                              + byte_off;
                void* s = (void*)&v_pack[stage][pos_in_blk][byte_off];
                unsigned si = __cvta_generic_to_shared(s);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(si), "l"(g));
            }
            if (tid < BS) {
                v_scale_b[stage][tid] = __ldg(v_scale_base + (size_t)(base_slot + tid));
            }
        } else {
            // Padding block: encode q=0 (nibble 0x8) for all bytes.
            if (tid < 64) {
                const int pos_in_blk = tid >> 2;
                const int byte_off   = (tid & 3) << 4;
                uint8_t* dst = (uint8_t*)&v_pack[stage][pos_in_blk][byte_off];
                #pragma unroll
                for (int i = 0; i < 16; ++i) dst[i] = 0x88;
            }
            if (tid < BS) v_scale_b[stage][tid] = __float2bfloat16(0.0f);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    // ── Iterate over K-tiles (2 BS-blocks each) ───────────────────────────
    // Pipelined with STAGES=4: stages (0,1) and (2,3) alternate between
    // consecutive K-tiles, so we can prefetch K+1 while computing K.

    const int n_blocks = blk_end - blk_start;
    const int n_ktiles = (n_blocks + K_BLOCKS_PER_TILE - 1) / K_BLOCKS_PER_TILE;

    // Prologue: issue loads for K-tile 0.
    if (n_ktiles > 0) {
        load_block_async(blk_start    , 0);
        load_block_async(blk_start + 1, 1);
    }

    for (int kt = 0; kt < n_ktiles; ++kt) {
        const int blk_a = blk_start + kt * K_BLOCKS_PER_TILE;
        const int blk_b = blk_a + 1;
        const bool has_b = (blk_b < blk_end);

        // Current K-tile stages.
        const int stage_a = (kt & 1) * 2 + 0;
        const int stage_b = (kt & 1) * 2 + 1;

        // Prefetch next K-tile.
        if (kt + 1 < n_ktiles) {
            const int next_a = blk_start + (kt + 1) * K_BLOCKS_PER_TILE;
            const int next_b = next_a + 1;
            const int nstage_a = ((kt + 1) & 1) * 2 + 0;
            const int nstage_b = ((kt + 1) & 1) * 2 + 1;
            load_block_async(next_a, nstage_a);
            load_block_async(next_b, nstage_b);
        }

        // Wait until cur K-tile's 2 cp.async groups are done (older ones).
        // If we prefetched, there are up to 4 pending groups; wait_group 2
        // keeps the prefetched pair pending.
        if (kt + 1 < n_ktiles) {
            asm volatile("cp.async.wait_group 2;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        // ── Unpack V_int4 → v_unpack[0..BS-1][:] for stage_a ───────────────
        for (int idx = tid; idx < BS * D_HALF; idx += THREADS_B) {
            int r  = idx / D_HALF;
            int bx = idx % D_HALF;
            uint8_t pk = v_pack[stage_a][r][bx];
            int q_even = int((pk >> 4) & 0xF) - 8;
            int q_odd  = int( pk       & 0xF) - 8;
            v_unpack[r][2 * bx + 0] = (int8_t)q_even;
            v_unpack[r][2 * bx + 1] = (int8_t)q_odd;
        }

        // ── Unpack/zero second half v_unpack[BS..2BS-1][:] ─────────────────
        if (has_b) {
            for (int idx = tid; idx < BS * D_HALF; idx += THREADS_B) {
                int r  = idx / D_HALF;
                int bx = idx % D_HALF;
                uint8_t pk = v_pack[stage_b][r][bx];
                int q_even = int((pk >> 4) & 0xF) - 8;
                int q_odd  = int( pk       & 0xF) - 8;
                v_unpack[BS + r][2 * bx + 0] = (int8_t)q_even;
                v_unpack[BS + r][2 * bx + 1] = (int8_t)q_odd;
            }
        } else {
            for (int idx = tid; idx < BS * V_STRIDE; idx += THREADS_B) {
                int r  = idx / V_STRIDE;
                int c  = idx % V_STRIDE;
                v_unpack[BS + r][c] = 0;
            }
        }

        // ── Build pq_int8[m, k] for k ∈ [0, K_TILE) ─────────────────────────
        //   For each query row m and each K-row k:
        //     pos = (blk_a + (k / BS)) * BS + (k % BS)  if k < n_blocks*BS
        //     p_raw = (pos < s_start || pos >= s_end || pos >= S) ? 0
        //           : exp(scores[m, pos] - mx[m]) * invZ[m]
        //     pv    = p_raw * v_scale[stage][k % BS]      (fp32)
        //   Then per-m: pq_scale[m] = max_k(|pv[m,k]|) / 127
        //               pq_int8[m,k] = round(pv[m,k] / pq_scale[m])
        //
        //   Layout: each warp owns one m row (warp_id maps to m).
        //   Lanes 0..K_TILE-1 do one K_TILE column each (we have 32 lanes,
        //   K_TILE = 32 — perfect 1:1 mapping).

        if (warp_id < num_groups) {
            const int m = warp_id;
            const int h_q = h_kv * num_groups + m;
            const float* row =
                scores + ((size_t)b * H_q + (size_t)h_q) * max_S;
            const float mx   = m_smem[m];
            const float invZ = invZ_smem[m];

            // Compute pv for this lane's k.
            const int k = lane;  // 0..31
            const int local_blk = k / BS;       // 0 or 1
            const int local_off = k % BS;       // 0..15
            const int abs_blk   = blk_a + local_blk;
            float pv = 0.0f;
            if (abs_blk < blk_end) {
                int pos = abs_blk * BS + local_off;
                if (pos >= s_start && pos < s_end && pos < S) {
                    int stage_x = (local_blk == 0) ? stage_a : stage_b;
                    float p = __expf(row[pos] - mx) * invZ;
                    float vs = __bfloat162float(v_scale_b[stage_x][local_off]);
                    pv = p * vs;
                }
            }

            // Per-row max abs.
            float a = fabsf(pv);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            }
            float row_scale = a / 127.0f;
            float inv_scale = (row_scale > 0.0f) ? (1.0f / row_scale) : 0.0f;
            int q = (int)__float2int_rn(pv * inv_scale);
            q = max(-127, min(127, q));
            pq_int8[m][k] = (int8_t)q;
            if (lane == 0) pq_scale[m] = row_scale;
        } else {
            // Padded rows: zero.
            if (lane < K_TILE) {
                pq_int8[warp_id][lane] = 0;
            }
            if (lane == 0) pq_scale[warp_id] = 0.0f;
        }
        __syncthreads();

        // ── MMA: per-warp 4 N-tiles ────────────────────────────────────────
        // A operand: pq_int8[M_TILE][K_TILE], row-major.
        //   Lane mapping for m16n8k32 .row .col s8:
        //   a0 = row=(lane/4)+0,  k=(lane%4)*4..+3
        //   a1 = row=(lane/4)+8,  k=(lane%4)*4..+3
        //   a2 = row=(lane/4)+0,  k=(lane%4)*4+16..+19
        //   a3 = row=(lane/4)+8,  k=(lane%4)*4+16..+19
        // B operand: v_unpack[K=32][N_per_warp_tile * 8 cols], col-major view.
        //   col-major means logical N-tile of 8 cols × 32 rows of K, treated as
        //   stored column-major.  Layout for b0/b1:
        //   b0 = col=(lane/4),    k=(lane%4)*4..+3
        //   b1 = col=(lane/4),    k=(lane%4)*4+16..+19

        const int n_base_warp = warp_id * N_TILES_PER_WARP;  // each warp owns 4 N-tiles

        // Load A from pq_int8 (same across N-tiles, shared between warps —
        // each warp reads a different M-row range but all M rows are in SMEM).
        uint32_t a_reg[4];
        {
            const int m_low  = (lane >> 2);       // row 0..7
            const int m_high = m_low + 8;         // row 8..15
            const int k_off0 = (lane & 3) * 4;    // 0,4,8,12
            const int k_off1 = k_off0 + 16;
            // 4 bytes packed → uint32.
            a_reg[0] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_low ][k_off0]);
            a_reg[1] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_high][k_off0]);
            a_reg[2] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_low ][k_off1]);
            a_reg[3] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_high][k_off1]);
        }

        // For each N-tile this warp owns: load B and do the MMA, then dequant
        // by pq_scale and add into c_acc.
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; ++nt) {
            const int n_tile = n_base_warp + nt;
            const int d_col_base = n_tile * N_TILE;     // first D column

            // Build B operand.  B is col-major (K rows × N cols).  Per PTX,
            // each lane reads two 4-byte K-strips at one N-column.
            uint32_t b_reg[2];
            {
                const int n_col = lane >> 2;                  // 0..7
                const int k_off0 = (lane & 3) * 4;            // 0,4,8,12
                const int k_off1 = k_off0 + 16;               // 16,20,24,28
                // pack 4 int8 from v_unpack[k_off0+0..3][d_col_base + n_col]
                int8_t v0 = v_unpack[k_off0    ][d_col_base + n_col];
                int8_t v1 = v_unpack[k_off0 + 1][d_col_base + n_col];
                int8_t v2 = v_unpack[k_off0 + 2][d_col_base + n_col];
                int8_t v3 = v_unpack[k_off0 + 3][d_col_base + n_col];
                int8_t w0 = v_unpack[k_off1    ][d_col_base + n_col];
                int8_t w1 = v_unpack[k_off1 + 1][d_col_base + n_col];
                int8_t w2 = v_unpack[k_off1 + 2][d_col_base + n_col];
                int8_t w3 = v_unpack[k_off1 + 3][d_col_base + n_col];
                b_reg[0] = (uint32_t)( (uint8_t)v0
                                    | ((uint8_t)v1 <<  8)
                                    | ((uint8_t)v2 << 16)
                                    | ((uint8_t)v3 << 24));
                b_reg[1] = (uint32_t)( (uint8_t)w0
                                    | ((uint8_t)w1 <<  8)
                                    | ((uint8_t)w2 << 16)
                                    | ((uint8_t)w3 << 24));
            }

            int32_t d_reg[4];
            int32_t c_zero[4] = {0, 0, 0, 0};
            mma_m16n8k32_s8s8s32(d_reg, a_reg, b_reg, c_zero);

            // Dequant d_reg by pq_scale[m_row].  Per lane:
            //   d0 = row=m_low,  col=(lane%4)*2 + 0
            //   d1 = row=m_low,  col=(lane%4)*2 + 1
            //   d2 = row=m_high, col=(lane%4)*2 + 0
            //   d3 = row=m_high, col=(lane%4)*2 + 1
            const int m_low  = lane >> 2;
            const int m_high = m_low + 8;
            float s_low  = pq_scale[m_low];
            float s_high = pq_scale[m_high];
            c_acc[nt][0] += (float)d_reg[0] * s_low;
            c_acc[nt][1] += (float)d_reg[1] * s_low;
            c_acc[nt][2] += (float)d_reg[2] * s_high;
            c_acc[nt][3] += (float)d_reg[3] * s_high;
        }
        __syncthreads();
    }

    // ── Writeback: atomicAdd c_acc into o_hat ───────────────────────────────
    // Per lane: 4 fp32 values per N-tile × 4 N-tiles = 16 atomicAdds.
    // d_reg → o_hat mapping:
    //   m_low/m_high < num_groups means real query row → write
    //   else (padding row) → skip
    {
        const int n_base_warp = warp_id * N_TILES_PER_WARP;
        const int m_low  = lane >> 2;
        const int m_high = m_low + 8;
        const int col_lo = (lane & 3) * 2;
        // For each N-tile owned by this warp:
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; ++nt) {
            const int n_tile = n_base_warp + nt;
            const int d_col_base = n_tile * N_TILE;

            if (m_low < num_groups) {
                const int h_q = h_kv * num_groups + m_low;
                const size_t base = ((size_t)b * H_q + (size_t)h_q) * D_FIXED;
                atomicAdd(o_hat + base + d_col_base + col_lo + 0, c_acc[nt][0]);
                atomicAdd(o_hat + base + d_col_base + col_lo + 1, c_acc[nt][1]);
            }
            if (m_high < num_groups) {
                const int h_q = h_kv * num_groups + m_high;
                const size_t base = ((size_t)b * H_q + (size_t)h_q) * D_FIXED;
                atomicAdd(o_hat + base + d_col_base + col_lo + 0, c_acc[nt][2]);
                atomicAdd(o_hat + base + d_col_base + col_lo + 1, c_acc[nt][3]);
            }
        }
    }

    // ── Phase 2: per-split top-K correction (o_old_unnorm + W) ──────────────
    // Each warp owns one query row m (warp_id < num_groups).  For each top-K
    // index k, if topk_idx[b, h_q, k] falls in this split's [s_start, s_end)
    // range, accumulate:
    //   a_hat = exp(scores[b, h_q, pos] - mx[m]) * invZ[m]
    //   slot  = block_table[b, pos / BS] * BS + (pos % BS)
    //   vs    = v_scales[h_kv, slot]
    //   acc_W      += a_hat                                (lane 0 only)
    //   acc_o_old[d] += a_hat * vs * dequant(V_int4[h_kv, slot, d])
    //                                                       (each lane: 4 d's)
    //
    // Lane → d mapping: lane l covers d ∈ {4*l + 0, 4*l + 1, 4*l + 2, 4*l + 3}.
    // We pack two lookups per byte (V is int4, packed 2 nibbles per byte).
    //
    // Split-partial: each CTA writes only its own slice of topK contributions
    // via atomicAdd to global memory (host pre-zeros the buffers).
    if (WITH_PRE_CORR && warp_id < num_groups) {
        const int m   = warp_id;
        const int h_q = h_kv * num_groups + m;
        const float* row =
            scores + ((size_t)b * H_q + (size_t)h_q) * max_S;
        const float mx   = m_smem[m];
        const float invZ = invZ_smem[m];
        const int64_t* topk_row =
            topk_idx + ((size_t)b * H_q + (size_t)h_q) * top_k;

        // 4 contiguous d positions per lane.
        const int d0 = lane * 4;
        const int d1 = d0 + 1;
        const int d2 = d0 + 2;
        const int d3 = d0 + 3;
        // Packed-byte indices and nibble shifts (V is 2 int4 per byte, even=hi).
        const int b0 = d0 >> 1, sh0 = (1 - (d0 & 1)) * 4;
        const int b1 = d1 >> 1, sh1 = (1 - (d1 & 1)) * 4;
        const int b2 = d2 >> 1, sh2 = (1 - (d2 & 1)) * 4;
        const int b3 = d3 >> 1, sh3 = (1 - (d3 & 1)) * 4;

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        float acc_W = 0.0f;

        for (int k = 0; k < top_k; ++k) {
            int64_t pos = topk_row[k];
            if (pos < 0) continue;
            int p = (int)pos;
            // Split filter + seq bound.
            if (p < s_start || p >= s_end || p >= S) continue;

            int blk_idx = p / BS;
            int in_blk  = p - blk_idx * BS;
            int64_t slot = bt_row[blk_idx] * (int64_t)BS + (int64_t)in_blk;

            // a_hat = softmax probability at this position.
            float a_hat = __expf(row[p] - mx) * invZ;
            float vs    = __bfloat162float(__ldg(v_scale_base + slot));
            float ws    = a_hat * vs;

            // Load 2 bytes from the packed V row (covers d0..d3).
            const uint8_t* v_slot_base = v_pack_base + (size_t)slot * D_HALF;
            uint8_t pk_a = __ldg(v_slot_base + b0);            // bytes for d0,d1
            uint8_t pk_b = (b2 == b0) ? pk_a : __ldg(v_slot_base + b2);
            (void)b1; (void)b3;  // b1==b0, b3==b2 by construction
            int q0 = int((pk_a >> sh0) & 0xF) - 8;
            int q1 = int((pk_a >> sh1) & 0xF) - 8;
            int q2 = int((pk_b >> sh2) & 0xF) - 8;
            int q3 = int((pk_b >> sh3) & 0xF) - 8;

            acc0 = __fmaf_rn(ws, (float)q0, acc0);
            acc1 = __fmaf_rn(ws, (float)q1, acc1);
            acc2 = __fmaf_rn(ws, (float)q2, acc2);
            acc3 = __fmaf_rn(ws, (float)q3, acc3);
            // a_hat is identical across all lanes (uniform row[p], mx, invZ),
            // so only lane 0 needs to accumulate it (avoids 32x over-count).
            if (lane == 0) acc_W += a_hat;
        }

        const size_t base = ((size_t)b * H_q + (size_t)h_q) * D_FIXED;
        if (acc0 != 0.0f) atomicAdd(o_old_unnorm + base + d0, acc0);
        if (acc1 != 0.0f) atomicAdd(o_old_unnorm + base + d1, acc1);
        if (acc2 != 0.0f) atomicAdd(o_old_unnorm + base + d2, acc2);
        if (acc3 != 0.0f) atomicAdd(o_old_unnorm + base + d3, acc3);
        if (lane == 0 && acc_W != 0.0f) {
            atomicAdd(W_out + ((size_t)b * H_q + (size_t)h_q), acc_W);
        }
    }
}


// ── Host launcher ────────────────────────────────────────────────────────────
torch::Tensor int8tc_softmax_v_fused(
    torch::Tensor scores,
    torch::Tensor seq_lens,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size)
{
    CHECK_CUDA(scores); CHECK_CUDA(seq_lens);
    CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table);
    TORCH_CHECK(scores.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(v_data.scalar_type()      == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF,
                "v_data must be (H_kv, num_slots, D/2=", D_HALF, ")");
    TORCH_CHECK(block_size == 16, "only block_size=16 supported");
    TORCH_CHECK(H_q % H_kv == 0);
    const int num_groups = (int)(H_q / H_kv);
    TORCH_CHECK(num_groups <= THREADS_B / 32, "legacy INT8TC Stage-B supports GQA <= 4; use the reg variant for GQA up to 16");

    auto sc = scores.contiguous();
    auto sl = seq_lens.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();
    auto bt = block_table.contiguous();

    const int B          = (int)sc.size(0);
    const int max_blocks = (int)bt.size(1);
    const int ng_vs      = (int)vs.size(2);
    TORCH_CHECK(ng_vs == 1, "num_groups in v_scales must be 1");

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());
    auto o_hat = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);

    constexpr int VPAD = 4;
    constexpr int V_STRIDE = D_FIXED + VPAD;
    constexpr int APAD = 4;
    constexpr int A_STRIDE = K_TILE + APAD;

    size_t smem = 0;
    smem += STAGES * BS * D_HALF;                                // v_pack
    smem  = (smem + 15) & ~size_t(15);
    smem += STAGES * BS * sizeof(__nv_bfloat16);                 // v_scale_b
    smem  = (smem + 15) & ~size_t(15);
    smem += K_TILE * V_STRIDE * sizeof(int8_t);                  // v_unpack
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * A_STRIDE * sizeof(int8_t);                  // pq_int8
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * sizeof(float);                              // pq_scale
    smem  = (smem + 15) & ~size_t(15);
    smem += 32 * sizeof(float);                                  // red32
    smem += 64;                                                  // slack

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)N_SPLITS);
    dim3 block(THREADS_B);

    cudaFuncSetAttribute(int8tc_softmax_v_fused_kernel<16, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    int8tc_softmax_v_fused_kernel<16, false><<<grid, block, smem, stream.stream()>>>(
        sc.data_ptr<float>(),
        sl.data_ptr<int32_t>(),
        vd.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        o_hat.data_ptr<float>(),
        /*topk_idx=*/nullptr, /*o_old_unnorm=*/nullptr, /*W_out=*/nullptr,
        /*top_k=*/0,
        (int)max_S, max_blocks, (int)H_q, (int)H_kv,
        (int)vd.stride(0), (int)vs.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o_hat;
}


// In-place variant — caller-allocated o_hat, no allocation in timed window.
void int8tc_softmax_v_fused_inplace(
    torch::Tensor scores,
    torch::Tensor seq_lens,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor o_hat,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size)
{
    CHECK_CUDA(scores); CHECK_CUDA(seq_lens);
    CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(o_hat);
    TORCH_CHECK(scores.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(v_data.scalar_type()      == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(o_hat.scalar_type()       == torch::kFloat32);
    TORCH_CHECK(block_size == 16);
    TORCH_CHECK(H_q % H_kv == 0);
    const int num_groups = (int)(H_q / H_kv);
    TORCH_CHECK(num_groups <= THREADS_B / 32, "legacy INT8TC Stage-B supports GQA <= 4; use the reg variant for GQA up to 16");

    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();

    const int B          = (int)scores.size(0);
    const int max_blocks = (int)block_table.size(1);

    constexpr int VPAD = 4;
    constexpr int V_STRIDE = D_FIXED + VPAD;
    constexpr int APAD = 4;
    constexpr int A_STRIDE = K_TILE + APAD;

    size_t smem = 0;
    smem += STAGES * BS * D_HALF;
    smem  = (smem + 15) & ~size_t(15);
    smem += STAGES * BS * sizeof(__nv_bfloat16);
    smem  = (smem + 15) & ~size_t(15);
    smem += K_TILE * V_STRIDE * sizeof(int8_t);
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * A_STRIDE * sizeof(int8_t);
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * sizeof(float);
    smem  = (smem + 15) & ~size_t(15);
    smem += 32 * sizeof(float);
    smem += 64;

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)N_SPLITS);
    dim3 block(THREADS_B);

    cudaFuncSetAttribute(int8tc_softmax_v_fused_kernel<16, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    int8tc_softmax_v_fused_kernel<16, false><<<grid, block, smem, stream.stream()>>>(
        scores.data_ptr<float>(),
        seq_lens.data_ptr<int32_t>(),
        v_data.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        block_table.data_ptr<int64_t>(),
        o_hat.data_ptr<float>(),
        /*topk_idx=*/nullptr, /*o_old_unnorm=*/nullptr, /*W_out=*/nullptr,
        /*top_k=*/0,
        (int)max_S, max_blocks, (int)H_q, (int)H_kv,
        (int)v_data.stride(0), (int)vs.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ── pre_corr variant: returns [o_hat, o_old_unnorm, W] ──────────────────────
std::vector<torch::Tensor> int8tc_softmax_v_with_pre_corr(
    torch::Tensor scores,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor topk_idx,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t top_k)
{
    CHECK_CUDA(scores); CHECK_CUDA(seq_lens);
    CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(topk_idx);
    TORCH_CHECK(scores.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(v_data.scalar_type()      == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(topk_idx.scalar_type()    == torch::kInt64);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(topk_idx.dim() == 3);
    TORCH_CHECK(topk_idx.size(2) == top_k);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF,
                "v_data must be (H_kv, num_slots, D/2=", D_HALF, ")");
    TORCH_CHECK(block_size == 16, "only block_size=16 supported");
    TORCH_CHECK(H_q % H_kv == 0);
    const int num_groups = (int)(H_q / H_kv);
    TORCH_CHECK(num_groups <= THREADS_B / 32, "legacy INT8TC Stage-B supports GQA <= 4; use the reg variant for GQA up to 16");

    auto sc = scores.contiguous();
    auto sl = seq_lens.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();
    auto bt = block_table.contiguous();
    auto ti = topk_idx.contiguous();

    const int B          = (int)sc.size(0);
    const int max_blocks = (int)bt.size(1);
    const int ng_vs      = (int)vs.size(2);
    TORCH_CHECK(ng_vs == 1, "num_groups in v_scales must be 1");

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());
    auto o_hat        = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto o_old_unnorm = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out        = torch::zeros({B, (int64_t)H_q}, opts_f32);

    constexpr int VPAD = 4;
    constexpr int V_STRIDE = D_FIXED + VPAD;
    constexpr int APAD = 4;
    constexpr int A_STRIDE = K_TILE + APAD;

    size_t smem = 0;
    smem += STAGES * BS * D_HALF;
    smem  = (smem + 15) & ~size_t(15);
    smem += STAGES * BS * sizeof(__nv_bfloat16);
    smem  = (smem + 15) & ~size_t(15);
    smem += K_TILE * V_STRIDE * sizeof(int8_t);
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * A_STRIDE * sizeof(int8_t);
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * sizeof(float);
    smem  = (smem + 15) & ~size_t(15);
    smem += 32 * sizeof(float);
    smem += 64;

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)N_SPLITS);
    dim3 block(THREADS_B);

    cudaFuncSetAttribute(int8tc_softmax_v_fused_kernel<16, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    int8tc_softmax_v_fused_kernel<16, true><<<grid, block, smem, stream.stream()>>>(
        sc.data_ptr<float>(),
        sl.data_ptr<int32_t>(),
        vd.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        o_hat.data_ptr<float>(),
        ti.data_ptr<int64_t>(),
        o_old_unnorm.data_ptr<float>(),
        W_out.data_ptr<float>(),
        (int)top_k,
        (int)max_S, max_blocks, (int)H_q, (int)H_kv,
        (int)vd.stride(0), (int)vs.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}


// In-place variant of the pre_corr entry — caller pre-allocates and zeroes
// all three outputs (matches int4 wvs2 semantics).
void int8tc_softmax_v_with_pre_corr_inplace(
    torch::Tensor scores,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor topk_idx,
    torch::Tensor o_hat,
    torch::Tensor o_old_unnorm,
    torch::Tensor W_out,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t top_k)
{
    CHECK_CUDA(scores); CHECK_CUDA(seq_lens);
    CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(topk_idx);
    CHECK_CUDA(o_hat); CHECK_CUDA(o_old_unnorm); CHECK_CUDA(W_out);
    TORCH_CHECK(scores.scalar_type()       == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type()     == torch::kInt32);
    TORCH_CHECK(v_data.scalar_type()       == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()     == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type()  == torch::kInt64);
    TORCH_CHECK(topk_idx.scalar_type()     == torch::kInt64);
    TORCH_CHECK(o_hat.scalar_type()        == torch::kFloat32);
    TORCH_CHECK(o_old_unnorm.scalar_type() == torch::kFloat32);
    TORCH_CHECK(W_out.scalar_type()        == torch::kFloat32);
    TORCH_CHECK(block_size == 16);
    TORCH_CHECK(H_q % H_kv == 0);
    const int num_groups = (int)(H_q / H_kv);
    TORCH_CHECK(num_groups <= THREADS_B / 32, "legacy INT8TC Stage-B supports GQA <= 4; use the reg variant for GQA up to 16");

    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();

    const int B          = (int)scores.size(0);
    const int max_blocks = (int)block_table.size(1);

    constexpr int VPAD = 4;
    constexpr int V_STRIDE = D_FIXED + VPAD;
    constexpr int APAD = 4;
    constexpr int A_STRIDE = K_TILE + APAD;

    size_t smem = 0;
    smem += STAGES * BS * D_HALF;
    smem  = (smem + 15) & ~size_t(15);
    smem += STAGES * BS * sizeof(__nv_bfloat16);
    smem  = (smem + 15) & ~size_t(15);
    smem += K_TILE * V_STRIDE * sizeof(int8_t);
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * A_STRIDE * sizeof(int8_t);
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * sizeof(float);
    smem  = (smem + 15) & ~size_t(15);
    smem += 32 * sizeof(float);
    smem += 64;

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)N_SPLITS);
    dim3 block(THREADS_B);

    cudaFuncSetAttribute(int8tc_softmax_v_fused_kernel<16, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    int8tc_softmax_v_fused_kernel<16, true><<<grid, block, smem, stream.stream()>>>(
        scores.data_ptr<float>(),
        seq_lens.data_ptr<int32_t>(),
        v_data.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        block_table.data_ptr<int64_t>(),
        o_hat.data_ptr<float>(),
        topk_idx.data_ptr<int64_t>(),
        o_old_unnorm.data_ptr<float>(),
        W_out.data_ptr<float>(),
        (int)top_k,
        (int)max_S, max_blocks, (int)H_q, (int)H_kv,
        (int)v_data.stride(0), (int)vs.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ── Register-resident V expansion variant ────────────────────────────────────
//
// Same softmax + pq_int8 build as the fused kernel, but skips the v_unpack
// SMEM round-trip: each lane builds its mma B-operand by loading V_int4
// packed bytes from v_pack directly into registers and extending to INT8
// in-register.
//
// Two critical micro-arch fixes vs the first cut of this kernel:
//   1. v_pack now has VPACK_PAD=4 padding bytes per row so the row stride
//      (D_HALF+VPAD = 68 = 17 banks/row) makes consecutive k_rows land on
//      different SMEM banks.  Without this, lane (l_mod4, l_div4) reading
//      4 k_rows at a fixed col_byte hit only banks {0, 16, 0, 16} → 2-way
//      bank conflict that doubles SMEM read latency.
//   2. The end-of-K-tile __syncthreads is redundant — the next iteration's
//      cp.async.wait_group + __syncthreads at the top of the loop already
//      orders the previous mma's reads before any new write.  Removing it
//      saves one CTA-wide barrier per K-tile (~33% of the barrier wait
//      that ncu attributes to this kernel family).
template <int NG_PER_HEAD_MAX = 16, bool WITH_PRE_CORR = false,
          bool USE_PRECOMPUTED_STATS = false>
__global__ void int8tc_softmax_v_fused_reg_kernel(
    const float*         __restrict__ scores,
    const float*         __restrict__ softmax_m,
    const float*         __restrict__ softmax_inv_z,
    const int32_t*       __restrict__ seq_lens,
    const uint8_t*       __restrict__ v_data,
    const __nv_bfloat16* __restrict__ v_scales,
    const int64_t*       __restrict__ block_table,
    float*               __restrict__ o_hat,
    const int64_t*       __restrict__ topk_idx,
    float*               __restrict__ o_old_unnorm,
    float*               __restrict__ W_out,
    int top_k,
    int max_S,
    int max_blocks,
    int H_q,
    int H_kv,
    int v_data_stride_h,
    int v_scales_stride_h)
{
    const int b       = blockIdx.x;
    const int h_kv    = blockIdx.y;
    const int s_split = blockIdx.z;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int num_groups = H_q / H_kv;
    const int S          = seq_lens[b];

    const int chunk_per_split = (max_S + N_SPLITS - 1) / N_SPLITS;
    const int s_start = s_split * chunk_per_split;
    const int s_end   = min(s_start + chunk_per_split, S);
    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    // ── Dynamic SMEM (no v_unpack) ──────────────────────────────────────────
    // VPACK_PAD = 16 → row stride = 80 bytes = 20 banks.  Picked over 4
    // (which would also break the conflict) because 16-byte cp.async requires
    // 16-byte alignment, and 68 mod 16 ≠ 0 makes row[1]+ misaligned.  With
    // stride 80, consecutive k_rows land on banks {0, 20, 8, 28} —
    // conflict-free.
    constexpr int VPACK_PAD    = 16;
    constexpr int PACK_STRIDE  = D_HALF + VPACK_PAD;  // 80 bytes per row
    constexpr int APAD = 4;
    constexpr int A_STRIDE = K_TILE + APAD;

    extern __shared__ __align__(16) uint8_t smem_raw[];
    size_t off = 0;
    auto v_pack    = reinterpret_cast<uint8_t(*)[BS][PACK_STRIDE]>(&smem_raw[off]);
    off += STAGES * BS * PACK_STRIDE;
    auto v_scale_b = reinterpret_cast<__nv_bfloat16(*)[BS]>(&smem_raw[off]);
    off += STAGES * BS * sizeof(__nv_bfloat16);
    off = (off + 15) & ~size_t(15);
    auto pq_int8   = reinterpret_cast<int8_t(*)[A_STRIDE]>(&smem_raw[off]);
    off += M_TILE * A_STRIDE * sizeof(int8_t);
    off = (off + 15) & ~size_t(15);
    float* pq_scale = reinterpret_cast<float*>(&smem_raw[off]);

    // Four physical warps serve up to sixteen GQA query rows. Every MMA
    // row outside the real group must stay zero, including rows 4..15.
    for (int i = tid; i < (M_TILE - num_groups) * A_STRIDE; i += THREADS_B) {
        pq_int8[num_groups + i / A_STRIDE][i % A_STRIDE] = 0;
    }
    for (int m = num_groups + tid; m < M_TILE; m += THREADS_B) {
        pq_scale[m] = 0.0f;
    }

    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    float c_acc[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int n = 0; n < N_TILES_PER_WARP; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) c_acc[n][i] = 0.0f;
    }

    // ── Pass 1: full-S softmax (identical to fused kernel) ──────────────────
    __shared__ float m_smem[NG_PER_HEAD_MAX];
    __shared__ float invZ_smem[NG_PER_HEAD_MAX];

    for (int m = warp_id; m < num_groups; m += THREADS_B / 32) {
        const int h_q = h_kv * num_groups + m;
        if constexpr (USE_PRECOMPUTED_STATS) {
            if (lane == 0) {
                const size_t off_stats = (size_t)b * H_q + (size_t)h_q;
                m_smem[m] = softmax_m[off_stats];
                invZ_smem[m] = softmax_inv_z[off_stats];
            }
        } else {
            const float* row = scores + ((size_t)b * H_q + (size_t)h_q) * max_S;

            float local_max = -INFINITY;
            for (int i = lane; i < S; i += 32) local_max = fmaxf(local_max, row[i]);
            float mx = warp_max(local_max);
            if (!isfinite(mx)) mx = 0.0f;

            float local_sum = 0.0f;
            for (int i = lane; i < S; i += 32) local_sum += __expf(row[i] - mx);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                local_sum += __shfl_xor_sync(0xffffffffu, local_sum, o);
            float invZ = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;

            if (lane == 0) { m_smem[m] = mx; invZ_smem[m] = invZ; }
        }
    }
    if (warp_id == 0 && lane < NG_PER_HEAD_MAX && lane >= num_groups) {
        m_smem[lane] = 0.0f;
        invZ_smem[lane] = 0.0f;
    }
    __syncthreads();

    if (s_start >= s_end) return;

    // ── cp.async helper (same as fused kernel) ─────────────────────────────
    auto load_block_async = [&](int blk, int stage) {
        if (blk < blk_end) {
            const int64_t base_slot = bt_row[blk] * (int64_t)BS;
            if (tid < 64) {
                const int pos_in_blk = tid >> 2;
                const int byte_off   = (tid & 3) << 4;
                const void* g = v_pack_base
                              + (size_t)(base_slot + pos_in_blk) * D_HALF
                              + byte_off;
                void* s = (void*)&v_pack[stage][pos_in_blk][byte_off];
                unsigned si = __cvta_generic_to_shared(s);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(si), "l"(g));
            }
            if (tid < BS) {
                v_scale_b[stage][tid] = __ldg(v_scale_base + (size_t)(base_slot + tid));
            }
        } else {
            // Padding block: encode q=0 (nibble 0x8) for all bytes so
            // register-extension yields signed INT8 0.
            if (tid < 64) {
                const int pos_in_blk = tid >> 2;
                const int byte_off   = (tid & 3) << 4;
                uint8_t* dst = (uint8_t*)&v_pack[stage][pos_in_blk][byte_off];
                #pragma unroll
                for (int i = 0; i < 16; ++i) dst[i] = 0x88;
            }
            if (tid < BS) v_scale_b[stage][tid] = __float2bfloat16(0.0f);
        }
        asm volatile("cp.async.commit_group;\n");
    };

    const int n_blocks = blk_end - blk_start;
    const int n_ktiles = (n_blocks + K_BLOCKS_PER_TILE - 1) / K_BLOCKS_PER_TILE;

    if (n_ktiles > 0) {
        load_block_async(blk_start    , 0);
        load_block_async(blk_start + 1, 1);
    }

    for (int kt = 0; kt < n_ktiles; ++kt) {
        const int blk_a = blk_start + kt * K_BLOCKS_PER_TILE;
        const int blk_b = blk_a + 1;
        const bool has_b = (blk_b < blk_end);

        const int stage_a = (kt & 1) * 2 + 0;
        const int stage_b = (kt & 1) * 2 + 1;

        if (kt + 1 < n_ktiles) {
            const int next_a = blk_start + (kt + 1) * K_BLOCKS_PER_TILE;
            const int next_b = next_a + 1;
            const int nstage_a = ((kt + 1) & 1) * 2 + 0;
            const int nstage_b = ((kt + 1) & 1) * 2 + 1;
            load_block_async(next_a, nstage_a);
            load_block_async(next_b, nstage_b);
        }

        if (kt + 1 < n_ktiles) {
            asm volatile("cp.async.wait_group 2;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        // ── NO v_unpack write — v_pack stays as INT4 source.

        // ── Build pq_int8[m, k]  (identical to fused kernel) ───────────────
        for (int m = warp_id; m < num_groups; m += THREADS_B / 32) {
            const int h_q = h_kv * num_groups + m;
            const float* row =
                scores + ((size_t)b * H_q + (size_t)h_q) * max_S;
            const float mx   = m_smem[m];
            const float invZ = invZ_smem[m];

            const int k = lane;
            const int local_blk = k / BS;
            const int local_off = k % BS;
            const int abs_blk   = blk_a + local_blk;
            float pv = 0.0f;
            if (abs_blk < blk_end) {
                int pos = abs_blk * BS + local_off;
                if (pos >= s_start && pos < s_end && pos < S) {
                    int stage_x = (local_blk == 0) ? stage_a : stage_b;
                    float p = __expf(row[pos] - mx) * invZ;
                    float vs = __bfloat162float(v_scale_b[stage_x][local_off]);
                    pv = p * vs;
                }
            }

            float a = fabsf(pv);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            float row_scale = a / 127.0f;
            float inv_scale = (row_scale > 0.0f) ? (1.0f / row_scale) : 0.0f;
            int q = (int)__float2int_rn(pv * inv_scale);
            q = max(-127, min(127, q));
            pq_int8[m][k] = (int8_t)q;
            if (lane == 0) pq_scale[m] = row_scale;
        }
        __syncthreads();

        // ── MMA: build B-operand directly from v_pack in registers ─────────
        // Each lane (l4=lane/4, l_mod4=lane%4) needs, for each N-tile in this
        // warp, 8 INT8 values from v_unpack-logical positions:
        //   b_reg[0]: 4 K-rows in [k_off0, k_off0+4) of stage_a,
        //             at col = n_tile*N_TILE + l4
        //   b_reg[1]: 4 K-rows in [k_off0, k_off0+4) of stage_b,
        //             at col = same as above (or zero if !has_b)
        // where k_off0 = l_mod4 * 4.
        //
        // For col c, the packed byte is v_pack[stage][k][c>>1]; the high
        // nibble (>>4) holds the even-col INT4 (matches existing dequant).
        // After extracting the nibble we subtract 8 to unbias into signed
        // INT8 ∈ [-8, 7].
        const int n_base_warp = warp_id * N_TILES_PER_WARP;
        const int lane_div4 = lane >> 2;
        const int lane_mod4 = lane & 3;
        const int k_off0    = lane_mod4 * 4;

        // Load A from pq_int8 (same as fused kernel).
        uint32_t a_reg[4];
        {
            const int m_low  = lane >> 2;
            const int m_high = m_low + 8;
            const int k_off_a0 = (lane & 3) * 4;
            const int k_off_a1 = k_off_a0 + 16;
            a_reg[0] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_low ][k_off_a0]);
            a_reg[1] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_high][k_off_a0]);
            a_reg[2] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_low ][k_off_a1]);
            a_reg[3] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_high][k_off_a1]);
        }

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; ++nt) {
            const int n_tile = n_base_warp + nt;
            const int d_col_base = n_tile * N_TILE;
            const int col = d_col_base + lane_div4;
            const int col_byte = col >> 1;
            const bool high_nibble = ((col & 1) == 0);

            uint32_t b_reg0 = 0;
            #pragma unroll
            for (int kr = 0; kr < 4; ++kr) {
                uint8_t byte = v_pack[stage_a][k_off0 + kr][col_byte];
                int nib = high_nibble ? ((int)byte >> 4) & 0xF : (int)byte & 0xF;
                int v_int8 = nib - 8;
                b_reg0 |= ((uint32_t)(uint8_t)(int8_t)v_int8) << (kr * 8);
            }

            uint32_t b_reg1 = 0;
            if (has_b) {
                #pragma unroll
                for (int kr = 0; kr < 4; ++kr) {
                    uint8_t byte = v_pack[stage_b][k_off0 + kr][col_byte];
                    int nib = high_nibble ? ((int)byte >> 4) & 0xF : (int)byte & 0xF;
                    int v_int8 = nib - 8;
                    b_reg1 |= ((uint32_t)(uint8_t)(int8_t)v_int8) << (kr * 8);
                }
            }

            uint32_t b_reg[2] = {b_reg0, b_reg1};
            int32_t d_reg[4];
            int32_t c_zero[4] = {0, 0, 0, 0};
            mma_m16n8k32_s8s8s32(d_reg, a_reg, b_reg, c_zero);

            const int m_low  = lane >> 2;
            const int m_high = m_low + 8;
            float s_low  = pq_scale[m_low];
            float s_high = pq_scale[m_high];
            c_acc[nt][0] += (float)d_reg[0] * s_low;
            c_acc[nt][1] += (float)d_reg[1] * s_low;
            c_acc[nt][2] += (float)d_reg[2] * s_high;
            c_acc[nt][3] += (float)d_reg[3] * s_high;
        }
        __syncthreads();
    }

    // ── Writeback: atomicAdd c_acc into o_hat (identical to fused) ──────────
    {
        const int n_base_warp = warp_id * N_TILES_PER_WARP;
        const int m_low  = lane >> 2;
        const int m_high = m_low + 8;
        const int col_lo = (lane & 3) * 2;
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; ++nt) {
            const int n_tile = n_base_warp + nt;
            const int d_col_base = n_tile * N_TILE;
            if (m_low < num_groups) {
                const int h_q = h_kv * num_groups + m_low;
                const size_t base = ((size_t)b * H_q + (size_t)h_q) * D_FIXED;
                atomicAdd(o_hat + base + d_col_base + col_lo + 0, c_acc[nt][0]);
                atomicAdd(o_hat + base + d_col_base + col_lo + 1, c_acc[nt][1]);
            }
            if (m_high < num_groups) {
                const int h_q = h_kv * num_groups + m_high;
                const size_t base = ((size_t)b * H_q + (size_t)h_q) * D_FIXED;
                atomicAdd(o_hat + base + d_col_base + col_lo + 0, c_acc[nt][2]);
                atomicAdd(o_hat + base + d_col_base + col_lo + 1, c_acc[nt][3]);
            }
        }
    }

    // ── Phase 2: top-K correction (split-local contribution) ────────────────
    for (int m = warp_id; WITH_PRE_CORR && m < num_groups; m += THREADS_B / 32) {
        const int h_q = h_kv * num_groups + m;
        const float* row =
            scores + ((size_t)b * H_q + (size_t)h_q) * max_S;
        const float mx   = m_smem[m];
        const float invZ = invZ_smem[m];
        const int64_t* topk_row =
            topk_idx + ((size_t)b * H_q + (size_t)h_q) * top_k;

        const int d0 = lane * 4;
        const int d1 = d0 + 1;
        const int d2 = d0 + 2;
        const int d3 = d0 + 3;
        const int b0 = d0 >> 1, sh0 = (1 - (d0 & 1)) * 4;
        const int b1 = d1 >> 1, sh1 = (1 - (d1 & 1)) * 4;
        const int b2 = d2 >> 1, sh2 = (1 - (d2 & 1)) * 4;
        const int b3 = d3 >> 1, sh3 = (1 - (d3 & 1)) * 4;

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        float acc_W = 0.0f;

        for (int k = 0; k < top_k; ++k) {
            int64_t pos = topk_row[k];
            if (pos < 0) continue;
            int p = (int)pos;
            if (p < s_start || p >= s_end || p >= S) continue;

            int blk_idx = p / BS;
            int in_blk  = p - blk_idx * BS;
            int64_t slot = bt_row[blk_idx] * (int64_t)BS + (int64_t)in_blk;

            float a_hat = __expf(row[p] - mx) * invZ;
            float vs    = __bfloat162float(__ldg(v_scale_base + slot));
            float ws    = a_hat * vs;

            const uint8_t* v_slot_base = v_pack_base + (size_t)slot * D_HALF;
            uint8_t pk_a = __ldg(v_slot_base + b0);
            uint8_t pk_b = (b2 == b0) ? pk_a : __ldg(v_slot_base + b2);
            (void)b1; (void)b3;
            int q0 = int((pk_a >> sh0) & 0xF) - 8;
            int q1 = int((pk_a >> sh1) & 0xF) - 8;
            int q2 = int((pk_b >> sh2) & 0xF) - 8;
            int q3 = int((pk_b >> sh3) & 0xF) - 8;

            acc0 = __fmaf_rn(ws, (float)q0, acc0);
            acc1 = __fmaf_rn(ws, (float)q1, acc1);
            acc2 = __fmaf_rn(ws, (float)q2, acc2);
            acc3 = __fmaf_rn(ws, (float)q3, acc3);
            if (lane == 0) acc_W += a_hat;
        }

        const size_t base = ((size_t)b * H_q + (size_t)h_q) * D_FIXED;
        if (acc0 != 0.0f) atomicAdd(o_old_unnorm + base + d0, acc0);
        if (acc1 != 0.0f) atomicAdd(o_old_unnorm + base + d1, acc1);
        if (acc2 != 0.0f) atomicAdd(o_old_unnorm + base + d2, acc2);
        if (acc3 != 0.0f) atomicAdd(o_old_unnorm + base + d3, acc3);
        if (lane == 0 && acc_W != 0.0f) {
            atomicAdd(W_out + ((size_t)b * H_q + (size_t)h_q), acc_W);
        }
    }
}


// Host launcher for the register-resident variant (pre_corr).  Mirrors
// int8tc_softmax_v_with_pre_corr but allocates LESS dyn SMEM (no v_unpack)
// and dispatches to int8tc_softmax_v_fused_reg_kernel.
std::vector<torch::Tensor> int8tc_softmax_v_with_pre_corr_reg(
    torch::Tensor scores,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor topk_idx,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t top_k)
{
    CHECK_CUDA(scores); CHECK_CUDA(seq_lens);
    CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(topk_idx);
    TORCH_CHECK(scores.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(v_data.scalar_type()      == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(topk_idx.scalar_type()    == torch::kInt64);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(topk_idx.dim() == 3);
    TORCH_CHECK(topk_idx.size(2) == top_k);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
    TORCH_CHECK(block_size == 16);
    TORCH_CHECK(H_q % H_kv == 0);
    const int num_groups = (int)(H_q / H_kv);
    TORCH_CHECK(num_groups <= 16);

    auto sc = scores.contiguous();
    auto sl = seq_lens.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();
    auto bt = block_table.contiguous();
    auto ti = topk_idx.contiguous();

    const int B          = (int)sc.size(0);
    const int max_blocks = (int)bt.size(1);
    TORCH_CHECK((int)vs.size(2) == 1, "num_groups in v_scales must be 1");

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());
    auto o_hat        = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto o_old_unnorm = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out        = torch::zeros({B, (int64_t)H_q}, opts_f32);

    constexpr int VPACK_PAD   = 16;
    constexpr int PACK_STRIDE = D_HALF + VPACK_PAD;  // 80 — bank-conflict-free + 16-byte aligned
    constexpr int APAD = 4;
    constexpr int A_STRIDE = K_TILE + APAD;

    // No v_unpack region — saves K_TILE × V_STRIDE = 32 × 132 = 4224 B/CTA.
    size_t smem = 0;
    smem += STAGES * BS * PACK_STRIDE;                           // v_pack (padded)
    smem  = (smem + 15) & ~size_t(15);
    smem += STAGES * BS * sizeof(__nv_bfloat16);                 // v_scale_b
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * A_STRIDE * sizeof(int8_t);                  // pq_int8
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * sizeof(float);                              // pq_scale
    smem += 64;                                                  // slack

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)N_SPLITS);
    dim3 block(THREADS_B);

    cudaFuncSetAttribute(int8tc_softmax_v_fused_reg_kernel<16, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    int8tc_softmax_v_fused_reg_kernel<16, true, false><<<grid, block, smem, stream.stream()>>>(
        sc.data_ptr<float>(),
        nullptr,
        nullptr,
        sl.data_ptr<int32_t>(),
        vd.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        o_hat.data_ptr<float>(),
        ti.data_ptr<int64_t>(),
        o_old_unnorm.data_ptr<float>(),
        W_out.data_ptr<float>(),
        (int)top_k,
        (int)max_S, max_blocks, (int)H_q, (int)H_kv,
        (int)vd.stride(0), (int)vs.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}


template <int TOPK_CHUNK = 256>
__global__ void int8tc_topk_corr_chunk_kernel(
    const float*         __restrict__ scores,
    const float*         __restrict__ softmax_m,
    const float*         __restrict__ softmax_inv_z,
    const int32_t*       __restrict__ seq_lens,
    const uint8_t*       __restrict__ v_data,
    const __nv_bfloat16* __restrict__ v_scales,
    const int64_t*       __restrict__ block_table,
    const int64_t*       __restrict__ topk_idx,
    float*               __restrict__ o_old_unnorm,
    float*               __restrict__ W_out,
    int top_k,
    int max_S,
    int max_blocks,
    int H_q,
    int H_kv,
    int v_data_stride_h,
    int v_scales_stride_h)
{
    const int b = blockIdx.x;
    const int h_q = blockIdx.y;
    const int chunk = blockIdx.z;
    const int lane = threadIdx.x & 31;
    const int num_groups = H_q / H_kv;
    const int h_kv = h_q / num_groups;
    const int S = seq_lens[b];
    const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;

    const float* row = scores + bh * (size_t)max_S;
    const int64_t* topk_row = topk_idx + bh * (size_t)top_k;
    const int64_t* bt_row = block_table + (size_t)b * (size_t)max_blocks;
    const uint8_t* v_pack_base = v_data + (size_t)h_kv * (size_t)v_data_stride_h;
    const __nv_bfloat16* v_scale_base =
        v_scales + (size_t)h_kv * (size_t)v_scales_stride_h;
    const float mx = softmax_m[bh];
    const float invZ = softmax_inv_z[bh];

    const int d0 = lane * 4;
    const int d1 = d0 + 1;
    const int d2 = d0 + 2;
    const int d3 = d0 + 3;
    const int b0 = d0 >> 1, sh0 = (1 - (d0 & 1)) * 4;
    const int b1 = d1 >> 1, sh1 = (1 - (d1 & 1)) * 4;
    const int b2 = d2 >> 1, sh2 = (1 - (d2 & 1)) * 4;
    const int b3 = d3 >> 1, sh3 = (1 - (d3 & 1)) * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float accW = 0.0f;
    const int k_start = chunk * TOPK_CHUNK;
    const int k_end = min(k_start + TOPK_CHUNK, top_k);
    for (int k = k_start; k < k_end; ++k) {
        int64_t pos64 = topk_row[k];
        if (pos64 < 0) continue;
        int pos = (int)pos64;
        if (pos >= S) continue;

        int blk_idx = pos / BS;
        int in_blk = pos - blk_idx * BS;
        int64_t block_id = bt_row[blk_idx];
        if (block_id < 0) continue;
        int64_t slot = block_id * (int64_t)BS + (int64_t)in_blk;

        float w = __expf(row[pos] - mx) * invZ;
        float vs = __bfloat162float(__ldg(v_scale_base + slot));
        float ws = w * vs;

        const uint8_t* v_slot = v_pack_base + (size_t)slot * D_HALF;
        uint8_t pk_a = __ldg(v_slot + b0);
        uint8_t pk_b = (b2 == b0) ? pk_a : __ldg(v_slot + b2);
        (void)b1; (void)b3;
        int q0 = int((pk_a >> sh0) & 0xF) - 8;
        int q1 = int((pk_a >> sh1) & 0xF) - 8;
        int q2 = int((pk_b >> sh2) & 0xF) - 8;
        int q3 = int((pk_b >> sh3) & 0xF) - 8;
        acc0 = __fmaf_rn(ws, (float)q0, acc0);
        acc1 = __fmaf_rn(ws, (float)q1, acc1);
        acc2 = __fmaf_rn(ws, (float)q2, acc2);
        acc3 = __fmaf_rn(ws, (float)q3, acc3);
        if (lane == 0) accW += w;
    }

    const size_t out_base = bh * D_FIXED;
    if (acc0 != 0.0f) atomicAdd(o_old_unnorm + out_base + d0, acc0);
    if (acc1 != 0.0f) atomicAdd(o_old_unnorm + out_base + d1, acc1);
    if (acc2 != 0.0f) atomicAdd(o_old_unnorm + out_base + d2, acc2);
    if (acc3 != 0.0f) atomicAdd(o_old_unnorm + out_base + d3, acc3);
    if (lane == 0 && accW != 0.0f) atomicAdd(W_out + bh, accW);
}


std::vector<torch::Tensor> int8tc_softmax_v_with_pre_corr_reg_stats(
    torch::Tensor scores,
    torch::Tensor softmax_m,
    torch::Tensor softmax_inv_z,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor topk_idx,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t top_k)
{
    CHECK_CUDA(scores); CHECK_CUDA(softmax_m); CHECK_CUDA(softmax_inv_z);
    CHECK_CUDA(seq_lens); CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(topk_idx);
    TORCH_CHECK(scores.scalar_type()         == torch::kFloat32);
    TORCH_CHECK(softmax_m.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(softmax_inv_z.scalar_type()  == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type()       == torch::kInt32);
    TORCH_CHECK(v_data.scalar_type()         == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()       == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type()    == torch::kInt64);
    TORCH_CHECK(topk_idx.scalar_type()       == torch::kInt64);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(softmax_m.dim() == 2);
    TORCH_CHECK(softmax_inv_z.dim() == 2);
    TORCH_CHECK(topk_idx.dim() == 3);
    TORCH_CHECK(topk_idx.size(2) == top_k);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
    TORCH_CHECK(block_size == 16);
    TORCH_CHECK(H_q % H_kv == 0);
    const int num_groups = (int)(H_q / H_kv);
    TORCH_CHECK(num_groups <= 16);

    auto sc = scores.contiguous();
    auto sm = softmax_m.contiguous();
    auto sz = softmax_inv_z.contiguous();
    auto sl = seq_lens.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();
    auto bt = block_table.contiguous();
    auto ti = topk_idx.contiguous();

    const int B          = (int)sc.size(0);
    const int max_blocks = (int)bt.size(1);
    TORCH_CHECK((int)sm.size(0) == B && (int)sm.size(1) == (int)H_q);
    TORCH_CHECK((int)sz.size(0) == B && (int)sz.size(1) == (int)H_q);
    TORCH_CHECK((int)vs.size(2) == 1, "num_groups in v_scales must be 1");

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());
    auto o_hat        = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto o_old_unnorm = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out        = torch::zeros({B, (int64_t)H_q}, opts_f32);

    constexpr int VPACK_PAD   = 16;
    constexpr int PACK_STRIDE = D_HALF + VPACK_PAD;
    constexpr int APAD = 4;
    constexpr int A_STRIDE = K_TILE + APAD;

    size_t smem = 0;
    smem += STAGES * BS * PACK_STRIDE;
    smem  = (smem + 15) & ~size_t(15);
    smem += STAGES * BS * sizeof(__nv_bfloat16);
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * A_STRIDE * sizeof(int8_t);
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * sizeof(float);
    smem += 64;

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)N_SPLITS);
    dim3 block(THREADS_B);
    const bool parallel_topk_corr = (top_k >= 1024);

    if (parallel_topk_corr) {
        cudaFuncSetAttribute(int8tc_softmax_v_fused_reg_kernel<16, false, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        int8tc_softmax_v_fused_reg_kernel<16, false, true><<<grid, block, smem, stream.stream()>>>(
            sc.data_ptr<float>(),
            sm.data_ptr<float>(),
            sz.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            o_hat.data_ptr<float>(),
            ti.data_ptr<int64_t>(),
            o_old_unnorm.data_ptr<float>(),
            W_out.data_ptr<float>(),
            (int)top_k,
            (int)max_S, max_blocks, (int)H_q, (int)H_kv,
            (int)vd.stride(0), (int)vs.stride(0)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        constexpr int TOPK_CORR_CHUNK = 256;
        const int n_topk_chunks = ((int)top_k + TOPK_CORR_CHUNK - 1) / TOPK_CORR_CHUNK;
        dim3 corr_grid((unsigned)B, (unsigned)H_q, (unsigned)n_topk_chunks);
        dim3 corr_block(32);
        int8tc_topk_corr_chunk_kernel<TOPK_CORR_CHUNK><<<corr_grid, corr_block, 0, stream.stream()>>>(
            sc.data_ptr<float>(),
            sm.data_ptr<float>(),
            sz.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            ti.data_ptr<int64_t>(),
            o_old_unnorm.data_ptr<float>(),
            W_out.data_ptr<float>(),
            (int)top_k,
            (int)max_S, max_blocks, (int)H_q, (int)H_kv,
            (int)vd.stride(0), (int)vs.stride(0)
        );
    } else {
        cudaFuncSetAttribute(int8tc_softmax_v_fused_reg_kernel<16, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
        int8tc_softmax_v_fused_reg_kernel<16, true, true><<<grid, block, smem, stream.stream()>>>(
            sc.data_ptr<float>(),
            sm.data_ptr<float>(),
            sz.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            o_hat.data_ptr<float>(),
            ti.data_ptr<int64_t>(),
            o_old_unnorm.data_ptr<float>(),
            W_out.data_ptr<float>(),
            (int)top_k,
            (int)max_S, max_blocks, (int)H_q, (int)H_kv,
            (int)vd.stride(0), (int)vs.stride(0)
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}


// Diagnostic / prototype helper: compute only the topK old-cache contribution
// used by JetFire correction:
//
//   o_old_unnorm = sum_{i in topK} softmax_full(score_i) * V_int4_i
//   W_out        = sum_{i in topK} softmax_full(score_i)
//
// This intentionally skips full o_hat.  It lets the unit benchmark estimate a
// BitDecode-style hybrid path:
//
//   JetFire StageA(topK + softmax stats)
//   + fast packed full-attention decode for o_hat
//   + this topK-only correction term.
std::vector<torch::Tensor> int8tc_topk_corr_only_stats(
    torch::Tensor scores,
    torch::Tensor softmax_m,
    torch::Tensor softmax_inv_z,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor topk_idx,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t top_k)
{
    CHECK_CUDA(scores); CHECK_CUDA(softmax_m); CHECK_CUDA(softmax_inv_z);
    CHECK_CUDA(seq_lens); CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(topk_idx);
    TORCH_CHECK(scores.scalar_type()         == torch::kFloat32);
    TORCH_CHECK(softmax_m.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(softmax_inv_z.scalar_type()  == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type()       == torch::kInt32);
    TORCH_CHECK(v_data.scalar_type()         == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()       == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type()    == torch::kInt64);
    TORCH_CHECK(topk_idx.scalar_type()       == torch::kInt64);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(softmax_m.dim() == 2);
    TORCH_CHECK(softmax_inv_z.dim() == 2);
    TORCH_CHECK(topk_idx.dim() == 3);
    TORCH_CHECK(topk_idx.size(2) == top_k);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
    TORCH_CHECK(block_size == 16);
    TORCH_CHECK(H_q % H_kv == 0);
    const int num_groups = (int)(H_q / H_kv);
    TORCH_CHECK(num_groups <= 16);

    auto sc = scores.contiguous();
    auto sm = softmax_m.contiguous();
    auto sz = softmax_inv_z.contiguous();
    auto sl = seq_lens.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();
    auto bt = block_table.contiguous();
    auto ti = topk_idx.contiguous();

    const int B          = (int)sc.size(0);
    const int max_blocks = (int)bt.size(1);
    TORCH_CHECK((int)sm.size(0) == B && (int)sm.size(1) == (int)H_q);
    TORCH_CHECK((int)sz.size(0) == B && (int)sz.size(1) == (int)H_q);
    TORCH_CHECK((int)vs.size(2) == 1, "num_groups in v_scales must be 1");

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());
    auto o_old_unnorm = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out        = torch::zeros({B, (int64_t)H_q}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int TOPK_CORR_CHUNK = 256;
    const int n_topk_chunks = ((int)top_k + TOPK_CORR_CHUNK - 1) / TOPK_CORR_CHUNK;
    dim3 corr_grid((unsigned)B, (unsigned)H_q, (unsigned)n_topk_chunks);
    dim3 corr_block(32);
    int8tc_topk_corr_chunk_kernel<TOPK_CORR_CHUNK><<<corr_grid, corr_block, 0, stream.stream()>>>(
        sc.data_ptr<float>(),
        sm.data_ptr<float>(),
        sz.data_ptr<float>(),
        sl.data_ptr<int32_t>(),
        vd.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        ti.data_ptr<int64_t>(),
        o_old_unnorm.data_ptr<float>(),
        W_out.data_ptr<float>(),
        (int)top_k,
        (int)max_S, max_blocks, (int)H_q, (int)H_kv,
        (int)vd.stride(0), (int)vs.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_old_unnorm, W_out};
}


// ── Producer/Consumer kernel ('pc') ───────────────────────────────────────
// Warp-specialized rewrite targeting the structural barrier-stall bottleneck
// found in the existing kernels by ncu:
//
//   * Existing kernels: 4 warps/CTA, 1 CTA/SM resident, 6.25% occupancy,
//     88% No-Eligible cycles.  Stalls split ~35% on L1TEX scoreboard +
//     ~31% on CTA-wide __syncthreads.  All 4 warps lock-step at every
//     barrier; one slow warp stalls the SM.
//
//   * 'pc' kernel: 8 warps/CTA (256 threads).
//     Warps 0..3 = CONSUMERS — pq build + mma + atomicAdd.
//     Warps 4..7 = PRODUCERS — cp.async + commit + wait + mbarrier signal.
//     The handshake is per-stage mbarrier (SM_80+ async barrier), NOT
//     __syncthreads — so producers can run ahead of consumers (and vice
//     versa) without lockstep stall.  Consumers only sync among themselves
//     via named barrier `bar.sync 0, 128`.
//
// Resource budget (target: 4 CTAs/SM @ ~50% occupancy):
//   regs: ~60/thread × 256 → 15360/CTA → 4 CTAs/SM (65536 regs/SM)
//   smem: ~11 KB/CTA → 20 CTAs/SM by SMEM
//   warp limit: 64/SM ÷ 8/CTA = 8 CTAs/SM
//   bottleneck: registers → 4 CTAs/SM × 8 warps = 32 warps/SM = 50% occ
//
// Stage organization: STAGES=4 ring buffer, each stage = 1 K-tile (2 BS
// blocks).  v_pack has VPACK_PAD=16 (PACK_STRIDE=80) for bank-conflict-free
// register-resident V dequant in mma B-build.
template <int NG_PER_HEAD_MAX = 16, bool WITH_PRE_CORR = false>
__global__ void int8tc_softmax_v_fused_pc_kernel(
    const float*         __restrict__ scores,
    const int32_t*       __restrict__ seq_lens,
    const uint8_t*       __restrict__ v_data,
    const __nv_bfloat16* __restrict__ v_scales,
    const int64_t*       __restrict__ block_table,
    float*               __restrict__ o_hat,
    const int64_t*       __restrict__ topk_idx,
    float*               __restrict__ o_old_unnorm,
    float*               __restrict__ W_out,
    int top_k,
    int max_S,
    int max_blocks,
    int H_q,
    int H_kv,
    int v_data_stride_h,
    int v_scales_stride_h)
{
    constexpr int CONS_WARPS    = 4;
    constexpr int PROD_WARPS    = 4;
    constexpr int TOTAL_WARPS   = CONS_WARPS + PROD_WARPS;
    constexpr int CONS_THREADS  = CONS_WARPS * 32;     // 128
    constexpr int PROD_THREADS  = PROD_WARPS * 32;     // 128
    constexpr int PC_STAGES     = 4;
    constexpr int VPACK_PAD     = 16;
    constexpr int PACK_STRIDE   = D_HALF + VPACK_PAD;  // 80
    constexpr int APAD          = 4;
    constexpr int A_STRIDE      = K_TILE + APAD;       // 36

    const int b       = blockIdx.x;
    const int h_kv    = blockIdx.y;
    const int s_split = blockIdx.z;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const bool is_producer = (warp_id >= CONS_WARPS);
    const int cons_tid_in_group = tid;                          // 0..127
    const int prod_tid_in_group = tid - CONS_THREADS;           // 0..127 (valid when is_producer)

    const int num_groups = H_q / H_kv;
    const int S          = seq_lens[b];

    const int chunk_per_split = (max_S + N_SPLITS - 1) / N_SPLITS;
    const int s_start = s_split * chunk_per_split;
    const int s_end   = min(s_start + chunk_per_split, S);
    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    // ── Dynamic SMEM layout ─────────────────────────────────────────────────
    // v_pack[STAGES][2][BS][PACK_STRIDE]  : 4 × 2 × 16 × 80 = 10240 B
    // v_scale_b[STAGES][2][BS]            : 4 × 2 × 16 × 2  = 256 B
    // pq_int8[M_TILE][A_STRIDE]           : 16 × 36         = 576 B
    // pq_scale[M_TILE]                    : 16 × 4          = 64 B
    // mb_prod[STAGES] (uint64)            : 4 × 8           = 32 B
    // mb_cons[STAGES] (uint64)            : 4 × 8           = 32 B
    extern __shared__ __align__(16) uint8_t smem_raw[];
    size_t off = 0;
    auto v_pack    = reinterpret_cast<uint8_t(*)[2][BS][PACK_STRIDE]>(&smem_raw[off]);
    off += PC_STAGES * 2 * BS * PACK_STRIDE;
    auto v_scale_b = reinterpret_cast<__nv_bfloat16(*)[2][BS]>(&smem_raw[off]);
    off += PC_STAGES * 2 * BS * sizeof(__nv_bfloat16);
    off = (off + 15) & ~size_t(15);
    auto pq_int8   = reinterpret_cast<int8_t(*)[A_STRIDE]>(&smem_raw[off]);
    off += M_TILE * A_STRIDE * sizeof(int8_t);
    off = (off + 15) & ~size_t(15);
    float* pq_scale = reinterpret_cast<float*>(&smem_raw[off]);
    off += M_TILE * sizeof(float);
    off = (off + 7) & ~size_t(7);
    uint64_t* mb_prod = reinterpret_cast<uint64_t*>(&smem_raw[off]);
    off += PC_STAGES * sizeof(uint64_t);
    uint64_t* mb_cons = reinterpret_cast<uint64_t*>(&smem_raw[off]);

    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    // ── mbarrier init (thread 0 only) ───────────────────────────────────────
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < PC_STAGES; ++s) {
            mbar_init(&mb_prod[s], PROD_THREADS);  // producer arrivals fill the stage
            mbar_init(&mb_cons[s], CONS_THREADS);  // consumer arrivals release the stage
        }
    }
    __syncthreads();

    // Per-warp FP32 accumulators (consumers only need these; producers
    // declare them so the kernel has a uniform stack layout).
    float c_acc[N_TILES_PER_WARP][4];
    #pragma unroll
    for (int n = 0; n < N_TILES_PER_WARP; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) c_acc[n][i] = 0.0f;
    }

    // ── Pass 1: full-S softmax (consumer warps only) ────────────────────────
    __shared__ float m_smem[NG_PER_HEAD_MAX];
    __shared__ float invZ_smem[NG_PER_HEAD_MAX];

    if (!is_producer && warp_id < num_groups) {
        const int m   = warp_id;
        const int h_q = h_kv * num_groups + m;
        const float* row = scores + ((size_t)b * H_q + (size_t)h_q) * max_S;

        float local_max = -INFINITY;
        for (int i = lane; i < S; i += 32) local_max = fmaxf(local_max, row[i]);
        float mx = warp_max(local_max);
        if (!isfinite(mx)) mx = 0.0f;

        float local_sum = 0.0f;
        for (int i = lane; i < S; i += 32) local_sum += __expf(row[i] - mx);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            local_sum += __shfl_xor_sync(0xffffffffu, local_sum, o);
        float invZ = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;

        if (lane == 0) { m_smem[m] = mx; invZ_smem[m] = invZ; }
    }
    if (!is_producer && warp_id == 0 && lane < NG_PER_HEAD_MAX && lane >= num_groups) {
        m_smem[lane] = 0.0f;
        invZ_smem[lane] = 0.0f;
    }
    __syncthreads();  // make softmax data visible to ALL warps + ensure
                      // mbarriers init is visible to all before producers
                      // start arriving.

    if (s_start >= s_end) return;

    const int n_blocks = blk_end - blk_start;
    const int n_ktiles = (n_blocks + K_BLOCKS_PER_TILE - 1) / K_BLOCKS_PER_TILE;

    // ────────────────────────────────────────────────────────────────────────
    // PRODUCER WARPS: cp.async loads + mbarrier signaling
    // ────────────────────────────────────────────────────────────────────────
    // Pipelined: each producer thread keeps up to PC_STAGES cp.async groups
    // in flight, then arrives on the OLDEST stage's mb_prod when its group
    // lands.  This is the standard cp.async pipeline pattern; my first cut
    // did wait_group(0) per K-tile, fully serializing HBM latency.
    if (is_producer) {
        const int block_select = prod_tid_in_group >> 6;          // 0 or 1
        const int row_in_block = (prod_tid_in_group & 63) >> 2;   // 0..15
        const int byte_off     = (prod_tid_in_group & 3) << 4;    // 0,16,32,48

        // ── load_one(kt, stage): issue THIS thread's chunk of K-tile kt
        // into stage[stage].  Caller emits commit_group after a batch.
        auto issue_chunk = [&](int kt, int stage) {
            const int blk_a = blk_start + kt * K_BLOCKS_PER_TILE;
            const int blk_b = blk_a + 1;
            const int blk_id = (block_select == 0) ? blk_a : blk_b;
            if (blk_id < blk_end) {
                const int64_t base_slot = bt_row[blk_id] * (int64_t)BS;
                const void* g = v_pack_base
                              + (size_t)(base_slot + row_in_block) * D_HALF
                              + byte_off;
                void* s = (void*)&v_pack[stage][block_select][row_in_block][byte_off];
                unsigned si = __cvta_generic_to_shared(s);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(si), "l"(g));
            } else {
                uint8_t* dst = (uint8_t*)&v_pack[stage][block_select][row_in_block][byte_off];
                #pragma unroll
                for (int i = 0; i < 16; ++i) dst[i] = 0x88;
            }
            // v_scale (16 threads do block_a, 16 threads do block_b — only
            // producer threads 0..31 contribute; idempotent across producer
            // warps using prod_tid_in_group's low 5 bits).
            if (prod_tid_in_group < 32) {
                const int scale_row = prod_tid_in_group & 15;
                const int scale_blk = (prod_tid_in_group >> 4) & 1;
                const int blk_id_s = (scale_blk == 0) ? blk_a : blk_b;
                if (blk_id_s < blk_end) {
                    const int64_t base_slot = bt_row[blk_id_s] * (int64_t)BS;
                    v_scale_b[stage][scale_blk][scale_row] =
                        __ldg(v_scale_base + (size_t)(base_slot + scale_row));
                } else {
                    v_scale_b[stage][scale_blk][scale_row] = __float2bfloat16(0.0f);
                }
            }
            asm volatile("cp.async.commit_group;\n");
        };

        // Prologue: issue first min(PC_STAGES, n_ktiles) K-tiles.
        const int n_prol = (n_ktiles < PC_STAGES) ? n_ktiles : PC_STAGES;
        #pragma unroll
        for (int kt = 0; kt < PC_STAGES; ++kt) {
            if (kt < n_prol) {
                issue_chunk(kt, kt);  // kt < PC_STAGES → stage == kt
            } else {
                // Pad commit groups so wait_group bookkeeping is uniform.
                asm volatile("cp.async.commit_group;\n");
            }
        }

        // Main loop: pipeline depth = PC_STAGES groups in flight.
        //
        // ORDER IS CRITICAL.  We must arrive on mb_prod[kt%PC_STAGES] BEFORE
        // waiting on mb_cons[kt%PC_STAGES] — otherwise consumer cannot make
        // progress (it waits on mb_prod), cannot arrive on mb_cons, and
        // producer deadlocks.  The pattern is: wait_group → arrive(prod) →
        // (optionally) wait(cons) → issue_chunk for the slot we just freed.
        for (int kt = 0; kt < n_ktiles; ++kt) {
            const int stage = kt % PC_STAGES;

            // Wait until ≤ PC_STAGES-1 groups remain in flight — the oldest
            // (= kt's) has now landed in SMEM.
            asm volatile("cp.async.wait_group %0;\n" :: "n"(PC_STAGES - 1));

            // Signal consumers that K-tile kt is ready in stage[stage].
            mbar_arrive(&mb_prod[stage]);

            // Issue next prefetch (kt + PC_STAGES) into the slot kt just
            // vacated, AFTER the consumer has consumed kt.  Skip when we're
            // out of K-tiles; emit empty commit_group so wait_group
            // bookkeeping (the count of pending groups) stays consistent.
            const int issue_kt = kt + PC_STAGES;
            if (issue_kt < n_ktiles) {
                // Wait for consumer to release this slot (it held kt before).
                const int wait_parity = (kt / PC_STAGES) & 1;
                mbar_wait(&mb_cons[stage], wait_parity);
                issue_chunk(issue_kt, stage);
            } else {
                asm volatile("cp.async.commit_group;\n");
            }
        }
        return;
    }

    // ────────────────────────────────────────────────────────────────────────
    // CONSUMER WARPS: pq build + mma + atomicAdd
    // ────────────────────────────────────────────────────────────────────────
    for (int kt = 0; kt < n_ktiles; ++kt) {
        const int stage = kt % PC_STAGES;
        const int prod_parity = (kt / PC_STAGES) & 1;

        // Wait for this stage's data to be produced.
        mbar_wait(&mb_prod[stage], prod_parity);

        const int blk_a = blk_start + kt * K_BLOCKS_PER_TILE;
        const int blk_b = blk_a + 1;
        const bool has_b = (blk_b < blk_end);

        // ── pq build: each consumer warp handles one M-row ──────────────────
        if (warp_id < num_groups) {
            const int m = warp_id;
            const int h_q = h_kv * num_groups + m;
            const float* row =
                scores + ((size_t)b * H_q + (size_t)h_q) * max_S;
            const float mx   = m_smem[m];
            const float invZ = invZ_smem[m];

            const int k = lane;                       // 0..31
            const int local_blk = k / BS;             // 0 or 1
            const int local_off = k % BS;             // 0..15
            const int abs_blk   = blk_a + local_blk;
            float pv = 0.0f;
            if (abs_blk < blk_end) {
                int pos = abs_blk * BS + local_off;
                if (pos >= s_start && pos < s_end && pos < S) {
                    float p = __expf(row[pos] - mx) * invZ;
                    float vs = __bfloat162float(v_scale_b[stage][local_blk][local_off]);
                    pv = p * vs;
                }
            }

            float a = fabsf(pv);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            float row_scale = a / 127.0f;
            float inv_scale = (row_scale > 0.0f) ? (1.0f / row_scale) : 0.0f;
            int q = (int)__float2int_rn(pv * inv_scale);
            q = max(-127, min(127, q));
            pq_int8[m][k] = (int8_t)q;
            if (lane == 0) pq_scale[m] = row_scale;
        } else if (warp_id < CONS_WARPS) {
            // Padded M-row: zero out so mma over those rows is 0.
            if (lane < K_TILE) pq_int8[warp_id][lane] = 0;
            if (lane == 0)     pq_scale[warp_id] = 0.0f;
        }

        // Sync among consumers ONLY (128 threads, named barrier 0).
        asm volatile("bar.sync 0, %0;\n" :: "n"(CONS_THREADS));

        // ── MMA: build B-operand from v_pack in registers ──────────────────
        const int n_base_warp = warp_id * N_TILES_PER_WARP;
        const int lane_div4 = lane >> 2;
        const int lane_mod4 = lane & 3;
        const int k_off0    = lane_mod4 * 4;

        uint32_t a_reg[4];
        {
            const int m_low  = lane >> 2;
            const int m_high = m_low + 8;
            const int k_off_a0 = (lane & 3) * 4;
            const int k_off_a1 = k_off_a0 + 16;
            a_reg[0] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_low ][k_off_a0]);
            a_reg[1] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_high][k_off_a0]);
            a_reg[2] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_low ][k_off_a1]);
            a_reg[3] = *reinterpret_cast<const uint32_t*>(&pq_int8[m_high][k_off_a1]);
        }

        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; ++nt) {
            const int n_tile = n_base_warp + nt;
            const int d_col_base = n_tile * N_TILE;
            const int col = d_col_base + lane_div4;
            const int col_byte = col >> 1;
            const bool high_nibble = ((col & 1) == 0);

            // b_reg[0]: K rows k_off0..k_off0+3 from v_pack[stage][block_a].
            uint32_t b_reg0 = 0;
            #pragma unroll
            for (int kr = 0; kr < 4; ++kr) {
                uint8_t byte = v_pack[stage][0][k_off0 + kr][col_byte];
                int nib = high_nibble ? ((int)byte >> 4) & 0xF : (int)byte & 0xF;
                int v_int8 = nib - 8;
                b_reg0 |= ((uint32_t)(uint8_t)(int8_t)v_int8) << (kr * 8);
            }

            // b_reg[1]: K rows k_off0..k_off0+3 from v_pack[stage][block_b]
            //          (or zero if !has_b).
            uint32_t b_reg1 = 0;
            if (has_b) {
                #pragma unroll
                for (int kr = 0; kr < 4; ++kr) {
                    uint8_t byte = v_pack[stage][1][k_off0 + kr][col_byte];
                    int nib = high_nibble ? ((int)byte >> 4) & 0xF : (int)byte & 0xF;
                    int v_int8 = nib - 8;
                    b_reg1 |= ((uint32_t)(uint8_t)(int8_t)v_int8) << (kr * 8);
                }
            }

            uint32_t b_reg[2] = {b_reg0, b_reg1};
            int32_t d_reg[4];
            int32_t c_zero[4] = {0, 0, 0, 0};
            mma_m16n8k32_s8s8s32(d_reg, a_reg, b_reg, c_zero);

            const int m_low  = lane >> 2;
            const int m_high = m_low + 8;
            float s_low  = pq_scale[m_low];
            float s_high = pq_scale[m_high];
            c_acc[nt][0] += (float)d_reg[0] * s_low;
            c_acc[nt][1] += (float)d_reg[1] * s_low;
            c_acc[nt][2] += (float)d_reg[2] * s_high;
            c_acc[nt][3] += (float)d_reg[3] * s_high;
        }

        // Signal producer that this stage is consumed.
        mbar_arrive(&mb_cons[stage]);
    }

    // ── Writeback: atomicAdd c_acc into o_hat ───────────────────────────────
    {
        const int n_base_warp = warp_id * N_TILES_PER_WARP;
        const int m_low  = lane >> 2;
        const int m_high = m_low + 8;
        const int col_lo = (lane & 3) * 2;
        #pragma unroll
        for (int nt = 0; nt < N_TILES_PER_WARP; ++nt) {
            const int n_tile = n_base_warp + nt;
            const int d_col_base = n_tile * N_TILE;
            if (m_low < num_groups) {
                const int h_q = h_kv * num_groups + m_low;
                const size_t base = ((size_t)b * H_q + (size_t)h_q) * D_FIXED;
                atomicAdd(o_hat + base + d_col_base + col_lo + 0, c_acc[nt][0]);
                atomicAdd(o_hat + base + d_col_base + col_lo + 1, c_acc[nt][1]);
            }
            if (m_high < num_groups) {
                const int h_q = h_kv * num_groups + m_high;
                const size_t base = ((size_t)b * H_q + (size_t)h_q) * D_FIXED;
                atomicAdd(o_hat + base + d_col_base + col_lo + 0, c_acc[nt][2]);
                atomicAdd(o_hat + base + d_col_base + col_lo + 1, c_acc[nt][3]);
            }
        }
    }

    // ── Phase 2: top-K correction (consumer warps only, warp_id < num_groups)
    if (WITH_PRE_CORR && warp_id < num_groups) {
        const int m   = warp_id;
        const int h_q = h_kv * num_groups + m;
        const float* row =
            scores + ((size_t)b * H_q + (size_t)h_q) * max_S;
        const float mx   = m_smem[m];
        const float invZ = invZ_smem[m];
        const int64_t* topk_row =
            topk_idx + ((size_t)b * H_q + (size_t)h_q) * top_k;

        const int d0 = lane * 4;
        const int d1 = d0 + 1;
        const int d2 = d0 + 2;
        const int d3 = d0 + 3;
        const int bx0 = d0 >> 1, sh0 = (1 - (d0 & 1)) * 4;
        const int bx1 = d1 >> 1, sh1 = (1 - (d1 & 1)) * 4;
        const int bx2 = d2 >> 1, sh2 = (1 - (d2 & 1)) * 4;
        const int bx3 = d3 >> 1, sh3 = (1 - (d3 & 1)) * 4;

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        float acc_W = 0.0f;

        for (int k = 0; k < top_k; ++k) {
            int64_t pos = topk_row[k];
            if (pos < 0) continue;
            int p = (int)pos;
            if (p < s_start || p >= s_end || p >= S) continue;

            int blk_idx = p / BS;
            int in_blk  = p - blk_idx * BS;
            int64_t slot = bt_row[blk_idx] * (int64_t)BS + (int64_t)in_blk;

            float a_hat = __expf(row[p] - mx) * invZ;
            float vs    = __bfloat162float(__ldg(v_scale_base + slot));
            float ws    = a_hat * vs;

            const uint8_t* v_slot_base = v_pack_base + (size_t)slot * D_HALF;
            uint8_t pk_a = __ldg(v_slot_base + bx0);
            uint8_t pk_b = (bx2 == bx0) ? pk_a : __ldg(v_slot_base + bx2);
            (void)bx1; (void)bx3;
            int q0 = int((pk_a >> sh0) & 0xF) - 8;
            int q1 = int((pk_a >> sh1) & 0xF) - 8;
            int q2 = int((pk_b >> sh2) & 0xF) - 8;
            int q3 = int((pk_b >> sh3) & 0xF) - 8;

            acc0 = __fmaf_rn(ws, (float)q0, acc0);
            acc1 = __fmaf_rn(ws, (float)q1, acc1);
            acc2 = __fmaf_rn(ws, (float)q2, acc2);
            acc3 = __fmaf_rn(ws, (float)q3, acc3);
            if (lane == 0) acc_W += a_hat;
        }

        const size_t base = ((size_t)b * H_q + (size_t)h_q) * D_FIXED;
        if (acc0 != 0.0f) atomicAdd(o_old_unnorm + base + d0, acc0);
        if (acc1 != 0.0f) atomicAdd(o_old_unnorm + base + d1, acc1);
        if (acc2 != 0.0f) atomicAdd(o_old_unnorm + base + d2, acc2);
        if (acc3 != 0.0f) atomicAdd(o_old_unnorm + base + d3, acc3);
        if (lane == 0 && acc_W != 0.0f) {
            atomicAdd(W_out + ((size_t)b * H_q + (size_t)h_q), acc_W);
        }
    }
}


// Host launcher for the pc kernel (pre_corr variant).
std::vector<torch::Tensor> int8tc_softmax_v_with_pre_corr_pc(
    torch::Tensor scores,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor topk_idx,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t top_k)
{
    CHECK_CUDA(scores); CHECK_CUDA(seq_lens);
    CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(topk_idx);
    TORCH_CHECK(scores.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(v_data.scalar_type()      == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(topk_idx.scalar_type()    == torch::kInt64);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(topk_idx.dim() == 3);
    TORCH_CHECK(topk_idx.size(2) == top_k);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
    TORCH_CHECK(block_size == 16);
    TORCH_CHECK(H_q % H_kv == 0);
    const int num_groups = (int)(H_q / H_kv);
    TORCH_CHECK(num_groups <= THREADS_B / 32, "legacy INT8TC Stage-B supports GQA <= 4; use the reg variant for GQA up to 16");

    auto sc = scores.contiguous();
    auto sl = seq_lens.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();
    auto bt = block_table.contiguous();
    auto ti = topk_idx.contiguous();

    const int B          = (int)sc.size(0);
    const int max_blocks = (int)bt.size(1);
    TORCH_CHECK((int)vs.size(2) == 1, "num_groups in v_scales must be 1");

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());
    auto o_hat        = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto o_old_unnorm = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out        = torch::zeros({B, (int64_t)H_q}, opts_f32);

    constexpr int PC_STAGES   = 4;
    constexpr int VPACK_PAD   = 16;
    constexpr int PACK_STRIDE = D_HALF + VPACK_PAD;
    constexpr int APAD = 4;
    constexpr int A_STRIDE = K_TILE + APAD;

    size_t smem = 0;
    smem += PC_STAGES * 2 * BS * PACK_STRIDE;                    // v_pack
    smem += PC_STAGES * 2 * BS * sizeof(__nv_bfloat16);          // v_scale_b
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * A_STRIDE * sizeof(int8_t);                  // pq_int8
    smem  = (smem + 15) & ~size_t(15);
    smem += M_TILE * sizeof(float);                              // pq_scale
    smem  = (smem + 7) & ~size_t(7);
    smem += PC_STAGES * sizeof(uint64_t);                        // mb_prod
    smem += PC_STAGES * sizeof(uint64_t);                        // mb_cons
    smem += 64;                                                  // slack

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)N_SPLITS);
    dim3 block(/*8 warps*/ 256);

    cudaFuncSetAttribute(int8tc_softmax_v_fused_pc_kernel<16, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    int8tc_softmax_v_fused_pc_kernel<16, true><<<grid, block, smem, stream.stream()>>>(
        sc.data_ptr<float>(),
        sl.data_ptr<int32_t>(),
        vd.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        o_hat.data_ptr<float>(),
        ti.data_ptr<int64_t>(),
        o_old_unnorm.data_ptr<float>(),
        W_out.data_ptr<float>(),
        (int)top_k,
        (int)max_S, max_blocks, (int)H_q, (int)H_kv,
        (int)vd.stride(0), (int)vs.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}


}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8tc_softmax_v_fused", &int8tc_softmax_v_fused,
          "JetFire fused INT8 tensor-core Stage B "
          "(softmax + V-int4 dequant + WMMA in a single launch).",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"), py::arg("seq_lens"),
          py::arg("v_data"), py::arg("v_scales"),
          py::arg("block_table"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"));

    m.def("int8tc_softmax_v_fused_inplace", &int8tc_softmax_v_fused_inplace,
          "In-place variant: caller supplies pre-zeroed o_hat (for microbench).",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"), py::arg("seq_lens"),
          py::arg("v_data"), py::arg("v_scales"),
          py::arg("block_table"), py::arg("o_hat"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"));

    m.def("int8tc_softmax_v_with_pre_corr",
          &int8tc_softmax_v_with_pre_corr,
          "JetFire fused INT8 tensor-core Stage B + top-K correction "
          "(returns [o_hat, o_old_unnorm, W]). Drop-in for "
          "int4_softmax_v_with_pre_corr_wvs2.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"),
          py::arg("v_data"),
          py::arg("v_scales"),
          py::arg("block_table"),
          py::arg("seq_lens"),
          py::arg("topk_idx"),
          py::arg("max_S"),
          py::arg("H_q"),
          py::arg("H_kv"),
          py::arg("block_size"),
          py::arg("top_k"));

    m.def("int8tc_softmax_v_with_pre_corr_pc",
          &int8tc_softmax_v_with_pre_corr_pc,
          "Producer/Consumer warp-specialized variant: 8 warps/CTA (4 prod "
          "+ 4 cons) with mbarrier async sync.  Targets the CTA-barrier "
          "stall (30.7% of stall cycles in the lockstep kernels).  Same "
          "I/O signature as int8tc_softmax_v_with_pre_corr.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"),
          py::arg("v_data"),
          py::arg("v_scales"),
          py::arg("block_table"),
          py::arg("seq_lens"),
          py::arg("topk_idx"),
          py::arg("max_S"),
          py::arg("H_q"),
          py::arg("H_kv"),
          py::arg("block_size"),
          py::arg("top_k"));

    m.def("int8tc_softmax_v_with_pre_corr_reg",
          &int8tc_softmax_v_with_pre_corr_reg,
          "Register-resident V expansion variant: skips the v_unpack SMEM "
          "round-trip used by the default fused kernel, building the mma "
          "B operand directly from v_pack (INT4) in registers.  Same I/O "
          "signature as int8tc_softmax_v_with_pre_corr.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"),
          py::arg("v_data"),
          py::arg("v_scales"),
          py::arg("block_table"),
          py::arg("seq_lens"),
          py::arg("topk_idx"),
          py::arg("max_S"),
          py::arg("H_q"),
          py::arg("H_kv"),
          py::arg("block_size"),
          py::arg("top_k"));

    m.def("int8tc_softmax_v_with_pre_corr_reg_stats",
          &int8tc_softmax_v_with_pre_corr_reg_stats,
          "Register-resident V expansion variant with precomputed "
          "softmax stats from Stage A.  Skips the internal full-S "
          "softmax pass and returns [o_hat, o_old_unnorm, W].",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"),
          py::arg("softmax_m"),
          py::arg("softmax_inv_z"),
          py::arg("v_data"),
          py::arg("v_scales"),
          py::arg("block_table"),
          py::arg("seq_lens"),
          py::arg("topk_idx"),
          py::arg("max_S"),
          py::arg("H_q"),
          py::arg("H_kv"),
          py::arg("block_size"),
          py::arg("top_k"));

    m.def("int8tc_topk_corr_only_stats",
          &int8tc_topk_corr_only_stats,
          "Diagnostic helper: compute only [o_old_unnorm, W] from "
          "topK using caller-provided softmax stats.  Skips full o_hat.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"),
          py::arg("softmax_m"),
          py::arg("softmax_inv_z"),
          py::arg("v_data"),
          py::arg("v_scales"),
          py::arg("block_table"),
          py::arg("seq_lens"),
          py::arg("topk_idx"),
          py::arg("max_S"),
          py::arg("H_q"),
          py::arg("H_kv"),
          py::arg("block_size"),
          py::arg("top_k"));

    m.def("int8tc_softmax_v_with_pre_corr_inplace",
          &int8tc_softmax_v_with_pre_corr_inplace,
          "In-place variant: caller supplies pre-zeroed "
          "[o_hat, o_old_unnorm, W] (for microbench / hot-loop integration).",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"),
          py::arg("v_data"),
          py::arg("v_scales"),
          py::arg("block_table"),
          py::arg("seq_lens"),
          py::arg("topk_idx"),
          py::arg("o_hat"),
          py::arg("o_old_unnorm"),
          py::arg("W_out"),
          py::arg("max_S"),
          py::arg("H_q"),
          py::arg("H_kv"),
          py::arg("block_size"),
          py::arg("top_k"));
}
