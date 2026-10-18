// WMMA-based INT2 decode kernels.  Production entrypoint:
// `fused_decode_wmma_v5_int2[_appendkv|_with_scores]` — full fused
// Q·K + online softmax + P·V with in-kernel INT2 dequant.
//
// Earlier variants (`fused_qk_wmma_bf16{,_v2,_v3}`, `fused_decode_wmma_v4`)
// are kept for A/B testing and unit tests; v5 is what decode_path uses.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace {

constexpr int D        = 128;
constexpr int BLOCK_S  = 128;
constexpr int THREADS  = 128;        // 4 warps per block
constexpr int WARP_N_TILES = 2;      // each warp handles 2 N-tiles of 16 cols
                                      // → 4 warps × 2 = 8 tiles × 16 = BLOCK_S
constexpr int STAGES   = 2;          // double-buffer for v3 cp.async pipeline

// SMEM row-stride padding for the v5 fused decode kernel.
//
// Why these specific pads: the natural row sizes alias the 32-bank SMEM
// pattern, causing massive WMMA-load bank conflicts.
//
//   q_smem / k_smem / v_smem :  D=128 bf16   = 256 B/row
//                              row stride mod 128 banks (×8 B) = 0
//                              ⇒ every row starts on bank 0 → 8-way conflict
//                              fix: +8 bf16 (16 B) → row r starts on bank
//                              (r·4) mod 32, 16 fragment rows fan across.
//   p_smem                   :  BLOCK_S_V4=64 bf16 = 128 B/row
//                              row stride mod 32-bank period = 0 → 16-way
//                              fix: +8 bf16 → 4-bank shift per row.
//   score_smem / o_smem      :  fp32 row stride = 256 B / 512 B
//                              ⇒ same bank-aligned aliasing
//                              fix: +4 fp32 (16 B).
//
// Empirical: ncu at B=4,S=16k showed 30.4M LD bank conflicts (pre-fix)
// driving 57% of dynamic instructions to short_scoreboard stall.  The
// standalone WMMA decode kernel uses the same padding pattern (see
// int2_flash_decode/csrc/int2_kv_flash_decode.cu lines 63-78).
constexpr int Q_PAD          = 8;
constexpr int Q_STRIDE_V5    = D + Q_PAD;             // bf16
constexpr int K_PAD          = 8;
constexpr int K_STRIDE_V5    = D + K_PAD;             // bf16
constexpr int V_PAD          = 8;
constexpr int V_STRIDE_V5    = D + V_PAD;             // bf16
constexpr int P_PAD          = 8;
constexpr int O_PAD          = 4;
constexpr int O_STRIDE_V5    = D + O_PAD;             // fp32
constexpr int SCORE_PAD      = 4;

// cp.async helpers.  `cp.async.ca.shared.global` (128-bit) routes the
// load through L2 cache; `ca` is fine for our tile sizes (we re-use
// each tile exactly once per block).
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned smem_int = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_int), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_group_1() {
    asm volatile("cp.async.wait_group 1;");
}
__device__ __forceinline__ void cp_async_wait_group_0() {
    asm volatile("cp.async.wait_group 0;");
}

// INT2 unpack helper — matches jetfire_vllm/kernels/intN_pack.py::_pack_int2.
// 1 byte holds 4 signed int2 values in big-endian order:
//   bits [7:6]=v0  [5:4]=v1  [3:2]=v2  [1:0]=v3,  offset-encoded (stored = q + 2).
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

__global__ void fused_qk_wmma_bf16_kernel(
    const __nv_bfloat16* __restrict__ Q,            // (B, H_q, D) bf16
    const __nv_bfloat16* __restrict__ K_bf16,       // (H_kv, num_slots, D) bf16
    const int64_t*       __restrict__ BlockTable,   // (B, max_blocks)
    const int32_t*       __restrict__ SeqLens,      // (B,)
    float*               __restrict__ Scores,       // (B, H_q, max_S) fp32
    int    max_S, int H_q, int H_kv,
    int    block_size, int block_table_stride,
    int    K_stride_h,                                 // stride of K_bf16 dim 0
    float  attn_scale)
{
    const int b        = blockIdx.x;
    const int h_q      = blockIdx.y;
    const int tile_idx = blockIdx.z;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid & 31;

    const int num_groups = H_q / H_kv;
    const int h_kv       = h_q / num_groups;

    const int S        = SeqLens[b];
    const int tile_s0  = tile_idx * BLOCK_S;
    const int tile_end = min(tile_s0 + BLOCK_S, S);
    if (tile_s0 >= tile_end) return;

    // ── SMEM ──────────────────────────────────────────────────────────────
    // Q padded to (16, D) with zeros in rows 1..15.  Plus 1 row of pad for
    // safety on store_matrix_sync (m16n16 stores 16 rows).
    __shared__ __nv_bfloat16 q_smem[16][D];
    __shared__ __nv_bfloat16 k_smem[BLOCK_S][D];
    // Scores staging for this tile (16 M-rows × BLOCK_S N-cols fp32).
    __shared__ float         score_smem[16][BLOCK_S];

    // ── Zero-pad Q rows 1..15 once ───────────────────────────────────────
    for (int i = tid; i < 15 * D; i += THREADS) {
        int r = 1 + (i / D);
        int c = i % D;
        q_smem[r][c] = __float2bfloat16(0.f);
    }
    // Copy the single real Q row (row 0).
    if (tid < D) {
        q_smem[0][tid] = Q[(size_t)b * H_q * D + h_q * D + tid];
    }

    // ── Load K tile (BLOCK_S × D) from the correct KV head + slots ───────
    // Each thread loads several (r, c) positions; K is stored (H_kv, slots, D)
    // so we index by slot from the block_table.
    const __nv_bfloat16* k_base = K_bf16 + (size_t)h_kv * K_stride_h;
    const int64_t* bt_row = BlockTable + (size_t)b * block_table_stride;
    for (int i = tid; i < BLOCK_S * D; i += THREADS) {
        int r = i / D;
        int c = i % D;
        int pos = tile_s0 + r;
        if (pos < tile_end) {
            int block_idx = pos / block_size;
            int block_off = pos % block_size;
            int64_t slot = bt_row[block_idx] * (int64_t)block_size + block_off;
            k_smem[r][c] = k_base[slot * (int64_t)D + c];
        } else {
            k_smem[r][c] = __float2bfloat16(0.f);
        }
    }
    __syncthreads();

    // ── WMMA Q·K^T: one M-tile of 16, eight N-tiles of 16 per block ──────
    // Q layout for mma: (M=16, K=D) row-major; A fragment uses row_major.
    // K layout: stored as (row=position, col=d) row-major.  To compute
    // Q · K^T, treat K as the B-operand with col_major layout of shape
    // (K=D, N=BLOCK_S) — which physically is the same (position, d) data.
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // Each warp handles WARP_N_TILES contiguous N-tiles.  4 warps × 2 tiles × 16 = 128.
    #pragma unroll
    for (int w_n = 0; w_n < WARP_N_TILES; ++w_n) {
        int n_tile = warp_id * WARP_N_TILES + w_n;   // 0..7
        int n_start = n_tile * 16;                   // 0, 16, 32, …, 112

        fill_fragment(c_frag, 0.f);

        #pragma unroll
        for (int k = 0; k < D; k += 16) {
            // A: q_smem[0..16][k..k+16] row-major stride D
            load_matrix_sync(a_frag, &q_smem[0][k], D);
            // B: K rows n_start..n_start+16 × cols k..k+16, col-major of
            // (K=16, N=16).  k_smem is (BLOCK_S=N, D=K) row-major; the
            // col-major load with leading stride D gives K·N^T per tile.
            load_matrix_sync(b_frag, &k_smem[n_start][k], D);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store the (16, 16) output tile to shared score staging.
        store_matrix_sync(&score_smem[0][n_start], c_frag, BLOCK_S, mem_row_major);
    }
    __syncthreads();

    // ── Extract row 0 (the one real query), apply scale, write to GMEM ───
    const size_t sc_base = (size_t)b * H_q * max_S + (size_t)h_q * max_S + tile_s0;
    for (int n = tid; n < BLOCK_S; n += THREADS) {
        int pos = tile_s0 + n;
        if (pos < tile_end) {
            Scores[sc_base + n] = score_smem[0][n] * attn_scale;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// V2: restructured grid for real WMMA utilisation.
//
// Grid:  (B, H_kv, num_splits)
// Block: one (batch, KV-head, split-of-S) triple per block.  Each block
//        processes num_groups=H_q/H_kv Q heads that share the same KV
//        head (typically 4 for LLaMA-3.1-8B).  Q is padded from M=4 to
//        WMMA M=16 with zero rows → 25% TC M-utilisation (vs 6% in v1).
//        Inside the block we loop over the split's BLOCK_S-sized tiles,
//        sharing the single Q load across all tiles and loading K fresh
//        per tile.  This amortises kernel-launch overhead (4× fewer
//        blocks than v1) and reuses the Q fragments 4× across tiles.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_qk_wmma_bf16_v2_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_bf16,
    const int64_t*       __restrict__ BlockTable,
    const int32_t*       __restrict__ SeqLens,
    float*               __restrict__ Scores,
    int    max_S, int H_q, int H_kv,
    int    block_size, int block_table_stride,
    int    K_stride_h,
    int    num_splits,
    float  attn_scale)
{
    const int b         = blockIdx.x;
    const int h_kv      = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 32;

    const int num_groups = H_q / H_kv;           // 4 for LLaMA-3.1-8B
    const int S          = SeqLens[b];
    const int split_size = (max_S + num_splits - 1) / num_splits;
    const int split_s0   = split_idx * split_size;
    const int split_end  = min(split_s0 + split_size, S);
    if (split_s0 >= split_end) return;

    // SMEM
    __shared__ __nv_bfloat16 q_smem[16][D];
    __shared__ __nv_bfloat16 k_smem[BLOCK_S][D];
    __shared__ float         score_smem[16][BLOCK_S];

    // One-time: zero rows 4..15 of q_smem (padding).
    for (int i = tid; i < (16 - 4) * D; i += THREADS) {
        int r = 4 + i / D;
        int c = i % D;
        q_smem[r][c] = __float2bfloat16(0.f);
    }
    // Load num_groups Q rows (h_q = h_kv*num_groups + g for g=0..num_groups-1).
    // At num_groups=4 and D=128 that's 4 × 128 = 512 loads; 128 threads do 4 each.
    for (int i = tid; i < num_groups * D; i += THREADS) {
        int r = i / D;
        int c = i % D;
        int h_q = h_kv * num_groups + r;
        q_smem[r][c] = Q[(size_t)b * H_q * D + (size_t)h_q * D + c];
    }

    const __nv_bfloat16* k_base = K_bf16 + (size_t)h_kv * K_stride_h;
    const int64_t*       bt_row = BlockTable + (size_t)b * block_table_stride;

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // ── Tile loop within this split ──────────────────────────────────────
    for (int tile_s0 = split_s0; tile_s0 < split_end; tile_s0 += BLOCK_S) {
        const int tile_end = min(tile_s0 + BLOCK_S, split_end);

        // Load K for this tile.
        for (int i = tid; i < BLOCK_S * D; i += THREADS) {
            int r = i / D;
            int c = i % D;
            int pos = tile_s0 + r;
            if (pos < tile_end) {
                int block_idx = pos / block_size;
                int block_off = pos % block_size;
                int64_t slot = bt_row[block_idx] * (int64_t)block_size + block_off;
                k_smem[r][c] = k_base[slot * (int64_t)D + c];
            } else {
                k_smem[r][c] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // WMMA: (M=16 padded from num_groups=4, N=BLOCK_S=128, K=D=128).
        // 4 warps × WARP_N_TILES=2 N-tiles of 16 = 8 tiles = 128 cols.
        #pragma unroll
        for (int w_n = 0; w_n < WARP_N_TILES; ++w_n) {
            int n_tile  = warp_id * WARP_N_TILES + w_n;
            int n_start = n_tile * 16;

            fill_fragment(c_frag, 0.f);
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                load_matrix_sync(a_frag, &q_smem[0][k], D);
                load_matrix_sync(b_frag, &k_smem[n_start][k], D);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            store_matrix_sync(&score_smem[0][n_start], c_frag, BLOCK_S, mem_row_major);
        }
        __syncthreads();

        // Write num_groups real rows of scores to GMEM (4 × BLOCK_S writes
        // per tile → 128 threads handle it in a few iterations).
        for (int i = tid; i < num_groups * BLOCK_S; i += THREADS) {
            int r = i / BLOCK_S;
            int n = i % BLOCK_S;
            int pos = tile_s0 + n;
            if (pos < tile_end) {
                int h_q = h_kv * num_groups + r;
                size_t sc_off = (size_t)b * H_q * max_S
                              + (size_t)h_q * max_S + (size_t)pos;
                Scores[sc_off] = score_smem[r][n] * attn_scale;
            }
        }
        __syncthreads();
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// V3: v2 + cp.async 2-stage pipeline for K loads.
//
// Same grid and M=4-padded-to-16 structure as v2.  The only difference is
// K loading: instead of a synchronous load before each tile's WMMA, we
// fire cp.async for the NEXT tile's K while the current tile's WMMA is
// in flight, then wait on the older group before consuming.  With enough
// tiles per split the HBM → SMEM traffic is fully hidden behind the
// mma.sync compute, so the kernel becomes compute-bound (or bank-conflict
// bound) instead of HBM-wait-bound.
// ─────────────────────────────────────────────────────────────────────────────
// v3 uses a smaller tile (64 instead of 128) to fit 2 stages of K in
// static SMEM alongside q_smem + score_smem.  Totals at BLOCK_S_V3=64:
//   q_smem       : 16 × 128 × 2 =  4 KB
//   k_smem[2]    : 2  ×  64 × 128 × 2 = 32 KB
//   score_smem   : 16 ×  64 × 4 =  4 KB
//   ── total                    = 40 KB  (< 48 KB static limit)
constexpr int BLOCK_S_V3     = 64;
constexpr int WARP_N_TILES_V3 = 1;      // 4 warps × 1 × 16 = 64 = BLOCK_S_V3

__global__ void fused_qk_wmma_bf16_v3_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_bf16,
    const int64_t*       __restrict__ BlockTable,
    const int32_t*       __restrict__ SeqLens,
    float*               __restrict__ Scores,
    int    max_S, int H_q, int H_kv,
    int    block_size, int block_table_stride,
    int    K_stride_h,
    int    num_splits,
    float  attn_scale)
{
    const int b         = blockIdx.x;
    const int h_kv      = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 32;

    const int num_groups = H_q / H_kv;
    const int S          = SeqLens[b];
    const int split_size = (max_S + num_splits - 1) / num_splits;
    const int split_s0   = split_idx * split_size;
    const int split_end  = min(split_s0 + split_size, S);
    if (split_s0 >= split_end) return;

    __shared__ __nv_bfloat16 q_smem[16][D];
    __shared__ __nv_bfloat16 k_smem[STAGES][BLOCK_S_V3][D];
    __shared__ float         score_smem[16][BLOCK_S_V3];

    // Pad rows 4..15 of q_smem.
    for (int i = tid; i < (16 - 4) * D; i += THREADS) {
        int r = 4 + i / D;
        int c = i % D;
        q_smem[r][c] = __float2bfloat16(0.f);
    }
    for (int i = tid; i < num_groups * D; i += THREADS) {
        int r = i / D;
        int c = i % D;
        int h_q = h_kv * num_groups + r;
        q_smem[r][c] = Q[(size_t)b * H_q * D + (size_t)h_q * D + c];
    }

    const __nv_bfloat16* k_base = K_bf16 + (size_t)h_kv * K_stride_h;
    const int64_t*       bt_row = BlockTable + (size_t)b * block_table_stride;

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // ── cp.async load helper: each of THREADS threads issues the loads
    // for its assigned row of the K tile.  Row = tid (128 rows × 128 threads
    // → 1 row per thread), and 16 × 16-byte copies fill the 256-byte row.
    // Each thread loads ONE row of K; BLOCK_S_V3=64 rows < 128 threads,
    // so threads ≥ 64 are idle during loads (they still issue commit to
    // stay in lockstep with cp.async.wait_group accounting).
    auto issue_load = [&](int stage, int tile_s0, int tile_end) {
        if (tid < BLOCK_S_V3) {
            int pos = tile_s0 + tid;
            if (pos < tile_end) {
                int block_idx = pos / block_size;
                int block_off = pos % block_size;
                int64_t slot = bt_row[block_idx] * (int64_t)block_size + block_off;
                const __nv_bfloat16* src = k_base + slot * (int64_t)D;
                __nv_bfloat16* dst = &k_smem[stage][tid][0];
                // D=128 bf16 = 256 B = 16 × cp.async.16
                #pragma unroll
                for (int i = 0; i < D * 2 / 16; ++i) {
                    cp_async16((uint8_t*)dst + i * 16,
                               (const uint8_t*)src + i * 16);
                }
            } else {
                // Out-of-range rows: synchronous zero-fill (rare).
                #pragma unroll
                for (int i = 0; i < D; ++i) {
                    k_smem[stage][tid][i] = __float2bfloat16(0.f);
                }
            }
        }
        cp_async_commit();
    };

    const int num_tiles = (split_end - split_s0 + BLOCK_S_V3 - 1) / BLOCK_S_V3;

    // Prologue: issue the first tile's load.
    int tile_s0_pref = split_s0;
    int tile_end_pref = min(tile_s0_pref + BLOCK_S_V3, split_end);
    issue_load(0, tile_s0_pref, tile_end_pref);
    int loaded = 1;

    for (int t = 0; t < num_tiles; ++t) {
        const int stage = t % STAGES;
        const int tile_s0  = split_s0 + t * BLOCK_S_V3;
        const int tile_end = min(tile_s0 + BLOCK_S_V3, split_end);

        if (loaded < num_tiles) {
            int next_s0 = split_s0 + loaded * BLOCK_S_V3;
            int next_end = min(next_s0 + BLOCK_S_V3, split_end);
            issue_load(loaded % STAGES, next_s0, next_end);
            loaded++;
        }

        if (loaded > t + 1) {
            cp_async_wait_group_1();
        } else {
            cp_async_wait_group_0();
        }
        __syncthreads();

        // WMMA Q·K^T, one M-tile × WARP_N_TILES_V3 N-tiles per warp.
        #pragma unroll
        for (int w_n = 0; w_n < WARP_N_TILES_V3; ++w_n) {
            int n_tile  = warp_id * WARP_N_TILES_V3 + w_n;
            int n_start = n_tile * 16;

            fill_fragment(c_frag, 0.f);
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                load_matrix_sync(a_frag, &q_smem[0][k], D);
                load_matrix_sync(b_frag, &k_smem[stage][n_start][k], D);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            store_matrix_sync(&score_smem[0][n_start], c_frag,
                              BLOCK_S_V3, mem_row_major);
        }
        __syncthreads();

        // Write num_groups real rows to GMEM.
        for (int i = tid; i < num_groups * BLOCK_S_V3; i += THREADS) {
            int r = i / BLOCK_S_V3;
            int n = i % BLOCK_S_V3;
            int pos = tile_s0 + n;
            if (pos < tile_end) {
                int h_q = h_kv * num_groups + r;
                size_t sc_off = (size_t)b * H_q * max_S
                              + (size_t)h_q * max_S + (size_t)pos;
                Scores[sc_off] = score_smem[r][n] * attn_scale;
            }
        }
        // No sync here — next iter's first action is cp_async_wait which
        // won't race with stores above (different smem regions).
    }
    // Drain any remaining in-flight loads at end of block.
    cp_async_wait_group_0();
}


// ─────────────────────────────────────────────────────────────────────────────
// V4: full fused decode — Q·K^T + online softmax + P·V + epilogue, all in
// Tensor Cores on pre-dequantized bf16 K/V.  Returns o_hat (B, H_q, D) fp32.
//
// Numerics: bf16 matmuls with fp32 accumulators (inside WMMA fragments) for
// both Q·K and P·V; softmax runs fp32, but P is cast to bf16 before the
// P·V mma (precision loss is small — typical flash-attention v2/v3 pattern).
//
// SMEM layout (dynamic; allocated by caller via `cudaFuncSetAttribute`):
//   q_smem       (16, D) bf16
//   k_smem[2]    (STAGES, BLOCK_S, D) bf16   — cp.async double-buffered
//   v_smem[2]    (STAGES, BLOCK_S, D) bf16   — cp.async double-buffered
//   score_smem   (16, BLOCK_S) fp32          — Q·K output, softmax inputs
//   p_smem       (16, BLOCK_S) bf16          — softmax output, P·V input
//   o_smem       (16, D) fp32                — running o accumulator
//   m_smem[16], l_smem[16]                    — per-row running max / sum
//
// Grid:  (B, H_kv)        — one split per (batch, KV head), whole S in one go.
// Block: 128 threads (4 warps).  M = num_groups Q heads padded to 16.
// Split-K (combine kernel across splits) deferred to Step 6.
// ─────────────────────────────────────────────────────────────────────────────
constexpr int BLOCK_S_V4 = 64;

__global__ void fused_decode_wmma_v4_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_bf16,
    const __nv_bfloat16* __restrict__ V_bf16,
    const int64_t*       __restrict__ BlockTable,
    const int32_t*       __restrict__ SeqLens,
    float*               __restrict__ O_partial,      // (B, H_q, num_splits, D)
    float*               __restrict__ M_per_split,    // (B, H_q, num_splits)
    float*               __restrict__ L_per_split,    // (B, H_q, num_splits)
    int    max_S, int H_q, int H_kv,
    int    block_size, int block_table_stride,
    int    K_stride_h,
    int    num_splits,
    float  attn_scale)
{
    const int b         = blockIdx.x;
    const int h_kv      = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 32;

    const int num_groups = H_q / H_kv;
    const int S          = SeqLens[b];
    const int split_size = (max_S + num_splits - 1) / num_splits;
    const int split_s0   = split_idx * split_size;
    const int split_end  = min(split_s0 + split_size, S);

    // ── Dynamic SMEM layout ──────────────────────────────────────────────
    extern __shared__ __align__(16) uint8_t dyn_smem[];
    size_t off = 0;
    auto q_smem     = reinterpret_cast<__nv_bfloat16(*)[D]>(&dyn_smem[off]);
    off += 16 * D * sizeof(__nv_bfloat16);                     // 4 KB
    auto k_smem     = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S_V4][D]>(&dyn_smem[off]);
    off += STAGES * BLOCK_S_V4 * D * sizeof(__nv_bfloat16);    // 32 KB
    auto v_smem     = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S_V4][D]>(&dyn_smem[off]);
    off += STAGES * BLOCK_S_V4 * D * sizeof(__nv_bfloat16);    // 32 KB
    auto score_smem = reinterpret_cast<float(*)[BLOCK_S_V4]>(&dyn_smem[off]);
    off += 16 * BLOCK_S_V4 * sizeof(float);                    //  4 KB
    auto p_smem     = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S_V4]>(&dyn_smem[off]);
    off += 16 * BLOCK_S_V4 * sizeof(__nv_bfloat16);            //  2 KB
    auto o_smem     = reinterpret_cast<float(*)[D]>(&dyn_smem[off]);
    off += 16 * D * sizeof(float);                              //  8 KB
    float* m_smem   = reinterpret_cast<float*>(&dyn_smem[off]);
    off += 16 * sizeof(float);                                  // 64 B
    float* l_smem   = reinterpret_cast<float*>(&dyn_smem[off]);
    off += 16 * sizeof(float);

    // ── Init Q (4 real rows + 12 zero pad) ───────────────────────────────
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
    // Init running state.
    if (tid < 16) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.0f;
    }
    // Zero o_smem.
    for (int i = tid; i < 16 * D; i += THREADS) {
        int r = i / D;
        int c = i % D;
        o_smem[r][c] = 0.0f;
    }

    const __nv_bfloat16* k_base = K_bf16 + (size_t)h_kv * K_stride_h;
    const __nv_bfloat16* v_base = V_bf16 + (size_t)h_kv * K_stride_h;
    const int64_t*       bt_row = BlockTable + (size_t)b * block_table_stride;

    // Issue-load helper: copies one tile's K AND V for a given stage.
    // Each thread owns one row (tid 0..63 for K, tid 64..127 for V).
    auto issue_load = [&](int stage, int tile_s0, int tile_end) {
        int tt = tid & (BLOCK_S_V4 - 1);  // 0..63
        bool load_v = (tid >= BLOCK_S_V4);
        __nv_bfloat16 (*dst_smem)[BLOCK_S_V4][D] = load_v ? v_smem : k_smem;
        const __nv_bfloat16* gmem_base = load_v ? v_base : k_base;

        int pos = tile_s0 + tt;
        if (pos < tile_end) {
            int block_idx = pos / block_size;
            int block_off = pos % block_size;
            int64_t slot = bt_row[block_idx] * (int64_t)block_size + block_off;
            const __nv_bfloat16* src = gmem_base + slot * (int64_t)D;
            __nv_bfloat16* dst = &dst_smem[stage][tt][0];
            #pragma unroll
            for (int i = 0; i < D * 2 / 16; ++i) {
                cp_async16((uint8_t*)dst + i * 16,
                           (const uint8_t*)src + i * 16);
            }
        } else {
            // Synchronous zero-fill for out-of-range rows.
            #pragma unroll
            for (int i = 0; i < D; ++i) {
                dst_smem[stage][tt][i] = __float2bfloat16(0.f);
            }
        }
        cp_async_commit();
    };

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>  a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major>  b_qk_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major>  b_pv_frag;
    fragment<accumulator, 16, 16, 16, float> qk_acc;
    fragment<accumulator, 16, 16, 16, float> pv_acc;

    // Early exit for empty splits (writes sentinels so combine can detect).
    const int num_tiles = (split_end - split_s0 + BLOCK_S_V4 - 1) / BLOCK_S_V4;
    if (split_s0 >= split_end || num_tiles <= 0) {
        // Write m=-inf, l=0, o_partial=0 for all real rows so combine can
        // skip this split safely.
        for (int i = tid; i < num_groups; i += THREADS) {
            int h_q = h_kv * num_groups + i;
            size_t ml_off = (size_t)b * H_q * num_splits
                          + (size_t)h_q * num_splits + split_idx;
            M_per_split[ml_off] = -INFINITY;
            L_per_split[ml_off] = 0.0f;
        }
        for (int i = tid; i < num_groups * D; i += THREADS) {
            int r = i / D, c = i % D;
            int h_q = h_kv * num_groups + r;
            size_t op_off = (size_t)b * H_q * num_splits * D
                          + (size_t)h_q * num_splits * D
                          + (size_t)split_idx * D + c;
            O_partial[op_off] = 0.0f;
        }
        return;
    }

    // Prologue: prefetch tile 0 of this split.
    int loaded = 0;
    {
        int tile_s0_first = split_s0;
        int tile_end_first = min(tile_s0_first + BLOCK_S_V4, split_end);
        issue_load(0, tile_s0_first, tile_end_first);
        loaded = 1;
    }

    // ── Main tile loop ───────────────────────────────────────────────────
    for (int t = 0; t < num_tiles; ++t) {
        const int stage     = t % STAGES;
        const int tile_s0   = split_s0 + t * BLOCK_S_V4;
        const int tile_end  = min(tile_s0 + BLOCK_S_V4, split_end);
        const int tile_rows = tile_end - tile_s0;

        // Issue next tile.
        if (loaded < num_tiles) {
            int next_s0 = split_s0 + loaded * BLOCK_S_V4;
            int next_end = min(next_s0 + BLOCK_S_V4, split_end);
            issue_load(loaded % STAGES, next_s0, next_end);
            loaded++;
        }
        if (loaded > t + 1) cp_async_wait_group_1();
        else                cp_async_wait_group_0();
        __syncthreads();

        // ─── Phase A: Q·K^T via WMMA, store to score_smem ────────────────
        // BLOCK_S_V4=64 → 4 N-tiles of 16.  4 warps × 1 N-tile per warp.
        {
            int n_tile  = warp_id;                 // 0..3
            int n_start = n_tile * 16;

            fill_fragment(qk_acc, 0.f);
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                load_matrix_sync(a_frag, &q_smem[0][k], D);
                load_matrix_sync(b_qk_frag, &k_smem[stage][n_start][k], D);
                mma_sync(qk_acc, a_frag, b_qk_frag, qk_acc);
            }
            store_matrix_sync(&score_smem[0][n_start], qk_acc,
                              BLOCK_S_V4, mem_row_major);
        }
        __syncthreads();

        // ─── Phase B: row-wise online softmax on score_smem ──────────────
        // Always zero-fill pad rows (num_groups..15) of p_smem first.  All
        // 4 warps cooperate: the (16 - num_groups) * BLOCK_S_V4 cells are
        // divided among THREADS=128 threads.
        for (int i = tid; i < (16 - num_groups) * BLOCK_S_V4; i += THREADS) {
            int r = num_groups + i / BLOCK_S_V4;
            int c = i % BLOCK_S_V4;
            p_smem[r][c] = __float2bfloat16(0.f);
        }

        // 4 real rows — one warp per row (works when num_groups == WARPS = 4).
        if (warp_id < num_groups) {
            int r = warp_id;
            float old_m = m_smem[r];
            float old_l = l_smem[r];

            float local_max = -INFINITY;
            int lane = tid & 31;
            for (int c = lane; c < tile_rows; c += 32) {
                float v = score_smem[r][c] * attn_scale;
                local_max = fmaxf(local_max, v);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                local_max = fmaxf(local_max,
                                   __shfl_xor_sync(0xffffffffu, local_max, off));
            }
            float new_m = fmaxf(old_m, local_max);
            float alpha = expf(old_m - new_m);

            float row_sum = 0.0f;
            for (int c = lane; c < BLOCK_S_V4; c += 32) {
                float v = (c < tile_rows)
                              ? __expf(score_smem[r][c] * attn_scale - new_m)
                              : 0.0f;
                p_smem[r][c] = __float2bfloat16(v);
                row_sum += v;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                row_sum += __shfl_xor_sync(0xffffffffu, row_sum, off);
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

        // ─── Phase C: P·V WMMA into o_smem ────────────────────────────────
        // M=16, N=D=128 (8 N-tiles), K=BLOCK_S_V4=64 (4 K-tiles).
        // 4 warps × 2 N-tiles per warp = 8 N-tiles total.  Each warp writes
        // its (16, 16) result to a DISJOINT region of score_smem (at column
        // warp_id*16), then adds its own tile into o_smem at the correct
        // N-start.  No cross-warp write conflicts.
        #pragma unroll
        for (int w_n = 0; w_n < 2; ++w_n) {
            int n_tile  = warp_id * 2 + w_n;       // 0..7
            int n_start = n_tile * 16;

            fill_fragment(pv_acc, 0.f);
            #pragma unroll
            for (int k = 0; k < BLOCK_S_V4; k += 16) {
                load_matrix_sync(a_frag, &p_smem[0][k], BLOCK_S_V4);
                load_matrix_sync(b_pv_frag, &v_smem[stage][k][n_start], D);
                mma_sync(pv_acc, a_frag, b_pv_frag, pv_acc);
            }
            // Each warp writes to and reads from its OWN (col slice) of
            // score_smem (cols warp_id*16..warp_id*16+16) and writes to
            // its OWN col slice of o_smem (cols n_start..n_start+16, also
            // per-warp distinct).  No cross-warp race → no __syncthreads
            // needed between store_matrix_sync and the element-wise add.
            store_matrix_sync(&score_smem[0][warp_id * 16], pv_acc,
                              BLOCK_S_V4, mem_row_major);
            // store_matrix_sync is warp-collective; subsequent lane reads
            // of the same SMEM region see the written values without a
            // fence (intra-warp order).
            int lane = tid & 31;
            #pragma unroll
            for (int i = lane; i < 16 * 16; i += 32) {
                int r = i / 16;
                int c = i % 16;
                o_smem[r][n_start + c] += score_smem[r][warp_id * 16 + c];
            }
        }
        // One block-wide sync at end of Phase C, before next tile's Phase A
        // overwrites k_smem and score_smem.
        __syncthreads();
    }

    // ── Epilogue: write per-split UNNORMALIZED partials (o without /l)
    // and per-split m, l.  The combine kernel does the cross-split merge.
    __syncthreads();
    for (int i = tid; i < num_groups * D; i += THREADS) {
        int r = i / D;
        int c = i % D;
        int h_q = h_kv * num_groups + r;
        size_t op_off = (size_t)b * H_q * num_splits * D
                      + (size_t)h_q * num_splits * D
                      + (size_t)split_idx * D + c;
        O_partial[op_off] = o_smem[r][c];
    }
    for (int i = tid; i < num_groups; i += THREADS) {
        int h_q = h_kv * num_groups + i;
        size_t ml_off = (size_t)b * H_q * num_splits
                      + (size_t)h_q * num_splits + split_idx;
        M_per_split[ml_off] = m_smem[i];
        L_per_split[ml_off] = l_smem[i];
    }
    cp_async_wait_group_0();
}


// ─────────────────────────────────────────────────────────────────────────────
// V5: v4 + in-kernel INT2 dequant.  Reads INT2 packed K (asymmetric) +
// INT2 packed V (symmetric) + scales + zp directly from HBM via cp.async,
// stores packed + scales in SMEM, unpacks to bf16 in a second SMEM region
// just before each tile's WMMA.  Same fused decode as v4 otherwise.
// ─────────────────────────────────────────────────────────────────────────────
constexpr int D_QUARTER = D / 4;

// NS_EQ_1: when true, the launch has num_splits == 1 — the kernel skips the
// per-split partial materialization and writes the final normalized o_hat
// directly, avoiding the combine pass entirely.  Host dispatches this when
// B * H_kv already saturates the SM count, so splits would be pure tax.
template <bool NS_EQ_1>
__global__ void fused_decode_wmma_v5_int2_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const uint8_t*       __restrict__ K_data,       // (H_kv, num_slots, D/4) uint8
    const __nv_bfloat16* __restrict__ K_scales,     // (H_kv, num_slots) bf16
    const __nv_bfloat16* __restrict__ K_zp,         // (H_kv, num_slots) bf16
    const uint8_t*       __restrict__ V_data,       // (H_kv, num_slots, D/4) uint8
    const __nv_bfloat16* __restrict__ V_scales,     // (H_kv, num_slots) bf16
    const int64_t*       __restrict__ BlockTable,
    const int32_t*       __restrict__ SeqLens,
    float*               __restrict__ O_partial,    // unused if NS_EQ_1
    float*               __restrict__ M_per_split,  // unused if NS_EQ_1
    float*               __restrict__ L_per_split,  // unused if NS_EQ_1
    float*               __restrict__ O_hat,        // used only if NS_EQ_1
    float*               __restrict__ Scores,       // (B, H_q, max_S) fp32 or nullptr
    int    max_S, int H_q, int H_kv,
    int    block_size, int block_table_stride,
    int    K_data_stride_h,                          // stride of K_data dim 0
    int    K_scales_stride_h,                        // stride of K_scales dim 0
    int    num_splits,
    float  attn_scale)
{
    const int b         = blockIdx.x;
    const int h_kv      = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 32;

    const int num_groups = H_q / H_kv;
    const int S          = SeqLens[b];
    const int split_size = (max_S + num_splits - 1) / num_splits;
    const int split_s0   = split_idx * split_size;
    const int split_end  = min(split_s0 + split_size, S);

    // ── Dynamic SMEM layout ──────────────────────────────────────────────
    // Padded strides break SMEM bank aliasing on WMMA fragment loads — see
    // the constants block at the top of this TU for the diagnostic + math.
    constexpr int P_STRIDE_V5     = BLOCK_S_V4 + P_PAD;       // bf16
    constexpr int SCORE_STRIDE_V5 = BLOCK_S_V4 + SCORE_PAD;   // fp32
    extern __shared__ __align__(16) uint8_t dyn_smem5[];
    size_t off = 0;
    auto q_smem     = reinterpret_cast<__nv_bfloat16(*)[Q_STRIDE_V5]>(&dyn_smem5[off]);
    off += 16 * Q_STRIDE_V5 * sizeof(__nv_bfloat16);
    // INT2 packed double-buffered scratch (+ scales/zp).
    auto k_pack     = reinterpret_cast<uint8_t(*)[BLOCK_S_V4][D_QUARTER]>(&dyn_smem5[off]);
    off += STAGES * BLOCK_S_V4 * D_QUARTER;                        //  4 KB
    auto v_pack     = reinterpret_cast<uint8_t(*)[BLOCK_S_V4][D_QUARTER]>(&dyn_smem5[off]);
    off += STAGES * BLOCK_S_V4 * D_QUARTER;                        //  4 KB
    auto k_scale    = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S_V4]>(&dyn_smem5[off]);
    off += STAGES * BLOCK_S_V4 * sizeof(__nv_bfloat16);            // 256 B
    auto k_zp       = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S_V4]>(&dyn_smem5[off]);
    off += STAGES * BLOCK_S_V4 * sizeof(__nv_bfloat16);            // 256 B
    auto v_scale    = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S_V4]>(&dyn_smem5[off]);
    off += STAGES * BLOCK_S_V4 * sizeof(__nv_bfloat16);            // 256 B
    // bf16 unpacked tiles (single-buffered — consumed immediately per tile).
    auto k_smem     = reinterpret_cast<__nv_bfloat16(*)[K_STRIDE_V5]>(&dyn_smem5[off]);
    off += BLOCK_S_V4 * K_STRIDE_V5 * sizeof(__nv_bfloat16);
    auto v_smem     = reinterpret_cast<__nv_bfloat16(*)[V_STRIDE_V5]>(&dyn_smem5[off]);
    off += BLOCK_S_V4 * V_STRIDE_V5 * sizeof(__nv_bfloat16);
    // Output / reduction scratch.
    auto score_smem = reinterpret_cast<float(*)[SCORE_STRIDE_V5]>(&dyn_smem5[off]);
    off += 16 * SCORE_STRIDE_V5 * sizeof(float);
    auto p_smem     = reinterpret_cast<__nv_bfloat16(*)[P_STRIDE_V5]>(&dyn_smem5[off]);
    off += 16 * P_STRIDE_V5 * sizeof(__nv_bfloat16);
    auto o_smem     = reinterpret_cast<float(*)[O_STRIDE_V5]>(&dyn_smem5[off]);
    off += 16 * O_STRIDE_V5 * sizeof(float);
    float* m_smem   = reinterpret_cast<float*>(&dyn_smem5[off]);
    off += 16 * sizeof(float);
    float* l_smem   = reinterpret_cast<float*>(&dyn_smem5[off]);
    off += 16 * sizeof(float);

    // ── Q init ───────────────────────────────────────────────────────────
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

    const uint8_t*       k_pack_base = K_data   + (size_t)h_kv * K_data_stride_h;
    const uint8_t*       v_pack_base = V_data   + (size_t)h_kv * K_data_stride_h;
    const __nv_bfloat16* ks_base     = K_scales + (size_t)h_kv * K_scales_stride_h;
    const __nv_bfloat16* kz_base     = K_zp     + (size_t)h_kv * K_scales_stride_h;
    const __nv_bfloat16* vs_base     = V_scales + (size_t)h_kv * K_scales_stride_h;
    const int64_t*       bt_row      = BlockTable + (size_t)b * block_table_stride;

    const int num_tiles = (split_end - split_s0 + BLOCK_S_V4 - 1) / BLOCK_S_V4;
    if (split_s0 >= split_end || num_tiles <= 0) {
        if constexpr (NS_EQ_1) {
            // ns=1 collapses the full sequence into one CTA per (b, h_kv), so
            // an empty-split here means S=0 — write zero o_hat and exit.
            for (int i = tid; i < num_groups * D; i += THREADS) {
                int r = i / D, c = i % D;
                int h_q = h_kv * num_groups + r;
                size_t off = (size_t)b * H_q * D + (size_t)h_q * D + c;
                O_hat[off] = 0.0f;
            }
        } else {
            for (int i = tid; i < num_groups; i += THREADS) {
                int h_q = h_kv * num_groups + i;
                size_t ml_off = (size_t)b * H_q * num_splits
                              + (size_t)h_q * num_splits + split_idx;
                M_per_split[ml_off] = -INFINITY;
                L_per_split[ml_off] = 0.0f;
            }
            for (int i = tid; i < num_groups * D; i += THREADS) {
                int r = i / D, c = i % D;
                int h_q = h_kv * num_groups + r;
                size_t op_off = (size_t)b * H_q * num_splits * D
                              + (size_t)h_q * num_splits * D
                              + (size_t)split_idx * D + c;
                O_partial[op_off] = 0.0f;
            }
        }
        return;
    }

    // Load helper: async-loads one tile's K_packed + V_packed + K/V scales
    // + K zp into the stage's SMEM slots.  Packed tensors are 32 B per row
    // (D_QUARTER = D/4 = 32) → 2 cp.async.16 per thread per row.
    auto issue_load = [&](int stage, int tile_s0, int tile_end) {
        int tt = tid & (BLOCK_S_V4 - 1);  // 0..63; threads >= 64 handle V.
        bool load_v = (tid >= BLOCK_S_V4);
        int pos = tile_s0 + tt;
        int64_t slot = (pos < tile_end)
                         ? (bt_row[pos / block_size] * (int64_t)block_size + pos % block_size)
                         : -1;

        if (!load_v) {
            // Load K packed + K scale + K zp.
            if (slot >= 0) {
                const uint8_t* src = k_pack_base + slot * (int64_t)D_QUARTER;
                uint8_t*       dst = &k_pack[stage][tt][0];
                cp_async16(dst,      src);            // bytes 0..15
                cp_async16(dst + 16, src + 16);       // bytes 16..31
                k_scale[stage][tt] = ks_base[slot];
                k_zp   [stage][tt] = kz_base[slot];
            } else {
                uint8_t* dst = &k_pack[stage][tt][0];
                #pragma unroll
                for (int i = 0; i < D_QUARTER; ++i) dst[i] = 0;
                k_scale[stage][tt] = __float2bfloat16(0.f);
                k_zp   [stage][tt] = __float2bfloat16(0.f);
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

    // Unpack helper: converts INT2 packed → bf16 in k_smem/v_smem.
    // Each thread unpacks one (row, byte) slot into 4 bf16 values.
    // Total per tile per tensor: BLOCK_S_V4 * D_QUARTER = 64 * 32 = 2048 threads-of-work;
    // with 128 threads, 16 iterations.
    auto unpack_tile = [&](int stage) {
        for (int i = tid; i < BLOCK_S_V4 * D_QUARTER; i += THREADS) {
            int r = i / D_QUARTER;
            int bx = i % D_QUARTER;
            uint8_t pk = k_pack[stage][r][bx];
            float  ks = __bfloat162float(k_scale[stage][r]);
            float  kz = __bfloat162float(k_zp   [stage][r]);
            __nv_bfloat16 v0, v1, v2, v3;
            int2_unpack_bf16(pk, ks, kz, v0, v1, v2, v3);
            k_smem[r][4*bx + 0] = v0;
            k_smem[r][4*bx + 1] = v1;
            k_smem[r][4*bx + 2] = v2;
            k_smem[r][4*bx + 3] = v3;
            // V (symmetric, no zp).
            uint8_t pv = v_pack[stage][r][bx];
            float  vs = __bfloat162float(v_scale[stage][r]);
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

    // Prologue.
    {
        int tile_end = min(split_s0 + BLOCK_S_V4, split_end);
        issue_load(0, split_s0, tile_end);
    }
    int loaded = 1;

    // Main tile loop.
    for (int t = 0; t < num_tiles; ++t) {
        const int stage     = t % STAGES;
        const int tile_s0   = split_s0 + t * BLOCK_S_V4;
        const int tile_end  = min(tile_s0 + BLOCK_S_V4, split_end);
        const int tile_rows = tile_end - tile_s0;

        if (loaded < num_tiles) {
            int next_s0 = split_s0 + loaded * BLOCK_S_V4;
            int next_end = min(next_s0 + BLOCK_S_V4, split_end);
            issue_load(loaded % STAGES, next_s0, next_end);
            loaded++;
        }
        if (loaded > t + 1) cp_async_wait_group_1();
        else                cp_async_wait_group_0();
        __syncthreads();

        // Unpack INT2 packed → bf16 in k_smem / v_smem for this tile.
        unpack_tile(stage);
        __syncthreads();

        // Phase A: Q·K^T
        {
            int n_tile  = warp_id;
            int n_start = n_tile * 16;
            fill_fragment(qk_acc, 0.f);
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                load_matrix_sync(a_frag, &q_smem[0][k], Q_STRIDE_V5);
                load_matrix_sync(b_qk_frag, &k_smem[n_start][k], K_STRIDE_V5);
                mma_sync(qk_acc, a_frag, b_qk_frag, qk_acc);
            }
            store_matrix_sync(&score_smem[0][n_start], qk_acc,
                              SCORE_STRIDE_V5, mem_row_major);
        }
        __syncthreads();

        // Optional: emit raw QK scores (× attn_scale) to global memory so the
        // correction path can run topk + softmax outside.  Each split CTA
        // owns the slice [tile_s0, tile_end) of the (B, H_q, max_S) tensor —
        // no cross-CTA aliasing.  Only num_groups rows × tile_rows cols are
        // valid in score_smem; padded entries are not touched.
        if (Scores != nullptr) {
            const int valid = num_groups * tile_rows;
            for (int i = tid; i < valid; i += THREADS) {
                const int r = i / tile_rows;
                const int c = i % tile_rows;
                const int h_q = h_kv * num_groups + r;
                const int s_global = tile_s0 + c;
                Scores[((size_t)b * H_q + h_q) * max_S + s_global] =
                    score_smem[r][c] * attn_scale;
            }
        }

        // Phase B: softmax (same as v4).
        for (int i = tid; i < (16 - num_groups) * BLOCK_S_V4; i += THREADS) {
            int r = num_groups + i / BLOCK_S_V4;
            int c = i % BLOCK_S_V4;
            p_smem[r][c] = __float2bfloat16(0.f);
        }
        if (warp_id < num_groups) {
            int r = warp_id;
            float old_m = m_smem[r];
            float old_l = l_smem[r];
            float local_max = -INFINITY;
            int lane = tid & 31;
            for (int c = lane; c < tile_rows; c += 32) {
                float v = score_smem[r][c] * attn_scale;
                local_max = fmaxf(local_max, v);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                local_max = fmaxf(local_max,
                                   __shfl_xor_sync(0xffffffffu, local_max, off));
            }
            float new_m = fmaxf(old_m, local_max);
            float alpha = expf(old_m - new_m);
            float row_sum = 0.0f;
            for (int c = lane; c < BLOCK_S_V4; c += 32) {
                float v = (c < tile_rows)
                              ? __expf(score_smem[r][c] * attn_scale - new_m)
                              : 0.0f;
                p_smem[r][c] = __float2bfloat16(v);
                row_sum += v;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                row_sum += __shfl_xor_sync(0xffffffffu, row_sum, off);
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

        // Phase C: P·V WMMA (same as v4).
        #pragma unroll
        for (int w_n = 0; w_n < 2; ++w_n) {
            int n_tile  = warp_id * 2 + w_n;
            int n_start = n_tile * 16;
            fill_fragment(pv_acc, 0.f);
            #pragma unroll
            for (int k = 0; k < BLOCK_S_V4; k += 16) {
                load_matrix_sync(a_frag, &p_smem[0][k], P_STRIDE_V5);
                load_matrix_sync(b_pv_frag, &v_smem[k][n_start], V_STRIDE_V5);
                mma_sync(pv_acc, a_frag, b_pv_frag, pv_acc);
            }
            store_matrix_sync(&score_smem[0][warp_id * 16], pv_acc,
                              SCORE_STRIDE_V5, mem_row_major);
            int lane = tid & 31;
            #pragma unroll
            for (int i = lane; i < 16 * 16; i += 32) {
                int r = i / 16;
                int c = i % 16;
                o_smem[r][n_start + c] += score_smem[r][warp_id * 16 + c];
            }
        }
        __syncthreads();
    }

    // Epilogue.
    if constexpr (NS_EQ_1) {
        // Single split → o_smem holds sum(P · V) with un-normalized P; divide
        // by l_smem to get the final softmax-normalized output, write directly
        // to o_hat (skipping o_partial / m / l + the combine pass).
        for (int i = tid; i < num_groups * D; i += THREADS) {
            int r = i / D, c = i % D;
            int h_q = h_kv * num_groups + r;
            float l   = l_smem[r];
            float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
            size_t off = (size_t)b * H_q * D + (size_t)h_q * D + c;
            O_hat[off] = o_smem[r][c] * inv;
        }
    } else {
        // Multi-split path — write per-split partials for combine kernel.
        for (int i = tid; i < num_groups * D; i += THREADS) {
            int r = i / D, c = i % D;
            int h_q = h_kv * num_groups + r;
            size_t op_off = (size_t)b * H_q * num_splits * D
                          + (size_t)h_q * num_splits * D
                          + (size_t)split_idx * D + c;
            O_partial[op_off] = o_smem[r][c];
        }
        for (int i = tid; i < num_groups; i += THREADS) {
            int h_q = h_kv * num_groups + i;
            size_t ml_off = (size_t)b * H_q * num_splits
                          + (size_t)h_q * num_splits + split_idx;
            M_per_split[ml_off] = m_smem[i];
            L_per_split[ml_off] = l_smem[i];
        }
    }
    cp_async_wait_group_0();
}


// ─────────────────────────────────────────────────────────────────────────────
// Combine kernel: reduce (o_partial, m, l) across splits → final o_hat.
// Grid: (B, H_q), block: 128 threads (one per output dim).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void combine_splits_v4_kernel(
    const float* __restrict__ O_partial,     // (B, H_q, num_splits, D)
    const float* __restrict__ M_per_split,   // (B, H_q, num_splits)
    const float* __restrict__ L_per_split,   // (B, H_q, num_splits)
    float*       __restrict__ O_hat,         // (B, H_q, D)
    int H_q, int num_splits)
{
    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;

    const size_t ml_base = (size_t)b * H_q * num_splits
                         + (size_t)h_q * num_splits;
    const size_t op_base = ml_base * D;

    // Pass 1: find global m across splits.
    extern __shared__ float smem_combine[];
    float* smem_m = smem_combine;             // num_splits floats
    float* smem_l = smem_m + num_splits;      // num_splits floats

    float global_m = -INFINITY;
    for (int s = tid; s < num_splits; s += blockDim.x) {
        float m = M_per_split[ml_base + s];
        float l = L_per_split[ml_base + s];
        smem_m[s] = m;
        smem_l[s] = l;
    }
    __syncthreads();
    for (int s = 0; s < num_splits; ++s) {
        global_m = fmaxf(global_m, smem_m[s]);
    }

    // Pass 2: compute rescale factors + global l.
    float global_l = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        global_l += smem_l[s] * expf(smem_m[s] - global_m);
    }
    const float inv_l = 1.0f / global_l;

    // Pass 3: combine o_partial with rescale, write final o_hat.
    if (tid < D) {
        float acc = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            float scale_s = expf(smem_m[s] - global_m);
            acc += O_partial[op_base + (size_t)s * D + tid] * scale_s;
        }
        size_t out_off = (size_t)b * H_q * D + (size_t)h_q * D + tid;
        O_hat[out_off] = acc * inv_l;
    }
}


} // namespace

torch::Tensor
fused_qk_wmma_bf16(
    torch::Tensor q,              // (B, H_q, D) bf16
    torch::Tensor k_bf16,         // (H_kv, num_slots, D) bf16, pre-dequanted
    torch::Tensor block_table,    // (B, max_blocks) int64
    torch::Tensor seq_lens,       // (B,) int32
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    double  attn_scale)
{
    TORCH_CHECK(q.is_cuda() && k_bf16.is_cuda(), "CUDA only");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k_bf16.scalar_type() == torch::kBFloat16, "k_bf16 must be bf16");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(q.dim() == 3 && q.size(2) == D);
    TORCH_CHECK(k_bf16.dim() == 3 && k_bf16.size(2) == D);

    const int64_t B = q.size(0);
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto kc = k_bf16.contiguous();

    auto options_f32 = q.options().dtype(torch::kFloat32);
    auto scores = torch::zeros({B, H_q, max_S}, options_f32);

    const int num_tiles_S = (int)((max_S + BLOCK_S - 1) / BLOCK_S);
    dim3 grid((unsigned)B, (unsigned)H_q, (unsigned)num_tiles_S);
    dim3 block(THREADS);

    const auto stream = at::cuda::getCurrentCUDAStream();
    fused_qk_wmma_bf16_kernel<<<grid, block, 0, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kc.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        sl.data_ptr<int32_t>(),
        scores.data_ptr<float>(),
        (int)max_S, (int)H_q, (int)H_kv,
        (int)block_size,
        (int)bt.stride(0),
        (int)kc.stride(0),
        (float)attn_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}


torch::Tensor
fused_qk_wmma_bf16_v2(
    torch::Tensor q,
    torch::Tensor k_bf16,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t num_splits,
    double  attn_scale)
{
    TORCH_CHECK(q.is_cuda() && k_bf16.is_cuda(), "CUDA only");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_bf16.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(q.dim() == 3 && q.size(2) == D);
    TORCH_CHECK(k_bf16.dim() == 3 && k_bf16.size(2) == D);
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be multiple of H_kv for GQA");

    const int64_t B = q.size(0);
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto kc = k_bf16.contiguous();

    if (num_splits <= 0) {
        // Mirror the scalar kernel's heuristic: target ~2048 blocks.
        // Grid = B * H_kv * num_splits.  At B=8, H_kv=8 → 64 base; want
        // num_splits ∈ [1, 32] to fill SMs without fragmenting tiles.
        num_splits = std::max<int64_t>(
            1, std::min<int64_t>(32, 2048 / (B * H_kv)));
    }

    auto options_f32 = q.options().dtype(torch::kFloat32);
    auto scores = torch::zeros({B, H_q, max_S}, options_f32);

    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)num_splits);
    dim3 block(THREADS);

    const auto stream = at::cuda::getCurrentCUDAStream();
    fused_qk_wmma_bf16_v2_kernel<<<grid, block, 0, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kc.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        sl.data_ptr<int32_t>(),
        scores.data_ptr<float>(),
        (int)max_S, (int)H_q, (int)H_kv,
        (int)block_size, (int)bt.stride(0), (int)kc.stride(0),
        (int)num_splits, (float)attn_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}


torch::Tensor
fused_qk_wmma_bf16_v3(
    torch::Tensor q,
    torch::Tensor k_bf16,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t num_splits,
    double  attn_scale)
{
    TORCH_CHECK(q.is_cuda() && k_bf16.is_cuda(), "CUDA only");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_bf16.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(q.dim() == 3 && q.size(2) == D);
    TORCH_CHECK(k_bf16.dim() == 3 && k_bf16.size(2) == D);
    TORCH_CHECK(H_q % H_kv == 0);

    const int64_t B = q.size(0);
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto kc = k_bf16.contiguous();

    if (num_splits <= 0) {
        num_splits = std::max<int64_t>(
            1, std::min<int64_t>(32, 2048 / (B * H_kv)));
    }

    auto options_f32 = q.options().dtype(torch::kFloat32);
    auto scores = torch::zeros({B, H_q, max_S}, options_f32);

    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)num_splits);
    dim3 block(THREADS);

    const auto stream = at::cuda::getCurrentCUDAStream();
    fused_qk_wmma_bf16_v3_kernel<<<grid, block, 0, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kc.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        sl.data_ptr<int32_t>(),
        scores.data_ptr<float>(),
        (int)max_S, (int)H_q, (int)H_kv,
        (int)block_size, (int)bt.stride(0), (int)kc.stride(0),
        (int)num_splits, (float)attn_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}


torch::Tensor
fused_decode_wmma_v4(
    torch::Tensor q,
    torch::Tensor k_bf16,
    torch::Tensor v_bf16,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t num_splits,
    double  attn_scale)
{
    TORCH_CHECK(q.is_cuda() && k_bf16.is_cuda() && v_bf16.is_cuda(), "CUDA only");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_bf16.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(v_bf16.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(q.dim() == 3 && q.size(2) == D);
    TORCH_CHECK(k_bf16.dim() == 3 && k_bf16.size(2) == D);
    TORCH_CHECK(v_bf16.dim() == 3 && v_bf16.size(2) == D);
    TORCH_CHECK(H_q % H_kv == 0);

    const int64_t B = q.size(0);
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto kc = k_bf16.contiguous();
    auto vc = v_bf16.contiguous();
    TORCH_CHECK(kc.stride(0) == vc.stride(0),
                "K and V must share h-stride (dequant helper guarantees this)");

    if (num_splits <= 0) {
        // Microbench sweep (kernel_bench.py) shows best num_splits sitting
        // at 16-32 across B ∈ {1,2,4,8} H_kv=8 S=2048.  Target ~1024 blocks
        // (B·H_kv·splits ≥ ~1024) to keep all 114 H100 SMs busy under the
        // 1-block-per-SM limit imposed by ~82 KB dyn SMEM.  Cap at 32 so
        // the combine kernel's per-split reduction doesn't explode.
        num_splits = std::max<int64_t>(
            2, std::min<int64_t>(32, 1024 / (B * H_kv)));
        num_splits = std::min<int64_t>(num_splits, max_S / BLOCK_S_V4);
        num_splits = std::max<int64_t>(1, num_splits);
    }

    auto options_f32 = q.options().dtype(torch::kFloat32);
    auto o_hat     = torch::empty({B, H_q, D}, options_f32);
    auto o_partial = torch::empty({B, H_q, num_splits, D}, options_f32);
    auto m_per_spl = torch::empty({B, H_q, num_splits}, options_f32);
    auto l_per_spl = torch::empty({B, H_q, num_splits}, options_f32);

    const size_t smem_bytes =
        16 * D * sizeof(__nv_bfloat16) +
        STAGES * BLOCK_S_V4 * D * sizeof(__nv_bfloat16) +
        STAGES * BLOCK_S_V4 * D * sizeof(__nv_bfloat16) +
        16 * BLOCK_S_V4 * sizeof(float) +
        16 * BLOCK_S_V4 * sizeof(__nv_bfloat16) +
        16 * D * sizeof(float) +
        32 * sizeof(float);

    cudaFuncSetAttribute(fused_decode_wmma_v4_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);

    dim3 grid1((unsigned)B, (unsigned)H_kv, (unsigned)num_splits);
    dim3 block1(THREADS);

    const auto stream = at::cuda::getCurrentCUDAStream();
    fused_decode_wmma_v4_kernel<<<grid1, block1, smem_bytes, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kc.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(vc.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        sl.data_ptr<int32_t>(),
        o_partial.data_ptr<float>(),
        m_per_spl.data_ptr<float>(),
        l_per_spl.data_ptr<float>(),
        (int)max_S, (int)H_q, (int)H_kv,
        (int)block_size, (int)bt.stride(0), (int)kc.stride(0),
        (int)num_splits, (float)attn_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Combine kernel.
    dim3 grid2((unsigned)B, (unsigned)H_q);
    dim3 block2(128);  // >= D=128
    size_t combine_smem = 2 * num_splits * sizeof(float);
    combine_splits_v4_kernel<<<grid2, block2, combine_smem, stream.stream()>>>(
        o_partial.data_ptr<float>(),
        m_per_spl.data_ptr<float>(),
        l_per_spl.data_ptr<float>(),
        o_hat.data_ptr<float>(),
        (int)H_q, (int)num_splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return o_hat;
}


// ──────────────────────────────────────────────────────────────────────────
// kv_quant_write_int2_kernel — fuse the new (K, V) token's quant+pack+scatter
// into a single kernel launched right before the attention kernel.
//
// Replaces the Python mirror_kv path (~250 µs/layer dominated by Python +
// triton.launch overhead at decode batch=1).  This C++ launch is ~5–10 µs.
// Grid: (B, H_kv).  Block: one warp (32 threads).  Lane t handles 4 of the
// 128 D positions, lanes 0..7 cover what becomes the single group=128 entry.
//
// vLLM cache layout (group=128, NG=1, MSB-first byte order, OFFSET=2):
//   K asym: full INT2 codebook, signed q=code-2 ∈ [-2,1].
//           scale=(xmax-xmin)/3, zp=xmin+2*scale,
//           code=round((x-zp)/scale)+2 clamped to [0,3].
//   V sym : scale=amax,                              code=round(x/scale)+2
// ──────────────────────────────────────────────────────────────────────────
__global__ void kv_quant_write_int2_kernel(
    const __nv_bfloat16* __restrict__ K_new,   // (B, H_kv, D) bf16
    const __nv_bfloat16* __restrict__ V_new,   // (B, H_kv, D) bf16
    uint8_t*             __restrict__ K_data,  // (H_kv, num_slots, D/4)
    __nv_bfloat16*       __restrict__ K_scales,// (H_kv, num_slots)
    __nv_bfloat16*       __restrict__ K_zp,    // (H_kv, num_slots)
    uint8_t*             __restrict__ V_data,  // (H_kv, num_slots, D/4)
    __nv_bfloat16*       __restrict__ V_scales,// (H_kv, num_slots)
    const int64_t*       __restrict__ block_table,  // (B, max_blocks)
    const int32_t*       __restrict__ seq_lens,     // (B,)
    int B, int H_kv, int block_size, int max_blocks_per_seq,
    int K_data_stride_h, int K_scales_stride_h)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int t = threadIdx.x;
    if (b >= B || h >= H_kv || t >= 32) return;

    // New token sits at logical position seq_lens[b] - 1.
    const int pos       = seq_lens[b] - 1;
    if (pos < 0) return;
    const int block_idx = pos / block_size;
    const int in_block  = pos % block_size;
    const int64_t block_id = block_table[(size_t)b * max_blocks_per_seq + block_idx];
    const int64_t slot     = block_id * block_size + in_block;

    // Each lane reads 4 consecutive bf16 values: positions [4t, 4t+3] of the 128-D row.
    const __nv_bfloat16* k_row = K_new + ((size_t)b * H_kv + h) * D;
    const __nv_bfloat16* v_row = V_new + ((size_t)b * H_kv + h) * D;
    const int base = 4 * t;
    float k0 = __bfloat162float(k_row[base + 0]);
    float k1 = __bfloat162float(k_row[base + 1]);
    float k2 = __bfloat162float(k_row[base + 2]);
    float k3 = __bfloat162float(k_row[base + 3]);
    float v0 = __bfloat162float(v_row[base + 0]);
    float v1 = __bfloat162float(v_row[base + 1]);
    float v2 = __bfloat162float(v_row[base + 2]);
    float v3 = __bfloat162float(v_row[base + 3]);

    // Per-lane partials, then warp-reduce across all 32 lanes (group_size=128).
    float k_lmin = fminf(fminf(k0, k1), fminf(k2, k3));
    float k_lmax = fmaxf(fmaxf(k0, k1), fmaxf(k2, k3));
    float v_amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        k_lmin = fminf(k_lmin, __shfl_xor_sync(0xffffffffu, k_lmin, off));
        k_lmax = fmaxf(k_lmax, __shfl_xor_sync(0xffffffffu, k_lmax, off));
        v_amax = fmaxf(v_amax, __shfl_xor_sync(0xffffffffu, v_amax, off));
    }

    // ── K asym pack: full-codebook INT2, code ∈ [0,3], q=code-2 ──
    float k_scale_f = fmaxf((k_lmax - k_lmin) / 3.0f, 1e-8f);
    float k_zp_f    = k_lmin + 2.0f * k_scale_f;
    __nv_bfloat16 k_scale_bf = __float2bfloat16(k_scale_f);
    __nv_bfloat16 k_zp_bf    = __float2bfloat16(k_zp_f);
    float k_scale_ru = fmaxf(__bfloat162float(k_scale_bf), 1e-8f);
    float k_zp_ru    = __bfloat162float(k_zp_bf);
    auto kquant = [&](float v) -> int {
        int q = __float2int_rn((v - k_zp_ru) / k_scale_ru) + 2;
        return max(0, min(3, q));
    };
    int kc0 = kquant(k0), kc1 = kquant(k1), kc2 = kquant(k2), kc3 = kquant(k3);
    // MSB-first byte: bits[7:6]=kc0, [5:4]=kc1, [3:2]=kc2, [1:0]=kc3.
    uint8_t k_byte = (uint8_t)((kc0 << 6) | (kc1 << 4) | (kc2 << 2) | kc3);

    // ── V sym pack: scale=amax, code=round(x/scale)+2 ─────────────────────
    float v_scale_f = fmaxf(v_amax, 1e-8f);
    __nv_bfloat16 v_scale_bf = __float2bfloat16(v_scale_f);
    float v_scale_ru = fmaxf(__bfloat162float(v_scale_bf), 1e-8f);
    auto vquant = [&](float v) -> int {
        int q = __float2int_rn(v / v_scale_ru) + 2;
        return max(1, min(3, q));
    };
    int vc0 = vquant(v0), vc1 = vquant(v1), vc2 = vquant(v2), vc3 = vquant(v3);
    uint8_t v_byte = (uint8_t)((vc0 << 6) | (vc1 << 4) | (vc2 << 2) | vc3);

    // ── Write packed bytes (lane t writes byte at offset t) ──────────────
    K_data[(size_t)h * K_data_stride_h + slot * D_QUARTER + t] = k_byte;
    V_data[(size_t)h * K_data_stride_h + slot * D_QUARTER + t] = v_byte;

    // Lane 0 writes the per-(slot, head) scale / zp once.
    if (t == 0) {
        K_scales[(size_t)h * K_scales_stride_h + slot] = k_scale_bf;
        K_zp    [(size_t)h * K_scales_stride_h + slot] = k_zp_bf;
        V_scales[(size_t)h * K_scales_stride_h + slot] = v_scale_bf;
    }
}


torch::Tensor
fused_decode_wmma_v5_int2(
    torch::Tensor q,              // (B, H_q, D) bf16
    torch::Tensor k_data,         // (H_kv, num_slots, D/4) uint8
    torch::Tensor k_scales,       // (H_kv, num_slots) or (H_kv, num_slots, 1) bf16
    torch::Tensor k_zp,
    torch::Tensor v_data,         // (H_kv, num_slots, D/4) uint8
    torch::Tensor v_scales,
    torch::Tensor block_table,    // (B, max_blocks) int64
    torch::Tensor seq_lens,       // (B,) int32
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t num_splits,
    double  attn_scale)
{
    TORCH_CHECK(q.is_cuda() && k_data.is_cuda() && v_data.is_cuda(), "CUDA only");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_data.scalar_type() == torch::kByte);
    TORCH_CHECK(v_data.scalar_type() == torch::kByte);
    TORCH_CHECK(k_scales.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_zp.scalar_type()     == torch::kBFloat16);
    TORCH_CHECK(v_scales.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(q.dim() == 3 && q.size(2) == D);
    TORCH_CHECK(H_q % H_kv == 0);

    auto ks = (k_scales.dim() == 3) ? k_scales.squeeze(-1).contiguous() : k_scales.contiguous();
    auto kz = (k_zp.dim()     == 3) ? k_zp.squeeze(-1).contiguous()     : k_zp.contiguous();
    auto vs = (v_scales.dim() == 3) ? v_scales.squeeze(-1).contiguous() : v_scales.contiguous();

    const int64_t B = q.size(0);
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto kd = k_data.contiguous();
    auto vd = v_data.contiguous();

    // num_splits heuristic.  Empirically validated on a 5×4 grid (B ∈ {1..32},
    // S ∈ {4k..32k}, H_q=32, H_kv=8): even at B*H_kv ≥ SM_count the
    // multi-split path wins by ~25%.  Reason: splits provide memory-level
    // parallelism (more concurrent CTAs across SMs → more in-flight cp.async,
    // higher HBM utilization), not just SM-filling.  With 1 block/SM occupancy
    // and STAGES=2 cp.async depth, a single CTA can only keep 2 tiles in
    // flight; only horizontal grid expansion via splits saturates HBM.
    //
    // The ns=1 fast path (template <true>) is kept as a user-explicit option
    // (callers can pass num_splits=1 when they know better, e.g. very large
    // B*H_kv where 1024-block heuristic naturally returns 1).
    if (num_splits <= 0) {
        num_splits = std::max<int64_t>(
            2, std::min<int64_t>(32, 1024 / (B * H_kv)));
        num_splits = std::min<int64_t>(num_splits, max_S / BLOCK_S_V4);
        num_splits = std::max<int64_t>(1, num_splits);
    }
    const bool ns_eq_1 = (num_splits == 1);

    auto options_f32 = q.options().dtype(torch::kFloat32);
    auto o_hat     = torch::empty({B, H_q, D}, options_f32);
    // ns=1 fast path skips o_partial / m / l materialization + combine pass.
    torch::Tensor o_partial, m_per_spl, l_per_spl;
    if (!ns_eq_1) {
        o_partial = torch::empty({B, H_q, num_splits, D}, options_f32);
        m_per_spl = torch::empty({B, H_q, num_splits}, options_f32);
        l_per_spl = torch::empty({B, H_q, num_splits}, options_f32);
    }

    // Padded SMEM layout — strides are bumped to break WMMA bank aliasing.
    // Must mirror the kernel-side layout exactly.
    constexpr int P_STRIDE_HOST     = BLOCK_S_V4 + P_PAD;
    constexpr int SCORE_STRIDE_HOST = BLOCK_S_V4 + SCORE_PAD;
    const size_t smem_bytes =
        16 * Q_STRIDE_V5 * sizeof(__nv_bfloat16) +             // q
        STAGES * BLOCK_S_V4 * D_QUARTER * 2 +                  // k_pack + v_pack
        STAGES * BLOCK_S_V4 * sizeof(__nv_bfloat16) * 3 +      // k_scale + k_zp + v_scale
        BLOCK_S_V4 * K_STRIDE_V5 * sizeof(__nv_bfloat16) +     // k_smem (bf16, padded)
        BLOCK_S_V4 * V_STRIDE_V5 * sizeof(__nv_bfloat16) +     // v_smem (bf16, padded)
        16 * SCORE_STRIDE_HOST * sizeof(float) +               // score (padded)
        16 * P_STRIDE_HOST * sizeof(__nv_bfloat16) +           // p (padded)
        16 * O_STRIDE_V5 * sizeof(float) +                     // o (padded)
        32 * sizeof(float);                                    // m+l

    cudaFuncSetAttribute(fused_decode_wmma_v5_int2_kernel<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
    cudaFuncSetAttribute(fused_decode_wmma_v5_int2_kernel<false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);

    dim3 grid1((unsigned)B, (unsigned)H_kv, (unsigned)num_splits);
    dim3 block1(THREADS);

    const auto stream = at::cuda::getCurrentCUDAStream();
    if (ns_eq_1) {
        fused_decode_wmma_v5_int2_kernel<true>
            <<<grid1, block1, smem_bytes, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            kd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kz.data_ptr<at::BFloat16>()),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            sl.data_ptr<int32_t>(),
            /*O_partial=*/nullptr,
            /*M_per_split=*/nullptr,
            /*L_per_split=*/nullptr,
            o_hat.data_ptr<float>(),
            /*Scores=*/nullptr,
            (int)max_S, (int)H_q, (int)H_kv,
            (int)block_size, (int)bt.stride(0),
            (int)kd.stride(0), (int)ks.stride(0),
            (int)num_splits, (float)attn_scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return o_hat;
    }

    fused_decode_wmma_v5_int2_kernel<false>
        <<<grid1, block1, smem_bytes, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        kd.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kz.data_ptr<at::BFloat16>()),
        vd.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        sl.data_ptr<int32_t>(),
        o_partial.data_ptr<float>(),
        m_per_spl.data_ptr<float>(),
        l_per_spl.data_ptr<float>(),
        /*O_hat=*/nullptr,
        /*Scores=*/nullptr,
        (int)max_S, (int)H_q, (int)H_kv,
        (int)block_size, (int)bt.stride(0),
        (int)kd.stride(0), (int)ks.stride(0),
        (int)num_splits, (float)attn_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid2((unsigned)B, (unsigned)H_q);
    dim3 block2(128);
    size_t combine_smem = 2 * num_splits * sizeof(float);
    combine_splits_v4_kernel<<<grid2, block2, combine_smem, stream.stream()>>>(
        o_partial.data_ptr<float>(),
        m_per_spl.data_ptr<float>(),
        l_per_spl.data_ptr<float>(),
        o_hat.data_ptr<float>(),
        (int)H_q, (int)num_splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return o_hat;
}


// ──────────────────────────────────────────────────────────────────────────
// fused_decode_wmma_v5_int2_with_scores — same as fused_decode_wmma_v5_int2
// but ALSO emits raw QK scores (× attn_scale) to a (B, H_q, max_S) fp32
// tensor.  Used by the CPU correction path which needs scores for top-K +
// softmax computed outside the attention kernel.
// Returns (o_hat, scores).
// ──────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor>
fused_decode_wmma_v5_int2_with_scores(
    torch::Tensor q,
    torch::Tensor k_data, torch::Tensor k_scales, torch::Tensor k_zp,
    torch::Tensor v_data, torch::Tensor v_scales,
    torch::Tensor block_table, torch::Tensor seq_lens,
    int64_t max_S, int64_t H_q, int64_t H_kv,
    int64_t block_size, int64_t num_splits, double attn_scale)
{
    TORCH_CHECK(q.is_cuda() && k_data.is_cuda() && v_data.is_cuda(), "CUDA only");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(q.dim() == 3 && q.size(2) == D);
    TORCH_CHECK(H_q % H_kv == 0);

    auto ks = (k_scales.dim() == 3) ? k_scales.squeeze(-1).contiguous() : k_scales.contiguous();
    auto kz = (k_zp.dim()     == 3) ? k_zp.squeeze(-1).contiguous()     : k_zp.contiguous();
    auto vs = (v_scales.dim() == 3) ? v_scales.squeeze(-1).contiguous() : v_scales.contiguous();
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto kd = k_data.contiguous();
    auto vd = v_data.contiguous();

    const int64_t B = q.size(0);
    if (num_splits <= 0) {
        num_splits = std::max<int64_t>(
            2, std::min<int64_t>(32, 1024 / (B * H_kv)));
        num_splits = std::min<int64_t>(num_splits, max_S / BLOCK_S_V4);
        num_splits = std::max<int64_t>(1, num_splits);
    }
    const bool ns_eq_1 = (num_splits == 1);

    auto options_f32 = q.options().dtype(torch::kFloat32);
    auto o_hat   = torch::empty({B, H_q, D}, options_f32);
    auto scores  = torch::empty({B, H_q, max_S}, options_f32);

    torch::Tensor o_partial, m_per_spl, l_per_spl;
    if (!ns_eq_1) {
        o_partial = torch::empty({B, H_q, num_splits, D}, options_f32);
        m_per_spl = torch::empty({B, H_q, num_splits}, options_f32);
        l_per_spl = torch::empty({B, H_q, num_splits}, options_f32);
    }

    constexpr int P_STRIDE_HOST     = BLOCK_S_V4 + P_PAD;
    constexpr int SCORE_STRIDE_HOST = BLOCK_S_V4 + SCORE_PAD;
    const size_t smem_bytes =
        16 * Q_STRIDE_V5 * sizeof(__nv_bfloat16) +
        STAGES * BLOCK_S_V4 * D_QUARTER * 2 +
        STAGES * BLOCK_S_V4 * sizeof(__nv_bfloat16) * 3 +
        BLOCK_S_V4 * K_STRIDE_V5 * sizeof(__nv_bfloat16) +
        BLOCK_S_V4 * V_STRIDE_V5 * sizeof(__nv_bfloat16) +
        16 * SCORE_STRIDE_HOST * sizeof(float) +
        16 * P_STRIDE_HOST * sizeof(__nv_bfloat16) +
        16 * O_STRIDE_V5 * sizeof(float) +
        32 * sizeof(float);

    cudaFuncSetAttribute(fused_decode_wmma_v5_int2_kernel<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    cudaFuncSetAttribute(fused_decode_wmma_v5_int2_kernel<false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);

    dim3 grid1((unsigned)B, (unsigned)H_kv, (unsigned)num_splits);
    dim3 block1(THREADS);
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (ns_eq_1) {
        fused_decode_wmma_v5_int2_kernel<true>
            <<<grid1, block1, smem_bytes, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            kd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kz.data_ptr<at::BFloat16>()),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            sl.data_ptr<int32_t>(),
            nullptr, nullptr, nullptr,
            o_hat.data_ptr<float>(),
            scores.data_ptr<float>(),
            (int)max_S, (int)H_q, (int)H_kv,
            (int)block_size, (int)bt.stride(0),
            (int)kd.stride(0), (int)ks.stride(0),
            (int)num_splits, (float)attn_scale);
    } else {
        fused_decode_wmma_v5_int2_kernel<false>
            <<<grid1, block1, smem_bytes, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            kd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(kz.data_ptr<at::BFloat16>()),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            sl.data_ptr<int32_t>(),
            o_partial.data_ptr<float>(),
            m_per_spl.data_ptr<float>(),
            l_per_spl.data_ptr<float>(),
            nullptr,
            scores.data_ptr<float>(),
            (int)max_S, (int)H_q, (int)H_kv,
            (int)block_size, (int)bt.stride(0),
            (int)kd.stride(0), (int)ks.stride(0),
            (int)num_splits, (float)attn_scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        dim3 grid2((unsigned)B, (unsigned)H_q);
        dim3 block2(128);
        size_t combine_smem = 2 * num_splits * sizeof(float);
        combine_splits_v4_kernel<<<grid2, block2, combine_smem, stream.stream()>>>(
            o_partial.data_ptr<float>(),
            m_per_spl.data_ptr<float>(),
            l_per_spl.data_ptr<float>(),
            o_hat.data_ptr<float>(),
            (int)H_q, (int)num_splits);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, scores};
}


// ──────────────────────────────────────────────────────────────────────────
// fused_decode_wmma_v5_int2_appendkv — same as v5 but takes the new (K, V) for
// the current decode step and writes them into the cache *inside this kernel
// launch*, before the attention CTAs read.  Replaces the Python mirror_kv path
// that costs ~13 ms / token of W_total at decode B=1 (= ~250 µs/layer × 32).
//
// Caller must NOT mirror_kv for layers using this path — otherwise double-write
// of the same slot.  See decode_path.py JETFIRE_WMMA_DECODE_APPENDKV gate.
// ──────────────────────────────────────────────────────────────────────────
torch::Tensor
fused_decode_wmma_v5_int2_appendkv(
    torch::Tensor q,              // (B, H_q, D) bf16
    torch::Tensor k_new,          // (B, H_kv, D) bf16 — new K for this decode step
    torch::Tensor v_new,          // (B, H_kv, D) bf16
    torch::Tensor k_data,         // (H_kv, num_slots, D/4) uint8
    torch::Tensor k_scales,
    torch::Tensor k_zp,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,    // (B, max_blocks) int64
    torch::Tensor seq_lens,       // (B,) int32 — includes the new token
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t num_splits,
    double  attn_scale)
{
    TORCH_CHECK(k_new.is_cuda() && v_new.is_cuda(), "CUDA only");
    TORCH_CHECK(k_new.scalar_type() == torch::kBFloat16, "k_new bf16");
    TORCH_CHECK(v_new.scalar_type() == torch::kBFloat16, "v_new bf16");
    TORCH_CHECK(k_new.dim() == 3 && k_new.size(2) == D, "k_new (B, H_kv, D)");
    TORCH_CHECK(v_new.dim() == 3 && v_new.size(2) == D, "v_new (B, H_kv, D)");
    const int64_t B = q.size(0);
    TORCH_CHECK(k_new.size(0) == B && k_new.size(1) == H_kv);
    TORCH_CHECK(v_new.size(0) == B && v_new.size(1) == H_kv);

    auto kn = k_new.contiguous();
    auto vn = v_new.contiguous();
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto kd = k_data.contiguous();
    auto vd = v_data.contiguous();
    auto ks = (k_scales.dim() == 3) ? k_scales.squeeze(-1).contiguous() : k_scales.contiguous();
    auto kz = (k_zp.dim()     == 3) ? k_zp.squeeze(-1).contiguous()     : k_zp.contiguous();
    auto vs = (v_scales.dim() == 3) ? v_scales.squeeze(-1).contiguous() : v_scales.contiguous();

    const auto stream = at::cuda::getCurrentCUDAStream();

    // 1) Quant + write the new (K, V) into cache.  Same stream as attention
    //    below, so memory ordering is automatic.
    dim3 wgrid((unsigned)B, (unsigned)H_kv);
    dim3 wblock(32);
    kv_quant_write_int2_kernel<<<wgrid, wblock, 0, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(kn.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(vn.data_ptr<at::BFloat16>()),
        kd.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(kz.data_ptr<at::BFloat16>()),
        vd.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        sl.data_ptr<int32_t>(),
        (int)B, (int)H_kv, (int)block_size, (int)bt.size(1),
        (int)kd.stride(0), (int)ks.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // 2) Forward to the existing v5 attention kernel — it now sees the freshly-
    //    written slot at position seq_lens[b]-1.
    return fused_decode_wmma_v5_int2(
        q, k_data, k_scales, k_zp, v_data, v_scales,
        block_table, seq_lens,
        max_S, H_q, H_kv, block_size, num_splits, attn_scale);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_qk_wmma_bf16", &fused_qk_wmma_bf16,
          "Q·K^T only, M=1 padded, single tile per block (A/B reference)");
    m.def("fused_qk_wmma_bf16_v2", &fused_qk_wmma_bf16_v2,
          "Q·K^T only, grid (B, H_kv, splits), M=num_groups packed");
    m.def("fused_qk_wmma_bf16_v3", &fused_qk_wmma_bf16_v3,
          "Q·K^T only, v2 + cp.async 2-stage pipeline for K loads");
    m.def("fused_decode_wmma_v4", &fused_decode_wmma_v4,
          "Full fused decode: Q·K + online softmax + P·V (bf16 inputs)");
    m.def("fused_decode_wmma_v5_int2", &fused_decode_wmma_v5_int2,
          "v4 + in-kernel INT2 dequant");
    m.def("fused_decode_wmma_v5_int2_appendkv", &fused_decode_wmma_v5_int2_appendkv,
          "v5 + fused new-(K,V) quant write");
    m.def("fused_decode_wmma_v5_int2_with_scores", &fused_decode_wmma_v5_int2_with_scores,
          "v5 + raw-scores output for CPU correction top-K (production)");
}
