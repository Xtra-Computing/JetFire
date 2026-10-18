// Int8 tensor-core variant of int4_qk_scores.cu (Stage A: Q·K^T → scores +
// top-K post-pass).
//
// Differences vs int4_qk_scores.cu:
//   * Q is dynamically quantized BF16 → Int8 with a **per-row × per-K-group**
//     scale.  D=128 is split into K_ITERS=4 groups of K_TILE=32 channels
//     each; for each (m, k_iter) we compute scale = max_d_in_iter |Q[m,d]|
//     / 127 and quantize.  This way every channel is at most ~7-8 bits
//     coarser than its row's local max in that K-group.
//   * K is INT4-packed on GMEM; the kernel decodes the centered int4 value
//     (range [-8, 7]) directly into a signed int8 SMEM tile.
//   * The MMA uses PTX `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
//     (H100 / sm_90 has 1979 TFLOPS int8 peak, vs 989 TFLOPS bf16).
//   * Post-mma rescale:
//       score[m, t] = k_scale[t] *
//                     Σ_k_iter (q_scale[m, k_iter] * int32_acc[k_iter, m, t])
//     Each K-iter has its own int32 accumulator so the per-K-iter q scale
//     can be applied before summing.
//
// Tile layout:
//   BLOCK_S = 64 token cols  (4 warps × 2 N-tiles of 8 cols each)
//   THREADS = 128            (4 warps × 32 lanes)
//   STAGES  = 2              (cp.async double-buffer for K_pack + k_scale)
//   D       = 128 = 4 × K32  (4 mma K-iterations per N-tile)
//   M-tile  = 16             (we pad rows num_groups..15 with zero int8)
//
// The top-K post-pass is bit-width-agnostic (operates on fp32 scores) and
// is copied verbatim from int4_qk_scores.cu.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_load.cuh>

#include <string>

namespace {

constexpr int D            = 128;
constexpr int D_HALF       = D / 2;          // 64 packed bytes per K row (INT4)
constexpr int BLOCK_S      = 128;            // token-col tile width.
// At 128: each CTA does 2× the BLOCK_S=64 work; grid shrinks by 2×.
constexpr int THREADS      = 128;            // 4 warps × 32 (tried 256 = 8
                                             // warps: slight regression).
constexpr int STAGES       = 2;              // double-buffered cp.async pipeline
constexpr int N_TILES_PER_WARP = BLOCK_S / 4 / 8;  // 4 N-tiles per warp

constexpr int M_TILE       = 16;             // mma M dim
// N_TILE = 8 mma N dim (implicit; baked into the mma asm).
constexpr int K_TILE       = 32;             // mma K dim
constexpr int K_ITERS      = D / K_TILE;     // 4
constexpr int MAX_Q_GROUPS = 4;              // current fused paths only support H_q/H_kv <= 4

// SMEM Q tile: int8 (M_TILE × D).  Pad K dimension to avoid 32B bank conflicts.
constexpr int Q_PAD_B      = 16;             // 16 bytes of padding per row
constexpr int Q_STRIDE_B   = D + Q_PAD_B;    // 144 bytes per Q row (int8)

// k_pack (INT4 packed, row-strided): pad row stride to 80 bytes (16 byte
// pad) so consecutive k_rows land on different SMEM banks for the
// register-resident MMA B-operand build (mirrors the Stage B sv_reg fix).
// Stride 80 = 20 banks per row → 4 consecutive k_rows hit banks 0,20,8,28.
constexpr int K_PACK_PAD   = 16;
constexpr int K_PACK_STRIDE = D_HALF + K_PACK_PAD;  // 80 bytes per packed K row

constexpr int SCORE_PAD    = 4;
constexpr int SCORE_STRIDE = BLOCK_S + SCORE_PAD;  // fp32 cols per row

// cp.async helpers (16B). H100 / sm_90.
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

// `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
//   A: (16, 32) int8 row-major.    Per thread holds 4 b32 regs = 16 int8.
//   B: (32, 8)  int8 col-major.    Per thread holds 2 b32 regs = 8  int8.
//   C/D: (16, 8) int32.            Per thread holds 4 s32 regs.
__device__ __forceinline__ void mma_m16n8k32_s8s8s32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int& c0, int& c1, int& c2, int& c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
        " {%0, %1, %2, %3},"
        " {%4, %5, %6, %7},"
        " {%8, %9},"
        " {%0, %1, %2, %3};"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}


__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ uint32_t f2r_desc(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void int8tc_qk_scores_kernel(
    const __nv_bfloat16* __restrict__ Q,           // (B, H_q, D)
    const uint8_t*       __restrict__ K_data,      // (H_kv, num_slots, D/2)
    const __nv_bfloat16* __restrict__ K_scales,    // (H_kv, num_slots)
    const int64_t*       __restrict__ BlockTable,  // (B, max_blocks)
    const int32_t*       __restrict__ SeqLens,     // (B,)
    float*               __restrict__ Scores,      // (B, H_q, max_S) fp32
    int    max_S, int H_q, int H_kv,
    int    block_size, int block_table_stride,
    int    K_data_stride_h,
    int    K_scales_stride_h,
    float  attn_scale)
{
    const int b        = blockIdx.x;
    const int h_kv     = blockIdx.y;
    const int tile_idx = blockIdx.z;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane     = tid & 31;

    const int num_groups = H_q / H_kv;          // up to M_TILE
    const int S          = SeqLens[b];
    const int tile_s0    = tile_idx * BLOCK_S;
    if (tile_s0 >= max_S) return;
    const int tile_end_S = min(tile_s0 + BLOCK_S, S);
    const int tile_end_M = min(tile_s0 + BLOCK_S, max_S);

    // ── Dynamic SMEM layout ──────────────────────────────────────────────
    extern __shared__ __align__(16) uint8_t dyn_smem[];
    size_t off = 0;

    // q_smem: int8 (M_TILE × Q_STRIDE_B)
    auto q_smem    = reinterpret_cast<int8_t(*)[Q_STRIDE_B]>(&dyn_smem[off]);
    off += (size_t)M_TILE * Q_STRIDE_B;
    // q_scale (per row × per K-iter): float — used to rescale int32 mma accum.
    // Layout: q_row_scale[m * K_ITERS + k_iter]
    auto q_row_scale = reinterpret_cast<float*>(&dyn_smem[off]);
    off += (size_t)M_TILE * K_ITERS * sizeof(float);
    // alignment to 16B
    off = (off + 15) & ~size_t(15);

    // k_pack: STAGES × BLOCK_S × K_PACK_STRIDE bytes (uint8, INT4 packed +
    // 16-byte row padding for bank-conflict-free register reads).
    auto k_pack    = reinterpret_cast<uint8_t(*)[BLOCK_S][K_PACK_STRIDE]>(&dyn_smem[off]);
    off += (size_t)STAGES * BLOCK_S * K_PACK_STRIDE;
    // k_scale: STAGES × BLOCK_S bfloat16
    auto k_scale_smem = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S]>(&dyn_smem[off]);
    off += (size_t)STAGES * BLOCK_S * sizeof(__nv_bfloat16);
    off = (off + 15) & ~size_t(15);

    // (No k_smem region — the mma B-operand is built directly from k_pack
    //  in registers, saving the SMEM round-trip + one __syncthreads.)

    // score_smem: M_TILE × SCORE_STRIDE float (intermediate int32 accum after
    // rescale into fp32).
    auto score_smem = reinterpret_cast<float(*)[SCORE_STRIDE]>(&dyn_smem[off]);
    off += (size_t)M_TILE * SCORE_STRIDE * sizeof(float);

    // ── Pointers (hoisted so cp.async can issue before Q-quantize) ───────
    const uint8_t*       k_pack_base = K_data   + (size_t)h_kv * K_data_stride_h;
    const __nv_bfloat16* ks_base     = K_scales + (size_t)h_kv * K_scales_stride_h;
    const int64_t*       bt_row      = BlockTable + (size_t)b * block_table_stride;

    auto issue_load = [&](int stage, int tile_s0_local, int tile_end_local) {
        // For BLOCK_S=128: cp.async16 ops cover BLOCK_S * D_HALF = 128 * 64
        // = 8192 bytes = 512 × 16-byte.  128 threads × 4 iters.
        constexpr int N_LOAD_ITERS = (BLOCK_S * D_HALF) / (THREADS * 16);
        #pragma unroll
        for (int it = 0; it < N_LOAD_ITERS; ++it) {
            int lin = it * THREADS + tid;
            int row   = lin >> 2;
            int chunk = lin & 3;
            if (row < BLOCK_S) {
                int pos = tile_s0_local + row;
                int64_t slot = -1;
                if (pos < tile_end_local) {
                    slot = bt_row[pos / block_size] * (int64_t)block_size
                         + (pos % block_size);
                }
                uint8_t* dst = &k_pack[stage][row][chunk * 16];
                if (slot >= 0) {
                    const uint8_t* src = k_pack_base + slot * (int64_t)D_HALF + chunk * 16;
                    cp_async16(dst, src);
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) dst[i] = 0;
                }
            }
        }
        // k_scale: 128 threads × 1 bf16 each.
        {
            int row = tid;
            if (row < BLOCK_S) {
                int pos = tile_s0_local + row;
                int64_t slot = -1;
                if (pos < tile_end_local) {
                    slot = bt_row[pos / block_size] * (int64_t)block_size
                         + (pos % block_size);
                }
                if (slot >= 0) {
                    k_scale_smem[stage][row] = ks_base[slot];
                } else {
                    k_scale_smem[stage][row] = __float2bfloat16(0.f);
                }
            }
        }
        cp_async_commit();
    };

    // Issue cp.async FIRST — fire-and-forget.  HBM latency is hidden by the
    // Q-quantize work that runs concurrently in the same threads (after the
    // cp.async commit, threads return to the Q-quantize loop).  This removes
    // the previously-required __syncthreads between Q-quantize and cp.async
    // by reordering — Q-quantize writes q_smem only, cp.async writes k_pack
    // only, so they touch disjoint SMEM.
    issue_load(0, tile_s0, tile_end_S);

    // ── Stage 1: quantize Q (BF16 → Int8) per row × per K-iter scale ─────
    // Zero-pad rows num_groups..M_TILE-1 first (all D bytes per row).
    for (int i = tid; i < (M_TILE - num_groups) * Q_STRIDE_B; i += THREADS) {
        int r = num_groups + i / Q_STRIDE_B;
        int c = i % Q_STRIDE_B;
        q_smem[r][c] = 0;
    }
    // Padded rows have zero scale (contributes 0 post-mma).
    for (int i = tid; i < (M_TILE - num_groups) * K_ITERS; i += THREADS) {
        q_row_scale[(num_groups + i / K_ITERS) * K_ITERS + (i % K_ITERS)] = 0.f;
    }

    if (num_groups <= 4) {
        // For standard GQA, one warp per row.  Inside the warp each lane covers
        // one of the 32 channels of a K-group; we sweep 4 K-groups sequentially.
        if (warp_id < num_groups) {
            const int r = warp_id;
            const int h_q = h_kv * num_groups + r;
            const __nv_bfloat16* qrow =
                Q + (size_t)b * H_q * D + (size_t)h_q * D;

            #pragma unroll
            for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
                const int d = k_iter * K_TILE + lane;
                float f = __bfloat162float(qrow[d]);
                float v = fabsf(f);

                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    float other = __shfl_xor_sync(0xFFFFFFFFu, v, off);
                    v = fmaxf(v, other);
                }
                const float q_scale = fmaxf(v / 127.f, 1e-12f);

                if (lane == 0) {
                    q_row_scale[r * K_ITERS + k_iter] = q_scale;
                }
                const float inv = 1.f / q_scale;
                int8_t qi = (int8_t)max(-128, min(127, __float2int_rn(f * inv)));
                q_smem[r][d] = qi;
            }
        }
    } else {
        // Fallback for rare num_groups > 4 (would need num_groups warps).
        if (tid == 0) {
            for (int r = 0; r < num_groups; ++r) {
                const int h_q = h_kv * num_groups + r;
                const __nv_bfloat16* qrow =
                    Q + (size_t)b * H_q * D + (size_t)h_q * D;
                for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
                    float row_max = 0.f;
                    for (int dd = 0; dd < K_TILE; ++dd) {
                        float f = __bfloat162float(qrow[k_iter * K_TILE + dd]);
                        row_max = fmaxf(row_max, fabsf(f));
                    }
                    const float q_scale = fmaxf(row_max / 127.f, 1e-12f);
                    q_row_scale[r * K_ITERS + k_iter] = q_scale;
                    const float inv = 1.f / q_scale;
                    for (int dd = 0; dd < K_TILE; ++dd) {
                        const int d = k_iter * K_TILE + dd;
                        float f = __bfloat162float(qrow[d]);
                        q_smem[r][d] = (int8_t)max(-128, min(127, __float2int_rn(f * inv)));
                    }
                }
            }
        }
    }
    // Q-quantize done.  Wait for the cp.async issued before Q-quantize to
    // land in k_pack, then sync so all warps see both q_smem and k_pack.
    cp_async_wait_group_0();
    __syncthreads();

    // ── Stage 3: SKIPPED — no SMEM K decode.  The mma loop below builds its
    //            B operand directly from k_pack in registers (INT4 → INT8
    //            in-register expansion).  Saves the BLOCK_S × K_STRIDE_B
    //            SMEM region (9 KB) and ONE __syncthreads per CTA — that
    //            __syncthreads was ~half of every CTA's barrier wait.

    // ── Stage 4: MMA loop.  Each warp computes 2 N-tiles of 8 cols.       ─
    //
    // N tile mapping:
    //   warp 0 → cols  0..15 (n_tile 0 = cols 0..7, n_tile 1 = cols 8..15)
    //   warp 1 → cols 16..31
    //   warp 2 → cols 32..47
    //   warp 3 → cols 48..63
    //
    // For each warp, we need to load:
    //   * Q (16 × 128 int8) for the A operand — shared across all warps.
    //     A operand layout for m16n8k32 (row-major 16×32 sub-tile per mma):
    //       per-lane: 4 × b32 = {a0, a1, a2, a3}
    //         a0 = q[gid    ][tid*4 + (0..3)]  (cols  0..3 of K-iter k0)
    //         a1 = q[gid + 8][tid*4 + (0..3)]
    //         a2 = q[gid    ][tid*4 + (16..19)]
    //         a3 = q[gid + 8][tid*4 + (16..19)]
    //     where gid = lane >> 2, tid = lane & 3.
    //   * K (8 × 128 int8) for the B operand for this warp's 8 cols.
    //     B operand layout (col-major 32×8):
    //       per-lane: 2 × b32 = {b0, b1}
    //         b0 = k[col = gid][k_off + tid*4 + (0..3)]
    //         b1 = k[col = gid][k_off + tid*4 + (16..19)]
    //     The "col" is the N dimension (= n_start + gid within 0..7).

    // Per (K-iter, N-tile) accumulator — 4 K-iters × N_TILES_PER_WARP × 4
    // int32 each.  We can't accumulate across K-iters because each K-iter
    // has its own q_scale, applied during the rescale-to-fp32 step below.
    int32_t acc[K_ITERS][N_TILES_PER_WARP][4];
    #pragma unroll
    for (int kk = 0; kk < K_ITERS; ++kk)
    #pragma unroll
    for (int n = 0; n < N_TILES_PER_WARP; ++n)
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc[kk][n][i] = 0;

    const int gid = lane >> 2;       // 0..7
    const int tid_in_g = lane & 3;   // 0..3
    const int n_start = warp_id * (N_TILES_PER_WARP * 8);
                                     // each warp covers N_TILES_PER_WARP × 8 cols

    #pragma unroll
    for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
        const int k_off = k_iter * K_TILE;
        // Load A (Q) fragments — shared across warps but each warp reads
        // its own copy (cheap shared-memory reads, no conflicts on int8).
        uint32_t a0, a1, a2, a3;
        {
            const int8_t* p0 = &q_smem[gid    ][k_off + tid_in_g * 4 +  0];
            const int8_t* p1 = &q_smem[gid + 8][k_off + tid_in_g * 4 +  0];
            const int8_t* p2 = &q_smem[gid    ][k_off + tid_in_g * 4 + 16];
            const int8_t* p3 = &q_smem[gid + 8][k_off + tid_in_g * 4 + 16];
            a0 = *reinterpret_cast<const uint32_t*>(p0);
            a1 = *reinterpret_cast<const uint32_t*>(p1);
            a2 = *reinterpret_cast<const uint32_t*>(p2);
            a3 = *reinterpret_cast<const uint32_t*>(p3);
        }

        // For each of the 2 N-tiles in this warp:
        //
        // B operand for m16n8k32 (per-lane 2× uint32 = 8 INT8 covering
        // K-direction K_TILE=32 with the second half offset by +16).  We
        // build it directly from k_pack (INT4 packed bytes) in registers —
        // no SMEM intermediate, no extra __syncthreads.
        //
        // Bank-conflict-free read pattern: each lane reads uint32 at a
        // 4-byte-aligned offset.  Adjacent lane-pairs (tid_in_g={0,1} and
        // {2,3}) share the same 4-byte chunk and get a SMEM broadcast (no
        // serialization).  Picking uint16 directly at the unaligned offset
        // would split bank 0 across lanes 0/1 and cause a 2-way conflict.
        const int byte_idx_b0_unaligned = (k_off >> 1) + (tid_in_g << 1);
        const int byte_idx_b0_aligned   = byte_idx_b0_unaligned & ~3;
        const int byte_idx_b1_aligned   = byte_idx_b0_aligned + 8;
        const int shift_within_u32      = (tid_in_g & 1) * 16;

        auto extract_4_int8 = [](uint16_t u16) -> uint32_t {
            // u16 layout: [byte_hi : byte_lo]
            // byte_lo = high_nibble (D+0) | low_nibble (D+1)
            // byte_hi = high_nibble (D+2) | low_nibble (D+3)
            const int n0 = ((int)(u16 >> 4)  & 0xF) - 8;
            const int n1 = ((int)(u16)       & 0xF) - 8;
            const int n2 = ((int)(u16 >> 12) & 0xF) - 8;
            const int n3 = ((int)(u16 >> 8)  & 0xF) - 8;
            return ((uint32_t)(uint8_t)(int8_t)n0)
                 | (((uint32_t)(uint8_t)(int8_t)n1) <<  8)
                 | (((uint32_t)(uint8_t)(int8_t)n2) << 16)
                 | (((uint32_t)(uint8_t)(int8_t)n3) << 24);
        };

        #pragma unroll
        for (int n = 0; n < N_TILES_PER_WARP; ++n) {
            const int n_off = n_start + n * 8;
            const int col = n_off + gid;
            const uint32_t u32_b0 = *reinterpret_cast<const uint32_t*>(
                &k_pack[0][col][byte_idx_b0_aligned]);
            const uint32_t u32_b1 = *reinterpret_cast<const uint32_t*>(
                &k_pack[0][col][byte_idx_b1_aligned]);
            uint16_t u16_b0 = (uint16_t)(u32_b0 >> shift_within_u32);
            uint16_t u16_b1 = (uint16_t)(u32_b1 >> shift_within_u32);
            uint32_t b0 = extract_4_int8(u16_b0);
            uint32_t b1 = extract_4_int8(u16_b1);
            mma_m16n8k32_s8s8s32(a0, a1, a2, a3,
                                 b0, b1,
                                 acc[k_iter][n][0], acc[k_iter][n][1],
                                 acc[k_iter][n][2], acc[k_iter][n][3]);
        }
    }

    // ── Stage 5: rescale int32 accum (per K-iter) and combine, write to
    //            score_smem.
    //   score[m, t] = k_scale[t] *
    //                 Σ_k_iter (q_row_scale[m, k_iter] * acc[k_iter, m, t])
    //
    // C/D layout (m16n8 row-major, lane = 4*gid + tid_in_g):
    //   c[0] = (row gid    , col tid_in_g*2 + 0)
    //   c[1] = (row gid    , col tid_in_g*2 + 1)
    //   c[2] = (row gid + 8, col tid_in_g*2 + 0)
    //   c[3] = (row gid + 8, col tid_in_g*2 + 1)
    #pragma unroll
    for (int n = 0; n < N_TILES_PER_WARP; ++n) {
        const int n_off = n_start + n * 8;
        const int m0 = gid;
        const int m1 = gid + 8;
        const int c0 = n_off + tid_in_g * 2 + 0;
        const int c1 = n_off + tid_in_g * 2 + 1;
        const float ks0 = __bfloat162float(k_scale_smem[0][c0]);
        const float ks1 = __bfloat162float(k_scale_smem[0][c1]);

        float s00 = 0.f, s01 = 0.f, s10 = 0.f, s11 = 0.f;
        #pragma unroll
        for (int kk = 0; kk < K_ITERS; ++kk) {
            const float qs0 = q_row_scale[m0 * K_ITERS + kk];
            const float qs1 = q_row_scale[m1 * K_ITERS + kk];
            s00 += (float)acc[kk][n][0] * qs0;
            s01 += (float)acc[kk][n][1] * qs0;
            s10 += (float)acc[kk][n][2] * qs1;
            s11 += (float)acc[kk][n][3] * qs1;
        }
        score_smem[m0][c0] = s00 * ks0;
        score_smem[m0][c1] = s01 * ks1;
        score_smem[m1][c0] = s10 * ks0;
        score_smem[m1][c1] = s11 * ks1;
    }
    __syncthreads();

    // ── Stage 6: emit to GMEM (× attn_scale) with -inf sentinel for OOR.  ─
    const int valid_rows = num_groups;
    const int tile_cols  = tile_end_M - tile_s0;
    const int total      = valid_rows * tile_cols;
    for (int i = tid; i < total; i += THREADS) {
        int r       = i / tile_cols;
        int c       = i % tile_cols;
        int h_q     = h_kv * num_groups + r;
        int s_glob  = tile_s0 + c;
        size_t out_off = ((size_t)b * H_q + h_q) * max_S + s_glob;
        if (s_glob < tile_end_S) {
            Scores[out_off] = score_smem[r][c] * attn_scale;
        } else {
            Scores[out_off] = -INFINITY;
        }
    }
}


template <int K_T>
__global__ void int8tc_qk_scores_topk_stats_fused_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const uint8_t*       __restrict__ K_data,
    const __nv_bfloat16* __restrict__ K_scales,
    const int64_t*       __restrict__ BlockTable,
    const int32_t*       __restrict__ SeqLens,
    float*               __restrict__ Scores,
    int64_t*             __restrict__ topk_idx,
    int64_t*             __restrict__ topk_slots,
    float*               __restrict__ softmax_m,
    float*               __restrict__ softmax_inv_z,
    int    max_S, int H_q, int H_kv,
    int    block_size, int block_table_stride,
    int    K_data_stride_h,
    int    K_scales_stride_h,
    float  attn_scale)
{
    static_assert(K_T % 32 == 0, "top-k must be warp-distributed");
    constexpr int K_PER_LANE = K_T / 32;

    const int b       = blockIdx.x;
    const int h_kv    = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int num_groups = H_q / H_kv;
    const int S          = SeqLens[b];
    const int num_tiles  = (max_S + BLOCK_S - 1) / BLOCK_S;

    extern __shared__ __align__(16) uint8_t dyn_smem[];
    size_t off = 0;
    auto q_smem = reinterpret_cast<int8_t(*)[Q_STRIDE_B]>(&dyn_smem[off]);
    off += (size_t)M_TILE * Q_STRIDE_B;
    auto q_row_scale = reinterpret_cast<float*>(&dyn_smem[off]);
    off += (size_t)M_TILE * K_ITERS * sizeof(float);
    off = (off + 15) & ~size_t(15);
    auto k_pack = reinterpret_cast<uint8_t(*)[BLOCK_S][K_PACK_STRIDE]>(&dyn_smem[off]);
    off += (size_t)STAGES * BLOCK_S * K_PACK_STRIDE;
    auto k_scale_smem = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S]>(&dyn_smem[off]);
    off += (size_t)STAGES * BLOCK_S * sizeof(__nv_bfloat16);
    off = (off + 15) & ~size_t(15);
    auto score_smem = reinterpret_cast<float(*)[SCORE_STRIDE]>(&dyn_smem[off]);

    const uint8_t*       k_pack_base = K_data   + (size_t)h_kv * K_data_stride_h;
    const __nv_bfloat16* ks_base     = K_scales + (size_t)h_kv * K_scales_stride_h;
    const int64_t*       bt_row      = BlockTable + (size_t)b * block_table_stride;

    auto issue_load = [&](int stage, int tile_idx_local) {
        const int tile_s0_local = tile_idx_local * BLOCK_S;
        const int tile_end_local = min(tile_s0_local + BLOCK_S, S);
        constexpr int N_LOAD_ITERS = (BLOCK_S * D_HALF) / (THREADS * 16);
        #pragma unroll
        for (int it = 0; it < N_LOAD_ITERS; ++it) {
            int lin = it * THREADS + tid;
            int row   = lin >> 2;
            int chunk = lin & 3;
            if (row < BLOCK_S) {
                int pos = tile_s0_local + row;
                int64_t slot = -1;
                if (pos < tile_end_local) {
                    slot = bt_row[pos / block_size] * (int64_t)block_size
                         + (pos % block_size);
                }
                uint8_t* dst = &k_pack[stage][row][chunk * 16];
                if (slot >= 0) {
                    const uint8_t* src = k_pack_base + slot * (int64_t)D_HALF + chunk * 16;
                    cp_async16(dst, src);
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) dst[i] = 0;
                }
            }
        }
        if (tid < BLOCK_S) {
            int pos = tile_s0_local + tid;
            int64_t slot = -1;
            if (pos < tile_end_local) {
                slot = bt_row[pos / block_size] * (int64_t)block_size
                     + (pos % block_size);
            }
            k_scale_smem[stage][tid] = (slot >= 0) ? ks_base[slot] : __float2bfloat16(0.0f);
        }
        cp_async_commit();
    };

    if (num_tiles > 0) {
        issue_load(0, 0);
    }

    for (int i = tid; i < (M_TILE - num_groups) * Q_STRIDE_B; i += THREADS) {
        int r = num_groups + i / Q_STRIDE_B;
        int c = i % Q_STRIDE_B;
        q_smem[r][c] = 0;
    }
    for (int i = tid; i < (M_TILE - num_groups) * K_ITERS; i += THREADS) {
        q_row_scale[(num_groups + i / K_ITERS) * K_ITERS + (i % K_ITERS)] = 0.f;
    }
    if (num_groups <= 4) {
        if (warp_id < num_groups) {
            const int r = warp_id;
            const int h_q = h_kv * num_groups + r;
            const __nv_bfloat16* qrow =
                Q + (size_t)b * H_q * D + (size_t)h_q * D;
            #pragma unroll
            for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
                const int d = k_iter * K_TILE + lane;
                float f = __bfloat162float(qrow[d]);
                float v = fabsf(f);
                v = warp_max(v);
                const float q_scale = fmaxf(v / 127.f, 1e-12f);
                if (lane == 0) q_row_scale[r * K_ITERS + k_iter] = q_scale;
                q_smem[r][d] = (int8_t)max(-128, min(127, __float2int_rn(f / q_scale)));
            }
        }
    } else {
        if (tid == 0) {
            for (int r = 0; r < num_groups; ++r) {
                const int h_q = h_kv * num_groups + r;
                const __nv_bfloat16* qrow =
                    Q + (size_t)b * H_q * D + (size_t)h_q * D;
                for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
                    float row_max = 0.f;
                    for (int dd = 0; dd < K_TILE; ++dd) {
                        float f = __bfloat162float(qrow[k_iter * K_TILE + dd]);
                        row_max = fmaxf(row_max, fabsf(f));
                    }
                    const float q_scale = fmaxf(row_max / 127.f, 1e-12f);
                    q_row_scale[r * K_ITERS + k_iter] = q_scale;
                    const float inv = 1.f / q_scale;
                    for (int dd = 0; dd < K_TILE; ++dd) {
                        const int d = k_iter * K_TILE + dd;
                        q_smem[r][d] = (int8_t)max(-128, min(127, __float2int_rn(__bfloat162float(qrow[d]) * inv)));
                    }
                }
            }
        }
    }

    cp_async_wait_group_0();
    __syncthreads();

    float row_topk_keys[K_PER_LANE];
    int   row_topk_vals[K_PER_LANE];
    #pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        row_topk_keys[i] = -INFINITY;
        row_topk_vals[i] = -1;
    }
    int filled_chunks = 0;
    float row_m = -INFINITY;
    float row_l = 0.0f;

    const int gid = lane >> 2;
    const int tid_in_g = lane & 3;
    const int n_start = warp_id * (N_TILES_PER_WARP * 8);

    auto update_topk_chunk = [&](float new_key, int new_idx) {
        if (filled_chunks < K_PER_LANE) {
            row_topk_keys[filled_chunks] = new_key;
            row_topk_vals[filled_chunks] = new_idx;
            return;
        }
        while (true) {
            float my_min = row_topk_keys[0];
            int   my_min_sub = 0;
            #pragma unroll
            for (int i = 1; i < K_PER_LANE; ++i) {
                if (row_topk_keys[i] < my_min) {
                    my_min = row_topk_keys[i];
                    my_min_sub = i;
                }
            }
            float global_min = my_min;
            int global_min_lane = lane;
            int global_min_sub = my_min_sub;
            #pragma unroll
            for (int off2 = 16; off2 > 0; off2 >>= 1) {
                float other_min = __shfl_xor_sync(0xffffffffu, global_min, off2);
                int other_lane = __shfl_xor_sync(0xffffffffu, global_min_lane, off2);
                int other_sub = __shfl_xor_sync(0xffffffffu, global_min_sub, off2);
                bool replace = (other_min < global_min)
                    || (other_min == global_min && other_lane < global_min_lane);
                if (replace) {
                    global_min = other_min;
                    global_min_lane = other_lane;
                    global_min_sub = other_sub;
                }
            }
            bool qual = (new_key > global_min);
            unsigned active = __ballot_sync(0xffffffffu, qual);
            if (active == 0) break;
            int pick_lane = __ffs(active) - 1;
            float picked_key = __shfl_sync(0xffffffffu, new_key, pick_lane);
            int picked_idx = __shfl_sync(0xffffffffu, new_idx, pick_lane);
            if (lane == global_min_lane) {
                row_topk_keys[global_min_sub] = picked_key;
                row_topk_vals[global_min_sub] = picked_idx;
            }
            if (lane == pick_lane) {
                new_key = -INFINITY;
                new_idx = -1;
            }
        }
    };

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_s0 = tile_idx * BLOCK_S;
        const int tile_end_S = min(tile_s0 + BLOCK_S, S);
        const int tile_end_M = min(tile_s0 + BLOCK_S, max_S);
        const int valid_cols = max(0, tile_end_S - tile_s0);
        const int tile_cols = max(0, tile_end_M - tile_s0);

        int32_t acc[K_ITERS][N_TILES_PER_WARP][4];
        #pragma unroll
        for (int kk = 0; kk < K_ITERS; ++kk)
        #pragma unroll
        for (int n = 0; n < N_TILES_PER_WARP; ++n)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[kk][n][i] = 0;

        #pragma unroll
        for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
            const int k_off = k_iter * K_TILE;
            uint32_t a0, a1, a2, a3;
            {
                const int8_t* p0 = &q_smem[gid    ][k_off + tid_in_g * 4 +  0];
                const int8_t* p1 = &q_smem[gid + 8][k_off + tid_in_g * 4 +  0];
                const int8_t* p2 = &q_smem[gid    ][k_off + tid_in_g * 4 + 16];
                const int8_t* p3 = &q_smem[gid + 8][k_off + tid_in_g * 4 + 16];
                a0 = *reinterpret_cast<const uint32_t*>(p0);
                a1 = *reinterpret_cast<const uint32_t*>(p1);
                a2 = *reinterpret_cast<const uint32_t*>(p2);
                a3 = *reinterpret_cast<const uint32_t*>(p3);
            }
            const int byte_idx_b0_unaligned = (k_off >> 1) + (tid_in_g << 1);
            const int byte_idx_b0_aligned   = byte_idx_b0_unaligned & ~3;
            const int byte_idx_b1_aligned   = byte_idx_b0_aligned + 8;
            const int shift_within_u32      = (tid_in_g & 1) * 16;

            auto extract_4_int8 = [](uint16_t u16) -> uint32_t {
                const int n0 = ((int)(u16 >> 4)  & 0xF) - 8;
                const int n1 = ((int)(u16)       & 0xF) - 8;
                const int n2 = ((int)(u16 >> 12) & 0xF) - 8;
                const int n3 = ((int)(u16 >> 8)  & 0xF) - 8;
                return ((uint32_t)(uint8_t)(int8_t)n0)
                     | (((uint32_t)(uint8_t)(int8_t)n1) <<  8)
                     | (((uint32_t)(uint8_t)(int8_t)n2) << 16)
                     | (((uint32_t)(uint8_t)(int8_t)n3) << 24);
            };

            #pragma unroll
            for (int n = 0; n < N_TILES_PER_WARP; ++n) {
                const int n_off = n_start + n * 8;
                const int col = n_off + gid;
                const uint32_t u32_b0 = *reinterpret_cast<const uint32_t*>(
                    &k_pack[stage][col][byte_idx_b0_aligned]);
                const uint32_t u32_b1 = *reinterpret_cast<const uint32_t*>(
                    &k_pack[stage][col][byte_idx_b1_aligned]);
                uint16_t u16_b0 = (uint16_t)(u32_b0 >> shift_within_u32);
                uint16_t u16_b1 = (uint16_t)(u32_b1 >> shift_within_u32);
                uint32_t b0 = extract_4_int8(u16_b0);
                uint32_t b1 = extract_4_int8(u16_b1);
                mma_m16n8k32_s8s8s32(
                    a0, a1, a2, a3, b0, b1,
                    acc[k_iter][n][0], acc[k_iter][n][1],
                    acc[k_iter][n][2], acc[k_iter][n][3]);
            }
        }

        #pragma unroll
        for (int n = 0; n < N_TILES_PER_WARP; ++n) {
            const int n_off = n_start + n * 8;
            const int m0 = gid;
            const int m1 = gid + 8;
            const int c0 = n_off + tid_in_g * 2 + 0;
            const int c1 = n_off + tid_in_g * 2 + 1;
            const float ks0 = __bfloat162float(k_scale_smem[stage][c0]);
            const float ks1 = __bfloat162float(k_scale_smem[stage][c1]);

            float s00 = 0.f, s01 = 0.f, s10 = 0.f, s11 = 0.f;
            #pragma unroll
            for (int kk = 0; kk < K_ITERS; ++kk) {
                const float qs0 = q_row_scale[m0 * K_ITERS + kk];
                const float qs1 = q_row_scale[m1 * K_ITERS + kk];
                s00 += (float)acc[kk][n][0] * qs0;
                s01 += (float)acc[kk][n][1] * qs0;
                s10 += (float)acc[kk][n][2] * qs1;
                s11 += (float)acc[kk][n][3] * qs1;
            }
            score_smem[m0][c0] = s00 * ks0 * attn_scale;
            score_smem[m0][c1] = s01 * ks1 * attn_scale;
            score_smem[m1][c0] = s10 * ks0 * attn_scale;
            score_smem[m1][c1] = s11 * ks1 * attn_scale;
        }
        __syncthreads();

        if (warp_id < num_groups) {
            float tile_max = -INFINITY;
            for (int c = lane; c < valid_cols; c += 32) {
                tile_max = fmaxf(tile_max, score_smem[warp_id][c]);
            }
            tile_max = warp_max(tile_max);
            if (valid_cols > 0) {
                float tile_sum = 0.0f;
                for (int c = lane; c < valid_cols; c += 32) {
                    tile_sum += __expf(score_smem[warp_id][c] - tile_max);
                }
                tile_sum = warp_sum(tile_sum);
                if (!isfinite(row_m)) {
                    row_m = tile_max;
                    row_l = tile_sum;
                } else {
                    float new_m = fmaxf(row_m, tile_max);
                    row_l = row_l * __expf(row_m - new_m)
                          + tile_sum * __expf(tile_max - new_m);
                    row_m = new_m;
                }
            }
            for (int chunk = 0; chunk < (valid_cols + 31) / 32; ++chunk) {
                int c = chunk * 32 + lane;
                float new_key = (c < valid_cols) ? score_smem[warp_id][c] : -INFINITY;
                int new_idx = (c < valid_cols) ? (tile_s0 + c) : -1;
                update_topk_chunk(new_key, new_idx);
                ++filled_chunks;
            }
        }

        const int valid_rows = num_groups;
        const int total = valid_rows * tile_cols;
        for (int i = tid; i < total; i += THREADS) {
            int r = i / tile_cols;
            int c = i % tile_cols;
            int h_q = h_kv * num_groups + r;
            int s_glob = tile_s0 + c;
            size_t out_off = ((size_t)b * H_q + h_q) * max_S + s_glob;
            Scores[out_off] = (c < valid_cols) ? score_smem[r][c] : -INFINITY;
        }

        if (tile_idx + 1 < num_tiles) {
            issue_load(stage ^ 1, tile_idx + 1);
            cp_async_wait_group_0();
        }
        __syncthreads();
    }

    if (warp_id < num_groups) {
        const int h_q = h_kv * num_groups + warp_id;
        const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;
        if (lane == 0) {
            softmax_m[bh] = isfinite(row_m) ? row_m : 0.0f;
            softmax_inv_z[bh] = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
        }
        const size_t topk_off = bh * (size_t)K_T;
        #pragma unroll
        for (int i = 0; i < K_PER_LANE; ++i) {
            int rank = lane * K_PER_LANE + i;
            int64_t orig_idx = (int64_t)row_topk_vals[i];
            topk_idx[topk_off + (size_t)rank] = orig_idx;
            int64_t slot = -1;
            if (orig_idx >= 0 && orig_idx < (int64_t)S && orig_idx < (int64_t)max_S) {
                const int64_t block_id = orig_idx / (int64_t)block_size;
                const int64_t in_block = orig_idx % (int64_t)block_size;
                if (block_id < (int64_t)block_table_stride) {
                    const int64_t block = bt_row[block_id];
                    slot = (block >= 0) ? block * (int64_t)block_size + in_block : -1;
                }
            }
            topk_slots[topk_off + (size_t)rank] = slot;
        }
    }
}


template <int K_T, int GROUP_TILES_T>
__global__ void int8tc_qk_scores_topk_stats_group_fused_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const uint8_t*       __restrict__ K_data,
    const __nv_bfloat16* __restrict__ K_scales,
    const int64_t*       __restrict__ BlockTable,
    const int32_t*       __restrict__ SeqLens,
    float*               __restrict__ Scores,
    int64_t*             __restrict__ group_topk_idx,
    float*               __restrict__ group_m,
    float*               __restrict__ group_sum,
    int    max_S, int H_q, int H_kv,
    int    block_size, int block_table_stride,
    int    K_data_stride_h,
    int    K_scales_stride_h,
    float  attn_scale,
    int    n_groups)
{
    static_assert(K_T % 32 == 0, "top-k must be warp-distributed");
    constexpr int GROUP_SPAN = GROUP_TILES_T * BLOCK_S;

    const int b       = blockIdx.x;
    const int h_kv    = blockIdx.y;
    const int group_i = blockIdx.z;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int num_groups = H_q / H_kv;
    const int S          = SeqLens[b];
    const int group_s0   = group_i * GROUP_SPAN;
    const int group_s1   = min(group_s0 + GROUP_SPAN, max_S);
    const int first_tile = group_s0 / BLOCK_S;
    const int num_tiles_local = max(0, (group_s1 - group_s0 + BLOCK_S - 1) / BLOCK_S);

    extern __shared__ __align__(16) uint8_t dyn_smem[];
    size_t off = 0;
    auto q_smem = reinterpret_cast<int8_t(*)[Q_STRIDE_B]>(&dyn_smem[off]);
    off += (size_t)M_TILE * Q_STRIDE_B;
    auto q_row_scale = reinterpret_cast<float*>(&dyn_smem[off]);
    off += (size_t)M_TILE * K_ITERS * sizeof(float);
    off = (off + 15) & ~size_t(15);
    auto k_pack = reinterpret_cast<uint8_t(*)[BLOCK_S][K_PACK_STRIDE]>(&dyn_smem[off]);
    off += (size_t)STAGES * BLOCK_S * K_PACK_STRIDE;
    auto k_scale_smem = reinterpret_cast<__nv_bfloat16(*)[BLOCK_S]>(&dyn_smem[off]);
    off += (size_t)STAGES * BLOCK_S * sizeof(__nv_bfloat16);
    off = (off + 15) & ~size_t(15);
    auto score_smem = reinterpret_cast<float(*)[SCORE_STRIDE]>(&dyn_smem[off]);
    off += (size_t)M_TILE * SCORE_STRIDE * sizeof(float);
    auto group_score_smem = reinterpret_cast<float(*)[GROUP_SPAN]>(&dyn_smem[off]);

    const uint8_t*       k_pack_base = K_data   + (size_t)h_kv * K_data_stride_h;
    const __nv_bfloat16* ks_base     = K_scales + (size_t)h_kv * K_scales_stride_h;
    const int64_t*       bt_row      = BlockTable + (size_t)b * block_table_stride;
    __shared__ int      warp_hist[4][256];
    __shared__ int      warp_cutoff_bin[4];
    __shared__ int      warp_k_above[4];
    __shared__ int      warp_k_remaining[4];
    __shared__ uint32_t warp_prefix[4];
    __shared__ int      warp_prefix_len[4];
    __shared__ int      warp_emit_pos[4];

    auto issue_load_abs_tile = [&](int stage, int abs_tile_idx) {
        const int tile_s0_local = abs_tile_idx * BLOCK_S;
        const int tile_end_local = min(tile_s0_local + BLOCK_S, S);
        constexpr int N_LOAD_ITERS = (BLOCK_S * D_HALF) / (THREADS * 16);
        #pragma unroll
        for (int it = 0; it < N_LOAD_ITERS; ++it) {
            int lin = it * THREADS + tid;
            int row   = lin >> 2;
            int chunk = lin & 3;
            if (row < BLOCK_S) {
                int pos = tile_s0_local + row;
                int64_t slot = -1;
                if (pos < tile_end_local) {
                    slot = bt_row[pos / block_size] * (int64_t)block_size
                         + (pos % block_size);
                }
                uint8_t* dst = &k_pack[stage][row][chunk * 16];
                if (slot >= 0) {
                    const uint8_t* src = k_pack_base + slot * (int64_t)D_HALF + chunk * 16;
                    cp_async16(dst, src);
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) dst[i] = 0;
                }
            }
        }
        if (tid < BLOCK_S) {
            int pos = tile_s0_local + tid;
            int64_t slot = -1;
            if (pos < tile_end_local) {
                slot = bt_row[pos / block_size] * (int64_t)block_size
                     + (pos % block_size);
            }
            k_scale_smem[stage][tid] = (slot >= 0) ? ks_base[slot] : __float2bfloat16(0.0f);
        }
        cp_async_commit();
    };

    if (num_tiles_local > 0) {
        issue_load_abs_tile(0, first_tile);
    }

    for (int i = tid; i < (M_TILE - num_groups) * Q_STRIDE_B; i += THREADS) {
        int r = num_groups + i / Q_STRIDE_B;
        int c = i % Q_STRIDE_B;
        q_smem[r][c] = 0;
    }
    for (int i = tid; i < (M_TILE - num_groups) * K_ITERS; i += THREADS) {
        q_row_scale[(num_groups + i / K_ITERS) * K_ITERS + (i % K_ITERS)] = 0.f;
    }
    if (num_groups <= 4) {
        if (warp_id < num_groups) {
            const int r = warp_id;
            const int h_q = h_kv * num_groups + r;
            const __nv_bfloat16* qrow =
                Q + (size_t)b * H_q * D + (size_t)h_q * D;
            #pragma unroll
            for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
                const int d = k_iter * K_TILE + lane;
                float f = __bfloat162float(qrow[d]);
                float v = fabsf(f);
                v = warp_max(v);
                const float q_scale = fmaxf(v / 127.f, 1e-12f);
                if (lane == 0) q_row_scale[r * K_ITERS + k_iter] = q_scale;
                q_smem[r][d] = (int8_t)max(-128, min(127, __float2int_rn(f / q_scale)));
            }
        }
    } else {
        if (tid == 0) {
            for (int r = 0; r < num_groups; ++r) {
                const int h_q = h_kv * num_groups + r;
                const __nv_bfloat16* qrow =
                    Q + (size_t)b * H_q * D + (size_t)h_q * D;
                for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
                    float row_max = 0.f;
                    for (int dd = 0; dd < K_TILE; ++dd) {
                        float f = __bfloat162float(qrow[k_iter * K_TILE + dd]);
                        row_max = fmaxf(row_max, fabsf(f));
                    }
                    const float q_scale = fmaxf(row_max / 127.f, 1e-12f);
                    q_row_scale[r * K_ITERS + k_iter] = q_scale;
                    const float inv = 1.f / q_scale;
                    for (int dd = 0; dd < K_TILE; ++dd) {
                        const int d = k_iter * K_TILE + dd;
                        q_smem[r][d] = (int8_t)max(-128, min(127, __float2int_rn(__bfloat162float(qrow[d]) * inv)));
                    }
                }
            }
        }
    }

    cp_async_wait_group_0();
    __syncthreads();

    float row_m = -INFINITY;
    float row_l = 0.0f;

    const int gid = lane >> 2;
    const int tid_in_g = lane & 3;
    const int n_start = warp_id * (N_TILES_PER_WARP * 8);

    for (int tile_local = 0; tile_local < num_tiles_local; ++tile_local) {
        const int stage = tile_local & 1;
        const int abs_tile = first_tile + tile_local;
        const int tile_s0 = abs_tile * BLOCK_S;
        const int tile_end_S = min(tile_s0 + BLOCK_S, S);
        const int tile_end_M = min(tile_s0 + BLOCK_S, max_S);
        const int valid_cols = max(0, tile_end_S - tile_s0);
        const int tile_cols = max(0, tile_end_M - tile_s0);

        int32_t acc[K_ITERS][N_TILES_PER_WARP][4];
        #pragma unroll
        for (int kk = 0; kk < K_ITERS; ++kk)
        #pragma unroll
        for (int n = 0; n < N_TILES_PER_WARP; ++n)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[kk][n][i] = 0;

        #pragma unroll
        for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
            const int k_off = k_iter * K_TILE;
            uint32_t a0, a1, a2, a3;
            {
                const int8_t* p0 = &q_smem[gid    ][k_off + tid_in_g * 4 +  0];
                const int8_t* p1 = &q_smem[gid + 8][k_off + tid_in_g * 4 +  0];
                const int8_t* p2 = &q_smem[gid    ][k_off + tid_in_g * 4 + 16];
                const int8_t* p3 = &q_smem[gid + 8][k_off + tid_in_g * 4 + 16];
                a0 = *reinterpret_cast<const uint32_t*>(p0);
                a1 = *reinterpret_cast<const uint32_t*>(p1);
                a2 = *reinterpret_cast<const uint32_t*>(p2);
                a3 = *reinterpret_cast<const uint32_t*>(p3);
            }
            const int byte_idx_b0_unaligned = (k_off >> 1) + (tid_in_g << 1);
            const int byte_idx_b0_aligned   = byte_idx_b0_unaligned & ~3;
            const int byte_idx_b1_aligned   = byte_idx_b0_aligned + 8;
            const int shift_within_u32      = (tid_in_g & 1) * 16;
            auto extract_4_int8 = [](uint16_t u16) -> uint32_t {
                const int n0 = ((int)(u16 >> 4)  & 0xF) - 8;
                const int n1 = ((int)(u16)       & 0xF) - 8;
                const int n2 = ((int)(u16 >> 12) & 0xF) - 8;
                const int n3 = ((int)(u16 >> 8)  & 0xF) - 8;
                return ((uint32_t)(uint8_t)(int8_t)n0)
                     | (((uint32_t)(uint8_t)(int8_t)n1) <<  8)
                     | (((uint32_t)(uint8_t)(int8_t)n2) << 16)
                     | (((uint32_t)(uint8_t)(int8_t)n3) << 24);
            };

            #pragma unroll
            for (int n = 0; n < N_TILES_PER_WARP; ++n) {
                const int n_off = n_start + n * 8;
                const int col = n_off + gid;
                const uint32_t u32_b0 = *reinterpret_cast<const uint32_t*>(
                    &k_pack[stage][col][byte_idx_b0_aligned]);
                const uint32_t u32_b1 = *reinterpret_cast<const uint32_t*>(
                    &k_pack[stage][col][byte_idx_b1_aligned]);
                uint16_t u16_b0 = (uint16_t)(u32_b0 >> shift_within_u32);
                uint16_t u16_b1 = (uint16_t)(u32_b1 >> shift_within_u32);
                uint32_t b0 = extract_4_int8(u16_b0);
                uint32_t b1 = extract_4_int8(u16_b1);
                mma_m16n8k32_s8s8s32(
                    a0, a1, a2, a3, b0, b1,
                    acc[k_iter][n][0], acc[k_iter][n][1],
                    acc[k_iter][n][2], acc[k_iter][n][3]);
            }
        }

        #pragma unroll
        for (int n = 0; n < N_TILES_PER_WARP; ++n) {
            const int n_off = n_start + n * 8;
            const int m0 = gid;
            const int m1 = gid + 8;
            const int c0 = n_off + tid_in_g * 2 + 0;
            const int c1 = n_off + tid_in_g * 2 + 1;
            const float ks0 = __bfloat162float(k_scale_smem[stage][c0]);
            const float ks1 = __bfloat162float(k_scale_smem[stage][c1]);
            float s00 = 0.f, s01 = 0.f, s10 = 0.f, s11 = 0.f;
            #pragma unroll
            for (int kk = 0; kk < K_ITERS; ++kk) {
                const float qs0 = q_row_scale[m0 * K_ITERS + kk];
                const float qs1 = q_row_scale[m1 * K_ITERS + kk];
                s00 += (float)acc[kk][n][0] * qs0;
                s01 += (float)acc[kk][n][1] * qs0;
                s10 += (float)acc[kk][n][2] * qs1;
                s11 += (float)acc[kk][n][3] * qs1;
            }
            score_smem[m0][c0] = s00 * ks0 * attn_scale;
            score_smem[m0][c1] = s01 * ks1 * attn_scale;
            score_smem[m1][c0] = s10 * ks0 * attn_scale;
            score_smem[m1][c1] = s11 * ks1 * attn_scale;
        }
        __syncthreads();

        if (warp_id < num_groups) {
            float tile_max = -INFINITY;
            for (int c = lane; c < valid_cols; c += 32) {
                float s = score_smem[warp_id][c];
                group_score_smem[warp_id][tile_local * BLOCK_S + c] = s;
                tile_max = fmaxf(tile_max, s);
            }
            tile_max = warp_max(tile_max);
            if (valid_cols > 0) {
                float tile_sum = 0.0f;
                for (int c = lane; c < valid_cols; c += 32) {
                    tile_sum += __expf(score_smem[warp_id][c] - tile_max);
                }
                tile_sum = warp_sum(tile_sum);
                if (!isfinite(row_m)) {
                    row_m = tile_max;
                    row_l = tile_sum;
                } else {
                    float new_m = fmaxf(row_m, tile_max);
                    row_l = row_l * __expf(row_m - new_m)
                          + tile_sum * __expf(tile_max - new_m);
                    row_m = new_m;
                }
            }
        }

        const int valid_rows = num_groups;
        const int total = valid_rows * tile_cols;
        for (int i = tid; i < total; i += THREADS) {
            int r = i / tile_cols;
            int c = i % tile_cols;
            int h_q = h_kv * num_groups + r;
            int s_glob = tile_s0 + c;
            size_t out_off = ((size_t)b * H_q + h_q) * max_S + s_glob;
            Scores[out_off] = (c < valid_cols) ? score_smem[r][c] : -INFINITY;
        }

        if (tile_local + 1 < num_tiles_local) {
            issue_load_abs_tile(stage ^ 1, first_tile + tile_local + 1);
            cp_async_wait_group_0();
        }
        __syncthreads();
    }

    if (warp_id < num_groups) {
        const int h_q = h_kv * num_groups + warp_id;
        const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;
        const size_t out = bh * (size_t)n_groups + (size_t)group_i;
        const int valid_count = max(0, min(GROUP_SPAN, min(max_S, S) - group_s0));
        int64_t* out_idx = group_topk_idx + out * (size_t)K_T;
        if (lane == 0) {
            group_m[out] = isfinite(row_m) ? row_m : 0.0f;
            group_sum[out] = row_l;
            warp_prefix[warp_id] = 0u;
            warp_prefix_len[warp_id] = 0;
            warp_k_remaining[warp_id] = K_T;
            warp_emit_pos[warp_id] = 0;
        }
        for (int rank = lane; rank < K_T; rank += 32) {
            out_idx[rank] = -1;
        }
        __syncwarp();

        #pragma unroll
        for (int level = 0; level < 4; ++level) {
            if (warp_k_remaining[warp_id] == 0) break;

            for (int i = lane; i < 256; i += 32) {
                warp_hist[warp_id][i] = 0;
            }
            __syncwarp();

            const int      cur_prefix_len = warp_prefix_len[warp_id];
            const uint32_t cur_prefix     = warp_prefix[warp_id];
            const int      shift_high     = 32 - cur_prefix_len;
            const int      shift_byte     = 32 - cur_prefix_len - 8;
            const uint32_t mask_high      = (cur_prefix_len == 0) ? 0u
                                            : (~((1u << shift_high) - 1u));

            for (int i = lane; i < valid_count; i += 32) {
                uint32_t r = f2r_desc(group_score_smem[warp_id][i]);
                uint32_t high = (cur_prefix_len == 0) ? 0u : ((r & mask_high) >> shift_high);
                if (high == cur_prefix) {
                    int bin = (int)((r >> shift_byte) & 0xFFu);
                    atomicAdd(&warp_hist[warp_id][bin], 1);
                }
            }
            __syncwarp();

            if (lane == 0) {
                int cum = 0;
                int cb  = 0;
                int Ka  = 0;
                for (int bin = 255; bin >= 0; --bin) {
                    int c = warp_hist[warp_id][bin];
                    if (cum + c >= warp_k_remaining[warp_id]) { cb = bin; Ka = cum; break; }
                    cum += c;
                }
                warp_cutoff_bin[warp_id] = cb;
                warp_k_above[warp_id] = Ka;
            }
            __syncwarp();

            const int cb_local = warp_cutoff_bin[warp_id];
            for (int i = lane; i < valid_count; i += 32) {
                uint32_t r = f2r_desc(group_score_smem[warp_id][i]);
                uint32_t high = (cur_prefix_len == 0) ? 0u : ((r & mask_high) >> shift_high);
                if (high == cur_prefix) {
                    int bin = (int)((r >> shift_byte) & 0xFFu);
                    if (bin > cb_local) {
                        int pos = atomicAdd(&warp_emit_pos[warp_id], 1);
                        if (pos < K_T) out_idx[pos] = (int64_t)(group_s0 + i);
                    }
                }
            }
            __syncwarp();

            if (lane == 0) {
                warp_k_remaining[warp_id] -= warp_k_above[warp_id];
                warp_prefix[warp_id] = (cur_prefix << 8) | (uint32_t)cb_local;
                warp_prefix_len[warp_id] = cur_prefix_len + 8;
            }
            __syncwarp();
        }

        if (warp_k_remaining[warp_id] > 0) {
            const uint32_t target = warp_prefix[warp_id];
            for (int i = lane; i < valid_count; i += 32) {
                uint32_t r = f2r_desc(group_score_smem[warp_id][i]);
                if (r == target) {
                    int pos = atomicAdd(&warp_emit_pos[warp_id], 1);
                    if (pos < K_T) out_idx[pos] = (int64_t)(group_s0 + i);
                }
            }
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Host launcher.
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor int8tc_qk_scores(
    torch::Tensor q,
    torch::Tensor k_data,
    torch::Tensor k_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    double  attn_scale)
{
    TORCH_CHECK(q.is_cuda() && k_data.is_cuda() && k_scales.is_cuda(),
                "all inputs must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k_data.scalar_type() == torch::kUInt8, "k_data must be uint8");
    TORCH_CHECK(k_scales.scalar_type() == torch::kBFloat16, "k_scales must be bf16");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64, "block_table must be int64");
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == D, "q must be (B, H_q, 128)");
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
    TORCH_CHECK(k_data.size(-1) == D_HALF, "k_data last dim must be D/2 = 64 for INT4");

    auto qc  = q.contiguous();
    auto kdc = k_data.contiguous();
    auto ks  = (k_scales.dim() == 3) ? k_scales.squeeze(-1).contiguous()
                                     : k_scales.contiguous();
    auto bt  = block_table.contiguous();
    auto sl  = seq_lens.contiguous();

    const int64_t B = qc.size(0);

    auto options_f32 = qc.options().dtype(torch::kFloat32);
    auto scores      = torch::empty({B, H_q, max_S}, options_f32);

    const int num_tiles = ((int)max_S + BLOCK_S - 1) / BLOCK_S;

    // SMEM layout (mirrors offsets in the kernel) — no k_smem region.
    size_t smem_bytes = 0;
    smem_bytes += (size_t)M_TILE * Q_STRIDE_B;                 // q_smem int8
    smem_bytes += (size_t)M_TILE * K_ITERS * sizeof(float);    // q_row_scale
    smem_bytes  = (smem_bytes + 15) & ~size_t(15);
    smem_bytes += (size_t)STAGES * BLOCK_S * K_PACK_STRIDE;    // k_pack (padded)
    smem_bytes += (size_t)STAGES * BLOCK_S * sizeof(__nv_bfloat16);  // k_scale
    smem_bytes  = (smem_bytes + 15) & ~size_t(15);
    smem_bytes += (size_t)M_TILE * SCORE_STRIDE * sizeof(float);

    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)num_tiles);
    dim3 block(THREADS);

    const auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(int8tc_qk_scores_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
    int8tc_qk_scores_kernel<<<grid, block, smem_bytes, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr<at::BFloat16>()),
        kdc.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        sl.data_ptr<int32_t>(),
        scores.data_ptr<float>(),
        (int)max_S, (int)H_q, (int)H_kv,
        (int)block_size, (int)bt.stride(0),
        (int)kdc.stride(0), (int)ks.stride(0),
        (float)attn_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return scores;
}


// ─────────────────────────────────────────────────────────────────────────
// Top-K + slot-translate post-pass — copied VERBATIM from int4_qk_scores.cu
// (bit-width agnostic, operates on fp32 scores).
// ─────────────────────────────────────────────────────────────────────────

constexpr int TOPK_BLOCK_THREADS = 128;
constexpr int GROUP_TILES = 8;
constexpr int GROUP_S = GROUP_TILES * BLOCK_S;  // 1024
constexpr int FUSED_GROUP_TILES = 8;
constexpr int FUSED_GROUP_S = FUSED_GROUP_TILES * BLOCK_S;  // 1024

__device__ __forceinline__ int64_t translate_topk_slot_or_invalid(
    int64_t orig_idx,
    int valid_S,
    const int64_t* __restrict__ block_table,
    int b,
    int max_blocks,
    int bs)
{
    if (orig_idx < 0 || orig_idx >= (int64_t)valid_S) {
        return -1;
    }
    const int64_t block_id = orig_idx / (int64_t)bs;
    const int64_t in_block = orig_idx % (int64_t)bs;
    if (block_id < 0 || block_id >= (int64_t)max_blocks) {
        return -1;
    }
    const int64_t block = block_table[
        (size_t)b * (size_t)max_blocks + (size_t)block_id];
    return (block >= 0) ? block * (int64_t)bs + in_block : -1;
}


template <int POW2_MAX_S, int N_WARPS_T>
__global__ void topk_slots_kernel_v3(
    const float*   __restrict__ scores,
    const int32_t* __restrict__ seq_lens,
    const int64_t* __restrict__ block_table,
    int max_S, int max_blocks, int H_q, int K, int bs,
    int64_t*       __restrict__ topk_idx,
    int64_t*       __restrict__ topk_slots)
{
    constexpr int N_WARPS     = N_WARPS_T;
    constexpr int LANES       = 32;
    constexpr int THREADS     = N_WARPS * LANES;
    constexpr int PER_WARP    = POW2_MAX_S / N_WARPS;
    constexpr int K_MAX       = 128;
    constexpr int K_PER_LANE  = K_MAX / LANES;
    constexpr int CAND_TOTAL  = N_WARPS * K_MAX;
    constexpr int FINAL_ITEMS = CAND_TOTAL / THREADS;

    using Sort = cub::BlockRadixSort<float, THREADS, FINAL_ITEMS, int>;
    __shared__ typename Sort::TempStorage cub_tmp;
    __shared__ float merge_keys[CAND_TOTAL];
    __shared__ int   merge_vals[CAND_TOTAL];

    const int b       = blockIdx.x;
    const int h_q     = blockIdx.y;
    const int tid     = threadIdx.x;
    const int lane    = tid & (LANES - 1);
    const int warp_id = tid >> 5;

    const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;
    const int S = seq_lens[b];
    const int valid_S = min(max_S, S);
    const float* score_row = scores + bh * (size_t)max_S;

    float keys[K_PER_LANE];
    int   vals[K_PER_LANE];
    #pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        keys[i] = -INFINITY;
        vals[i] = -1;
    }

    constexpr int FILL_CHUNKS = K_PER_LANE;
    #pragma unroll
    for (int fc = 0; fc < FILL_CHUNKS; ++fc) {
        int chunk_off = fc * LANES;
        int idx = warp_id * PER_WARP + chunk_off + lane;
        bool valid = (idx < max_S) && (idx < S);
        keys[fc] = valid ? score_row[idx] : -INFINITY;
        vals[fc] = valid ? idx : -1;
    }

    for (int chunk_off = FILL_CHUNKS * LANES; chunk_off < PER_WARP; chunk_off += LANES) {
        int idx = warp_id * PER_WARP + chunk_off + lane;
        bool valid = (idx < max_S) && (idx < S);
        float new_key = valid ? score_row[idx] : -INFINITY;
        int   new_idx = idx;

        while (true) {
            float my_min = keys[0];
            int   my_min_sub = 0;
            #pragma unroll
            for (int i = 1; i < K_PER_LANE; ++i) {
                if (keys[i] < my_min) { my_min = keys[i]; my_min_sub = i; }
            }

            float global_min      = my_min;
            int   global_min_lane = lane;
            int   global_min_sub  = my_min_sub;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float other_min  = __shfl_xor_sync(0xFFFFFFFFu, global_min, off);
                int   other_lane = __shfl_xor_sync(0xFFFFFFFFu, global_min_lane, off);
                int   other_sub  = __shfl_xor_sync(0xFFFFFFFFu, global_min_sub, off);
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
            int   picked_idx = __shfl_sync(0xFFFFFFFFu, new_idx, pick_lane);

            if (lane == global_min_lane) {
                keys[global_min_sub] = picked_key;
                vals[global_min_sub] = picked_idx;
            }
            if (lane == pick_lane) {
                new_key = -INFINITY;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        int slot = warp_id * K_MAX + lane * K_PER_LANE + i;
        merge_keys[slot] = keys[i];
        merge_vals[slot] = vals[i];
    }
    __syncthreads();

    float thread_keys[FINAL_ITEMS];
    int   thread_vals[FINAL_ITEMS];
    #pragma unroll
    for (int i = 0; i < FINAL_ITEMS; ++i) {
        int cand_idx = tid * FINAL_ITEMS + i;
        if (cand_idx < CAND_TOTAL) {
            thread_keys[i] = merge_keys[cand_idx];
            thread_vals[i] = merge_vals[cand_idx];
        } else {
            thread_keys[i] = -INFINITY;
            thread_vals[i] = -1;
        }
    }
    __syncthreads();

    Sort(cub_tmp).SortDescending(thread_keys, thread_vals);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < FINAL_ITEMS; ++i) {
        int rank = tid * FINAL_ITEMS + i;
        if (rank < K) {
            int64_t orig_idx = (int64_t)thread_vals[i];
            topk_idx[bh * (size_t)K + rank] = orig_idx;
            topk_slots[bh * (size_t)K + rank] =
                translate_topk_slot_or_invalid(
                    orig_idx, valid_S, block_table, b, max_blocks, bs);
        }
    }
}


__device__ __forceinline__ float block_reduce_max_v5(float v, float* smem) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 16) ? smem[lane] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, off));
        }
        if (lane == 0) smem[0] = t;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_reduce_sum_v5(float v, float* smem) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 16) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            t += __shfl_xor_sync(0xffffffffu, t, off);
        }
        if (lane == 0) smem[0] = t;
    }
    __syncthreads();
    return smem[0];
}

template <int N_WARPS_T, int K_T>
__global__ void topk_slots_kernel_v5(
    const float*   __restrict__ scores,
    const int32_t* __restrict__ seq_lens,
    const int64_t* __restrict__ block_table,
    int max_S, int max_blocks, int H_q, int bs,
    int64_t*       __restrict__ topk_idx,
    int64_t*       __restrict__ topk_slots)
{
    constexpr int K       = K_T;
    constexpr int N_WARPS = N_WARPS_T;
    constexpr int THREADS = N_WARPS * 32;

    __shared__ int      hist[256];
    __shared__ int      cutoff_bin;
    __shared__ int      K_above;
    __shared__ int      K_remaining;
    __shared__ uint32_t prefix;
    __shared__ int      prefix_len;
    __shared__ int      emit_pos;

    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;

    const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;
    const int S = seq_lens[b];
    const int valid_S = min(max_S, S);
    const float* score_row = scores + bh * (size_t)max_S;

    for (int rank = tid; rank < K; rank += THREADS) {
        topk_idx[bh * (size_t)K + rank] = -1;
        topk_slots[bh * (size_t)K + rank] = -1;
    }
    __syncthreads();

    if (tid == 0) {
        prefix      = 0;
        prefix_len  = 0;
        K_remaining = K;
        emit_pos    = 0;
    }
    __syncthreads();

    #pragma unroll
    for (int level = 0; level < 4; ++level) {
        if (K_remaining == 0) break;

        for (int i = tid; i < 256; i += THREADS) hist[i] = 0;
        __syncthreads();

        const int      cur_prefix_len = prefix_len;
        const uint32_t cur_prefix     = prefix;
        const int      shift_high     = 32 - cur_prefix_len;
        const int      shift_byte     = 32 - cur_prefix_len - 8;
        const uint32_t mask_high      = (cur_prefix_len == 0) ? 0u
                                        : (~((1u << shift_high) - 1u));

        for (int i = tid; i < valid_S; i += THREADS) {
            uint32_t r = f2r_desc(score_row[i]);
            uint32_t high = (cur_prefix_len == 0) ? 0u : ((r & mask_high) >> shift_high);
            if (high == cur_prefix) {
                int bin = (int)((r >> shift_byte) & 0xFFu);
                atomicAdd(&hist[bin], 1);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int cum = 0;
            int cb  = 0;
            int Ka  = 0;
            for (int bin = 255; bin >= 0; --bin) {
                int c = hist[bin];
                if (cum + c >= K_remaining) { cb = bin; Ka = cum; break; }
                cum += c;
            }
            cutoff_bin = cb;
            K_above    = Ka;
        }
        __syncthreads();

        const int cb_local = cutoff_bin;
        for (int i = tid; i < valid_S; i += THREADS) {
            uint32_t r = f2r_desc(score_row[i]);
            uint32_t high = (cur_prefix_len == 0) ? 0u : ((r & mask_high) >> shift_high);
            if (high == cur_prefix) {
                int bin = (int)((r >> shift_byte) & 0xFFu);
                if (bin > cb_local) {
                    int pos = atomicAdd(&emit_pos, 1);
                    if (pos < K) {
                        topk_idx[bh * (size_t)K + pos] = (int64_t)i;
                    }
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            K_remaining -= K_above;
            prefix       = (cur_prefix << 8) | (uint32_t)cb_local;
            prefix_len   = cur_prefix_len + 8;
        }
        __syncthreads();
    }

    if (K_remaining > 0) {
        const uint32_t target = prefix;
        for (int i = tid; i < valid_S; i += THREADS) {
            uint32_t r = f2r_desc(score_row[i]);
            if (r == target) {
                int pos = atomicAdd(&emit_pos, 1);
                if (pos < K) {
                    topk_idx[bh * (size_t)K + pos] = (int64_t)i;
                }
            }
        }
        __syncthreads();
    }

    for (int rank = tid; rank < K; rank += THREADS) {
        int64_t orig_idx = topk_idx[bh * (size_t)K + rank];
        topk_slots[bh * (size_t)K + rank] =
            translate_topk_slot_or_invalid(
                orig_idx, valid_S, block_table, b, max_blocks, bs);
    }
}

template <int N_WARPS_T, int K_T>
__global__ void topk_slots_stats_kernel_v5(
    const float*   __restrict__ scores,
    const int32_t* __restrict__ seq_lens,
    const int64_t* __restrict__ block_table,
    int max_S, int max_blocks, int H_q, int bs,
    int64_t*       __restrict__ topk_idx,
    int64_t*       __restrict__ topk_slots,
    float*         __restrict__ softmax_m,
    float*         __restrict__ softmax_inv_z)
{
    constexpr int K       = K_T;
    constexpr int N_WARPS = N_WARPS_T;
    constexpr int THREADS = N_WARPS * 32;

    __shared__ int      hist[256];
    __shared__ int      cutoff_bin;
    __shared__ int      K_above;
    __shared__ int      K_remaining;
    __shared__ uint32_t prefix;
    __shared__ int      prefix_len;
    __shared__ int      emit_pos;
    __shared__ float    red_buf[16];

    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;

    const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;
    const int S = seq_lens[b];
    const int valid_S = min(max_S, S);
    const float* score_row = scores + bh * (size_t)max_S;

    for (int rank = tid; rank < K; rank += THREADS) {
        topk_idx[bh * (size_t)K + rank] = -1;
        topk_slots[bh * (size_t)K + rank] = -1;
    }
    __syncthreads();

    if (tid == 0) {
        prefix      = 0;
        prefix_len  = 0;
        K_remaining = K;
        emit_pos    = 0;
    }
    __syncthreads();

    float local_max = -INFINITY;
    #pragma unroll
    for (int level = 0; level < 4; ++level) {
        if (K_remaining == 0) break;

        for (int i = tid; i < 256; i += THREADS) hist[i] = 0;
        __syncthreads();

        const int      cur_prefix_len = prefix_len;
        const uint32_t cur_prefix     = prefix;
        const int      shift_high     = 32 - cur_prefix_len;
        const int      shift_byte     = 32 - cur_prefix_len - 8;
        const uint32_t mask_high      = (cur_prefix_len == 0) ? 0u
                                        : (~((1u << shift_high) - 1u));

        for (int i = tid; i < valid_S; i += THREADS) {
            float s = score_row[i];
            if (level == 0) local_max = fmaxf(local_max, s);
            uint32_t r = f2r_desc(s);
            uint32_t high = (cur_prefix_len == 0) ? 0u : ((r & mask_high) >> shift_high);
            if (high == cur_prefix) {
                int bin = (int)((r >> shift_byte) & 0xFFu);
                atomicAdd(&hist[bin], 1);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int cum = 0;
            int cb  = 0;
            int Ka  = 0;
            for (int bin = 255; bin >= 0; --bin) {
                int c = hist[bin];
                if (cum + c >= K_remaining) { cb = bin; Ka = cum; break; }
                cum += c;
            }
            cutoff_bin = cb;
            K_above    = Ka;
        }
        __syncthreads();

        const int cb_local = cutoff_bin;
        for (int i = tid; i < valid_S; i += THREADS) {
            uint32_t r = f2r_desc(score_row[i]);
            uint32_t high = (cur_prefix_len == 0) ? 0u : ((r & mask_high) >> shift_high);
            if (high == cur_prefix) {
                int bin = (int)((r >> shift_byte) & 0xFFu);
                if (bin > cb_local) {
                    int pos = atomicAdd(&emit_pos, 1);
                    if (pos < K) {
                        topk_idx[bh * (size_t)K + pos] = (int64_t)i;
                    }
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            K_remaining -= K_above;
            prefix       = (cur_prefix << 8) | (uint32_t)cb_local;
            prefix_len   = cur_prefix_len + 8;
        }
        __syncthreads();
    }

    float m = block_reduce_max_v5(local_max, red_buf);
    if (!isfinite(m)) m = 0.0f;
    float local_sum = 0.0f;
    for (int i = tid; i < valid_S; i += THREADS) {
        local_sum += __expf(score_row[i] - m);
    }
    float z = block_reduce_sum_v5(local_sum, red_buf);
    if (tid == 0) {
        softmax_m[bh] = m;
        softmax_inv_z[bh] = (z > 0.0f) ? (1.0f / z) : 0.0f;
    }
    __syncthreads();

    if (K_remaining > 0) {
        const uint32_t target = prefix;
        for (int i = tid; i < valid_S; i += THREADS) {
            uint32_t r = f2r_desc(score_row[i]);
            if (r == target) {
                int pos = atomicAdd(&emit_pos, 1);
                if (pos < K) {
                    topk_idx[bh * (size_t)K + pos] = (int64_t)i;
                }
            }
        }
        __syncthreads();
    }

    for (int rank = tid; rank < K; rank += THREADS) {
        int64_t orig_idx = topk_idx[bh * (size_t)K + rank];
        topk_slots[bh * (size_t)K + rank] =
            translate_topk_slot_or_invalid(
                orig_idx, valid_S, block_table, b, max_blocks, bs);
    }
}


template <int POW2_MAX_S, int THREADS = TOPK_BLOCK_THREADS>
__global__ void topk_slots_kernel(
    const float*   __restrict__ scores,
    const int32_t* __restrict__ seq_lens,
    const int64_t* __restrict__ block_table,
    int max_S, int max_blocks, int H_q, int K, int bs,
    int64_t*       __restrict__ topk_idx,
    int64_t*       __restrict__ topk_slots)
{
    constexpr int ITEMS = POW2_MAX_S / THREADS;

    constexpr cub::BlockLoadAlgorithm LOAD_ALGO =
        (ITEMS % 32 == 0)
            ? cub::BLOCK_LOAD_WARP_TRANSPOSE
            : cub::BLOCK_LOAD_TRANSPOSE;
    using LoadKeys = cub::BlockLoad<float, THREADS, ITEMS, LOAD_ALGO>;

    using Sort = cub::BlockRadixSort<
        float, THREADS, ITEMS, int,
        4,
        true,
        cub::BLOCK_SCAN_RAKING_MEMOIZE
    >;

    __shared__ union {
        typename LoadKeys::TempStorage load;
        typename Sort::TempStorage     sort;
    } tmp;

    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= gridDim.x || h_q >= H_q) return;

    const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;
    const int S = seq_lens[b];
    const float* score_row = scores + bh * (size_t)max_S;

    const int valid_count = min(max_S, S);
    float thread_keys[ITEMS];
    LoadKeys(tmp.load).Load(score_row, thread_keys, valid_count, -INFINITY);
    __syncthreads();

    int thread_vals[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        thread_vals[i] = tid * ITEMS + i;
    }

    Sort(tmp.sort).SortDescending(thread_keys, thread_vals);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int rank = tid * ITEMS + i;
        if (rank < K) {
            int64_t orig_idx = (int64_t)thread_vals[i];
            topk_idx[bh * (size_t)K + rank] = orig_idx;
            topk_slots[bh * (size_t)K + rank] =
                translate_topk_slot_or_invalid(
                    orig_idx, valid_count, block_table, b, max_blocks, bs);
        }
    }
}


template <int POW2_GROUP_S, int K_T, int THREADS = 512>
__global__ void group_topk_stats_kernel(
    const float*   __restrict__ scores,
    const int32_t* __restrict__ seq_lens,
    int max_S, int H_q,
    int64_t*       __restrict__ group_topk_idx,
    float*         __restrict__ group_m,
    float*         __restrict__ group_sum,
    int n_groups)
{
    constexpr int ITEMS = POW2_GROUP_S / THREADS;
    constexpr cub::BlockLoadAlgorithm LOAD_ALGO =
        (ITEMS % 32 == 0)
            ? cub::BLOCK_LOAD_WARP_TRANSPOSE
            : cub::BLOCK_LOAD_TRANSPOSE;
    using LoadKeys = cub::BlockLoad<float, THREADS, ITEMS, LOAD_ALGO>;
    using Sort = cub::BlockRadixSort<
        float, THREADS, ITEMS, int,
        4,
        true,
        cub::BLOCK_SCAN_RAKING_MEMOIZE
    >;

    __shared__ union {
        typename LoadKeys::TempStorage load;
        typename Sort::TempStorage     sort;
    } tmp;
    __shared__ float red_buf[16];

    const int b       = blockIdx.x;
    const int h_q     = blockIdx.y;
    const int group_i = blockIdx.z;
    const int tid     = threadIdx.x;

    const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;
    const int S = min(max_S, seq_lens[b]);
    const int start = group_i * POW2_GROUP_S;
    const int valid_count = max(0, min(POW2_GROUP_S, S - start));
    const float* score_row = scores + bh * (size_t)max_S + start;

    float thread_keys[ITEMS];
    LoadKeys(tmp.load).Load(score_row, thread_keys, valid_count, -INFINITY);
    __syncthreads();

    float local_max = -INFINITY;
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        local_max = fmaxf(local_max, thread_keys[i]);
    }
    float gm = block_reduce_max_v5(local_max, red_buf);
    if (!isfinite(gm)) gm = 0.0f;
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        float v = thread_keys[i];
        if (isfinite(v)) local_sum += __expf(v - gm);
    }
    float gsum = block_reduce_sum_v5(local_sum, red_buf);
    if (tid == 0) {
        const size_t out = bh * (size_t)n_groups + (size_t)group_i;
        group_m[out] = gm;
        group_sum[out] = gsum;
    }
    __syncthreads();

    int thread_vals[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        thread_vals[i] = tid * ITEMS + i;
    }

    Sort(tmp.sort).SortDescending(thread_keys, thread_vals);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int rank = tid * ITEMS + i;
        if (rank < K_T) {
            int idx_local = thread_vals[i];
            int64_t out_idx = (idx_local < valid_count)
                ? (int64_t)(start + idx_local)
                : (int64_t)-1;
            group_topk_idx[
                ((bh * (size_t)n_groups) + (size_t)group_i) * (size_t)K_T
                + (size_t)rank
            ] = out_idx;
        }
    }
}


template <int CAND_POW2, int K_T, int THREADS = 512>
__global__ void merge_group_topk_stats_kernel(
    const float*   __restrict__ scores,
    const int64_t* __restrict__ group_topk_idx,
    const float*   __restrict__ group_m,
    const float*   __restrict__ group_sum,
    const int32_t* __restrict__ seq_lens,
    const int64_t* __restrict__ block_table,
    int max_S, int max_blocks, int H_q, int bs, int n_groups,
    int64_t*       __restrict__ topk_idx,
    int64_t*       __restrict__ topk_slots,
    float*         __restrict__ softmax_m,
    float*         __restrict__ softmax_inv_z)
{
    constexpr int ITEMS = CAND_POW2 / THREADS;
    using Sort = cub::BlockRadixSort<
        float, THREADS, ITEMS, int,
        4,
        true,
        cub::BLOCK_SCAN_RAKING_MEMOIZE
    >;
    __shared__ typename Sort::TempStorage sort_tmp;
    __shared__ float red_buf[16];

    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;
    const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;
    const float* score_row = scores + bh * (size_t)max_S;

    float local_max = -INFINITY;
    for (int g = tid; g < n_groups; g += THREADS) {
        local_max = fmaxf(local_max, group_m[bh * (size_t)n_groups + (size_t)g]);
    }
    float gm = block_reduce_max_v5(local_max, red_buf);
    if (!isfinite(gm)) gm = 0.0f;

    float local_sum = 0.0f;
    for (int g = tid; g < n_groups; g += THREADS) {
        size_t off = bh * (size_t)n_groups + (size_t)g;
        local_sum += group_sum[off] * __expf(group_m[off] - gm);
    }
    float gsum = block_reduce_sum_v5(local_sum, red_buf);
    if (tid == 0) {
        softmax_m[bh] = gm;
        softmax_inv_z[bh] = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
    }
    __syncthreads();

    float thread_keys[ITEMS];
    int thread_vals[ITEMS];
    const int valid_cands = n_groups * K_T;
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int flat = tid * ITEMS + i;
        int idx = -1;
        float score = -INFINITY;
        if (flat < valid_cands) {
            idx = (int)group_topk_idx[bh * (size_t)valid_cands + (size_t)flat];
            if (idx >= 0) score = score_row[idx];
        }
        thread_keys[i] = score;
        thread_vals[i] = idx;
    }

    Sort(sort_tmp).SortDescending(thread_keys, thread_vals);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int rank = tid * ITEMS + i;
        if (rank < K_T) {
            int64_t orig_idx = (int64_t)thread_vals[i];
            topk_idx[bh * (size_t)K_T + (size_t)rank] = orig_idx;
            topk_slots[bh * (size_t)K_T + (size_t)rank] =
                translate_topk_slot_or_invalid(
                    orig_idx, max_S, block_table, b, max_blocks, bs);
        }
    }
}


template <int N_WARPS_T, int K_T>
__global__ void group_topk_stats_kernel_v5(
    const float*   __restrict__ scores,
    const int32_t* __restrict__ seq_lens,
    int max_S, int H_q, int group_span, int n_groups,
    int64_t*       __restrict__ group_topk_idx,
    float*         __restrict__ group_m,
    float*         __restrict__ group_sum)
{
    constexpr int K       = K_T;
    constexpr int N_WARPS = N_WARPS_T;
    constexpr int THREADS = N_WARPS * 32;

    __shared__ int      hist[256];
    __shared__ int      cutoff_bin;
    __shared__ int      K_above;
    __shared__ int      K_remaining;
    __shared__ uint32_t prefix;
    __shared__ int      prefix_len;
    __shared__ int      emit_pos;
    __shared__ float    red_buf[16];

    const int b       = blockIdx.x;
    const int h_q     = blockIdx.y;
    const int group_i = blockIdx.z;
    const int tid     = threadIdx.x;

    const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;
    const int S = seq_lens[b];
    const int start = group_i * group_span;
    const int valid_count = max(0, min(group_span, min(max_S, S) - start));
    const float* score_row = scores + bh * (size_t)max_S + start;
    int64_t* out_idx = group_topk_idx + ((bh * (size_t)n_groups) + (size_t)group_i) * (size_t)K;

    for (int rank = tid; rank < K; rank += THREADS) out_idx[rank] = -1;
    if (tid == 0) {
        prefix      = 0;
        prefix_len  = 0;
        K_remaining = K;
        emit_pos    = 0;
    }
    __syncthreads();

    float local_max = -INFINITY;
    #pragma unroll
    for (int level = 0; level < 4; ++level) {
        if (K_remaining == 0) break;

        for (int i = tid; i < 256; i += THREADS) hist[i] = 0;
        __syncthreads();

        const int      cur_prefix_len = prefix_len;
        const uint32_t cur_prefix     = prefix;
        const int      shift_high     = 32 - cur_prefix_len;
        const int      shift_byte     = 32 - cur_prefix_len - 8;
        const uint32_t mask_high      = (cur_prefix_len == 0) ? 0u
                                        : (~((1u << shift_high) - 1u));

        for (int i = tid; i < valid_count; i += THREADS) {
            float s = score_row[i];
            if (level == 0) local_max = fmaxf(local_max, s);
            uint32_t r = f2r_desc(s);
            uint32_t high = (cur_prefix_len == 0) ? 0u : ((r & mask_high) >> shift_high);
            if (high == cur_prefix) {
                int bin = (int)((r >> shift_byte) & 0xFFu);
                atomicAdd(&hist[bin], 1);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int cum = 0;
            int cb  = 0;
            int Ka  = 0;
            for (int bin = 255; bin >= 0; --bin) {
                int c = hist[bin];
                if (cum + c >= K_remaining) { cb = bin; Ka = cum; break; }
                cum += c;
            }
            cutoff_bin = cb;
            K_above    = Ka;
        }
        __syncthreads();

        const int cb_local = cutoff_bin;
        for (int i = tid; i < valid_count; i += THREADS) {
            uint32_t r = f2r_desc(score_row[i]);
            uint32_t high = (cur_prefix_len == 0) ? 0u : ((r & mask_high) >> shift_high);
            if (high == cur_prefix) {
                int bin = (int)((r >> shift_byte) & 0xFFu);
                if (bin > cb_local) {
                    int pos = atomicAdd(&emit_pos, 1);
                    if (pos < K) out_idx[pos] = (int64_t)(start + i);
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            K_remaining -= K_above;
            prefix       = (cur_prefix << 8) | (uint32_t)cb_local;
            prefix_len   = cur_prefix_len + 8;
        }
        __syncthreads();
    }

    float gm = block_reduce_max_v5(local_max, red_buf);
    if (!isfinite(gm)) gm = 0.0f;
    float local_sum = 0.0f;
    for (int i = tid; i < valid_count; i += THREADS) {
        local_sum += __expf(score_row[i] - gm);
    }
    float gsum = block_reduce_sum_v5(local_sum, red_buf);
    if (tid == 0) {
        group_m[bh * (size_t)n_groups + (size_t)group_i] = gm;
        group_sum[bh * (size_t)n_groups + (size_t)group_i] = gsum;
    }
    __syncthreads();

    if (K_remaining > 0) {
        const uint32_t target = prefix;
        for (int i = tid; i < valid_count; i += THREADS) {
            uint32_t r = f2r_desc(score_row[i]);
            if (r == target) {
                int pos = atomicAdd(&emit_pos, 1);
                if (pos < K) out_idx[pos] = (int64_t)(start + i);
            }
        }
    }
}


template <int N_WARPS_T, int K_T>
__global__ void merge_group_topk_stats_kernel_v5(
    const float*   __restrict__ scores,
    const int64_t* __restrict__ group_topk_idx,
    const float*   __restrict__ group_m,
    const float*   __restrict__ group_sum,
    const int64_t* __restrict__ block_table,
    int max_S, int max_blocks, int H_q, int bs, int n_groups,
    int64_t*       __restrict__ topk_idx,
    int64_t*       __restrict__ topk_slots,
    float*         __restrict__ softmax_m,
    float*         __restrict__ softmax_inv_z)
{
    constexpr int K       = K_T;
    constexpr int N_WARPS = N_WARPS_T;
    constexpr int THREADS = N_WARPS * 32;

    __shared__ int      hist[256];
    __shared__ int      cutoff_bin;
    __shared__ int      K_above;
    __shared__ int      K_remaining;
    __shared__ uint32_t prefix;
    __shared__ int      prefix_len;
    __shared__ int      emit_pos;
    __shared__ float    red_buf[16];

    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;
    const size_t bh = (size_t)b * (size_t)H_q + (size_t)h_q;

    const float* score_row = scores + bh * (size_t)max_S;
    const int valid_cands = n_groups * K;
    const int64_t* cand_row = group_topk_idx + bh * (size_t)valid_cands;
    int64_t* out_idx = topk_idx + bh * (size_t)K;
    int64_t* out_slots = topk_slots + bh * (size_t)K;

    for (int rank = tid; rank < K; rank += THREADS) out_idx[rank] = -1;
    if (tid == 0) {
        prefix      = 0;
        prefix_len  = 0;
        K_remaining = K;
        emit_pos    = 0;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int g = tid; g < n_groups; g += THREADS) {
        local_max = fmaxf(local_max, group_m[bh * (size_t)n_groups + (size_t)g]);
    }
    float gm = block_reduce_max_v5(local_max, red_buf);
    if (!isfinite(gm)) gm = 0.0f;

    float local_sum = 0.0f;
    for (int g = tid; g < n_groups; g += THREADS) {
        size_t off = bh * (size_t)n_groups + (size_t)g;
        local_sum += group_sum[off] * __expf(group_m[off] - gm);
    }
    float gsum = block_reduce_sum_v5(local_sum, red_buf);
    if (tid == 0) {
        softmax_m[bh] = gm;
        softmax_inv_z[bh] = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int level = 0; level < 4; ++level) {
        if (K_remaining == 0) break;

        for (int i = tid; i < 256; i += THREADS) hist[i] = 0;
        __syncthreads();

        const int      cur_prefix_len = prefix_len;
        const uint32_t cur_prefix     = prefix;
        const int      shift_high     = 32 - cur_prefix_len;
        const int      shift_byte     = 32 - cur_prefix_len - 8;
        const uint32_t mask_high      = (cur_prefix_len == 0) ? 0u
                                        : (~((1u << shift_high) - 1u));

        for (int i = tid; i < valid_cands; i += THREADS) {
            int idx = (int)cand_row[i];
            float s = (idx >= 0) ? score_row[idx] : -INFINITY;
            uint32_t r = f2r_desc(s);
            uint32_t high = (cur_prefix_len == 0) ? 0u : ((r & mask_high) >> shift_high);
            if (high == cur_prefix) {
                int bin = (int)((r >> shift_byte) & 0xFFu);
                atomicAdd(&hist[bin], 1);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int cum = 0;
            int cb  = 0;
            int Ka  = 0;
            for (int bin = 255; bin >= 0; --bin) {
                int c = hist[bin];
                if (cum + c >= K_remaining) { cb = bin; Ka = cum; break; }
                cum += c;
            }
            cutoff_bin = cb;
            K_above    = Ka;
        }
        __syncthreads();

        const int cb_local = cutoff_bin;
        for (int i = tid; i < valid_cands; i += THREADS) {
            int idx = (int)cand_row[i];
            float s = (idx >= 0) ? score_row[idx] : -INFINITY;
            uint32_t r = f2r_desc(s);
            uint32_t high = (cur_prefix_len == 0) ? 0u : ((r & mask_high) >> shift_high);
            if (high == cur_prefix) {
                int bin = (int)((r >> shift_byte) & 0xFFu);
                if (bin > cb_local) {
                    int pos = atomicAdd(&emit_pos, 1);
                    if (pos < K) out_idx[pos] = (int64_t)idx;
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            K_remaining -= K_above;
            prefix       = (cur_prefix << 8) | (uint32_t)cb_local;
            prefix_len   = cur_prefix_len + 8;
        }
        __syncthreads();
    }

    if (K_remaining > 0) {
        const uint32_t target = prefix;
        for (int i = tid; i < valid_cands; i += THREADS) {
            int idx = (int)cand_row[i];
            float s = (idx >= 0) ? score_row[idx] : -INFINITY;
            if (f2r_desc(s) == target) {
                int pos = atomicAdd(&emit_pos, 1);
                if (pos < K) out_idx[pos] = (int64_t)idx;
            }
        }
        __syncthreads();
    }

    for (int rank = tid; rank < K; rank += THREADS) {
        int64_t orig_idx = out_idx[rank];
        out_slots[rank] = translate_topk_slot_or_invalid(
            orig_idx, max_S, block_table, b, max_blocks, bs);
    }
}


// Wrapper: int8tc_qk_scores_topk runs Stage A + topk_slots post-pass.
std::vector<torch::Tensor> int8tc_qk_scores_topk(
    torch::Tensor q,
    torch::Tensor k_data,
    torch::Tensor k_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    double  attn_scale,
    int64_t top_k)
{
    torch::Tensor scores = int8tc_qk_scores(
        q, k_data, k_scales, block_table, seq_lens,
        max_S, H_q, H_kv, block_size, attn_scale
    );

    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();

    const int64_t B = scores.size(0);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(scores.device());
    auto topk_idx   = torch::empty({B, H_q, (int64_t)top_k}, opts_i64);
    auto topk_slots = torch::empty({B, H_q, (int64_t)top_k}, opts_i64);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_q);

    auto launch_v5 = [&](auto K_const) {
        constexpr int KV = K_const.value;
        dim3 block(16 * 32);
        topk_slots_kernel_v5<16, KV><<<grid, block, 0, stream.stream()>>>(
            scores.data_ptr<float>(), sl.data_ptr<int32_t>(),
            bt.data_ptr<int64_t>(),
            (int)max_S, (int)bt.size(1), (int)H_q, (int)block_size,
            topk_idx.data_ptr<int64_t>(), topk_slots.data_ptr<int64_t>()
        );
    };

    auto launch_small = [&](auto P_const) {
        constexpr int P = P_const.value;
        static_assert(P <= 8192, "");
        dim3 block(256);
        topk_slots_kernel<P, 256><<<grid, block, 0, stream.stream()>>>(
            scores.data_ptr<float>(), sl.data_ptr<int32_t>(),
            bt.data_ptr<int64_t>(),
            (int)max_S, (int)bt.size(1), (int)H_q, (int)top_k, (int)block_size,
            topk_idx.data_ptr<int64_t>(), topk_slots.data_ptr<int64_t>()
        );
    };

    auto launch_large = [&]() -> bool {
        const int K = (int)top_k;
        switch (K) {
            case   64: launch_v5(std::integral_constant<int,   64>{}); return true;
            case  128: launch_v5(std::integral_constant<int,  128>{}); return true;
            case  256: launch_v5(std::integral_constant<int,  256>{}); return true;
            case  512: launch_v5(std::integral_constant<int,  512>{}); return true;
            case 1024: launch_v5(std::integral_constant<int, 1024>{}); return true;
            case 2048: launch_v5(std::integral_constant<int, 2048>{}); return true;
            case 4096: launch_v5(std::integral_constant<int, 4096>{}); return true;
            case 8192: launch_v5(std::integral_constant<int, 8192>{}); return true;
            default:   return false;
        }
    };

    if      (max_S <= 4096) launch_small(std::integral_constant<int, 4096>{});
    else if (max_S <= 8192) launch_small(std::integral_constant<int, 8192>{});
    else if (launch_large())  { /* used v5 */ }
    else {
        TORCH_CHECK(false, "int8tc_qk_scores_topk: unsupported top_k=", top_k);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {scores, topk_idx, topk_slots};
}

std::vector<torch::Tensor> int8tc_qk_scores_topk_stats(
    torch::Tensor q,
    torch::Tensor k_data,
    torch::Tensor k_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    double  attn_scale,
    int64_t top_k)
{
    torch::Tensor scores = int8tc_qk_scores(
        q, k_data, k_scales, block_table, seq_lens,
        max_S, H_q, H_kv, block_size, attn_scale
    );

    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();

    const int64_t B = scores.size(0);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(scores.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(scores.device());
    auto topk_idx      = torch::empty({B, H_q, (int64_t)top_k}, opts_i64);
    auto topk_slots    = torch::empty({B, H_q, (int64_t)top_k}, opts_i64);
    auto softmax_m     = torch::empty({B, H_q}, opts_f32);
    auto softmax_inv_z = torch::empty({B, H_q}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_q);
    dim3 block(16 * 32);
    auto launch = [&](auto K_const) {
        constexpr int KV = K_const.value;
        topk_slots_stats_kernel_v5<16, KV><<<grid, block, 0, stream.stream()>>>(
            scores.data_ptr<float>(), sl.data_ptr<int32_t>(),
            bt.data_ptr<int64_t>(),
            (int)max_S, (int)bt.size(1), (int)H_q, (int)block_size,
            topk_idx.data_ptr<int64_t>(), topk_slots.data_ptr<int64_t>(),
            softmax_m.data_ptr<float>(), softmax_inv_z.data_ptr<float>()
        );
    };
    switch ((int)top_k) {
        case   64: launch(std::integral_constant<int,   64>{}); break;
        case  128: launch(std::integral_constant<int,  128>{}); break;
        case  256: launch(std::integral_constant<int,  256>{}); break;
        case  512: launch(std::integral_constant<int,  512>{}); break;
        case 1024: launch(std::integral_constant<int, 1024>{}); break;
        case 2048: launch(std::integral_constant<int, 2048>{}); break;
        case 4096: launch(std::integral_constant<int, 4096>{}); break;
        case 8192: launch(std::integral_constant<int, 8192>{}); break;
        default:
            TORCH_CHECK(false, "int8tc_qk_scores_topk_stats: unsupported top_k=", top_k);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scores, topk_idx, topk_slots, softmax_m, softmax_inv_z};
}


std::vector<torch::Tensor> int8tc_qk_scores_topk_stats_fused(
    torch::Tensor q,
    torch::Tensor k_data,
    torch::Tensor k_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    double  attn_scale,
    int64_t top_k)
{
    TORCH_CHECK((H_q / H_kv) <= 4,
                "int8tc_qk_scores_topk_stats_fused currently supports H_q/H_kv <= 4 only");
    TORCH_CHECK(block_size == 16, "only block_size=16 supported");

    auto qc  = q.contiguous();
    auto kdc = k_data.contiguous();
    auto ks  = (k_scales.dim() == 3) ? k_scales.squeeze(-1).contiguous()
                                     : k_scales.contiguous();
    auto bt  = block_table.contiguous();
    auto sl  = seq_lens.contiguous();

    const int64_t B = qc.size(0);
    auto opts_f32 = qc.options().dtype(torch::kFloat32);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(qc.device());
    auto scores        = torch::empty({B, H_q, max_S}, opts_f32);
    auto topk_idx      = torch::empty({B, H_q, top_k}, opts_i64);
    auto topk_slots    = torch::empty({B, H_q, top_k}, opts_i64);
    auto softmax_m     = torch::empty({B, H_q}, opts_f32);
    auto softmax_inv_z = torch::empty({B, H_q}, opts_f32);

    size_t smem_bytes = 0;
    smem_bytes += (size_t)M_TILE * Q_STRIDE_B;
    smem_bytes += (size_t)M_TILE * K_ITERS * sizeof(float);
    smem_bytes  = (smem_bytes + 15) & ~size_t(15);
    smem_bytes += (size_t)STAGES * BLOCK_S * K_PACK_STRIDE;
    smem_bytes += (size_t)STAGES * BLOCK_S * sizeof(__nv_bfloat16);
    smem_bytes  = (smem_bytes + 15) & ~size_t(15);
    smem_bytes += (size_t)M_TILE * SCORE_STRIDE * sizeof(float);
    smem_bytes += (size_t)MAX_Q_GROUPS * FUSED_GROUP_S * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_kv);
    dim3 block(THREADS);

    auto launch = [&](auto K_const) {
        constexpr int KV = K_const.value;
        cudaFuncSetAttribute(int8tc_qk_scores_topk_stats_fused_kernel<KV>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
        int8tc_qk_scores_topk_stats_fused_kernel<KV><<<grid, block, smem_bytes, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr<at::BFloat16>()),
            kdc.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            sl.data_ptr<int32_t>(),
            scores.data_ptr<float>(),
            topk_idx.data_ptr<int64_t>(),
            topk_slots.data_ptr<int64_t>(),
            softmax_m.data_ptr<float>(),
            softmax_inv_z.data_ptr<float>(),
            (int)max_S, (int)H_q, (int)H_kv,
            (int)block_size, (int)bt.stride(0),
            (int)kdc.stride(0), (int)ks.stride(0),
            (float)attn_scale
        );
    };
    switch ((int)top_k) {
        case   64: launch(std::integral_constant<int,   64>{}); break;
        case  128: launch(std::integral_constant<int,  128>{}); break;
        case  256: launch(std::integral_constant<int,  256>{}); break;
        case  512: launch(std::integral_constant<int,  512>{}); break;
        case 1024: launch(std::integral_constant<int, 1024>{}); break;
        default:
            TORCH_CHECK(false, "int8tc_qk_scores_topk_stats_fused: unsupported top_k=", top_k);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scores, topk_idx, topk_slots, softmax_m, softmax_inv_z};
}


std::vector<torch::Tensor> int8tc_qk_scores_topk_stats_group_fused(
    torch::Tensor q,
    torch::Tensor k_data,
    torch::Tensor k_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    double  attn_scale,
    int64_t top_k)
{
    TORCH_CHECK((H_q / H_kv) <= 4,
                "int8tc_qk_scores_topk_stats_group_fused currently supports H_q/H_kv <= 4 only");
    TORCH_CHECK(block_size == 16, "only block_size=16 supported");

    auto qc  = q.contiguous();
    auto kdc = k_data.contiguous();
    auto ks  = (k_scales.dim() == 3) ? k_scales.squeeze(-1).contiguous()
                                     : k_scales.contiguous();
    auto bt  = block_table.contiguous();
    auto sl  = seq_lens.contiguous();

    const int64_t B = qc.size(0);
    const int64_t n_groups = (max_S + FUSED_GROUP_S - 1) / FUSED_GROUP_S;
    auto opts_f32 = qc.options().dtype(torch::kFloat32);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(qc.device());
    auto scores         = torch::empty({B, H_q, max_S}, opts_f32);
    auto group_topk_idx = torch::empty({B, H_q, n_groups, top_k}, opts_i64);
    auto group_m        = torch::empty({B, H_q, n_groups}, opts_f32);
    auto group_sum      = torch::empty({B, H_q, n_groups}, opts_f32);
    auto topk_idx       = torch::empty({B, H_q, top_k}, opts_i64);
    auto topk_slots     = torch::empty({B, H_q, top_k}, opts_i64);
    auto softmax_m      = torch::empty({B, H_q}, opts_f32);
    auto softmax_inv_z  = torch::empty({B, H_q}, opts_f32);

    size_t smem_bytes = 0;
    smem_bytes += (size_t)M_TILE * Q_STRIDE_B;
    smem_bytes += (size_t)M_TILE * K_ITERS * sizeof(float);
    smem_bytes  = (smem_bytes + 15) & ~size_t(15);
    smem_bytes += (size_t)STAGES * BLOCK_S * K_PACK_STRIDE;
    smem_bytes += (size_t)STAGES * BLOCK_S * sizeof(__nv_bfloat16);
    smem_bytes  = (smem_bytes + 15) & ~size_t(15);
    smem_bytes += (size_t)M_TILE * SCORE_STRIDE * sizeof(float);
    smem_bytes += (size_t)MAX_Q_GROUPS * FUSED_GROUP_S * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 group_grid((unsigned)B, (unsigned)H_kv, (unsigned)n_groups);
    dim3 block(THREADS);
    auto launch_group = [&](auto K_const) {
        constexpr int KV = K_const.value;
        cudaFuncSetAttribute(int8tc_qk_scores_topk_stats_group_fused_kernel<KV, FUSED_GROUP_TILES>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);
        int8tc_qk_scores_topk_stats_group_fused_kernel<KV, FUSED_GROUP_TILES><<<group_grid, block, smem_bytes, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr<at::BFloat16>()),
            kdc.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            sl.data_ptr<int32_t>(),
            scores.data_ptr<float>(),
            group_topk_idx.data_ptr<int64_t>(),
            group_m.data_ptr<float>(),
            group_sum.data_ptr<float>(),
            (int)max_S, (int)H_q, (int)H_kv,
            (int)block_size, (int)bt.stride(0),
            (int)kdc.stride(0), (int)ks.stride(0),
            (float)attn_scale, (int)n_groups
        );
    };
    switch ((int)top_k) {
        case   64: launch_group(std::integral_constant<int,   64>{}); break;
        case  128: launch_group(std::integral_constant<int,  128>{}); break;
        case  256: launch_group(std::integral_constant<int,  256>{}); break;
        case  512: launch_group(std::integral_constant<int,  512>{}); break;
        case 1024: launch_group(std::integral_constant<int, 1024>{}); break;
        default:
            TORCH_CHECK(false, "int8tc_qk_scores_topk_stats_group_fused: unsupported top_k=", top_k);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 merge_grid((unsigned)B, (unsigned)H_q);
    dim3 merge_block(16 * 32);
    auto launch_merge = [&](auto K_const) {
        constexpr int KV = K_const.value;
        merge_group_topk_stats_kernel_v5<16, KV><<<merge_grid, merge_block, 0, stream.stream()>>>(
            scores.data_ptr<float>(),
            group_topk_idx.data_ptr<int64_t>(),
            group_m.data_ptr<float>(),
            group_sum.data_ptr<float>(),
            bt.data_ptr<int64_t>(),
            (int)max_S, (int)bt.size(1), (int)H_q, (int)block_size, (int)n_groups,
            topk_idx.data_ptr<int64_t>(),
            topk_slots.data_ptr<int64_t>(),
            softmax_m.data_ptr<float>(),
            softmax_inv_z.data_ptr<float>()
        );
    };
    switch ((int)top_k) {
        case   64: launch_merge(std::integral_constant<int,   64>{}); break;
        case  128: launch_merge(std::integral_constant<int,  128>{}); break;
        case  256: launch_merge(std::integral_constant<int,  256>{}); break;
        case  512: launch_merge(std::integral_constant<int,  512>{}); break;
        case 1024: launch_merge(std::integral_constant<int, 1024>{}); break;
        default:
            TORCH_CHECK(false, "int8tc_qk_scores_topk_stats_group_fused merge: unsupported top_k=", top_k);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scores, topk_idx, topk_slots, softmax_m, softmax_inv_z};
}


std::vector<torch::Tensor> int8tc_qk_scores_topk_stats_grouped(
    torch::Tensor q,
    torch::Tensor k_data,
    torch::Tensor k_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    double  attn_scale,
    int64_t top_k)
{
    torch::Tensor scores = int8tc_qk_scores(
        q, k_data, k_scales, block_table, seq_lens,
        max_S, H_q, H_kv, block_size, attn_scale
    );

    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();

    const int64_t B = scores.size(0);
    const int64_t n_groups = (max_S + GROUP_S - 1) / GROUP_S;
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(scores.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(scores.device());

    auto group_topk_idx = torch::empty({B, H_q, n_groups, top_k}, opts_i64);
    auto group_m        = torch::empty({B, H_q, n_groups}, opts_f32);
    auto group_sum      = torch::empty({B, H_q, n_groups}, opts_f32);
    auto topk_idx       = torch::empty({B, H_q, top_k}, opts_i64);
    auto topk_slots     = torch::empty({B, H_q, top_k}, opts_i64);
    auto softmax_m      = torch::empty({B, H_q}, opts_f32);
    auto softmax_inv_z  = torch::empty({B, H_q}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 group_grid((unsigned)B, (unsigned)H_q, (unsigned)n_groups);
    dim3 block(16 * 32);
    auto launch_group = [&](auto K_const) {
        constexpr int KV = K_const.value;
        group_topk_stats_kernel_v5<16, KV><<<group_grid, block, 0, stream.stream()>>>(
            scores.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            (int)max_S, (int)H_q, GROUP_S, (int)n_groups,
            group_topk_idx.data_ptr<int64_t>(),
            group_m.data_ptr<float>(),
            group_sum.data_ptr<float>()
        );
    };
    switch ((int)top_k) {
        case   64: launch_group(std::integral_constant<int,   64>{}); break;
        case  128: launch_group(std::integral_constant<int,  128>{}); break;
        case  256: launch_group(std::integral_constant<int,  256>{}); break;
        case  512: launch_group(std::integral_constant<int,  512>{}); break;
        case 1024: launch_group(std::integral_constant<int, 1024>{}); break;
        default:
            TORCH_CHECK(false, "int8tc_qk_scores_topk_stats_grouped group: unsupported top_k=", top_k);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 merge_grid((unsigned)B, (unsigned)H_q);
    auto launch_merge = [&](auto K_const) {
        constexpr int KV = K_const.value;
        merge_group_topk_stats_kernel_v5<16, KV><<<merge_grid, block, 0, stream.stream()>>>(
            scores.data_ptr<float>(),
            group_topk_idx.data_ptr<int64_t>(),
            group_m.data_ptr<float>(),
            group_sum.data_ptr<float>(),
            bt.data_ptr<int64_t>(),
            (int)max_S, (int)bt.size(1), (int)H_q, (int)block_size, (int)n_groups,
            topk_idx.data_ptr<int64_t>(),
            topk_slots.data_ptr<int64_t>(),
            softmax_m.data_ptr<float>(),
            softmax_inv_z.data_ptr<float>()
        );
    };
    switch ((int)top_k) {
        case   64: launch_merge(std::integral_constant<int,   64>{}); break;
        case  128: launch_merge(std::integral_constant<int,  128>{}); break;
        case  256: launch_merge(std::integral_constant<int,  256>{}); break;
        case  512: launch_merge(std::integral_constant<int,  512>{}); break;
        case 1024: launch_merge(std::integral_constant<int, 1024>{}); break;
        default:
            TORCH_CHECK(false, "int8tc_qk_scores_topk_stats_grouped merge: unsupported top_k=", top_k);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scores, topk_idx, topk_slots, softmax_m, softmax_inv_z};
}


}  // anonymous namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8tc_qk_scores", &int8tc_qk_scores,
          "Stage A (Int8 tensor cores): Q·K^T → scores. GIL released.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"));

    m.def("int8tc_qk_scores_topk", &int8tc_qk_scores_topk,
          "Stage A (Int8 TC): Q·K^T → scores + top-K + slot translate. "
          "Returns [scores, topk_idx, topk_slots]. GIL released.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"), py::arg("top_k"));

    m.def("int8tc_qk_scores_topk_stats", &int8tc_qk_scores_topk_stats,
          "Stage A (Int8 TC): Q·K^T → scores + top-K + slot translate + softmax stats. "
          "Returns [scores, topk_idx, topk_slots, softmax_m, softmax_inv_z].",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"), py::arg("top_k"));

    m.def("int8tc_qk_scores_topk_stats_fused", &int8tc_qk_scores_topk_stats_fused,
          "Stage A (Int8 TC): one CTA scans the full sequence for a (B,H_kv) "
          "pair, reusing Q quantization and maintaining online softmax + "
          "top-K in-kernel. Returns [scores, topk_idx, topk_slots, "
          "softmax_m, softmax_inv_z].",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"), py::arg("top_k"));

    m.def("int8tc_qk_scores_topk_stats_group_fused", &int8tc_qk_scores_topk_stats_group_fused,
          "Stage A (Int8 TC): group-of-tiles fused path. One CTA handles a "
          "tile group, reuses Q quantization, maintains group-local online "
          "softmax + top-K, then a small merge kernel combines groups into "
          "final [scores, topk_idx, topk_slots, softmax_m, softmax_inv_z].",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"), py::arg("top_k"));

    m.def("int8tc_qk_scores_topk_stats_grouped", &int8tc_qk_scores_topk_stats_grouped,
          "Stage A (Int8 TC): Q·K^T → scores + grouped exact top-K + merged softmax stats. "
          "Returns [scores, topk_idx, topk_slots, softmax_m, softmax_inv_z].",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"), py::arg("top_k"));
}
