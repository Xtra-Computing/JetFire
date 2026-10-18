// INT4 SYMMETRIC variant of int2_qk_scores.cu (Stage A: Q·K^T → scores +
// optional top-K).
//
// Differences vs the INT2 ASYMMETRIC kernel:
//   * K is INT4 (4 bits per signed value), packed 2 values per byte:
//       byte = ((q_even + 8) << 4) | (q_odd + 8)
//     where q_even = xq[..., 0::2] (high nibble), q_odd = xq[..., 1::2]
//     (low nibble), and q ∈ {-8, ..., 7}.
//   * SYMMETRIC quantization: dequant = q * scale (no zero-point).
//     The k_zp tensor is dropped from all signatures.
//   * K row stride is D / 2 = 64 bytes (vs D / 4 = 32 for INT2).
//
// Inputs:
//   q         : (B, H_q, D=128) bf16
//   k_data    : (H_kv, num_slots, D/2=64) uint8   (symmetric INT4)
//   k_scales  : (H_kv, num_slots, K_num_quant_groups) bf16
//   block_table: (B, max_blocks) int64
//   seq_lens   : (B,) int32
//
// Output:
//   scores    : (B, H_q, max_S) fp32, with positions >= seq_lens[b] = -INFINITY.
//
// The top-K post-pass is bit-width-agnostic and uses the release
// radix-select implementation.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <string>

namespace {

constexpr int D            = 128;
constexpr int D_HALF       = D / 2;          // 64 packed bytes per row (INT4)
constexpr int BLOCK_S      = 64;             // WMMA tile width (matches v5)
constexpr int THREADS      = 128;            // 4 warps × 32
constexpr int STAGES       = 2;              // double-buffered cp.async pipeline

constexpr int Q_PAD        = 8;
constexpr int K_PAD        = 8;
constexpr int Q_STRIDE     = D + Q_PAD;       // bf16
constexpr int K_STRIDE     = D + K_PAD;       // bf16
constexpr int SCORE_PAD    = 4;
constexpr int SCORE_STRIDE = BLOCK_S + SCORE_PAD;  // fp32

// cp.async helpers — same wiring as the INT2 variant.
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

// INT4 SYMMETRIC unpack: one byte = 2 signed int4 values.
//   high nibble (shift=4) = even index (xq[..., 0::2])
//   low  nibble (shift=0) = odd  index (xq[..., 1::2])
//   q ∈ {-8, ..., 7}, dequant = q * scale (no zp).
__device__ __forceinline__ void int4_unpack_bf16(
    uint8_t p, float scale,
    __nv_bfloat16& v_even, __nv_bfloat16& v_odd)
{
    int q_even = int((p >> 4) & 0xF) - 8;
    int q_odd  = int( p       & 0xF) - 8;
    v_even = __float2bfloat16(float(q_even) * scale);
    v_odd  = __float2bfloat16(float(q_odd)  * scale);
}


__global__ void int4_qk_scores_kernel(
    const __nv_bfloat16* __restrict__ Q,           // (B, H_q, D)
    const uint8_t*       __restrict__ K_data,      // (H_kv, num_slots, D/2)
    const __nv_bfloat16* __restrict__ K_scales,    // (H_kv, num_slots) bf16
    const int64_t*       __restrict__ BlockTable,  // (B, max_blocks)
    const int32_t*       __restrict__ SeqLens,     // (B,)
    float*               __restrict__ Scores,      // (B, H_q, max_S) fp32
    int    max_S, int H_q, int H_kv,
    int    block_size, int block_table_stride,
    int    K_data_stride_h,
    int    K_scales_stride_h,
    int    K_scales_stride_slot,
    int    K_num_quant_groups,
    float  attn_scale)
{
    const int b        = blockIdx.x;
    const int h_kv     = blockIdx.y;
    const int tile_idx = blockIdx.z;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;

    const int num_groups = H_q / H_kv;          // 4 for LLaMA-3.1-8B
    const int S          = SeqLens[b];
    const int tile_s0    = tile_idx * BLOCK_S;
    if (tile_s0 >= max_S) return;
    const int tile_end_S = min(tile_s0 + BLOCK_S, S);
    const int tile_end_M = min(tile_s0 + BLOCK_S, max_S);  // for sentinel writes

    // ── Dynamic SMEM layout ──────────────────────────────────────────────
    extern __shared__ __align__(16) uint8_t dyn_smem[];
    size_t off = 0;
    auto q_smem     = reinterpret_cast<__nv_bfloat16(*)[Q_STRIDE]>(&dyn_smem[off]);
    off += 16 * Q_STRIDE * sizeof(__nv_bfloat16);

    // INT4 packed rows: D_HALF = 64 bytes (vs 32 for INT2).
    auto k_pack     = reinterpret_cast<uint8_t(*)[BLOCK_S][D_HALF]>(&dyn_smem[off]);
    off += STAGES * BLOCK_S * D_HALF;
    auto k_scale    = reinterpret_cast<__nv_bfloat16*>(&dyn_smem[off]);
    off += STAGES * BLOCK_S * K_num_quant_groups * sizeof(__nv_bfloat16);
    // No k_zp — symmetric.

    auto k_smem     = reinterpret_cast<__nv_bfloat16(*)[K_STRIDE]>(&dyn_smem[off]);
    off += BLOCK_S * K_STRIDE * sizeof(__nv_bfloat16);
    auto score_smem = reinterpret_cast<float(*)[SCORE_STRIDE]>(&dyn_smem[off]);
    off += 16 * SCORE_STRIDE * sizeof(float);

    // ── Q init ───────────────────────────────────────────────────────────
    // Pad rows num_groups..15 to zero so WMMA mma_sync gives 0 for padded rows.
    for (int i = tid; i < (16 - num_groups) * D; i += THREADS) {
        int r = num_groups + i / D;
        int c = i % D;
        q_smem[r][c] = __float2bfloat16(0.f);
    }
    // Load num_groups Q rows for this h_kv group.
    for (int i = tid; i < num_groups * D; i += THREADS) {
        int r = i / D;
        int c = i % D;
        int h_q = h_kv * num_groups + r;
        q_smem[r][c] = Q[(size_t)b * H_q * D + (size_t)h_q * D + c];
    }

    const uint8_t*       k_pack_base = K_data   + (size_t)h_kv * K_data_stride_h;
    const __nv_bfloat16* ks_base     = K_scales + (size_t)h_kv * K_scales_stride_h;
    const int64_t*       bt_row      = BlockTable + (size_t)b * block_table_stride;

    // We process exactly one tile per CTA (grid.z = num_tiles).  Issue the
    // load, wait, unpack, do WMMA.
    auto issue_load = [&](int stage, int tile_s0_local, int tile_end_local) {
        // Each row needs D_HALF = 64 bytes → 4 cp.async16 per row.
        // 128 threads × 2 ops = 256 cp.async16 ops to cover BLOCK_S × 4 = 256.
        // Layout: linear_id = iter * THREADS + tid;
        //         row    = linear_id >> 2  (4 chunks per row)
        //         chunk  = linear_id & 3   (chunk ∈ {0,1,2,3} → bytes [chunk*16 .. chunk*16+15])
        #pragma unroll
        for (int it = 0; it < 2; ++it) {
            int lin = it * THREADS + tid;
            int row   = lin >> 2;        // 0..63
            int chunk = lin & 3;         // 0..3
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
        // Scales: BLOCK_S = 64 entries; 128 threads → first 64 threads each
        // handle one row.
        if (tid < BLOCK_S) {
            int row = tid;
            int pos = tile_s0_local + row;
            int64_t slot = -1;
            if (pos < tile_end_local) {
                slot = bt_row[pos / block_size] * (int64_t)block_size
                     + (pos % block_size);
            }
            if (slot >= 0) {
                for (int g = 0; g < K_num_quant_groups; ++g) {
                    const int scale_off =
                        (stage * BLOCK_S + row) * K_num_quant_groups + g;
                    const int64_t src_off =
                        slot * (int64_t)K_scales_stride_slot + g;
                    k_scale[scale_off] = ks_base[src_off];
                }
            } else {
                for (int g = 0; g < K_num_quant_groups; ++g) {
                    const int scale_off =
                        (stage * BLOCK_S + row) * K_num_quant_groups + g;
                    k_scale[scale_off] = __float2bfloat16(0.f);
                }
            }
        }
        cp_async_commit();
    };

    auto unpack_tile = [&](int stage) {
        // Each thread unpacks one (row, byte) → 2 bf16 K values.
        // Total work: BLOCK_S * D_HALF = 64 * 64 = 4096 bytes → 32 iters at 128 threads.
        for (int i = tid; i < BLOCK_S * D_HALF; i += THREADS) {
            int r  = i / D_HALF;
            int bx = i % D_HALF;
            uint8_t pk = k_pack[stage][r][bx];
            const int elem0 = 2 * bx;
            int qg = (elem0 * K_num_quant_groups) / D;
            qg = min(max(qg, 0), K_num_quant_groups - 1);
            const int scale_off =
                (stage * BLOCK_S + r) * K_num_quant_groups + qg;
            float ks = __bfloat162float(k_scale[scale_off]);
            __nv_bfloat16 v_even, v_odd;
            int4_unpack_bf16(pk, ks, v_even, v_odd);
            // bx-th packed byte expands to D-positions {2*bx, 2*bx+1}.
            k_smem[r][2*bx + 0] = v_even;
            k_smem[r][2*bx + 1] = v_odd;
        }
    };

    // ── Issue load for this CTA's (single) tile ──────────────────────────
    issue_load(0, tile_s0, tile_end_S);
    cp_async_wait_group_0();
    __syncthreads();

    // Unpack INT4 → bf16.
    unpack_tile(0);
    __syncthreads();

    // ── WMMA Q·K^T ───────────────────────────────────────────────────────
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // 4 warps × one N-tile of 16 cols each = BLOCK_S = 64.
    int n_tile  = warp_id;             // 0..3
    int n_start = n_tile * 16;
    fill_fragment(c_frag, 0.f);
    #pragma unroll
    for (int k = 0; k < D; k += 16) {
        load_matrix_sync(a_frag, &q_smem[0][k], Q_STRIDE);
        load_matrix_sync(b_frag, &k_smem[n_start][k], K_STRIDE);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    store_matrix_sync(&score_smem[0][n_start], c_frag,
                      SCORE_STRIDE, mem_row_major);
    __syncthreads();

    // ── Emit (× attn_scale) to global memory; sentinel = -INFINITY for
    // positions in [tile_s0, tile_end_M) that are beyond seq_lens[b]. ──
    const int valid_rows = num_groups;
    const int tile_cols  = tile_end_M - tile_s0;   // could be < BLOCK_S in the last tile
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


// ─────────────────────────────────────────────────────────────────────────────
// Host launcher.
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor int4_qk_scores(
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
    auto ks  = (k_scales.dim() == 2) ? k_scales.unsqueeze(-1).contiguous()
                                     : k_scales.contiguous();
    auto bt  = block_table.contiguous();
    auto sl  = seq_lens.contiguous();

    const int64_t B = qc.size(0);
    const int K_num_quant_groups = (int)ks.size(2);
    TORCH_CHECK(K_num_quant_groups == 1 || K_num_quant_groups == 2,
                "int4_qk_scores currently supports K quant groups 1 or 2, got ",
                K_num_quant_groups);

    auto options_f32 = qc.options().dtype(torch::kFloat32);
    auto scores      = torch::empty({B, H_q, max_S}, options_f32);

    const int num_tiles = ((int)max_S + BLOCK_S - 1) / BLOCK_S;

    const size_t smem_bytes =
        16 * Q_STRIDE * sizeof(__nv_bfloat16) +              // q_smem
        STAGES * BLOCK_S * D_HALF +                          // k_pack (2x INT2)
        STAGES * BLOCK_S * K_num_quant_groups * sizeof(__nv_bfloat16) + // k_scale
        BLOCK_S * K_STRIDE * sizeof(__nv_bfloat16) +         // k_smem (bf16)
        16 * SCORE_STRIDE * sizeof(float);                   // score_smem

    cudaFuncSetAttribute(int4_qk_scores_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);

    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)num_tiles);
    dim3 block(THREADS);

    const auto stream = at::cuda::getCurrentCUDAStream();
    int4_qk_scores_kernel<<<grid, block, smem_bytes, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr<at::BFloat16>()),
        kdc.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
        bt.data_ptr<int64_t>(),
        sl.data_ptr<int32_t>(),
        scores.data_ptr<float>(),
        (int)max_S, (int)H_q, (int)H_kv,
        (int)block_size, (int)bt.stride(0),
        (int)kdc.stride(0), (int)ks.stride(0),
        (int)ks.stride(1), K_num_quant_groups,
        (float)attn_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return scores;
}


// ─────────────────────────────────────────────────────────────────────────
// Radix-select Top-K + slot-translate post-pass.
// One CTA per (b, h_q).  Reads Stage-A fp32 scores, selects top-K indices
// with radix buckets, then translates selected score positions to KV slots.
// ─────────────────────────────────────────────────────────────────────────

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
    return block_table[(size_t)b * (size_t)max_blocks + (size_t)block_id]
         * (int64_t)bs + in_block;
}


// ── topk_slots_kernel_v5 (radix-select) ───────────────────────────────
__device__ __forceinline__ uint32_t f2r_desc(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
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

    // A heterogeneous batch can contain S < K. Keep every unselected
    // lane invalid rather than reading uninitialized allocator contents.
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

    // A heterogeneous batch can contain S < K. Keep every unselected
    // lane invalid rather than reading uninitialized allocator contents.
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


// ─────────────────────────────────────────────────────────────────────────
// Wrapper: int4_qk_scores_topk runs Stage A + the topk_slots post-pass.
// Returns (scores, topk_idx, topk_slots).
// ─────────────────────────────────────────────────────────────────────────

std::vector<torch::Tensor> int4_qk_scores_topk(
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
    // Phase 1: Stage A main kernel → scores.
    torch::Tensor scores = int4_qk_scores(
        q, k_data, k_scales, block_table, seq_lens,
        max_S, H_q, H_kv, block_size, attn_scale
    );

    // Phase 2: top-K + slot translate → topk_idx, topk_slots.
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();

    const int64_t B = scores.size(0);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(scores.device());
    auto topk_idx   = torch::empty({B, H_q, (int64_t)top_k}, opts_i64);
    auto topk_slots = torch::empty({B, H_q, (int64_t)top_k}, opts_i64);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_q);

    // Release topK is radix-select only.  Register-heap topK variants were
    // small-K experiments and are no longer dispatchable from the release
    // launcher.
    auto launch_large_v5 = [&]() -> bool {
        dim3 block(16 * 32);
        const int K = (int)top_k;
        auto launch = [&](auto K_const) {
            constexpr int KV = K_const.value;
            topk_slots_kernel_v5<16, KV><<<grid, block, 0, stream.stream()>>>(
                scores.data_ptr<float>(), sl.data_ptr<int32_t>(),
                bt.data_ptr<int64_t>(),
                (int)max_S, (int)bt.size(1), (int)H_q, (int)block_size,
                topk_idx.data_ptr<int64_t>(), topk_slots.data_ptr<int64_t>()
            );
        };
        switch (K) {
            case   32: launch(std::integral_constant<int,   32>{}); return true;
            case   64: launch(std::integral_constant<int,   64>{}); return true;
            case  128: launch(std::integral_constant<int,  128>{}); return true;
            case  256: launch(std::integral_constant<int,  256>{}); return true;
            case  512: launch(std::integral_constant<int,  512>{}); return true;
            case 1024: launch(std::integral_constant<int, 1024>{}); return true;
            case 2048: launch(std::integral_constant<int, 2048>{}); return true;
            case 4096: launch(std::integral_constant<int, 4096>{}); return true;
            case 8192: launch(std::integral_constant<int, 8192>{}); return true;
            default:   return false;
        }
    };
    if (max_S <= 262144) {
        TORCH_CHECK(launch_large_v5(),
                    "int4_qk_scores_topk: unsupported radix-select top_k=",
                    top_k, " (supported: 32,64,128,256,512,1024,2048,4096,8192)");
    } else {
        TORCH_CHECK(false, "int4_qk_scores_topk: max_S=", max_S,
                    " exceeds supported limit (262144).  Add a larger "
                    "POW2_MAX_S template instantiation if needed.");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {scores, topk_idx, topk_slots};
}


std::vector<torch::Tensor> int4_qk_scores_topk_stats(
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
    torch::Tensor scores = int4_qk_scores(
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
            TORCH_CHECK(false, "int4_qk_scores_topk_stats: unsupported top_k=", top_k);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scores, topk_idx, topk_slots, softmax_m, softmax_inv_z};
}


}  // anonymous namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_qk_scores", &int4_qk_scores,
          "Stage A: Q·K^T → scores (INT4 SYM K, bf16 Q, fp32 scores). GIL released.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"));

    m.def("int4_qk_scores_topk", &int4_qk_scores_topk,
          "Stage A v4: Q·K^T → scores + in-kernel mask + top-K + slot translate. "
          "Returns [scores, topk_idx, topk_slots]. GIL released.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"), py::arg("top_k"));

    m.def("int4_qk_scores_topk_stats", &int4_qk_scores_topk_stats,
          "Stage A v5: Q·K^T → scores + top-K + slot translate + softmax stats. "
          "Returns [scores, topk_idx, topk_slots, softmax_m, softmax_inv_z].",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"), py::arg("top_k"));
}
