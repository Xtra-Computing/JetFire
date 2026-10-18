// Phase 2 of decode-stage split: standalone Q·K^T → scores kernel.
//
// Mirrors the Q·K^T half of `fused_decode_wmma_v5_int2_kernel` in
// `jetfire_vllm/csrc/fused_decode_wmma.cu` but emits ONLY raw QK scores
// (× attn_scale) so the GPU's V-cache stage (softmax + P·V) can run
// concurrently with the CPU's top-K bf16 attention path.
//
// Inputs  (matching `fused_decode_wmma_v5_int2_with_scores`):
//   q         : (B, H_q, D=128) bf16
//   k_data    : (H_kv, num_slots, D/4=32) uint8   (offset INT2, per-slot scale)
//   k_scales  : (H_kv, num_slots, K_num_quant_groups) bf16
//   k_zp      : (H_kv, num_slots[, 1]) bf16 for asymmetric K, or empty for
//               symmetric K (treated as zero additive bias)
//   block_table: (B, max_blocks) int64
//   seq_lens   : (B,) int32
//
// Output:
//   scores    : (B, H_q, max_S) fp32, with positions >= seq_lens[b] = -INFINITY.
//
// Implementation: WMMA bf16 16×16×16 mma.sync.  Path 1 from the brief.

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
constexpr int D_QUARTER    = D / 4;          // 32 packed bytes per row
constexpr int BLOCK_S      = 64;             // WMMA tile width (matches v5 BLOCK_S_V4)
constexpr int THREADS      = 128;            // 4 warps × 32
constexpr int STAGES       = 2;              // double-buffered cp.async pipeline

constexpr int Q_PAD        = 8;
constexpr int K_PAD        = 8;
constexpr int Q_STRIDE     = D + Q_PAD;       // bf16
constexpr int K_STRIDE     = D + K_PAD;       // bf16
constexpr int SCORE_PAD    = 4;
constexpr int SCORE_STRIDE = BLOCK_S + SCORE_PAD;  // fp32

// cp.async helpers — same wiring as fused_decode_wmma.cu.
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

// Same INT2 unpack helper as the v5 kernel.  1 byte = 4 signed int2's,
// big-endian, offset-encoded (stored = q + 2).
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


__global__ void int2_qk_scores_kernel(
    const __nv_bfloat16* __restrict__ Q,           // (B, H_q, D)
    const uint8_t*       __restrict__ K_data,      // (H_kv, num_slots, D/4)
    const __nv_bfloat16* __restrict__ K_scales,    // (H_kv, num_slots) bf16
    const __nv_bfloat16* __restrict__ K_zp,        // (H_kv, num_slots) bf16
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

    auto k_pack     = reinterpret_cast<uint8_t(*)[BLOCK_S][D_QUARTER]>(&dyn_smem[off]);
    off += STAGES * BLOCK_S * D_QUARTER;
    auto k_scale    = reinterpret_cast<__nv_bfloat16*>(&dyn_smem[off]);
    off += STAGES * BLOCK_S * K_num_quant_groups * sizeof(__nv_bfloat16);
    auto k_zp_smem  = reinterpret_cast<__nv_bfloat16*>(&dyn_smem[off]);
    off += STAGES * BLOCK_S * K_num_quant_groups * sizeof(__nv_bfloat16);

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
    // Pad columns Q_STRIDE-1..D (only the row 0..15 padding columns) — not
    // strictly needed since WMMA reads [0..D), but defensive.

    const uint8_t*       k_pack_base = K_data   + (size_t)h_kv * K_data_stride_h;
    const __nv_bfloat16* ks_base     = K_scales + (size_t)h_kv * K_scales_stride_h;
    const __nv_bfloat16* kz_base =
        K_zp == nullptr ? nullptr : K_zp + (size_t)h_kv * K_scales_stride_h;
    const int64_t*       bt_row      = BlockTable + (size_t)b * block_table_stride;

    // We process exactly one tile per CTA (grid.z = num_tiles).  Use a
    // 2-stage cp.async only if it pays off — for one tile the prologue is
    // the entire load, so a single-stage path suffices.  Issue the load,
    // wait, unpack, do WMMA.
    auto issue_load = [&](int stage, int tile_s0_local, int tile_end_local) {
        // Each row needs D_QUARTER=32 bytes → 2 cp.async16 per row.
        // 128 threads load 64 rows; assign one row per pair (tid/2).
        int row    = tid / 2;            // 0..63
        int half   = tid & 1;            // 0 or 1 → bytes [0..15] or [16..31]
        if (row < BLOCK_S) {
            int pos = tile_s0_local + row;
            int64_t slot = -1;
            if (pos < tile_end_local) {
                slot = bt_row[pos / block_size] * (int64_t)block_size
                     + (pos % block_size);
            }
            uint8_t* dst = &k_pack[stage][row][half * 16];
            if (slot >= 0) {
                const uint8_t* src = k_pack_base + slot * (int64_t)D_QUARTER + half * 16;
                cp_async16(dst, src);
                if (half == 0) {
                    for (int g = 0; g < K_num_quant_groups; ++g) {
                        const int scale_off =
                            (stage * BLOCK_S + row) * K_num_quant_groups + g;
                        const int64_t src_off =
                            slot * (int64_t)K_scales_stride_slot + g;
                        k_scale  [scale_off] = ks_base[src_off];
                        k_zp_smem[scale_off] =
                            kz_base == nullptr ? __float2bfloat16(0.f)
                                               : kz_base[src_off];
                    }
                }
            } else {
                // No cp.async for invalid slots — just zero-fill via a
                // shared write.  cp.async16 of 16 zero bytes is also fine
                // but we don't have an obvious zero source pointer; using
                // direct SMEM stores keeps this simple.
                #pragma unroll
                for (int i = 0; i < 16; ++i) dst[i] = 0;
                if (half == 0) {
                    for (int g = 0; g < K_num_quant_groups; ++g) {
                        const int scale_off =
                            (stage * BLOCK_S + row) * K_num_quant_groups + g;
                        k_scale  [scale_off] = __float2bfloat16(0.f);
                        k_zp_smem[scale_off] = __float2bfloat16(0.f);
                    }
                }
            }
        }
        cp_async_commit();
    };

    auto unpack_tile = [&](int stage) {
        // Each thread unpacks one (row, byte) → 4 bf16 K values.
        // Total work: BLOCK_S * D_QUARTER = 64 * 32 = 2048 → 16 iters at 128 threads.
        for (int i = tid; i < BLOCK_S * D_QUARTER; i += THREADS) {
            int r  = i / D_QUARTER;
            int bx = i % D_QUARTER;
            uint8_t pk = k_pack[stage][r][bx];
            const int elem0 = 4 * bx;
            int qg = (elem0 * K_num_quant_groups) / D;
            qg = min(max(qg, 0), K_num_quant_groups - 1);
            const int scale_off =
                (stage * BLOCK_S + r) * K_num_quant_groups + qg;
            float ks = __bfloat162float(k_scale  [scale_off]);
            float kz = __bfloat162float(k_zp_smem[scale_off]);
            __nv_bfloat16 v0, v1, v2, v3;
            int2_unpack_bf16(pk, ks, kz, v0, v1, v2, v3);
            k_smem[r][4*bx + 0] = v0;
            k_smem[r][4*bx + 1] = v1;
            k_smem[r][4*bx + 2] = v2;
            k_smem[r][4*bx + 3] = v3;
        }
    };

    // ── Issue load for this CTA's (single) tile ──────────────────────────
    issue_load(0, tile_s0, tile_end_S);
    cp_async_wait_group_0();
    __syncthreads();

    // Unpack INT2 → bf16.
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
torch::Tensor int2_qk_scores(
    torch::Tensor q,
    torch::Tensor k_data,
    torch::Tensor k_scales,
    torch::Tensor k_zp,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    double  attn_scale)
{
    const bool has_k_zp = k_zp.defined() && k_zp.numel() > 0;
    TORCH_CHECK(q.is_cuda() && k_data.is_cuda() && k_scales.is_cuda(),
                "q, k_data, and k_scales must be CUDA");
    TORCH_CHECK(!has_k_zp || k_zp.is_cuda(), "k_zp must be CUDA when provided");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(k_data.scalar_type() == torch::kUInt8, "k_data must be uint8");
    TORCH_CHECK(k_scales.scalar_type() == torch::kBFloat16, "k_scales must be bf16");
    TORCH_CHECK(!has_k_zp || k_zp.scalar_type() == torch::kBFloat16,
                "k_zp must be bf16 when provided");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64, "block_table must be int64");
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == D, "q must be (B, H_q, 128)");
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");
    TORCH_CHECK(H_q / H_kv <= 16,
                "int2_qk_scores: H_q/H_kv=", (H_q / H_kv),
                " exceeds the 16-row WMMA tile limit");

    auto qc  = q.contiguous();
    auto kdc = k_data.contiguous();
    auto ks  = (k_scales.dim() == 2) ? k_scales.unsqueeze(-1).contiguous()
                                     : k_scales.contiguous();
    torch::Tensor kz;
    if (has_k_zp) {
        kz = (k_zp.dim() == 2) ? k_zp.unsqueeze(-1).contiguous()
                               : k_zp.contiguous();
    }
    auto bt  = block_table.contiguous();
    auto sl  = seq_lens.contiguous();

    const int64_t B = qc.size(0);
    const int K_num_quant_groups = (int)ks.size(2);
    TORCH_CHECK(K_num_quant_groups == 1 || K_num_quant_groups == 2,
                "int2_qk_scores currently supports K quant groups 1 or 2, got ",
                K_num_quant_groups);

    auto options_f32 = qc.options().dtype(torch::kFloat32);
    auto scores      = torch::empty({B, H_q, max_S}, options_f32);

    const int num_tiles = ((int)max_S + BLOCK_S - 1) / BLOCK_S;

    const size_t smem_bytes =
        16 * Q_STRIDE * sizeof(__nv_bfloat16) +              // q_smem
        STAGES * BLOCK_S * D_QUARTER +                       // k_pack
        STAGES * BLOCK_S * K_num_quant_groups * sizeof(__nv_bfloat16) + // k_scale
        STAGES * BLOCK_S * K_num_quant_groups * sizeof(__nv_bfloat16) + // k_zp
        BLOCK_S * K_STRIDE * sizeof(__nv_bfloat16) +         // k_smem (bf16)
        16 * SCORE_STRIDE * sizeof(float);                   // score_smem

    cudaFuncSetAttribute(int2_qk_scores_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);

    dim3 grid((unsigned)B, (unsigned)H_kv, (unsigned)num_tiles);
    dim3 block(THREADS);

    const auto stream = at::cuda::getCurrentCUDAStream();
    int2_qk_scores_kernel<<<grid, block, smem_bytes, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr<at::BFloat16>()),
        kdc.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ks.data_ptr<at::BFloat16>()),
        has_k_zp
            ? reinterpret_cast<const __nv_bfloat16*>(kz.data_ptr<at::BFloat16>())
            : nullptr,
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
// Positions outside seq_lens[b] are ignored for topK and translated to -1 if
// they appear only because K exceeds the valid sequence length.

__device__ __forceinline__ int64_t translate_topk_slot_or_invalid(
    int64_t orig_idx,
    int valid_S,
    const int64_t* __restrict__ block_table,
    int b,
    int max_blocks,
    int bs)
{
    // topK is computed over a row where positions >= seq_len carry -inf.
    // Those padding entries can still appear when K > seq_len; keep their
    // score index for the GPU softmax path, but mark the CPU shadow slot
    // invalid so CPU correction skips them instead of attending to slot 0.
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
// Recursive 8-bit radix-select.  SMEM cost is INDEPENDENT of K (just a
// 256-int histogram + a few control words), so this kernel covers the
// full K ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192} range — including
// K's that overflow v4's 48 KB static-SMEM cap.
//
// fp32 → uint32 monotone map (descending): negatives flipped, positives
// have sign-bit set.  After the map, larger float ↔ larger uint32, so
// the radix bytes can be indexed with a plain `>>` and bin numbers
// compared with `>`.
//
// Per pass:
//   1. Stream all S elements; for those whose top `prefix_len` bits match
//      `prefix`, atomicAdd into the bin selected by the next 8 bits.
//   2. Find the cutoff bin B* (descending-cumulative scan) such that all
//      elements in bins > B* are guaranteed top-K.
//   3. Stream again; emit any element with prefix-match AND next-byte > B*
//      to topk_idx via an atomic counter.
//   4. K_remaining -= K_above; prefix = prefix·256 + B*; prefix_len += 8.
//   5. If K_remaining == 0 or prefix_len == 32, exit.
//
// At prefix_len == 32 with K_remaining > 0, all surviving elements share
// an identical 32-bit prefix → pick any K_remaining of them (all have
// equal float value, so the "tie-break" is a no-op for any consumer that
// only cares about the score-set).
//
// HBM cost: up to 4 hist + 4 emit = 8 reads of S per CTA (vs 1 for v3/v4).
// L2 (40 MB on H100) absorbs most of the repeats since scores per launch
// total ~64 MB at B=16, ctx=32k, so passes 2+ are mostly L2 hits.

__device__ __forceinline__ uint32_t f2r_desc(float f) {
    uint32_t u = __float_as_uint(f);
    // For descending order: positives → high MSB; negatives → flip all.
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
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
    __shared__ uint32_t prefix;       // accumulated determined high bits
    __shared__ int      prefix_len;   // bits already determined (0, 8, 16, 24, 32)
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

    // Up to 4 levels of 8-bit refinement.
    #pragma unroll
    for (int level = 0; level < 4; ++level) {
        if (K_remaining == 0) break;

        // 1) Histogram: count next-byte for prefix-matching elements.
        for (int i = tid; i < 256; i += THREADS) hist[i] = 0;
        __syncthreads();

        const int      cur_prefix_len = prefix_len;
        const uint32_t cur_prefix     = prefix;
        const int      shift_high     = 32 - cur_prefix_len;        // bits already decided
        const int      shift_byte     = 32 - cur_prefix_len - 8;    // bits for this level
        // Mask to extract just the prefix bits (top `cur_prefix_len` bits).
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

        // 2) Find cutoff bin (single thread; 256 ops).
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

        // 3) Emit elements above cutoff bin.
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

        // 4) Update state for next level.
        if (tid == 0) {
            K_remaining -= K_above;
            prefix       = (cur_prefix << 8) | (uint32_t)cb_local;
            prefix_len   = cur_prefix_len + 8;
        }
        __syncthreads();
    }

    // Tail: if K_remaining > 0 after 4 levels, all remaining elements
    // share an identical 32-bit prefix (i.e., identical fp32 value).
    // Emit any K_remaining of them.
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

    // Write topk_slots[] from topk_idx[] (block-table translation).
    for (int rank = tid; rank < K; rank += THREADS) {
        int64_t orig_idx = topk_idx[bh * (size_t)K + rank];
        topk_slots[bh * (size_t)K + rank] =
            translate_topk_slot_or_invalid(
                orig_idx, valid_S, block_table, b, max_blocks, bs);
    }
}


// Wrapper that runs Stage A + the topk_slots post-pass and returns
// (scores, topk_idx, topk_slots).  Equivalent to the chain of Python ops
// in V2_02_topk_d2h_fork (mask → topk → //bs/%bs → gather → muladd) but
// done in two CUDA launches instead of ~9.

std::vector<torch::Tensor> int2_qk_scores_topk(
    torch::Tensor q,
    torch::Tensor k_data,
    torch::Tensor k_scales,
    torch::Tensor k_zp,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    double  attn_scale,
    int64_t top_k)
{
    // Phase 1: Stage A main kernel → scores (re-uses the existing wrapper).
    torch::Tensor scores = int2_qk_scores(
        q, k_data, k_scales, k_zp, block_table, seq_lens,
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

    // Release topK is radix-select only. JetFire's release design uses
    // large correction sets (default K=256), where radix-select is the
    // stable algorithmic fit.
    auto launch_large_v5 = [&]() -> bool {
        // v5 doesn't depend on POW2_MAX_S — its outer loop just runs over
        // valid_S = min(max_S, seq_lens[b]).  N_WARPS=16 (THREADS=512).
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
                    "int2_qk_scores_topk: unsupported radix-select top_k=",
                    top_k, " (supported: 32,64,128,256,512,1024,2048,4096,8192)");
    } else {
        TORCH_CHECK(false, "int2_qk_scores_topk: max_S=", max_S,
                    " exceeds supported limit (262144).  Add a larger "
                    "POW2_MAX_S template instantiation if needed.");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {scores, topk_idx, topk_slots};
}


}  // anonymous namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int2_qk_scores", &int2_qk_scores,
          "Stage A: Q·K^T → scores (INT2 K, bf16 Q, fp32 scores). GIL released.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"), py::arg("k_zp"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"));

    m.def("int2_qk_scores_topk", &int2_qk_scores_topk,
          "Stage A v4: Q·K^T → scores + in-kernel mask + top-K + slot translate. "
          "Returns [scores, topk_idx, topk_slots]. GIL released.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("q"), py::arg("k_data"), py::arg("k_scales"), py::arg("k_zp"),
          py::arg("block_table"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("attn_scale"), py::arg("top_k"));
}
