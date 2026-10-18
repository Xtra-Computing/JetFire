// Stage B kernel for the JetFire fused-attention split — INT3 V-cache fork.
//   o_hat = softmax(scores) · dequant(V_int3)
//
// This is the INT3 sibling of int2_softmax_v.cu / int4_softmax_v.cu.
// Differences vs the INT2 / INT4 production kernels:
//   * V cache row width: D*3/8 bytes — 48 bytes/row at D=128.
//   * Storage layout: PLANAR (NOT byte-interleaved):
//         row = [plane0: D/8 bytes | plane1: D/8 bytes | plane2: D/8 bytes]
//                    bytes 0..15        bytes 16..31        bytes 32..47
//     Each group of 8 INT3 codes shares one byte in each plane:
//        combined_24b = (plane0_byte << 16) | (plane1_byte << 8) | plane2_byte
//     and val[i] for i in 0..7 lives at bits (21 - 3*i) +.. 23 (MSB-first).
//     See _pack_int3_planar / _unpack_int3_planar in jetfire_vllm/kernels/intN_pack.py.
//   * Per-thread inner unpack:
//        group_idx = d / 8;  in_group_idx = d % 8;
//        b0 = v_buf[group_idx +  0];          // plane 0
//        b1 = v_buf[group_idx + 16];          // plane 1
//        b2 = v_buf[group_idx + 32];          // plane 2
//        combined = (b0 << 16) | (b1 << 8) | b2;
//        q = ((combined >> (21 - 3*in_group_idx)) & 0x7) - 4;   // q ∈ [-4, 3]
//   * cp.async double-buffer: chunk = BS × 48 = 768 B at BS=16.  We issue 4-byte
//     transactions: 128 threads × 4 B = 512 B (covers planes 0..1 fully and the
//     first half of plane 2), then threads with tid<64 issue a second 4-byte
//     transaction for the remaining 256 B (second half of plane 2 — but per row).
//     Concretely, we treat the 768-byte chunk as 192 × 4-byte transactions and
//     dispatch them as { tid<128 → one transaction; tid<64 → another } both
//     within the same cp.async.commit_group.
//
// All other architecture (split-S, double-buffered SMEM staging, pre-multiplied
// wvs2 path, v2 weights-elimination path) is preserved verbatim.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <type_traits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(),  #x " must be CUDA")

constexpr int D_FIXED        = 128;
constexpr int D_PACKED_INT3  = D_FIXED * 3 / 8;   // 48 bytes per row (INT3 planar)
constexpr int N_GROUPS_INT3  = D_FIXED / 8;       // 16 groups of 8 vals → 16 bytes/plane/row
constexpr int THREADS_B      = 128;
constexpr int N_SPLITS       = 8;                 // split-S factor for kernel B
constexpr int N_SPLITS_WVS2_SMALL_B = 16;
constexpr int N_SPLITS_WVS2_B1 = 128;             // B=1 decode is CTA-starved at long ctx.
constexpr int WVS2_TOPK_CHUNK = 128;
constexpr size_t WVS2_TOPK_SMEM =
    (size_t)WVS2_TOPK_CHUNK * (sizeof(int64_t) + 3 * sizeof(float));


__device__ __forceinline__ float block_max(float v, float* smem32) {
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


__device__ __forceinline__ float block_sum(float v, float* smem32) {
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


// Kernel A: per (b, h_q), produce weights = softmax(scores) into HBM.
// Bit-width-agnostic — operates on fp32 scores. Identical to INT2/INT4 path.
__global__ void softmax_prep_kernel(
    const float*   __restrict__ scores,    // (B, H_q, max_S)
    const int32_t* __restrict__ seq_lens,  // (B,)
    float*         __restrict__ weights,   // (B, H_q, max_S) [out]
    int max_S, int H_q)
{
    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;
    const int S   = seq_lens[b];

    extern __shared__ __align__(16) uint8_t smem_raw[];
    float* red_buf = reinterpret_cast<float*>(smem_raw);

    const float* score_row =
        scores  + ((size_t)b * H_q + (size_t)h_q) * (size_t)max_S;
    float* weight_row =
        weights + ((size_t)b * H_q + (size_t)h_q) * (size_t)max_S;

    float local_max = -INFINITY;
    for (int i = tid; i < S; i += THREADS_B) {
        float s = score_row[i];
        local_max = fmaxf(local_max, s);
    }
    float m = block_max(local_max, red_buf);
    if (!isfinite(m)) m = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < max_S; i += THREADS_B) {
        float v = (i < S) ? __expf(score_row[i] - m) : 0.0f;
        weight_row[i] = v;
        local_sum += v;
    }
    float Z = block_sum(local_sum, red_buf);
    float invZ = (Z > 0.0f) ? (1.0f / Z) : 0.0f;
    __syncthreads();

    for (int i = tid; i < max_S; i += THREADS_B) {
        weight_row[i] = weight_row[i] * invZ;
    }
}


// ────────────────────────────────────────────────────────────────────────
// Helper: load a 768-byte INT3 chunk (BS=16 rows × 48 B) into SMEM via
// cp.async, using 192 × 4-byte transactions.
//   - Threads tid in [0, 128) issue transaction 0 covering byte_off = tid*4.
//   - Threads tid in [0, 64)  issue transaction 1 covering byte_off = tid*4 + 512.
// All transactions go into the same cp.async commit_group.
// `chunk_smem`/`chunk_gmem` point to the start of the BS-row block.
// ────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void load_int3_chunk_async(
    uint8_t* __restrict__ chunk_smem,            // [BS * 48] in SMEM (per-buf slice)
    const uint8_t* __restrict__ chunk_gmem,      // [BS * 48] in HBM (per-block slice)
    int tid)
{
    constexpr int BS = 16;
    constexpr int CHUNK_BYTES = BS * D_PACKED_INT3;  // 768

    // Transaction 0: byte_off = tid * 4, valid for tid in [0, 128).
    {
        const int byte_off = tid * 4;
        const int pos_in_blk = byte_off / D_PACKED_INT3;       // 0..15
        const int row_off    = byte_off - pos_in_blk * D_PACKED_INT3;
        const void* gmem_ptr = chunk_gmem + (size_t)pos_in_blk * D_PACKED_INT3 + row_off;
        void* smem_ptr       = chunk_smem + (size_t)pos_in_blk * D_PACKED_INT3 + row_off;
        unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(smem_int), "l"(gmem_ptr));
        (void)CHUNK_BYTES;
    }
    // Transaction 1: byte_off = tid * 4 + 512, valid for tid in [0, 64).
    if (tid < 64) {
        const int byte_off = tid * 4 + 512;
        const int pos_in_blk = byte_off / D_PACKED_INT3;       // 10..15
        const int row_off    = byte_off - pos_in_blk * D_PACKED_INT3;
        const void* gmem_ptr = chunk_gmem + (size_t)pos_in_blk * D_PACKED_INT3 + row_off;
        void* smem_ptr       = chunk_smem + (size_t)pos_in_blk * D_PACKED_INT3 + row_off;
        unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(smem_int), "l"(gmem_ptr));
    }
}


// Kernel B (INT3): split-S SMEM-staged V multiply + Phase-2 top-K.
//
// Per-row V is now D*3/8 = 48 bytes (planar).  Chunk SMEM v_buf: 2 × BS × 48 B
// = 1.5 KB at BS=16 (vs 1 KB INT2, 2 KB INT4).
template <int BS, int NG>
__global__ void weighted_v_split_kernel(
    const float*         __restrict__ weights,      // (B, H_q, max_S)
    const uint8_t*       __restrict__ v_data,       // (H_kv, num_slots, D*3/8)
    const __nv_bfloat16* __restrict__ v_scales,     // (H_kv, num_slots, num_groups)
    const int64_t*       __restrict__ block_table,  // (B, max_blocks)
    const int32_t*       __restrict__ seq_lens,     // (B,)
    const int64_t*       __restrict__ topk_idx,     // (B, H_q, K) — null if pre_corr disabled
    float*               __restrict__ o_hat,        // (B, H_q, D) — pre-zeroed
    float*               __restrict__ o_old_unnorm, // (B, H_q, D) — pre-zeroed (null = skip)
    float*               __restrict__ W_out,        // (B, H_q)    — pre-zeroed (null = skip)
    int max_S,
    int max_blocks,
    int H_q,
    int H_kv,
    int K,
    int v_data_stride_h,
    int v_scales_stride_h)
{
    constexpr int GROUP_SIZE = D_FIXED / NG;
    const int b       = blockIdx.x;
    const int h_q     = blockIdx.y;
    const int s_split = blockIdx.z;
    const int tid     = threadIdx.x;

    const int n_groups_per_head = H_q / H_kv;
    const int h_kv              = h_q / n_groups_per_head;
    const int S                 = seq_lens[b];

    const int chunk_per_split = (max_S + N_SPLITS - 1) / N_SPLITS;
    const int s_start = s_split * chunk_per_split;
    const int s_end   = min(s_start + chunk_per_split, S);
    if (s_start >= S) return;

    // SMEM layout (DOUBLE-BUFFERED — INT3 sizes):
    //   v_buf  : 2 × BS * 48 bytes   (1536 B for BS=16)
    //   vs_buf : 2 × BS × NG * 2 bytes
    //   w_buf  : 2 × BS * 4 bytes    (128 B)
    const bool do_pre_corr = (topk_idx != nullptr);

    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* v_buf = smem_raw;                                              // [2][BS*48]
    __nv_bfloat16* vs_buf =
        reinterpret_cast<__nv_bfloat16*>(v_buf + 2 * BS * D_PACKED_INT3);   // [2][BS][NG]
    float* w_buf = reinterpret_cast<float*>(vs_buf + 2 * BS * NG);          // [2][BS]

    const int d = tid;
    if (d >= D_FIXED) return;
    // INT3 PLANAR unpack: 8 codes per (group_idx) triplet of bytes, MSB-first.
    const int group_idx    = d >> 3;          // 0..15
    const int in_group_idx = d & 7;           // 0..7
    const int shift        = 21 - 3 * in_group_idx;
    const int group_id     = (NG == 1) ? 0 : (d / GROUP_SIZE);

    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const float*         w_row        = weights  + ((size_t)b * H_q + (size_t)h_q) * max_S;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    auto load_chunk_async = [&](int blk, int buf) {
        const int64_t base_slot = bt_row[blk] * (int64_t)BS;

        const uint8_t* gmem_chunk =
            v_pack_base + (size_t)base_slot * D_PACKED_INT3;
        uint8_t* smem_chunk =
            v_buf + (size_t)buf * BS * D_PACKED_INT3;
        load_int3_chunk_async(smem_chunk, gmem_chunk, tid);

        // V scales — BS * NG bf16 values.  For NG=2, stage both
        // groups so dimensions 0..63 and 64..127 use their own scale.
        if (tid < BS * NG) {
            int scale_pos = tid / NG;
            int scale_gid = tid % NG;
            vs_buf[(buf * BS + scale_pos) * NG + scale_gid] = __ldg(
                v_scale_base + (size_t)(base_slot + scale_pos) * NG + scale_gid
            );
        }

        // Weights — 16 fp32, masked to [s_start, s_end).
        if (tid < BS) {
            int p = blk * BS + tid;
            w_buf[buf * BS + tid] =
                (p >= s_start && p < s_end) ? w_row[p] : 0.0f;
        }
    };

    auto compute_chunk = [&](int buf, float& acc_io) {
        const uint8_t* v_cur = v_buf + (size_t)buf * BS * D_PACKED_INT3;
        const __nv_bfloat16* vs_cur = vs_buf + buf * BS * NG;
        const float* w_cur = w_buf + buf * BS;
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            const uint8_t* row = v_cur + j * D_PACKED_INT3;
            uint32_t b0 = row[group_idx +  0];                    // plane 0
            uint32_t b1 = row[group_idx + N_GROUPS_INT3];         // plane 1
            uint32_t b2 = row[group_idx + 2 * N_GROUPS_INT3];     // plane 2
            uint32_t combined = (b0 << 16) | (b1 << 8) | b2;
            int q = int((combined >> shift) & 0x7u) - 4;          // q ∈ [-4, 3]
            float vs = __bfloat162float(vs_cur[j * NG + group_id]);
            float v_f = (float)q * vs;
            float w   = w_cur[j];
            acc_io = __fmaf_rn(w, v_f, acc_io);
        }
    };

    // ────────────────────────── Phase 1: V mul → o_hat ──────────────────────
    float acc = 0.0f;

    if (blk_start < blk_end) {
        load_chunk_async(blk_start, 0);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        int cur_buf = 0;
        for (int blk = blk_start; blk < blk_end - 1; ++blk) {
            int nxt_buf = 1 - cur_buf;
            load_chunk_async(blk + 1, nxt_buf);
            asm volatile("cp.async.commit_group;\n");

            compute_chunk(cur_buf, acc);

            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            cur_buf = nxt_buf;
        }
        compute_chunk(cur_buf, acc);
        __syncthreads();
    }

    const size_t bh = (size_t)b * H_q + (size_t)h_q;
    atomicAdd(o_hat + bh * D_FIXED + d, acc);

    // ──────── Phase 2: top-K only loop, exclusive to s_split == 0 ───────
    if (do_pre_corr && s_split == 0) {
        const int64_t* topk_row = topk_idx + bh * (size_t)K;
        float acc_o_old = 0.0f;
        float acc_W     = 0.0f;
        for (int k = 0; k < K; ++k) {
            int64_t pos = topk_row[k];
            if (pos < 0 || (int)pos >= S) continue;
            int blk_idx = (int)(pos / BS);
            int in_blk  = (int)(pos % BS);
            int64_t slot = bt_row[blk_idx] * (int64_t)BS + (int64_t)in_blk;

            const uint8_t* row = v_pack_base + slot * D_PACKED_INT3;
            uint32_t b0 = __ldg(row + group_idx +  0);
            uint32_t b1 = __ldg(row + group_idx + N_GROUPS_INT3);
            uint32_t b2 = __ldg(row + group_idx + 2 * N_GROUPS_INT3);
            uint32_t combined = (b0 << 16) | (b1 << 8) | b2;
            int q = int((combined >> shift) & 0x7u) - 4;
            float vs = __bfloat162float(__ldg(
                v_scale_base + slot * NG + group_id));
            float v_f = (float)q * vs;
            float w   = w_row[(size_t)pos];

            acc_o_old = __fmaf_rn(w, v_f, acc_o_old);
            if (d == 0) acc_W += w;
        }
        o_old_unnorm[bh * D_FIXED + d] = acc_o_old;
        if (d == 0) W_out[bh] = acc_W;
    }
}


torch::Tensor int3_softmax_v(
    torch::Tensor scores,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size)
{
    CHECK_CUDA(scores); CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(seq_lens);
    TORCH_CHECK(scores.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(v_data.scalar_type()      == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_PACKED_INT3,
        "INT3 v_data must be (H_kv, num_slots, D*3/8=", D_PACKED_INT3, ")");
    TORCH_CHECK(H_q % H_kv == 0);

    auto sc = scores.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();

    const int B          = sc.size(0);
    const int num_groups = (int)vs.size(2);
    const int max_blocks = (int)bt.size(1);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());

    auto weights = torch::empty({B, (int64_t)H_q, (int64_t)max_S}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();

    {
        dim3 grid_a((unsigned)B, (unsigned)H_q);
        dim3 block_a(THREADS_B);
        const size_t smem_a = 16 * sizeof(float);
        softmax_prep_kernel<<<grid_a, block_a, smem_a, stream.stream()>>>(
            sc.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            weights.data_ptr<float>(),
            (int)max_S, (int)H_q
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto o_hat = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);

    auto launch_b = [&](auto BS_const, auto NG_const) {
        constexpr int BS = BS_const.value;
        constexpr int NG = NG_const.value;
        dim3 grid_b((unsigned)B, (unsigned)H_q, N_SPLITS);
        dim3 block_b(THREADS_B);
        const size_t smem_b = (size_t)2 * BS * D_PACKED_INT3  // v_buf [2][BS*48]
                            + (size_t)2 * BS * NG * 2         // vs_buf [2][BS][NG]
                            + (size_t)2 * BS * 4              // w_buf
                            + 16;
        cudaFuncSetAttribute(weighted_v_split_kernel<BS, NG>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_b);
        weighted_v_split_kernel<BS, NG><<<grid_b, block_b, smem_b, stream.stream()>>>(
            weights.data_ptr<float>(),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            sl.data_ptr<int32_t>(),
            /*topk_idx*/ nullptr,
            o_hat.data_ptr<float>(),
            /*o_old_unnorm*/ nullptr,
            /*W_out*/ nullptr,
            (int)max_S, max_blocks, (int)H_q, (int)H_kv,
            /*K=*/0,
            (int)vd.stride(0), (int)vs.stride(0)
        );
    };

    if (block_size == 16 && num_groups == 1) {
        launch_b(std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
    } else if (block_size == 16 && num_groups == 2) {
        launch_b(std::integral_constant<int, 16>{}, std::integral_constant<int, 2>{});
    } else {
        TORCH_CHECK(false,
            "int3_softmax_v: unsupported (block_size=", block_size,
            ", num_groups=", num_groups, ").  Supported: (16,1), (16,2).");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return o_hat;
}


// pre_corr fork — adds top-K bookkeeping (o_old_unnorm, W).
std::vector<torch::Tensor> int3_softmax_v_with_pre_corr(
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
    CHECK_CUDA(scores); CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(seq_lens); CHECK_CUDA(topk_idx);
    TORCH_CHECK(scores.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(v_data.scalar_type()      == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(topk_idx.scalar_type()    == torch::kInt64);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(topk_idx.dim() == 3);
    TORCH_CHECK(topk_idx.size(2) == top_k);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_PACKED_INT3);
    TORCH_CHECK(H_q % H_kv == 0);

    auto sc = scores.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto ti = topk_idx.contiguous();

    const int B          = sc.size(0);
    const int num_groups = (int)vs.size(2);
    const int max_blocks = (int)bt.size(1);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());

    auto weights = torch::empty({B, (int64_t)H_q, (int64_t)max_S}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    {
        dim3 grid_a((unsigned)B, (unsigned)H_q);
        dim3 block_a(THREADS_B);
        const size_t smem_a = 16 * sizeof(float);
        softmax_prep_kernel<<<grid_a, block_a, smem_a, stream.stream()>>>(
            sc.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            weights.data_ptr<float>(),
            (int)max_S, (int)H_q
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto o_hat        = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto o_old_unnorm = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out        = torch::zeros({B, (int64_t)H_q}, opts_f32);

    auto launch_b = [&](auto BS_const, auto NG_const) {
        constexpr int BS = BS_const.value;
        constexpr int NG = NG_const.value;
        dim3 grid_b((unsigned)B, (unsigned)H_q, N_SPLITS);
        dim3 block_b(THREADS_B);
        const size_t smem_b = (size_t)2 * BS * D_PACKED_INT3
                            + (size_t)2 * BS * NG * 2
                            + (size_t)2 * BS * 4
                            + 16;
        cudaFuncSetAttribute(weighted_v_split_kernel<BS, NG>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_b);
        weighted_v_split_kernel<BS, NG><<<grid_b, block_b, smem_b, stream.stream()>>>(
            weights.data_ptr<float>(),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            sl.data_ptr<int32_t>(),
            ti.data_ptr<int64_t>(),
            o_hat.data_ptr<float>(),
            o_old_unnorm.data_ptr<float>(),
            W_out.data_ptr<float>(),
            (int)max_S, max_blocks, (int)H_q, (int)H_kv,
            (int)top_k,
            (int)vd.stride(0), (int)vs.stride(0)
        );
    };

    if (block_size == 16 && num_groups == 1) {
        launch_b(std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
    } else if (block_size == 16 && num_groups == 2) {
        launch_b(std::integral_constant<int, 16>{}, std::integral_constant<int, 2>{});
    } else {
        TORCH_CHECK(false,
            "int3_softmax_v_with_pre_corr: unsupported (block_size=", block_size,
            ", num_groups=", num_groups, ").");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {o_hat, o_old_unnorm, W_out};
}


// ────────────────────────────────────────────────────────────────────────
// wvs2 path: pre-multiplied (w · vs) at chunk-load time, INT3 sibling.
// Same semantics as INT2/INT4 wvs2 but with INT3 byte width / planar unpack.
// ────────────────────────────────────────────────────────────────────────

template <int BS, int SPLITS = N_SPLITS, int NG = 1>
__global__ void weighted_v_split_kernel_wvs2(
    const float*         __restrict__ weights,      // (B, H_q, max_S)
    const uint8_t*       __restrict__ v_data,       // (H_kv, num_slots, D*3/8)
    const __nv_bfloat16* __restrict__ v_scales,     // (H_kv, num_slots, NG)
    const int64_t*       __restrict__ block_table,
    const int32_t*       __restrict__ seq_lens,
    const int64_t*       __restrict__ topk_idx,
    float*               __restrict__ o_hat,
    float*               __restrict__ o_old_unnorm,
    float*               __restrict__ W_out,
    int max_S,
    int max_blocks,
    int H_q,
    int H_kv,
    int K,
    int v_data_stride_h,
    int v_scales_stride_h)
{
    const int b       = blockIdx.x;
    const int h_q     = blockIdx.y;
    const int s_split = blockIdx.z;
    const int tid     = threadIdx.x;

    const int n_groups_per_head = H_q / H_kv;
    const int h_kv              = h_q / n_groups_per_head;
    const int S                 = seq_lens[b];

    const int chunk_per_split = (max_S + SPLITS - 1) / SPLITS;
    const int s_start = s_split * chunk_per_split;
    const int s_end   = min(s_start + chunk_per_split, S);
    if (s_start >= S) return;

    const bool do_pre_corr = (topk_idx != nullptr);

    // SMEM layout (INT3 sizes):
    //   v_buf : 2 × BS × 48 bytes   (1536 B for BS=16)
    //   ws_buf: 2 × BS × NG × 4 bytes.
    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* v_buf = smem_raw;                                                   // [2][BS*48]
    float*   ws_buf = reinterpret_cast<float*>(v_buf + 2 * BS * D_PACKED_INT3);  // [2][BS][NG]
    uint8_t* topk_raw = reinterpret_cast<uint8_t*>(ws_buf + 2 * BS * NG);
    int64_t* topk_slot_buf = reinterpret_cast<int64_t*>(topk_raw);
    float*   topk_ws_buf = reinterpret_cast<float*>(topk_slot_buf + WVS2_TOPK_CHUNK);
    float*   topk_w_buf = topk_ws_buf + WVS2_TOPK_CHUNK * NG;

    const int d = tid;
    if (d >= D_FIXED) return;
    const int group_idx    = d >> 3;
    const int in_group_idx = d & 7;
    const int shift        = 21 - 3 * in_group_idx;
    constexpr int GROUP_SIZE = D_FIXED / NG;
    const int scale_group_id = (NG == 1) ? 0 : (d / GROUP_SIZE);

    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const float*         w_row        = weights  + ((size_t)b * H_q + (size_t)h_q) * max_S;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    auto load_chunk_async = [&](int blk, int buf) {
        const int64_t base_slot = bt_row[blk] * (int64_t)BS;

        const uint8_t* gmem_chunk =
            v_pack_base + (size_t)base_slot * D_PACKED_INT3;
        uint8_t* smem_chunk =
            v_buf + (size_t)buf * BS * D_PACKED_INT3;
        load_int3_chunk_async(smem_chunk, gmem_chunk, tid);

        // Pre-multiplied (w * vs), one value per slot and scale group.
        if (tid < BS * NG) {
            const int j = tid / NG;
            const int g = tid - j * NG;
            int p = blk * BS + j;
            float ws = 0.0f;
            if (p >= s_start && p < s_end) {
                float vs = __bfloat162float(__ldg(
                    v_scale_base + ((size_t)(base_slot + j) * NG + g)));
                float w  = w_row[p];
                ws = w * vs;
            }
            ws_buf[(buf * BS + j) * NG + g] = ws;
        }
    };

    auto compute_chunk = [&](int buf, float& acc_io) {
        const uint8_t* v_cur  = v_buf + (size_t)buf * BS * D_PACKED_INT3;
        const float*   ws_cur = ws_buf + buf * BS * NG;
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            const uint8_t* row = v_cur + j * D_PACKED_INT3;
            uint32_t b0 = row[group_idx +  0];
            uint32_t b1 = row[group_idx + N_GROUPS_INT3];
            uint32_t b2 = row[group_idx + 2 * N_GROUPS_INT3];
            uint32_t combined = (b0 << 16) | (b1 << 8) | b2;
            int q = int((combined >> shift) & 0x7u) - 4;
            acc_io = __fmaf_rn(ws_cur[j * NG + scale_group_id], (float)q, acc_io);
        }
    };

    // Phase 1.
    float acc = 0.0f;
    if (blk_start < blk_end) {
        load_chunk_async(blk_start, 0);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        int cur_buf = 0;
        for (int blk = blk_start; blk < blk_end - 1; ++blk) {
            int nxt_buf = 1 - cur_buf;
            load_chunk_async(blk + 1, nxt_buf);
            asm volatile("cp.async.commit_group;\n");

            compute_chunk(cur_buf, acc);

            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            cur_buf = nxt_buf;
        }
        compute_chunk(cur_buf, acc);
        __syncthreads();
    }

    const size_t bh = (size_t)b * H_q + (size_t)h_q;
    atomicAdd(o_hat + bh * D_FIXED + d, acc);

    // Phase 2.
    if (do_pre_corr && s_split == 0) {
        constexpr int TOPK_STEP = WVS2_TOPK_CHUNK / NG;
        const int64_t* topk_row = topk_idx + bh * (size_t)K;
        float acc_o_old = 0.0f;
        float acc_W     = 0.0f;
        for (int k0 = 0; k0 < K; k0 += TOPK_STEP) {
            if (tid < TOPK_STEP * NG) {
                const int k_local = tid / NG;
                const int g = tid - k_local * NG;
                const int kk = k0 + k_local;
                int64_t slot = -1;
                float w = 0.0f;
                float ws = 0.0f;
                if (kk < K) {
                    int64_t pos = topk_row[kk];
                    if (pos >= 0 && (int)pos < S) {
                        int blk_idx = (int)(pos / BS);
                        int in_blk  = (int)(pos % BS);
                        int64_t block_id = bt_row[blk_idx];
                        if (block_id >= 0) {
                            slot = block_id * (int64_t)BS + (int64_t)in_blk;
                            float vs = __bfloat162float(__ldg(
                                v_scale_base + (size_t)slot * NG + g));
                            w = w_row[(size_t)pos];
                            ws = w * vs;
                        }
                    }
                }
                topk_ws_buf[k_local * NG + g] = ws;
                if (g == 0) {
                    topk_slot_buf[k_local] = slot;
                    topk_w_buf[k_local] = w;
                }
            }
            __syncthreads();

            const int kend = min(TOPK_STEP, K - k0);
            #pragma unroll 1
            for (int j = 0; j < kend; ++j) {
                int64_t slot = topk_slot_buf[j];
                if (slot < 0) continue;
                const uint8_t* row = v_pack_base + slot * D_PACKED_INT3;
                uint32_t b0 = __ldg(row + group_idx +  0);
                uint32_t b1 = __ldg(row + group_idx + N_GROUPS_INT3);
                uint32_t b2 = __ldg(row + group_idx + 2 * N_GROUPS_INT3);
                uint32_t combined = (b0 << 16) | (b1 << 8) | b2;
                int q = int((combined >> shift) & 0x7u) - 4;
                acc_o_old = __fmaf_rn(topk_ws_buf[j * NG + scale_group_id], (float)q, acc_o_old);
                if (d == 0) acc_W += topk_w_buf[j];
            }
            __syncthreads();
        }
        o_old_unnorm[bh * D_FIXED + d] = acc_o_old;
        if (d == 0) W_out[bh] = acc_W;
    }
}


std::vector<torch::Tensor> int3_softmax_v_with_pre_corr_wvs2(
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
    CHECK_CUDA(scores); CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(seq_lens); CHECK_CUDA(topk_idx);
    TORCH_CHECK(scores.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(v_data.scalar_type()      == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(topk_idx.scalar_type()    == torch::kInt64);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(topk_idx.dim() == 3);
    TORCH_CHECK(topk_idx.size(2) == top_k);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_PACKED_INT3);
    TORCH_CHECK(H_q % H_kv == 0);
    TORCH_CHECK(block_size == 16, "wvs2 only supports block_size=16");

    auto sc = scores.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto ti = topk_idx.contiguous();

    const int B          = sc.size(0);
    const int num_groups = (int)vs.size(2);
    const int max_blocks = (int)bt.size(1);
    TORCH_CHECK(num_groups == 1 || num_groups == 2,
                "wvs2 requires num_groups=1 or 2 (got ", num_groups, ")");

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());

    auto weights = torch::empty({B, (int64_t)H_q, (int64_t)max_S}, opts_f32);
    auto stream  = at::cuda::getCurrentCUDAStream();
    {
        dim3 grid_a((unsigned)B, (unsigned)H_q);
        dim3 block_a(THREADS_B);
        const size_t smem_a = 16 * sizeof(float);
        softmax_prep_kernel<<<grid_a, block_a, smem_a, stream.stream()>>>(
            sc.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            weights.data_ptr<float>(),
            (int)max_S, (int)H_q
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto o_hat        = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto o_old_unnorm = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out        = torch::zeros({B, (int64_t)H_q}, opts_f32);

    dim3 block_b(THREADS_B);
    constexpr int BS_C = 16;
    const size_t smem_b = (size_t)2 * BS_C * D_PACKED_INT3   // v_buf
                        + (size_t)2 * BS_C * num_groups * 4  // ws_buf (fp32)
                        + WVS2_TOPK_SMEM                     // Phase-2 topK slot/ws/w
                        + 16;
    auto launch_wvs2 = [&](auto ng_const, auto split_const) {
        constexpr int NG_C = ng_const.value;
        constexpr int SPLITS = split_const.value;
        dim3 grid_b((unsigned)B, (unsigned)H_q, (unsigned)SPLITS);
        cudaFuncSetAttribute(weighted_v_split_kernel_wvs2<BS_C, SPLITS, NG_C>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_b);
        weighted_v_split_kernel_wvs2<BS_C, SPLITS, NG_C><<<grid_b, block_b, smem_b, stream.stream()>>>(
            weights.data_ptr<float>(),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            bt.data_ptr<int64_t>(),
            sl.data_ptr<int32_t>(),
            ti.data_ptr<int64_t>(),
            o_hat.data_ptr<float>(),
            o_old_unnorm.data_ptr<float>(),
            W_out.data_ptr<float>(),
            (int)max_S, max_blocks, (int)H_q, (int)H_kv,
            (int)top_k,
            (int)vd.stride(0), (int)vs.stride(0)
        );
    };
    auto launch_by_ng = [&](auto split_const) {
        if (num_groups == 1) {
            launch_wvs2(std::integral_constant<int, 1>{}, split_const);
        } else {
            launch_wvs2(std::integral_constant<int, 2>{}, split_const);
        }
    };
    if (B == 1) {
        launch_by_ng(std::integral_constant<int, N_SPLITS_WVS2_B1>{});
    } else if (B <= 16) {
        launch_by_ng(std::integral_constant<int, N_SPLITS_WVS2_SMALL_B>{});
    } else {
        launch_by_ng(std::integral_constant<int, N_SPLITS>{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}

void int3_softmax_v_with_pre_corr_wvs2_inplace(
    torch::Tensor scores,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor topk_idx,
    torch::Tensor weights,
    torch::Tensor o_hat,
    torch::Tensor o_old_unnorm,
    torch::Tensor W_out,
    int64_t max_S,
    int64_t H_q,
    int64_t H_kv,
    int64_t block_size,
    int64_t top_k)
{
    CHECK_CUDA(scores); CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(seq_lens); CHECK_CUDA(topk_idx);
    CHECK_CUDA(weights); CHECK_CUDA(o_hat); CHECK_CUDA(o_old_unnorm); CHECK_CUDA(W_out);
    TORCH_CHECK(scores.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(v_data.scalar_type()      == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(topk_idx.scalar_type()    == torch::kInt64);
    TORCH_CHECK(weights.scalar_type()     == torch::kFloat32);
    TORCH_CHECK(o_hat.scalar_type()       == torch::kFloat32);
    TORCH_CHECK(o_old_unnorm.scalar_type()== torch::kFloat32);
    TORCH_CHECK(W_out.scalar_type()       == torch::kFloat32);
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(v_data.is_contiguous(), "v_data must be contiguous");
    TORCH_CHECK(v_scales.is_contiguous(), "v_scales must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
    TORCH_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(o_hat.is_contiguous(), "o_hat must be contiguous");
    TORCH_CHECK(o_old_unnorm.is_contiguous(), "o_old_unnorm must be contiguous");
    TORCH_CHECK(W_out.is_contiguous(), "W_out must be contiguous");
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(v_scales.dim() == 3, "graph inplace path expects 3D v_scales");
    TORCH_CHECK(topk_idx.dim() == 3);
    TORCH_CHECK(topk_idx.size(2) == top_k);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_PACKED_INT3);
    TORCH_CHECK(H_q % H_kv == 0);
    TORCH_CHECK(block_size == 16, "wvs2 only supports block_size=16");

    const int B          = scores.size(0);
    const int num_groups = (int)v_scales.size(2);
    const int max_blocks = (int)block_table.size(1);
    TORCH_CHECK(num_groups == 1 || num_groups == 2,
                "wvs2 requires num_groups=1 or 2 (got ", num_groups, ")");
    TORCH_CHECK(weights.sizes() == torch::IntArrayRef({B, H_q, max_S}));
    TORCH_CHECK(o_hat.sizes() == torch::IntArrayRef({B, H_q, D_FIXED}));
    TORCH_CHECK(o_old_unnorm.sizes() == torch::IntArrayRef({B, H_q, D_FIXED}));
    TORCH_CHECK(W_out.sizes() == torch::IntArrayRef({B, H_q}));

    auto stream = at::cuda::getCurrentCUDAStream();
    {
        dim3 grid_a((unsigned)B, (unsigned)H_q);
        dim3 block_a(THREADS_B);
        const size_t smem_a = 16 * sizeof(float);
        softmax_prep_kernel<<<grid_a, block_a, smem_a, stream.stream()>>>(
            scores.data_ptr<float>(),
            seq_lens.data_ptr<int32_t>(),
            weights.data_ptr<float>(),
            (int)max_S, (int)H_q
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaMemsetAsync(
        o_hat.data_ptr<float>(), 0, o_hat.numel() * sizeof(float), stream.stream()));
    C10_CUDA_CHECK(cudaMemsetAsync(
        o_old_unnorm.data_ptr<float>(), 0, o_old_unnorm.numel() * sizeof(float), stream.stream()));
    C10_CUDA_CHECK(cudaMemsetAsync(
        W_out.data_ptr<float>(), 0, W_out.numel() * sizeof(float), stream.stream()));

    dim3 block_b(THREADS_B);
    constexpr int BS_C = 16;
    const size_t smem_b = (size_t)2 * BS_C * D_PACKED_INT3
                        + (size_t)2 * BS_C * num_groups * 4
                        + WVS2_TOPK_SMEM
                        + 16;
    auto launch_wvs2 = [&](auto ng_const, auto split_const) {
        constexpr int NG_C = ng_const.value;
        constexpr int SPLITS = split_const.value;
        dim3 grid_b((unsigned)B, (unsigned)H_q, (unsigned)SPLITS);
        cudaFuncSetAttribute(weighted_v_split_kernel_wvs2<BS_C, SPLITS, NG_C>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_b);
        weighted_v_split_kernel_wvs2<BS_C, SPLITS, NG_C><<<grid_b, block_b, smem_b, stream.stream()>>>(
            weights.data_ptr<float>(),
            v_data.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(v_scales.data_ptr<at::BFloat16>()),
            block_table.data_ptr<int64_t>(),
            seq_lens.data_ptr<int32_t>(),
            topk_idx.data_ptr<int64_t>(),
            o_hat.data_ptr<float>(),
            o_old_unnorm.data_ptr<float>(),
            W_out.data_ptr<float>(),
            (int)max_S, max_blocks, (int)H_q, (int)H_kv,
            (int)top_k,
            (int)v_data.stride(0), (int)v_scales.stride(0)
        );
    };
    auto launch_by_ng = [&](auto split_const) {
        if (num_groups == 1) {
            launch_wvs2(std::integral_constant<int, 1>{}, split_const);
        } else {
            launch_wvs2(std::integral_constant<int, 2>{}, split_const);
        }
    };
    if (B == 1) {
        launch_by_ng(std::integral_constant<int, N_SPLITS_WVS2_B1>{});
    } else if (B <= 16) {
        launch_by_ng(std::integral_constant<int, N_SPLITS_WVS2_SMALL_B>{});
    } else {
        launch_by_ng(std::integral_constant<int, N_SPLITS>{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int3_softmax_v", &int3_softmax_v,
          "Stage B INT3 (planar): 2-kernel split-S + SMEM staging "
          "(softmax_prep + weighted_v_split with atomicAdd)",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"),
          py::arg("v_data"),
          py::arg("v_scales"),
          py::arg("block_table"),
          py::arg("seq_lens"),
          py::arg("max_S"),
          py::arg("H_q"),
          py::arg("H_kv"),
          py::arg("block_size"));

    m.def("int3_softmax_v_with_pre_corr", &int3_softmax_v_with_pre_corr,
          "Stage B INT3 (planar) with pre_corr: returns [o_hat, o_old_unnorm, W].",
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

    m.def("int3_softmax_v_with_pre_corr_wvs2", &int3_softmax_v_with_pre_corr_wvs2,
          "Stage B INT3 (planar) + wvs2 weighted_v_split with pre-multiplied w·vs at "
          "chunk-load time. NG=1 only. Same outputs as int3_softmax_v_with_pre_corr.",
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
    m.def("int3_softmax_v_with_pre_corr_wvs2_inplace",
          &int3_softmax_v_with_pre_corr_wvs2_inplace,
          "Graph-safe INT3 wvs2 Stage B using caller-owned workspace.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"),
          py::arg("v_data"),
          py::arg("v_scales"),
          py::arg("block_table"),
          py::arg("seq_lens"),
          py::arg("topk_idx"),
          py::arg("weights"),
          py::arg("o_hat"),
          py::arg("o_old_unnorm"),
          py::arg("W_out"),
          py::arg("max_S"),
          py::arg("H_q"),
          py::arg("H_kv"),
          py::arg("block_size"),
          py::arg("top_k"));
}
