// Stage B kernel for the JetFire fused-attention split — INT4 V-cache fork.
//   o_hat = softmax(scores) · dequant(V_int4)
//
// This is the INT4 sibling of int2_softmax_v.cu.  Differences vs the INT2
// production kernel:
//   * V cache row width: D/2 bytes (vs D/4 for INT2) — 64 bytes/row at D=128.
//   * Inner unpack: 2 codes per byte (vs 4 for INT2).
//       byte = ((q_even + 8) << 4) | (q_odd + 8)   // _pack_int4 in intN_pack.py
//       => high nibble (shift=4) = even lane (d=2*i+0)
//          low  nibble (shift=0) = odd  lane (d=2*i+1)
//       q ∈ [-8, 7]  (signed nibble), offset = 8.
//   * cp.async double-buffer: each thread issues 2× 4-byte loads per chunk
//     (128 threads × 4 B = 512 B; chunk = BS × 64 B = 1024 B at BS=16).
//
// All other architecture (split-S, double-buffered SMEM staging,
// pre-multiplied wvs2 path, v2 weights-elimination path) is preserved.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <type_traits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(),  #x " must be CUDA")

constexpr int D_FIXED   = 128;
constexpr int D_HALF    = D_FIXED / 2;     // 64 bytes per row (INT4)
constexpr int THREADS_B = 128;
constexpr int N_SPLITS  = 8;             // default split-S factor for kernel B
constexpr int N_SPLITS_WVS2_B1 = 128;    // B=1 decode is CTA-starved at 32k ctx.
constexpr int WVS2_TOPK_CHUNK = 128;     // Phase-2 topK staging chunk.
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
// Bit-width-agnostic — operates on fp32 scores. Identical to INT2 path.
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

// Stats-only softmax pass.  This keeps the stable max/sum reduction from
// softmax_prep_kernel but writes just two fp32 scalars per (B,Hq) row.
__global__ void softmax_stats_kernel(
    const float*   __restrict__ scores,      // (B, H_q, max_S)
    const int32_t* __restrict__ seq_lens,    // (B,)
    float*         __restrict__ softmax_m,   // (B, H_q)
    float*         __restrict__ softmax_inv_z,
    int max_S, int H_q)
{
    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;
    const int S   = seq_lens[b];

    extern __shared__ __align__(16) uint8_t smem_raw[];
    float* red_buf = reinterpret_cast<float*>(smem_raw);

    const size_t bh = (size_t)b * H_q + (size_t)h_q;
    const float* score_row = scores + bh * (size_t)max_S;

    float local_max = -INFINITY;
    for (int i = tid; i < S; i += THREADS_B) {
        local_max = fmaxf(local_max, score_row[i]);
    }
    float m = block_max(local_max, red_buf);
    if (!isfinite(m)) m = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < S; i += THREADS_B) {
        local_sum += __expf(score_row[i] - m);
    }
    float Z = block_sum(local_sum, red_buf);
    const float invZ = (Z > 0.0f) ? (1.0f / Z) : 0.0f;

    if (tid == 0) {
        softmax_m[bh] = m;
        softmax_inv_z[bh] = invZ;
    }
}

__device__ __forceinline__ float load_weight_or_score(
    const float* __restrict__ weights,
    const float* __restrict__ scores,
    const float* __restrict__ softmax_m,
    const float* __restrict__ softmax_inv_z,
    size_t bh,
    int max_S,
    int pos)
{
    if (weights != nullptr) {
        return weights[bh * (size_t)max_S + (size_t)pos];
    }
    return __expf(scores[bh * (size_t)max_S + (size_t)pos] - softmax_m[bh])
         * softmax_inv_z[bh];
}


// Kernel B (INT4): split-S SMEM-staged V multiply + Phase-2 top-K.
//
// Per-row V is now D/2 = 64 bytes.  Chunk SMEM v_buf: 2 × BS × 64 bytes
// (2 KB at BS=16 — vs 1 KB for INT2).
//
// cp.async strategy: 128 threads, 1024 bytes per chunk → 8 bytes per thread,
// issued as 2× 4-byte cp.async transactions (within the same commit_group).
//   thread tid → pos_in_blk = tid >> 3,  byte_off = (tid & 7) * 8
//   transaction 0 covers bytes [byte_off,   byte_off+4)
//   transaction 1 covers bytes [byte_off+4, byte_off+8)
template <int BS, int NG>
__global__ void weighted_v_split_kernel(
    const float*         __restrict__ weights,      // (B, H_q, max_S)
    const uint8_t*       __restrict__ v_data,       // (H_kv, num_slots, D/2)
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

    // SMEM layout (DOUBLE-BUFFERED — INT4 sizes):
    //   v_buf  : 2 × BS * 64 bytes   (2048 B for BS=16)
    //   vs_buf : 2 × BS × NG * 2 bytes
    //   w_buf  : 2 × BS * 4 bytes    (128 B)
    const bool do_pre_corr = (topk_idx != nullptr);

    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* v_buf = smem_raw;                                       // [2][BS*64]
    __nv_bfloat16* vs_buf =
        reinterpret_cast<__nv_bfloat16*>(v_buf + 2 * BS * D_HALF);   // [2][BS][NG]
    float* w_buf = reinterpret_cast<float*>(vs_buf + 2 * BS * NG);   // [2][BS]

    const int d = tid;
    if (d >= D_FIXED) return;
    // INT4 unpack: 2 codes per byte, high nibble = even lane, low = odd.
    const int byte_x = d >> 1;            // 0..63
    const int sub    = d &  1;            // 0=even (high), 1=odd (low)
    const int shift  = (1 - sub) * 4;     // 4 or 0
    const int group_id = (NG == 1) ? 0 : (d / GROUP_SIZE);

    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const float*         w_row        = weights  + ((size_t)b * H_q + (size_t)h_q) * max_S;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    auto load_chunk_async = [&](int blk, int buf) {
        const int64_t block_id = bt_row[blk];
        const int64_t base_slot = block_id * (int64_t)BS;
        const bool valid_block = (block_id >= 0);

        // V data: 128 threads × 8 bytes = 1024 B (= BS * 64).
        // Each thread issues 2 cp.async at byte_off and byte_off+4.
        const int pos_in_blk = tid >> 3;          // 0..15  (BS rows)
        const int byte_off   = (tid & 7) << 3;    // 0..56  (8 byte-groups × 8 B)

        // Transaction 0: [byte_off, byte_off+4)
        {
            void* smem_ptr =
                v_buf + (size_t)buf * BS * D_HALF
                      + (size_t)pos_in_blk * D_HALF + byte_off;
            if (valid_block) {
                const void* gmem_ptr =
                    v_pack_base + (size_t)(base_slot + pos_in_blk) * D_HALF + byte_off;
                unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 4;\n"
                    :: "r"(smem_int), "l"(gmem_ptr));
            } else {
                uint8_t* dst = reinterpret_cast<uint8_t*>(smem_ptr);
                #pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = 0x88;
            }
        }
        // Transaction 1: [byte_off+4, byte_off+8)
        {
            void* smem_ptr =
                v_buf + (size_t)buf * BS * D_HALF
                      + (size_t)pos_in_blk * D_HALF + byte_off + 4;
            if (valid_block) {
                const void* gmem_ptr =
                    v_pack_base + (size_t)(base_slot + pos_in_blk) * D_HALF + byte_off + 4;
                unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 4;\n"
                    :: "r"(smem_int), "l"(gmem_ptr));
            } else {
                uint8_t* dst = reinterpret_cast<uint8_t*>(smem_ptr);
                #pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = 0x88;
            }
        }

        // V scales — BS * NG bf16 values.  For NG=2, stage both
        // groups so dimensions 0..63 and 64..127 use their own scale.
        if (tid < BS * NG) {
            int scale_pos = tid / NG;
            int scale_gid = tid % NG;
            vs_buf[(buf * BS + scale_pos) * NG + scale_gid] = valid_block
                ? __ldg(v_scale_base + (size_t)(base_slot + scale_pos) * NG + scale_gid)
                : __float2bfloat16(0.0f);
        }

        // Weights — 16 fp32, masked to [s_start, s_end).
        if (tid < BS) {
            int p = blk * BS + tid;
            w_buf[buf * BS + tid] =
                (p >= s_start && p < s_end) ? w_row[p] : 0.0f;
        }
    };

    auto compute_chunk = [&](int buf, float& acc_io) {
        const uint8_t* v_cur = v_buf + (size_t)buf * BS * D_HALF;
        const __nv_bfloat16* vs_cur = vs_buf + buf * BS * NG;
        const float* w_cur = w_buf + buf * BS;
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            uint8_t pk = v_cur[j * D_HALF + byte_x];
            int q = int((pk >> shift) & 0xF) - 8;
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
            int64_t block_id = bt_row[blk_idx];
            if (block_id < 0) continue;
            int64_t slot = block_id * (int64_t)BS + (int64_t)in_blk;

            uint8_t pk = __ldg(v_pack_base + slot * D_HALF + byte_x);
            int q = int((pk >> shift) & 0xF) - 8;
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


torch::Tensor int4_softmax_v(
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF,
        "INT4 v_data must be (H_kv, num_slots, D/2=", D_HALF, ")");
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
        const size_t smem_b = (size_t)2 * BS * D_HALF  // v_buf  [2][BS*D_HALF]
                            + (size_t)2 * BS * NG * 2  // vs_buf [2][BS][NG]
                            + (size_t)2 * BS * 4       // w_buf
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
            "int4_softmax_v: unsupported (block_size=", block_size,
            ", num_groups=", num_groups, ").  Supported: (16,1), (16,2).");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return o_hat;
}


// pre_corr fork — adds top-K bookkeeping (o_old_unnorm, W).
std::vector<torch::Tensor> int4_softmax_v_with_pre_corr(
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
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
        const size_t smem_b = (size_t)2 * BS * D_HALF
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
            "int4_softmax_v_with_pre_corr: unsupported (block_size=", block_size,
            ", num_groups=", num_groups, ").");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {o_hat, o_old_unnorm, W_out};
}


// ────────────────────────────────────────────────────────────────────────
// wvs2 path: pre-multiplied (w · vs) at chunk-load time, INT4 sibling.
// Same semantics as INT2 wvs2 but with INT4 byte width / unpack.
// ────────────────────────────────────────────────────────────────────────

template <int BS, int SPLITS = N_SPLITS, int NG = 1>
__global__ void weighted_v_split_kernel_wvs2(
    const float*         __restrict__ weights,      // (B, H_q, max_S)
    const float*         __restrict__ scores,       // optional when weights == nullptr
    const float*         __restrict__ softmax_m,    // (B, H_q)
    const float*         __restrict__ softmax_inv_z,
    const uint8_t*       __restrict__ v_data,       // (H_kv, num_slots, D/2)
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

    // SMEM layout (INT4 sizes):
    //   v_buf : 2 × BS × 64 bytes   (2048 B for BS=16)
    //   ws_buf: 2 × BS × NG × 4 bytes.
    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* v_buf = smem_raw;                                          // [2][BS*D_HALF]
    float*   ws_buf = reinterpret_cast<float*>(v_buf + 2 * BS * D_HALF); // [2][BS][NG]
    uint8_t* topk_raw = reinterpret_cast<uint8_t*>(ws_buf + 2 * BS * NG);
    int64_t* topk_slot_buf = reinterpret_cast<int64_t*>(topk_raw);
    float*   topk_ws_buf =
        reinterpret_cast<float*>(topk_slot_buf + WVS2_TOPK_CHUNK);
    float*   topk_w_buf = topk_ws_buf + WVS2_TOPK_CHUNK * NG;

    const int d = tid;
    if (d >= D_FIXED) return;
    const int byte_x = d >> 1;
    const int sub    = d &  1;
    const int shift  = (1 - sub) * 4;
    constexpr int GROUP_SIZE = D_FIXED / NG;
    const int group_id = (NG == 1) ? 0 : (d / GROUP_SIZE);

    const size_t bh = (size_t)b * H_q + (size_t)h_q;
    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    auto load_chunk_async = [&](int blk, int buf) {
        const int64_t block_id = bt_row[blk];
        const int64_t base_slot = block_id * (int64_t)BS;
        const bool valid_block = (block_id >= 0);

        // V data: 128 thr × 8 B = 1024 B (= BS × 64). Two cp.async per thread.
        const int pos_in_blk = tid >> 3;
        const int byte_off   = (tid & 7) << 3;

        {
            void* smem_ptr =
                v_buf + (size_t)buf * BS * D_HALF
                      + (size_t)pos_in_blk * D_HALF + byte_off;
            if (valid_block) {
                const void* gmem_ptr =
                    v_pack_base + (size_t)(base_slot + pos_in_blk) * D_HALF + byte_off;
                unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 4;\n"
                    :: "r"(smem_int), "l"(gmem_ptr));
            } else {
                uint8_t* dst = reinterpret_cast<uint8_t*>(smem_ptr);
                #pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = 0x88;
            }
        }
        {
            void* smem_ptr =
                v_buf + (size_t)buf * BS * D_HALF
                      + (size_t)pos_in_blk * D_HALF + byte_off + 4;
            if (valid_block) {
                const void* gmem_ptr =
                    v_pack_base + (size_t)(base_slot + pos_in_blk) * D_HALF + byte_off + 4;
                unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 4;\n"
                    :: "r"(smem_int), "l"(gmem_ptr));
            } else {
                uint8_t* dst = reinterpret_cast<uint8_t*>(smem_ptr);
                #pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = 0x88;
            }
        }

        // Pre-multiplied (w * vs), one value per slot and scale group.
        if (tid < BS * NG) {
            const int j = tid / NG;
            const int g = tid - j * NG;
            int p = blk * BS + j;
            float ws = 0.0f;
            if (valid_block && p >= s_start && p < s_end) {
                float vs = __bfloat162float(__ldg(
                    v_scale_base + ((size_t)(base_slot + j) * NG + g)));
                float w  = load_weight_or_score(
                    weights, scores, softmax_m, softmax_inv_z, bh, max_S, p);
                ws = w * vs;
            }
            ws_buf[(buf * BS + j) * NG + g] = ws;
        }
    };

    auto compute_chunk = [&](int buf, float& acc_io) {
        const uint8_t* v_cur  = v_buf + (size_t)buf * BS * D_HALF;
        const float*   ws_cur = ws_buf + buf * BS * NG;
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            uint8_t pk = v_cur[j * D_HALF + byte_x];
            int q = int((pk >> shift) & 0xF) - 8;
            acc_io = __fmaf_rn(ws_cur[j * NG + group_id], (float)q, acc_io);
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
                            w = load_weight_or_score(
                                weights, scores, softmax_m, softmax_inv_z,
                                bh, max_S, (int)pos);
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
                uint8_t pk = __ldg(v_pack_base + slot * D_HALF + byte_x);
                int q = int((pk >> shift) & 0xF) - 8;
                acc_o_old = __fmaf_rn(topk_ws_buf[j * NG + group_id], (float)q, acc_o_old);
                if (d == 0) acc_W += topk_w_buf[j];
            }
            __syncthreads();
        }
        o_old_unnorm[bh * D_FIXED + d] = acc_o_old;
        if (d == 0) W_out[bh] = acc_W;
    }
}


std::vector<torch::Tensor> int4_softmax_v_with_pre_corr_wvs2(
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
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
    const size_t smem_b = (size_t)2 * BS_C * D_HALF   // v_buf
                        + (size_t)2 * BS_C * num_groups * 4  // ws_buf (fp32)
                        + WVS2_TOPK_SMEM              // Phase-2 topK slot/ws/w
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
            static_cast<const float*>(nullptr),
            static_cast<const float*>(nullptr),
            static_cast<const float*>(nullptr),
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
    } else {
        launch_by_ng(std::integral_constant<int, N_SPLITS>{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}

void int4_softmax_v_with_pre_corr_wvs2_inplace(
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
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
    const size_t smem_b = (size_t)2 * BS_C * D_HALF
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
            static_cast<const float*>(nullptr),
            static_cast<const float*>(nullptr),
            static_cast<const float*>(nullptr),
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
    } else {
        launch_by_ng(std::integral_constant<int, N_SPLITS>{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ────────────────────────────────────────────────────────────────────────
// v2 (Phase 1): WMMA-based weighted_v_split.  Promotes the per-thread
// scalar FMA loop in wvs2 to a 16x16x16 BF16 WMMA tile.
//
// Grid: (B, H_kv, N_SPLITS) — groups H_q/H_kv = 4 query heads per CTA so
//       the WMMA M dimension carries 4 real rows (12 zero-padded rows).
// Block: 128 threads (4 warps).  Each warp owns 2 N-tiles (cols 0..15 +
//       cols 16..31, etc.); 4 warps × 2 tiles = 8 tiles cover D=128.
//
// Numerical note vs wvs2: wvs2 keeps `ws = w*v_scale` in FP32; v2 must
// downcast w to BF16 for the WMMA A operand.  Per-FMA error ~4e-3,
// accumulated as RMS √BS·eps ≈ 1.6e-2 worst-case.  Unit test atol/rtol
// loosened accordingly.
//
// Pass 2 (top-K correction) is preserved scalar — it processes only K
// positions (default 256) so it is not a critical-path target.
//
// Constraints:
//   * BS = 16 (matches wvs2 production constraint and WMMA K_TILE)
//   * NG = 1 (per-slot scale; matches wvs2)
//   * H_q must be a multiple of H_kv with n_groups_per_head ∈ {1, 2, 4, 8}
//     (the M-padding tolerates any n_groups_per_head ≤ 16; LLaMA-3.1-8B = 4).
// ────────────────────────────────────────────────────────────────────────

template <int BS>
__global__ void weighted_v_split_kernel_wmma_v2(
    const float*         __restrict__ weights,      // (B, H_q, max_S)
    const float*         __restrict__ scores,       // optional when weights == nullptr
    const float*         __restrict__ softmax_m,    // (B, H_q)
    const float*         __restrict__ softmax_inv_z,
    const uint8_t*       __restrict__ v_data,       // (H_kv, num_slots, D/2)
    const __nv_bfloat16* __restrict__ v_scales,     // (H_kv, num_slots, 1) — NG=1
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
    using namespace nvcuda::wmma;

    constexpr int M_TILE = 16;
    constexpr int N_TILE = 16;
    constexpr int K_TILE = 16;                 // = BS
    constexpr int N_TILES_TOTAL    = D_FIXED / N_TILE;            // 8
    constexpr int N_TILES_PER_WARP = N_TILES_TOTAL / 4;           // 2
    constexpr int V_PAD            = 8;                           // bf16 pad cols
    constexpr int V_STRIDE         = D_FIXED + V_PAD;             // 136
    static_assert(BS == K_TILE, "v2 requires BS=K_TILE=16");

    const int b       = blockIdx.x;
    const int h_kv    = blockIdx.y;
    const int s_split = blockIdx.z;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;

    const int n_groups_per_head = H_q / H_kv;     // 4 for LLaMA-3.1-8B
    const int S = seq_lens[b];

    const int chunk_per_split = (max_S + N_SPLITS - 1) / N_SPLITS;
    const int s_start = s_split * chunk_per_split;
    const int s_end   = min(s_start + chunk_per_split, S);
    if (s_start >= S) return;

    const bool do_pre_corr = (topk_idx != nullptr);

    // ── Dynamic SMEM layout ──────────────────────────────────────────────
    // v_pack    [2][BS][D_HALF]                bytes   2 * 16 * 64 = 2048
    // v_scl     [2][BS]                  bf16          2 * 16 * 2  =   64
    // w_bf      [M_TILE][BS]             bf16              16 * 16 * 2 =  512
    // v_bf      [BS][V_STRIDE]           bf16              16 * 136 * 2 = 4352
    // c_scratch [N_TILES_TOTAL][M_TILE][N_TILE]  fp32  8 * 16 * 16 * 4 = 8192
    extern __shared__ __align__(16) uint8_t smem_raw[];
    size_t off = 0;
    auto v_pack = reinterpret_cast<uint8_t(*)[BS][D_HALF]>(&smem_raw[off]);
    off += 2 * BS * D_HALF;
    auto v_scl  = reinterpret_cast<__nv_bfloat16(*)[BS]>(&smem_raw[off]);
    off += 2 * BS * sizeof(__nv_bfloat16);
    auto w_bf   = reinterpret_cast<__nv_bfloat16(*)[BS]>(&smem_raw[off]);
    off += M_TILE * BS * sizeof(__nv_bfloat16);
    auto v_bf   = reinterpret_cast<__nv_bfloat16(*)[V_STRIDE]>(&smem_raw[off]);
    off += BS * V_STRIDE * sizeof(__nv_bfloat16);
    auto c_scratch =
        reinterpret_cast<float(*)[M_TILE][N_TILE]>(&smem_raw[off]);
    // off += N_TILES_TOTAL * M_TILE * N_TILE * sizeof(float);  // unused after

    // ── Source pointers ──────────────────────────────────────────────────
    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    // ── Async load V_int4 + scales for one block ─────────────────────────
    auto load_chunk_async = [&](int blk, int buf) {
        const int64_t base_slot = bt_row[blk] * (int64_t)BS;
        // V data: 1024 B = 128 thr × 8 B; 2× 4-byte cp.async per thread.
        const int pos_in_blk = tid >> 3;          // 0..15  (BS rows)
        const int byte_off   = (tid & 7) << 3;    // 0..56  (8 byte-groups × 8 B)
        {
            const void* g = v_pack_base + (size_t)(base_slot + pos_in_blk) * D_HALF + byte_off;
            void* s = (char*)&v_pack[buf][pos_in_blk][byte_off];
            unsigned si = __cvta_generic_to_shared(s);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                         :: "r"(si), "l"(g));
        }
        {
            const void* g = (const char*)v_pack_base + (size_t)(base_slot + pos_in_blk) * D_HALF + byte_off + 4;
            void* s = (char*)&v_pack[buf][pos_in_blk][byte_off] + 4;
            unsigned si = __cvta_generic_to_shared(s);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                         :: "r"(si), "l"(g));
        }
        // V scales (BF16) — sync ldg (16 entries).
        if (tid < BS) {
            v_scl[buf][tid] = __ldg(v_scale_base + (size_t)(base_slot + tid));
        }
    };

    // ── Sync-load weights for one block, FP32 → BF16, padded to (16,BS).
    // Real rows: [0, n_groups_per_head); padded rows: [n_groups_per_head, M_TILE).
    auto load_weights_sync = [&](int blk) {
        // 128 threads cover 16 rows × BS=16 = 256 elements (2 per thread, but
        // we use only one assignment per thread for clarity since BS=16).
        const int total = M_TILE * BS;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            int h_local = idx / BS;
            int j_local = idx % BS;
            float w = 0.0f;
            if (h_local < n_groups_per_head) {
                int p = blk * BS + j_local;
                if (p >= s_start && p < s_end) {
                    int h_q = h_kv * n_groups_per_head + h_local;
                    size_t bh = (size_t)b * H_q + (size_t)h_q;
                    w = load_weight_or_score(
                        weights, scores, softmax_m, softmax_inv_z,
                        bh, max_S, p);
                }
            }
            w_bf[h_local][j_local] = __float2bfloat16(w);
        }
    };

    // ── Dequantize INT4 V → BF16 v_bf, applying per-row scale.
    // 16 rows × 64 packed bytes → 16 rows × 128 bf16 values.
    auto dequant_to_bf16 = [&](int buf) {
        const int total = BS * D_HALF;            // 1024 byte-positions
        for (int idx = tid; idx < total; idx += blockDim.x) {
            int r  = idx / D_HALF;
            int bx = idx % D_HALF;
            uint8_t pk = v_pack[buf][r][bx];
            float vs = __bfloat162float(v_scl[buf][r]);
            int q_even = int((pk >> 4) & 0xF) - 8;
            int q_odd  = int( pk       & 0xF) - 8;
            v_bf[r][2 * bx + 0] = __float2bfloat16((float)q_even * vs);
            v_bf[r][2 * bx + 1] = __float2bfloat16((float)q_odd  * vs);
        }
    };

    // ── WMMA fragments ───────────────────────────────────────────────────
    // A = w_bf  (M=16, K=16, row_major).
    // B = v_bf  (K=16, N=16, row_major) — view as (K rows × N cols), where
    //   v_bf[k, n] = dequantized V at S-position k, head-dim n.  Since each
    //   warp covers 2 N-tiles, we load B fresh for each tile.
    fragment<matrix_a, M_TILE, N_TILE, K_TILE, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, M_TILE, N_TILE, K_TILE, __nv_bfloat16, row_major> b_frag;
    fragment<accumulator, M_TILE, N_TILE, K_TILE, float> c_frag[N_TILES_PER_WARP];
    #pragma unroll
    for (int n = 0; n < N_TILES_PER_WARP; ++n) fill_fragment(c_frag[n], 0.f);

    // ── Pass 1: pipelined V-load + WMMA accumulation ──────────────────────
    if (blk_start < blk_end) {
        // Prologue: prime buffer 0.
        load_chunk_async(blk_start, 0);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        int cur = 0;
        for (int blk = blk_start; blk < blk_end; ++blk) {
            const int nxt = 1 - cur;
            const bool has_next = (blk + 1 < blk_end);
            if (has_next) {
                load_chunk_async(blk + 1, nxt);
                asm volatile("cp.async.commit_group;\n");
            }

            // Sync-load weights for THIS block (small).
            load_weights_sync(blk);

            // Dequantize V_int4 (cur buf) → v_bf.
            dequant_to_bf16(cur);
            __syncthreads();          // ensure w_bf and v_bf visible to all warps

            // WMMA: each warp owns 2 N-tiles starting at warp_id*N_TILES_PER_WARP.
            const int n_base = warp_id * N_TILES_PER_WARP;
            load_matrix_sync(a_frag, &w_bf[0][0], BS);
            #pragma unroll
            for (int n = 0; n < N_TILES_PER_WARP; ++n) {
                int n_tile = n_base + n;
                load_matrix_sync(b_frag, &v_bf[0][n_tile * N_TILE], V_STRIDE);
                mma_sync(c_frag[n], a_frag, b_frag, c_frag[n]);
            }

            if (has_next) {
                asm volatile("cp.async.wait_group 0;\n");
                __syncthreads();
                cur = nxt;
            } else {
                __syncthreads();      // drain before scratch reuse
            }
        }
    }

    // ── Writeback: store fragments to c_scratch, atomicAdd to o_hat ──────
    {
        const int n_base = warp_id * N_TILES_PER_WARP;
        #pragma unroll
        for (int n = 0; n < N_TILES_PER_WARP; ++n) {
            int n_tile = n_base + n;
            store_matrix_sync(&c_scratch[n_tile][0][0], c_frag[n],
                              N_TILE, mem_row_major);
        }
        __syncthreads();

        // 4 valid heads × 128 cols = 512 atomicAdds per CTA, distributed over
        // 128 threads → 4 atomicAdds per thread.
        const int total = n_groups_per_head * D_FIXED;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            int h_local = idx / D_FIXED;
            int d       = idx % D_FIXED;
            int n_tile  = d / N_TILE;
            int n_off   = d % N_TILE;
            float v = c_scratch[n_tile][h_local][n_off];
            int h_q = h_kv * n_groups_per_head + h_local;
            atomicAdd(o_hat + ((size_t)b * H_q + h_q) * D_FIXED + d, v);
        }
    }

    // ── Pass 2: top-K correction (scalar; only on s_split == 0) ──────────
    if (do_pre_corr && s_split == 0) {
        // Each thread handles d=tid for ALL 4 query heads (sequential).
        const int d = tid;
        if (d < D_FIXED) {
            const int byte_x = d >> 1;
            const int sub    = d &  1;
            const int shift  = (1 - sub) * 4;
            #pragma unroll 1
            for (int h_local = 0; h_local < n_groups_per_head; ++h_local) {
                int h_q = h_kv * n_groups_per_head + h_local;
                const size_t bh = (size_t)b * H_q + (size_t)h_q;
                const int64_t* topk_row =
                    topk_idx + ((size_t)b * H_q + (size_t)h_q) * (size_t)K;

                float acc_o_old = 0.0f;
                float acc_W     = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int64_t pos = topk_row[k];
                    if (pos < 0 || (int)pos >= S) continue;
                    int blk_idx = (int)(pos / BS);
                    int in_blk  = (int)(pos % BS);
                    int64_t slot = bt_row[blk_idx] * (int64_t)BS + (int64_t)in_blk;

                    uint8_t pk = __ldg(v_pack_base + slot * D_HALF + byte_x);
                    int q = int((pk >> shift) & 0xF) - 8;
                    float vs = __bfloat162float(__ldg(v_scale_base + slot));
                    float w  = load_weight_or_score(
                        weights, scores, softmax_m, softmax_inv_z,
                        bh, max_S, (int)pos);
                    float ws = w * vs;

                    acc_o_old = __fmaf_rn(ws, (float)q, acc_o_old);
                    if (d == 0) acc_W += w;
                }
                o_old_unnorm[bh * D_FIXED + d] = acc_o_old;
                if (d == 0) W_out[bh] = acc_W;
            }
        }
    }
}


// ────────────────────────────────────────────────────────────────────────
// v3 (Phase 2): occupancy-tuned WMMA weighted_v_split.
//
// Diagnosis (ncu, H100 PCIe, B=32 ctx=32k):
//   * Achieved occupancy 62.6 % (40 warps/SM, theoretical max 75 %).
//   * Compute SM throughput 72.6 % — the actual bottleneck.
//   * DRAM throughput 15.2 % — HBM is NOT saturated.
//   * No-Eligible cycles 23.7 % (not 80 % — the small-scale 80 % figure
//     was from a B=4 ctx=4k grid that does not even fill the SMs once).
//   * ALU pipe 58.5 % utilised — dominated by int4-byte unpack and the
//     per-element FP32→BF16 conversion in dequant.
//
// Changes vs v2 (additive — v2 left untouched):
//   1.  Reuse the c_scratch SMEM region during the WMMA loop as v_bf
//       (BF16 16×136 = 4352 B).  v_bf is dead after the loop, so the
//       same 8192 B slot serves both purposes.  Net SMEM: 13 KB vs v2's
//       15.2 KB.
//   2.  Hoist the per-byte BF16→FP32 scale conversion: walk r-outer and
//       compute `vs` ONCE per (thread, row) per call (vs v2's redundant
//       8× per byte).  Cuts ALU pipe pressure measurably.
//   3.  8-byte cp.async (cp.async.ca.shared.global ... 8) instead of v2's
//       two 4-byte issues.  Halves the cp.async instruction count from
//       the issue pipe (v2: 256 issues / CTA → v3: 128).
//   4.  uint64 SMEM read + uint32 packed BF16 SMEM store in dequant —
//       cuts shared-memory instruction count vs v2's byte-strided pattern.
//   5.  Static M-padding: zero rows [n_groups_per_head, M_TILE) of w_bf
//       ONCE at CTA start instead of every block (saves ~192 stores per
//       block × 256 blocks = 49 K wasted stores per CTA).
//   6.  3-stage cp.async pipeline (vs v2's 2-stage) — gives one extra
//       prefetch in flight to hide gather latency without the spilling
//       seen at 4+ stages.
//
// Total dynamic SMEM @ BS=16, STAGES=3:
//   v_pack [3][BS][D_HALF]                        3 * 16 * 64  = 3072 B
//   v_scl  [3][BS]                       bf16     3 * 16 * 2   =   96 B
//   w_bf   [M_TILE][BS]                  bf16    16 * 16 * 2   =  512 B
//   reuse  max(v_bf 4352, c_scratch 8192)                      = 8192 B
//                                                              -------
//                                                               11872 B
//   + alignment slack                                              64 B
//                                                              -------
//                                                              ~12 KB
//
// vs v2 = 15168 B → ~3.2 KB cut.  Same achieved occupancy (compute-bound,
// not SMEM-bound), so the SMEM cut is a wash at production — the actual
// gain comes from instruction-count reductions in dequant + cp.async.
// ────────────────────────────────────────────────────────────────────────

constexpr int N_SPLITS_V3 = 8;       // match v2
constexpr int CP_STAGES_V3 = 3;      // 3-stage cp.async pipeline — small bump
                                     // over v2's 2-stage helps overlap dequant
                                     // with HBM gather without paying spill.

template <int BS>
__global__ void weighted_v_split_kernel_wmma_v3(
    const float*         __restrict__ weights,      // (B, H_q, max_S)
    const float*         __restrict__ scores,       // optional when weights == nullptr
    const float*         __restrict__ softmax_m,    // (B, H_q)
    const float*         __restrict__ softmax_inv_z,
    const uint8_t*       __restrict__ v_data,       // (H_kv, num_slots, D/2)
    const __nv_bfloat16* __restrict__ v_scales,     // (H_kv, num_slots, 1) — NG=1
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
    using namespace nvcuda::wmma;

    constexpr int M_TILE = 16;
    constexpr int N_TILE = 16;
    constexpr int K_TILE = 16;                              // = BS
    constexpr int N_TILES_TOTAL    = D_FIXED / N_TILE;      // 8
    constexpr int N_TILES_PER_WARP = N_TILES_TOTAL / 4;     // 2
    constexpr int V_PAD            = 8;                     // bf16 pad cols
    constexpr int V_STRIDE         = D_FIXED + V_PAD;       // 136 — matches v2;
                                                            //  drops the 16-way
                                                            //  SMEM bank conflict
                                                            //  that came from
                                                            //  row stride = 32×4 B.
    constexpr int STAGES           = CP_STAGES_V3;
    static_assert(BS == K_TILE, "v3 requires BS=K_TILE=16");

    const int b       = blockIdx.x;
    const int h_kv    = blockIdx.y;
    const int s_split = blockIdx.z;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;

    const int n_groups_per_head = H_q / H_kv;
    const int S = seq_lens[b];

    const int chunk_per_split = (max_S + N_SPLITS_V3 - 1) / N_SPLITS_V3;
    const int s_start = s_split * chunk_per_split;
    const int s_end   = min(s_start + chunk_per_split, S);
    if (s_start >= S) return;

    const bool do_pre_corr = (topk_idx != nullptr);

    // ── Dynamic SMEM layout ──────────────────────────────────────────────
    // v_pack [STAGES][BS][D_HALF]            STAGES * 16 * 64
    // v_scl  [STAGES][BS]              bf16  STAGES * 16 * 2
    // w_bf   [M_TILE][BS]              bf16  16 * 16 * 2
    // v_bf   [BS][V_STRIDE]            bf16  16 * 128 * 2   (REUSED as
    //                                       4×(M_TILE×N_TILE) FP32 c-staging
    //                                       after the loop)
    extern __shared__ __align__(16) uint8_t smem_raw[];
    size_t off = 0;
    auto v_pack = reinterpret_cast<uint8_t(*)[BS][D_HALF]>(&smem_raw[off]);
    off += (size_t)STAGES * BS * D_HALF;
    auto v_scl  = reinterpret_cast<__nv_bfloat16(*)[BS]>(&smem_raw[off]);
    off += (size_t)STAGES * BS * sizeof(__nv_bfloat16);
    auto w_bf   = reinterpret_cast<__nv_bfloat16(*)[BS]>(&smem_raw[off]);
    off += M_TILE * BS * sizeof(__nv_bfloat16);
    uint8_t* v_bf_raw = &smem_raw[off];
    auto v_bf   = reinterpret_cast<__nv_bfloat16(*)[V_STRIDE]>(v_bf_raw);
    // The same SMEM region is reused after the WMMA loop as c_scratch
    // (FP32 [N_TILES_TOTAL][M_TILE][N_TILE] = 8192 B).  Host launcher
    // reserves max(v_bf=4096 B, c_scratch=8192 B) = 8192 B for this slot.
    // Saving vs v2 = V_PAD removal (256 B) + double-allocation removal (no
    // need to keep v_bf alive after WMMA).  Net SMEM: 10880 B vs v2's
    // 15232 B → ~4.4 KB cut per CTA.

    // ── Source pointers ──────────────────────────────────────────────────
    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    // ── Async load V_int4 + scales for one block into a staging slot ─────
    // 1024 B / block = 128 threads × 8 B.  One 8-byte cp.async per thread
    // halves the cp.async instruction count vs v2's two 4-byte issues,
    // reducing issue-pipe pressure (No-Eligible cycles cited by ncu).
    auto load_chunk_async = [&](int blk, int slot) {
        const int64_t base_slot = bt_row[blk] * (int64_t)BS;
        const int pos_in_blk = tid >> 3;          // 0..15  (BS rows)
        const int byte_off   = (tid & 7) << 3;    // 0..56  (8-byte groups)
        const void* g = v_pack_base + (size_t)(base_slot + pos_in_blk) * D_HALF + byte_off;
        void* s = (char*)&v_pack[slot][pos_in_blk][byte_off];
        unsigned si = __cvta_generic_to_shared(s);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                     :: "r"(si), "l"(g));
        if (tid < BS) {
            v_scl[slot][tid] = __ldg(v_scale_base + (size_t)(base_slot + tid));
        }
    };

    // Sync-load weights for one block, FP32 → BF16, padded to (M_TILE,BS).
    // We zero the padded rows ONCE outside the loop (below); inside this
    // call we only touch h_local ∈ [0, n_groups_per_head).
    auto load_weights_sync = [&](int blk) {
        // 128 threads × (n_groups_per_head × BS / 128) entries.  For
        // production (n_groups_per_head=4) that is 64 entries → 2 lanes
        // load 0, but more commonly each thread loads at most 1.
        const int total = n_groups_per_head * BS;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            int h_local = idx / BS;
            int j_local = idx % BS;
            int p = blk * BS + j_local;
            float w = 0.0f;
            if (p >= s_start && p < s_end) {
                int h_q = h_kv * n_groups_per_head + h_local;
                size_t bh = (size_t)b * H_q + (size_t)h_q;
                w = load_weight_or_score(
                    weights, scores, softmax_m, softmax_inv_z,
                    bh, max_S, p);
            }
            w_bf[h_local][j_local] = __float2bfloat16(w);
        }
    };

    // ── Dequantize INT4 V → BF16 v_bf, applying per-row scale.
    //
    // Warp-partitioned layout: warp w (w ∈ [0,4)) owns v_bf columns
    // 32w..32w+31, which come from v_pack bytes 16w..16w+15 (each byte
    // produces 2 v_bf cols).  Each warp's 32 lanes cover its 16 rows ×
    // 16 bytes = 256 byte-positions = 8 per lane.  Because each warp
    // writes only its own band of v_bf and ALSO READS only its own band
    // via WMMA load_matrix_sync, no cross-warp sync is needed between
    // dequant and WMMA — the post-dequant __syncthreads in v2 is
    // replaced by __syncwarp() here, halving the sync count in the
    // inner loop.
    //
    // Each lane:
    //   * computes vs = bf16 row scale  ONCE per call.
    //   * 8-byte SMEM load (uint64_t) from v_pack for its 8 bytes.
    //   * writes 8 __nv_bfloat162 (= 16 BF16 = 32 B) into v_bf.
    auto dequant_to_bf16 = [&](int slot) {
        constexpr int BYTES_PER_LANE = 8;
        constexpr int BYTES_PER_WARP = 16;
        // lane_in_warp = tid % 32, w = warp_id.
        const int lane_in_warp = tid & 31;
        const int w_bx_base   = warp_id * BYTES_PER_WARP;
        // 32 lanes, 16 rows × 2 byte-sub-lanes per row.  Map:
        //   r          = lane_in_warp >> 1   (0..15)
        //   bx_local   = (lane_in_warp & 1) * BYTES_PER_LANE  (0 or 8)
        const int r        = lane_in_warp >> 1;
        const int bx_off   = w_bx_base + (lane_in_warp & 1) * BYTES_PER_LANE;

        const __nv_bfloat16 vs_h = v_scl[slot][r];
        const __nv_bfloat162 vs2 = __halves2bfloat162(vs_h, vs_h);

        // 8-byte SMEM load.
        const uint64_t pack8 =
            *reinterpret_cast<const uint64_t*>(&v_pack[slot][r][bx_off]);

        __nv_bfloat162* outp =
            reinterpret_cast<__nv_bfloat162*>(&v_bf[r][2 * bx_off]);

        #pragma unroll
        for (int k = 0; k < BYTES_PER_LANE; ++k) {
            uint8_t pk = (uint8_t)((pack8 >> (k * 8)) & 0xFF);
            int q_even = int((pk >> 4) & 0xF) - 8;
            int q_odd  = int( pk       & 0xF) - 8;
            __nv_bfloat16 be0 = __float2bfloat16((float)q_even);
            __nv_bfloat16 bo0 = __float2bfloat16((float)q_odd);
            __nv_bfloat162 q2 = __halves2bfloat162(be0, bo0);
            outp[k] = __hmul2(q2, vs2);
        }
    };

    // ── WMMA fragments ───────────────────────────────────────────────────
    fragment<matrix_a, M_TILE, N_TILE, K_TILE, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, M_TILE, N_TILE, K_TILE, __nv_bfloat16, row_major> b_frag;
    fragment<accumulator, M_TILE, N_TILE, K_TILE, float> c_frag[N_TILES_PER_WARP];
    #pragma unroll
    for (int n = 0; n < N_TILES_PER_WARP; ++n) fill_fragment(c_frag[n], 0.f);

    // Zero the M-padding rows of w_bf ONCE — they are static across all
    // blocks (only the first n_groups_per_head rows are reloaded each
    // iter).  Saves writing 192 zeros per block in load_weights_sync.
    {
        const int pad_total = (M_TILE - n_groups_per_head) * BS;
        for (int idx = tid; idx < pad_total; idx += blockDim.x) {
            int h_local = idx / BS + n_groups_per_head;
            int j_local = idx % BS;
            w_bf[h_local][j_local] = __float2bfloat16(0.0f);
        }
    }

    // ── Pass 1: pipelined V-load + WMMA accumulation (STAGES-deep) ───────
    // Ring-buffer with STAGES slots.  At any point, up to STAGES-1 prefetch
    // commits are in flight; we wait_group(STAGES-1) after each iteration's
    // commit to bring the oldest home.  Deeper pipeline → more HBM bytes
    // in flight, addressing the 80% No-Eligible cycle stall on v2.
    const int n_base = warp_id * N_TILES_PER_WARP;
    if (blk_start < blk_end) {
        const int n_blk = blk_end - blk_start;

        // Prologue: issue prefetches for the first min(STAGES, n_blk) blocks.
        const int n_prime = min(STAGES, n_blk);
        for (int p = 0; p < n_prime; ++p) {
            load_chunk_async(blk_start + p, p % STAGES);
            asm volatile("cp.async.commit_group;\n");
        }
        // Pad: emit no-op commits so the wait_group bookkeeping is uniform.
        for (int p = n_prime; p < STAGES; ++p) {
            asm volatile("cp.async.commit_group;\n");
        }

        int issue = n_prime;          // next blk-offset to issue (from blk_start)

        // STAGES≥3 invariant: the slot we are about to dequant
        // (cur_slot = i % STAGES) is DIFFERENT from the slot we issue a
        // prefetch into (issue % STAGES) as long as (issue - i) is in
        // [1, STAGES-1].  At iteration i:
        //   * issue = n_prime + i = i + STAGES   (in steady state)
        //   * issue - i = STAGES  → issue%STAGES == i%STAGES = cur_slot
        //   * BUT we issue BEFORE incrementing issue, so the slot we just
        //     issued into is (i + STAGES) % STAGES = cur_slot.
        // We DO write to the same slot we just consumed.  However, the
        // wait_group + sync at the top of the next iteration provides the
        // necessary barrier: the cp.async writes (issued AT iter i) will
        // not be visible until we explicitly wait_group for them at the
        // start of iter i+STAGES, which is the next time we consume that
        // slot.  Dequant in iter i finishes reading v_pack[cur_slot]
        // before the cp.async at end-of-iter-i is issued (issue happens
        // strictly AFTER load_weights+dequant in this body), so the
        // overwrite race is structurally avoided WITHOUT the second sync.
        for (int i = 0; i < n_blk; ++i) {
            const int blk = blk_start + i;
            const int cur_slot = i % STAGES;

            // Start writing weights for THIS block BEFORE the wait_group so
            // it overlaps with the cp.async-completion stall.  w_bf writes
            // are independent of v_pack — no race.
            load_weights_sync(blk);

            // Bring slot `cur_slot` home (leave up to STAGES-1 in flight).
            asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1));
            __syncthreads();          // make cp.async data AND w_bf visible
                                     // to ALL warps (single barrier covers
                                     // both fences).

            // Dequantize V_int4 (cur slot) → v_bf.  Warp-partitioned: warp
            // w writes only v_bf cols [32w, 32w+32) — no cross-warp
            // dependence on this region.
            dequant_to_bf16(cur_slot);
            __syncwarp();             // make v_bf visible within THIS warp
                                     // (sufficient: WMMA only reads warp w's
                                     // own 2 N-tile band).

            // WMMA: each warp owns N_TILES_PER_WARP=2 N-tiles.  Runs
            // concurrently with the cp.async issued below (cp.async is
            // non-blocking).
            load_matrix_sync(a_frag, &w_bf[0][0], BS);
            #pragma unroll
            for (int n = 0; n < N_TILES_PER_WARP; ++n) {
                int n_tile = n_base + n;
                load_matrix_sync(b_frag, &v_bf[0][n_tile * N_TILE], V_STRIDE);
                mma_sync(c_frag[n], a_frag, b_frag, c_frag[n]);
            }

            // Issue the next prefetch.  Race-free with the dequant above:
            // dequant's read of v_pack[cur_slot] completed before the
            // __syncthreads after dequant, and WMMA reads v_bf (not
            // v_pack), so v_pack[cur_slot] is dead.  The new cp.async
            // overwrites v_pack[cur_slot] which will not be read again
            // until iter i+STAGES (next time cur_slot comes round), at
            // which point we wait_group for it.
            if (issue < n_blk) {
                load_chunk_async(blk_start + issue, issue % STAGES);
                asm volatile("cp.async.commit_group;\n");
                ++issue;
            } else {
                asm volatile("cp.async.commit_group;\n");
            }
        }

        // Drain remaining commits before SMEM reuse for c_scratch.
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    } else {
        __syncthreads();
    }

    // ── Writeback: store fragments to c_stage, atomicAdd to o_hat ──────
    // c_stage[N_TILES_TOTAL][M_TILE][N_TILE] aliases v_bf (4096 B FP32 vs
    // 4096 B BF16).  Each warp owns 2 N-tiles in this region (different
    // N-tile indices, no aliasing).  Layout matches v2's c_scratch so the
    // writeback fan-out across 128 threads is identical (no per-warp
    // serialisation, no extra barriers).
    {
        // Reinterpret the v_bf region as c_stage[N_TILES_TOTAL][M_TILE][N_TILE].
        auto c_scratch =
            reinterpret_cast<float(*)[M_TILE][N_TILE]>(v_bf_raw);

        const int n_base = warp_id * N_TILES_PER_WARP;
        #pragma unroll
        for (int n = 0; n < N_TILES_PER_WARP; ++n) {
            int n_tile = n_base + n;
            store_matrix_sync(&c_scratch[n_tile][0][0], c_frag[n],
                              N_TILE, mem_row_major);
        }
        __syncthreads();

        // 4 valid heads × 128 cols = 512 atomicAdds, 4 per thread.
        const int total = n_groups_per_head * D_FIXED;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            int h_local = idx / D_FIXED;
            int d       = idx % D_FIXED;
            int n_tile  = d / N_TILE;
            int n_off   = d % N_TILE;
            float v = c_scratch[n_tile][h_local][n_off];
            int h_q = h_kv * n_groups_per_head + h_local;
            atomicAdd(o_hat + ((size_t)b * H_q + h_q) * D_FIXED + d, v);
        }
    }

    // ── Pass 2: top-K correction (scalar; only on s_split == 0) ──────────
    if (do_pre_corr && s_split == 0) {
        const int d = tid;
        if (d < D_FIXED) {
            const int byte_x = d >> 1;
            const int sub    = d &  1;
            const int shift  = (1 - sub) * 4;
            #pragma unroll 1
            for (int h_local = 0; h_local < n_groups_per_head; ++h_local) {
                int h_q = h_kv * n_groups_per_head + h_local;
                const size_t bh = (size_t)b * H_q + (size_t)h_q;
                const int64_t* topk_row =
                    topk_idx + ((size_t)b * H_q + (size_t)h_q) * (size_t)K;

                float acc_o_old = 0.0f;
                float acc_W     = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int64_t pos = topk_row[k];
                    if (pos < 0 || (int)pos >= S) continue;
                    int blk_idx = (int)(pos / BS);
                    int in_blk  = (int)(pos % BS);
                    int64_t slot = bt_row[blk_idx] * (int64_t)BS + (int64_t)in_blk;

                    uint8_t pk = __ldg(v_pack_base + slot * D_HALF + byte_x);
                    int q = int((pk >> shift) & 0xF) - 8;
                    float vs = __bfloat162float(__ldg(v_scale_base + slot));
                    float w  = load_weight_or_score(
                        weights, scores, softmax_m, softmax_inv_z,
                        bh, max_S, (int)pos);
                    float ws = w * vs;

                    acc_o_old = __fmaf_rn(ws, (float)q, acc_o_old);
                    if (d == 0) acc_W += w;
                }
                o_old_unnorm[bh * D_FIXED + d] = acc_o_old;
                if (d == 0) W_out[bh] = acc_W;
            }
        }
    }
}


// Host launcher mirroring int4_softmax_v_with_pre_corr_wmma_v2 but
// dispatching to the v3 occupancy-tuned kernel.
std::vector<torch::Tensor> int4_softmax_v_with_pre_corr_wmma_v3(
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
    TORCH_CHECK(H_q % H_kv == 0);
    TORCH_CHECK(block_size == 16, "wmma_v3 requires block_size=16");
    const int n_groups_per_head_check = (int)(H_q / H_kv);
    TORCH_CHECK(n_groups_per_head_check <= 16,
                "wmma_v3 requires H_q/H_kv <= 16 (got ", n_groups_per_head_check, ")");

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
    TORCH_CHECK(num_groups == 1, "wmma_v3 requires num_groups=1 (got ", num_groups, ")");

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

    constexpr int BS_C = 16;
    constexpr int M_TILE = 16, N_TILE = 16;
    constexpr int N_TILES_TOTAL = D_FIXED / N_TILE;
    constexpr int STAGES = CP_STAGES_V3;
    // v_pack + v_scl + w_bf + max(v_bf 4096 B, c_scratch 8192 B).  The
    // last region is reused: v_bf during the WMMA loop (4 KB BF16) and
    // c_scratch after (8 KB FP32).  Reserve the larger size.
    const size_t reuse_region =
        max((size_t)BS_C * D_FIXED * sizeof(__nv_bfloat16),
            (size_t)N_TILES_TOTAL * M_TILE * N_TILE * sizeof(float));
    const size_t smem_b =
          (size_t)STAGES * BS_C * D_HALF                      // v_pack
        + (size_t)STAGES * BS_C * sizeof(__nv_bfloat16)       // v_scl
        + (size_t)M_TILE * BS_C * sizeof(__nv_bfloat16)       // w_bf
        + reuse_region                                         // v_bf | c_scratch
        + 64;                                                  // align slack

    cudaFuncSetAttribute(weighted_v_split_kernel_wmma_v3<BS_C>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_b);

    dim3 grid_b((unsigned)B, (unsigned)H_kv, N_SPLITS_V3);
    dim3 block_b(THREADS_B);
    weighted_v_split_kernel_wmma_v3<BS_C><<<grid_b, block_b, smem_b, stream.stream()>>>(
        weights.data_ptr<float>(),
        static_cast<const float*>(nullptr),
        static_cast<const float*>(nullptr),
        static_cast<const float*>(nullptr),
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}


// Host launcher mirroring int4_softmax_v_with_pre_corr_wvs2 but dispatching
// to the WMMA v2 kernel.
std::vector<torch::Tensor> int4_softmax_v_with_pre_corr_wmma_v2(
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
    TORCH_CHECK(H_q % H_kv == 0);
    TORCH_CHECK(block_size == 16, "wmma_v2 requires block_size=16");
    const int n_groups_per_head_check = (int)(H_q / H_kv);
    TORCH_CHECK(n_groups_per_head_check <= 16,
                "wmma_v2 requires H_q/H_kv <= 16 (got ", n_groups_per_head_check, ")");

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
    TORCH_CHECK(num_groups == 1, "wmma_v2 requires num_groups=1 (got ", num_groups, ")");

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

    constexpr int BS_C = 16;
    constexpr int M_TILE = 16, N_TILE = 16, V_PAD = 8;
    constexpr int N_TILES_TOTAL = D_FIXED / N_TILE;     // 8
    constexpr int V_STRIDE = D_FIXED + V_PAD;
    const size_t smem_b =
          (size_t)2 * BS_C * D_HALF                     // v_pack
        + (size_t)2 * BS_C * sizeof(__nv_bfloat16)      // v_scl
        + (size_t)M_TILE * BS_C * sizeof(__nv_bfloat16) // w_bf
        + (size_t)BS_C * V_STRIDE * sizeof(__nv_bfloat16) // v_bf
        + (size_t)N_TILES_TOTAL * M_TILE * N_TILE * sizeof(float)  // c_scratch
        + 64;                                                       // align slack

    cudaFuncSetAttribute(weighted_v_split_kernel_wmma_v2<BS_C>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_b);

    dim3 grid_b((unsigned)B, (unsigned)H_kv, N_SPLITS);
    dim3 block_b(THREADS_B);
    weighted_v_split_kernel_wmma_v2<BS_C><<<grid_b, block_b, smem_b, stream.stream()>>>(
        weights.data_ptr<float>(),
        static_cast<const float*>(nullptr),
        static_cast<const float*>(nullptr),
        static_cast<const float*>(nullptr),
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}


std::vector<torch::Tensor> int4_softmax_stats_only(
    torch::Tensor scores,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q)
{
    CHECK_CUDA(scores); CHECK_CUDA(seq_lens);
    TORCH_CHECK(scores.scalar_type()   == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(scores.dim() == 3);

    auto sc = scores.contiguous();
    auto sl = seq_lens.contiguous();
    const int B = sc.size(0);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());
    auto softmax_m     = torch::empty({B, (int64_t)H_q}, opts_f32);
    auto softmax_inv_z = torch::empty({B, (int64_t)H_q}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)B, (unsigned)H_q);
    dim3 block(THREADS_B);
    const size_t smem = 16 * sizeof(float);
    softmax_stats_kernel<<<grid, block, smem, stream.stream()>>>(
        sc.data_ptr<float>(), sl.data_ptr<int32_t>(),
        softmax_m.data_ptr<float>(), softmax_inv_z.data_ptr<float>(),
        (int)max_S, (int)H_q
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {softmax_m, softmax_inv_z};
}


template <typename KernelFn>
static std::vector<torch::Tensor> _with_scores_dispatch(
    KernelFn kernel,
    torch::Tensor scores,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor topk_idx,
    int64_t max_S, int64_t H_q, int64_t H_kv,
    int64_t block_size, int64_t top_k,
    dim3 grid_b, size_t smem_b)
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
    TORCH_CHECK(H_q % H_kv == 0);
    TORCH_CHECK(block_size == 16, "with_scores requires block_size=16");
    const int n_groups_per_head_check = (int)(H_q / H_kv);
    TORCH_CHECK(n_groups_per_head_check <= 16,
                "with_scores requires H_q/H_kv <= 16 (got ",
                n_groups_per_head_check, ")");

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
    TORCH_CHECK(num_groups == 1, "with_scores requires num_groups=1 (got ",
                num_groups, ")");

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());
    auto softmax_m     = torch::empty({B, (int64_t)H_q}, opts_f32);
    auto softmax_inv_z = torch::empty({B, (int64_t)H_q}, opts_f32);
    auto o_hat         = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto o_old_unnorm  = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out         = torch::zeros({B, (int64_t)H_q}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    {
        dim3 grid_a((unsigned)B, (unsigned)H_q);
        dim3 block_a(THREADS_B);
        const size_t smem_a = 16 * sizeof(float);
        softmax_stats_kernel<<<grid_a, block_a, smem_a, stream.stream()>>>(
            sc.data_ptr<float>(), sl.data_ptr<int32_t>(),
            softmax_m.data_ptr<float>(), softmax_inv_z.data_ptr<float>(),
            (int)max_S, (int)H_q
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    cudaFuncSetAttribute(kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_b);
    dim3 block_b(THREADS_B);
    kernel<<<grid_b, block_b, smem_b, stream.stream()>>>(
        static_cast<const float*>(nullptr),
        sc.data_ptr<float>(),
        softmax_m.data_ptr<float>(),
        softmax_inv_z.data_ptr<float>(),
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}

template <typename KernelFn>
static std::vector<torch::Tensor> _with_scores_stats_dispatch(
    KernelFn kernel,
    torch::Tensor scores,
    torch::Tensor softmax_m,
    torch::Tensor softmax_inv_z,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor topk_idx,
    int64_t max_S, int64_t H_q, int64_t H_kv,
    int64_t block_size, int64_t top_k,
    dim3 grid_b, size_t smem_b)
{
    CHECK_CUDA(scores); CHECK_CUDA(softmax_m); CHECK_CUDA(softmax_inv_z);
    CHECK_CUDA(v_data); CHECK_CUDA(v_scales);
    CHECK_CUDA(block_table); CHECK_CUDA(seq_lens); CHECK_CUDA(topk_idx);
    TORCH_CHECK(scores.scalar_type()        == torch::kFloat32);
    TORCH_CHECK(softmax_m.scalar_type()     == torch::kFloat32);
    TORCH_CHECK(softmax_inv_z.scalar_type() == torch::kFloat32);
    TORCH_CHECK(v_data.scalar_type()        == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()      == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type()   == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type()      == torch::kInt32);
    TORCH_CHECK(topk_idx.scalar_type()      == torch::kInt64);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(softmax_m.dim() == 2 && softmax_inv_z.dim() == 2);
    TORCH_CHECK(topk_idx.dim() == 3);
    TORCH_CHECK(topk_idx.size(2) == top_k);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_HALF);
    TORCH_CHECK(H_q % H_kv == 0);
    TORCH_CHECK(block_size == 16, "with_scores_stats requires block_size=16");
    const int n_groups_per_head_check = (int)(H_q / H_kv);
    TORCH_CHECK(n_groups_per_head_check <= 16,
                "with_scores_stats requires H_q/H_kv <= 16 (got ",
                n_groups_per_head_check, ")");

    auto sc = scores.contiguous();
    auto sm = softmax_m.contiguous();
    auto sz = softmax_inv_z.contiguous();
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
    TORCH_CHECK(num_groups == 1, "with_scores_stats requires num_groups=1 (got ",
                num_groups, ")");
    TORCH_CHECK(sm.size(0) == B && sm.size(1) == H_q);
    TORCH_CHECK(sz.size(0) == B && sz.size(1) == H_q);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());
    auto o_hat         = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto o_old_unnorm  = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out         = torch::zeros({B, (int64_t)H_q}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_b);
    dim3 block_b(THREADS_B);
    kernel<<<grid_b, block_b, smem_b, stream.stream()>>>(
        static_cast<const float*>(nullptr),
        sc.data_ptr<float>(),
        sm.data_ptr<float>(),
        sz.data_ptr<float>(),
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}


std::vector<torch::Tensor> int4_softmax_v_with_pre_corr_wvs2_scores(
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
    constexpr int BS_C = 16;
    const size_t smem_b =
          (size_t)2 * BS_C * D_HALF
        + (size_t)2 * BS_C * 4
        + WVS2_TOPK_SMEM
        + 16;
    const int B = (int)scores.size(0);
    if (B == 1) {
        dim3 grid_b((unsigned)B, (unsigned)H_q, (unsigned)N_SPLITS_WVS2_B1);
        return _with_scores_dispatch(
            weighted_v_split_kernel_wvs2<BS_C, N_SPLITS_WVS2_B1>,
            scores, v_data, v_scales, block_table, seq_lens, topk_idx,
            max_S, H_q, H_kv, block_size, top_k, grid_b, smem_b);
    }
    dim3 grid_b((unsigned)B, (unsigned)H_q, (unsigned)N_SPLITS);
    return _with_scores_dispatch(
        weighted_v_split_kernel_wvs2<BS_C, N_SPLITS>,
        scores, v_data, v_scales, block_table, seq_lens, topk_idx,
        max_S, H_q, H_kv, block_size, top_k, grid_b, smem_b);
}


std::vector<torch::Tensor> int4_softmax_v_with_pre_corr_wmma_v2_scores(
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
    constexpr int BS_C = 16;
    constexpr int M_TILE = 16, N_TILE = 16, V_PAD = 8;
    constexpr int N_TILES_TOTAL = D_FIXED / N_TILE;
    constexpr int V_STRIDE = D_FIXED + V_PAD;
    const size_t smem_b =
          (size_t)2 * BS_C * D_HALF
        + (size_t)2 * BS_C * sizeof(__nv_bfloat16)
        + (size_t)M_TILE * BS_C * sizeof(__nv_bfloat16)
        + (size_t)BS_C * V_STRIDE * sizeof(__nv_bfloat16)
        + (size_t)N_TILES_TOTAL * M_TILE * N_TILE * sizeof(float)
        + 64;
    dim3 grid_b((unsigned)scores.size(0), (unsigned)H_kv, (unsigned)N_SPLITS);
    return _with_scores_dispatch(
        weighted_v_split_kernel_wmma_v2<BS_C>,
        scores, v_data, v_scales, block_table, seq_lens, topk_idx,
        max_S, H_q, H_kv, block_size, top_k, grid_b, smem_b);
}


std::vector<torch::Tensor> int4_softmax_v_with_pre_corr_wmma_v3_scores(
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
    constexpr int BS_C = 16;
    constexpr int M_TILE = 16, N_TILE = 16;
    constexpr int N_TILES_TOTAL = D_FIXED / N_TILE;
    constexpr int STAGES = CP_STAGES_V3;
    const size_t reuse_region =
        max((size_t)BS_C * D_FIXED * sizeof(__nv_bfloat16),
            (size_t)N_TILES_TOTAL * M_TILE * N_TILE * sizeof(float));
    const size_t smem_b =
          (size_t)STAGES * BS_C * D_HALF
        + (size_t)STAGES * BS_C * sizeof(__nv_bfloat16)
        + (size_t)M_TILE * BS_C * sizeof(__nv_bfloat16)
        + reuse_region
        + 64;
    dim3 grid_b((unsigned)scores.size(0), (unsigned)H_kv, (unsigned)N_SPLITS_V3);
    return _with_scores_dispatch(
        weighted_v_split_kernel_wmma_v3<BS_C>,
        scores, v_data, v_scales, block_table, seq_lens, topk_idx,
        max_S, H_q, H_kv, block_size, top_k, grid_b, smem_b);
}


std::vector<torch::Tensor> int4_softmax_v_with_pre_corr_wmma_v3_scores_stats(
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
    constexpr int BS_C = 16;
    constexpr int M_TILE = 16, N_TILE = 16;
    constexpr int N_TILES_TOTAL = D_FIXED / N_TILE;
    constexpr int STAGES = CP_STAGES_V3;
    const size_t reuse_region =
        max((size_t)BS_C * D_FIXED * sizeof(__nv_bfloat16),
            (size_t)N_TILES_TOTAL * M_TILE * N_TILE * sizeof(float));
    const size_t smem_b =
          (size_t)STAGES * BS_C * D_HALF
        + (size_t)STAGES * BS_C * sizeof(__nv_bfloat16)
        + (size_t)M_TILE * BS_C * sizeof(__nv_bfloat16)
        + reuse_region
        + 64;
    dim3 grid_b((unsigned)scores.size(0), (unsigned)H_kv, (unsigned)N_SPLITS_V3);
    return _with_scores_stats_dispatch(
        weighted_v_split_kernel_wmma_v3<BS_C>,
        scores, softmax_m, softmax_inv_z,
        v_data, v_scales, block_table, seq_lens, topk_idx,
        max_S, H_q, H_kv, block_size, top_k, grid_b, smem_b);
}


// ────────────────────────────────────────────────────────────────────────
// Diagnostic-only stand-alone entries.  These split the bundled
// `softmax_prep + weighted_v_split` pipeline so each kernel can be timed
// in isolation.  Used by bench/microbench_int4_v_kernels.py to attribute
// V2_01b stage time between softmax_prep and weighted_v_split.
//
// Not used in the production decode path (which still calls the bundled
// _wvs2 / _wmma_v2 entries above).
// ────────────────────────────────────────────────────────────────────────

torch::Tensor int4_softmax_prep_only(
    torch::Tensor scores,
    torch::Tensor seq_lens,
    int64_t max_S,
    int64_t H_q)
{
    CHECK_CUDA(scores); CHECK_CUDA(seq_lens);
    TORCH_CHECK(scores.scalar_type()   == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(scores.dim() == 3);

    auto sc = scores.contiguous();
    auto sl = seq_lens.contiguous();
    const int B = sc.size(0);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(sc.device());
    auto weights = torch::empty({B, (int64_t)H_q, (int64_t)max_S}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
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
    return weights;
}


// Common machinery for the stand-alone weighted_v_split entries: takes
// pre-computed `weights`, runs the V-multiply kernel only.
template <typename KernelFn>
static std::vector<torch::Tensor> _wvs_only_dispatch(
    KernelFn kernel,
    torch::Tensor weights,
    torch::Tensor v_data,
    torch::Tensor v_scales,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    torch::Tensor topk_idx,
    int64_t max_S, int64_t H_q, int64_t H_kv,
    int64_t block_size, int64_t top_k,
    dim3 grid_b, size_t smem_b)
{
    auto wt = weights.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 2)
                  ? v_scales.unsqueeze(-1).contiguous()
                  : v_scales.contiguous();
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto ti = topk_idx.contiguous();

    const int B          = wt.size(0);
    const int max_blocks = (int)bt.size(1);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(wt.device());
    auto o_hat        = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto o_old_unnorm = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out        = torch::zeros({B, (int64_t)H_q}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_b);
    dim3 block_b(THREADS_B);
    kernel<<<grid_b, block_b, smem_b, stream.stream()>>>(
        wt.data_ptr<float>(),
        static_cast<const float*>(nullptr),
        static_cast<const float*>(nullptr),
        static_cast<const float*>(nullptr),
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_hat, o_old_unnorm, W_out};
}


std::vector<torch::Tensor> int4_weighted_v_split_wvs2_only(
    torch::Tensor weights, torch::Tensor v_data, torch::Tensor v_scales,
    torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor topk_idx,
    int64_t max_S, int64_t H_q, int64_t H_kv, int64_t block_size, int64_t top_k)
{
    TORCH_CHECK(block_size == 16, "wvs2_only requires block_size=16");
    constexpr int BS_C = 16;
    const size_t smem_b =
          (size_t)2 * BS_C * D_HALF
        + (size_t)2 * BS_C * 4
        + WVS2_TOPK_SMEM
        + 16;
    const int B = (int)weights.size(0);
    if (B == 1) {
        dim3 grid_b((unsigned)B, (unsigned)H_q, (unsigned)N_SPLITS_WVS2_B1);
        return _wvs_only_dispatch(
            weighted_v_split_kernel_wvs2<BS_C, N_SPLITS_WVS2_B1>,
            weights, v_data, v_scales, block_table, seq_lens, topk_idx,
            max_S, H_q, H_kv, block_size, top_k,
            grid_b, smem_b);
    }
    dim3 grid_b((unsigned)B, (unsigned)H_q, (unsigned)N_SPLITS);
    return _wvs_only_dispatch(
        weighted_v_split_kernel_wvs2<BS_C, N_SPLITS>,
        weights, v_data, v_scales, block_table, seq_lens, topk_idx,
        max_S, H_q, H_kv, block_size, top_k,
        grid_b, smem_b);
}


std::vector<torch::Tensor> int4_weighted_v_split_wmma_v2_only(
    torch::Tensor weights, torch::Tensor v_data, torch::Tensor v_scales,
    torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor topk_idx,
    int64_t max_S, int64_t H_q, int64_t H_kv, int64_t block_size, int64_t top_k)
{
    TORCH_CHECK(block_size == 16, "wmma_v2_only requires block_size=16");
    constexpr int BS_C = 16;
    constexpr int M_TILE = 16, N_TILE = 16, V_PAD = 8;
    constexpr int N_TILES_TOTAL = D_FIXED / N_TILE;
    constexpr int V_STRIDE = D_FIXED + V_PAD;
    const size_t smem_b =
          (size_t)2 * BS_C * D_HALF
        + (size_t)2 * BS_C * sizeof(__nv_bfloat16)
        + (size_t)M_TILE * BS_C * sizeof(__nv_bfloat16)
        + (size_t)BS_C * V_STRIDE * sizeof(__nv_bfloat16)
        + (size_t)N_TILES_TOTAL * M_TILE * N_TILE * sizeof(float)
        + 64;
    dim3 grid_b((unsigned)weights.size(0), (unsigned)H_kv, (unsigned)N_SPLITS);
    return _wvs_only_dispatch(
        weighted_v_split_kernel_wmma_v2<BS_C>,
        weights, v_data, v_scales, block_table, seq_lens, topk_idx,
        max_S, H_q, H_kv, block_size, top_k,
        grid_b, smem_b);
}


// ────────────────────────────────────────────────────────────────────────
// Diagnostic inplace entries — microbench-only.  Take pre-allocated output
// tensors as in/out so that allocation cost is not in the timed window.
// Caller must zero o_hat/o_old_unnorm/W_out before each call (atomicAdd
// semantics).  No new kernels — these reuse the existing v1/v2 launchers.
// ────────────────────────────────────────────────────────────────────────

template <typename KernelFn>
static void _wvs_inplace_dispatch(
    KernelFn kernel,
    torch::Tensor weights, torch::Tensor v_data, torch::Tensor v_scales,
    torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor topk_idx,
    torch::Tensor o_hat, torch::Tensor o_old_unnorm, torch::Tensor W_out,
    int64_t max_S, int64_t H_q, int64_t H_kv,
    int64_t block_size, int64_t top_k,
    dim3 grid_b, size_t smem_b)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_b);
    dim3 block_b(THREADS_B);
    const int max_blocks = (int)block_table.size(1);
    kernel<<<grid_b, block_b, smem_b, stream.stream()>>>(
        weights.data_ptr<float>(),
        static_cast<const float*>(nullptr),
        static_cast<const float*>(nullptr),
        static_cast<const float*>(nullptr),
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void int4_weighted_v_split_wvs2_inplace(
    torch::Tensor weights, torch::Tensor v_data, torch::Tensor v_scales,
    torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor topk_idx,
    torch::Tensor o_hat, torch::Tensor o_old_unnorm, torch::Tensor W_out,
    int64_t max_S, int64_t H_q, int64_t H_kv, int64_t block_size, int64_t top_k)
{
    TORCH_CHECK(block_size == 16, "wvs2_inplace requires block_size=16");
    constexpr int BS_C = 16;
    const size_t smem_b =
          (size_t)2 * BS_C * D_HALF
        + (size_t)2 * BS_C * 4
        + WVS2_TOPK_SMEM
        + 16;
    const int B = (int)weights.size(0);
    if (B == 1) {
        dim3 grid_b((unsigned)B, (unsigned)H_q, (unsigned)N_SPLITS_WVS2_B1);
        _wvs_inplace_dispatch(
            weighted_v_split_kernel_wvs2<BS_C, N_SPLITS_WVS2_B1>,
            weights, v_data, v_scales, block_table, seq_lens, topk_idx,
            o_hat, o_old_unnorm, W_out,
            max_S, H_q, H_kv, block_size, top_k, grid_b, smem_b);
        return;
    }
    dim3 grid_b((unsigned)B, (unsigned)H_q, (unsigned)N_SPLITS);
    _wvs_inplace_dispatch(
        weighted_v_split_kernel_wvs2<BS_C, N_SPLITS>,
        weights, v_data, v_scales, block_table, seq_lens, topk_idx,
        o_hat, o_old_unnorm, W_out,
        max_S, H_q, H_kv, block_size, top_k, grid_b, smem_b);
}


void int4_weighted_v_split_wmma_v2_inplace(
    torch::Tensor weights, torch::Tensor v_data, torch::Tensor v_scales,
    torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor topk_idx,
    torch::Tensor o_hat, torch::Tensor o_old_unnorm, torch::Tensor W_out,
    int64_t max_S, int64_t H_q, int64_t H_kv, int64_t block_size, int64_t top_k)
{
    TORCH_CHECK(block_size == 16, "wmma_v2_inplace requires block_size=16");
    constexpr int BS_C = 16;
    constexpr int M_TILE = 16, N_TILE = 16, V_PAD = 8;
    constexpr int N_TILES_TOTAL = D_FIXED / N_TILE;
    constexpr int V_STRIDE = D_FIXED + V_PAD;
    const size_t smem_b =
          (size_t)2 * BS_C * D_HALF
        + (size_t)2 * BS_C * sizeof(__nv_bfloat16)
        + (size_t)M_TILE * BS_C * sizeof(__nv_bfloat16)
        + (size_t)BS_C * V_STRIDE * sizeof(__nv_bfloat16)
        + (size_t)N_TILES_TOTAL * M_TILE * N_TILE * sizeof(float)
        + 64;
    dim3 grid_b((unsigned)weights.size(0), (unsigned)H_kv, (unsigned)N_SPLITS);
    _wvs_inplace_dispatch(
        weighted_v_split_kernel_wmma_v2<BS_C>,
        weights, v_data, v_scales, block_table, seq_lens, topk_idx,
        o_hat, o_old_unnorm, W_out,
        max_S, H_q, H_kv, block_size, top_k, grid_b, smem_b);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_softmax_v", &int4_softmax_v,
          "Stage B INT4: 2-kernel split-S + SMEM staging "
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

    m.def("int4_softmax_v_with_pre_corr", &int4_softmax_v_with_pre_corr,
          "Stage B INT4 with pre_corr: returns [o_hat, o_old_unnorm, W].",
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

    m.def("int4_softmax_v_with_pre_corr_wvs2", &int4_softmax_v_with_pre_corr_wvs2,
          "Stage B INT4 + wvs2 weighted_v_split with pre-multiplied w·vs at "
          "chunk-load time. NG=1 only. Same outputs as int4_softmax_v_with_pre_corr.",
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
    m.def("int4_softmax_v_with_pre_corr_wvs2_inplace",
          &int4_softmax_v_with_pre_corr_wvs2_inplace,
          "Graph-safe INT4 wvs2 Stage B using caller-owned workspace.",
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

    m.def("int4_softmax_v_with_pre_corr_wmma_v2",
          &int4_softmax_v_with_pre_corr_wmma_v2,
          "Stage B INT4 + WMMA-promoted weighted_v_split (Phase 1 of INT4 "
          "decode optimisation).  BS=16, NG=1 only.  Same outputs as "
          "int4_softmax_v_with_pre_corr.",
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

    m.def("int4_softmax_v_with_pre_corr_wmma_v3",
          &int4_softmax_v_with_pre_corr_wmma_v3,
          "Stage B INT4 + occupancy-tuned WMMA weighted_v_split (Phase 2). "
          "Drops c_scratch + V_PAD, deepens cp.async pipeline to 3 stages, "
          "halves N_SPLITS, applies __launch_bounds__.  BS=16, NG=1 only.  "
          "Same outputs as int4_softmax_v_with_pre_corr.",
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

    m.def("int4_softmax_v_with_pre_corr_wvs2_scores",
          &int4_softmax_v_with_pre_corr_wvs2_scores,
          "Stage B INT4 wvs2 path without materializing weights: "
          "softmax_stats + weighted_v reads scores directly.",
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

    m.def("int4_softmax_v_with_pre_corr_wmma_v2_scores",
          &int4_softmax_v_with_pre_corr_wmma_v2_scores,
          "Stage B INT4 WMMA v2 without materializing weights: "
          "softmax_stats + weighted_v reads scores directly.",
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

    m.def("int4_softmax_v_with_pre_corr_wmma_v3_scores",
          &int4_softmax_v_with_pre_corr_wmma_v3_scores,
          "Stage B INT4 WMMA v3 without materializing weights: "
          "softmax_stats + weighted_v reads scores directly.",
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

    m.def("int4_softmax_v_with_pre_corr_wmma_v3_scores_stats",
          &int4_softmax_v_with_pre_corr_wmma_v3_scores_stats,
          "Stage B INT4 WMMA v3 using caller-provided softmax stats: "
          "weighted_v reads scores directly and skips softmax_stats launch.",
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

    // ── Diagnostic stand-alone entries (kernel isolation timing) ─────────
    m.def("int4_softmax_prep_only", &int4_softmax_prep_only,
          "Diagnostic: run softmax_prep_kernel alone, returns weights.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"));

    m.def("int4_softmax_stats_only", &int4_softmax_stats_only,
          "Diagnostic: run softmax_stats_kernel alone, returns [m, invZ].",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("scores"), py::arg("seq_lens"),
          py::arg("max_S"), py::arg("H_q"));

    m.def("int4_weighted_v_split_wvs2_only",
          &int4_weighted_v_split_wvs2_only,
          "Diagnostic: run weighted_v_split_wvs2 alone given pre-computed "
          "weights.  Returns (o_hat, o_old_unnorm, W_out).",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("weights"), py::arg("v_data"), py::arg("v_scales"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("topk_idx"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("top_k"));

    m.def("int4_weighted_v_split_wmma_v2_only",
          &int4_weighted_v_split_wmma_v2_only,
          "Diagnostic: run weighted_v_split_wmma_v2 alone given pre-computed "
          "weights.  Returns (o_hat, o_old_unnorm, W_out).",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("weights"), py::arg("v_data"), py::arg("v_scales"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("topk_idx"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("top_k"));

    // Diagnostic: in-place variants (caller-provided buffers, no allocation
    // in the timed window).  Used by microbench for accurate kernel-only
    // per-call timing.  No new kernels.
    m.def("int4_weighted_v_split_wvs2_inplace",
          &int4_weighted_v_split_wvs2_inplace,
          "Diagnostic in-place: weighted_v_split_wvs2 into pre-allocated "
          "(o_hat, o_old_unnorm, W_out).  Caller must zero before call.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("weights"), py::arg("v_data"), py::arg("v_scales"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("topk_idx"),
          py::arg("o_hat"), py::arg("o_old_unnorm"), py::arg("W_out"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("top_k"));

    m.def("int4_weighted_v_split_wmma_v2_inplace",
          &int4_weighted_v_split_wmma_v2_inplace,
          "Diagnostic in-place: weighted_v_split_wmma_v2 into pre-allocated "
          "(o_hat, o_old_unnorm, W_out).  Caller must zero before call.",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("weights"), py::arg("v_data"), py::arg("v_scales"),
          py::arg("block_table"), py::arg("seq_lens"), py::arg("topk_idx"),
          py::arg("o_hat"), py::arg("o_old_unnorm"), py::arg("W_out"),
          py::arg("max_S"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("block_size"), py::arg("top_k"));
}
