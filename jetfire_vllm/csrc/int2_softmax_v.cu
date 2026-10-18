// Stage B kernel for the JetFire fused-attention split, v3 (split-S + SMEM staging).
//   o_hat = softmax(scores) · dequant(V_int)
//
// Design (vs v1/v2 single-CTA-per-(b,h_q)):
//   * 2 kernels.
//   * Kernel A (softmax_prep_kernel): computes softmax weights  → HBM tensor
//     `weights = exp(score - max) / Z` of shape (B, H_q, max_S) fp32.
//     1 CTA per (b, h_q), 128 threads.  Same Phase 1+2 as v2 but writes
//     normalized weights instead of keeping them in SMEM.  Cheap (~10 µs total
//     for B=4 H_q=32, dominated by 2 MB write to HBM).
//   * Kernel B (weighted_v_split_kernel): split-S + SMEM staging.
//     gridDim = (B, H_q, N_SPLITS).  Each CTA processes max_S/N_SPLITS
//     positions.  V chunks are loaded BS positions at a time into SMEM
//     (cooperative LDG.U32, 4 bytes per thread) and re-used across the
//     128-thread inner loop instead of each thread doing its own LDG.U8
//     per position.  Final per-CTA partial accumulator is atomicAdd'd to
//     o_hat which must be zeroed before launch.
//
// Why ncu said this:
//   v2 had 4.45 active warps / SM (theoretical 16) → only 0.08 eligible
//   warps/cycle, every 13 cycles a single instruction issued.  L1TEX
//   scoreboard stalls accounted for 80% of the wall.  Splitting S 8× and
//   adding SMEM staging brings warp count up and replaces gmem LDG with
//   SMEM LDS in the inner loop.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <type_traits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(),  #x " must be CUDA")

constexpr int D_FIXED   = 128;
constexpr int D_QUARTER = D_FIXED / 4;
constexpr int THREADS_B = 128;
constexpr int N_SPLITS  = 8;             // split-S factor for kernel B
constexpr int N_SPLITS_WVS2_SMALL_B = 16;
constexpr int N_SPLITS_WVS2_B1 = 128;    // B=1 decode is CTA-starved at long ctx.
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
//
// SMEM-free design: only uses a small (~16 floats) reduction buffer, NOT a
// max_S-sized cache.  This makes the kernel work at any max_S (originally
// limited to ~57k by H100's 228 KB dynamic SMEM cap).  Trade-off: an extra
// HBM round-trip on the weights buffer (read+write twice instead of once).
//
//   Phase 1: read scores → block_max.
//   Phase 2: read scores → write exp(score - max) to weights, accumulate sum.
//   Phase 3: read weights, divide by sum, write back normalized weights.
//
// HBM traffic vs old design: old = 2 read(scores) + 1 write(weights);
// new = 2 read(scores) + 1 write(weights) + 1 read(weights) + 1 write(weights).
// ~1.5× more HBM access on the weights buffer.  Acceptable for unblocking
// long-ctx; can be optimized later by fusing with weighted_v_split.
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
    float* red_buf = reinterpret_cast<float*>(smem_raw);  // ~16 floats only

    const float* score_row =
        scores  + ((size_t)b * H_q + (size_t)h_q) * (size_t)max_S;
    float* weight_row =
        weights + ((size_t)b * H_q + (size_t)h_q) * (size_t)max_S;

    // Phase 1: row max.
    float local_max = -INFINITY;
    for (int i = tid; i < S; i += THREADS_B) {
        float s = score_row[i];
        local_max = fmaxf(local_max, s);
    }
    float m = block_max(local_max, red_buf);
    if (!isfinite(m)) m = 0.0f;

    // Phase 2: exp(score - max) → weights HBM, accumulate sum.
    float local_sum = 0.0f;
    for (int i = tid; i < max_S; i += THREADS_B) {
        float v = (i < S) ? __expf(score_row[i] - m) : 0.0f;
        weight_row[i] = v;
        local_sum += v;
    }
    float Z = block_sum(local_sum, red_buf);
    float invZ = (Z > 0.0f) ? (1.0f / Z) : 0.0f;
    __syncthreads();

    // Phase 3: read weights, normalize, write back.
    for (int i = tid; i < max_S; i += THREADS_B) {
        weight_row[i] = weight_row[i] * invZ;
    }
}


// Kernel B v5: split-S SMEM-staged V multiply (Phase 1) plus a separate
// top-K-only Phase 2 that runs in s_split=0 only.  Replaces v4's branch-
// free conditional inside the unrolled inner loop, which wasted ~98% of
// the o_old FMAs on positions that aren't in top-K.
//
// Phase 1 (all splits):
//   acc_hat = Σ_{p in [s_start, s_end)} w[p] · dequant(V[p])
//
// Phase 2 (s_split == 0 only):
//   acc_o_old = Σ_{k in topK} w[topk_idx[k]] · dequant(V[topk_idx[k]])
//   acc_W     = Σ_{k in topK} w[topk_idx[k]]
//
// Phase 2 reads V by per-position __ldg (only K=64 iters total per (b,h_q),
// done once across all splits), so we can't reuse the cooperative SMEM
// staging — but the work is so small (~64 byte-loads × 128 threads vs
// 4096 in Phase 1) that this barely matters.
//
// Templated on BS (block_size, typically 16) and NG (num_groups, typically 1).
template <int BS, int NG>
__global__ void weighted_v_split_kernel(
    const float*         __restrict__ weights,      // (B, H_q, max_S)
    const uint8_t*       __restrict__ v_data,       // (H_kv, num_slots, D/4)
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

    // S range for this CTA.
    const int chunk_per_split = (max_S + N_SPLITS - 1) / N_SPLITS;
    const int s_start = s_split * chunk_per_split;
    const int s_end   = min(s_start + chunk_per_split, S);
    if (s_start >= S) return;

    // SMEM layout (DOUBLE-BUFFERED — ping-pong for load/compute overlap):
    //   v_buf  : 2 × BS * 32 bytes  (1024 B for BS=16)
    //   vs_buf : 2 × BS × NG * 2 bytes
    //   w_buf  : 2 × BS * 4 bytes   (128 B)
    // While threads compute on buf[cur], LDG.E.U32 for buf[1-cur] is in
    // flight; one __syncthreads per chunk (vs two in single-buffer v5)
    // and the gmem latency overlaps with the 16-FMA inner loop.
    const bool do_pre_corr = (topk_idx != nullptr);

    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* v_buf = smem_raw;                                       // [2][BS*32]
    __nv_bfloat16* vs_buf =
        reinterpret_cast<__nv_bfloat16*>(v_buf + 2 * BS * 32);       // [2][BS][NG]
    float* w_buf = reinterpret_cast<float*>(vs_buf + 2 * BS * NG);   // [2][BS]

    const int d = tid;
    if (d >= D_FIXED) return;
    const int byte_x = d >> 2;
    const int sub    = d &  3;
    const int shift  = (3 - sub) * 2;
    const int group_id = (NG == 1) ? 0 : (d / GROUP_SIZE);

    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const float*         w_row        = weights  + ((size_t)b * H_q + (size_t)h_q) * max_S;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    // ── Helper lambdas: issue ASYNC load for chunk `blk` into `buf` ─────
    // V data goes through cp.async.ca.shared.global (4-byte per thread)
    // — non-blocking, gmem → SMEM directly without register stop-over,
    // so the inner FMA loop on a different buffer can run concurrently.
    // V scales and weights are small (16 elements each) and stay on
    // regular __ldg.
    auto load_chunk_async = [&](int blk, int buf) {
        const int64_t base_slot = bt_row[blk] * (int64_t)BS;

        // V data: 128 threads × 4-byte cp.async = 512B exactly.
        const int pos_in_blk = tid >> 3;
        const int byte_off   = (tid & 7) << 2;
        const void* gmem_ptr =
            v_pack_base + (size_t)(base_slot + pos_in_blk) * 32 + byte_off;
        void* smem_ptr =
            v_buf + (size_t)buf * BS * 32
                  + (size_t)pos_in_blk * 32 + byte_off;
        unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(smem_int), "l"(gmem_ptr));

        // V scales — BS * NG bf16 values.  For NG=2, stage both groups so
        // dimensions 0..63 and 64..127 use their matching per-group scale.
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
        const uint8_t* v_cur = v_buf + (size_t)buf * BS * 32;
        const __nv_bfloat16* vs_cur = vs_buf + buf * BS * NG;
        const float* w_cur = w_buf + buf * BS;
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            uint8_t pk = v_cur[j * 32 + byte_x];
            int q = int((pk >> shift) & 0x3) - 2;
            float vs = __bfloat162float(vs_cur[j * NG + group_id]);
            float v_f = (float)q * vs;
            float w   = w_cur[j];
            acc_io = __fmaf_rn(w, v_f, acc_io);
        }
    };

    // ────────────────────────── Phase 1: V mul → o_hat ──────────────────────
    float acc = 0.0f;

    if (blk_start < blk_end) {
        // Prologue: issue cp.async for chunk[blk_start] into buf 0,
        // commit, then wait so the data is available before first compute.
        load_chunk_async(blk_start, 0);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        // Main loop: issue async load for chunk[blk+1] into buf[1-cur],
        // commit, then compute on buf[cur].  The cp.async transfer runs
        // concurrently with the 16-FMA inner loop because cp.async does
        // NOT block the issuing thread (vs regular __ldg which stalls
        // until the data lands in a register).  After compute, wait for
        // the most recent cp.async group to complete and __syncthreads
        // so all warps see the new buf[nxt] before next iter.
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
        // Tail compute on the last loaded buffer.
        compute_chunk(cur_buf, acc);
        __syncthreads();
    }

    const size_t bh = (size_t)b * H_q + (size_t)h_q;
    atomicAdd(o_hat + bh * D_FIXED + d, acc);

    // ──────── Phase 2: top-K only loop, exclusive to s_split == 0 ───────
    // Iterates K positions (typically 64), each thread `d` does one FMA per
    // top-K entry into its own output element.  Only s_split=0 writes here,
    // so direct stores (output was zero-initialized by torch::zeros).
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

            uint8_t pk = __ldg(v_pack_base + slot * 32 + byte_x);
            int q = int((pk >> shift) & 0x3) - 2;
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


torch::Tensor int2_softmax_v(
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_QUARTER);
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

    // Kernel A: prep softmax weights.
    auto weights = torch::empty({B, (int64_t)H_q, (int64_t)max_S}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();

    {
        dim3 grid_a((unsigned)B, (unsigned)H_q);
        dim3 block_a(THREADS_B);
        // SMEM-free softmax_prep: only ~16 floats reduction buffer.
        const size_t smem_a = 16 * sizeof(float);
        softmax_prep_kernel<<<grid_a, block_a, smem_a, stream.stream()>>>(
            sc.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            weights.data_ptr<float>(),
            (int)max_S, (int)H_q
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Kernel B: split-S V mul → o_hat (must be zeroed; atomicAdd target).
    auto o_hat = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);

    auto launch_b = [&](auto BS_const, auto NG_const) {
        constexpr int BS = BS_const.value;
        constexpr int NG = NG_const.value;
        dim3 grid_b((unsigned)B, (unsigned)H_q, N_SPLITS);
        dim3 block_b(THREADS_B);
        // Double-buffered SMEM (ping-pong for load/compute overlap).
        const size_t smem_b = (size_t)2 * BS * 32  // v_buf  [2][BS*32]
                            + (size_t)2 * BS * NG * 2 // vs_buf [2][BS][NG]
                            + (size_t)2 * BS * 4   // w_buf  [2][BS]
                            + 16;                  // padding
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
    } else if (block_size == 32 && num_groups == 1) {
        launch_b(std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{});
    } else {
        TORCH_CHECK(false,
            "int2_softmax_v: unsupported (block_size=", block_size,
            ", num_groups=", num_groups, ").  Supported: (16,1), (16,2), (32,1).");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return o_hat;
}


// Stage B v4 with pre_corr fused.  Returns (o_hat, o_old_unnorm, W).
//
// o_old_unnorm = Σ_topK softmax_full(s_i) · dequant(V_int[i])  (unnormalized)
// W            = Σ_topK softmax_full(s_i)                     (top-K mass)
//
// The blend formula consuming these is:
//     o_corr = o_hat + W·o_new − o_old_unnorm
// (equivalent to the legacy `o_hat + W·(o_new − o_old)` where
//  o_old = o_old_unnorm / W).
std::vector<torch::Tensor> int2_softmax_v_with_pre_corr(
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_QUARTER);
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

    // Kernel A: prep softmax weights.
    auto weights = torch::empty({B, (int64_t)H_q, (int64_t)max_S}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    {
        dim3 grid_a((unsigned)B, (unsigned)H_q);
        dim3 block_a(THREADS_B);
        // SMEM-free softmax_prep: only ~16 floats reduction buffer.
        const size_t smem_a = 16 * sizeof(float);
        softmax_prep_kernel<<<grid_a, block_a, smem_a, stream.stream()>>>(
            sc.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            weights.data_ptr<float>(),
            (int)max_S, (int)H_q
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Kernel B: split-S, V mul + pre_corr fused → o_hat, o_old_unnorm, W.
    auto o_hat        = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto o_old_unnorm = torch::zeros({B, (int64_t)H_q, (int64_t)D_FIXED}, opts_f32);
    auto W_out        = torch::zeros({B, (int64_t)H_q}, opts_f32);

    auto launch_b = [&](auto BS_const, auto NG_const) {
        constexpr int BS = BS_const.value;
        constexpr int NG = NG_const.value;
        dim3 grid_b((unsigned)B, (unsigned)H_q, N_SPLITS);
        dim3 block_b(THREADS_B);
        // Double-buffered SMEM.
        const size_t smem_b = (size_t)2 * BS * 32
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
    } else if (block_size == 32 && num_groups == 1) {
        launch_b(std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{});
    } else {
        TORCH_CHECK(false,
            "int2_softmax_v_with_pre_corr: unsupported (block_size=", block_size,
            ", num_groups=", num_groups, ").");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {o_hat, o_old_unnorm, W_out};
}


// ────────────────────────────────────────────────────────────────────────
// wvs2 path: legacy `weighted_v_split_kernel` with the inner-loop
// hot path simplified. Each chunk-position scale `vs[j]` (bf16) and
// weight `w[j]` (fp32) is identical across all 128 threads in a CTA, so
// at chunk-load time the 16 BS threads compute `ws[j] = w[j] * vs[j]`
// once into SMEM. The inner FMA loop drops the bf16→fp32 conversion
// and the `q · vs` FMUL — what remains per FMA is:
//
//   uint8_t pk  = v_buf[j*32 + byte_x];           // 1 SMEM byte load
//   int     q   = (pk >> shift) & 0x3) - 2;       // 3 ALU
//   float   ws  = ws_buf[j];                      // 1 SMEM fp32 load
//   acc = __fmaf_rn(ws, (float)q, acc);           // 1 FMA (int→float fused)
//
// vs the legacy 6-op chain (byte load + 3 ALU + bf16 conv + FMUL + fp32
// load + FMA). Same numerics — `w * vs * q` is associative-equivalent
// to `(w*vs) * q` modulo fp32 fma rounding (~1 ULP). NG=1 only (the
// production case for LLaMA-3.1-8B GQA).
//
// Selected at the C++ wrapper level via int2_softmax_v_with_pre_corr_wvs2.
// The Python adapter keeps this off the release path unless
// JETFIRE_ALLOW_UNSTABLE_INT2_STAGEB=1.
// ────────────────────────────────────────────────────────────────────────

template <int BS, int SPLITS = N_SPLITS, int NG = 1>
__global__ void weighted_v_split_kernel_wvs2(
    const float*         __restrict__ weights,      // (B, H_q, max_S)
    const uint8_t*       __restrict__ v_data,       // (H_kv, num_slots, D/4)
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

    // SMEM layout: same byte budget as legacy (v_buf + ws_buf where ws
    // replaces the original vs_buf+w_buf pair). Net SMEM bytes:
    //   v_buf : 2 × BS × 32 bytes   (1024 B for BS=16)
    //   ws_buf: 2 × BS × NG × 4 bytes.
    //   NG=1 is the g128 path; NG=2 is the g64 multi-group path.
    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* v_buf = smem_raw;                                    // [2][BS*32]
    float*   ws_buf = reinterpret_cast<float*>(v_buf + 2 * BS * 32);  // [2][BS][NG]
    uint8_t* topk_raw = reinterpret_cast<uint8_t*>(ws_buf + 2 * BS * NG);
    int64_t* topk_slot_buf = reinterpret_cast<int64_t*>(topk_raw);
    float*   topk_ws_buf = reinterpret_cast<float*>(topk_slot_buf + WVS2_TOPK_CHUNK);
    float*   topk_w_buf = topk_ws_buf + WVS2_TOPK_CHUNK * NG;

    const int d = tid;
    if (d >= D_FIXED) return;
    const int byte_x = d >> 2;
    const int sub    = d &  3;
    const int shift  = (3 - sub) * 2;
    constexpr int GROUP_SIZE = D_FIXED / NG;
    const int group_id = (NG == 1) ? 0 : (d / GROUP_SIZE);

    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const float*         w_row        = weights  + ((size_t)b * H_q + (size_t)h_q) * max_S;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    auto load_chunk_async = [&](int blk, int buf) {
        const int64_t base_slot = bt_row[blk] * (int64_t)BS;

        // V data via cp.async (identical to legacy — 128 thr × 4 byte = 512 B).
        const int pos_in_blk = tid >> 3;
        const int byte_off   = (tid & 7) << 2;
        const void* gmem_ptr =
            v_pack_base + (size_t)(base_slot + pos_in_blk) * 32 + byte_off;
        void* smem_ptr =
            v_buf + (size_t)buf * BS * 32
                  + (size_t)pos_in_blk * 32 + byte_off;
        unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(smem_int), "l"(gmem_ptr));

        // Pre-multiplied weights × scales, one value per slot and scale group.
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
        const uint8_t* v_cur  = v_buf + (size_t)buf * BS * 32;
        const float*   ws_cur = ws_buf + buf * BS * NG;
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            uint8_t pk = v_cur[j * 32 + byte_x];
            int q = int((pk >> shift) & 0x3) - 2;
            acc_io = __fmaf_rn(ws_cur[j * NG + group_id], (float)q, acc_io);
        }
    };

    // ── Phase 1: V mul → o_hat (cp.async double-buffer, identical to v3).
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

    // ── Phase 2: top-K only loop, exclusive to s_split == 0.
    // Same numerics as legacy. K=64 iters (small) — pre-multiplication
    // happens inline (no SMEM staging, slots are scattered).
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
                uint8_t pk = __ldg(v_pack_base + slot * 32 + byte_x);
                int q = int((pk >> shift) & 0x3) - 2;
                acc_o_old = __fmaf_rn(topk_ws_buf[j * NG + group_id], (float)q, acc_o_old);
                if (d == 0) acc_W += topk_w_buf[j];
            }
            __syncthreads();
        }
        o_old_unnorm[bh * D_FIXED + d] = acc_o_old;
        if (d == 0) W_out[bh] = acc_W;
    }
}


std::vector<torch::Tensor> int2_softmax_v_with_pre_corr_wvs2(
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_QUARTER);
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

    // Phase 0: same softmax_prep as legacy (v_softmax_v_with_pre_corr).
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
    const size_t smem_b = (size_t)2 * BS_C * 32       // v_buf
                        + (size_t)2 * BS_C * num_groups * 4  // ws_buf (fp32)
                        + WVS2_TOPK_SMEM              // Phase-2 topK slot/ws/w
                        + 16;                         // alignment slack
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

void int2_softmax_v_with_pre_corr_wvs2_inplace(
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
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_QUARTER);
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
    const size_t smem_b = (size_t)2 * BS_C * 32
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


// ────────────────────────────────────────────────────────────────────────
// v2 path: eliminate the (B, H_q, max_S) FP32 ``weights`` tensor.
//
// Legacy:  softmax_prep_kernel writes weights (192 MB HBM traffic at B=16,
//          ctx=32k) → weighted_v_split_kernel reads weights.
//
// v2:      softmax_norm_kernel writes only (row_max, row_invsum) — 4 KB
//          total — and weighted_v_split_v2_kernel computes
//          ``w = expf(score - row_max) * row_invsum`` on the fly per chunk.
//          Net HBM saved: ~192 MB / layer-iter.
//
// Same blend math, same outputs. Selected at the Python wrapper level via
// JETFIRE_FUSED_SOFTMAX_V=v2 (see decode_path.py).
// ────────────────────────────────────────────────────────────────────────

__global__ void softmax_norm_kernel(
    const float*   __restrict__ scores,    // (B, H_q, max_S)
    const int32_t* __restrict__ seq_lens,  // (B,)
    float*         __restrict__ row_max,    // (B, H_q)     [out]
    float*         __restrict__ row_invsum, // (B, H_q)     [out]
    int max_S, int H_q)
{
    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;
    const int S   = seq_lens[b];

    extern __shared__ __align__(16) uint8_t smem_raw[];
    float* red_buf = reinterpret_cast<float*>(smem_raw);  // ~16 floats only

    const float* score_row =
        scores + ((size_t)b * H_q + (size_t)h_q) * (size_t)max_S;

    // Pass 1: row max.
    float local_max = -INFINITY;
    for (int i = tid; i < S; i += THREADS_B) {
        float s = score_row[i];
        local_max = fmaxf(local_max, s);
    }
    float m = block_max(local_max, red_buf);
    if (!isfinite(m)) m = 0.0f;

    // Pass 2: sum of exp(score - max).
    float local_sum = 0.0f;
    for (int i = tid; i < S; i += THREADS_B) {
        local_sum += __expf(score_row[i] - m);
    }
    float Z = block_sum(local_sum, red_buf);

    if (tid == 0) {
        const size_t bh = (size_t)b * H_q + (size_t)h_q;
        row_max[bh]    = m;
        row_invsum[bh] = (Z > 0.0f) ? (1.0f / Z) : 0.0f;
    }
}


// v2 weighted V mul: same structure as weighted_v_split_kernel but reads
// raw scores + (row_max, row_invsum) instead of pre-normalized weights.
// Computes weight = expf(score - row_max) * row_invsum inline per chunk.
template <int BS, int NG>
__global__ void weighted_v_split_v2_kernel(
    const float*         __restrict__ scores,       // (B, H_q, max_S)
    const float*         __restrict__ row_max,      // (B, H_q)
    const float*         __restrict__ row_invsum,   // (B, H_q)
    const uint8_t*       __restrict__ v_data,
    const __nv_bfloat16* __restrict__ v_scales,
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

    const bool do_pre_corr = (topk_idx != nullptr);

    // SMEM layout: same as legacy kernel. We re-use w_buf to stage scores
    // (raw fp32) instead of pre-normalized weights — same byte layout.
    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* v_buf = smem_raw;                                       // [2][BS*32]
    __nv_bfloat16* vs_buf =
        reinterpret_cast<__nv_bfloat16*>(v_buf + 2 * BS * 32);       // [2][BS][NG]
    float* s_buf = reinterpret_cast<float*>(vs_buf + 2 * BS * NG);   // [2][BS] — scores

    const int d = tid;
    if (d >= D_FIXED) return;
    const int byte_x = d >> 2;
    const int sub    = d &  3;
    const int shift  = (3 - sub) * 2;
    const int group_id = (NG == 1) ? 0 : (d / GROUP_SIZE);

    const uint8_t*       v_pack_base  = v_data   + (size_t)h_kv * v_data_stride_h;
    const __nv_bfloat16* v_scale_base = v_scales + (size_t)h_kv * v_scales_stride_h;
    const float*         s_row        = scores   + ((size_t)b * H_q + (size_t)h_q) * max_S;
    const int64_t*       bt_row       = block_table + (size_t)b * max_blocks;

    // Load (max, invsum) for this row — broadcast to all threads via L1.
    const size_t bh = (size_t)b * H_q + (size_t)h_q;
    const float row_m_val   = row_max[bh];
    const float row_inv_val = row_invsum[bh];

    const int blk_start = s_start / BS;
    const int blk_end   = (s_end + BS - 1) / BS;

    auto load_chunk_async = [&](int blk, int buf) {
        const int64_t base_slot = bt_row[blk] * (int64_t)BS;

        // V data: 128 threads × 4-byte cp.async = 512B exactly.
        const int pos_in_blk = tid >> 3;
        const int byte_off   = (tid & 7) << 2;
        const void* gmem_ptr =
            v_pack_base + (size_t)(base_slot + pos_in_blk) * 32 + byte_off;
        void* smem_ptr =
            v_buf + (size_t)buf * BS * 32
                  + (size_t)pos_in_blk * 32 + byte_off;
        unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(smem_int), "l"(gmem_ptr));

        // V scales — BS * NG bf16 values.  For NG=2, stage both groups so
        // dimensions 0..63 and 64..127 use their matching per-group scale.
        if (tid < BS * NG) {
            int scale_pos = tid / NG;
            int scale_gid = tid % NG;
            vs_buf[(buf * BS + scale_pos) * NG + scale_gid] = __ldg(
                v_scale_base + (size_t)(base_slot + scale_pos) * NG + scale_gid
            );
        }

        // Compute the 16 normalized weights ONCE per chunk and stage them
        // in SMEM (s_buf). The inner FMA loop then broadcasts these from
        // SMEM (1 expf per position, NOT per (position × output-dim)).
        // BS threads do the exp work; the remaining 112 threads of the
        // block don't participate in the load (handles the V-data load
        // via cp.async above).
        if (tid < BS) {
            int p = blk * BS + tid;
            float w = 0.0f;
            if (p >= s_start && p < s_end) {
                w = __expf(s_row[p] - row_m_val) * row_inv_val;
            }
            s_buf[buf * BS + tid] = w;
        }
    };

    auto compute_chunk = [&](int buf, float& acc_io) {
        const uint8_t* v_cur = v_buf + (size_t)buf * BS * 32;
        const __nv_bfloat16* vs_cur = vs_buf + buf * BS * NG;
        const float* w_cur = s_buf + buf * BS;  // pre-computed weights
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            uint8_t pk = v_cur[j * 32 + byte_x];
            int q = int((pk >> shift) & 0x3) - 2;
            float vs = __bfloat162float(vs_cur[j * NG + group_id]);
            float v_f = (float)q * vs;
            float w = w_cur[j];
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

            uint8_t pk = __ldg(v_pack_base + slot * 32 + byte_x);
            int q = int((pk >> shift) & 0x3) - 2;
            float vs = __bfloat162float(__ldg(
                v_scale_base + slot * NG + group_id));
            float v_f = (float)q * vs;
            // Inline softmax.
            float w = __expf(__ldg(s_row + (size_t)pos) - row_m_val) * row_inv_val;

            acc_o_old = __fmaf_rn(w, v_f, acc_o_old);
            if (d == 0) acc_W += w;
        }
        o_old_unnorm[bh * D_FIXED + d] = acc_o_old;
        if (d == 0) W_out[bh] = acc_W;
    }
}


// v2 wrappers — same external signatures as legacy, but route through
// softmax_norm + weighted_v_split_v2.

torch::Tensor int2_softmax_v_v2(
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
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32);
    TORCH_CHECK(v_data.scalar_type() == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_QUARTER);
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

    // Row reductions: (B, H_q) each. ~4 KB total.
    auto row_max    = torch::empty({B, (int64_t)H_q}, opts_f32);
    auto row_invsum = torch::empty({B, (int64_t)H_q}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();

    {
        dim3 grid_a((unsigned)B, (unsigned)H_q);
        dim3 block_a(THREADS_B);
        const size_t smem_a = 16 * sizeof(float);
        softmax_norm_kernel<<<grid_a, block_a, smem_a, stream.stream()>>>(
            sc.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            row_max.data_ptr<float>(),
            row_invsum.data_ptr<float>(),
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
        const size_t smem_b = (size_t)2 * BS * 32  // v_buf
                            + (size_t)2 * BS * NG * 2 // vs_buf [2][BS][NG]
                            + (size_t)2 * BS * 4   // s_buf (scores fp32)
                            + 16;
        cudaFuncSetAttribute(weighted_v_split_v2_kernel<BS, NG>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_b);
        weighted_v_split_v2_kernel<BS, NG><<<grid_b, block_b, smem_b, stream.stream()>>>(
            sc.data_ptr<float>(),
            row_max.data_ptr<float>(),
            row_invsum.data_ptr<float>(),
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
    } else if (block_size == 32 && num_groups == 1) {
        launch_b(std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{});
    } else {
        TORCH_CHECK(false,
            "int2_softmax_v_v2: unsupported (block_size=", block_size,
            ", num_groups=", num_groups, ").  Supported: (16,1), (16,2), (32,1).");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return o_hat;
}


std::vector<torch::Tensor> int2_softmax_v_with_pre_corr_v2(
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
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32);
    TORCH_CHECK(v_data.scalar_type() == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64);
    TORCH_CHECK(scores.dim() == 3);
    TORCH_CHECK(topk_idx.dim() == 3);
    TORCH_CHECK(topk_idx.size(2) == top_k);
    TORCH_CHECK(v_data.dim() == 3 && v_data.size(2) == D_QUARTER);
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

    auto row_max    = torch::empty({B, (int64_t)H_q}, opts_f32);
    auto row_invsum = torch::empty({B, (int64_t)H_q}, opts_f32);

    auto stream = at::cuda::getCurrentCUDAStream();
    {
        dim3 grid_a((unsigned)B, (unsigned)H_q);
        dim3 block_a(THREADS_B);
        const size_t smem_a = 16 * sizeof(float);
        softmax_norm_kernel<<<grid_a, block_a, smem_a, stream.stream()>>>(
            sc.data_ptr<float>(),
            sl.data_ptr<int32_t>(),
            row_max.data_ptr<float>(),
            row_invsum.data_ptr<float>(),
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
        const size_t smem_b = (size_t)2 * BS * 32
                            + (size_t)2 * BS * NG * 2
                            + (size_t)2 * BS * 4
                            + 16;
        cudaFuncSetAttribute(weighted_v_split_v2_kernel<BS, NG>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_b);
        weighted_v_split_v2_kernel<BS, NG><<<grid_b, block_b, smem_b, stream.stream()>>>(
            sc.data_ptr<float>(),
            row_max.data_ptr<float>(),
            row_invsum.data_ptr<float>(),
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
    } else if (block_size == 32 && num_groups == 1) {
        launch_b(std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{});
    } else {
        TORCH_CHECK(false,
            "int2_softmax_v_with_pre_corr_v2: unsupported (block_size=", block_size,
            ", num_groups=", num_groups, ").");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {o_hat, o_old_unnorm, W_out};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int2_softmax_v", &int2_softmax_v,
          "Stage B v3: 2-kernel split-S + SMEM staging "
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

    m.def("int2_softmax_v_with_pre_corr", &int2_softmax_v_with_pre_corr,
          "Stage B v4: also accumulates o_old_unnorm + W via in-kernel "
          "top-K bitmap.  Returns [o_hat, o_old_unnorm, W]. "
          "GIL released so the CPU worker thread (cpu_topk_attn) can run "
          "concurrently with this kernel's launches.",
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

    m.def("int2_softmax_v_with_pre_corr_wvs2", &int2_softmax_v_with_pre_corr_wvs2,
          "Stage B v4 + wvs2 weighted_v_split with pre-multiplied w·vs at "
          "chunk-load time (drops bf16 conv + FMUL in inner FMA). NG=1 "
          "only. Same outputs as int2_softmax_v_with_pre_corr.",
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
    m.def("int2_softmax_v_with_pre_corr_wvs2_inplace",
          &int2_softmax_v_with_pre_corr_wvs2_inplace,
          "Graph-safe INT2 wvs2 Stage B using caller-owned workspace.",
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

    m.def("int2_softmax_v_v2", &int2_softmax_v_v2,
          "Stage B v2: softmax_norm + weighted_v_split_v2. "
          "Eliminates the (B, H_q, max_S) FP32 weights tensor — only "
          "(row_max, row_invsum) cross HBM, weight computed inline.",
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

    m.def("int2_softmax_v_with_pre_corr_v2", &int2_softmax_v_with_pre_corr_v2,
          "Stage B v4 + v2 weights elimination. Same outputs as "
          "int2_softmax_v_with_pre_corr but no (B,H_q,max_S) weights tensor.",
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
}
