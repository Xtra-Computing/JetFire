// Fused correction kernels for the JetFire CPU-correction decode path.
//
// fused_post_correction collapses stages 08 (o_new) and 09 (blend) into a
// single CUDA launch.  Replaces this PyTorch op chain (~10 GPU launches +
// matching Python dispatch):
//
//   s_new = bmm(Q.unsqueeze(1), K_topk.transpose(-2,-1)).squeeze(1) * scale
//   a_new = softmax(s_new, dim=-1, dtype=fp32)
//   o_new = bmm(a_new.unsqueeze(1), V_topk).squeeze(1)
//   o_corrected = o_hat + W * (o_new - o_old)
//
// Per (b, h_q) block:
//   1. Load Q_row (D bf16 → fp32) into SMEM.
//   2. s_new[k] = scale · Σ_d Q[d] · K_topk[k, d]    for k in [0, K).
//   3. softmax over K (max reduce → exp → sum reduce → divide).
//   4. o_new[d] = Σ_k a_new[k] · V_topk[k, d]        for d in [0, D).
//   5. o_corr[d] = o_hat[d] + W · (o_new[d] − o_old[d]).
//
// Shapes targeted at the JetFire production config: H_q=32, H_kv=8, D=128,
// top_K up to 256.  K and D are template params; specialise per call.
//
// Compute is tiny (K·D + K·D ≈ 16 KMACs per (b,h_q) ≈ 16 MFLOPs / token at
// H_q=32) — kernel is launch-overhead bound, which is the entire point.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>


namespace {

template <int K_, int D_>
__global__ void fused_post_correction_kernel(
    const __nv_bfloat16* __restrict__ Q,        // (B, H_q, D) bf16
    const float*         __restrict__ K_topk,   // (B, H_q, K, D) fp32 (already on GPU)
    const float*         __restrict__ V_topk,   // (B, H_q, K, D) fp32
    const float*         __restrict__ O_hat,    // (B, H_q, D) fp32
    const float*         __restrict__ O_old,    // (B, H_q, D) fp32
    const float*         __restrict__ W_buf,    // (B, H_q, 1) fp32 (or (B, H_q))
    float*               __restrict__ O_corr,   // (B, H_q, D) fp32
    int H_q, float scale)
{
    constexpr int K = K_;
    constexpr int D = D_;

    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;

    const size_t bh        = (size_t)b * H_q + h_q;
    const __nv_bfloat16* q_row = Q       + bh * D;
    const float*         K_row = K_topk  + bh * K * D;
    const float*         V_row = V_topk  + bh * K * D;
    const float*         o_hat_row = O_hat + bh * D;
    const float*         o_old_row = O_old + bh * D;
    float*               o_out     = O_corr + bh * D;
    const float          w_val     = W_buf[bh];

    extern __shared__ float smem[];
    float* q_smem    = smem;          // [D]
    float* a_smem    = smem + D;      // [K] — first holds s_new, then a_new
    float* reduce_smem = a_smem + K;  // [2] for max + sum

    // 1. Load Q (bf16 → fp32) into SMEM.
    for (int d = tid; d < D; d += blockSize) {
        q_smem[d] = __bfloat162float(q_row[d]);
    }
    __syncthreads();

    // 2. s_new[k] = scale * sum_d Q[d] * K_topk[k, d].
    //    One thread handles one k for K <= 256.
    if (tid < K) {
        const float* k_ptr = K_row + (size_t)tid * D;
        float acc = 0.0f;
        #pragma unroll 16
        for (int d = 0; d < D; ++d) {
            acc += q_smem[d] * k_ptr[d];
        }
        a_smem[tid] = acc * scale;
    }
    __syncthreads();

    // 3a. max(s_new) over K.  Keep the reduction single-warp and let each
    // lane scan multiple K entries so K=128/256 are handled correctly.
    if (tid < 32) {
        float my_max = -INFINITY;
        for (int k = tid; k < K; k += 32) {
            my_max = fmaxf(my_max, a_smem[k]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffffu, my_max, off));
        }
        if (tid == 0) reduce_smem[0] = my_max;
    }
    __syncthreads();
    const float max_s = reduce_smem[0];

    // 3b. exp + sum.
    if (tid < K) {
        a_smem[tid] = __expf(a_smem[tid] - max_s);
    }
    __syncthreads();
    if (tid < 32) {
        float my_sum = 0.0f;
        for (int k = tid; k < K; k += 32) {
            my_sum += a_smem[k];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            my_sum += __shfl_xor_sync(0xffffffffu, my_sum, off);
        }
        if (tid == 0) reduce_smem[1] = my_sum;
    }
    __syncthreads();
    const float l = reduce_smem[1];
    const float inv_l = 1.0f / l;

    // 3c. divide → a_new in a_smem.
    if (tid < K) {
        a_smem[tid] = a_smem[tid] * inv_l;
    }
    __syncthreads();

    // 4. o_new[d] = Σ_k a_new[k] · V_topk[k, d].
    //    Each thread handles one d (D = 128 == blockSize).
    if (tid < D) {
        const int d = tid;
        float acc = 0.0f;
        #pragma unroll 16
        for (int k = 0; k < K; ++k) {
            acc += a_smem[k] * V_row[(size_t)k * D + d];
        }
        // 5. blend: o_corr[d] = o_hat[d] + W · (o_new[d] − o_old[d]).
        o_out[d] = o_hat_row[d] + w_val * (acc - o_old_row[d]);
    }
}

}  // namespace


torch::Tensor fused_post_correction(
    torch::Tensor q,           // (B, H_q, D) bf16
    torch::Tensor k_topk,      // (B, H_q, K, D) fp32
    torch::Tensor v_topk,      // (B, H_q, K, D) fp32
    torch::Tensor o_hat,       // (B, H_q, D) fp32
    torch::Tensor o_old,       // (B, H_q, D) fp32
    torch::Tensor w_tensor,    // (B, H_q, 1) or (B, H_q) fp32
    double scale)
{
    TORCH_CHECK(q.is_cuda() && k_topk.is_cuda() && v_topk.is_cuda(), "CUDA only");
    TORCH_CHECK(o_hat.is_cuda() && o_old.is_cuda() && w_tensor.is_cuda());
    TORCH_CHECK(q.scalar_type()      == torch::kBFloat16, "q bf16");
    TORCH_CHECK(k_topk.scalar_type() == torch::kFloat32,  "k_topk fp32");
    TORCH_CHECK(v_topk.scalar_type() == torch::kFloat32,  "v_topk fp32");
    TORCH_CHECK(o_hat.scalar_type()  == torch::kFloat32,  "o_hat fp32");
    TORCH_CHECK(o_old.scalar_type()  == torch::kFloat32,  "o_old fp32");
    TORCH_CHECK(w_tensor.scalar_type() == torch::kFloat32, "w fp32");
    TORCH_CHECK(q.dim() == 3 && k_topk.dim() == 4 && v_topk.dim() == 4);

    auto qc      = q.contiguous();
    auto kc      = k_topk.contiguous();
    auto vc      = v_topk.contiguous();
    auto oh      = o_hat.contiguous();
    auto oo      = o_old.contiguous();
    auto wc_view = w_tensor.contiguous().view(-1);  // flatten (B*H_q,)

    const int B   = (int)qc.size(0);
    const int H_q = (int)qc.size(1);
    const int D   = (int)qc.size(2);
    const int K   = (int)kc.size(2);

    TORCH_CHECK(kc.size(0) == B && kc.size(1) == H_q && kc.size(3) == D, "k_topk shape mismatch");
    TORCH_CHECK(vc.size(0) == B && vc.size(1) == H_q && vc.size(2) == K && vc.size(3) == D);
    TORCH_CHECK(oh.size(0) == B && oh.size(1) == H_q && oh.size(2) == D);
    TORCH_CHECK(oo.size(0) == B && oo.size(1) == H_q && oo.size(2) == D);
    TORCH_CHECK((int)wc_view.size(0) == B * H_q);

    auto o_corr = torch::empty({B, H_q, D}, qc.options().dtype(torch::kFloat32));

    dim3 grid((unsigned)B, (unsigned)H_q);
    dim3 block(256);
    const size_t smem_bytes = (D + K + 2) * sizeof(float);

    const auto stream = at::cuda::getCurrentCUDAStream();
    if (K == 64 && D == 128) {
        fused_post_correction_kernel<64, 128>
            <<<grid, block, smem_bytes, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr<at::BFloat16>()),
            kc.data_ptr<float>(), vc.data_ptr<float>(),
            oh.data_ptr<float>(), oo.data_ptr<float>(),
            wc_view.data_ptr<float>(),
            o_corr.data_ptr<float>(),
            H_q, (float)scale);
    } else if (K == 32 && D == 128) {
        fused_post_correction_kernel<32, 128>
            <<<grid, block, smem_bytes, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr<at::BFloat16>()),
            kc.data_ptr<float>(), vc.data_ptr<float>(),
            oh.data_ptr<float>(), oo.data_ptr<float>(),
            wc_view.data_ptr<float>(),
            o_corr.data_ptr<float>(),
            H_q, (float)scale);
    } else if (K == 128 && D == 128) {
        fused_post_correction_kernel<128, 128>
            <<<grid, block, smem_bytes, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr<at::BFloat16>()),
            kc.data_ptr<float>(), vc.data_ptr<float>(),
            oh.data_ptr<float>(), oo.data_ptr<float>(),
            wc_view.data_ptr<float>(),
            o_corr.data_ptr<float>(),
            H_q, (float)scale);
    } else if (K == 256 && D == 128) {
        fused_post_correction_kernel<256, 128>
            <<<grid, block, smem_bytes, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(qc.data_ptr<at::BFloat16>()),
            kc.data_ptr<float>(), vc.data_ptr<float>(),
            oh.data_ptr<float>(), oo.data_ptr<float>(),
            wc_view.data_ptr<float>(),
            o_corr.data_ptr<float>(),
            H_q, (float)scale);
    } else {
        TORCH_CHECK(false,
            "fused_post_correction: unsupported (K,D)=(", K, ",", D, "); ",
            "supported: (32,128), (64,128), (128,128), (256,128)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o_corr;
}


// ──────────────────────────────────────────────────────────────────────────
// fused_pre_correction collapses stages 03 (softmax) + 05 (topk_anorm) +
// 06 (V_int2 gather + dequant + bmm) into a single CUDA launch.  Replaces:
//
//   a_hat   = F.softmax(scores)                          # 1 kernel
//   a_topk  = a_hat.gather(-1, topk_idx)                 # 1 op
//   W       = a_topk.sum(-1, keepdim=True).clamp(min=1e-8)
//   a_norm  = a_topk / W
//   topk_slots  = seq_slots[topk_idx]
//   v_packed_topk = V_data[head_map, topk_slots]         # fancy-index gather
//   v_scale_topk  = V_scales[head_map, topk_slots]
//   v_topk_int    = dequant_int2_fp32(v_packed_topk, v_scale_topk, ...)
//   o_old   = bmm(a_norm.unsqueeze(1), v_topk_int).squeeze(1)
//
// Per (h_q) block (B=1 per-seq path):
//   1. Reduce max over scores[h_q, :S] (block-wide).
//   2. Reduce sum of exp(scores - max) → softmax denominator l.
//   3. For each k in [0, K): a_topk[k] = exp(scores[topk_idx[k]] - max) / l,
//      cache topk_slot[k] = seq_slots[topk_idx[k]] in SMEM.
//   4. W = max(Σ_k a_topk[k], ε), a_norm[k] = a_topk[k] / W.
//   5. For each d in [0, D):
//      o_old[d] = Σ_k a_norm[k] · dequant(V_int2[h_kv, topk_slot[k], byte(d)])
//      where dequant = (code - 2) · v_scale[h_kv, topk_slot[k]] for sym V at NG=1.
//
// Shapes/constants matched to the JetFire production config:
//   • V layout: (H_kv, num_slots, D/4) uint8 packed, MSB-first 4-codes-per-byte
//   • V scales: (H_kv, num_slots) bf16, NG=1 (group_size=128)
//   • V is symmetric (no zp); OFFSET=2.
//   • H_q/H_kv = num_groups; head_map[h_q] = h_q / num_groups.
// ──────────────────────────────────────────────────────────────────────────
namespace {

template <int K_, int D_>
__global__ void fused_pre_correction_kernel(
    const float*         __restrict__ Scores,    // (H_q, S) fp32
    const int64_t*       __restrict__ TopkIdx,   // (H_q, K) int64
    const int64_t*       __restrict__ SeqSlots,  // (S,) int64 - per-seq slot map
    const uint8_t*       __restrict__ V_data,    // (H_kv, num_slots, D/4) uint8
    const __nv_bfloat16* __restrict__ V_scales,  // (H_kv, num_slots) bf16
    float*               __restrict__ O_old,     // (H_q, D) fp32
    float*               __restrict__ W_out,     // (H_q,) fp32
    int H_q, int H_kv, int S, int num_groups,
    int v_data_h_stride, int v_scales_h_stride)
{
    constexpr int K = K_;
    constexpr int D = D_;
    constexpr int D_QUARTER = D / 4;   // bytes per token-head in V_data

    const int h_q = blockIdx.x;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = blockSize / 32;
    const int h_kv = h_q / num_groups;

    if (h_q >= H_q) return;

    const float*   sc_row = Scores  + (size_t)h_q * S;
    const int64_t* tk_row = TopkIdx + (size_t)h_q * K;

    extern __shared__ float smem_pre[];
    float* a_smem      = smem_pre;            // [K]
    int*   tslot_smem  = (int*)(a_smem + K);  // [K]
    float* warp_buf    = (float*)(tslot_smem + K);  // [n_warps] for cross-warp reduce

    // ── 1. max(scores) over S ────────────────────────────────────────────
    float my_max = -INFINITY;
    for (int s = tid; s < S; s += blockSize) {
        float v = sc_row[s];
        if (v > my_max) my_max = v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffffu, my_max, off));
    }
    if (lane == 0) warp_buf[warp_id] = my_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < n_warps) ? warp_buf[lane] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        }
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    const float max_s = warp_buf[0];

    // ── 2. l = Σ exp(scores - max) ───────────────────────────────────────
    float my_sum = 0.0f;
    for (int s = tid; s < S; s += blockSize) {
        my_sum += __expf(sc_row[s] - max_s);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        my_sum += __shfl_xor_sync(0xffffffffu, my_sum, off);
    }
    if (lane == 0) warp_buf[warp_id] = my_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < n_warps) ? warp_buf[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off);
        }
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    const float l = warp_buf[0];
    const float inv_l = 1.0f / fmaxf(l, 1e-30f);

    // ── 3. a_topk[k] = exp(scores[topk_idx[k]] - max) / l, cache slots ───
    if (tid < K) {
        int64_t idx = tk_row[tid];
        a_smem[tid] = __expf(sc_row[idx] - max_s) * inv_l;
        tslot_smem[tid] = (int)SeqSlots[idx];
    }
    __syncthreads();

    // ── 4. W = clamp(Σ a_topk, ε), a_norm = a_topk / W ───────────────────
    if (tid == 0) {
        float ws = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; ++k) ws += a_smem[k];
        ws = fmaxf(ws, 1e-8f);
        warp_buf[0] = ws;
        W_out[h_q] = ws;
    }
    __syncthreads();
    const float W = warp_buf[0];
    if (tid < K) {
        a_smem[tid] = a_smem[tid] / W;
    }
    __syncthreads();

    // ── 5. o_old[d] = Σ_k a_norm[k] · dequant(V_int2[h_kv, slot[k], d]) ──
    //   • V byte layout MSB-first: bits [7:6]=elem 0 of the 4-tuple
    //     starting at column 4·byte_idx, bits [1:0]=elem 3.
    //   • Sym V: code in {1, 2, 3} → signed (code - 2) ∈ {-1, 0, +1}.
    //   • value = (code - 2) · v_scale[h_kv, slot]
    if (tid < D) {
        const int d         = tid;
        const int byte_idx  = d >> 2;       // d / 4
        const int pos       = d & 3;        // d % 4
        const int shift_amt = 6 - 2 * pos;  // MSB-first

        const uint8_t*       V_h  = V_data   + (size_t)h_kv * v_data_h_stride;
        const __nv_bfloat16* Vs_h = V_scales + (size_t)h_kv * v_scales_h_stride;

        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; ++k) {
            const int slot_k = tslot_smem[k];
            const uint8_t byte_v = V_h[(size_t)slot_k * D_QUARTER + byte_idx];
            const int code = (byte_v >> shift_amt) & 0x3;
            const float v_signed = (float)(code - 2);
            const float vs = __bfloat162float(Vs_h[slot_k]);
            acc += a_smem[k] * (v_signed * vs);
        }
        O_old[(size_t)h_q * D + d] = acc;
    }
}

}  // namespace


std::tuple<torch::Tensor, torch::Tensor>
fused_pre_correction(
    torch::Tensor scores,       // (H_q, S) fp32
    torch::Tensor topk_idx,     // (H_q, K) int64
    torch::Tensor seq_slots,    // (S,) int64
    torch::Tensor v_data,       // (H_kv, num_slots, D/4) uint8
    torch::Tensor v_scales,     // (H_kv, num_slots) or (H_kv, num_slots, 1) bf16
    int64_t H_kv,
    int64_t num_groups)
{
    TORCH_CHECK(scores.is_cuda() && topk_idx.is_cuda());
    TORCH_CHECK(seq_slots.is_cuda() && v_data.is_cuda() && v_scales.is_cuda());
    TORCH_CHECK(scores.scalar_type()   == torch::kFloat32, "scores fp32");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt64,    "topk_idx int64");
    TORCH_CHECK(seq_slots.scalar_type()== torch::kInt64,    "seq_slots int64");
    TORCH_CHECK(v_data.scalar_type()   == torch::kByte,     "v_data uint8");
    TORCH_CHECK(v_scales.scalar_type() == torch::kBFloat16, "v_scales bf16");
    TORCH_CHECK(scores.dim() == 2 && topk_idx.dim() == 2);

    auto sc = scores.contiguous();
    auto ti = topk_idx.contiguous();
    auto ss = seq_slots.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 3) ? v_scales.squeeze(-1).contiguous()
                                     : v_scales.contiguous();

    const int H_q = (int)sc.size(0);
    const int S   = (int)sc.size(1);
    const int K   = (int)ti.size(1);
    const int D   = 128;  // hardcoded; matches the production config

    TORCH_CHECK(ss.size(0) == S);
    TORCH_CHECK(vd.dim() == 3 && vd.size(0) == (int)H_kv && vd.size(2) == D / 4);
    TORCH_CHECK(vs.dim() == 2 && vs.size(0) == (int)H_kv);

    const int num_slots = (int)vd.size(1);
    auto opts_f32 = sc.options().dtype(torch::kFloat32);
    auto o_old = torch::empty({H_q, D}, opts_f32);
    auto W_out = torch::empty({H_q},    opts_f32);

    dim3 grid((unsigned)H_q);
    dim3 block(256);
    const int n_warps = 256 / 32;
    const size_t smem_bytes = (K + K + n_warps) * sizeof(float);

    const auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto k_const, auto d_const) {
        fused_pre_correction_kernel<decltype(k_const)::value, decltype(d_const)::value>
            <<<grid, block, smem_bytes, stream.stream()>>>(
            sc.data_ptr<float>(),
            ti.data_ptr<int64_t>(),
            ss.data_ptr<int64_t>(),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            o_old.data_ptr<float>(),
            W_out.data_ptr<float>(),
            H_q, (int)H_kv, S, (int)num_groups,
            (int)vd.stride(0), (int)vs.stride(0));
    };

    if (K == 64 && D == 128) {
        launch(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{});
    } else if (K == 32 && D == 128) {
        launch(std::integral_constant<int, 32>{}, std::integral_constant<int, 128>{});
    } else if (K == 128 && D == 128) {
        launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{});
    } else if (K == 256 && D == 128) {
        launch(std::integral_constant<int, 256>{}, std::integral_constant<int, 128>{});
    } else {
        TORCH_CHECK(false, "fused_pre_correction: unsupported (K,D)=(", K, ",", D, ")");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_old, W_out};
}


// ──────────────────────────────────────────────────────────────────────────
// fused_pre_correction_batched — same as fused_pre_correction but processes
// B sequences in one launch, enabling true batched decode under correction.
//
// Grid: (B, H_q).  Per (b, h_q):
//   • S_b = SeqLens[b]
//   • Reduce max + sum over Scores[b, h_q, 0..S_b)  (mask positions ≥ S_b)
//   • Compute slots from BlockTable + block_size:
//       slot = BlockTable[b, idx/bs] · bs + (idx % bs)
//   • Sym-V dequant + bmm, output O_old[b, h_q, :], W[b, h_q]
//
// The Scores tensor is (B, H_q, max_S) but only [0, S_b) is valid for each b.
// fused_decode_wmma_v5_int2 only writes valid positions, so we skip s ≥ S_b.
// ──────────────────────────────────────────────────────────────────────────
namespace {

template <int K_, int D_>
__global__ void fused_pre_correction_batched_kernel(
    const float*         __restrict__ Scores,     // (B, H_q, max_S) fp32
    const int64_t*       __restrict__ TopkIdx,    // (B, H_q, K) int64
    const int64_t*       __restrict__ BlockTable, // (B, max_blocks) int64
    const int32_t*       __restrict__ SeqLens,    // (B,) int32
    const uint8_t*       __restrict__ V_data,     // (H_kv, num_slots, D/4)
    const __nv_bfloat16* __restrict__ V_scales,   // (H_kv, num_slots) bf16
    float*               __restrict__ O_old,      // (B, H_q, D) fp32
    float*               __restrict__ W_out,      // (B, H_q) fp32
    int H_q, int H_kv, int max_S, int block_size, int num_groups,
    int v_data_h_stride, int v_scales_h_stride, int block_table_stride)
{
    constexpr int K = K_;
    constexpr int D = D_;
    constexpr int D_QUARTER = D / 4;

    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n_warps = blockSize / 32;
    const int h_kv = h_q / num_groups;

    const int S_b = SeqLens[b];

    const size_t bh = (size_t)b * H_q + h_q;
    const float*   sc_row = Scores  + bh * max_S;
    const int64_t* tk_row = TopkIdx + bh * K;
    const int64_t* bt_row = BlockTable + (size_t)b * block_table_stride;

    extern __shared__ float smem_pre_b[];
    float* a_smem      = smem_pre_b;
    int*   tslot_smem  = (int*)(a_smem + K);
    float* warp_buf    = (float*)(tslot_smem + K);

    // 1. max(scores) over [0, S_b)
    float my_max = -INFINITY;
    for (int s = tid; s < S_b; s += blockSize) {
        float v = sc_row[s];
        if (v > my_max) my_max = v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffffu, my_max, off));
    }
    if (lane == 0) warp_buf[warp_id] = my_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < n_warps) ? warp_buf[lane] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        }
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    const float max_s = warp_buf[0];

    // 2. l = Σ exp(scores - max)
    float my_sum = 0.0f;
    for (int s = tid; s < S_b; s += blockSize) {
        my_sum += __expf(sc_row[s] - max_s);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        my_sum += __shfl_xor_sync(0xffffffffu, my_sum, off);
    }
    if (lane == 0) warp_buf[warp_id] = my_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < n_warps) ? warp_buf[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off);
        }
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    const float l = warp_buf[0];
    const float inv_l = 1.0f / fmaxf(l, 1e-30f);

    // 3. a_topk[k] = exp(scores[topk_idx[k]] - max) / l, cache slots
    if (tid < K) {
        int64_t logical_pos = tk_row[tid];
        a_smem[tid] = __expf(sc_row[logical_pos] - max_s) * inv_l;
        // slot = block_table[b, pos/bs] * bs + pos%bs
        int blk_idx = (int)(logical_pos / block_size);
        int in_blk  = (int)(logical_pos % block_size);
        int64_t block_id = bt_row[blk_idx];
        tslot_smem[tid] = (int)(block_id * block_size + in_blk);
    }
    __syncthreads();

    // 4. W = Σ a_topk, a_norm = a_topk / W
    if (tid == 0) {
        float ws = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; ++k) ws += a_smem[k];
        ws = fmaxf(ws, 1e-8f);
        warp_buf[0] = ws;
        W_out[bh] = ws;
    }
    __syncthreads();
    const float W = warp_buf[0];
    if (tid < K) {
        a_smem[tid] = a_smem[tid] / W;
    }
    __syncthreads();

    // 5. o_old[d] = Σ_k a_norm[k] · dequant(V_int2[h_kv, slot[k], d])
    if (tid < D) {
        const int d         = tid;
        const int byte_idx  = d >> 2;
        const int pos       = d & 3;
        const int shift_amt = 6 - 2 * pos;

        const uint8_t*       V_h  = V_data   + (size_t)h_kv * v_data_h_stride;
        const __nv_bfloat16* Vs_h = V_scales + (size_t)h_kv * v_scales_h_stride;

        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; ++k) {
            const int slot_k = tslot_smem[k];
            const uint8_t byte_v = V_h[(size_t)slot_k * D_QUARTER + byte_idx];
            const int code = (byte_v >> shift_amt) & 0x3;
            const float v_signed = (float)(code - 2);
            const float vs = __bfloat162float(Vs_h[slot_k]);
            acc += a_smem[k] * (v_signed * vs);
        }
        O_old[bh * D + d] = acc;
    }
}

}  // namespace


std::tuple<torch::Tensor, torch::Tensor>
fused_pre_correction_batched(
    torch::Tensor scores,        // (B, H_q, max_S) fp32
    torch::Tensor topk_idx,      // (B, H_q, K) int64
    torch::Tensor block_table,   // (B, max_blocks) int64
    torch::Tensor seq_lens,      // (B,) int32
    torch::Tensor v_data,        // (H_kv, num_slots, D/4) uint8
    torch::Tensor v_scales,      // (H_kv, num_slots) or (H_kv, num_slots, 1) bf16
    int64_t block_size,
    int64_t H_kv,
    int64_t num_groups)
{
    TORCH_CHECK(scores.is_cuda() && topk_idx.is_cuda() && block_table.is_cuda());
    TORCH_CHECK(seq_lens.is_cuda() && v_data.is_cuda() && v_scales.is_cuda());
    TORCH_CHECK(scores.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(topk_idx.scalar_type()    == torch::kInt64);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt64);
    TORCH_CHECK(seq_lens.scalar_type()    == torch::kInt32);
    TORCH_CHECK(v_data.scalar_type()      == torch::kByte);
    TORCH_CHECK(v_scales.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(scores.dim() == 3 && topk_idx.dim() == 3);

    auto sc = scores.contiguous();
    auto ti = topk_idx.contiguous();
    auto bt = block_table.contiguous();
    auto sl = seq_lens.contiguous();
    auto vd = v_data.contiguous();
    auto vs = (v_scales.dim() == 3) ? v_scales.squeeze(-1).contiguous()
                                     : v_scales.contiguous();

    const int B     = (int)sc.size(0);
    const int H_q   = (int)sc.size(1);
    const int max_S = (int)sc.size(2);
    const int K     = (int)ti.size(2);
    const int D     = 128;

    TORCH_CHECK(ti.size(0) == B && ti.size(1) == H_q);
    TORCH_CHECK(bt.size(0) == B);
    TORCH_CHECK(sl.size(0) == B);
    TORCH_CHECK(vd.dim() == 3 && vd.size(0) == (int)H_kv && vd.size(2) == D / 4);
    TORCH_CHECK(vs.dim() == 2 && vs.size(0) == (int)H_kv);

    auto opts_f32 = sc.options().dtype(torch::kFloat32);
    auto o_old = torch::empty({B, H_q, D}, opts_f32);
    auto W_out = torch::empty({B, H_q},    opts_f32);

    dim3 grid((unsigned)B, (unsigned)H_q);
    dim3 block(256);
    const int n_warps = 256 / 32;
    const size_t smem_bytes = (K + K + n_warps) * sizeof(float);

    const auto stream = at::cuda::getCurrentCUDAStream();
    auto launch = [&](auto k_const, auto d_const) {
        fused_pre_correction_batched_kernel<decltype(k_const)::value, decltype(d_const)::value>
            <<<grid, block, smem_bytes, stream.stream()>>>(
            sc.data_ptr<float>(),
            ti.data_ptr<int64_t>(),
            bt.data_ptr<int64_t>(),
            sl.data_ptr<int32_t>(),
            vd.data_ptr<uint8_t>(),
            reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr<at::BFloat16>()),
            o_old.data_ptr<float>(),
            W_out.data_ptr<float>(),
            H_q, (int)H_kv, max_S, (int)block_size, (int)num_groups,
            (int)vd.stride(0), (int)vs.stride(0), (int)bt.stride(0));
    };

    if (K == 64 && D == 128) {
        launch(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{});
    } else if (K == 32 && D == 128) {
        launch(std::integral_constant<int, 32>{}, std::integral_constant<int, 128>{});
    } else if (K == 128 && D == 128) {
        launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{});
    } else if (K == 256 && D == 128) {
        launch(std::integral_constant<int, 256>{}, std::integral_constant<int, 128>{});
    } else {
        TORCH_CHECK(false, "fused_pre_correction_batched: unsupported (K,D)=(", K, ",", D, ")");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o_old, W_out};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_correction", &fused_post_correction,
          "Fused post-correction (08 o_new + 09 blend) — 1 launch instead of ~10",
          py::arg("q"), py::arg("k_topk"), py::arg("v_topk"),
          py::arg("o_hat"), py::arg("o_old"), py::arg("w"), py::arg("scale"));
    m.def("fused_pre_correction", &fused_pre_correction,
          "Fused pre-correction (03 softmax + 05 topk_anorm + 06 o_old) — "
          "1 launch instead of ~15.  Returns (o_old, W).",
          py::arg("scores"), py::arg("topk_idx"), py::arg("seq_slots"),
          py::arg("v_data"), py::arg("v_scales"),
          py::arg("H_kv"), py::arg("num_groups"));
    m.def("fused_pre_correction_batched", &fused_pre_correction_batched,
          "Batched fused pre-correction across B sequences (jetfire_int2 v2). "
          "Grid (B, H_q) with per-batch seq_lens masking.  Returns (o_old, W).",
          py::arg("scores"), py::arg("topk_idx"), py::arg("block_table"),
          py::arg("seq_lens"), py::arg("v_data"), py::arg("v_scales"),
          py::arg("block_size"), py::arg("H_kv"), py::arg("num_groups"));
}
