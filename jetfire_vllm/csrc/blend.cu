// JetFire blend kernel — Unit 3 of decode-orchestrator refactor.
//
// Purpose: replace the in-Python expression
//   o_corr_fp32 = o_hat + W·(o_new − o_old)
//   output = o_corr_fp32.to(bf16)
// with a single fused CUDA kernel.  Saves one launch per layer, avoids
// the fp32 intermediate.
//
// Shapes (all device-resident):
//   o_hat: (B, H_q, D) fp32 — INT2 attention output
//   o_old: (B, H_q, D) fp32 — INT pre-correction (top-K weighted, INT V)
//   W:     (B, H_q)    fp32 — top-K mass (sum of attention weights at top-K)
//   o_new: (B, H_q, D) bf16 — CPU-computed top-K attention (full bf16)
// Output:
//   out:   (B, H_q, D) bf16

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>


static inline bool blend_trace_enabled() {
    const char* env = std::getenv("JETFIRE_BLEND_TRACE");
    return env != nullptr && env[0] == '1';
}


static inline void blend_trace(const char* msg) {
    if (blend_trace_enabled()) {
        std::fprintf(stderr, "[jetfire][blend.cu] %s\n", msg);
        std::fflush(stderr);
    }
}


__global__ void blend_kernel(
    const float* __restrict__ o_hat,
    const float* __restrict__ o_old,
    const float* __restrict__ W,
    const __nv_bfloat16* __restrict__ o_new,
    __nv_bfloat16* __restrict__ out,
    int H_q, int D)
{
    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;

    const size_t bh = (size_t)b * H_q + h;
    const float* hat_p = o_hat + bh * D;
    const float* old_p = o_old + bh * D;
    const __nv_bfloat16* new_p = o_new + bh * D;
    __nv_bfloat16* out_p = out + bh * D;
    const float w = W[bh];

    for (int d = tid; d < D; d += blockDim.x) {
        const float n_f = __bfloat162float(new_p[d]);
        const float v   = hat_p[d] + w * (n_f - old_p[d]);
        out_p[d] = __float2bfloat16(v);
    }
}


torch::Tensor blend(
    torch::Tensor o_hat,
    torch::Tensor o_old,
    torch::Tensor W,
    torch::Tensor o_new)
{
    TORCH_CHECK(o_hat.is_cuda() && o_old.is_cuda() && W.is_cuda() && o_new.is_cuda(),
                "blend: all inputs must be CUDA tensors");
    TORCH_CHECK(o_hat.scalar_type() == torch::kFloat32, "o_hat must be fp32");
    TORCH_CHECK(o_old.scalar_type() == torch::kFloat32, "o_old must be fp32");
    TORCH_CHECK(W.scalar_type()     == torch::kFloat32, "W must be fp32");
    TORCH_CHECK(o_new.scalar_type() == torch::kBFloat16, "o_new must be bf16");
    TORCH_CHECK(o_hat.dim() == 3 && o_old.dim() == 3 && o_new.dim() == 3,
                "o_hat/o_old/o_new must be 3D (B, H_q, D)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (B, H_q)");

    auto hc = o_hat.contiguous();
    auto oc = o_old.contiguous();
    auto wc = W.contiguous();
    auto nc = o_new.contiguous();

    const int B   = hc.size(0);
    const int H_q = hc.size(1);
    const int D   = hc.size(2);

    TORCH_CHECK(oc.sizes() == hc.sizes(), "o_old shape mismatch");
    TORCH_CHECK(nc.sizes() == hc.sizes(), "o_new shape mismatch");
    TORCH_CHECK(wc.size(0) == B && wc.size(1) == H_q, "W shape mismatch");

    auto out = torch::empty(
        {B, H_q, D},
        torch::TensorOptions().dtype(torch::kBFloat16).device(hc.device())
    );

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(B, H_q);
    int threads = D < 128 ? D : 128;

    blend_kernel<<<grid, threads, 0, stream.stream()>>>(
        hc.data_ptr<float>(),
        oc.data_ptr<float>(),
        wc.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(nc.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        H_q, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}


// In-place variant: writes directly into the caller-provided `out` buffer.
// Skips the per-call bf16 tensor alloc that ``blend`` would otherwise do.
// `out` must already be (B, H_q, D) bf16 on the same device as o_hat.
void blend_into(
    torch::Tensor o_hat,
    torch::Tensor o_old,
    torch::Tensor W,
    torch::Tensor o_new,
    torch::Tensor out)
{
    blend_trace("blend_into enter");
    TORCH_CHECK(o_hat.is_cuda() && o_old.is_cuda() && W.is_cuda() &&
                o_new.is_cuda() && out.is_cuda(),
                "blend_into: all tensors must be CUDA");
    TORCH_CHECK(o_hat.scalar_type() == torch::kFloat32);
    TORCH_CHECK(o_old.scalar_type() == torch::kFloat32);
    TORCH_CHECK(W.scalar_type()     == torch::kFloat32);
    TORCH_CHECK(o_new.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(out.scalar_type()   == torch::kBFloat16);
    TORCH_CHECK(o_hat.dim() == 3 && o_old.dim() == 3 && o_new.dim() == 3 &&
                out.dim() == 3, "o_hat/o_old/o_new/out must be 3D");
    TORCH_CHECK(W.dim() == 2);

    blend_trace("blend_into checks_done");
    auto hc = o_hat.contiguous();
    blend_trace("blend_into o_hat_contiguous_done");
    auto oc = o_old.contiguous();
    blend_trace("blend_into o_old_contiguous_done");
    auto wc = W.contiguous();
    blend_trace("blend_into W_contiguous_done");
    auto nc = o_new.contiguous();
    blend_trace("blend_into o_new_contiguous_done");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int B   = hc.size(0);
    const int H_q = hc.size(1);
    const int D   = hc.size(2);
    TORCH_CHECK(oc.sizes() == hc.sizes(), "o_old shape mismatch");
    TORCH_CHECK(nc.sizes() == hc.sizes(), "o_new shape mismatch");
    TORCH_CHECK(out.sizes() == hc.sizes(), "out shape mismatch");
    TORCH_CHECK(wc.size(0) == B && wc.size(1) == H_q, "W shape mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(B, H_q);
    int threads = D < 128 ? D : 128;
    if (blend_trace_enabled()) {
        std::fprintf(
            stderr,
            "[jetfire][blend.cu] blend_into launch B=%d H_q=%d D=%d threads=%d\n",
            B, H_q, D, threads);
        std::fflush(stderr);
    }

    blend_kernel<<<grid, threads, 0, stream.stream()>>>(
        hc.data_ptr<float>(),
        oc.data_ptr<float>(),
        wc.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(nc.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        H_q, D
    );
    blend_trace("blend_into launch_returned");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    blend_trace("blend_into launch_check_done");
}


// New formula variant: out = o_hat + W·o_new − o_old_unnorm.  This is what
// the Stage-B-v4 fused-pre_corr path produces (o_old kept unnormalized so
// the kernel doesn't need to do a divide-by-W).  Mathematically:
//     o_corr = o_hat + W·(o_new − o_old_norm)
//            = o_hat + W·o_new − W·o_old_norm
//            = o_hat + W·o_new − o_old_unnorm   (since o_old_unnorm = W · o_old_norm)
//
// Inputs:
//   o_hat       : (B, H_q, D) fp32
//   o_old_unnorm: (B, H_q, D) fp32
//   W           : (B, H_q)    fp32
//   o_new       : (B, H_q, D) bf16
//   out         : (B, H_q, D) bf16 (in-place)

__global__ void blend_unnorm_kernel(
    const float* __restrict__ o_hat,
    const float* __restrict__ o_old_unnorm,
    const float* __restrict__ W,
    const __nv_bfloat16* __restrict__ o_new,
    __nv_bfloat16* __restrict__ out,
    int H_q, int D)
{
    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;

    const size_t bh = (size_t)b * H_q + h;
    const float* hat_p = o_hat        + bh * D;
    const float* old_p = o_old_unnorm + bh * D;
    const __nv_bfloat16* new_p = o_new + bh * D;
    __nv_bfloat16* out_p = out + bh * D;
    const float w = W[bh];

    for (int d = tid; d < D; d += blockDim.x) {
        const float n_f = __bfloat162float(new_p[d]);
        const float v   = hat_p[d] + w * n_f - old_p[d];
        out_p[d] = __float2bfloat16(v);
    }
}


void blend_unnorm_into(
    torch::Tensor o_hat,
    torch::Tensor o_old_unnorm,
    torch::Tensor W,
    torch::Tensor o_new,
    torch::Tensor out)
{
    blend_trace("blend_unnorm_into enter");
    TORCH_CHECK(o_hat.is_cuda() && o_old_unnorm.is_cuda() && W.is_cuda() &&
                o_new.is_cuda() && out.is_cuda(),
                "blend_unnorm_into: all tensors must be CUDA");
    TORCH_CHECK(o_hat.scalar_type() == torch::kFloat32);
    TORCH_CHECK(o_old_unnorm.scalar_type() == torch::kFloat32);
    TORCH_CHECK(W.scalar_type()     == torch::kFloat32);
    TORCH_CHECK(o_new.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(out.scalar_type()   == torch::kBFloat16);
    TORCH_CHECK(o_hat.dim() == 3 && o_old_unnorm.dim() == 3 && o_new.dim() == 3 &&
                out.dim() == 3, "tensors must be 3D");
    TORCH_CHECK(W.dim() == 2);

    blend_trace("blend_unnorm_into checks_done");
    auto hc = o_hat.contiguous();
    blend_trace("blend_unnorm_into o_hat_contiguous_done");
    auto oc = o_old_unnorm.contiguous();
    blend_trace("blend_unnorm_into o_old_contiguous_done");
    auto wc = W.contiguous();
    blend_trace("blend_unnorm_into W_contiguous_done");
    auto nc = o_new.contiguous();
    blend_trace("blend_unnorm_into o_new_contiguous_done");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int B   = hc.size(0);
    const int H_q = hc.size(1);
    const int D   = hc.size(2);
    TORCH_CHECK(oc.sizes() == hc.sizes());
    TORCH_CHECK(nc.sizes() == hc.sizes());
    TORCH_CHECK(out.sizes() == hc.sizes());
    TORCH_CHECK(wc.size(0) == B && wc.size(1) == H_q);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(B, H_q);
    int threads = D < 128 ? D : 128;
    if (blend_trace_enabled()) {
        std::fprintf(
            stderr,
            "[jetfire][blend.cu] blend_unnorm_into launch B=%d H_q=%d D=%d threads=%d\n",
            B, H_q, D, threads);
        std::fflush(stderr);
    }

    blend_unnorm_kernel<<<grid, threads, 0, stream.stream()>>>(
        hc.data_ptr<float>(),
        oc.data_ptr<float>(),
        wc.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(nc.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        H_q, D
    );
    blend_trace("blend_unnorm_into launch_returned");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    blend_trace("blend_unnorm_into launch_check_done");
}


// Fused V2_04 join + V2_05 post_corr.  Replaces two Python-level calls
// (Future.result + blend_unnorm_into) with one C++ entry that holds the
// GIL released throughout.  Skips the ~25 µs cv-wakeup of threading.Event
// used inside ThreadPoolExecutor.Future.result().
//
// Synchronization model:
//   1. CPU worker (cpu_topk_attn) writes 1 to `done_flag` with release
//      ordering AFTER its _mm_sfence().  Acquire-load of 1 here means all
//      worker writes (incl. WC stores to mapped pinned o_new) are flushed.
//   2. Main thread spin-waits on the flag with _mm_pause() — work is
//      ~50 µs so spin is far cheaper than a futex wakeup.
//   3. blend_unnorm_kernel launched on the current (compute) stream.
//
// Caller is required to be in mapped-o_new mode (the only path that calls
// this fn today).  Mapped means o_new lives in cudaHostAllocMapped pinned
// memory; the GPU reads it via PCIe and the CPU's _mm_sfence + release
// store on done_flag is the only ordering primitive needed.  No
// cross-stream cudaStreamWaitEvent is required — comm stream's q/ts D2H
// is only consumed by the CPU worker, and blend reads o_hat/o_old/W from
// the same compute stream that produced them.  Removing that wait eliminates
// the cross-stream chain that saturates the CUDA driver queue at high B
// (verified: B=32 gen=64 went from hang ≥600s to 9.3s wall).
//
// No exception path: if the worker dies the spin would hang.  Acceptable
// trade-off in our hot decode loop where worker exceptions are vanishingly
// rare; revert to JETFIRE_FUSED_JOIN_BLEND=0 for debugging.

void wait_and_blend_unnorm_into(
    torch::Tensor done_flag,           // (1,) int64 CPU
    torch::Tensor o_hat,               // (B, H_q, D) fp32 CUDA
    torch::Tensor o_old_unnorm,        // (B, H_q, D) fp32 CUDA
    torch::Tensor W,                   // (B, H_q)    fp32 CUDA
    torch::Tensor o_new,               // (B, H_q, D) bf16 CUDA (mapped)
    torch::Tensor out)                 // (B, H_q, D) bf16 CUDA [out]
{
    TORCH_CHECK(done_flag.is_cpu() && done_flag.numel() == 1
                && done_flag.scalar_type() == torch::kInt64,
                "done_flag must be a 1-element int64 CPU tensor");
    TORCH_CHECK(o_hat.is_cuda() && o_old_unnorm.is_cuda() && W.is_cuda() &&
                o_new.is_cuda() && out.is_cuda(),
                "wait_and_blend_unnorm_into: all device tensors must be CUDA");
    TORCH_CHECK(o_hat.scalar_type() == torch::kFloat32);
    TORCH_CHECK(o_old_unnorm.scalar_type() == torch::kFloat32);
    TORCH_CHECK(W.scalar_type()     == torch::kFloat32);
    TORCH_CHECK(o_new.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(out.scalar_type()   == torch::kBFloat16);

    // 1. Spin-wait on completion flag (acquire-load).  No syscall, no GIL.
    auto* flag_p = reinterpret_cast<std::atomic<int64_t>*>(
        done_flag.data_ptr<int64_t>());
    while (flag_p->load(std::memory_order_acquire) == 0) {
        _mm_pause();
    }

    // 2. Shape checks (kept after the wait so they don't add to the
    //    main-thread arrival latency on the spin loop).
    auto hc = o_hat.contiguous();
    auto oc = o_old_unnorm.contiguous();
    auto wc = W.contiguous();
    auto nc = o_new.contiguous();
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int B   = hc.size(0);
    const int H_q = hc.size(1);
    const int D   = hc.size(2);
    TORCH_CHECK(oc.sizes() == hc.sizes());
    TORCH_CHECK(nc.sizes() == hc.sizes());
    TORCH_CHECK(out.sizes() == hc.sizes());
    TORCH_CHECK(wc.size(0) == B && wc.size(1) == H_q);

    // 3. Launch blend kernel inline on the compute stream.
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(B, H_q);
    int threads = D < 128 ? D : 128;
    blend_unnorm_kernel<<<grid, threads, 0, compute_stream.stream()>>>(
        hc.data_ptr<float>(),
        oc.data_ptr<float>(),
        wc.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(nc.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        H_q, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blend", &blend,
          "JetFire blend kernel: o_corr = o_hat + W*(o_new - o_old) (bf16 out)",
          py::arg("o_hat"), py::arg("o_old"), py::arg("W"), py::arg("o_new"));
    m.def("blend_into", &blend_into,
          "Same as blend but writes into pre-allocated `out` (no alloc)",
          py::arg("o_hat"), py::arg("o_old"), py::arg("W"),
          py::arg("o_new"), py::arg("out"));
    m.def("blend_unnorm_into", &blend_unnorm_into,
          "Variant for Stage B v4: o_corr = o_hat + W·o_new − o_old_unnorm "
          "(o_old kept unnormalized; saves a divide).",
          py::arg("o_hat"), py::arg("o_old_unnorm"), py::arg("W"),
          py::arg("o_new"), py::arg("out"));
    m.def("wait_and_blend_unnorm_into", &wait_and_blend_unnorm_into,
          py::call_guard<py::gil_scoped_release>(),
          "Fused V2_04+V2_05: spin-wait on done_flag then launch the "
          "blend_unnorm kernel.  Caller must be in mapped-o_new mode so no "
          "cross-stream wait is needed.  Replaces Future.result + blend_into.",
          py::arg("done_flag"),
          py::arg("o_hat"), py::arg("o_old_unnorm"), py::arg("W"),
          py::arg("o_new"), py::arg("out"));
}
