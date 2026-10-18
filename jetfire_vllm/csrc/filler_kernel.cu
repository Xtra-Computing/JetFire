// JetFire Phase 0 — GPU filler kernel.
// Artifact provenance: copied from unit_test/csrc/filler_kernel.cu in the
// JetFire development workspace; included here to keep the loader standalone.
//
// Purpose: insert a configurable-duration GPU "busywork" kernel into the
// decode pipeline to test whether the CPU overlap window has enough
// headroom to absorb additional GPU work.  If wall-clock throughput is
// flat as filler_us grows from 0 to ~500us, then Path A (split INT2 attn
// to fill that window) has measurable headroom.  If throughput drops
// sharply, the window is already saturated.
//
// Implementation: 1-block, 1-thread kernel that spins on clock64 until
// the requested wall time has elapsed.  H100 SM clock is ~1.6 GHz so we
// convert µs → cycles ≈ us * 1600.  Accuracy is ~30%, fine for the knee.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>


__device__ __forceinline__ uint64_t get_clock64() {
    uint64_t v;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(v));
    return v;
}


__global__ void filler_spin_kernel(uint64_t cycles) {
    if (threadIdx.x == 0) {
        const uint64_t start = get_clock64();
        const uint64_t end   = start + cycles;
        while (get_clock64() < end) {
            // intentional spin; __nanosleep(64) avoids 100% issue rate
            __nanosleep(64);
        }
    }
}


// Approximation: H100 PCIe SM clock ≈ 1.6 GHz under load.
// 1 µs ≈ 1600 cycles.  Calibrated good enough for finding a knee point.
static constexpr uint64_t H100_CYCLES_PER_US = 1600ULL;


void launch_filler(int64_t us) {
    if (us <= 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint64_t cycles = (uint64_t)us * H100_CYCLES_PER_US;
    filler_spin_kernel<<<1, 32, 0, stream.stream()>>>(cycles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// For unit-test calibration: launch + measure wall time, verify accuracy.
double calibrate_filler(int64_t us) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEventRecord(a, stream.stream());
    launch_filler(us);
    cudaEventRecord(b, stream.stream());
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return (double)ms * 1000.0;  // returns measured µs
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_filler", &launch_filler, py::arg("us"),
          "Launch a GPU spin kernel that occupies the current stream "
          "for approximately `us` microseconds");
    m.def("calibrate_filler", &calibrate_filler, py::arg("us"),
          "Launch + sync + measure: returns actual measured µs");
}
