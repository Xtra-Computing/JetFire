// Parallel CPU shadow K/V gather for the JetFire decode CPU correction path.
//
// Replaces two `torch.index_select` calls (Python-dispatched) with a single
// C++ entry that:
//   • Releases the GIL via py::call_guard at the binding level — so a
//     worker-thread caller can actually run concurrently with the main
//     thread issuing CUDA ops.  The Step-A Python ThreadPoolExecutor fork
//     showed +12 % TPOT regression because torch.index_select holds the
//     GIL through Python dispatch.
//   • Spreads the H_q × top_k rows across CPU threads via at::parallel_for
//     (OpenMP under the hood).  Each row is a 256-byte (D=128 bf16) memcpy.
//   • Computes the source offset `hkv * num_slots + slot` inline, saving
//     the (head_map * num_slots + ts_buf) integer-multiply tensor build
//     that the Python path performed.
//
// API (matches the legacy `_decode_one_seq` gather inputs):
//   k_shadow, v_shadow : (H_kv, num_slots, D) bf16, CPU
//   head_map           : (H_q, top_k) int64 — query-head → kv-head map
//   ts_buf             : (H_q, top_k) int64 — top-k slot indices for each q-head
//   k_out, v_out       : (H_q, top_k, D) bf16, CPU (pinned recommended)

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cstring>


static void gather_kv_topk(
    torch::Tensor k_shadow,
    torch::Tensor v_shadow,
    torch::Tensor head_map,
    torch::Tensor ts_buf,
    torch::Tensor k_out,
    torch::Tensor v_out)
{
    TORCH_CHECK(k_shadow.is_cpu() && v_shadow.is_cpu(),  "shadow tensors must be CPU");
    TORCH_CHECK(head_map.is_cpu() && ts_buf.is_cpu(),     "indices must be CPU");
    TORCH_CHECK(k_out.is_cpu()    && v_out.is_cpu(),      "outputs must be CPU");
    TORCH_CHECK(k_shadow.scalar_type() == torch::kBFloat16, "k_shadow bf16");
    TORCH_CHECK(v_shadow.scalar_type() == torch::kBFloat16, "v_shadow bf16");
    TORCH_CHECK(k_out.scalar_type()    == torch::kBFloat16, "k_out bf16");
    TORCH_CHECK(v_out.scalar_type()    == torch::kBFloat16, "v_out bf16");
    TORCH_CHECK(head_map.scalar_type() == torch::kInt64, "head_map int64");
    TORCH_CHECK(ts_buf.scalar_type()   == torch::kInt64, "ts_buf int64");
    TORCH_CHECK(k_shadow.dim() == 3 && v_shadow.dim() == 3,
                "shadow tensors must be (H_kv, num_slots, D)");
    TORCH_CHECK(head_map.sizes() == ts_buf.sizes(),
                "head_map and ts_buf must have the same shape");
    TORCH_CHECK(k_out.dim() == 3 && v_out.dim() == 3,
                "out tensors must be (H_q, top_k, D)");

    auto ks = k_shadow.contiguous();
    auto vs = v_shadow.contiguous();
    auto hm = head_map.contiguous();
    auto ts = ts_buf.contiguous();

    const int64_t H_kv      = ks.size(0);
    const int64_t num_slots = ks.size(1);
    const int64_t D         = ks.size(2);
    const int64_t H_q       = hm.size(0);
    const int64_t top_k     = hm.size(1);

    TORCH_CHECK(vs.size(0) == H_kv && vs.size(1) == num_slots && vs.size(2) == D,
                "v_shadow shape must match k_shadow");
    TORCH_CHECK(k_out.size(0) == H_q && k_out.size(1) == top_k && k_out.size(2) == D,
                "k_out shape mismatch");
    TORCH_CHECK(v_out.size(0) == H_q && v_out.size(1) == top_k && v_out.size(2) == D,
                "v_out shape mismatch");

    const at::BFloat16* k_src = ks.data_ptr<at::BFloat16>();
    const at::BFloat16* v_src = vs.data_ptr<at::BFloat16>();
    const int64_t*      hm_p  = hm.data_ptr<int64_t>();
    const int64_t*      ts_p  = ts.data_ptr<int64_t>();
    at::BFloat16*       k_dst = k_out.data_ptr<at::BFloat16>();
    at::BFloat16*       v_dst = v_out.data_ptr<at::BFloat16>();

    const int64_t total              = H_q * top_k;
    const int64_t kshadow_head_stride = num_slots * D;
    const int64_t row_bytes           = D * sizeof(at::BFloat16);

    // GRAIN tuned for D=128 → 256 B / row.  16 rows per task = 4 KB chunk,
    // small enough that every thread gets work but big enough to amortise
    // OpenMP thread-spawn overhead.
    constexpr int64_t GRAIN = 16;
    at::parallel_for(0, total, GRAIN, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t hkv  = hm_p[i];
            const int64_t slot = ts_p[i];
            const int64_t src_off = hkv * kshadow_head_stride + slot * D;
            std::memcpy(k_dst + i * D, k_src + src_off, row_bytes);
            std::memcpy(v_dst + i * D, v_src + src_off, row_bytes);
        }
    });
}


// Batched version: indices and outputs add a leading B dim.
//   head_map / ts_buf : (B, H_q, K) int64
//   k_out / v_out     : (B, H_q, K, D) bf16
// Same algorithm — at::parallel_for over B*H_q*K rows.
static void gather_kv_topk_batched(
    torch::Tensor k_shadow,
    torch::Tensor v_shadow,
    torch::Tensor head_map,
    torch::Tensor ts_buf,
    torch::Tensor k_out,
    torch::Tensor v_out)
{
    TORCH_CHECK(k_shadow.is_cpu() && v_shadow.is_cpu());
    TORCH_CHECK(head_map.is_cpu() && ts_buf.is_cpu());
    TORCH_CHECK(k_out.is_cpu() && v_out.is_cpu());
    TORCH_CHECK(k_shadow.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(v_shadow.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(head_map.scalar_type() == torch::kInt64);
    TORCH_CHECK(ts_buf.scalar_type()   == torch::kInt64);
    TORCH_CHECK(k_out.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(v_out.scalar_type()    == torch::kBFloat16);
    TORCH_CHECK(k_shadow.dim() == 3 && head_map.dim() == 3 && ts_buf.dim() == 3);
    TORCH_CHECK(k_out.dim() == 4 && v_out.dim() == 4);

    auto ks = k_shadow.contiguous();
    auto vs = v_shadow.contiguous();
    auto hm = head_map.contiguous();
    auto ts = ts_buf.contiguous();

    const int64_t H_kv      = ks.size(0);
    const int64_t num_slots = ks.size(1);
    const int64_t D         = ks.size(2);
    const int64_t B         = hm.size(0);
    const int64_t H_q       = hm.size(1);
    const int64_t K         = hm.size(2);
    const int64_t row_bytes = D * sizeof(at::BFloat16);
    const int64_t kshadow_head_stride = num_slots * D;

    TORCH_CHECK(vs.size(0) == H_kv && vs.size(1) == num_slots && vs.size(2) == D);
    TORCH_CHECK(k_out.size(0) == B && k_out.size(1) == H_q
                && k_out.size(2) == K && k_out.size(3) == D);
    TORCH_CHECK(v_out.size(0) == B && v_out.size(1) == H_q
                && v_out.size(2) == K && v_out.size(3) == D);

    const at::BFloat16* k_src = ks.data_ptr<at::BFloat16>();
    const at::BFloat16* v_src = vs.data_ptr<at::BFloat16>();
    const int64_t*      hm_p  = hm.data_ptr<int64_t>();
    const int64_t*      ts_p  = ts.data_ptr<int64_t>();
    at::BFloat16*       k_dst = k_out.data_ptr<at::BFloat16>();
    at::BFloat16*       v_dst = v_out.data_ptr<at::BFloat16>();

    const int64_t total = B * H_q * K;
    constexpr int64_t GRAIN = 16;
    at::parallel_for(0, total, GRAIN, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t hkv  = hm_p[i];
            const int64_t slot = ts_p[i];
            const int64_t src_off = hkv * kshadow_head_stride + slot * D;
            std::memcpy(k_dst + i * D, k_src + src_off, row_bytes);
            std::memcpy(v_dst + i * D, v_src + src_off, row_bytes);
        }
    });
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_kv_topk", &gather_kv_topk,
          py::call_guard<py::gil_scoped_release>(),
          "Parallel CPU shadow K/V gather; releases the GIL so a worker "
          "thread can run concurrently with main-thread CUDA ops.",
          py::arg("k_shadow"), py::arg("v_shadow"),
          py::arg("head_map"), py::arg("ts_buf"),
          py::arg("k_out"),    py::arg("v_out"));
    m.def("gather_kv_topk_batched", &gather_kv_topk_batched,
          py::call_guard<py::gil_scoped_release>(),
          "Batched CPU shadow K/V gather (B, H_q, K) — jetfire_int2 v2.",
          py::arg("k_shadow"), py::arg("v_shadow"),
          py::arg("head_map"), py::arg("ts_buf"),
          py::arg("k_out"),    py::arg("v_out"));
}
