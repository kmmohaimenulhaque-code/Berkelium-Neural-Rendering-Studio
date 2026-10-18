// PyTorch (ROCm) binding for the Berkelium deterministic rasterization backward. Original Berkelium code (MIT).
// Built only by torch.utils.cpp_extension on a ROCm PyTorch (torch.version.hip set). Written with the CUDA-named
// ATen API that PyTorch's hipify maps to HIP; the kernel source itself is native HIP.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "bk_det_bwd.h"

#define BK_CHECK(x, ...) TORCH_CHECK(x, "berkelium_det_bwd: ", __VA_ARGS__)

static void check_f32(const at::Tensor& t, const char* name) {
    BK_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous(), name,
             " must be a contiguous float32 tensor on the GPU");
}
static void check_i32(const at::Tensor& t, const char* name) {
    BK_CHECK(t.is_cuda() && t.scalar_type() == at::kInt && t.is_contiguous(), name,
             " must be a contiguous int32 tensor on the GPU");
}

// Returns {v_means2d [N,2], v_conics [N,3], v_colors [N,C], v_opacities [N], v_means2d_abs [N,2] or empty}.
std::vector<at::Tensor> rasterize_bwd(const at::Tensor& means2d, const at::Tensor& conics, const at::Tensor& colors,
                                      const at::Tensor& opacities, const c10::optional<at::Tensor>& backgrounds,
                                      int64_t width, int64_t height, int64_t tile_size,
                                      const at::Tensor& tile_offsets, const at::Tensor& flatten_ids,
                                      const at::Tensor& render_alphas, const at::Tensor& last_ids,
                                      const at::Tensor& v_render_colors,
                                      const c10::optional<at::Tensor>& v_render_alphas, bool absgrad) {
    const c10::OptionalDeviceGuard device_guard(at::device_of(means2d));   // launch on the tensors' device
    BK_CHECK(tile_size == bk::kTile, "only tile_size == 16 is supported");
    BK_CHECK(means2d.dim() == 2 && means2d.size(1) == 2, "packed / batched layouts are UNSUPPORTED_FEATURE");
    check_f32(means2d, "means2d"); check_f32(conics, "conics"); check_f32(colors, "colors");
    check_f32(opacities, "opacities"); check_f32(render_alphas, "render_alphas");
    check_f32(v_render_colors, "v_render_colors");
    check_i32(tile_offsets, "tile_offsets"); check_i32(flatten_ids, "flatten_ids"); check_i32(last_ids, "last_ids");
    const int64_t N = means2d.size(0), M = flatten_ids.numel(), C = colors.size(-1);
    const int64_t P = width * height, tw = (width + bk::kTile - 1) / bk::kTile, th = (height + bk::kTile - 1) / bk::kTile;
    BK_CHECK(C >= 1 && C <= 4, "colors channels must be 1..4");
    BK_CHECK(M <= INT32_MAX && N <= INT32_MAX, "intersection / Gaussian count exceeds int32 (RESOURCE_LIMIT)");
    BK_CHECK(tile_offsets.dim() == 2 && tile_offsets.size(0) == th && tile_offsets.size(1) == tw,
             "tile_offsets must be [ceil(H/16), ceil(W/16)]");
    BK_CHECK(conics.numel() == 3 * N && colors.numel() == C * N && opacities.numel() == N, "per-Gaussian shapes");
    BK_CHECK(render_alphas.numel() == P && last_ids.numel() == P && v_render_colors.numel() == C * P,
             "per-pixel shapes must match width*height");
    const auto dev = means2d.device();
    for (const at::Tensor* t : {&conics, &colors, &opacities, &tile_offsets, &flatten_ids, &render_alphas, &last_ids,
                                &v_render_colors})
        BK_CHECK(t->device() == dev, "all tensors must be on the same device");
    if (backgrounds.has_value()) {
        check_f32(*backgrounds, "backgrounds");
        BK_CHECK(backgrounds->numel() == C && backgrounds->device() == dev, "backgrounds must be [C] on the device");
    }
    if (v_render_alphas.has_value()) {
        check_f32(*v_render_alphas, "v_render_alphas");
        BK_CHECK(v_render_alphas->numel() == P && v_render_alphas->device() == dev, "v_render_alphas shape/device");
    }

    // Deterministic per-Gaussian segments: integer counts + stable sort are order-independent.
    auto order = at::argsort(flatten_ids.to(at::kLong), /*stable=*/true, /*dim=*/0, /*descending=*/false)
                     .to(at::kInt).contiguous();
    auto counts = at::bincount(flatten_ids.to(at::kLong), {}, N);
    auto seg = at::cat({at::zeros({1}, counts.options()), at::cumsum(counts, 0)}).to(at::kLong).contiguous();

    auto f = means2d.options();
    auto pair_grads = at::empty({M * bk::pair_width((int)C)}, f);
    auto v_means2d = at::empty({N, 2}, f), v_conics = at::empty({N, 3}, f), v_colors = at::empty({N, C}, f);
    auto v_opac = at::empty({N}, f);
    at::Tensor v_abs = absgrad ? at::empty({N, 2}, f) : at::Tensor();

    bk::BwdInputs in{(int)N, (int)M, (int)width, (int)height, (int)tile_offsets.size(1), (int)tile_offsets.size(0),
                     (int)C, means2d.data_ptr<float>(), conics.data_ptr<float>(), colors.data_ptr<float>(),
                     opacities.data_ptr<float>(), backgrounds.has_value() ? backgrounds->data_ptr<float>() : nullptr,
                     tile_offsets.data_ptr<int32_t>(), flatten_ids.data_ptr<int32_t>(),
                     render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(), v_render_colors.data_ptr<float>(),
                     v_render_alphas.has_value() ? v_render_alphas->data_ptr<float>() : nullptr,
                     order.data_ptr<int32_t>(), seg.data_ptr<int64_t>()};
    bk::BwdOutputs out{pair_grads.data_ptr<float>(), v_means2d.data_ptr<float>(), v_conics.data_ptr<float>(),
                       v_colors.data_ptr<float>(), v_opac.data_ptr<float>(),
                       absgrad ? v_abs.data_ptr<float>() : nullptr};
    auto err = bk::rasterize_bwd_deterministic(in, out, at::cuda::getCurrentCUDAStream());
    BK_CHECK(err == hipSuccess, "kernel launch failed: ", hipGetErrorString(err));
    return {v_means2d, v_conics, v_colors, v_opac, v_abs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_bwd", &rasterize_bwd, "Berkelium deterministic rasterize_to_pixels backward (HIP)");
}
