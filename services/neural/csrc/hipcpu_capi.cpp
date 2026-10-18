// C ABI around the Berkelium HIP kernels for the HIP-CPU runtime (CPU execution of the UNMODIFIED .hip source).
// Used only for CPU validation; it proves the kernel logic, NOT GPU execution. Original Berkelium code (MIT).
#include "bk_det_bwd.h"

extern "C" int bk_rasterize_bwd_deterministic(
    int N, int n_isects, int width, int height, int tile_w, int tile_h, int cdim,
    const float* means2d, const float* conics, const float* colors, const float* opacities, const float* backgrounds,
    const int32_t* tile_offsets, const int32_t* flatten_ids, const float* render_alphas, const int32_t* last_ids,
    const float* v_render_colors, const float* v_render_alphas, const int32_t* pair_order, const int64_t* seg_offsets,
    float* pair_grads, float* v_means2d, float* v_conics, float* v_colors, float* v_opacities, float* v_means2d_abs) {
    bk::BwdInputs in{N, n_isects, width, height, tile_w, tile_h, cdim, means2d, conics, colors, opacities, backgrounds,
                     tile_offsets, flatten_ids, render_alphas, last_ids, v_render_colors, v_render_alphas, pair_order,
                     seg_offsets};
    bk::BwdOutputs out{pair_grads, v_means2d, v_conics, v_colors, v_opacities, v_means2d_abs};
    hipError_t e = bk::rasterize_bwd_deterministic(in, out, nullptr);
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}
