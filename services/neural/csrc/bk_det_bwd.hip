// Berkelium deterministic rasterization backward (original Berkelium code, MIT). See bk_det_bwd.h.
//
// Per-pixel gradient math is the same as gsplat 1.5.3 rasterize_to_pixels_3dgs_bwd (Apache-2.0 / AMD MIT), which
// it must agree with numerically; the accumulation scheme (no atomics, fixed reduction order, per-intersection
// slots) is the Berkelium contribution. Builds unchanged with hipcc (ROCm) and with the HIP-CPU runtime.
#include "bk_det_bwd.h"

#include <cmath>

#if defined(__HIP_CPU_RT__) || !defined(__HIP_DEVICE_COMPILE__)
#define BK_EXP(x) expf(x)
#else
#define BK_EXP(x) __expf(x)   // same fast exponential as gsplat's kernel on the GPU
#endif

namespace bk {

constexpr float kAlphaThreshold = 1.f / 255.f;

template <int CDIM>
__global__ void __launch_bounds__(kBlock) pairs_kernel(BwdInputs in, float* __restrict__ pair_grads) {
    constexpr int G = pair_width(CDIM);
    __shared__ float xyo_batch[kBlock * 3];
    __shared__ float conic_batch[kBlock * 3];
    __shared__ float rgb_batch[kBlock * CDIM];
    __shared__ float red[G][kBlock];
    __shared__ int any_flag[3];
    __shared__ int32_t bin_max[kBlock];

    const int tile_id = blockIdx.y * in.tile_w + blockIdx.x;
    const int tr = threadIdx.y * kTile + threadIdx.x;
    const int i = blockIdx.y * kTile + threadIdx.y;
    const int j = blockIdx.x * kTile + threadIdx.x;
    const bool inside = (i < in.height) && (j < in.width);
    const int pix = inside ? i * in.width + j : 0;
    const float px = (float)j + 0.5f, py = (float)i + 0.5f;

    const int range_start = in.tile_offsets[tile_id];
    const int range_end = (tile_id == in.tile_w * in.tile_h - 1) ? in.n_isects : in.tile_offsets[tile_id + 1];
    const int num_batches = (range_end - range_start + kBlock - 1) / kBlock;

    const float T_final = inside ? 1.f - in.render_alphas[pix] : 1.f;
    float T = T_final;
    float buffer[CDIM];
    float v_c[CDIM];
    float bg_dot = 0.f;
    for (int k = 0; k < CDIM; ++k) {
        buffer[k] = 0.f;
        v_c[k] = inside ? in.v_render_colors[pix * CDIM + k] : 0.f;
        if (in.backgrounds != nullptr) bg_dot += in.backgrounds[k] * v_c[k];
    }
    const float v_a = (inside && in.v_render_alphas != nullptr) ? in.v_render_alphas[pix] : 0.f;
    const int bin_final = inside ? in.last_ids[pix] : -1;

    // Tile-wide last contributing intersection (deterministic integer max tree). Elements behind it are used by no
    // pixel of this tile, so they are skipped without barriers (same skip as gsplat's warp_bin_final, per block).
    bin_max[tr] = bin_final;
    if (tr == 0) { any_flag[0] = 0; any_flag[1] = 0; any_flag[2] = 0; }
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (tr < s && bin_max[tr + s] > bin_max[tr]) bin_max[tr] = bin_max[tr + s];
        __syncthreads();
    }
    const int tile_bin_final = bin_max[0];
    int e = 0;  // running element counter across batches (selects the rotating "any" flag)

    for (int b = 0; b < num_batches; ++b) {
        __syncthreads();
        const int batch_end = range_end - 1 - kBlock * b;
        const int remaining = batch_end + 1 - range_start;
        const int batch_size = remaining < kBlock ? remaining : kBlock;
        const int idx = batch_end - tr;
        if (idx >= range_start) {
            const int g = in.flatten_ids[idx];
            xyo_batch[3 * tr + 0] = in.means2d[2 * g + 0];
            xyo_batch[3 * tr + 1] = in.means2d[2 * g + 1];
            xyo_batch[3 * tr + 2] = in.opacities[g];
            conic_batch[3 * tr + 0] = in.conics[3 * g + 0];
            conic_batch[3 * tr + 1] = in.conics[3 * g + 1];
            conic_batch[3 * tr + 2] = in.conics[3 * g + 2];
            for (int k = 0; k < CDIM; ++k) rgb_batch[tr * CDIM + k] = in.colors[g * CDIM + k];
        }
        __syncthreads();

        const int t_first = batch_end - tile_bin_final > 0 ? batch_end - tile_bin_final : 0;   // block-uniform
        for (int t = t_first; t < batch_size; ++t, ++e) {
            const int gidx = batch_end - t;  // global intersection index of this element
            float lg[G];
            for (int k = 0; k < G; ++k) lg[k] = 0.f;
            bool valid = inside && (gidx <= bin_final);
            if (valid) {
                const float cx = conic_batch[3 * t + 0], cy = conic_batch[3 * t + 1], cz = conic_batch[3 * t + 2];
                const float opac = xyo_batch[3 * t + 2];
                const float dx = xyo_batch[3 * t + 0] - px, dy = xyo_batch[3 * t + 1] - py;
                const float sigma = 0.5f * (cx * dx * dx + cz * dy * dy) + cy * dx * dy;
                const float vis = BK_EXP(-sigma);
                const float alpha = fminf(0.999f, opac * vis);
                if (sigma < 0.f || alpha < kAlphaThreshold) {
                    valid = false;
                } else {
                    const float ra = 1.f / (1.f - alpha);
                    T *= ra;
                    const float fac = alpha * T;
                    float v_alpha = 0.f;
                    for (int k = 0; k < CDIM; ++k) {
                        lg[5 + k] = fac * v_c[k];
                        v_alpha += (rgb_batch[t * CDIM + k] * T - buffer[k] * ra) * v_c[k];
                    }
                    v_alpha += T_final * ra * v_a;
                    if (in.backgrounds != nullptr) v_alpha += -T_final * ra * bg_dot;
                    if (opac * vis <= 0.999f) {
                        const float v_sigma = -opac * vis * v_alpha;
                        lg[2] = 0.5f * v_sigma * dx * dx;
                        lg[3] = v_sigma * dx * dy;
                        lg[4] = 0.5f * v_sigma * dy * dy;
                        lg[0] = v_sigma * (cx * dx + cy * dy);
                        lg[1] = v_sigma * (cy * dx + cz * dy);
                        lg[6 + CDIM] = fabsf(lg[0]);
                        lg[7 + CDIM] = fabsf(lg[1]);
                        lg[5 + CDIM] = vis * v_alpha;
                    }
                    for (int k = 0; k < CDIM; ++k) buffer[k] += rgb_batch[t * CDIM + k] * fac;
                }
            }
            // Block-wide "any pixel uses this element" with a 3-slot rotating flag: slot (e+1)%3 is cleared
            // before this barrier, slot e%3 is written by valid lanes, read by all after the barrier.
            if (tr == 0) any_flag[(e + 1) % 3] = 0;
            if (valid) any_flag[e % 3] = 1;
            __syncthreads();
            if (any_flag[e % 3] == 0) continue;  // uniform branch: slot stays zero (pre-cleared workspace)

            for (int k = 0; k < G; ++k) red[k][tr] = lg[k];
            __syncthreads();
            for (int s = kBlock / 2; s > 0; s >>= 1) {  // fixed-shape tree => fixed summation order
                if (tr < s)
                    for (int k = 0; k < G; ++k) red[k][tr] += red[k][tr + s];
                __syncthreads();
            }
            if (tr == 0)
                for (int k = 0; k < G; ++k) pair_grads[(size_t)gidx * G + k] = red[k][0];
        }
    }
}

template <int CDIM>
__global__ void reduce_kernel(BwdInputs in, const float* __restrict__ pair_grads, BwdOutputs out) {
    constexpr int G = pair_width(CDIM);
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= in.N) return;
    float acc[G];
    for (int k = 0; k < G; ++k) acc[k] = 0.f;
    for (int64_t p = in.seg_offsets[g]; p < in.seg_offsets[g + 1]; ++p) {  // ascending intersection order
        const float* r = pair_grads + (size_t)in.pair_order[p] * G;
        for (int k = 0; k < G; ++k) acc[k] += r[k];
    }
    out.v_means2d[2 * g + 0] = acc[0];
    out.v_means2d[2 * g + 1] = acc[1];
    out.v_conics[3 * g + 0] = acc[2];
    out.v_conics[3 * g + 1] = acc[3];
    out.v_conics[3 * g + 2] = acc[4];
    for (int k = 0; k < CDIM; ++k) out.v_colors[g * CDIM + k] = acc[5 + k];
    out.v_opacities[g] = acc[5 + CDIM];
    if (out.v_means2d_abs != nullptr) {
        out.v_means2d_abs[2 * g + 0] = acc[6 + CDIM];
        out.v_means2d_abs[2 * g + 1] = acc[7 + CDIM];
    }
}

template <int CDIM>
static hipError_t launch(const BwdInputs& in, const BwdOutputs& out, hipStream_t stream) {
    const size_t pair_bytes = (size_t)in.n_isects * pair_width(CDIM) * sizeof(float);
    hipError_t err = pair_bytes ? hipMemsetAsync(out.pair_grads, 0, pair_bytes, stream) : hipSuccess;
    if (err != hipSuccess) return err;
    if (in.n_isects > 0) {
        hipLaunchKernelGGL(pairs_kernel<CDIM>, dim3(in.tile_w, in.tile_h, 1), dim3(kTile, kTile, 1), 0, stream, in,
                           out.pair_grads);
        if ((err = hipGetLastError()) != hipSuccess) return err;
    }
    if (in.N > 0) {
        hipLaunchKernelGGL(reduce_kernel<CDIM>, dim3((in.N + 255) / 256, 1, 1), dim3(256, 1, 1), 0, stream, in,
                           (const float*)out.pair_grads, out);
        err = hipGetLastError();
    }
    return err;
}

hipError_t rasterize_bwd_deterministic(const BwdInputs& in, const BwdOutputs& out, hipStream_t stream) {
    switch (in.cdim) {
        case 1: return launch<1>(in, out, stream);
        case 2: return launch<2>(in, out, stream);
        case 3: return launch<3>(in, out, stream);
        case 4: return launch<4>(in, out, stream);
        default: return hipErrorInvalidValue;
    }
}

}  // namespace bk
