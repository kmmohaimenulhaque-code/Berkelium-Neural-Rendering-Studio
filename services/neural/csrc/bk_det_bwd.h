// Berkelium deterministic rasterization backward — launcher API (original Berkelium code, MIT).
//
// Replaces the atomicAdd gradient accumulation of gsplat's rasterize_to_pixels_3dgs_bwd with a two-stage,
// atomic-free scheme:
//   stage 1  one block per 16x16 tile; for each Gaussian/tile intersection, the 256 per-pixel gradient terms are
//            summed with a fixed-shape shared-memory tree and written to that intersection's OWN slot
//            (pair_grads[isect]) — exactly one writer per slot.
//   stage 2  one thread per Gaussian sums its intersection slots in ascending intersection order
//            (pair_order = stable argsort of flatten_ids).
// Same build + device + inputs  =>  bitwise-identical gradients. Math follows gsplat 1.5.3 (profile gsplat_v1).
//
// All pointers are device pointers (host pointers under the HIP-CPU runtime). Layouts follow gsplat (packed=false,
// one image): means2d [N,2], conics [N,3], colors [N,CDIM], opacities [N], tile_offsets [tile_h,tile_w],
// flatten_ids [n_isects], render_alphas/last_ids/v_render_alphas [H,W], v_render_colors [H,W,CDIM].
#pragma once
#include <cstdint>

#include <hip/hip_runtime.h>

namespace bk {

constexpr int kTile = 16;          // tile edge; block = kTile x kTile threads
constexpr int kBlock = kTile * kTile;

// gradient record per intersection: xy(2) conic(3) color(CDIM) opacity(1) |xy|(2)
constexpr int pair_width(int cdim) { return 2 + 3 + cdim + 1 + 2; }

struct BwdInputs {
    int N, n_isects, width, height, tile_w, tile_h, cdim;
    const float* means2d;
    const float* conics;
    const float* colors;
    const float* opacities;
    const float* backgrounds;      // [CDIM] or nullptr
    const int32_t* tile_offsets;
    const int32_t* flatten_ids;
    const float* render_alphas;
    const int32_t* last_ids;
    const float* v_render_colors;
    const float* v_render_alphas;  // may be nullptr (treated as zero)
    const int32_t* pair_order;     // [n_isects] stable argsort of flatten_ids
    const int64_t* seg_offsets;    // [N+1]  segment of Gaussian g in pair_order: [seg[g], seg[g+1])
};

struct BwdOutputs {
    float* pair_grads;     // workspace [n_isects * pair_width(cdim)] (zeroed by the launcher)
    float* v_means2d;      // [N,2]  (overwritten)
    float* v_conics;       // [N,3]
    float* v_colors;       // [N,CDIM]
    float* v_opacities;    // [N]
    float* v_means2d_abs;  // [N,2] or nullptr
};

// Returns hipSuccess or the first HIP error. cdim must be 1..4 (3 = RGB, 4 = RGB + expected depth).
hipError_t rasterize_bwd_deterministic(const BwdInputs& in, const BwdOutputs& out, hipStream_t stream);

}  // namespace bk
