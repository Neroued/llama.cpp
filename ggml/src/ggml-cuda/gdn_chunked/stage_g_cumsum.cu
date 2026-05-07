// Stage 1: chunk-local cumsum of the gate.
//
// Math + I/O layouts: see chunked_gdn.cuh::stages::g_cumsum_config.
// Block / smem total: H_TILE x BT/2 threads, smem (BT/2)*(H_TILE+1) fp32.
// Tuning history, ncu numbers, sync audit, footguns:
//   chunked/PERF_LOG.md#stage-1-g_cumsum-stage_g_cumsumcu

#include "chunked_gdn.cuh"
#include "chunked_internals.cuh"
#include "chunked_utils.cuh"

#include <cstdio>

namespace chunked_gdn::stages {

namespace {

using chunked_gdn::detail::BT;

static_assert(chunked_gdn::kChunkSize == 64,
              "stage_g_cumsum: kChunkSize must be 64 (kernel hard-codes "
              "BT=64 = 2 * WARP_SIZE)");
constexpr int BT_HALF = BT / 2;  // 32 t-slots, 2 t per thread

static_assert(BT_HALF == gdn::WARP_SIZE,
              "stage_g_cumsum: kernel's cross-y scan assumes BT_HALF == 32");

template <int H_TILE>
__global__ void g_cumsum_gdn_kernel(const float * __restrict__ g_in,
                                    float       * __restrict__ g_out,
                                    int64_t T,
                                    int64_t T_total,
                                    int64_t H_v) {
    static_assert(H_TILE >= 4 && H_TILE <= 32,
                  "stage_g_cumsum: H_TILE must be in [4, 32]");
    static_assert(H_TILE * BT_HALF % gdn::WARP_SIZE == 0,
                  "stage_g_cumsum: block must split into whole warps");

    // +1 padding makes Phase C column reads conflict-free (gcd(H_TILE+1, 32)=1
    // for H_TILE in {4,8,16,32}).
    constexpr int SMEM_STRIDE = H_TILE + 1;
    __shared__ float smem[BT_HALF][SMEM_STRIDE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int chunk      = blockIdx.x;
    const int b          = blockIdx.y;
    const int h_tile_idx = blockIdx.z;
    const int h_v_off    = h_tile_idx * H_TILE + tx;

    const auto cb = gdn::chunk_bounds_t::of(chunk, T, BT);
    const int64_t cs = cb.cs;
    const int     cl = cb.cl;

    const int t0 = 2 * ty;
    const int t1 = t0 + 1;

    const bool valid_h = (h_v_off < H_v);

    // === Phase A: coalesced load (32 lanes x 4 B = 128 B sector) ===
    // g_in is the user-provided gate buffer with batch stride = T_total * H_v;
    // g_out is workspace allocated for [B, T, H_v] (T = chunked length, may be
    // smaller than T_total when chunked covers only the head of a longer
    // buffer in split dispatch).
    float a = 0.0f, bv = 0.0f;
    if (valid_h) {
        const int64_t in_base = ((int64_t) b * T_total + cs) * H_v + h_v_off;
        if (t0 < cl) a  = g_in[in_base + (int64_t) t0 * H_v];
        if (t1 < cl) bv = g_in[in_base + (int64_t) t1 * H_v];
    }

    // === Phase B: per-thread 2-element sum -> partial in smem ===
    smem[ty][tx] = a + bv;
    __syncthreads();

    // === Phase C: cross-y exclusive scan, one warp per tx column ===
    //
    // Linear thread id is laid out row-major (ty * H_TILE + tx). With
    // blockDim = (H_TILE, 32), the block contains exactly H_TILE warps, so
    // mapping `warp_id -> tx column` and `lane -> ty row` is well-defined
    // for every supported H_TILE.
    const int lid     = ty * H_TILE + tx;
    const int warp_id = lid >> 5;
    const int lane    = lid & 31;
    {
        float v = smem[lane][warp_id];

        // Hillis-Steele inclusive scan.
        #pragma unroll
        for (int offset = 1; offset < gdn::WARP_SIZE; offset <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, v, offset);
            if (lane >= offset) v += n;
        }

        // Inclusive -> exclusive: shift down by one lane; lane 0's prefix is 0.
        const float prev = __shfl_up_sync(0xffffffffu, v, 1);
        const float prefix_above = (lane == 0) ? 0.0f : prev;

        smem[lane][warp_id] = prefix_above;
    }
    __syncthreads();

    // === Phase D: apply prefix + coalesced store ===
    if (valid_h) {
        const float pre = smem[ty][tx];
        const int64_t out_base = ((int64_t) b * T + cs) * H_v + h_v_off;
        if (t0 < cl) g_out[out_base + (int64_t) t0 * H_v] = pre + a;
        if (t1 < cl) g_out[out_base + (int64_t) t1 * H_v] = pre + a + bv;
    }
}

template <int H_TILE>
cudaError_t launch_typed(const g_cumsum_config & cfg, dim3 grid) {
    const dim3 block(H_TILE, BT_HALF, 1);
    const int64_t L_total = (cfg.L_total != 0) ? cfg.L_total : cfg.L;
    g_cumsum_gdn_kernel<H_TILE><<<grid, block, 0, cfg.stream>>>(
        cfg.g_in, cfg.g_out, cfg.L, L_total, cfg.H_v);
    return cudaGetLastError();
}

// Pick the largest H_TILE in {32, 16, 8, 4} that divides H_v; coalesced load
// efficiency drops 2x at every step below WARP_SIZE.
inline int pick_h_tile(int64_t H_v) {
    if ((H_v & 31) == 0) return 32;
    if ((H_v & 15) == 0) return 16;
    if ((H_v & 7)  == 0) return 8;
    if ((H_v & 3)  == 0) return 4;
    return 0;
}

}  // namespace

cudaError_t launch_g_cumsum(const g_cumsum_config & cfg) {
    detail::stage_validator v{"launch_g_cumsum",
                              cfg.S, /*H_qk=*/0, cfg.H_v, cfg.L, cfg.B,
                              cfg.kda, /*require_h_qk=*/false};
    GDN_PROPAGATE(v.check_shape());
    GDN_PROPAGATE(v.check_gdn_full_chunks());
    if (cfg.g_in == nullptr || cfg.g_out == nullptr) return cudaErrorInvalidValue;
    if (cfg.kda) return cudaErrorNotYetImplemented;  // Phase C

    const int H_TILE = pick_h_tile(cfg.H_v);
    if (H_TILE == 0) {
        std::fprintf(stderr,
                     "launch_g_cumsum: H_v=%lld is not divisible by 4 (smallest "
                     "supported H_TILE is 4)\n", (long long) cfg.H_v);
        return cudaErrorInvalidValue;
    }

    const int64_t NT      = (cfg.L + BT - 1) / BT;
    const int64_t n_tiles = (cfg.H_v + H_TILE - 1) / H_TILE;
    GDN_PROPAGATE(v.check_grid(NT, cfg.B, n_tiles));

    const dim3 grid((unsigned) NT, (unsigned) cfg.B, (unsigned) n_tiles);

    switch (H_TILE) {
        case 32: return launch_typed<32>(cfg, grid);
        case 16: return launch_typed<16>(cfg, grid);
        case 8:  return launch_typed< 8>(cfg, grid);
        case 4:  return launch_typed< 4>(cfg, grid);
        default: return cudaErrorInvalidValue;  // pick_h_tile already filtered
    }
}

}  // namespace chunked_gdn::stages
