// Stage 3: chunk-sequential state passing.
//
// Math + I/O layouts: see chunked_gdn.cuh::stages::state_passing_config.
// Block / smem total: 8 warps (256 t) at S=128 with __launch_bounds__(256, 2);
//                     ~33 KB smem (W_half + UVD alias + k_half + snap + g).
// Tuning history (V1->V4), ncu numbers, sync audit, SnapView swizzle,
// cross-chunk prefetch placement, two-commit-group design:
//   chunked/PERF_LOG.md#stage-3-state_passing-stage_state_passingcu

#include "chunked_gdn.cuh"
#include "chunked_internals.cuh"
#include "chunked_utils.cuh"

#include <cstdio>

namespace chunked_gdn::stages {

namespace {

using chunked_gdn::detail::BT;
using chunked_gdn::detail::MMA_M;
using chunked_gdn::detail::MMA_N;
using chunked_gdn::detail::MMA_K;
using chunked_gdn::detail::bh_decode_t;
using chunked_gdn::detail::zero_frag;
using gdn::SmemTile;
using gdn::mma_m16n8k8_tf32;
using gdn::async_copy_commit;
using gdn::async_copy_wait;
using gdn::async_copy_wait_all;
using gdn::exp2_fast;
using gdn::RCP_LN2_F;

static_assert(chunked_gdn::kChunkSize == 64,
              "stage_state_passing: kChunkSize must be 64");

// ---------------------------------------------------------------------------
// Per-S kernel dimensions for the d-strip + BT-split design.
//
//   N_STRIP_PER_BLOCK  = min(S, 16)         d-cols owned by a block
//   D_STRIPS           = S / N_STRIP_PER_BLOCK
//   DT_TILES_PER_BLOCK = N_STRIP_PER_BLOCK / MMA_N        (1 or 2)
//   BT_SPLITS          = chosen so total warps <= 8
//   N_WARPS            = DT_TILES_PER_BLOCK * BT_SPLITS
//   THREADS            = N_WARPS * 32
//   BT_PER_WARP        = BT / BT_SPLITS                   (matmul1 M-rows)
//   S_PER_WARP         = S  / BT_SPLITS                   (matmul2 M-rows)
//   M_TILES_MM1_PW     = BT_PER_WARP / MMA_M
//   M_TILES_H_PW       = S_PER_WARP  / MMA_M
//   M_TILES_H_GLOB     = S / MMA_M
//   K_TILES_MM2_HALF   = (BT / 2) / MMA_K                 (matmul2 per k-half)
//   W_HALVES           = how many K-axis halves W is loaded in
// ---------------------------------------------------------------------------
template <int S> struct kernel_dims;

template <> struct kernel_dims<128> {
    static constexpr int N_STRIP_PER_BLOCK   = 16;
    static constexpr int D_STRIPS            = 8;
    static constexpr int DT_TILES_PER_BLOCK  = 2;
    static constexpr int BT_SPLITS           = 4;
    static constexpr int N_WARPS             = DT_TILES_PER_BLOCK * BT_SPLITS;  // 8
    static constexpr int THREADS             = N_WARPS * gdn::WARP_SIZE;
    static constexpr int W_HALVES            = 2;
};

template <> struct kernel_dims<64> {
    static constexpr int N_STRIP_PER_BLOCK   = 16;
    static constexpr int D_STRIPS            = 4;
    static constexpr int DT_TILES_PER_BLOCK  = 2;
    static constexpr int BT_SPLITS           = 4;
    static constexpr int N_WARPS             = DT_TILES_PER_BLOCK * BT_SPLITS;
    static constexpr int THREADS             = N_WARPS * gdn::WARP_SIZE;
    static constexpr int W_HALVES            = 2;
};

template <> struct kernel_dims<32> {
    static constexpr int N_STRIP_PER_BLOCK   = 16;
    static constexpr int D_STRIPS            = 2;
    static constexpr int DT_TILES_PER_BLOCK  = 2;
    static constexpr int BT_SPLITS           = 2;
    static constexpr int N_WARPS             = DT_TILES_PER_BLOCK * BT_SPLITS;
    static constexpr int THREADS             = N_WARPS * gdn::WARP_SIZE;
    static constexpr int W_HALVES            = 2;
};

template <> struct kernel_dims<16> {
    static constexpr int N_STRIP_PER_BLOCK   = 16;
    static constexpr int D_STRIPS            = 1;
    static constexpr int DT_TILES_PER_BLOCK  = 2;
    static constexpr int BT_SPLITS           = 1;
    static constexpr int N_WARPS             = DT_TILES_PER_BLOCK * BT_SPLITS;
    static constexpr int THREADS             = N_WARPS * gdn::WARP_SIZE;
    // S=16 W half stride would be 8 -- SmemTile requires >= 16. Load full W.
    static constexpr int W_HALVES            = 1;
};

// Smem floats per block, derived purely from kernel_dims<S>.
template <int S>
struct smem_layout {
    using D = kernel_dims<S>;
    static constexpr int W_STRIDE      = S / D::W_HALVES;
    static constexpr int W_HALF_FLT    = BT * W_STRIDE;
    static constexpr int UVD_FLT       = BT * D::N_STRIP_PER_BLOCK;   // U-vd alias
    static constexpr int K_HALF_ROWS   = BT / 2;
    static constexpr int K_HALF_FLT    = K_HALF_ROWS * S;
    static constexpr int M_TILES_H_PW  = (S / D::BT_SPLITS) / MMA_M;
    static constexpr int SNAP_K_ROWS   = MMA_M * M_TILES_H_PW;
    static constexpr int SNAP_FLT      = SNAP_K_ROWS * D::N_STRIP_PER_BLOCK;
    static constexpr int M_TILES_H_GLOB= S / MMA_M;
    static constexpr int N_SNAP_ITERS  = M_TILES_H_GLOB / M_TILES_H_PW;
    static constexpr int N_SNAP_BUF    = (N_SNAP_ITERS > 1) ? 2 : 1;
    static constexpr int SMEM_FLOATS   = W_HALF_FLT + UVD_FLT + K_HALF_FLT
                                       + SNAP_FLT * N_SNAP_BUF + BT;
};

// ---------------------------------------------------------------------------
// SnapView<STRIDE>: custom swizzle for snap_smem (transposed h[d][k] layout).
// Different from SmemTile<32>'s default swizzle to keep the 4-d-row scatter
// conflict-free. See PERF_LOG.md for the full bank-conflict analysis; do
// NOT merge with gdn::SmemTile.
// ---------------------------------------------------------------------------
template <int STRIDE>
struct SnapView {
    float * __restrict__ base;
    static_assert(STRIDE == 32 || STRIDE == 16,
                  "SnapView: only STRIDE in {16, 32} supported");

    __device__ __forceinline__ int swz_xor(int row) const {
        if constexpr (STRIDE == 32) {
            return (row & 7) << 2;        // {0, 4, 8, ..., 28}
        } else {
            return ((row >> 1) & 3) << 2; // fall back to default for STRIDE=16
        }
    }

    __device__ __forceinline__ float & at(int row, int col) const {
        return base[row * STRIDE + (col ^ swz_xor(row))];
    }

    __device__ __forceinline__ float4 & vec4_at(int row, int col) const {
        return *reinterpret_cast<float4 *>(
            &base[row * STRIDE + (col ^ swz_xor(row))]);
    }
};

// __launch_bounds__: 256 threads + min_blocks=2 -> 128 reg cap; min_blocks=3
// costs spills at the current grid (1.5 blocks/SM). See PERF_LOG.md.
template <int S>
__launch_bounds__(kernel_dims<S>::THREADS, 2)
__global__ void state_passing_gdn_kernel(const float * __restrict__ W_in,
                                         const float * __restrict__ U_in,
                                         const float * __restrict__ k_in,
                                         const float * __restrict__ g_cumsum,
                                         const float * __restrict__ state_in,
                                         float       * __restrict__ v_new,
                                         float       * __restrict__ h_chunk,
                                         float       * __restrict__ state_out,
                                         int64_t T,
                                         int64_t T_total,
                                         int64_t H_v,
                                         gdn::head_map qk_map,
                                         // Token-axis stride for k (in floats).
                                         // Caller passes the materialised value
                                         // (launcher handles 0 -> H_qk * S).
                                         int64_t k_stride_t,
                                         int     NT) {
    using D = kernel_dims<S>;
    using L = smem_layout<S>;
    constexpr int N_STRIP_PER_BLOCK = D::N_STRIP_PER_BLOCK;
    constexpr int D_STRIPS          = D::D_STRIPS;
    constexpr int BT_SPLITS         = D::BT_SPLITS;
    constexpr int THREADS_K         = D::THREADS;
    constexpr int W_HALVES          = D::W_HALVES;
    constexpr int W_STRIDE          = L::W_STRIDE;

    constexpr int BT_PER_WARP      = BT / BT_SPLITS;
    constexpr int S_PER_WARP       = S  / BT_SPLITS;
    constexpr int M_TILES_MM1_PW   = BT_PER_WARP / MMA_M;
    constexpr int M_TILES_H_PW     = S_PER_WARP  / MMA_M;
    constexpr int M_TILES_H_GLOB   = S / MMA_M;
    constexpr int K_TILES_MM2_HALF = (BT / 2) / MMA_K;

    static_assert(M_TILES_H_PW   >= 1, "S_PER_WARP must yield >= 1 M-tile per warp");
    static_assert(M_TILES_MM1_PW >= 1, "BT_PER_WARP must yield >= 1 M-tile per warp");
    static_assert(M_TILES_H_GLOB == BT_SPLITS * M_TILES_H_PW,
                  "BT_SPLITS partition of S must be exact");
    static_assert(W_STRIDE >= 16, "SmemTile<W_STRIDE> requires stride >= 16");

    constexpr int SNAP_K_ROWS    = L::SNAP_K_ROWS;
    constexpr int N_SNAP_ITERS   = L::N_SNAP_ITERS;
    constexpr int N_SNAP_HALF    = N_SNAP_ITERS / W_HALVES;
    constexpr int K_TILES_PER_SNAP_ITER = SNAP_K_ROWS / MMA_K;

    static_assert(N_SNAP_ITERS == BT_SPLITS, "design assumes one snap iter per s_idx");
    static_assert(N_SNAP_HALF >= 1, "W_HALVES must divide N_SNAP_ITERS");

    constexpr int W_HALF_FLT     = L::W_HALF_FLT;
    constexpr int UVD_FLT        = L::UVD_FLT;
    constexpr int K_HALF_ROWS    = L::K_HALF_ROWS;
    constexpr int K_HALF_FLT     = L::K_HALF_FLT;
    constexpr int SNAP_FLT       = L::SNAP_FLT;
    constexpr int N_SNAP_BUF     = L::N_SNAP_BUF;

    // Smem partition. U and vd alias (`uvd_smem`) -- disjoint phases. snap is
    // double-buffered when N_SNAP_ITERS > 1 so iter i+1 scatter overlaps
    // iter i's K-tiles without an extra sync. See PERF_LOG.md.
    extern __shared__ float smem[];
    float * const W_smem    = smem;                              // W_HALF_FLT
    float * const uvd_smem  = W_smem + W_HALF_FLT;               // UVD_FLT (U & vd alias)
    float * const k_smem    = uvd_smem + UVD_FLT;                // K_HALF_FLT
    float * const snap_smem = k_smem + K_HALF_FLT;               // SNAP_FLT * N_SNAP_BUF
    float * const g_smem    = snap_smem + SNAP_FLT * N_SNAP_BUF; // BT

    SmemTile<W_STRIDE>           W_view {W_smem};
    SmemTile<N_STRIP_PER_BLOCK>  vd_view{uvd_smem};
    SmemTile<S>                  k_view {k_smem};
    SmemTile<N_STRIP_PER_BLOCK>  U_view {uvd_smem};
    SnapView<SNAP_K_ROWS>        snap_views[2] = {
        SnapView<SNAP_K_ROWS>{snap_smem},
        SnapView<SNAP_K_ROWS>{snap_smem + (N_SNAP_BUF >= 2 ? SNAP_FLT : 0)}
    };

    // Block / lane indexing.
    //   grid.x = bhd in [0, B*H_v*D_STRIPS).
    //   warp = dt_idx * BT_SPLITS + s_idx.
    const int tid = threadIdx.x;
    const auto lanes = gdn::mma_lane_t::decode(tid);
    const int warp   = lanes.warp;
    const int lane_g = lanes.lane_g;
    const int lane_t = lanes.lane_t;

    const int s_idx        = warp % BT_SPLITS;
    const int dt_idx       = warp / BT_SPLITS;
    const int warp_d_local = dt_idx * MMA_N;

    const int bhd        = blockIdx.x;
    const int chain_idx  = bhd / D_STRIPS;
    const int strip_idx  = bhd - chain_idx * D_STRIPS;
    const auto bh        = bh_decode_t::of(chain_idx, (int) H_v);
    const int b          = bh.b;
    const int h_v        = bh.h_v;
    const int d_off      = strip_idx * N_STRIP_PER_BLOCK;
    const int warp_d_global = d_off + warp_d_local;

    // === Phase 0: load state_in (AR-transposed) -> per-warp h_frag ===
    float h_frag[M_TILES_H_PW][4];
    {
        const int64_t st_base = ((int64_t) b * H_v + h_v) * S * S;
        const int     row_off = s_idx * S_PER_WARP;
        #pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
            const int row_g0 = row_off + m * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_global + 2 * lane_t;
            const int col_d1 = col_d0 + 1;
            h_frag[m][0] = __ldg(state_in + st_base + (int64_t) col_d0 * S + row_g0);
            h_frag[m][1] = __ldg(state_in + st_base + (int64_t) col_d1 * S + row_g0);
            h_frag[m][2] = __ldg(state_in + st_base + (int64_t) col_d0 * S + row_g1);
            h_frag[m][3] = __ldg(state_in + st_base + (int64_t) col_d1 * S + row_g1);
        }
    }

    // Cross-chunk cp.async prefetch. Chunk 0's W_half_0 + U + k_half_0 issued
    // BEFORE the loop; subsequent chunks issue at end of B / E.
    const int64_t W_stride = (int64_t) H_v * S;
    const int64_t k_stride = k_stride_t;
    auto chunk_W_base = [&](int chunk_idx) -> int64_t {
        // W / U / v_new / g_cumsum / h_chunk live in workspace sized to T.
        return ((int64_t) b * T + (int64_t) chunk_idx * BT) * H_v  * S + (int64_t) h_v  * S;
    };
    auto chunk_k_base = [&](int chunk_idx) -> int64_t {
        // k_in is the user buffer with batch stride T_total * k_stride_t.
        return ((int64_t) b * T_total + (int64_t) chunk_idx * BT) * k_stride_t
             + (int64_t) qk_map.qk_head(h_v) * S;
    };

    // Two cp.async commit groups per chunk-boundary issue:
    //   Group A (early) = W_half_0 + U   -- consumed in Phase B / C
    //   Group B (late)  = k_half_0       -- consumed in Phase E1
    // Phase A drains group A only; group B is drained just before matmul2.
    // See PERF_LOG.md for the timing rationale.
    {
        const auto cb0 = gdn::chunk_bounds_t::of(0, T, BT);
        const int  cl0 = cb0.cl;
        gdn::issue_async_load_vec4<BT, W_STRIDE, THREADS_K>(
            W_view, W_in + chunk_W_base(0), W_stride, cl0, tid);
        gdn::issue_async_load_vec4<BT, N_STRIP_PER_BLOCK, THREADS_K>(
            U_view, U_in + chunk_W_base(0) + d_off, W_stride, cl0, tid);
        async_copy_commit();
        const int cl_kh0_0 = (cl0 < K_HALF_ROWS) ? cl0 : K_HALF_ROWS;
        gdn::issue_async_load_vec4<K_HALF_ROWS, S, THREADS_K>(
            k_view, k_in + chunk_k_base(0), k_stride, cl_kh0_0, tid);
        async_copy_commit();
    }

    // === Main chunk loop ===
    for (int chunk = 0; chunk < NT; ++chunk) {
        const auto cb = gdn::chunk_bounds_t::of(chunk, T, BT);
        const int64_t cs = cb.cs;
        const int     cl = cb.cl;

        const int64_t W_base = chunk_W_base(chunk);
        const int64_t k_base = chunk_k_base(chunk);

        // === Phase A: drain early group (W + U); k stays in flight ===
        if (tid < BT) {
            float val = 0.0f;
            if (tid < cl) {
                val = g_cumsum[((int64_t) b * T + cs + tid) * H_v + h_v];
            }
            g_smem[tid] = val;
        }

        async_copy_wait<1>();   // wait for W + U; k_half_0 keeps loading
        __syncthreads();

        // === Phase B: matmul1 (M = vnew tiles) over W_HALVES K-axis halves ===
        //
        // Snap is double-buffered: iter i+1 scatters snap[(i+1)%2] while
        // iter i's K-tiles read snap[i%2] -- no inter-iter sync needed.
        float vnew_frag[M_TILES_MM1_PW][4];
        zero_frag(vnew_frag);

        const int64_t hc_base = (((int64_t) b * NT + chunk) * H_v + h_v) * S * S;

        #pragma unroll 1
        for (int half = 0; half < W_HALVES; ++half) {
            if (half >= 1) {
                __syncthreads();   // gates prev half's mma reads of W_smem
                gdn::issue_async_load_vec4<BT, W_STRIDE, THREADS_K>(
                    W_view, W_in + W_base + (int64_t) (half * W_STRIDE),
                    W_stride, cl, tid);
                async_copy_commit();
                async_copy_wait_all();
                // Visibility covered by next sit's scatter-SYNC below.
            }

            const int sit_lo = half * N_SNAP_HALF;
            const int sit_hi = sit_lo + N_SNAP_HALF;

            #pragma unroll
            for (int sit = sit_lo; sit < sit_hi; ++sit) {
                const int owning_s_idx = sit;
                const int k_row_off    = sit * SNAP_K_ROWS;
                const int buf          = sit & (N_SNAP_BUF - 1);
                SnapView<SNAP_K_ROWS> snap = snap_views[buf];

                if (s_idx == owning_s_idx) {
                    // Scatter h_frag into snap[d][k] (transposed).
                    #pragma unroll
                    for (int m = 0; m < M_TILES_H_PW; ++m) {
                        const int k_g0 = m * MMA_M + lane_g;
                        const int k_g1 = k_g0 + 8;
                        const int d0   = warp_d_local + 2 * lane_t;
                        const int d1   = d0 + 1;
                        snap.at(d0, k_g0) = h_frag[m][0];
                        snap.at(d1, k_g0) = h_frag[m][1];
                        snap.at(d0, k_g1) = h_frag[m][2];
                        snap.at(d1, k_g1) = h_frag[m][3];
                    }
                }
                __syncthreads();   // gates scatter -> mma + coop_write_h_chunk

                // Coop float4 gmem write of h_chunk for this snap block.
                {
                    constexpr int K_VEC_PER_D = SNAP_K_ROWS / 4;
                    constexpr int N_VEC_SNAP  = SNAP_FLT / 4;
                    #pragma unroll
                    for (int v = tid; v < N_VEC_SNAP; v += THREADS_K) {
                        const int d_local = v / K_VEC_PER_D;
                        const int kvec    = v - d_local * K_VEC_PER_D;
                        const int k_off   = kvec * 4;
                        float4 val = snap.vec4_at(d_local, k_off);
                        const int d_global = d_off + d_local;
                        *reinterpret_cast<float4 *>(
                            &h_chunk[hc_base + (int64_t) d_global * S + k_row_off + k_off]) = val;
                    }
                }

                // matmul1 inner: K_TILES_PER_SNAP_ITER mma K-tiles.
                #pragma unroll
                for (int kt = 0; kt < K_TILES_PER_SNAP_ITER; ++kt) {
                    const int W_k_local = (k_row_off + kt * MMA_K) - half * W_STRIDE;
                    const int snap_k    = kt * MMA_K;

                    const int col_g  = warp_d_local + lane_g;
                    const float b0 = snap.at(col_g, snap_k + lane_t);
                    const float b1 = snap.at(col_g, snap_k + lane_t + 4);

                    #pragma unroll
                    for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
                        const int row_g0 = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
                        const int row_g1 = row_g0 + 8;
                        const int col_t0 = W_k_local + lane_t;
                        const int col_t1 = col_t0 + 4;

                        const float a0 = W_view.at(row_g0, col_t0);
                        const float a1 = W_view.at(row_g1, col_t0);
                        const float a2 = W_view.at(row_g0, col_t1);
                        const float a3 = W_view.at(row_g1, col_t1);

                        mma_m16n8k8_tf32(
                            vnew_frag[m_mm1][0],
                            vnew_frag[m_mm1][1],
                            vnew_frag[m_mm1][2],
                            vnew_frag[m_mm1][3],
                            a0, a1, a2, a3, b0, b1);
                    }
                }

                // No sync: next iter writes the OTHER snap buffer.
            }
        }

        // Cross-chunk W prefetch right after Phase B finishes consuming
        // W_smem -- hides W's HBM read behind C+D+E. ~22 us vs chunk-end
        // batching (see PERF_LOG.md). The __syncthreads gates the half=1
        // sit's mma reads of W_smem (matmul1 inner LDS.32 in slow warps)
        // BEFORE this thread's cp.async DMA starts overwriting W_smem with
        // the next chunk's W_half_0. Without it, racecheck reports a
        // cp.async-write vs f32_to_tf32-read race on W_smem at sm_120, which
        // surfaces as ~25% flaky v_new at S=128/L=256 (the chunk-end barrier
        // before next iter's Phase A only fences the *future* read of the
        // arriving W, not the *prior* read of the outgoing W).
        if (chunk + 1 < NT) {
            __syncthreads();
            const int64_t cs_next     = cs + BT;
            const int64_t ce_next_64  = cs_next + BT;
            const int64_t ce_next     = (ce_next_64 < T) ? ce_next_64 : T;
            const int     cl_next     = (int) (ce_next - cs_next);
            const int64_t W_base_next = chunk_W_base(chunk + 1);
            gdn::issue_async_load_vec4<BT, W_STRIDE, THREADS_K>(
                W_view, W_in + W_base_next, W_stride, cl_next, tid);
            async_copy_commit();
        }

        // === Phase C: subtract U from U_smem (no global wait, U landed in A) ===
        #pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0 = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_local + 2 * lane_t;
            const float2 u_top = *reinterpret_cast<const float2 *>(
                &U_view.at(row_g0, col_d0));
            const float2 u_bot = *reinterpret_cast<const float2 *>(
                &U_view.at(row_g1, col_d0));
            vnew_frag[m_mm1][0] = u_top.x - vnew_frag[m_mm1][0];
            vnew_frag[m_mm1][1] = u_top.y - vnew_frag[m_mm1][1];
            vnew_frag[m_mm1][2] = u_bot.x - vnew_frag[m_mm1][2];
            vnew_frag[m_mm1][3] = u_bot.y - vnew_frag[m_mm1][3];
        }

        // === Phase D: STG vnew (UNDECAYED), STS v_decay -> vd_view, scale h_frag ===
        const float g_C     = g_smem[cl - 1];
        const float gamma_C = exp2_fast(g_C * RCP_LN2_F);

        const int64_t vn_base   = ((int64_t) b * T + cs) * H_v * S
                                + (int64_t) h_v * S;
        const int64_t vn_stride = (int64_t) H_v * S;

        #pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0 = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_global + 2 * lane_t;

            const bool top_in_chunk = (row_g0 < cl);
            const bool bot_in_chunk = (row_g1 < cl);

            const float g_top = top_in_chunk ? g_smem[row_g0] : g_C;
            const float g_bot = bot_in_chunk ? g_smem[row_g1] : g_C;
            const float dec_top = top_in_chunk
                ? exp2_fast((g_C - g_top) * RCP_LN2_F) : 0.0f;
            const float dec_bot = bot_in_chunk
                ? exp2_fast((g_C - g_bot) * RCP_LN2_F) : 0.0f;

            const float v0 = vnew_frag[m_mm1][0];
            const float v1 = vnew_frag[m_mm1][1];
            const float v2 = vnew_frag[m_mm1][2];
            const float v3 = vnew_frag[m_mm1][3];

            if (top_in_chunk) {
                *reinterpret_cast<float2 *>(
                    &v_new[vn_base + (int64_t) row_g0 * vn_stride + col_d0]) =
                    make_float2(v0, v1);
            }
            if (bot_in_chunk) {
                *reinterpret_cast<float2 *>(
                    &v_new[vn_base + (int64_t) row_g1 * vn_stride + col_d0]) =
                    make_float2(v2, v3);
            }

            const int row_g0_loc = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1_loc = row_g0_loc + 8;
            const int col_d0_loc = warp_d_local + 2 * lane_t;
            *reinterpret_cast<float2 *>(&vd_view.at(row_g0_loc, col_d0_loc)) =
                make_float2(v0 * dec_top, v1 * dec_top);
            *reinterpret_cast<float2 *>(&vd_view.at(row_g1_loc, col_d0_loc)) =
                make_float2(v2 * dec_bot, v3 * dec_bot);
        }

        #pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                h_frag[m][e] *= gamma_C;
            }
        }

        // Drain late group (k_half_0). Wait queue at this point is
        // {k (older), W_next (newer)}; wait<1> drains k while leaving
        // W_next in flight. The barrier below also gates matmul2's reads
        // of vd_view (Phase D write -> E1 read).
        async_copy_wait<1>();
        __syncthreads();

        // === Phase E1: matmul2 first half (k_half_0, vd rows 0..BT/2-1) ===
        #pragma unroll
        for (int kt = 0; kt < K_TILES_MM2_HALF; ++kt) {
            const int k_off_local = kt * MMA_K;

            const int row_t0 = k_off_local + lane_t;
            const int row_t1 = row_t0 + 4;
            const int col_g  = warp_d_local + lane_g;
            const float b0 = vd_view.at(row_t0, col_g);
            const float b1 = vd_view.at(row_t1, col_g);

            #pragma unroll
            for (int m = 0; m < M_TILES_H_PW; ++m) {
                const int row_a0    = k_off_local + lane_t;
                const int row_a1    = row_a0 + 4;
                const int col_a_top = s_idx * S_PER_WARP + m * MMA_M + lane_g;
                const int col_a_bot = col_a_top + 8;

                const float a0 = k_view.at(row_a0, col_a_top);
                const float a1 = k_view.at(row_a0, col_a_bot);
                const float a2 = k_view.at(row_a1, col_a_top);
                const float a3 = k_view.at(row_a1, col_a_bot);

                mma_m16n8k8_tf32(
                    h_frag[m][0],
                    h_frag[m][1],
                    h_frag[m][2],
                    h_frag[m][3],
                    a0, a1, a2, a3, b0, b1);
            }
        }

        __syncthreads();    // before k_half_1 cp.async overwrites k_smem

        // === Phase E2: cp.async k_half_1 -> k_smem ===
        const int64_t k_base_h1 = k_base + (int64_t) K_HALF_ROWS * k_stride;
        const int     cl_kh1    = (cl > K_HALF_ROWS) ? (cl - K_HALF_ROWS) : 0;
        gdn::issue_async_load_vec4<K_HALF_ROWS, S, THREADS_K>(
            k_view, k_in + k_base_h1, k_stride, cl_kh1, tid);
        async_copy_commit();
        async_copy_wait_all();
        __syncthreads();

        // === Phase E3: matmul2 second half (k_half_1, vd rows BT/2..BT-1) ===
        #pragma unroll
        for (int kt = 0; kt < K_TILES_MM2_HALF; ++kt) {
            const int k_off_local  = kt * MMA_K;
            const int k_off_global = k_off_local + K_HALF_ROWS;   // vd row

            const int row_t0 = k_off_global + lane_t;
            const int row_t1 = row_t0 + 4;
            const int col_g  = warp_d_local + lane_g;
            const float b0 = vd_view.at(row_t0, col_g);
            const float b1 = vd_view.at(row_t1, col_g);

            #pragma unroll
            for (int m = 0; m < M_TILES_H_PW; ++m) {
                const int row_a0    = k_off_local + lane_t;
                const int row_a1    = row_a0 + 4;
                const int col_a_top = s_idx * S_PER_WARP + m * MMA_M + lane_g;
                const int col_a_bot = col_a_top + 8;

                const float a0 = k_view.at(row_a0, col_a_top);
                const float a1 = k_view.at(row_a0, col_a_bot);
                const float a2 = k_view.at(row_a1, col_a_top);
                const float a3 = k_view.at(row_a1, col_a_bot);

                mma_m16n8k8_tf32(
                    h_frag[m][0],
                    h_frag[m][1],
                    h_frag[m][2],
                    h_frag[m][3],
                    a0, a1, a2, a3, b0, b1);
            }
        }

        __syncthreads();    // before chunk-end cp.async overwrites k/U smem

        // Cross-chunk prefetch: chunk t+1 U (small) + k_half_0 (large) split
        // into two commit groups so Phase A only waits for U.
        if (chunk + 1 < NT) {
            const int64_t cs_next     = cs + BT;
            const int64_t ce_next_64  = cs_next + BT;
            const int64_t ce_next     = (ce_next_64 < T) ? ce_next_64 : T;
            const int     cl_next     = (int) (ce_next - cs_next);
            const int64_t k_base_next = chunk_k_base(chunk + 1);
            const int64_t U_base_next = chunk_W_base(chunk + 1);
            const int     cl_kh0_next = (cl_next < K_HALF_ROWS) ? cl_next : K_HALF_ROWS;
            gdn::issue_async_load_vec4<BT, N_STRIP_PER_BLOCK, THREADS_K>(
                U_view, U_in + U_base_next + d_off, W_stride, cl_next, tid);
            async_copy_commit();
            gdn::issue_async_load_vec4<K_HALF_ROWS, S, THREADS_K>(
                k_view, k_in + k_base_next, k_stride, cl_kh0_next, tid);
            async_copy_commit();
        }
    }

    // === Phase Z: store h_frag -> state_out (AR-transposed) via snap+coop ===
    const int64_t st_base = ((int64_t) b * H_v + h_v) * S * S;

    #pragma unroll
    for (int sit = 0; sit < N_SNAP_ITERS; ++sit) {
        const int owning_s_idx = sit;
        const int k_row_off    = sit * SNAP_K_ROWS;
        const int buf          = sit & (N_SNAP_BUF - 1);
        SnapView<SNAP_K_ROWS> snap = snap_views[buf];

        if (s_idx == owning_s_idx) {
            #pragma unroll
            for (int m = 0; m < M_TILES_H_PW; ++m) {
                const int k_g0 = m * MMA_M + lane_g;
                const int k_g1 = k_g0 + 8;
                const int d0   = warp_d_local + 2 * lane_t;
                const int d1   = d0 + 1;
                snap.at(d0, k_g0) = h_frag[m][0];
                snap.at(d1, k_g0) = h_frag[m][1];
                snap.at(d0, k_g1) = h_frag[m][2];
                snap.at(d1, k_g1) = h_frag[m][3];
            }
        }
        __syncthreads();

        constexpr int K_VEC_PER_D = SNAP_K_ROWS / 4;
        constexpr int N_VEC_SNAP  = SNAP_FLT / 4;
        #pragma unroll
        for (int v = tid; v < N_VEC_SNAP; v += THREADS_K) {
            const int d_local = v / K_VEC_PER_D;
            const int kvec    = v - d_local * K_VEC_PER_D;
            const int k_off   = kvec * 4;
            float4 val = snap.vec4_at(d_local, k_off);
            const int d_global = d_off + d_local;
            *reinterpret_cast<float4 *>(
                &state_out[st_base + (int64_t) d_global * S + k_row_off + k_off]) = val;
        }
        // No sync: next sit writes the OTHER snap buffer.
    }
}

template <int S>
cudaError_t launch_typed(const state_passing_config & cfg,
                         gdn::head_map qk_map, int NT) {
    using D = kernel_dims<S>;
    constexpr int smem_bytes = smem_layout<S>::SMEM_FLOATS * (int) sizeof(float);

    cudaError_t err = cudaFuncSetAttribute(
        state_passing_gdn_kernel<S>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);
    if (err != cudaSuccess) return err;

    const dim3 grid((unsigned) (cfg.B * cfg.H_v * D::D_STRIPS), 1, 1);
    const dim3 block(D::THREADS, 1, 1);

    const int64_t k_stride_t = (cfg.k_stride_t_floats != 0)
                                ? cfg.k_stride_t_floats
                                : (int64_t) cfg.H_qk * cfg.S;

    const int64_t L_total = (cfg.L_total != 0) ? cfg.L_total : cfg.L;

    state_passing_gdn_kernel<S><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.W, cfg.U, cfg.k_or_kg, cfg.g_cumsum, cfg.state_in,
        cfg.v_new, cfg.h_chunk, cfg.state_out,
        cfg.L, L_total, cfg.H_v, qk_map, k_stride_t, NT);
    return cudaGetLastError();
}

}  // namespace

cudaError_t launch_state_passing(const state_passing_config & cfg) {
    detail::stage_validator v{"launch_state_passing",
                              cfg.S, cfg.H_qk, cfg.H_v, cfg.L, cfg.B, cfg.kda};
    GDN_PROPAGATE(v.check_shape());
    GDN_PROPAGATE(v.check_gdn_full_chunks());
    if (cfg.W == nullptr || cfg.U == nullptr || cfg.k_or_kg == nullptr ||
        cfg.g_cumsum == nullptr || cfg.state_in == nullptr ||
        cfg.v_new == nullptr || cfg.h_chunk == nullptr ||
        cfg.state_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (cfg.kda) return cudaErrorNotYetImplemented;  // Phase C

    const auto    qk_map = gdn::head_map::of((int) cfg.H_qk, (int) cfg.H_v);
    const int64_t NT     = (cfg.L + BT - 1) / BT;

    // grid_x = B * H_v * D_STRIPS depends on S; check inside each case.
    auto check_grid_for = [&](int d_strips) -> cudaError_t {
        return v.check_grid(cfg.B * cfg.H_v * d_strips, /*grid_y=*/1, /*grid_z=*/1);
    };

    switch (cfg.S) {
        case 16:  GDN_PROPAGATE(check_grid_for(kernel_dims<16> ::D_STRIPS));
                  return launch_typed<16> (cfg, qk_map, (int) NT);
        case 32:  GDN_PROPAGATE(check_grid_for(kernel_dims<32> ::D_STRIPS));
                  return launch_typed<32> (cfg, qk_map, (int) NT);
        case 64:  GDN_PROPAGATE(check_grid_for(kernel_dims<64> ::D_STRIPS));
                  return launch_typed<64> (cfg, qk_map, (int) NT);
        case 128: GDN_PROPAGATE(check_grid_for(kernel_dims<128>::D_STRIPS));
                  return launch_typed<128>(cfg, qk_map, (int) NT);
        default:  return cudaErrorInvalidValue;  // check_shape already filtered
    }
}

}  // namespace chunked_gdn::stages
