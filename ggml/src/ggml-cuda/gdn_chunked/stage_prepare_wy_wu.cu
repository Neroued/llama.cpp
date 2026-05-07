// Stage 2 (fused): build T_inv in smem, immediately consume it to produce
// W and U. T_inv never crosses HBM (workspace T_inv region is 0 bytes).
//
// Math + I/O layouts: see chunked_gdn.cuh::stages::prepare_wy_wu_config.
// Block / smem total: 4 warps (128 t), smem ~50 KB at S=128 (16 KB T_inv +
//                     32 KB VK alias + 0.75 KB g/beta/bg).
// Tuning history, ncu numbers, sync audit, footguns:
//   chunked/PERF_LOG.md#stage-2-prepare_wy_wu-fused-stage_prepare_wy_wucu

#include "chunked_gdn.cuh"
#include "chunked_internals.cuh"
#include "chunked_utils.cuh"

#include <cstdio>

namespace chunked_gdn::stages {

namespace {

using chunked_gdn::detail::BT;
using chunked_gdn::detail::BC;
using chunked_gdn::detail::MMA_M;
using chunked_gdn::detail::MMA_N;
using chunked_gdn::detail::MMA_K;
using chunked_gdn::detail::bh_decode_t;
using chunked_gdn::detail::zero_frag;
using gdn::SmemTile;
using gdn::mma_m16n8k8_tf32;
using gdn::async_copy_commit;
using gdn::async_copy_wait_all;

static_assert(chunked_gdn::kChunkSize == 64,
              "stage_prepare_wy_wu: kChunkSize must be 64 (kernel hard-codes "
              "BT=64 = 4 * BC=16)");

constexpr int N_SUB     = BT / BC;                              // 4
constexpr int N_WARPS   = N_SUB;                                // 4 warps
constexpr int THREADS   = N_WARPS * gdn::WARP_SIZE;             // 128
constexpr int N_K_TILES = BT / MMA_K;                           // 8 (recompute_wu)

static_assert(MMA_M == BC, "kernel assumes MMA m == BC");

// Phase D scratch row stride. BC+1=17 is mutually prime with 32 -- breaks
// the 4-way bank conflict that stride=16 induces on the scatter writes.
constexpr int SCR_STRIDE = BC + 1;                              // 17

// ---------------------------------------------------------------------------
// kernel_dims<S>: K-chunking, output N-chunking, fused smem layout.
//
// VK_floats = max(BT*S, scratch). At S<=16 the natural BT*S=1024 floats is
// LESS than scratch=1088 -- without max() Phase D scatter would silently
// overflow into g_smem/beta_smem (see PERF_LOG.md Stage 2 footgun).
// ---------------------------------------------------------------------------
template <int S> struct kernel_dims {
    // K-chunk dims for the prepare_wy half (K loaded in chunks of <= 64 cols).
    static constexpr int K_CHUNK         = (S > 64) ? 64 : S;
    static constexpr int N_K_CHUNKS      = S / K_CHUNK;
    static constexpr int K_TILES_CHUNK   = K_CHUNK / MMA_K;
    static_assert(S % K_CHUNK == 0, "S must be a multiple of the K chunk size");

    // recompute_wu N-axis chunking (caps D fragment at 16 fp32/lane).
    static constexpr int N_TILES_TOTAL     = S / MMA_N;
    static constexpr int N_TILES_PER_CHUNK = (N_TILES_TOTAL < 4) ? N_TILES_TOTAL : 4;
    static constexpr int N_CHUNKS          = N_TILES_TOTAL / N_TILES_PER_CHUNK;
    static_assert(N_CHUNKS * N_TILES_PER_CHUNK == N_TILES_TOTAL, "must divide");

    // Fused smem layout (floats).
    static constexpr int T_inv_floats   = BT * BT;                 // 16 KB
    static constexpr int natural_VK     = BT * S;
    static constexpr int scratch_floats = N_WARPS * BC * (BC + 1); // 1088
    static constexpr int VK_floats      = natural_VK > scratch_floats
                                          ? natural_VK : scratch_floats;
    static constexpr int g_floats     = BT;
    static constexpr int beta_floats  = BT;
    static constexpr int bg_floats    = BT;
    static constexpr int SMEM_FLOATS  = T_inv_floats + VK_floats
                                      + g_floats + beta_floats + bg_floats;
};

// ---------------------------------------------------------------------------
// Phase D helpers. Refactored from in-kernel lambdas to namespace-scope
// __device__ functions to drop the spill-via-lambda-capture pattern that
// nvcc's register allocator could not see through. See PERF_LOG.md.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void scatter_frag_to_scr(
        const float frag[8],
        float * __restrict__ scr_smem,
        int warp,
        int lane) {
    float * Sptr = scr_smem + warp * BC * SCR_STRIDE;
    const int lane_g = lane >> 2;
    const int col_2t = (lane & 3) << 1;
    Sptr[ lane_g      * SCR_STRIDE + col_2t    ] = frag[0];
    Sptr[ lane_g      * SCR_STRIDE + col_2t + 1] = frag[1];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t    ] = frag[2];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 1] = frag[3];
    Sptr[ lane_g      * SCR_STRIDE + col_2t + 8] = frag[4];
    Sptr[ lane_g      * SCR_STRIDE + col_2t + 9] = frag[5];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 8] = frag[6];
    Sptr[(lane_g + 8) * SCR_STRIDE + col_2t + 9] = frag[7];
}

// 16x16x16 mma: A from raw row-major scratch (stride SCR_STRIDE),
// B from swizzled M_view at offset (M_row_off, M_col_off).
__device__ __forceinline__ void mma16_raw_x_swiz(
        float D[8],
        int lane,
        const float * __restrict__ A_buf,
        SmemTile<BT> M_view,
        int M_row_off,
        int M_col_off) {
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
    #pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0 = A_buf[ lane_g      * SCR_STRIDE + (k_off + lane_t    )];
        const float a1 = A_buf[(lane_g + 8) * SCR_STRIDE + (k_off + lane_t    )];
        const float a2 = A_buf[ lane_g      * SCR_STRIDE + (k_off + lane_t + 4)];
        const float a3 = A_buf[(lane_g + 8) * SCR_STRIDE + (k_off + lane_t + 4)];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n_off = nt * MMA_N;
            const float b0 = M_view.at(M_row_off + k_off + lane_t,     M_col_off + n_off + lane_g);
            const float b1 = M_view.at(M_row_off + k_off + lane_t + 4, M_col_off + n_off + lane_g);
            mma_m16n8k8_tf32(
                D[nt * 4 + 0], D[nt * 4 + 1],
                D[nt * 4 + 2], D[nt * 4 + 3],
                a0, a1, a2, a3, b0, b1);
        }
    }
}

// 16x16x16 mma: A from swizzled M_view, B from raw row-major scratch.
__device__ __forceinline__ void mma16_swiz_x_raw(
        float D[8],
        int lane,
        SmemTile<BT> M_view,
        int M_row_off,
        int M_col_off,
        const float * __restrict__ B_buf) {
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;
    #pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const int k_off = kt * MMA_K;
        const float a0 = M_view.at(M_row_off + lane_g,     M_col_off + k_off + lane_t    );
        const float a1 = M_view.at(M_row_off + lane_g + 8, M_col_off + k_off + lane_t    );
        const float a2 = M_view.at(M_row_off + lane_g,     M_col_off + k_off + lane_t + 4);
        const float a3 = M_view.at(M_row_off + lane_g + 8, M_col_off + k_off + lane_t + 4);
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n_off = nt * MMA_N;
            const float b0 = B_buf[(k_off + lane_t    ) * SCR_STRIDE + (n_off + lane_g)];
            const float b1 = B_buf[(k_off + lane_t + 4) * SCR_STRIDE + (n_off + lane_g)];
            mma_m16n8k8_tf32(
                D[nt * 4 + 0], D[nt * 4 + 1],
                D[nt * 4 + 2], D[nt * 4 + 3],
                a0, a1, a2, a3, b0, b1);
        }
    }
}

__device__ __forceinline__ void compute_off_diag(
        float out[8],
        int my_w,
        int my_j,
        int warp,
        int lane,
        const float A_reg[N_SUB][8],
        float * __restrict__ scr_smem,
        SmemTile<BT> M_view) {
    float sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int k = my_j; k < my_w; ++k) {
        scatter_frag_to_scr(A_reg[k], scr_smem, warp, lane);
        __syncwarp();

        const float * A_buf = scr_smem + warp * BC * SCR_STRIDE;
        if (k == my_j) {
            mma16_raw_x_swiz(sum, lane, A_buf, M_view, my_j * BC, my_j * BC);
            #pragma unroll
            for (int e = 0; e < 8; ++e) sum[e] += A_reg[k][e];
        } else {
            mma16_raw_x_swiz(sum, lane, A_buf, M_view, k * BC, my_j * BC);
        }
    }

    scatter_frag_to_scr(sum, scr_smem, warp, lane);
    __syncwarp();
    const float * B_buf = scr_smem + warp * BC * SCR_STRIDE;

    float prod[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    mma16_swiz_x_raw(prod, lane, M_view, my_w * BC, my_w * BC, B_buf);

    #pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = prod[e] + sum[e];
}

__device__ __forceinline__ void store_frag_to_M(
        const float frag[8],
        int my_w,
        int my_j,
        int lane_g,
        int lane_t,
        SmemTile<BT> M_view) {
    const int row_g0   = my_w * BC + lane_g;
    const int row_g1   = row_g0 + 8;
    const int col_base = my_j * BC + 2 * lane_t;
    M_view.at(row_g0, col_base    ) = frag[0];
    M_view.at(row_g0, col_base + 1) = frag[1];
    M_view.at(row_g1, col_base    ) = frag[2];
    M_view.at(row_g1, col_base + 1) = frag[3];
    M_view.at(row_g0, col_base + 8) = frag[4];
    M_view.at(row_g0, col_base + 9) = frag[5];
    M_view.at(row_g1, col_base + 8) = frag[6];
    M_view.at(row_g1, col_base + 9) = frag[7];
}

// __launch_bounds__ NOT applied: nvcc's natural reg=203 is the sweet spot.
// Capping at 256 (min_blocks=2) just spills the overflow. See PERF_LOG.md.
template <int S>
__global__ void
prepare_wy_wu_gdn_kernel(const float * __restrict__ k_in,
                         const float * __restrict__ v_in,
                         const float * __restrict__ g_cumsum,
                         const float * __restrict__ beta_in,
                         float       * __restrict__ W,
                         float       * __restrict__ U,
                         int64_t T,
                         int64_t T_total,
                         int64_t H_v,
                         gdn::head_map qk_map,
                         // Token-axis strides (in floats) for k / v. Caller
                         // passes materialised values (launch_typed handles
                         // 0 -> packed defaults H_qk*S / H_v*S).
                         int64_t k_stride_t,
                         int64_t v_stride_t) {
    using KD = kernel_dims<S>;
    constexpr int K_STRIDE          = KD::K_CHUNK;
    constexpr int K_CHUNK           = KD::K_CHUNK;
    constexpr int K_TILES_CHUNK     = KD::K_TILES_CHUNK;
    constexpr int N_K_CHUNKS        = KD::N_K_CHUNKS;
    constexpr int N_TILES_PER_CHUNK = KD::N_TILES_PER_CHUNK;
    constexpr int N_CHUNKS          = KD::N_CHUNKS;

    extern __shared__ float smem[];
    float * const T_inv_smem = smem;
    float * const VK_smem    = smem + KD::T_inv_floats;
    float * const g_smem     = VK_smem + KD::VK_floats;
    float * const beta_smem  = g_smem + BT;
    float * const bg_smem    = beta_smem + BT;

    // VK_smem aliases: K (Phase WY-B) -> scr (Phase WY-D) -> V (Phase WU-A)
    // -> K (Phase WU-C). Each transition is gated by an existing barrier.
    float * const K_smem   = VK_smem;
    float * const scr_smem = VK_smem;

    SmemTile<BT>       M_view{T_inv_smem};   // T_inv from Phase WY-C onward
    SmemTile<BT>       T_view{T_inv_smem};   // recompute_wu reads via T_view
    SmemTile<K_STRIDE> K_view{K_smem};       // prepare_wy Phase B
    SmemTile<S>        VK_view{VK_smem};     // recompute_wu Phase A..D

    const int tid = threadIdx.x;
    const auto lanes = gdn::mma_lane_t::decode(tid);
    const int lane   = lanes.lane;
    const int warp   = lanes.warp;
    const int lane_g = lanes.lane_g;
    const int lane_t = lanes.lane_t;

    const int chunk = blockIdx.x;
    // L2-reuse permutation: CTAs 0..G-1 in a wave share qk_head=0, etc.
    // See `head_map::cta_h_v` / `bh_decode_t::of(bh, head_map)`.
    const auto bh   = bh_decode_t::of(blockIdx.y, qk_map);
    const int b     = bh.b;
    const int h_v   = bh.h_v;

    const auto cb = gdn::chunk_bounds_t::of(chunk, T, BT);
    const int64_t cs = cb.cs;
    const int     cl = cb.cl;

    // === Phase WY-A: cooperative load of g, beta (K loaded per-chunk in B) ===
    // g_cumsum is workspace ([B, T, H_v], packed at T = chunked length).
    // beta_in is the user buffer ([B, T_total, H_v]); in split dispatch
    // T_total >= T so we must use T_total for batch stride here.
    if (tid < BT) {
        if (tid < cl) {
            const int64_t g_off    = ((int64_t) b * T       + cs + tid) * H_v + h_v;
            const int64_t beta_off = ((int64_t) b * T_total + cs + tid) * H_v + h_v;
            g_smem[tid]    = g_cumsum[g_off];
            beta_smem[tid] = beta_in[beta_off];
        } else {
            g_smem[tid]    = 0.0f;
            beta_smem[tid] = 0.0f;
        }
    }
    // No __syncthreads here -- per-chunk K loader below issues one.

    // === Phase WY-B: KKT via TF32 MMA on lower-tri 4x4 sub-block grid ===
    float A_reg[N_SUB][8];
    zero_frag(A_reg);

    const int row_g0 = warp * BC + lane_g;          // strip rows 0..7
    const int row_g1 = row_g0 + 8;                  // strip rows 8..15

    // k_in is the user buffer (batch stride = T_total * k_stride_t).
    const int64_t k_base = ((int64_t) b * T_total + cs) * k_stride_t
                         + (int64_t) qk_map.qk_head(h_v) * S;
    constexpr int VEC_PER_ROW_CHUNK = K_CHUNK / 4;
    constexpr int N_VEC_CHUNK       = BT * VEC_PER_ROW_CHUNK;

    auto kkt_strip = [&]<int N_OWNED>() {
        #pragma unroll
        for (int k_tile = 0; k_tile < K_TILES_CHUNK; ++k_tile) {
            const int k_off  = k_tile * MMA_K;
            const int col_t0 = k_off + lane_t;
            const int col_t1 = col_t0 + 4;

            const float a0 = K_view.at(row_g0, col_t0);
            const float a1 = K_view.at(row_g1, col_t0);
            const float a2 = K_view.at(row_g0, col_t1);
            const float a3 = K_view.at(row_g1, col_t1);

            #pragma unroll
            for (int j_sub = 0; j_sub < N_OWNED; ++j_sub) {
                #pragma unroll
                for (int n_tile = 0; n_tile < 2; ++n_tile) {
                    const int n_off = n_tile * MMA_N;
                    const int row_b = j_sub * BC + n_off + lane_g;
                    const float b0 = K_view.at(row_b, col_t0);
                    const float b1 = K_view.at(row_b, col_t1);

                    mma_m16n8k8_tf32(
                        A_reg[j_sub][n_tile * 4 + 0],
                        A_reg[j_sub][n_tile * 4 + 1],
                        A_reg[j_sub][n_tile * 4 + 2],
                        A_reg[j_sub][n_tile * 4 + 3],
                        a0, a1, a2, a3,
                        b0, b1);
                }
            }
        }
    };

    #pragma unroll
    for (int kc = 0; kc < N_K_CHUNKS; ++kc) {
        const int chunk_col = kc * K_CHUNK;

        #pragma unroll
        for (int v = tid; v < N_VEC_CHUNK; v += THREADS) {
            const int row  = v / VEC_PER_ROW_CHUNK;
            const int col4 = v - row * VEC_PER_ROW_CHUNK;
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (row < cl) {
                val = *reinterpret_cast<const float4 *>(
                    k_in + k_base + (int64_t) row * k_stride_t + chunk_col + col4 * 4);
            }
            K_view.vec4_at(row, col4 * 4) = val;
        }
        __syncthreads();

        switch (warp) {
            case 0: kkt_strip.template operator()<1>(); break;
            case 1: kkt_strip.template operator()<2>(); break;
            case 2: kkt_strip.template operator()<3>(); break;
            case 3: kkt_strip.template operator()<4>(); break;
        }

        if (kc + 1 < N_K_CHUNKS) {
            __syncthreads();
        }
    }

    // === Phase WY-C reg: gate decay + beta scaling on fragments ===
    const int   r_g0     = warp * BC + lane_g;
    const int   r_g1     = r_g0 + 8;
    const float beta_r0  = beta_smem[r_g0];
    const float beta_r1  = beta_smem[r_g1];
    const float g_r0     = g_smem[r_g0];
    const float g_r1     = g_smem[r_g1];
    const float nbeta_r0 = -beta_r0;
    const float nbeta_r1 = -beta_r1;

    #pragma unroll
    for (int j_sub = 0; j_sub < N_SUB; ++j_sub) {
        if (j_sub > warp) continue;

        const int c_base = j_sub * BC + 2 * lane_t;
        const int c0 = c_base;
        const int c1 = c_base + 1;
        const int c2 = c_base + 8;
        const int c3 = c_base + 9;

        const float g_c0 = g_smem[c0];
        const float g_c1 = g_smem[c1];
        const float g_c2 = g_smem[c2];
        const float g_c3 = g_smem[c3];

        const bool is_diag = (j_sub == warp);

        const bool m00 = !is_diag || r_g0 > c0;
        const bool m01 = !is_diag || r_g0 > c1;
        const bool m10 = !is_diag || r_g1 > c0;
        const bool m11 = !is_diag || r_g1 > c1;
        const bool m02 = !is_diag || r_g0 > c2;
        const bool m03 = !is_diag || r_g0 > c3;
        const bool m12 = !is_diag || r_g1 > c2;
        const bool m13 = !is_diag || r_g1 > c3;

        // NaN-guard: g_cumsum is monotone non-increasing (g <= 0). On the
        // lower triangle (m == true) g_r - g_c <= 0 and expf is bounded by 1.
        // On the upper triangle (m == false) g_r - g_c >= 0 and with realistic
        // per-token |g| ~ 1..5 over BT=64 it can exceed expf's safe range
        // (~88), so expf overflows to +inf. Routing those lanes through a
        // selp (?:) bypass to 0.0 keeps the result finite without going
        // through the inf * 0 = NaN trap, while also folding the mask into
        // the same selp (saves one fmul per element vs. an explicit `* m`).
        A_reg[j_sub][0] = m00 ? nbeta_r0 * A_reg[j_sub][0] * expf(g_r0 - g_c0) : 0.0f;
        A_reg[j_sub][1] = m01 ? nbeta_r0 * A_reg[j_sub][1] * expf(g_r0 - g_c1) : 0.0f;
        A_reg[j_sub][2] = m10 ? nbeta_r1 * A_reg[j_sub][2] * expf(g_r1 - g_c0) : 0.0f;
        A_reg[j_sub][3] = m11 ? nbeta_r1 * A_reg[j_sub][3] * expf(g_r1 - g_c1) : 0.0f;
        A_reg[j_sub][4] = m02 ? nbeta_r0 * A_reg[j_sub][4] * expf(g_r0 - g_c2) : 0.0f;
        A_reg[j_sub][5] = m03 ? nbeta_r0 * A_reg[j_sub][5] * expf(g_r0 - g_c3) : 0.0f;
        A_reg[j_sub][6] = m12 ? nbeta_r1 * A_reg[j_sub][6] * expf(g_r1 - g_c2) : 0.0f;
        A_reg[j_sub][7] = m13 ? nbeta_r1 * A_reg[j_sub][7] * expf(g_r1 - g_c3) : 0.0f;
    }

    // === Phase WY-C diag: scatter A_reg[w][w] -> M_view, in-place forward sub ===
    __syncthreads();   // gates K_smem (Phase B reads) -> M_smem alias handover

    {
        constexpr int N = BT * BT;
        #pragma unroll
        for (int idx = tid; idx < N; idx += THREADS) T_inv_smem[idx] = 0.0f;
    }
    __syncthreads();

    {
        const int c_base = warp * BC + 2 * lane_t;
        M_view.at(r_g0, c_base    ) = A_reg[warp][0];
        M_view.at(r_g0, c_base + 1) = A_reg[warp][1];
        M_view.at(r_g1, c_base    ) = A_reg[warp][2];
        M_view.at(r_g1, c_base + 1) = A_reg[warp][3];
        M_view.at(r_g0, c_base + 8) = A_reg[warp][4];
        M_view.at(r_g0, c_base + 9) = A_reg[warp][5];
        M_view.at(r_g1, c_base + 8) = A_reg[warp][6];
        M_view.at(r_g1, c_base + 9) = A_reg[warp][7];
    }
    __syncwarp();

    {
        const int diag_off = warp * BC;
        const int wcol     = lane & 15;
        for (int i = 1; i < BC; ++i) {
            const int row_i = diag_off + i;
            const int col   = diag_off + wcol;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < BC - 1; ++j) {
                if (j < i) {
                    sum += M_view.at(row_i,         diag_off + j) *
                           M_view.at(diag_off + j,  col);
                }
            }
            __syncwarp();
            if (wcol < i) {
                M_view.at(row_i, col) += sum;
            }
            __syncwarp();
        }
    }
    __syncthreads();

    // === Phase WY-D: block-Schur off-diagonal completion (3 waves) ===
    {
        float out[8];
        if (warp >= 1) {
            compute_off_diag(out, warp, warp - 1, warp, lane,
                             A_reg, scr_smem, M_view);
            store_frag_to_M(out, warp, warp - 1, lane_g, lane_t, M_view);
        }
    }
    __syncthreads();

    {
        float out[8];
        if (warp >= 2) {
            compute_off_diag(out, warp, warp - 2, warp, lane,
                             A_reg, scr_smem, M_view);
            store_frag_to_M(out, warp, warp - 2, lane_g, lane_t, M_view);
        }
    }
    __syncthreads();

    {
        float out[8];
        if (warp == 3) {
            compute_off_diag(out, 3, 0, warp, lane,
                             A_reg, scr_smem, M_view);
            store_frag_to_M(out, 3, 0, lane_g, lane_t, M_view);
        }
    }
    __syncthreads();

    // === Phase WY-E: +I on diagonal of T_inv (no HBM write; sync fused into WU-A) ===
    if (tid < cl) {
        M_view.at(tid, tid) += 1.0f;
    }

    // === Phase WU-A: cp.async V into VK_smem; pre-compute bg = beta * exp(g) ===
    {
        // v_in is the user buffer (batch stride = T_total * v_stride_t).
        const int64_t row_base   = (int64_t) b * T_total * v_stride_t
                                 + (int64_t) cs * v_stride_t
                                 + (int64_t) h_v * S;
        const int64_t row_stride = v_stride_t;
        gdn::issue_async_load_vec4<BT, S, THREADS>(VK_view, v_in + row_base,
                                                   row_stride, cl, tid);
    }

    if (tid < BT) {
        bg_smem[tid] = beta_smem[tid] * expf(g_smem[tid]);
    }

    async_copy_commit();
    async_copy_wait_all();
    __syncthreads();   // gates +I, bg writes, and V smem visibility

    // Per-thread fragment indexing for the recompute_wu matmuls.
    const int t_row_g0 = warp * MMA_M + lane_g;
    const int t_row_g1 = t_row_g0 + 8;
    const int col_d0   = (lane & 3) << 1;

    const int64_t out_base       = ((int64_t) b * T + cs) * H_v * S
                                 + (int64_t) h_v * S;
    const int64_t out_row_stride = (int64_t) H_v * S;

    // T_inv @ (scale * VK) -> out_gmem
    auto matmul_and_store = [&](float * __restrict__ out_gmem,
                                const float (& scale_kt)[2 * N_K_TILES]) {
        #pragma unroll
        for (int n_chunk = 0; n_chunk < N_CHUNKS; ++n_chunk) {
            const int n_chunk_off = n_chunk * (N_TILES_PER_CHUNK * MMA_N);

            float D[N_TILES_PER_CHUNK][4];
            zero_frag(D);

            #pragma unroll
            for (int k_tile = 0; k_tile < N_K_TILES; ++k_tile) {
                const int k_off  = k_tile * MMA_K;
                const int col_t0 = k_off + lane_t;
                const int col_t1 = col_t0 + 4;
                const float a0 = T_view.at(t_row_g0, col_t0);
                const float a1 = T_view.at(t_row_g1, col_t0);
                const float a2 = T_view.at(t_row_g0, col_t1);
                const float a3 = T_view.at(t_row_g1, col_t1);

                const int row_t0 = k_off + lane_t;
                const int row_t1 = row_t0 + 4;
                const float s0 = scale_kt[2 * k_tile + 0];
                const float s1 = scale_kt[2 * k_tile + 1];

                #pragma unroll
                for (int t = 0; t < N_TILES_PER_CHUNK; ++t) {
                    const int n_off = n_chunk_off + t * MMA_N;
                    const int col_g = n_off + lane_g;
                    const float b0 = VK_view.at(row_t0, col_g) * s0;
                    const float b1 = VK_view.at(row_t1, col_g) * s1;

                    mma_m16n8k8_tf32(
                        D[t][0], D[t][1], D[t][2], D[t][3],
                        a0, a1, a2, a3,
                        b0, b1);
                }
            }

            #pragma unroll
            for (int t = 0; t < N_TILES_PER_CHUNK; ++t) {
                const int col = n_chunk_off + t * MMA_N + col_d0;
                if (t_row_g0 < cl) {
                    float2 v0 = make_float2(D[t][0], D[t][1]);
                    *reinterpret_cast<float2 *>(
                        &out_gmem[(int64_t) t_row_g0 * out_row_stride + col]) = v0;
                }
                if (t_row_g1 < cl) {
                    float2 v1 = make_float2(D[t][2], D[t][3]);
                    *reinterpret_cast<float2 *>(
                        &out_gmem[(int64_t) t_row_g1 * out_row_stride + col]) = v1;
                }
            }
        }
    };

    // Pre-load per-thread row-scale fragments for the U matmul.
    float beta_kt[2 * N_K_TILES];
    #pragma unroll
    for (int k_tile = 0; k_tile < N_K_TILES; ++k_tile) {
        const int row_t0 = k_tile * MMA_K + lane_t;
        const int row_t1 = row_t0 + 4;
        beta_kt[2 * k_tile + 0] = beta_smem[row_t0];
        beta_kt[2 * k_tile + 1] = beta_smem[row_t1];
    }

    // === Phase WU-B: U = T_inv @ (beta * V) ===
    matmul_and_store(U + out_base, beta_kt);

    // === Phase WU-C: cp.async K -> VK_smem (overwrite V) ===
    __syncthreads();   // gates V reads done before K cp.async overwrites

    {
        // k_in is the user buffer (batch stride = T_total * k_stride_t).
        const int64_t row_base = ((int64_t) b * T_total + cs) * k_stride_t
                               + (int64_t) qk_map.qk_head(h_v) * S;
        gdn::issue_async_load_vec4<BT, S, THREADS>(VK_view, k_in + row_base,
                                                   k_stride_t, cl, tid);
    }
    async_copy_commit();

    // Pre-load bg into per-thread slots while K cp.async is in flight.
    float bg_kt[2 * N_K_TILES];
    #pragma unroll
    for (int k_tile = 0; k_tile < N_K_TILES; ++k_tile) {
        const int row_t0 = k_tile * MMA_K + lane_t;
        const int row_t1 = row_t0 + 4;
        bg_kt[2 * k_tile + 0] = bg_smem[row_t0];
        bg_kt[2 * k_tile + 1] = bg_smem[row_t1];
    }

    async_copy_wait_all();
    __syncthreads();

    // === Phase WU-D: W = T_inv @ (beta * gamma * K) ===
    matmul_and_store(W + out_base, bg_kt);
}

template <int S>
cudaError_t launch_typed(const prepare_wy_wu_config & cfg,
                         dim3 grid, dim3 block, gdn::head_map qk_map) {
    constexpr int smem_bytes = kernel_dims<S>::SMEM_FLOATS * (int) sizeof(float);

    cudaError_t err = cudaFuncSetAttribute(
        prepare_wy_wu_gdn_kernel<S>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);
    if (err != cudaSuccess) return err;

    // 0 means "use packed default" -- materialise here so the kernel only
    // ever sees a real stride.
    const int64_t k_stride_t = (cfg.k_stride_t_floats != 0)
                                ? cfg.k_stride_t_floats
                                : (int64_t) cfg.H_qk * cfg.S;
    const int64_t v_stride_t = (cfg.v_stride_t_floats != 0)
                                ? cfg.v_stride_t_floats
                                : (int64_t) cfg.H_v * cfg.S;

    const int64_t L_total = (cfg.L_total != 0) ? cfg.L_total : cfg.L;

    prepare_wy_wu_gdn_kernel<S><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.k, cfg.v, cfg.g_cumsum, cfg.beta,
        cfg.W, cfg.U,
        cfg.L, L_total, cfg.H_v, qk_map, k_stride_t, v_stride_t);
    return cudaGetLastError();
}

}  // namespace

cudaError_t launch_prepare_wy_wu(const prepare_wy_wu_config & cfg) {
    detail::stage_validator v{"launch_prepare_wy_wu",
                              cfg.S, cfg.H_qk, cfg.H_v, cfg.L, cfg.B, cfg.kda};
    GDN_PROPAGATE(v.check_shape());
    GDN_PROPAGATE(v.check_gdn_full_chunks());
    if (cfg.k == nullptr || cfg.v == nullptr || cfg.g_cumsum == nullptr ||
        cfg.beta == nullptr || cfg.W == nullptr || cfg.U == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (cfg.kda && (cfg.q == nullptr || cfg.Aqk == nullptr || cfg.kg == nullptr)) {
        return cudaErrorInvalidValue;
    }
    if (cfg.kda) return cudaErrorNotYetImplemented;  // Phase C

    const auto    qk_map = gdn::head_map::of((int) cfg.H_qk, (int) cfg.H_v);
    const int64_t NT     = (cfg.L + BT - 1) / BT;
    const int64_t bh     = cfg.B * cfg.H_v;
    GDN_PROPAGATE(v.check_grid(NT, bh));

    const dim3 grid((unsigned) NT, (unsigned) bh, 1);
    const dim3 block(THREADS, 1, 1);

    switch (cfg.S) {
        case 16:  return launch_typed<16> (cfg, grid, block, qk_map);
        case 32:  return launch_typed<32> (cfg, grid, block, qk_map);
        case 64:  return launch_typed<64> (cfg, grid, block, qk_map);
        case 128: return launch_typed<128>(cfg, grid, block, qk_map);
        default:  return cudaErrorInvalidValue;  // check_shape already filtered
    }
}

}  // namespace chunked_gdn::stages
