// gated_delta_net_v2.cu
//
// Alternative CUDA kernel for the fused Gated Delta Net AR op.
//
// Differences vs the baseline kernel in gated_delta_net.cu:
//   - row-per-warp tiling: each warp owns D_V_PER_WARP=4 rows of the v dimension
//     (instead of one column).
//   - lane-stride is along the d_qk dimension in contiguous chunks of
//     D_QK_PER_LANE = S_v / warp_size elements.
//   - state update and attention are split into two stages; per stage performs
//     one warp_reduce_sum per v-row tile.
//   - K prefetch: next token's K loaded ahead of attention stage so its memory
//     latency overlaps with output store / pointer advancement.
//   - float4/float2 vectorized loads for contiguous state/K/Q/alpha elements.
//   - S_v supported: 16 (16 active lanes), 32, 64, 128, 256.
//
// Trade-offs vs v1 (NV 32-warp):
//   - Reduce count per token: v1 = 2; v2 = 2 * D_V_PER_WARP = 8.
//   - Memory access: lane-stride D_QK_PER_LANE is friendlier to vec load/store
//     and to multi-element K prefetch.
//   - block count: v2 has ceil(S_v/16) vs v1's ceil(S_v/4); fewer blocks but
//     more work per block (register-heavy, good ILP).
//
// Selected at runtime via the GGML_GDN_AR_V2=1 environment variable.
//
// F32 only. KDA on/off supported.

#include "gated_delta_net.cuh"

namespace gdn_v2 {

constexpr int WARP_SIZE_V2 = 32;
constexpr int NUM_WARPS    = 4;     // warps per block
constexpr int D_V_PER_WARP = 4;     // v-rows owned by each warp
constexpr int BLOCK_DV     = NUM_WARPS * D_V_PER_WARP;  // 16

template <int S_v, bool KDA>
__global__ void __launch_bounds__(WARP_SIZE_V2 * NUM_WARPS, 2)
gated_delta_net_v2_cuda(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ g,
    const float * __restrict__ beta,
    const float * __restrict__ curr_state,
    float *       __restrict__ dst,
    int64_t H, int64_t n_tokens, int64_t n_seqs,
    int64_t sq1, int64_t sq2, int64_t sq3,
    int64_t sv1, int64_t sv2, int64_t sv3,
    int64_t sb1, int64_t sb2, int64_t sb3,
    uint3   neqk1_magic,
    uint3   rq3_magic,
    float   scale)
{
    constexpr int ACTIVE_LANES   = (S_v < WARP_SIZE_V2) ? S_v : WARP_SIZE_V2;
    constexpr int D_QK_PER_LANE  = S_v / ACTIVE_LANES;
    static_assert(D_QK_PER_LANE >= 1, "v2 requires S_v >= 1");
    static_assert(S_v % ACTIVE_LANES == 0, "S_v must be a multiple of ACTIVE_LANES");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp_id  = threadIdx.y;

    const uint32_t dv_base  = blockIdx.z * BLOCK_DV + warp_id * D_V_PER_WARP;
    if (dv_base >= S_v) return;

    const uint32_t dqk_base = lane * D_QK_PER_LANE;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    // dst layout: [attn_data | final_state]
    const int64_t attn_score_elems = (int64_t) S_v * H * n_tokens * n_seqs;
    float * attn_data_base = dst;
    float * state_out      = dst + attn_score_elems;

    const int64_t state_off = ((int64_t) sequence * H + h_idx) * S_v * S_v;
    state_out      += state_off;
    curr_state     += state_off;
    attn_data_base += ((int64_t) sequence * n_tokens * H + h_idx) * S_v;

    // === STATE TILE: D_V_PER_WARP v-rows × D_QK_PER_LANE columns ==========
    float s_tile[D_V_PER_WARP][D_QK_PER_LANE];

    #pragma unroll
    for (int r = 0; r < D_V_PER_WARP; ++r) {
        const uint32_t dv = dv_base + r;
        const float * row = curr_state + (int64_t) dv * S_v + dqk_base;
        bool valid = (dv < S_v);
        if constexpr (S_v < WARP_SIZE_V2) { valid = valid && (lane < ACTIVE_LANES); }
        if (valid) {
            if constexpr (D_QK_PER_LANE == 4) {
                *reinterpret_cast<float4*>(&s_tile[r][0]) = *reinterpret_cast<const float4*>(row);
            } else if constexpr (D_QK_PER_LANE == 2) {
                *reinterpret_cast<float2*>(&s_tile[r][0]) = *reinterpret_cast<const float2*>(row);
            } else if constexpr (D_QK_PER_LANE == 8) {
                *reinterpret_cast<float4*>(&s_tile[r][0]) = *reinterpret_cast<const float4*>(row);
                *reinterpret_cast<float4*>(&s_tile[r][4]) = *reinterpret_cast<const float4*>(row + 4);
            } else {
                s_tile[r][0] = row[0];
            }
        } else {
            if constexpr (D_QK_PER_LANE == 4) {
                *reinterpret_cast<float4*>(&s_tile[r][0]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else if constexpr (D_QK_PER_LANE == 2) {
                *reinterpret_cast<float2*>(&s_tile[r][0]) = make_float2(0.0f, 0.0f);
            } else if constexpr (D_QK_PER_LANE == 8) {
                *reinterpret_cast<float4*>(&s_tile[r][0]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                *reinterpret_cast<float4*>(&s_tile[r][4]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            } else {
                s_tile[r][0] = 0.0f;
            }
        }
    }

    // === K prefetch t=0 ==================================================
    float k_reg[D_QK_PER_LANE];
    {
        const float * k_t = k + (int64_t) iq3 * sq3 + (int64_t) iq1 * sq1;
        if constexpr (S_v < WARP_SIZE_V2) {
            if (lane < ACTIVE_LANES) { k_reg[0] = k_t[lane]; }
            else                     { k_reg[0] = 0.0f; }
        } else {
            if constexpr (D_QK_PER_LANE == 4) {
                *reinterpret_cast<float4*>(&k_reg[0]) = *reinterpret_cast<const float4*>(&k_t[dqk_base]);
            } else if constexpr (D_QK_PER_LANE == 2) {
                *reinterpret_cast<float2*>(&k_reg[0]) = *reinterpret_cast<const float2*>(&k_t[dqk_base]);
            } else if constexpr (D_QK_PER_LANE == 8) {
                *reinterpret_cast<float4*>(&k_reg[0]) = *reinterpret_cast<const float4*>(&k_t[dqk_base]);
                *reinterpret_cast<float4*>(&k_reg[4]) = *reinterpret_cast<const float4*>(&k_t[dqk_base + 4]);
            } else {
                k_reg[0] = k_t[dqk_base];
            }
        }
    }

    // === TOKEN LOOP ======================================================
    for (int t = 0; t < n_tokens; ++t) {
        const float * v_t = v + (int64_t) sequence * sv3
                              + (int64_t) t * sv2
                              + (int64_t) h_idx * sv1;
        const int64_t gb_off = (int64_t) sequence * sb3
                             + (int64_t) t * sb2
                             + (int64_t) h_idx * sb1;
        const float beta_val = beta[gb_off];

        // alpha: KDA -> per-lane vector; non-KDA -> warp-wide scalar
        float alpha_lane[D_QK_PER_LANE];
        float alpha_scalar = 0.0f;
        if constexpr (KDA) {
            const float * g_t = g + gb_off * S_v;
            float g_reg[D_QK_PER_LANE];
            if constexpr (S_v < WARP_SIZE_V2) {
                if (lane < ACTIVE_LANES) { g_reg[0] = g_t[lane]; }
                else                     { g_reg[0] = 0.0f; }
            } else {
                if constexpr (D_QK_PER_LANE == 4) {
                    *reinterpret_cast<float4*>(&g_reg[0]) = *reinterpret_cast<const float4*>(&g_t[dqk_base]);
                } else if constexpr (D_QK_PER_LANE == 2) {
                    *reinterpret_cast<float2*>(&g_reg[0]) = *reinterpret_cast<const float2*>(&g_t[dqk_base]);
                } else if constexpr (D_QK_PER_LANE == 8) {
                    *reinterpret_cast<float4*>(&g_reg[0]) = *reinterpret_cast<const float4*>(&g_t[dqk_base]);
                    *reinterpret_cast<float4*>(&g_reg[4]) = *reinterpret_cast<const float4*>(&g_t[dqk_base + 4]);
                } else {
                    g_reg[0] = g_t[dqk_base];
                }
            }
            #pragma unroll
            for (int c = 0; c < D_QK_PER_LANE; ++c) {
                alpha_lane[c] = expf(g_reg[c]);
            }
        } else {
            alpha_scalar = expf(g[gb_off]);
        }

        // v_local: lane in [0, D_V_PER_WARP) holds v[dv_base + lane]
        float v_local = 0.0f;
        if (lane < D_V_PER_WARP && dv_base + lane < S_v) {
            v_local = v_t[dv_base + lane];
        }

        // === STAGE A: state update (one warp_reduce per r) ================
        #pragma unroll
        for (int r = 0; r < D_V_PER_WARP; ++r) {
            float partial = 0.0f;
            #pragma unroll
            for (int c = 0; c < D_QK_PER_LANE; ++c) {
                if constexpr (KDA) {
                    partial += alpha_lane[c] * s_tile[r][c] * k_reg[c];
                } else {
                    partial += s_tile[r][c] * k_reg[c];
                }
            }
            partial = warp_reduce_sum<WARP_SIZE_V2>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r);
            const float delta = beta_val *
                (v_r - (KDA ? 1.0f : alpha_scalar) * partial);

            #pragma unroll
            for (int c = 0; c < D_QK_PER_LANE; ++c) {
                if constexpr (KDA) {
                    s_tile[r][c] = alpha_lane[c] * s_tile[r][c]
                                 + delta * k_reg[c];
                } else {
                    s_tile[r][c] = alpha_scalar * s_tile[r][c]
                                 + delta * k_reg[c];
                }
            }
        }

        // === K prefetch t+1 ===============================================
        if (t + 1 < n_tokens) {
            const float * k_t1 = k + (int64_t) iq3 * sq3
                                   + (int64_t) (t + 1) * sq2
                                   + (int64_t) iq1 * sq1;
            if constexpr (S_v < WARP_SIZE_V2) {
                if (lane < ACTIVE_LANES) { k_reg[0] = k_t1[lane]; }
                else                     { k_reg[0] = 0.0f; }
            } else {
                if constexpr (D_QK_PER_LANE == 4) {
                    *reinterpret_cast<float4*>(&k_reg[0]) = *reinterpret_cast<const float4*>(&k_t1[dqk_base]);
                } else if constexpr (D_QK_PER_LANE == 2) {
                    *reinterpret_cast<float2*>(&k_reg[0]) = *reinterpret_cast<const float2*>(&k_t1[dqk_base]);
                } else if constexpr (D_QK_PER_LANE == 8) {
                    *reinterpret_cast<float4*>(&k_reg[0]) = *reinterpret_cast<const float4*>(&k_t1[dqk_base]);
                    *reinterpret_cast<float4*>(&k_reg[4]) = *reinterpret_cast<const float4*>(&k_t1[dqk_base + 4]);
                } else {
                    k_reg[0] = k_t1[dqk_base];
                }
            }
        }

        // === STAGE B: attn (one warp_reduce per r) ========================
        const float * q_t = q + (int64_t) iq3 * sq3
                              + (int64_t) t * sq2
                              + (int64_t) iq1 * sq1;
        float q_reg[D_QK_PER_LANE];
        if constexpr (S_v < WARP_SIZE_V2) {
            if (lane < ACTIVE_LANES) { q_reg[0] = q_t[lane]; }
            else                     { q_reg[0] = 0.0f; }
        } else {
            if constexpr (D_QK_PER_LANE == 4) {
                *reinterpret_cast<float4*>(&q_reg[0]) = *reinterpret_cast<const float4*>(&q_t[dqk_base]);
            } else if constexpr (D_QK_PER_LANE == 2) {
                *reinterpret_cast<float2*>(&q_reg[0]) = *reinterpret_cast<const float2*>(&q_t[dqk_base]);
            } else if constexpr (D_QK_PER_LANE == 8) {
                *reinterpret_cast<float4*>(&q_reg[0]) = *reinterpret_cast<const float4*>(&q_t[dqk_base]);
                *reinterpret_cast<float4*>(&q_reg[4]) = *reinterpret_cast<const float4*>(&q_t[dqk_base + 4]);
            } else {
                q_reg[0] = q_t[dqk_base];
            }
        }

        float my_out = 0.0f;
        #pragma unroll
        for (int r = 0; r < D_V_PER_WARP; ++r) {
            float partial = 0.0f;
            #pragma unroll
            for (int c = 0; c < D_QK_PER_LANE; ++c) {
                partial += s_tile[r][c] * q_reg[c];
            }
            partial = warp_reduce_sum<WARP_SIZE_V2>(partial);
            if (lane == r) my_out = partial;
        }

        if (lane < D_V_PER_WARP && dv_base + lane < S_v) {
            float * attn_t = attn_data_base + (int64_t) t * S_v * H;
            attn_t[dv_base + lane] = my_out * scale;
        }
    }

    // === FINAL STATE WRITEBACK ===========================================
    #pragma unroll
    for (int r = 0; r < D_V_PER_WARP; ++r) {
        const uint32_t dv = dv_base + r;
        bool valid = (dv < S_v);
        if constexpr (S_v < WARP_SIZE_V2) { valid = valid && (lane < ACTIVE_LANES); }
        if (valid) {
            float * row = state_out + (int64_t) dv * S_v + dqk_base;
            if constexpr (D_QK_PER_LANE == 4) {
                *reinterpret_cast<float4*>(row) = *reinterpret_cast<const float4*>(&s_tile[r][0]);
            } else if constexpr (D_QK_PER_LANE == 2) {
                *reinterpret_cast<float2*>(row) = *reinterpret_cast<const float2*>(&s_tile[r][0]);
            } else if constexpr (D_QK_PER_LANE == 8) {
                *reinterpret_cast<float4*>(row)     = *reinterpret_cast<const float4*>(&s_tile[r][0]);
                *reinterpret_cast<float4*>(row + 4) = *reinterpret_cast<const float4*>(&s_tile[r][4]);
            } else {
                row[0] = s_tile[r][0];
            }
        }
    }
}

} // namespace gdn_v2

template <bool KDA>
void launch_gated_delta_net_v2(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float *       dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream)
{
    using namespace gdn_v2;

    const int n_block_dv = (int) ((S_v + BLOCK_DV - 1) / BLOCK_DV);
    dim3 grid_dims((unsigned) H, (unsigned) n_seqs, (unsigned) n_block_dv);
    dim3 block_dims(WARP_SIZE_V2, NUM_WARPS, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16:
            gated_delta_net_v2_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_v2_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64:
            gated_delta_net_v2_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 128:
            gated_delta_net_v2_cuda<128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 256:
            gated_delta_net_v2_cuda<256, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        default:
            GGML_ABORT("gated_delta_net_v2: unsupported S_v=%lld (need 16/32/64/128/256)",
                       (long long) S_v);
    }
}

// Explicit template instantiations to make the symbol available to the
// dispatch site in gated_delta_net.cu (forward-declared there).
template void launch_gated_delta_net_v2<true >(
    const float *, const float *, const float *,
    const float *, const float *, const float *, float *,
    int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    int64_t, int64_t,
    float, cudaStream_t);

template void launch_gated_delta_net_v2<false>(
    const float *, const float *, const float *,
    const float *, const float *, const float *, float *,
    int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    int64_t, int64_t,
    float, cudaStream_t);
