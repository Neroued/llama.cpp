#include "gated_delta_net.cuh"

namespace gdn_v2 {

template <int n> static __device__ __forceinline__ void vec_load(float (&reg)[n], const float * ptr) {
    if constexpr (n == 4) {
        *reinterpret_cast<float4 *>(&reg[0]) = *reinterpret_cast<const float4 *>(ptr);
    } else if constexpr (n == 2) {
        *reinterpret_cast<float2 *>(&reg[0]) = *reinterpret_cast<const float2 *>(ptr);
    } else if constexpr (n == 8) {
        *reinterpret_cast<float4 *>(&reg[0]) = *reinterpret_cast<const float4 *>(ptr);
        *reinterpret_cast<float4 *>(&reg[4]) = *reinterpret_cast<const float4 *>(ptr + 4);
    } else {
        reg[0] = ptr[0];
    }
}

template <int n> static __device__ __forceinline__ void vec_store(float * ptr, const float (&reg)[n]) {
    if constexpr (n == 4) {
        *reinterpret_cast<float4 *>(ptr) = *reinterpret_cast<const float4 *>(&reg[0]);
    } else if constexpr (n == 2) {
        *reinterpret_cast<float2 *>(ptr) = *reinterpret_cast<const float2 *>(&reg[0]);
    } else if constexpr (n == 8) {
        *reinterpret_cast<float4 *>(ptr)     = *reinterpret_cast<const float4 *>(&reg[0]);
        *reinterpret_cast<float4 *>(ptr + 4) = *reinterpret_cast<const float4 *>(&reg[4]);
    } else {
        ptr[0] = reg[0];
    }
}

template <int n> static __device__ __forceinline__ void vec_zero(float (&reg)[n]) {
    if constexpr (n == 4) {
        *reinterpret_cast<float4 *>(&reg[0]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else if constexpr (n == 2) {
        *reinterpret_cast<float2 *>(&reg[0]) = make_float2(0.0f, 0.0f);
    } else if constexpr (n == 8) {
        *reinterpret_cast<float4 *>(&reg[0]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *reinterpret_cast<float4 *>(&reg[4]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        reg[0] = 0.0f;
    }
}

constexpr int d_v_per_warp = 4;
constexpr int num_warps    = 4;
constexpr int block_dv     = num_warps * d_v_per_warp;  // 16

template <int S_v, bool KDA>
__global__ void __launch_bounds__(ggml_cuda_get_physical_warp_size() * num_warps, 2) gated_delta_net_v2_cuda(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ g,
    const float * __restrict__ beta,
    const float * __restrict__ curr_state,
    float * __restrict__ dst,
    int64_t H,
    int64_t n_tokens,
    int64_t n_seqs,
    int64_t sq1,
    int64_t sq2,
    int64_t sq3,
    int64_t sv1,
    int64_t sv2,
    int64_t sv3,
    int64_t sb1,
    int64_t sb2,
    int64_t sb3,
    uint3   neqk1_magic,
    uint3   rq3_magic,
    float   scale) {
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    constexpr int active_lanes  = (S_v < warp_size) ? S_v : warp_size;
    constexpr int d_qk_per_lane = S_v / active_lanes;
    static_assert(d_qk_per_lane >= 1, "v2 requires S_v >= 1");
    static_assert(S_v % active_lanes == 0, "S_v must be a multiple of active_lanes");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp_id  = threadIdx.y;

    const uint32_t dv_base = blockIdx.z * block_dv + warp_id * d_v_per_warp;
    if (dv_base >= S_v) {
        return;
    }

    const uint32_t dqk_base = lane * d_qk_per_lane;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    // dst layout: [attn_data | final_state]
    const int64_t attn_score_elems = (int64_t) S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state_out        = dst + attn_score_elems;

    const int64_t state_off = ((int64_t) sequence * H + h_idx) * S_v * S_v;
    state_out += state_off;
    curr_state += state_off;
    attn_data += ((int64_t) sequence * n_tokens * H + h_idx) * S_v;

    // Load state into register tile: d_v_per_warp rows x d_qk_per_lane columns
    float s_tile[d_v_per_warp][d_qk_per_lane];

#pragma unroll
    for (int r = 0; r < d_v_per_warp; ++r) {
        const uint32_t dv    = dv_base + r;
        const float *  row   = curr_state + (int64_t) dv * S_v + dqk_base;
        bool           valid = (dv < S_v);
        if constexpr (S_v < warp_size) {
            valid = valid && (lane < active_lanes);
        }
        if (valid) {
            vec_load<d_qk_per_lane>(s_tile[r], row);
        } else {
            vec_zero<d_qk_per_lane>(s_tile[r]);
        }
    }

    // Prefetch k for t=0
    float k_reg[d_qk_per_lane];
    {
        const float * k_t = k + (int64_t) iq3 * sq3 + (int64_t) iq1 * sq1;
        if constexpr (S_v < warp_size) {
            if (lane < active_lanes) {
                k_reg[0] = k_t[lane];
            } else {
                k_reg[0] = 0.0f;
            }
        } else {
            vec_load<d_qk_per_lane>(k_reg, &k_t[dqk_base]);
        }
    }

    for (int t = 0; t < n_tokens; ++t) {
        const float * v_t = v + (int64_t) sequence * sv3 + (int64_t) t * sv2 + (int64_t) h_idx * sv1;

        const int64_t gb_off   = (int64_t) sequence * sb3 + (int64_t) t * sb2 + (int64_t) h_idx * sb1;
        const float   beta_val = beta[gb_off];

        // alpha: KDA -> per-lane vector, non-KDA -> warp-wide scalar
        float alpha_lane[d_qk_per_lane];
        float alpha_scalar = 0.0f;
        if constexpr (KDA) {
            const float * g_t = g + gb_off * S_v;
            float         g_reg[d_qk_per_lane];
            if constexpr (S_v < warp_size) {
                if (lane < active_lanes) {
                    g_reg[0] = g_t[lane];
                } else {
                    g_reg[0] = 0.0f;
                }
            } else {
                vec_load<d_qk_per_lane>(g_reg, &g_t[dqk_base]);
            }
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                alpha_lane[c] = expf(g_reg[c]);
            }
        } else {
            alpha_scalar = expf(g[gb_off]);
        }

        float v_local = 0.0f;
        if (lane < d_v_per_warp && dv_base + lane < S_v) {
            v_local = v_t[dv_base + lane];
        }

#pragma unroll
        for (int r = 0; r < d_v_per_warp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                if constexpr (KDA) {
                    partial += alpha_lane[c] * s_tile[r][c] * k_reg[c];
                } else {
                    partial += s_tile[r][c] * k_reg[c];
                }
            }
            partial = warp_reduce_sum<warp_size>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r);
            const float delta = beta_val * (v_r - (KDA ? 1.0f : alpha_scalar) * partial);

#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                if constexpr (KDA) {
                    s_tile[r][c] = alpha_lane[c] * s_tile[r][c] + delta * k_reg[c];
                } else {
                    s_tile[r][c] = alpha_scalar * s_tile[r][c] + delta * k_reg[c];
                }
            }
        }

        // Prefetch k for next token
        if (t + 1 < n_tokens) {
            const float * k_t1 = k + (int64_t) iq3 * sq3 + (int64_t) (t + 1) * sq2 + (int64_t) iq1 * sq1;
            if constexpr (S_v < warp_size) {
                if (lane < active_lanes) {
                    k_reg[0] = k_t1[lane];
                } else {
                    k_reg[0] = 0.0f;
                }
            } else {
                vec_load<d_qk_per_lane>(k_reg, &k_t1[dqk_base]);
            }
        }

        // Stage B: attention output
        const float * q_t = q + (int64_t) iq3 * sq3 + (int64_t) t * sq2 + (int64_t) iq1 * sq1;
        float         q_reg[d_qk_per_lane];
        if constexpr (S_v < warp_size) {
            if (lane < active_lanes) {
                q_reg[0] = q_t[lane];
            } else {
                q_reg[0] = 0.0f;
            }
        } else {
            vec_load<d_qk_per_lane>(q_reg, &q_t[dqk_base]);
        }

        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < d_v_per_warp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                partial += s_tile[r][c] * q_reg[c];
            }
            partial = warp_reduce_sum<warp_size>(partial);
            if (lane == r) {
                attn_val = partial;
            }
        }

        if (lane < d_v_per_warp && dv_base + lane < S_v) {
            float * attn_t         = attn_data + (int64_t) t * S_v * H;
            attn_t[dv_base + lane] = attn_val * scale;
        }
    }

#pragma unroll
    for (int r = 0; r < d_v_per_warp; ++r) {
        const uint32_t dv    = dv_base + r;
        bool           valid = (dv < S_v);
        if constexpr (S_v < warp_size) {
            valid = valid && (lane < active_lanes);
        }
        if (valid) {
            float * row = state_out + (int64_t) dv * S_v + dqk_base;
            vec_store<d_qk_per_lane>(row, s_tile[r]);
        }
    }
}

}  // namespace gdn_v2

template <bool KDA>
void launch_gated_delta_net_v2(const float * q_d,
                               const float * k_d,
                               const float * v_d,
                               const float * g_d,
                               const float * b_d,
                               const float * s_d,
                               float *       dst_d,
                               int64_t       S_v,
                               int64_t       H,
                               int64_t       n_tokens,
                               int64_t       n_seqs,
                               int64_t       sq1,
                               int64_t       sq2,
                               int64_t       sq3,
                               int64_t       sv1,
                               int64_t       sv2,
                               int64_t       sv3,
                               int64_t       sb1,
                               int64_t       sb2,
                               int64_t       sb3,
                               int64_t       neqk1,
                               int64_t       rq3,
                               float         scale,
                               cudaStream_t  stream) {
    using namespace gdn_v2;

    const int   warp_size   = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int   n_block_dv  = (int) ((S_v + block_dv - 1) / block_dv);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    dim3 grid_dims((unsigned) H, (unsigned) n_seqs, (unsigned) n_block_dv);
    dim3 block_dims(warp_size, num_warps, 1);

    switch (S_v) {
        case 16:
            gated_delta_net_v2_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_v2_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale);
            break;
        case 64:
            gated_delta_net_v2_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale);
            break;
        case 128:
            gated_delta_net_v2_cuda<128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale);
            break;
        case 256:
            gated_delta_net_v2_cuda<256, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale);
            break;
        default:
            GGML_ABORT("gated_delta_net_v2: unsupported S_v=%lld (need 16/32/64/128/256)", (long long) S_v);
    }
}

template void launch_gated_delta_net_v2<true>(const float *,
                                              const float *,
                                              const float *,
                                              const float *,
                                              const float *,
                                              const float *,
                                              float *,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              int64_t,
                                              float,
                                              cudaStream_t);

template void launch_gated_delta_net_v2<false>(const float *,
                                               const float *,
                                               const float *,
                                               const float *,
                                               const float *,
                                               const float *,
                                               float *,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               int64_t,
                                               float,
                                               cudaStream_t);
