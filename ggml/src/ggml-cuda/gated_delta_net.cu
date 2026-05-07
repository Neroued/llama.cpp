#include "gated_delta_net.cuh"

#include "gdn_chunked/chunked_gdn.cuh"

// AR (autoregressive / recurrent-form) Gated DeltaNet CUDA kernel.
//
// Row-per-warp implementation ported from the standalone ar_gdn project:
//   ~/chunked_gdn/ar/ar_gdn.cu
// Replaces the previous column-per-warp kernel; same algorithm, faster.
//
// Per block:
//   blockIdx.x = h_idx  in [0, H_v)        (one V head)
//   blockIdx.y = seq    in [0, B)
//   blockIdx.z = block_dv slice over S_v rows
//   4 warps per block, each owning d_v_per_warp = 4 V-rows.
//   Lanes within a warp shard the QK axis (d_qk_per_lane = S_v / active_lanes).
//   The (S_v x S_v) state tile is held in registers across the L token axis.

namespace {

constexpr int d_v_per_warp = 4;
constexpr int num_warps    = 4;
constexpr int block_dv     = num_warps * d_v_per_warp;  // 16

template <int S_v, bool KDA>
__global__ void __launch_bounds__(WARP_SIZE * num_warps, 2)
gated_delta_net_cuda(const float * __restrict__ q,
                     const float * __restrict__ k,
                     const float * __restrict__ v,
                     const float * __restrict__ g,
                     const float * __restrict__ beta,
                     const float * __restrict__ curr_state,
                     float * __restrict__       attn_data,
                     float * __restrict__       state_out,
                     int64_t H,
                     int64_t n_tokens,
                     // Per-sequence stride (in tokens) of the attn_data
                     // output buffer. Equal to n_tokens for a stand-alone
                     // call; for the chunked split (chunked covers [0,n_full)
                     // and AR covers [n_full, n_tokens)) this is the FULL
                     // sequence length so each batch's tail tokens land
                     // at attn_data + seq * n_tokens_full * H * S_v +
                     // (n_full + t) * H * S_v -- the host pre-shifts
                     // attn_data by n_full * H * S_v before the launch.
                     int64_t n_tokens_attn_stride,
                     int64_t n_seqs,
                     int64_t sq1,
                     int64_t sq2,
                     int64_t sq3,
                     int64_t sk1,
                     int64_t sk2,
                     int64_t sk3,
                     int64_t sv1,
                     int64_t sv2,
                     int64_t sv3,
                     int64_t sb1,
                     int64_t sb2,
                     int64_t sb3,
                     uint3   neqk1_magic,
                     uint3   rq3_magic,
                     float   scale) {
    constexpr int warp_size     = WARP_SIZE;
    constexpr int active_lanes  = (S_v < warp_size) ? S_v : warp_size;
    constexpr int d_qk_per_lane = S_v / active_lanes;
    static_assert(d_qk_per_lane >= 1, "S_v must >= 1");
    static_assert(S_v % active_lanes == 0, "S_v must be a multiple of active_lanes");
    static_assert(S_v % block_dv == 0, "S_v must be a multiple of block_dv");

    GGML_UNUSED(n_seqs);

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp_id  = threadIdx.y;

    // each warp owns d_v_per_warp rows of the state matrix; warp's lanes shard the QK axis
    const uint32_t dv_base  = blockIdx.z * block_dv + warp_id * d_v_per_warp;
    const uint32_t dqk_base = lane * d_qk_per_lane;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    // attn_data and state_out are independent base pointers (state_out comes
    // from dst->src[6], a view of an ssm-state cache).
    const int64_t state_off = ((int64_t) sequence * H + h_idx) * S_v * S_v;
    state_out  += state_off;
    curr_state += state_off;
    attn_data  += ((int64_t) sequence * n_tokens_attn_stride * H + h_idx) * S_v;

    auto load_qk_lane = [&] __device__(float (&reg)[d_qk_per_lane], const float * base) {
        if constexpr (S_v < warp_size) {
            reg[0] = (lane < active_lanes) ? base[lane] : 0.0f;
        } else {
            ggml_cuda_memcpy_1<d_qk_per_lane * sizeof(float)>(reg, base + dqk_base);
        }
    };
    auto store_qk_lane = [&] __device__(const float (&reg)[d_qk_per_lane], float * base) {
        if constexpr (S_v < warp_size) {
            if (lane < active_lanes) {
                base[lane] = reg[0];
            }
        } else {
            ggml_cuda_memcpy_1<d_qk_per_lane * sizeof(float)>(base + dqk_base, reg);
        }
    };

    // state is stored transposed: M[r][c] = S[c][r]
    __align__(16) float s_tile[d_v_per_warp][d_qk_per_lane];
#pragma unroll
    for (int r = 0; r < d_v_per_warp; ++r) {
        load_qk_lane(s_tile[r], curr_state + (int64_t) (dv_base + r) * S_v);
    }

    __align__(16) float k_reg[d_qk_per_lane];
    load_qk_lane(k_reg, k + (int64_t) iq3 * sk3 + (int64_t) iq1 * sk1);

    for (int t = 0; t < n_tokens; ++t) {
        const float * v_t = v + (int64_t) sequence * sv3 + (int64_t) t * sv2 + (int64_t) h_idx * sv1;

        const int64_t gb_off   = (int64_t) sequence * sb3 + (int64_t) t * sb2 + (int64_t) h_idx * sb1;
        const float   beta_val = beta[gb_off];

        __align__(16) float alpha_lane[d_qk_per_lane];
        float               alpha_scalar = 0.0f;
        if constexpr (KDA) {
            __align__(16) float g_reg[d_qk_per_lane];
            load_qk_lane(g_reg, g + gb_off * S_v);
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                alpha_lane[c] = expf(g_reg[c]);
            }
        } else {
            alpha_scalar = expf(g[gb_off]);
        }

        // only first d_v_per_warp lanes hold a real v[dv_base + lane]; broadcast via __shfl_sync
        float v_local = 0.0f;
        if (lane < d_v_per_warp) {
            v_local = v_t[dv_base + lane];
        }

        // stage A: state update
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

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, warp_size);
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

        // prefetch k for next token while issuing the q load for this token
        if (t + 1 < n_tokens) {
            load_qk_lane(k_reg, k + (int64_t) iq3 * sk3 + (int64_t) (t + 1) * sk2 + (int64_t) iq1 * sk1);
        }

        __align__(16) float q_reg[d_qk_per_lane];
        load_qk_lane(q_reg, q + (int64_t) iq3 * sq3 + (int64_t) t * sq2 + (int64_t) iq1 * sq1);

        // stage B: attention output
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

        if (lane < d_v_per_warp) {
            float * attn_t         = attn_data + (int64_t) t * S_v * H;
            attn_t[dv_base + lane] = attn_val * scale;
        }
    }

    // store final state
#pragma unroll
    for (int r = 0; r < d_v_per_warp; ++r) {
        store_qk_lane(s_tile[r], state_out + (int64_t) (dv_base + r) * S_v);
    }
}

template <bool KDA>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * attn_out_d, float * state_out_d,
        int64_t S_v,   int64_t H, int64_t n_tokens,
        // Per-sequence stride (in tokens) of attn_out_d. Pass 0 for the
        // default packed case (== n_tokens). Chunked-tail dispatch passes
        // n_tokens_full so AR-tail picks the right per-batch stride; the
        // host pre-shifts attn_out_d by n_full * H * S_v in that case.
        int64_t n_tokens_attn_stride,
        int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sk1,   int64_t sk2, int64_t sk3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    const int n_block_dv = (int) ((S_v + block_dv - 1) / block_dv);

    const int64_t n_tk_attn = (n_tokens_attn_stride != 0) ? n_tokens_attn_stride : n_tokens;

    const uint3 neqk1_magic = init_fastdiv_values((uint64_t) neqk1);
    const uint3 rq3_magic   = init_fastdiv_values((uint64_t) rq3);

    dim3 grid_dims((unsigned) H, (unsigned) n_seqs, (unsigned) n_block_dv);
    dim3 block_dims(WARP_SIZE, num_warps, 1);

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, attn_out_d, state_out_d, H,
                n_tokens, n_tk_attn, n_seqs,
                sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, attn_out_d, state_out_d, H,
                n_tokens, n_tk_attn, n_seqs,
                sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64:
            gated_delta_net_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, attn_out_d, state_out_d, H,
                n_tokens, n_tk_attn, n_seqs,
                sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 128:
            gated_delta_net_cuda<128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, attn_out_d, state_out_d, H,
                n_tokens, n_tk_attn, n_seqs,
                sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

}  // anonymous namespace

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q         = dst->src[0];
    ggml_tensor * src_k         = dst->src[1];
    ggml_tensor * src_v         = dst->src[2];
    ggml_tensor * src_g         = dst->src[3];
    ggml_tensor * src_beta      = dst->src[4];
    ggml_tensor * src_state     = dst->src[5];
    ggml_tensor * src_state_out = dst->src[6];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;
    const float * s_d = (const float *) src_state->data;

    float * attn_out_d  = (float *) dst->data;
    float * state_out_d = (float *) src_state_out->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));
    GGML_ASSERT(ggml_is_contiguous(src_state_out));
    GGML_ASSERT(src_state_out->view_src != nullptr);

    // strides in floats. q and k carry independent strides (the new kernel
    // accepts q/k sliced from a [Q | K | V] mix tensor with different
    // per-token strides).
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sk1 = nbk1 / sizeof(float);
    const int64_t sk2 = nbk2 / sizeof(float);
    const int64_t sk3 = nbk3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // Decide chunked eligibility. Chunked path requires:
    //   - sm_80+ (cp.async + tf32 mma)
    //   - !KDA (chunked KDA is NYI; AR handles all KDA)
    //   - S_v in {16, 32, 64, 128}
    //   - n_tokens >= 64 (else there is no full chunk)
    //   - g/beta packed (chunked stages assume packed token stride for both)
    //   - rq3 == 1 (no q/v batch broadcast; chunked treats q-batch == v-batch)
    //   - state_in / state_out contiguous (asserted above for state_out)
    const int      cc            = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool     chunked_arch  = ampere_mma_available(cc);
    const bool     packed_g_beta = ggml_is_contiguous(src_g) && ggml_is_contiguous(src_beta);
    const bool     supported_s   = (S_v == 16 || S_v == 32 || S_v == 64 || S_v == 128);
    const bool     chunked_ok    = !kda
                                && chunked_arch
                                && supported_s
                                && n_tokens >= 64
                                && rq3 == 1
                                && packed_g_beta;

    if (!chunked_ok) {
        if (kda) {
            launch_gated_delta_net<true>(q_d, k_d, v_d, g_d, b_d, s_d,
                attn_out_d, state_out_d,
                S_v, H, n_tokens, /*n_tokens_attn_stride=*/0, n_seqs,
                sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, stream);
        } else {
            launch_gated_delta_net<false>(q_d, k_d, v_d, g_d, b_d, s_d,
                attn_out_d, state_out_d,
                S_v, H, n_tokens, /*n_tokens_attn_stride=*/0, n_seqs,
                sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, stream);
        }
        return;
    }

    const int64_t n_full = (n_tokens / 64) * 64;
    const int64_t n_tail = n_tokens - n_full;

    // state_mid: chunked end-state staging slot. We chain it through the
    // existing state_out cache view rather than allocating a fresh buffer:
    //   1) chunked writes its end-state into state_out_d.
    //   2) AR-tail (when n_tail > 0) reads state_in = state_out_d, runs the
    //      r remaining tokens, writes state_out = state_out_d in place.
    // The AR kernel loads its (per-block) state slice into registers once
    // at entry and stores once at exit; per-block slices are disjoint, so
    // aliasing state_in == state_out is safe and saves one
    // n_seqs * H * S_v * S_v float buffer plus the implicit copy to it.
    float * const state_mid_d = state_out_d;

    // Single chunked launch covering n_seqs * n_full tokens. Chunked stages
    // are now L_total-aware: cfg.L drives chunk validity bounds + workspace
    // sizing, while cfg.L_total feeds batch-stride addressing for input /
    // output buffers (q/k/v/g/beta/attn_out). This lets chunked process only
    // the [0, n_full) head of each batch's region in a single kernel call,
    // even when the buffer was allocated for n_tokens > n_full per batch.
    chunked_gdn::config cfg{};
    cfg.S       = S_v;
    cfg.H_qk    = neqk1;
    cfg.H_v     = H;
    cfg.L       = n_full;
    cfg.L_total = n_tokens;
    cfg.B       = n_seqs;
    cfg.kda     = false;
    cfg.q        = q_d;
    cfg.k        = k_d;
    cfg.v        = v_d;
    cfg.g        = g_d;
    cfg.beta     = b_d;
    cfg.state_in = s_d;
    cfg.q_stride_t_floats = sq2;
    cfg.k_stride_t_floats = sk2;
    cfg.v_stride_t_floats = sv2;
    cfg.attn_out  = attn_out_d;
    cfg.state_out = state_mid_d;
    cfg.stream = stream;

    const int64_t need_ws =
        chunked_gdn::workspace_bytes(S_v, neqk1, H, n_full, n_seqs, false);
    ggml_cuda_pool_alloc<char> ws_buf(ctx.pool(), (size_t) need_ws);
    cfg.workspace            = ws_buf.get();
    cfg.workspace_size_bytes = need_ws;

    cudaError_t err = chunked_gdn::launch(cfg);
    GGML_ASSERT(err == cudaSuccess);

    if (n_tail > 0) {
        // Single AR launch over the trailing r = n_tail tokens for ALL
        // sequences. AR's per-seq attn stride is now an explicit kernel
        // parameter (n_tokens_attn_stride) decoupled from the loop bound
        // (n_tokens), so passing n_tokens_full lets AR address attn_out at
        // seq * n_tokens_full * H * S_v + (n_full + t) * H * S_v after we
        // pre-shift the base pointer by n_full * H * S_v. Per-seq strides
        // for q/k/v/g/beta are already encoded in sq3/sk3/sv3/sb3.
        launch_gated_delta_net<false>(
            q_d + n_full * sq2,
            k_d + n_full * sk2,
            v_d + n_full * sv2,
            g_d + n_full * sb2,
            b_d + n_full * sb2,
            state_mid_d,
            attn_out_d + n_full * S_v * H,
            state_out_d,
            S_v, H, n_tail, /*n_tokens_attn_stride=*/n_tokens, n_seqs,
            sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    }
}
