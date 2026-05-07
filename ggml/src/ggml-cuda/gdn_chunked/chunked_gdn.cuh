// Public-facing launcher for the chunked GDN port (was chunked_gdn project's
// chunked/chunked_gdn.cuh; ported to live inside ggml-cuda).
//
// Differences from the standalone project:
//   * KDA path is rejected (`cudaErrorNotYetImplemented`); ggml-cuda's
//     dispatch in gated_delta_net.cu only routes !kda inputs into chunked.
//   * AR fallback is owned by the dispatcher (gated_delta_net.cu), not by
//     this launcher. Callers split tail tokens (n_tokens % 64 != 0) into a
//     chunked + AR-tail pair using `state_mid` as the chunked-end-state
//     staging slot; this launcher only sees the n_full <= n_tokens chunked
//     prefix.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace chunked_gdn {

inline constexpr int64_t kChunkSize    = 64;
inline constexpr int64_t kSubChunkSize = 16;

static_assert(kChunkSize  % kSubChunkSize == 0,
              "kChunkSize must be a multiple of kSubChunkSize");

inline constexpr int64_t kWorkspaceAlign = 256;

struct workspace_layout {
    int64_t g_cumsum_off,  g_cumsum_bytes;
    int64_t Aqk_off,       Aqk_bytes;
    int64_t W_off,         W_bytes;
    int64_t U_off,         U_bytes;
    int64_t kg_off,        kg_bytes;
    int64_t v_new_off,     v_new_bytes;
    int64_t h_chunk_off,   h_chunk_bytes;
    int64_t total_bytes;
};

namespace detail {

inline int64_t align_up(int64_t x) {
    return (x + kWorkspaceAlign - 1) & ~(kWorkspaceAlign - 1);
}

inline int64_t reserve(int64_t & cursor, int64_t bytes) {
    if (bytes == 0) return cursor;
    const int64_t off = cursor;
    cursor = align_up(off + bytes);
    return off;
}

} // namespace detail

inline workspace_layout compute_workspace_layout(int64_t S, int64_t H_qk, int64_t H_v,
                                                 int64_t L, int64_t B, bool kda) {
    (void) H_qk;
    constexpr int64_t f = (int64_t) sizeof(float);
    const int64_t BT  = kChunkSize;
    const int64_t T   = L;
    const int64_t NT  = (T + BT - 1) / BT;

    const int64_t per_token_S    = B * T * H_v * S;
    const int64_t per_token_BT   = B * T * H_v * BT;
    const int64_t per_token_g    = B * T * H_v * (kda ? S : 1);
    const int64_t per_chunk_SxS  = B * NT * H_v * S * S;

    workspace_layout w{};
    w.g_cumsum_bytes = per_token_g   * f;
    w.Aqk_bytes      = kda ? per_token_BT * f : 0;
    w.W_bytes        = per_token_S   * f;
    w.U_bytes        = per_token_S   * f;
    w.kg_bytes       = kda ? per_token_S * f : 0;
    w.v_new_bytes    = per_token_S   * f;
    w.h_chunk_bytes  = per_chunk_SxS * f;

    int64_t cur = 0;
    w.g_cumsum_off = detail::reserve(cur, w.g_cumsum_bytes);
    w.Aqk_off      = detail::reserve(cur, w.Aqk_bytes);
    w.W_off        = detail::reserve(cur, w.W_bytes);
    w.U_off        = detail::reserve(cur, w.U_bytes);
    w.kg_off       = detail::reserve(cur, w.kg_bytes);
    w.v_new_off    = detail::reserve(cur, w.v_new_bytes);
    w.h_chunk_off  = detail::reserve(cur, w.h_chunk_bytes);
    w.total_bytes  = cur;
    return w;
}

inline int64_t workspace_bytes(int64_t S, int64_t H_qk, int64_t H_v,
                               int64_t L, int64_t B, bool kda) {
    return compute_workspace_layout(S, H_qk, H_v, L, B, kda).total_bytes;
}

struct config {
    int64_t S    = 0;
    int64_t H_qk = 0;
    int64_t H_v  = 0;
    int64_t L    = 0;     // tokens chunked actually processes (must be multiple of kChunkSize)
    // Optional batch stride in tokens for input/output buffers (q/k/v/g/beta/attn_out).
    // Set this when chunked is processing only the first L tokens of a buffer that
    // was allocated for L_total > L tokens per batch (split dispatch: chunked covers
    // [0, L) and an AR-tail covers [L, L_total) over the same B>=1 buffer). When 0
    // the launcher defaults to L (single-stage call, no split). Workspace addressing
    // always uses L; only inputs/outputs use L_total for batch stride.
    int64_t L_total = 0;
    int64_t B    = 0;
    bool    kda  = false;

    const float * q        = nullptr;
    const float * k        = nullptr;
    const float * v        = nullptr;
    const float * g        = nullptr;
    const float * beta     = nullptr;
    const float * state_in = nullptr;

    int64_t q_stride_t_floats = 0;
    int64_t k_stride_t_floats = 0;
    int64_t v_stride_t_floats = 0;

    float * attn_out  = nullptr;
    float * state_out = nullptr;

    void   * workspace            = nullptr;
    int64_t  workspace_size_bytes = 0;

    cudaStream_t stream = nullptr;
};

cudaError_t launch(const config & cfg);

namespace stages {

struct g_cumsum_config {
    int64_t S    = 0;
    int64_t H_v  = 0;
    int64_t L    = 0;
    int64_t L_total = 0;   // batch stride in tokens for g_in (defaults to L when 0)
    int64_t B    = 0;
    bool    kda  = false;

    const float * g_in  = nullptr;
    float       * g_out = nullptr;

    cudaStream_t  stream = nullptr;
};
cudaError_t launch_g_cumsum(const g_cumsum_config & cfg);

struct prepare_wy_wu_config {
    int64_t S    = 0;
    int64_t H_qk = 0;
    int64_t H_v  = 0;
    int64_t L    = 0;
    int64_t L_total = 0;   // batch stride in tokens for k_in / v_in / beta_in (defaults to L)
    int64_t B    = 0;
    bool    kda  = false;

    const float * q        = nullptr;
    const float * k        = nullptr;
    const float * v        = nullptr;
    const float * g_cumsum = nullptr;
    const float * beta     = nullptr;

    float       * W        = nullptr;
    float       * U        = nullptr;
    float       * Aqk      = nullptr;
    float       * kg       = nullptr;

    int64_t       k_stride_t_floats = 0;
    int64_t       v_stride_t_floats = 0;

    cudaStream_t  stream   = nullptr;
};
cudaError_t launch_prepare_wy_wu(const prepare_wy_wu_config & cfg);

struct state_passing_config {
    int64_t S    = 0;
    int64_t H_qk = 0;
    int64_t H_v  = 0;
    int64_t L    = 0;
    int64_t L_total = 0;   // batch stride in tokens for k_in (defaults to L)
    int64_t B    = 0;
    bool    kda  = false;

    const float * W        = nullptr;
    const float * U        = nullptr;
    const float * k_or_kg  = nullptr;
    const float * g_cumsum = nullptr;
    const float * state_in = nullptr;

    float       * v_new    = nullptr;
    float       * h_chunk  = nullptr;
    float       * state_out = nullptr;

    int64_t       k_stride_t_floats = 0;

    cudaStream_t  stream   = nullptr;
};
cudaError_t launch_state_passing(const state_passing_config & cfg);

struct chunk_output_config {
    int64_t S    = 0;
    int64_t H_qk = 0;
    int64_t H_v  = 0;
    int64_t L    = 0;
    int64_t L_total = 0;   // batch stride in tokens for q_in / k_in / attn_out (defaults to L)
    int64_t B    = 0;
    bool    kda  = false;

    const float * q        = nullptr;
    const float * k        = nullptr;
    const float * v_new    = nullptr;
    const float * g_cumsum = nullptr;
    const float * Aqk      = nullptr;
    const float * h_chunk  = nullptr;

    float       * attn_out = nullptr;

    int64_t       q_stride_t_floats = 0;
    int64_t       k_stride_t_floats = 0;

    cudaStream_t  stream   = nullptr;
};
cudaError_t launch_chunk_output(const chunk_output_config & cfg);

} // namespace stages

} // namespace chunked_gdn
