// Public launcher for chunked_gdn.
//
// GDN path: real 4-stage pipeline (g_cumsum -> prepare_wy_wu (fused
// prepare_wy + recompute_wu, T_inv stays in smem) -> state_passing ->
// chunk_output), each stage owns its own kernel(s) and reads/writes the
// matching workspace region. Modulo GQA mapping h_qk = h_v % H_qk is
// applied inside each kernel via `gdn::head_map`; q/k stay at H_qk heads
// throughout.
//
// KDA path: stages 1-5 are NYI for KDA. The standalone chunked_gdn project
// fell back to ar_gdn::launch here, but in this port the dispatch site
// (ggml-cuda/gated_delta_net.cu) routes KDA inputs to the existing AR kernel
// directly; this launcher just rejects KDA so the caller never gets a
// silent fallback.

#include "chunked_gdn.cuh"

#include "chunked_internals.cuh"
#include "chunked_utils.cuh"     // GDN_PROPAGATE
#include "gdn_common.h"

#include <cstdio>

namespace chunked_gdn {

namespace {

cudaError_t launch_gdn_pipeline(const config & cfg,
                                const workspace_layout & wl,
                                char * ws_base) {
    float * const d_g_cumsum = (float *) (ws_base + wl.g_cumsum_off);
    float * const d_W        = (float *) (ws_base + wl.W_off);
    float * const d_U        = (float *) (ws_base + wl.U_off);
    float * const d_v_new    = (float *) (ws_base + wl.v_new_off);
    float * const d_h_chunk  = (float *) (ws_base + wl.h_chunk_off);

    float * const d_attn_out  = cfg.attn_out;
    float * const d_state_out = cfg.state_out;

    const int64_t L_total = (cfg.L_total != 0) ? cfg.L_total : cfg.L;

    // Stage 1: g_cumsum
    {
        stages::g_cumsum_config c{};
        c.S       = cfg.S;
        c.H_v     = cfg.H_v;
        c.L       = cfg.L;
        c.L_total = L_total;
        c.B       = cfg.B;
        c.kda     = cfg.kda;
        c.g_in    = cfg.g;
        c.g_out   = d_g_cumsum;
        c.stream  = cfg.stream;
        GDN_PROPAGATE(stages::launch_g_cumsum(c));
    }

    // Stage 2+3 fused: prepare_wy_wu (T_inv lives only in smem)
    {
        stages::prepare_wy_wu_config c{};
        c.S        = cfg.S;
        c.H_qk     = cfg.H_qk;
        c.H_v      = cfg.H_v;
        c.L        = cfg.L;
        c.L_total  = L_total;
        c.B        = cfg.B;
        c.kda      = cfg.kda;
        c.q        = cfg.q;
        c.k        = cfg.k;
        c.v        = cfg.v;
        c.g_cumsum = d_g_cumsum;
        c.beta     = cfg.beta;
        c.W        = d_W;
        c.U        = d_U;
        c.Aqk      = nullptr;
        c.kg       = nullptr;
        c.k_stride_t_floats = cfg.k_stride_t_floats;
        c.v_stride_t_floats = cfg.v_stride_t_floats;
        c.stream   = cfg.stream;
        GDN_PROPAGATE(stages::launch_prepare_wy_wu(c));
    }

    // Stage 4: state_passing
    {
        stages::state_passing_config c{};
        c.S         = cfg.S;
        c.H_qk      = cfg.H_qk;
        c.H_v       = cfg.H_v;
        c.L         = cfg.L;
        c.L_total   = L_total;
        c.B         = cfg.B;
        c.kda       = cfg.kda;
        c.W         = d_W;
        c.U         = d_U;
        c.k_or_kg   = cfg.k;
        c.g_cumsum  = d_g_cumsum;
        c.state_in  = cfg.state_in;
        c.v_new     = d_v_new;
        c.h_chunk   = d_h_chunk;
        c.state_out = d_state_out;
        c.k_stride_t_floats = cfg.k_stride_t_floats;
        c.stream    = cfg.stream;
        GDN_PROPAGATE(stages::launch_state_passing(c));
    }

    // Stage 5: chunk_output
    {
        stages::chunk_output_config c{};
        c.S        = cfg.S;
        c.H_qk     = cfg.H_qk;
        c.H_v      = cfg.H_v;
        c.L        = cfg.L;
        c.L_total  = L_total;
        c.B        = cfg.B;
        c.kda      = cfg.kda;
        c.q        = cfg.q;
        c.k        = cfg.k;
        c.v_new    = d_v_new;
        c.g_cumsum = d_g_cumsum;
        c.Aqk      = nullptr;
        c.h_chunk  = d_h_chunk;
        c.attn_out = d_attn_out;
        c.q_stride_t_floats = cfg.q_stride_t_floats;
        c.k_stride_t_floats = cfg.k_stride_t_floats;
        c.stream   = cfg.stream;
        GDN_PROPAGATE(stages::launch_chunk_output(c));
    }

    return cudaSuccess;
}

} // anonymous namespace

cudaError_t launch(const config & cfg) {
    if (!gdn::is_supported_head_dim(cfg.S)) {
        std::fprintf(stderr,
                     "chunked_gdn::launch: unsupported S=%lld (allowed: 16, 32, 64, 128)\n",
                     (long long) cfg.S);
        return cudaErrorInvalidValue;
    }
    if (!gdn::are_head_counts_valid(cfg.H_qk, cfg.H_v)) {
        std::fprintf(stderr,
                     "chunked_gdn::launch: invalid H_qk=%lld H_v=%lld "
                     "(need H_qk >= 1, H_v >= H_qk, H_v %% H_qk == 0)\n",
                     (long long) cfg.H_qk, (long long) cfg.H_v);
        return cudaErrorInvalidValue;
    }
    if (cfg.L <= 0 || cfg.B <= 0) {
        return cudaErrorInvalidValue;
    }
    if (!cfg.kda && (cfg.L % detail::BT) != 0) {
        std::fprintf(stderr,
                     "chunked_gdn::launch: GDN chunked path requires L to be a "
                     "multiple of %d; route tail tokens through AR instead (L=%lld)\n",
                     detail::BT, (long long) cfg.L);
        return cudaErrorInvalidValue;
    }
    if (cfg.attn_out == nullptr || cfg.state_out == nullptr) {
        std::fprintf(stderr,
                     "chunked_gdn::launch: attn_out and state_out must both be non-null\n");
        return cudaErrorInvalidValue;
    }

    const workspace_layout wl =
        compute_workspace_layout(cfg.S, cfg.H_qk, cfg.H_v, cfg.L, cfg.B, cfg.kda);
    if (wl.total_bytes > 0) {
        if (cfg.workspace == nullptr || cfg.workspace_size_bytes < wl.total_bytes) {
            std::fprintf(stderr,
                         "chunked_gdn::launch: workspace too small "
                         "(need %lld bytes, got %lld)\n",
                         (long long) wl.total_bytes, (long long) cfg.workspace_size_bytes);
            return cudaErrorInvalidValue;
        }
    }
    char * const ws_base = (char *) cfg.workspace;

    if (cfg.kda) {
        // Caller (gated_delta_net.cu) is expected to route KDA to the AR
        // kernel before reaching this launcher. Reject explicitly so a
        // mis-dispatch surfaces as a hard error rather than a silent
        // miscompute.
        return cudaErrorNotYetImplemented;
    }

    return launch_gdn_pipeline(cfg, wl, ws_base);
}

} // namespace chunked_gdn
