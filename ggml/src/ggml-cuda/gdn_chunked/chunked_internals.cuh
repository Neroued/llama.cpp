// Internal constants shared by every chunked-stage .cu in this port.
//
// Ported from chunked_gdn/chunked/chunked_internals.cuh. Only the chunked
// stage .cu files in this directory may include this header.

#pragma once

#include "chunked_gdn.cuh"
#include "chunked_utils.cuh"
#include "gdn_common.h"

#include <cstdint>
#include <cstdio>

namespace chunked_gdn::detail {

struct bh_decode_t {
    int b;
    int h_v;

    static __device__ __forceinline__ bh_decode_t of(int bh, int H_v) {
        bh_decode_t r;
        r.b   = bh / H_v;
        r.h_v = bh - r.b * H_v;
        return r;
    }

    static __device__ __forceinline__ bh_decode_t of(int bh, gdn::head_map qk_map) {
        bh_decode_t r;
        const int H_v   = qk_map.H_v;
        r.b             = bh / H_v;
        const int cta_h = bh - r.b * H_v;
        r.h_v           = qk_map.cta_h_v(cta_h);
        return r;
    }
};

template <int TILES, int N>
__device__ __forceinline__ void zero_frag(float (& frag)[TILES][N]) {
    #pragma unroll
    for (int t = 0; t < TILES; ++t) {
        #pragma unroll
        for (int e = 0; e < N; ++e) frag[t][e] = 0.0f;
    }
}

inline constexpr int BT     = (int) chunked_gdn::kChunkSize;
inline constexpr int BC     = (int) chunked_gdn::kSubChunkSize;

inline constexpr int MMA_M  = 16;
inline constexpr int MMA_N  = 8;
inline constexpr int MMA_K  = 8;

static_assert(BT % BC == 0, "chunked_gdn::detail::BT must be a multiple of BC");
static_assert(BT % MMA_M == 0, "chunked_gdn::detail::BT must be a multiple of MMA_M");

struct stage_validator {
    const char * name;
    int64_t S, H_qk, H_v, T, B;
    bool kda;
    bool require_h_qk = true;

    cudaError_t check_shape() const {
        const bool bad_shape = S <= 0 || H_v <= 0 || T <= 0 || B <= 0
                            || (require_h_qk && H_qk <= 0);
        if (bad_shape) {
            std::fprintf(stderr,
                "%s: invalid shape (S=%lld H_qk=%lld H_v=%lld T=%lld B=%lld)\n",
                name,
                (long long) S, (long long) H_qk, (long long) H_v,
                (long long) T, (long long) B);
            return cudaErrorInvalidValue;
        }
        if (!gdn::is_supported_head_dim(S)) {
            std::fprintf(stderr,
                "%s: unsupported S=%lld (allowed: 16, 32, 64, 128)\n",
                name, (long long) S);
            return cudaErrorInvalidValue;
        }
        if (require_h_qk && !gdn::are_head_counts_valid(H_qk, H_v)) {
            std::fprintf(stderr,
                "%s: invalid head counts H_qk=%lld H_v=%lld "
                "(need H_qk >= 1, H_v >= H_qk, H_v %% H_qk == 0)\n",
                name, (long long) H_qk, (long long) H_v);
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    cudaError_t check_gdn_full_chunks() const {
        if (!kda && (T % BT) != 0) {
            std::fprintf(stderr,
                "%s: GDN chunked path requires T to be a multiple of %d; "
                "route tail tokens through AR instead (T=%lld)\n",
                name, BT, (long long) T);
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    cudaError_t check_grid(int64_t grid_x, int64_t grid_y, int64_t grid_z = 1) const {
        if (grid_x > (int64_t) 0xffffffff) {
            std::fprintf(stderr,
                "%s: grid.x too large (%lld); splitting along NT/H_v/B is a follow-up\n",
                name, (long long) grid_x);
            return cudaErrorInvalidConfiguration;
        }
        if (grid_y > (int64_t) 0xffff) {
            std::fprintf(stderr,
                "%s: grid.y too large (%lld); splitting along H_v/B is a follow-up\n",
                name, (long long) grid_y);
            return cudaErrorInvalidConfiguration;
        }
        if (grid_z > (int64_t) 0xffff) {
            std::fprintf(stderr,
                "%s: grid.z too large (%lld); splitting along H_v/B is a follow-up\n",
                name, (long long) grid_z);
            return cudaErrorInvalidConfiguration;
        }
        return cudaSuccess;
    }
};

} // namespace chunked_gdn::detail
