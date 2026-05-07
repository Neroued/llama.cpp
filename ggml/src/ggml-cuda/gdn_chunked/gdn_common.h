// Common host-side types and validators for chunked_gdn.
//
// Ported from chunked_gdn/include/gdn_common.h. Kept in the `gdn::` namespace
// (sub-namespace of `chunked_gdn` ports) so device-side helpers in
// chunked_utils.cuh can share `head_map` / `init_fastdiv_values` / size
// helpers without dragging device intrinsics into host TUs.
//
// HOST-SAFE: this header includes only `<cuda_runtime.h>` for `uint3`. Never
// include `chunked_utils.cuh` here -- it pulls in NVCC-only intrinsics that
// would break host-only TUs (test_*, bench_*, ...). Keep the device-only
// `fastmodulo` body in chunked_utils.cuh; this header forward-declares it
// inside the `__CUDA_ARCH__` branch of `head_map::qk_head` only.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#  define GDN_HOST_DEVICE __host__ __device__
#else
#  define GDN_HOST_DEVICE
#endif

#ifdef __CUDA_ARCH__
namespace gdn {
__device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values);
}
#endif

namespace gdn {

inline uint3 init_fastdiv_values(uint64_t d_64) {
    if (d_64 == 0 || d_64 > (uint64_t) 0xffffffffu) {
        std::fprintf(stderr, "gdn::init_fastdiv_values: invalid divisor %llu\n",
                     (unsigned long long) d_64);
        std::abort();
    }
    const uint32_t d = (uint32_t) d_64;
    uint32_t L = 0;
    while (L < 32 && ((uint32_t) 1 << L) < d) L++;
    const uint32_t mp =
        (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}

struct sizes {
    int64_t q_floats;
    int64_t k_floats;
    int64_t v_floats;
    int64_t g_floats;
    int64_t beta_floats;
    int64_t state_floats;
    int64_t attn_out_floats;
};

inline sizes compute_sizes(int64_t S, int64_t H_qk, int64_t H_v,
                           int64_t L, int64_t B, bool kda) {
    sizes s{};
    s.q_floats        = B * L * H_qk * S;
    s.k_floats        = B * L * H_qk * S;
    s.v_floats        = B * L * H_v  * S;
    s.g_floats        = B * L * H_v  * (kda ? S : 1);
    s.beta_floats     = B * L * H_v;
    s.state_floats    = B * H_v * S * S;
    s.attn_out_floats = B * L * H_v * S;
    return s;
}

inline bool is_supported_head_dim(int64_t S) {
    return S == 16 || S == 32 || S == 64 || S == 128;
}

inline bool are_head_counts_valid(int64_t H_qk, int64_t H_v) {
    return H_qk > 0 && H_v >= H_qk && (H_v % H_qk) == 0;
}

struct head_map {
    int   H_qk;
    int   H_v;
    uint3 mod_magic;

    static head_map of(int H_qk_, int H_v_) {
        return head_map{H_qk_, H_v_,
                        gdn::init_fastdiv_values((uint64_t) H_qk_)};
    }

    GDN_HOST_DEVICE int qk_head(int h_v) const {
    #ifdef __CUDA_ARCH__
        return (int) gdn::fastmodulo((uint32_t) h_v, mod_magic);
    #else
        return h_v % H_qk;
    #endif
    }

    GDN_HOST_DEVICE int cta_h_v(int cta_h) const {
        const int G       = H_v / H_qk;
        const int g_idx   = cta_h % G;
        const int qk_idx  = cta_h / G;
        return g_idx * H_qk + qk_idx;
    }
};

} // namespace gdn
