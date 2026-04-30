// bench_gdn_ar: standalone benchmark for the fused Gated Delta Net AR (autoregressive) CUDA kernel.
//
// Targets ggml's `GGML_OP_GATED_DELTA_NET` op, which on CUDA is dispatched to the
// `gated_delta_net_cuda<S_v, KDA>` kernel in ggml/src/ggml-cuda/gated_delta_net.cu.
// This is the same fused path used at runtime when `cparams.fused_gdn_ar=true`.
//
// Reference design / counterpart kernel: ar.cu's `autoregressive_fwd_kernel`.
//
// Phase 1 scope:
//   - measure end-to-end ggml_backend_graph_compute latency for a single GDN op
//   - support shape sweep over n_seq_tokens (decode -> chunked-prefill range)
//   - output console (default) or CSV
//   - F32 only (current kernel limitation)
//
// Out of scope (future phases):
//   - kernel-only timing via cudaEvent
//   - F16/BF16 dtype
//   - multi-step / spec-decode style kernel

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// CLI options
// ---------------------------------------------------------------------------

struct cli_opts {
    int64_t hk       = 16;     // num_k_heads
    int64_t hv       = 32;     // num_v_heads (must be multiple of hk)
    int64_t hs       = 128;    // head_size (kernel supports {16, 32, 64, 128})
    int64_t L        = 1;      // n_seq_tokens
    int64_t B        = 1;      // n_seqs
    bool    kda      = false;  // vector gate (one g per row)
    bool    permuted = false;  // non-contiguous q/k/v via permute

    std::string backend = "";  // empty: auto select first GPU, fallback CPU
    std::string dtype   = "f32";

    bool     v2          = false;  // enable GGML_GDN_AR_V2=1
    bool     compare     = false;  // run v1 then v2, print side-by-side
    int      warmup      = 10;
    int      repeat      = 100;
    int      min_time_ms = 1000;
    bool     sweep       = false;
    bool     csv         = false;
    bool     md          = false;  // markdown table output
    uint32_t seed        = 42;
};

static void print_help(const char * prog) {
    printf(
        "bench_gdn_ar - Gated Delta Net AutoRegressive CUDA kernel benchmark\n"
        "\n"
        "Benchmarks the fused GDN op (GGML_OP_GATED_DELTA_NET) on the selected backend.\n"
        "On CUDA this hits gated_delta_net.cu's gated_delta_net_cuda<S_v, KDA> kernel,\n"
        "the same path used by the runtime when cparams.fused_gdn_ar=true.\n"
        "\n"
        "Although the kernel is named for the autoregressive (n_seq_tokens=1) decode\n"
        "case, it also serves chunked prefill (n_seq_tokens>1). Use --sweep to scan\n"
        "the L dimension and observe the decode -> prefill latency curve.\n"
        "\n"
        "Reference / comparison implementation: ar.cu's autoregressive_fwd_kernel.\n"
        "\n"
        "Usage: %s [options]\n"
        "\n"
        "Shape (defaults follow Qwen3-Next style):\n"
        "  --hk N          num_k_heads               (default: 16)\n"
        "  --hv N          num_v_heads               (default: 32; must be multiple of hk)\n"
        "  --hs N          head_size                 (default: 128; kernel supports 16/32/64/128)\n"
        "  --L N           n_seq_tokens              (default: 1)\n"
        "  --B N           n_seqs                    (default: 1)\n"
        "  --kda           use vector gate (KDA)     (default: off)\n"
        "  --permuted      use non-contiguous QKV    (default: off)\n"
        "  --dtype TYPE    f32 only for now          (default: f32)\n"
        "  --v2            enable v2 row-per-warp kernel (GGML_GDN_AR_V2=1)\n"
        "  --compare       run v1 then v2, print side-by-side table\n"
        "\n"
        "Bench loop:\n"
        "  --warmup N      warmup iterations         (default: 10)\n"
        "  --repeat N      max timed iterations      (default: 100)\n"
        "  --min-time-ms N minimum total bench time  (default: 1000)\n"
        "  --sweep         sweep L over {1,2,4,8,16,32,64,128,256,512,1024}\n"
        "  --csv           emit CSV instead of console\n"
        "  --md            emit markdown table (for PR)\n"
        "  --seed N        random seed               (default: 42)\n"
        "\n"
        "Backend:\n"
        "  --backend NAME  ggml backend name (e.g. CUDA0, CPU). Default: first GPU, else CPU.\n"
        "\n"
        "Misc:\n"
        "  -h, --help      print this help\n"
        "\n"
        "Examples:\n"
        "  %s --help\n"
        "  %s --backend CUDA0 --hk 16 --hv 32 --hs 128 --L 1 --B 1\n"
        "  %s --backend CUDA0 --sweep --csv > gdn_ar_baseline.csv\n"
        "  %s --backend CUDA0 --sweep --v2 --md   # v2 kernel, markdown table\n",
        prog, prog, prog, prog, prog);
}

static bool parse_int(const char * s, int64_t & out) {
    if (!s || !*s) {
        return false;
    }
    char *    end = nullptr;
    long long v   = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0') {
        return false;
    }
    out = (int64_t) v;
    return true;
}

static bool parse_uint(const char * s, uint32_t & out) {
    int64_t v;
    if (!parse_int(s, v) || v < 0) {
        return false;
    }
    out = (uint32_t) v;
    return true;
}

// returns true on success, false if --help was requested or parse error
static int parse_cli(int argc, char ** argv, cli_opts & o) {
    auto need_arg = [&](int i) -> const char * {
        if (i + 1 >= argc) {
            fprintf(stderr, "missing argument for %s\n", argv[i]);
            return nullptr;
        }
        return argv[i + 1];
    };

    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            print_help(argv[0]);
            return 1;
        } else if (!strcmp(a, "--hk")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            if (!parse_int(v, o.hk) || o.hk <= 0) {
                fprintf(stderr, "invalid --hk\n");
                return -1;
            }
            ++i;
        } else if (!strcmp(a, "--hv")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            if (!parse_int(v, o.hv) || o.hv <= 0) {
                fprintf(stderr, "invalid --hv\n");
                return -1;
            }
            ++i;
        } else if (!strcmp(a, "--hs")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            if (!parse_int(v, o.hs) || o.hs <= 0) {
                fprintf(stderr, "invalid --hs\n");
                return -1;
            }
            ++i;
        } else if (!strcmp(a, "--L")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            if (!parse_int(v, o.L) || o.L <= 0) {
                fprintf(stderr, "invalid --L\n");
                return -1;
            }
            ++i;
        } else if (!strcmp(a, "--B")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            if (!parse_int(v, o.B) || o.B <= 0) {
                fprintf(stderr, "invalid --B\n");
                return -1;
            }
            ++i;
        } else if (!strcmp(a, "--kda")) {
            o.kda = true;
        } else if (!strcmp(a, "--permuted")) {
            o.permuted = true;
        } else if (!strcmp(a, "--dtype")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            o.dtype = v;
            if (o.dtype != "f32") {
                fprintf(stderr, "only --dtype f32 is supported\n");
                return -1;
            }
            ++i;
        } else if (!strcmp(a, "--warmup")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            int64_t tmp;
            if (!parse_int(v, tmp) || tmp < 0) {
                fprintf(stderr, "invalid --warmup\n");
                return -1;
            }
            o.warmup = (int) tmp;
            ++i;
        } else if (!strcmp(a, "--repeat")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            int64_t tmp;
            if (!parse_int(v, tmp) || tmp <= 0) {
                fprintf(stderr, "invalid --repeat\n");
                return -1;
            }
            o.repeat = (int) tmp;
            ++i;
        } else if (!strcmp(a, "--min-time-ms")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            int64_t tmp;
            if (!parse_int(v, tmp) || tmp < 0) {
                fprintf(stderr, "invalid --min-time-ms\n");
                return -1;
            }
            o.min_time_ms = (int) tmp;
            ++i;
        } else if (!strcmp(a, "--sweep")) {
            o.sweep = true;
        } else if (!strcmp(a, "--csv")) {
            o.csv = true;
        } else if (!strcmp(a, "--md")) {
            o.md = true;
        } else if (!strcmp(a, "--v2")) {
            o.v2 = true;
        } else if (!strcmp(a, "--compare")) {
            o.compare = true;
        } else if (!strcmp(a, "--seed")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            if (!parse_uint(v, o.seed)) {
                fprintf(stderr, "invalid --seed\n");
                return -1;
            }
            ++i;
        } else if (!strcmp(a, "--backend")) {
            const char * v = need_arg(i);
            if (!v) {
                return -1;
            }
            o.backend = v;
            ++i;
        } else {
            fprintf(stderr, "unknown argument: %s (try --help)\n", a);
            return -1;
        }
    }

    if (o.hv % o.hk != 0) {
        fprintf(stderr, "hv (%lld) must be a multiple of hk (%lld)\n", (long long) o.hv, (long long) o.hk);
        return -1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

static ggml_backend_t select_backend(const std::string & name) {
    const size_t n = ggml_backend_dev_count();

    // explicit by name
    if (!name.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (!dev) {
            fprintf(stderr, "backend '%s' not found. Available:\n", name.c_str());
            for (size_t i = 0; i < n; i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                fprintf(stderr, "  - %s (%s)\n", ggml_backend_dev_name(d), ggml_backend_dev_description(d));
            }
            return nullptr;
        }
        return ggml_backend_dev_init(dev, nullptr);
    }

    // auto: prefer first non-CPU device
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return ggml_backend_dev_init(d, nullptr);
        }
    }
    // fallback to CPU
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

// ---------------------------------------------------------------------------
// Tensor init helpers (single-threaded, deterministic)
// ---------------------------------------------------------------------------

static void fill_uniform_f32(ggml_tensor * t, std::mt19937 & gen, float lo, float hi) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const size_t                          n = ggml_nelements(t);
    std::vector<float>                    buf(n);
    std::uniform_real_distribution<float> d(lo, hi);
    for (size_t i = 0; i < n; i++) {
        buf[i] = d(gen);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, n * sizeof(float));
}

static void fill_zero_f32(ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const size_t       n = ggml_nelements(t);
    std::vector<float> buf(n, 0.0f);
    ggml_backend_tensor_set(t, buf.data(), 0, n * sizeof(float));
}

// ---------------------------------------------------------------------------
// Bench result for one shape
// ---------------------------------------------------------------------------

struct bench_result {
    bool        ok        = false;
    int         n_runs    = 0;
    double      mean_us   = 0.0;
    double      min_us    = 0.0;
    double      max_us    = 0.0;
    double      stddev_us = 0.0;
    double      gflops    = 0.0;
    double      gbs       = 0.0;
    std::string error;
};

// FLOPs estimate per token, per (head_v, col_j):
//   - kv reduction: ~2 * hs (mul+add per row) [non-KDA] or 3*hs [KDA, includes expf bookkeeping]
//   - delta:        ~2
//   - state update: ~2 * hs (mul+add per row)
//   - attn reduction: ~2 * hs
// Per (head_v, col_j) per token: roughly 6*hs FLOPs (non-KDA), 7*hs (KDA, expf counted as 1 FLOP).
// Total: B * L * hv * hs * (6*hs or 7*hs)
static double estimate_gdn_flops(const cli_opts & o) {
    const double per_col = o.kda ? (7.0 * o.hs) : (6.0 * o.hs);
    return (double) o.B * o.L * o.hv * o.hs * per_col;
}

// Bytes moved per call (HBM): main contributors
//   - q, k:       2 * B * L * hk * hs * sizeof(float)   (read)
//   - v:          B * L * hv * hs * sizeof(float)        (read)
//   - g, beta:    B * L * hv * (hs or 1) * sizeof(float) + B * L * hv * sizeof(float)
//   - state in:   B * hv * hs * hs * sizeof(float)
//   - state out:  B * hv * hs * hs * sizeof(float)
//   - out (attn): B * L * hv * hs * sizeof(float)
static double estimate_gdn_bytes(const cli_opts & o) {
    const double f     = sizeof(float);
    double       bytes = 0.0;
    bytes += 2.0 * o.B * o.L * o.hk * o.hs * f;                // q + k
    bytes += 1.0 * o.B * o.L * o.hv * o.hs * f;                // v
    bytes += 1.0 * o.B * o.L * o.hv * (o.kda ? o.hs : 1) * f;  // g
    bytes += 1.0 * o.B * o.L * o.hv * f;                       // beta
    bytes += 2.0 * o.B * o.hv * o.hs * o.hs * f;               // state in + out
    bytes += 1.0 * o.B * o.L * o.hv * o.hs * f;                // attn output
    return bytes;
}

// ---------------------------------------------------------------------------
// Build + run a single shape
// ---------------------------------------------------------------------------

static bench_result run_one(ggml_backend_t backend, const cli_opts & o) {
    bench_result res;

    // 1. context (no_alloc=true, backend allocates)
    ggml_init_params p = {};
    p.mem_size         = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(64, false);
    p.mem_buffer       = nullptr;
    p.no_alloc         = true;
    ggml_context_ptr ctx(ggml_init(p));
    if (!ctx) {
        res.error = "ggml_init failed";
        return res;
    }

    // 2. build tensors (mirrors test_gated_delta_net::build_graph)
    const int64_t   hk = o.hk, hv = o.hv, hs = o.hs, L = o.L, B = o.B;
    const ggml_type t = GGML_TYPE_F32;

    ggml_tensor *q, *k, *v;
    if (o.permuted) {
        // create with dims 1<->2 swapped, then permute back -> non-contiguous
        q = ggml_permute(ctx.get(), ggml_new_tensor_4d(ctx.get(), t, hs, L, hk, B), 0, 2, 1, 3);
        k = ggml_permute(ctx.get(), ggml_new_tensor_4d(ctx.get(), t, hs, L, hk, B), 0, 2, 1, 3);
        v = ggml_permute(ctx.get(), ggml_new_tensor_4d(ctx.get(), t, hs, L, hv, B), 0, 2, 1, 3);
    } else {
        q = ggml_new_tensor_4d(ctx.get(), t, hs, hk, L, B);
        k = ggml_new_tensor_4d(ctx.get(), t, hs, hk, L, B);
        v = ggml_new_tensor_4d(ctx.get(), t, hs, hv, L, B);
    }
    ggml_set_name(q, "q");
    ggml_set_name(k, "k");
    ggml_set_name(v, "v");

    const int64_t g_ne0 = o.kda ? hs : 1;
    ggml_tensor * g     = ggml_new_tensor_4d(ctx.get(), t, g_ne0, hv, L, B);
    ggml_tensor * beta  = ggml_new_tensor_4d(ctx.get(), t, 1, hv, L, B);
    ggml_tensor * state = ggml_new_tensor_2d(ctx.get(), t, hs * hs * hv, B);
    ggml_set_name(g, "g");
    ggml_set_name(beta, "beta");
    ggml_set_name(state, "state");

    ggml_tensor * out = ggml_gated_delta_net(ctx.get(), q, k, v, g, beta, state);
    ggml_set_name(out, "gdn_out");

    // 3. backend support check (after building so the op tensor exists)
    if (!ggml_backend_supports_op(backend, out)) {
        res.error = "backend does not support GATED_DELTA_NET for these dims";
        return res;
    }

    // 4. allocate
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        res.error = "ggml_backend_alloc_ctx_tensors failed";
        return res;
    }

    // 5. randomize inputs (single-threaded, seeded)
    std::mt19937 gen(o.seed);
    fill_uniform_f32(q, gen, -1.0f, 1.0f);
    fill_uniform_f32(k, gen, -1.0f, 1.0f);
    fill_uniform_f32(v, gen, -1.0f, 1.0f);
    // gate is in log-space, kernel takes expf(g); keep g <= 0 so alpha in (0,1]
    fill_uniform_f32(g, gen, -4.0f, 0.0f);
    // beta has been sigmoid'd at graph build time in the model; but the op itself
    // uses *beta_t directly, so for an isolated bench we fill it in (0,1) range.
    fill_uniform_f32(beta, gen, 0.05f, 0.95f);
    // initial state: small random (zero is also fine, but small noise stresses
    // numerics more realistically across long sweeps)
    fill_uniform_f32(state, gen, -0.1f, 0.1f);

    // 6. graph
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(gf, out);

    // 7. warmup
    for (int i = 0; i < o.warmup; i++) {
        ggml_status s = ggml_backend_graph_compute(backend, gf);
        if (s != GGML_STATUS_SUCCESS) {
            res.error = std::string("warmup compute failed: ") + ggml_status_to_string(s);
            return res;
        }
    }
    ggml_backend_synchronize(backend);

    // 8. timing loop: keep going until both --repeat and --min-time-ms are met
    std::vector<double> samples;
    samples.reserve(std::max(o.repeat, 16));
    int64_t total_us = 0;
    while ((int) samples.size() < o.repeat || total_us < (int64_t) o.min_time_ms * 1000) {
        const int64_t t0 = ggml_time_us();
        ggml_status   s  = ggml_backend_graph_compute(backend, gf);
        if (s != GGML_STATUS_SUCCESS) {
            res.error = std::string("compute failed: ") + ggml_status_to_string(s);
            return res;
        }
        ggml_backend_synchronize(backend);
        const int64_t t1 = ggml_time_us();
        const double  dt = (double) (t1 - t0);
        samples.push_back(dt);
        total_us += (int64_t) (t1 - t0);

        // safety cap to avoid runaway on extreme shapes
        if (samples.size() > 100000) {
            break;
        }
    }

    // 9. stats
    double sum = 0.0, mn = samples.front(), mx = samples.front();
    for (double s : samples) {
        sum += s;
        mn = std::min(mn, s);
        mx = std::max(mx, s);
    }
    const double mean = sum / samples.size();
    double       var  = 0.0;
    for (double s : samples) {
        const double d = s - mean;
        var += d * d;
    }
    var /= samples.size();
    const double sd = std::sqrt(var);

    res.ok        = true;
    res.n_runs    = (int) samples.size();
    res.mean_us   = mean;
    res.min_us    = mn;
    res.max_us    = mx;
    res.stddev_us = sd;

    // 10. derived metrics (use mean)
    const double sec   = mean * 1e-6;
    const double flops = estimate_gdn_flops(o);
    const double bytes = estimate_gdn_bytes(o);
    res.gflops         = (sec > 0.0) ? (flops / sec) / 1e9 : 0.0;
    res.gbs            = (sec > 0.0) ? (bytes / sec) / (1024.0 * 1024.0 * 1024.0) : 0.0;

    return res;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

static void print_console_header(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const bool is_gpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
    printf("backend: %s (%s)\n",
           ggml_backend_name(backend),
           ggml_backend_dev_description(dev));
    if (is_gpu) {
        printf("op:      GATED_DELTA_NET (fused AR path; e.g. ggml-cuda gated_delta_net_cuda<S_v, KDA>)\n\n");
    } else {
        printf("op:      GATED_DELTA_NET (CPU forward; ggml-cpu/ops.cpp ggml_compute_forward_gated_delta_net)\n\n");
    }
}

static void print_console_one(const cli_opts & o, const bench_result & r) {
    printf("shape: hk=%lld hv=%lld hs=%lld L=%lld B=%lld kda=%d permuted=%d dtype=%s\n", (long long) o.hk,
           (long long) o.hv, (long long) o.hs, (long long) o.L, (long long) o.B, o.kda ? 1 : 0, o.permuted ? 1 : 0,
           o.dtype.c_str());
    if (!r.ok) {
        printf("  ERROR: %s\n\n", r.error.c_str());
        return;
    }
    printf("  runs:    %8d\n", r.n_runs);
    printf("  mean:    %8.2f us\n", r.mean_us);
    printf("  min:     %8.2f us\n", r.min_us);
    printf("  max:     %8.2f us\n", r.max_us);
    printf("  stddev:  %8.2f us\n", r.stddev_us);
    printf("  GFLOPS:  %8.2f\n", r.gflops);
    printf("  GB/s:    %8.2f\n", r.gbs);
    printf("\n");
}

static void print_csv_header() {
    printf("backend,hk,hv,hs,L,B,kda,permuted,dtype,runs,mean_us,min_us,max_us,stddev_us,gflops,gbs,error\n");
}

static void print_csv_one(ggml_backend_t backend, const cli_opts & o, const bench_result & r) {
    printf("%s,%lld,%lld,%lld,%lld,%lld,%d,%d,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n", ggml_backend_name(backend),
           (long long) o.hk, (long long) o.hv, (long long) o.hs, (long long) o.L, (long long) o.B, o.kda ? 1 : 0,
           o.permuted ? 1 : 0, o.dtype.c_str(), r.n_runs, r.mean_us, r.min_us, r.max_us, r.stddev_us, r.gflops, r.gbs,
           r.ok ? "" : r.error.c_str());
}

// ---------------------------------------------------------------------------
// Markdown table output (for PR-ready tables)
// ---------------------------------------------------------------------------

static const char * md_kernel_label(const cli_opts & o) {
    return o.v2 ? "v2 (row-per-warp + float4)" : "v1 (baseline)";
}

static void print_md_table(const std::vector<int64_t> & L_vals,
                           const std::vector<bench_result> & results,
                           const cli_opts & o) {
    // Header
    printf("| L (tokens) | mean (us) | min (us) | max (us) | stddev (us) | GFLOPS | GB/s |\n");
    printf("|-----------:|----------:|---------:|---------:|------------:|-------:|-----:|\n");
    for (size_t i = 0; i < results.size(); i++) {
        const bench_result & r = results[i];
        if (!r.ok) {
            printf("| %-9lld | ERROR: %s |\n", (long long)L_vals[i], r.error.c_str());
            continue;
        }
        printf("| %9lld | %9.2f | %8.2f | %8.2f | %11.2f | %6.1f | %5.1f |\n",
               (long long)L_vals[i], r.mean_us, r.min_us, r.max_us, r.stddev_us, r.gflops, r.gbs);
    }
}

static void print_md_header(const cli_opts & o) {
    printf("### GDN kernel: %s\n\n", md_kernel_label(o));
    printf("Shape: hk=%lld hv=%lld hs=%lld B=%lld kda=%d dtype=%s\n\n",
           (long long)o.hk, (long long)o.hv, (long long)o.hs, (long long)o.B,
           o.kda ? 1 : 0, o.dtype.c_str());
}

// ---------------------------------------------------------------------------
// --compare mode: run v1 and v2 in separate processes (to bypass static
// env-var caching), parse CSV, print side-by-side comparison.
// ---------------------------------------------------------------------------

#include <map>
#include <unistd.h>

static std::string build_child_cmd(int argc, char ** argv, bool is_v2) {
    char exe_path[4096];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) { perror("readlink"); exit(1); }
    exe_path[len] = '\0';

    std::string cmd = is_v2 ? "GGML_GDN_AR_V2=1 " : "GGML_GDN_AR_V2=0 ";
    cmd += exe_path;
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if (!strcmp(a, "--compare") || !strcmp(a, "--v2") || !strcmp(a, "--md")) {
            if (!strcmp(a, "--md")) continue;  // skip --md
            continue;
        }
        cmd += " ";
        cmd += a;
    }
    cmd += " --csv";
    return cmd;
}

static int run_compare(int argc, char ** argv, const cli_opts & o) {
    // v1 run
    std::string cmd_v1 = build_child_cmd(argc, argv, false);
    FILE *      p_v1   = popen(cmd_v1.c_str(), "r");
    if (!p_v1) { perror("popen v1"); return 1; }

    std::map<int64_t, double> v1_us;  // L -> mean_us
    {
        char * line = nullptr; size_t n = 0;
        while (getline(&line, &n, p_v1) != -1) {
            if (line[0] == 'b') continue;  // skip header
            int64_t L = 0;
            double  us = 0.0;
            sscanf(line, "%*[^,],%*[^,],%*[^,],%*[^,],%lld,%*[^,],%*[^,],%*[^,],%*[^,],%*[^,],%lf",
                   (long long *)&L, &us);
            v1_us[L] = us;
        }
        free(line);
        pclose(p_v1);
    }

    // v2 run
    std::string cmd_v2 = build_child_cmd(argc, argv, true);
    FILE *      p_v2   = popen(cmd_v2.c_str(), "r");
    if (!p_v2) { perror("popen v2"); return 1; }

    std::map<int64_t, double> v2_us;
    {
        char * line = nullptr; size_t n = 0;
        while (getline(&line, &n, p_v2) != -1) {
            if (line[0] == 'b') continue;
            int64_t L = 0;
            double  us = 0.0;
            sscanf(line, "%*[^,],%*[^,],%*[^,],%*[^,],%lld,%*[^,],%*[^,],%*[^,],%*[^,],%*[^,],%lf",
                   (long long *)&L, &us);
            v2_us[L] = us;
        }
        free(line);
        pclose(p_v2);
    }

    bool md = o.md;
    if (md) {
        printf("### v1 vs v2 comparison\n\n");
        printf("Shape: hk=%lld hv=%lld hs=%lld B=%lld kda=%d dtype=%s\n\n",
               (long long)o.hk, (long long)o.hv, (long long)o.hs, (long long)o.B,
               o.kda ? 1 : 0, o.dtype.c_str());
        printf("| L | v1 (us) | v2 (us) | speedup |\n");
        printf("|---:|--------:|--------:|--------:|\n");
    } else {
        printf("%6s %10s %10s %8s\n", "L", "v1(us)", "v2(us)", "speedup");
        printf("%6s %10s %10s %8s\n", "------", "----------", "----------", "--------");
    }

    for (const auto & kv : v1_us) {
        int64_t L  = kv.first;
        double  u1 = kv.second;
        auto    it = v2_us.find(L);
        if (it == v2_us.end()) continue;
        double u2 = it->second;
        double sp = (u1 / u2 - 1.0) * 100.0;
        if (md) {
            printf("| %4lld | %8.1f | %8.1f | %+6.1f%% |\n", (long long)L, u1, u2, sp);
        } else {
            printf("%6lld %10.1f %10.1f %+7.1f%%\n", (long long)L, u1, u2, sp);
        }
    }

    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    cli_opts o;
    int      rc = parse_cli(argc, argv, o);
    if (rc == 1) {
        return 0;  // --help
    }
    if (rc < 0) {
        return 1;  // parse error
    }

    // --compare: re-exec self twice (v1/v2), parse CSV, print comparison
    if (o.compare) {
        return run_compare(argc, argv, o);
    }

    // Must be set before backend init: the dispatch decision is cached
    // on first call (static local).
    if (o.v2) {
        setenv("GGML_GDN_AR_V2", "1", 1);
    }

    ggml_backend_load_all();

    ggml_backend_t backend = select_backend(o.backend);
    if (!backend) {
        fprintf(stderr, "failed to initialize backend\n");
        return 1;
    }

    // sweep over L if requested; otherwise just run the user's shape
    std::vector<int64_t> L_values;
    if (o.sweep) {
        L_values = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
    } else {
        L_values = { o.L };
    }

    if (o.md) {
        // markdown: collect all results, print table at end
        std::vector<bench_result> results;
        int n_failed = 0;
        for (int64_t L : L_values) {
            cli_opts s     = o;
            s.L            = L;
            bench_result r = run_one(backend, s);
            if (!r.ok) ++n_failed;
            results.push_back(r);
        }
        print_md_header(o);
        print_md_table(L_values, results, o);
        ggml_backend_free(backend);
        return n_failed == 0 ? 0 : 1;
    }

    if (o.csv) {
        print_csv_header();
    } else {
        print_console_header(backend);
    }

    int n_failed = 0;
    for (int64_t L : L_values) {
        cli_opts s     = o;
        s.L            = L;
        bench_result r = run_one(backend, s);
        if (!r.ok) {
            ++n_failed;
        }
        if (o.csv) {
            print_csv_one(backend, s, r);
        } else {
            print_console_one(s, r);
        }
        fflush(stdout);
    }

    ggml_backend_free(backend);
    return n_failed == 0 ? 0 : 1;
}
