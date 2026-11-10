#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "json.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using clock_type = std::chrono::steady_clock;
using json       = common_json;

struct rendered {
    std::string   text;
    jinja::string parts;
};

struct sample {
    double setup;
    double execute;
    double flatten;
    double cleanup;
    double total;
};

static double elapsed(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        std::cout
            << "Read JSON lines from stdin: source, context, name, repeat (5), warmup (1), threads (1), mark_input "
               "(false).\n"
               "Write JSON lines with compile time, per-render times, output text and input-marked string parts.\n"
               "Timing includes context construction and destruction; excludes input JSON parsing and template "
               "compilation.\n";
        return std::string(argv[1]) == "--help" ? 0 : 1;
    }
    int failures = 0;
    for (std::string line; std::getline(std::cin, line);) {
        json report;
        try {
            const auto   request    = json::parse(line);
            const auto   source     = request.at("source").get<std::string>();
            const auto & input      = request.at("context");
            const auto   repeat     = request.value("repeat", 5);
            const auto   warmup     = request.value("warmup", 1);
            const auto   threads    = request.value("threads", 1);
            const auto   mark_input = request.value("mark_input", false);
            report["name"]          = request.value("name", std::string("case"));
            if (repeat <= 0 || warmup < 0 || threads <= 0) {
                throw std::invalid_argument("repeat and threads must be positive; warmup must be nonnegative");
            }
            const auto start     = clock_type::now();
            const auto program   = jinja::parse_from_tokens(jinja::lexer().tokenize(source));
            report["compile_ms"] = elapsed(start, clock_type::now());

            auto render = [&](sample & timing) {
                const auto             begin = clock_type::now();
                clock_type::time_point setup, execute, flatten;
                rendered               output;
                {
                    jinja::context ctx(source);
                    ctx.current_time = 0;
                    jinja::global_from_json(ctx, input, mark_input);
                    setup             = clock_type::now();
                    const auto result = jinja::runtime(ctx).execute(program);
                    execute           = clock_type::now();
                    output.parts      = jinja::runtime::gather_string_parts(result)->as_string();
                    output.text       = output.parts.str();
                    flatten           = clock_type::now();
                }
                const auto end = clock_type::now();
                timing         = { elapsed(begin, setup), elapsed(setup, execute), elapsed(execute, flatten),
                                   elapsed(flatten, end), elapsed(begin, end) };
                return output;
            };
            sample                           unused;
            const auto                       reference      = render(unused);
            const auto &                     reference_text = reference.text;
            std::vector<std::vector<sample>> samples(threads);
            std::vector<std::exception_ptr>  errors(threads);
            std::atomic<int>                 ready{ 0 };
            std::atomic<bool>                run{ threads == 1 };
            auto                             worker = [&](int id) {
                try {
                    sample warmup_sample;
                    for (int i = 0; i < warmup; ++i) {
                        render(warmup_sample);
                    }
                    samples[id].reserve(repeat);
                } catch (...) {
                    errors[id] = std::current_exception();
                }
                ready.fetch_add(1);
                while (!run.load()) {
                    std::this_thread::yield();
                }
                if (errors[id]) {
                    return;
                }
                try {
                    for (int i = 0; i < repeat; ++i) {
                        sample     timing;
                        const auto text = render(timing);
                        samples[id].push_back(timing);
                        if (text.text != reference_text) {
                            throw std::runtime_error("render changed across repetitions");
                        }
                    }
                } catch (...) {
                    errors[id] = std::current_exception();
                }
            };
            if (threads == 1) {
                worker(0);
            } else {
                std::vector<std::thread> workers;
                for (int id = 0; id < threads; ++id) {
                    workers.emplace_back(worker, id);
                }
                while (ready.load() != threads) {
                    std::this_thread::yield();
                }
                run.store(true);
                for (auto & thread : workers) {
                    thread.join();
                }
            }
            for (const auto & error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
            report["threads"]    = threads;
            report["mark_input"] = mark_input;
            report["samples"]    = json::array();
            for (int id = 0; id < threads; ++id) {
                for (const auto & timing : samples[id]) {
                    report["samples"].push_back({
                        { "thread",     id             },
                        { "setup_ms",   timing.setup   },
                        { "execute_ms", timing.execute },
                        { "flatten_ms", timing.flatten },
                        { "cleanup_ms", timing.cleanup },
                        { "total_ms",   timing.total   }
                    });
                }
            }
            report["text"]  = reference_text;
            report["parts"] = json::array();
            for (const auto & part : reference.parts.parts) {
                report["parts"].push_back({
                    { "text",     part.val      },
                    { "is_input", part.is_input }
                });
            }
            report["ok"] = true;
        } catch (const std::exception & error) {
            ++failures;
            report["ok"]    = false;
            report["error"] = error.what();
        }
        std::cout << report.dump() << '\n';
    }
    return failures ? 1 : 0;
}
