#!/usr/bin/env python3
"""Generate reproducible Jinja workloads and compare bench-jinja with Python Jinja2."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path


def history(count):
    """Synthetic tool-use history with deterministic contents; no user data."""
    messages = [{"role": "system", "content": "Use the supplied tools to inspect and explain source code."}]
    for i in range(count - 1):
        turn, slot = divmod(i, 4)
        name = f"inspect_{turn % 16}"
        if slot == 0:
            message = {
                "role": "user",
                "content": f"Inspect module {turn}. " + "Explain the result and its implications. " * 32,
            }
        elif slot == 1:
            message = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I will inspect the relevant module. " * 8,
                "tool_calls": [
                    {
                        "id": f"call_{turn}",
                        "type": "function",
                        "function": {"name": name, "arguments": {"path": f"src/module_{turn}.cpp", "lines": 80}},
                    }
                ],
            }
        elif slot == 2:
            message = {
                "role": "tool",
                "tool_call_id": f"call_{turn}",
                "content": "int process(int value) { return value + 1; }\n" * 32,
            }
        else:
            message = {
                "role": "assistant",
                "content": "The module increments the input value. " * 16,
                "reasoning_content": "The implementation matches the requested contract. " * 8,
            }
        messages.append(message)
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"inspect_{i}",
                "description": "Inspect source code.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}},
                    "required": ["path"],
                },
            },
        }
        for i in range(16)
    ]
    return {
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": True,
        "clear_thinking": False,
        "preserve_thinking": True,
        "reasoning_effort": "high",
        "bos_token": "",
        "eos_token": "",
    }


def cases(args):
    for count in (1000, 10000, 100000):
        yield {
            "name": f"range_break/{count}",
            "source": "{% for i in range(n) %}{{ i }}{% break %}{% endfor %}",
            "context": {"n": count},
        }
    for count in (128, 512, 2048, 4096):
        yield {
            "name": f"suffix_break/{count}",
            "source": "{% set ns=namespace(n=0) %}{% for i in range(n) %}{% for j in range(i,n) %}{% set ns.n=ns.n+1 %}{% break %}{% endfor %}{% endfor %}{{ ns.n }}",
            "context": {"n": count},
        }
        yield {
            "name": f"linear/{count}",
            "source": "{% set ns=namespace(n=0) %}{% for i in range(n) %}{% set ns.n=ns.n+1 %}{% endfor %}{{ ns.n }}",
            "context": {"n": count},
        }
    for path in args.template:
        source = path.read_text()
        for count in args.messages:
            yield {"name": f"{path.stem}/{count}", "source": source, "context": history(count)}


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0,
    }


def python_render(case, warmup, repeat):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, **{"ensure_ascii": False, **kwargs})

    def fail(message):
        raise RuntimeError(message)

    env.globals["raise_exception"] = fail
    start = time.perf_counter_ns()
    template = env.from_string(case["source"])
    compile_ms = (time.perf_counter_ns() - start) / 1e6
    samples = []
    text = ""
    for i in range(-warmup, repeat):
        text = ""
        start = time.perf_counter_ns()
        text = template.render(**case["context"])
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if i >= 0:
            samples.append(elapsed)
    return {"compile_ms": compile_ms, "samples_ms": samples, **summarize(samples)}, text


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSONL measurements, excluding rendered text")
    parser.add_argument("--cases-out", type=Path, help="Save inputs accepted directly by bench-jinja")
    parser.add_argument(
        "--template", action="append", type=Path, help="Jinja file; defaults to the repository GLM and Qwen templates"
    )
    parser.add_argument("--messages", type=int, nargs="+", default=[101, 401, 1601])
    parser.add_argument(
        "--threads", type=int, nargs="+", default=[1], help="Concurrent renders of one shared compiled template"
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--jinja2", action="store_true", help="Also measure Python Jinja2 and compare output")
    parser.add_argument("--mark-input", action="store_true")
    parser.add_argument("--filter", default="", help="Only run case names containing this text")
    args = parser.parse_args()
    if args.repeat <= 0 or args.warmup < 0 or any(n <= 0 for n in args.threads + args.messages):
        parser.error("repeat, threads and messages must be positive; warmup must be nonnegative")
    if args.template is None:
        args.template = [root / "models/templates/GLM-4.7-Flash.jinja", root / "models/templates/Qwen3.5-4B.jinja"]
    selected = [case for case in cases(args) if args.filter in case["name"]]
    if not selected:
        parser.error("no matching cases")
    requests = [
        {**case, "threads": threads, "repeat": args.repeat, "warmup": args.warmup, "mark_input": args.mark_input}
        for case in selected
        for threads in args.threads
    ]
    if args.cases_out:
        args.cases_out.write_text("".join(json.dumps(case) + "\n" for case in requests))
    with args.output.open("w") as output:
        for request in requests:
            run = subprocess.run(
                [str(args.binary.resolve())], input=json.dumps(request) + "\n", text=True, capture_output=True
            )
            if run.returncode:
                raise RuntimeError(f"{request['name']}: {run.stdout}\n{run.stderr}")
            result = json.loads(run.stdout)
            text = result.pop("text")
            parts = result.pop("parts")
            result["sha256"] = hashlib.sha256(text.encode()).hexdigest()
            result["parts_sha256"] = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()
            result["output_bytes"] = len(text.encode())
            result.update(summarize([s["total_ms"] for s in result["samples"]]))
            if args.jinja2 and request["threads"] == 1:
                timing, expected = python_render(request, args.warmup, args.repeat)
                result["jinja2"] = timing
                result["matches_jinja2"] = text == expected
                if text != expected:
                    raise RuntimeError(f"{request['name']}: output differs from Jinja2")
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(f"{result['name']} threads={result['threads']} median={result['median_ms']:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
