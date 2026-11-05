#!/usr/bin/env python3
"""Run and validate independent CPU benchmark executables.

This is orchestration code, never part of the timed numerical kernels.
Program output: RESULT,implementation,operation,n,iterations,elapsed_ns,checksum.
Produces machine metadata, raw run records and a Markdown summary.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
import pathlib
import platform
import random
import statistics
import subprocess
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPECS = {
    "smoke": [
        ("add", 1024, 200), ("relu", 1024, 200), ("dot", 1024, 200),
        ("add", 65536, 10), ("relu", 65536, 10), ("dot", 65536, 10),
        ("matmul", 32, 3), ("matmul", 64, 2), ("matmul", 128, 1),
        ("matmul_ikj", 32, 3), ("matmul_ikj", 64, 2), ("matmul_ikj", 128, 1),
    ],
    "full": [
        ("add", 1024, 1000), ("relu", 1024, 1000), ("dot", 1024, 1000),
        ("add", 65536, 50), ("relu", 65536, 50), ("dot", 65536, 50),
        ("add", 1048576, 5), ("relu", 1048576, 5), ("dot", 1048576, 5),
        ("matmul", 32, 10), ("matmul", 64, 5),
        ("matmul", 128, 3), ("matmul", 256, 1),
        ("matmul_ikj", 32, 10), ("matmul_ikj", 64, 5),
        ("matmul_ikj", 128, 3), ("matmul_ikj", 256, 1),
    ],
}
ALL_IMPLS = ("goose", "c", "python_loop", "numpy", "torch_cpu", "goose_blas", "c_blas", "c_blas_adapter")


def command_for(impl, op, n, iters, goose, c, goose_blas, c_blas, c_blas_adapter):
    if impl == "goose":
        return [str(goose), op, str(n), str(iters)]
    if impl == "c":
        return [str(c), op, str(n), str(iters)]
    if impl == "goose_blas":
        return [str(goose_blas), op, str(n), str(iters)]
    if impl == "c_blas":
        return [str(c_blas), op, str(n), str(iters)]
    if impl == "c_blas_adapter":
        return [str(c_blas_adapter), op, str(n), str(iters)]
    return [sys.executable, str(ROOT / "benchmarks/python/cpu_bench.py"),
            impl, op, str(n), str(iters)]


def metadata():
    def version(command):
        try:
            return subprocess.run(command, text=True, capture_output=True,
                                  timeout=10, check=True).stdout.strip().splitlines()[0]
        except Exception:
            return "unavailable"
    return {
        "utc_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": version([sys.executable, "-c", "import numpy; print(numpy.__version__)"]),
        "torch": version([sys.executable, "-c", "import torch; print(torch.__version__)"]),
        "cc": version(["cc", "--version"]),
        "clang": version(["clang", "--version"]),
        "goose": version(["goose", "version"]),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_environment": os.environ.get("RUNNER_ENVIRONMENT"),
        "warning": "Shared CI/virtualized hosts are exploratory, not controlled research hardware.",
    }


def parse_result(stdout, expected_impl, op, n, iters):
    rows = [row for row in stdout.splitlines() if row.startswith("RESULT,")]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one RESULT row, got: {stdout[:400]}")
    parts = rows[0].split(",")
    if len(parts) != 7:
        raise ValueError(f"expected seven fields, got: {parts}")
    _, impl, reported_op, reported_n, reported_iters, ns, checksum = parts
    if (impl, reported_op, int(reported_n), int(reported_iters)) != (expected_impl, op, n, iters):
        raise ValueError(f"mismatched result metadata: {parts}")
    elapsed = int(ns)
    value = float(checksum)
    if elapsed < 0 or not math.isfinite(value):
        raise ValueError(f"invalid result: {parts}")
    return {"implementation": impl, "operation": op, "n": n, "iterations": iters,
            "elapsed_ns": elapsed, "ns_per_iteration": elapsed / iters,
            "checksum": value}


def checksum_ok(op, n, expected, actual):
    # Aggregate checksums are only a smoke test, NOT elementwise correctness.
    # GEMM and dot can differ due to f32 accumulation and FMA/BLAS order.
    atol = (0.1 if op == "dot" else (0.02 if op.startswith("matmul") else 0.001))
    rtol = (0.015 if op == "dot" else 0.0005)
    return math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol)


def run(args):
    implementations = [x.strip() for x in args.implementations.split(",") if x.strip()]
    if not implementations or any(x not in ALL_IMPLS for x in implementations):
        raise SystemExit(f"select implementations from: {ALL_IMPLS}")
    goose = pathlib.Path(args.goose).resolve()
    c = pathlib.Path(args.c).resolve()
    goose_blas = pathlib.Path(args.goose_blas).resolve()
    c_blas = pathlib.Path(args.c_blas).resolve()
    c_blas_adapter = pathlib.Path(args.c_blas_adapter).resolve()
    for impl, exe in (("goose", goose), ("c", c),
                      ("goose_blas", goose_blas), ("c_blas", c_blas), ("c_blas_adapter", c_blas_adapter)):
        if impl in implementations and not exe.is_file():
            raise SystemExit(f"{impl} executable not found: {exe}")
    cases = [(op,n,iters) for op,n,iters in SPECS[args.mode]
             if n <= args.max_vector_size or op.startswith("matmul")]
    records = []
    failures = []
    rng = random.Random(20261105)
    for op,n,iters in cases:
        selected = [
            impl for impl in implementations
            if not (impl == "python_loop" and ((op.startswith("matmul") and n > 64) or (not op.startswith("matmul") and n > 65536)))
            and not (impl in ("goose_blas", "c_blas", "c_blas_adapter") and op != "matmul")
        ]
        # Compare programming-language performance with randomized process order.
        for trial in range(args.trials):
            order = list(selected)
            rng.shuffle(order)
            for impl in order:
                cmd = command_for(impl, op, n, iters, goose, c, goose_blas, c_blas, c_blas_adapter)
                print(f"[{op} n={n} trial={trial}] {impl}", flush=True)
                try:
                    finished = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout, check=True)
                    row = parse_result(finished.stdout, impl, op, n, iters)
                    row["trial"] = trial
                    records.append(row)
                    print("  {:.3f} us/iteration checksum={:.9g}".format(row["ns_per_iteration"]/1000, row["checksum"]), flush=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as ex:
                    failures.append(f"{impl} {op} n={n} trial={trial}: {ex}")
                    print("  ERROR: " + failures[-1], file=sys.stderr, flush=True)
        checks = [r for r in records if r["operation"] == op and r["n"] == n]
        reference = next((r["checksum"] for r in checks if r["implementation"] == "c"),
                         next((r["checksum"] for r in checks if r["implementation"] == "python_loop"), None))
        if reference is not None:
            for r in checks:
                if not checksum_ok(op, n, reference, r["checksum"]):
                    failures.append(f"checksum mismatch: {op} n={n} {r['implementation']}: ref={reference}, got={r['checksum']}")
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["operation"], row["n"], row["implementation"])].append(row["ns_per_iteration"])
    summary = []
    for (op,n,impl), times in sorted(grouped.items()):
        summary.append({"operation": op, "n": n, "implementation": impl,
                        "median_us": statistics.median(times)/1000,
                        "minimum_us": min(times)/1000,
                        "maximum_us": max(times)/1000,
                        "trials": len(times)})
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"hardware.json").write_text(json.dumps(metadata(), indent=2)+"\n")
    (output/"raw_results.json").write_text(json.dumps(records, indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output/"failures.json").write_text(json.dumps(failures, indent=2)+"\n")
    table = [
        "# Exploratory CPU Baseline", "",
        "> These measurements were collected on a runner/VM. They are NOT controlled hardware performance claims.",
        "> Kernel times exclude allocation, initialization, warmup, checksum and process startup.",
        "> Pure Python uses Python floats; C/Goose/NumPy/PyTorch use f32.",
        "> NumPy/PyTorch dispatch to optimized native kernels; their algorithms may not match naive C/Goose.",
        "> ReLU is fused in C/Goose, but two operations in NumPy/PyTorch. Do not attribute such differences to language alone.",
        "> Dot/GEMM checksums are aggregated smoke checks, not per-element correctness verification.",
        "", "| Operation | N | Implementation | Median us/op | Min us/op | Max us/op | Trials |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        table.append(f"| {row['operation']} | {row['n']} | {row['implementation']} | "
                     f"{row['median_us']:.3f} | {row['minimum_us']:.3f} | "
                     f"{row['maximum_us']:.3f} | {row['trials']} |")
    table.extend(["", f"Validation/execution failures: **{len(failures)}**.", ""])
    for failure in failures:
        table.append(f"- {failure}")
    (output/"report.md").write_text("\n".join(table)+"\n")
    print(f"Saved {len(records)} samples, {len(summary)} summaries, {len(failures)} failures to {output}", flush=True)
    if failures:
        return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goose", default="build/goose_cpu_bench")
    p.add_argument("--c", default="build/c_cpu_bench")
    p.add_argument("--goose-blas", default="build/goose_blas_bench")
    p.add_argument("--c-blas", default="build/c_blas_bench")
    p.add_argument("--c-blas-adapter", default="build/c_blas_adapter_bench")
    p.add_argument("--implementations", default="goose,c,python_loop,numpy")
    p.add_argument("--mode", choices=sorted(SPECS), default="smoke")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-vector-size", type=int, default=1048576)
    p.add_argument("--output", default="benchmarks/reports/latest")
    args = p.parse_args()
    if args.trials < 1 or args.timeout < 1:
        raise SystemExit("trials and timeout must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
