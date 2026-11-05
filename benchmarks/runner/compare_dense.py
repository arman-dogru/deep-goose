#!/usr/bin/env python3
"""Compare Goose and C dense-forward kernels with deterministic f32 inputs.

Each executable independently checks every output against an f64 reference.
Raw per-process timings and host metadata are saved, not just medians.
Hosted CI data are exploratory, not publication-grade benchmarking.
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

SIZES = {
    "smoke": [(1, 7, 9, 12), (7, 65, 67, 5), (64, 64, 64, 3), (64, 128, 128, 2)],
    "study": [
        (1, 128, 128, 250), (7, 65, 67, 100),
        (32, 64, 64, 100), (64, 128, 128, 40),
        (128, 128, 128, 20), (64, 256, 128, 20),
    ],
}
VARIANTS = ("naive", "rows")


def parse(text, impl, variant, B, K, N, repetitions):
    rows = [s for s in text.splitlines() if s.startswith("DENSE_RESULT,")]
    if len(rows) != 1:
        raise ValueError(f"expected one DENSE_RESULT line: {text[:250]}")
    parts = rows[0].split(",")
    if len(parts) != 10:
        raise ValueError(f"expected 10 CSV fields: {parts}")
    _, reported_impl, reported_variant, b, k, n, reps, elapsed, error, checksum = parts
    if (reported_impl, reported_variant, int(b), int(k), int(n), int(reps)) != (
        impl, variant, B, K, N, repetitions
    ):
        raise ValueError(f"incorrect result identifiers: {parts}")
    elapsed_ns, max_error, check = int(elapsed), float(error), float(checksum)
    if elapsed_ns <= 0 or max_error < 0 or not math.isfinite(max_error) or not math.isfinite(check):
        raise ValueError(f"invalid numerical result: {parts}")
    return {
        "implementation": impl, "variant": variant, "batch": B,
        "in_features": K, "out_features": N, "iterations": repetitions,
        "elapsed_ns": elapsed_ns, "us_per_forward": elapsed_ns/repetitions/1000,
        "max_abs_error": max_error, "checksum": check,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goose", default="build/goose_dense_study")
    parser.add_argument("--c", default="build/c_dense_study")
    parser.add_argument("--mode", choices=SIZES, default="smoke")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", default="benchmarks/reports/dense-study")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    executables = {impl: str(pathlib.Path(getattr(args, impl)).resolve()) for impl in ("goose","c")}
    rng = random.Random(20261105)
    records, failures = [], []
    for B,K,N,reps in SIZES[args.mode]:
        for variant in VARIANTS:
            for trial in range(args.trials):
                order = ["goose","c"]
                rng.shuffle(order)
                for impl in order:
                    command = [executables[impl], variant, str(B), str(K), str(N), str(reps)]
                    try:
                        r = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
                        row = parse(r.stdout, impl, variant, B, K, N, reps)
                        row["trial"] = trial
                        records.append(row)
                        print(f"{impl} {variant} {B}x{K}x{N}: "
                              f"{row['us_per_forward']:.3f} us/forward "
                              f"max_error={row['max_abs_error']:.7g}", flush=True)
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        failures.append(f"{impl} {variant} {B}x{K}x{N} trial={trial}: {error}")
                        print("FAIL: " + failures[-1], flush=True)
    summary = []
    for B,K,N,_ in SIZES[args.mode]:
        for variant in VARIANTS:
            times = {}
            for impl in ("goose","c"):
                sample = [r["us_per_forward"] for r in records
                          if (r["implementation"],r["variant"],r["batch"],r["in_features"],r["out_features"])
                             == (impl,variant,B,K,N)]
                if len(sample) != args.trials:
                    failures.append(f"incomplete trials: {impl} {variant} {B}x{K}x{N}")
                    continue
                times[impl] = {"median_us": statistics.median(sample),
                               "min_us": min(sample), "max_us": max(sample),
                               "trials": len(sample)}
            if len(times) == 2:
                summary.append({"batch":B,"in_features":K,"out_features":N,"variant":variant,
                                **times,"goose_over_c_latency":times["goose"]["median_us"]/times["c"]["median_us"]})
    def run_version(*command):
        try:
            return subprocess.run(command, check=True, text=True, capture_output=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
    meta = {"timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),
            "github_sha":os.environ.get("GITHUB_SHA"),
            "platform":platform.platform(),"machine":platform.machine(),
            "lscpu":run_version("lscpu"),
            "goose":run_version("goose","version"),
            "clang":run_version("clang","--version"),
            "warning":"Measurements on shared hosted runner are exploratory."}
    out=pathlib.Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    for filename,thing in [
        ("raw_results.json",records),("summary.json",summary),
        ("hardware.json",meta),("failures.json",failures)
    ]:
        (out/filename).write_text(json.dumps(thing,indent=2)+"\n")
    table = [
        "# Goose/C dense-forward CPU comparison","",
        "> Preliminary GitHub-hosted runner measurements; do not infer universal speedups.",
        "> Every output checked against a separate f64 reference outside the timed region.",
        "> Both languages use f32 and Clang; no Python framework used in these matched comparisons.",
        "",
        "| B×K×N | Kernel | Goose median µs | C median µs | Goose/C |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary:
        dims=f"{row['batch']}×{row['in_features']}×{row['out_features']}"
        table.append(f"| {dims} | {row['variant']} | {row['goose']['median_us']:.3f} | "
                     f"{row['c']['median_us']:.3f} | {row['goose_over_c_latency']:.2f}× |")
    table+=["",f"Successful samples: **{len(records)}**. Failures: **{len(failures)}**."]
    if failures: table += [f"- {fail}" for fail in failures]
    (out/"report.md").write_text("\n".join(table)+"\n")
    print(f"Saved {len(records)} samples, {len(summary)} groups, "
          f"{len(failures)} failures",flush=True)
    raise SystemExit(1 if failures else 0)


if __name__=="__main__":
    main()
