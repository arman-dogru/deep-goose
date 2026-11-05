#!/usr/bin/env python3
"""Independent-process Goose/C GEMM investigation.

Each language implements ijk, ikj, and 32x32x32 blocked GEMM, each validates
every result element against an f64 reference OUTSIDE its timed interval.
Generated raw samples, summary table, and host metadata are retained.
This is experimental shared-host data, not publication-quality performance.
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
    "smoke": [(31, 4), (32, 4), (33, 4), (64, 3), (65, 3), (128, 1)],
    "study": [
        (31, 250), (32, 250), (33, 200),
        (63, 100), (64, 100), (65, 90),
        (127, 20), (128, 20), (129, 18), (256, 4),
    ],
}
VARIANTS = ("ijk", "ikj", "blocked32")
IMPLEMENTATIONS = ("goose", "c")


def parse_result(output: str, impl: str, variant: str, n: int, reps: int):
    rows = [line for line in output.splitlines() if line.startswith("GEMM_RESULT,")]
    if len(rows) != 1:
        raise ValueError(f"expected one GEMM_RESULT row: {output[:300]}")
    fields = rows[0].split(",")
    if len(fields) != 8:
        raise ValueError(f"expected 8 CSV fields, got {len(fields)}")
    _, i, v, shape, repetitions, elapsed, max_error, checksum = fields
    if (i, v, int(shape), int(repetitions)) != (impl, variant, n, reps):
        raise ValueError(f"incorrect output identifiers: {fields}")
    elapsed_ns = int(elapsed)
    error, check = float(max_error), float(checksum)
    if elapsed_ns <= 0 or not math.isfinite(error) or not math.isfinite(check):
        raise ValueError(f"invalid timing or numerical result: {fields}")
    if error < 0 or error > 0.05:
        raise ValueError(f"excessive absolute error {error}")
    return {
        "implementation": i, "variant": v, "n": n,
        "repetitions": reps, "elapsed_ns": elapsed_ns,
        "us_per_gemm": elapsed_ns / reps / 1000.0,
        "max_abs_error": error, "checksum": check,
    }


def machine_info():
    def command(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  check=True, timeout=10).stdout.strip()[:12000]
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else None,
        "lscpu": command("lscpu"),
        "clang": command("clang", "--version"),
        "goose": command("goose", "version"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "warning": "GitHub-hosted runners are not dedicated controlled hardware.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goose", type=pathlib.Path, default=pathlib.Path("build/goose_gemm_study"))
    p.add_argument("--c", type=pathlib.Path, default=pathlib.Path("build/c_gemm_study"))
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("benchmarks/reports/gemm-study"))
    p.add_argument("--mode", choices=SIZES, default="smoke")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()
    if args.trials < 1 or args.timeout < 1:
        p.error("trials and timeout must be positive")
    binaries = {"goose": args.goose.resolve(), "c": args.c.resolve()}
    for name, binary in binaries.items():
        if not binary.is_file():
            p.error(f"{name} executable not found: {binary}")

    rng = random.Random(20261105)
    records, failures = [], []
    for n, reps in SIZES[args.mode]:
        for variant in VARIANTS:
            for trial in range(args.trials):
                order = list(IMPLEMENTATIONS)
                rng.shuffle(order)
                for impl in order:
                    cmd = [str(binaries[impl]), variant, str(n), str(reps)]
                    try:
                        result = subprocess.run(
                            cmd, check=True, capture_output=True, text=True,
                            timeout=args.timeout)
                        row = parse_result(result.stdout, impl, variant, n, reps)
                        row["trial"] = trial
                        records.append(row)
                        print(f"{impl} {variant} {n} trial={trial} "
                              f"{row['us_per_gemm']:.4f} us/GEMM "
                              f"error={row['max_abs_error']:.6g}", flush=True)
                    except (OSError, subprocess.SubprocessError, ValueError) as exc:
                        message = f"{impl} {variant} n={n} trial={trial}: {exc}"
                        failures.append(message)
                        print("FAIL " + message, flush=True)

    summary = []
    for n, _ in SIZES[args.mode]:
        for variant in VARIANTS:
            data = {}
            for impl in IMPLEMENTATIONS:
                timings = [
                    r["us_per_gemm"] for r in records
                    if r["n"] == n and r["variant"] == variant
                    and r["implementation"] == impl]
                if len(timings) != args.trials:
                    failures.append(f"incomplete trials: {impl} {variant} n={n}")
                    continue
                data[impl] = {
                    "median_us": statistics.median(timings),
                    "min_us": min(timings),
                    "max_us": max(timings),
                    "samples": len(timings),
                }
            if len(data) == 2:
                summary.append({
                    "n": n, "variant": variant,
                    "goose": data["goose"], "c": data["c"],
                    "goose_over_c_latency": data["goose"]["median_us"] / data["c"]["median_us"],
                })

    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in [
        ("raw_results.json", records),
        ("summary.json", summary),
        ("hardware.json", machine_info()),
        ("failures.json", failures),
    ]:
        (args.output / name).write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Matched Goose versus C GEMM loop-order and blocking study",
        "",
        "> Exploratory GitHub-hosted CPU execution, NOT controlled performance evidence.",
        "> Matched Clang optimization and f32 row-major inputs. Output validated elementwise",
        "> against an independent f64 reference OUTSIDE the timed region.",
        "> All variants accumulate each C[i,j] in increasing k order. i-k-j and blocked",
        "> versions include output zeroing inside the timed call; i-j-k overwrites output.",
        "",
        f"Mode: `{args.mode}`; trials per language/shape/variant: `{args.trials}`.",
        "",
        "| N | Algorithm | Goose median (us) | C median (us) | Goose/C latency | "
        "Goose range (us) | C range (us) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        g, c = r["goose"], r["c"]
        lines.append(
            f"| {r['n']} | {r['variant']} | {g['median_us']:.3f} | "
            f"{c['median_us']:.3f} | {r['goose_over_c_latency']:.2f}x | "
            f"{g['min_us']:.3f}–{g['max_us']:.3f} | "
            f"{c['min_us']:.3f}–{c['max_us']:.3f} |"
        )
    lines += [
        "",
        f"Total successful execution records: **{len(records)}**.",
        f"Reported validation/execution failures: **{len(failures)}**.",
        "",
        "The same mathematical operation and initial values are used. This still does",
        "not isolate compiler effect from Goose runtime checks/array representations.",
        "Use the generated C, assembly, bounds-check logs and Clang vectorization",
        "reports to investigate any sustained difference.",
    ]
    if failures:
        lines += ["", "## Failures", ""] + [f"- {x}" for x in failures]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {len(records)} valid records, {len(summary)} comparison groups, "
          f"{len(failures)} failures to {args.output}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
