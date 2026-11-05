#!/usr/bin/env python3
"""Measure a complete deterministic 2->1 MSE/SGD training loop.

Benchmark inputs and the training specification are in examples/regression/train.goose.
Results are exploratory timings from the runner; not dedicated-machine benchmarks.
"""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
ALL = ("goose", "c", "python_loop", "numpy", "torch_cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", choices=("sgd","adam"), default="sgd")
    p.add_argument("--goose", default="build/goose_regression_train")
    p.add_argument("--c", default="build/c_regression_train")
    p.add_argument("--implementations", default="goose,c,python_loop,numpy,torch_cpu")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--output", default="benchmarks/reports/regression")
    args = p.parse_args()
    expected_tag = "ADAM_RESULT" if args.optimizer == "adam" else "TRAIN_RESULT"
    expected_epochs = "250" if args.optimizer == "adam" else "200"
    impls = [x.strip() for x in args.implementations.split(",") if x.strip()]
    if args.trials < 1 or not impls or any(i not in ALL for i in impls):
        raise SystemExit("invalid trials/implementations")
    records, failures = [], []
    rng = random.Random(20261105)
    for trial in range(args.trials):
        order = impls[:]
        rng.shuffle(order)
        for impl in order:
            if impl == "goose":
                cmd = [str(pathlib.Path(args.goose).resolve())]
            elif impl == "c":
                cmd = [str(pathlib.Path(args.c).resolve())]
            else:
                cmd = [sys.executable, str(ROOT/("benchmarks/python/regression_adam.py" if args.optimizer == "adam" else "benchmarks/python/regression_train.py")), impl]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
                rows = [r for r in result.stdout.splitlines() if r.startswith(expected_tag+",")]
                if len(rows) != 1:
                    raise ValueError(f"expected one result row: {result.stdout[:300]}")
                parts = rows[0].split(",")
                if len(parts) != 8 or parts[1] != impl or parts[2] != expected_epochs:
                    raise ValueError(f"malformed record: {parts}")
                _, _, epochs, ns, loss, w0, w1, bias = parts
                record = {
                    "trial": trial, "implementation": impl, "epochs": int(epochs),
                    "elapsed_ns": int(ns), "loss": float(loss),
                    "w0": float(w0), "w1": float(w1), "bias": float(bias),
                }
                if (record["elapsed_ns"] < 0 or
                    not all(math.isfinite(record[k]) for k in ("loss","w0","w1","bias")) or
                    record["loss"] > 1e-4 or
                    abs(record["w0"]-1.5) > 0.05 or
                    abs(record["w1"]+2.0) > 0.05 or
                    abs(record["bias"]-0.25) > 0.05):
                    raise ValueError(f"incorrect training result: {record}")
                records.append(record)
                print(f"{impl} trial={trial}: {record['elapsed_ns']/1e6:.3f} ms; loss={record['loss']:.6g}", flush=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as ex:
                failures.append(f"{impl} trial={trial}: {ex}")
                print(f"ERROR: {failures[-1]}", file=sys.stderr, flush=True)
    groups=defaultdict(list)
    for record in records:
        groups[record["implementation"]].append(record)
    summary=[]
    for impl in impls:
        if impl not in groups:
            continue
        rows=groups[impl]
        ms=[r["elapsed_ns"]/1e6 for r in rows]
        summary.append({
            "implementation": impl,
            "median_training_ms": statistics.median(ms),
            "min_ms": min(ms), "max_ms": max(ms),
            "median_loss": statistics.median(r["loss"] for r in rows),
            "trials": len(rows),
        })
    output=pathlib.Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"raw.json").write_text(json.dumps(records,indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (output/"failures.json").write_text(json.dumps(failures,indent=2)+"\n")
    lines=[
        f"# Exploratory CPU regression-training comparison ({args.optimizer.upper()})", "",
        f"256 examples, two input features, one output, MSE, full-batch {args.optimizer.upper()}, {expected_epochs} epochs.",
        "Weights initialize to zero; target is \`1.5*x0 - 2*x1 + 0.25\`.",
        f"Reported time includes forward, backward and {args.optimizer.upper()}, excludes dataset creation and final evaluation.",
        "Goose/C use f32 accumulation; Python loop uses f64; NumPy/PyTorch use optimized tensor ops.",
        "Shared CI hardware is **not** controlled benchmark infrastructure.",
        "", "| Implementation | Median training ms | Min ms | Max ms | Median MSE | Trials |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(f"| {s['implementation']} | {s['median_training_ms']:.4f} | "
                     f"{s['min_ms']:.4f} | {s['max_ms']:.4f} | "
                     f"{s['median_loss']:.3g} | {s['trials']} |")
    lines += ["",f"Failures: **{len(failures)}**"]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
