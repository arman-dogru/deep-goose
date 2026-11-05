#!/usr/bin/env python3
"""Matched Goose/C dense backward performance and numerical validation.

Every subprocess checks dW, db, dX elementwise against f64 references.
Sources use the same f32 inputs and accumulation order per gradient element.
All reported latencies are exploratory on GitHub-hosted virtualized CPUs.
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

CASES={
    "smoke":[(1,1,1,4),(1,7,9,8),(7,65,67,3),(32,64,64,2)],
    "study":[
        (1,128,128,300),(7,65,67,125),(32,64,64,75),
        (64,128,128,25),(128,128,128,15),
        (64,256,128,12),(128,256,128,10),
    ],
}
VARIANTS=("naive","rows_dw","rows_all")
IMPLS=("goose","c")


def parsed(stdout, impl, variant, B,K,N,iterations):
    lines=[x for x in stdout.splitlines() if x.startswith("BACKWARD_RESULT,")]
    if len(lines)!=1:
        raise ValueError(f"expected one BACKWARD_RESULT: {stdout[:400]}")
    fields=lines[0].split(",")
    if len(fields)!=10:
        raise ValueError(f"expected 10 CSV fields: {fields}")
    _,who,which,b,k,n,reps,elapsed,error,checksum=fields
    if (who,which,int(b),int(k),int(n),int(reps)) != (impl,variant,B,K,N,iterations):
        raise ValueError(f"wrong output identifiers: {fields}")
    elapsed_ns=int(elapsed)
    max_error=float(error)
    checked_sum=float(checksum)
    if elapsed_ns<=0 or max_error<0 or not math.isfinite(max_error) or not math.isfinite(checked_sum):
        raise ValueError(f"invalid timing or numerical result: {fields}")
    return {
        "implementation":impl,"variant":variant,"batch":B,
        "in_features":K,"out_features":N,"iterations":iterations,
        "elapsed_ns":elapsed_ns,"us_per_backward":elapsed_ns/iterations/1000.0,
        "max_abs_error":max_error,"checksum":checked_sum,
    }


def command_text(*args):
    try:
        return subprocess.run(args, check=True, capture_output=True,
                              text=True,timeout=10).stdout.strip()[:16000]
    except (OSError,subprocess.SubprocessError):
        return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--goose",default="build/goose_backward_study")
    parser.add_argument("--c",default="build/c_backward_study")
    parser.add_argument("--mode",choices=CASES,default="smoke")
    parser.add_argument("--trials",type=int,default=5)
    parser.add_argument("--output",default="benchmarks/reports/backward-study")
    args=parser.parse_args()
    if args.trials<1:
        parser.error("trials must be positive")
    binaries={impl:str(pathlib.Path(getattr(args,impl)).resolve()) for impl in IMPLS}
    for impl, binary in binaries.items():
        if not pathlib.Path(binary).is_file():
            parser.error(f"missing executable for {impl}: {binary}")

    rng=random.Random(20261105)
    raw,failures=[],[]
    for B,K,N,iterations in CASES[args.mode]:
        for variant in VARIANTS:
            for trial in range(args.trials):
                order=list(IMPLS)
                rng.shuffle(order)
                for impl in order:
                    cmd=[binaries[impl],variant,str(B),str(K),str(N),str(iterations)]
                    try:
                        p=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=120)
                        record=parsed(p.stdout,impl,variant,B,K,N,iterations)
                        record["trial"]=trial
                        raw.append(record)
                        print(f"{impl} {variant} {B}x{K}x{N} trial={trial}: "
                              f"{record['us_per_backward']:.3f} us "
                              f"max_error={record['max_abs_error']:.8g}",flush=True)
                    except (OSError,ValueError,subprocess.SubprocessError) as e:
                        message=f"{impl} {variant} {B}x{K}x{N} trial={trial}: {e}"
                        failures.append(message)
                        print(f"FAIL: {message}",flush=True)

    summary=[]
    for B,K,N,_ in CASES[args.mode]:
        for variant in VARIANTS:
            stats={}
            for impl in IMPLS:
                records=[r for r in raw if
                         (r["implementation"],r["variant"],r["batch"],
                          r["in_features"],r["out_features"])==(impl,variant,B,K,N)]
                if len(records)!=args.trials:
                    failures.append(f"incomplete trials: {impl} {variant} {B}x{K}x{N}")
                    continue
                sample=[r["us_per_backward"] for r in records]
                stats[impl]={"median_us":statistics.median(sample),
                             "min_us":min(sample),"max_us":max(sample),
                             "trials":len(sample),
                             "max_abs_error":max(r["max_abs_error"] for r in records)}
            if len(stats)==2:
                summary.append({
                    "batch":B,"in_features":K,"out_features":N,
                    "variant":variant,
                    "goose":stats["goose"],"c":stats["c"],
                    "goose_over_c_latency":stats["goose"]["median_us"]/stats["c"]["median_us"],
                })
    output=pathlib.Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    hardware={
        "timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "github_sha":os.environ.get("GITHUB_SHA"),
        "platform":platform.platform(),
        "cpu_count":os.cpu_count(),
        "lscpu":command_text("lscpu"),
        "goose":command_text("goose","version"),
        "clang":command_text("clang","--version"),
        "caveat":"Shared CI runner, not a controlled benchmark host",
    }
    for path,value in [
        ("raw_results.json",raw),("summary.json",summary),
        ("failures.json",failures),("hardware.json",hardware),
    ]:
        (output/path).write_text(json.dumps(value,indent=2)+"\n")
    rows=[
        "# Dense backward: matched Goose/C CPU performance","",
        "> GitHub-hosted virtualized CPU; these are exploratory timings.",
        "> f32 weight, input, output gradients; all checked elementwise against",
        "> independent f64 references outside the timed region.",
        "",
        f"Mode: `{args.mode}`, process trials per configuration: {args.trials}.","",
        "| B×K×N | Kernel | Goose median µs | C median µs | Goose/C latency |",
        "|---|---|---:|---:|---:|",
    ]
    for s in summary:
        rows.append(f"| {s['batch']}×{s['in_features']}×{s['out_features']} "
                    f"| {s['variant']} | {s['goose']['median_us']:.3f} "
                    f"| {s['c']['median_us']:.3f} | {s['goose_over_c_latency']:.2f}× |")
    rows.extend(["",f"Samples: **{len(raw)}**. Failures: **{len(failures)}**."])
    if failures: rows.extend(f"- {x}" for x in failures)
    (output/"report.md").write_text("\n".join(rows)+"\n")
    print(f"Saved {len(raw)} samples, {len(summary)} groups, {len(failures)} failures",flush=True)
    raise SystemExit(1 if failures else 0)


if __name__=="__main__":
    main()
