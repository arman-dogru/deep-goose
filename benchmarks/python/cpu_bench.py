#!/usr/bin/env python3
"""Pure Python, NumPy, PyTorch CPU baseline; output matches the C/Goose CSV schema.

Usage: python cpu_bench.py <python_loop|numpy|torch_cpu> <add|relu|dot|matmul|matmul_ikj> <n> <iterations>
The timed region excludes input initialization, warmup, and checksum.
Python loop arithmetic uses double-precision Python floats rather than strict f32.
NumPy/PyTorch reductions and GEMM use their optimized implementation; this is
a framework comparison, NOT a matched-algorithm/compiler comparison.
"""
from __future__ import annotations

import sys
import time


def inputs(n: int, matmul: bool):
    count = n * n if matmul else n
    a = [((i % 257) - 128) * 0.0078125 for i in range(count)]
    b = [((i % 199) - 99) * 0.015625 for i in range(count)]
    return a, b


def python_kernel(op: str, a: list[float], b: list[float], out: list[float], n: int):
    if op == "add":
        for i in range(n):
            out[i] = a[i] + b[i]
        return None
    if op == "relu":
        for i in range(n):
            x = a[i] + b[i]
            out[i] = x if x > 0.0 else 0.0
        return None
    if op == "dot":
        s = 0.0
        for i in range(n):
            s += a[i] * b[i]
        return s
    if op == "matmul_ikj":
        for pos in range(n*n):
            out[pos] = 0.0
        for i in range(n):
            for k in range(n):
                v = a[i*n+k]
                for j in range(n):
                    out[i*n+j] += v*b[k*n+j]
        return None
    if op == "matmul":
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += a[i * n + k] * b[k * n + j]
                out[i * n + j] = s
        return None
    raise ValueError(op)


def benchmark(impl: str, op: str, n: int, iterations: int) -> None:
    a_list, b_list = inputs(n, op in ("matmul", "matmul_ikj"))
    count = len(a_list)
    if impl == "python_loop":
        out = [0.0] * count
        def run():
            return python_kernel(op, a_list, b_list, out, n)
        def checksum(last):
            return float(last) if op == "dot" else float(sum(out))
    elif impl == "numpy":
        import numpy as np
        a = np.asarray(a_list, dtype=np.float32)
        b = np.asarray(b_list, dtype=np.float32)
        out = np.empty_like(a)
        if op in ("matmul", "matmul_ikj"):
            a = a.reshape(n, n)
            b = b.reshape(n, n)
            out = out.reshape(n, n)
            def run():
                np.matmul(a, b, out=out)
                return None
        elif op == "add":
            def run():
                np.add(a, b, out=out)
                return None
        elif op == "relu":
            # Separate add and maximum: unlike C/Goose fused scalar loop.
            def run():
                np.add(a, b, out=out)
                np.maximum(out, 0, out=out)
                return None
        else:
            def run():
                return np.dot(a, b)
        def checksum(last):
            return float(last) if op == "dot" else float(np.sum(out, dtype=np.float64))
    elif impl == "torch_cpu":
        import torch
        torch.set_num_threads(1)
        a = torch.tensor(a_list, dtype=torch.float32)
        b = torch.tensor(b_list, dtype=torch.float32)
        out = torch.empty_like(a)
        if op in ("matmul", "matmul_ikj"):
            a = a.reshape(n, n)
            b = b.reshape(n, n)
            out = out.reshape(n, n)
            def run():
                torch.mm(a, b, out=out)
                return None
        elif op == "add":
            def run():
                torch.add(a, b, out=out)
                return None
        elif op == "relu":
            def run():
                torch.add(a, b, out=out)
                out.relu_()
                return None
        else:
            def run():
                return torch.dot(a, b)
        def checksum(last):
            return float(last.item()) if op == "dot" else float(out.double().sum().item())
    else:
        raise ValueError(f"unknown implementation: {impl}")

    run()  # warmup
    start = time.perf_counter_ns()
    last = None
    for _ in range(iterations):
        last = run()
    elapsed_ns = time.perf_counter_ns() - start
    total = checksum(last)
    print(f"RESULT,{impl},{op},{n},{iterations},{elapsed_ns},{total:.17g}", flush=True)


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: cpu_bench.py <python_loop|numpy|torch_cpu> <add|relu|dot|matmul|matmul_ikj> <n> <iterations>")
    impl, op, n_raw, iterations_raw = sys.argv[1:]
    n, iterations = int(n_raw), int(iterations_raw)
    if not 0 < n <= 1048576 or not 0 < iterations <= 1048576:
        raise SystemExit("dimensions / iterations outside safety limits")
    if op == "matmul" and n > 4096:
        raise SystemExit("matrix dimension outside safety limit")
    if op not in {"add", "relu", "dot", "matmul", "matmul_ikj"}:
        raise SystemExit(f"unknown operation: {op}")
    benchmark(impl, op, n, iterations)


if __name__ == "__main__":
    main()
