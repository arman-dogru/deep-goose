# Deep Goose

**Deep Goose** is an experimental, CPU-first AI and reinforcement-learning library implemented primarily in [Goose](https://aardappel.github.io/goose/), with matched C, Python, NumPy, and PyTorch baselines. The project is evaluating Goose's numerical performance, memory/lifetime model, native-library interoperability, and usefulness for AI systems. A CUDA backend is the next major phase, not a currently available feature.

**Status (2026-11-05):** working and CI-tested CPU mathematical operators, dense forward/backward passes, SGD and Adam, nonlinear XOR training, tabular Q-learning, fixed-capacity replay storage, and optional OpenBLAS GEMM. A complete native CPU DQN example is present and under CI validation. See [implementation status](docs/IMPLEMENTATION_STATUS.md) for verification status and remaining work.

## Project documents

- [Engineering plan](ENGINEERING_PLAN.md) — detailed CPU-to-CUDA roadmap, acceptance gates, benchmark protocol.
- [Implementation status](docs/IMPLEMENTATION_STATUS.md) — delivered, tested, and remaining work.
- [Latest OpenBLAS benchmark](benchmarks/reports/2026-11-05-openblas.md) — real Goose/C/NumPy/PyTorch exploratory measurements.
- [CPU MLP training benchmark](benchmarks/reports/2026-11-05-regression-training.md) — matched regression workload.
- [GEMM loop order benchmark](benchmarks/reports/2026-11-05-gemm-loop-order.md) — naive versus i-k-j.
- [Tabular LineWorld comparison](benchmarks/reports/2026-11-05-lineworld.md) — Goose versus C/Python learning.

All reported GitHub-hosted benchmark numbers are **exploratory**. Dedicated hardware is required for publication-quality performance claims.

## Implemented capabilities

| Component | Current implementation |
|---|---|
| CPU tensor operations | Contiguous row-major `f32` storage, add, fill, dot, GEMM, ReLU |
| Neural networks | Dense forward, dense backward, ReLU backward, MSE and derivative |
| Optimizers | SGD and Adam primitive with explicit first/second moments |
| CPU training | Synthetic regression; nonlinear XOR MLP |
| Numerical tests | Small tensor references, finite-difference dense gradients |
| RL | Standard Q-learning, replay buffer, tabular LineWorld, DQN LineWorld example |
| Optional BLAS | Goose/C adapter to OpenBLAS CBLAS SGEMM |
| Benchmarking | Same input generation, randomized trial order, raw JSON, summary reports |
| Comparison backends | Native Goose, handwritten C, Python loops, NumPy, PyTorch CPU, OpenBLAS |

### Build prerequisites

- Linux x86-64 reference environment.
- [Goose v2026.3](https://github.com/aardappel/goose/releases/tag/v2026.3) binary on `PATH`.
- Clang and standard C build tools.
- Python 3 plus `numpy` and optionally CPU `torch`.
- Optional: OpenBLAS development headers and library (`libopenblas-dev` on Ubuntu).

The CI workflow downloads pinned Goose v2026.3 and validates the examples automatically.

### Build, test, and train

```bash
mkdir -p build

# CPU tensor/optimizer and RL tests
goose check tests/goose/cpu_ops.goose
goose build --cc clang -o build/cpu_ops_test tests/goose/cpu_ops.goose
./build/cpu_ops_test

# Numerical gradient checks
goose build --cc clang -o build/gradient_check tests/goose/gradient_check.goose
./build/gradient_check

# Nonlinear supervised training
goose build --cc clang -o build/xor examples/xor/train.goose
./build/xor

# Full-batch regression benchmark (also built in C and Python)
goose build --cc clang -o build/regression examples/regression/train.goose
./build/regression

# Tabular RL
goose build --cc clang -o build/lineworld examples/rl/lineworld.goose
./build/lineworld

# Neural DQN example — check CI verification status before relying on it
goose build --cc clang -o build/dqn examples/dqn/lineworld.goose
./build/dqn
```

### Compare Goose, C, Python, NumPy, and PyTorch

```bash
# Native CPU kernels
goose build --cc clang -o build/goose_cpu_bench benchmarks/goose/cpu_bench.goose
clang -O2 -std=c11 -D_POSIX_C_SOURCE=200809L \
    -o build/c_cpu_bench benchmarks/c/cpu_bench.c

# Optional BLAS-backed kernels
goose build --cc clang -o build/goose_blas_bench benchmarks/goose/blas_bench.goose
clang -O2 -std=c11 -D_POSIX_C_SOURCE=200809L \
    -o build/c_blas_bench benchmarks/c/blas_bench.c -lopenblas

# Run on a stable dedicated machine for performance claims
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 benchmarks/runner/compare.py \
    --goose build/goose_cpu_bench \
    --c build/c_cpu_bench \
    --goose-blas build/goose_blas_bench \
    --c-blas build/c_blas_bench \
    --implementations goose,c,goose_blas,c_blas,python_loop,numpy,torch_cpu \
    --mode smoke --trials 5 \
    --output benchmarks/reports/local
```

The runner saves `hardware.json`, `raw_results.json`, `summary.json`, `failures.json`, and `report.md`. For a larger workload matrix, use `--mode full`.

### Benchmark integrity

Results distinguish **matched implementations** (same loop and data layout in Goose/C) from **framework comparisons** (NumPy/PyTorch/BLAS call optimized native kernels). The runner excludes initialization, warm-up, process launch and output checksum from the kernel timing. Pure Python loops use Python binary64 floats; the native tensor programs use f32. ReLU is fused in Goose/C but not in the NumPy/PyTorch baseline.

Do **not** generalize a tiny XOR/regression or shared-runner timing to large-model performance. End-to-end model training, numerical checks, memory behavior, and controlled hardware are all required for a defensible conclusion.

## Contributing and scope

The [engineering plan](ENGINEERING_PLAN.md) prioritizes reproducible CPU measurements and correctness before the CUDA phase. The immediate next CPU work includes wider matrix shapes, blocked GEMM and profiler diagnostics, an adapter-matched CBLAS control, fully validated Adam training, stronger deterministic tests, and deep RL benchmarks against equivalent Python/C implementations.

CUDA is deliberately not included in the CPU package dependencies.
