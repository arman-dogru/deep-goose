# Deep Goose

An experimental, CPU-first AI and reinforcement-learning library written primarily in [Goose](https://aardappel.github.io/goose/), with reproducible comparisons against C, pure Python, NumPy and PyTorch. A CUDA backend is planned after CPU correctness and performance gates.

**Status:** early CPU benchmark implementation. **No Goose performance claim has been established.**

Read the [engineering plan](ENGINEERING_PLAN.md) for architecture, work phases, acceptance gates and the CUDA roadmap.

## Initial CPU benchmarks

The repository currently contains equivalent Goose and C implementations of:

- \`add\`: \`out[i] = a[i] + b[i]\`.
- \`relu\`: \`out[i] = max(a[i] + b[i], 0)\`.
- \`dot\`: sequential \`f32\` dot product.
- \`matmul\`: naive row-major square GEMM with \`i, j, k\` loop order.

Python baselines include pure Python loops and NumPy, plus optional CPU PyTorch. Benchmarks use deterministic input formulas and exclude input construction, warmup, output checksum and process startup from timed regions.

### Prerequisites

- Linux x86-64, GCC or Clang, Python 3.
- [Goose v2026.3](https://github.com/aardappel/goose/releases/tag/v2026.3) on PATH.
- \`numpy\` (required for automated comparison), \`torch\` (optional).

### Build and run

\`\`\`bash
mkdir -p build
goose check benchmarks/goose/cpu_bench.goose
goose build --cc clang -o build/goose_cpu_bench benchmarks/goose/cpu_bench.goose
cc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -o build/c_cpu_bench benchmarks/c/cpu_bench.c

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 benchmarks/runner/compare.py \
    --goose build/goose_cpu_bench \
    --c build/c_cpu_bench \
    --implementations goose,c,python_loop,numpy,torch_cpu \
    --mode smoke --trials 3 \
    --output benchmarks/reports/latest
\`\`\`

Choose \`--mode full\` for larger workloads and \`--trials 5\` (or more) for additional samples. **Do not treat GitHub-hosted CI timing as controlled benchmark data.** It is primarily for validation and regression smoke tests.

Each implementation outputs:

\`\`\`text
RESULT,implementation,operation,n,iterations,elapsed_ns,checksum
\`\`\`

The runner saves \`hardware.json\`, \`raw_results.json\`, \`summary.json\`, \`failures.json\` and \`report.md\`.

### Comparison limitations

- Pure Python computes using Python \`float\` (binary64), while compiled and tensor implementations use \`f32\`.
- NumPy and PyTorch invoke optimized numerical routines that are **not algorithmically equivalent** to naive GEMM.
- NumPy/PyTorch ReLU uses two operations, while Goose/C fuse add and ReLU in a single loop.
- All results are single-thread oriented; the runner limits common BLAS thread variables.
- Aggregate checksums are only an initial smoke check, not full elementwise correctness tests.
- The benchmark host, build flags, actual compiler revision, warmup, variability and operation granularity are crucial to interpretation.

The next milestones are comprehensive correctness tests, optimized GEMM comparisons, tensor storage and a CPU MLP.

## Directory layout

\`\`\`text
ENGINEERING_PLAN.md
benchmarks/
  c/cpu_bench.c
  goose/cpu_bench.goose
  python/cpu_bench.py
  runner/compare.py
  reports/              # generated, not committed unless reviewed
.github/workflows/cpu-baseline.yml
\`\`\`
