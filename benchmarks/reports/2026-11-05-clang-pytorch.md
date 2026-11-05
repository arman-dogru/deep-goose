# CPU baseline: corrected Clang and PyTorch comparison (2026-11-05)

**Status:** exploratory GitHub-hosted CPU benchmark. **Not** controlled research hardware; compiler-matched to a greater extent than the first run.

- [CI run](https://github.com/arman-dogru/deep-goose/actions/runs/42053827913)
- [Raw benchmark artifact](https://github.com/arman-dogru/deep-goose/actions/runs/42053827913/artifacts/13428540945)
- Source commit: `0b5eab84624c81d6748f2cc6371254a7b014ab8f`.
- Goose release: `v2026.3`, compiled with Clang 18.1.3.
- Handwritten C: compiled with **the same Clang 18.1.3** at `-O2`.
- Python 3.12.3; NumPy 2.5.4; PyTorch 2.15.0+cpu; common BLAS/OpenMP thread counts fixed to one.
- Runner: GitHub-hosted Ubuntu 24.04, x86-64 VM, two visible CPUs.
- 132 sampled invocations parsed from CI logs, zero checksum/execution failures reported.
- Separately, the Goose CPU library regression test passed (tensor storage, dense forward/backward, SGD, Q-learning and replay buffer).

## Selected measured medians

All values are **microseconds per kernel invocation**; lower is better. Process launch and input setup excluded. Three trials per implementation/case.

| Operation | Elements/dimension | Goose (Clang) | C (Clang) | NumPy | PyTorch CPU | Pure Python |
|---|---:|---:|---:|---:|---:|---:|
| add | 65536 | 27.374 | 20.403 | 20.441 | 22.954 | 3144.396 |
| relu | 65536 | 22.987 | 23.946 | 54.424 | 28.907 | 3982.168 |
| dot | 65536 | 64.131 | 63.525 | 11.732 | 15.496 | 3018.722 |
| matmul | 64 | 318.219 | 147.365 | 12.709 | 14.226 | 23419.378 |
| matmul | 128 | 3501.025 | 2962.279 | 68.468 | 66.875 | — |


## What the results actually say

1. With the same Clang toolchain, native Goose and handwritten C sequential `f32` dot products are close at N=65,536. The optimized NumPy/PyTorch implementations remain much faster.
2. Goose's naive row-major GEMM is slower than matched C and substantially slower than optimized NumPy/PyTorch matrix operations. Cache tiling, loop order, vectorization and BLAS integration remain major optimization work.
3. Elementwise results vary by size and workflow; the small-operation timings are particularly sensitive to measurement noise and compiler transformations. Do **not** publish a universal Goose-vs-C speedup from this small exploratory run.
4. NumPy/PyTorch dispatch to optimized numerical backends; ReLU is not fused across add+ReLU in those versions, unlike the single-loop Goose/C code. This is not an equal-algorithm comparison.
5. The `f32` checksums match within the runner's aggregate tolerances, but cancellation can mask elementwise mistakes. The independent Goose library regression test checks explicit expected values on small tensor examples, and broader numerical reference tests are still required.

## Required next experiments

- [ ] Dedicated CPU benchmark machine with stable clock/thermal/power conditions.
- [ ] GCC-vs-GCC and Clang-vs-Clang matched comparisons on the same source-level optimization choices.
- [ ] Compiler-generated C and SIMD/bounds-check diagnostics for Goose versus handwritten C.
- [ ] GEMM loop-reordered, blocked and shared BLAS backends.
- [ ] Larger workloads and randomized trial ordering with more independent repetitions.
- [ ] Full tensor elementwise numerical checks, gradient checks and end-to-end training experiments.

Raw evidence (hardware JSON, 132 or fewer sample records, and generated report) is attached to the CI run. This Markdown is a reviewed interpretation rather than a fabricated benchmark.
