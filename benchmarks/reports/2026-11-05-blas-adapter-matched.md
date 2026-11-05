# Adapter-matched OpenBLAS CPU comparison (2026-11-05)

**Exploratory GitHub-hosted CPU timing, not controlled hardware evidence.**

- [Passing CI run](https://github.com/arman-dogru/deep-goose/actions/runs/42061105147)
- Goose v2026.3, Clang, OpenBLAS single-thread.
- 201 samples across benchmark cases; zero reported failures.
- Separate OpenBLAS correctness checks passed for rectangular (3×4 @ 4×2) and K=1 matrices.

## Median f32 GEMM latency, microseconds

| Matrix | Goose → C adapter → BLAS | C → identical adapter → BLAS | C → direct BLAS |
|---|---:|---:|---:|
| 32×32 | 1.923 | 1.689 | 1.693 |
| 64×64 | 10.171 | 8.889 | 8.934 |
| 128×128 | 76.355 | 69.635 | 70.146 |

C through the adapter closely tracks CBLAS directly. Goose remains slower: at 128×128 the observed median difference is approximately 9.6% relative to the adapter-matched C control; at 32×32 it is 0.234 µs.

**Limitations:** the Goose-facing function performs shape assertions that the C control does not. Compiler code generation, tiny sample sizes, and virtual-host scheduling also differ. These measurements cannot isolate a precise Goose FFI overhead.

Further work: match shape validation in C, inspect generated C/assembly, collect longer repeat timings and use a dedicated benchmark machine.
