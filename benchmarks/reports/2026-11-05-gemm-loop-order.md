# CPU GEMM loop-order experiment (2026-11-05)

**Status:** measured on a GitHub-hosted virtualized CPU, not a controlled performance machine.

- [Passing CI](https://github.com/arman-dogru/deep-goose/actions/runs/42055298389)
- [Raw benchmark artifact](https://github.com/arman-dogru/deep-goose/actions/runs/42055298389/artifacts/13429246370)
- Commit under test: `f8161621359f4c9740b5584a0c928046559a1350`.
- Goose v2026.3 → Clang 18 `-O2`, C → Clang 18 `-O2`; single-thread f32, same deterministic matrices, same numerical k-summation order.
- Three process repetitions per shape/implementation. All results passed smoke checksum checks.

## Core comparison

Median **microseconds per matrix multiply**, lower is better.

| Matrix size | Goose naive i-j-k | Goose loop-reordered i-k-j | C naive i-j-k | C loop-reordered i-k-j | NumPy matmul |
|---|---:|---:|---:|---:|---:|
| 32×32 | 34.361 | 17.346 | 14.831 | 8.038 | 4.205 |
| 64×64 | 301.185 | 118.102 | 170.806 | 32.376 | 11.161 |
| 128×128 | 3,521.202 | 598.983 | 3,073.171 | 250.560 | 54.853 |

Both native implementations benefit from reordering the loops so the innermost loop writes adjacent output elements and reads adjacent elements of the right-hand matrix. **At 128×128, Goose improved by approximately 5.9× and C by approximately 12.3×** compared with their own naive loops.

However, the Goose i-k-j implementation is still approximately **2.4× slower than the corresponding C implementation** at 128×128. That is an actionable optimization gap, not a reason to assert a global language ranking.

## What to investigate next

- [ ] Goose `--bce-lines` versus emitted C: remaining bounds checks inside inner loops.
- [ ] Compare Clang vectorization reports, assembly, alias analysis and instruction counts.
- [ ] Evaluate loop blocking/tile sizes and register-level accumulation.
- [ ] Shared BLAS (e.g., OpenBLAS) from both C and Goose; test FFI overhead separately.
- [ ] Larger cases and dedicated-machine repetitions, with confidence intervals.
- [ ] Verify individual output elements, not just aggregate checksum.

**Benchmark validity:** Output initialization is inside the timed `matmul_ikj` implementation for both C and Goose; all input allocations are outside timed regions. NumPy calls optimized BLAS, so its line is a framework baseline, not a matched-algorithm control.
