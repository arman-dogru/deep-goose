# Register-blocked GEMM in Goose and C — 2026-11-05

**Result: Neither tested register microkernel improves native Goose GEMM over row slicing. Retain row slicing for the library.**

## Reproducible evidence

- [Passing GitHub Actions run](https://github.com/arman-dogru/deep-goose/actions/runs/42085836403)
- [Raw benchmarks, generated C, bounds-check diagnostics and assembly](https://github.com/arman-dogru/deep-goose/actions/runs/42085836403/artifacts/13442676914)
- [Matched Goose kernels](../goose/gemm_study.goose)
- [Matched handwritten C kernels](../c/gemm_study.c)
- [Experiment runner](../runner/compare_gemm.py)
- Source commit tested: `9bd96c1be5ad7eb25f3167bb72b83a55d5d90b51`.

All measurements are **exploratory GitHub-hosted CPU timings**, not controlled research benchmarks.

## Methods

Two new experimental kernels were compared with the established row-slice `i-k-j` implementation:

1. **`micro4`:** a 4×4 output tile, 16 scalar accumulators held across the full `k` dimension, stored once per tile. Partial rows/columns use a correct row-wise fallback.
2. **`micro1x4`:** a 1×4 tile with four accumulators, using a four-element Goose slice of each B row in an attempt to hoist inner-loop bounds validation.

All kernels use the same row-major `f32` input matrices, deterministic **non-dyadic input scaling**, increasing-`k` accumulation order per output, and output initialization **inside** the timed operation. Identical C implementations use row pointers. Every computed element is compared against an independent `f64` reference outside the timed region, with irregular dimensions `31,32,33,63,64,65,127,128,129,256`.

Execution evidence:

- Smoke check: 216 successful executions.
- Main O1 Goose/C comparison: 600 executions, five independent process trials per language/shape/algorithm.
- Goose O2/C comparison: 600 executions.
- **1,416 successful executions in total; zero numerical or execution failures.**

## Selected main-run medians

Microseconds per square GEMM, lower is better. These numbers are taken from one passing CI run; do not combine absolute values with prior runs on different virtual CPU allocations.

| N | Variant | Goose (µs) | C (µs) | Goose/C |
|---:|---|---:|---:|---:|
| 32 | rows | **4.604** | 3.544 | 1.30× |
| 32 | micro1x4 | 10.330 | 3.419 | 3.02× |
| 32 | micro4 | 8.924 | **1.717** | 5.20× |
| 64 | rows | **29.132** | 29.616 | 0.98× |
| 64 | micro1x4 | 65.419 | 28.295 | 2.31× |
| 64 | micro4 | 65.763 | **18.658** | 3.52× |
| 128 | rows | **218.049** | 204.792 | 1.06× |
| 128 | micro1x4 | 500.426 | 349.966 | 1.43× |
| 128 | micro4 | 549.100 | **141.770** | 3.87× |
| 256 | rows | **1578.147** | **1142.923** | 1.38× |
| 256 | micro1x4 | 4127.190 | 2956.468 | 1.40× |
| 256 | micro4 | 4334.681 | 1168.954 | 3.71× |

### What changed

At 128×128, `micro4` is 31% faster than C's `rows` implementation but **152% slower** than Goose's `rows` implementation. The 1×4 variant is also ~130% slower than Goose `rows`. At 256×256, both Goose register microkernels are more than 2.6× slower than Goose `rows`.

Goose O2 did not rescue these kernels: at 128×128, medians were `micro4` 563.35 µs and `micro1x4` 504.98 µs, versus row slices 211.24 µs.

**Important observation:** the same general register-blocking idea yields different outcomes in Goose and C. An optimization considered successful in C is not automatically successful when translated to Goose syntax.

## Compiler evidence and plausible causes

The uploaded Goose-generated C for `gemm_micro4` retains explicit overflow-checked integer arithmetic and per-element `GS_IDX` range checks inside the inner `k` loop, as well as a check when creating each B-row slice.

For example, the generated hot loop contains multiple `gs_mul_i64`, `gs_add_i64`, and `GS_IDX` calls/expressions for the four A-row values and four B-column values for each `k`. The handwritten C version uses raw pointer arithmetic and plain array indexing. These are **static code differences**; they do not by themselves quantify executed instructions, CPU pipeline stalls, or the fraction of the slowdown attributable to checks.

The `micro1x4` source tries to reduce the checked access cost via a four-element B slice, but remained slower than row slices. The likely additional problem is the scalar register reduction's serial dependence and loss of effective vectorization compared with row-slice `i-k-j` traversal. A controlled profiler and actual hot-loop assembly comparison would be needed to establish causality.

## Engineering decision

- Keep native Goose `matmul_f32_into` **row-sliced**; its tests and training examples remain passing.
- Keep `micro4` and `micro1x4` in the **benchmark suite only** as reproducible negative results.
- Continue to use the optional optimized BLAS backend for serious dense-matrix workloads.
- Do not adopt an unchecked/unsound variant merely to make a benchmark look competitive.
- The next CPU work should address **dense-layer forward/backward bottlenecks**, larger batch shapes, and compiler-safe ways to expose contiguous access to SIMD.

## Reproduce

```bash
goose build --cc clang -o build/goose_gemm_study benchmarks/goose/gemm_study.goose
clang -O2 -std=c11 -D_POSIX_C_SOURCE=200809L \
  -Wall -Wextra -Werror -o build/c_gemm_study benchmarks/c/gemm_study.c -lm
python3 benchmarks/runner/compare_gemm.py \
  --goose build/goose_gemm_study --c build/c_gemm_study \
  --mode study --trials 5 --output benchmarks/reports/local-gemm
```
