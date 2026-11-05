# Native CPU GEMM optimization investigation — 2026-11-05

**Status: Implemented, compiled and benchmarked in Goose v2026.3 and Clang.** All numerical correctness checks passed. **Performance figures are exploratory measurements on a shared GitHub-hosted CPU and must not be interpreted as controlled hardware results.**

## Evidence

- [Passing full GEMM study, codegen diagnostics and raw artifact — CI #42083819305](https://github.com/arman-dogru/deep-goose/actions/runs/42083819305)
- [Download raw benchmark/diagnostics artifact](https://github.com/arman-dogru/deep-goose/actions/runs/42083819305/artifacts/13442446179)
- [Goose benchmark source](../goose/gemm_study.goose)
- [Algorithm-matched C benchmark](../c/gemm_study.c)
- [Independent-process runner and reporting](../runner/compare_gemm.py)
- [Production CPU matmul implementation](../../src/ops/cpu.goose)
- [Library matrix-shape tests](../../tests/goose/matmul_shapes.goose)
- [Dedicated GEMM study workflow](../../.github/workflows/gemm-study.yml)

## What was implemented

Four mathematically equivalent row-major f32 GEMM variants, in both Goose and handwritten C:

1. **`ijk`:** naive outer `i,j`, inner reduction over `k`.
2. **`ikj`:** reorder loops for contiguous innermost reads/writes of `B` and `C`.
3. **`blocked32`:** cache tiling with 32×32×32 tiles, including partial tiles.
4. **`rows`:** `i,k,j` with an entire row represented as a Goose slice or C row pointer before its innermost loop.

The `rows` variant is implemented in the **actual Goose CPU library** (`matmul_f32_into`), not just the benchmark. Its inputs and output must not overlap. No bounds checking was disabled.

All experiments use the same inputs, f32 arithmetic, row-major layout, Clang, and single-thread execution. They use a non-power-of-two scaling pattern (`0.00713` and `0.01517`) so the output is not artificially exact under f32 accumulation. Each program checks **every result element** against an independent f64 dot-product reference after the timed region.

## Experimental protocol

- Platform: GitHub-hosted Ubuntu 24.04 x86-64; **two visible virtual CPUs**, AMD EPYC 9V74.
- Goose compiler: pinned v2026.3; Clang native compiler; Goose language optimization `-O1` by default.
- C: `clang -O2`, `-std=c11`; identical operation order per output element.
- A separate comparison compiles Goose with language-level `-O2`.
- Input construction, warmup, output checking and program startup are outside the timed region.
- Both `ikj`, `blocked32` and `rows` zero the output inside their timed operation. Naive `ijk` overwrites its output.
- Five separate process trials per implementation/algorithm/shape, randomized implementation order within each trial.
- Ten dimensions: 31, 32, 33, 63, 64, 65, 127, 128, 129, 256.
- `144` smoke samples, `400` Goose-O1/C study samples, and `400` Goose-O2/C study samples, **zero reported failures**.
- Largest observed absolute error against the independent reference: approximately **1.05×10⁻⁵**.

### Selected Goose-O1 and C medians

Microseconds per matrix multiplication; smaller is faster. All numbers below come from **the same passing CI execution**.

| N | Algorithm | Goose (µs) | C (µs) | Goose/C latency |
|---:|---|---:|---:|---:|
| 32 | ijk | 20.322 | 8.209 | 2.48× |
| 32 | ikj | 17.420 | 4.982 | 3.50× |
| 32 | blocked32 | 18.899 | 4.937 | 3.83× |
| 32 | rows | **4.555** | **3.785** | **1.20×** |
| 64 | ijk | 156.020 | 81.684 | 1.91× |
| 64 | ikj | 59.323 | 27.868 | 2.13× |
| 64 | blocked32 | 85.901 | 33.476 | 2.57× |
| 64 | rows | **29.903** | **30.451** | **0.98×** |
| 128 | ijk | 1593.557 | 812.948 | 1.96× |
| 128 | ikj | 417.545 | 153.202 | 2.73× |
| 128 | blocked32 | 712.760 | 234.207 | 3.04× |
| 128 | rows | **216.670** | **196.185** | **1.10×** |
| 256 | ijk | 12835.880 | 12919.401 | 0.99× |
| 256 | ikj | 2712.422 | 1201.405 | 2.26× |
| 256 | blocked32 | 5939.687 | 1958.012 | 3.03× |
| 256 | rows | **1569.439** | **1180.149** | **1.33×** |

**Finding 1 — the strongest optimization was how the Goose source expresses row access.** At 128×128, row slices reduced Goose time from ~418 µs (`ikj`) to ~217 µs, a **1.93× improvement** within the same run. At 64×64, they improved ~1.98×. At 256×256, they improved ~1.73×.

**Finding 2 — naive 32³ blocking lost to ordinary row-wise traversal.** At 128×128, `blocked32` took 713 µs versus 217 µs for row slices; the same ordering held in C. Adding tiles without an optimized microkernel, parameter tuning, or register blocking is not automatically beneficial.

**Finding 3 — C remains ahead for the larger `rows` cases but the gap is much narrower.** At 128×128, Goose was ~10% slower in this run; at 256×256, ~33% slower. Earlier GitHub-hosted runs showed different ratios, illustrating why a dedicated benchmark machine and confidence intervals remain essential.

### Goose language optimizer O1 versus O2

The same CI run independently built Goose with `-O2`. Selected medians:

| N | Variant | Goose default O1 (µs) | Goose O2 (µs) |
|---:|---|---:|---:|
| 64 | rows | 29.903 | 30.756 |
| 128 | rows | 216.670 | 214.940 |
| 256 | rows | 1569.439 | 1606.868 |

Higher Goose language optimization did **not consistently improve** the row-slice GEMM kernel in this experiment. It is not an alternative to source-level numerical layout optimization.

## Compiler diagnostics and what they establish

Collected evidence is in the archived `gemm-codegen/` directory:

- Goose `--bce-lines` source-line bounds-check diagnostics.
- Source C emitted by Goose (the native compiler's input).
- Clang loop-vectorization analysis for **both** handwritten C and Goose-generated C.
- Optimized executable disassemblies and symbol tables.
- CPU/compiler versions and raw timing records.

The Goose-O1 bounds-check elimination output reports:

```text
bce: elided 12/36 index and 0/7 slice checks
```

This is the number of **static checks/sites considered by the compiler**, not an estimate of how many checks executed at runtime. Some of these checks occur outside GEMM. In the emitted C, matrix row calculations include checked integer arithmetic and some `GS_IDX` bounds checks, while handwritten C uses direct pointer arithmetic.

Clang's vectorization diagnostics are important:

- C `ikj`, `blocked32`, and `rows` inner `j` loops: **vectorized (width 4, interleave count 2)**.
- Goose-generated C: Clang also reports vectorizing the corresponding loops (width 4). The generated `rows` loop is **interleaved by 2**, while the generated `ikj`/`blocked32` loops were interleaved by 1 in this diagnostic compilation.
- Neither source's `ijk` innermost `k` reduction was reported vectorized under strict floating-point reassociation settings.

**Correct interpretation:** It is *false* that Goose merely fails to vectorize. Both languages achieve SIMD for appropriate loop shapes. The smaller Goose row-slice gap plausibly relates to simpler memory expressions and better vectorization opportunities, while checked indexing, integer-overflow checks, extra control flow, alias analysis, and runner noise could affect remaining differences. **This experiment does not causally quantify any one mechanism.**

The separate vectorization diagnostic compilation of Goose-emitted C is close to but not necessarily identical to the exact build invocation used to produce the Goose executable. Inspect the assembly and actual binary for definitive machine-code conclusions.

## Code shipped to Deep Goose

The native `matmul_f32_into` function in `src/ops/cpu.goose` now uses row slices for `A`, `B` and output rows, with contiguous `j` updates and output initialization. It preserves the numerical accumulation order per output element. This is an **algorithm implementation improvement**, not a benchmark-only change.

The standalone `tests/goose/matmul_shapes.goose` exercises:

- `1×1 @ 1×1`
- `2×3 @ 3×4`
- `7×5 @ 5×9`
- `33×34 @ 34×31`
- `65×63 @ 63×67`
- Zero-inner-dimension and zero-output-dimension cases.

Each output element is checked against an f64 reference. Non-dyadic input scaling was also introduced for these tests.

## Reproduce the investigation

```bash
goose check benchmarks/goose/gemm_study.goose
goose build --cc clang -o build/goose_gemm_study benchmarks/goose/gemm_study.goose
clang -O2 -std=c11 -D_POSIX_C_SOURCE=200809L \
  -Wall -Wextra -Werror -o build/c_gemm_study benchmarks/c/gemm_study.c -lm

python3 benchmarks/runner/compare_gemm.py \
  --goose build/goose_gemm_study --c build/c_gemm_study \
  --mode study --trials 5 --output benchmarks/reports/local-gemm

goose check benchmarks/goose/gemm_study.goose --bce-lines
goose build --cc clang -O2 -o build/goose_gemm_study_o2 \
  benchmarks/goose/gemm_study.goose

python3 benchmarks/runner/compare_gemm.py \
  --goose build/goose_gemm_study_o2 --c build/c_gemm_study \
  --mode study --trials 5 --output benchmarks/reports/local-gemm-o2
```

## Remaining performance investigation

1. Run on dedicated hardware with CPU affinity, stable clocks and many independent trials.
2. Inspect the hot `rows` machine-code loop, check count and alias conditions in detail; quantify differences through profiler counter data.
3. Compare a C control that performs **equivalent** dimension/overflow/bounds validation, not merely unchecked C pointers.
4. Test a tuned GEMM microkernel with register blocking and tile sizes selected by actual cache architecture.
5. Benchmark larger real MLP inference/training with the native operator and OpenBLAS before CUDA work.

**Decision:** Adopt row slicing for native CPU GEMM; retain OpenBLAS as the optimized GEMM backend for serious dense workloads. Do **not** adopt the current blocked32 implementation as a production optimization.
