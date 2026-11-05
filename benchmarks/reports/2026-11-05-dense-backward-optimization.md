# Native Goose dense-backward CPU optimization — 2026-11-05

**Status:** Matched Goose/C benchmark and numerical reference tests passed. The optimized implementation is committed in the CPU library; full integration and learning regression CI is tracked separately below.

## Evidence and reproduction

- [Passing backward benchmark CI](https://github.com/arman-dogru/deep-goose/actions/runs/42088202217)
- [Raw benchmark samples, per-shape medians, compiler diagnostics and hardware metadata](https://github.com/arman-dogru/deep-goose/actions/runs/42088202217/artifacts/13443483561)
- [Goose benchmark](../goose/backward_study.goose)
- [Matched C benchmark](../c/backward_study.c)
- [Independent-process comparison runner](../runner/compare_backward.py)
- [CPU library implementation](../../src/ops/cpu.goose)
- [Full-element API regression test](../../tests/goose/dense_backward_shapes.goose)
- [Numerical finite differences for dX, dW, db](../../tests/goose/gradient_check.goose)

All timings are **exploratory** and originate from a shared GitHub-hosted Ubuntu 24.04 virtual CPU. The machine in this run reports an Intel Xeon Platinum 8370C, two visible CPU cores, Clang 18.1.3 and pinned Goose v2026.3. Different GitHub Actions runs may use different processors. Do not combine these results with independent runs to calculate speedups.

## Mathematical operation

For a dense forward layer `Y=XW+b`, backward computes three outputs in `f32`:

- `dW = Xᵀ dY`, shape `in_features × out_features`.
- `db = sum_rows(dY)`, shape `out_features`.
- `dX = dY Wᵀ`, shape `batch × in_features`.

Gradients are **sums**, not batch averages. The caller provides non-overlapping buffers for all inputs and outputs. Every call overwrites all gradient buffers, including in-place accumulation from explicit zero initialization.

The three matched Goose/C implementations are:

1. **`naive`** — original column-first `dW` accumulation, repeated multi-index arithmetic, scalar `dX` reductions.
2. **`rows_dw`** — output-column-contiguous row slices for `dW` and `db`; retains the original `dX` loop.
3. **`rows_all`** — row-sliced `dW`, `db` and `dX` computations.

In all variants the reduction order for each gradient element is preserved, so this is a source-layout optimization rather than a changed numerical algorithm.

## Benchmark protocol

- Deterministic non-dyadic `f32` input, weight and upstream gradient values, equal across Goose and C.
- Warm-up run preceding a timed loop; excludes allocation, initialization, process startup and numerical reference calculation.
- Repeats the **complete** backward operation per timing sample, including output initialization in the row-sliced variants.
- Five independent subprocess trials for seven representative shapes, with randomized Goose/C order per trial; a separate three-trial smoke matrix tests small/irregular shapes.
- Every element of `dW`, `db`, and `dX` checked independently against `f64` references **after** timing, in both languages.
- **72 smoke executions and 210 study executions, zero failures**. Largest observed absolute error in the 210-run main study: approximately `4.11×10⁻⁷`.

## Measured median times for entire dense backward

Microseconds per complete backward invocation, lower is better. All values are from the **same CI run**.

| Batch × Input × Output | Goose naive | Goose rows dW/db | Goose rows all | Goose speedup | C rows all |
|---|---:|---:|---:|---:|---:|
| 1×128×128 | 100.528 | 45.493 | **25.149** | **4.00×** | 22.643 |
| 7×65×67 | 151.137 | 126.487 | **46.799** | **3.23×** | 47.260 |
| 32×64×64 | 425.684 | 237.557 | **190.526** | **2.23×** | 151.367 |
| 64×128×128 | 3,420.165 | 1,949.913 | **1,666.111** | **2.05×** | 1,366.284 |
| 128×128×128 | 7,148.609 | 3,802.500 | **3,160.528** | **2.26×** | 2,235.027 |
| 64×256×128 | 7,085.971 | 3,917.634 | **3,099.457** | **2.29×** | 2,583.758 |
| 128×256×128 | 14,582.756 | 7,951.896 | **6,351.167** | **2.30×** | 4,016.121 |

In every shape tested, `rows_all` was the fastest Goose backward implementation. Reordering `dW` alone helped, but row slices for the `dX` reduction also provided a consistent improvement.

C remains ahead on most larger shapes. At `128×128×128`, the Goose-to-C ratio for `rows_all` was about **1.41×**. The unusually close `7×65×67` timing should not be generalized.

## Correctness and safety

The new public function in `src/ops/cpu.goose` implements the `rows_all` method with the same API and explicit dimensional checks as before. It does not disable the language's slice/range validation.

The public tests cover:

- Matched `dX`, `dW`, and `db` against independent `f64` references on singleton, rectangular, irregular and batched shapes.
- Repeated calls into the same output buffers to ensure all three gradients are overwritten.
- Finite-difference gradients for **inputs, weights and bias** on a small MSE loss.

The pre-existing CI suite additionally exercises the implementation indirectly through SGD regression, Adam regression, XOR nonlinear MLP training, multiclass softmax learning and a native Deep Q-Network. Passing those checks is required before promoting integration to verified status.

## Interpretation

The primary performance issue in the previous `dW` loop was traversing `dY` with a row-sized stride inside the reduction, alongside repeated checked multi-index arithmetic. Reordering to process a contiguous output column span per sampled input feature avoids that access pattern and allows more efficient memory operations.

The `dX` loop still performs a scalar reduction per output element, but pre-established row slices reduce repeated index arithmetic and make the input arrays contiguous.

**No causal percentage can be assigned to bounds checks, vectorization or cache behavior from these timings alone.** The CI artifact includes compiler bounds-check and C vectorization diagnostics. A dedicated CPU, pinned frequencies, more trials and machine-code/profiler analysis remain necessary.

## Decision and next steps

- **Adopt** row-sliced `dense_backward_f32_into` for the public native CPU library if the full regression suite passes.
- Retain the three reference implementations in the benchmark for reproducibility.
- Continue CPU numerical validation on more shapes/initializations, especially gradients through larger nonlinear networks.
- Compare native CPU training at realistic batch sizes and benchmark the optimized OpenBLAS backend; extend the eventual CUDA backend through a shared tensor operation interface.

The reported performance is about this specific row-major `f32` CPU operator, not a universal Goose-versus-C or Goose-versus-PyTorch performance claim.
