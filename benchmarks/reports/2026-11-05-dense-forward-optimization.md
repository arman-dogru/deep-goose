# Dense-forward CPU optimization: Goose versus C — 2026-11-05

**Result: Row-slice traversal accelerates the native Goose dense-layer forward operation by 3.0×–8.9× across tested shapes.** The implementation has been committed to the public `dense_forward_f32_into` API, preserving explicit shape validation.

## Reproducible evidence

- [Passing dense benchmark CI run](https://github.com/arman-dogru/deep-goose/actions/runs/42086561338)
- [Raw machine metadata, all samples and vectorization diagnostic artifacts](https://github.com/arman-dogru/deep-goose/actions/runs/42086561338/artifacts/13442513243)
- [Standalone Goose benchmark](../goose/dense_study.goose)
- [Matched C benchmark](../c/dense_study.c)
- [Independent-process runner](../runner/compare_dense.py)
- [Optimized library implementation](../../src/ops/cpu.goose)
- [Public API elementwise shape tests](../../tests/goose/dense_shapes.goose)

**Performance qualification:** all timings below are measurements on a GitHub-hosted virtual CPU, not a controlled hardware system. Compare algorithms within the same run, rather than treating microsecond differences as reproducible hardware facts.

## Problem

The existing Goose `dense_forward_f32_into` implementation calculated one output element at a time:

```text
for row:
    for output_col:
        sum = bias[output_col]
        for input_feature:
            sum += X[row, input_feature] * W[input_feature, output_col]
        Y[row, output_col] = sum
```

With row-major weights, the inner loop walks memory with stride `out_features` and performs a scalar reduction. The revised implementation traverses contiguous output and weight rows:

```text
for row:
    Y[row, :] = bias[:]
    for input_feature:
        value = X[row, input_feature]
        Y[row, :] += value * W[input_feature, :]
```

Row slices allow Goose's compiler to reason about contiguous ranges. The algorithm preserves the increasing-input-feature accumulation order for each output. Every call initializes the output from the bias, so repeated calls do not accumulate stale results.

## Matched C and Goose results

Both languages used f32 arrays, the same deterministic non-dyadic inputs and bias, the same operation order per output element, and Clang. Each run excludes allocations and reference validation from the timed interval and checks **every output element against a separate f64 reference**.

**Median microseconds per forward pass** (smaller is better; five process trials per implementation, shape and variant):

| Batch×Input×Output | Goose old | Goose row-sliced | Goose speedup | C old | C row-sliced |
|---|---:|---:|---:|---:|---:|
| 1×128×128 | 14.607 | **1.636** | **8.93×** | 11.208 | 1.733 |
| 7×65×67 | 18.857 | **3.866** | **4.88×** | 9.650 | 3.432 |
| 32×64×64 | 72.819 | **24.287** | **3.00×** | 38.801 | 14.606 |
| 64×128×128 | 925.511 | **181.137** | **5.11×** | 695.222 | 197.627 |
| 128×128×128 | 1674.023 | **218.113** | **7.68×** | 948.679 | 228.850 |
| 64×256×128 | 1672.424 | **228.311** | **7.33×** | 1405.260 | 204.242 |

The updated Goose function is close to the matched C row-slice implementation in four of six cases, and both remain much faster than their old column-first versions. The `32×64×64` case still exhibits a noticeable Goose/C gap. Hosted-runner noise, compiler code generation and loop overhead can influence these ratios.

### Validation scope

- `48` smoke benchmark executions (four shapes), `120` full study executions (six shapes).
- **168 successful runs and zero numerical/execution failures.**
- Largest observed absolute error across the benchmark suite: approximately **6.44×10⁻⁷**, relative to an independent f64 reference.
- Reference validation is applied **element by element**, not via checksum alone.
- Shapes include singleton batch, odd dimensions and larger batch/feature counts.
- The public library now additionally has `tests/goose/dense_shapes.goose`, covering non-square dimensions and repeated writes to the same output.

### Compiler evidence

Clang's report for the handwritten C version confirms vectorizing the innermost contiguous output-column loop (`width 4, interleave 2`). The naive version's inner reduction over input features was not vectorized under these compile settings.

This provides a plausible explanation for the large speedup: contiguous memory access and opportunities for SIMD vectorization, rather than merely a faster compiler setting. The Goose source also avoids repeated multi-index arithmetic in the hottest loop. The experiment does not isolate the individual impact of vectorization, bounds checks, integer overflow checks or cache misses.

## Engineering decision

1. **Adopt row-sliced dense forward** in `src/ops/cpu.goose`, because it wins across all tested Goose cases.
2. Keep both versions in the benchmark source for reproducibility.
3. Preserve the independent dense shape tests and all existing neural-network training regression tests.
4. Maintain non-aliasing as a documented API precondition for distinct input, weight, bias and output buffers.
5. Continue optimizing dense backward, larger MLP training and BLAS-backed execution.

The row-sliced algorithm is a native CPU improvement, not a replacement for tuned BLAS on larger matrices. Dedicated hardware and confidence intervals remain necessary before making general speedup claims.
