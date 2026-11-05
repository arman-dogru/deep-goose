# First end-to-end CPU training result (2026-11-05)

**Status:** verified correctness and exploratory end-to-end timing on GitHub-hosted CPU runner. **Not controlled benchmark hardware.**

- [Passing CI run](https://github.com/arman-dogru/deep-goose/actions/runs/42054487814)
- [Raw output artifact](https://github.com/arman-dogru/deep-goose/actions/runs/42054487814/artifacts/13428925644)
- Pinned Goose compiler: v2026.3, Clang 18.
- C compiled with Clang 18, `-O2`.
- Python 3.12, NumPy and PyTorch CPU with one thread.
- Three independent process executions for each implementation.
- **All tests passed:** library regression tests, training convergence checks, CPU smoke benchmarks.

## Workload

- Full-batch linear regression with 256 examples and two input features.
- Target `y = 1.5*x0 - 2*x1 + 0.25`.
- Dense forward, MSE backward, dense backward (`dW`, `db`, `dX`), SGD update.
- Zero initial weights and bias, `f32` model in Goose/C/NumPy/PyTorch.
- Learning rate 0.1, 200 epochs, exact same synthetic dataset.
- Times exclude dataset preparation and final evaluation, include **all 200 training iterations**.
- Final losses around `4e-13` for compiled/`f32` tensor implementations (Python loop `1e-13`).

## Measured results

| Implementation | Full training median | Approximate final MSE |
|---|---:|---:|
| Goose (Clang) | **0.157 ms** | 4.20e-13 |
| C (Clang) | **0.151 ms** | 4.20e-13 |
| NumPy CPU | **2.967 ms** | 4.46e-13 |
| PyTorch CPU (manual gradients) | **6.512 ms** | 4.46e-13 |
| Pure Python loops | **21.025 ms** | 1.26e-13 |

Goose and equivalent C are within a few percent on this small fixed-shape workload. The Python tensor frameworks are slower because 200 short iterations perform multiple framework-dispatched operations, so dispatch overhead dominates the tiny mathematical workload.

**Do not extrapolate these numbers to large neural networks.** The benchmark intentionally uses a small 2→1 dense layer, no general autograd, and a single CPU thread. This is a proof of the CPU forward/backward/optimizer integration and a real end-to-end latency comparison, not a claim that Goose universally outperforms PyTorch.

## Implementation files

- [Goose training](../../examples/regression/train.goose)
- [C training](../c/regression_train.c)
- [Python/NumPy/PyTorch training](../python/regression_train.py)
- [Comparison runner](../runner/train_compare.py)

## Next work

- [ ] MLP with nonlinear hidden layers and gradient checking.
- [ ] Adam optimizer and tests.
- [ ] Numerical reference checks on complete tensor outputs.
- [ ] Cache-blocked and BLAS-backed GEMM comparisons.
- [ ] More seeds, larger batches, dedicated CPU benchmark host.
