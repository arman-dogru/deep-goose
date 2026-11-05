# CPU Adam and classification milestones — 2026-11-05

**Exploratory shared GitHub CI measurements, not dedicated-hardware performance claims.**

## End-to-end Adam optimizer benchmark

The same 256-example, two-feature full-batch linear regression was trained for **250 epochs** from zero parameters using Adam with bias correction and manual dense backward gradients.

- [Passing five-implementation CI comparison](https://github.com/arman-dogru/deep-goose/actions/runs/42060658305)
- [Goose implementation](../../examples/regression/adam.goose)
- [C implementation](../c/regression_adam.c)
- [Python/NumPy/PyTorch implementation](../python/regression_adam.py)
- [Benchmark runner](../runner/train_compare.py)

| Implementation | Median full-training time, ms | Final MSE |
|---|---:|---:|
| Goose / Clang | 0.185 | ~4.09e-10 |
| C / Clang | 0.165 | ~4.09e-10 |
| NumPy CPU | 6.709 | ~4.12e-10 |
| PyTorch CPU | 16.745 | ~4.12e-10 |
| Pure Python loops | 20.282 | ~4.08e-10 |

These medians use three independently executed processes on GitHub-hosted CPUs. The small model incurs heavy Python tensor-operation dispatch overhead relative to the numerical work; the experiment does not establish large-model training competitiveness.

## Stable classification operators

- [Stable Goose softmax, log-softmax, cross-entropy and gradients](../../src/ops/classification.goose).
- [Extreme-input and finite-difference gradient tests](../../tests/goose/classification.goose).
- [Passing numeric CI](https://github.com/arman-dogru/deep-goose/actions/runs/42060092564).

## End-to-end three-class training

- [Goose example](../../examples/classification/train.goose).
- [Passing CI](https://github.com/arman-dogru/deep-goose/actions/runs/42061275452).
- Three linearly separable synthetic classes; 90 examples; 2-to-3 dense layer; 300 SGD updates.
- **90/90 training examples classified correctly**.
- Final mean cross-entropy: **0.016333317**.
- One CI run's reported training-loop time: **0.993494 ms**; not a controlled performance result.

This demonstrates classification learning beyond regression and XOR, while remaining a small deterministic smoke test.

## Portable dense-parameter checkpoints

A versioned, Goose-serialized f32 dense-parameter format now saves and loads weights and biases. This uses **Goose v2026.3's supported array serialization** rather than newer binary write helpers.

- [Implementation](../../src/nn/checkpoint.goose)
- [Round-trip and rejection tests](../../tests/goose/checkpoint.goose)
- [Passing CI](https://github.com/arman-dogru/deep-goose/actions/runs/42061959784)

Format records a magic header, version and parameter lengths followed by a Goose-framed f32 array. It rejects invalid headers, mismatched counts and truncated inputs. **It does not contain optimizer state or a cryptographic integrity check.** It should not be described as a full resumable training checkpoint.

## Remaining CPU work

- Broader output-wise numerical references and more random shapes.
- Larger neural networks, multiclass dataset evaluation, and robust optimizer-state checkpoints.
- Blocked/optimized native GEMM and compiler inspection.
- C/PyTorch DQN parity benchmarks.
- Dedicated CPU performance hardware with multiple seeds and confidence intervals.
