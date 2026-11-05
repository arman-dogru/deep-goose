# Native Goose CPU DQN: first end-to-end learning validation (2026-11-05)

**Status:** CI-verified complete CPU Deep Q-Network training run. This is a functional milestone, **not** an apples-to-apples framework performance comparison.

- [Passing GitHub Actions run](https://github.com/arman-dogru/deep-goose/actions/runs/42058760404)
- [Implementation](../../examples/dqn/lineworld.goose)
- [CPU mathematical kernels](../../src/ops/cpu.goose)
- [Replay and transition implementation](../../src/rl/tabular.goose)

## Algorithm and architecture

- Discrete eight-state LineWorld with two possible actions.
- One-hot state input (length 8).
- Online Q network: `Linear(8,16) → ReLU → Linear(16,2)`.
- Separate target network with hard parameter synchronization every 100 optimizer updates.
- Fixed-capacity 1,024-transition replay buffer; minibatch size 32, deterministic LCG sampling.
- True terminal transitions do not bootstrap.
- Explicit dense backward gradients, chosen-action squared TD error, full-batch SGD; learning rate 0.03, discount 0.95.
- Epsilon-style exploration (15% action exploration under the fixed LCG); deterministic initial weights.
- All neural-network math performed in **native Goose CPU**, no external BLAS, PyTorch, or NumPy dependency for this example.

## Verified result

The first checked end-to-end CI execution reported:

```text
DQN_RESULT,goose,12000,144867010,11969,1227,7,0.73508453
```

| Measure | Measured |
|---|---:|
| Environment interactions | 12,000 |
| Optimizer updates | 11,969 |
| Completed episodes | 1,227 |
| Training loop elapsed CPU wall-clock | 144.867 ms |
| Greedy evaluation steps to terminal | 7 (shortest path) |
| Learned Q(state=0, right) | 0.735085 |
| CI outcome | Passed |

The learned value is consistent with six remaining reward-discount steps at `gamma=0.95`; `0.95^6 ≈ 0.73509`.

This validates model execution, forward/backward computation, replay sampling, target updates, terminal handling, and a learned policy on a small problem. It does not demonstrate general DQN stability across tasks.

## Remaining validation before broad claims

- [ ] Equivalent C and PyTorch implementations with serialized matching initial parameters, RNG action sequence and replay indices.
- [ ] Controlled multi-seed learning curves, convergence confidence intervals, time-to-target reward.
- [ ] Larger networks, batched environments and optimizer variants.
- [ ] Gradient and tensor reference testing across more dimensions.
- [ ] Dedicated CPU hardware and profiled bottlenecks.
- [ ] Save/restore optimizer and DQN network checkpoints.
