# Native Goose Q-learning: LineWorld CPU control experiment (2026-11-05)

**Status:** CI-verified tabular reinforcement learning correctness, with exploratory CPU timings. These are not controlled performance claims.

- [Passing GitHub Actions run](https://github.com/arman-dogru/deep-goose/actions/runs/42058069523)
- Source code: [Goose](../../examples/rl/lineworld.goose), [C](../c/lineworld.c), [Python](../python/lineworld.py)

## Workload and validation

Eight-state deterministic LineWorld with two actions, terminal state 7, reward 1 on reaching the terminal. Learning uses standard tabular Q-learning (`alpha=0.2`, `gamma=0.95`), an identical deterministic 32-bit linear congruential generator, 10% exploratory actions, 3,000 training episodes and a 64-step episode limit.

The three implementations executed the same **23,116 environment transitions**. After learning, all three select the shortest greedy path (seven steps). The learned `Q(state=0, action=right)` is about 0.73509, consistent with six remaining bootstrapped discounts.

## Exploratory measurement

| Implementation | Total 3,000-episode training time | Transitions | Greedy evaluation |
|---|---:|---:|---:|
| Goose/Clang | 69.385 µs | 23,116 | 7 steps |
| C/Clang | 102.974 µs | 23,116 | 7 steps |
| Pure Python | 11,823.988 µs | 23,116 | 7 steps |

**Interpretation:** Both native implementations are orders of magnitude faster than pure Python loops in this very small tabular workload. The ~70–100 µs native times are extremely short, so shared-runner variability and compiler optimization can overwhelm the true implementation gap. Do **not** interpret the Goose/C difference as a general language speedup.

This experiment demonstrates a working Goose RL transition/update loop and confirms consistent outcomes against C and Python. It does not benchmark neural-network DQN or parallel rollout collection.

## Next experiments

- [ ] Repeat many independent runs, adjust inner repetition count and measure median with confidence bounds.
- [ ] Add vectorized environment stepping and replay insertion/sampling benchmarking.
- [ ] Compare complete DQN learning on a neural network with controlled seeds and equivalent PyTorch/C implementations.
- [ ] Run on dedicated CPU hardware and profile native differences.
