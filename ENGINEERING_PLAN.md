# Deep Goose — AI Library Engineering Plan

> **Status:** Proposed engineering roadmap  
> **Prepared:** 2026-11-05  
> **Scope:** Goose-native CPU numerical and AI library; reproducible Python/C/Goose comparisons; optional CUDA backend; GPU benchmarking  
> **Repository:** [arman-dogru/deep-goose](https://github.com/arman-dogru/deep-goose)  
> **Primary research question:** Can Goose provide competitive performance and a practical development model for numerical AI and reinforcement learning workloads?

This document is the development plan and benchmark specification. **Nothing in the roadmap is an achieved performance result.** Targets and dates below are provisional decision criteria, not measurements or delivery guarantees.

---

## 1. Executive strategy

Build an AI library primarily in [Goose](https://aardappel.github.io/goose/), beginning with optimized native CPU execution and rigorous performance comparisons against Python and C, then implement a CUDA backend and benchmark it against native CUDA C++ and PyTorch CUDA.

Treat this as **a systems research project that produces a reusable library**, rather than an unverified attempt to replace PyTorch.

### Questions to answer

1. **Language competitiveness:** How does Goose-generated CPU code compare with algorithmically equivalent handwritten C, pure Python, and established numerical Python frameworks?
2. **Framework competitiveness:** Can a Goose tensor/neural-network/RL implementation be correct, efficient, maintainable, and reproducible?
3. **CUDA interoperability:** What overhead and engineering constraints arise when Goose controls CUDA memory, kernel launches, libraries, and asynchronous execution?
4. **End-to-end usefulness:** How fast are inference, training, and RL tasks when all relevant overheads are included?

### Core research principle

Never conflate these comparisons:

- Goose native kernels **vs** equivalent handwritten C kernels: language/compiler/code-generation assessment.
- Goose calling BLAS **vs** C calling the *same BLAS*: host interface/FFI assessment.
- Goose tensor framework **vs** NumPy/PyTorch: framework-level assessment.
- Goose calling identical CUDA kernels **vs** C++ calling them: CUDA host overhead assessment.
- Goose + cuBLAS **vs** CUDA C++ + the same cuBLAS configuration: integration assessment.
- Goose CUDA training **vs** PyTorch CUDA training: full-framework assessment.

A fast external BLAS/CUDA library does not prove that Goose's own numerical kernels are fast. Conversely, native Goose kernels need not outperform highly tuned vendor libraries for the overall Goose framework to be useful.

### Technical direction

\`\`\`text
                         Goose application layer
                   Inference | Training | Reinforcement learning
                                        |
                         Neural networks / optimizers
                                        |
                      Tensor operations and metadata
                                        |
                     Explicit backend selection/specialization
                         /               |               \
                 Native Goose CPU     C / BLAS          CUDA
                         \               |               /
                    Backend-specific storage and lifetime rules
                                        |
                   Independent, reproducible benchmark suite
                    Python | NumPy | PyTorch | C | CUDA C++
\`\`\`

## 2. Success criteria and scope

### Mandatory outcomes

| Outcome | Evidence required |
|---|---|
| Correctness | Cross-language reference tests, numerical tolerances, memory/lifetime tests |
| CPU competitiveness | Raw timing samples, throughput, memory behavior, compiler diagnostics |
| CPU AI capability | Reproducible MLP inference and training, correct gradients, convergence |
| CPU RL capability | Reproducible tabular RL and DQN with environment/learner profiling |
| CUDA control | Working device allocation, copies, synchronization, kernel launches, error handling |
| GPU AI capability | Persistent GPU tensors, numerical ops, MLP forward/backward and optimizer |
| GPU competitiveness | Fair comparison with identical CUDA kernels/cuBLAS, plus PyTorch CUDA |
| Research reproducibility | Pinned toolchains, hardware manifest, source commit, raw results, scripts |

### First-release scope

- Linux x86-64 primary development/benchmark environment.
- Native Goose CPU backend and optional C/BLAS backend.
- \`f32\` initially; \`f64\` for numerical references and CPU validation.
- Contiguous, row-major, primarily 1D/2D tensors.
- Elementwise operations, reductions, matrix multiplication, activations.
- Dense neural networks, explicit backpropagation, SGD and Adam.
- Basic RL environment API, replay buffer, Q-learning, DQN.
- Separate and independently runnable benchmark programs.
- Optional NVIDIA CUDA backend after the CPU research gate.

### Explicit non-goals for the first release

- General-purpose dynamic autograd comparable to PyTorch.
- Native Goose-to-PTX/CUDA compiler backend.
- LLM pretraining, distributed clusters, multi-GPU collectives.
- General-purpose convolutional model zoo or complete cuDNN coverage.
- Arbitrary dynamic graph composition and fully general strided tensors from day one.
- Claiming that Goose outperforms C/PyTorch before the evidence exists.

## 3. Verify Goose's current capabilities before coding

The implementation and benchmark design must follow **the exact Goose compiler revision we pin**, not assumptions imported from Rust, C++, or Python.

Relevant currently documented facilities to verify in a small test program:

- Whole-program compilation to C and optimized native executables.
- Inline arrays, slices, generics, scope-based data lifetimes.
- Current allocation/pool facilities, including \`slot_pool<T>\` and \`slice_pool<T>\`.
- Array growth and stable reference rules.
- Optimizer and bounds-check diagnostics (including \`--bce-lines\` and \`checkfree fn\`, where supported by the pinned revision).
- \`extern fn\` C interoperability and native package/CMake integration.
- Worker threads with isolated memory and typed communication queues.
- Available monotonic timing and seeded random-number facilities.
- Build options for Clang/GCC and output inspection.

**Constraints to plan around:** no documented Goose-to-CUDA device-code backend; no general Python-like dynamic dispatch; shared mutable memory between Goose workers is not the default model; C ABI layout and external pointer lifetimes require explicit verification.

Keep a short \`docs/GOOSE_COMPATIBILITY.md\` documenting every relied-upon feature, the compiler commit, proof-of-concept snippet, and exact build command. Review it when upgrading Goose.

---

# Part I — Software design

## 4. Library architecture

Use the following subsystems:

1. **Core tensor/storage:** dtype, shape, strides, offset, ownership, storage identity, device identity.
2. **Numerical operators:** elementwise, reduction, GEMM, activations and basic batch operations.
3. **CPU backends:** Goose-native kernels and optional C/BLAS adapter, distinguishable in benchmarks.
4. **Neural networks:** dense layers, explicit forward/backward, losses, optimizers, model checkpointing.
5. **Reinforcement learning:** environment interface, collectors, replay, tabular methods, DQN; PPO later.
6. **CUDA backend:** native C ABI adapter, CUDA Runtime, custom kernels, cuBLAS, future optimized libraries.

A shared mathematical operation contract is useful; a runtime-polymorphic object interface is **not required**. Prefer compile-time specialization or explicit backend entry points consistent with Goose's type system.

## 5. Tensor design and invariants

Conceptual tensor metadata:

| Property | Meaning |
|---|---|
| \`dtype\` | Initially \`f32\`/\`f64\`; additional types later |
| \`shape\` | Per-axis logical dimensions |
| \`strides\` | Physical element offsets per axis |
| \`storage\` | Owning storage object or opaque backend handle |
| \`offset\` | Starting element within storage |
| \`device\` | CPU or CUDA device identity |
| \`contiguous\` | Whether logical iteration matches contiguous storage |
| \`owns_storage\` | Whether this object controls storage lifetime |

These are **conceptual properties**, not a claimed compiler-tested Goose struct layout.

Mandatory invariants:

- Dimensions and element counts are validated; multiplication must not overflow.
- Every view is within the underlying allocation.
- Mutable views obey explicit aliasing rules.
- Shapes and dtypes are checked before arithmetic.
- Tensor storage outlives all dependent views and pending operations.
- No device pointer is interpreted as a host pointer.
- CPU/GPU transfer and synchronization behavior is explicit.
- An output buffer may never be silently reallocated while another view depends on it.

### Start with a restricted API

Initial tensor representation: contiguous row-major 1D/2D, fixed dtype per operation, explicit output buffers for low-level kernels. General strides, broadcasting, arbitrary ranks and implicit allocations are later extensions.

Provide two API levels:

- **Low level:** \`matmul_into(A, B, output)\`, \`relu_into(input, output)\`, etc. Explicit memory and timing.
- **Convenience level:** allocates results, composes operators, and owns temporary workspaces.

## 6. Goose memory-lifetime strategy

Goose's scope-based data stacks and pools suggest designing around lifetime categories rather than copying a reference-counted Python tensor model.

\`\`\`text
Training session
├── Persistent weights and biases
├── Persistent optimizer moments/state
├── Replay or dataset storage
├── Per-minibatch activations
├── Per-minibatch gradients
└── Reusable numerical workspace/scratch buffers
\`\`\`

### Storage policy

- Short-lived activations: scope-local buffers or reusable scratch storage.
- Weights/optimizer state: persistent owner or suitable pool.
- Replay buffers: explicit fixed-capacity storage with stable indexing.
- Views: non-owning and never permitted to outlive their storage.
- Operations: avoid hidden temporary allocation where feasible.
- Track reserved virtual address space separately from committed/resident physical memory; Goose stack reservations must not be confused with RSS.

The first implementation should **not** attempt arbitrary graph-shaped tensor ownership. Validate the restricted design with training loops and checkpoint restoration before broadening it.

## 7. Numerical backend API

Initial operation contract:

\`\`\`text
Storage:     allocate, release, copy, fill
Elementwise: add, subtract, multiply, divide, relu, tanh, sigmoid, exp
Reduction:   sum, mean, max, argmax
Linear:      dot, matvec, matmul, transpose
Learning:    dense_forward, dense_backward, softmax, log_softmax
\`\`\`

Document layout (row-major), strides, dtypes, numerical stability, aliasing, error behavior, shape validation, and whether operations allocate.

**Do not hide backend differences:** expose native-Goose versus BLAS/CUDA selection in every benchmark result.

## 8. Neural-network architecture

Start with static or bounded-shape multilayer perceptrons:

\`\`\`text
Input -> Linear -> ReLU/tanh -> Linear -> ReLU/tanh -> Output
\`\`\`

Implement in order:

1. Dense forward: \`Y = XW + b\`.
2. ReLU and tanh.
3. MSE loss.
4. Dense backward:
   - \`dW = X^T dY\`
   - \`db = sum_batch(dY)\`
   - \`dX = dY W^T\`
5. SGD.
6. Adam, including bias correction.
7. Stable softmax/log-softmax and cross-entropy.
8. Sequential model composition.
9. Serialization and full training checkpoint restore.

An explicit backward implementation is enough for a functioning MLP/DQN and avoids prematurely building a full reverse-mode autodiff engine.

## 9. Reinforcement learning architecture

Define an environment interface equivalent to:

\`\`\`text
reset(seed) -> observation
step(action) -> (observation, reward, terminated, truncated)
\`\`\`

Keep environment stepping independent of tensor and neural-network code.

Use explicit RNG state and deterministic seed propagation. Separate \`terminated\` (true terminal state) from \`truncated\` (time limit or external interruption). Correct bootstrapping depends on that distinction.

Core components:

- Bandit and GridWorld examples.
- Q-learning and SARSA.
- Deterministic CartPole dynamics.
- Circular replay buffer, sampling without replacement when required by the workload.
- DQN with target-network updates, epsilon-greedy policy, gradient steps.
- Batched environment stepping.
- PPO only once rollout storage, stochastic action log probabilities, GAE, and policy-gradient training are working.

### Suggested repository layout

\`\`\`text
deep-goose/
├── README.md
├── ENGINEERING_PLAN.md
├── docs/
│   ├── GOOSE_COMPATIBILITY.md
│   ├── ARCHITECTURE.md
│   ├── NUMERICAL_CONTRACT.md
│   ├── BENCHMARK_PROTOCOL.md
│   └── DECISIONS.md
├── src/
│   ├── core/         # tensor, shape, storage, RNG
│   ├── ops/          # elementwise, reduction, matmul, activation
│   ├── nn/           # linear, loss, model composition
│   ├── optim/        # SGD, Adam
│   └── rl/           # environment, replay, Q-learning, DQN
├── backends/
│   ├── cpu/native/
│   ├── blas/         # Goose bindings, C adapter, build sidecar
│   └── cuda/         # Goose bindings + CMake/CUDA C++ adapter
├── benchmarks/
│   ├── specs/
│   ├── goose/
│   ├── c/
│   ├── python/
│   ├── pytorch/
│   ├── cuda_c/
│   ├── runner/
│   └── reports/
├── tests/
│   ├── correctness/
│   ├── gradients/
│   ├── numerical/
│   ├── memory/
│   └── integration/
├── examples/
│   ├── tensor_ops/
│   ├── xor/
│   ├── classification/
│   ├── gridworld/
│   └── dqn/
└── scripts/
    ├── build_cpu.sh
    ├── test_cpu.sh
    ├── benchmark_cpu.sh
    └── benchmark_cuda.sh
\`\`\`

Layout is a proposal; actual Goose package discovery and native-build sidecars must be validated against the pinned compiler.

---

# Part II — CPU research

## 10. Phase 0: Reproducible baseline environment (indicative weeks 1–2)

### Setup and pinning

Establish one reference Linux x86-64 benchmark workstation. Install and pin:

- Goose compiler (exact git commit) and its required build dependencies.
- GCC and Clang.
- CMake; C and C++ toolchains.
- Python interpreter.
- NumPy and its exact BLAS implementation.
- PyTorch CPU build.
- \`pytest\`, profiler tools, \`perf\`.
- Optional Google Benchmark for specific native baselines.

Record:

| Area | Required metadata |
|---|---|
| CPU | Exact model, microarchitecture, socket/core count, clock policy |
| Instruction set | SSE/AVX2/AVX-512 or other ISA features |
| Memory | RAM size/speed, channels, NUMA layout |
| Cache | L1/L2/L3 sizes and sharing |
| Software | OS/kernel, Goose commit, GCC/Clang versions |
| Python | Python/NumPy/PyTorch versions, BLAS vendor/version |
| Build | Optimization flags, target ISA, LTO/fast-math decisions |
| Execution | Thread counts, affinity, governor, environment variables |

### Benchmark host control

- Use the same machine and dataset for all compared implementations.
- Record background load, CPU throttling, temperature, governor/frequency.
- Pin CPU affinity and NUMA placement or record unconstrained placement.
- Separate 1-thread and multi-thread results; cap BLAS/PyTorch thread counts explicitly.
- Perform cache-warm and cache-cold experiments separately.
- Distinguish process startup time from kernel runtime.
- Exclude input generation from kernel microbenchmarks.
- Report both optimized Goose with Clang and Goose with GCC.
- Do not use the convenient TinyCC development route as the primary optimized baseline.

### Deliverables

- Working Goose, C and Python programs computing the same validated result.
- Pinned environment manifest and build scripts.
- Benchmark harness producing raw timing JSON.
- Written benchmark protocol.
- At least one repeatability experiment on an unchanged implementation.

**Gate 0:** Independent programs execute correctly with consistent metadata and measurable repeatability.

## 11. Phase 1: CPU microbenchmarks (indicative weeks 3–5)

Create these implementation families:

| Implementation | What it evaluates |
|---|---|
| Pure Python loops | Interpreter overhead |
| NumPy | Python-facing, optimized numerical backend |
| PyTorch CPU | Production tensor framework |
| Handwritten C | Comparable native loops and memory layout |
| Native Goose | Goose code generation and runtime behavior |
| C + BLAS | External optimized numerical backend |
| Goose + same BLAS | Goose-to-native integration overhead |

**Matched implementation track:** same algorithm, dtype, layout, threading, loop order and optimization model (where possible).  
**Best-available framework track:** allow BLAS, library fusion, multi-threading and optimized implementations; report separately.

### A. Elementwise and reductions

| Operation | Initial sizes | Report |
|---|---|---|
| Vector add/multiply | 1K, 64K, 1M, 16M | Latency, GB/s |
| Dot product | Same | Latency, GFLOP/s |
| Sum/mean | Same | Latency, bandwidth |
| ReLU | Same | Latency, bandwidth |
| Sigmoid/tanh | 1K, 64K, 1M | Latency, scalar/SIMD math effects |
| Softmax | 128, 1K, 16K, batched | Latency, numerical error |

For \`f32\` vector add, nominal logical bytes transferred are \`12N\` (two 4-byte reads and a 4-byte write), so the **logical** effective bandwidth is \`12N / t\`. Hardware traffic can differ due to caching/write allocation.

### B. Matrix and tensor workloads

| Operation | Initial shapes |
|---|---|
| Matrix-vector | 128×128; 1024×1024; 4096×4096 |
| Square GEMM | 64²; 128²; 256²; 512²; 1024²; 2048² |
| Rectangular GEMM | 32×128 @ 128×256; 256×1024 @ 1024×512 |
| Batched GEMM | B=1,16,64 |
| Transpose | Cache-resident and memory-sized matrices |
| Matrix+bias broadcast | Small, medium, large |
| Gather/scatter | Sequential and randomized indices; no conflicting writes |
| Copy/fill | 4KB to hundreds of MB |
| Tensor creation | Small and large allocations |

For \`A[m,k] @ B[k,n]\`, report approximate floating-point work \`2mnk\` and \`GFLOP/s = 2mnk / (1e9 * seconds)\`. Also report latency: GFLOP/s alone is misleading for small matrices.

### C. AI workloads

- Dense forward and dense backward.
- ReLU backward.
- Stable softmax + cross-entropy.
- Layer normalization.
- Adam parameter update.
- Embedding-like random gather.
- Later: quantized inference experiments.

### D. Memory workloads

- Temporary tensor construction/destruction in a loop.
- Explicit workspace reuse versus repeated allocation.
- Growing transition arrays.
- Circular replay insertion and overwrite.
- Random replay minibatch retrieval.
- Tensor view creation versus copy.
- Peak RSS, committed memory, and virtual address-space size **as distinct values**.

### GEMM implementation progression

1. Naive triple loop.
2. Loop-reordered implementation.
3. Cache-blocked/tiled implementation.
4. Auto-vectorized version.
5. Optional explicit SIMD C reference.
6. BLAS-backed version.

Do not compare a highly tuned Goose version against deliberately naive C and attribute the difference to the language. Compare both matched algorithms and best optimized forms.

### Compiler/profiler investigation

- Inspect generated C and flags.
- Run Goose bounds-check diagnostics where available.
- Inspect Clang/GCC vectorization reports.
- Disassemble selected loops and inspect SIMD instructions.
- Measure instructions, cycles, cache misses and bandwidth with appropriate tooling.
- Investigate alignment, loop-carried dependencies, aliasing, and unnecessary data movement.
- Explain large speed differences with evidence rather than speculation.

### Provisional acceptance targets

- All correctness and numerical tolerance tests pass.
- Aim for native Goose throughput at least **80% of matched C** on core supported scalar/loop kernels (project threshold, not an expected result).
- Aim for Goose+BLAS within **10% of C+identical BLAS** on sufficiently large operations (not a guarantee).
- A sustained **5%** performance change triggers investigation, subject to measured noise.
- Publish every important result with raw samples and a hardware/build manifest.

**Gate 1:** A reproducible CPU kernel report, including at least one investigated bottleneck and explanation of where Goose succeeds or underperforms.

## 12. Benchmark methodology: mandatory protocol

Use three distinct levels:

1. **Microbenchmark:** preallocated data, isolated numerical kernel.
2. **Operator pipeline:** e.g., \`matmul + bias + ReLU\`, including real temporary-buffer behavior.
3. **End-to-end:** full inference or training step, including relevant framework/algorithm overhead.

### Timing protocol

- Generate and validate inputs outside the timed kernel region.
- Warm up execution; use an initial default of 10 warm-up iterations.
- Take at least 30 measured samples for short microbenchmarks.
- Adapt the inner repetition count to limit timer resolution error.
- Execute implementation order in randomized/rotated order.
- Keep raw samples; report median, spread (e.g., p10/p90) and confidence intervals where justified.
- Record whether caches are warm and whether allocations are timed.
- Prevent dead-code elimination: consume/validate observable outputs without making checksum computation dominate timing.
- Use independent repetitions for uncertainty; thousands of in-process repeats are not independent machine conditions.
- Use multiple seeds for stochastic ML/RL convergence experiments.
- Never invent or interpolate missing benchmark values.

### Numerical reference checks

For appropriately conditioned elementwise \`f32\` tests, a provisional rule is:

\`\`\`text
abs(actual - reference) <= atol + rtol * abs(reference)
example starting tolerances: atol=1e-6, rtol=1e-5
\`\`\`

These are *not* universal. Reductions, long dot products, gradients and mixed precision need operation-specific tolerances; consider \`f64\` and higher-precision reference calculations.

Also test:

- Shape mismatch, zero dimensions, integer overflow in element counts.
- NaN/infinity, signed zero, tiny/huge magnitudes.
- Non-multiple-of-tile/ISA dimensions.
- Aliasing and overlapping views where supported.
- Reproducibility across seeds and repeated training.

### Result metrics

- Median latency and uncertainty.
- Throughput: elements/s, samples/s, RL steps/s.
- GFLOP/s or TFLOP/s where mathematically applicable.
- Effective logical bandwidth.
- Peak RSS, allocated/committed bytes, VRAM.
- Correctness, maximum error and gradient-check error.
- Time-to-target reward/accuracy.
- Compiler version, flags, commit, CPU topology/affinity.

Goose/C speedup should be consistently defined as:

\`\`\`text
speedup_Goose_vs_C = latency_C / latency_Goose
> 1.0 means Goose is faster
\`\`\`

Report results by workload and shape; avoid a single aggregate score without documented weighting.

### Result schema (illustrative, unmeasured fields explicitly null)

\`\`\`json
{
  "experiment": "matmul_f32",
  "implementation": "goose_native_cpu",
  "compiler": "clang",
  "shape": [1024, 1024, 1024],
  "dtype": "f32",
  "threads": 1,
  "iterations": 100,
  "latency_ms_median": null,
  "latency_ms_p90": null,
  "gflops": null,
  "max_abs_error": null,
  "peak_rss_mb": null,
  "correct": null,
  "git_commit": null,
  "compiler_version": null
}
\`\`\`

A separate manifest should capture hardware, OS, dependency versions, exact commands, environment and precision settings. Store raw timings (JSON/Parquet) and reports generated from them, not just screenshots.

## 13. Phase 2: Native CPU tensor library (indicative weeks 6–10)

Implementation sequence and test gates:

| Step | Deliverable | Minimum validation |
|---|---|---|
| 1 | Shape/dtype/element-count handling | Invalid sizes and overflow |
| 2 | Contiguous owning CPU storage | Bounds, lifetime, stable ownership |
| 3 | Copy/fill and explicit output buffers | Aliasing, correctness |
| 4 | Elementwise math | Multiple sizes and extreme values |
| 5 | Dot/reductions | Stable numerical checks |
| 6 | Row-major GEMM | Square and rectangular, tail sizes |
| 7 | Activation functions | Edge cases |
| 8 | Softmax/log-softmax | Overflow/underflow resistance |
| 9 | Reshape/basic views | Storage/view lifetime |
| 10 | Batched operations | Shapes and batch behavior |
| 11 | Optional C/BLAS backend | Identical inputs/reference |
| 12 | Serialization | Round-trip, malformed input |

Keep the package useful without BLAS or CUDA. Build convenience APIs only after correctness of low-level operations.

**Gate 2:** Native Goose CPU tensor operations run without external numerical code and pass the reference suite; BLAS is optional and measured separately.

## 14. Phase 3: CPU neural networks (indicative weeks 11–15)

### Inference models

| Model | Architecture | Purpose |
|---|---|---|
| Tiny MLP | 32 → 64 → 16 → 4 | Dispatch and small-operation overhead |
| Medium MLP | 128 → 256 → 256 → 10 | Typical MLP CPU inference |
| Large MLP | 1024 → 2048 → 2048 → 1000 | Dense throughput/memory |

For every model, benchmark batches \`1, 8, 32, 128, 512\` if memory allows.

### Gradient checks

Use central finite differences on selected parameters:

\`\`\`text
g_numeric = (L(theta_i + epsilon) - L(theta_i - epsilon)) / (2*epsilon)
\`\`\`

Use \`f64\` where possible for gradient tests, tune epsilon to the scale and operation, and report absolute and relative errors.

### Training tasks

1. **XOR:** nonlinear-function smoke test; correctness before timing.
2. **Deterministic synthetic regression:** fully controlled forward/backward/optimizer benchmark.
3. **MNIST-like classification (e.g., MNIST):** fixed dataset split, preprocessing, architecture, parameter initialization and optimizer.

Compare Goose native, Goose+BLAS, C, NumPy where appropriate and PyTorch CPU. Measure forward latency, backward latency, optimizer time, full step time, samples/second, RSS, final loss/accuracy and **wall-clock time to target accuracy**.

Initial model parameters should be serialized once and loaded identically by implementations. Do not rely on different language RNG algorithms to produce identical weights from a shared seed.

**Gate 3:** Working CPU MLP training with passing gradient checks, checkpoint restore and independently validated learning curves.

## 15. Phase 4: CPU reinforcement learning (indicative weeks 16–18)

### Environments

- Multi-armed bandit.
- Deterministic GridWorld.
- CartPole.
- Batched CartPole.
- Later: small continuous-control environment.

Implement matched dynamics in Goose, C and Python, with controlled initial conditions and action sequences.

### Algorithms

1. Tabular Q-learning.
2. SARSA.
3. DQN with replay sampling and target network.
4. Batched DQN.
5. Later: PPO and advantage estimation.

### Metrics

| Benchmark | Primary measures |
|---|---|
| Environment step | Steps/second and latency |
| Batched environment step | Scaling versus batch count |
| Replay insertion | Transitions/second |
| Replay sampling | Samples/second; bytes copied |
| DQN optimization | Gradient updates/second |
| Complete DQN | Environment steps to target reward; wall-clock time |
| Parallel collection | Worker scaling and queue communication overhead |

Profile environment simulation, sampling, action selection, forward, backward, optimizer and worker communication separately.

Goose workers use isolated memory and typed queues. Account for trajectory copying, synchronization, policy-version consistency and backpressure. Do not assume shared-memory zero-copy training.

**Gate 4 / CPU review:** DQN learns CartPole with correct truncation/termination semantics; publish a CPU research report before starting CUDA.

## 16. CPU go/no-go evaluation

Answer these with measurements:

- Is Goose native arithmetic competitive with matched C?
- Can optimized Goose-generated C vectorize?
- Is calling BLAS from Goose competitive with calling the same BLAS from C?
- Does Goose's memory model improve or hinder temporary tensors/replay?
- Does MLP training remain efficient beyond isolated kernels?
- Are Goose package/lifetime constraints manageable?

| Finding | Proposed direction |
|---|---|
| Native Goose competitive with C | Invest in native CPU kernels |
| GEMM weak, rest good | Keep native Goose operators; integrate BLAS for GEMM |
| BLAS FFI efficient, native math weak | Emphasize systems orchestration and library integration |
| Small-op FFI expensive | Fuse operations and reduce call count |
| Tensor ownership costly/awkward | Simplify tensor model and workspace strategy |
| Broad severe regressions | Reassess architecture before committing to CUDA |

---

# Part III — CUDA backend

## 17. Phase 5: CUDA interoperability (indicative weeks 19–22)

### Core decision

For the first version, **Goose is host code** and **CUDA kernels are written in CUDA C++** or sourced from established CUDA libraries. A Goose-to-PTX compiler is a different, much larger project.

Integration path:

\`\`\`text
Goose API
  -> stable C ABI / extern fn
  -> C/C++ native adapter
  -> CUDA Runtime / Driver + cuBLAS + custom .cu kernels
  -> NVIDIA GPU
\`\`\`

Use Goose native build/CMake capabilities to link CUDA code. Keep CPU-only builds independent of CUDA toolkit availability.

### Candidate package tree

\`\`\`text
backends/cuda/
├── cuda.goose
├── cuda.build
├── CMakeLists.txt
├── include/cuda_api.h
├── native/
│   ├── cuda_api.cpp
│   ├── memory.cu
│   ├── elementwise.cu
│   ├── reduction.cu
│   ├── matmul.cu
│   ├── activation.cu
│   └── optimizer.cu
└── tests/
\`\`\`

Illustrative CMake starting point (not yet tested against the chosen Goose revision):

\`\`\`cmake
cmake_minimum_required(VERSION 3.27)
project(goose_cuda LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
add_library(goose_cuda STATIC
    native/cuda_api.cpp
    native/memory.cu
    native/elementwise.cu
    native/reduction.cu
    native/matmul.cu
)
target_include_directories(goose_cuda PUBLIC
    \${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(goose_cuda PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(goose_cuda PUBLIC CUDA::cudart CUDA::cublas)
\`\`\`

If supported by the pinned Goose package system, the companion sidecar selects \`cmake_target = "goose_cuda";\`. Validate exact invocation and ABI with a minimal build before introducing tensor types.

### First API surface

| Operation | Required semantics |
|---|---|
| Enumerate devices | Return device properties and failure codes |
| Set current device | Explicit device selection |
| Allocate/free device bytes | Clear ownership and error behavior |
| Copy H2D/D2H | Initially synchronous, correct lifetime |
| Synchronize device/stream | Explicit completion |
| Create/destroy stream | Initially optional or restricted |
| Launch vector add | Simple end-to-end proof |
| Return errors | C ABI error values plus descriptive translation |

**Gate 5:** Goose allocates GPU memory, copies inputs, launches vector add, synchronizes, validates results and releases all resources.

## 18. GPU memory ownership: mandatory design

External CUDA allocations are not automatically governed by Goose's ordinary memory safety.

Use **opaque native allocation handles**, not raw device pointers in application code, to avoid exposing unsafe pointer arithmetic.

Conceptual native handle registry:

\`\`\`text
allocation identifier
generation counter
device identity
byte capacity
native device pointer (private)
allocation state
outstanding operations / synchronization state
\`\`\`

Enforce:

- Generation-checked stale-handle rejection.
- Explicit ownership for allocated resources.
- Buffer size and device identity validation.
- No release while work still depends on an allocation.
- Host buffer lifetime retained for asynchronous copies.
- Device/stream synchronization semantics documented.
- Stable, ABI-safe integer/scalar arguments and explicitly supported buffer encodings.
- No assumption that Goose packed structures match C ABI struct layout.

### Initial synchronization model

Start conservatively: one default stream and synchronous host-facing operations. Once correctness and lifetime checks are stable, add streams, events, pinned-memory transfers, and asynchronous operations with explicit completion tracking.

Later optimize allocation with device pools or CUDA stream-ordered allocation as justified by profiling.

## 19. Phase 6: CUDA numerical kernels (indicative weeks 23–25)

Implementation order:

| Priority | Operation | Backend |
|---|---|---|
| P0 | Vector add/multiply | Custom CUDA |
| P0 | ReLU | Custom CUDA |
| P0 | Sum reduction | Custom CUDA |
| P0 | GEMM | cuBLAS |
| P1 | GEMM + bias | cuBLAS + kernel |
| P1 | Stable softmax | Custom CUDA |
| P1 | Layer normalization | Custom CUDA |
| P1 | Dense backward | cuBLAS + reductions |
| P1 | SGD/Adam | Custom CUDA |
| P2 | cuBLASLt / fusion | Tuned library |
| P2 | cuDNN | Selected neural-network operations |
| P2 | CUDA Graphs | Reduce launch overhead |

Persist intermediate tensors on the GPU. **Do not copy outputs back to the host between each operator.**

### Matrix layout

CPU tensors start row-major; traditional cuBLAS conventions are column-major. Handle leading dimensions and transposition algebraically, or use an API supporting the required layout. Never add hidden full-matrix transposes without accounting for their time and memory cost.

### Fusion

Compare:

\`\`\`text
GEMM -> bias kernel -> ReLU kernel
vs
GEMM -> fused bias+ReLU
\`\`\`

Keep fused/unfused baselines clearly labeled. A fusion win is not automatically a Goose-language win.

**Gate 6:** GPU tensor operations are correct, keep data resident between ops, and produce defensible timing relative to native CUDA counterparts.

## 20. Phase 7: GPU neural-network training (indicative weeks 26–28)

Reuse the mathematical training logic, not a separate algorithm fork.

- GPU MLP forward pass.
- Dense backward using GEMM and reductions.
- Activation backward kernels.
- Loss evaluation.
- SGD then Adam.
- GPU-resident parameters, gradients and optimizer moments.
- Full checkpoint round-trip through host storage.
- Validation of learning equivalence to CPU references.

Use the same tiny/medium/large MLP architectures and batch-size sweep as CPU experiments. Compare Goose CUDA, native CUDA C++, PyTorch CUDA, Goose CPU and PyTorch CPU.

**Gate 7:** Fully GPU-executed MLP training with numerical correctness, gradient tests and comparable learning results.

## 21. Phase 8: GPU reinforcement learning (indicative weeks 29–32)

Start with GPU DQN.

\`\`\`text
Goose CPU environments
       |
Transition collection -> replay buffer / minibatch sampling
       |
Selected minibatches transferred to GPU
       |
GPU DQN: online Q -> target Q -> loss -> backward -> Adam
       |
Updated policy / actions for future environment steps
\`\`\`

Benchmark single CartPole, batched CartPole, small DQN and larger-network DQN separately.

A tiny RL agent may run **slower** on GPU due to launch/copy overhead. This is a valid result. Identify batch/model-size break-even points and distinguish rollout throughput from learner throughput.

PPO and broader RL algorithms follow as an advanced release, not prerequisites for the first GPU results.

---

# Part IV — GPU benchmark protocol

## 22. Required GPU comparison tracks

| Implementation | Interpretation |
|---|---|
| CUDA C++ direct kernel | Device/kernel baseline |
| Goose invoking identical kernel | Host-side integration overhead |
| Goose CUDA tensor operation | Framework dispatch + metadata overhead |
| CUDA C++ + cuBLAS | Optimized numerical baseline |
| Goose + identical cuBLAS configuration | Native library integration overhead |
| PyTorch CUDA | Established production framework |

Do not compare unlike algorithms or precision modes and attribute the result to language selection.

### Device timing and transfer timing

CUDA is asynchronous. Distinguish:

1. **Host submission latency:** CPU time to enqueue/dispatch.
2. **Device elapsed time:** CUDA-event measurement around completed GPU work.
3. **Transfer time:** H2D and D2H, payload/host allocation type clearly specified.
4. **End-to-end latency:** application invocation to synchronized valid result, including relevant transfers/allocations.

Conceptually:

\`\`\`text
T_total involves host, transfer, device and synchronization work.
These components can overlap, so they are not always simply additive.
\`\`\`

For CPU timers, synchronize when measuring completion. For kernel-device measurements, use appropriate CUDA events.

### GPU workload matrix

| Operation | Sizes | Metrics |
|---|---|---|
| Minimal kernel launch | Tiny/no-op | Host enqueue and completion latency |
| H2D / D2H transfer | 4KB–1GB | GB/s; pinned/pageable distinction |
| Vector add | 1K–16M elements | Latency, effective GB/s |
| Sum reduction | 1K–16M | Latency, bandwidth |
| GEMM | 128–4096 where VRAM allows | Latency, TFLOP/s |
| MLP inference | B=1–512 | Latency, samples/s |
| MLP training | B=32–1024 | Full step time, samples/s |
| Adam | 1M–100M parameters if feasible | Update throughput |
| DQN | Small and larger models | Wall-clock time to reward, steps/s |

### Configuration controls

- GPU model, memory, compute capability.
- Driver and CUDA toolkit.
- SM architecture compilation flags.
- Power/clock/thermal conditions.
- Precision, accumulation type, TF32/Tensor Core settings.
- Exact cuBLAS algorithms and workspace policies where controllable.
- Graphs/fusion enabled or disabled.
- Allocation inside/outside timed region.
- Residency of all inputs and outputs.
- CUDA stream, synchronization and warm-up.
- Relevant host CPU and data-transfer bus.

Use Nsight Systems for launch/transfer timelines and Nsight Compute for kernel-level behavior.

### Provisional GPU targets

- Numerical correctness mandatory.
- Goose calling the same CUDA kernels should approach direct C++ performance for GPU-dominated workloads; initial acceptance threshold **within 10%**, recognizing small kernels are dominated by overhead.
- Goose+cuBLAS should approach direct cuBLAS for large GEMM; initial target **within 10%**.
- Report microsecond-scale overhead directly for launch-dominated cases.
- GPU DQN must reach comparable reward, not merely high optimizer throughput.

These are **targets for engineering evaluation**, not predictions.

---

# Part V — Quality, risks, and research outputs

## 23. Testing plan

| Layer | What to test | Cadence |
|---|---|---|
| Unit | Shape/dtype, indexing, arithmetic, errors | Every commit |
| Property-based | Random shapes and algebraic properties | Every commit/scheduled |
| Cross-language reference | Against NumPy, PyTorch, or higher precision | Every commit |
| Gradients | Finite differences vs analytical backward | Every relevant change |
| Memory/lifetime | Views, pools, stale handles, pending operations | Every relevant change |
| FFI | ABI layout, native adapter errors, buffer transfer | Every relevant change |
| Integration | MLP, checkpoints, GridWorld, DQN | CI |
| Performance | CPU/GPU regression suite on dedicated machines | Scheduled |

Special cases: zero-sized buffers, overlarge dimensions, tail sizes, NaN/Inf, numerical overflow, malformed checkpoints, allocation failure, invalid CUDA device, double free, stale handle and release-before-completion.

## 24. Major risks and mitigation

| Risk | Consequence | Mitigation |
|---|---|---|
| Goose tensor lifetimes are too restrictive | Complex training API | Explicit workspace, bounded shapes, pools |
| Native numerical code underperforms C | Poor CPU backend | Generated C/SIMD profiling, use BLAS selectively |
| Mismatched C ABI/struct packing | Corruption/undefined behavior | Primitive ABI + tested adapters |
| GPU asynchronous lifetime errors | Use-after-free/data races | Opaque handles, explicit completion tracking |
| FFI overhead on small kernels | Slow operator pipelines | Fuse operations, benchmark dispatch |
| Unfair cross-framework comparisons | False performance claims | Matched and best-available tracks |
| Floating-point variation | Divergent learning curves | Serialized initial params, tolerance tests, multi-seed evaluations |
| Goose compiler changes | Broken builds/results | Pin commits, compatibility tests |
| Thermal/OS benchmark noise | Unreproducible claims | Dedicated host, repeat trials, environment manifests |
| Scope creep | No finished research artifact | Fixed gates, defer general autodiff/compiler work |

## 25. Automated testing and CI

**CPU CI:** build, format/lint if available, unit tests, reference checks, finite-difference gradient tests, short deterministic integration tests.

**GPU CI:** CUDA build and native adapter checks on a GPU-enabled runner; memory/stream/handle tests; kernel correctness; short MLP training smoke test.

**Scheduled performance CI:** full runs on dedicated hardware; performance deltas annotated with variance; no noisy public regressions based on one measurement.

## 26. Experiment artifacts

Each benchmark run should generate:

\`\`\`text
experiments/<run-id>/
├── hardware.json
├── environment.json
├── compiler_flags.json
├── workload.json
├── reference_outputs/
├── raw_timing_samples/
├── summary_metrics.json
├── profiling/
├── charts/
└── report.md
\`\`\`

ML/RL experiments additionally retain initial weights, optimizer configuration, datasets/splits, RNG seeds, checkpoint metadata and learning curves.

Keep Python for **benchmark orchestration/reporting**; this does not contaminate Goose/C kernel timings if orchestration is outside the timed region.

## 27. Required research reports

1. **CPU microbenchmarks:** matched Goose/C and comparisons with Python/NumPy/PyTorch.
2. **CPU tensor and MLP:** operator and end-to-end latency, memory, correctness.
3. **CPU RL:** rollout, replay and optimizer breakdowns; steps-to-reward.
4. **CUDA interoperability:** host dispatch overhead, transfers, errors/lifetimes.
5. **GPU tensors and training:** identical-kernel/cublas comparisons; PyTorch CUDA.
6. **CPU vs GPU break-even:** batch/model dimensions where CUDA starts winning.

Required charts include latency by size, effective bandwidth, GEMM throughput, memory usage, training time-to-target, RL learning curves with seed variation, and CUDA transfer/launch breakdown.

Avoid averaging unrelated workloads into one headline speedup without a documented weighting scheme.

---

# Part VI — Milestones, release gates and first sprint

## 28. Indicative 32-week schedule

| Window | Phase | Deliverable / gate |
|---|---|---|
| Weeks 1–2 | Phase 0 | Toolchain, independent harness, metadata, repeatable smoke benchmark |
| Weeks 3–5 | Phase 1 | Matched Goose/C/Python CPU microbenchmarks, diagnostics and report |
| Weeks 6–10 | Phase 2 | Native CPU tensor operations; optional BLAS |
| Weeks 11–15 | Phase 3 | MLP inference/backprop/SGD/Adam; complete training comparison |
| Weeks 16–18 | Phase 4 | CPU RL, replay, DQN; CPU go/no-go review |
| Weeks 19–22 | Phase 5 | CUDA adapter, handles, copies, synchronization, first kernel |
| Weeks 23–25 | Phase 6 | CUDA tensor operations and cuBLAS benchmarks |
| Weeks 26–28 | Phase 7 | CUDA MLP forward/backward/optimizer training |
| Weeks 29–32 | Phase 8 | GPU DQN, comparisons, profiling and first complete evaluation |

**These are effort estimates for an experienced individual engineer, not completion promises.** Significant compiler work, broader compatibility or serious CUDA memory bugs can change the schedule. Gate results, not the calendar, determine advancement.

## 29. Release structure

### Release 1 — CPU

- Native Goose numerical/tensor API.
- Optional C/BLAS backend.
- Dense neural-network inference and training.
- CPU RL examples, replay and DQN.
- Fully reproducible CPU comparisons and report.

### Release 2 — CUDA

- Device discovery/handles/allocations/transfers/streams.
- CUDA tensor operations and cuBLAS.
- GPU MLP inference, backward and optimization.
- CUDA C++ and PyTorch CUDA comparisons.
- Published GPU profiling and limitations.

### Release 3 — Advanced (conditional)

- More shapes/dtypes/layouts and fused operations.
- PPO and broader RL tasks.
- General reverse-mode autodiff, if justified.
- cuBLASLt/cuDNN, CUDA Graphs, memory pools.
- Optional DLPack interoperability.
- Optional native Goose GPU codegen research as a **separate** project.

## 30. First sprint: exact implementation checklist

### Environment

- [ ] Pin Goose compiler revision; document build and run commands.
- [ ] Install/pin GCC, Clang, CMake, Python, NumPy, PyTorch.
- [ ] Record CPU/cache/RAM/ISA/hardware metadata and build flags.
- [ ] Write \`docs/BENCHMARK_PROTOCOL.md\`.
- [ ] Establish JSON benchmark results and raw-sample storage.

### First matched numerical kernels

- [ ] Vector addition in native Goose.
- [ ] Equivalent C and pure Python implementations.
- [ ] NumPy and PyTorch CPU equivalents.
- [ ] Dot product and ReLU across all implementations.
- [ ] Test 1K, 64K, 1M and 16M elements where feasible.
- [ ] Check output accuracy; prevent dead-code elimination.
- [ ] Measure median latency, spread, logical bandwidth, and RSS.

### First AI-relevant kernel

- [ ] Row-major naive GEMM in Goose and C, with matching loop order.
- [ ] Evaluate 64×64, 256×256 and 1024×1024 cases.
- [ ] Compare with NumPy and PyTorch CPU GEMM.
- [ ] Inspect generated C, compiler optimization and bounds-check diagnostics.
- [ ] Profile at least one gap to understand the cause.
- [ ] Produce \`benchmarks/reports/cpu-baseline.md\` with raw evidence.

### Sprint exit gate

- [ ] All correctness tests pass.
- [ ] Every benchmark has exact hardware/compiler metadata.
- [ ] Repeatability and observed variance are documented.
- [ ] At least one performance bottleneck is explained.
- [ ] No unsupported performance claim appears in the report.

**First research deliverable:** On one fixed CPU, with matched algorithms and optimized builds, how close is Goose to C for vector operations, reductions and GEMM, and what explains the differences?

## 31. Initial technical decisions

| Decision | Starting choice |
|---|---|
| Primary language | Goose |
| Native adapter | C/C++ |
| CUDA kernel language | CUDA C++ |
| CPU platform | Linux x86-64 |
| GPU platform | NVIDIA CUDA |
| Tensor layout | Contiguous row-major |
| Primary training dtype | \`f32\` |
| Reference/gradient-check dtype | \`f64\` where supported |
| Initial CPU high-performance backend | BLAS |
| Initial GPU GEMM backend | cuBLAS |
| First NN | MLP |
| First optimizers | SGD, Adam |
| First deep RL algorithm | DQN |
| Initial differentiation | Explicit backward |
| Backend dispatch | Compile-time specialization or explicit entry point |
| Benchmark runner | Python, outside timed code |
| Release policy | CPU-only must remain usable without CUDA |
| Licensing | Choose before adding third-party code; audit dependencies |

## 32. Final decision rule

**Benchmarks first, CPU tensor correctness second, training third, CUDA fourth.**

At the end of this roadmap, the project must be able to support—or falsify—with reproducible evidence the following assertions:

1. Goose-generated CPU kernels can/cannot approach equivalent C performance on selected workloads.
2. Goose can/cannot call optimized CPU libraries with acceptably low overhead.
3. Goose's memory model is/is not practical for long-lived parameters, ephemeral activations and replay storage.
4. CPU Goose neural-network training and RL can/cannot achieve competitive end-to-end throughput.
5. Goose CUDA bindings can/cannot approach native CUDA C++ when calling identical kernels/libraries.
6. GPU acceleration helps/hurts under explicitly measured model and batch configurations.

Do not preselect the desired outcome. If results favor Goose primarily as an orchestrator for BLAS/CUDA rather than a native kernel competitor, the project should report that plainly and adjust its architecture.

---

## References and external documentation

Use official documentation as the primary source of truth, pinned to specific versions/commits during implementation:

- [Goose language documentation](https://aardappel.github.io/goose/)
- [Goose source repository](https://github.com/aardappel/goose)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUDA Toolkit CMake integration](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)
- [PyTorch benchmark utilities](https://pytorch.org/docs/stable/benchmark_utils.html)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [Google Benchmark](https://github.com/google/benchmark)
- [DLPack specification](https://github.com/dmlc/dlpack)

**Document governance:** update this file when scope or phase gates change. Keep implementation details, raw results and decision records in separate tracked files. Update claims only when supported by benchmark evidence.
