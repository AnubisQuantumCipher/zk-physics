# zk-physics

Zero-knowledge proof of a sonoluminescence physics simulation. Proves that a bubble collapse simulation was executed correctly -- without revealing the physical parameters.

Built with [halo2](https://github.com/privacy-scaling-explorations/halo2) (PLONK + KZG).

## What This Does

A sonoluminescence bubble collapses violently under pressure, reaching temperatures above 10,000 K and emitting light. This project encodes the governing Rayleigh-Plesset ODE into a halo2 circuit, generates a zero-knowledge proof of the entire simulation, and verifies it in constant time.

**The verifier learns only**: the equilibrium radius R0 and the final temperature. All other parameters (pressure, viscosity, surface tension, time step, intermediate states) remain hidden.

## Quick Start

```bash
# Prove a 100-step collapse simulation
cargo run --release --bin prove -- --steps 100

# Verify the proof
cargo run --release --bin verify -- --proof output/proof.bin

# Run benchmarks
cargo run --release --bin benchmark
```

## Architecture

| File | Lines | Role |
|------|------:|------|
| `src/chips/field_arith.rs` | 707 | Scaled fixed-point multiply/divide gates |
| `src/chips/physics_step.rs` | 532 | Rayleigh-Plesset ODE step as halo2 chip |
| `src/circuits/sonoluminescence.rs` | 372 | Full circuit: N chained physics steps |
| `src/witness.rs` | 398 | BigInt witness generation with field replay |
| `src/field_utils.rs` | 166 | Fixed-point scaling (S = 10^30) and conversions |
| `src/bin/prove.rs` | 156 | CLI: generate KZG proof |
| `src/bin/verify.rs` | 192 | CLI: verify proof from file |
| `src/bin/benchmark.rs` | 142 | Benchmark across step counts |
| `tests/privacy.rs` | 222 | Privacy tests: verify hidden parameters stay hidden |
| `reference/simulate.py` | -- | Python reference simulation |
| `reference/simulate_scaled.py` | -- | Python scaled-integer simulation |
| `reference/test_simulate.py` | -- | Cross-validation: Python vs Rust |
| `docs/PHYSICS.md` | -- | Physics background and equations |
| `docs/CIRCUIT.md` | -- | Circuit architecture and constraint analysis |
| `docs/BENCHMARKS.md` | -- | Full benchmark results |

## Benchmarks

Apple Silicon, `--release`, bn256 curve, KZG + SHPLONK:

| Steps | k | Rows | Prove (ms) | Verify (ms) | Proof Size |
|------:|--:|-----:|-----------:|------------:|----------:|
| 10 | 9 | 512 | 26 | 3 | 1,088 B |
| 100 | 13 | 8,192 | 133 | 1 | 1,088 B |
| 500 | 15 | 32,768 | 395 | 1 | 1,088 B |
| 1,000 | 16 | 65,536 | 728 | 3 | 1,088 B |
| 3,000 | 18 | 262,144 | 2,412 | 3 | 1,088 B |

Proof size is **constant at 1,088 bytes** regardless of simulation length. Verification is 1-3 ms.

## The Physics

The simulation solves the Rayleigh-Plesset equation for bubble wall dynamics:

```
R'' = [P_gas - P0 + 2sigma/R - 4mu*Rdot/R] / (rho*R) - 1.5*Rdot^2/R
```

Numerical integration uses Stormer-Verlet (symplectic, energy-preserving). Fixed-point scaling at S = 10^30 keeps all intermediate products well below the bn256 field modulus (~2^255).

At ~2,359 steps the bubble reaches minimum radius (~0.77 um) with peak temperature ~12,348 K -- above the 5,000 K sonoluminescence threshold.

## Article Series

This project is the basis for an 8-part technical series on [Jacobian](https://jacobian.ghost.io):

1. [Why Fixed-Point Arithmetic is the Hardest Part of ZK](https://jacobian.ghost.io/why-fixed-point-arithmetic-is-the-hardest-part-of-zk/)
2. [Building Custom halo2 Chips: A Field Arithmetic Walkthrough](https://jacobian.ghost.io/building-custom-halo2-chips/) *(Pro)*
3. [Proving Physics: Encoding Differential Equations in ZK](https://jacobian.ghost.io/proving-physics-encoding-differential-equations-in-zk/)
4. [1,088 Bytes to Prove a Star: Constant-Size Proofs with KZG](https://jacobian.ghost.io/1088-bytes-to-prove-a-star/) *(Pro)*
5. [The Witness Problem: When BigInt Precision Breaks Your Proof](https://jacobian.ghost.io/the-witness-problem/)
6. [Testing ZK Privacy: How to Verify Proofs Hide What They Should](https://jacobian.ghost.io/testing-zk-privacy/) *(Pro)*
7. [Python to Rust to Proof: Cross-Validating a ZK System](https://jacobian.ghost.io/python-to-rust-to-proof/)
8. [Sonoluminescence: The Physics of Light from Nothing](https://jacobian.ghost.io/sonoluminescence-the-physics-of-light-from-nothing/) *(Pro)*

New here? [Get the free halo2 circuit guide](https://jacobian.ghost.io/build-your-first-halo2-circuit/) -- build your first custom chip step by step.

## License

MIT
