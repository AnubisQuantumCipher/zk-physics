# Benchmarks

## System

- **CPU**: Apple Silicon (M-series)
- **Build**: `--release` (optimized)
- **Curve**: bn256 (BN254)
- **Scheme**: KZG + SHPLONK

## Circuit

- **Constraints per step**: 44 regions (15 scaled_mul, 6 scaled_div, 7 add/sub, 5 emission, 2 acoustic, 3 Pythagorean removed, 6 core physics)
- **Public inputs**: 3 (R0, final_temperature, total_emission)
- **Features**: Rayleigh-Plesset dynamics, adiabatic temperature, acoustic driving (witness sin/cos), Stefan-Boltzmann emission

## Results

| Steps | k | Rows | Witness (ms) | Prove (ms) | Verify (ms) | Proof (bytes) |
|------:|--:|-----:|-------------:|-----------:|------------:|--------------:|
| 10 | 9 | 512 | <1 | 26 | 3 | 1,088 |
| 50 | 12 | 4,096 | <1 | 79 | 1 | 1,088 |
| 100 | 13 | 8,192 | 1 | 133 | 1 | 1,088 |
| 500 | 15 | 32,768 | 6 | 395 | 1 | 1,088 |
| 1,000 | 16 | 65,536 | 24 | 728 | 3 | 1,088 |
| 2,000 | 17 | 131,072 | 40 | 1,337 | 3 | 1,088 |
| 3,000 | 18 | 262,144 | 50 | 2,412 | 3 | 1,088 |

## Key Observations

1. **Constant proof size**: 1,088 bytes regardless of step count — a fundamental property of PLONK/KZG.

2. **Proving time scales linearly**: ~0.7 ms per step, dominated by MSM (multi-scalar multiplication) over the bn256 curve.

3. **Verification is fast**: 1-3 ms, essentially independent of circuit size (single pairing check).

4. **Witness generation is negligible**: BigInt arithmetic for the simulation trace is orders of magnitude cheaper than cryptographic operations.

5. **3,000-step proofs are practical**: Full collapse simulations complete in under 2.5 seconds with constant-size proofs.

## Reproduce

```bash
# Quick benchmark (10, 50, 100, 500, 1000 steps)
cargo run --release --bin benchmark

# Custom step counts
cargo run --release --bin benchmark -- --steps 10,100,500,1000,2000,3000

# Full proof + verify cycle
cargo run --release --bin prove -- --steps 100
cargo run --release --bin verify -- --proof output/proof.bin
```
