# Circuit Architecture

## Framework

- **halo2** (PSE fork v0.3.0) — PLONK-based ZK proof system
- **Curve**: bn256 (BN254)
- **Commitment**: KZG with SHPLONK multiopen
- **Transcript**: Blake2b
- **No trusted setup** (universal SRS via `ParamsKZG::new(k)`)

## Circuit Structure

```
SonoluminescenceCircuit
├── FieldArithChip (scaled multiply, divide, power)
│   ├── Gate: s_mul  →  a × b = c × S
│   └── Gate: s_div  →  c × b = a × S
└── PhysicsStep (chained N times)
    ├── Velocity: Ṙ = (R_curr - R_prev) × S / dt
    ├── Gas pressure: P_gas = P_initial × (R₀/R)^5
    ├── Temperature: T = T₀ × (R₀/R)²
    ├── RP acceleration: R'' = δP/(ρR) - 1.5Ṙ²/R
    └── Verlet update: R_next = 2R_curr - R_prev + R''×dt²
```

## Public Inputs

Only two values are exposed:
1. **R₀** (row 0) — equilibrium bubble radius
2. **Final temperature** (row 1) — temperature at the last simulation step

All other parameters (pressure, viscosity, surface tension, etc.) remain private.

## Constraint Analysis

### Per Step (~15 constrained operations)
| Operation | Constraints | Description |
|-----------|------------|-------------|
| Velocity | 2 | sub + scaled_div |
| Ratio R₀/R | 1 | scaled_div |
| Ratio^5 | 4 | chain of scaled_mul |
| P_gas | 1 | scaled_mul |
| Temperature | 2 | ratio² + scaled_mul |
| Surface tension | 2 | scaled_mul + scaled_div |
| Viscosity | 3 | scaled_mul chain + scaled_div |
| δP | 1 | sub chain |
| ρR | 1 | scaled_mul |
| term1 (δP/ρR) | 1 | scaled_div |
| term2 (1.5Ṙ²/R) | 3 | scaled_mul + scaled_div + scaled_mul |
| R'' | 1 | sub |
| Verlet | 3 | scaled_mul + add + sub |

**Total: ~30 regions per step** (including overhead for loading intermediates)

### Circuit Sizing

```
k = min k such that 2^k ≥ n_steps × 30 + 30
```

| Steps | k | Rows (2^k) |
|-------|---|------------|
| 10 | 9 | 512 |
| 50 | 11 | 2,048 |
| 100 | 12 | 4,096 |
| 200 | 13 | 8,192 |
| 500 | 14 | 16,384 |

## Chips

### FieldArithChip

Core arithmetic with two custom gates:

**Scaled Multiply** (`s_mul` selector):
```
a × b - c × S = 0  (mod p)
```
Witness: `c = a × b × S⁻¹ mod p`

**Scaled Divide** (`s_div` selector):
```
c × b - a × S = 0  (mod p)
```
Witness: `c = a × S × b⁻¹ mod p`

Methods: `scaled_mul`, `scaled_div`, `scaled_pow`, `load_private`, `load_constant`, `add`, `sub`

### PhysicsStep

Chains FieldArithChip operations to implement one Rayleigh-Plesset timestep. Takes `PhysicsParams` (all physical constants as assigned cells) and `StepState` (R_curr, R_prev), returns `StepResult` (R_next, temperature, etc.).

## Field Arithmetic vs Integer Arithmetic

A critical subtlety: the circuit performs field arithmetic (modular inverse for division), which differs from integer truncating division:

```
field:   a × b × S⁻¹ mod p   (exact in the field)
integer: floor(a × b / S)     (truncates)
```

These give different results when `a × b` is not exactly divisible by S. Public inputs must be computed using the same field arithmetic as the circuit (via `compute_public_inputs<Fr>()`), not by replaying integer arithmetic.

## Proof Properties

- **Proof size**: 1056 bytes (constant, independent of step count)
- **Verification time**: ~1-6 ms
- **Proving time**: ~25 ms (10 steps) to ~264 ms (500 steps)
- **Zero-knowledge**: private parameters hidden behind randomized blinding factors
- **Soundness**: PLONK with KZG commitments on bn256
