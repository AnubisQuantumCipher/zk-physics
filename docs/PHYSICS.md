# Physics: Sonoluminescence Simulation

## Overview

This project proves the correct execution of a sonoluminescence simulation using zero-knowledge proofs. The simulation models the collapse of a gas bubble in liquid under pressure, following the Rayleigh-Plesset equation.

## Rayleigh-Plesset Equation

The governing equation for bubble wall dynamics:

```
R'' = [P_gas - P_0 + 2σ/R - 4μṘ/R] / (ρR) - (3/2)(Ṙ²/R)
```

Where:
- `R` — bubble radius (m)
- `R'`, `R''` — wall velocity and acceleration
- `P_gas` — internal gas pressure (Pa)
- `P_0` — ambient liquid pressure (101325 Pa)
- `σ` — surface tension (0.0728 N/m for water)
- `μ` — dynamic viscosity (1.002×10⁻³ Pa·s)
- `ρ` — liquid density (998 kg/m³)

## Gas Pressure Model

Polytropic compression with exponent γ = 5/3 (noble gas):

```
P_gas = P_initial × (R₀/R)^5
```

where `P_initial = P₀ + 2σ/R₀` (initial equilibrium pressure).

The exponent 5 arises from 3γ = 3 × 5/3 = 5 for the pressure-radius relationship in 3D compression, but in our simplified model we use the direct power law with exponent 5 for the pressure ratio.

## Temperature Model

Adiabatic compression heating:

```
T = T₀ × (R₀/R)²
```

Sonoluminescence is detected when peak temperature exceeds 5000 K.

## Numerical Integration: Störmer-Verlet

We use the Störmer-Verlet (velocity Verlet) method, a symplectic integrator that conserves energy over long timescales:

```
R_{n+1} = 2R_n - R_{n-1} + R''_n × dt²
```

Velocity is derived (not stored):
```
Ṙ_n = (R_n - R_{n-1}) / dt
```

Advantages over forward Euler:
- Energy-preserving (symplectic)
- Second-order accurate
- Same constraint cost in ZK (no extra state variables)
- Stable through the violent bubble collapse

## Collapse Preset

Parameters for the default simulation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| R₀ | 5 μm | Equilibrium radius |
| R_start | 25 μm (5×R₀) | Initial expanded radius |
| P₀ | 101325 Pa | Ambient pressure |
| σ | 0.0728 N/m | Surface tension |
| μ | 1.002×10⁻³ Pa·s | Viscosity |
| ρ | 998 kg/m³ | Water density |
| T₀ | 293.15 K | Ambient temperature |
| dt | 1 ns | Time step |

At ~2359 steps, the bubble reaches minimum radius (~0.77 μm) with peak temperature ~12,348 K — well above the 5000 K sonoluminescence threshold.

## Fixed-Point Scaling

All physical values are represented as integers scaled by S = 10³⁰:

```
value_scaled = value_float × 10³⁰
```

This scaling was chosen because:
- Sonoluminescence spans ~200× pressure variation and ~170× temperature variation
- Products of two scaled values reach ~10⁶⁰, still 16 orders of magnitude below the bn256 field modulus (~2²⁵⁵)
- Division is implemented as field multiplicative inverse in the circuit

## Witness Generation

The witness is generated using arbitrary-precision integers (`num-bigint` in Rust, native integers in Python) to avoid u128 overflow on intermediate products. The BigInt trace is converted to u128 values for halo2 circuit loading.

Key subtlety: field arithmetic (modular inverse) differs from integer arithmetic (truncating division). Public inputs are computed by replaying the circuit's exact field operations using Fr, not by integer arithmetic.
