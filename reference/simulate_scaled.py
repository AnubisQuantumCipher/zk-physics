#!/usr/bin/env python3
"""
Scaled Integer Rayleigh-Plesset Sonoluminescence Simulator
==========================================================
EXACT mirror of the halo2 ZK circuit arithmetic.

ALL computation uses Python integers with S = 10^30 scaling.
No floating point anywhere in the physics. This is the GROUND TRUTH
that the halo2 circuit must reproduce bit-for-bit.

Fixed-point convention:
  - Real value v is stored as V = round(v * S)
  - Multiply: scaled_mul(A, B) = A * B // S
  - Divide:   scaled_div(A, B) = A * S // B
  - Power:    chain of scaled_mul

Usage:
  python simulate_scaled.py --params collapse
  python simulate_scaled.py --params collapse --export trace_scaled.json
  python simulate_scaled.py --params collapse --compare  # compare vs float
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple


# ============================================================================
# Scaling Constants
# ============================================================================

S = 10**30  # Scaling factor

# Physical constants (scaled)
SIGMA_SB_S = round(5.670374419e-8 * S)    # Stefan-Boltzmann (W m^-2 K^-4)
RHO_WATER_S = round(998.0 * S)             # Water density (kg/m^3)
SURFACE_TENSION_S = round(0.0728 * S)      # Surface tension (N/m)
VISCOSITY_S = round(1.002e-3 * S)          # Dynamic viscosity (Pa*s)
PI_S = round(3.14159265358979323846 * S)   # pi

# Useful constants
TWO_S = 2 * S
THREE_HALVES_S = 3 * S // 2  # 1.5 in scaled
FOUR_S = 4 * S
SIGMA_4PI_S = round(5.670374419e-8 * 4 * 3.14159265358979323846 * S)  # sigma_SB * 4 * pi


# ============================================================================
# Scaled Arithmetic
# ============================================================================

def scaled_mul(a: int, b: int) -> int:
    """Scaled multiplication: (a * b) // S.

    In the ZK circuit, this is constrained as: c * S == a * b
    Python integers have unlimited precision, so this is exact.
    """
    return a * b // S


def scaled_div(a: int, b: int) -> int:
    """Scaled division: (a * S) // b.

    In the ZK circuit, this is constrained as: c * b == a * S
    """
    if b == 0:
        raise ZeroDivisionError("scaled_div: division by zero")
    return a * S // b


def scaled_pow(base: int, exp: int) -> int:
    """Scaled integer power: base^exp with descaling at each step."""
    if exp == 0:
        return S  # 1.0 in scaled
    result = base
    for _ in range(1, exp):
        result = scaled_mul(result, base)
    return result


def to_scaled(v: float) -> int:
    """Convert float to scaled integer."""
    return round(v * S)


def from_scaled(v: int) -> float:
    """Convert scaled integer back to float (for comparison)."""
    return v / S


# ============================================================================
# Simulation Parameters (Scaled)
# ============================================================================

@dataclass
class ScaledParams:
    """All parameters as scaled integers."""
    R0: int           # Equilibrium radius
    P0: int           # Ambient pressure
    Pa: int           # Acoustic amplitude
    gamma_num: int    # gamma numerator (e.g., 5)
    gamma_den: int    # gamma denominator (e.g., 3)
    sigma: int        # Surface tension
    mu: int           # Viscosity
    rho: int          # Liquid density
    T0: int           # Initial temperature
    n_steps: int      # Number of timesteps (not scaled)
    dt: int           # Timestep (scaled)
    R_start: int      # Starting radius

    @staticmethod
    def collapse() -> 'ScaledParams':
        """Collapse-only preset matching simulate.py's collapse()."""
        R0 = to_scaled(5.0e-6)
        return ScaledParams(
            R0=R0,
            P0=to_scaled(101325.0),
            Pa=0,
            gamma_num=5,
            gamma_den=3,
            sigma=SURFACE_TENSION_S,
            mu=VISCOSITY_S,
            rho=RHO_WATER_S,
            T0=to_scaled(293.15),
            n_steps=3000,
            dt=to_scaled(1.0e-9),
            R_start=5 * R0,
        )

    @staticmethod
    def acoustic() -> 'ScaledParams':
        """Acoustic driving preset: Pa=135kPa, freq=26.5kHz."""
        R0 = to_scaled(5.0e-6)
        return ScaledParams(
            R0=R0,
            P0=to_scaled(101325.0),
            Pa=to_scaled(135000.0),
            gamma_num=5,
            gamma_den=3,
            sigma=SURFACE_TENSION_S,
            mu=VISCOSITY_S,
            rho=RHO_WATER_S,
            T0=to_scaled(293.15),
            n_steps=3000,
            dt=to_scaled(1.0e-9),
            R_start=5 * R0,
        )

    @staticmethod
    def small_test() -> 'ScaledParams':
        """Small test (10 steps)."""
        R0 = to_scaled(5.0e-6)
        return ScaledParams(
            R0=R0,
            P0=to_scaled(101325.0),
            Pa=0,
            gamma_num=5,
            gamma_den=3,
            sigma=SURFACE_TENSION_S,
            mu=VISCOSITY_S,
            rho=RHO_WATER_S,
            T0=to_scaled(293.15),
            n_steps=10,
            dt=to_scaled(1.0e-9),
            R_start=5 * R0,
        )


# ============================================================================
# Scaled Step State
# ============================================================================

@dataclass
class ScaledStepState:
    """State at a single timestep, all values scaled."""
    step: int
    R: int
    R_prev: int
    Rdot: int
    P_gas: int
    T: int
    Rddot: int
    emission: int = 0


# ============================================================================
# Physics Functions (Integer Arithmetic Only)
# ============================================================================

def compute_gas_pressure_scaled(R: int, R0: int, P0: int,
                                 gamma_num: int, gamma_den: int,
                                 sigma: int) -> int:
    """P_gas = (P0 + 2*sigma/R0) * (R0/R)^(3*gamma)

    For gamma = 5/3: exponent = 3 * 5/3 = 5
    P_gas = P_initial * (R0/R)^5

    We compute (R0/R)^5 using integer arithmetic:
    ratio = R0 * S / R  (scaled R0/R)
    ratio^5 with 4 descalings
    """
    # P_initial = P0 + 2*sigma/R0
    two_sigma_over_R0 = scaled_div(2 * sigma, R0)
    P_initial = P0 + two_sigma_over_R0

    # Exponent: 3 * gamma_num / gamma_den
    # For gamma = 5/3: 3*5/3 = 5 (integer)
    exponent = 3 * gamma_num // gamma_den

    # ratio = R0/R (scaled)
    ratio = scaled_div(R0, R)

    # ratio^exponent
    ratio_pow = scaled_pow(ratio, exponent)

    return scaled_mul(P_initial, ratio_pow)


def compute_temperature_scaled(R: int, R0: int, T0: int,
                                gamma_num: int, gamma_den: int) -> int:
    """T = T0 * (R0/R)^(3*(gamma-1))

    For gamma = 5/3: exponent = 3*(5/3 - 1) = 3*2/3 = 2
    T = T0 * (R0/R)^2
    """
    # Exponent: 3 * (gamma_num - gamma_den) / gamma_den
    # For gamma = 5/3: 3*(5-3)/3 = 2
    exponent = 3 * (gamma_num - gamma_den) // gamma_den

    ratio = scaled_div(R0, R)
    ratio_pow = scaled_pow(ratio, exponent)

    return scaled_mul(T0, ratio_pow)


def compute_acoustic_pressure_scaled(Pa: int, sin_val: int) -> int:
    """Acoustic driving pressure: Pa * sin(2*pi*freq*t).

    The sin value is pre-computed and passed as a scaled integer.
    Returns Pa * sin_val (scaled).
    """
    return scaled_mul(Pa, sin_val)


def compute_emission_scaled(R: int, T: int) -> int:
    """Stefan-Boltzmann emission: E = sigma_SB * 4*pi * T^4 * R^2.

    Computed unconditionally every step (negligible when T is low).
    """
    T_sq = scaled_mul(T, T)
    T_4 = scaled_mul(T_sq, T_sq)
    R_sq = scaled_mul(R, R)
    T4_R2 = scaled_mul(T_4, R_sq)
    return scaled_mul(SIGMA_4PI_S, T4_R2)


def compute_acceleration_scaled(R: int, Rdot: int, P_gas: int,
                                 P0: int, sigma: int, mu: int,
                                 rho: int) -> int:
    """Rayleigh-Plesset acceleration (no acoustic driving for collapse-only).

    R'' = (P_gas - P0 + 2*sigma/R - 4*mu*Rdot/R) / (rho*R) - 3/2 * Rdot^2/R

    All terms computed in scaled integer arithmetic.
    """
    if abs(R) < 1:
        return 0  # Prevent division by zero

    # 2*sigma/R
    two_sigma_over_R = scaled_div(2 * sigma, R)

    # 4*mu*Rdot/R
    four_mu = 4 * mu  # Not scaled_mul because 4 is not scaled
    four_mu_rdot = scaled_mul(four_mu, Rdot)
    four_mu_rdot_over_R = scaled_div(four_mu_rdot, R)

    # delta_P = P_gas - P0 + 2*sigma/R - 4*mu*Rdot/R
    delta_P = P_gas - P0 + two_sigma_over_R - four_mu_rdot_over_R

    # rho * R
    rho_R = scaled_mul(rho, R)

    # delta_P / (rho * R)
    term1 = scaled_div(delta_P, rho_R)

    # 3/2 * Rdot^2 / R
    Rdot_sq = scaled_mul(Rdot, Rdot)
    Rdot_sq_over_R = scaled_div(Rdot_sq, R)
    # 3/2 as integer: multiply by 3, divide by 2
    term2 = 3 * Rdot_sq_over_R // 2

    return term1 - term2


# ============================================================================
# Simulation (Störmer-Verlet, Integer Only)
# ============================================================================

def simulate_scaled(params: ScaledParams) -> List[ScaledStepState]:
    """Run RP simulation using Störmer-Verlet with pure integer arithmetic.

    R_{n+1} = 2*R_n - R_{n-1} + a_n * dt^2
    Rdot_n  = (R_n - R_{n-1}) / dt
    """
    states: List[ScaledStepState] = []
    dt = params.dt
    dt2 = scaled_mul(dt, dt)  # dt^2 (scaled)

    R_min = params.R0 // 1000  # Minimum radius clamp

    R_curr = params.R_start
    R_prev = params.R_start  # Rdot_0 = 0

    for i in range(params.n_steps + 1):
        # Derive velocity
        Rdot = scaled_div(R_curr - R_prev, dt) if i > 0 else 0

        # Compute derived quantities
        P_gas = compute_gas_pressure_scaled(
            R_curr, params.R0, params.P0,
            params.gamma_num, params.gamma_den, params.sigma
        )
        T = compute_temperature_scaled(
            R_curr, params.R0, params.T0,
            params.gamma_num, params.gamma_den
        )

        # Acceleration
        Rddot = compute_acceleration_scaled(
            R_curr, Rdot, P_gas, params.P0,
            params.sigma, params.mu, params.rho
        )

        # Emission
        emission = compute_emission_scaled(R_curr, T)

        states.append(ScaledStepState(
            step=i, R=R_curr, R_prev=R_prev, Rdot=Rdot,
            P_gas=P_gas, T=T, Rddot=Rddot, emission=emission
        ))

        # Verlet update
        if i < params.n_steps:
            R_next = 2 * R_curr - R_prev + scaled_mul(Rddot, dt2)

            if R_next < R_min:
                R_next = R_min

            R_prev = R_curr
            R_curr = R_next

    return states


# ============================================================================
# Comparison with Float Version
# ============================================================================

def compare_with_float(scaled_states: List[ScaledStepState],
                        n_steps: int = None) -> dict:
    """Compare scaled integer simulation against float simulate.py."""
    from simulate import SimulationParams, simulate as simulate_float

    params_float = SimulationParams.collapse()
    float_states = simulate_float(params_float)

    n = min(len(scaled_states), len(float_states))
    if n_steps:
        n = min(n, n_steps + 1)

    max_R_err = 0.0
    max_T_err = 0.0
    max_Rdot_err = 0.0

    for i in range(n):
        sf = float_states[i]
        ss = scaled_states[i]

        R_float = sf.R
        R_scaled = from_scaled(ss.R)
        if R_float > 0:
            R_rel_err = abs(R_scaled - R_float) / R_float
            max_R_err = max(max_R_err, R_rel_err)

        T_float = sf.T
        T_scaled = from_scaled(ss.T)
        if T_float > 0:
            T_rel_err = abs(T_scaled - T_float) / T_float
            max_T_err = max(max_T_err, T_rel_err)

        Rdot_float = sf.Rdot
        Rdot_scaled = from_scaled(ss.Rdot)
        if abs(Rdot_float) > 1e-10:
            Rdot_rel_err = abs(Rdot_scaled - Rdot_float) / abs(Rdot_float)
            max_Rdot_err = max(max_Rdot_err, Rdot_rel_err)

    return {
        'max_R_relative_error': max_R_err,
        'max_T_relative_error': max_T_err,
        'max_Rdot_relative_error': max_Rdot_err,
        'steps_compared': n,
    }


# ============================================================================
# Output
# ============================================================================

def print_summary(params: ScaledParams, states: List[ScaledStepState]) -> None:
    """Print summary of scaled simulation."""
    min_R = min(s.R for s in states)
    min_idx = next(i for i, s in enumerate(states) if s.R == min_R)
    max_T = states[min_idx].T

    print("=" * 60)
    print("SCALED INTEGER SIMULATION (S = 10^30)")
    print("  Integration: Störmer-Verlet (symplectic)")
    print("=" * 60)
    print(f"\nParameters (raw scaled integers):")
    print(f"  R0     = {params.R0}")
    print(f"  R_start= {params.R_start}")
    print(f"  P0     = {params.P0}")
    print(f"  T0     = {params.T0}")
    print(f"  dt     = {params.dt}")
    print(f"  Steps  = {params.n_steps}")
    print(f"\nResults (converted to physical units):")
    print(f"  Min R  = {from_scaled(min_R)*1e6:.4f} µm (step {min_idx})")
    print(f"  Max T  = {from_scaled(max_T):.0f} K")
    print(f"  R0/R_min = {from_scaled(scaled_div(params.R0, min_R)):.1f}x")

    if from_scaled(max_T) > 5000:
        print(f"\n  ** SONOLUMINESCENCE DETECTED **")
    print("=" * 60)


def export_trace(params: ScaledParams, states: List[ScaledStepState],
                  filename: str) -> None:
    """Export trace as JSON with both raw scaled and float values."""
    trace = {
        'scale_factor': S,
        'params': {
            'R0': params.R0,
            'R0_float': from_scaled(params.R0),
            'P0': params.P0,
            'R_start': params.R_start,
            'T0': params.T0,
            'dt': params.dt,
            'n_steps': params.n_steps,
            'gamma_num': params.gamma_num,
            'gamma_den': params.gamma_den,
        },
        'states': [
            {
                'step': s.step,
                'R': s.R,
                'R_float': from_scaled(s.R),
                'Rdot': s.Rdot,
                'Rdot_float': from_scaled(s.Rdot),
                'P_gas': s.P_gas,
                'T': s.T,
                'T_float': from_scaled(s.T),
                'Rddot': s.Rddot,
            }
            for s in states
        ],
    }
    with open(filename, 'w') as f:
        json.dump(trace, f, indent=2, default=str)
    print(f"\nScaled trace exported to {filename} ({len(states)} steps)")


def main():
    parser = argparse.ArgumentParser(
        description='Scaled Integer RP Sonoluminescence Simulator')
    parser.add_argument('--params', choices=['collapse', 'small'],
                        default='collapse', help='Parameter preset')
    parser.add_argument('--export', type=str, help='Export trace to JSON')
    parser.add_argument('--compare', action='store_true',
                        help='Compare against float simulation')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    if args.params == 'small':
        params = ScaledParams.small_test()
    else:
        params = ScaledParams.collapse()

    states = simulate_scaled(params)

    if not args.quiet:
        print_summary(params, states)

    if args.compare:
        print("\nComparison with float simulation:")
        result = compare_with_float(states)
        print(f"  Steps compared:  {result['steps_compared']}")
        print(f"  Max R error:     {result['max_R_relative_error']:.2e}")
        print(f"  Max T error:     {result['max_T_relative_error']:.2e}")
        print(f"  Max Rdot error:  {result['max_Rdot_relative_error']:.2e}")
        if result['max_R_relative_error'] < 1e-6:
            print("  PASS: Integer arithmetic matches float within tolerance")
        else:
            print("  WARNING: Significant divergence detected")

    if args.export:
        export_trace(params, states, args.export)

    return 0


if __name__ == '__main__':
    sys.exit(main())
