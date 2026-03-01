#!/usr/bin/env python3
"""
Rayleigh-Plesset Sonoluminescence Simulator
============================================
Reference implementation for the zk-physics ZK circuit.
Every value computed here is the ground truth that the circuit must match.

Physics Model:
  - Rayleigh-Plesset equation for bubble dynamics
  - Polytropic gas compression (adiabatic)
  - Acoustic driving pressure (sinusoidal)
  - Stefan-Boltzmann emission at collapse
  - Störmer-Verlet integration (symplectic, energy-preserving)

Integration Method:
  Störmer-Verlet:
    R_{n+1} = 2*R_n - R_{n-1} + a_n * dt^2
    Rdot_n  = (R_n - R_{n-1}) / dt

  This is intentionally used over forward Euler because:
  1. Symplectic: preserves energy over long timescales
  2. Same constraint cost in the ZK circuit
  3. Forward Euler blows up at bubble collapse

Usage:
  python simulate.py --params default
  python simulate.py --params collapse
  python simulate.py --R0 5e-6 --P0 101325 --Pa 130000 --freq 26500 --gamma 1.667 --steps 1000
  python simulate.py --params collapse --export trace.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

SIGMA_SB = 5.670374419e-8    # Stefan-Boltzmann constant (W m^-2 K^-4)
RHO_WATER = 998.0             # Water density (kg/m^3)
SURFACE_TENSION = 0.0728      # Surface tension of water (N/m)
VISCOSITY = 1.002e-3           # Dynamic viscosity of water (Pa*s)


@dataclass
class SimulationParams:
    """All parameters needed for the Rayleigh-Plesset simulation."""
    R0: float          # Equilibrium bubble radius (m)
    P0: float          # Ambient pressure (Pa)
    Pa: float          # Acoustic driving amplitude (Pa)
    freq: float        # Acoustic frequency (Hz)
    gamma: float       # Polytropic exponent (5/3 for monatomic)
    sigma: float       # Surface tension (N/m)
    mu: float          # Dynamic viscosity (Pa*s)
    rho: float         # Liquid density (kg/m^3)
    T0: float          # Initial gas temperature (K)
    n_steps: int       # Number of timesteps
    dt: float          # Timestep size (s)
    R_start: float = 0.0  # Starting radius (0 means use R0)

    def effective_R_start(self) -> float:
        """Return the actual starting radius."""
        return self.R_start if self.R_start > 0 else self.R0

    @staticmethod
    def default() -> 'SimulationParams':
        """Default parameters for a typical sonoluminescence experiment.

        Physically realistic values for single-bubble sonoluminescence (SBSL)
        in water with argon gas:
        - 5 micron equilibrium radius
        - ~1 atm ambient pressure
        - ~1.3 atm acoustic driving (above Blake threshold)
        - 26.5 kHz ultrasonic frequency
        - gamma = 5/3 for noble gas (argon) -- monatomic
        """
        freq = 26500.0
        n_steps = 5000
        dt = 1.0 / (freq * n_steps)  # One acoustic cycle

        return SimulationParams(
            R0=5.0e-6,           # 5 microns
            P0=101325.0,         # 1 atm
            Pa=135000.0,         # 1.33 atm (above Blake threshold)
            freq=freq,           # 26.5 kHz
            gamma=5.0/3.0,       # Monatomic gas (argon)
            sigma=SURFACE_TENSION,
            mu=VISCOSITY,
            rho=RHO_WATER,
            T0=293.15,           # 20°C
            n_steps=n_steps,
            dt=dt,
        )

    @staticmethod
    def collapse() -> 'SimulationParams':
        """Collapse-only preset: bubble starts at 10*R0, no acoustic driving.

        The bubble is inflated to 5x its equilibrium radius and released.
        Ambient pressure drives the collapse. No acoustic forcing.
        This produces sonoluminescence (T > 5000K) via pure implosion.

        This is the v1 MVP scenario for the ZK circuit: fewer parameters,
        no sin/cos witnesses needed, cleaner proof.

        Tuned parameters:
        - 5x R0 start: enough energy for sonoluminescence without
          extreme compression that causes numerical instability
        - dt = 1 ns: resolves collapse dynamics and bounce-back
        - 3000 steps: captures full collapse + rebound
        - Peak T ~ 12,000 K at step ~2359, min R ~ 0.77 µm
        """
        R0 = 5.0e-6
        return SimulationParams(
            R0=R0,
            P0=101325.0,
            Pa=0.0,              # No acoustic driving
            freq=0.0,            # No frequency needed
            gamma=5.0/3.0,
            sigma=SURFACE_TENSION,
            mu=VISCOSITY,
            rho=RHO_WATER,
            T0=293.15,
            n_steps=3000,
            dt=1.0e-9,           # 1 ns steps — resolves collapse bounce
            R_start=5.0 * R0,    # Start at 5x equilibrium
        )

    @staticmethod
    def small_test() -> 'SimulationParams':
        """Small test case for unit testing (10 steps)."""
        return SimulationParams(
            R0=5.0e-6,
            P0=101325.0,
            Pa=0.0,
            freq=0.0,
            gamma=5.0/3.0,
            sigma=SURFACE_TENSION,
            mu=VISCOSITY,
            rho=RHO_WATER,
            T0=293.15,
            n_steps=10,
            dt=1.0e-9,
            R_start=10.0 * 5.0e-6,
        )


@dataclass
class StepState:
    """State at a single timestep."""
    step: int          # Step index
    t: float           # Time (s)
    R: float           # Bubble radius (m)
    R_prev: float      # Previous radius for Verlet (m)
    Rdot: float        # Radial velocity (m/s)
    P_gas: float       # Internal gas pressure (Pa)
    P_acoustic: float  # Acoustic driving pressure (Pa)
    T: float           # Gas temperature (K)
    E_emission: float  # Instantaneous emission power (W)
    Rddot: float       # Radial acceleration (m/s^2)


def compute_gas_pressure(R: float, R0: float, P0: float, gamma: float,
                          sigma: float) -> float:
    """Compute internal gas pressure via polytropic law.

    P_gas = (P0 + 2*sigma/R0) * (R0/R)^(3*gamma)

    The initial pressure includes the Laplace pressure correction.
    """
    P_initial = P0 + 2.0 * sigma / R0
    ratio = R0 / R
    return P_initial * (ratio ** (3.0 * gamma))


def compute_temperature(R: float, R0: float, T0: float, gamma: float) -> float:
    """Compute gas temperature via adiabatic compression.

    T = T0 * (R0/R)^(3*(gamma-1))

    For gamma = 5/3:  exponent = 3*(5/3 - 1) = 2
    So T = T0 * (R0/R)^2
    """
    ratio = R0 / R
    return T0 * (ratio ** (3.0 * (gamma - 1.0)))


def compute_acoustic_pressure(t: float, Pa: float, freq: float) -> float:
    """Compute acoustic driving pressure.

    P_acoustic = -Pa * sin(2*pi*freq*t)

    Negative sign: rarefaction (negative P) causes expansion,
    followed by compression driving collapse.
    """
    if Pa == 0.0 or freq == 0.0:
        return 0.0
    return -Pa * np.sin(2.0 * np.pi * freq * t)


def compute_emission(R: float, T: float, Rdot: float,
                     v_threshold: float = 100.0) -> float:
    """Compute Stefan-Boltzmann emission power.

    E = sigma_SB * T^4 * 4*pi*R^2

    Only emits when bubble is collapsing faster than v_threshold.
    Models the flash occurring only during violent implosion.
    """
    if Rdot < -v_threshold:
        surface_area = 4.0 * np.pi * R * R
        return SIGMA_SB * (T ** 4) * surface_area
    return 0.0


def compute_acceleration(R: float, Rdot: float, P_gas: float,
                          P0: float, P_acoustic: float,
                          sigma: float, mu: float, rho: float) -> float:
    """Compute bubble wall acceleration from Rayleigh-Plesset equation.

    rho * (R * R'' + 3/2 * R'^2) = P_gas - P0 - P_acoustic - 4*mu*R'/R + 2*sigma/R

    Solving for R'':
    R'' = (1/(rho*R)) * (P_gas - P0 + P_acoustic + 2*sigma/R - 4*mu*R'/R) - 3/2 * R'^2 / R
    """
    if abs(R) < 1e-20:
        return 0.0  # Prevent division by zero

    # Pressure difference driving the bubble
    delta_P = P_gas - P0 + P_acoustic + 2.0 * sigma / R - 4.0 * mu * Rdot / R

    # Rayleigh-Plesset: R'' = delta_P / (rho * R) - 3/2 * R'^2 / R
    Rddot = delta_P / (rho * R) - 1.5 * Rdot * Rdot / R

    return Rddot


def simulate(params: SimulationParams) -> List[StepState]:
    """Run the full Rayleigh-Plesset simulation using Störmer-Verlet.

    Störmer-Verlet integration:
      R_{n+1} = 2*R_n - R_{n-1} + a_n * dt^2
      Rdot_n  = (R_n - R_{n-1}) / dt

    This is symplectic (energy-preserving) and handles the stiff
    collapse dynamics much better than forward Euler.

    Returns a list of StepState for each timestep.
    """
    states: List[StepState] = []
    dt = params.dt
    dt2 = dt * dt

    R_start = params.effective_R_start()
    R_min = params.R0 * 1e-3  # Minimum radius clamp (prevent singularity)

    # Initialize: R_curr = R_start, Rdot_0 = 0
    # For Verlet, we need R_prev. With Rdot_0 = 0:
    #   R_prev = R_curr - Rdot_0 * dt = R_curr
    R_curr = R_start
    R_prev = R_start  # Rdot_0 = 0

    for i in range(params.n_steps + 1):
        t = i * dt

        # Derive velocity from Verlet history
        Rdot = (R_curr - R_prev) / dt if i > 0 else 0.0

        # Compute derived quantities at current state
        P_gas = compute_gas_pressure(R_curr, params.R0, params.P0,
                                      params.gamma, params.sigma)
        P_acoustic = compute_acoustic_pressure(t, params.Pa, params.freq)
        T = compute_temperature(R_curr, params.R0, params.T0, params.gamma)
        E = compute_emission(R_curr, T, Rdot)

        # Compute acceleration
        Rddot = compute_acceleration(R_curr, Rdot, P_gas, params.P0,
                                      P_acoustic, params.sigma, params.mu,
                                      params.rho)

        # Record state
        states.append(StepState(
            step=i, t=t, R=R_curr, R_prev=R_prev, Rdot=Rdot,
            P_gas=P_gas, P_acoustic=P_acoustic,
            T=T, E_emission=E, Rddot=Rddot
        ))

        # Störmer-Verlet update (skip on last step)
        if i < params.n_steps:
            R_next = 2.0 * R_curr - R_prev + Rddot * dt2

            # Clamp: prevent negative or near-zero radius
            if R_next < R_min:
                R_next = R_min

            R_prev = R_curr
            R_curr = R_next

    return states


def compute_total_emission(states: List[StepState], dt: float) -> float:
    """Compute total radiated energy by integrating emission power over time."""
    total = 0.0
    for s in states:
        total += s.E_emission * dt
    return total


def find_minimum_radius(states: List[StepState]) -> Tuple[int, float, float]:
    """Find the timestep with minimum radius (maximum compression)."""
    min_idx = 0
    min_R = states[0].R
    for i, s in enumerate(states):
        if s.R < min_R:
            min_R = s.R
            min_idx = i
    return min_idx, min_R, states[min_idx].T


def print_summary(params: SimulationParams, states: List[StepState]) -> None:
    """Print a summary of the simulation results."""
    total_E = compute_total_emission(states, params.dt)
    min_idx, min_R, max_T = find_minimum_radius(states)
    R_start = params.effective_R_start()

    print("=" * 60)
    print("RAYLEIGH-PLESSET SONOLUMINESCENCE SIMULATION")
    print("  Integration: Störmer-Verlet (symplectic)")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  R0          = {params.R0*1e6:.2f} µm")
    print(f"  R_start     = {R_start*1e6:.2f} µm ({R_start/params.R0:.1f}x R0)")
    print(f"  P0          = {params.P0:.0f} Pa ({params.P0/101325:.3f} atm)")
    if params.Pa > 0:
        print(f"  Pa          = {params.Pa:.0f} Pa ({params.Pa/101325:.3f} atm)")
        print(f"  Frequency   = {params.freq:.0f} Hz")
    else:
        print(f"  Pa          = 0 (collapse-only, no acoustic driving)")
    print(f"  Gamma       = {params.gamma:.4f}")
    print(f"  Sigma       = {params.sigma:.4f} N/m")
    print(f"  Mu          = {params.mu:.6f} Pa·s")
    print(f"  T0          = {params.T0:.2f} K")
    print(f"  Steps       = {params.n_steps}")
    print(f"  dt          = {params.dt:.2e} s")
    print(f"  Total time  = {params.n_steps * params.dt:.2e} s")
    if params.freq > 0:
        print(f"  Cycles      = {params.n_steps * params.dt * params.freq:.2f}")

    print(f"\nResults:")
    print(f"  Min radius  = {min_R*1e6:.4f} µm (at step {min_idx})")
    print(f"  Compression = {R_start/min_R:.1f}x (from start)")
    print(f"  R0/R_min    = {params.R0/min_R:.1f}x (from equilibrium)")
    print(f"  Max temp    = {max_T:.0f} K")
    print(f"  Total emission = {total_E:.4e} J")

    # Check if sonoluminescence occurred
    if max_T > 5000:
        print(f"\n  ** SONOLUMINESCENCE DETECTED **")
        print(f"  Peak temperature {max_T:.0f} K exceeds 5000 K threshold")
    else:
        print(f"\n  No sonoluminescence (peak temp {max_T:.0f} K < 5000 K)")

    print("=" * 60)


def export_trace(params: SimulationParams, states: List[StepState],
                  filename: str) -> None:
    """Export the full simulation trace as JSON for circuit comparison.

    Every intermediate value is included so the halo2 circuit
    can use this as witness data and compare against it.
    """
    trace = {
        'params': {
            **asdict(params),
            'R_start_effective': params.effective_R_start(),
        },
        'states': [asdict(s) for s in states],
        'summary': {
            'total_emission': compute_total_emission(states, params.dt),
            'min_radius': {
                'step': find_minimum_radius(states)[0],
                'R': find_minimum_radius(states)[1],
                'T': find_minimum_radius(states)[2],
            },
            'sonoluminescence': find_minimum_radius(states)[2] > 5000,
        }
    }
    with open(filename, 'w') as f:
        json.dump(trace, f, indent=2, default=str)
    print(f"\nTrace exported to {filename} ({len(states)} steps)")


def main():
    parser = argparse.ArgumentParser(
        description='Rayleigh-Plesset Sonoluminescence Simulator (Störmer-Verlet)')
    parser.add_argument('--params', choices=['default', 'collapse', 'small'],
                        help='Use preset parameters')
    parser.add_argument('--R0', type=float, help='Equilibrium radius (m)')
    parser.add_argument('--P0', type=float, help='Ambient pressure (Pa)')
    parser.add_argument('--Pa', type=float, help='Acoustic amplitude (Pa)')
    parser.add_argument('--freq', type=float, help='Acoustic frequency (Hz)')
    parser.add_argument('--gamma', type=float, help='Polytropic exponent')
    parser.add_argument('--steps', type=int, help='Number of timesteps')
    parser.add_argument('--dt', type=float, help='Timestep size (s)')
    parser.add_argument('--T0', type=float, help='Initial temperature (K)')
    parser.add_argument('--R-start', type=float, dest='R_start',
                        help='Starting radius (m), default=R0')
    parser.add_argument('--export', type=str,
                        help='Export trace to JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress summary output')

    args = parser.parse_args()

    if args.params == 'small':
        params = SimulationParams.small_test()
    elif args.params == 'collapse':
        params = SimulationParams.collapse()
    else:
        params = SimulationParams.default()

    # Override individual parameters if provided
    if args.R0 is not None: params.R0 = args.R0
    if args.P0 is not None: params.P0 = args.P0
    if args.Pa is not None: params.Pa = args.Pa
    if args.freq is not None: params.freq = args.freq
    if args.gamma is not None: params.gamma = args.gamma
    if args.steps is not None: params.n_steps = args.steps
    if args.dt is not None: params.dt = args.dt
    if args.T0 is not None: params.T0 = args.T0
    if args.R_start is not None: params.R_start = args.R_start

    # Run simulation
    states = simulate(params)

    if not args.quiet:
        print_summary(params, states)

    if args.export:
        export_trace(params, states, args.export)

    return 0


if __name__ == '__main__':
    sys.exit(main())
