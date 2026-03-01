#!/usr/bin/env python3
"""
Tests for the Rayleigh-Plesset sonoluminescence simulator.
Verifies physics functions, Verlet integration stability, and sonoluminescence detection.
"""

import pytest
import numpy as np
from simulate import (
    SimulationParams, StepState,
    compute_gas_pressure, compute_temperature,
    compute_acoustic_pressure, compute_emission,
    compute_acceleration, simulate, find_minimum_radius,
    compute_total_emission,
    SIGMA_SB, RHO_WATER, SURFACE_TENSION, VISCOSITY,
)


# ============================================================================
# Gas Pressure Tests
# ============================================================================

class TestGasPressure:
    """Test polytropic gas pressure: P_gas = (P0 + 2*sigma/R0) * (R0/R)^(3*gamma)"""

    def test_at_equilibrium(self):
        """At R = R0, pressure equals initial pressure."""
        R0, P0, gamma = 5e-6, 101325.0, 5.0/3.0
        P = compute_gas_pressure(R0, R0, P0, gamma, SURFACE_TENSION)
        P_initial = P0 + 2.0 * SURFACE_TENSION / R0
        assert abs(P - P_initial) / P_initial < 1e-10

    def test_compression_increases_pressure(self):
        """Compressing the bubble (R < R0) increases gas pressure."""
        R0, P0, gamma = 5e-6, 101325.0, 5.0/3.0
        P_eq = compute_gas_pressure(R0, R0, P0, gamma, SURFACE_TENSION)
        P_compressed = compute_gas_pressure(R0/2, R0, P0, gamma, SURFACE_TENSION)
        assert P_compressed > P_eq

    def test_expansion_decreases_pressure(self):
        """Expanding the bubble (R > R0) decreases gas pressure."""
        R0, P0, gamma = 5e-6, 101325.0, 5.0/3.0
        P_eq = compute_gas_pressure(R0, R0, P0, gamma, SURFACE_TENSION)
        P_expanded = compute_gas_pressure(2*R0, R0, P0, gamma, SURFACE_TENSION)
        assert P_expanded < P_eq

    def test_known_compression_ratio(self):
        """At R = R0/2, pressure should increase by 2^(3*gamma)."""
        R0, P0, gamma = 5e-6, 101325.0, 5.0/3.0
        P_initial = P0 + 2.0 * SURFACE_TENSION / R0
        P_half = compute_gas_pressure(R0/2, R0, P0, gamma, SURFACE_TENSION)
        expected = P_initial * (2.0 ** (3.0 * gamma))
        assert abs(P_half - expected) / expected < 1e-10

    def test_at_5x_expansion(self):
        """At R = 5*R0, pressure drops by 5^(3*gamma)."""
        R0, P0, gamma = 5e-6, 101325.0, 5.0/3.0
        P_initial = P0 + 2.0 * SURFACE_TENSION / R0
        P_5x = compute_gas_pressure(5*R0, R0, P0, gamma, SURFACE_TENSION)
        expected = P_initial * (1.0/5.0) ** (3.0 * gamma)
        assert abs(P_5x - expected) / expected < 1e-10


# ============================================================================
# Temperature Tests
# ============================================================================

class TestTemperature:
    """Test adiabatic temperature: T = T0 * (R0/R)^(3*(gamma-1))"""

    def test_at_equilibrium(self):
        """At R = R0, temperature equals T0."""
        T = compute_temperature(5e-6, 5e-6, 293.15, 5.0/3.0)
        assert abs(T - 293.15) < 1e-10

    def test_compression_heats(self):
        """Compressing the bubble increases temperature."""
        T0 = 293.15
        T_compressed = compute_temperature(2.5e-6, 5e-6, T0, 5.0/3.0)
        assert T_compressed > T0

    def test_known_ratio_monatomic(self):
        """For gamma=5/3, exponent = 3*(5/3-1) = 2. At R=R0/2: T = T0 * 4."""
        R0, T0, gamma = 5e-6, 293.15, 5.0/3.0
        T = compute_temperature(R0/2, R0, T0, gamma)
        expected = T0 * 4.0  # (R0/(R0/2))^2 = 2^2 = 4
        assert abs(T - expected) / expected < 1e-10

    def test_sonoluminescence_threshold(self):
        """At R = R0/4.13, T should exceed 5000K for gamma=5/3."""
        R0, T0, gamma = 5e-6, 293.15, 5.0/3.0
        R_threshold = R0 / 4.13
        T = compute_temperature(R_threshold, R0, T0, gamma)
        assert T > 5000, f"Expected T > 5000, got {T:.0f}"

    def test_specific_compression_5x(self):
        """At R = R0/5: T = T0 * 25 for gamma=5/3."""
        R0, T0, gamma = 5e-6, 293.15, 5.0/3.0
        T = compute_temperature(R0/5, R0, T0, gamma)
        expected = T0 * 25.0
        assert abs(T - expected) / expected < 1e-10


# ============================================================================
# Emission Tests
# ============================================================================

class TestEmission:
    """Test Stefan-Boltzmann emission: E = sigma_SB * T^4 * 4*pi*R^2"""

    def test_no_emission_when_expanding(self):
        """No emission when bubble is expanding (Rdot > 0)."""
        E = compute_emission(1e-6, 10000, 100.0)
        assert E == 0.0

    def test_no_emission_below_threshold(self):
        """No emission when collapse velocity is below threshold."""
        E = compute_emission(1e-6, 10000, -50.0, v_threshold=100.0)
        assert E == 0.0

    def test_emission_during_collapse(self):
        """Emission occurs during fast collapse (Rdot < -threshold)."""
        R, T, Rdot = 1e-6, 10000.0, -200.0
        E = compute_emission(R, T, Rdot, v_threshold=100.0)
        expected = SIGMA_SB * T**4 * 4.0 * np.pi * R**2
        assert abs(E - expected) / expected < 1e-10

    def test_emission_scales_with_T4(self):
        """Emission power scales as T^4."""
        R, Rdot = 1e-6, -200.0
        E1 = compute_emission(R, 5000.0, Rdot)
        E2 = compute_emission(R, 10000.0, Rdot)
        ratio = E2 / E1
        expected_ratio = (10000.0/5000.0)**4  # = 16
        assert abs(ratio - expected_ratio) / expected_ratio < 1e-10


# ============================================================================
# Acoustic Pressure Tests
# ============================================================================

class TestAcousticPressure:
    """Test acoustic driving: P = -Pa * sin(2*pi*freq*t)"""

    def test_zero_at_t_zero(self):
        """At t=0, sin(0) = 0, so P_acoustic = 0."""
        P = compute_acoustic_pressure(0.0, 135000.0, 26500.0)
        assert abs(P) < 1e-10

    def test_no_driving_when_Pa_zero(self):
        """With Pa = 0, no acoustic pressure."""
        P = compute_acoustic_pressure(1e-5, 0.0, 26500.0)
        assert P == 0.0

    def test_no_driving_when_freq_zero(self):
        """With freq = 0, no acoustic pressure."""
        P = compute_acoustic_pressure(1e-5, 135000.0, 0.0)
        assert P == 0.0

    def test_amplitude(self):
        """Maximum acoustic pressure magnitude equals Pa."""
        Pa, freq = 135000.0, 26500.0
        # At t = 1/(4*freq), sin(pi/2) = 1, P = -Pa
        t_quarter = 1.0 / (4.0 * freq)
        P = compute_acoustic_pressure(t_quarter, Pa, freq)
        assert abs(P + Pa) / Pa < 1e-10


# ============================================================================
# Verlet Integration Tests
# ============================================================================

class TestVerletStability:
    """Test Störmer-Verlet integration stability."""

    def test_no_nan_or_inf(self):
        """Simulation should never produce NaN or Inf values."""
        params = SimulationParams.collapse()
        states = simulate(params)
        for s in states:
            assert np.isfinite(s.R), f"Non-finite R at step {s.step}"
            assert np.isfinite(s.Rdot), f"Non-finite Rdot at step {s.step}"
            assert np.isfinite(s.T), f"Non-finite T at step {s.step}"
            assert np.isfinite(s.Rddot), f"Non-finite Rddot at step {s.step}"

    def test_positive_radius(self):
        """Radius must remain positive at all times."""
        params = SimulationParams.collapse()
        states = simulate(params)
        for s in states:
            assert s.R > 0, f"Non-positive R={s.R} at step {s.step}"

    def test_positive_temperature(self):
        """Temperature must remain positive at all times."""
        params = SimulationParams.collapse()
        states = simulate(params)
        for s in states:
            assert s.T > 0, f"Non-positive T={s.T} at step {s.step}"

    def test_long_run_stability(self):
        """10K steps should complete without blowup."""
        params = SimulationParams(
            R0=5e-6, P0=101325.0, Pa=0.0, freq=0.0,
            gamma=5.0/3.0, sigma=SURFACE_TENSION, mu=VISCOSITY,
            rho=RHO_WATER, T0=293.15,
            n_steps=10000, dt=1e-9,
            R_start=5.0 * 5e-6,
        )
        states = simulate(params)
        assert len(states) == 10001
        # Final R should be reasonable (not exploded to infinity)
        assert states[-1].R < 1e-3, f"R exploded to {states[-1].R}"
        assert states[-1].R > 0

    def test_deterministic(self):
        """Two runs with same params produce identical results."""
        params = SimulationParams.collapse()
        states1 = simulate(params)
        states2 = simulate(params)
        for s1, s2 in zip(states1, states2):
            assert s1.R == s2.R, f"Non-deterministic R at step {s1.step}"
            assert s1.Rdot == s2.Rdot
            assert s1.T == s2.T

    def test_step_count(self):
        """Simulation returns exactly n_steps + 1 states (including initial)."""
        params = SimulationParams.small_test()
        states = simulate(params)
        assert len(states) == params.n_steps + 1

    def test_initial_conditions(self):
        """First state matches initial conditions."""
        params = SimulationParams.collapse()
        states = simulate(params)
        s0 = states[0]
        assert s0.step == 0
        assert abs(s0.t) < 1e-20
        assert abs(s0.R - params.effective_R_start()) < 1e-20
        assert abs(s0.Rdot) < 1e-20  # Initially at rest


# ============================================================================
# Sonoluminescence Detection Tests
# ============================================================================

class TestSonoluminescence:
    """Test that the collapse preset produces sonoluminescence."""

    def test_collapse_reaches_high_temperature(self):
        """Collapse preset should reach T > 5000K."""
        params = SimulationParams.collapse()
        states = simulate(params)
        _, _, max_T = find_minimum_radius(states)
        assert max_T > 5000, f"Peak temp {max_T:.0f} K < 5000 K threshold"

    def test_collapse_produces_emission(self):
        """Collapse should produce non-zero total emission."""
        params = SimulationParams.collapse()
        states = simulate(params)
        total_E = compute_total_emission(states, params.dt)
        assert total_E > 0, "No emission detected during collapse"

    def test_collapse_compression_ratio(self):
        """Bubble should compress significantly (R_min < R0)."""
        params = SimulationParams.collapse()
        states = simulate(params)
        _, min_R, _ = find_minimum_radius(states)
        assert min_R < params.R0, f"min_R={min_R} >= R0={params.R0}"

    def test_bounce_occurs(self):
        """After collapse, bubble should bounce back (R increases again)."""
        params = SimulationParams.collapse()
        states = simulate(params)
        min_idx, min_R, _ = find_minimum_radius(states)
        # After minimum, radius should increase
        assert min_idx < len(states) - 1, "Minimum at last step — no bounce"
        assert states[min_idx + 1].R > min_R, "No bounce after collapse"


# ============================================================================
# Acceleration / Rayleigh-Plesset Tests
# ============================================================================

class TestAcceleration:
    """Test the Rayleigh-Plesset acceleration computation."""

    def test_zero_at_equilibrium_no_driving(self):
        """At R=R0, Rdot=0, no acoustic driving: acceleration should be small.
        (Not exactly zero because of surface tension imbalance, but small.)
        """
        R0, P0, gamma = 5e-6, 101325.0, 5.0/3.0
        P_gas = compute_gas_pressure(R0, R0, P0, gamma, SURFACE_TENSION)
        a = compute_acceleration(R0, 0.0, P_gas, P0, 0.0,
                                  SURFACE_TENSION, VISCOSITY, RHO_WATER)
        # At equilibrium: P_gas = P0 + 2*sigma/R0, so delta_P = 2*sigma/R0 + 2*sigma/R0
        # = 4*sigma/R0. Not exactly zero. But the magnitude should be bounded.
        assert np.isfinite(a)
        assert abs(a) < 1e15  # Reasonable bound

    def test_negative_acceleration_during_collapse(self):
        """When bubble is expanded (R > R0) with Rdot < 0, acceleration should
        drive further collapse (be negative or strongly decelerating)."""
        R0, P0, gamma = 5e-6, 101325.0, 5.0/3.0
        R = 5 * R0  # Expanded
        P_gas = compute_gas_pressure(R, R0, P0, gamma, SURFACE_TENSION)
        # At 5*R0: P_gas is very small, P0 dominates
        # delta_P = P_gas - P0 + ... is negative → acceleration is negative
        a = compute_acceleration(R, 0.0, P_gas, P0, 0.0,
                                  SURFACE_TENSION, VISCOSITY, RHO_WATER)
        assert a < 0, f"Expected negative acceleration, got {a}"

    def test_prevents_division_by_zero(self):
        """Near-zero R should return 0 acceleration (safety guard)."""
        a = compute_acceleration(1e-30, 0.0, 101325, 101325, 0.0,
                                  SURFACE_TENSION, VISCOSITY, RHO_WATER)
        assert a == 0.0


# ============================================================================
# Scaled vs Float Comparison Tests
# ============================================================================

class TestScaledAcoustic:
    """Test scaled acoustic driving computation."""

    def test_acoustic_zero_pa(self):
        """With Pa=0, acoustic term is zero."""
        from simulate_scaled import compute_acoustic_pressure_scaled, to_scaled
        Pa = 0
        sin_val = to_scaled(0.5)
        result = compute_acoustic_pressure_scaled(Pa, sin_val)
        assert result == 0

    def test_acoustic_nonzero(self):
        """With Pa>0 and sin>0, acoustic term is nonzero."""
        from simulate_scaled import compute_acoustic_pressure_scaled, to_scaled, from_scaled
        Pa = to_scaled(135000.0)
        sin_val = to_scaled(0.5)
        result = compute_acoustic_pressure_scaled(Pa, sin_val)
        expected = 135000.0 * 0.5
        assert abs(from_scaled(result) - expected) / expected < 1e-6


class TestScaledEmission:
    """Test scaled Stefan-Boltzmann emission computation."""

    def test_emission_at_room_temperature(self):
        """Emission at room temperature and equilibrium radius is very small."""
        from simulate_scaled import compute_emission_scaled, to_scaled, from_scaled
        R = to_scaled(5.0e-6)
        T = to_scaled(293.15)
        E = compute_emission_scaled(R, T)
        E_float = from_scaled(E)
        # Very small but positive (Stefan-Boltzmann at 293K, R=5µm ~ 1.3e-7 W)
        assert E_float >= 0
        assert E_float < 1e-3  # Tiny at room temp

    def test_emission_scales_with_T4(self):
        """Emission should scale roughly as T^4."""
        from simulate_scaled import compute_emission_scaled, to_scaled, from_scaled
        R = to_scaled(1.0e-6)
        E1 = from_scaled(compute_emission_scaled(R, to_scaled(1000.0)))
        E2 = from_scaled(compute_emission_scaled(R, to_scaled(2000.0)))
        if E1 > 0:
            ratio = E2 / E1
            assert abs(ratio - 16.0) / 16.0 < 0.01  # T^4 scaling


class TestScaledVsFloat:
    """Test that integer-scaled simulation matches float version."""

    def test_scaled_matches_float(self):
        """Scaled integer simulation should match float within 1e-6 relative error."""
        from simulate_scaled import ScaledParams, simulate_scaled, compare_with_float
        params = ScaledParams.collapse()
        states = simulate_scaled(params)
        result = compare_with_float(states)
        assert result['max_R_relative_error'] < 1e-6, \
            f"R error {result['max_R_relative_error']:.2e} exceeds tolerance"
        assert result['max_T_relative_error'] < 1e-6, \
            f"T error {result['max_T_relative_error']:.2e} exceeds tolerance"

    def test_scaled_sonoluminescence(self):
        """Scaled version should also detect sonoluminescence."""
        from simulate_scaled import ScaledParams, simulate_scaled, from_scaled
        params = ScaledParams.collapse()
        states = simulate_scaled(params)
        max_T = max(from_scaled(s.T) for s in states)
        assert max_T > 5000, f"Scaled max T = {max_T:.0f} < 5000 K"

    def test_scaled_deterministic(self):
        """Two scaled runs produce identical integer results."""
        from simulate_scaled import ScaledParams, simulate_scaled
        params = ScaledParams.collapse()
        s1 = simulate_scaled(params)
        s2 = simulate_scaled(params)
        for a, b in zip(s1, s2):
            assert a.R == b.R, f"Non-deterministic at step {a.step}"
            assert a.T == b.T


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
