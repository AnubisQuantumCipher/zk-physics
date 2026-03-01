//! Witness generation for the sonoluminescence circuit.
//!
//! Uses arbitrary-precision integers (BigInt) for all arithmetic to match
//! the ZK circuit's 254-bit field exactly. Values are converted to u128
//! only when loading into halo2 cells.

use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use std::f64::consts::PI;
/// Scaling factor as BigInt.
fn scale() -> BigInt {
    BigInt::from(10u64).pow(30)
}

/// Scaled multiply: (a * b) / S
fn smul(a: &BigInt, b: &BigInt) -> BigInt {
    (a * b) / scale()
}

/// Scaled divide: (a * S) / b
fn sdiv(a: &BigInt, b: &BigInt) -> BigInt {
    (a * scale()) / b
}

/// Convert float to scaled BigInt.
fn to_scaled(v: f64) -> BigInt {
    BigInt::from((v * 1e30) as i128)
}

/// Convert BigInt to u128 for halo2.
fn to_u128(v: &BigInt) -> u128 {
    v.to_u128().expect("Value too large for u128")
}

/// Convert a float to a scaled BigInt value using BigInt precision.
/// Uses two-stage multiplication to avoid f64 precision loss at large scales.
fn float_to_scaled_big(v: f64) -> BigInt {
    // f64 has ~15 significant digits. Split: v * 10^15 as i64, then * 10^15
    let hi = (v * 1e15).round() as i64;
    BigInt::from(hi) * BigInt::from(10u64).pow(15)
}

/// Full witness for the sonoluminescence simulation.
#[derive(Clone, Debug)]
pub struct SimulationWitness {
    pub r0: u128,
    pub p0: u128,
    pub p_initial: u128,
    pub t0: u128,
    pub sigma: u128,
    pub mu: u128,
    pub rho: u128,
    pub dt: u128,
    pub dt2: u128,
    pub r_trace: Vec<u128>,
    pub n_steps: usize,
    pub pa: u128,
    pub sin_values: Vec<i128>,
    pub cos_values: Vec<i128>,
}

impl SimulationWitness {
    /// Generate witness for the collapse preset.
    pub fn collapse_preset(n_steps: usize) -> Self {
        let s = scale();

        let r0 = to_scaled(5.0e-6);
        let p0 = to_scaled(101325.0);
        let sigma = to_scaled(0.0728);
        let mu = to_scaled(1.002e-3);
        let rho = to_scaled(998.0);
        let t0 = to_scaled(293.15);
        let dt = to_scaled(1.0e-9);
        let r_start: BigInt = &r0 * 5; // 5 * R0

        // P_initial = P0 + 2*sigma/R0
        let two_s = &s * 2;
        let two_sigma = smul(&two_s, &sigma);
        let two_sigma_over_r0 = sdiv(&two_sigma, &r0);
        let p_initial = &p0 + &two_sigma_over_r0;

        // dt^2
        let dt2 = smul(&dt, &dt);

        let r_min: BigInt = &r0 / 100;

        let four_s = &s * 4;
        let three_halves_s = &s * 3 / 2;

        let mut r_trace_big: Vec<BigInt> = Vec::with_capacity(n_steps + 1);
        let mut r_curr = r_start;
        let mut r_prev = r_curr.clone();

        for i in 0..=n_steps {
            r_trace_big.push(r_curr.clone());

            if i < n_steps {
                // Velocity
                let (rdot, rdot_neg) = if i == 0 {
                    (BigInt::zero(), false)
                } else if r_curr >= r_prev {
                    (sdiv(&(&r_curr - &r_prev), &dt), false)
                } else {
                    (sdiv(&(&r_prev - &r_curr), &dt), true)
                };

                // Gas pressure: P_initial * (R0/R)^5
                let ratio = sdiv(&r0, &r_curr);
                let ratio2 = smul(&ratio, &ratio);
                let ratio3 = smul(&ratio2, &ratio);
                let ratio4 = smul(&ratio3, &ratio);
                let ratio5 = smul(&ratio4, &ratio);
                let p_gas = smul(&p_initial, &ratio5);

                // 2*sigma/R
                let two_sigma_over_r = sdiv(&two_sigma, &r_curr);

                // 4*mu*Rdot/R
                let four_mu = smul(&four_s, &mu);
                let four_mu_rdot = smul(&four_mu, &rdot);
                let viscous = sdiv(&four_mu_rdot, &r_curr);

                // delta_P
                let mut dp_pos = &p_gas + &two_sigma_over_r;
                let mut dp_neg = p0.clone();
                if rdot_neg {
                    dp_pos = dp_pos + &viscous;
                } else {
                    dp_neg = dp_neg + &viscous;
                }

                let (delta_p, delta_p_neg) = if dp_pos >= dp_neg {
                    (&dp_pos - &dp_neg, false)
                } else {
                    (&dp_neg - &dp_pos, true)
                };

                // rho * R
                let rho_r = smul(&rho, &r_curr);

                // term1 = delta_P / (rho * R)
                let term1 = sdiv(&delta_p, &rho_r);
                let term1_neg = delta_p_neg;

                // term2 = 1.5 * Rdot^2 / R
                let rdot_sq = smul(&rdot, &rdot);
                let rdot_sq_over_r = sdiv(&rdot_sq, &r_curr);
                let term2 = smul(&three_halves_s, &rdot_sq_over_r);

                // Rddot = term1 - term2
                let (rddot, rddot_neg) = if term1_neg {
                    (&term1 + &term2, true)
                } else if term1 >= term2 {
                    (&term1 - &term2, false)
                } else {
                    (&term2 - &term1, true)
                };

                // Verlet: R_next = 2*R_curr - R_prev + Rddot * dt^2
                let accel_term = smul(&rddot, &dt2);

                let r_next = if rddot_neg {
                    let base = &r_curr * 2 - &r_prev;
                    if base > accel_term {
                        base - &accel_term
                    } else {
                        r_min.clone()
                    }
                } else {
                    &r_curr * 2 - &r_prev + &accel_term
                };

                let r_next = if r_next < r_min { r_min.clone() } else { r_next };

                r_prev = r_curr;
                r_curr = r_next;
            }
        }

        // Convert to u128 for halo2
        let r_trace: Vec<u128> = r_trace_big.iter().map(|r| to_u128(r)).collect();

        // Collapse preset: no acoustic driving
        // sin = 0, cos = 1.0 (scaled) for all steps → Pythagorean identity holds
        let sin_values = vec![0i128; n_steps];
        let cos_values = vec![scale().to_i128().unwrap(); n_steps];

        SimulationWitness {
            r0: to_u128(&r0),
            p0: to_u128(&p0),
            p_initial: to_u128(&p_initial),
            t0: to_u128(&t0),
            sigma: to_u128(&sigma),
            mu: to_u128(&mu),
            rho: to_u128(&rho),
            dt: to_u128(&dt),
            dt2: to_u128(&dt2),
            r_trace,
            n_steps,
            pa: 0,
            sin_values,
            cos_values,
        }
    }

    /// Generate witness for acoustic driving preset.
    /// Pa = 135 kPa, freq = 26.5 kHz
    pub fn acoustic_preset(n_steps: usize) -> Self {
        let s = scale();

        let r0 = to_scaled(5.0e-6);
        let p0 = to_scaled(101325.0);
        let sigma = to_scaled(0.0728);
        let mu = to_scaled(1.002e-3);
        let rho = to_scaled(998.0);
        let t0 = to_scaled(293.15);
        let dt_f64 = 1.0e-9_f64;
        let dt = to_scaled(dt_f64);
        let r_start: BigInt = &r0 * 5;
        let pa_f64 = 135000.0_f64;
        let freq = 26500.0_f64;

        // P_initial = P0 + 2*sigma/R0
        let two_s = &s * 2;
        let two_sigma = smul(&two_s, &sigma);
        let two_sigma_over_r0 = sdiv(&two_sigma, &r0);
        let p_initial = &p0 + &two_sigma_over_r0;

        let dt2 = smul(&dt, &dt);

        let r_min: BigInt = &r0 / 100;

        let four_s = &s * 4;
        let three_halves_s = &s * 3 / 2;

        // Compute sin/cos values for each step
        let mut sin_values = Vec::with_capacity(n_steps);
        let mut cos_values = Vec::with_capacity(n_steps);
        for i in 0..n_steps {
            let t = (i as f64) * dt_f64;
            let angle = 2.0 * PI * freq * t;
            let sin_big = float_to_scaled_big(angle.sin());
            let cos_big = float_to_scaled_big(angle.cos());
            sin_values.push(sin_big.to_i128().unwrap());
            cos_values.push(cos_big.to_i128().unwrap());
        }

        // Simulate with acoustic driving
        let mut r_trace_big: Vec<BigInt> = Vec::with_capacity(n_steps + 1);
        let mut r_curr = r_start;
        let mut r_prev = r_curr.clone();
        let pa_big = to_scaled(pa_f64);

        for i in 0..=n_steps {
            r_trace_big.push(r_curr.clone());

            if i < n_steps {
                // Velocity
                let (rdot, rdot_neg) = if i == 0 {
                    (BigInt::zero(), false)
                } else if r_curr >= r_prev {
                    (sdiv(&(&r_curr - &r_prev), &dt), false)
                } else {
                    (sdiv(&(&r_prev - &r_curr), &dt), true)
                };

                // Gas pressure: P_initial * (R0/R)^5
                let ratio = sdiv(&r0, &r_curr);
                let ratio2 = smul(&ratio, &ratio);
                let ratio3 = smul(&ratio2, &ratio);
                let ratio4 = smul(&ratio3, &ratio);
                let ratio5 = smul(&ratio4, &ratio);
                let p_gas = smul(&p_initial, &ratio5);

                // 2*sigma/R
                let two_sigma_over_r = sdiv(&two_sigma, &r_curr);

                // 4*mu*Rdot/R
                let four_mu = smul(&four_s, &mu);
                let four_mu_rdot = smul(&four_mu, &rdot);
                let viscous = sdiv(&four_mu_rdot, &r_curr);

                // delta_P = P_gas - P0 + 2sigma/R - 4mu*Rdot/R
                let mut dp_pos = &p_gas + &two_sigma_over_r;
                let mut dp_neg = p0.clone();
                if rdot_neg {
                    dp_pos = dp_pos + &viscous;
                } else {
                    dp_neg = dp_neg + &viscous;
                }

                // Acoustic term: -Pa * sin(2*pi*freq*t)
                // pa_sin = Pa * sin_i
                let sin_big = BigInt::from(sin_values[i]);
                let pa_sin = smul(&pa_big, &sin_big);
                // delta_p_final = old_delta_p - pa_sin
                // (subtracting pa_sin means: if sin > 0, we subtract, giving negative driving)
                if pa_sin >= BigInt::zero() {
                    dp_neg = dp_neg + &pa_sin;
                } else {
                    dp_pos = dp_pos + (-&pa_sin);
                }

                let (delta_p, delta_p_neg) = if dp_pos >= dp_neg {
                    (&dp_pos - &dp_neg, false)
                } else {
                    (&dp_neg - &dp_pos, true)
                };

                // rho * R
                let rho_r = smul(&rho, &r_curr);

                // term1 = delta_P / (rho * R)
                let term1 = sdiv(&delta_p, &rho_r);
                let term1_neg = delta_p_neg;

                // term2 = 1.5 * Rdot^2 / R
                let rdot_sq = smul(&rdot, &rdot);
                let rdot_sq_over_r = sdiv(&rdot_sq, &r_curr);
                let term2 = smul(&three_halves_s, &rdot_sq_over_r);

                // Rddot = term1 - term2
                let (rddot, rddot_neg) = if term1_neg {
                    (&term1 + &term2, true)
                } else if term1 >= term2 {
                    (&term1 - &term2, false)
                } else {
                    (&term2 - &term1, true)
                };

                // Verlet: R_next = 2*R_curr - R_prev + Rddot * dt^2
                let accel_term = smul(&rddot, &dt2);

                let r_next = if rddot_neg {
                    let base = &r_curr * 2 - &r_prev;
                    if base > accel_term {
                        base - &accel_term
                    } else {
                        r_min.clone()
                    }
                } else {
                    &r_curr * 2 - &r_prev + &accel_term
                };

                let r_next = if r_next < r_min { r_min.clone() } else { r_next };

                r_prev = r_curr;
                r_curr = r_next;
            }
        }

        let r_trace: Vec<u128> = r_trace_big.iter().map(|r| to_u128(r)).collect();

        SimulationWitness {
            r0: to_u128(&r0),
            p0: to_u128(&p0),
            p_initial: to_u128(&p_initial),
            t0: to_u128(&t0),
            sigma: to_u128(&sigma),
            mu: to_u128(&mu),
            rho: to_u128(&rho),
            dt: to_u128(&dt),
            dt2: to_u128(&dt2),
            r_trace,
            n_steps,
            pa: to_u128(&to_scaled(pa_f64)),
            sin_values,
            cos_values,
        }
    }

    /// Compute temperature at a given R: T = T0 * (R0/R)^2
    fn temperature_at_big(t0: &BigInt, r0: &BigInt, r: &BigInt) -> BigInt {
        let ratio = sdiv(r0, r);
        let ratio_sq = smul(&ratio, &ratio);
        smul(t0, &ratio_sq)
    }

    /// Compute temperature at the final step.
    pub fn compute_final_temperature(&self) -> u128 {
        let t0 = BigInt::from(self.t0);
        let r0 = BigInt::from(self.r0);
        let r_final = BigInt::from(*self.r_trace.last().unwrap());
        to_u128(&Self::temperature_at_big(&t0, &r0, &r_final))
    }

    /// Compute peak temperature across all steps.
    pub fn compute_peak_temperature(&self) -> u128 {
        let t0 = BigInt::from(self.t0);
        let r0 = BigInt::from(self.r0);
        let peak = self.r_trace.iter()
            .map(|r| Self::temperature_at_big(&t0, &r0, &BigInt::from(*r)))
            .max()
            .unwrap();
        to_u128(&peak)
    }

    pub fn min_radius(&self) -> u128 {
        *self.r_trace.iter().min().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_utils;

    #[test]
    fn test_collapse_witness_10_steps() {
        let w = SimulationWitness::collapse_preset(10);
        assert_eq!(w.r_trace.len(), 11);
        assert_eq!(w.n_steps, 10);
        assert_eq!(w.r_trace[0], 5 * w.r0);
    }

    #[test]
    fn test_collapse_witness_produces_sonoluminescence() {
        let w = SimulationWitness::collapse_preset(3000);
        let peak_t = w.compute_peak_temperature();
        let peak_t_float = field_utils::scaled_to_float(peak_t as i128);
        assert!(
            peak_t_float > 5000.0,
            "Peak temperature {:.0} K < 5000 K",
            peak_t_float
        );
    }

    #[test]
    fn test_witness_deterministic() {
        let w1 = SimulationWitness::collapse_preset(100);
        let w2 = SimulationWitness::collapse_preset(100);
        for (a, b) in w1.r_trace.iter().zip(w2.r_trace.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_witness_r_start() {
        let w = SimulationWitness::collapse_preset(10);
        let r0_f = field_utils::scaled_to_float(w.r_trace[0] as i128);
        assert!(
            (r0_f - 25.0e-6).abs() < 1e-12,
            "R_start = {}, expected 25e-6",
            r0_f
        );
    }

    #[test]
    fn test_witness_all_positive() {
        let w = SimulationWitness::collapse_preset(100);
        for (i, r) in w.r_trace.iter().enumerate() {
            assert!(*r > 0, "R <= 0 at step {}", i);
        }
    }
}
