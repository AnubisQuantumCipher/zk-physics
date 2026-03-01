//! Full sonoluminescence simulation circuit.
//!
//! Chains N physics steps together, proving that a bubble collapse
//! simulation was computed correctly without revealing the parameters.
//!
//! Public inputs: R0, final_temperature
//! Private inputs: all physical parameters + full state trace

use ff::PrimeField;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
};

use crate::chips::field_arith::{FieldArithChip, FieldArithConfig};
use crate::chips::physics_step::{self, PhysicsParams, StepState};
use crate::field_utils::{SCALE, SIGMA_4PI_SCALED};
use crate::witness::SimulationWitness;

#[derive(Clone, Debug)]
pub struct SonoluminescenceConfig {
    field_arith: FieldArithConfig,
    instance: Column<Instance>,
}

#[derive(Clone)]
pub struct SonoluminescenceCircuit {
    pub witness: Option<SimulationWitness>,
    pub n_steps: usize,
}

impl SonoluminescenceCircuit {
    pub fn new(witness: SimulationWitness) -> Self {
        let n_steps = witness.n_steps;
        Self {
            witness: Some(witness),
            n_steps,
        }
    }

    pub fn min_k(n_steps: usize) -> u32 {
        let regions_per_step = 44; // 28 base + 2 acoustic + 1 sin load + 5 emission + 2 accum + 6 margin
        let overhead = 20;
        let total = n_steps * regions_per_step + overhead;
        let mut k = 1u32;
        while (1usize << k) < total + 10 {
            k += 1;
        }
        k
    }

    fn val<F: PrimeField>(
        opt_witness: &Option<SimulationWitness>,
        f: impl Fn(&SimulationWitness) -> u128,
    ) -> Value<F> {
        match opt_witness {
            Some(w) => Value::known(F::from_u128(f(w))),
            None => Value::unknown(),
        }
    }

    fn signed_val<F: PrimeField>(
        opt_witness: &Option<SimulationWitness>,
        f: impl Fn(&SimulationWitness) -> i128,
    ) -> Value<F> {
        match opt_witness {
            Some(w) => {
                let v = f(w);
                if v >= 0 {
                    Value::known(F::from_u128(v as u128))
                } else {
                    Value::known(-F::from_u128((-v) as u128))
                }
            }
            None => Value::unknown(),
        }
    }
}

/// Convert a signed i128 to a field element.
fn signed_to_field<F: PrimeField>(v: i128) -> F {
    if v >= 0 {
        F::from_u128(v as u128)
    } else {
        -F::from_u128((-v) as u128)
    }
}

/// Compute the expected public outputs by replaying the circuit's field arithmetic.
/// This ensures the public inputs exactly match the circuit's computation.
pub fn compute_public_inputs<F: PrimeField>(witness: &SimulationWitness) -> Vec<F> {
    let s = F::from_u128(SCALE);
    let s_inv = s.invert().unwrap();

    let r0 = F::from_u128(witness.r0);
    let p0 = F::from_u128(witness.p0);
    let p_initial = F::from_u128(witness.p_initial);
    let t0 = F::from_u128(witness.t0);
    let sigma = F::from_u128(witness.sigma);
    let mu = F::from_u128(witness.mu);
    let rho = F::from_u128(witness.rho);
    let dt = F::from_u128(witness.dt);
    let dt2 = F::from_u128(witness.dt2);
    let pa = F::from_u128(witness.pa);
    let two = F::from_u128(2 * SCALE);
    let four = F::from_u128(4 * SCALE);
    let three_halves = F::from_u128(3 * SCALE / 2);

    let sigma_4pi = F::from_u128(SIGMA_4PI_SCALED);

    let mut r_curr = F::from_u128(witness.r_trace[0]);
    let mut r_prev = r_curr;
    let mut last_temp = F::ZERO;
    let mut cumulative_emission = F::ZERO;

    for step in 0..witness.n_steps {
        let sin_i: F = signed_to_field(witness.sin_values[step]);

        // Rdot = (R_curr - R_prev) / dt = (R_curr - R_prev) * S * dt^{-1}
        let r_diff = r_curr - r_prev;
        let rdot = r_diff * s * dt.invert().unwrap();

        // ratio = R0 / R = R0 * S * R^{-1}
        let ratio = r0 * s * r_curr.invert().unwrap();

        // ratio^5
        let r2 = ratio * ratio * s_inv;
        let r3 = r2 * ratio * s_inv;
        let r4 = r3 * ratio * s_inv;
        let r5 = r4 * ratio * s_inv;

        // P_gas = P_initial * ratio^5
        let p_gas = p_initial * r5 * s_inv;

        // T = T0 * ratio^2
        let ratio_sq = ratio * ratio * s_inv;
        last_temp = t0 * ratio_sq * s_inv;

        // 2*sigma / R
        let two_sigma = two * sigma * s_inv;
        let two_sigma_over_r = two_sigma * s * r_curr.invert().unwrap();

        // 4*mu*Rdot/R
        let four_mu = four * mu * s_inv;
        let four_mu_rdot = four_mu * rdot * s_inv;
        let viscous = four_mu_rdot * s * r_curr.invert().unwrap();

        // delta_P = P_gas - P0 + 2sigma/R - viscous - Pa*sin
        let dp1 = p_gas - p0;
        let dp2 = dp1 + two_sigma_over_r;
        let dp3 = dp2 - viscous;

        // Acoustic term
        let pa_sin = pa * sin_i * s_inv;
        let delta_p = dp3 - pa_sin;

        // rho * R
        let rho_r = rho * r_curr * s_inv;

        // term1 = delta_P / (rho*R)
        let term1 = delta_p * s * rho_r.invert().unwrap();

        // term2 = 1.5 * Rdot^2 / R
        let rdot_sq = rdot * rdot * s_inv;
        let rdot_sq_over_r = rdot_sq * s * r_curr.invert().unwrap();
        let term2 = three_halves * rdot_sq_over_r * s_inv;

        // Rddot
        let rddot = term1 - term2;

        // Verlet update
        let two_r = two * r_curr * s_inv;
        let verlet_pos = two_r - r_prev;
        let accel_term = rddot * dt2 * s_inv;
        let r_next = verlet_pos + accel_term;

        // Emission: sigma_4pi * T^4 * R^2
        let t_sq = last_temp * last_temp * s_inv;
        let t_4 = t_sq * t_sq * s_inv;
        let r_sq = r_curr * r_curr * s_inv;
        let t4_r2 = t_4 * r_sq * s_inv;
        let emission = sigma_4pi * t4_r2 * s_inv;
        let emission_dt = emission * dt * s_inv;
        cumulative_emission = cumulative_emission + emission_dt;

        r_prev = r_curr;
        r_curr = r_next;
    }

    vec![r0, last_temp, cumulative_emission]
}

impl<F: PrimeField> Circuit<F> for SonoluminescenceCircuit {
    type Config = SonoluminescenceConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            witness: None,
            n_steps: self.n_steps,
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let instance = meta.instance_column();

        meta.enable_equality(instance);

        let field_arith = FieldArithChip::<F>::configure(meta, a, b, c);

        SonoluminescenceConfig {
            field_arith,
            instance,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = FieldArithChip::<F>::construct(config.field_arith.clone());
        let w = &self.witness;

        // Load parameters
        let r0 = chip.load_private(layouter.namespace(|| "R0"), Self::val(w, |w| w.r0))?;
        let p0 = chip.load_private(layouter.namespace(|| "P0"), Self::val(w, |w| w.p0))?;
        let p_initial = chip.load_private(layouter.namespace(|| "P_initial"), Self::val(w, |w| w.p_initial))?;
        let t0 = chip.load_private(layouter.namespace(|| "T0"), Self::val(w, |w| w.t0))?;
        let sigma = chip.load_private(layouter.namespace(|| "sigma"), Self::val(w, |w| w.sigma))?;
        let mu = chip.load_private(layouter.namespace(|| "mu"), Self::val(w, |w| w.mu))?;
        let rho = chip.load_private(layouter.namespace(|| "rho"), Self::val(w, |w| w.rho))?;
        let dt = chip.load_private(layouter.namespace(|| "dt"), Self::val(w, |w| w.dt))?;
        let dt2 = chip.load_private(layouter.namespace(|| "dt2"), Self::val(w, |w| w.dt2))?;
        let pa = chip.load_private(layouter.namespace(|| "Pa"), Self::val(w, |w| w.pa))?;

        let two = chip.load_constant(layouter.namespace(|| "two"), F::from_u128(2 * SCALE))?;
        let four = chip.load_constant(layouter.namespace(|| "four"), F::from_u128(4 * SCALE))?;
        let three_halves = chip.load_constant(layouter.namespace(|| "1.5"), F::from_u128(3 * SCALE / 2))?;
        let sigma_4pi = chip.load_constant(layouter.namespace(|| "sigma_4pi"), F::from_u128(SIGMA_4PI_SCALED))?;

        let params = PhysicsParams {
            r0: r0.clone(),
            p0,
            p_initial,
            t0,
            sigma,
            mu,
            rho,
            dt,
            dt2,
            two,
            four,
            three_halves,
            pa,
            sigma_4pi,
        };

        // Expose R0 as public input (row 0)
        layouter.constrain_instance(r0.cell(), config.instance, 0)?;

        // Load initial state
        let r_start_val: Value<F> = match w {
            Some(w) => Value::known(F::from_u128(w.r_trace[0])),
            None => Value::unknown(),
        };
        let mut r_curr = chip.load_private(layouter.namespace(|| "R_start"), r_start_val)?;
        let mut r_prev = r_curr.clone();

        let mut last_temp = chip.load_constant(layouter.namespace(|| "zero_temp"), F::ZERO)?;
        let mut cumulative_emission = chip.load_constant(layouter.namespace(|| "zero_emission"), F::ZERO)?;

        for step in 0..self.n_steps {
            // Load per-step sin as private witness
            let sin_i = chip.load_private(
                layouter.namespace(|| format!("sin_{}", step)),
                Self::signed_val(w, |w| w.sin_values[step]),
            )?;

            let state = StepState {
                r_curr: r_curr.clone(),
                r_prev: r_prev.clone(),
            };

            let result = physics_step::physics_step(
                &chip,
                layouter.namespace(|| format!("step_{}", step)),
                &params,
                &state,
                &sin_i,
            )?;

            // Accumulate emission energy: emission * dt
            let emission_dt = chip.scaled_mul(
                layouter.namespace(|| format!("emission_dt_{}", step)),
                &result.emission,
                &params.dt,
            )?;
            cumulative_emission = chip.add(
                layouter.namespace(|| format!("cum_emission_{}", step)),
                &cumulative_emission,
                &emission_dt,
            )?;

            last_temp = result.temperature;
            r_prev = result.r_curr;
            r_curr = result.r_next;
        }

        // Expose final temperature as public output (row 1)
        layouter.constrain_instance(last_temp.cell(), config.instance, 1)?;

        // Expose cumulative emission as public output (row 2)
        layouter.constrain_instance(cumulative_emission.cell(), config.instance, 2)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witness::SimulationWitness;
    use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr};

    #[test]
    fn test_circuit_10_steps() {
        let witness = SimulationWitness::collapse_preset(10);
        let public_inputs = compute_public_inputs::<Fr>(&witness);
        let circuit = SonoluminescenceCircuit::new(witness);
        let k = SonoluminescenceCircuit::min_k(10);

        let prover = MockProver::run(k, &circuit, vec![public_inputs]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_circuit_50_steps() {
        let witness = SimulationWitness::collapse_preset(50);
        let public_inputs = compute_public_inputs::<Fr>(&witness);
        let circuit = SonoluminescenceCircuit::new(witness);
        let k = SonoluminescenceCircuit::min_k(50);

        let prover = MockProver::run(k, &circuit, vec![public_inputs]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_circuit_10_steps_acoustic() {
        let witness = SimulationWitness::acoustic_preset(10);
        let public_inputs = compute_public_inputs::<Fr>(&witness);
        let circuit = SonoluminescenceCircuit::new(witness);
        let k = SonoluminescenceCircuit::min_k(10);

        let prover = MockProver::run(k, &circuit, vec![public_inputs]).unwrap();
        prover.assert_satisfied();
    }
}
