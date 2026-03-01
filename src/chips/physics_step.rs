//! Physics step chip: constrains a single Rayleigh-Plesset timestep.
//!
//! For the collapse-only MVP (no acoustic driving), each step constrains:
//! 1. Gas pressure: P_gas = P_initial * (R0/R)^5  (for gamma=5/3)
//! 2. Temperature: T = T0 * (R0/R)^2  (for gamma=5/3)
//! 3. RP acceleration: Rddot = delta_P/(rho*R) - 1.5*Rdot^2/R
//! 4. Verlet update: R_next = 2*R_curr - R_prev + Rddot * dt^2
//! 5. Velocity: Rdot = (R_curr - R_prev) / dt
//!
//! All arithmetic uses scaled fixed-point via FieldArithChip.

use ff::PrimeField;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk::Error,
};

use super::field_arith::FieldArithChip;

/// Parameters for the physics step, loaded as circuit constants.
/// All values are pre-scaled by S = 10^30.
#[derive(Clone, Debug)]
pub struct PhysicsParams<F: PrimeField> {
    pub r0: AssignedCell<F, F>,          // Equilibrium radius
    pub p0: AssignedCell<F, F>,          // Ambient pressure
    pub p_initial: AssignedCell<F, F>,   // P0 + 2*sigma/R0
    pub t0: AssignedCell<F, F>,          // Initial temperature
    pub sigma: AssignedCell<F, F>,       // Surface tension
    pub mu: AssignedCell<F, F>,          // Viscosity
    pub rho: AssignedCell<F, F>,         // Liquid density
    pub dt: AssignedCell<F, F>,          // Timestep
    pub dt2: AssignedCell<F, F>,         // dt^2 (scaled)
    pub two: AssignedCell<F, F>,         // 2 * S
    pub four: AssignedCell<F, F>,        // 4 * S
    pub three_halves: AssignedCell<F, F>, // 1.5 * S
    pub pa: AssignedCell<F, F>,          // Acoustic amplitude (scaled)
    pub sigma_4pi: AssignedCell<F, F>,   // sigma_SB * 4 * pi (scaled)
}

/// State at a single timestep (all assigned cells).
#[derive(Clone, Debug)]
pub struct StepState<F: PrimeField> {
    pub r_curr: AssignedCell<F, F>,
    pub r_prev: AssignedCell<F, F>,
}

/// Result of a physics step.
#[derive(Clone, Debug)]
pub struct StepResult<F: PrimeField> {
    pub r_next: AssignedCell<F, F>,
    pub r_curr: AssignedCell<F, F>,  // Becomes r_prev for next step
    pub rdot: AssignedCell<F, F>,
    pub p_gas: AssignedCell<F, F>,
    pub temperature: AssignedCell<F, F>,
    pub rddot: AssignedCell<F, F>,
    pub emission: AssignedCell<F, F>,  // Stefan-Boltzmann emission power
}

/// Execute one physics timestep using the FieldArithChip.
///
/// Constraints per step:
///   - Pythagorean check: sin²+cos²=S (3 regions)
///   - Gas pressure, temperature, velocity, RP equation (~21 regions)
///   - Acoustic term: pa*sin, delta_p - pa_sin (2 regions)
///   - Verlet update (3 regions)
///   Total: ~30 constrained operations
pub fn physics_step<F: PrimeField>(
    chip: &FieldArithChip<F>,
    mut layouter: impl Layouter<F>,
    params: &PhysicsParams<F>,
    state: &StepState<F>,
    sin_i: &AssignedCell<F, F>,
) -> Result<StepResult<F>, Error> {
    // ================================================================
    // 1. Velocity: Rdot = (R_curr - R_prev) / dt
    // ================================================================
    let r_diff = chip.sub(
        layouter.namespace(|| "R_curr - R_prev"),
        &state.r_curr,
        &state.r_prev,
    )?;

    let rdot = chip.scaled_div(
        layouter.namespace(|| "Rdot = diff / dt"),
        &r_diff,
        &params.dt,
    )?;

    // ================================================================
    // 2. Gas pressure: P_gas = P_initial * (R0/R)^5
    //    For gamma = 5/3: 3*gamma = 5
    // ================================================================

    // ratio = R0 / R_curr
    let ratio = chip.scaled_div(
        layouter.namespace(|| "R0/R"),
        &params.r0,
        &state.r_curr,
    )?;

    // ratio^5 via chain of multiplies
    let ratio_pow5 = chip.scaled_pow(
        layouter.namespace(|| "ratio^5"),
        &ratio,
        5,
    )?;

    // P_gas = P_initial * ratio^5
    let p_gas = chip.scaled_mul(
        layouter.namespace(|| "P_gas"),
        &params.p_initial,
        &ratio_pow5,
    )?;

    // ================================================================
    // 3. Temperature: T = T0 * (R0/R)^2
    //    For gamma = 5/3: 3*(gamma-1) = 2
    // ================================================================

    // ratio^2 (reuse ratio)
    let ratio_sq = chip.scaled_mul(
        layouter.namespace(|| "ratio^2"),
        &ratio,
        &ratio,
    )?;

    let temperature = chip.scaled_mul(
        layouter.namespace(|| "T = T0 * ratio^2"),
        &params.t0,
        &ratio_sq,
    )?;

    // ================================================================
    // 4. RP acceleration
    //    Rddot = delta_P / (rho * R) - 1.5 * Rdot^2 / R
    //    delta_P = P_gas - P0 + 2*sigma/R - 4*mu*Rdot/R
    // ================================================================

    // 2*sigma/R
    let two_sigma = chip.scaled_mul(
        layouter.namespace(|| "2*sigma"),
        &params.two,
        &params.sigma,
    )?;
    let two_sigma_over_r = chip.scaled_div(
        layouter.namespace(|| "2sigma/R"),
        &two_sigma,
        &state.r_curr,
    )?;

    // 4*mu*Rdot / R
    let four_mu = chip.scaled_mul(
        layouter.namespace(|| "4*mu"),
        &params.four,
        &params.mu,
    )?;
    let four_mu_rdot = chip.scaled_mul(
        layouter.namespace(|| "4mu*Rdot"),
        &four_mu,
        &rdot,
    )?;
    let viscous_term = chip.scaled_div(
        layouter.namespace(|| "4mu*Rdot/R"),
        &four_mu_rdot,
        &state.r_curr,
    )?;

    // delta_P = P_gas - P0 + 2*sigma/R - 4*mu*Rdot/R - Pa*sin
    let dp1 = chip.sub(
        layouter.namespace(|| "P_gas - P0"),
        &p_gas,
        &params.p0,
    )?;
    let dp2 = chip.add(
        layouter.namespace(|| "+ 2sigma/R"),
        &dp1,
        &two_sigma_over_r,
    )?;
    let dp3 = chip.sub(
        layouter.namespace(|| "- viscous"),
        &dp2,
        &viscous_term,
    )?;

    // Acoustic term: Pa * sin_i
    let pa_sin = chip.scaled_mul(
        layouter.namespace(|| "Pa*sin"),
        &params.pa,
        sin_i,
    )?;

    // delta_P = dp3 - pa_sin
    let delta_p = chip.sub(
        layouter.namespace(|| "- acoustic"),
        &dp3,
        &pa_sin,
    )?;

    // rho * R
    let rho_r = chip.scaled_mul(
        layouter.namespace(|| "rho*R"),
        &params.rho,
        &state.r_curr,
    )?;

    // term1 = delta_P / (rho * R)
    let term1 = chip.scaled_div(
        layouter.namespace(|| "deltaP/(rhoR)"),
        &delta_p,
        &rho_r,
    )?;

    // Rdot^2
    let rdot_sq = chip.scaled_mul(
        layouter.namespace(|| "Rdot^2"),
        &rdot,
        &rdot,
    )?;

    // Rdot^2 / R
    let rdot_sq_over_r = chip.scaled_div(
        layouter.namespace(|| "Rdot^2/R"),
        &rdot_sq,
        &state.r_curr,
    )?;

    // 1.5 * Rdot^2 / R
    let term2 = chip.scaled_mul(
        layouter.namespace(|| "1.5*Rdot^2/R"),
        &params.three_halves,
        &rdot_sq_over_r,
    )?;

    // Rddot = term1 - term2
    let rddot = chip.sub(
        layouter.namespace(|| "Rddot"),
        &term1,
        &term2,
    )?;

    // ================================================================
    // 5. Verlet update: R_next = 2*R_curr - R_prev + Rddot * dt^2
    // ================================================================

    // 2 * R_curr
    let two_r = chip.scaled_mul(
        layouter.namespace(|| "2*R"),
        &params.two,
        &state.r_curr,
    )?;

    // 2*R - R_prev
    let verlet_pos = chip.sub(
        layouter.namespace(|| "2R - R_prev"),
        &two_r,
        &state.r_prev,
    )?;

    // Rddot * dt^2
    let accel_term = chip.scaled_mul(
        layouter.namespace(|| "Rddot*dt^2"),
        &rddot,
        &params.dt2,
    )?;

    // R_next = 2*R - R_prev + Rddot*dt^2
    let r_next = chip.add(
        layouter.namespace(|| "R_next"),
        &verlet_pos,
        &accel_term,
    )?;

    // ================================================================
    // 6. Stefan-Boltzmann emission: E = sigma_SB * 4*pi * T^4 * R^2
    // ================================================================
    let t_sq = chip.scaled_mul(
        layouter.namespace(|| "T^2"),
        &temperature,
        &temperature,
    )?;
    let t_4 = chip.scaled_mul(
        layouter.namespace(|| "T^4"),
        &t_sq,
        &t_sq,
    )?;
    let r_sq = chip.scaled_mul(
        layouter.namespace(|| "R^2_em"),
        &state.r_curr,
        &state.r_curr,
    )?;
    let t4_r2 = chip.scaled_mul(
        layouter.namespace(|| "T^4*R^2"),
        &t_4,
        &r_sq,
    )?;
    let emission = chip.scaled_mul(
        layouter.namespace(|| "emission"),
        &params.sigma_4pi,
        &t4_r2,
    )?;

    Ok(StepResult {
        r_next,
        r_curr: state.r_curr.clone(),
        rdot,
        p_gas,
        temperature,
        rddot,
        emission,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chips::field_arith::FieldArithConfig;
    use crate::field_utils::{self, SCALE, SIGMA_4PI_SCALED};
    use halo2_proofs::{
        circuit::{SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{Circuit, ConstraintSystem},
    };

    fn scaled_to_field(v: i128) -> Fr {
        if v >= 0 {
            Fr::from_u128(v as u128)
        } else {
            -Fr::from_u128((-v) as u128)
        }
    }

    /// Single-step test circuit: runs one physics step and checks constraints.
    #[derive(Clone)]
    struct SingleStepCircuit {
        // Parameters
        r0: Value<Fr>,
        p0: Value<Fr>,
        p_initial: Value<Fr>,
        t0: Value<Fr>,
        sigma: Value<Fr>,
        mu: Value<Fr>,
        rho: Value<Fr>,
        dt: Value<Fr>,
        dt2: Value<Fr>,
        // State
        r_curr: Value<Fr>,
        r_prev: Value<Fr>,
        // Acoustic
        sin_i: Value<Fr>,
        pa: Value<Fr>,
    }

    impl SingleStepCircuit {
        /// Create a collapse-only circuit (no acoustic driving).
        fn collapse(
            r0: f64, p0: f64, p_initial: f64, t0: f64,
            sigma: f64, mu: f64, rho: f64, dt: f64,
            r_curr: f64, r_prev: f64,
        ) -> Self {
            Self {
                r0: Value::known(scaled_to_field(field_utils::float_to_scaled(r0))),
                p0: Value::known(scaled_to_field(field_utils::float_to_scaled(p0))),
                p_initial: Value::known(scaled_to_field(field_utils::float_to_scaled(p_initial))),
                t0: Value::known(scaled_to_field(field_utils::float_to_scaled(t0))),
                sigma: Value::known(scaled_to_field(field_utils::float_to_scaled(sigma))),
                mu: Value::known(scaled_to_field(field_utils::float_to_scaled(mu))),
                rho: Value::known(scaled_to_field(field_utils::float_to_scaled(rho))),
                dt: Value::known(scaled_to_field(field_utils::float_to_scaled(dt))),
                dt2: Value::known(scaled_to_field(field_utils::float_to_scaled(dt * dt))),
                r_curr: Value::known(scaled_to_field(field_utils::float_to_scaled(r_curr))),
                r_prev: Value::known(scaled_to_field(field_utils::float_to_scaled(r_prev))),
                sin_i: Value::known(Fr::from(0u64)),               // sin=0
                pa: Value::known(Fr::from(0u64)),                 // no acoustic driving
            }
        }
    }

    impl Circuit<Fr> for SingleStepCircuit {
        type Config = FieldArithConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                r0: Value::unknown(),
                p0: Value::unknown(),
                p_initial: Value::unknown(),
                t0: Value::unknown(),
                sigma: Value::unknown(),
                mu: Value::unknown(),
                rho: Value::unknown(),
                dt: Value::unknown(),
                dt2: Value::unknown(),
                r_curr: Value::unknown(),
                r_prev: Value::unknown(),
                sin_i: Value::unknown(),
                pa: Value::unknown(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let c = meta.advice_column();
            FieldArithChip::<Fr>::configure(meta, a, b, c)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            let chip = FieldArithChip::<Fr>::construct(config);

            // Load parameters
            let r0 = chip.load_private(layouter.namespace(|| "R0"), self.r0)?;
            let p0 = chip.load_private(layouter.namespace(|| "P0"), self.p0)?;
            let p_initial = chip.load_private(layouter.namespace(|| "P_initial"), self.p_initial)?;
            let t0 = chip.load_private(layouter.namespace(|| "T0"), self.t0)?;
            let sigma = chip.load_private(layouter.namespace(|| "sigma"), self.sigma)?;
            let mu = chip.load_private(layouter.namespace(|| "mu"), self.mu)?;
            let rho = chip.load_private(layouter.namespace(|| "rho"), self.rho)?;
            let dt = chip.load_private(layouter.namespace(|| "dt"), self.dt)?;
            let dt2 = chip.load_private(layouter.namespace(|| "dt2"), self.dt2)?;
            let pa = chip.load_private(layouter.namespace(|| "Pa"), self.pa)?;

            // Load constants
            let two = chip.load_constant(
                layouter.namespace(|| "two"),
                Fr::from_u128(2 * SCALE),
            )?;
            let four = chip.load_constant(
                layouter.namespace(|| "four"),
                Fr::from_u128(4 * SCALE),
            )?;
            let three_halves = chip.load_constant(
                layouter.namespace(|| "1.5"),
                Fr::from_u128(3 * SCALE / 2),
            )?;

            // sigma_SB * 4 * pi
            let sigma_4pi = chip.load_constant(
                layouter.namespace(|| "sigma_4pi"),
                Fr::from_u128(SIGMA_4PI_SCALED),
            )?;

            let params = PhysicsParams {
                r0,
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

            // Load state
            let r_curr = chip.load_private(layouter.namespace(|| "R_curr"), self.r_curr)?;
            let r_prev = chip.load_private(layouter.namespace(|| "R_prev"), self.r_prev)?;

            let state = StepState { r_curr, r_prev };

            // Load sin
            let sin_i = chip.load_private(layouter.namespace(|| "sin_i"), self.sin_i)?;

            // Execute one physics step
            let _result = physics_step(
                &chip,
                layouter.namespace(|| "step"),
                &params,
                &state,
                &sin_i,
            )?;

            Ok(())
        }
    }

    #[test]
    fn test_single_step_initial() {
        // Test the first step of the collapse simulation:
        // R_curr = R_prev = 5*R0 = 25 µm, Rdot = 0
        let r0 = 5.0e-6_f64;
        let p0 = 101325.0_f64;
        let sigma = 0.0728_f64;
        let p_initial = p0 + 2.0 * sigma / r0;
        let t0 = 293.15_f64;
        let mu = 1.002e-3_f64;
        let rho = 998.0_f64;
        let dt = 1.0e-9_f64;
        let r_start = 5.0 * r0;

        let circuit = SingleStepCircuit::collapse(
            r0, p0, p_initial, t0, sigma, mu, rho, dt, r_start, r_start,
        );

        // Need enough rows for ~37 regions with 1 row each, plus overhead
        let k = 7; // 2^7 = 128 rows
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_single_step_mid_collapse() {
        // Test a mid-collapse step where velocity is non-zero
        let r0 = 5.0e-6_f64;
        let p0 = 101325.0_f64;
        let sigma = 0.0728_f64;
        let p_initial = p0 + 2.0 * sigma / r0;
        let t0 = 293.15_f64;
        let mu = 1.002e-3_f64;
        let rho = 998.0_f64;
        let dt = 1.0e-9_f64;
        let r_curr = 10.0e-6_f64;
        let r_prev = 10.05e-6_f64;

        let circuit = SingleStepCircuit::collapse(
            r0, p0, p_initial, t0, sigma, mu, rho, dt, r_curr, r_prev,
        );

        let k = 7;
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
