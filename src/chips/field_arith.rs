//! Field arithmetic chips for scaled fixed-point operations in halo2.
//!
//! These chips constrain scaled multiply and divide operations:
//! - ScaledMulChip: c * S == a * b  (descaled product)
//! - ScaledDivChip: c * b == a * S  (field division with scaling)
//! - PowerChip: chain of scaled multiplies for integer exponents
//!
//! All arithmetic operates on field elements in the bn256 scalar field
//! (the curve used by halo2's KZG backend).

use ff::PrimeField;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed, Selector},
    poly::Rotation,
};
use std::marker::PhantomData;

use crate::field_utils::SCALE;

/// Configuration for scaled arithmetic operations.
/// Uses 3 advice columns (a, b, c) and selectors for mul/div gates.
#[derive(Clone, Debug)]
pub struct FieldArithConfig {
    pub a: Column<Advice>,
    pub b: Column<Advice>,
    pub c: Column<Advice>,
    pub constant: Column<Fixed>,
    pub s_mul: Selector,
    pub s_div: Selector,
    pub s_add: Selector,
    pub s_sub: Selector,
}

/// Chip for scaled field arithmetic.
pub struct FieldArithChip<F: PrimeField> {
    config: FieldArithConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> FieldArithChip<F> {
    pub fn construct(config: FieldArithConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn config(&self) -> &FieldArithConfig {
        &self.config
    }

    /// Configure the chip's columns and gates.
    ///
    /// Gates:
    /// - s_mul: a * b == c * S  (scaled multiply: c = a*b/S)
    /// - s_div: c * b == a * S  (scaled divide: c = a*S/b)
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
    ) -> FieldArithConfig {
        let s_mul = meta.selector();
        let s_div = meta.selector();
        let s_add = meta.selector();
        let s_sub = meta.selector();

        let constant = meta.fixed_column();
        meta.enable_constant(constant);

        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(c);

        let scale = Expression::Constant(F::from_u128(SCALE));

        // Scaled multiply gate: a * b == c * S
        // Equivalently: a * b - c * S == 0
        meta.create_gate("scaled_mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a_val = meta.query_advice(a, Rotation::cur());
            let b_val = meta.query_advice(b, Rotation::cur());
            let c_val = meta.query_advice(c, Rotation::cur());
            vec![s * (a_val * b_val - c_val * scale.clone())]
        });

        // Scaled divide gate: c * b == a * S
        // Equivalently: c * b - a * S == 0
        meta.create_gate("scaled_div", |meta| {
            let s = meta.query_selector(s_div);
            let a_val = meta.query_advice(a, Rotation::cur());
            let b_val = meta.query_advice(b, Rotation::cur());
            let c_val = meta.query_advice(c, Rotation::cur());
            vec![s * (c_val * b_val - a_val * scale.clone())]
        });

        // Add gate: a + b - c == 0
        meta.create_gate("add", |meta| {
            let s = meta.query_selector(s_add);
            let a_val = meta.query_advice(a, Rotation::cur());
            let b_val = meta.query_advice(b, Rotation::cur());
            let c_val = meta.query_advice(c, Rotation::cur());
            vec![s * (a_val + b_val - c_val)]
        });

        // Sub gate: a - b - c == 0
        meta.create_gate("sub", |meta| {
            let s = meta.query_selector(s_sub);
            let a_val = meta.query_advice(a, Rotation::cur());
            let b_val = meta.query_advice(b, Rotation::cur());
            let c_val = meta.query_advice(c, Rotation::cur());
            vec![s * (a_val - b_val - c_val)]
        });

        FieldArithConfig {
            a,
            b,
            c,
            constant,
            s_mul,
            s_div,
            s_add,
            s_sub,
        }
    }

    /// Assign a scaled multiplication: c = a * b / S
    /// The prover provides the witness value; the gate constrains a*b == c*S.
    pub fn scaled_mul(
        &self,
        mut layouter: impl Layouter<F>,
        a: &AssignedCell<F, F>,
        b: &AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        layouter.assign_region(
            || "scaled_mul",
            |mut region| {
                self.config.s_mul.enable(&mut region, 0)?;

                a.copy_advice(|| "a", &mut region, self.config.a, 0)?;
                b.copy_advice(|| "b", &mut region, self.config.b, 0)?;

                let c_val = a.value().zip(b.value()).map(|(a_v, b_v)| {
                    // c = a * b * S^{-1} in the field
                    let s_inv = F::from_u128(SCALE).invert().unwrap();
                    *a_v * *b_v * s_inv
                });

                region.assign_advice(|| "c = a*b/S", self.config.c, 0, || c_val)
            },
        )
    }

    /// Assign a scaled division: c = a * S / b
    /// The gate constrains c * b == a * S.
    pub fn scaled_div(
        &self,
        mut layouter: impl Layouter<F>,
        a: &AssignedCell<F, F>,
        b: &AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        layouter.assign_region(
            || "scaled_div",
            |mut region| {
                self.config.s_div.enable(&mut region, 0)?;

                a.copy_advice(|| "a", &mut region, self.config.a, 0)?;
                b.copy_advice(|| "b", &mut region, self.config.b, 0)?;

                let c_val = a.value().zip(b.value()).map(|(a_v, b_v)| {
                    // c = a * S / b = a * S * b^{-1}
                    let s = F::from_u128(SCALE);
                    let b_inv = b_v.invert().unwrap();
                    *a_v * s * b_inv
                });

                region.assign_advice(|| "c = a*S/b", self.config.c, 0, || c_val)
            },
        )
    }

    /// Assign a scaled power: base^exp with descaling at each step.
    pub fn scaled_pow(
        &self,
        mut layouter: impl Layouter<F>,
        base: &AssignedCell<F, F>,
        exp: u32,
    ) -> Result<AssignedCell<F, F>, Error> {
        if exp == 0 {
            return self.load_constant(
                layouter.namespace(|| "pow_0"),
                F::from_u128(SCALE),
            );
        }

        let mut result = base.clone();
        for i in 1..exp {
            result = self.scaled_mul(
                layouter.namespace(|| format!("pow_step_{}", i)),
                &result,
                base,
            )?;
        }
        Ok(result)
    }

    /// Load a private witness value into an advice cell.
    pub fn load_private(
        &self,
        mut layouter: impl Layouter<F>,
        value: Value<F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        layouter.assign_region(
            || "load_private",
            |mut region| {
                region.assign_advice(|| "private", self.config.a, 0, || value)
            },
        )
    }

    /// Load a constant value into an advice cell, constrained by a fixed column.
    pub fn load_constant(
        &self,
        mut layouter: impl Layouter<F>,
        value: F,
    ) -> Result<AssignedCell<F, F>, Error> {
        layouter.assign_region(
            || "load_constant",
            |mut region| {
                region.assign_advice_from_constant(|| "constant", self.config.a, 0, value)
            },
        )
    }

    /// Add two values: c = a + b (no scaling needed)
    /// Constrained by s_add gate: a + b - c == 0
    pub fn add(
        &self,
        mut layouter: impl Layouter<F>,
        a: &AssignedCell<F, F>,
        b: &AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        layouter.assign_region(
            || "add",
            |mut region| {
                self.config.s_add.enable(&mut region, 0)?;

                a.copy_advice(|| "a", &mut region, self.config.a, 0)?;
                b.copy_advice(|| "b", &mut region, self.config.b, 0)?;

                let c_val = a.value().zip(b.value()).map(|(a_v, b_v)| *a_v + *b_v);
                region.assign_advice(|| "a+b", self.config.c, 0, || c_val)
            },
        )
    }

    /// Subtract: c = a - b (no scaling needed)
    /// Constrained by s_sub gate: a - b - c == 0
    pub fn sub(
        &self,
        mut layouter: impl Layouter<F>,
        a: &AssignedCell<F, F>,
        b: &AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        layouter.assign_region(
            || "sub",
            |mut region| {
                self.config.s_sub.enable(&mut region, 0)?;

                a.copy_advice(|| "a", &mut region, self.config.a, 0)?;
                b.copy_advice(|| "b", &mut region, self.config.b, 0)?;

                let c_val = a.value().zip(b.value()).map(|(a_v, b_v)| *a_v - *b_v);
                region.assign_advice(|| "a-b", self.config.c, 0, || c_val)
            },
        )
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_utils;
    use halo2_proofs::{
        circuit::SimpleFloorPlanner,
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::Circuit,
    };

    /// Convert a scaled i128 to a field element.
    fn scaled_to_field(v: i128) -> Fr {
        if v >= 0 {
            Fr::from_u128(v as u128)
        } else {
            -Fr::from_u128((-v) as u128)
        }
    }

    // ====================================================================
    // Test circuit for scaled multiplication
    // ====================================================================

    #[derive(Clone)]
    struct MulTestCircuit {
        a: Value<Fr>,
        b: Value<Fr>,
    }

    impl Circuit<Fr> for MulTestCircuit {
        type Config = FieldArithConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                a: Value::unknown(),
                b: Value::unknown(),
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
            let a = chip.load_private(layouter.namespace(|| "a"), self.a)?;
            let b = chip.load_private(layouter.namespace(|| "b"), self.b)?;
            let _c = chip.scaled_mul(layouter.namespace(|| "mul"), &a, &b)?;
            Ok(())
        }
    }

    #[test]
    fn test_scaled_mul_circuit() {
        let k = 4;
        let a_scaled = field_utils::float_to_scaled(2.0);
        let b_scaled = field_utils::float_to_scaled(3.0);

        let circuit = MulTestCircuit {
            a: Value::known(scaled_to_field(a_scaled)),
            b: Value::known(scaled_to_field(b_scaled)),
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_scaled_mul_small_values() {
        let k = 4;
        let a_scaled = field_utils::float_to_scaled(0.001);
        let b_scaled = field_utils::float_to_scaled(0.002);

        let circuit = MulTestCircuit {
            a: Value::known(scaled_to_field(a_scaled)),
            b: Value::known(scaled_to_field(b_scaled)),
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_scaled_mul_physics_values() {
        let k = 4;
        let a_scaled = field_utils::float_to_scaled(101325.0);
        let b_scaled = field_utils::float_to_scaled(0.2);

        let circuit = MulTestCircuit {
            a: Value::known(scaled_to_field(a_scaled)),
            b: Value::known(scaled_to_field(b_scaled)),
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    // ====================================================================
    // Test circuit for scaled division
    // ====================================================================

    #[derive(Clone)]
    struct DivTestCircuit {
        a: Value<Fr>,
        b: Value<Fr>,
    }

    impl Circuit<Fr> for DivTestCircuit {
        type Config = FieldArithConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                a: Value::unknown(),
                b: Value::unknown(),
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
            let a = chip.load_private(layouter.namespace(|| "a"), self.a)?;
            let b = chip.load_private(layouter.namespace(|| "b"), self.b)?;
            let _c = chip.scaled_div(layouter.namespace(|| "div"), &a, &b)?;
            Ok(())
        }
    }

    #[test]
    fn test_scaled_div_circuit() {
        let k = 4;
        let a_scaled = field_utils::float_to_scaled(10.0);
        let b_scaled = field_utils::float_to_scaled(3.0);

        let circuit = DivTestCircuit {
            a: Value::known(scaled_to_field(a_scaled)),
            b: Value::known(scaled_to_field(b_scaled)),
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_scaled_div_exact() {
        let k = 4;
        let a_scaled = field_utils::float_to_scaled(6.0);
        let b_scaled = field_utils::float_to_scaled(2.0);

        let circuit = DivTestCircuit {
            a: Value::known(scaled_to_field(a_scaled)),
            b: Value::known(scaled_to_field(b_scaled)),
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    // ====================================================================
    // Test circuit for scaled power
    // ====================================================================

    #[derive(Clone)]
    struct PowTestCircuit {
        base: Value<Fr>,
        exp: u32,
    }

    impl Circuit<Fr> for PowTestCircuit {
        type Config = FieldArithConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                base: Value::unknown(),
                exp: self.exp,
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
            let base = chip.load_private(layouter.namespace(|| "base"), self.base)?;
            let _result = chip.scaled_pow(layouter.namespace(|| "pow"), &base, self.exp)?;
            Ok(())
        }
    }

    #[test]
    fn test_scaled_pow_square() {
        let k = 4;
        let base_scaled = field_utils::float_to_scaled(3.0);

        let circuit = PowTestCircuit {
            base: Value::known(scaled_to_field(base_scaled)),
            exp: 2,
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_scaled_pow_fifth() {
        let k = 5;
        let base_scaled = field_utils::float_to_scaled(2.0);

        let circuit = PowTestCircuit {
            base: Value::known(scaled_to_field(base_scaled)),
            exp: 5,
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_scaled_pow_zero() {
        let k = 4;
        let base_scaled = field_utils::float_to_scaled(42.0);

        let circuit = PowTestCircuit {
            base: Value::known(scaled_to_field(base_scaled)),
            exp: 0,
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    // ====================================================================
    // Test circuit for add/sub (constrained gates)
    // ====================================================================

    #[derive(Clone)]
    struct AddSubTestCircuit {
        a: Value<Fr>,
        b: Value<Fr>,
    }

    impl Circuit<Fr> for AddSubTestCircuit {
        type Config = FieldArithConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                a: Value::unknown(),
                b: Value::unknown(),
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
            let a = chip.load_private(layouter.namespace(|| "a"), self.a)?;
            let b = chip.load_private(layouter.namespace(|| "b"), self.b)?;
            let _sum = chip.add(layouter.namespace(|| "add"), &a, &b)?;
            let _diff = chip.sub(layouter.namespace(|| "sub"), &a, &b)?;
            Ok(())
        }
    }

    #[test]
    fn test_add_sub_circuit() {
        let k = 4;
        let a_scaled = field_utils::float_to_scaled(10.0);
        let b_scaled = field_utils::float_to_scaled(3.0);

        let circuit = AddSubTestCircuit {
            a: Value::known(scaled_to_field(a_scaled)),
            b: Value::known(scaled_to_field(b_scaled)),
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    /// Negative test: a malicious prover who provides wrong sum should be rejected.
    #[derive(Clone)]
    struct BadAddCircuit {
        a: Value<Fr>,
        b: Value<Fr>,
        bad_sum: Value<Fr>,
    }

    impl Circuit<Fr> for BadAddCircuit {
        type Config = FieldArithConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                a: Value::unknown(),
                b: Value::unknown(),
                bad_sum: Value::unknown(),
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
            let chip = FieldArithChip::<Fr>::construct(config.clone());
            let a = chip.load_private(layouter.namespace(|| "a"), self.a)?;
            let b = chip.load_private(layouter.namespace(|| "b"), self.b)?;

            // Manually assign a wrong sum with the add gate enabled
            layouter.assign_region(
                || "bad_add",
                |mut region| {
                    config.s_add.enable(&mut region, 0)?;
                    a.copy_advice(|| "a", &mut region, config.a, 0)?;
                    b.copy_advice(|| "b", &mut region, config.b, 0)?;
                    region.assign_advice(|| "bad_sum", config.c, 0, || self.bad_sum)
                },
            )?;
            Ok(())
        }
    }

    #[test]
    fn test_bad_add_rejected() {
        let k = 4;
        let a = Fr::from_u128(field_utils::float_to_scaled(10.0) as u128);
        let b = Fr::from_u128(field_utils::float_to_scaled(3.0) as u128);
        let wrong = Fr::from(99u64); // Clearly not a + b

        let circuit = BadAddCircuit {
            a: Value::known(a),
            b: Value::known(b),
            bad_sum: Value::known(wrong),
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        assert!(
            prover.verify().is_err(),
            "MockProver should reject a wrong sum"
        );
    }

    // ====================================================================
    // Verify Rust field_utils matches expected values
    // ====================================================================

    #[test]
    fn test_field_mul_matches_expected() {
        let a = 101325.0_f64;
        let b = 0.2_f64;
        let a_s = field_utils::float_to_scaled(a);
        let b_s = field_utils::float_to_scaled(b);
        let c_s = field_utils::scaled_mul(a_s, b_s);
        let result = field_utils::scaled_to_float(c_s);
        assert!(
            (result - a * b).abs() < 1e-6,
            "scaled_mul diverges: {} vs {}",
            result,
            a * b
        );
    }

    #[test]
    fn test_field_div_matches_expected() {
        let a = 5.0e-6_f64;
        let b = 25.0e-6_f64;
        let a_s = field_utils::float_to_scaled(a);
        let b_s = field_utils::float_to_scaled(b);
        let c_s = field_utils::scaled_div(a_s, b_s);
        let result = field_utils::scaled_to_float(c_s);
        assert!(
            (result - a / b).abs() < 1e-10,
            "scaled_div diverges: {} vs {}",
            result,
            a / b
        );
    }

    #[test]
    fn test_field_pow_matches_expected() {
        let base = 0.2_f64;
        let base_s = field_utils::float_to_scaled(base);
        let result_s = field_utils::scaled_pow(base_s, 5);
        let result = field_utils::scaled_to_float(result_s);
        assert!(
            (result - base.powi(5)).abs() < 1e-10,
            "scaled_pow diverges: {} vs {}",
            result,
            base.powi(5)
        );
    }
}
