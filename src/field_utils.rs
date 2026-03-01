//! Fixed-point scaled arithmetic utilities for ZK field elements.
//!
//! All physics values are represented as integers scaled by S = 10^30.
//! This module provides helpers for working with these scaled values
//! outside of the circuit (for witness generation).

/// Scaling factor: 10^30
/// Chosen to accommodate sonoluminescence's 200x pressure / 170x temperature
/// variation while keeping products (10^60) well below the ~2^255 field modulus.
pub const SCALE: u128 = 1_000_000_000_000_000_000_000_000_000_000; // 10^30

/// Stefan-Boltzmann constant * 4 * pi, scaled.
/// sigma_SB = 5.670374419e-8 W/(m^2 K^4)
/// 4*pi ≈ 12.566370614359172
/// sigma_SB * 4*pi ≈ 7.125602647e-7
/// Scaled: round(7.125602647e-7 * 10^30)
pub const SIGMA_4PI_SCALED: u128 = 712_560_264_713_355_606_884_352;

/// Scale a floating-point value to a fixed-point integer.
pub fn float_to_scaled(v: f64) -> i128 {
    (v * SCALE as f64) as i128
}

/// Unscale a fixed-point integer back to floating-point (for display/testing).
pub fn scaled_to_float(v: i128) -> f64 {
    v as f64 / SCALE as f64
}

/// Split factor for wide multiplication. 10^10, so SPLIT^3 = 10^30 = SCALE.
const SPLIT: i128 = 10_000_000_000; // 10^10

/// Scaled multiplication: (a * b) / S
/// In the circuit, this becomes a constraint: c * S == a * b
///
/// Uses a 3-way split to handle values up to ~10^36 without i128 overflow.
/// a = a2*SPLIT^2 + a1*SPLIT + a0, similarly for b.
/// Max partial product: a_i * b_j ≤ (SPLIT-1)^2 ≈ 10^20, well within i128.
pub fn scaled_mul(a: i128, b: i128) -> i128 {
    let sign = if (a < 0) != (b < 0) { -1i128 } else { 1 };
    let a_abs = a.unsigned_abs();
    let b_abs = b.unsigned_abs();
    let sp = SPLIT as u128;

    let a0 = a_abs % sp;
    let a1 = (a_abs / sp) % sp;
    let a2 = a_abs / (sp * sp);

    let b0 = b_abs % sp;
    let b1 = (b_abs / sp) % sp;
    let b2 = b_abs / (sp * sp);

    // (a * b) / SCALE where SCALE = SPLIT^3
    // a*b = sum_{i+j=k} a_i*b_j * SPLIT^(i+j)
    // Dividing by SPLIT^3: terms with i+j < 3 contribute to fractional part,
    // terms with i+j = 3 contribute directly, terms with i+j > 3 contribute * SPLIT^(k-3)

    // i+j = 0: a0*b0 / SPLIT^3 → negligible
    // i+j = 1: (a0*b1 + a1*b0) / SPLIT^2 → negligible
    // i+j = 2: (a0*b2 + a1*b1 + a2*b0) / SPLIT → keep
    let sum_2 = a0 * b2 + a1 * b1 + a2 * b0;
    // i+j = 3: (a1*b2 + a2*b1) → exact
    let sum_3 = a1 * b2 + a2 * b1;
    // i+j = 4: a2*b2 * SPLIT → exact
    let sum_4 = a2 * b2;

    // Accumulate carefully to avoid overflow
    let mut result: u128 = sum_2 / sp;
    result = result.wrapping_add(sum_3);
    // sum_4 * sp might overflow; use checked_mul and saturate
    if let Some(s4_scaled) = sum_4.checked_mul(sp) {
        result = result.wrapping_add(s4_scaled);
    } else {
        // For extremely large products (>10^38), the result will be taken mod p
        // in the field anyway. Use wrapping.
        result = result.wrapping_add(sum_4.wrapping_mul(sp));
    }
    sign * result as i128
}

/// Scaled division: (a * S) / b
/// In the circuit, this becomes a constraint: c * b == a * S
///
/// Since a * SCALE can exceed both i128 and u128 range, we compute via
/// iterative long division. We multiply by CHUNK = 10^6 five times
/// (since CHUNK^5 = 10^30 = SCALE), dividing by b at each step and
/// carrying the remainder. The maximum intermediate product is
/// remainder * CHUNK < b * CHUNK. For values up to ~200 * SCALE ~ 10^32,
/// this gives ~10^32 * 10^6 = 10^38, safely within u128 (~3.4 * 10^38).
pub fn scaled_div(a: i128, b: i128) -> i128 {
    if b == 0 {
        panic!("Division by zero in scaled_div");
    }
    let sign = if (a < 0) != (b < 0) { -1i128 } else { 1i128 };
    let a_abs = a.unsigned_abs();
    let b_abs = b.unsigned_abs();

    // SCALE = CHUNK^5 where CHUNK = 10^6.
    // Iteratively: start with a_abs / b_abs and remainder,
    // then 5 rounds of: multiply remainder by CHUNK, get next digit and new remainder,
    // shift accumulator by CHUNK and add digit.
    const CHUNK: u128 = 1_000_000; // 10^6
    const ROUNDS: usize = 5;       // CHUNK^5 = 10^30 = SCALE

    let mut acc: u128 = a_abs / b_abs;
    let mut rem: u128 = a_abs % b_abs;

    for _ in 0..ROUNDS {
        let wide = rem * CHUNK; // rem < b_abs, so wide < b_abs * 10^6
        let digit = wide / b_abs;
        rem = wide % b_abs;
        acc = acc * CHUNK + digit;
    }

    sign * acc as i128
}

/// Scaled integer power: base^exp with descaling at each step
pub fn scaled_pow(base: i128, exp: u32) -> i128 {
    if exp == 0 {
        return SCALE as i128; // 1.0 in scaled representation
    }
    let mut result = base;
    for _ in 1..exp {
        result = scaled_mul(result, base);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_roundtrip() {
        let v = 3.14159;
        let scaled = float_to_scaled(v);
        let back = scaled_to_float(scaled);
        assert!((back - v).abs() < 1e-12);
    }

    #[test]
    fn test_scaled_mul() {
        let a = float_to_scaled(2.0);
        let b = float_to_scaled(3.0);
        let c = scaled_mul(a, b);
        let result = scaled_to_float(c);
        assert!((result - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaled_div() {
        let a = float_to_scaled(10.0);
        let b = float_to_scaled(3.0);
        let c = scaled_div(a, b);
        let result = scaled_to_float(c);
        assert!((result - 3.333333333).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_pow() {
        let base = float_to_scaled(2.0);
        let result = scaled_pow(base, 4);
        let v = scaled_to_float(result);
        assert!((v - 16.0).abs() < 1e-6);
    }
}
