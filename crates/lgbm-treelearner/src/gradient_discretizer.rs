//! `gradient_discretizer` — the numeric core of the opt-in `use_quantized_grad`
//! APPROXIMATE training mode (phase-10, Wave 1).
//!
//! Verbatim port of the DETERMINISTIC path of `LightGBM/src/treelearner/
//! gradient_discretizer.cpp` (`DiscretizeGradients`, `stochastic_rounding=false`):
//! gradients/hessians are quantized to `i8` via a per-iteration scale derived from the
//! max-abs value, accumulated as integers, and de-quantized (× scale) for split-finding.
//!
//! **This is APPROXIMATE by construction** (spike-008: even full int16 drifts ~3e-4 — far
//! above the exact ~1e-6 contract). It is reached ONLY when `use_quantized_grad=true`; the
//! default exact path never touches this code. The parity target is C++
//! `use_quantized_grad=true, stochastic_rounding=false`, NOT the f64 exact anchor.
//!
//! Stochastic rounding + `quant_train_renew_leaf` are deferred (Wave 6) — they need
//! RNG-matching, a separate parity problem. This module is deterministic-only.

/// Quantization scales + the deterministic quantize/de-quantize math.
///
/// Layout of the discretized buffer mirrors C++ exactly: pairs `[hess_i, grad_i]`, i.e.
/// `out[2*i] = quantized hessian`, `out[2*i + 1] = quantized gradient` (`gradient_discretizer.cpp:124-157`).
#[derive(Debug, Clone)]
pub struct GradientDiscretizer {
    num_grad_quant_bins: i32,
    is_constant_hessian: bool,
    // Set per iteration by `discretize`.
    grad_scale: f64,
    hess_scale: f64,
    inv_grad_scale: f64,
    inv_hess_scale: f64,
    max_grad_abs: f64,
    max_hess_abs: f64,
}

impl GradientDiscretizer {
    /// `num_grad_quant_bins` and the constant-hessian flag are fixed for the run
    /// (`use_quantized_grad` config). Scales are computed per iteration in [`Self::discretize`].
    #[must_use]
    pub fn new(num_grad_quant_bins: i32, is_constant_hessian: bool) -> Self {
        Self {
            num_grad_quant_bins,
            is_constant_hessian,
            grad_scale: 0.0,
            hess_scale: 0.0,
            inv_grad_scale: 0.0,
            inv_hess_scale: 0.0,
            max_grad_abs: 0.0,
            max_hess_abs: 0.0,
        }
    }

    #[must_use]
    pub fn grad_scale(&self) -> f64 {
        self.grad_scale
    }
    #[must_use]
    pub fn hess_scale(&self) -> f64 {
        self.hess_scale
    }

    /// Compute this iteration's scales from the max-abs gradient/hessian, then quantize
    /// every row to an `i8` pair `[hess_i, grad_i]`. Deterministic round-half-away-from-zero,
    /// verbatim from C++ (`grad>=0 ? g*inv+0.5 : g*inv-0.5`, then truncate-toward-zero).
    ///
    /// Returns the discretized buffer of length `2 * grad.len()`. `grad`/`hess` are the f32
    /// `score_t` objective outputs (read in f64, matching the C++ `double` promotion).
    ///
    /// # Panics
    /// Never panics on empty input — returns an empty buffer (scales stay 0).
    pub fn discretize(&mut self, grad: &[f32], hess: &[f32]) -> Vec<i8> {
        debug_assert_eq!(grad.len(), hess.len(), "grad/hess length mismatch");
        let n = grad.len();
        if n == 0 {
            return Vec::new();
        }

        // max-abs over the iteration (f64 fabs, matching `DiscretizeGradients:72-99`).
        let mut max_grad = f64::from(grad[0]).abs();
        let mut max_hess = f64::from(hess[0]).abs();
        for i in 0..n {
            max_grad = max_grad.max(f64::from(grad[i]).abs());
            max_hess = max_hess.max(f64::from(hess[i]).abs());
        }
        self.max_grad_abs = max_grad;
        self.max_hess_abs = max_hess;

        // Scales (`:107-114`): grad uses bins/2 (INTEGER division, as in C++), hess uses
        // the full bins unless constant (then hess_scale = max_hess, quantized hess ≡ 1).
        let half_bins = (self.num_grad_quant_bins / 2) as f64; // integer div then widen
        self.grad_scale = max_grad / half_bins;
        self.hess_scale = if self.is_constant_hessian {
            max_hess
        } else {
            max_hess / f64::from(self.num_grad_quant_bins)
        };
        self.inv_grad_scale = 1.0 / self.grad_scale;
        self.inv_hess_scale = 1.0 / self.hess_scale;

        let mut out = vec![0i8; 2 * n];
        for i in 0..n {
            out[2 * i + 1] = quantize_signed(f64::from(grad[i]), self.inv_grad_scale);
            out[2 * i] = if self.is_constant_hessian {
                1
            } else {
                // hessian is non-negative; C++ always adds +0.5 (no sign branch).
                (f64::from(hess[i]) * self.inv_hess_scale + 0.5) as i8
            };
        }
        out
    }

    /// De-quantize an integer (grad-sum, hess-sum) bin cell back to f64 — `int_sum * scale`,
    /// the value split-finding consumes (C++ multiplies the int histogram sums by the scales).
    #[must_use]
    pub fn dequantize_grad(&self, int_grad_sum: i64) -> f64 {
        int_grad_sum as f64 * self.grad_scale
    }
    #[must_use]
    pub fn dequantize_hess(&self, int_hess_sum: i64) -> f64 {
        int_hess_sum as f64 * self.hess_scale
    }
}

/// Sign-aware deterministic quantization (`gradient_discretizer.cpp:145-147`):
/// `g >= 0 ? trunc(g*inv + 0.5) : trunc(g*inv - 0.5)`. Rust `as i8` truncates toward zero
/// for in-range values (matching C++ `static_cast<int8_t>`) and saturates out-of-range
/// (C++ is UB there; valid `num_grad_quant_bins` keep values in range).
#[inline]
fn quantize_signed(value: f64, inv_scale: f64) -> i8 {
    let scaled = value * inv_scale;
    let biased = if value >= 0.0 { scaled + 0.5 } else { scaled - 0.5 };
    biased as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed: grads on/off the quantization grid, constant hessian, bins=4.
    /// max|g|=1 → grad_scale = 1/(4/2) = 0.5, inv = 2.0.
    ///   1.0 → 1.0*2+0.5=2.5 → 2 ;  -1.0 → -1.0*2-0.5=-2.5 → -2
    ///   0.5 → 0.5*2+0.5=1.5 → 1 ;  -0.5 → -0.5*2-0.5=-1.5 → -1
    ///   0.3 → 0.3*2+0.5=1.1 → 1 ;  -0.3 → -0.3*2-0.5=-1.1 → -1
    #[test]
    fn deterministic_quantize_constant_hessian_bins4() {
        let grad = [1.0f32, -1.0, 0.5, -0.5, 0.3, -0.3];
        let hess = [1.0f32; 6];
        let mut d = GradientDiscretizer::new(4, true);
        let q = d.discretize(&grad, &hess);
        assert_eq!(d.grad_scale(), 0.5);
        // pairs [hess, grad]; hess ≡ 1 (constant).
        let grads_q: Vec<i8> = (0..6).map(|i| q[2 * i + 1]).collect();
        let hess_q: Vec<i8> = (0..6).map(|i| q[2 * i]).collect();
        assert_eq!(grads_q, vec![2, -2, 1, -1, 1, -1]);
        assert_eq!(hess_q, vec![1, 1, 1, 1, 1, 1]);
    }

    /// Round-half-away-from-zero boundary: ±1.5 (in grid units) round to ±2.
    #[test]
    fn half_rounds_away_from_zero() {
        // grad_scale 0.5, inv 2.0: 0.75*2=1.5 → +0.5 = 2.0 → 2 ; -0.75 → -2.0 → -2
        let grad = [0.75f32, -0.75, 1.0];
        let hess = [1.0f32; 3];
        let mut d = GradientDiscretizer::new(4, true);
        let q = d.discretize(&grad, &hess);
        assert_eq!(q[1], 2); // 0.75
        assert_eq!(q[3], -2); // -0.75
    }

    /// Non-constant hessian: hess uses the FULL bins (not bins/2). max|h|=0.25, bins=4 →
    /// hess_scale = 0.25/4 = 0.0625, inv = 16. h=0.25 → 0.25*16+0.5=4.5 → 4. h=0.125 → 2.5 → 2.
    #[test]
    fn nonconstant_hessian_uses_full_bins() {
        let grad = [0.5f32, -0.5];
        let hess = [0.25f32, 0.125];
        let mut d = GradientDiscretizer::new(4, false);
        let q = d.discretize(&grad, &hess);
        assert_eq!(d.hess_scale(), 0.0625);
        assert_eq!(q[0], 4); // hess 0.25
        assert_eq!(q[2], 2); // hess 0.125
    }

    /// De-quant inverts on-grid values exactly: int_sum × scale.
    #[test]
    fn dequant_roundtrips_on_grid() {
        let grad = [1.0f32, 0.5, -0.5];
        let hess = [1.0f32; 3];
        let mut d = GradientDiscretizer::new(4, true);
        let q = d.discretize(&grad, &hess);
        // sum the int grads of all 3 rows into one "bin": 2 + 1 + (-1) = 2 → ×0.5 = 1.0,
        // which equals the exact grad sum 1.0+0.5-0.5 = 1.0 (these happen to be on-grid).
        let int_sum: i64 = (0..3).map(|i| i64::from(q[2 * i + 1])).sum();
        assert_eq!(int_sum, 2);
        assert!((d.dequantize_grad(int_sum) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_input_is_safe() {
        let mut d = GradientDiscretizer::new(4, true);
        assert!(d.discretize(&[], &[]).is_empty());
    }
}
