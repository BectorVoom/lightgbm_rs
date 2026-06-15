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
    /// `num_grad_quant_bins` MUST be ≤ 254: the discretized gradient is `i8` and maps to
    /// `±bins/2`, so `bins/2` must fit `i8::MAX` (127). C++ overflows silently (UB) and the
    /// model learns nothing — verified against LightGBM 4.6 (256 → constant model). The
    /// config default is 4.
    #[must_use]
    pub fn new(num_grad_quant_bins: i32, is_constant_hessian: bool) -> Self {
        debug_assert!(
            (1..=254).contains(&num_grad_quant_bins),
            "num_grad_quant_bins {num_grad_quant_bins} must be in 1..=254 (i8 discretized storage)"
        );
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

    /// De-quantize a whole integer histogram (`[g0,h0,g1,h1,…]`, even=grad/odd=hess) into the
    /// f64 stride-2 layout the existing split-finder + `fix_histogram` consume. Per-cell
    /// `int_sum × scale`. Because scaling distributes over the sum, this equals building an
    /// f64 histogram from per-row de-quantized values — so the quantized path slots straight
    /// into the existing downstream with only the build swapped.
    #[must_use]
    pub fn dequantize_histogram(&self, int_hist: &[i64]) -> Vec<f64> {
        int_hist
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i % 2 == 0 {
                    v as f64 * self.grad_scale
                } else {
                    v as f64 * self.hess_scale
                }
            })
            .collect()
    }
}

/// Build the integer histogram (exact i64 grad/hess sums per bin) from a quantized buffer
/// over `leaf_rows` — the quantized analog of `accumulate_histogram_into` (Wave 2).
///
/// `discretized` holds `[hess_i, grad_i]` i8 pairs (from [`GradientDiscretizer::discretize`]):
/// `discretized[2*row] = hess`, `discretized[2*row + 1] = grad`. `binned[row]` is the row's
/// bin. Output is interleaved `[g0,h0,g1,h1,…]` of length `2 * num_bin` (grad at `bin*2`, hess
/// at `bin*2 + 1` — the SAME layout as the exact f64 histogram, so de-quant feeds downstream
/// unchanged).
///
/// i64 accumulation is exact and overflow-free — it equals C++'s int32/int16 histogram VALUE
/// (the dynamic bit-width in `SetNumBitsInHistogramBin` is a storage/perf choice; the integer
/// sum is identical whenever the narrow path doesn't overflow, which C++ guarantees by gating).
#[must_use]
pub fn construct_int_histogram(
    binned: &[u32],
    discretized: &[i8],
    leaf_rows: &[u32],
    num_bin: u32,
) -> Vec<i64> {
    let mut out = vec![0i64; 2 * num_bin as usize];
    for &row in leaf_rows {
        let r = row as usize;
        let bin = binned[r] as usize;
        let ti = bin * 2;
        out[ti] += i64::from(discretized[2 * r + 1]); // grad
        out[ti + 1] += i64::from(discretized[2 * r]); // hess
    }
    out
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

    // ---- Wave 2: integer histogram ----

    /// Int histogram sums the quantized grads/hess per bin (grad at bin*2, hess at +1).
    /// grads [1,-1,0.5,-0.5] (constant hess) at bins=4 → ints [2,-2,1,-1], hess ≡1.
    /// Place rows in bins [0,0,1,1]: bin0 grad=2+(-2)=0 hess=2 ; bin1 grad=1+(-1)=0 hess=2.
    #[test]
    fn int_histogram_sums_per_bin() {
        let grad = [1.0f32, -1.0, 0.5, -0.5];
        let hess = [1.0f32; 4];
        let mut d = GradientDiscretizer::new(4, true);
        let q = d.discretize(&grad, &hess);
        let binned = [0u32, 0, 1, 1];
        let rows = [0u32, 1, 2, 3];
        let h = construct_int_histogram(&binned, &q, &rows, 2);
        assert_eq!(h, vec![0, 2, 0, 2]); // [g0,h0,g1,h1]
    }

    /// Self-consistency: de-quant(Σ int) == Σ (int × scale) built per-row — the property
    /// that lets the quantized path reuse the exact downstream (scaling distributes over Σ).
    #[test]
    fn dequant_histogram_distributes_over_sum() {
        let grad = [0.7f32, -0.3, 0.9, -0.8, 0.2];
        let hess = [0.1f32, 0.2, 0.15, 0.25, 0.05];
        let mut d = GradientDiscretizer::new(16, false);
        let q = d.discretize(&grad, &hess);
        let binned = [0u32, 1, 0, 1, 0];
        let rows = [0u32, 1, 2, 3, 4];
        let int_h = construct_int_histogram(&binned, &q, &rows, 2);
        let deq = d.dequantize_histogram(&int_h);

        // Per-row reference: de-quant each row's int grad/hess, accumulate in f64.
        let mut ref_h = vec![0.0f64; 4];
        for (k, &row) in rows.iter().enumerate() {
            let _ = k;
            let r = row as usize;
            let b = binned[r] as usize;
            ref_h[b * 2] += f64::from(q[2 * r + 1]) * d.grad_scale();
            ref_h[b * 2 + 1] += f64::from(q[2 * r]) * d.hess_scale();
        }
        for (a, b) in deq.iter().zip(ref_h.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    /// Histogram subtraction still works on int sums (parent − child = sibling), exactly —
    /// integers subtract without error, so the quantized path keeps the subtraction trick.
    #[test]
    fn int_histogram_subtraction_is_exact() {
        let grad: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.1).collect();
        let hess = vec![1.0f32; 20];
        let mut d = GradientDiscretizer::new(8, true);
        let q = d.discretize(&grad, &hess);
        let binned: Vec<u32> = (0..20).map(|i| (i % 3) as u32).collect();
        let all: Vec<u32> = (0..20).collect();
        let left: Vec<u32> = (0..20).filter(|i| i % 2 == 0).collect();
        let right: Vec<u32> = (0..20).filter(|i| i % 2 == 1).collect();
        let parent = construct_int_histogram(&binned, &q, &all, 3);
        let lh = construct_int_histogram(&binned, &q, &left, 3);
        let rh = construct_int_histogram(&binned, &q, &right, 3);
        for i in 0..parent.len() {
            assert_eq!(parent[i] - lh[i], rh[i], "subtraction mismatch at {i}");
        }
    }
}
