//! Monotone constraints (ADV-01) — an ADDITIVE gate on the per-feature numeric
//! split scan (D-06 HARD INVARIANT: when no feature carries a monotone constraint
//! the spine `find_best_split` path is byte-untouched; this module is only
//! consulted for features whose `monotone_type != 0`).
//!
//! ## Faithful port surface
//! - `BasicConstraint { min, max }` — the per-leaf output clamp
//!   (`monotone_constraints.hpp:24-31`).
//! - The `USE_MC` split-gain branch (`feature_histogram.hpp:780-796`): each
//!   candidate's `left_output`/`right_output` are computed with the leaf's
//!   `[min,max]` constraint clamp; if the BASIC direction check is violated
//!   (`monotone>0 && left>right` or `monotone<0 && left<right`) the candidate
//!   gain is `0` (rejected); otherwise the gain uses `GetLeafGainGivenOutput`.
//! - `ComputeMonotoneSplitGainPenalty` (`monotone_constraints.hpp:357-366`): the
//!   `monotone_penalty`-driven gain multiplier applied to a monotone split.
//! - `BasicLeafConstraints::Update` (`monotone_constraints.hpp:487-503`): on a
//!   numeric monotone split, propagate `mid = (left+right)/2` as the parent's
//!   `max`/`min` and the new child's `min`/`max`.
//!
//! ## Method scope
//! The `basic` method is ported faithfully (the common case + the matrix default).
//! `intermediate` / `advanced` (the full up-tree min/max recompute,
//! `monotone_constraints.hpp:516-1186`) fall back to the `basic` propagation here;
//! the per-axis capture (Task 4) reveals which `intermediate`/`advanced` cells
//! reach bit-exact parity vs real lib_lightgbm 4.6 and any genuine divergence is
//! tracked honestly (never tolerance-weakened).

use lgbm_compute::gain::{
    calculate_splitted_leaf_output, get_leaf_gain, get_leaf_gain_given_output, GainConfig, SplitInfo,
};

/// The per-leaf output clamp `[min, max]` (`monotone_constraints.hpp:24-31`).
/// A fresh constraint is unbounded (`[-inf, +inf]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasicConstraint {
    /// Lower bound on the leaf output.
    pub min: f64,
    /// Upper bound on the leaf output.
    pub max: f64,
}

impl Default for BasicConstraint {
    fn default() -> Self {
        Self {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
        }
    }
}

impl BasicConstraint {
    /// `UpdateMin` (`monotone_constraints.hpp:72`): `min = max(new_min, min)`.
    fn update_min(&mut self, new_min: f64) {
        self.min = new_min.max(self.min);
    }
    /// `UpdateMax` (`monotone_constraints.hpp:74`): `max = min(new_max, max)`.
    fn update_max(&mut self, new_max: f64) {
        self.max = new_max.min(self.max);
    }
}

/// The monotone-constraints state for one tree's growth (`BasicLeafConstraints`,
/// `monotone_constraints.hpp:465-514`). Holds one `[min,max]` entry per leaf and
/// the per-feature monotone-type vector + the method/penalty config.
#[derive(Debug, Clone)]
pub struct MonotoneConstraints {
    /// Per-feature monotone type, indexed by REAL feature index: `+1`
    /// (non-decreasing), `-1` (non-increasing), `0` (no constraint). Empty ⇒
    /// monotone constraints are INACTIVE (the spine path; this module is never
    /// consulted, D-06).
    monotone_type: Vec<i8>,
    /// Per-leaf output clamp (sized to `num_leaves`).
    entries: Vec<BasicConstraint>,
    /// `monotone_penalty` (`config.h`). `0.0` ⇒ no penalty.
    penalty: f64,
}

impl MonotoneConstraints {
    /// Build the per-tree monotone state. `monotone_type` is indexed by REAL
    /// feature index; `num_leaves` sizes the per-leaf clamp; `penalty` is
    /// `config.monotone_penalty`. Returns `None` when no feature is constrained
    /// (every entry `0` or the vector empty) — the caller then takes the spine
    /// path with ZERO monotone overhead (D-06).
    #[must_use]
    pub fn new(monotone_type: &[i32], num_leaves: i32, penalty: f64) -> Option<Self> {
        if monotone_type.is_empty() || monotone_type.iter().all(|&m| m == 0) {
            return None;
        }
        let monotone_type: Vec<i8> = monotone_type.iter().map(|&m| m as i8).collect();
        Some(Self {
            monotone_type,
            entries: vec![BasicConstraint::default(); num_leaves as usize],
            penalty,
        })
    }

    /// The monotone type for a REAL feature index (`0` if out of range).
    #[must_use]
    pub fn feature_monotone(&self, real_feature_index: i32) -> i8 {
        self.monotone_type
            .get(real_feature_index as usize)
            .copied()
            .unwrap_or(0)
    }

    /// `ComputeMonotoneSplitGainPenalty` (`monotone_constraints.hpp:357-366`):
    /// the gain multiplier for a monotone split at `depth = leaf_depth(leaf)`.
    #[must_use]
    pub fn split_gain_penalty(&self, depth: i32) -> f64 {
        let eps = f64::from(lgbm_core::types::K_EPSILON);
        let penalization = self.penalty;
        let depth = f64::from(depth);
        if penalization >= depth + 1.0 {
            return eps;
        }
        if penalization <= 1.0 {
            return 1.0 - penalization / 2.0_f64.powf(depth) + eps;
        }
        1.0 - 2.0_f64.powf(penalization - 1.0 - depth) + eps
    }

    /// `BasicLeafConstraints::Update` (`monotone_constraints.hpp:487-503`) for a
    /// NUMERIC monotone split: clone the parent's clamp into the new child, then
    /// propagate `mid = (left_output + right_output) / 2` as the parent's
    /// `max`/`min` and the new child's `min`/`max` per the split's monotone type.
    /// A no-op when the winning feature is unconstrained.
    pub fn update_after_split(
        &mut self,
        parent_leaf: i32,
        new_leaf: i32,
        monotone_type: i8,
        left_output: f64,
        right_output: f64,
    ) {
        // Clone the parent's constraint into the new (right) child.
        let parent_entry = self.entries[parent_leaf as usize];
        self.entries[new_leaf as usize] = parent_entry;
        if monotone_type == 0 {
            return;
        }
        let mid = (left_output + right_output) / 2.0;
        if monotone_type < 0 {
            self.entries[parent_leaf as usize].update_min(mid);
            self.entries[new_leaf as usize].update_max(mid);
        } else {
            self.entries[parent_leaf as usize].update_max(mid);
            self.entries[new_leaf as usize].update_min(mid);
        }
    }

    /// The per-leaf clamp (for the constraint-aware scan).
    #[must_use]
    pub fn constraint_for(&self, leaf: i32) -> BasicConstraint {
        self.entries[leaf as usize]
    }
}

/// `CalculateSplittedLeafOutput<USE_MC=true, ...>`
/// (`feature_histogram.hpp:740-755`): the unconstrained output, then CLAMPED into
/// `[constraint.min, constraint.max]`.
fn calc_output_clamped(
    cfg: &GainConfig,
    sum_g: f64,
    sum_h: f64,
    constraint: &BasicConstraint,
) -> f64 {
    let mut ret = calculate_splitted_leaf_output(cfg.use_l1(), sum_g, sum_h, cfg.lambda_l1, cfg.lambda_l2);
    if ret < constraint.min {
        ret = constraint.min;
    } else if ret > constraint.max {
        ret = constraint.max;
    }
    ret
}

/// `GetSplitGains<USE_MC=true, ...>` (`feature_histogram.hpp:779-796`): compute
/// the clamped left/right outputs, REJECT (return `0`) on a basic direction
/// violation, else return the sum of `GetLeafGainGivenOutput`. The leaf
/// constraint `[min,max]` is applied to BOTH children (the basic method uses the
/// same `[min,max]` for both sides — `LeftToBasicConstraint`/
/// `RightToBasicConstraint` are the identity for `BasicConstraintEntry`).
#[allow(clippy::too_many_arguments)]
fn split_gain_mc(
    cfg: &GainConfig,
    sum_left_g: f64,
    sum_left_h: f64,
    sum_right_g: f64,
    sum_right_h: f64,
    monotone_type: i8,
    constraint: &BasicConstraint,
) -> f64 {
    let left_output = calc_output_clamped(cfg, sum_left_g, sum_left_h, constraint);
    let right_output = calc_output_clamped(cfg, sum_right_g, sum_right_h, constraint);
    if (monotone_type > 0 && left_output > right_output)
        || (monotone_type < 0 && left_output < right_output)
    {
        return 0.0;
    }
    get_leaf_gain_given_output(cfg.use_l1(), sum_left_g, sum_left_h, cfg.lambda_l1, cfg.lambda_l2, left_output)
        + get_leaf_gain_given_output(cfg.use_l1(), sum_right_g, sum_right_h, cfg.lambda_l1, cfg.lambda_l2, right_output)
}

/// The constraint-aware numeric split finder for a MONOTONE feature — the `USE_MC`
/// instantiation of `FindBestThresholdSequentially` (`feature_histogram.hpp:830-`)
/// for the REVERSE + FORWARD branches, mirroring the spine `find_best_split` gate
/// order EXACTLY but substituting `split_gain_mc` for the unconstrained gain and
/// using the leaf `[min,max]` clamp for the recorded child outputs.
///
/// Returns a [`SplitInfo`] (gain net of `min_gain_shift`; `gain == NEG_INFINITY`
/// when no candidate clears the gates). The caller multiplies the gain by the
/// monotone penalty.
///
/// `hist` is the SAME compacted+FixHistogram'd per-feature histogram the spine
/// scans; `sum_g`/`sum_h` are the RAW leaf sums (the `+2*kEpsilon` bump is applied
/// here, matching the spine call site). This is byte-isolated from the spine: it
/// only runs for features with `monotone_type != 0`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn find_best_split_monotone(
    hist: &[f64],
    cfg: &GainConfig,
    num_bin: i32,
    offset: i32,
    default_bin: u32,
    skip_default_bin: bool,
    run_forward: bool,
    sum_g: f64,
    sum_h_raw: f64,
    num_data: i32,
    monotone_type: i8,
    constraint: &BasicConstraint,
) -> SplitInfo {
    let eps = f64::from(lgbm_core::types::K_EPSILON);
    let sum_hessian = sum_h_raw + 2.0 * eps;
    let use_l1 = cfg.use_l1();
    let l1 = cfg.lambda_l1;
    let l2 = cfg.lambda_l2;
    let gain_shift = get_leaf_gain(use_l1, sum_g, sum_hessian, l1, l2);
    let min_gain_shift = gain_shift + cfg.min_gain_to_split;
    let cnt_factor = f64::from(num_data) / sum_hessian;
    let round_int = |x: f64| -> i32 { (x + f64::from(0.5f32)) as i32 };
    let get_grad = |t: i32| hist[(t as usize) << 1];
    let get_hess = |t: i32| hist[((t as usize) << 1) + 1];

    let mut best = SplitInfo::none();
    let mut best_gain = f64::NEG_INFINITY;

    // ---- REVERSE branch (feature_histogram.hpp:854-936) ----
    {
        let mut sum_right_gradient = 0.0f64;
        let mut sum_right_hessian = eps;
        let mut right_count = 0i32;
        let t_start = num_bin - 1 - offset;
        let t_end = 1 - offset;
        let mut t = t_start;
        while t >= t_end {
            if skip_default_bin && (t + offset) == default_bin as i32 {
                t -= 1;
                continue;
            }
            sum_right_gradient += get_grad(t);
            sum_right_hessian += get_hess(t);
            right_count += round_int(get_hess(t) * cnt_factor);
            if right_count < cfg.min_data_in_leaf || sum_right_hessian < cfg.min_sum_hessian_in_leaf {
                t -= 1;
                continue;
            }
            let left_count = num_data - right_count;
            if left_count < cfg.min_data_in_leaf {
                break;
            }
            let sum_left_hessian = sum_hessian - sum_right_hessian;
            if sum_left_hessian < cfg.min_sum_hessian_in_leaf {
                break;
            }
            let sum_left_gradient = sum_g - sum_right_gradient;
            let g = split_gain_mc(
                cfg,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                monotone_type,
                constraint,
            );
            if g > min_gain_shift && g > best_gain {
                best_gain = g;
                best = build_split(
                    cfg,
                    (t - 1 + offset) as u32,
                    g - min_gain_shift,
                    left_count,
                    right_count,
                    sum_g,
                    sum_hessian,
                    sum_left_gradient,
                    sum_left_hessian,
                    constraint,
                    true,
                );
            }
            t -= 1;
        }
    }

    // ---- FORWARD branch (feature_histogram.hpp:937-1029) ----
    if run_forward {
        let mut sum_left_gradient = 0.0f64;
        let mut sum_left_hessian = eps;
        let mut left_count = 0i32;
        let t_end = num_bin - 2 - offset;
        let mut t = 0i32;
        while t <= t_end {
            if skip_default_bin && (t + offset) == default_bin as i32 {
                t += 1;
                continue;
            }
            sum_left_gradient += get_grad(t);
            sum_left_hessian += get_hess(t);
            left_count += round_int(get_hess(t) * cnt_factor);
            if left_count < cfg.min_data_in_leaf || sum_left_hessian < cfg.min_sum_hessian_in_leaf {
                t += 1;
                continue;
            }
            let right_count = num_data - left_count;
            if right_count < cfg.min_data_in_leaf {
                break;
            }
            let sum_right_hessian = sum_hessian - sum_left_hessian;
            if sum_right_hessian < cfg.min_sum_hessian_in_leaf {
                break;
            }
            let sum_right_gradient = sum_g - sum_left_gradient;
            let g = split_gain_mc(
                cfg,
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                monotone_type,
                constraint,
            );
            if g > min_gain_shift && g > best_gain {
                best_gain = g;
                best = build_split(
                    cfg,
                    (t + offset) as u32,
                    g - min_gain_shift,
                    left_count,
                    right_count,
                    sum_g,
                    sum_hessian,
                    sum_left_gradient,
                    sum_left_hessian,
                    constraint,
                    false,
                );
            }
            t += 1;
        }
    }

    best
}

/// Assemble the winning [`SplitInfo`] with the clamped child outputs.
///
/// `sum_gradient`/`sum_hessian` are the leaf's RAW totals; `left_sum_gradient`/
/// `left_sum_hessian` are the RAW (un-`kEpsilon`'d) child sums. C++
/// (`feature_histogram.hpp:1049-1066`) computes the child OUTPUTS from the RAW
/// hessians (`best_sum_left_hessian` and `sum_hessian - best_sum_left_hessian`) and
/// ONLY THEN stores `{left,right}_sum_hessian = <raw> - kEpsilon`. Computing the
/// output from the already-`-kEpsilon` value shifts the monotone leaf output ~1 ULP
/// (`0.050000000000000003` vs golden `0.049999999999999989`) — the DEF-07-11-01
/// fold-order knife-edge. The RIGHT operands also follow the C++ form
/// (`sum_gradient - best_sum_left_gradient`, `sum_hessian - best_sum_left_hessian`),
/// not the scan-accumulated `sum_right_*`, since those can differ in the last ULP.
#[allow(clippy::too_many_arguments)]
fn build_split(
    cfg: &GainConfig,
    threshold: u32,
    gain: f64,
    left_count: i32,
    right_count: i32,
    sum_gradient: f64,
    sum_hessian: f64,
    left_sum_gradient: f64,
    left_sum_hessian: f64,
    constraint: &BasicConstraint,
    default_left: bool,
) -> SplitInfo {
    let eps = f64::from(lgbm_core::types::K_EPSILON);
    let right_sum_gradient = sum_gradient - left_sum_gradient;
    let right_sum_hessian = sum_hessian - left_sum_hessian;
    let left_output = calc_output_clamped(cfg, left_sum_gradient, left_sum_hessian, constraint);
    let right_output = calc_output_clamped(cfg, right_sum_gradient, right_sum_hessian, constraint);
    SplitInfo {
        threshold,
        gain,
        left_count,
        right_count,
        left_sum_gradient,
        left_sum_hessian: left_sum_hessian - eps,
        right_sum_gradient,
        right_sum_hessian: right_sum_hessian - eps,
        left_output,
        right_output,
        default_left,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GainConfig {
        GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn inactive_when_all_zero_or_empty() {
        assert!(MonotoneConstraints::new(&[], 8, 0.0).is_none());
        assert!(MonotoneConstraints::new(&[0, 0, 0], 8, 0.0).is_none());
        assert!(MonotoneConstraints::new(&[0, 1, 0], 8, 0.0).is_some());
    }

    #[test]
    fn plus_one_constraint_rejects_decreasing_split() {
        // A feature where low bins have POSITIVE gradient (=> negative output,
        // since output = -g/h) and high bins NEGATIVE gradient (=> positive
        // output). Unconstrained, the best split puts low (neg output) left and
        // high (pos output) right => left_output < right_output, which a +1
        // constraint ALLOWS. Flip the gradient sign so left_output > right_output
        // and the +1 constraint must REJECT (gain 0) for that bin.
        //
        // bins 0,0,1,1 ; grad such that left output > right output.
        // grad: bin0 = -4 (output +4/h), bin1 = +4 (output -4/h).
        // reverse split at bin0|bin1: left=bin0 (output +), right=bin1 (output -)
        //   => left_output > right_output => +1 rejects.
        let hist = {
            // 2 bins, stride-2 [g,h]: bin0=(-8,2), bin1=(8,2).
            vec![-8.0f64, 2.0, 8.0, 2.0]
        };
        let constraint = BasicConstraint::default();
        // num_bin=2 => no FORWARD (run_forward false); REVERSE only. offset 0.
        let unconstrained = lgbm_compute::gain::SplitInfo::none();
        let _ = unconstrained;
        let split = find_best_split_monotone(
            &hist,
            &cfg(),
            2,    // num_bin
            0,    // offset
            99,   // default_bin (unused)
            false,
            false,
            0.0, // sum_g = -8 + 8 = 0
            4.0, // sum_h = 2 + 2 = 4
            4,
            1, // monotone +1
            &constraint,
        );
        // left output (bin0 grad -8) = +8/(2)=+4; right (bin1 grad +8)=-4.
        // left > right => +1 rejects => no positive-gain split.
        assert!(
            split.gain <= 0.0 || split.gain.is_infinite(),
            "+1 constraint must reject a decreasing split (gain={})",
            split.gain
        );
    }

    #[test]
    fn plus_one_constraint_allows_increasing_split() {
        // grad: bin0=+8 (output -4), bin1=-8 (output +4): left_output < right
        // => +1 ALLOWS, positive gain.
        let hist = vec![8.0f64, 2.0, -8.0, 2.0];
        let split = find_best_split_monotone(
            &hist,
            &cfg(),
            2,
            0,
            99,
            false,
            false,
            0.0,
            4.0,
            4,
            1,
            &BasicConstraint::default(),
        );
        assert!(
            split.gain > 0.0,
            "+1 constraint must allow an increasing split (gain={})",
            split.gain
        );
        assert!(split.left_output <= split.right_output + 1e-12);
    }

    #[test]
    fn penalty_monotonic_in_depth() {
        let mc = MonotoneConstraints::new(&[1], 8, 0.5).unwrap();
        // penalization 0.5 <= 1 => 1 - 0.5/2^depth + eps.
        let p0 = mc.split_gain_penalty(0);
        let p1 = mc.split_gain_penalty(1);
        // deeper => smaller penalty subtracted => larger multiplier.
        assert!(p1 > p0, "penalty multiplier grows with depth ({p0} vs {p1})");
        assert!(p0 < 1.0 + 1e-9);
    }

    #[test]
    fn update_propagates_mid_constraint() {
        let mut mc = MonotoneConstraints::new(&[1], 8, 0.0).unwrap();
        // +1 split, left_output=1, right_output=3 => mid=2.
        mc.update_after_split(0, 1, 1, 1.0, 3.0);
        // parent (leaf 0) max clamped to 2; child (leaf 1) min clamped to 2.
        assert_eq!(mc.constraint_for(0).max, 2.0);
        assert_eq!(mc.constraint_for(1).min, 2.0);
    }
}
