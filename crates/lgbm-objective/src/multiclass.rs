//! `multiclass` objectives — `softmax` (the redundant-form K-class cross-entropy)
//! and `multiclassova` (one-vs-all binary), ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/objective/multiclass_objective.hpp:24-181` — `MulticlassSoftmax`:
//!   - ctor (`:26-32`): `factor_ = num_class / (num_class - 1.0f)` (the Friedman
//!     redundant-form rescale).
//!   - `Init` (`:53-84`): `label_int_[i] = (int)label_[i]`; the range check
//!     `label < 0 || label >= num_class` → `Log::Fatal` (`:62`) — surfaced here as
//!     [`ObjectiveError::LabelOutOfRange`] (Security V5, T-06-04-01); the
//!     `class_init_probs_[k] = count(label==k) / num_data` baseline.
//!   - `GetGradients` (`:86-130`, no-weights): the STRIDED class-major gather
//!     `rec[k] = score[num_data * k + i]`, `Common::Softmax(&rec)`, then
//!     `grad[idx] = p - 1` if `label_int_[i] == k` else `p`;
//!     `hess[idx] = factor_ * p * (1 - p)`; all f32-cast on store.
//!   - `BoostFromScore(class_id)` (`:155-157`):
//!     `log(max(kEpsilon, class_init_probs_[class_id]))`.
//!   - `ClassNeedTrain(class_id)` (`:159-166`): false when
//!     `|class_init_probs_[k]| <= kEpsilon || >= 1 - kEpsilon`.
//!   - `NumModelPerIteration` / `NumPredictOneRow` (`:149,151`) = `num_class`.
//! - `multiclass_objective.hpp:186-276` — `MulticlassOVA`: holds `num_class`
//!   independent `BinaryLogloss`, each with `is_pos = (int)label == i`. `Init`
//!   inits each (`:221-226`); `GetGradients` (`:228-233`) calls binary `i` at
//!   `offset = num_data * i` over the class-major slices; `BoostFromScore` /
//!   `ClassNeedTrain` delegate to the per-class binary (`:261-267`).
//! - `LightGBM/include/LightGBM/utils/common.h:571-586` — `Common::Softmax`
//!   (the `std::vector<double>*` in-place overload): `wmax = max(rec)`,
//!   `rec[i] = exp(rec[i] - wmax)`, `rec[i] /= wsum` (max-subtraction stability).
//!   Re-used VERBATIM via [`lgbm_model::objective::softmax`] (the out-of-place
//!   overload `common.h:587`), which is the same algorithm — NOT re-ported here.
//!
//! The predict-side transform (`ConvertOutput` softmax / per-class sigmoid) lives
//! in [`lgbm_model::ObjectiveKind::convert`] and is reused for predict + the
//! multi_logloss metric — this crate owns only the TRAINING side.
//!
//! The spine corpora are unweighted; the `weights_ != nullptr` branches
//! (multiclass_objective.hpp:108-129) are recorded in the formula but a later
//! wave's surface (the per-row weight multiply). `deterministic = true` strips the
//! C++ OpenMP parallelism, so this is an ordered sequential f64 fold — the
//! bit-exact deterministic anchor.

use lgbm_core::types::K_EPSILON;
use lgbm_model::objective::softmax;

use crate::binary::Binary;
use crate::error::ObjectiveError;

/// The `multiclass` (softmax) objective, training side.
///
/// Carries the validated `num_class`, the `factor_` redundant-form rescale, the
/// per-row integer labels (`label_int_`), and the per-class init probabilities
/// (`class_init_probs_`), all computed once at [`MulticlassSoftmax::new`] (the C++
/// `Init`).
#[derive(Debug, Clone, PartialEq)]
pub struct MulticlassSoftmax {
    /// `num_class_` (`config.num_class`, `>= 1`).
    num_class: i32,
    /// `factor_ = num_class / (num_class - 1.0f)` (multiclass_objective.hpp:31).
    factor: f64,
    /// `label_int_[i] = (int)label_[i]` — the per-row integer class (Init :61),
    /// range-validated to `[0, num_class)`.
    label_int: Vec<i32>,
    /// `class_init_probs_[k] = count(label==k) / num_data` (Init :81-83).
    class_init_probs: Vec<f64>,
}

impl MulticlassSoftmax {
    /// Construct + run the C++ `Init` over `labels` (compute `label_int_` +
    /// `class_init_probs_`), validating `num_class >= 1` and every label in
    /// `[0, num_class)` (multiclass_objective.hpp:62 `Log::Fatal` → typed error).
    ///
    /// # Errors
    /// - [`ObjectiveError::Unsupported`] when `num_class < 1`.
    /// - [`ObjectiveError::LabelOutOfRange`] when any `label` (truncated to int) is
    ///   `< 0` or `>= num_class` (Security V5 — never an OOB index / panic).
    pub fn new(num_class: i32, labels: &[f32]) -> Result<Self, ObjectiveError> {
        if num_class < 1 {
            return Err(ObjectiveError::Unsupported {
                name: format!("multiclass num_class {num_class} must be >= 1"),
            });
        }
        // factor_ = num_class / (num_class - 1.0f) — note the C++ subtracts an f32
        // literal `1.0f` (immaterial for integers up to 2^24).
        let factor = num_class as f64 / (num_class as f64 - 1.0f32 as f64);
        let mut label_int = Vec::with_capacity(labels.len());
        let mut class_init_probs = vec![0.0f64; num_class as usize];
        // num_data == labels.len() (unweighted: sum_weight = num_data).
        for &l in labels {
            // C++ `static_cast<int>(label_[i])` truncates toward zero.
            let li = l as i32;
            if li < 0 || li >= num_class {
                return Err(ObjectiveError::LabelOutOfRange {
                    label: li,
                    num_class,
                });
            }
            class_init_probs[li as usize] += 1.0;
            label_int.push(li);
        }
        let sum_weight = labels.len() as f64;
        if sum_weight > 0.0 {
            for p in &mut class_init_probs {
                *p /= sum_weight;
            }
        }
        Ok(Self {
            num_class,
            factor,
            label_int,
            class_init_probs,
        })
    }

    /// `num_class_`.
    pub fn num_class(&self) -> i32 {
        self.num_class
    }

    /// C++ `ObjectiveFunction::NumModelPerIteration` (multiclass_objective.hpp:149)
    /// = `num_class` — the number of trees the GBDT loop grows per iteration.
    pub fn num_model_per_iteration(&self) -> i32 {
        self.num_class
    }

    /// C++ `MulticlassSoftmax::GetGradients` (multiclass_objective.hpp:86-130,
    /// no-weights). `score` is the WHOLE class-major f64 score buffer (length
    /// `num_data * num_class`, class `k`'s rows at `[k*num_data, (k+1)*num_data)`);
    /// `gradients`/`hessians` receive the same layout (f32). Per row `i`:
    /// - gather `rec[k] = score[num_data * k + i]` (STRIDED — Pattern 4: a
    ///   row-major `[row][class]` layout silently diverges here);
    /// - `Common::Softmax(&rec)` (max-subtraction);
    /// - `grad[idx] = p - 1` if `label_int_[i] == k` else `p`;
    ///   `hess[idx] = factor_ * p * (1 - p)`; f32-cast on store.
    ///
    /// # Errors
    /// [`ObjectiveError::LengthMismatch`] (V5 boundary) if the slices are not
    /// `num_data * num_class` long — validated before any per-row write.
    pub fn get_gradients(
        &self,
        score: &[f64],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<(), ObjectiveError> {
        let num_data = self.label_int.len();
        let k_classes = self.num_class as usize;
        let total = num_data * k_classes;
        if score.len() != total {
            return Err(ObjectiveError::LengthMismatch { expected: total, actual: score.len() });
        }
        if gradients.len() != total {
            return Err(ObjectiveError::LengthMismatch { expected: total, actual: gradients.len() });
        }
        if hessians.len() != total {
            return Err(ObjectiveError::LengthMismatch { expected: total, actual: hessians.len() });
        }
        let mut rec = vec![0.0f64; k_classes];
        let mut prob = vec![0.0f64; k_classes];
        for i in 0..num_data {
            // STRIDED class-major gather: rec[k] = score[num_data * k + i].
            for k in 0..k_classes {
                rec[k] = score[num_data * k + i];
            }
            // Common::Softmax (in-place semantics; out-of-place form here).
            softmax(&rec, &mut prob);
            for k in 0..k_classes {
                let p = prob[k];
                let idx = num_data * k + i;
                // `p - 1.0f`: the C++ subtracts an f32 literal but `p` is f64; the
                // subtraction is in f64 then cast (matches C++ `(score_t)(p - 1.0f)`
                // promotion: 1.0f promotes to 1.0 double before the subtract).
                if self.label_int[i] == k as i32 {
                    gradients[idx] = (p - 1.0f32 as f64) as f32;
                } else {
                    gradients[idx] = p as f32;
                }
                hessians[idx] = (self.factor * p * (1.0f32 as f64 - p)) as f32;
            }
        }
        Ok(())
    }

    /// C++ `MulticlassSoftmax::BoostFromScore(class_id)` (multiclass_objective.hpp:
    /// 155-157): `log(max(kEpsilon, class_init_probs_[class_id]))`.
    pub fn boost_from_score(&self, class_id: i32) -> f64 {
        let p = self.class_init_probs[class_id as usize];
        p.max(K_EPSILON as f64).ln()
    }

    /// C++ `MulticlassSoftmax::ClassNeedTrain(class_id)` (multiclass_objective.hpp:
    /// 159-166): false when `|class_init_probs_[k]| <= kEpsilon` (the class is
    /// absent) or `>= 1 - kEpsilon` (the class is the entire corpus) — then a
    /// constant tree is pushed instead of training (Pitfall 6).
    pub fn class_need_train(&self, class_id: i32) -> bool {
        let p = self.class_init_probs[class_id as usize].abs();
        let eps = K_EPSILON as f64;
        !(p <= eps || p >= 1.0 - eps)
    }
}

/// The `multiclassova` (one-vs-all) objective, training side.
///
/// Holds `num_class` independent [`Binary`] objectives; class `i`'s binary treats
/// `is_pos = (int)label == i` (multiclass_objective.hpp:192). Each operates on its
/// own class-major score slice at `offset = num_data * i`.
#[derive(Debug, Clone, PartialEq)]
pub struct MulticlassOva {
    /// `num_class_`.
    num_class: i32,
    /// The shared `sigmoid_` for every per-class binary.
    binary: Binary,
    /// Per-row integer labels (`(int)label`), used to derive each class's
    /// `is_pos` mapping at grad time. Validated `>= 0` (a negative label would map
    /// to no positive class — C++ `static_cast<int>` keeps it but it is never a
    /// positive for any class; we keep the C++ behavior: no range fatal in OVA).
    label_int: Vec<i32>,
}

impl MulticlassOva {
    /// Construct with a validated `num_class >= 1` and `sigmoid > 0`. Unlike
    /// softmax, the C++ `MulticlassOVA` does NOT range-check the label (it only maps
    /// `is_pos = (int)label == i` per class) — we mirror that, recording the
    /// per-row int label for the grad-time `is_pos` derivation.
    ///
    /// # Errors
    /// [`ObjectiveError::Unsupported`] when `num_class < 1` or `sigmoid <= 0`.
    pub fn new(num_class: i32, sigmoid: f64, labels: &[f32]) -> Result<Self, ObjectiveError> {
        if num_class < 1 {
            return Err(ObjectiveError::Unsupported {
                name: format!("multiclassova num_class {num_class} must be >= 1"),
            });
        }
        let binary = Binary::new(sigmoid)?;
        let label_int = labels.iter().map(|&l| l as i32).collect();
        Ok(Self {
            num_class,
            binary,
            label_int,
        })
    }

    /// `num_class_`.
    pub fn num_class(&self) -> i32 {
        self.num_class
    }

    /// C++ `NumModelPerIteration` (multiclass_objective.hpp:255) = `num_class`.
    pub fn num_model_per_iteration(&self) -> i32 {
        self.num_class
    }

    /// The per-class one-vs-all binary labels (`is_pos = (int)label == class_id`)
    /// as `{0.0, 1.0}` f32 — what binary `class_id` trains against. Exposed so the
    /// per-class init / metric paths can reuse the exact mapping.
    pub fn class_labels(&self, class_id: i32) -> Vec<f32> {
        self.label_int
            .iter()
            .map(|&li| if li == class_id { 1.0f32 } else { 0.0f32 })
            .collect()
    }

    /// C++ `MulticlassOVA::GetGradients` (multiclass_objective.hpp:228-233): for
    /// each class `i`, call binary `i` over the class-major slice at
    /// `offset = num_data * i` with `is_pos = (label == i)`.
    ///
    /// # Errors
    /// [`ObjectiveError::LengthMismatch`] when the buffers are not
    /// `num_data * num_class` long.
    pub fn get_gradients(
        &self,
        score: &[f64],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<(), ObjectiveError> {
        let num_data = self.label_int.len();
        let k_classes = self.num_class as usize;
        let total = num_data * k_classes;
        if score.len() != total {
            return Err(ObjectiveError::LengthMismatch { expected: total, actual: score.len() });
        }
        if gradients.len() != total {
            return Err(ObjectiveError::LengthMismatch { expected: total, actual: gradients.len() });
        }
        if hessians.len() != total {
            return Err(ObjectiveError::LengthMismatch { expected: total, actual: hessians.len() });
        }
        for i in 0..k_classes {
            let offset = num_data * i;
            let cls_labels = self.class_labels(i as i32);
            self.binary.get_gradients(
                &score[offset..offset + num_data],
                &cls_labels,
                &mut gradients[offset..offset + num_data],
                &mut hessians[offset..offset + num_data],
            )?;
        }
        Ok(())
    }

    /// C++ `MulticlassOVA::BoostFromScore(class_id)` (multiclass_objective.hpp:
    /// 261-263): the per-class binary's `BoostFromScore` over its one-vs-all labels.
    pub fn boost_from_score(&self, class_id: i32) -> f64 {
        let cls_labels = self.class_labels(class_id);
        self.binary.boost_from_score(&cls_labels)
    }

    /// C++ `MulticlassOVA::ClassNeedTrain(class_id)` (multiclass_objective.hpp:
    /// 265-267): the per-class binary's `ClassNeedTrain` (false when one class is
    /// absent across the one-vs-all split).
    pub fn class_need_train(&self, class_id: i32) -> bool {
        let cls_labels = self.class_labels(class_id);
        self.binary.class_need_train(&cls_labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oracle_harness::comparator::ORACLE_TOL;

    const TOL: f64 = ORACLE_TOL as f64;

    /// Hand-compute softmax with the max-subtraction algorithm (the bit-exact
    /// expected), mirroring `Common::Softmax`.
    fn hand_softmax(rec: &[f64]) -> Vec<f64> {
        let mut wmax = rec[0];
        for &v in &rec[1..] {
            wmax = v.max(wmax);
        }
        let mut wsum = 0.0f64;
        let mut out: Vec<f64> = rec.iter().map(|&v| {
            let e = (v - wmax).exp();
            wsum += e;
            e
        }).collect();
        for o in &mut out {
            *o /= wsum;
        }
        out
    }

    #[test]
    fn softmax_max_subtraction_bit_exact() {
        // The reused softmax (lgbm_model::objective::softmax) must equal the
        // hand-computed max-subtraction algorithm bit-for-bit.
        let rec = [0.5f64, -1.2, 2.3];
        let mut got = [0.0f64; 3];
        softmax(&rec, &mut got);
        let expect = hand_softmax(&rec);
        for k in 0..3 {
            assert_eq!(got[k].to_bits(), expect[k].to_bits(), "softmax[{k}] bit-exact");
        }
    }

    #[test]
    fn multiclass_strided_gather_grad_hess() {
        // 3 classes, 2 rows. score buffer is class-major: [c0r0,c0r1, c1r0,c1r1,
        // c2r0,c2r1]. labels: row0=class1, row1=class2.
        let num_class = 3;
        let num_data = 2;
        let labels = [1.0f32, 2.0];
        let obj = MulticlassSoftmax::new(num_class, &labels).unwrap();
        // arbitrary scores per (class,row).
        let score = vec![
            0.1f64, 0.4, // class 0: r0, r1
            -0.3, 1.2,   // class 1
            0.7, -0.5,   // class 2
        ];
        let mut grad = vec![0.0f32; 6];
        let mut hess = vec![0.0f32; 6];
        obj.get_gradients(&score, &mut grad, &mut hess).unwrap();
        let factor = 3.0 / (3.0 - 1.0f32 as f64);
        for i in 0..num_data as usize {
            // gather rec strided.
            let rec: Vec<f64> = (0..num_class as usize)
                .map(|k| score[num_data as usize * k + i])
                .collect();
            let p = hand_softmax(&rec);
            for k in 0..num_class as usize {
                let idx = num_data as usize * k + i;
                let label_i = labels[i] as i32;
                let eg = if label_i == k as i32 {
                    (p[k] - 1.0f32 as f64) as f32
                } else {
                    p[k] as f32
                };
                let eh = (factor * p[k] * (1.0f32 as f64 - p[k])) as f32;
                assert_eq!(grad[idx], eg, "grad[{idx}] (class {k}, row {i}) bit-exact");
                assert_eq!(hess[idx], eh, "hess[{idx}] bit-exact");
            }
        }
    }

    #[test]
    fn multiclass_boost_from_score_is_log_class_prob() {
        // 3 classes; labels: two class-0, one class-1, one class-2 (num_data=4).
        let labels = [0.0f32, 0.0, 1.0, 2.0];
        let obj = MulticlassSoftmax::new(3, &labels).unwrap();
        // class_init_probs = [2/4, 1/4, 1/4].
        let expect0 = (2.0f64 / 4.0).max(K_EPSILON as f64).ln();
        let expect1 = (1.0f64 / 4.0).max(K_EPSILON as f64).ln();
        assert!((obj.boost_from_score(0) - expect0).abs() < TOL);
        assert!((obj.boost_from_score(1) - expect1).abs() < TOL);
        assert!((obj.boost_from_score(2) - expect1).abs() < TOL);
    }

    #[test]
    fn multiclass_boost_from_score_clamps_absent_class() {
        // class 2 absent -> prob 0 -> clamped to kEpsilon -> log(kEpsilon) finite.
        let labels = [0.0f32, 1.0, 0.0, 1.0];
        let obj = MulticlassSoftmax::new(3, &labels).unwrap();
        let init = obj.boost_from_score(2);
        assert!(init.is_finite(), "absent-class init must be finite: {init}");
        assert_eq!(init.to_bits(), (K_EPSILON as f64).ln().to_bits());
        // and class_need_train is false for the absent class.
        assert!(!obj.class_need_train(2));
        assert!(obj.class_need_train(0));
    }

    #[test]
    fn multiclass_label_out_of_range_is_typed_error() {
        // label 3 with num_class=3 -> out of [0,3) -> typed error, not a panic.
        let labels = [0.0f32, 1.0, 3.0];
        let err = MulticlassSoftmax::new(3, &labels).unwrap_err();
        assert!(matches!(err, ObjectiveError::LabelOutOfRange { label: 3, num_class: 3 }));
        // negative label too.
        let neg = [0.0f32, -1.0];
        assert!(matches!(
            MulticlassSoftmax::new(2, &neg).unwrap_err(),
            ObjectiveError::LabelOutOfRange { label: -1, num_class: 2 }
        ));
    }

    #[test]
    fn multiclass_rejects_bad_num_class() {
        assert!(MulticlassSoftmax::new(0, &[0.0]).is_err());
        assert!(MulticlassSoftmax::new(-1, &[0.0]).is_err());
    }

    #[test]
    fn multiclassova_class_i_equals_binary_with_is_pos() {
        // 3 classes, 4 rows. ova class i's g/h must equal a binary objective with
        // is_pos = (label == i), at offset num_data * i.
        let num_class = 3;
        let num_data = 4;
        let labels = [0.0f32, 1.0, 2.0, 1.0];
        let sigmoid = 1.0;
        let ova = MulticlassOva::new(num_class, sigmoid, &labels).unwrap();
        let score = vec![
            0.2f64, -0.1, 0.5, 0.0, // class 0 rows
            0.3, 0.8, -0.4, 0.1,    // class 1 rows
            -0.7, 0.2, 0.9, -0.2,   // class 2 rows
        ];
        let mut grad = vec![0.0f32; 12];
        let mut hess = vec![0.0f32; 12];
        ova.get_gradients(&score, &mut grad, &mut hess).unwrap();

        let binary = Binary::new(sigmoid).unwrap();
        for i in 0..num_class as usize {
            let offset = num_data as usize * i;
            // is_pos labels for class i.
            let cls_labels: Vec<f32> = labels
                .iter()
                .map(|&l| if l as i32 == i as i32 { 1.0 } else { 0.0 })
                .collect();
            let mut eg = vec![0.0f32; num_data as usize];
            let mut eh = vec![0.0f32; num_data as usize];
            binary
                .get_gradients(
                    &score[offset..offset + num_data as usize],
                    &cls_labels,
                    &mut eg,
                    &mut eh,
                )
                .unwrap();
            for r in 0..num_data as usize {
                assert_eq!(grad[offset + r], eg[r], "ova grad class {i} row {r}");
                assert_eq!(hess[offset + r], eh[r], "ova hess class {i} row {r}");
            }
        }
    }

    #[test]
    fn multiclassova_boost_from_score_is_per_class_binary() {
        let labels = [0.0f32, 1.0, 2.0, 1.0];
        let ova = MulticlassOva::new(3, 1.0, &labels).unwrap();
        let binary = Binary::new(1.0).unwrap();
        for c in 0..3 {
            let cls_labels: Vec<f32> = labels
                .iter()
                .map(|&l| if l as i32 == c { 1.0 } else { 0.0 })
                .collect();
            assert_eq!(
                ova.boost_from_score(c).to_bits(),
                binary.boost_from_score(&cls_labels).to_bits(),
                "ova boost_from_score class {c}"
            );
        }
    }

    #[test]
    fn num_model_per_iteration_is_num_class() {
        let labels = [0.0f32, 1.0, 2.0];
        assert_eq!(MulticlassSoftmax::new(3, &labels).unwrap().num_model_per_iteration(), 3);
        assert_eq!(MulticlassOva::new(3, 1.0, &labels).unwrap().num_model_per_iteration(), 3);
    }
}
