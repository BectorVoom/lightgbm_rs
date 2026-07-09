//! Core `ConvertOutput` objective shim (PRD-02) — the four in-scope transforms
//! parsed from the model's `objective=` line (NOT a training `Config`).
//!
//! Faithful 1:1 port of the C++ `ObjectiveFunction::ConvertOutput` overrides that
//! Phase 3 needs, transcribed from:
//! - `LightGBM/src/objective/regression_objective.hpp` `ConvertOutput` (148, the
//!   identity / `Common::Sign(x)*x*x` sqrt variant) + string-ctor `sqrt` token.
//! - `LightGBM/src/objective/binary_objective.hpp` `ConvertOutput` (175,
//!   `1/(1+exp(-sigmoid*input))`) + string-ctor `sigmoid:` parse (43-53).
//! - `LightGBM/src/objective/multiclass_objective.hpp` `ConvertOutput` (132 the
//!   softmax, 239 the per-class ova sigmoid) + string-ctor `num_class:` parse
//!   (34-47).
//! - `LightGBM/include/LightGBM/utils/common.h` `Softmax` (587, WITH the
//!   `wmax` max-subtraction) + `Sign` (873, `(x>0)-(x<0)`).
//!
//! # The objective string is the source of truth
//! `GBDT::Predict` (`gbdt_prediction.cpp:55`) calls `objective_function_->
//! ConvertOutput`. The objective function's parameters (`sigmoid_`, `num_class_`,
//! `sqrt_`) come from the model's `objective=` line, which LightGBM re-creates the
//! objective from on load (`gbdt.cpp` `LoadModelFromString` → `objective=` →
//! `ObjectiveFunction::CreateObjectiveFunction(name, strs)`). We mirror that: the
//! parser reads `sigmoid:`/`num_class:`/`sqrt` tokens straight off the line —
//! never from a training `Config` (RESEARCH line 370).
//!
//! # Scope (Phase 7 boundary)
//! Only the CORE objectives are in scope: `regression`/`regression_l1` (identity),
//! `binary` (sigmoid), `multiclass` (softmax), `multiclassova`/`ova`/`ovr`
//! (per-class sigmoid). The OBJ-04/05 exp/log family adds `poisson`/`gamma`/
//! `tweedie` (`ConvertOutput = exp`), `cross_entropy`/`xentropy` (sigmoid), and
//! `cross_entropy_lambda`/`xentlambda` (`log1p(exp)`). The identity-`ConvertOutput`
//! regression objectives (`huber`/`fair`/`quantile`/`mape`) are NOT parsed here —
//! the caller maps them to [`ObjectiveKind::Regression`] (their predict-side
//! transform IS the identity). Any other objective (lambdarank/…) is rejected with
//! [`ModelError::MalformedModel`] — never silently defaulted-to-identity (T-03-09).

use crate::error::ModelError;

/// C++ `Common::Sign` (`common.h:873`): `(x > 0) - (x < 0)` → `-1 | 0 | 1`.
#[inline]
fn sign(x: f64) -> f64 {
    ((x > 0.0) as i32 - (x < 0.0) as i32) as f64
}

/// The parsed objective kind + its `ConvertOutput` parameters, read from the
/// model's `objective=` line. Mirrors which C++ `ObjectiveFunction` subclass the
/// loader instantiates.
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveKind {
    /// `regression`, `regression_l1`, `regression_l2`, `mse`, `l2`, `l1`, `mae`,
    /// `mean_absolute_error`, `mean_squared_error` — identity (or `sqrt` variant
    /// for `regression`). `ConvertOutput`: identity / `Sign(x)*x*x`.
    Regression {
        /// The `sqrt` token was present on the `objective=` line.
        sqrt: bool,
    },
    /// `binary` — single-output logistic. `ConvertOutput`:
    /// `1/(1+exp(-sigmoid*input))`.
    Binary {
        /// `sigmoid:` value (C++ default 1.0). Must be `> 0`.
        sigmoid: f64,
    },
    /// `multiclass`/`softmax` — `num_class` outputs, softmax with
    /// max-subtraction.
    Multiclass {
        /// `num_class:` value (`>= 1`).
        num_class: i32,
    },
    /// `multiclassova`/`multiclass_ova`/`ova`/`ovr` — per-class independent
    /// sigmoid over `num_class` outputs.
    MulticlassOva {
        /// `num_class:` value (`>= 1`).
        num_class: i32,
        /// `sigmoid:` value (C++ default 1.0). Must be `> 0`.
        sigmoid: f64,
    },
    /// `poisson` / `gamma` / `tweedie` (OBJ-04 exp/log family) — single-output
    /// log-mean regression. `ConvertOutput`: `exp(input)`
    /// (`regression_objective.hpp:460` poisson; gamma/tweedie subclass it).
    Poisson,
    /// `cross_entropy` / `xentropy` (OBJ-05) — single-output cross-entropy.
    /// `ConvertOutput`: `1/(1+exp(-input))` (`xentropy_objective.hpp:131`).
    CrossEntropy,
    /// `cross_entropy_lambda` / `xentlambda` (OBJ-05) — single-output
    /// alternative-parameterization cross-entropy. `ConvertOutput`:
    /// `log1p(exp(input))` (the lambda, NOT a probability;
    /// `xentropy_objective.hpp:251`).
    CrossEntropyLambda,
}

impl ObjectiveKind {
    /// C++ `ObjectiveFunction::NeedAccuratePrediction` (`objective_function.h:76`)
    /// plus the per-objective overrides. Returns `false` ONLY for objectives whose
    /// raw margin is a decisive classification score — `binary`
    /// (`binary_objective.hpp:188`), `multiclass` / `multiclassova`
    /// (`multiclass_objective.hpp:153,259`), and ranking (not represented here).
    /// Every regression-like objective (regression, poisson/gamma/tweedie,
    /// cross_entropy / cross_entropy_lambda) inherits the base `true`.
    ///
    /// This is the GATE that decides whether prediction early stopping is allowed:
    /// `Predictor` installs the active early-stop instance only when
    /// `!NeedAccuratePrediction()` (`predictor.hpp:46`). For an
    /// accurate-prediction objective the early-stop request is silently ignored and
    /// the full ensemble is always evaluated.
    pub fn need_accurate_prediction(&self) -> bool {
        match self {
            ObjectiveKind::Binary { .. }
            | ObjectiveKind::Multiclass { .. }
            | ObjectiveKind::MulticlassOva { .. } => false,
            ObjectiveKind::Regression { .. }
            | ObjectiveKind::Poisson
            | ObjectiveKind::CrossEntropy
            | ObjectiveKind::CrossEntropyLambda => true,
        }
    }

    /// Parse the model's `objective=` line value (e.g. `binary sigmoid:1`,
    /// `multiclass num_class:3`, `regression sqrt`, `regression`) into a kind +
    /// params. Mirrors the C++ string-ctors: the first whitespace token is the
    /// objective name, the remaining tokens are `key:value` (or the bare `sqrt`
    /// flag), parsed by splitting on `:` exactly as `Common::Split(str, ':')`.
    ///
    /// Returns [`ModelError::MalformedModel`] for any non-core objective (Phase 7
    /// scope) or for an invalid `sigmoid`/`num_class` value — never a panic, never
    /// a silent identity default (T-03-09).
    pub fn parse(objective_line: &str) -> Result<ObjectiveKind, ModelError> {
        let mut tokens = objective_line.split_whitespace();
        let name = tokens.next().ok_or_else(|| ModelError::MalformedModel {
            detail: "empty objective= line".to_string(),
        })?;

        // Collect `key:value` tokens (and the bare `sqrt` flag) from the rest.
        let mut sigmoid: f64 = 1.0;
        let mut num_class: Option<i32> = None;
        let mut sqrt = false;
        for tok in tokens {
            if tok == "sqrt" {
                sqrt = true;
                continue;
            }
            // C++ `Common::Split(str, ':')`; only `tokens.size()==2` is honored.
            if let Some((key, val)) = tok.split_once(':') {
                match key {
                    "sigmoid" => {
                        sigmoid = val.parse::<f64>().map_err(|_| ModelError::MalformedModel {
                            detail: format!("objective sigmoid: not a float: {val:?}"),
                        })?;
                    }
                    "num_class" => {
                        let nc =
                            val.parse::<i32>().map_err(|_| ModelError::MalformedModel {
                                detail: format!("objective num_class: not an int: {val:?}"),
                            })?;
                        num_class = Some(nc);
                    }
                    // Unknown key:value tokens are ignored (forward-compatible),
                    // matching the C++ ctor which only matches known keys.
                    _ => {}
                }
            }
        }

        match name {
            "regression" | "regression_l2" | "regression_l1" | "mean_squared_error"
            | "mse" | "l2" | "mean_absolute_error" | "mae" | "l1" => {
                // Only `regression` honors a `sqrt` token in C++, but accepting
                // it on any l2/l1 alias is harmless (the token only appears for
                // `regression sqrt`).
                Ok(ObjectiveKind::Regression { sqrt })
            }
            "binary" => {
                if sigmoid <= 0.0 {
                    return Err(ModelError::MalformedModel {
                        detail: format!("binary sigmoid {sigmoid} must be > 0"),
                    });
                }
                Ok(ObjectiveKind::Binary { sigmoid })
            }
            "multiclass" | "softmax" => {
                let nc = num_class.ok_or_else(|| ModelError::MalformedModel {
                    detail: "multiclass objective must contain num_class".to_string(),
                })?;
                if nc < 1 {
                    return Err(ModelError::MalformedModel {
                        detail: format!("multiclass num_class {nc} must be >= 1"),
                    });
                }
                Ok(ObjectiveKind::Multiclass { num_class: nc })
            }
            "multiclassova" | "multiclass_ova" | "ova" | "ovr" => {
                let nc = num_class.ok_or_else(|| ModelError::MalformedModel {
                    detail: "multiclassova objective must contain num_class".to_string(),
                })?;
                if nc < 1 {
                    return Err(ModelError::MalformedModel {
                        detail: format!("multiclassova num_class {nc} must be >= 1"),
                    });
                }
                if sigmoid <= 0.0 {
                    return Err(ModelError::MalformedModel {
                        detail: format!("multiclassova sigmoid {sigmoid} must be > 0"),
                    });
                }
                Ok(ObjectiveKind::MulticlassOva {
                    num_class: nc,
                    sigmoid,
                })
            }
            // OBJ-04 exp/log family: ConvertOutput = exp(input). gamma/tweedie
            // subclass Poisson and share the exp transform.
            "poisson" | "gamma" | "tweedie" => Ok(ObjectiveKind::Poisson),
            // OBJ-05 cross-entropy: sigmoid ConvertOutput.
            "cross_entropy" | "xentropy" => Ok(ObjectiveKind::CrossEntropy),
            // OBJ-05 cross-entropy lambda: log1p(exp) ConvertOutput.
            "cross_entropy_lambda" | "xentlambda" => Ok(ObjectiveKind::CrossEntropyLambda),
            other => Err(ModelError::MalformedModel {
                detail: format!("objective '{other}' out of Phase-3 scope"),
            }),
        }
    }

    /// The number of transformed outputs this objective produces per row
    /// (`num_class` for multiclass kinds, else 1) — equals
    /// `num_tree_per_iteration` for a well-formed model.
    pub fn num_output(&self) -> usize {
        match self {
            ObjectiveKind::Regression { .. }
            | ObjectiveKind::Binary { .. }
            | ObjectiveKind::Poisson
            | ObjectiveKind::CrossEntropy
            | ObjectiveKind::CrossEntropyLambda => 1,
            ObjectiveKind::Multiclass { num_class }
            | ObjectiveKind::MulticlassOva { num_class, .. } => (*num_class).max(0) as usize,
        }
    }

    /// C++ `ObjectiveFunction::ConvertOutput(input, output)` for the in-scope
    /// objectives. `input` is the raw per-class score vector (length
    /// [`Self::num_output`]); `output` receives the transformed vector (same
    /// length). Accumulation/transform is in f64; the f32 cast happens at the
    /// predict boundary, not here.
    pub fn convert(&self, input: &[f64], output: &mut [f64]) {
        match self {
            ObjectiveKind::Regression { sqrt } => {
                output[0] = convert_regression(input[0], *sqrt);
            }
            ObjectiveKind::Binary { sigmoid } => {
                output[0] = convert_binary(input[0], *sigmoid);
            }
            ObjectiveKind::Multiclass { .. } => {
                softmax(input, output);
            }
            ObjectiveKind::MulticlassOva { sigmoid, .. } => {
                convert_multiclassova(input, output, *sigmoid);
            }
            ObjectiveKind::Poisson => {
                output[0] = convert_poisson(input[0]);
            }
            ObjectiveKind::CrossEntropy => {
                // sigmoid with sigmoid_ == 1.0 (xentropy has no sigmoid param).
                output[0] = convert_binary(input[0], 1.0);
            }
            ObjectiveKind::CrossEntropyLambda => {
                output[0] = convert_cross_entropy_lambda(input[0]);
            }
        }
    }
}

/// C++ `RegressionPoissonLoss::ConvertOutput` (`regression_objective.hpp:460`):
/// `exp(input)`. gamma/tweedie subclass Poisson and share it.
#[inline]
pub fn convert_poisson(input: f64) -> f64 {
    input.exp()
}

/// C++ `CrossEntropyLambda::ConvertOutput` (`xentropy_objective.hpp:251`):
/// `log1p(exp(input))` (the lambda > 0, NOT a probability).
#[inline]
pub fn convert_cross_entropy_lambda(input: f64) -> f64 {
    input.exp().ln_1p()
}

/// C++ `RegressionL2loss::ConvertOutput` (`regression_objective.hpp:148`):
/// identity, or `Common::Sign(input)*input*input` for the `sqrt` variant.
#[inline]
pub fn convert_regression(input: f64, sqrt: bool) -> f64 {
    if sqrt {
        sign(input) * input * input
    } else {
        input
    }
}

/// C++ `BinaryLogloss::ConvertOutput` (`binary_objective.hpp:175`):
/// `1/(1+exp(-sigmoid*input))`.
#[inline]
pub fn convert_binary(input: f64, sigmoid: f64) -> f64 {
    1.0 / (1.0 + (-sigmoid * input).exp())
}

/// C++ `Common::Softmax(input, output, len)` (`common.h:587`) — softmax WITH the
/// `wmax` max-subtraction for numerical stability. Naive `exp/sum` is forbidden
/// (RESEARCH Don't-Hand-Roll line 275; T-03-08): on a large-magnitude raw score
/// `exp(input)` overflows to `inf`, but `exp(input - wmax)` stays finite.
///
/// `output.len()` must equal `input.len()`.
pub fn softmax(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    // wmax = max(input[..]) — C++ seeds with input[0] then folds the rest.
    let mut wmax = input[0];
    for &v in &input[1..] {
        wmax = v.max(wmax);
    }
    let mut wsum = 0.0f64;
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = (v - wmax).exp();
        wsum += *o;
    }
    for o in output.iter_mut() {
        *o /= wsum;
    }
}

/// C++ `MulticlassOVA::ConvertOutput` (`multiclass_objective.hpp:239`): per-class
/// independent sigmoid over `num_class` outputs.
pub fn convert_multiclassova(input: &[f64], output: &mut [f64], sigmoid: f64) {
    debug_assert_eq!(input.len(), output.len());
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = convert_binary(v, sigmoid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Tree;

    #[test]
    fn need_accurate_prediction_gates_early_stop() {
        // Only binary / multiclass / multiclassova return false (early-stop allowed).
        assert!(!ObjectiveKind::parse("binary").unwrap().need_accurate_prediction());
        assert!(!ObjectiveKind::parse("multiclass num_class:3")
            .unwrap()
            .need_accurate_prediction());
        assert!(!ObjectiveKind::parse("multiclassova num_class:3")
            .unwrap()
            .need_accurate_prediction());
        // Regression-like objectives inherit the base `true` (early-stop ignored).
        assert!(ObjectiveKind::parse("regression").unwrap().need_accurate_prediction());
        assert!(ObjectiveKind::parse("poisson").unwrap().need_accurate_prediction());
        assert!(ObjectiveKind::parse("cross_entropy")
            .unwrap()
            .need_accurate_prediction());
        assert!(ObjectiveKind::parse("cross_entropy_lambda")
            .unwrap()
            .need_accurate_prediction());
    }

    // --- Test 1 + 2: regression identity / sqrt ---

    #[test]
    fn regression_identity_is_passthrough() {
        for x in [0.0, 1.5, -3.25, 1e9, -1e-9] {
            assert_eq!(convert_regression(x, false), x);
        }
        let k = ObjectiveKind::parse("regression").unwrap();
        assert_eq!(k, ObjectiveKind::Regression { sqrt: false });
        let mut out = [0.0];
        k.convert(&[2.5], &mut out);
        assert_eq!(out[0], 2.5);
    }

    #[test]
    fn regression_sqrt_is_sign_times_square() {
        // sign(x)*x*x: +4 -> +16, -4 -> -16, 0 -> 0.
        assert_eq!(convert_regression(4.0, true), 16.0);
        assert_eq!(convert_regression(-4.0, true), -16.0);
        assert_eq!(convert_regression(0.0, true), 0.0);
        let k = ObjectiveKind::parse("regression sqrt").unwrap();
        assert_eq!(k, ObjectiveKind::Regression { sqrt: true });
        let mut out = [0.0];
        k.convert(&[-3.0], &mut out);
        assert_eq!(out[0], -9.0);
    }

    // --- Test 3: binary sigmoid ---

    #[test]
    fn binary_sigmoid_matches_formula() {
        // sigmoid(0) = 0.5 exactly.
        assert_eq!(convert_binary(0.0, 1.0), 0.5);
        // 1/(1+exp(-sigmoid*x)).
        let s = 1.5f64;
        let x = 0.7f64;
        let expect = 1.0 / (1.0 + (-s * x).exp());
        assert!((convert_binary(x, s) - expect).abs() < 1e-15);

        let k = ObjectiveKind::parse("binary sigmoid:1").unwrap();
        assert_eq!(k, ObjectiveKind::Binary { sigmoid: 1.0 });
        let mut out = [0.0];
        k.convert(&[0.0], &mut out);
        assert_eq!(out[0], 0.5);
    }

    #[test]
    fn binary_default_sigmoid_is_one() {
        // No sigmoid: token -> default 1.0 (C++ default).
        let k = ObjectiveKind::parse("binary").unwrap();
        assert_eq!(k, ObjectiveKind::Binary { sigmoid: 1.0 });
    }

    // --- Test 4: softmax with max-subtraction (stability) ---

    #[test]
    fn softmax_sums_to_one() {
        let input = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        softmax(&input, &mut out);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-15, "softmax must sum to 1, got {sum}");
        // Monotone: larger input -> larger probability.
        assert!(out[0] < out[1] && out[1] < out[2]);
    }

    #[test]
    fn softmax_uses_max_subtraction_no_overflow() {
        // Large-magnitude input: naive exp(1000) overflows to inf, making a
        // naive exp/sum produce NaN. Max-subtraction keeps it finite.
        let input = [1000.0, 1001.0, 1002.0];
        let mut out = [0.0; 3];
        softmax(&input, &mut out);
        let sum: f64 = out.iter().sum();
        assert!(sum.is_finite(), "sum must be finite under max-subtraction");
        assert!((sum - 1.0).abs() < 1e-12, "softmax must still sum to 1, got {sum}");
        for o in out {
            assert!(o.is_finite() && !o.is_nan(), "no inf/NaN with max-subtraction");
        }
        // The probabilities depend only on differences, so [1000,1001,1002]
        // gives the same result as [0,1,2].
        let mut ref_out = [0.0; 3];
        softmax(&[0.0, 1.0, 2.0], &mut ref_out);
        for i in 0..3 {
            assert!((out[i] - ref_out[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn multiclass_parse_and_convert() {
        let k = ObjectiveKind::parse("multiclass num_class:3").unwrap();
        assert_eq!(k, ObjectiveKind::Multiclass { num_class: 3 });
        assert_eq!(k.num_output(), 3);
        let mut out = [0.0; 3];
        k.convert(&[1.0, 2.0, 3.0], &mut out);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-15);
    }

    // --- Test 5: multiclassova ---

    #[test]
    fn multiclassova_is_per_class_sigmoid() {
        let input = [0.0, 1.0, -1.0];
        let mut out = [0.0; 3];
        convert_multiclassova(&input, &mut out, 1.0);
        assert_eq!(out[0], 0.5);
        assert!((out[1] - 1.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-15);
        assert!((out[2] - 1.0 / (1.0 + (1.0f64).exp())).abs() < 1e-15);

        let k = ObjectiveKind::parse("multiclassova num_class:3 sigmoid:1").unwrap();
        assert_eq!(
            k,
            ObjectiveKind::MulticlassOva {
                num_class: 3,
                sigmoid: 1.0
            }
        );
        let mut out2 = [0.0; 3];
        k.convert(&input, &mut out2);
        assert_eq!(out, out2);
    }

    // --- Test 6: objective parse + out-of-scope rejection ---

    #[test]
    fn parse_recognizes_core_objectives() {
        assert_eq!(
            ObjectiveKind::parse("regression").unwrap(),
            ObjectiveKind::Regression { sqrt: false }
        );
        assert_eq!(
            ObjectiveKind::parse("regression_l1").unwrap(),
            ObjectiveKind::Regression { sqrt: false }
        );
        assert_eq!(
            ObjectiveKind::parse("binary sigmoid:2").unwrap(),
            ObjectiveKind::Binary { sigmoid: 2.0 }
        );
        assert_eq!(
            ObjectiveKind::parse("multiclass num_class:3").unwrap(),
            ObjectiveKind::Multiclass { num_class: 3 }
        );
    }

    #[test]
    fn parse_rejects_out_of_scope_objective() {
        // lambdarank / rank_xendcg are out of scope. huber/quantile have an identity
        // ConvertOutput — they are NOT parsed here (the caller maps them to
        // ObjectiveKind::Regression), so the model-text parser rejects the bare names.
        for line in ["lambdarank", "huber", "quantile alpha:0.9", "rank_xendcg"] {
            let err = ObjectiveKind::parse(line).unwrap_err();
            assert!(
                matches!(err, ModelError::MalformedModel { .. }),
                "{line} must be rejected, got {err:?}"
            );
        }
    }

    #[test]
    fn parse_exp_log_family_convert_kinds() {
        // poisson/gamma/tweedie -> exp ConvertOutput.
        for line in ["poisson", "gamma", "tweedie"] {
            assert_eq!(
                ObjectiveKind::parse(line).unwrap(),
                ObjectiveKind::Poisson,
                "{line} must parse to Poisson (exp ConvertOutput)"
            );
        }
        // exp(input): ln(2) -> 2.
        let mut out = [0.0];
        ObjectiveKind::Poisson.convert(&[2.0f64.ln()], &mut out);
        assert!((out[0] - 2.0).abs() < 1e-12);

        // cross_entropy / xentropy -> sigmoid ConvertOutput.
        for line in ["cross_entropy", "xentropy"] {
            assert_eq!(ObjectiveKind::parse(line).unwrap(), ObjectiveKind::CrossEntropy);
        }
        ObjectiveKind::CrossEntropy.convert(&[0.0], &mut out);
        assert!((out[0] - 0.5).abs() < 1e-12, "sigmoid(0) = 0.5");

        // cross_entropy_lambda / xentlambda -> log1p(exp) ConvertOutput.
        for line in ["cross_entropy_lambda", "xentlambda"] {
            assert_eq!(
                ObjectiveKind::parse(line).unwrap(),
                ObjectiveKind::CrossEntropyLambda
            );
        }
        ObjectiveKind::CrossEntropyLambda.convert(&[0.0], &mut out);
        assert!((out[0] - 2.0f64.ln()).abs() < 1e-12, "log1p(exp(0)) = ln(2)");
    }

    #[test]
    fn parse_rejects_bad_params_without_panic() {
        // multiclass without num_class.
        assert!(ObjectiveKind::parse("multiclass").is_err());
        // binary with non-positive sigmoid.
        assert!(ObjectiveKind::parse("binary sigmoid:0").is_err());
        // garbage sigmoid value.
        assert!(ObjectiveKind::parse("binary sigmoid:abc").is_err());
        // empty line.
        assert!(ObjectiveKind::parse("").is_err());
    }

    // --- Test 7: categorical-split parity (the 03-02 Tree decode) ---

    /// A hand-constructed categorical tree mirroring the C++ layout: node 0 is a
    /// categorical split on feature 0. `cat_boundaries=[0, 1]` (one category
    /// group spanning bitset block [0,1)), `cat_threshold=[0b...]` selecting
    /// categories {1, 3}. find_in_bitset(bits, cat) true -> left_child, else
    /// right_child.
    fn categorical_stump() -> Tree {
        // bitset with bits 1 and 3 set: (1<<1)|(1<<3) = 0b1010 = 10.
        Tree {
            num_leaves: 2,
            num_cat: 1,
            left_child: vec![-1],  // ~(-1) = leaf 0
            right_child: vec![-2], // ~(-2) = leaf 1
            split_feature: vec![0],
            // threshold holds the cat_idx (0) for a categorical split.
            threshold: vec![0.0],
            decision_type: vec![1], // categorical mask bit set
            split_gain: vec![0.0],
            leaf_value: vec![100.0, 200.0],
            leaf_weight: vec![1.0, 1.0],
            leaf_count: vec![1, 1],
            internal_value: vec![0.0],
            internal_weight: vec![0.0],
            internal_count: vec![2],
            cat_boundaries: vec![0, 1],
            cat_threshold: vec![0b1010],
            shrinkage: 1.0,
            is_linear: false,
            leaf_depth: vec![1, 1],
            leaf_parent: vec![0, 0],
            split_feature_inner: vec![-1],
            threshold_in_bin: vec![0],
        }
    }

    #[test]
    fn categorical_decode_routes_known_category() {
        let t = categorical_stump();
        // category 1 -> in bitset -> left_child -> leaf 0 (value 100.0).
        assert_eq!(t.get_leaf(&[1.0]), 0);
        assert_eq!(t.predict(&[1.0]), 100.0);
        // category 3 -> in bitset -> left -> leaf 0.
        assert_eq!(t.predict(&[3.0]), 100.0);
        // category 2 -> NOT in bitset -> right_child -> leaf 1 (value 200.0).
        assert_eq!(t.get_leaf(&[2.0]), 1);
        assert_eq!(t.predict(&[2.0]), 200.0);
        // category 0 -> NOT in bitset -> right -> leaf 1.
        assert_eq!(t.predict(&[0.0]), 200.0);
        // negative / NaN category -> right_child (C++ CategoricalDecision).
        assert_eq!(t.predict(&[-1.0]), 200.0);
        assert_eq!(t.predict(&[f64::NAN]), 200.0);
    }
}
