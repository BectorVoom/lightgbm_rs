//! Phase-20 20-04 (ODL-16, D-01/D-06 layer 2) — the resident-score A/B.
//!
//! Proves the SECOND D-06 parity layer: after a FULL multi-iteration GBDT train the
//! resident `cuda_score_` (kept on device across the whole train via the
//! `boosting_on_cuda_` toggle — the §16 Shrinkage → UpdateScore(§11) → optional
//! RenewTreeOutput → Metric.Eval order) equals the pure-host `score_` accumulation.
//!
//! Two arms driven on the SAME cpu backend inside ONE process (the
//! `cuda_on_device_enabled()` env is a process-global `OnceLock`, so per-arm env
//! toggling is impossible — the arms are selected via the `Gbdt::set_boosting_on_cuda`
//! driver seam the 20-04 wiring exposes):
//!   - HOST arm  (`set_boosting_on_cuda(false)`): the byte-unchanged partition-scatter
//!     score path — the ORACLE.
//!   - RESIDENT arm (`set_boosting_on_cuda(true)`): the per-leaf `AddScore` routes
//!     through the Phase-18 `add_prediction_to_score_on_device` tree-walk delegate
//!     (D-02), keeping the score resident and mirroring it back to `score_`.
//!
//! Anchor discipline (D-05/D-07, def-f8u-01): the reference is ALWAYS the host/cpu-fold
//! accumulation — NEVER a second GPU f32 path. On the cpu f64 anchor the two arms are
//! BIT-EXACT (`compare_exact_f64_bits`); the optional `rocm` gpu cell holds the resident
//! hip arm to the ~1e-6 f32 envelope AGAINST the same cpu-anchor host accumulation.
//!
//! Scope: the L2 continuous-feature proving slice (identity-binned, numeric splits, no
//! missing values, no RenewTreeOutput refit) — the regime where the device tree-walk is
//! bit-exact to the host partition scatter.

use lgbm_boosting::Gbdt;
use lgbm_compute::gain::GainConfig;
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::ComputeClientReexport as ComputeClient;
use lgbm_compute::{Backend, BinColumn, CpuBackend};
use lgbm_dataset::bin_mapper::MissingType;
use lgbm_objective::Objective;
use lgbm_treelearner::{FeatureColumn, SerialTreeLearner};
use oracle_harness::comparator::compare_exact_f64_bits;

/// The number of boosting iterations the A/B run trains (a handful of L2 continuous
/// trees — enough that the resident score buffer is accumulated across many iters, the
/// whole point of the residency proof).
const ITERS: i32 = 6;

/// A small identity-binned L2 continuous corpus (bin == raw value, no missing values).
/// Feature 0 carries a strong monotone signal (labels rise with `f0`) so every tree
/// splits (`num_leaves > 1`) and the per-leaf `AddScore` is exercised each iter; feature
/// 1 is a weak secondary. Returns `(features, labels)`.
fn corpus() -> (Vec<FeatureColumn>, Vec<f32>) {
    // 8 rows, 2 features.
    let f0_bins = vec![0u32, 0, 1, 1, 2, 2, 3, 3];
    let f1_bins = vec![0u32, 1, 0, 1, 0, 1, 0, 1];
    let labels = vec![1.0f32, 1.0, 3.0, 3.0, 8.0, 8.0, 12.0, 13.0];

    let f0 = FeatureColumn {
        bins: BinColumn::new(f0_bins, 4),
        num_bin: 4,
        offset: lgbm_treelearner::offset_for_most_freq_bin(0),
        min_bin: 0,
        max_bin: 3,
        default_bin: 4,
        most_freq_bin: 0,
        missing_type: MissingType::None,
        bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5],
        real_feature_index: 0,
        ..Default::default()
    };
    let f1 = FeatureColumn {
        bins: BinColumn::new(f1_bins, 2),
        num_bin: 2,
        offset: lgbm_treelearner::offset_for_most_freq_bin(0),
        min_bin: 0,
        max_bin: 1,
        default_bin: 2,
        most_freq_bin: 0,
        missing_type: MissingType::None,
        bin_upper_bound: vec![0.5, 1.5],
        real_feature_index: 1,
        ..Default::default()
    };
    (vec![f0, f1], labels)
}

/// The shared gain/cap config (permissive so the tiny corpus splits every iter).
fn gain_config() -> GainConfig {
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

/// Train `ITERS` L2 continuous trees on `backend`/`client`, returning the post-train
/// f64 `score_` buffer. `resident` selects the arm via the 20-04 `set_boosting_on_cuda`
/// driver seam: `true` keeps `cuda_score_` resident (device tree-walk delegate), `false`
/// is the byte-unchanged host partition-scatter reference.
fn train_scores<B: Backend>(
    backend: &B,
    client: &ComputeClient<B::Runtime>,
    resident: bool,
) -> Vec<f64> {
    let (features, labels) = corpus();
    let num_features = features.len();
    let num_data = labels.len() as i32;

    let mut learner =
        SerialTreeLearner::new(backend, client, gain_config(), 4, -1).with_features(features);
    // Regression L2 spine, boost_from_average on (the init-score add is host in both
    // arms, so both start from the identical baseline).
    let mut gbdt = Gbdt::new(
        Objective::Regression { sqrt: false },
        0.3,
        1,
        num_data,
        true,
        None,
    );
    gbdt.set_boosting_on_cuda(resident);
    assert_eq!(
        gbdt.boosting_on_cuda(),
        resident,
        "the driver seam must force the arm's boosting_on_cuda toggle"
    );
    gbdt.train(&mut learner, &labels, num_features, ITERS)
        .expect("multi-iteration train");
    gbdt.scores().to_vec()
}

/// D-06 layer 2 — after a full multi-iteration train the RESIDENT `cuda_score_` equals
/// the pure-HOST `score_` BIT-FOR-BIT on the cpu f64 anchor.
#[test]
fn resident_score_matches_host_after_full_train_cpu_anchor() {
    let backend = CpuBackend;
    let client = cpu_client();

    // ORACLE: the byte-unchanged host partition-scatter accumulation.
    let host = train_scores(&backend, &client, false);
    // RESIDENT: the device tree-walk delegate kept resident across the whole train.
    let resident = train_scores(&backend, &client, true);

    // Sanity: the train actually moved the score (non-trivial residency proof).
    assert_eq!(host.len(), 8, "one f64 score per row");
    assert!(
        host.iter().any(|&s| s.abs() > 1e-6),
        "the score must be non-zero after a real multi-iter train"
    );
    // Monotone signal survived: the label-1 rows score below the label-13 rows.
    assert!(host[0] < host[7], "host score must track the label signal");

    compare_exact_f64_bits(&resident, &host).unwrap_or_else(|m| {
        panic!(
            "resident cuda_score_ != host score_ after a full train (cpu f64 anchor); \
             the resident device tree-walk diverged from the host partition scatter on \
             the identity-binned L2 slice: {m:?}"
        )
    });
}

/// The env-unset default `Gbdt` reports the host path (D-09): with `LGBM_CUDA_ON_DEVICE`
/// unset the internal score updater defaults `boosting_on_cuda = false`, so the GBDT
/// score/eval path is byte-unchanged. (This cell forces neither arm; it reads the
/// env-derived default.)
#[test]
fn env_unset_default_is_host_path() {
    // Only meaningful when the env is actually unset; when the A/B is run under
    // LGBM_CUDA_ON_DEVICE=1 the default flips on by design, so skip the assertion.
    if lgbm_compute::cuda_on_device_enabled() {
        eprintln!(
            "resident_score_ab: SKIP env-default check — LGBM_CUDA_ON_DEVICE=1 flips the \
             default resident ON by design (D-09)."
        );
        return;
    }
    let num_data = 8i32;
    let gbdt = Gbdt::new(
        Objective::Regression { sqrt: false },
        0.3,
        1,
        num_data,
        true,
        None,
    );
    assert!(
        !gbdt.boosting_on_cuda(),
        "LGBM_CUDA_ON_DEVICE unset ⇒ GBDT boosting_on_cuda must default OFF \
         (byte-unchanged host score/eval path, D-09/ODL-19)"
    );
}

/// Optional ROCm/HIP gpu cell (opt-in `--features rocm`): the resident hip arm held to
/// the ~1e-6 f32 envelope AGAINST the cpu-anchor host accumulation — NEVER a second GPU
/// path (D-07/def-f8u-01). The cpu f64 host arm is the anchor; the hip resident arm is
/// the f32 candidate.
#[cfg(feature = "rocm")]
mod hip {
    use super::*;
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::RocmBackend;

    /// The f32 leaf-accumulation envelope (mirrors `learner_parity::ROCM_LEAF_VALUE_TOL`).
    const ROCM_SCORE_TOL: f64 = 1e-5;

    #[test]
    fn resident_score_matches_cpu_anchor_on_hip_within_envelope() {
        // Anchor: the cpu f64 host accumulation (NEVER a second GPU path).
        let cpu_backend = CpuBackend;
        let cpu = cpu_client();
        let anchor = train_scores(&cpu_backend, &cpu, false);

        // Candidate: the resident hip arm.
        let hip_backend = RocmBackend::default();
        let hip = rocm_client();
        let resident = train_scores(&hip_backend, &hip, true);

        assert_eq!(resident.len(), anchor.len(), "score length parity");
        for (i, (&r, &a)) in resident.iter().zip(anchor.iter()).enumerate() {
            let d = (r - a).abs();
            assert!(
                d <= ROCM_SCORE_TOL,
                "resident hip score[{i}] = {r} diverged from cpu-anchor host {a} by {d} \
                 > {ROCM_SCORE_TOL} (f32 accumulation envelope)"
            );
        }
    }
}
