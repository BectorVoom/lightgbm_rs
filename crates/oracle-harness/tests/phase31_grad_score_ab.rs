//! The GBDT-level GRAD_NS/SCORE_NS same-session ledger + bit-exactness gate for the
//! on-device grad/hess + resident-score path.
//!
//! This is the crate-boundary counterpart to `phase31_ab.rs` (an `lgbm-treelearner` EXAMPLE,
//! which covers the same comparison at the Backend+SerialTreeLearner level). `lgbm-treelearner`
//! does NOT depend on `lgbm-boosting` (the dependency runs the other way), so it CANNOT
//! construct a `Gbdt` or exercise the resident grad/hess branch — and therefore cannot
//! populate the `GRAD_NS`/`SCORE_NS` statics, which only a GBDT-level train drives.
//! `oracle-harness` is the ONLY crate depending on all three of
//! `lgbm-compute`/`lgbm-treelearner`/`lgbm-boosting`, so the GBDT-level GRAD_NS/SCORE_NS
//! measurement lives HERE.
//!
//! What it proves (measure via GRAD_NS/SCORE_NS, NOT the ONDEV_GROW ledger which wraps only
//! the grow call):
//!   - CORRECTNESS GATE (asserted): training a `num_class==1`, no-bagging, no-GOSS `binary`
//!     GBDT with `boosting_on_cuda()=true` (the resident grad/hess + resident-score path
//!     ACTIVE) produces a model + score BIT-IDENTICAL, leaf-for-leaf, to the same config with
//!     `boosting_on_cuda()=false` (byte-unchanged host path) — on the cpu-f64 anchor.
//!   - MEASUREMENT ARTIFACT (printed, NOT asserted for magnitude/direction): both runs' drained
//!     `GRAD_NS`/`SCORE_NS` (ns) are dumped side-by-side in the SAME process (measuring both
//!     arms in one process avoids inventing fake deltas from cross-process noise). No specific
//!     magnitude or direction is asserted — this test asserts only the bit-exactness; the ns
//!     delta is left for a human reader to interpret.
//!
//! The `GRAD_NS`/`SCORE_NS` statics are gated by `LGBM_PHASE_PROF=1` (read-once at process
//! start), so under a plain `cargo test` they stay 0 and the ledger line prints zeros — the
//! bit-exactness gate holds regardless. Run with `LGBM_PHASE_PROF=1 -- --nocapture` to see the
//! real ns values. This is the SOLE test in this binary, so the process-global counters are
//! never contended by a parallel test thread.

use lgbm_boosting::{BoostObjective, Gbdt};
use lgbm_compute::gain::GainConfig;
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::ComputeClientReexport as ComputeClient;
use lgbm_compute::{Backend, BinColumn, CpuBackend};
use lgbm_dataset::bin_mapper::MissingType;
use lgbm_objective::Binary;
use lgbm_treelearner::phase_prof::{GRAD_NS, SCORE_NS};
use lgbm_treelearner::{FeatureColumn, SerialTreeLearner};
use oracle_harness::comparator::compare_exact_f64_bits;
use std::sync::atomic::Ordering;

/// Boosting iterations — enough that the resident grad/hess + score path is exercised across
/// many iters (the whole point of the residency proof), matching `resident_score_ab.rs`.
const ITERS: i32 = 6;

/// A small identity-binned corpus (bin == raw value, no missing values), field-for-field the
/// same fixture `resident_score_ab.rs` trains on. Feature 0 carries a strong monotone signal so
/// every tree splits (`num_leaves > 1`) and the per-leaf grad/hess + AddScore fires each iter.
/// (Duplicated here rather than imported: sibling integration-test binaries are separate
/// compilation units and cannot share a module-private helper.)
fn corpus() -> Vec<FeatureColumn> {
    let f0_bins = vec![0u32, 0, 1, 1, 2, 2, 3, 3];
    let f1_bins = vec![0u32, 1, 0, 1, 0, 1, 0, 1];

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
    vec![f0, f1]
}

/// Binary labels ({0,1}) separable by feature 0 (rows with `f0 >= 2` are class 1) — every tree
/// splits, so the per-leaf grad/hess + AddScore (and its resident mirror) fires each iteration.
fn binary_labels() -> Vec<f32> {
    vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
}

/// The shared permissive gain/cap config (so the tiny corpus splits every iter).
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

/// One arm: train a fresh `binary` num_class==1 GBDT for `ITERS` iterations with the given
/// `boosting_on_cuda` toggle, draining `GRAD_NS`/`SCORE_NS` immediately AROUND this train (clear
/// before, read after) so the returned ns belong to THIS arm only. `resident=true` activates
/// the GBDT-owned resident score + on-device grad/hess path; `false` is the byte-unchanged
/// host path. Returns `(trees, scores, resident_active, grad_ns, score_ns)`.
fn train_arm<B: Backend>(
    backend: &B,
    client: &ComputeClient<B::Runtime>,
    resident: bool,
) -> (Vec<lgbm_model::Tree>, Vec<f64>, Option<bool>, u64, u64) {
    // Clear the process-global counters so the post-train drain reflects ONLY this arm.
    GRAD_NS.swap(0, Ordering::Relaxed);
    SCORE_NS.swap(0, Ordering::Relaxed);

    let features = corpus();
    let labels = binary_labels();
    let num_features = features.len();
    let num_data = labels.len() as i32;

    let mut learner =
        SerialTreeLearner::new(backend, client, gain_config(), 4, -1).with_features(features);
    let mut gbdt = Gbdt::with_objective(
        BoostObjective::Binary(Binary::new(1.0).unwrap()),
        0.3,
        1, // num_class == 1
        num_data,
        true,
        None,
    );
    gbdt.set_boosting_on_cuda(resident);
    gbdt.train(&mut learner, &labels, num_features, ITERS)
        .expect("multi-iteration binary train");

    // Drain (swap-to-read) this arm's grad/hess + score ns.
    let grad_ns = GRAD_NS.swap(0, Ordering::Relaxed);
    let score_ns = SCORE_NS.swap(0, Ordering::Relaxed);
    (
        gbdt.trees().to_vec(),
        gbdt.scores().to_vec(),
        gbdt.resident_score_active(),
        grad_ns,
        score_ns,
    )
}

/// SAME-SESSION GBDT-level ledger: the resident grad/hess +
/// resident-score path is BIT-EXACT to the host path (the asserted correctness gate), and its
/// `GRAD_NS`/`SCORE_NS` are reported side-by-side from ONE process run (the printed measurement
/// artifact — no magnitude/direction is asserted; do not assume a transfer number without
/// measuring).
#[test]
fn resident_grad_hess_score_same_session_ledger_bit_exact() {
    let backend = CpuBackend;
    let client = cpu_client();

    // SAME process, back-to-back: resident arm active, then the byte-unchanged host arm.
    let (dev_trees, dev_scores, dev_active, dev_grad, dev_score) =
        train_arm(&backend, &client, true);
    let (host_trees, host_scores, host_active, host_grad, host_score) =
        train_arm(&backend, &client, false);

    let phase_prof_on = std::env::var("LGBM_PHASE_PROF").map(|v| v == "1").unwrap_or(false);

    // MEASUREMENT ARTIFACT — printed, not asserted.
    println!(
        "[phase31_grad_score_ab] resident_on: grad_ns={dev_grad} score_ns={dev_score} | \
         host: grad_ns={host_grad} score_ns={host_score} | phase_prof={phase_prof_on} \
         (ns are 0 unless LGBM_PHASE_PROF=1 at process start; magnitudes are NOT asserted — \
         ODS-04: do not assume a transfer number without measuring)"
    );

    // Activation sanity: the binary num_class==1 envelope MUST activate the resident path on the
    // `true` arm and MUST stay host on the `false` arm.
    assert_eq!(
        dev_active,
        Some(true),
        "the binary num_class==1 envelope must activate Plan 04's resident grad/hess + score path"
    );
    assert_eq!(
        host_active,
        Some(false),
        "the boosting_on_cuda=false arm must take the byte-unchanged host path"
    );

    // Non-vacuous: a real 6-iter binary train actually moved the score.
    assert!(
        host_scores.iter().any(|&s| s.abs() > 1e-6),
        "a real multi-iter binary train must move the score (non-vacuous ledger)"
    );

    // THE CORRECTNESS GATE: bit-identical model + score vs the host path.
    assert_eq!(
        dev_trees, host_trees,
        "resident on-device grad/hess + score model diverged from the host path (leaf-for-leaf) \
         across the full binary train"
    );
    compare_exact_f64_bits(&dev_scores, &host_scores).unwrap_or_else(|m| {
        panic!(
            "resident-arm score != host score after a full binary train (cpu f64 anchor); Plan \
             04's persistent resident score drifted from the host accumulation: {m:?}"
        )
    });

    // When the phase-prof gate IS on, the ledger must be non-vacuous (the drain actually saw the
    // grad/hess + score timers fire) — otherwise the printed measurement is meaningless. This is
    // a gate on the LEDGER's honesty, NOT on any delta magnitude.
    if phase_prof_on {
        assert!(
            dev_grad > 0 && dev_score > 0 && host_grad > 0 && host_score > 0,
            "LGBM_PHASE_PROF=1 but a grad/score counter drained to 0 (dev_grad={dev_grad} \
             dev_score={dev_score} host_grad={host_grad} host_score={host_score}) — the \
             same-session ledger is vacuous"
        );
    }
}
