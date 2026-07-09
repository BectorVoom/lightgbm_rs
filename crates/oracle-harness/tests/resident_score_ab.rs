//! The resident-score A/B.
//!
//! Proves that after a FULL multi-iteration GBDT train the resident `cuda_score_`
//! (kept on device across the whole train via the `boosting_on_cuda_` toggle — the §16
//! Shrinkage → UpdateScore(§11) → optional RenewTreeOutput → Metric.Eval order) equals
//! the pure-host `score_` accumulation.
//!
//! Two arms driven on the SAME cpu backend inside ONE process (the
//! `cuda_on_device_enabled()` env is a process-global `OnceLock`, so per-arm env
//! toggling is impossible — the arms are selected via the `Gbdt::set_boosting_on_cuda`
//! driver seam):
//!   - HOST arm  (`set_boosting_on_cuda(false)`): the byte-unchanged partition-scatter
//!     score path — the ORACLE.
//!   - RESIDENT arm (`set_boosting_on_cuda(true)`): the per-leaf `AddScore` routes
//!     through the `add_prediction_to_score_on_device` tree-walk delegate, keeping the
//!     score resident and mirroring it back to `score_`.
//!
//! Anchor discipline: the reference is ALWAYS the host/cpu-fold accumulation — NEVER a
//! second GPU f32 path. On the cpu f64 anchor the two arms are BIT-EXACT
//! (`compare_exact_f64_bits`); the optional `rocm` gpu cell holds the resident hip arm to
//! the ~1e-6 f32 envelope AGAINST the same cpu-anchor host accumulation.
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
/// f64 `score_` buffer. `resident` selects the arm via the `set_boosting_on_cuda`
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

/// After a full multi-iteration train the RESIDENT `cuda_score_` equals
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

/// Train `ITERS` L2 continuous trees on `backend`/`client` using the ENV-DERIVED default
/// `boosting_on_cuda` (NO explicit `set_boosting_on_cuda`) — the path a normal caller takes.
/// With `LGBM_CUDA_ON_DEVICE` unset the default is `false`, so this is the byte-unchanged host
/// score path; the test below proves it equals the explicit host arm bit-for-bit.
fn train_scores_env_default<B: Backend>(
    backend: &B,
    client: &ComputeClient<B::Runtime>,
) -> Vec<f64> {
    let (features, labels) = corpus();
    let num_features = features.len();
    let num_data = labels.len() as i32;
    let mut learner =
        SerialTreeLearner::new(backend, client, gain_config(), 4, -1).with_features(features);
    let mut gbdt = Gbdt::new(
        Objective::Regression { sqrt: false },
        0.3,
        1,
        num_data,
        true,
        None,
    );
    // NOTE: deliberately NO `set_boosting_on_cuda` — take the env-derived default.
    gbdt.train(&mut learner, &labels, num_features, ITERS)
        .expect("multi-iteration train (env-default arm)");
    gbdt.scores().to_vec()
}

/// With `LGBM_CUDA_ON_DEVICE` UNSET, the env-DERIVED default GBDT train
/// produces a score BIT-IDENTICAL to the explicit byte-unchanged host arm. This proves the
/// resident rewrite is CONFINED to the resident arm (reached only under the
/// env-gated `boosting_on_cuda_` seam) and does NOT leak into the default path — "every backend
/// byte unchanged with the env unset". When the A/B is run under `LGBM_CUDA_ON_DEVICE=1` the
/// default flips resident ON by design, so the assertion is skipped there.
#[test]
fn sc4_env_unset_default_train_is_byte_identical_to_host_arm() {
    if lgbm_compute::cuda_on_device_enabled() {
        eprintln!(
            "resident_score_ab: SKIP SC-4 byte-identical check — LGBM_CUDA_ON_DEVICE=1 flips the \
             default resident ON by design (D-09)."
        );
        return;
    }
    let backend = CpuBackend;
    let client = cpu_client();

    // The env-derived default arm (no explicit toggle) vs the EXPLICIT byte-unchanged host arm.
    let default_scores = train_scores_env_default(&backend, &client);
    let host = train_scores(&backend, &client, false);

    // Sanity: a real multi-iter train moved the score (non-vacuous proof).
    assert!(
        host.iter().any(|&s| s.abs() > 1e-6),
        "the score must be non-zero after a real multi-iter train"
    );
    compare_exact_f64_bits(&default_scores, &host).unwrap_or_else(|m| {
        panic!(
            "SC-4: env-unset default train score != explicit host arm (byte-unchanged path); the \
             28-03/28-04 resident rewrite leaked into the LGBM_CUDA_ON_DEVICE-unset default path: {m:?}"
        )
    });
}

/// Build the on-device grow driver's `GrowFeature` carriers from the same [`corpus`]
/// the resident-score A/B trains on — a field-by-field mirror of the spine columns
/// (exactly what `SerialTreeLearner`'s on_device_eligible block builds at the seam),
/// with the config-default categorical scalars (inert on this numeric slice).
fn grow_features() -> Vec<lgbm_compute::GrowFeature> {
    let (features, _labels) = corpus();
    features
        .iter()
        .map(|f| lgbm_compute::GrowFeature {
            bins: f.bins.clone(),
            num_bin: f.num_bin,
            offset: f.offset,
            min_bin: f.min_bin,
            max_bin: f.max_bin,
            default_bin: f.default_bin,
            most_freq_bin: f.most_freq_bin,
            missing_type: f.missing_type,
            bin_upper_bound: f.bin_upper_bound.clone(),
            real_feature_index: f.real_feature_index,
            bin_type: f.bin_type,
            bin_to_category: f.bin_to_category.clone(),
            cat_smooth: 10.0,
            cat_l2: 10.0,
            max_cat_threshold: 32,
            max_cat_to_onehot: 4,
            min_data_per_group: 100,
        })
        .collect()
}

/// The COMBINED on-device GROW → on-device SCORE path
/// (not just the score delegate in isolation) is BIT-EXACT to the host partition
/// scatter on the cpu-f64 anchor.
///
/// Grows the whole L2 tree on device via the driver
/// ([`grow_tree_on_device_driver`], the cubecl-cpu STRUCTURE anchor), then applies
/// its per-leaf `AddScore` on device via the resident §11 partition scatter
/// [`add_prediction_to_score_on_device_resident`] over the resident row→leaf layout
/// the grow produced. The oracle is the byte-unchanged host partition scatter
/// [`SerialTreeLearner::add_prediction_to_score`] over the SAME grown `(tree,
/// layout)`. This proves the on-device score move is correctness-preserving before a
/// real-CUDA A/B measures the speedup (single-process, no `LGBM_CUDA_ON_DEVICE`
/// OnceLock fight — the driver runs regardless of the env).
#[test]
fn combined_on_device_grow_and_score_matches_host_scatter_cpu_anchor() {
    use lgbm_compute::add_prediction_to_score_on_device_resident;
    use lgbm_compute::kernels::grow_driver::grow_tree_on_device_driver;
    use lgbm_treelearner::DataPartition;

    let backend = CpuBackend;
    let client = cpu_client();
    let (features, labels) = corpus();
    let nd = labels.len();

    // L2 gradients from a zero init (grad = score − label = −label; hess = 1). Any
    // g/h that splits works — the score parity is over WHATEVER `(tree, layout)` grows.
    let gradients: Vec<f32> = labels.iter().map(|&l| -l).collect();
    let hessians: Vec<f32> = vec![1.0f32; nd];

    // GROW on device → grown tree + the resident row→leaf partition layout.
    let gf = grow_features();
    let (tree, layout) =
        grow_tree_on_device_driver(&backend, &client, &gradients, &hessians, &gf, 4, -1)
            .expect("on-device grow");
    assert!(
        tree.num_leaves > 1,
        "the monotone corpus must split (non-trivial score)"
    );

    // SCORE on device: resident §11 AddPredictionToScore partition scatter.
    let device = add_prediction_to_score_on_device_resident(&client, &layout, &tree.leaf_value)
        .expect("on-device resident score scatter");

    // ORACLE: the byte-unchanged HOST partition scatter over the SAME (tree, layout).
    let part = DataPartition::from_payload(layout);
    let learner =
        SerialTreeLearner::new(&backend, &client, gain_config(), 4, -1).with_features(features);
    let mut host = vec![0.0f64; nd];
    learner.add_prediction_to_score(&tree, &part, &mut host);

    assert!(
        host.iter().any(|&s| s.abs() > 1e-6),
        "the grown tree must move the score (non-trivial parity proof)"
    );
    compare_exact_f64_bits(&device, &host).unwrap_or_else(|m| {
        panic!(
            "combined on-device grow+score != host partition scatter (cpu f64 anchor); \
             the resident §11 scatter diverged from the host add_prediction_to_score on \
             the identity-binned L2 slice: {m:?}"
        )
    });
}

/// The per-row leaf map derived ON DEVICE
/// from the resident leaf-grouped partition layout equals the retired host `O(num_data)`
/// inversion loop, BIT-FOR-BIT, on the cpu-f64 anchor. Uses a multi-leaf fixture with
/// rows deliberately shuffled inside each leaf (so the derivation is index-driven, not
/// position-driven), and asserts the covered-once total-function contract (every row
/// mapped to a real leaf, no `-1` sentinel survives on a fully-covered partition).
#[test]
fn resident_leaf_map_device_matches_host_inversion() {
    use lgbm_compute::derive_leaf_map_device;

    let client = cpu_client();
    // 6 rows, 3 leaves. `indices` is leaf-grouped (leaf 0 = rows {4,1}, leaf 1 = {0,5},
    // leaf 2 = {3,2}) — intentionally shuffled within each leaf.
    let indices = vec![4u32, 1, 0, 5, 3, 2];
    let leaf_begin = vec![0i32, 2, 4];
    let leaf_count = vec![2i32, 2, 2];
    let num_data = 6usize;

    let device = derive_leaf_map_device(&client, &indices, &leaf_begin, &leaf_count, num_data)
        .expect("device leaf-map derivation from the resident partition ranges");

    // ORACLE: the retired host inversion loop (add_prediction_to_score_on_device_resident
    // 1248-1281): `data_index_to_leaf[row] = leaf` for every leaf's rows.
    let mut host = vec![-1i32; num_data];
    for (leaf, (&b, &c)) in leaf_begin.iter().zip(leaf_count.iter()).enumerate() {
        for &row in &indices[b as usize..(b + c) as usize] {
            host[row as usize] = leaf as i32;
        }
    }

    assert_eq!(
        device, host,
        "device-derived data_index_to_leaf must equal the host inversion output"
    );
    // Covered-once total function over [0,num_data): no uncovered `-1` slot survives.
    assert!(
        device.iter().all(|&l| l >= 0),
        "every row must be covered exactly once (no -1 sentinel on a full partition)"
    );
}

/// A malformed partition layout whose permutation
/// contains a row id `>= num_data` yields a clean `ComputeError` (mirroring the retired
/// host inversion's `r >= num_data` guard) instead of an out-of-bounds device write inside
/// the `unsafe` scatter launch (UB). The bound is enforced at the host boundary
/// (`upload_leaf_map_inputs`) BEFORE any launch, so no partial device-memory corruption is
/// reachable through the `pub` `derive_leaf_map_device` / `derive_leaf_map_device_handle`.
#[test]
fn derive_leaf_map_rejects_out_of_range_row_id() {
    use lgbm_compute::derive_leaf_map_device;

    let client = cpu_client();
    // 6 rows, 3 leaves — but leaf 2's second row id is 6, one PAST the end of a 6-row
    // `data_index_to_leaf` buffer (valid row ids are `[0, 6)`). The per-leaf `[begin, begin+
    // count) <= indices.len()` range check passes (the sub-ranges cover all 6 positions);
    // only the row-id VALUE bound catches the `6`.
    let indices = vec![4u32, 1, 0, 5, 3, 6];
    let leaf_begin = vec![0i32, 2, 4];
    let leaf_count = vec![2i32, 2, 2];
    let num_data = 6usize;

    let result = derive_leaf_map_device(&client, &indices, &leaf_begin, &leaf_count, num_data);
    let err = result.expect_err(
        "an out-of-range partition row id (6 >= num_data 6) must be rejected with a clean \
         ComputeError, NOT launch the out-of-bounds device scatter (WR-02)",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("row id") && msg.contains("num_data"),
        "the rejection must name the per-value invariant (got: {msg})"
    );
}

/// The DEFERRED-readback resident score
/// mirror accumulated ACROSS trees (one final readback) is BIT-EXACT to the EAGER
/// per-tree-readback accumulation, on the cpu-f64 anchor.
///
/// EAGER (the per-tree contract): for each grown tree, call
/// `add_prediction_to_score_on_device_resident` and sum the returned `num_data` f64 delta
/// into a host accumulator — a full readback PER TREE.
///
/// DEFERRED (the resident approach): allocate ONE `ResidentScore` mirror, apply every tree's
/// `AddScore` on device (`add_tree_on_device`, NO intermediate readback), and read the
/// resident buffer back ONCE at the end. The two must agree bit-for-bit: the mirror adds
/// the same exact f64 leaf values into `score[row]` in the same tree order the eager host
/// sum does.
#[test]
fn resident_score_deferred_readback_matches_eager_cpu_anchor() {
    use lgbm_compute::{add_prediction_to_score_on_device_resident, ResidentScore};
    let client = cpu_client();
    let num_data = 6usize;

    // Three distinct "trees" over the SAME 6 rows: different leaf groupings + distinct
    // exact-representable f64 leaf values (no rounding — the scatter must reproduce them
    // bit-for-bit). Rows are shuffled within each leaf (index-driven scatter).
    let trees: Vec<(lgbm_dataset::LeafPartitionLayout, Vec<f64>)> = vec![
        (
            lgbm_dataset::LeafPartitionLayout {
                num_data: 6,
                indices: vec![4, 1, 0, 5, 3, 2],
                leaf_begin: vec![0, 2, 4],
                leaf_count: vec![2, 2, 2],
            },
            vec![-0.5f64, 0.25, 1.75],
        ),
        (
            lgbm_dataset::LeafPartitionLayout {
                num_data: 6,
                indices: vec![0, 2, 4, 1, 3, 5],
                leaf_begin: vec![0, 3],
                leaf_count: vec![3, 3],
            },
            vec![0.125f64, -0.375],
        ),
        (
            lgbm_dataset::LeafPartitionLayout {
                num_data: 6,
                indices: vec![5, 4, 3, 2, 1, 0],
                leaf_begin: vec![0],
                leaf_count: vec![6],
            },
            vec![2.0f64],
        ),
    ];

    // EAGER: full readback per tree, summed on host (the prior contract).
    let mut eager = vec![0.0f64; num_data];
    for (layout, leaf_values) in &trees {
        let delta = add_prediction_to_score_on_device_resident(&client, layout, leaf_values)
            .expect("eager per-tree resident score");
        for (i, d) in delta.iter().enumerate() {
            eager[i] += *d;
        }
    }

    // DEFERRED: one resident mirror, no intermediate readback, single final read.
    let mut mirror = ResidentScore::new(&client, num_data);
    for (layout, leaf_values) in &trees {
        mirror
            .add_tree_on_device(&client, layout, leaf_values)
            .expect("deferred resident add_tree_on_device");
    }
    let deferred = mirror.read_resident_score(&client);

    // Sanity: multiple trees actually moved the score.
    assert!(
        eager.iter().any(|&s| s.abs() > 1e-6),
        "the three trees must move the score (non-trivial residency proof)"
    );
    compare_exact_f64_bits(&deferred, &eager).unwrap_or_else(|m| {
        panic!(
            "deferred resident-mirror score != eager per-tree-readback accumulation \
             (cpu f64 anchor); the across-trees device accumulation diverged from the \
             per-tree host sum: {m:?}"
        )
    });
}

/// The env-unset default `Gbdt` reports the host path: with `LGBM_CUDA_ON_DEVICE`
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

/// Binary labels ({0,1}) for the same [`corpus`] rows, separable by feature 0 (f0 bins
/// `[0,0,1,1,2,2,3,3]` — rows with `f0 >= 2` are class 1). Every tree splits, so the
/// per-leaf `AddScore` (and its resident mirror) is exercised each of the 6 iterations.
fn binary_labels() -> Vec<f32> {
    vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
}

/// Train a fresh binary GBDT for `ITERS` iterations on `backend`/`client`, returning
/// `(trees, scores, resident_active)`. `resident` selects the arm via the
/// `set_boosting_on_cuda` seam: `true` is the GBDT-owned resident score
/// + on-device grad/hess path, `false` the byte-unchanged host path.
fn train_binary<B: Backend>(
    backend: &B,
    client: &ComputeClient<B::Runtime>,
    resident: bool,
) -> (Vec<lgbm_model::Tree>, Vec<f64>, Option<bool>) {
    use lgbm_boosting::BoostObjective;
    use lgbm_objective::Binary;

    let (features, _l2_labels) = corpus();
    let labels = binary_labels();
    let num_features = features.len();
    let num_data = labels.len() as i32;

    let mut learner =
        SerialTreeLearner::new(backend, client, gain_config(), 4, -1).with_features(features);
    let mut gbdt = Gbdt::with_objective(
        BoostObjective::Binary(Binary::new(1.0).unwrap()),
        0.3,
        1,
        num_data,
        true,
        None,
    );
    gbdt.set_boosting_on_cuda(resident);
    gbdt.train(&mut learner, &labels, num_features, ITERS)
        .expect("multi-iteration binary train");
    (
        gbdt.trees().to_vec(),
        gbdt.scores().to_vec(),
        gbdt.resident_score_active(),
    )
}

/// The GBDT-owned, TRAIN-LIFETIME resident score +
/// on-device grad/hess path produces a model BIT-IDENTICAL (every leaf value, every
/// split) to the host path across a 6-iteration binary train. This proves the resident
/// score never drifts across a persistent multi-tree train (not just a single tree): the
/// score is uploaded to the device ONCE (after BoostFromAverage), then accumulated in
/// place tree-by-tree, and read only via the on-device grad/hess launcher — yet stays
/// bit-exact to the host `score_updater.scores()` accumulation on the cpu-f64 anchor.
///
/// Reset-discipline note: this envelope sidesteps the
/// length-mismatch/aliasing class entirely. Bagging/GOSS are the only mid-train
/// row-count changers and are EXCLUDED from activation (they fall back to host, proven
/// by `resident_score_falls_back_completely_for_bagging_and_multiclass`), so the resident
/// buffer never has to reset on a row-subset swap. The `ResidentScore` is a per-GBDT-
/// instance field, so a fresh `Gbdt` (a different `Vec<FeatureColumn>` / feature set)
/// gets a fresh resident buffer with no cross-train aliasing — there is no GBDT-level
/// `with_features` mid-train swap at this layer, so the closest boundary (independent
/// GBDT constructions, each matching its host arm) is what this crate can drive.
#[test]
fn resident_score_persists_across_full_train_bit_exact() {
    let backend = CpuBackend;
    let client = cpu_client();

    let (host_trees, host_scores, host_active) = train_binary(&backend, &client, false);
    let (dev_trees, dev_scores, dev_active) = train_binary(&backend, &client, true);

    assert_eq!(
        dev_active,
        Some(true),
        "the binary num_class==1 envelope must activate the resident path"
    );
    assert_eq!(host_active, Some(false), "the host arm must not activate");

    // A real 6-iter train moved the score (non-vacuous residency proof).
    assert!(
        host_scores.iter().any(|&s| s.abs() > 1e-6),
        "a real multi-iter binary train must move the score"
    );
    // BIT-EXACT model + score vs the host path.
    assert_eq!(
        dev_trees, host_trees,
        "resident on-device grad/hess model diverged from the host path across the full train"
    );
    compare_exact_f64_bits(&dev_scores, &host_scores).unwrap_or_else(|m| {
        panic!(
            "resident-arm score != host score after a full binary train (cpu f64 anchor); \
             the persistent resident score drifted from the host accumulation: {m:?}"
        )
    });
}

/// Bagging and multiclass fall back to the host path for the
/// ENTIRE train (`resident_score_active() == Some(false)`), never a partial-resident
/// state. Proves the num_class==1 / no-bagging / no-GOSS scope guard is complete, not
/// partial (no silent host/device divergence).
#[test]
fn resident_score_falls_back_completely_for_bagging_and_multiclass() {
    use lgbm_boosting::BoostObjective;
    use lgbm_compute::gain::GainConfig;
    use lgbm_objective::MulticlassSoftmax;

    let backend = CpuBackend;
    let client = cpu_client();
    let (features, _l2_labels) = corpus();
    let num_features = features.len();
    let labels = binary_labels();
    let num_data = labels.len() as i32;

    // (a) MULTICLASS (num_class=3) — never activates, even with boosting_on_cuda=true.
    let mc_labels = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0];
    let mut mc_learner = SerialTreeLearner::new(&backend, &client, gain_config(), 4, -1)
        .with_features(features.clone());
    let mut mc = Gbdt::with_objective(
        BoostObjective::Multiclass(MulticlassSoftmax::new(3, &mc_labels).unwrap()),
        0.1,
        3,
        num_data,
        true,
        None,
    );
    mc.set_boosting_on_cuda(true);
    mc.train(&mut mc_learner, &mc_labels, num_features, 3)
        .expect("multiclass train");
    assert_eq!(
        mc.resident_score_active(),
        Some(false),
        "multiclass (num_class=3) must fall back to host for the ENTIRE train"
    );

    // (b) BAGGING — never activates; the boosting_on_cuda toggle is byte-identical
    // (both arms take the host subset path, which never touches the resident branch).
    let bag_cfg = GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 0.0,
        ..Default::default()
    };
    let train_bag = |on: bool| {
        let bc = lgbm_boosting::sample_strategy::BaggingConfig::new(0.5, 1.0, 1.0, 1, 3, false)
            .unwrap();
        let strat = lgbm_boosting::sample_strategy::BaggingSampleStrategy::reset_sample_config(
            bc, num_data, &labels,
        );
        let mut learner = SerialTreeLearner::new(&backend, &client, bag_cfg.clone(), 4, -1)
            .with_features(features.clone());
        let mut gbdt =
            Gbdt::new(lgbm_objective::Objective::Regression { sqrt: false }, 0.3, 1, num_data, true, None)
                .with_bagging(strat, features.clone());
        gbdt.set_boosting_on_cuda(on);
        gbdt.train(&mut learner, &labels, num_features, 3)
            .expect("bagging train");
        (gbdt.trees().to_vec(), gbdt.resident_score_active())
    };
    let (bag_on_trees, bag_on_active) = train_bag(true);
    let (bag_off_trees, bag_off_active) = train_bag(false);
    assert_eq!(bag_on_active, Some(false), "bagging must fall back to host completely");
    assert_eq!(bag_off_active, Some(false), "bagging host arm inactive too");
    assert_eq!(
        bag_on_trees, bag_off_trees,
        "bagging: the boosting_on_cuda toggle must be byte-identical (no resident path)"
    );
}

/// Optional ROCm/HIP gpu cell (opt-in `--features rocm`): the resident hip arm held to
/// the ~1e-6 f32 envelope AGAINST the cpu-anchor host accumulation — NEVER a second GPU
/// path. The cpu f64 host arm is the anchor; the hip resident arm is
/// the f32 candidate.
#[cfg(feature = "rocm")]
mod hip {
    use super::*;
    use lgbm_compute::runtime::rocm_client;
    use lgbm_compute::RocmBackend;

    /// The f32 leaf-accumulation envelope (mirrors `learner_parity::ROCM_LEAF_VALUE_TOL`).
    /// The ROCm/HIP arm is held to this ~1e-6-order envelope (the CLAUDE.md ROCm bar is
    /// ~1e-6; f32 leaf accumulation widens it to `1e-5`) AGAINST the cpu-f64 host anchor
    /// — NEVER bit-exact to a second GPU f32 path.
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

    /// The COMBINED on-device GROW → on-device SCORE path
    /// on hip: grow the L2 tree on the hip runtime, score it via the resident §11
    /// partition scatter, and hold that on-device arm to the ~1e-6-order f32 envelope
    /// (`ROCM_SCORE_TOL`, CLAUDE.md ROCm bar ~1e-6) AGAINST the cpu-f64 host partition
    /// scatter — NEVER bit-exact to a second GPU f32 path. Mirrors the
    /// cpu-anchor combined test, with the hip runtime as the on-device candidate.
    #[test]
    fn combined_on_device_grow_and_score_on_hip_within_envelope() {
        use lgbm_compute::add_prediction_to_score_on_device_resident;
        use lgbm_compute::kernels::grow_driver::grow_tree_on_device_driver;
        use lgbm_treelearner::DataPartition;

        let (features, labels) = corpus();
        let nd = labels.len();
        let gradients: Vec<f32> = labels.iter().map(|&l| -l).collect();
        let hessians: Vec<f32> = vec![1.0f32; nd];
        let gf = grow_features();

        // CANDIDATE: grow + score on the hip runtime.
        let hip_backend = RocmBackend::default();
        let hip = rocm_client();
        let (tree, layout) =
            grow_tree_on_device_driver(&hip_backend, &hip, &gradients, &hessians, &gf, 4, -1)
                .expect("hip on-device grow");
        let device = add_prediction_to_score_on_device_resident(&hip, &layout, &tree.leaf_value)
            .expect("hip on-device resident score");

        // ANCHOR: the cpu-f64 host partition scatter over the cpu-grown tree.
        let cpu_backend = CpuBackend;
        let cpu = cpu_client();
        let (cpu_tree, cpu_layout) =
            grow_tree_on_device_driver(&cpu_backend, &cpu, &gradients, &hessians, &gf, 4, -1)
                .expect("cpu on-device grow");
        let part = DataPartition::from_payload(cpu_layout);
        let learner = SerialTreeLearner::new(&cpu_backend, &cpu, gain_config(), 4, -1)
            .with_features(features);
        let mut anchor = vec![0.0f64; nd];
        learner.add_prediction_to_score(&cpu_tree, &part, &mut anchor);

        assert_eq!(device.len(), anchor.len(), "score length parity");
        for (i, (&r, &a)) in device.iter().zip(anchor.iter()).enumerate() {
            let d = (r - a).abs();
            assert!(
                d <= ROCM_SCORE_TOL,
                "hip combined grow+score[{i}] = {r} diverged from cpu-anchor host {a} by \
                 {d} > {ROCM_SCORE_TOL} (~1e-6 f32 accumulation envelope)"
            );
        }
    }
}
