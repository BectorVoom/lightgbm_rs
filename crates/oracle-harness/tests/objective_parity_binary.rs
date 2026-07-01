//! Binary-family on-device objective parity (Phase 19, 19-02 / ODL-06).
//!
//! Cells:
//! - `binary` — the sigmoid grad/hess kernel bit-exact vs BOTH the
//!   `lgbm_objective::binary::Binary` f64 anchor (D-05) AND the real-lib_lightgbm
//!   `binary_gh_iter{1,N}.txt` goldens (score-derivation route). Plus the
//!   `<USE_WEIGHT>` equivalence + twice-run determinism properties. The per-row math
//!   carries NO accumulation → bit-exact (`compare_exact_u32`) despite the sigmoid
//!   `exp` (the f64 exp's ~1-ULP residual is absorbed by the `score_t` f32 cast, D-01).
//! - `convert_binary` — the sigmoid ConvertOutput inverse-link `1/(1+exp(-σx))` vs the
//!   host `lgbm_model::objective::convert_binary` anchor (bit-exact, same f64 libm exp).
//! - `binary_boost` — the two-stage BoostFromScore logit init (Task 2) held to
//!   `ORACLE_TOL` (the documented atomicAdd-order residual, D-05 / Pitfall 5, NOT
//!   bit-exact) + the OVA `±1` label-reset kernel (bit-exact).
//!
//! Anchor discipline: NEVER GPU-vs-GPU (D-05). The cubecl-cpu f64 fold is the
//! reference.

mod objective_common;

use lgbm_compute::kernels::objective_binary as obj;
use lgbm_compute::runtime::cpu_client;
use lgbm_objective::binary::Binary;
use objective_common::{parse_gh, read_boosting_golden};
use oracle_harness::comparator::{compare_exact_u32, compare_within, ORACLE_TOL};

/// The cubecl-cpu f64 anchor client type (oracle-harness stays cubecl-free at the
/// crate level, CMP-01 — only the re-exported client type is named here).
type Client = lgbm_compute::ComputeClientReexport<lgbm_compute::runtime::ActiveRuntime>;

/// The binary spine corpus labels (identical to
/// `boosting_oracle_capture.py::binary_corpus`): 5 positives of 12, both classes
/// present (`ClassNeedTrain == true`, `pavg = 5/12 != 0/1`).
const BINARY_LABELS: [f32; 12] =
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];

/// The capture's `LATER_ITER` for the `binary_gh_iterN` golden (`write_layered`
/// default). `score_prev` is the raw score at `num_iteration = LATER_ITER - 1`, i.e.
/// `binary_scores.txt` data-line index `LATER_ITER - 2`.
const LATER_ITER: usize = 5;

/// The spine cell sigmoid (`config.sigmoid` default 1.0).
const SIGMOID: f64 = 1.0;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The host f64 anchor grad/hess for the balanced default (`label_weights_ = 1.0`).
fn host_gh(score: &[f64], labels: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let obj = Binary::new(SIGMOID).unwrap();
    let mut g = vec![0.0f32; labels.len()];
    let mut h = vec![0.0f32; labels.len()];
    obj.get_gradients(score, labels, &mut g, &mut h).expect("host anchor grad/hess");
    (g, h)
}

/// Read the `binary_scores.txt` per-iter raw-score golden and return the data line at
/// `idx` (0-based, skipping `#` comment lines), parsed from u64 f64-bits. `None`
/// (skip-pass) if the fixture is absent.
fn read_scores_line(idx: usize) -> Option<Vec<f64>> {
    let text = read_boosting_golden("binary_scores.txt")?;
    let line = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .nth(idx)?;
    Some(
        line.split_whitespace()
            .map(|t| f64::from_bits(t.parse::<u64>().expect("u64 bits")))
            .collect(),
    )
}

/// Run one score vector: assert device == host f64 anchor (D-05) AND device == golden
/// g/h (score-derivation route), all bit-exact (no accumulation).
fn check_at(client: &Client, scores: &[f64], labels: &[f32], gh_file: &str, tag: &str) {
    let (dg, dh) = obj::get_gradients_on(client, scores, labels, None, SIGMOID, None)
        .expect("device grad/hess launch");

    // D-05 PRIMARY gate: device == host f64 anchor bit-exact (both f64 libm math).
    let (hg, hh) = host_gh(scores, labels);
    compare_exact_u32(&bits(&dg), &bits(&hg))
        .unwrap_or_else(|m| panic!("{tag} grad vs host anchor not bit-exact: {m:?}"));
    compare_exact_u32(&bits(&dh), &bits(&hh))
        .unwrap_or_else(|m| panic!("{tag} hess vs host anchor not bit-exact: {m:?}"));

    // Real-lib_lightgbm golden (skip-pass if absent). No accumulation → bit-exact.
    if let Some(text) = read_boosting_golden(gh_file) {
        let (g_golden, h_golden) = parse_gh(&text);
        compare_exact_u32(&bits(&dg), &bits(&g_golden))
            .unwrap_or_else(|m| panic!("{tag} grad vs golden not bit-exact: {m:?}"));
        compare_exact_u32(&bits(&dh), &bits(&h_golden))
            .unwrap_or_else(|m| panic!("{tag} hess vs golden not bit-exact: {m:?}"));
    }
}

#[test]
fn binary() {
    let client = cpu_client();
    let labels = BINARY_LABELS.to_vec();

    // iter-1: score0 = full(init) where init = BoostFromScore (the capture's
    // `score0 = np.full(n, init)`).
    let init = Binary::new(SIGMOID).unwrap().boost_from_score(&labels);
    let score0 = vec![init; labels.len()];
    check_at(&client, &score0, &labels, "binary_gh_iter1.txt", "iter1");

    // iter-N: score_prev = the captured raw score at num_iteration = LATER_ITER - 1
    // (`binary_scores.txt` data line `LATER_ITER - 2`).
    if let Some(score_prev) = read_scores_line(LATER_ITER - 2) {
        assert_eq!(score_prev.len(), labels.len(), "scores line width");
        check_at(&client, &score_prev, &labels, "binary_gh_iterN.txt", "iterN");
    }
}

#[test]
fn binary_weight_branch_and_determinism() {
    let client = cpu_client();
    let labels = BINARY_LABELS.to_vec();
    let ones = vec![1.0f32; labels.len()];
    let init = Binary::new(SIGMOID).unwrap().boost_from_score(&labels);
    let score = vec![init + 0.37; labels.len()]; // non-trivial residuals

    // Weight-branch equivalence: use_weight=true with all-1.0 weights is bit-identical
    // to use_weight=false (device-vs-device, same backend → exact).
    let (g_nw, h_nw) = obj::get_gradients_on(&client, &score, &labels, None, SIGMOID, None)
        .expect("device grad/hess");
    let (g_w, h_w) = obj::get_gradients_on(&client, &score, &labels, Some(&ones), SIGMOID, None)
        .expect("device grad/hess weighted");
    compare_exact_u32(&bits(&g_nw), &bits(&g_w))
        .unwrap_or_else(|m| panic!("weight-branch grad equivalence: {m:?}"));
    compare_exact_u32(&bits(&h_nw), &bits(&h_w))
        .unwrap_or_else(|m| panic!("weight-branch hess equivalence: {m:?}"));

    // Label-weight-branch equivalence: use_label_weight with the balanced 1.0/1.0
    // default is bit-identical to the off branch (both classes weight 1.0).
    let (g_lw, h_lw) =
        obj::get_gradients_on(&client, &score, &labels, None, SIGMOID, Some((1.0, 1.0)))
            .expect("device grad/hess label-weighted");
    compare_exact_u32(&bits(&g_nw), &bits(&g_lw))
        .unwrap_or_else(|m| panic!("label-weight-branch grad equivalence: {m:?}"));
    compare_exact_u32(&bits(&h_nw), &bits(&h_lw))
        .unwrap_or_else(|m| panic!("label-weight-branch hess equivalence: {m:?}"));

    // Twice-run determinism on the cpu anchor.
    let (g2, h2) = obj::get_gradients_on(&client, &score, &labels, None, SIGMOID, None)
        .expect("device grad/hess rerun");
    compare_exact_u32(&bits(&g_nw), &bits(&g2))
        .unwrap_or_else(|m| panic!("grad determinism: {m:?}"));
    compare_exact_u32(&bits(&h_nw), &bits(&h2))
        .unwrap_or_else(|m| panic!("hess determinism: {m:?}"));
}

#[test]
fn binary_boost() {
    let client = cpu_client();
    let labels = BINARY_LABELS.to_vec();

    // Two-stage BoostFromScore: device Σ is_pos reduce + host logit finalize. The
    // device reduce is the documented atomicAdd-order residual (D-05 / Pitfall 5), so
    // the init scalar is held to ORACLE_TOL vs the host anchor — NOT compare_exact.
    let host_init = Binary::new(SIGMOID).unwrap().boost_from_score(&labels);
    let device_init =
        obj::boost_from_score_on(&client, &labels, None, SIGMOID).expect("device boost");
    compare_within(&[device_init as f32], &[host_init as f32], ORACLE_TOL).unwrap_or_else(|m| {
        panic!("binary BoostFromScore init not within ORACLE_TOL: device {device_init} host {host_init}: {m:?}")
    });

    // Weighted prior (all-1.0 weights) collapses to the unweighted logit (sumw = N),
    // still within ORACLE_TOL of the host anchor.
    let ones = vec![1.0f32; labels.len()];
    let device_init_w = obj::boost_from_score_on(&client, &labels, Some(&ones), SIGMOID)
        .expect("device boost weighted");
    compare_within(&[device_init_w as f32], &[host_init as f32], ORACLE_TOL)
        .unwrap_or_else(|m| panic!("weighted BoostFromScore init not within ORACLE_TOL: {m:?}"));

    // Length-mismatched weights are a typed V5 error.
    assert!(obj::boost_from_score_on(&client, &labels, Some(&[1.0f32, 2.0]), SIGMOID).is_err());

    // OVA (one-vs-all) label reset: out[i] = (label == class) ? +1 : -1, bit-exact.
    let multiclass_labels = [0.0f32, 1.0, 2.0, 1.0, 0.0, 2.0];
    for class_id in 0..3i32 {
        let got = obj::reset_ova_label_on(&client, &multiclass_labels, class_id).expect("ova");
        let want: Vec<f32> = multiclass_labels
            .iter()
            .map(|&l| if l as i32 == class_id { 1.0f32 } else { -1.0 })
            .collect();
        compare_exact_u32(&bits(&got), &bits(&want))
            .unwrap_or_else(|m| panic!("OVA label reset class {class_id} not bit-exact: {m:?}"));
    }
}

#[test]
fn convert_binary() {
    use lgbm_model::objective::convert_binary;
    let client = cpu_client();
    let raw = vec![-3.0f64, -0.5, 0.0, 0.5, 2.0, 11.0];

    // Sigmoid ConvertOutput `1/(1+exp(-σx))`: bit-exact vs the host anchor (same f64
    // libm exp, same op order).
    for &sigmoid in &[1.0f64, 2.0] {
        let prob = obj::sigmoid_convert_output_on(&client, &raw, sigmoid).expect("convert");
        for (i, &x) in raw.iter().enumerate() {
            assert_eq!(
                prob[i].to_bits(),
                convert_binary(x, sigmoid).to_bits(),
                "sigmoid[{i}] (σ={sigmoid}) not bit-exact"
            );
        }
    }
}
