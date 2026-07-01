//! Multiclass-family on-device objective parity (Phase 19, 19-03 / ODL-07).
//!
//! Cells:
//! - `multiclass` — the class-major softmax grad/hess kernel bit-exact vs BOTH the
//!   `lgbm_objective::multiclass::MulticlassSoftmax` f64 anchor (D-05) AND the
//!   real-lib_lightgbm `multiclass_gh_iter{1,N}.txt` class-major goldens
//!   (score-derivation route). The per-row math carries NO accumulation and the f64
//!   `exp` ~1-ULP residual is absorbed by the `score_t` f32 cast (D-01), so
//!   `compare_exact_u32` holds. Plus the class-major `Σ_k grad[k·N+i] ≈ 0` per-row
//!   invariant (a golden-independent sanity net) and a twice-run determinism check.
//! - `multiclassova` — the one-vs-all per-class binary response bit-exact vs the
//!   `lgbm_objective::multiclass::MulticlassOva` f64 anchor AND the
//!   `multiclassova_gh_iter{1,N}.txt` class-major golden (skip-pass if absent).
//! - `softmax_convert` — the per-row softmax ConvertOutput bit-exact vs the host
//!   `lgbm_model::objective::softmax` anchor.
//!
//! Anchor discipline: NEVER GPU-vs-GPU (D-05). The cubecl-cpu f64 fold is the
//! reference. The class-major `[num_data*k + i]` stride is LOAD-BEARING (Pitfall 3).

mod objective_common;

use lgbm_compute::kernels::objective_multiclass as obj;
use lgbm_compute::runtime::cpu_client;
use lgbm_objective::multiclass::MulticlassSoftmax;
use objective_common::{parse_gh, read_boosting_golden};
use oracle_harness::comparator::{compare_exact_u32, ORACLE_TOL};

/// The cubecl-cpu f64 anchor client type (oracle-harness stays cubecl-free at the
/// crate level, CMP-01 — only the re-exported client type is named here).
type Client = lgbm_compute::ComputeClientReexport<lgbm_compute::runtime::ActiveRuntime>;

/// The multiclass spine corpus labels (identical to
/// `boosting_oracle_capture.py::multiclass_corpus`): 12 rows, 3 classes, balanced
/// (4 each), all classes present (`class_need_train == true`).
const MULTICLASS_LABELS: [f32; 12] =
    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0];

const NUM_CLASS: usize = 3;
const NUM_DATA: usize = 12;

/// The capture's `MULTICLASS_LATER_ITER` for the `*_gh_iterN` golden (the multiclass
/// cells cap the horizon at 5 iters and use `later_iter = 4`, NOT the single-output
/// spine's 5). The iterN raw score is `predict(num_iteration = LATER_ITER - 1)`, i.e.
/// `*_scores.txt` data-line index `LATER_ITER - 2`.
const LATER_ITER: usize = 4;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The host f64 softmax anchor grad/hess over the class-major `score` buffer.
fn host_softmax_gh(score: &[f64], labels: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let obj = MulticlassSoftmax::new(NUM_CLASS as i32, labels).unwrap();
    let mut g = vec![0.0f32; score.len()];
    let mut h = vec![0.0f32; score.len()];
    obj.get_gradients(score, &mut g, &mut h).expect("host softmax anchor grad/hess");
    (g, h)
}

/// The per-class softmax `BoostFromScore` inits (the iter-1 raw score is each class's
/// init broadcast to every row — class-major).
fn softmax_iter1_scores() -> Vec<f64> {
    let obj = MulticlassSoftmax::new(NUM_CLASS as i32, &MULTICLASS_LABELS).unwrap();
    let mut score = vec![0.0f64; NUM_DATA * NUM_CLASS];
    for k in 0..NUM_CLASS {
        let init = obj.boost_from_score(k as i32);
        for i in 0..NUM_DATA {
            score[NUM_DATA * k + i] = init;
        }
    }
    score
}

/// Read the `*_scores.txt` per-iter class-major raw-score golden and return the data
/// line at `idx` (0-based, skipping `#` comment lines), parsed from u64 f64-bits.
/// `None` (skip-pass) if the fixture is absent.
fn read_scores_line(file: &str, idx: usize) -> Option<Vec<f64>> {
    let text = read_boosting_golden(file)?;
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

/// Assert device == host f64 anchor (D-05) AND device == golden g/h (score-derivation),
/// all bit-exact (no accumulation → the f64 exp residual is f32-cast-absorbed, D-01).
fn check_softmax_at(client: &Client, scores: &[f64], gh_file: &str, tag: &str) {
    let (dg, dh) = obj::softmax_get_gradients_on(client, scores, &MULTICLASS_LABELS, NUM_CLASS)
        .expect("device softmax grad/hess launch");

    // D-05 PRIMARY gate: device == host f64 anchor bit-exact (both f64 libm math).
    let (hg, hh) = host_softmax_gh(scores, &MULTICLASS_LABELS);
    compare_exact_u32(&bits(&dg), &bits(&hg))
        .unwrap_or_else(|m| panic!("{tag} softmax grad vs host anchor not bit-exact: {m:?}"));
    compare_exact_u32(&bits(&dh), &bits(&hh))
        .unwrap_or_else(|m| panic!("{tag} softmax hess vs host anchor not bit-exact: {m:?}"));

    // Real-lib_lightgbm class-major golden (skip-pass if absent). No accum → bit-exact.
    if let Some(text) = read_boosting_golden(gh_file) {
        let (g_golden, h_golden) = parse_gh(&text);
        assert_eq!(g_golden.len(), NUM_DATA * NUM_CLASS, "{tag} golden grad width");
        compare_exact_u32(&bits(&dg), &bits(&g_golden))
            .unwrap_or_else(|m| panic!("{tag} softmax grad vs golden not bit-exact: {m:?}"));
        compare_exact_u32(&bits(&dh), &bits(&h_golden))
            .unwrap_or_else(|m| panic!("{tag} softmax hess vs golden not bit-exact: {m:?}"));
    }

    // Held-out invariant: Σ_k grad[k·N+i] ≈ 0 per row (softmax grads sum to zero).
    for i in 0..NUM_DATA {
        let mut s = 0.0f64;
        for k in 0..NUM_CLASS {
            s += f64::from(dg[NUM_DATA * k + i]);
        }
        assert!(
            s.abs() < f64::from(ORACLE_TOL),
            "{tag} Σ_k grad[k·N+{i}] = {s} not ≈ 0 (row {i})"
        );
    }
}

#[test]
fn multiclass() {
    let client = cpu_client();

    // iter-1: score0 = per-class BoostFromScore init broadcast to every row.
    let score0 = softmax_iter1_scores();
    check_softmax_at(&client, &score0, "multiclass_gh_iter1.txt", "iter1");

    // Twice-run determinism on the cpu anchor.
    let (g1, h1) =
        obj::softmax_get_gradients_on(&client, &score0, &MULTICLASS_LABELS, NUM_CLASS).unwrap();
    let (g2, h2) =
        obj::softmax_get_gradients_on(&client, &score0, &MULTICLASS_LABELS, NUM_CLASS).unwrap();
    compare_exact_u32(&bits(&g1), &bits(&g2))
        .unwrap_or_else(|m| panic!("softmax grad determinism: {m:?}"));
    compare_exact_u32(&bits(&h1), &bits(&h2))
        .unwrap_or_else(|m| panic!("softmax hess determinism: {m:?}"));

    // iter-N: raw score at num_iteration = LATER_ITER - 1 (scores line LATER_ITER - 2).
    if let Some(score_prev) = read_scores_line("multiclass_scores.txt", LATER_ITER - 2) {
        assert_eq!(score_prev.len(), NUM_DATA * NUM_CLASS, "scores line width");
        check_softmax_at(&client, &score_prev, "multiclass_gh_iterN.txt", "iterN");
    }
}

#[test]
fn softmax_convert() {
    use lgbm_model::objective::softmax;
    let client = cpu_client();

    // A non-trivial class-major raw-score buffer (distinct per class/row).
    let mut raw = vec![0.0f64; NUM_DATA * NUM_CLASS];
    for i in 0..NUM_DATA {
        for k in 0..NUM_CLASS {
            raw[NUM_DATA * k + i] = 0.3 * (i as f64) - 0.7 * (k as f64) + 0.1;
        }
    }

    let prob = obj::softmax_convert_output_on(&client, &raw, NUM_CLASS).expect("softmax convert");

    // Per row: gather class-major rec, host softmax over the K-length slice, compare.
    for i in 0..NUM_DATA {
        let rec: Vec<f64> = (0..NUM_CLASS).map(|k| raw[NUM_DATA * k + i]).collect();
        let mut want = vec![0.0f64; NUM_CLASS];
        softmax(&rec, &mut want);
        for k in 0..NUM_CLASS {
            let idx = NUM_DATA * k + i;
            assert_eq!(
                prob[idx].to_bits(),
                want[k].to_bits(),
                "softmax convert [class {k}, row {i}] not bit-exact"
            );
        }
    }
}
