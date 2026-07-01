//! Regression-family on-device objective parity (Phase 19, 19-01 / ODL-05).
//!
//! Cells:
//! - `regression` — the six grad/hess kernels (L2/L1/Quantile/Huber/Fair/Poisson)
//!   bit-exact vs BOTH the `lgbm_objective::regression::Objective` f64 anchor (D-05)
//!   AND the real-lib_lightgbm `*_gh_iter{1,N}.txt` goldens (score-derivation
//!   route). Plus the `<USE_WEIGHT>` equivalence + twice-run determinism properties.
//! - `convert_regression` — the ConvertOutput inverse-link (passthrough / sqrt·x² /
//!   exp) vs the host `lgbm_model::objective::convert_{regression,poisson}` anchor.
//! - `boost_from_score` — the BoostFromScore init composition (Task 2).
//! - `renew_leaf` — the L1/Quantile per-leaf median refit (Task 3, host-orchestrated).
//!
//! Anchor discipline: NEVER GPU-vs-GPU (D-05). The cubecl-cpu f64 fold is the
//! reference; the poisson `exp` path is held to `ORACLE_TOL` vs the numpy-derived
//! golden (the documented exp-libm residual, Pitfall 5 / D-05) while every
//! arithmetic-only objective is bit-exact (`compare_exact_u32`).

mod objective_common;

use lgbm_compute::kernels::objective_regression as obj;
use lgbm_compute::runtime::cpu_client;
use lgbm_objective::regression::Objective;
use objective_common::{parse_gh, read_boosting_golden};
use oracle_harness::comparator::{compare_exact_u32, compare_within, ORACLE_TOL};

/// The cubecl-cpu f64 anchor client type (oracle-harness stays cubecl-free at the
/// crate level, CMP-01 — only the re-exported client type is named here).
type Client = lgbm_compute::ComputeClientReexport<lgbm_compute::runtime::ActiveRuntime>;

/// The spine corpus labels (identical to `boosting_oracle_capture.py::spine_corpus`
/// and `boosting_parity.rs::spine_corpus`). Also the poisson corpus labels
/// (`exp_log_corpus("poisson")` returns the spine corpus).
const SPINE_LABELS: [f32; 12] =
    [2.0, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0];

/// The six regression objectives under test.
#[derive(Clone, Copy, Debug)]
enum Reg {
    L2,
    L1,
    Quantile,
    Huber,
    Fair,
    Poisson,
}

use Reg::{Fair, Huber, L1, L2, Poisson, Quantile};

const ALL: [Reg; 6] = [L2, L1, Quantile, Huber, Fair, Poisson];

impl Reg {
    /// The golden fixture prefix (`<prefix>_gh_iter{1,N}.txt` / `<prefix>_scores.txt`).
    fn prefix(self) -> &'static str {
        match self {
            L2 => "regression",
            L1 => "regression_l1",
            Quantile => "quantile",
            Huber => "huber",
            Fair => "fair",
            Poisson => "poisson",
        }
    }

    /// The capture's `later_iter` for the `gh_iterN` golden: poisson (exp/log family)
    /// uses `EXP_LOG_LATER_ITER = 4`; the rest use `LATER_ITER = 5`. `score_prev` is
    /// the raw score at `num_iteration = later_iter - 1`, i.e. `_scores.txt` data-line
    /// index `later_iter - 2`.
    fn later_iter(self) -> usize {
        match self {
            Poisson => 4,
            _ => 5,
        }
    }

    /// The host f64 anchor objective (default params matching the capture cells).
    fn host(self) -> Objective {
        match self {
            L2 => Objective::Regression { sqrt: false },
            L1 => Objective::RegressionL1,
            Quantile => Objective::Quantile { alpha: 0.9 },
            Huber => Objective::Huber { alpha: 0.9 },
            Fair => Objective::Fair { c: 1.0 },
            Poisson => Objective::Poisson { max_delta_step: 0.7 },
        }
    }

    /// Launch the matching device grad/hess kernel on the cpu anchor.
    fn device(
        self,
        client: &Client,
        scores: &[f64],
        labels: &[f32],
        weights: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>) {
        match self {
            L2 => obj::get_gradients_l2_on(client, scores, labels, weights),
            L1 => obj::get_gradients_l1_on(client, scores, labels, weights),
            Quantile => obj::get_gradients_quantile_on(client, scores, labels, weights, 0.9),
            Huber => obj::get_gradients_huber_on(client, scores, labels, weights, 0.9),
            Fair => obj::get_gradients_fair_on(client, scores, labels, weights, 1.0),
            Poisson => obj::get_gradients_poisson_on(client, scores, labels, weights, 0.7),
        }
        .expect("device grad/hess launch")
    }

    /// Arithmetic-only objectives are bit-exact; poisson's `exp` is held to
    /// `ORACLE_TOL` (the documented numpy-vs-libm exp residual, Pitfall 5).
    fn bit_exact(self) -> bool {
        !matches!(self, Poisson)
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Assert `got == want` bit-exact for arithmetic objectives, else within
/// `ORACLE_TOL` (poisson exp).
fn assert_gh(reg: Reg, got: &[f32], want: &[f32], what: &str) {
    if reg.bit_exact() {
        compare_exact_u32(&bits(got), &bits(want))
            .unwrap_or_else(|m| panic!("{reg:?} {what}: not bit-exact: {m:?}"));
    } else {
        compare_within(got, want, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("{reg:?} {what}: not within ORACLE_TOL: {m:?}"));
    }
}

/// Read the `<prefix>_scores.txt` per-iter raw-score golden and return the data line
/// at `idx` (0-based, skipping `#` comment lines), parsed from u64 f64-bits. `None`
/// (skip-pass) if the fixture is absent.
fn read_scores_line(prefix: &str, idx: usize) -> Option<Vec<f64>> {
    let text = read_boosting_golden(&format!("{prefix}_scores.txt"))?;
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

/// Run one objective at one score vector: assert device == host f64 anchor (D-05)
/// AND device == golden g/h (score-derivation route), reusing the same inputs.
fn check_at(
    reg: Reg,
    client: &Client,
    scores: &[f64],
    labels: &[f32],
    gh_file: &str,
    tag: &str,
) {
    let (dg, dh) = reg.device(client, scores, labels, None);

    // D-05 PRIMARY gate: device == host f64 anchor bit-exact (both f64 math; the
    // poisson exp source is the only non-arithmetic op → held to ORACLE_TOL).
    let mut hg = vec![0.0f32; labels.len()];
    let mut hh = vec![0.0f32; labels.len()];
    reg.host()
        .get_gradients(scores, labels, &mut hg, &mut hh)
        .expect("host anchor grad/hess");
    assert_gh(reg, &dg, &hg, &format!("{tag} grad vs host anchor"));
    assert_gh(reg, &dh, &hh, &format!("{tag} hess vs host anchor"));

    // Real-lib_lightgbm golden (skip-pass if absent).
    if let Some(text) = read_boosting_golden(gh_file) {
        let (g_golden, h_golden) = parse_gh(&text);
        assert_gh(reg, &dg, &g_golden, &format!("{tag} grad vs golden"));
        assert_gh(reg, &dh, &h_golden, &format!("{tag} hess vs golden"));
    }
}

#[test]
fn regression() {
    let client = cpu_client();
    let labels = SPINE_LABELS.to_vec();

    for reg in ALL {
        let host = reg.host();
        // iter-1: score0 = full(init) where init = BoostFromScore (the capture's
        // `score0 = np.full(n, init)`). Quantile's grad is sign-invariant so the
        // f32-rounded-alpha init difference never flips a per-row grad.
        let init = host.boost_from_score(&labels);
        let score0 = vec![init; labels.len()];
        check_at(
            reg,
            &client,
            &score0,
            &labels,
            &format!("{}_gh_iter1.txt", reg.prefix()),
            "iter1",
        );

        // iter-N: score_prev = the captured raw score at num_iteration = later-1
        // (`_scores.txt` data line `later - 2`).
        if let Some(score_prev) = read_scores_line(reg.prefix(), reg.later_iter() - 2) {
            assert_eq!(score_prev.len(), labels.len(), "{reg:?} scores line width");
            check_at(
                reg,
                &client,
                &score_prev,
                &labels,
                &format!("{}_gh_iterN.txt", reg.prefix()),
                "iterN",
            );
        }
    }
}

#[test]
fn regression_weight_branch_and_determinism() {
    let client = cpu_client();
    let labels = SPINE_LABELS.to_vec();
    let ones = vec![1.0f32; labels.len()];

    for reg in ALL {
        let init = reg.host().boost_from_score(&labels);
        let score = vec![init + 0.37; labels.len()]; // non-trivial residuals

        // Weight-branch equivalence: use_weight=true with all-1.0 weights is
        // bit-identical to use_weight=false (device-vs-device, same backend → exact).
        let (g_nw, h_nw) = reg.device(&client, &score, &labels, None);
        let (g_w, h_w) = reg.device(&client, &score, &labels, Some(&ones));
        compare_exact_u32(&bits(&g_nw), &bits(&g_w))
            .unwrap_or_else(|m| panic!("{reg:?} weight-branch grad equivalence: {m:?}"));
        compare_exact_u32(&bits(&h_nw), &bits(&h_w))
            .unwrap_or_else(|m| panic!("{reg:?} weight-branch hess equivalence: {m:?}"));

        // Twice-run determinism on the cpu anchor.
        let (g2, h2) = reg.device(&client, &score, &labels, None);
        compare_exact_u32(&bits(&g_nw), &bits(&g2))
            .unwrap_or_else(|m| panic!("{reg:?} grad determinism: {m:?}"));
        compare_exact_u32(&bits(&h_nw), &bits(&h2))
            .unwrap_or_else(|m| panic!("{reg:?} hess determinism: {m:?}"));
    }
}

impl Reg {
    /// The device BoostFromScore composition tag + alpha for this objective.
    fn boost_tag_alpha(self) -> (u32, f64) {
        match self {
            L2 => (obj::TAG_L2, 0.0),
            L1 => (obj::TAG_L1, 0.0),
            Quantile => (obj::TAG_QUANTILE, 0.9),
            Huber => (obj::TAG_HUBER, 0.0),
            Fair => (obj::TAG_FAIR, 0.0),
            Poisson => (obj::TAG_POISSON, 0.0),
        }
    }
}

#[test]
fn boost_from_score() {
    let client = cpu_client();
    let labels = SPINE_LABELS.to_vec();

    for reg in ALL {
        let (tag, alpha) = reg.boost_tag_alpha();
        let device_init =
            obj::boost_from_score_on(&client, tag, &labels, alpha).expect("device boost");
        let host_init = reg.host().boost_from_score(&labels);
        // Deterministic reduction order (reduce_sum ascending fold / percentile) →
        // bit-exact vs the host f64 anchor for all six.
        assert_eq!(
            device_init.to_bits(),
            host_init.to_bits(),
            "{reg:?} BoostFromScore not bit-exact vs host anchor: device {device_init} host {host_init}"
        );
    }
}

#[test]
fn boost_from_score_poisson_rejects_bad_labels() {
    let client = cpu_client();
    // A negative label is rejected (Pitfall 6 label-domain guard).
    assert!(obj::boost_from_score_on(&client, obj::TAG_POISSON, &[1.0, -0.5, 2.0], 0.0).is_err());
    // A zero label-sum is rejected.
    assert!(obj::boost_from_score_on(&client, obj::TAG_POISSON, &[0.0, 0.0], 0.0).is_err());
    // A valid non-negative, non-zero-sum corpus is accepted.
    assert!(obj::boost_from_score_on(&client, obj::TAG_POISSON, &[0.0, 1.0, 2.0], 0.0).is_ok());
    // The non-poisson objectives have no label-domain guard.
    assert!(obj::boost_from_score_on(&client, obj::TAG_L2, &[-1.0, 2.0], 0.0).is_ok());
}

#[test]
fn convert_regression() {
    use lgbm_model::objective::{convert_poisson, convert_regression};
    let client = cpu_client();
    let raw = vec![-3.0f64, -0.5, 0.0, 0.5, 2.0, 11.0];

    // Passthrough (non-sqrt regression / L1 / huber / fair / quantile): bit-exact.
    let pass = obj::convert_output_on(&client, &raw, obj::CONVERT_PASSTHROUGH).expect("convert");
    for (i, &x) in raw.iter().enumerate() {
        assert_eq!(
            pass[i].to_bits(),
            convert_regression(x, false).to_bits(),
            "passthrough[{i}] not bit-exact"
        );
    }

    // sqrt variant: Sign(x)*x*x, bit-exact vs the host anchor.
    let sq = obj::convert_output_on(&client, &raw, obj::CONVERT_SQRT_SQUARE).expect("convert");
    for (i, &x) in raw.iter().enumerate() {
        assert_eq!(
            sq[i].to_bits(),
            convert_regression(x, true).to_bits(),
            "sqrt-square[{i}] not bit-exact"
        );
    }

    // Poisson inverse-link: exp(x) — held to ORACLE_TOL (exp-libm residual, D-05).
    let ex = obj::convert_output_on(&client, &raw, obj::CONVERT_EXP).expect("convert");
    for (i, &x) in raw.iter().enumerate() {
        let want = convert_poisson(x);
        assert!(
            (ex[i] - want).abs() <= f64::from(ORACLE_TOL) * want.abs().max(1.0),
            "exp[{i}] = {} not within ORACLE_TOL of {want}",
            ex[i]
        );
    }
}
