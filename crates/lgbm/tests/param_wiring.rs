//! Facade-level proof that the per-feature constraint/penalty parameters reach
//! the tree learner through the PUBLIC training entry points.
//!
//! # Why this file exists
//!
//! `monotone_constraints`, `interaction_constraints`, `cegb_*`, `extra_trees` and
//! `feature_contri` are all implemented in `lgbm-treelearner` and oracle-tested
//! there — but the learner reads them from `LearnerConstraints`, which the public
//! `train` / `train_raw` path did not populate. The parameters therefore parsed,
//! validated, and then did NOTHING for any caller of the facade or the Python
//! binding. Every test below fails against that unwired state.
//!
//! Each test is a BEHAVIORAL assertion (the grown model differs / obeys the
//! constraint), not a plumbing assertion, so it stays honest if the wiring moves.

use std::collections::HashMap;

use lgbm::{train_raw, Config, RawCorpus};

/// A 2-feature corpus whose label is monotonically DECREASING in feature 0 and
/// increasing in feature 1, so a `+1` monotone constraint on feature 0 is
/// genuinely binding: the unconstrained fit wants a decreasing step there.
fn corpus(config: &Config) -> RawCorpus {
    let n = 200usize;
    let f0: Vec<f64> = (0..n).map(|i| (i % 20) as f64).collect();
    let f1: Vec<f64> = (0..n).map(|i| ((i / 20) % 10) as f64).collect();
    let labels: Vec<f32> = (0..n)
        .map(|i| (-2.0 * f0[i] + 1.0 * f1[i]) as f32)
        .collect();
    let mut raw = RawCorpus::from_columns(vec![f0, f1], labels);
    raw.config = config.clone();
    raw
}

fn cfg(pairs: &[(&str, &str)]) -> Config {
    let mut params: HashMap<String, String> = [
        ("objective", "regression"),
        ("num_iterations", "8"),
        ("num_leaves", "8"),
        ("learning_rate", "0.3"),
        ("min_data_in_leaf", "5"),
        ("seed", "1"),
        ("deterministic", "true"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    for (k, v) in pairs {
        params.insert(k.to_string(), v.to_string());
    }
    Config::from_params(&params).expect("config builds")
}

fn model_text(config: &Config) -> String {
    let c = corpus(config);
    train_raw(config, &c).expect("train ok").model_to_string()
}

/// Every `split_feature=` value appearing in the model text.
fn split_features(model: &str) -> Vec<i32> {
    model
        .lines()
        .filter_map(|l| l.strip_prefix("split_feature="))
        .flat_map(|v| v.split_whitespace())
        .filter_map(|t| t.parse::<i32>().ok())
        .collect()
}

#[test]
fn feature_contri_zero_removes_a_feature_from_every_split() {
    // `gain[i] *= feature_contri[i]` with a 0 multiplier drives feature 0's gain
    // to 0, below every competing feature — so it can never win a split.
    let baseline = model_text(&cfg(&[]));
    assert!(
        split_features(&baseline).contains(&0),
        "precondition: feature 0 wins splits without a penalty"
    );

    let penalized = model_text(&cfg(&[("feature_contri", "0.0,1.0")]));
    assert!(
        !split_features(&penalized).contains(&0),
        "feature_contri=0 must exclude feature 0 from all splits, got: {:?}",
        split_features(&penalized)
    );
}

#[test]
fn feature_contri_alias_feature_penalty_has_the_same_effect() {
    let a = model_text(&cfg(&[("feature_contri", "0.0,1.0")]));
    let b = model_text(&cfg(&[("feature_penalty", "0.0,1.0")]));
    assert_eq!(a, b, "the `feature_penalty` alias must behave identically");
}

#[test]
fn feature_contri_changes_the_model_without_excluding_a_feature() {
    // A partial (non-zero) penalty must still shift the split choice/order.
    let baseline = model_text(&cfg(&[]));
    let damped = model_text(&cfg(&[("feature_contri", "0.05,1.0")]));
    assert_ne!(baseline, damped, "a 0.05 gain multiplier must change the fit");
}

#[test]
fn monotone_constraints_reach_the_learner_and_change_the_model() {
    // Feature 0's true relationship is DECREASING, so forcing `+1`
    // (non-decreasing) must produce a different model than the unconstrained fit.
    let baseline = model_text(&cfg(&[]));
    let constrained = model_text(&cfg(&[("monotone_constraints", "1,0")]));
    assert_ne!(
        baseline, constrained,
        "monotone_constraints=+1 on a decreasing feature must change the model"
    );
}

#[test]
fn monotone_constraints_produce_a_non_decreasing_prediction() {
    // The real contract, not just "the model differs": under `+1` on feature 0,
    // predictions must be non-decreasing as feature 0 grows (feature 1 fixed).
    let config = cfg(&[("monotone_constraints", "1,0")]);
    let booster = train_raw(&config, &corpus(&config)).expect("train ok");
    let preds: Vec<f32> = (0..20)
        .map(|v| booster.predict_row(&[v as f64, 5.0])[0])
        .collect();
    for w in preds.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "prediction must be non-decreasing in feature 0 under a +1 constraint: {preds:?}"
        );
    }
}

#[test]
fn interaction_constraints_reach_the_learner_and_change_the_model() {
    // `[[0],[1]]` forbids features 0 and 1 from co-occurring on one root-to-leaf
    // path; the unconstrained fit uses both, so the model must differ.
    let baseline = model_text(&cfg(&[]));
    assert!(
        split_features(&baseline).contains(&0) && split_features(&baseline).contains(&1),
        "precondition: the unconstrained fit splits on both features"
    );
    let constrained = model_text(&cfg(&[("interaction_constraints", "[[0],[1]]")]));
    assert_ne!(
        baseline, constrained,
        "interaction_constraints must change the grown trees"
    );
}

#[test]
fn interaction_constraints_string_parses_to_the_cpp_groups() {
    // `Common::StringToArrayofArrays<int>(s, '[', ']', ',')` semantics.
    let c = cfg(&[("interaction_constraints", "[[0,1],[2]]")]);
    assert_eq!(c.interaction_constraints_vector(), vec![vec![0, 1], vec![2]]);
    let c = cfg(&[("interaction_constraints", "")]);
    assert!(c.interaction_constraints_vector().is_empty());
    // Stray/unbalanced brackets are skipped by SplitBrackets, not an error.
    let c = cfg(&[("interaction_constraints", "[0,1")]);
    assert!(c.interaction_constraints_vector().is_empty());
}

#[test]
fn extra_trees_reaches_the_learner_and_changes_the_model() {
    // extra_trees draws a RANDOM threshold per feature instead of the best one.
    let baseline = model_text(&cfg(&[]));
    let extra = model_text(&cfg(&[("extra_trees", "true")]));
    assert_ne!(baseline, extra, "extra_trees must change the grown trees");
}

#[test]
fn cegb_penalties_reach_the_learner_and_change_the_model() {
    let baseline = model_text(&cfg(&[]));
    let cegb = model_text(&cfg(&[
        ("cegb_tradeoff", "1.0"),
        ("cegb_penalty_split", "5.0"),
    ]));
    assert_ne!(baseline, cegb, "cegb_penalty_split must change the grown trees");

    let coupled = model_text(&cfg(&[
        ("cegb_tradeoff", "1.0"),
        ("cegb_penalty_feature_coupled", "10.0,0.0"),
    ]));
    assert_ne!(
        baseline, coupled,
        "cegb_penalty_feature_coupled must change the grown trees"
    );
}

#[test]
fn a_wrong_length_feature_contri_is_a_typed_error_not_a_silent_partial_penalty() {
    // `GBDT::Init` CHECK_EQs the length against num_total_features.
    let config = cfg(&[("feature_contri", "1.0,1.0,1.0")]); // 3 entries, 2 features
    let err = train_raw(&config, &corpus(&config)).expect_err("must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("feature_contri") && msg.contains('3'),
        "expected a length error naming feature_contri, got: {msg}"
    );
}

// --- `categorical_feature` (the PARAMETER, not the corpus field) -------------
//
// `RawCorpus::categorical_features` is set by the typed ingest paths (polars/Arrow),
// but real LightGBM also accepts `params={"categorical_feature": "0,2"}` at the
// Dataset level (`dataset_loader.cpp:168-189`). That parameter parsed and then did
// nothing here: every column was binned NUMERIC regardless.

/// The parameter must select the same categorical binning the corpus field does.
/// Equivalence is the strongest available assertion — it pins the parameter to the
/// already-oracle-tested `find_bin_categorical` path rather than to a plumbing detail.
#[test]
fn categorical_feature_param_matches_setting_the_corpus_field() {
    let via_param = cfg(&[("categorical_feature", "0")]);
    let by_param = train_raw(&via_param, &corpus(&via_param))
        .expect("train ok")
        .model_to_string();

    let plain = cfg(&[]);
    let mut c = corpus(&plain);
    c.categorical_features = vec![0];
    let by_field = train_raw(&plain, &c).expect("train ok").model_to_string();

    assert_eq!(
        by_param, by_field,
        "categorical_feature=0 must bin feature 0 exactly as corpus.categorical_features=[0]"
    );
    // …and both must differ from the all-numeric model, so the assertion has teeth.
    let numeric = train_raw(&plain, &corpus(&plain))
        .expect("train ok")
        .model_to_string();
    assert_ne!(
        by_param, numeric,
        "categorical binning must change the grown trees"
    );
}

#[test]
fn an_out_of_range_categorical_feature_index_is_a_typed_error() {
    let config = cfg(&[("categorical_feature", "7")]); // only 2 features
    let err = train_raw(&config, &corpus(&config)).expect_err("must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("categorical_feature") && msg.contains('7'),
        "expected an out-of-range error naming categorical_feature, got: {msg}"
    );
}

/// C++ accepts `categorical_feature=name:a,b`; an in-memory corpus carries no column
/// names, so the port must SAY SO rather than silently train an all-numeric model.
#[test]
fn the_name_prefixed_categorical_feature_form_is_an_explicit_error() {
    let config = cfg(&[("categorical_feature", "name:f0")]);
    let err = train_raw(&config, &corpus(&config)).expect_err("must be rejected");
    assert!(
        err.to_string().contains("name:"),
        "expected an explicit unsupported error, got: {err}"
    );
}

// --- `bagging_by_query` -------------------------------------------------------
//
// The query-grouped draw (`BaggingSampleStrategy::bagging_by_query`, RNG-replay
// tested in `oracle-harness/tests/rank_parity.rs`) was unreachable from the facade:
// `train_raw` rejected `bagging_by_query=true` outright because the facade used to
// drop the corpus's query boundaries. It now forwards them, so the parameter selects
// the query-grouped draw for any grouped corpus.

/// A grouped corpus: 200 rows in 10 queries of 20.
fn grouped_corpus(config: &Config) -> RawCorpus {
    let mut c = corpus(config);
    c.query_boundaries = (0..=10).map(|q| q * 20).collect();
    c
}

#[test]
fn bagging_by_query_draws_a_different_bag_than_row_bagging() {
    let row = cfg(&[("bagging_freq", "1"), ("bagging_fraction", "0.5")]);
    let by_row = train_raw(&row, &grouped_corpus(&row))
        .expect("row bagging trains")
        .model_to_string();

    let query = cfg(&[
        ("bagging_freq", "1"),
        ("bagging_fraction", "0.5"),
        ("bagging_by_query", "true"),
    ]);
    let by_query = train_raw(&query, &grouped_corpus(&query))
        .expect("query bagging trains")
        .model_to_string();

    assert_ne!(
        by_row, by_query,
        "bagging_by_query must select whole QUERIES, producing a different bag \
         (and therefore a different model) than the per-row draw"
    );
}

/// C++ `Log::Fatal("Ranking tasks require query information")`. An UNGROUPED corpus
/// must be a typed error, never a silent fall-through to row bagging.
#[test]
fn bagging_by_query_without_query_boundaries_is_a_typed_error() {
    let config = cfg(&[
        ("bagging_freq", "1"),
        ("bagging_fraction", "0.5"),
        ("bagging_by_query", "true"),
    ]);
    let err = train_raw(&config, &corpus(&config)).expect_err("must be rejected");
    assert!(
        err.to_string().contains("query/group boundaries"),
        "expected a query-boundary error, got: {err}"
    );
}
