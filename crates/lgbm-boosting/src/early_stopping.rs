//! Early stopping (BST-07) + metric-evaluation cadence infrastructure (MET-02),
//! ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/boosting/gbdt.cpp`:
//!   - `kMinScore` (gbdt.cpp:215): the per-(valid_set, metric) `best_score_` is
//!     initialized to the most-negative value so the FIRST eval always improves.
//!   - the early-stop decision (gbdt.cpp:591-608): per (valid_set i, metric j),
//!     `cur = metric.factor_to_bigger_better() * score.back()`; if
//!     `cur - best_score > early_stopping_min_delta` → update `best_score` +
//!     `best_iter` (improvement); else if `iter - best_iter >= early_stopping_round`
//!     → STOP. `first_metric_only` evaluates only metric `j == 0` for the decision.
//!   - on stop the trailing `early_stopping_round * num_class` trees are popped so
//!     prediction defaults to `best_iteration` (gbdt.cpp:484).
//!   - `OutputMetric` / eval cadence (gbdt.cpp:551-609): `metric_freq` gates which
//!     iters evaluate; `is_provide_training_metric` adds the training set; the
//!     multi-metric list is evaluated per round (MET-02).
//!
//! IMPORTANT (CR-02, 06-06): under early stopping the caller must eval the valid
//! set EVERY iteration — `metric_freq` gates only the recorded eval-HISTORY, NOT the
//! ES decision cadence. In C++ the valid-metric + ES block is
//! `if (need_output || early_stopping_round_ > 0)` (gbdt.cpp:574); `need_output`
//! (`iter % metric_freq == 0`) only guards the `Log::Info` line, so when ES is on
//! the valid eval + stop check run on every iter regardless of `metric_freq`. The
//! facade (booster.rs train loop) implements this: it computes the valid `row` and
//! calls `EarlyStopping::update` whenever ES is enabled, and pushes to
//! `valid_eval_history` only on the `metric_freq` cadence.
//!
//! This module owns the DECISION + cadence state machine; the actual metric values
//! are computed by the caller (the facade, which holds the metric enums + the valid
//! sets) and fed in via [`EvalSnapshot`]. Keeping the value computation in the
//! facade avoids inverting the crate dependency (lgbm-boosting would otherwise need
//! every metric's `Eval`), while this module stays the single source of the C++
//! stop arithmetic.

/// The most-negative f64 — C++ `kMinScore` (gbdt.cpp:215), the `best_score_` init
/// so the first evaluation always counts as an improvement.
pub const K_MIN_SCORE: f64 = f64::NEG_INFINITY;

/// One metric's identity for the stop decision: its `factor_to_bigger_better`
/// (`+1` for AUC, `-1` for losses) and its name (for the eval-history key). The
/// caller builds these from the configured metric list (MET-02 multi-metric).
#[derive(Debug, Clone, PartialEq)]
pub struct MetricSpec {
    /// The metric name (the eval-history key, e.g. `"l2"`, `"auc"`).
    pub name: String,
    /// C++ `factor_to_bigger_better` — multiplies the raw metric so "bigger is
    /// always better" for the comparison (`+1` AUC, `-1` losses).
    pub factor_to_bigger_better: f64,
}

/// One eval round's snapshot: the metric values for each (valid_set, metric).
/// `values[i][j]` is valid set `i`, metric `j`'s raw value at this iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalSnapshot {
    /// `values[valid_set][metric]` — the raw metric value (the caller already
    /// applied `Eval`; the factor is applied here in the decision).
    pub values: Vec<Vec<f64>>,
}

/// The early-stopping decision state machine (BST-07). Tracks `best_score` +
/// `best_iter` per (valid_set, metric) and decides, each eval round, whether to
/// STOP. Mirrors the C++ `GBDT` early-stop members + decision (gbdt.cpp:591-608).
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// `early_stopping_round` — STOP after this many non-improving rounds. `<= 0`
    /// disables early stopping entirely.
    round: i32,
    /// `early_stopping_min_delta` — an improvement must exceed this to count.
    min_delta: f64,
    /// `first_metric_only` — evaluate only metric `j == 0` for the stop decision.
    first_metric_only: bool,
    /// `num_class` — the trailing-tree-pop multiplier (`round * num_class`).
    num_class: i32,
    /// Per-metric specs (the factor + name), length = #metrics.
    metrics: Vec<MetricSpec>,
    /// `best_score_[valid_set][metric]` (factor-applied), init `K_MIN_SCORE`.
    best_score: Vec<Vec<f64>>,
    /// `best_iter_[valid_set][metric]` (the iter that achieved `best_score`).
    best_iter: Vec<Vec<i32>>,
    /// The recorded `best_iteration` (1-based round) once a stop fires, else the
    /// last evaluated round. `0` until the first eval.
    best_iteration: i32,
    /// Whether a stop has fired.
    stopped: bool,
}

impl EarlyStopping {
    /// Construct the decision machine for `num_valid_sets` validation sets and the
    /// configured `metrics` (MET-02 order). `round <= 0` ⇒ early stopping disabled
    /// ([`update`](Self::update) never stops). Mirrors the C++ `best_score_` /
    /// `best_iter_` allocation + `kMinScore` init.
    pub fn new(
        round: i32,
        min_delta: f64,
        first_metric_only: bool,
        num_class: i32,
        num_valid_sets: usize,
        metrics: Vec<MetricSpec>,
    ) -> Self {
        let n_metrics = metrics.len();
        Self {
            round,
            min_delta,
            first_metric_only,
            num_class: num_class.max(1),
            metrics,
            best_score: vec![vec![K_MIN_SCORE; n_metrics]; num_valid_sets],
            best_iter: vec![vec![0; n_metrics]; num_valid_sets],
            best_iteration: 0,
            stopped: false,
        }
    }

    /// Whether early stopping is active (`round > 0`).
    pub fn is_enabled(&self) -> bool {
        self.round > 0
    }

    /// Feed one eval round's [`EvalSnapshot`] at `iter` (0-based completed-iteration
    /// index) and return `true` if training should STOP after this round.
    ///
    /// Mirrors gbdt.cpp:591-608 exactly: per (valid_set i, metric j) (skipping
    /// `j > 0` when `first_metric_only`), `cur = factor[j] * values[i][j]`; if
    /// `cur - best_score[i][j] > min_delta` → improvement (update best + best_iter);
    /// else if `iter - best_iter[i][j] >= round` → STOP. The first STOP across all
    /// (i, j) ends training; `best_iteration` is set to `best_iter + 1` (1-based)
    /// of the metric that triggered.
    pub fn update(&mut self, iter: i32, snapshot: &EvalSnapshot) -> bool {
        // Always advance the running best_iteration to the latest evaluated round
        // (the C++ best_iteration_ defaults to the last round when no stop fires).
        self.best_iteration = iter + 1;
        if !self.is_enabled() {
            return false;
        }
        let n_metrics = self.metrics.len();
        let metric_upper = if self.first_metric_only { 1.min(n_metrics) } else { n_metrics };
        for (i, vals) in snapshot.values.iter().enumerate() {
            for j in 0..metric_upper {
                let factor = self.metrics[j].factor_to_bigger_better;
                let cur = factor * vals[j];
                if cur - self.best_score[i][j] > self.min_delta {
                    // improvement.
                    self.best_score[i][j] = cur;
                    self.best_iter[i][j] = iter;
                } else if iter - self.best_iter[i][j] >= self.round {
                    // STOP: best iteration is best_iter (1-based).
                    self.best_iteration = self.best_iter[i][j] + 1;
                    self.stopped = true;
                    return true;
                }
            }
        }
        false
    }

    /// The recorded `best_iteration` (1-based round). When a stop fired this is the
    /// best metric's round; otherwise it is the last evaluated round.
    pub fn best_iteration(&self) -> i32 {
        self.best_iteration
    }

    /// Whether a stop has fired.
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// The number of trailing trees to POP on a stop — C++ `early_stopping_round *
    /// num_tree_per_iteration` (gbdt.cpp:484): the trees grown AFTER `best_iteration`
    /// up to the stopping round, so prediction defaults to `best_iteration`.
    ///
    /// `total_iters` is the number of iterations actually run (the loop ends the
    /// iteration on which `update` returned true). The trailing trees are those from
    /// `best_iteration` (exclusive) to `total_iters`, i.e.
    /// `(total_iters - best_iteration) * num_class`.
    pub fn trailing_trees_to_pop(&self, total_iters: i32) -> i32 {
        if !self.stopped {
            return 0;
        }
        ((total_iters - self.best_iteration).max(0)) * self.num_class
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loss_specs(n: usize) -> Vec<MetricSpec> {
        (0..n)
            .map(|j| MetricSpec {
                name: format!("loss{j}"),
                factor_to_bigger_better: -1.0,
            })
            .collect()
    }

    #[test]
    fn improvement_updates_best_then_stops_after_round() {
        // One valid set, one loss metric, round=2, min_delta=0.
        let mut es = EarlyStopping::new(2, 0.0, false, 1, 1, loss_specs(1));
        // Decreasing loss (improving) at iters 0,1,2 then plateau.
        // factor=-1, so a smaller loss => larger cur => improvement.
        assert!(!es.update(0, &EvalSnapshot { values: vec![vec![5.0]] }));
        assert!(!es.update(1, &EvalSnapshot { values: vec![vec![4.0]] }));
        assert!(!es.update(2, &EvalSnapshot { values: vec![vec![3.0]] })); // best_iter=2
        // Now plateau (no improvement). iter3: 3-2=1 < 2, no stop.
        assert!(!es.update(3, &EvalSnapshot { values: vec![vec![3.0]] }));
        // iter4: 4-2=2 >= 2 => STOP. best_iteration = best_iter+1 = 3.
        assert!(es.update(4, &EvalSnapshot { values: vec![vec![3.0]] }));
        assert!(es.stopped());
        assert_eq!(es.best_iteration(), 3);
    }

    #[test]
    fn min_delta_gates_marginal_improvement() {
        // round=1, min_delta=0.5: a loss drop of 0.3 is NOT an improvement.
        let mut es = EarlyStopping::new(1, 0.5, false, 1, 1, loss_specs(1));
        es.update(0, &EvalSnapshot { values: vec![vec![5.0]] }); // best_iter=0
        // iter1: loss 4.7 => cur=-4.7; cur-(-5.0)=0.3 <= 0.5 => no improvement;
        // 1-0=1 >= 1 => STOP. best_iteration = 0+1 = 1.
        assert!(es.update(1, &EvalSnapshot { values: vec![vec![4.7]] }));
        assert_eq!(es.best_iteration(), 1);
    }

    #[test]
    fn first_metric_only_ignores_later_metrics() {
        // Two metrics; first_metric_only => only metric 0 drives the stop decision.
        let mut es = EarlyStopping::new(1, 0.0, true, 1, 1, loss_specs(2));
        // metric0 improves every round (no stop), metric1 plateaus (would stop if
        // evaluated). With first_metric_only, metric1 is ignored => no stop.
        es.update(0, &EvalSnapshot { values: vec![vec![5.0, 9.0]] });
        assert!(!es.update(1, &EvalSnapshot { values: vec![vec![4.0, 9.0]] }));
        assert!(!es.update(2, &EvalSnapshot { values: vec![vec![3.0, 9.0]] }));
        assert!(!es.stopped());
        // Without first_metric_only, metric1's plateau would stop at iter1 (round=1).
        let mut es2 = EarlyStopping::new(1, 0.0, false, 1, 1, loss_specs(2));
        es2.update(0, &EvalSnapshot { values: vec![vec![5.0, 9.0]] });
        assert!(es2.update(1, &EvalSnapshot { values: vec![vec![4.0, 9.0]] }));
    }

    #[test]
    fn disabled_when_round_not_positive() {
        let mut es = EarlyStopping::new(0, 0.0, false, 1, 1, loss_specs(1));
        assert!(!es.is_enabled());
        for k in 0..10 {
            assert!(!es.update(k, &EvalSnapshot { values: vec![vec![5.0]] }));
        }
        assert!(!es.stopped());
        assert_eq!(es.best_iteration(), 10); // last evaluated round
    }

    #[test]
    fn trailing_trees_pop_count() {
        // round=2, num_class=3, best_iteration=3, total_iters=5 =>
        // (5-3)*3 = 6 trailing trees popped.
        let mut es = EarlyStopping::new(2, 0.0, false, 3, 1, loss_specs(1));
        es.update(0, &EvalSnapshot { values: vec![vec![5.0]] });
        es.update(1, &EvalSnapshot { values: vec![vec![4.0]] });
        es.update(2, &EvalSnapshot { values: vec![vec![3.0]] }); // best_iter=2 (round3)
        es.update(3, &EvalSnapshot { values: vec![vec![3.0]] });
        assert!(es.update(4, &EvalSnapshot { values: vec![vec![3.0]] })); // stop, total=5
        assert_eq!(es.best_iteration(), 3);
        assert_eq!(es.trailing_trees_to_pop(5), 6);
    }

    // ===================== metric_infra (MET-02) =====================
    // The metric-eval cadence (metric_freq) + training-metric inclusion lives in the
    // facade loop; here we prove the decision-machine pieces it relies on: the
    // multi-metric EvalSnapshot shape and per-metric factor handling.

    #[test]
    fn metric_infra_multi_metric_snapshot_all_evaluated() {
        // Two metrics, NOT first_metric_only: BOTH metrics participate in the stop
        // decision (a plateau in EITHER can trigger the round-based stop).
        let mut es = EarlyStopping::new(1, 0.0, false, 1, 1, loss_specs(2));
        es.update(0, &EvalSnapshot { values: vec![vec![5.0, 5.0]] }); // best_iter both=0
        // metric0 keeps improving, metric1 plateaus => metric1 stops at iter1.
        assert!(es.update(1, &EvalSnapshot { values: vec![vec![4.0, 5.0]] }));
    }

    #[test]
    fn metric_infra_multiple_valid_sets_tracked_independently() {
        // Two valid sets, one metric: each set's best is tracked independently; a
        // plateau in set 0 triggers even while set 1 improves.
        let mut es = EarlyStopping::new(1, 0.0, false, 1, 2, loss_specs(1));
        es.update(0, &EvalSnapshot { values: vec![vec![5.0], vec![9.0]] });
        // set0 plateaus (5.0), set1 improves (8.0) => set0 stops at iter1.
        assert!(es.update(1, &EvalSnapshot { values: vec![vec![5.0], vec![8.0]] }));
    }

    #[test]
    fn auc_factor_plus_one_bigger_is_better() {
        // AUC: factor +1, so a LARGER auc => larger cur => improvement.
        let specs = vec![MetricSpec { name: "auc".into(), factor_to_bigger_better: 1.0 }];
        let mut es = EarlyStopping::new(1, 0.0, false, 1, 1, specs);
        es.update(0, &EvalSnapshot { values: vec![vec![0.7]] }); // best_iter=0
        // iter1 auc 0.8 > 0.7 => improvement (factor +1), no stop.
        assert!(!es.update(1, &EvalSnapshot { values: vec![vec![0.8]] }));
        // iter2 auc 0.8 (no improvement) => 2-1=1 >= 1 => STOP, best_iteration=2.
        assert!(es.update(2, &EvalSnapshot { values: vec![vec![0.8]] }));
        assert_eq!(es.best_iteration(), 2);
    }
}
