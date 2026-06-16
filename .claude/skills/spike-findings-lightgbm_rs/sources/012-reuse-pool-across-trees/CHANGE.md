# Spike 012 — production change (shipped)

The change is a 3-edit refactor in `crates/lgbm-treelearner/src/learner.rs`
(field `hist_pool: Option<HistogramPool>` + take/reset/store-back in `train_inner`),
not a standalone script. See the commit and the README. To inspect:

    git show <commit> -- crates/lgbm-treelearner/src/learner.rs
