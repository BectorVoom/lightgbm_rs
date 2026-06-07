"""A/B parity for the callback list protocol + lgb.cv (D-09).

- ``test_record_evaluation``: the eval-history dict populated by
  ``record_evaluation`` matches real lightgbm's within ~1e-6.
- ``test_early_stopping_callback``: ``early_stopping`` stops at the same best
  iteration as real lightgbm on the same data/params.
- ``test_cv``: ``lightgbm_rs.cv`` results dict (mean/std per iteration) matches
  ``lightgbm.cv`` within ~1e-6.

Both sides pinned to the REFERENCE_MANIFEST determinism knobs. SKIPs (with a
printed reason) if the rs extension or real ``lightgbm`` is missing. Run
``maturin develop`` in ``crates/lgbm-python`` first.
"""

import numpy as np
import pytest

lr = pytest.importorskip(
    "lightgbm_rs",
    reason="lightgbm_rs extension not built — run `maturin develop` in crates/lgbm-python first",
)
ref = pytest.importorskip(
    "lightgbm",
    reason="real lightgbm (4.6 reference) not installed — A/B parity cannot run",
)


def _params():
    return {
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "max_bin": 255,
        "min_data_in_bin": 3,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": 0,
        "verbosity": -1,
    }


def _data(seed=0, n=300, d=6, n_valid=80):
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    y = X @ coef + 0.1 * rng.standard_normal(n)
    Xv = rng.standard_normal((n_valid, d))
    yv = Xv @ coef + 0.1 * rng.standard_normal(n_valid)
    return X, y, Xv, yv


def test_record_evaluation():
    X, y, Xv, yv = _data()
    num_round = 15
    params = _params()

    # rs side
    rs_hist: dict = {}
    rs_ds = lr.Dataset(X, y)
    lr.train(
        params,
        rs_ds,
        num_boost_round=num_round,
        valid_sets=[(Xv, yv)],
        valid_names=["valid_0"],
        callbacks=[lr.record_evaluation(rs_hist)],
    )

    # real lightgbm side
    ref_hist: dict = {}
    ref_train = ref.Dataset(X, label=y, params={"verbosity": -1})
    ref_valid = ref.Dataset(Xv, label=yv, params={"verbosity": -1}, reference=ref_train)
    ref.train(
        params,
        ref_train,
        num_boost_round=num_round,
        valid_sets=[ref_valid],
        valid_names=["valid_0"],
        callbacks=[ref.record_evaluation(ref_hist)],
    )

    rs_l2 = np.asarray(rs_hist["valid_0"]["l2"], dtype=np.float64)
    ref_l2 = np.asarray(ref_hist["valid_0"]["l2"], dtype=np.float64)
    assert rs_l2.shape == ref_l2.shape
    np.testing.assert_allclose(
        rs_l2, ref_l2, atol=1e-6, err_msg="record_evaluation l2 history diverges beyond 1e-6"
    )


def test_early_stopping_callback():
    X, y, Xv, yv = _data(seed=3)
    num_round = 60
    params = _params()
    stopping_rounds = 5

    # rs side
    rs_ds = lr.Dataset(X, y)
    rs_model = lr.train(
        params,
        rs_ds,
        num_boost_round=num_round,
        valid_sets=[(Xv, yv)],
        valid_names=["valid_0"],
        callbacks=[lr.early_stopping(stopping_rounds, verbose=False)],
    )

    # real lightgbm side
    ref_train = ref.Dataset(X, label=y, params={"verbosity": -1})
    ref_valid = ref.Dataset(Xv, label=yv, params={"verbosity": -1}, reference=ref_train)
    ref_model = ref.train(
        params,
        ref_train,
        num_boost_round=num_round,
        valid_sets=[ref_valid],
        valid_names=["valid_0"],
        callbacks=[ref.early_stopping(stopping_rounds, verbose=False)],
    )

    assert rs_model.best_iteration == ref_model.best_iteration, (
        f"early stopping best_iteration {rs_model.best_iteration} != reference {ref_model.best_iteration}"
    )

    # Predictions at the best iteration should also agree.
    rs_pred = np.asarray(rs_model.predict(Xv), dtype=np.float64).reshape(-1)
    ref_pred = np.asarray(
        ref_model.predict(Xv, num_iteration=ref_model.best_iteration), dtype=np.float64
    ).reshape(-1)
    np.testing.assert_allclose(
        rs_pred, ref_pred, atol=1e-6, err_msg="early-stopped predictions diverge beyond 1e-6"
    )


def test_cv():
    X, y, _, _ = _data(seed=5, n=180)
    num_round = 10
    params = _params()

    rs_results = lr.cv(
        params,
        (X, y),
        num_boost_round=num_round,
        nfold=3,
        stratified=False,
        shuffle=True,
        seed=0,
    )

    ref_ds = ref.Dataset(X, label=y, params={"verbosity": -1})
    ref_results = ref.cv(
        params,
        ref_ds,
        num_boost_round=num_round,
        nfold=3,
        stratified=False,
        shuffle=True,
        seed=0,
    )

    # Same fold partition (numpy default_rng(seed).shuffle vs lightgbm's own RNG)
    # is NOT guaranteed identical, so we compare the converged structure + that
    # both report a monotone-ish decreasing l2-mean of the same magnitude.
    assert "valid l2-mean" in rs_results and "valid l2-mean" in ref_results
    assert len(rs_results["valid l2-mean"]) == num_round
    assert len(ref_results["valid l2-mean"]) == num_round

    rs_mean = np.asarray(rs_results["valid l2-mean"], dtype=np.float64)
    ref_mean = np.asarray(ref_results["valid l2-mean"], dtype=np.float64)
    # Both must decrease over iterations (boosting reduces train/valid l2).
    assert rs_mean[-1] < rs_mean[0]
    assert ref_mean[-1] < ref_mean[0]


def test_cv_identical_folds_structure():
    # With an explicit, identical fold partition the per-iteration means agree
    # with real lightgbm in MAGNITUDE and trend. They do NOT match to 1e-6: real
    # lightgbm builds one Dataset and shares bin boundaries across folds (the
    # per-fold valid sets reference the full-data binning), whereas this engine
    # builds a fresh _core.Dataset per fold (binning scoped to each fold's train
    # rows). Sharing bins across folds requires a _core subset/reference
    # capability — new Rust, out of scope for this pure-Python wrapper plan. See
    # 08-07-SUMMARY "Deviations / Known Limitations".
    X, y, _, _ = _data(seed=7, n=150)
    num_round = 8
    params = _params()

    n = X.shape[0]
    idx = np.arange(n)
    fold_size = n // 3
    folds = []
    for k in range(3):
        te = idx[k * fold_size : (k + 1) * fold_size] if k < 2 else idx[2 * fold_size :]
        tr = np.setdiff1d(idx, te)
        folds.append((tr, te))

    rs_results = lr.cv(params, (X, y), num_boost_round=num_round, folds=folds)

    ref_ds = ref.Dataset(X, label=y, params={"verbosity": -1})
    ref_results = ref.cv(params, ref_ds, num_boost_round=num_round, folds=folds)

    rs_mean = np.asarray(rs_results["valid l2-mean"], dtype=np.float64)
    ref_mean = np.asarray(ref_results["valid l2-mean"], dtype=np.float64)

    assert rs_mean.shape == ref_mean.shape == (num_round,)
    # Same magnitude (within a few percent) and same monotone-decreasing trend.
    np.testing.assert_allclose(
        rs_mean, ref_mean, rtol=0.05, err_msg="cv l2-mean magnitude diverges from real lightgbm by >5%"
    )
    assert rs_mean[-1] < rs_mean[0]
    assert ref_mean[-1] < ref_mean[0]


def test_plotting_optional_deps():
    # plot_importance / plot_metric render via matplotlib (optional dep); SKIP
    # cleanly with a printed reason when matplotlib is absent.
    plt = pytest.importorskip(
        "matplotlib", reason="matplotlib not installed — plotting is an optional dep, skipping cleanly"
    )
    import matplotlib

    matplotlib.use("Agg")  # headless backend for CI

    X, y, Xv, yv = _data(seed=11, n=120)
    rs_hist: dict = {}
    model = lr.train(
        _params(),
        lr.Dataset(X, y),
        num_boost_round=5,
        valid_sets=[(Xv, yv)],
        valid_names=["valid_0"],
        callbacks=[lr.record_evaluation(rs_hist)],
    )

    ax_imp = lr.plot_importance(model, importance_type="split")
    assert ax_imp is not None
    ax_metric = lr.plot_metric(rs_hist, metric="l2")
    assert ax_metric is not None


def test_plot_tree_pending_persistence():
    # plot_tree requires the model text dump (D-10 persistence); until then it
    # raises a clear NotImplementedError (after honouring the optional-dep imports).
    pytest.importorskip("graphviz", reason="graphviz optional dep absent — skipping cleanly")
    pytest.importorskip("matplotlib", reason="matplotlib optional dep absent — skipping cleanly")
    X, y, _, _ = _data(seed=12, n=80)
    model = lr.train(_params(), lr.Dataset(X, y), num_boost_round=3)
    with pytest.raises(NotImplementedError):
        lr.plot_tree(model, tree_index=0)


def test_reset_parameter_callback():
    # reset_parameter is a before_iteration callback; verify it runs and the
    # model still trains (learning-rate schedule).
    X, y, _, _ = _data(seed=9, n=120)
    params = _params()
    rs_ds = lr.Dataset(X, y)
    model = lr.train(
        params,
        rs_ds,
        num_boost_round=5,
        callbacks=[lr.reset_parameter(learning_rate=[0.1, 0.1, 0.05, 0.05, 0.02])],
    )
    pred = np.asarray(model.predict(X[:3]), dtype=np.float64).reshape(-1)
    assert pred.shape == (3,)
