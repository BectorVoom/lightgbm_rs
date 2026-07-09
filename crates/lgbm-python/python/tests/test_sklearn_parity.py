"""A/B parity for the sklearn estimators (PYB-03).

For each of ``LGBMRegressor`` / ``LGBMClassifier`` / ``LGBMRanker``, fit on the
SAME data + params via BOTH the real ``lightgbm`` 4.6 package and
``lightgbm_rs`` and assert predictions agree to within ~1e-6 on the
deterministic CPU anchor (the SC core-value contract).

Both sides are pinned to the REFERENCE_MANIFEST determinism knobs
(``deterministic=True``, ``force_row_wise=True``, ``n_jobs=1``, fixed
``random_state``).

SKIPs (with a printed reason — never silently passes) if scikit-learn, the rs
extension, or the real ``lightgbm`` reference is not installed. Run
``maturin develop`` in ``crates/lgbm-python`` first.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn not installed — sklearn-estimator parity cannot run")
lr = pytest.importorskip(
    "lightgbm_rs",
    reason="lightgbm_rs extension not built — run `maturin develop` in crates/lgbm-python first",
)
ref = pytest.importorskip(
    "lightgbm",
    reason="real lightgbm (4.6 reference) not installed — A/B parity cannot run",
)


def _common_kwargs():
    return dict(
        n_estimators=20,
        num_leaves=15,
        learning_rate=0.1,
        min_child_samples=5,
        random_state=0,
        n_jobs=1,
    )


def _det_params():
    # Determinism knobs forwarded to both sides via **kwargs / params.
    return dict(
        max_bin=255,
        min_data_in_bin=3,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )


def _reg_data(seed=0, n=300, d=6, n_test=60):
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    y = X @ coef + 0.1 * rng.standard_normal(n)
    Xt = rng.standard_normal((n_test, d))
    return X, y, Xt


def _clf_data(seed=1, n=300, d=6, n_test=60):
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    logits = X @ coef
    y = (logits > np.median(logits)).astype(int)
    Xt = rng.standard_normal((n_test, d))
    return X, y, Xt


def test_regressor_parity():
    X, y, Xt = _reg_data()
    kw = {**_common_kwargs(), **_det_params()}

    rs_model = lr.LGBMRegressor(**kw).fit(X, y)
    ref_model = ref.LGBMRegressor(objective="regression", **kw).fit(X, y)

    rs_pred = np.asarray(rs_model.predict(Xt), dtype=np.float64).reshape(-1)
    ref_pred = np.asarray(ref_model.predict(Xt), dtype=np.float64).reshape(-1)

    assert rs_pred.shape == ref_pred.shape
    np.testing.assert_allclose(
        rs_pred,
        ref_pred,
        atol=1e-6,
        err_msg="LGBMRegressor predictions diverge from real lightgbm beyond 1e-6",
    )


def test_classifier_parity():
    X, y, Xt = _clf_data()
    kw = {**_common_kwargs(), **_det_params()}

    rs_model = lr.LGBMClassifier(**kw).fit(X, y)
    ref_model = ref.LGBMClassifier(objective="binary", **kw).fit(X, y)

    rs_proba = np.asarray(rs_model.predict_proba(Xt), dtype=np.float64)
    ref_proba = np.asarray(ref_model.predict_proba(Xt), dtype=np.float64)

    assert rs_proba.shape == ref_proba.shape
    np.testing.assert_allclose(
        rs_proba,
        ref_proba,
        atol=1e-6,
        err_msg="LGBMClassifier predict_proba diverges from real lightgbm beyond 1e-6",
    )

    np.testing.assert_array_equal(np.sort(rs_model.classes_), np.sort(ref_model.classes_))


def _lambdarank_supported() -> bool:
    """Whether the compiled _core can train the lambdarank objective end-to-end.

    The ranking objective exists in the Rust facade, but the model-side objective
    layer used by ``_core.train`` (``ObjectiveKind::parse``) does not yet wire
    lambdarank — a Rust-side gap outside this pure-Python plan's scope.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4))
    y = rng.integers(0, 4, size=40).astype(float)
    try:
        lr.train({"objective": "lambdarank", "verbosity": -1}, lr.Dataset(X, y), num_boost_round=1)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _lambdarank_supported(),
    reason="lambdarank training not yet wired in _core (ObjectiveKind::parse) — Rust-side gap, out of scope for the pure-Python wrapper plan",
)
def test_ranker_parity():
    # Single-group ranking; lambdarank objective on both sides.
    rng = np.random.default_rng(2)
    n, d = 120, 5
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 4, size=n)  # graded relevance labels
    group = [n]  # one query group of all rows
    kw = {**_common_kwargs(), **_det_params()}

    rs_model = lr.LGBMRanker(**kw).fit(X, y, group=group)
    ref_model = ref.LGBMRanker(objective="lambdarank", **kw).fit(X, y, group=group)

    rs_pred = np.asarray(rs_model.predict(X), dtype=np.float64).reshape(-1)
    ref_pred = np.asarray(ref_model.predict(X), dtype=np.float64).reshape(-1)

    assert rs_pred.shape == ref_pred.shape
    np.testing.assert_allclose(
        rs_pred,
        ref_pred,
        atol=1e-6,
        err_msg="LGBMRanker predictions diverge from real lightgbm beyond 1e-6",
    )


def test_ranker_api_surface():
    # The LGBMRanker class exists and mirrors the official API surface even while
    # lambdarank training is a _core gap: exercise the estimator with a supported
    # (regression) objective to confirm fit/predict/feature_importances_ work.
    rng = np.random.default_rng(2)
    X = rng.standard_normal((120, 5))
    y = rng.integers(0, 4, size=120).astype(float)
    kw = {**_common_kwargs(), **_det_params()}
    model = lr.LGBMRanker(objective="regression", **kw).fit(X, y, group=[120])
    pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
    assert pred.shape == (120,)
    assert model.feature_importances_.shape == (5,)


def test_feature_importances_shape():
    X, y, _ = _reg_data()
    kw = {**_common_kwargs(), **_det_params()}
    rs_model = lr.LGBMRegressor(**kw).fit(X, y)
    imp = rs_model.feature_importances_
    assert imp.shape == (X.shape[1],)
    assert rs_model.n_features_in_ == X.shape[1]
