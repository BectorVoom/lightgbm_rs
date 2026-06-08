"""Smoke test (PYB-01): import the extension, train on a numpy dense matrix,
and confirm ``predict`` returns an OWNED numpy array of the right shape.

SKIPs cleanly (never silently passes) if the extension has not been built with
``maturin develop`` in the active venv.
"""

import numpy as np
import pytest

lightgbm_rs = pytest.importorskip(
    "lightgbm_rs",
    reason="lightgbm_rs extension not built — run `maturin develop` in crates/lgbm-python first",
)


def _params():
    # Deterministic L2 regression — the bit-exact CPU spine objective.
    return {
        "objective": "regression",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": 0,
        "verbosity": -1,
    }


def test_train_predict_owned_array():
    rng = np.random.default_rng(42)
    n, d = 200, 5
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + 0.1 * rng.standard_normal(n)

    ds = lightgbm_rs.Dataset(X, y)
    assert ds.num_data() == n
    assert ds.num_feature() == d

    model = lightgbm_rs.train(_params(), ds, num_boost_round=10)
    assert model.num_iteration() == 10

    pred = model.predict(X)
    assert isinstance(pred, np.ndarray), "predict must return an owned numpy array"
    assert pred.shape[0] == n
    assert np.all(np.isfinite(pred))

    # Owned output: mutating the returned array must not raise (it is writable,
    # not a read-only view lent from Rust) — SC#1.
    pred[0] = 123.0
    assert pred[0] == 123.0


def test_non_contiguous_input_is_handled():
    # A Fortran-ordered (column-major) array exercises the contiguity branch in
    # the marshaller (Pitfall 3 / SC#2): the result must match the C-order copy.
    rng = np.random.default_rng(7)
    X = np.asfortranarray(rng.standard_normal((120, 4)))
    y = rng.standard_normal(120)

    ds = lightgbm_rs.Dataset(X, y)
    model = lightgbm_rs.train(_params(), ds, num_boost_round=5)

    pred_f = np.asarray(model.predict(X)).reshape(-1)
    pred_c = np.asarray(model.predict(np.ascontiguousarray(X))).reshape(-1)
    np.testing.assert_array_equal(pred_f, pred_c)


def test_label_length_mismatch_raises_valueerror():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    y = rng.standard_normal(9)  # wrong length
    with pytest.raises(ValueError):
        lightgbm_rs.Dataset(X, y)


def test_predict_too_few_features_raises_valueerror():
    # Code-review CR-01: a too-narrow predict matrix must raise ValueError at the
    # boundary, NOT panic out-of-bounds inside the GIL-released predict.
    rng = np.random.default_rng(11)
    n, d = 150, 5
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d)
    model = lightgbm_rs.train(_params(), lightgbm_rs.Dataset(X, y), num_boost_round=10)

    narrow = rng.standard_normal((4, d - 2))  # fewer columns than trained
    with pytest.raises(ValueError):
        model.predict(narrow)


def test_refit_too_few_features_raises_valueerror():
    # Code-review CR-02: same width guard on the refit data matrix.
    rng = np.random.default_rng(12)
    n, d = 150, 5
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d)
    model = lightgbm_rs.train(_params(), lightgbm_rs.Dataset(X, y), num_boost_round=10)

    narrow = rng.standard_normal((20, d - 1))
    ylab = rng.standard_normal(20)
    with pytest.raises(ValueError):
        model.refit(narrow, ylab)
