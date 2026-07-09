"""Model persistence + pickle (D-10): the final Phase-8 user-facing surface.

Proves the C++-compatible model text round-trips through the Python surface
(``model_to_string`` / ``save_model`` / ``from_model_string`` / ``from_model_file``),
that a model trained by REAL ``lightgbm`` 4.6 cross-loads in ``lightgbm_rs`` and
predicts within ~1e-6 (the D-10 cross-format contract), that both a ``Booster``
and a fitted ``LGBMClassifier`` survive a ``pickle`` round-trip (sklearn-pipeline
requirement), and that malformed model text raises a TYPED exception rather than
crashing the interpreter (Security V5, T-08-08-01).

Run ``maturin develop`` in ``crates/lgbm-python`` first so the ``_core`` extension
reflects the persistence ``#[pymethods]``. SKIPs (with a printed reason — never a
silent pass) when the rs extension or the real ``lightgbm`` reference is absent.

NOTE: ``LightGBM/`` (the C++ reference tree) is never git-added by this test.
"""

import pickle

import numpy as np
import pytest

lightgbm_rs = pytest.importorskip(
    "lightgbm_rs",
    reason="lightgbm_rs extension not built — run `maturin develop` in crates/lgbm-python first",
)


def _params():
    # Determinism knobs pinned to REFERENCE_MANIFEST on both sides.
    return {
        "objective": "regression",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "max_bin": 255,
        "min_data_in_bin": 3,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": 0,
        "verbosity": -1,
    }


def _data(seed=0, n=300, d=6, n_test=50):
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    y = X @ coef + 0.1 * rng.standard_normal(n)
    Xt = rng.standard_normal((n_test, d))
    return X, y, Xt


def _rs_train(X, y, num_round=20):
    ds = lightgbm_rs.Dataset(X, y)
    return lightgbm_rs.train(_params(), ds, num_boost_round=num_round)


def test_text_roundtrip():
    """model_to_string -> Booster.from_model_string predicts byte-identically."""
    X, y, Xt = _data()
    model = _rs_train(X, y)
    text = model.model_to_string()
    assert "tree" in text  # sanity: real model text, not empty

    loaded = lightgbm_rs.Booster.from_model_string(text)

    p0 = np.asarray(model.predict(Xt), dtype=np.float64).reshape(-1)
    p1 = np.asarray(loaded.predict(np.ascontiguousarray(Xt, dtype=np.float64)), dtype=np.float64).reshape(-1)
    assert p0.shape == p1.shape
    np.testing.assert_array_equal(
        p0, p1, err_msg="text round-trip predictions are not byte-identical"
    )


def test_save_load_file(tmp_path):
    """save_model(path) -> Booster.from_model_file predicts identically."""
    X, y, Xt = _data(seed=1)
    model = _rs_train(X, y)
    path = tmp_path / "model.txt"
    model.save_model(str(path))
    assert path.exists() and path.stat().st_size > 0

    loaded = lightgbm_rs.Booster.from_model_file(str(path))

    p0 = np.asarray(model.predict(Xt), dtype=np.float64).reshape(-1)
    p1 = np.asarray(loaded.predict(np.ascontiguousarray(Xt, dtype=np.float64)), dtype=np.float64).reshape(-1)
    np.testing.assert_array_equal(
        p0, p1, err_msg="file save/load predictions are not byte-identical"
    )


def test_cross_format_load(tmp_path):
    """A model trained by REAL lightgbm 4.6 loads in lightgbm_rs within ~1e-6 (D-10)."""
    ref = pytest.importorskip(
        "lightgbm",
        reason="real lightgbm (4.6 reference) not installed — cross-format parity cannot run",
    )
    X, y, Xt = _data(seed=2)
    num_round = 20

    ref_ds = ref.Dataset(X, label=y, params={"verbosity": -1})
    ref_model = ref.train(_params(), ref_ds, num_boost_round=num_round)
    ref_pred = np.asarray(ref_model.predict(Xt), dtype=np.float64).reshape(-1)
    assert ref_pred.std() > 0.0, "reference predictions constant — cross-format check would be vacuous"

    # Save the REAL lightgbm model to its text format, load it in lightgbm_rs.
    ref_path = tmp_path / "ref_model.txt"
    ref_model.save_model(str(ref_path))
    rs_loaded = lightgbm_rs.Booster.from_model_file(str(ref_path))
    rs_pred = np.asarray(
        rs_loaded.predict(np.ascontiguousarray(Xt, dtype=np.float64)), dtype=np.float64
    ).reshape(-1)

    assert rs_pred.shape == ref_pred.shape
    np.testing.assert_allclose(
        rs_pred,
        ref_pred,
        atol=1e-6,
        err_msg="cross-format: a real-lightgbm model does not predict identically in lightgbm_rs (D-10)",
    )


def test_pickle_roundtrip_booster():
    """pickle.dumps/loads a trained Booster -> identical predictions (sklearn-pipeline req)."""
    X, y, Xt = _data(seed=4)
    model = _rs_train(X, y)
    core = lightgbm_rs.Booster.from_model_string(model.model_to_string())

    blob = pickle.dumps(core)
    restored = pickle.loads(blob)

    p0 = np.asarray(core.predict(np.ascontiguousarray(Xt, dtype=np.float64)), dtype=np.float64).reshape(-1)
    p1 = np.asarray(restored.predict(np.ascontiguousarray(Xt, dtype=np.float64)), dtype=np.float64).reshape(-1)
    np.testing.assert_array_equal(
        p0, p1, err_msg="Booster pickle round-trip predictions differ"
    )


def test_pickle_roundtrip_estimator():
    """pickle.dumps/loads a fitted LGBMClassifier -> identical predictions (D-10)."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 5))
    y = (X[:, 0] + 0.3 * rng.standard_normal(200) > 0).astype(int)
    Xt = rng.standard_normal((40, 5))

    clf = lightgbm_rs.LGBMClassifier(n_estimators=15, num_leaves=7, random_state=0, verbosity=-1)
    clf.fit(X, y)
    p0 = clf.predict_proba(Xt)

    blob = pickle.dumps(clf)
    restored = pickle.loads(blob)
    p1 = restored.predict_proba(Xt)

    assert p0.shape == p1.shape
    np.testing.assert_array_equal(
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
        err_msg="LGBMClassifier pickle round-trip predictions differ",
    )


def test_malformed_model_text_raises():
    """Garbage model text raises a typed exception, never a panic (Security V5, T-08-08-01)."""
    with pytest.raises((ValueError, lightgbm_rs.LightGBMError)):
        lightgbm_rs.Booster.from_model_string("this is not a valid lightgbm model\n@@@\x00garbage")


def test_malformed_model_file_raises(tmp_path):
    """A nonexistent / malformed model file raises a typed exception (T-08-08-02)."""
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises((ValueError, lightgbm_rs.LightGBMError, OSError)):
        lightgbm_rs.Booster.from_model_file(str(missing))
