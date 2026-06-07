"""A/B parity (PYB-01, SC core value): the same ``(X, y, params)`` fed to BOTH
the real ``lightgbm`` 4.6 package and ``lightgbm_rs`` must produce predictions
agreeing to within ~1e-6 on the deterministic CPU anchor.

Both sides are pinned to the REFERENCE_MANIFEST determinism knobs
(``deterministic=True``, ``force_row_wise=True``, ``num_threads=1``, fixed
``seed``). The objective is L2 regression — the bit-exact CPU spine.

SKIPs (with a printed reason — never silently passes) if either the rs extension
or the real ``lightgbm`` reference is not installed.
"""

import numpy as np
import pytest

lightgbm_rs = pytest.importorskip(
    "lightgbm_rs",
    reason="lightgbm_rs extension not built — run `maturin develop` in crates/lgbm-python first",
)
ref = pytest.importorskip(
    "lightgbm",
    reason="real lightgbm (4.6 reference) not installed — A/B parity cannot run",
)


# Determinism knobs pinned to REFERENCE_MANIFEST on BOTH sides.
def _params():
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


def test_ab_parity_regression_l2():
    X, y, Xt = _data()
    num_round = 20
    params = _params()

    # rs side
    rs_ds = lightgbm_rs.Dataset(X, y)
    rs_model = lightgbm_rs.train(params, rs_ds, num_boost_round=num_round)
    rs_pred = np.asarray(rs_model.predict(Xt), dtype=np.float64).reshape(-1)

    # real lightgbm 4.6 side
    ref_ds = ref.Dataset(X, label=y, params={"verbosity": -1})
    ref_model = ref.train(params, ref_ds, num_boost_round=num_round)
    ref_pred = np.asarray(ref_model.predict(Xt), dtype=np.float64).reshape(-1)

    assert rs_pred.shape == ref_pred.shape
    np.testing.assert_allclose(
        rs_pred,
        ref_pred,
        atol=1e-6,
        err_msg="lightgbm_rs predictions diverge from real lightgbm 4.6 beyond 1e-6",
    )


def test_ab_parity_is_actually_running():
    # Guard against a vacuous pass: confirm the reference really is the LightGBM
    # package and produces non-trivial (non-constant) predictions.
    assert hasattr(ref, "train") and hasattr(ref, "Dataset")
    X, y, Xt = _data(seed=3)
    ref_ds = ref.Dataset(X, label=y, params={"verbosity": -1})
    ref_model = ref.train(_params(), ref_ds, num_boost_round=10)
    ref_pred = np.asarray(ref_model.predict(Xt)).reshape(-1)
    assert ref_pred.std() > 0.0, "reference predictions are constant — A/B would be vacuous"
