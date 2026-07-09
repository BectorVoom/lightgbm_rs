"""polars DataFrame input (PYB-02, D-03/D-04): numeric columns equal the numpy
path; ``Categorical``/``Enum`` columns route to LightGBM categorical features
matching real lightgbm 4.6; an explicit ``categorical_feature`` override takes
precedence over dtype routing.

SKIPs cleanly (printed reason, never silently passes) if the rs extension,
polars, or the real lightgbm reference is not installed.
"""

import numpy as np
import pytest

lightgbm_rs = pytest.importorskip(
    "lightgbm_rs",
    reason="lightgbm_rs extension not built — run `maturin develop` in crates/lgbm-python first",
)
pl = pytest.importorskip("polars", reason="polars not installed")


def _params():
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


def test_polars_numeric_matches_numpy():
    # A pure-numeric polars frame must densify to the SAME f64 rows as the numpy
    # array, hence bit-identical predictions.
    rng = np.random.default_rng(0)
    n, d = 250, 5
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + 0.05 * rng.standard_normal(n)
    Xt = rng.standard_normal((40, d))

    m_np = lightgbm_rs.train(_params(), lightgbm_rs.Dataset(X, y), num_boost_round=15)
    p_np = np.asarray(m_np.predict(Xt)).reshape(-1)

    df = pl.DataFrame({f"f{j}": X[:, j] for j in range(d)})
    m_pl = lightgbm_rs.train(_params(), lightgbm_rs.dataset_from_polars(df, y), num_boost_round=15)
    p_pl = np.asarray(m_pl.predict(Xt)).reshape(-1)

    np.testing.assert_array_equal(p_np, p_pl)


def test_enum_dtype_routes_like_integer_codes():
    # An Enum column with an explicit category order has physical codes 0..k, so
    # AUTO dtype-routing on the Enum column must produce the SAME model as the
    # equivalent integer-code column forced categorical via the override. This
    # proves auto-routing extracts the correct physical codes (no numpy round-trip).
    rng = np.random.default_rng(1)
    n = 400
    codes = rng.integers(0, 4, n)
    num = rng.standard_normal(n)
    y = 0.5 * num + codes.astype(float) + 0.05 * rng.standard_normal(n)
    letters = np.array(["a", "b", "c", "d"])

    df_enum = pl.DataFrame(
        {"cat": pl.Series(letters[codes]).cast(pl.Enum(["a", "b", "c", "d"])), "num": num}
    )
    m_enum = lightgbm_rs.train(
        _params(), lightgbm_rs.dataset_from_polars(df_enum, y), num_boost_round=20
    )

    df_int = pl.DataFrame({"cat": codes, "num": num})
    m_int = lightgbm_rs.train(
        _params(),
        lightgbm_rs.dataset_from_polars(df_int, y, categorical_feature=["cat"]),
        num_boost_round=20,
    )

    Xt = np.column_stack([codes[:60].astype(float), num[:60]])
    pe = np.asarray(m_enum.predict(Xt)).reshape(-1)
    pi = np.asarray(m_int.predict(Xt)).reshape(-1)
    np.testing.assert_allclose(pe, pi, atol=1e-6)


def test_categorical_ab_matches_real_lightgbm():
    ref = pytest.importorskip(
        "lightgbm", reason="real lightgbm (4.6 reference) not installed — A/B cannot run"
    )
    rng = np.random.default_rng(2)
    n = 500
    codes = rng.integers(0, 5, n)
    num = rng.standard_normal(n)
    y = 0.3 * num + np.array([0.0, 2.0, -1.5, 1.0, -0.5])[codes] + 0.05 * rng.standard_normal(n)
    X = np.column_stack([codes.astype(float), num])
    Xt = np.column_stack([rng.integers(0, 5, 60).astype(float), rng.standard_normal(60)])

    df = pl.DataFrame({"cat": codes, "num": num})
    ds = lightgbm_rs.dataset_from_polars(df, y, categorical_feature=["cat"])
    p_rs = np.asarray(
        lightgbm_rs.train(_params(), ds, num_boost_round=20).predict(Xt)
    ).reshape(-1)

    ref_ds = ref.Dataset(X, label=y, categorical_feature=[0], params={"verbosity": -1})
    p_ref = np.asarray(
        ref.train(_params(), ref_ds, num_boost_round=20).predict(Xt)
    ).reshape(-1)

    np.testing.assert_allclose(
        p_rs, p_ref, atol=1e-6, err_msg="polars categorical splits diverge from real lightgbm 4.6"
    )


def test_categorical_override_changes_routing():
    # A non-monotone per-category effect: treating the integer column as
    # categorical (override) must yield a DIFFERENT model than the auto numeric
    # routing — proving the override actually takes effect.
    rng = np.random.default_rng(3)
    n = 400
    codes = rng.integers(0, 4, n)
    num = rng.standard_normal(n)
    y = np.array([0.0, 3.0, -3.0, 1.0])[codes] + 0.2 * num + 0.03 * rng.standard_normal(n)
    Xt = np.column_stack([rng.integers(0, 4, 50).astype(float), rng.standard_normal(50)])

    df = pl.DataFrame({"cat": codes, "num": num})
    p_numeric = np.asarray(
        lightgbm_rs.train(_params(), lightgbm_rs.dataset_from_polars(df, y), num_boost_round=20)
        .predict(Xt)
    ).reshape(-1)
    p_cat = np.asarray(
        lightgbm_rs.train(
            _params(),
            lightgbm_rs.dataset_from_polars(df, y, categorical_feature=["cat"]),
            num_boost_round=20,
        ).predict(Xt)
    ).reshape(-1)

    assert not np.allclose(p_numeric, p_cat, atol=1e-6), (
        "categorical override had no effect — numeric and categorical models identical"
    )
