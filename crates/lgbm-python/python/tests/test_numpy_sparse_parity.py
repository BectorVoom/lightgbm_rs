"""A/B parity for the widened input surface (PYB-02): f32 + f64 dense numpy AND
scipy CSR/CSC sparse all fed to BOTH real ``lightgbm`` 4.6 and ``lightgbm_rs``
must agree to within ~1e-6 on the deterministic CPU anchor.

Also asserts the two single-width invariants this slice guarantees:
- ``test_f32_f64_self_consistency`` — f32 and f64 of the SAME data give identical
  ``lightgbm_rs`` predictions (the single f32->f64 widen site, T-08-03-03).
- ``test_malformed_csr_raises`` — a non-monotone CSR ``indptr`` raises a Python
  ``ValueError`` BEFORE any indexing, never a panic/abort (Security V5, T-08-03-01).

Both sides pin the REFERENCE_MANIFEST determinism knobs (``deterministic=True``,
``force_row_wise=True``, ``num_threads=1``, fixed ``seed``); objective is L2
regression — the bit-exact CPU spine.

Run ``maturin develop`` (with the project ``.venv`` active) before this suite so
the ``_core`` extension reflects the f32/sparse marshalling. SKIPs with a printed
reason (never silently passes) if ``lightgbm_rs``, real ``lightgbm``, or ``scipy``
is absent. Do NOT git-add ``LightGBM/``.
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
sparse = pytest.importorskip(
    "scipy.sparse",
    reason="scipy not installed — CSR/CSC sparse parity cannot run",
)


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


def _data(seed=0, n=300, d=6, n_test=50, sparsity=0.0):
    """Dense (n, d) design + targets. With ``sparsity`` > 0, zero out that
    fraction of entries so the sparse matrices have genuine structural zeros (and
    the dense-equivalent has the same zeros)."""
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    if sparsity > 0.0:
        mask = rng.random((n, d)) < sparsity
        X[mask] = 0.0
    y = X @ coef + 0.1 * rng.standard_normal(n)
    Xt = rng.standard_normal((n_test, d))
    return X.astype(np.float64), y.astype(np.float64), Xt.astype(np.float64)


def _rs_predict(rs_ds, Xt, num_round):
    model = lightgbm_rs.train(_params(), rs_ds, num_boost_round=num_round)
    return np.asarray(model.predict(Xt), dtype=np.float64).reshape(-1)


def _ref_predict(X, y, Xt, num_round):
    ref_ds = ref.Dataset(X, label=y, params={"verbosity": -1})
    model = ref.train(_params(), ref_ds, num_boost_round=num_round)
    return np.asarray(model.predict(Xt), dtype=np.float64).reshape(-1)


@pytest.mark.parametrize("kind", ["f32", "f64", "csr", "csc"])
def test_ab_parity_input_kinds(kind):
    """f32/f64 dense and CSR/CSC sparse each match real lightgbm within 1e-6."""
    num_round = 20
    # Sparse cases use a structurally-sparse X so CSR/CSC carry real zeros; the
    # reference is always fed the equivalent dense f64 X (same numbers).
    sparsity = 0.4 if kind in ("csr", "csc") else 0.0
    X, y, Xt = _data(sparsity=sparsity)

    if kind == "f64":
        rs_ds = lightgbm_rs.Dataset(np.ascontiguousarray(X, dtype=np.float64), y)
    elif kind == "f32":
        rs_ds = lightgbm_rs.Dataset(np.ascontiguousarray(X, dtype=np.float32), y)
    elif kind == "csr":
        rs_ds = lightgbm_rs.dataset_from_csr(sparse.csr_matrix(X), y)
    elif kind == "csc":
        rs_ds = lightgbm_rs.dataset_from_csc(sparse.csc_matrix(X), y)
    else:  # pragma: no cover - parametrize guards this
        raise AssertionError(kind)

    rs_pred = _rs_predict(rs_ds, Xt, num_round)
    ref_pred = _ref_predict(X, y, Xt, num_round)

    assert rs_pred.shape == ref_pred.shape
    np.testing.assert_allclose(
        rs_pred,
        ref_pred,
        atol=1e-6,
        err_msg=f"lightgbm_rs ({kind}) diverges from real lightgbm 4.6 beyond 1e-6",
    )


def test_f32_f64_self_consistency():
    """f32 and f64 of the SAME data train to identical lightgbm_rs predictions
    (the single f32->f64 widen site — T-08-03-03)."""
    X, y, Xt = _data(seed=7)
    num_round = 15

    ds64 = lightgbm_rs.Dataset(np.ascontiguousarray(X, dtype=np.float64), y)
    ds32 = lightgbm_rs.Dataset(np.ascontiguousarray(X, dtype=np.float32), y)
    pred64 = _rs_predict(ds64, Xt, num_round)
    pred32 = _rs_predict(ds32, Xt, num_round)

    # f32 input widened at one site to the SAME f64 rows => bit-identical bins =>
    # identical models. (X is float64 source, so the f64 path is exact and the f32
    # path is the f64 source rounded to f32 then widened — both deterministic.)
    np.testing.assert_array_equal(
        pred32,
        pred64,
        err_msg="f32 and f64 inputs of the same data must yield identical predictions",
    )


def test_csr_csc_self_consistency():
    """A CSR matrix and its CSC transpose of the SAME data produce identical
    lightgbm_rs predictions (both densify to the same f64 rows)."""
    X, y, Xt = _data(seed=11, sparsity=0.5)
    num_round = 15
    pred_csr = _rs_predict(lightgbm_rs.dataset_from_csr(sparse.csr_matrix(X), y), Xt, num_round)
    pred_csc = _rs_predict(lightgbm_rs.dataset_from_csc(sparse.csc_matrix(X), y), Xt, num_round)
    np.testing.assert_array_equal(
        pred_csr,
        pred_csc,
        err_msg="CSR and CSC of the same data must yield identical predictions",
    )


def test_malformed_csr_raises():
    """A non-monotone CSR ``indptr`` is rejected as ``ValueError`` BEFORE any
    indexing — never a panic/abort (Security V5, T-08-03-01)."""
    # indptr 0,2,1 is not monotone non-decreasing.
    bad_indptr = np.array([0, 2, 1], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    label = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError):
        lightgbm_rs.Dataset.from_csr(bad_indptr, indices, data, (2, 2), label)


def test_out_of_range_column_index_raises():
    """A CSR column index >= num_cols raises ``ValueError`` (bounds checked before
    any column write, Security V5)."""
    indptr = np.array([0, 1], dtype=np.int64)
    indices = np.array([5], dtype=np.int32)  # 5 >= num_cols 2
    data = np.array([1.0], dtype=np.float32)
    label = np.array([0.0], dtype=np.float64)
    with pytest.raises(ValueError):
        lightgbm_rs.Dataset.from_csr(indptr, indices, data, (1, 2), label)


def test_wrong_dtype_dense_raises():
    """A dense numpy array of an unsupported dtype (e.g. int64) raises
    ``ValueError`` rather than panicking (dtype dispatch, SC#2)."""
    X = np.arange(12, dtype=np.int64).reshape(3, 4)
    y = np.zeros(3, dtype=np.float64)
    with pytest.raises(ValueError):
        lightgbm_rs.Dataset(X, y)
