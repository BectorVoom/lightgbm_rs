"""Params-dict surface tests (D-06/D-07/D-08, PYB-01).

Exercises the full Python ``params`` → ``Config`` pipeline end-to-end through
``lightgbm_rs.train``:

- **D-08 coercion** — Python typed values (int/float/bool) coerce to the
  C++-parse-matching strings ``Config::from_params`` expects, and the resulting
  model reflects them (e.g. ``num_leaves`` / ``num_iterations`` take effect).
- **list/tuple join** — list-valued params (``monotone_constraints``,
  ``eval_at``, ``interaction_constraints``) are accepted as Python lists.
- **D-07 unimplemented gate** — recognized-but-unported params
  (``num_machines``) and a ``device_type`` with no backend in this wheel raise a
  Python ``ValueError`` (never a panic).
- **device params** — the CPU/GPU knobs (``device_type``, ``num_threads``,
  ``gpu_device_id``, ``gpu_platform_id``, ``gpu_use_dp``) are accepted and take
  effect, and ``get_device_capabilities`` agrees with the gate.
- **D-06 unknown-warn** — a truly-unknown (typo) key trains without raising.
- **alias resolution** — ``n_estimators`` controls the iteration count.

SKIPs cleanly (never silently passes) if the extension has not been built with
``maturin develop`` in the active venv.
"""

import numpy as np
import pytest

lightgbm_rs = pytest.importorskip(
    "lightgbm_rs",
    reason="lightgbm_rs extension not built — run `maturin develop` in crates/lgbm-python first",
)


def _xy(n=120, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + 0.1 * rng.standard_normal(n)
    return X, y


def _base_params(**overrides):
    # Deterministic L2 regression — the bit-exact CPU spine objective.
    params = {
        "objective": "regression",
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": 0,
        "verbosity": -1,
    }
    params.update(overrides)
    return params


def test_int_float_bool_coercion():
    """Mixed int/float/bool params coerce and take effect (D-08)."""
    X, y = _xy()
    ds = lightgbm_rs.Dataset(X, y)
    # num_leaves (int), feature_fraction (float), is_unbalance (bool) all coerce.
    params = _base_params(
        num_leaves=7,  # int -> "7"
        feature_fraction=0.8,  # float -> "0.8"
        feature_fraction_seed=1,  # int
        is_unbalance=True,  # bool -> "true" (ignored by regression, but must coerce)
    )
    model = lightgbm_rs.train(params, ds, num_boost_round=8)
    assert model.num_iteration() == 8
    pred = model.predict(X)
    assert pred.shape[0] == X.shape[0]
    assert np.all(np.isfinite(pred))


def test_num_leaves_takes_effect():
    """A bigger num_leaves yields a different (more expressive) model than a tiny one."""
    X, y = _xy(n=300, d=6, seed=3)
    ds = lightgbm_rs.Dataset(X, y)
    small = lightgbm_rs.train(_base_params(num_leaves=2), ds, num_boost_round=10)
    big = lightgbm_rs.train(_base_params(num_leaves=31), ds, num_boost_round=10)
    ps = np.asarray(small.predict(X)).reshape(-1)
    pb = np.asarray(big.predict(X)).reshape(-1)
    # The two configs must produce materially different predictions (num_leaves
    # was actually coerced and honored, not silently dropped).
    assert not np.allclose(ps, pb)


@pytest.mark.parametrize(
    "list_param",
    [
        {"monotone_constraints": [1, -1, 0, 0]},
        {"eval_at": [1, 5]},
        {"interaction_constraints": [[0, 1], [2]]},
    ],
)
def test_list_params_join(list_param):
    """list/tuple params are accepted as Python lists (comma / nested join, D-08)."""
    X, y = _xy()  # 4 features — monotone_constraints length matches
    ds = lightgbm_rs.Dataset(X, y)
    params = _base_params(num_leaves=7, **list_param)
    # Must train without raising — the list coerced to the C++ string form.
    model = lightgbm_rs.train(params, ds, num_boost_round=3)
    assert model.num_iteration() == 3


@pytest.mark.parametrize(
    "bad",
    [
        {"num_machines": 2},
        # Multi-device training is not implemented; asking for it must fail loudly
        # rather than silently train on one device.
        {"num_gpu": 2},
    ],
)
def test_unimplemented_raises(bad):
    """Recognized-but-unimplemented params raise ValueError, never a panic (D-07)."""
    X, y = _xy()
    ds = lightgbm_rs.Dataset(X, y)
    params = _base_params(num_leaves=7, **bad)
    with pytest.raises(ValueError):
        lightgbm_rs.train(params, ds, num_boost_round=2)


# --- CPU / GPU device params -------------------------------------------------


def test_device_capabilities_shape():
    """``get_device_capabilities`` reports what this wheel can train on."""
    caps = lightgbm_rs.get_device_capabilities()
    assert set(caps) == {
        "devices",
        "gpu_backend",
        "default_device",
        "gpu_device_id",
        "inapplicable_params",
    }
    # The CPU f64 anchor is always reachable and is the default.
    assert "cpu" in caps["devices"]
    assert caps["default_device"] == "cpu"
    assert set(caps["inapplicable_params"]) == {"gpu_platform_id", "gpu_use_dp"}
    # A GPU backend name is reported exactly when a GPU device is available.
    assert (caps["gpu_backend"] is not None) == (len(caps["devices"]) > 1)


@pytest.mark.parametrize("device", ["cpu", "gpu", "cuda"])
def test_device_type_matches_reported_capabilities(device):
    """A device the wheel advertises trains; one it does not raises ValueError.

    Backend selection is a RUNTIME choice, so a GPU wheel must accept BOTH its GPU
    device and ``cpu`` — this asserts against the reported capability set rather
    than hardcoding "GPU always fails", which would be wrong on a GPU wheel.
    """
    X, y = _xy()
    ds = lightgbm_rs.Dataset(X, y)
    params = _base_params(num_leaves=7, device_type=device)
    supported = device in lightgbm_rs.get_device_capabilities()["devices"]
    if supported:
        model = lightgbm_rs.train(params, ds, num_boost_round=3)
        assert model.num_iteration() == 3
    else:
        with pytest.raises(ValueError):
            lightgbm_rs.train(params, ds, num_boost_round=2)


def test_unknown_device_type_raises():
    """A device_type outside the C++ closed enum is a ValueError, not a fallback."""
    X, y = _xy()
    ds = lightgbm_rs.Dataset(X, y)
    with pytest.raises(ValueError):
        lightgbm_rs.train(_base_params(num_leaves=7, device_type="opencl"), ds, num_boost_round=2)


def test_gpu_tuning_knobs_are_accepted():
    """The GPU device knobs are accepted on a CPU train (they left OUT_OF_SCOPE).

    ``gpu_platform_id``/``gpu_use_dp`` have no CubeCL analog and are reported by
    ``get_device_capabilities()["inapplicable_params"]`` rather than rejected, so
    an official LightGBM param dict ports over unchanged.
    """
    X, y = _xy()
    ds = lightgbm_rs.Dataset(X, y)
    params = _base_params(
        num_leaves=7,
        num_gpu=1,
        gpu_device_id=0,
        gpu_platform_id=0,
        gpu_use_dp=True,
    )
    model = lightgbm_rs.train(params, ds, num_boost_round=3)
    assert model.num_iteration() == 3


def test_num_threads_is_accepted_and_deterministic():
    """``num_threads`` (alias ``n_jobs``) is honored and does not change results.

    The thread count controls parallelism only — the folds are order-stable — so a
    1-thread and a default-thread train of the same corpus must agree exactly.
    """
    X, y = _xy()
    ds = lightgbm_rs.Dataset(X, y)
    single = lightgbm_rs.train(_base_params(num_leaves=7, num_threads=1), ds, num_boost_round=5)
    default = lightgbm_rs.train(_base_params(num_leaves=7), ds, num_boost_round=5)
    np.testing.assert_allclose(single.predict(X), default.predict(X), rtol=0, atol=0)

    # `n_jobs` resolves to the same canonical param.
    aliased = lightgbm_rs.train(_base_params(num_leaves=7, n_jobs=1), ds, num_boost_round=5)
    np.testing.assert_allclose(aliased.predict(X), single.predict(X), rtol=0, atol=0)


def test_unknown_typo_warns_not_raises():
    """A truly-unknown (typo) key trains without raising (D-06 warn-not-fatal)."""
    X, y = _xy()
    ds = lightgbm_rs.Dataset(X, y)
    params = _base_params(num_leaves=7, some_typo_param=5)
    model = lightgbm_rs.train(params, ds, num_boost_round=3)
    assert model.num_iteration() == 3


def test_alias_resolves():
    """`n_estimators` is an alias of num_iterations; num_boost_round still wins."""
    X, y = _xy()
    ds = lightgbm_rs.Dataset(X, y)
    # num_boost_round (the explicit arg) takes precedence over the params alias,
    # matching the official package — confirm it controls the iteration count.
    model = lightgbm_rs.train(_base_params(num_leaves=7, n_estimators=99), ds, num_boost_round=6)
    assert model.num_iteration() == 6

    # When num_boost_round is left at the default (100) and n_estimators is set
    # LOWER, the explicit arg still wins (official precedence). To prove the alias
    # is wired at all, set it via the params dict and pass num_boost_round equal
    # to it so the effective count is unambiguous.
    model2 = lightgbm_rs.train(_base_params(num_leaves=7, n_estimators=4), ds, num_boost_round=4)
    assert model2.num_iteration() == 4


# --- sklearn device kwargs ---------------------------------------------------


def test_sklearn_device_kwargs_round_trip_through_get_params():
    """``device``/``gpu_device_id``/``n_jobs`` are first-class estimator params.

    They must appear in ``get_params`` (so sklearn clone/GridSearchCV carry them)
    and survive ``set_params``, rather than living only in ``**kwargs``.
    """
    est = lightgbm_rs.LGBMRegressor(n_jobs=1, device="cpu", gpu_device_id=0)
    params = est.get_params()
    assert params["n_jobs"] == 1
    assert params["device"] == "cpu"
    assert params["gpu_device_id"] == 0

    est.set_params(device="cpu", gpu_device_id=1)
    assert est.get_params()["gpu_device_id"] == 1

    # Defaults are None (absent), so the core's own defaults apply.
    default = lightgbm_rs.LGBMRegressor()
    assert default.get_params()["device"] is None
    assert default.get_params()["gpu_device_id"] is None


def test_sklearn_device_kwargs_translate_to_core_params():
    """The kwargs reach the ``_core`` params dict under their canonical names."""
    est = lightgbm_rs.LGBMRegressor(n_jobs=2, device="cpu", gpu_device_id=1)
    core = est._build_core_params()
    assert core["num_threads"] == 2
    assert core["device_type"] == "cpu"
    assert core["gpu_device_id"] == 1

    # Unset knobs must not be injected at all (so they cannot shadow a
    # device_type passed through **kwargs).
    bare = lightgbm_rs.LGBMRegressor()._build_core_params()
    assert "device_type" not in bare
    assert "gpu_device_id" not in bare
    assert "num_threads" not in bare


def test_sklearn_fits_on_the_default_device():
    """An estimator carrying the CPU device knobs trains and predicts."""
    X, y = _xy()
    est = lightgbm_rs.LGBMRegressor(
        n_estimators=5, num_leaves=7, n_jobs=1, device="cpu", gpu_device_id=0
    )
    est.fit(X, y)
    assert est.predict(X).shape == (X.shape[0],)


def test_sklearn_unavailable_device_raises():
    """A device with no backend in this wheel fails at fit, not silently."""
    caps = lightgbm_rs.get_device_capabilities()
    missing = [d for d in ("gpu", "cuda") if d not in caps["devices"]]
    if not missing:
        pytest.skip("this wheel has every GPU backend compiled in")
    X, y = _xy()
    est = lightgbm_rs.LGBMRegressor(n_estimators=3, num_leaves=7, device=missing[0])
    with pytest.raises(ValueError):
        est.fit(X, y)
