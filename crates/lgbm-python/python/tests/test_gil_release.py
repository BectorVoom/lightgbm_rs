"""GIL-release proof (D-13 / SC#1): ``train`` runs inside ``Python::detach``, so
a background Python thread must make progress *during* training. If the GIL were
held across the CPU-bound train, the background thread could not advance.

SKIPs cleanly if the extension is not built.
"""

import threading
import time

import numpy as np
import pytest

lightgbm_rs = pytest.importorskip(
    "lightgbm_rs",
    reason="lightgbm_rs extension not built — run `maturin develop` in crates/lgbm-python first",
)


def _heavy_params():
    return {
        "objective": "regression",
        "num_leaves": 63,
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": 0,
        "verbosity": -1,
    }


def test_background_thread_advances_during_train():
    rng = np.random.default_rng(1)
    n, d = 6000, 12
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + 0.1 * rng.standard_normal(n)
    ds = lightgbm_rs.Dataset(X, y)

    counter = {"n": 0}
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter["n"] += 1
            time.sleep(0.0005)

    worker = threading.Thread(target=spin, daemon=True)
    worker.start()
    # Let the spinner reach a steady state, then snapshot.
    time.sleep(0.02)
    try:
        before = counter["n"]
        # Enough work that training takes clearly longer than the spinner tick.
        lightgbm_rs.train(_heavy_params(), ds, num_boost_round=80)
        after = counter["n"]
    finally:
        stop.set()
        worker.join(timeout=2.0)

    advanced = after - before
    assert advanced > 0, (
        "background Python thread made no progress during train — "
        "the GIL was NOT released (Python::detach missing/ineffective)"
    )
