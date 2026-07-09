#!/usr/bin/env python
"""Export a HIGH-DIMENSIONAL (>=100 feature) real dataset to a label-first TSV for the
`bench_real` harness (quick task 260620-d3v).

Run with the project's uv venv (sklearn 1.9.0; memory: phase8-python-venv):

    .venv/bin/python scripts/export_highdim_dataset.py

Output (gitignored — NOT committed):

    target/bench_data/highdim.tsv

Layout: each line is `label feat0 feat1 ... featN` (whitespace/TSV separated, label in
column 0) so the harness "tsv-label0" mode parses it directly. This is a SPEED benchmark,
not an accuracy benchmark, so the digit/identity label is fit with the regression objective
(documented in FINDINGS); the only requirement is real, non-uniform, BinMapper-binned
features with feature count >=100 so the split-fusion gate FIRES.

Fallback chain (documented in stderr provenance + FINDINGS):

  1. PRIMARY — sklearn.datasets.fetch_openml('mnist_784', version=1, as_frame=False):
     784 real pixel features. On success (cached or network reachable) we SUBSAMPLE rows
     via a fixed-seed permutation (default 15000) so training is fast but representative.
  2. FALLBACK (no network / fetch raises) — sklearn.datasets.fetch_olivetti_faces():
     4096 features x 400 rows, BUNDLED with sklearn (no download).

The actual dataset used + exact rows x features + label semantics are printed to stderr.
"""

import os
import sys

import numpy as np

OUT_DIR = os.path.join("target", "bench_data")
OUT_PATH = os.path.join(OUT_DIR, "highdim.tsv")
MNIST_SUBSAMPLE = 15000
SEED = 1


def _log(msg: str) -> None:
    print(f"[export_highdim] {msg}", file=sys.stderr)


def _load_mnist():
    """Return (X, y, name) for MNIST-784, subsampled. Raises on any fetch failure."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("mnist_784", version=1, as_frame=False)
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.float64)  # digit 0..9, used as regression target
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    take = min(MNIST_SUBSAMPLE, n)
    idx = rng.permutation(n)[:take]
    return X[idx], y[idx], "mnist_784 (openml v1)"


def _load_olivetti():
    """Return (X, y, name) for the bundled olivetti-faces (no network)."""
    from sklearn.datasets import fetch_olivetti_faces

    bunch = fetch_olivetti_faces(shuffle=False)
    X = np.asarray(bunch.data, dtype=np.float64)  # 400 x 4096
    y = np.asarray(bunch.target, dtype=np.float64)  # face id 0..39, regression target
    return X, y, "olivetti_faces (bundled)"


def main() -> int:
    name = None
    X = y = None
    try:
        _log("attempting PRIMARY dataset: mnist_784 via fetch_openml ...")
        X, y, name = _load_mnist()
    except Exception as exc:  # noqa: BLE001 - any fetch/network failure → fallback
        _log(f"mnist_784 unavailable ({type(exc).__name__}: {exc}); falling back to olivetti-faces")
        X, y, name = _load_olivetti()

    n_rows, n_feat = X.shape
    if n_feat < 100:
        _log(f"ERROR: dataset {name} has only {n_feat} features (<100); cannot exercise the fusion gate")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    # Write label-first, full f64 repr, whitespace-separated. Use np.savetxt for speed.
    out = np.concatenate([y.reshape(-1, 1), X], axis=1)
    np.savetxt(OUT_PATH, out, fmt="%.9g", delimiter="\t")

    _log(f"dataset = {name}")
    _log(f"rows x features = {n_rows} x {n_feat}  (label = column 0, regression objective)")
    _log(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)} bytes) — gitignored, NOT committed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
