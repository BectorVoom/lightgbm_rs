#!/usr/bin/env python
"""Phase-10 Wave 4 — C++ `use_quantized_grad` parity oracle generator.

Trains LightGBM 4.6 in DETERMINISTIC quantized mode (the parity-tractable path:
`stochastic_rounding=False`) on a fixed seeded dataset and dumps a golden JSON: the
inputs, the exact params, the model text, and per-row predictions + raw scores.

Wave 3b (Rust production quantized training) gates against this: the Rust quantized model's
predictions must match `pred` within the quantized contract (target: bit-exact-tractable for
deterministic rounding; any residual documented — NOT the exact ~1e-6 anchor, see spike-008).

Run (from repo root): .venv/bin/python crates/oracle-harness/tests/fixtures/quantized/gen_golden.py
"""
import json
import pathlib

import numpy as np
import lightgbm as lgb

OUT = pathlib.Path(__file__).with_name("quant_binary.json")

# Fixed, small, fully reproducible corpus.
rng = np.random.default_rng(20260615)
N, F = 512, 6
X = rng.random((N, F)).astype(np.float64)
# A learnable signal + mild noise so trees actually split.
logit = 2.5 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] - 0.9 + 0.3 * rng.standard_normal(N)
y = (logit > 0).astype(np.float64)

params = dict(
    objective="binary",
    num_leaves=7,
    min_data_in_leaf=5,
    max_bin=63,
    learning_rate=0.1,
    # The quantized mode under test — DETERMINISTIC rounding (parity-tractable).
    use_quantized_grad=True,
    # MUST be <= 254: the discretized gradient is int8, and the value maps to ±bins/2, so
    # bins/2 must fit int8 (127). 256 overflows → corrupted grads → the model learns nothing.
    num_grad_quant_bins=128,
    stochastic_rounding=False,
    quant_train_renew_leaf=False,
    # Pin everything for reproducibility.
    deterministic=True,
    force_row_wise=True,
    num_threads=1,
    seed=1,
    feature_pre_filter=False,
    verbose=-1,
)
NUM_ROUND = 10

ds = lgb.Dataset(X, label=y)
model = lgb.train(params, ds, num_boost_round=NUM_ROUND)
pred = model.predict(X)                       # probability
raw = model.predict(X, raw_score=True)        # margin (pre-sigmoid)

golden = {
    "_about": "LightGBM 4.6 use_quantized_grad=True, stochastic_rounding=False golden (phase-10 W4)",
    "lightgbm_version": lgb.__version__,
    "params": params,
    "num_round": NUM_ROUND,
    "n": N,
    "num_features": F,
    # X/y/pred live in the plain-text companions (xy.csv, .pred); keep the JSON to the
    # params + model text (the structural reference for Wave 3b) to avoid duplicating data.
    "model_text": model.model_to_string(),
}
OUT.write_text(json.dumps(golden))

# Plain-text companions the Rust harness reads WITHOUT a JSON dependency (the workspace
# deliberately carries none). Wave 3b trains from `*.xy.csv` + the params here and compares
# its predictions to `*.pred`.
xy = pathlib.Path(__file__).with_name("quant_binary.xy.csv")
with xy.open("w") as f:
    f.write("# " + ",".join([f"x{j}" for j in range(F)] + ["y"]) + "\n")
    for i in range(N):
        f.write(",".join(f"{v:.17g}" for v in X[i]) + f",{y[i]:.0f}\n")
predf = pathlib.Path(__file__).with_name("quant_binary.pred")
predf.write_text("".join(f"{p:.17g}\n" for p in pred))

print(f"wrote {OUT.name}, {xy.name}, {predf.name}  (lightgbm {lgb.__version__}, n={N}, rounds={NUM_ROUND})")
print("pred[:5] =", np.round(pred[:5], 8))
