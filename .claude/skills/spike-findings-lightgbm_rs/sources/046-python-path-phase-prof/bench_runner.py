#!/usr/bin/env python3
"""Spike-046 inner benchmark runner — one backend per subprocess invocation.

Run as:  python3 bench_runner.py <which> <device_type>
  which        = "rs" (lightgbm_rs)  | "off" (official lightgbm)
  device_type  = "cpu" | "cuda"

Prints the wall-clock total to STDOUT and (for lightgbm_rs, when
LGBM_PHASE_PROF=1 is set in the environment) the per-phase BUDGET/LOOP/COUNTS
attribution to STDERR via the spike-046 dump("train") hook.

Each backend runs in its OWN process so the phase_prof process-global atomics
never bleed between runs and stderr is cleanly attributable.
"""
import sys
import time

import numpy as np
from sklearn.datasets import make_classification

which = sys.argv[1]
device_type = sys.argv[2]
# Optional argv[3]: metric_freq override (spike-048 A/B — isolates the per-iter
# training-metric eval cost. metric_freq=N_ESTIMATORS => eval only on the last
# iter, i.e. ~1 eval instead of 100. Default 1 = current behavior.)
metric_freq = int(sys.argv[3]) if len(sys.argv) > 3 else 1

# Match the reported repro EXACTLY: 500k samples, 50 features, 100 trees.
N_SAMPLES = 500_000
N_FEATURES = 50
SEED = 42

print(f"[{which}/{device_type}] generating {N_SAMPLES}x{N_FEATURES}...", flush=True)
X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=SEED)
X = np.ascontiguousarray(X, dtype=np.float64)

params = dict(
    objective="binary",
    metric="binary_logloss",
    device_type=device_type,
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=100,
    metric_freq=metric_freq,
    verbose=-1,
)

if which == "rs":
    import lightgbm_rs as lgb
else:
    import lightgbm as lgb

# Warm pass discarded (allocator/JIT amortization — the warm-vs-cold rule), then
# a timed pass. Keep it to ONE warm + ONE timed to bound Kaggle wall-clock; the
# phase_prof dump reflects the LAST (timed) train.
print(f"[{which}/{device_type}] warmup...", flush=True)
warm = lgb.LGBMClassifier(**{**params, "n_estimators": 5})
warm.fit(X, y)

print(f"[{which}/{device_type}] timed train...", flush=True)
start = time.time()
model = lgb.LGBMClassifier(**params)
model.fit(X, y)
elapsed = time.time() - start

print(f"RESULT {which} {device_type} mfreq={metric_freq} train_time_s={elapsed:.3f}", flush=True)
