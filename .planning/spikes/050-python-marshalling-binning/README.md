---
spike: 050
name: python-marshalling-binning
type: standard
validates: "Given the spike-049 'Python marshalling ~25%' chunk, when attributed, then the real cost is identified and the win prototyped + proven bit-exact"
verdict: VALIDATED
related: [048, 049]
tags: [python, pyo3, binning, marshalling, cpu, parallel, rayon, bit-exact, shipped]
---

# Spike 050: The "Python marshalling ~25%" is actually single-threaded BINNING

## Why
Spike-049 mapped the post-metric-fix CUDA wall and left one chunk un-attributed:
"Python marshalling ~25%" (~2.78s). It's all host/CPU work (backend-independent), so
this attributes it LOCALLY at 500k×50 — no Kaggle needed.

## Method
Traced the pyo3 `fit` path: `dense_any_to_rows` (numpy→`Vec<Vec<f64>>`) → `RawCorpus` →
`lgbm::train_raw` → `build_feature_columns_from_raw_with_config` (raw→bin) →
`train_inner_columns` (the boosting loop, already attributed). A pure-Rust proxy
(`spike050_marshalling.rs`) times the marshalling conversion and the binning (now
wrapped in `BINNING_NS` — it was uninstrumented, which is why the Kaggle BUDGET showed
`binning=0`).

## Results (500k×50, local, 16 cores)

| Component | Time | Verdict |
|---|---|---|
| numpy→`Vec<Vec<f64>>` marshalling proxy | **43ms** | NOT the bottleneck — refutes the "marshalling" framing |
| raw→bin (`build_feature_columns_from_raw_with_config`), serial | **624ms** | the real cost; was hidden as `binning=0` |
| raw→bin, **feature-parallel (rayon)** | **96ms** | **6.5× — the win** |

`train_raw` total (bin + 10-iter loop) 1029ms → 520ms with parallel binning.

### Root cause
The per-feature binning loop (`for j in 0..num_features`) was **single-threaded**, yet
each feature's BinMapper construction (sample + sort + find split points) + bin
assignment is fully INDEPENDENT. C++ LightGBM bins OpenMP-parallel over features; we
didn't. (The numpy→Vec-of-Vecs conversion I initially suspected is only ~43ms.)

### The fix (prototyped + SHIPPED in this spike — bit-exact)
`build_feature_columns_from_raw_with_config`: extract the per-feature body into a
closure and run `(0..num_features).into_par_iter().map(bin_feature).collect()`
(order-preserving ⇒ `columns[j]` == feature j; per-feature BinMapper is deterministic
with a fixed `data_random_seed` ⇒ bit-exact). Env A/B gate `LGBM_PAR_BIN=0` forces serial.
Also wrapped the step in `BINNING_NS` so it's attributed (parity-neutral).

## Verification — BIT-EXACT + gate-green (parallel default-on)
- `raw_bin_train_matches_cpp_golden` ✓ (the C++ golden for the raw→bin→train path)
- `boosting_parity` 75/0, `lgbm` 41/0, `lgbm-treelearner` 77/0
- `cargo test --workspace` → 0 failures

## Impact / signal
- The "Python marshalling 25%" is **binning**, now ~6.5× faster (matches C++). On the
  real CUDA wall this directly trims the chunk (less on Kaggle's few vCPUs, but binning
  was a bigger fraction there). Confirm e2e on Kaggle when convenient.
- Marshalling (numpy→Vec<Vec>) is a non-issue (~43ms) — do NOT chase it.
- Remaining big fish (spike-049): the GPU histogram phases (53%, architectural on-device
  learner). That's the only major lever left after this.

## Verdict
**VALIDATED + SHIPPED** (bit-exact, gate-green). Files: `spike050_marshalling.rs`,
`booster.rs` (parallel binning + BINNING_NS wrap), `Cargo.toml` (rayon dep).
