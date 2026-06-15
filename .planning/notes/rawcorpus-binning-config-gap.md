---
title: RawCorpus continuous-data gap vs C++ — root cause + fix (config-propagation footgun)
date: 2026-06-15
context: phase-10 follow-up investigation
type: note
---

# RawCorpus continuous-data gap vs C++ (RESOLVED)

## Symptom

Rust `train_raw` on fresh CONTINUOUS data diverged from C++ by ~0.28 (predictions), for BOTH
exact and quantized training (surfaced during phase-10 W3b as the "pre-existing exact-path gap").
The committed corpora are bit-exact — but they are IDENTITY-binned (values already = bin indices),
so the continuous→bin path was untested.

## Investigation (1-iteration A/B on a tiny continuous corpus)

1. Rust tree #1 split at feat0 ≤ **0.45326**; C++ at feat0 ≤ **0.414929** → structural divergence
   from tree 1 (not accumulation).
2. `BinMapper::find_bin_numeric` on the column gave bounds **bit-identical to C++**
   (`0.033015, …, 0.956048`) — so the binning ALGORITHM is correct.
3. But the TRAINED model split at thresholds (`0.45326`, …) that are NOT in those bounds — they are
   midpoints of *adjacent data values* (≈255-bin binning). So training binned with `max_bin≈255`,
   ignoring the `max_bin=15` passed to `train_raw`.

## Root cause

`build_feature_columns_from_raw(corpus)` binned with `&corpus.config`, NOT the `config` passed to
`train_raw(config, corpus)`. `RawCorpus::new(rows, labels)` defaults `config` to `Config::default()`
(`max_bin=255`), and its doc says "set config on the returned value." So a caller who sets `max_bin`
on the TRAINING config (the natural thing) has it **silently ignored for binning** → wrong bins →
different trees → ~0.28 divergence. A pure config-propagation footgun; the numerics were always faithful.

## Fix

`build_feature_columns_from_raw_with_config(corpus, &Config)` bins with an EXPLICIT config; the
`train_raw` / `train_custom_raw_with_metric` paths pass their TRAINING config (categorical indices
still from the corpus). `build_feature_columns_from_raw(corpus)` kept as a back-compat shim
(`= _with_config(corpus, &corpus.config)`).

## Result (validated)

- **Rust-exact vs C++-exact on continuous data: max 0.285 → 2.9e-8** (f32-bit-faithful — the exact
  RawCorpus path now matches C++).
- **Rust-quant vs C++-quant (absolute): 0.291 → 3.5e-3** (pure quantization residual).
- W4 gate strengthened to ABSOLUTE bounds (`xe_max < 1e-5`, `qe_max < 1e-2`); new self-contained
  regression `rawcorpus_binning_config.rs` (max_bin=2 ⇒ ≤1 split threshold). raw_bin_train_parity,
  lgbm (41), python (5), quantized (3) all green.

**Impact:** this fixed ALL continuous-data RawCorpus training vs C++, not just quantized. Anyone
who passed `max_bin`/`min_data_in_bin` to `train_raw` without also setting `corpus.config` was
training with default binning.
