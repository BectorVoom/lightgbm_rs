# 260620-8v4 — FINDINGS: 2-lane split-scan lever

**Verdict: NULL (regression).** The candidate optimisation — running the
split-finding kernel's two independent passes (REVERSE ∥ FORWARD) on two lanes
concurrently — is **bit-exact-safe but a severe wall-clock regression** on the
production CPU anchor path. The serial `find_best_split_cpu_native` stays the
default and the single source of truth (the 2-lane variant is opt-in, OFF by
default, behind `LGBM_SPLIT_2LANE`).

This is consistent with the prior overhead campaign (q2z / sgu / tlk / mwr / ol8),
which produced repeated hardware-confirmed NULLs because the production hot path
was already idiomatic, and with project memory `l3-on-gpu-fixhistogram-deferred`
("profile before the next perf assumption").

---

## Task 1 — go/no-go measurement (SCAN fraction)

Harness: `crates/lgbm-treelearner/examples/bench_split_scan.rs`, driving the full
per-leaf BUILD/SCAN hot path through the facade `lgbm::train` and reading the
existing `lgbm_treelearner::phase_prof` `BUILD_NS` / `SCAN_NS` counters
(`LGBM_PHASE_PROF=1`). Warm window (cold iteration discarded), `cubecl-cpu`
f64 anchor backend, release build, identity-binned synthetic corpora
(20 000 rows, 255 bins, regression GBDT, 30 iters, 31 leaves).

Decision rule: **GO if wide `SCAN ≥ 20%` of `(BUILD+SCAN)`** (the 2-lane lever can
at best halve the scan, so a <20% scan caps the per-leaf win at <10%, below the
project bar for a parity-risky additive kernel variant).

Verbatim (two runs, stable):

```
SCAN_NS=33696508 BUILD_NS=64588803 SCAN%=34.28 config=narrow feat=10  backend=cubecl-cpu(f64-anchor)
SCAN_NS=358026556 BUILD_NS=450121804 SCAN%=44.30 config=wide  feat=120 backend=cubecl-cpu(f64-anchor)
SUMMARY: wide SCAN%=44.30  threshold=20.00  verdict=GO  (2-lane lever ceiling on per-leaf win ≈ 22.15%)

SCAN_NS=33928205 BUILD_NS=62751439 SCAN%=35.09 config=narrow feat=10  backend=cubecl-cpu(f64-anchor)
SCAN_NS=349071373 BUILD_NS=437980775 SCAN%=44.35 config=wide  feat=120 backend=cubecl-cpu(f64-anchor)
SUMMARY: wide SCAN%=44.35  threshold=20.00  verdict=GO  (2-lane lever ceiling on per-leaf win ≈ 22.18%)
```

**Gate cleared: wide SCAN% ≈ 44.3% ≥ 20% → GO.** The scan IS a material fraction
of per-leaf compute, so the lever was pursued (Task 2) rather than NULLed on
fraction alone.

### Note on the measured path (deviation from the plan's stated context)

The plan's `<context>` named `build_fix_scan_resident_f64_on` (histogram.rs:2444)
as the production per-leaf hot path. That function — and the fused
`find_best_splits_fused_kernel` — are **`#[cfg(feature = "rocm")]`-gated**. On the
authoritative **CPU f64 anchor** (the merge gate), `CpuBackend` does NOT override
`find_best_splits_batched`; it uses the trait default, which calls
`CpuBackend::find_best_split` → **`find_best_split_cpu_native`** (split.rs:1355) —
a plain-Rust serial REVERSE-then-FORWARD scan, ONE feature at a time, single
thread (260608-mc5 Task-3 kept the native path because the cubecl-cpu per-leaf
launch dispatch regressed CPU train time). So the 2-lane lever was applied to that
native function, the actual CPU SCAN hot path the `SCAN_NS` counter measures.

---

## Task 2 — additive 2-lane variant + A/B measurement

Implemented `find_best_split_cpu_native_2lane` (split.rs) as an **additive**
alternative to the serial `find_best_split_cpu_native`:

- Each pass (REVERSE / FORWARD) is byte-for-byte its serial counterpart — no
  intra-pass reordering of the f64 loop-carried accumulation. The only structural
  change is that the two passes no longer share a running `best_gain`; they run on
  two `rayon::join` lanes and are combined by a final argmax.
- Cross-lane combine `(fwd.gain > rev.gain) ? fwd : rev` (strict `>`) reproduces
  the serial cross-pass semantics exactly: **REVERSE wins ties** (the serial
  FORWARD uses strict `>` against the carried reverse gain), and the winner
  identity is preserved (the first forward candidate attaining the forward max is
  the same whether the running best was seeded at `0.0` or at `rev_gain < fwd_max`).
- `is_splittable` is the OR of the two passes. All other host steps (V5,
  `2*kEpsilon` bump, `min_gain_shift`, `cnt_factor`, finalization eps subtraction,
  accept gate) are identical to the serial function.

### Bit-exactness: PASS

`cargo test -p lgbm-compute --lib split_2lane` (2 tests, both green):

- `split_2lane_equals_serial_matrix` — sweeps 3 histograms (separable / flat /
  noisy) × offset∈{0,1} × skip_default_bin × run_forward × L1 on/off × 4
  min_data_in_leaf gates; every cell is **byte-identical** (f64 fields compared by
  `to_bits()`).
- `split_2lane_reverse_wins_exact_tie` — engineered cross-pass gain tie; the
  2-lane combine keeps the REVERSE winner (`default_left == true`), bit-identical
  to serial.

The argmax IS bit-identical to the serial body's tie-break. Parity is not the
blocker.

### Wall-clock A/B: REGRESSION (the lever is NULL)

`bench_split_scan` run 3× each, serial (`LGBM_SPLIT_2LANE` unset) vs 2-lane
(`LGBM_SPLIT_2LANE=1`), warm window, `cubecl-cpu` f64 anchor, release. Verbatim
`SCAN_NS` (the scan-only phase) and `warm_wall` (full 3-rep train):

```
###### SERIAL ######
SCAN_NS=34647926  ... warm_wall=173.306ms  config=narrow
SCAN_NS=376742463 ... warm_wall=1116.027ms config=wide
SCAN_NS=33952823  ... warm_wall=170.691ms  config=narrow
SCAN_NS=377724209 ... warm_wall=1137.498ms config=wide
SCAN_NS=34106096  ... warm_wall=168.400ms  config=narrow
SCAN_NS=379403047 ... warm_wall=1126.784ms config=wide

###### 2-LANE (LGBM_SPLIT_2LANE=1) ######
SCAN_NS=463733709  ... warm_wall=672.972ms  config=narrow
SCAN_NS=5010723570 ... warm_wall=6475.390ms config=wide
SCAN_NS=484368898  ... warm_wall=701.550ms  config=narrow
SCAN_NS=4907513778 ... warm_wall=6287.127ms config=wide
SCAN_NS=451903801  ... warm_wall=655.251ms  config=narrow
SCAN_NS=5086121844 ... warm_wall=6455.132ms config=wide
```

| Config | Serial SCAN (median) | 2-lane SCAN (median) | SCAN ratio | Serial warm_wall | 2-lane warm_wall | wall ratio |
|--------|---------------------:|---------------------:|-----------:|-----------------:|-----------------:|-----------:|
| narrow (10 feat)  | ~34.1 ms  | ~464 ms   | **~13.6× slower** | ~170.7 ms  | ~673 ms   | **~3.9× slower** |
| wide (120 feat)   | ~377.7 ms | ~5010 ms  | **~13.3× slower** | ~1126.8 ms | ~6455 ms  | **~5.7× slower** |

**The 2-lane SCAN is ~13× SLOWER, not faster. There is no win.**

### Root cause

`find_best_split` is called **once per (feature, leaf)** — millions of times over a
30-iter × 31-leaf × 120-feature train. Each call's REVERSE/FORWARD passes are only
~microseconds. `rayon::join` imposes a fork/join + work-stealing cost per call that
**dwarfs** the µs-scale passes; multiplied over millions of calls it produces the
~13× scan blow-up. The 22% per-leaf ceiling the SCAN fraction promised is
unreachable because the parallelisation granularity (one tiny scan) is far below
rayon's break-even point — the overhead is orders of magnitude larger than the work
being parallelised. This is the same class of finding as the prior NULL campaign:
the serial path is already at the right granularity; adding concurrency at this
level only adds dispatch cost.

(A coarser-grained alternative — parallelising the per-leaf FEATURE loop across
rayon instead of the two passes within one feature — is a *different* lever, not the
one this task scoped, and the learner's `find_best_splits_batched` trait-default
feature loop already runs serially by deliberate 260608-mc5 design; exploring
inter-feature rayon is out of scope here and would need its own audit.)

---

## Outcome

- **Default path unchanged**: `find_best_split_cpu_native` (serial) remains the
  production CPU SCAN path and the bit-exact source of truth. `split.rs`'s serial
  `split_scan_body` and `find_best_split_cpu_native` are untouched in their
  numerics.
- **Additive variant retained, OFF by default**: `find_best_split_cpu_native_2lane`
  is gated behind `LGBM_SPLIT_2LANE=1`, proven bit-identical, and kept as evidence
  + a re-runnable A/B. It is NOT wired into any default code path.
- **Reusable deliverable**: `bench_split_scan.rs` is a deterministic, re-runnable
  SCAN-fraction + serial-vs-2-lane A/B harness for future split-path perf audits.
- **Merge gate**: CPU f64 bit-exact parity preserved (see SUMMARY for verbatim
  kernel_parity / learner_parity / unit-suite results).

The honest verdict is **NULL** — the lever is parity-safe but a wall-clock
regression at this call granularity. No win was manufactured.
