# T0 — BEFORE baseline (260608-s2b)

Captured at HEAD `b8a7080` (no tracked source modified; only untracked planning/serena/LightGBM dirs).
All numbers are REAL measured output on the local gfx1100 ROCm GPU.

## Builds

- `cargo build --workspace` (cpu) — exit 0.
- `cargo build --workspace --features rocm` — exit 0.

## CPU bit-exact gates (default `cargo test`)

- `kernel_parity` (cpu): 6 passed / 0 failed / 0 ignored.
- `learner_parity` (cpu): 29 passed / 0 failed / 0 ignored (incl. spine_real, routing_self_consistency).
- `boosting_parity`: 75 passed / 0 failed / 0 ignored (incl. `mfb_zero_offset_histogram_contract`, `goss_parity_matrix`).
- Full `cargo test --workspace`: all suites GREEN, 0 failed, 0 ignored.

NOTE on pre-existing labels:
- DEF-07-02 ignored cells: NOT present in this default run — the out-of-scope fixtures
  are parked under `.out-of-scope-fixtures-holding/`, so boosting_parity shows 0 ignored.
- DEF-08-OOS-01 `goss_parity_matrix`: currently PASSES at HEAD (the parked OOS fixtures
  are what tripped it previously). Recorded as PASS here.

## hip oracles (`--features rocm`)

The on-hip oracles live in `mod hip` gated by `#[cfg(feature="rocm")]` (they are NOT
`#[ignore]`-marked, so they run as normal tests under `--features rocm`, not via `--ignored`).

kernel_parity (`--features rocm`): 13 passed / 1 failed.
- GREEN (relevant to this change):
  - `hip::kernel_parity_fix_compact_equals_host_on_hip` — PASS (bit-exact, compare_exact_f64_bits)
  - `hip::kernel_parity_resident_build_fix_compact_equals_host_on_hip` — PASS (within ~1e-6)
  - `hip::find_best_splits_batched_from_handle_equals_host_buf_on_hip` — PASS
  - `hip::kernel_parity_resident_gather_equals_host_gather_on_hip` — PASS
  - others (subtract/partition/histogram within-tol) — PASS
- **PRE-EXISTING FAILURE (OUT OF SCOPE, D-03a):**
  - `hip::kernel_parity_split_within_tol_on_hip` — FAIL. Documented f32-vs-f64 accumulation
    gap (abs_diff ~3.8e-6 / 7.6e-6 > ORACLE_TOL 1e-6, plus a `default_left` flip on a
    knife-edge). Labeled in-test as "documented f32-vs-f64 accumulation gap,
    04-ROCM-GAPS.md / D-03a". This is the split-scan f32 gap, unrelated to the
    widen-fold / size-gate change. NOT introduced here.

learner_parity (`--features rocm`): 29 passed / 1 failed.
- **PRE-EXISTING FAILURE (borderline f32-atomic ROCm gap):**
  - `hip::learner_parity_resident_equals_host_tree_on_hip` — FAIL at baseline.
    Structural fields (topology/split_feature/decision_type/threshold/counts) are BIT-EXACT;
    the ONLY divergence is a leaf-VALUE: leaf 11 resident=0.7184174571718487
    host=0.7184157371520994, abs_diff=1.72e-6 > the test's 1e-6 tol.
    Reproducible/deterministic across 3 consecutive runs (NOT GPU noise).
    This is the same f32-vs-f64 ROCm-gap family as D-03a (the resident path's f32-atomic
    RAW build accumulates in a different order than the host f32-atomic build, and on this
    one leaf the gap lands at 1.72e-6). The p90 SUMMARY's "resident==host trees GREEN"
    claim does NOT hold at this exact tolerance on the committed corpus at HEAD.
    **Pre-existing at HEAD b8a7080 — not introduced by 260608-s2b.**

    Bearing on this task: Lever A is bit-identical-by-construction (same f32→f64 cast,
    same fold order ⇒ same resident f64 buffer), so it CANNOT change this leaf value;
    Lever B routes between resident and host but does not alter the resident numerics.
    The acceptance bar for s2b is therefore "does not WORSEN this baseline failure",
    verified after T1/T2 by re-running and confirming the identical leaf-11 value.

    **CORRECTION (discovered during T1):** this test is actually FLAKY / non-deterministic,
    NOT deterministically failing. Re-running it 4× post-Lever-A gave PASS/PASS/PASS/FAIL
    with the leaf-11 value VARYING run-to-run (0.71841746, 0.71841608, 0.71841437, ...).
    Root cause: the f32-ATOMIC histogram build (the construct kernel + the host f32-atomic
    build, BOTH unchanged by this task) accumulates atomic-adds in a GPU-scheduler-dependent
    order, so each launch yields a slightly different f32 sum and the abs_diff hovers right
    at the 1e-6 threshold. This is GPU atomic non-determinism in a layer s2b does NOT touch
    (the documented ~1e-6 f32 ROCm gap, CLAUDE.md). The T0 "deterministic 3×" reading was a
    coincidental 3-fail streak. Lever A's folded kernel is bit-identical FOR A GIVEN f32 RAW
    (proven by `kernel_parity_fix_compact_equals_host_on_hip`, compare_exact_f64_bits, GREEN);
    it neither causes nor cures this pre-existing flakiness.

## AFTER Lever A (T1) — bench + oracles

hip oracles (`--features rocm`):
- `kernel_parity_fix_compact_equals_host_on_hip` — PASS (BIT-EXACT, folded kernel).
- `kernel_parity_resident_build_fix_compact_equals_host_on_hip` — PASS (within ~1e-6).
- `kernel_parity_split_within_tol_on_hip` — FAIL (UNCHANGED pre-existing D-03a gap).
- `learner_parity_resident_equals_host_tree_on_hip` — FLAKY around 1e-6 (pre-existing GPU
  atomic non-determinism, see correction above; NOT worsened — same value distribution).

CPU bit-exact gates: kernel_parity 6/6, learner_parity 29/29, boosting_parity 75/75 — all GREEN.

GPU bench AFTER Lever A (2 runs, train_median):
| size   | T0 (before) | Lever A run1 | Lever A run2 |
|--------|-------------|--------------|--------------|
| small  | 1.60s       | 1.46s        | 1.42s        |
| medium | 4.98s       | 4.89s        | 4.67s        |
| large  | 11.14s      | 11.54s       | 11.55s       |

small: consistently faster (−9 to −11%) — the launch-bound case benefits most from 3→2 launches.
medium: flat-to-faster. large: +3-4% (compute-bound; within the ±0.4s run-to-run GPU noise).

## GPU bench BEFORE (resident path, `RocmBackend::default()` resident_enabled=true)

`cargo run --release --features rocm --example bench_train` (system allocator, iters 100, leaves 31, TRAIN_REPS=5 median):

| size   | rows  | feat | bins | train_median | predict_med |
|--------|-------|------|------|--------------|-------------|
| small  | 2000  | 12   | 32   | 1.60s        | 4.26ms      |
| medium | 8000  | 30   | 64   | 4.98s        | 26.62ms     |
| large  | 20000 | 50   | 128  | 11.14s       | 69.23ms     |

These match the STATE p90 note (small ~1.6s regressed, medium ~4.8s, large ~11.6s).
This is the p90 resident baseline that levers A (3→2 launches) and B (size-gate) target.
