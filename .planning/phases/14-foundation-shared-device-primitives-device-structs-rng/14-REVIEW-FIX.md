---
phase: 14-foundation-shared-device-primitives-device-structs-rng
fixed_at: 2026-06-29T12:03:14Z
review_path: .planning/phases/14-foundation-shared-device-primitives-device-structs-rng/14-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 14: Code Review Fix Report

**Fixed at:** 2026-06-29
**Source review:** `.planning/phases/14-foundation-shared-device-primitives-device-structs-rng/14-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (3 warning, 3 info — `fix_scope=all`)
- Fixed: 6
- Skipped: 0

**Important context on the three WARNINGs.** WR-01/02/03 are all C++
cross-validation *coverage* gaps. Each finding's *primary* remedy is to
recapture/capture real C++ (HIP/NVIDIA) goldens via the xtask harness, which
requires a CUDA box and a local `lib_lightgbm` build — not feasible in this
environment, and the project contract forbids committing unverified fixtures.
Each finding's Fix section, however, explicitly offers a documentation/guard
alternative ("at minimum, document…", "until then, mark…", "or document
that…"). Those alternatives are what these commits implement: they make the
coverage gap loud and unmissable in code (so no future consumer mistakes
self-validation or a capture artifact for C++ reference parity) and add a guard
that fails loudly on a mismatched recapture. The genuine golden capture remains
owed to the named Phase-19/22 consumer (D-02) and is recorded as such in the
docstrings/comments. No fixture was fabricated.

All fixes were verified by Tier 1 (re-read) and Tier 2 (full `cargo test`):
`oracle-harness::primitive_parity` (4 passed), `lgbm-compute::primitives_self`
(22 passed), and `lgbm-compute::split_info` (9 passed) all green after the
changes.

## Fixed Issues

### WR-01: max/min reduction goldens encode a sub-1024-thread 0-identity artifact

**Files modified:** `crates/oracle-harness/tests/primitive_parity.rs`
**Commit:** bebc5bc
**Applied fix:** Replaced the misleading "0-identity bridge" comment in
`primitive_parity_reductions` with an explicit WR-01 note: the committed
max/min goldens were captured at `<<<1,n>>>` with `n<1024` (`num_warp <
warpSize`), where the verbatim C++ reduction folds a `0` for out-of-range warp
lanes — so every `op=max` golden over the all-negative inputs is exactly `0`.
The `.max(0.0)`/`.min(0.0)` reconciliations are now documented as
capture-artifact bridges, NOT production reference semantics, and genuine
reduction parity is flagged as untested until recapture at full 1024-thread
block width (owned by the D-02 consumer). Full golden recapture deferred (needs
CUDA box).

### WR-02: weighted percentile has no C++ oracle (self-validated only)

**Files modified:** `crates/lgbm-compute/src/kernels/primitives.rs`
**Commit:** 705cb99
**Applied fix:** Added a prominent docstring block to `percentile_weighted_f32_on`
stating there is no independent C++ oracle yet — it is validated only by a
serial paraphrase of itself in `primitives_self.rs`, a shared transcription
error would pass green, and the edge-position `values[pos]` quirk is unconfirmed
vs C++. A future consumer is explicitly told not to assume reference parity
until real weighted goldens are captured on a CUDA box
(`LGBM_PRIMITIVE_WEIGHTED_PERCENTILE=1`). Full golden capture deferred (needs
CUDA box).

### WR-03: exclusive prefix-sum cross-warp lanes never cross-validated against C++

**Files modified:** `crates/oracle-harness/tests/primitive_parity.rs`,
`crates/oracle-harness/fixtures/primitives/prefix_sum.txt`
**Commit:** 1c05381 (combined with IN-02 — they share the `WARP` doc hunk)
**Applied fix:** Documented on the `WARP` const that `ShufflePrefixSumExclusive`
is a within-warp building block, so for `n > WARP` the cross-warp combination of
the Rust exclusive scan is NOT cross-validated against C++ (only within-warp
lanes are); its cross-warp correctness rests entirely on the serial self-test,
with the true-global-scan golden capture owned by the Phase-19/22 consumer.
Capture of the `ShufflePrefixSumGlobal` exclusive golden deferred (needs CUDA
box).

### IN-01: `DeviceSplitInfo` read accessors panic while write paths return `Result`

**Files modified:** `crates/lgbm-compute/src/kernels/split_info.rs`
**Commit:** 5755eb2
**Applied fix:** Documented `scalars()`, `cat_threshold()`, and
`cat_threshold_real()` as infallible-by-contract host read helpers that follow
standard Rust slice-indexing convention (an out-of-range slot is a caller bug,
asserted) and are explicitly exempt from the V5 typed-error boundary
(T-14-04-01) the device-facing write paths observe. Took the "document the
asymmetry as intentional" alternative rather than adding `try_*` variants, to
avoid widening the public API in a skeleton phase.

### IN-02: `primitive_parity` hardcodes `WARP = 32`, silently coupling to warp-32 capture

**Files modified:** `crates/oracle-harness/tests/primitive_parity.rs`,
`crates/oracle-harness/fixtures/primitives/prefix_sum.txt`
**Commit:** 1c05381 (combined with WR-03)
**Applied fix:** Added a `WARP_WIDTH 32` metadata record to `prefix_sum.txt`
(skipped by the `records` parser alongside `MASTER_SEED`) and a new
`assert_capture_warp_width` helper called at the start of
`primitive_parity_prefix_sum`, which asserts the fixture's recorded
`WARP_WIDTH` equals the test's hardcoded `WARP`. A warp-64 (GFX9) recapture now
fails loudly at parse time instead of silently mis-validating the
exclusive-scan boundary-lane logic. Verified: `primitive_parity_prefix_sum`
passes with the assertion active.

### IN-03: `bitonic_argsort_global_on` cap (1<<20) permits pathologically slow single-owner sorts

**Files modified:** `crates/lgbm-compute/src/kernels/primitives.rs`
**Commit:** ba3546e
**Applied fix:** Lowered `MAX_GLOBAL_ARGSORT_ELEMENTS` from `1 << 20` to
`16 * BITONIC_SORT_NUM_ELEMENTS` (16384) — a few × the single-block tile,
generous over the largest exercised input (1500) but far below the
multi-megabyte regime that would hang the single-owner serial bitonic network.
Documented that the cap is to be raised when the genuine multi-cube
decomposition lands (Phase-19/22). Verified: the 1500- and 1100-element
multi-block self-tests still pass under the lowered cap.

---

_Fixed: 2026-06-29_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
