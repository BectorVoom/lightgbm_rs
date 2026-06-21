---
phase: 04-compute-backend-cpu-first-integer-histograms-rocm
fixed_at: 2026-06-06T00:00:00Z
review_path: .planning/phases/04-compute-backend-cpu-first-integer-histograms-rocm/04-REVIEW.md
iteration: 1
findings_in_scope: 8
fixed: 8
skipped: 0
status: all_fixed
---

# Phase 4: Code Review Fix Report

**Fixed at:** 2026-06-06
**Source review:** .planning/phases/04-compute-backend-cpu-first-integer-histograms-rocm/04-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 8 (CR-01, CR-02, WR-01..WR-06)
- Fixed: 8
- Skipped: 0
- Info findings (IN-01..IN-04): out of scope (`critical_warning`), not addressed.

The CPU bit-exact parity gate (`kernel_parity_{histogram,split,partition,subtract}_bit_exact_on_cpu`)
is green after all fixes; `cargo test --workspace` reports 0 failures. The C++ golden corpus
was confirmed byte-idempotent (`cargo run -p xtask -- kernel-capture` twice → identical
sha256, empty `git diff`).

## Fixed Issues

### CR-01 / CR-02: Oracle/host L1 sign uses `signum()` (≠ C++ `Sign`)

**Files modified:** `crates/oracle-harness/tests/kernel_parity.rs`
**Commit:** d09b26a
**Applied fix:** Replaced the parity test's `leaf_gain` / `leaf_gain_f32` host mirrors —
which used Rust `f64::signum` / `f32::signum` (returns `+1.0` at `0.0`, `-1.0` at `-0.0`,
never `0.0`) — with direct delegation to the production `#[cube]` primitives
`lgbm_compute::gain::get_leaf_gain` and `get_leaf_gain_f32` (called as plain host fns).
This eliminates the second source of truth entirely, so the bit-exact parity gate now uses
the same `Common::Sign(s) = (s>0)-(s<0)` semantics as the kernel and C++ reference. The
`get_leaf_gain_f32` import is scoped to the `#[cfg(feature = "rocm")] mod hip` block (its
only user) so the default build has no unused import. CR-01 and CR-02 are the same defect
surface (whole-leaf `gain_shift` and the per-candidate path both routed through these
helpers); a single fix closes both.

**Note (human verification suggested):** This is a numerical-fidelity correctness fix.
It is exercised and passes the existing parity gate, but the golden corpus does not yet
contain a zero-gradient L1 case (`sum_gradient == 0.0`, `lambda_l1 > 0`) that would directly
discriminate `signum` from `Sign`. See Follow-ups below.

### WR-02: Negative-`t` OOB in REVERSE split scan

**Files modified:** `crates/lgbm-compute/src/kernels/split.rs`
**Commit:** fc40302
**Applied fix:** The REVERSE scan (both the f64 and f32 kernels) replaced the C++
`for (; t >= 1 - offset; --t)` loop with a `0..rev_count` forward counter, dropping the
`t >= 1 - offset` lower bound. For an out-of-contract `offset >= 2`, `t` would go negative and
`(t as usize)` would wrap to an enormous index → OOB read on the cubecl-lowered array access.
Restored the bound with an `in_range = t >= (1 - offset)` gate (folded into `active`, so an
out-of-range iteration contributes nothing) plus a branchless index clamp
`t_safe = select(t < 0, 0i32, t)` so a negative `t` reads bin 0 inertly. For the valid
`offset ∈ {0,1}` cases (the only values C++ ever produces — `int8_t offset` is set to 0 or 1
in `FeatureMetainfo`) `t` never drops below `t_end ∈ {0,1}`, so this is a strict no-op there.
Updated the host `SAFETY` comment to document the new in-range guarantee. Split parity remains
bit-exact, including the `default_bin_skip` (offset=1) case.

**Note (human verification suggested):** Logic/bounds fix in production kernel code; passes
bit-exact parity but the negative-`t` path is unexercised by the corpus (no `offset >= 2`
golden, by design). The guard is provably inert for the in-contract cases.

### WR-03: `SplitInfo::none()` `default_left` invariant undocumented

**Files modified:** `crates/lgbm-compute/src/gain.rs`
**Commit:** d62595a
**Applied fix:** Documented on `SplitInfo::none()` that `default_left` is a don't-care
sentinel on a no-split result (value `true` matches the C++ default-constructed
`SplitInfo::default_left`), and that consumers MUST gate on `gain != kMinScore`
(`is_splittable`) before reading it. Explains why the parity gate only asserts `default_left`
on splittable winners. Documentation only; no behavior change.

### WR-05: f32 split kernel relies on literal-type inference

**Files modified:** `crates/lgbm-compute/src/gain.rs`
**Commit:** 51d32e0
**Applied fix:** Pinned every float literal in `threshold_l1_f32` to f32
(`f32::max(0.0f32, ...)`, `select(s > 0.0f32, 1.0f32, 0.0f32)`), removing any ambiguity that
cubecl `#[cube]` literal inference could resolve a bare `1.0`/`0.0` to f64 and silently widen
or mix precision on the hip path the f32 mirror exists to keep f64-free. The f64 anchor is
untouched. Verified the `--features rocm` build still compiles (ROCm 7.1.1 toolchain present).

### WR-01 / WR-06: C++ capture-harness hardening

**Files modified:** `xtask/cpp/kernel_capture.cpp`
**Commit:** 629af47
**Applied fix:**
- **WR-01:** Added a `threshold < num_bin` assertion at the top of `EmitPCase` (with
  `<cstdlib>` for `std::abort`), mirroring the kernel contract (`data_partition_on` returns
  `ComputeError::Runtime` for `threshold >= num_bin`). An out-of-contract `PCaseSpec` now
  aborts at capture time instead of producing a golden that panics at replay.
- **WR-06:** Reordered `SparseBin::Data` to check `i_delta >= num_vals_` BEFORE dereferencing
  `deltas_[i_delta]`, matching upstream `sparse_bin.hpp` bound discipline, so the golden
  generator cannot read past the delta stream.

Both are dev-only generator hardening. Confirmed `cargo run -p xtask -- kernel-capture` still
emits byte-identical goldens (sha256 unchanged for histogram/split/partition/subtract; empty
`git diff` on the fixtures and `REFERENCE_MANIFEST.md`).

### WR-04: `cfg_skip_default_bin` heuristic may not match C++ `SKIP_DEFAULT_BIN`

**Files modified:** `crates/lgbm-compute/src/kernels/split.rs`
**Commit:** aa5c34f
**Applied fix:** Documented the authoritative C++ predicate this heuristic approximates.
From `FuncForNumricalL3` (feature_histogram.hpp:396-429), the numeric (non-quantized) path
sets `SKIP_DEFAULT_BIN == true` **iff** `num_bin > 2 && missing_type == MissingType::Zero`
(NaN → `NA_AS_MISSING=true, skip=false`; None → both false) — a function of `missing_type`,
not of `default_bin` vs `num_bin`. Recorded the Phase-4 precondition under which the
`default_bin < num_bin` heuristic is sound (satisfied by the committed corpus, verified by the
bit-exact gate) and the Phase-5 follow-up to thread the authoritative flag through the public
signature.

**Rationale for documentation-only fix (per fix guidance):** Threading the true flag requires
adding `missing_type` (or a `skip_default_bin` bool) to `find_best_split_cpu`,
`find_best_split_raw_f32_on`, and the public `Backend::find_best_split` trait + all callers —
a public-signature change. The current parity is bit-exact with the heuristic, and the
guidance explicitly permits documenting the precondition + recording a follow-up rather than
destabilizing green parity. No divergence introduced.

## Skipped Issues

None — all 8 in-scope findings were fixed.

## Follow-ups (recorded, not done in Phase 4)

1. **Zero-gradient L1 golden (CR-01/CR-02 reinforcement):** Add a `BuildSplitCorpus` case with
   `sum_gradient == 0.0` and `lambda_l1 > 0` in `xtask/cpp/kernel_capture.cpp` and regenerate
   `split.txt`, to directly exercise the `Sign(0) == 0` boundary that discriminates `Sign` from
   `signum`. Deferred because adding a corpus case must be proven byte-idempotent and only
   change the intended fixture; the sign-helper fix already closes the actual defect, so the
   golden addition is reinforcement, not a blocker.

2. **Authoritative `skip_default_bin` threading (WR-04):** Replace the `default_bin < num_bin`
   heuristic with the C++ predicate (`num_bin > 2 && missing_type == MissingType::Zero`)
   threaded from the caller through `find_best_split_cpu` / `Backend::find_best_split`, and add
   a golden where `default_bin < num_bin` but `skip_default_bin == false`. Deferred to Phase-5
   because it changes the public trait signature.

---

_Fixed: 2026-06-06_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
