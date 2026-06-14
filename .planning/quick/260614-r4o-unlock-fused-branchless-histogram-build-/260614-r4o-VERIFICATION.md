---
phase: quick-260614-r4o
verified: 2026-06-14T00:00:00Z
status: human_needed
score: 6/7 must-haves verified (7th is a wall-clock A/B measurement requiring human re-run)
overrides_applied: 0
human_verification:
  - test: "Interleaved A/B (2 rounds) at small (2k) AND large (200k) with LGBM_PHASE_PROF=1, comparing the committed fused-branchless HEAD vs the spike-003 once-gather baseline, then confirm the LARGE result does NOT regress vs the ~2.89s spike-003 baseline."
    expected: "~-17% small / ~-4.5% large train_median (SUMMARY reports -17.0% small / -6.6% large); the large train_median (~2.73s) must NOT regress past the ~2.89s spike-003 baseline."
    why_human: "Wall-clock benchmark — non-deterministic, machine-variance-dependent, and requires temporarily reverting build_leaf_histograms_raw to the baseline form + two release builds. Cannot be re-run non-destructively within the verifier's no-state-mutation / <10s constraints. The bench harness exists at crates/lgbm/examples/bench_crossover.rs; re-run: LGBM_PHASE_PROF=1 BENCH_SIZES=\"small:2000:12:32\" BENCH_ITERS=100 BENCH_REPS=9 cargo run --release --example bench_crossover (and the large variant)."
---

# Phase quick-260614-r4o: Unlock Fused Branchless Histogram Build Verification Report

**Phase Goal:** Unlock the fused branchless histogram build (spike 003b) by relocating the per-element bin-range validation (V5/T-04-01) out of the hot fold to the once-per-train upstream check; CPU `build_leaf_histograms_raw` folds branchless from the bin column into a reused per-feature hot scratch; bit-exact preserved; no large-row regression; the relocated validation still rejects out-of-range bins with a typed error.

**Verified:** 2026-06-14
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | CPU default `build_leaf_histograms_raw` folds DIRECTLY from the bin column into a reused per-feature hot scratch, branchless, then stream-copies into `out` | ✓ VERIFIED | lib.rs:261-285 — single `scratch` Vec sized `2*max_num_bin` (261-266), per-feature `bin = bins[row]` read inline (274), no per-element runtime `if bin < num_bin` branch, `out[..].copy_from_slice(&scratch[..cells])` stream-copy (284). No `ord_bins` materialization, no per-feature alloc. |
| 2 | `ord_g`/`ord_h` gathered ONCE per leaf (spike-003 preserved); only the bin read is fused into the fold | ✓ VERIFIED | lib.rs:245-251 — the `for &row in leaf_rows { ord_g.push(...); ord_h.push(...); }` gather sits BEFORE the `for (fpos, &bins)` feature loop (267); the fold reads `ord_g[k]`/`ord_h[k]` (281-282). Gathered once, reused across features. |
| 3 | f64 fold ORDER byte-unchanged; CPU bit-exact merge gate (oracle learner_parity 29/0) stays BIT-EXACT | ✓ VERIFIED | Fold: ascending `leaf_rows` (273), grad at `bin*2`, hess at `+1` (280-282), `f64::from(...)` f32-read→f64-accumulate. Re-ran `cargo test -p oracle-harness --test learner_parity` → **29 passed / 0 failed**. lgbm-compute lib 21/0; full oracle all suites 0-failed. |
| 4 | Bin-range validity established ONCE per train by the upstream check in `train_inner`, rejecting out-of-range bins with `TreeLearnerError::BinIndexOutOfRange`; the V5/T-04-01 mitigation MOVES there, not removed | ✓ VERIFIED | learner.rs:707-726 — the `for &b in &f.bins { if b >= f.num_bin { return Err(BinIndexOutOfRange{..}) } }` loop with the documented "SINGLE bin-range gate, RELOCATED" comment (707-718). The per-element check was removed from the fold (truth #1) and survives here. |
| 5 | `debug_assert!(bin < num_bin)` guards the hot fold as defense-in-depth | ✓ VERIFIED | lib.rs:275-279 — `debug_assert!(bin < num_bin, "...T-04-01 relocation")` inside the fold; free in release, fires in debug/test. |
| 6 | A test proves the once-per-train upstream validation REJECTS an out-of-range bin with the typed `BinIndexOutOfRange` | ✓ VERIFIED | learner.rs:3423-3459 — `train_rejects_out_of_range_bin_with_typed_error`: bins `[0,1,3]` with num_bin 3, asserts `Err(BinIndexOutOfRange{index==3, num_bin==3})`, panics on `Ok` or any other variant (teeth present). Re-ran: test **passed**, treelearner 66/0. |
| 7 | Interleaved A/B (2 rounds) small+large shows ~-17% small / ~-4.5% large; LARGE does NOT regress vs the ~2.89s spike-003 baseline | ? UNCERTAIN | SUMMARY reports -17.0% small / -6.6% large (2.73s, no regression). Bench harness exists (crates/lgbm/examples/bench_crossover.rs). Wall-clock A/B requires destructive code revert + two release builds + machine-variance-sensitive timing — cannot re-run within verifier constraints. Routed to human verification. |

**Score:** 6/7 truths verified; 1 routed to human (wall-clock benchmark).

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/lgbm-compute/src/lib.rs` | Fused branchless `build_leaf_histograms_raw` + relocated-validation doc/threat contract | ✓ VERIFIED | Fused fold at 261-286; precondition doc + `# Errors` rewrite at 202-226 (cites learner.rs:700-714, C++ `dense_bin.hpp`, 003b). Contains `build_leaf_histograms_raw`. Substantive (57 inserted lines, real fold logic), wired (it IS the trait default used by `CpuBackend`). |
| `crates/lgbm-treelearner/src/learner.rs` | Once-per-train upstream validation (kept/strengthened) + typed-rejection test | ✓ VERIFIED | Gate + documenting comment at 700-726; rejection test at 3423-3459. Contains `BinIndexOutOfRange`. Substantive, wired (gate runs in `train_inner` before leaf growth). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| learner.rs `train_inner` upstream bin-range check | lib.rs `build_leaf_histograms_raw` branchless fold precondition | caller-guaranteed-precondition contract (doc + T-04-01 relocation) | ✓ WIRED | Both sites cross-reference: lib.rs:207-210 cites "`train_inner`, learner.rs:700-714"; learner.rs:712-717 cites "lgbm-compute/src/lib.rs ... build_leaf_histograms_raw ... Bin-range precondition doc". Shared `BinIndexOutOfRange` token couples them. The gate runs once-per-train BEFORE any leaf build; the fold trusts the established invariant. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `build_leaf_histograms_raw` | `out` (f64 histogram buffer) | folds real `gradients`/`hessians` (via `ord_g`/`ord_h`) keyed by real `bins[row]` | ✓ Yes — verified by 29/0 bit-exact parity against C++ lib_lightgbm 4.6 | ✓ FLOWING |

### Scope Containment (GPU/RocmBackend/split-scan/partition untouched)

| Concern | Status | Evidence |
|---------|--------|----------|
| RocmBackend override (lib.rs:900) untouched | ✓ VERIFIED | `git show f48adac` diff hunks confined to lines ~199-285 (trait default + doc); no `+` lines mention Rocm/cubecl/split_scan/data_partition. |
| GPU/cubecl/split-scan/partition untouched | ✓ VERIFIED | Both commits (f48adac, 1a09a04) touch ONLY crates/lgbm-compute/src/lib.rs (+57/-11) and crates/lgbm-treelearner/src/learner.rs (+50). |
| LightGBM/ never git-added | ✓ VERIFIED | `git ls-files LightGBM/` → 0 tracked; `git status` → `?? LightGBM/` (untracked). |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Compute lib tests | `cargo test -p lgbm-compute --lib` | 21 passed / 0 failed | ✓ PASS |
| Treelearner lib + new rejection test | `cargo test -p lgbm-treelearner` | 66 passed / 0 failed; `train_rejects_out_of_range_bin_with_typed_error` ... ok | ✓ PASS |
| PRIMARY bit-exact gate | `cargo test -p oracle-harness --test learner_parity` | 29 passed / 0 failed — BIT-EXACT | ✓ PASS |
| Full oracle harness (regression check) | `cargo test -p oracle-harness` | every suite 0-failed (boosting 75/0, kernel 6/0, learner 29/0, metric 15/0, predict 5/0, rank 4/0, raw_bin 2/0, rng 1/0, advanced 5/0, comparator 5/0, config_drift 3/0) | ✓ PASS |
| Clippy on edited region | `cargo clippy -p lgbm-compute` | only warning at lib.rs:319 (`find_best_splits_batched`, NOT this task's region); fold 261-286 clean | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| PERF-R3-FUSED-HIST | 260614-r4o-PLAN | Fused branchless CPU histogram build (R3 lever) at zero parity cost | ✓ SATISFIED (perf number → human) | Fused fold present + bit-exact (truths 1-6); the perf magnitude is the human-verify item (truth 7). |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None in edited regions | — | No TODO/FIXME/XXX/TBD/unimplemented! in the fold (lib.rs 228-287) or the gate/test (learner.rs 700-726, 3423-3459). The `Ok(out)` return is correct (no per-element fallible call by design). |

### Human Verification Required

#### 1. Interleaved A/B benchmark (small + large), large-no-regression

**Test:** Run the interleaved A/B (2 rounds) comparing the committed fused-branchless HEAD vs the spike-003 once-gather baseline, at small (2k) and large (200k):
```
LGBM_PHASE_PROF=1 BENCH_SIZES="small:2000:12:32" BENCH_ITERS=100 BENCH_REPS=9 cargo run --release --example bench_crossover
LGBM_PHASE_PROF=1 BENCH_SIZES="large:200000:32:64" BENCH_ITERS=50 BENCH_REPS=5 cargo run --release --example bench_crossover
```
(The baseline arm requires temporarily reverting only the `build_leaf_histograms_raw` body to the spike-003 once-gather form, then restoring.)

**Expected:** ~-17% small / ~-4.5% large train_median (SUMMARY reports -17.0% / -6.6%); the large train_median (~2.73s) must NOT regress past the ~2.89s spike-003 baseline. If large regresses, the result should be rejected/reverted.

**Why human:** Wall-clock measurement — non-deterministic, machine-variance-sensitive, requires destructive code revert + two release builds. Cannot be re-run non-destructively within the verifier's no-state-mutation / <10s constraints.

### Gaps Summary

No blocking gaps. All structural, correctness, scope, and bit-exact must-haves are VERIFIED against the codebase:

- The fused branchless fold exists exactly as specified (reused per-feature scratch, inline `bins[row]` read, no per-element branch, `debug_assert!` defense-in-depth, once-per-leaf `ord_g`/`ord_h` gather, byte-unchanged f64 fold order, stream-copy into `out`).
- The V5/T-04-01 mitigation is RELOCATED (not removed): the once-per-train upstream gate in `train_inner` rejects out-of-range bins with the typed `BinIndexOutOfRange`, proven by a test with teeth.
- The doc + threat-model contract cross-references both sites.
- The PRIMARY bit-exact merge gate is re-run-green (oracle learner_parity 29/0 BIT-EXACT; full oracle 0-failed; compute 21/0; treelearner 66/0).
- GPU/RocmBackend/split-scan/partition untouched; LightGBM/ never git-added; working tree matches committed HEAD.

The sole unverifiable item is the wall-clock A/B performance magnitude (truth 7), which is inherently a human/benchmark measurement and is routed to human verification. The codebase is functionally complete and the phase goal is structurally achieved; only the performance claim awaits human confirmation.

---

_Verified: 2026-06-14_
_Verifier: Claude (gsd-verifier)_
