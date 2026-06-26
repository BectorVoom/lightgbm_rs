---
phase: quick-260627-bw7
plan: 01
subsystem: lgbm-compute (rocm histogram kernels)
tags: [performance, gpu, rocm, vectorization, vector, dequant, fix-compact, bit-exact, measure-and-revert, documented-negative]
requirements: ["QUICK-260627-bw7"]
dependency_graph:
  requires: ["spike-044 (Vector<f64,2> cross-type-cast dequant recipe)", "spike-045 (Vector-regresses-inside-these-kernels precedent + cube-macro gotchas)"]
  provides: ["evidence: fix_compact_kernel dequant vectorizes bit-exactly on the LIVE resident path but is net-neutral e2e → reverted to scalar"]
  affects: ["crates/lgbm-compute/src/kernels/histogram.rs (fix_compact_kernel — final state = SCALAR, unchanged from HEAD)"]
tech_stack:
  added: []
  patterns: ["measure-and-revert gate (spike-043/045 precedent)", "git-checkout-file A/B baseline (no git stash — shared-ref hazard avoided)"]
key_files:
  created: []
  modified: []
decisions:
  - "REVERTED — the Vector<f64,2> dequant is bit-exact on every CPU+ROCm gate but the live resident wide-path A/B shows NO SEP-WIN (net-neutral within APU noise); per the measure-and-revert mandate (net-neutral-or-slower → revert) + spike-044 bounded-ROI + spike-045 Vector-regresses priors, the scalar kernel stays."
metrics:
  duration: "~16 min"
  completed: "2026-06-26"
  tasks: 3
  files_changed: 0
status: complete
---

# Quick 260627-bw7: Wire spike-044 `Vector<f64,2>` dequant into the live ROCm resident fix/compact — Summary

**Measured the spike-044 `Vector<f64,2>` cross-type-cast dequant in-place on the LIVE wide-rocm resident `fix_compact_kernel`: bit-exact on every CPU+ROCm parity gate, but the resident wide-500-feat A/B shows no robust win (net-neutral within the spoofed-APU noise) — so per the measure-and-revert mandate the kernel was REVERTED to scalar and the negative result documented. The honest, predicted deliverable per spike-044's bounded-ROI verdict.**

## Disposition: REVERTED (measured net-neutral, documented negative)

Final tree state: `fix_compact_kernel` is the original **SCALAR** kernel, byte-identical to HEAD (`git diff` empty). No code change shipped. The CPU f64 anchor was never touched (rocm-track only).

## STEP 0 — live-path investigation (confirmed)

- The dequant `hist[i] = f64::cast_from(i64::cast_from(h_raw[i])) / 2^30` lives ONLY in `fix_compact_kernel` (`histogram.rs:2347-2351`), the u64-fixed-point dequant pass.
- `fix_compact_kernel` is launched at TWO sites, both updated during the wire attempt:
  1. `fix_compact_f64_on` (`histogram.rs:2527`) — oracle/test path.
  2. `build_fix_compact_resident_f64_on` (`histogram.rs:2736`) — the **LIVE** wide-rocm resident directly-built-leaf path (root + smaller children at `num_data >= RESIDENT_MIN_NUM_DATA=12_000`), reached via `learner.rs::build_resident_leaf_into → Backend::build_resident_leaf`.
- The other fused kernel `build_fix_scan_fused_kernel` (`histogram.rs:2846`) does a SEQUENTIAL f64 build with NO dequant step AND is OFF by default (`resident_pool.rs::FUSED_MAX_NUM_DATA = -1`, never auto-engages) — so it is NOT the live dequant carrier and was NOT touched.
- Conclusion: `fix_compact_kernel`'s dequant IS on the live wide-rocm resident path; the wire had a real (if bounded) production effect — NOT a no-op wire.

## What was attempted (Task 1, then reverted in Task 3)

Rewrote `fix_compact_kernel` to the `Vector<f64,2>` pair layout (generic `<N: Size>`, N hard-coded to 2 by both callers — the `[g,h]` pair width = hip f64's max `io_optimized_vector_sizes` lane count):

- **DEQUANT vectorized** (spike-044 recipe): per bin pair `Vector<u64,2> → <i64,2> → <f64,2>` with a broadcast divide `Vector::<f64,N>::new(SCALE_F64)` (2^30). Per-lane cast bit-identical to the scalar path ⇒ bit-exact by construction.
- **FIX stayed SCALAR** (load-bearing dependent f64 reduction, never reordered): read each pair as a vector, extract comptime lanes `p[0]`/`p[1]`, accumulate the ascending branchless `select` fold; write the most-freq pair via the proven load-modify-store-lanes pattern (`kernel_vector_loop_unroll`, cubecl `runtime_tests/vector.rs`).
- **COMPACT vectorized**: whole-pair `Vector<f64,2>` copies / `Vector::new(0.0)` zeros, same ascending guard structure.
- Both launch sites updated to the spike-044 ABI: `2usize` vector_size right after `CubeDim::new_1d(1)`, array lengths in vector units (`raw.len()/2`, `slot_len/2` — both even). Underlying byte buffers unchanged; V5 validation + overflow guard + `launch_unchecked` SAFETY comments + `#[cfg(feature="rocm")]` all preserved.

`cargo build --release --features rocm` compiled clean on the first attempt — no fallback needed (the cube-macro gotchas from spike-045 were pre-empted: generic `N:Size` not a literal `Vector<_,2>`; comptime lane indices not runtime).

## Task 2 — bit-exact parity gate (on the VECTORIZED kernel): ALL GREEN

| Gate | Result |
|------|--------|
| `cargo test -p lgbm-treelearner --lib` | 77 passed, 0 failed |
| `cargo test -p lgbm` | 41 passed, 0 failed |
| `cargo test -p oracle-harness` (incl. `raw_bin_train_matches_cpp_golden`) | all green; golden ✓ |
| `cargo test -p oracle-harness --features rocm` | all green |

Decisive ROCm bit-exact cells (the dequant/fix/compact carriers), all `ok` on the vectorized kernel:
- `hip::kernel_parity_fix_compact_equals_host_on_hip`
- `hip::kernel_parity_resident_build_fix_compact_equals_host_on_hip`
- `hip::kernel_parity_resident_build_fix_compact_p_gt_1_equals_host_on_hip`
- `hip::learner_parity_resident_equals_host_tree_on_hip`

⇒ The vectorized dequant + the scalar fix fold + the pair-copy compact are **byte-identical** to the scalar kernel. Bit-exact confirmed (as predicted by construction).

## Task 2 — A/B sign measure (resident wide-500-feat path): NET-NEUTRAL (no SEP-WIN)

Method: two `--release --features rocm` builds — SCALAR baseline (restored via `git checkout -- <file>`) vs VECTORIZED — each run `LGBM_BENCH_SWEEP=wide LGBM_RESIDENT_FORCE=1 cargo run --release --features rocm --example bench_gpu_vs_cpu` (250k/500k/1M × 500 feat; `LGBM_RESIDENT_FORCE=1` pins the live resident path so the dequant is exercised). **3 process restarts per arm.** Whole-train median (s); sign-only (spoofed 8-CU APU confounds magnitude).

| bucket | vec median | scalar median | vec/scalar | vec range | scalar range | SEP-WIN? |
|--------|-----------:|--------------:|-----------:|:---------:|:------------:|:--------:|
| 250k×500 | 3.30 | 3.46 | 0.954 | [3.16, 3.39] | [3.36, 3.59] | **NO** |
| 500k×500 | 5.53 | 6.29 | 0.879 | [4.94, 5.98] | [5.26, 6.30] | **NO** |
| 1M×500   | 11.26 | 11.66 | 0.966 | [10.16, 11.40] | [9.72, 11.78] | **NO** |

**Read:** the vectorized median leans slightly faster in all 3 buckets, but **no bucket shows a SEP-WIN** (vec p75 < scalar p25, the CONVENTIONS robustness bar) — the ranges overlap heavily, and at 1M the scalar's best run (9.72s) beats the vectorized best (10.16s). The dequant is a sub-1% fused-minority fraction of whole-train (spike-044), so a 5–12% median lean **cannot** be the dequant vectorization — it is APU run-to-run / thermal noise (the scalar 500k arm was bimodal: 5.26 / 6.29 / 6.30). Classification: **NET-NEUTRAL** (no robust, attributable win).

## Task 3 — ship-or-revert decision

**REVERT.** Per the controlling measure-and-revert mandate ("if the vectorized kernel is NET-NEUTRAL-OR-SLOWER, REVERT"), a net-neutral A/B with no SEP-WIN does not clear the ship bar, and the mandate is explicit: do not ship a regression/non-win for a sub-1% theoretical gain. This matches the strong priors — spike-044's bounded-ROI DON'T-WIRE verdict (vec2-capped weak hip win on a fused minority fraction) and spike-045's "Vector regresses inside these kernels" finding. Shipping would add generic-`N`/vector-ABI complexity (with a hard `N==2` caller coupling) to a ROCm-parity-track path that already loses to the 16-core CPU ~4× at wide, for no robust e2e benefit.

The kernel was restored to scalar (`git checkout -- crates/lgbm-compute/src/kernels/histogram.rs`) and the reverted tree re-verified GREEN:
- `cargo build --release --features rocm` ✓
- `cargo test -p oracle-harness --features rocm` ✓ (incl. `kernel_parity_fix_compact_*`, `resident_build_fix_compact_*`, `learner_parity_resident_equals_host_tree_on_hip`)

`git diff` for `histogram.rs` is empty (byte-identical to HEAD). CPU f64 anchor (`subtract_histograms_cpu_native`, native host `fix_histogram`/`build_fix_scan`) byte-untouched.

## Deviations from Plan

None. The plan explicitly framed "measured, net-negative/neutral, reverted, documented" as a fully acceptable honest deliverable, and that is the outcome. The in-place rewrite compiled cleanly so the 2-launch FALLBACK was never needed.

## Known Stubs

None.

## Threat Flags

None — internal ROCm compute-kernel change only; no new external input, network, dependency, or trust boundary. Final tree carries no net change.

## Self-Check: PASSED

- `crates/lgbm-compute/src/kernels/histogram.rs` exists and is byte-identical to HEAD (scalar `fix_compact_kernel`) — `git diff --stat` empty. ✓
- No code commit was made (correct — REVERT disposition leaves the tree == HEAD). ✓
- CPU anchor untouched; reverted tree green on `cargo build --release --features rocm` + `cargo test -p oracle-harness --features rocm`. ✓
