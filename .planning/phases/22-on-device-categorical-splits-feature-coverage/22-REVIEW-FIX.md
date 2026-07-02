---
phase: 22-on-device-categorical-splits-feature-coverage
fixed_at: 2026-07-02T11:15:33Z
review_path: .planning/phases/22-on-device-categorical-splits-feature-coverage/22-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 22: Code Review Fix Report

**Fixed at:** 2026-07-02T11:15:33Z
**Source review:** .planning/phases/22-on-device-categorical-splits-feature-coverage/22-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (fix_scope = all: CR + WR + IN)
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: Device categorical partition off-by-`offset` for `most_freq_bin == 0`

**Files modified:** `crates/lgbm-compute/src/kernels/categorical_split.rs`, `crates/lgbm-compute/src/kernels/grow_driver.rs`
**Commit:** c5fca8b
**Applied fix:** The device router (`route_to_left_categorical`) keys a member row's raw
bin as `bin - min_bin + offset`, but `set_real_threshold` built the inner routing
bitset with bits at the *raw* winning bin, so device routing diverged whenever
`min_bin != offset` (i.e. `most_freq_bin == 0 ⇒ offset 1`). Verified the winning-bin
value the finder emits (`cat_threshold = t + offset`, where compacted slot `t` maps to
raw bin `t + offset`) equals the member row's raw bin, so building the inner bitset over
the transformed keys `winning_bin - min_bin + offset` (option (a)) places each bit
exactly where the router looks it up — provably key-consistent for every offset. Threaded
`min_bin` into `set_real_threshold` and updated the call site to pass `f.min_bin`. The two
committed fixtures (`min_bin == offset == 0`) are byte-unchanged; added `offset == 1` and
nonzero-`min_bin` regression unit tests. Note: option (b) (route the device partition by
the real category-value bitset + `bin_to_category`) was NOT taken because the device
routing kernel only supports inner-bin routing (`bin - min_bin + offset`) — the verbatim
C++ `dense_bin.hpp` convention — and option (a) is the smaller, key-consistent fix.

### WR-03: No coverage for `most_freq_bin == 0` (offset 1) categorical

**Files modified:** `crates/lgbm-compute/src/kernels/grow_driver.rs`
**Commit:** 79cc39c
**Applied fix:** Added `categorical_offset1_full_chain_routes_by_membership`, a one-hot
`offset == 1` fixture driven through the whole evaluator → `set_real_threshold` →
`partition_categorical_on_device` chain, pinned to the **category-membership golden** (a
row routes LEFT iff its raw bin is a winning category; one-hot with `most_freq_bin == 0`
defaults non-members RIGHT). This assertion is independent of the router's internal key
arithmetic and therefore *fails* under the pre-CR-01 inner bitset (members would route
right, `split_point == 0`) and passes after the key-consistency fix. Also added the two
CR-01 unit tests (`set_real_threshold_inner_key_offset1_matches_router_key`,
`set_real_threshold_inner_key_nonzero_min_bin`) in the CR-01 commit.

### WR-01: On-device categorical parity gates never run in the default merge gate

**Files modified:** `crates/oracle-harness/tests/learner_parity.rs`
**Commit:** 23cf762
**Applied fix:** The categorical STRUCTURE gate (`assert_categorical_device_structure_gate`)
is a pure f64-vs-f64 comparison on the cubecl-cpu lane via the direct driver entrypoint
(no GPU, no env). Removed its `if env_on` guard so it runs unconditionally in the default
merge gate, restoring coverage of the on-device categorical grow-driver wiring (partition
routing, slab staging, bitset construction) — the gap that let CR-01 ship green. The
real-4.6-golden FIDELITY gate (`run_categorical_cell_on_device`) stays env-gated. Verified
the gate runs and passes with `LGBM_CUDA_ON_DEVICE` unset.

### WR-02: Stage-1 categorical seam uses default categorical config

**Files modified:** `crates/lgbm-compute/src/kernels/best_split.rs`
**Commit:** 54fa5a9
**Applied fix:** Took option (b) — the three `pub` stage-1 launchers
(`find_best_splits_stage1_on` / `_f32_on` / `_globalmem_f32_on`) now return a typed
`ComputeError` (`categorical_seam_unsupported`) on a categorical task instead of silently
evaluating with `GainConfig::default()` categorical knobs (10/10/32/4/100). The live grow
driver (`grow_driver::scan_leaf`) bypasses this seam entirely; confirmed no caller passes a
categorical task to these launchers (all use `is_categorical: false`). Removed the now-dead
`categorical_gain_config` / `categorical_split_scalars` helpers and unused imports; updated
the launcher doc comments.

### WR-04: `both_too_small` combined gate diverges from C++ per-leaf `min_data*2`

**Files modified:** `crates/lgbm-compute/src/kernels/grow_driver.rs`
**Commit:** 9ba68ac
**Applied fix:** Replaced the single combined `both_too_small` predicate with a per-child
gate (`(leaves[child].rows.len() as i32) < min_data_x2`) inside the child loop, mirroring
C++ `BeforeFindBestSplit`'s per-leaf `num_data < min_data_in_leaf * 2` check, and used
`i32::saturating_mul` for the `* 2` (guards a pathological `min_data > i32::MAX/2`
overflow). Tree structure is unchanged (the small child's splits are already rejected by
`scan_leaf`'s per-side `min_data_in_leaf` guards); the on-device mindata + structure parity
gates confirm no regression.

### IN-01: `cat_threshold_real` slab staged but dead on the grow path

**Files modified:** `crates/lgbm-compute/src/kernels/categorical_split.rs`, `crates/lgbm-compute/src/kernels/grow_driver.rs`
**Commit:** 9d3aa06
**Applied fix:** Took the reviewer's first option — the driver now builds `real_bitset` by
**consuming** the `cat_threshold_real` slab (`dsi.cat_threshold_real(best_leaf)`, the
`win_real` mapping already staged) instead of re-mapping bin→category a second time inside
`set_real_threshold`. Extracted the CR-01 inner-key transform into a dedicated
`construct_inner_bitset` helper the driver calls directly (single source of truth for the
transform). `set_real_threshold` is retained (delegates to the same helper) for the unit
tests. Behavior unchanged: the slab real equals `set_real_threshold`'s `cat_values`
bit-for-bit; categorical + structure parity gates green.

### IN-02: Unreachable `.expect` in the categorical branch

**Files modified:** `crates/lgbm-compute/src/kernels/grow_driver.rs`
**Commit:** ca2ca3b
**Applied fix:** Replaced `split_info.as_mut().expect(...)` with
`split_info.as_mut().ok_or_else(|| ComputeError::Runtime { .. })?`, keeping the grow loop's
error boundary uniformly typed. The branch remains provably unreachable (categorical branch
⇒ `has_categorical` ⇒ `split_info == Some`).

## Verification

- `cargo test -p lgbm-compute --lib` — 134 passed (all categorical / partition / split_info
  / best_split unit tests, including the new CR-01 + WR-03 tests).
- `cargo test -p oracle-harness --test learner_parity` — 34 passed (default merge gate,
  including the now-unconditional categorical structure gate).
- `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity categorical` —
  5 passed, including the real-4.6-golden on-device fidelity gates
  (`learner_parity_categorical_{onehot,manyvsmany}_on_device`) that anchor CR-01 to the C++
  reference.
- `cargo test -p oracle-harness --test best_split_parity` — 4 passed (stage-1 numeric
  launchers unaffected by the WR-02 categorical-branch change).
- `cargo test -p lgbm-compute --test split_info` — 9 passed.

**Parity note:** CR-01, WR-04, IN-01 touch split/partition/bitset logic. The two committed
categorical fixtures are `offset == 0` (byte-unchanged), the new `offset == 1` coverage is
pinned to a category-membership golden, and the deterministic cubecl-cpu f64 anchor gates
(structure + real-4.6 fidelity) all reproduce bit-exact — the non-negotiable parity contract
holds.

---

_Fixed: 2026-07-02T11:15:33Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
