---
phase: 18-on-device-data-partition-tree-mutation-prediction
plan: 02
subsystem: compute
tags: [cubecl, data-partition, mark-prefix-sum-scatter, categorical, subtraction-trick, ODL-13]

# Dependency graph
requires:
  - phase: 18-on-device-data-partition-tree-mutation-prediction
    plan: 01
    provides: u16/u32 integer block prefix-sum launchers (PrepareOffset/AggregateBlockOffset), partition.txt fan-out/PCAT/PPACKET goldens, partition_parity.rs scaffold
  - phase: 16-histogram-constructor
    provides: HistArena {parent,smaller,larger} rotate() + counted-alloc discipline
  - phase: 14-foundation
    provides: split_info.rs SplitScalars (CUDASplitInfo per-side sums)
provides:
  - data_partition.rs §9 device mark→prefix-sum→scatter (numeric + categorical) + 16-int SplitTreeStructure packet + cpu f64 stable-partition anchor + shared route/find_in_bitset #[cube] fns
  - histogram_arena.rs leaf-indexed whole-pool swap() alongside rotate()
  - partition_parity order/cat/packet cells un-ignored, driven live vs the golden
affects: [18-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Shared pub(crate) #[cube] route fn with #[comptime] bool flag fan-out passed as plain bools at the launch site (cubecl auto-specializes+caches per combination — no host match)"
    - "Branchless select stores throughout the route decision (SP-2 cubecl-cpu MLIR constraint); comptime bools folded to i32 consts at fn top"
    - "Global stable-partition scatter (D-04 order-equivalence) reusing the 18-01 u32 exclusive + u16 inclusive prefix-sum primitives; exclusive rank = the [tid-1] derivation"
    - "cpu f64 anchor is a plain-Rust route mirror + stable partition (never launches a kernel, D-12); device fold cross-checked byte-equal to it"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/data_partition.rs
    - crates/lgbm-compute/src/kernels/histogram_arena.rs
    - crates/oracle-harness/tests/partition_parity.rs

key-decisions:
  - "Route flags are #[comptime] bools but passed as plain host-computed bool values at the launch call site — cubecl specializes+caches per unique combination, so NO 128-way host match is needed while still honoring the D-02 comptime fan-out"
  - "The device scatter uses the GLOBAL exclusive/inclusive scan (D-04 order-equivalence to a plain stable partition) rather than a per-block block-tiled scatter — bit-identical output, reuses the 18-01 primitives directly, and the exclusive in-block rank IS the [tid-1] inclusive derivation (Pitfall 2)"
  - "Scatter scratch / mark buffers are allocated per partition-call (not in a persistent driver `new`) because this plan delivers the partition OP, not the growth-loop driver — pre-allocation-once-outside-the-hot-loop (D-15) lands with that driver in a later plan; the functions are structured so a driver can hoist the scratch"

metrics:
  duration: 21min
  completed: 2026-07-01
status: complete
---

# Phase 18 Plan 02: §9 On-Device Data Partition (ODL-13) Summary

**The §9-faithful device `mark → prefix-sum → scatter` row router (numeric full-flag-fan-out + categorical membership), the cpu f64 stable-partition anchor, the 16-int SplitTreeStructure child-stats packet, and the HistArena leaf-indexed whole-pool swap — all three `partition_parity` cells + `histogram_arena::swap` green, byte-exact vs the golden, with the merge gate green (`LGBM_CUDA_ON_DEVICE` unset).**

## Performance
- **Duration:** ~21 min
- **Completed:** 2026-07-01
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- **Shared route decision (Task 1, D-02/D-03):** `route_to_left` transcribes the VERBATIM `SplitInner` full flag fan-out (`SplitRouteFanout`) as a `pub(crate) #[cube]` fn taking the seven comptime flags (MIN_IS_MAX, MISSING_IS_ZERO/NA, MFB_IS_ZERO/NA, MAX_TO_LEFT via the derived targets, USE_MIN_BIN, plus BIN_TYPE via the `<B: Int>` mark monomorph), using **branchless `select` stores only** (SP-2). `find_in_bitset` + `route_to_left_categorical` add the D-03 membership route with the preserved `pos/32 >= n → 0` guard (T-18-03). Both are `pub(crate)` — the single source the 18-04 predict walk will call (Pitfall 4).
- **Mark + scatter (Tasks 1–2):** `gen_data_to_left_kernel<B>` / `_categorical_kernel<B>` do the per-row native-width (u8/u16/u32) mark; `split_inner_scatter_kernel` does the stable scatter deriving the exclusive left rank from the inclusive scan `[tid-1]` (Pitfall 2) with the `i < n` guard (T-18-01); `update_data_index_to_leaf_kernel` writes the row→leaf map consuming `right_leaf_index` (the §1/§10 ordering invariant, Pitfall 3). The device fold reuses the 18-01 `prefix_sum_exclusive_u32_on` (AggregateBlockOffset) + `prefix_sum_inclusive_u16_on` (PrepareOffset) primitives.
- **cpu f64 anchor (D-04 CONFIRMED):** `partition_leaf_stable` / `partition_categorical_stable` are plain-Rust route mirrors + a stable partition (left-keepers in original order, then right-keepers), never launching a kernel (D-12). Hand-verified against 5 goldens (numeric basic/missing_zero/min_eq_max + cat_onehot/cat_oor_default) before coding.
- **16-int packet (D-08):** `split_tree_structure_packet` packs the 8 ints + 4 f64 (per-side sums from `SplitScalars`) with the `left_num < right_num` smaller/larger branch.
- **HistArena swap (Task 3, D-09):** a leaf-index → slot-handle table + `swap(parent_leaf, left_leaf, right_leaf, smaller_is_left)` reassigning INDICES only (zero `client.empty`) — the larger child inherits the parent slot in-place for the `parent − smaller` subtraction-trick reuse, the smaller takes a fresh non-aliasing slot. `rotate()` untouched (D-09).
- **Parity green:** `partition_parity` order (12 PCASE fan-out cases) + cat (3 PCAT) + packet (2 PPACKET) un-ignored and driven live — both the anchor AND the device fold compared byte-exact to the golden via `compare_exact_u32` / `compare_exact_f64_bits`. `histogram_arena::swap` round-trip green on the cpu f64 anchor (never GPU-vs-GPU).

## Task Commits
1. **Tasks 1–2 + categorical routing (data_partition.rs + partition_parity.rs)** — `38da26a` (feat)
2. **Task 3 HistArena leaf-indexed swap (histogram_arena.rs)** — `a634a48` (feat)

## Deviations from Plan

### Deliberate scoping / simplifications (no user permission needed — Rule 3 class)

**1. [Scoping] Route flags are comptime bools passed as plain host-computed values (no 128-way match)**
- **Where:** `route_to_left` / `gen_data_to_left_kernel`.
- **What:** The seven flags are `#[comptime] bool` params, but the host driver computes them from the runtime split params and passes them as plain `bool` at the `launch::<B,R>(...)` call site. cubecl specializes + caches a kernel per unique flag combination automatically, so the D-02 comptime fan-out is honored WITHOUT an unwieldy host `match` over 128 monomorphs.

**2. [Scoping] Device scatter uses the global scan (D-04 order-equivalence), not a per-block block-tiled scatter**
- **What:** Because D-04 CONFIRMED the reference block-tiled scatter is order-equivalent to a plain stable partition, the device fold runs the GLOBAL `prefix_sum_exclusive_u32_on` (AggregateBlockOffset class) + `prefix_sum_inclusive_u16_on` (PrepareOffset class) and scatters with `excl[i]` (= the `[tid-1]` exclusive left rank). Bit-identical output, directly reuses the 18-01 primitives, and preserves the Pitfall-2 inclusive↔exclusive relation (asserted in a debug cross-check).

**3. [Scoping, D-15] Scatter/mark scratch allocated per partition-call, not in a persistent driver `new`**
- **What:** This plan delivers the partition OP; there is no growth-loop hot loop yet (that driver is a later plan). The three `client.empty` sites (scatter `h_out`, two mark `h_to_left`) allocate per call. Pre-allocating "once outside the hot loop" (D-15) requires that loop to exist — the functions are structured so the future driver hoists the scratch. The 16-int packet is a host struct (no device alloc). Consistent with the shipped host-gather `partition.rs`, which also allocates per call.

**Total deviations:** 3 deliberate scoping decisions, 0 bugs. No architectural changes (Rule 4 not triggered).

## Verification
- `cargo test -p oracle-harness --test partition_parity` — 3/3 (order + cat + packet) green vs the golden.
- `cargo test -p lgbm-compute --lib data_partition` — 7/7; `histogram_arena` — 13/13 (incl. the 3 new swap tests).
- `cargo test --workspace` — GREEN with `LGBM_CUDA_ON_DEVICE` unset (ODL-19 merge gate).
- clippy clean on all new code (`data_partition.rs`, `histogram_arena.rs`, `partition_parity.rs`).
- Grep invariants: `partition.rs` and `HistArena::rotate` untouched (D-01/D-09); no f64 in the mark/scatter kernels (D-14 — f64 confined to the packet's scalar sum fields + the host anchor); no `LightGBM/` changes.
- hip parity: the partition output is an integer permutation (no f32 divergence), fully covered by the cpu f64 fold golden + the 18-01 scan-lowering cell — no separate `--features hip` cell owed (per plan `<verification>`).

## Known Stubs
None — the numeric + categorical device paths are fully wired to the goldens; no placeholder/empty-data sinks.

## Next Phase Readiness
- **18-04 (Wave 2, ODL-15):** `route_to_left` + `find_in_bitset` are `pub(crate)` and ready to share with the predict tree-walk (Pitfall 4 — one route transcription). The 16-int packet + row→leaf map are available for the driver integration.
- ODL-13 device data-partition is complete: the §9 mark→prefix-sum→scatter reproduces the reference stable partition bit-for-bit across the full numeric flag fan-out and categorical membership; the 16-int packet matches; the HistArena leaf-indexed swap makes the subtraction-trick reuse correct.

---
*Phase: 18-on-device-data-partition-tree-mutation-prediction*
*Completed: 2026-07-01*

## Self-Check: PASSED
All 3 modified files exist on disk + the SUMMARY; both task commits (38da26a, a634a48) present in git history.
