---
title: Device-Resident Grow-Loop Sync Deferral (Rank 2 → Rank 1 → Rank 3, Kaggle-validated)
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-14T00:00:00Z
source_requirements:
  - "User (this session, REVISED 2026-07-14): STOP after Wave 2 — bank the
     Rank-1 sync-halving win (incremental Rank 2 → Rank 1), and DROP Rank 3
     (the fully device-resident loop) entirely. Supersedes the earlier
     same-session 'target the full Rank-3 loop' framing; see §1."
  - ".planning/plans/device-resident-grow-loop/research.md (this phase's
     research, read in full)"
  - "docs/cuda-kernel-design.md (C++ CUDA reference prior art, §6/§907-935)"
  - "CLAUDE.md / AGENTS.md (numeric contract, dependency-first rule)"
  - ".planning/PROJECT.md (Key Decisions, Out of Scope — SEE §9 R-0 CONFLICT)"
---

> **PageIndex note:** PageIndex MCP was unreachable this session (connection
> error on `get_folder_structure`), consistent with the precedent already
> recorded in `.planning/plans/parity-gap-closure/SPEC.md` (PageIndex library
> empty / no target document for this workspace). This SPEC is therefore
> staged **locally** as the authoritative draft. **Pending PageIndex update:**
> upsert this file into the project's PageIndex collection once a
> collection/document id exists. No id was invented.

> **Reconciliation note (2026-07-14).** After this SPEC was first drafted
> (2026-07-13), the phase `research.md` was independently **re-verified against
> source and restructured** (now §1 Context … §10 Confidence). Every
> load-bearing claim this SPEC cites was re-confirmed at the file:line level;
> the `research §N` citations below therefore reference the phase's *evidence*
> (which stands), but the *section numbering* now maps into the reorganized
> research.md as: grow-loop/sync mechanics → research §2; pick/read_leaf
> deep-dive → §3; the 23% provenance → §4; parity + opt-in-flag pattern → §5;
> test/perf-measurement surface → §6; prior decisions & the CUDA-graph
> contradiction → §7; open questions → §8. Two findings were **sharpened** by
> the re-verification and are now first-class gates here (see §9 R-3 and the
> Wave-4 free-run-vs-drain pre-check in PLAN.md T-11): (a) the "~23%" is a **local 8-CU APU drain
> bucket that bundles device-compute time with the readback**, so the
> sync-deferrable share is **strictly less than 23%** and its transfer to P100
> is unproven — a **free-run-vs-drain A/B decomposition on real hardware is
> mandatory before Wave 3 commits**; (b) §9 R-0's PROJECT.md conflict is
> **confirmed verbatim** (`.planning/PROJECT.md:70` Out-of-Scope + `:115` Key
> Decision "Real-CUDA A/B found the fully-resident path 1.12–2.2× slower").
> **Scope trimmed (2026-07-14, same session):** with the confirmed
> `PROJECT.md:115` "1.12–2.2× slower" prior measurement in view, the user
> **dropped Rank 3 (the fully device-resident loop) and chose to stop after
> Wave 2** — banking the Rank-1 sync-halving win, which is a behavior-
> preserving optimization of the *existing* resident arm that does **not**
> touch the shelved fully-resident capability. This **dissolves the §9 R-0
> PROJECT.md conflict** (Waves 1–2 are not the Out-of-Scope capability). The
> "P100 verdict before shipping the perf change" gate is retained as Wave 4,
> re-scoped to validate the final Wave 1→2 chain. All Wave-3 (Rank-3)
> material below is retained only as struck **[DROPPED 2026-07-14]** context
> for a possible future phase; it is NOT part of this plan's acceptance.

# 1. Context

`lightgbm_rs`'s device-resident grow loop
(`grow_tree_on_device_resident<B, R>`,
`crates/lgbm-compute/src/kernels/grow_driver.rs:2329-3250`) grows one tree's
best-first leaf-wise structure entirely via device-resident histograms and a
device-resident best-split frontier (`DeviceFrontier<R>`, `lib.rs:511-622`),
but still performs **two blocking device→host readbacks per split** inside
the `for _split in 0..(num_leaves-1)` loop (`grow_driver.rs:2677-3213`):

1. **`pick`** (`frontier_pick_best_leaf_device` → `PickExport`,
   `grow_driver.rs:2700-2712`, `best_split.rs:2296-2423`) — the cross-leaf
   argmax winner (~8 `i64` + 10 `f64`), consumed for 2 genuine loop-control
   decisions (`best_leaf < 0` stop, `!(gain>0.0)` stop) plus a batch of
   pure launch-parameter values.
2. **`read_leaf`** (`DeviceLeafSplits::read_leaf`, `partition.rs:393-407`,
   called at `grow_driver.rs:2846` on the default resident-perm partition
   arm) — the just-partitioned parent's 6-int child ranges, consumed for
   `smaller_is_left` slot/role bookkeeping (real host branch,
   `grow_driver.rs:2928`) and the next build's row-range view offsets.

This is the machine-asserted **`2*num_leaves`** sync closed form on the
default resident-perm arm
(`crates/lgbm-compute/tests/on_device_sync_count.rs:222-230`,
`analytic_rp = 2 * NUM_LEAVES` asserted exactly at line 306-325) — measured
on real gfx1152 to be ~23% of drained wall time (pick ~13% + partition ~10%,
`[PROJECT: memory/local-rocm-gpu.md]`, `[PROJECT:
memory/ondevice-perf-campaign.md]`).

The research (`.planning/plans/device-resident-grow-loop/research.md`, read
in full — §1, §4-§10, §16, §18 are load-bearing here) ranked four
sub-approaches and recommended **Rank 1** (fuse `read_leaf(i)` +
`pick(i+1)` into one batched readback) as the first, lowest-risk target,
explicitly flagging **Rank 3** (fully device-resident loop, host polls only
a stop flag) as a **user-decision item, not a first phase**
(research §10 Rank 3 "Recommendation"), because:
- it is **never exercised by the CPU f64-fold merge gate**
  (`grow_tree_on_device_resident` only runs when
  `backend.resident_pool_supported()==true`, which `CpuBackend` never
  returns — `grow_driver.rs:1135-1141`, `lib.rs:1275`/`:3567`), so bugs
  surface only on real GPU hardware;
- its payoff is **unproven** (research §10 Rank 3, §16 "Rank 3 scope
  creep" risk) — the drain ledger conflates syncable-away time with
  genuine device compute, and no A/B has measured the achievable residual;
- it is comparable in scale to the already-attempted CUDA-graph
  `vendor/cubecl-cuda` fork effort (`[[cudagraph-campaign]]`).

**Locked user decision (this session, REVISED — supersedes the original
Rank-3 target):** build the **incremental Rank 2 → Rank 1 sequence and STOP
after Wave 2.** Rank 3 (the fully device-resident loop) is **dropped** — the
user reviewed the confirmed `PROJECT.md:115` prior measurement (real-CUDA A/B
found the fully-resident path **1.12–2.2× slower**, per-leaf sync floor) and
chose to bank the lower-risk Rank-1 sync-halving win rather than bet an
incremental redesign can overturn a measured net-negative. This SPEC and its
companion PLAN.md therefore build **Waves 0 → 1 → 2 → 4** (Wave 4 = the P100
perf verdict for the Wave 1→2 chain). **Consequence:** the earlier §9 R-0
conflict with `.planning/PROJECT.md`'s Out-of-Scope entry is **RESOLVED by
this scope trim** — Waves 1–2 are behavior-preserving optimizations of the
*existing* resident arm and are **not** the shelved "fully GPU-resident
(no-host-round-trip) best-first grow loop." The Wave-3 (Rank-3) specs are
retained below only as **[DROPPED 2026-07-14]**-marked reference for a
possible future phase.

# 2. Scope and Non-goals

**In scope (this SPEC, all three ranks, ordered):**
- **Wave 0**: commit the pre-existing uncommitted parallel-prefix-scan diff
  (`split.rs`, `scan_pargain_parity.rs`, `phase_prof.rs`,
  `rocm_drain_profile.rs`) as its own commit, unmodified, BEFORE any new
  grow-loop edit (locked decision #2, AGENTS.md dependency-first rule).
- **Wave 1 (Rank 2)**: widen `DeviceLeafSplits` to a per-split append-only
  buffer (locked decision #3); add an on-device smaller/larger role-
  assignment kernel (mirrors C++ `SplitTreeStructureKernel`,
  `docs/cuda-kernel-design.md:907-914`); remove the host branch on raw
  `read_leaf` counts.
- **Wave 2 (Rank 1)**: fixed-worst-case-grid + device early-exit
  build/subtract/scan kernel variants; batch `read_leaf(i)` +
  `pick(i+1)` into one `client.read(Vec<Handle>)` call; re-derive the
  sync closed form.
- **~~Wave 3 (Rank 3 extension)~~ [DROPPED 2026-07-14]**: moving per-leaf
  bookkeeping device-resident + a fixed host schedule polling only a stop
  flag. Removed from scope by the revised user decision (§1); the fully
  device-resident loop stays `PROJECT.md`-Out-of-Scope. Its specs
  (SPEC-DRGL-07..10) remain below marked **[DROPPED]** for future reference.
- **Wave 4 (perf verdict for the Wave 1→2 chain)**: a Kaggle CUDA/P100 perf
  validation run (retained "P100 verdict before shipping" gate),
  order-alternated warm-median-of-3 A/B, counts-ledger proof, preds
  bit-identical or within the documented envelope — mirrors the
  established `[[kaggle-bench-workflow]]` protocol used throughout
  `[[ondevice-perf-campaign]]`. Validates the Wave-1→2 (Rank-1) chain vs the
  pre-phase baseline, NOT a Rank-3 arm.

**Non-goals (explicit):**
- **The fully device-resident (Rank 3) grow loop** — dropped this session
  (§1); it remains `PROJECT.md`-Out-of-Scope with a prior 1.12–2.2×-slower
  real-CUDA measurement. No spec in this plan's *acceptance* (DRGL-00..06,
  DRGL-11) moves per-leaf bookkeeping device-resident or collapses the host
  loop to a fixed stop-flag-polled schedule.
- A true **zero-sync / persistent-kernel** loop — research §8/§11 (finding
  3 in §1) establishes cubecl 0.10 has no device-indirect-dispatch
  primitive; even the C++ CUDA reference crosses scalars back to the host
  per iteration (`docs/cuda-kernel-design.md:196-198`). Rank 3 as specified
  here still issues a host-driven fixed schedule of launches; it is NOT a
  literal zero-crossing loop.
- Changing leaf-wise growth to level-wise/batched-N-splits growth —
  research §9 rules this out as a correctness/parity change, not a perf
  lever. No spec in this SPEC batches future splits' PICK decisions.
- CUDA-Graph capture/replay (`vendor/cubecl-cuda/`) — a separate,
  already-concluded (~1.04×, not-worth-it) lever
  (`[[cudagraph-campaign]]`); out of scope here.
- Distributed/MPI, C-API, OpenCL `gpu` device knobs — project-wide out of
  scope (`[PROJECT: CLAUDE.md]`).
- Any change to the CPU f64-fold anchor's own code path
  (`grow_tree_on_device_driver_with_cfg`'s non-resident inline loop) — the
  anchor never executes `grow_tree_on_device_resident` and this SPEC does
  not touch the anchor's dispatch guard.
- Categorical-feature GPU kernels (`grow_driver.rs:1135-1141`'s
  `!features.iter().any(Categorical)` gate already routes categorical
  grows away from the resident arm) — unaffected, untouched.

# 3. Dependencies

**AGENTS.md rule — dependencies confirmed FIRST:**
- **No new external crate is required for any wave.** `[VERIFIED: research
  §11 "No new external crate is required for Rank 1 or Rank 2"; Rank 3
  stays within `#[cube]` code + existing `Backend` trait extension
  patterns per §11's own hedge]`.
- `cubecl = "0.10.0"` (workspace dep, `Cargo.lock` confirms;
  `[VERIFIED: LOCAL Cargo.toml:25]`) — `client.read(Vec<Handle>)` batched
  multi-handle read already exists and is the exact mechanism Wave 2 needs
  (`Backend::read_batched`, tested `grow_driver.rs:3317-3358`,
  `supports_async_device_copy()==true`,
  `supports_multi_stream_overlap()==false`).
- `thiserror` `2.0.18` (workspace) — `ComputeError::Runtime` is the
  existing typed-error pattern any new fallible device call (widened
  `DeviceLeafSplits`, new `Backend` trait methods) must follow, mirroring
  `DeviceLeafSplits::new`'s existing `num_leaves == 0` guard
  (`partition.rs:359-364`).
- Reused symbols: `DeviceFrontier<R>` (`lib.rs:511-622`), `SplitSoa`
  (`best_split.rs`), `DeviceLeafSplits<R>` (`partition.rs:345-408`),
  `ChildRanges`/`LEAF_SPLIT_STRIDE` (`partition.rs:319,336`),
  `ResidentDriverLeaf` (`grow_driver.rs:1860-1882`, host-only today),
  `Backend` trait (`lib.rs:624-…`), the `OnceLock`-cached env-gate +
  `AtomicU8` same-session-override pattern
  (`resident_perm_partition_enabled`/`PARTITION_RESIDENT_OVERRIDE`,
  `grow_driver.rs:417-445`), the `GROW_*_NS` phase-profiling ledger +
  `bump_launch`/`bump_sync` counters (`grow_driver.rs:214-225`), the
  `COUNTS` dump line (`phase_prof.rs`, extended by the uncommitted Wave-0
  diff with `scan_parprefix`).
- **Do Not Hand-Roll** (research §15, carried forward verbatim): do not
  reimplement `client.read(Vec<Handle>)`; do not reimplement the
  `split_gt` tie-break rule (`best_split.rs:2330-2354`) — any Wave-1/3
  device kernel must call into or exactly mirror it; do not build a new
  device-side reduction/argmax primitive — extend `DeviceFrontier`/
  `SplitSoa`, not a parallel family; do not add a new env-gate mechanism —
  reuse the `OnceLock`+`AtomicU8` template; do not reimplement the
  phase-profiling/counts-tripwire ledger — extend it (a new
  `read_leaf_deferred=`/`rank3_schedule=` tripwire per wave, matching the
  `scan_parprefix=` precedent).

**C++ reference dependency (partially available):** `docs/cuda-kernel-design.md`
was read in full for the relevant sections (§1-2, §6, §8-9,
`:155-232`, `:907-935`) and is the authoritative porting reference for this
phase (unlike the parity-gap-closure phase, the raw `LightGBM/` C++ tree is
also absent from this sandbox — `[VERIFIED: LOCAL ls -d LightGBM → not
found]` — so `docs/cuda-kernel-design.md`'s own `[PROJECT: ...]`-labeled
claims are the best available evidence, not independently re-verified
against `LightGBM/src/treelearner/cuda/`).

# 4. Typed Contracts

```rust
// ============================================================
// Wave 1 (Rank 2) — proposed widened DeviceLeafSplits + role-assignment
// ============================================================

// crates/lgbm-compute/src/kernels/partition.rs
// REPLACES the existing per-leaf-id `DeviceLeafSplits` (partition.rs:345-408)
// with a per-SPLIT, append-only, non-overwriting layout (locked decision #3
// — NOT a generation/consumed-guard on the existing per-leaf-id buffer).
pub struct DeviceLeafSplits<R: cubecl::Runtime> {
    /// `LEAF_SPLIT_STRIDE * capacity` i32 cells; split index `s` (0-based,
    /// assigned in split-loop iteration order) owns `[STRIDE*s, STRIDE*s+STRIDE)`.
    /// `capacity == 2*(num_leaves-1)` (proposed: existing width doubled — one
    /// record per split rather than one per leaf-id; sized so every split in a
    /// full grow gets a distinct, never-recycled slot — see SPEC-DRGL-01).
    ranges: Handle,
    /// Growing count of splits recorded so far this grow (host-tracked; NOT
    /// read back — used only to size the next write index).
    next_split_idx: usize,
    capacity: usize,
    _runtime: PhantomData<fn() -> R>,
}
impl<R: cubecl::Runtime> DeviceLeafSplits<R> {
    /// `capacity` replaces today's `num_leaves` sizing argument.
    pub fn new(client: &ComputeClient<R>, capacity: usize) -> Result<Self, ComputeError>;
    /// Proposed: write this split's 6-int child ranges into split index
    /// `self.next_split_idx` (device kernel or host-driven single-cell write,
    /// design TBD in SPEC-DRGL-01 Red step), then increment. Distinct from
    /// today's `write_leaf`-by-`leaf_id` (partition.rs:410) which OVERWRITES.
    pub fn record_split(&mut self, client: &ComputeClient<R>, /* ... */) -> Result<(), ComputeError>;
    /// Read split index `s`'s child ranges (replaces `read_leaf(leaf_id)`).
    /// Panics if `s >= self.next_split_idx`.
    pub fn read_split(&self, client: &ComputeClient<R>, s: usize) -> ChildRanges;
}

// crates/lgbm-compute/src/kernels/partition.rs (or best_split.rs — co-located
// with the existing device-resident reduce family per "Do Not Hand-Roll")
// proposed: new #[cube] kernel + host launcher, mirrors C++
// SplitTreeStructureKernel's role-assignment (docs/cuda-kernel-design.md:907-914).
/// Resolves `smaller_is_left` (pure integer comparison: `left_count <
/// right_count`) from the already-resident child-range record and writes the
/// resolved `smaller_leaf`/`larger_leaf`/`smaller_slot`/`larger_slot`
/// assignment into a resident output cell — the host no longer branches on
/// raw counts for role bookkeeping (SPEC-DRGL-02/03).
pub fn assign_smaller_larger_roles_device<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    leaf_splits: &DeviceLeafSplits<R>,
    split_idx: usize,
    next_slot: i32,
    parent_slot: i32,
) -> Result<(), ComputeError>;

// ============================================================
// Wave 2 (Rank 1) — proposed fixed-grid Backend trait extension
// ============================================================

// crates/lgbm-compute/src/lib.rs, `Backend` trait — CpuBackend-stub pattern
// (mirrors build_resident_leaf_rows_handle, lib.rs:1127-1145: real-device-only,
// default body returns ComputeError::Runtime).
pub trait Backend {
    // ... existing methods unchanged ...
    /// Proposed: launch the smaller-child build/subtract/scan with a grid
    /// sized off `parent_row_count_upper_bound` (a safe upper bound for
    /// either child, host-known with no new sync) instead of the exact
    /// child count, with a device-side early-exit reading the real count
    /// from `leaf_splits`'s resident record. Real-device-only
    /// (`<R as Runtime>::name(client) != "cpu"` gated, following the
    /// `partition_bc_fused` SMEM-fusion precedent, `partition.rs:1302-1305`).
    fn build_subtract_scan_fixed_grid_into_frontier(
        &self,
        client: &ComputeClient<Self::Runtime>,
        parent_slot: usize,
        smaller_slot: usize,
        larger_slot: usize,
        slot_len: usize,
        parent_row_count_upper_bound: i32,
        leaf_splits: &kernels::partition::DeviceLeafSplits<Self::Runtime>,
        split_idx: usize,
        feats: &[kernels::split::BatchedSplitFeature],
        real_feats: &[i32],
        cfg: &GainConfig,
        frontier: &DeviceFrontier<Self::Runtime>,
        smaller_leaf: usize,
        larger_leaf: usize,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::Runtime {
            detail: "build_subtract_scan_fixed_grid_into_frontier: backend has no \
                     fixed-grid resident build (real-device-only)".to_string(),
        })
    }
}

// crates/lgbm-compute/src/kernels/grow_driver.rs — batched readback fusion
// (proposed helper, replaces the two separate `bump_sync` sites at
// grow_driver.rs:2701 and :2845):
/// Issues ONE `client.read(vec![leaf_splits_handle, frontier_pick_handle])`
/// per iteration, returning BOTH the deferred `read_split(i-1)` result and
/// the current iteration's `PickExport`. Called once per loop iteration
/// (the LAST split's deferred `read_split` folds into the grow-tail readback
/// instead, per research §5 "read_leaf's host use").
fn read_deferred_split_and_pick<B, R>(
    backend: &B,
    client: &ComputeClient<R>,
    leaf_splits: &kernels::partition::DeviceLeafSplits<R>,
    deferred_split_idx: Option<usize>,
    frontier: &DeviceFrontier<R>,
    prev_smaller: i32,
    prev_larger: i32,
    cur_num_leaves: usize,
) -> Result<(Option<ChildRanges>, kernels::best_split::PickExport), ComputeError>;

// ============================================================
// Wave 3 (Rank 3) typed contracts — [DROPPED 2026-07-14]
// The device-resident DeviceFrontier extension (DeviceLeafState) and the
// `grow_tree_on_device_resident_rank3` fixed-schedule entry point are NO
// LONGER part of this plan (§1 scope trim). Their contracts are omitted
// here; see SPEC-DRGL-07..10 below (retained, [DROPPED]-marked) for the
// prose design should a future phase revive them.
// ============================================================
```

# 5. Failure-Isolated Behavioral Specifications

Each spec has ONE behavioral responsibility with one primary failure cause.
Status: **draft**, implementation state: **unimplemented** for all. IDs are
stable and referenced by PLAN.md tasks.

---

## Wave 0 — Prior-work hygiene (procedural)

### SPEC-DRGL-00 — Commit the uncommitted parprefix/rocm-drain-profile diff as its own prior commit
- **status:** draft
- **implementation_state:** unimplemented (this spec's "implementation" is a
  git commit, not new production code — no red/green test, just a
  procedural verification)
- **principal failure reason:** the uncommitted diff (`split.rs`
  `+752/-0`, `scan_pargain_parity.rs` `+258/-0`, `phase_prof.rs` `+6/-3`,
  new `rocm_drain_profile.rs`) is co-mingled with subsequent Wave 1-2
  edits, making git history dishonest and risking merge conflicts in the
  frequently-touched `split.rs` (AGENTS.md dependency-first rule).
- **scope:** repository git history only; NO content change to the diff.
- **dependencies:**
  - `git status --porcelain` / `git diff --stat` — confirms the diff's
    exact current shape before committing `[VERIFIED: LOCAL, this session,
    matches research §3 exactly]`.
- **input type:** the current uncommitted working-tree diff (4 files:
  `crates/lgbm-compute/src/kernels/split.rs`,
  `crates/lgbm-compute/tests/scan_pargain_parity.rs`,
  `crates/lgbm-treelearner/src/phase_prof.rs`,
  `crates/lgbm-treelearner/examples/rocm_drain_profile.rs`).
- **output type:** one new git commit on the current branch containing
  exactly these 4 files, unmodified from their current working-tree
  content.
- **preconditions:** `git status --porcelain` matches the 4-file diff
  documented in research §3 (no drift since research was written).
- **behavior:**
  - *Given* the current uncommitted 4-file diff, *when* the commit task
    runs, *then* `git log -1 --stat` shows exactly these 4 files with the
    same insertion/deletion counts as the pre-commit `git diff --stat`.
- **postconditions:** `git status --porcelain` no longer lists these 4
  files as modified/untracked; `.planning/plans/device-resident-grow-loop/`
  itself and `planning/`/`vendor/` (unrelated, per research §3) remain
  untouched/uncommitted.
- **errors:** none (procedural git operation; abort and re-verify if
  `git status` has drifted from the documented 4-file shape before
  committing).
- **side effects:** one new commit; no working-tree content change.
- **acceptance examples:**
  - Given the working tree exactly matches `git status --porcelain` output
    captured in research §3, when `git add` + `git commit` runs scoped to
    the 4 files, then `git show --stat HEAD` lists exactly those 4 files
    with unchanged line counts.
- **evidence:**
  - `git status --porcelain` (this session) — matches research §3's
    4-file list exactly.
  - `[PROJECT: research.md §3, §18 "Suggested ordering" item (1)]`
- **non-goals:** touching the diff's content; committing `planning/` or
  `vendor/` (both unrelated per research §3).

---

## Wave 1 — Rank 2: on-device role assignment (prerequisite, low-risk stepping stone)

### SPEC-DRGL-01 — `DeviceLeafSplits` widened to a per-split, non-overwriting, append-only buffer
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** deferring `read_leaf`'s consumption past
  the point where a leaf id (`new_left`) is picked again silently reads
  STALE/overwritten child-range data on the CURRENT per-leaf-id buffer
  (research §16 "`DeviceLeafSplits` overwrite race" risk row) — the
  widened layout is what makes ANY later deferral (Wave 2) safe.
- **scope:** `crates/lgbm-compute/src/kernels/partition.rs`
  (`DeviceLeafSplits<R>`, `LEAF_SPLIT_STRIDE`, `ChildRanges`,
  `read_leaf`/`write_leaf`).
- **dependencies:**
  - `LEAF_SPLIT_STRIDE = 6` (`partition.rs:336`) — reused stride per
    record.
  - `ComputeError::Runtime` — existing typed-error pattern for the
    `capacity == 0` guard (mirrors `partition.rs:359-364`).
  - 5 existing callers of `DeviceLeafSplits`/`read_leaf`/`write_leaf`
    (`grow_driver.rs`, per CodeGraph blast-radius) — all must be re-audited
    for the `leaf_id`→`split_idx` addressing change.
- **input type:** `capacity: usize` (proposed: `2*(num_leaves-1)`,
  locked decision #3 — per-split, NOT per-leaf-id sizing).
- **output type:** `DeviceLeafSplits<R>` (widened struct, §4 typed
  contract) exposing `record_split`/`read_split` in place of today's
  `write_leaf`/`read_leaf`.
- **preconditions:** `capacity >= 1` (mirrors today's `num_leaves >= 1`
  guard).
- **behavior:**
  - Every split's child-range record is written to a **fresh, never-reused**
    slot (`next_split_idx`, monotonically incrementing), never overwriting
    an earlier split's record.
- **postconditions:** for any two splits `i != j` recorded in the same
  grow, `read_split(i)` and `read_split(j)` return independently correct,
  non-aliased data even if both splits' `new_left` happened to reuse the
  same leaf id.
- **errors:**
  - `ComputeError::Runtime` when `capacity == 0` (mirrors existing `new`
    guard).
  - Panic (mirrors today's `assert!` on `read_leaf`) when
    `read_split(s)` is called with `s >= next_split_idx`.
- **side effects:** device allocation of `LEAF_SPLIT_STRIDE * capacity`
  i32 cells (was `LEAF_SPLIT_STRIDE * num_leaves`) — a strictly larger
  buffer, bounded and finite.
- **acceptance examples:**
  - Given a grow where leaf id `L` is `new_left` of split 2 and is picked
    again (becomes `best_leaf`) at split 5, when both splits' records are
    read via `read_split(2)` and `read_split(5)`, then both return the
    CORRECT (not aliased) data — a dedicated regression test intentionally
    constructs this re-pick scenario (research §16 verification column).
- **evidence:**
  - `[LOCAL partition.rs:342-408]` — current per-leaf-id struct + docs
    ("indices are reused/overwritten").
  - `[PROJECT: research.md §5 "read_leaf ... Conclusion", §16 row 1]`.
- **non-goals:** the on-device role-assignment kernel that WRITES into
  this buffer (SPEC-DRGL-02); the deferred-read consumption itself
  (SPEC-DRGL-05, Wave 2).

### SPEC-DRGL-02 — On-device smaller/larger role-assignment kernel (mirrors C++ `SplitTreeStructureKernel`)
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** today's `smaller_is_left = left_count <
  right_count` (`grow_driver.rs:2928`) is a HOST branch that requires the
  host to have already synchronously read `read_leaf`'s raw counts — this
  spec moves the pure-integer decision on-device so the host no longer
  needs the raw counts for role/slot bookkeeping, which is the
  prerequisite Wave 2 needs to defer the read (research §10 Rank 1
  mechanism (a); §18 "Rank 1 strictly depends on Rank 2's role-assignment
  kernel").
- **scope:** new `#[cube]` kernel + host launcher, co-located near the
  existing device-resident reduce family (`partition.rs` or
  `best_split.rs`, following the "Do Not Hand-Roll" precedent of extending
  the existing family rather than inventing a new one).
- **dependencies:**
  - `DeviceLeafSplits` (widened, SPEC-DRGL-01) — the resolved role
    assignment is written into (or alongside) this buffer's split record.
  - C++ prior art: `SplitTreeStructureKernel` "assigns smaller-vs-larger
    child roles and swaps the histogram-pool pointers"
    `[PROJECT: docs/cuda-kernel-design.md:907-914]`.
- **input type:** the just-partitioned split's `left_count`/`right_count`
  (already resident, from SPEC-DRGL-01's record), `next_slot: i32`,
  `parent_slot: i32` (host-known launch parameters, no new readback).
- **output type:** a resident role-assignment record —
  `(smaller_is_left: bool, smaller_slot: i32, larger_slot: i32)`,
  proposed encoded as 3 additional i32 cells appended to the per-split
  record (extends `LEAF_SPLIT_STRIDE` from 6 to 9, or a parallel
  same-indexed buffer — resolved in the Red step per research §19 Q3's
  "needs a short design spike" caveat, applied here to the role fields
  specifically since research did not spike this).
- **preconditions:** the split's `ChildRanges` record already exists
  (SPEC-DRGL-01's `record_split` has run for this split index).
- **behavior:**
  - *Given* `left_count < right_count`, *when* the role kernel runs,
    *then* it writes `smaller_is_left=true`, `smaller_slot=next_slot`,
    `larger_slot=parent_slot` — a pure integer comparison, bit-identical
    decision to today's host branch for every input (no FP reorder;
    research §7 "Deferring/fusing pick+read_leaf ... is a PURE
    host/device-orchestration change").
- **postconditions:** the role decision is available to a LATER host read
  (or a later kernel) without the host branching on raw counts.
- **errors:** `ComputeError::Runtime` on backend/device-dispatch failure
  (mirrors every other `#[cube]` kernel launcher in this crate).
- **side effects:** one additional device dispatch per split (real-device
  gated — a `bump_launch()` per the existing counting convention).
- **acceptance examples:**
  - Given `left_count=3, right_count=7`, when the role kernel runs, then
    the resident record decodes to `smaller_is_left=true`.
  - Given `left_count=7, right_count=3`, when the role kernel runs, then
    the resident record decodes to `smaller_is_left=false`.
  - Given `left_count == right_count` (tie), when the role kernel runs,
    then the decision matches today's host `<` (not `<=`) semantics
    exactly (`left_count < right_count` is `false` on a tie ⇒
    `smaller_is_left=false`) — a dedicated tie-case unit test, since a
    `<` vs `<=` mismatch would silently swap which child gets the fresh
    pool slot.
- **evidence:**
  - `[LOCAL grow_driver.rs:2925-2938]` — the current host role-assignment
    logic to be mirrored exactly.
  - `[PROJECT: docs/cuda-kernel-design.md:907-914]`.
- **non-goals:** the fixed-worst-case-grid build change (SPEC-DRGL-04);
  removing the host's synchronous `read_leaf`/`read_split` call itself
  (still synchronous in this Wave — only the ROLE branch moves on-device).

### SPEC-DRGL-03 — Host loop consumes the device-resolved role assignment (drops the raw-count host branch)
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** `grow_driver.rs:2925-2938`'s host
  `smaller_is_left`/slot-selection block must be replaced by reading the
  SPEC-DRGL-02 resolved record instead of recomputing the comparison from
  raw counts — a wiring change with exactly one behavior (source of the
  decision), independently testable from SPEC-DRGL-02's kernel
  correctness.
- **scope:** `crates/lgbm-compute/src/kernels/grow_driver.rs`
  (`grow_tree_on_device_resident`, `:2925-2938`).
- **dependencies:** SPEC-DRGL-01, SPEC-DRGL-02 (this spec cannot ship
  without both).
- **input type:** the resolved role record from SPEC-DRGL-02 (read via
  the SAME synchronous `read_split` call this Wave — the sync itself is
  NOT removed yet, only its consumption changes).
- **output type:** `(smaller_leaf, larger_leaf, smaller_slot,
  larger_slot, left_slot, right_slot)` — identical shape/semantics to
  today's host-computed tuple (`grow_driver.rs:2929-2938`), now SOURCED
  from the device record instead of recomputed.
- **preconditions:** SPEC-DRGL-01/02 are wired in ahead of this leaf's
  split iteration.
- **behavior:**
  - *Given* any split, *when* the host consumes the role record, *then*
    the resulting `(smaller_leaf, larger_leaf, smaller_slot,
    larger_slot)` tuple is bit-identical, for every input, to what
    today's `left_count < right_count` host branch would have produced.
- **postconditions:** `grow_driver.rs`'s `left_count < right_count`
  expression is deleted (dead code after this change — a regression
  guard for "no double-source of truth").
- **errors:** none new.
- **side effects:** none beyond SPEC-DRGL-02's.
- **acceptance examples:**
  - Given a fixed small corpus that currently grows a known N-leaf tree
    on the resident-perm arm, when this spec's change lands, then the
    grown tree (structure, leaf values, layout) is byte-identical to the
    pre-change tree — a byte-identity regression test following the
    `partition_bc_fusion_byte_identical_to_three_launch`
    (`resident_perm_partition.rs:88`) pattern.
- **evidence:**
  - `[LOCAL grow_driver.rs:2925-2938]`.
  - `[LOCAL resident_perm_partition.rs:1-25]` — the established
    byte-identity-gate idiom for this exact class of change ("device fold
    is pinned byte-equal to the host anchor").
- **non-goals:** removing the synchronous read itself (Wave 2,
  SPEC-DRGL-05).

---

## Wave 2 — Rank 1: fixed-grid builds + batched readback fusion

### SPEC-DRGL-04 — Fixed-worst-case-grid + device early-exit build/subtract/scan `Backend` trait variants
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** deferring `read_leaf`/`read_split`'s
  consumption to the NEXT iteration means the host no longer knows the
  child's EXACT row count at build-launch time — the launch grid must be
  sized off a safe UPPER bound (the already-host-known PARENT row count)
  with a device-side early-exit reading the real count from the resident
  buffer, and an off-by-one or wrong-field early-exit guard either
  double-processes rows (wrong histogram) or skips valid rows
  (undercounts) — research §16 "Fixed-worst-case-grid under-early-exit
  bug" risk row, the SINGLE highest-risk item in Wave 2.
- **scope:** new `Backend` trait method(s) (`lib.rs`, CpuBackend-stub
  pattern, real-device-only per `<R as Runtime>::name(client) != "cpu"`
  gating, mirroring `partition_bc_fused`, `partition.rs:1302-1305`);
  `RocmBackend`/`CudaBackend` implementors.
- **dependencies:**
  - SPEC-DRGL-01 (the widened buffer the early-exit reads from).
  - Existing `build_resident_leaf`/`subtract_resident`/
    `scan_resident_leaf_into_frontier` family (`lib.rs`,
    `grow_driver.rs:2986-3208`) — the fixed-grid variant must produce
    BYTE-IDENTICAL histograms to these, just launched with a different
    grid size.
- **input type:** `parent_row_count_upper_bound: i32` (host-known, no new
  sync) in place of the exact `s_count`/`l_count`.
- **output type:** identical device-resident histogram/frontier state to
  today's exact-grid build (§4 typed contract).
- **preconditions:** the widened `DeviceLeafSplits` record for this split
  is available (may be from THIS iteration if not yet deferred, or the
  PREVIOUS iteration once SPEC-DRGL-05 lands — this spec is
  grid-mechanism-only, deferral timing is SPEC-DRGL-05's concern).
- **behavior:**
  - *Given* a parent with `p_count` rows and a child with the true count
    `c_count <= p_count`, *when* the fixed-grid build launches with grid
    size `p_count` and a device early-exit reading `c_count` from the
    resident buffer, *then* the resulting histogram is byte-identical to
    the exact-`c_count`-grid build's histogram.
- **postconditions:** none beyond byte-identity.
- **errors:** `ComputeError::Runtime` on dispatch failure (existing
  pattern).
- **side effects:** none beyond the existing build/subtract/scan side
  effects (device histogram writes); grid over-provisioning has no
  observable effect beyond wasted (idle, early-exited) threads.
- **acceptance examples:**
  - Given a fixed corpus/split sequence, when the fixed-grid variant runs
    with real device hardware, then its output is byte-identical to the
    exact-grid variant's output for every split in the sequence — a
    dedicated `--features rocm` byte-identity test, following
    `partition_bc_fusion_byte_identical_to_three_launch`'s pattern
    (`resident_perm_partition.rs:88`).
- **evidence:**
  - `[LOCAL grow_driver.rs:2986-3208]` — the exact-grid reference
    behavior this must match byte-for-byte.
  - `[PROJECT: research.md §10 Rank 1 "Bit-exactness risk: LOW... only new
    concern is proving the fixed-worst-case-grid + early-exit kernels do
    EXACTLY the same per-row work as today's exact-sized grid"]`.
- **non-goals:** the batched readback itself (SPEC-DRGL-05); Wave 3's
  fixed schedule (a different, later grid-sizing concern for the WHOLE
  loop, not just one build).

### SPEC-DRGL-05 — Batched `read_split(i)` + `pick(i+1)` fusion into one `client.read` call
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** today's driver issues two SEPARATE
  blocking `client.read*` calls per split (`bump_sync` at
  `grow_driver.rs:2701` for pick, `:2845` for read_leaf) — this spec
  replaces them with ONE `client.read(vec![leaf_splits_handle,
  frontier_pick_handle])` batched call per iteration, deferring the
  PREVIOUS split's child-range consumption to coincide with the CURRENT
  split's pick readback (research §10 Rank 1 mechanism (c); §5 "read_leaf
  ... Conclusion").
- **scope:** `crates/lgbm-compute/src/kernels/grow_driver.rs`
  (`grow_tree_on_device_resident`'s per-split loop, `:2677-3213`) — the
  `read_deferred_split_and_pick` helper (§4 typed contract), **gated behind
  a new opt-in `LGBM_GROW_DEFER_SYNC` env flag (default OFF)** following the
  exact `OnceLock`+`AtomicU8`-override template
  (`resident_perm_partition_enabled`, `grow_driver.rs:417-445`). This is
  the plan's terminal perf-changing deliverable; per the locked "P100
  verdict before default-ON" decision, it ships OFF and is only flipped
  default-ON by a separate follow-up commit after SPEC-DRGL-11's P100 A/B.
  When the flag is OFF, the driver takes today's two-separate-reads path
  unchanged.
- **dependencies:**
  - SPEC-DRGL-01..04 (the widened buffer, on-device role assignment, and
    fixed-grid builds must all be in place — a deferred read is only
    safe once the buffer no longer overwrites, per SPEC-DRGL-01's own
    postcondition).
  - `Backend::read_batched` / `supports_async_device_copy()`
    (`grow_driver.rs:3317-3358`) — the existing batched-multi-handle-read
    contract this spec's helper calls into (not reimplements, per "Do Not
    Hand-Roll").
- **input type:** `deferred_split_idx: Option<usize>` (the PREVIOUS
  split's index, `None` for the root iteration) + the current iteration's
  pick inputs (`prev_smaller`, `prev_larger`, `cur_num_leaves`).
- **output type:** `(Option<ChildRanges>, PickExport)` — both values from
  ONE readback.
- **preconditions:** the previous split's `record_split` (SPEC-DRGL-01)
  and role-assignment (SPEC-DRGL-02) have completed device-side before
  this iteration's combined read (device-side ordering, no new sync).
- **behavior:**
  - *Given* split `i`'s child-range record and split `i+1`'s pick
    decision are both device-resident, *when* the combined read runs,
    *then* exactly ONE blocking `client.read` call retrieves both, and
    the returned `ChildRanges`/`PickExport` values are bit-identical to
    what two SEPARATE reads would have returned for the same device
    state.
  - *Given* the LAST split in a grow, *when* there is no next iteration's
    pick to fuse with, *then* its `read_split` folds into the existing
    grow-tail perm readback (`grow_driver.rs:3224-3227`) instead — no
    orphaned split ever goes unread.
- **postconditions:** the per-split sync count on the resident-perm arm
  drops from `2*num_leaves` to the Rank-1 closed form re-derived in
  SPEC-DRGL-06 (`~num_leaves + O(1)`, per research §10 Rank 1 "removes
  roughly HALF the per-split sync count").
- **errors:** `ComputeError::Runtime` on the batched read failing
  (existing `Backend::read_batched` error surface).
- **side effects:** none beyond the removed/reordered sync itself — no
  new device state.
- **acceptance examples:**
  - Given a grow that currently produces a byte-identical tree with 2
    separate reads/split, when the batched-fusion change lands, then the
    SAME corpus grows a byte-identical tree with 1 combined read/split
    (minus the root/tail edge iterations) — real-device (`--features
    rocm`) regression test.
  - Given a leaf id `L` that is `new_left` of split `i` and is picked
    again at split `i+1` (the SPEC-DRGL-01 re-pick scenario), when the
    combined read for iteration `i+1` runs, then it correctly retrieves
    BOTH split `i`'s deferred child-range AND split `i+1`'s pick,
    unaliased.
- **evidence:**
  - `[LOCAL grow_driver.rs:2700-2712,2843-2846,3317-3358]`.
  - `[PROJECT: research.md §5, §10 Rank 1 mechanism (c)]`.
- **non-goals:** any FP reorder (explicitly excluded — research §7 "no
  floating-point value is recomputed or reordered... only WHEN the host
  reads them back changes").

### SPEC-DRGL-06 — Rank-1 sync-count closed-form re-derivation (both `on_device_sync_count.rs` files)
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** the machine-asserted closed forms in
  `crates/lgbm-compute/tests/on_device_sync_count.rs:222-230,306-325`
  (`analytic_rp = 2 * NUM_LEAVES`, asserted EXACT) and the stale copy in
  `crates/oracle-harness/tests/on_device_sync_count.rs`
  (`COLLAPSE_NUM_LEAVES`-based, per research §4's noted "closed-form
  drift" pattern) both go STALE the moment SPEC-DRGL-05 changes the sync
  pattern — a single wrong or unchanged constant either false-fails the
  suite or (worse) silently stops proving what it claims (research §16
  "Closed-form sync-count drift" risk row).
- **scope:** `crates/lgbm-compute/tests/on_device_sync_count.rs`,
  `crates/oracle-harness/tests/on_device_sync_count.rs`.
- **dependencies:** SPEC-DRGL-01..05 (the closed form is derived FROM the
  shipped sync pattern, not designed ahead of it).
- **input type:** the resident-perm arm's grow-loop sync trace after
  SPEC-DRGL-05 lands (re-derived from source, per research §4's own
  method: "Re-derived here from a fresh `bump_sync()` grep of
  `grow_driver.rs`" — not trusted from either existing test file blindly).
- **output type:** an updated `analytic_rp` (or equivalently-named)
  constant + doc-comment closed form in BOTH files, EXACT-equality
  asserted (not `<=`).
- **preconditions:** SPEC-DRGL-05 has shipped and is stable.
- **behavior:**
  - *Given* the Rank-1 sync pattern (`~num_leaves + O(1)`, exact form
    re-derived from the actual `bump_sync()` call sites), *when*
    `on_device_sync_count_is_num_features_independent`
    (`crates/lgbm-compute/tests/on_device_sync_count.rs:140`) and
    `on_device_sync_count_collapses_to_num_leaves`
    (`crates/oracle-harness/tests/on_device_sync_count.rs:144`) run on
    real ROCm hardware, *then* both assert the NEW exact constant, not
    the pre-Rank-1 `2*num_leaves`.
- **postconditions:** both files' documented closed forms agree with each
  other and with a fresh source-level `bump_sync()` grep.
- **errors:** test failure (assertion mismatch) is the intended detector
  for any sync-count drift.
- **side effects:** none (test-only change).
- **acceptance examples:**
  - Given the Rank-1-complete driver, when
    `cargo test -p lgbm-compute --features rocm -- --exact
    on_device_sync_count_is_num_features_independent` runs on real
    gfx1152, then it passes with the NEW exact constant.
- **evidence:**
  - `[LOCAL crates/lgbm-compute/tests/on_device_sync_count.rs:182-326]`.
  - `[PROJECT: research.md §4 "Exact sync-count closed form", §16 row]`.
- **non-goals:** the Wave-3 re-derivation (SPEC-DRGL-10, a separate,
  later closed form).

---

## ~~Wave 3 — Rank 3 extension: device-resident per-leaf bookkeeping + fixed host schedule~~ — [DROPPED 2026-07-14]

> **⚠️ DROPPED FROM SCOPE (2026-07-14, §1).** SPEC-DRGL-07..10 below are **NOT
> part of this plan's acceptance** — the user chose to stop after Wave 2 and
> not pursue the fully device-resident (Rank 3) loop (`PROJECT.md`-Out-of-Scope,
> prior real-CUDA A/B 1.12–2.2× slower). This entire subsection is retained
> **verbatim as reference only** for a possible future phase; none of these
> specs is implemented, tested, or gated by this plan. The plan's acceptance
> is DRGL-00..06 (Waves 0–2) plus DRGL-11 (Wave 4 perf verdict). Skip to
> "Wave 4" when reading for the live plan.

### SPEC-DRGL-07 — Device-resident per-leaf bookkeeping state (extends `DeviceFrontier`)
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** today's `ResidentDriverLeaf` bookkeeping
  (`row_begin`, `row_count`, `sum_g`, `sum_h`, `slot`, `best`,
  `best_fpos`, `depth`) lives ENTIRELY host-side
  (`Vec<ResidentDriverLeaf>`, `grow_driver.rs:1860-1882,2595`) — Rank 3
  requires this state to be device-resident so the host loop no longer
  needs a per-split readback to know a leaf's row range/sums/slot;
  porting this Rust `Vec` state machine into `#[cube]`-addressable device
  memory is the single highest-risk item in this SPEC (research §10
  Rank 3 "HIGH in practice because of test-coverage exposure").
- **scope:** `crates/lgbm-compute/src/lib.rs` (`DeviceFrontier<R>`
  extension, §4 typed contract, proposed `DeviceLeafState` new type in
  `grow_driver.rs` or a new `kernels/leaf_state.rs` module).
- **dependencies:**
  - SPEC-DRGL-01 (`LEAF_SPLIT_STRIDE`-style fixed-record precedent to
    follow for the new per-leaf SoA layout).
  - `DeviceFrontier<R>`'s existing `records`/`best_leaf`/`stop` fields
    (`lib.rs:511-522`) — this spec adds a SIBLING field, does not replace
    them.
  - `ResidentDriverLeaf` (`grow_driver.rs:1860-1882`) — the exact field
    set and semantics this device structure must mirror bit-for-bit.
- **input type:** `num_leaves: usize` (sizing, mirrors
  `DeviceFrontier::new`).
- **output type:** a `DeviceLeafState` resident SoA (proposed: fixed
  i32/f64-stride record per leaf id, following `LEAF_SPLIT_STRIDE`'s
  precedent) carrying `row_begin, row_count, sum_g, sum_h, slot, depth,
  smaller_scannable, larger_scannable` per open leaf, seedable and
  updatable entirely device-side.
- **preconditions:** none beyond `num_leaves >= 1`.
- **behavior:**
  - *Given* the root leaf's seed values (host-known: `row_begin=0,
    row_count=num_data, sum_g=root_sum_g, sum_h=root_sum_h, slot=0,
    depth=0`), *when* `DeviceLeafState` is seeded, *then* a device read
    of slot 0 returns exactly these values (round-trip correctness).
- **postconditions:** the structure supports device-side UPDATE (a split
  writing its two children's bookkeeping) without a host round-trip,
  mirroring today's host `leaves[best_leaf] = ...` / `leaves.push(...)`
  pattern (`grow_driver.rs:2945-2965`) but on-device.
- **errors:** `ComputeError::Runtime` on allocation/dispatch failure
  (existing pattern).
- **side effects:** device allocation, sized `O(num_leaves)` (bounded,
  finite — matches every other resident structure in this crate).
- **acceptance examples:**
  - Given a device-seeded root record, when read back via a
    TEST/DEBUG-only accessor (mirrors `DeviceFrontier::read_best_leaf`,
    `lib.rs:615-621`, explicitly documented "the control loop keeps it
    resident"), then the returned values match the host seed exactly.
- **evidence:**
  - `[LOCAL grow_driver.rs:1857-1882]` — the host struct this device
    structure must reproduce.
  - `[LOCAL lib.rs:495-622]` — `DeviceFrontier`'s existing SoA precedent.
  - `[PROJECT: research.md §5 "Device-side infrastructure already
    available to build on" — "the leaf bookkeeping loop itself is still
    entirely host-side Rust... This is the gap between 'defer
    pick/read_leaf' (bounded, Rank 1) and 'fully device-resident loop'
    (Rank 3)"]`.
- **non-goals:** the scannability-gate COMPUTATION (SPEC-DRGL-08, a
  separate device kernel reading this state); the fixed host schedule
  that stops reading `leaves: Vec<...>` entirely (SPEC-DRGL-09).

### SPEC-DRGL-08 — Device-resident scannability-gate computation (`smaller_scannable`/`larger_scannable` on-device)
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** today's `smaller_scannable`/
  `larger_scannable` gates (`grow_driver.rs:3021-3034`: `s_n >=
  min_data_x2 && !(max_depth>0 && depth>=max_depth) && s_h > 0.0 && s_n >
  0`) are HOST comparisons over host-cached `s_n`/`s_h`/`depth` values —
  Rank 3 requires these to run on-device against SPEC-DRGL-07's resident
  state so the host loop does not need to read them back per split to
  decide whether to launch the next scan.
- **scope:** new `#[cube]` kernel reading `DeviceLeafState`
  (SPEC-DRGL-07), writing a resident scannable-flag pair per split.
- **dependencies:** SPEC-DRGL-07 (the resident state this gate reads);
  `min_data_in_leaf`/`max_depth` (host-constant for the whole grow — safe
  kernel-launch parameters, not per-split reads).
- **input type:** `(min_data_x2: i32, max_depth: i32)` (grow-constant
  launch parameters) + the resident leaf record for the leaf under test.
- **output type:** a resident `bool` (or i32 0/1) pair
  `(smaller_scannable, larger_scannable)` per split.
- **preconditions:** SPEC-DRGL-07's resident state for both children is
  seeded before this gate evaluates.
- **behavior:**
  - *Given* a child with `row_count < 2*min_data_in_leaf`, *when* the
    gate kernel runs, *then* it writes `scannable=false` — bit-identical
    decision to today's host `s_n >= min_data_x2` branch for every input
    (pure integer/float threshold comparison, no FP reorder, per research
    §7 "integer/threshold comparisons — no new FP reorder by itself").
- **postconditions:** the resident scan-dispatch kernels (build/subtract/
  scan) can read this flag device-side to skip an unscannable child's
  scan, matching today's host `if smaller_scannable { ... } else {
  reduce_winner_into_frontier(...) }` branch structure
  (`grow_driver.rs:3179-3193`) but without a host round-trip.
- **errors:** `ComputeError::Runtime` on dispatch failure.
- **side effects:** none beyond the resident flag write.
- **acceptance examples:**
  - Given the 4 boundary conditions of today's host gate
    (`row_count`-too-small, `depth`-capped, `sum_h<=0`, `row_count<=0`),
    when the device gate evaluates each independently, then each
    produces the SAME `false` result as the host branch — 4 dedicated
    boundary-condition unit/kernel tests.
- **evidence:**
  - `[LOCAL grow_driver.rs:3021-3034]` — the exact host predicate this
    kernel must reproduce bit-for-bit.
- **non-goals:** the fixed host schedule that stops reading these flags
  back per split (SPEC-DRGL-09).

### SPEC-DRGL-09 — Host loop collapses to a fixed `num_leaves-1` schedule polling only the device stop flag
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** this is the capstone wiring change —
  `grow_tree_on_device_resident`'s per-split loop body
  (`grow_driver.rs:2677-3213`) currently branches on FIVE host-visible
  values per split (`best_leaf<0`, `!(gain>0.0)`, `smaller_is_left`,
  `smaller_scannable`, `larger_scannable`); after SPEC-DRGL-01..08, only
  the loop-continuation stop signal remains a genuine host read (research
  §5 "pick's host use is a mix... 2 scalar loop-continuation decisions...
  cannot be pushed to the device without moving the loop itself off the
  host") — an off-by-one in the fixed `num_leaves-1` schedule vs the
  actual grown leaf count, or a stale device state reused before its
  producing kernel completes, silently grows a WRONG tree (research §10
  Rank 3 "plausible and hard to catch outside real-GPU CI").
- **scope:** `crates/lgbm-compute/src/kernels/grow_driver.rs` — a new
  `grow_tree_on_device_resident_rank3` entry point (§4 typed contract),
  gated behind its own opt-in env hatch (mirrors
  `resident_perm_partition_enabled`'s `OnceLock` + `AtomicU8`
  same-session-override template, `grow_driver.rs:417-445` — proposed
  `LGBM_GROW_RANK3` gate, default OFF until Wave 4's Kaggle validation
  passes, per the project's own "opt-in and known-slow until proven"
  convention for on-device paths, `[PROJECT: .planning/PROJECT.md Key
  Decisions "on_device_default() stays false"]`).
- **dependencies:** SPEC-DRGL-01..08 (every device-resident piece Rank 3
  depends on must already be in place).
- **input type:** identical to `grow_tree_on_device_resident`'s existing
  signature (§4 typed contract) — no new PUBLIC input shape, only the
  internal loop body changes.
- **output type:** identical `(lgbm_model::Tree, LeafPartitionLayout)` —
  a drop-in structural equivalent, gated behind the new opt-in hatch so
  the existing arm remains the default and is not removed by this spec.
- **preconditions:** the device stop flag (`frontier.stop_handle()`,
  already resident, `lib.rs:517`) is written by the on-device pick
  (mirrors today's `best_leaf` self-invalidation write,
  `best_split.rs:2365-2370`, extended to also set `stop=1` when
  `best_leaf==-1` OR `!(gain>0.0)`).
- **behavior:**
  - *Given* a grow that today stops early (best_leaf<0 or gain<=0 before
    `num_leaves-1` splits), *when* the Rank-3 schedule runs, *then* it
    launches the SAME number of REAL splits as today (the fixed-schedule
    ITERATIONS beyond the real stop point are device no-ops, detected by
    the stop-flag poll, and do not corrupt tree state).
  - *Given* a grow that fills all `num_leaves-1` splits, *when* the
    Rank-3 schedule runs, *then* it produces a tree structurally
    identical to `grow_tree_on_device_resident`'s current output for the
    SAME corpus.
- **postconditions:** the resulting tree/layout is BYTE-IDENTICAL to the
  Wave-2 (Rank 1) driver's output for every corpus in the existing
  regression suite (structure-parity, not a new/different tree).
- **errors:** `ComputeError::Runtime` on dispatch failure (existing
  pattern); the existing `check_root_seed_finite`/`check_tree_leaves_finite`
  tripwires (`grow_driver.rs:2496,3248`) are PRESERVED, not bypassed, by
  this spec.
- **side effects:** the new `LGBM_GROW_RANK3` opt-in env hatch (default
  OFF); no change to the existing (default) resident driver's behavior
  when the hatch is unset.
- **acceptance examples:**
  - Given the existing `resident_score_within_envelope_of_host_cuda`
    (`cuda_on_device.rs:374`) and
    `learner_parity_on_device_resident_fast_path_gate`
    (`learner_parity.rs:3061`) corpora, when run with `LGBM_GROW_RANK3=1`
    on real ROCm hardware, then both continue to pass UNCHANGED (this is
    the load-bearing "still grows the same tree" gate, SPEC-DRGL-10).
- **evidence:**
  - `[LOCAL grow_driver.rs:2677-3213]` — the full per-split branch
    inventory this spec collapses.
  - `[PROJECT: research.md §10 Rank 3 mechanism, §16 "Rank 3 scope
    creep" + "Resident-arm code path invisibility on CPU CI" risk rows]`.
- **non-goals:** removing the Wave-1/2 driver (`grow_tree_on_device_resident`
  stays the default; Rank 3 is opt-in via `LGBM_GROW_RANK3` until Wave 4's
  Kaggle validation, per the locked-decision conflict noted in §9 R-0);
  a true zero-host-crossing loop (explicitly out of scope, §2).

### SPEC-DRGL-10 — Rank-3 sync-count re-derivation + structure-parity re-validation against the CPU anchor and existing envelope gates
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** two independent verification concerns
  bundled under one acceptance gate because they must BOTH pass before
  Rank 3 can be considered shippable, and BOTH are re-verification of
  EXISTING contracts (not new behavior) against the Rank-3 code path: (a)
  the sync closed form changes AGAIN (a third form, distinct from
  SPEC-DRGL-06's Rank-1 form) and must be re-derived exactly in both
  `on_device_sync_count.rs` files; (b) the existing bit-exactness/envelope
  test suite
  (`resident_tree_bit_exact_to_u64_integer_path`,
  `resident_score_within_envelope_of_host_cuda`,
  `learner_parity_on_device_resident_fast_path_gate`) must continue
  passing UNCHANGED against the NEW Rank-3 code path, per research §7's
  explicit list of "what any redesign must not break".
- **scope:** `crates/lgbm-compute/tests/on_device_sync_count.rs`,
  `crates/oracle-harness/tests/on_device_sync_count.rs`,
  `crates/lgbm-compute/tests/cuda_on_device.rs:261,374`,
  `crates/oracle-harness/tests/learner_parity.rs:3061`.
- **dependencies:** SPEC-DRGL-09 (Rank 3 must be complete and stable
  before its closed form/parity gates are re-derived).
- **input type:** the Rank-3 driver's actual sync trace (`LGBM_GROW_RANK3=1`
  on real ROCm hardware) + the existing 4 gate tests' current pass/fail
  status against it.
- **output type:** an updated exact closed-form constant (Rank-3 form,
  distinct from Rank-1's) in both `on_device_sync_count.rs` files, run
  under the `LGBM_GROW_RANK3=1` hatch; and a PASSING result (unchanged
  assertions) from all 4 existing structure/envelope gates.
- **preconditions:** SPEC-DRGL-09 ships behind the opt-in hatch.
- **behavior:**
  - *Given* `LGBM_GROW_RANK3=1`, *when* the sync-count tests run on real
    ROCm hardware, *then* they assert the NEW exact Rank-3 closed form
    (re-derived from a fresh `bump_sync()`/stop-flag-poll grep, not
    assumed).
  - *Given* `LGBM_GROW_RANK3=1`, *when*
    `learner_parity_on_device_resident_fast_path_gate` runs, *then* it
    continues to assert the Rank-3-grown tree's structure is bit-exact to
    the CPU f64 anchor (unchanged assertion, new code path).
- **postconditions:** the Rank-3 driver is provably NOT a silent
  divergence from the anchor, closing the exact risk research §10 Rank 3
  flags as its "HIGH in practice" concern.
- **errors:** test failure (structure/envelope mismatch) is the intended
  detector — per research §16 "Resident-arm code path invisibility on CPU
  CI" risk, this gate MUST be run with `--features rocm` locally BEFORE
  considering Rank 3 done; it is NOT caught by the default `cargo test`
  merge gate.
- **side effects:** none (test-only).
- **acceptance examples:**
  - `LGBM_GROW_RANK3=1 cargo test -p lgbm-compute --features rocm` and
    `LGBM_GROW_RANK3=1 cargo test -p oracle-harness --features rocm` both
    pass on real gfx1152.
- **evidence:**
  - `[LOCAL cuda_on_device.rs:256-417]`, `[LOCAL learner_parity.rs:3053-3111]`.
  - `[PROJECT: research.md §7 "What the resident arm IS held to"]`.
- **non-goals:** the Kaggle CUDA perf run itself (SPEC-DRGL-11, a
  separate, perf-only concern — this spec is correctness-only).

---

## Wave 4 — Perf validation (required before phase completion; retained "P100 verdict before default-ON" gate)

### SPEC-DRGL-11 — Kaggle CUDA/P100 perf validation of the Wave 1→2 (Rank-1) chain
- **status:** draft
- **implementation_state:** unimplemented
- **principal failure reason:** local ROCm validation (a spoofed 8-CU APU,
  `[PROJECT: .planning/PROJECT.md Constraints "local GPU is a spoofed 8-CU
  APU — valid for parity gates, NOT for perf numbers"]`) proves
  correctness but NOT whether the Wave-2 sync-count reduction (batched
  `read_split(i)`+`pick(i+1)` fusion) actually reduces WALL time on a real
  discrete GPU — research §4/§6 "Mistaking 'reduces drain-ledger bucket' for
  'reduces wall time'" is the exact failure this spec's protocol guards
  against, sharpened by the finding that the 23% drain bucket bundles device
  compute with the readback (§9 R-3).
- **scope:** a Kaggle CUDA/P100 bench run driving `rocm_drain_profile.rs`
  (or its CUDA-feature equivalent) with the Wave-2 deferral flag OFF
  (baseline) vs ON (treatment) — a same-session runtime toggle, not two
  builds — on the SAME P100 session.
- **dependencies:**
  - SPEC-DRGL-00..06 (this is the final gate, run only once every prior
    wave is complete and locally ROCm-validated). *(SPEC-DRGL-07..10 are
    dropped and NOT dependencies.)*
  - `[PROJECT: memory/kaggle-bench-workflow.md]` — the established
    account/log-fetch/embed-patch protocol used throughout
    `[[ondevice-perf-campaign]]`.
- **input type:** two run configurations on ONE session — (a) baseline
  (`LGBM_GROW_DEFER_SYNC` unset/OFF — the two-separate-reads path that
  remains the default until this verdict), (b) treatment
  (`LGBM_GROW_DEFER_SYNC=1` — the Wave-2 batched-fusion path).
- **output type:** an order-alternated, warm-median-of-3 wall-time A/B
  result, a COUNTS-ledger proof both arms actually ran their intended
  code path (mirrors the `partition_resident=`/`scan_parprefix=`-style
  positive tripwire convention — here the `deferred_read_fused=` tripwire
  from SPEC-DRGL-05 fires on arm (b) and NOT (a)), and a preds comparison
  (bit-identical on the CPU-anchor-vs-CUDA structural check, or within the
  documented `resident_score_within_envelope_of_host_cuda` envelope for the
  CUDA-vs-CUDA numeric check).
- **preconditions:** SPEC-DRGL-00..10 are all locally ROCm-validated and
  passing.
- **behavior:**
  - *Given* the two arms run order-alternated (A/B/B/A or equivalent, per
    `[[ondevice-perf-campaign]]`'s established discipline) for 3 warm
    reps each, *when* the median wall times are compared, *then* a
    verdict (faster / slower / inconclusive) is recorded with the
    measured ratio, NOT assumed from the a-priori "~23% tail" figure
    (research §16 "always A/B on FREE-RUN wall time... per the
    established bench-protocol lesson").
- **postconditions:** the phase's completion criteria (§8) require this
  spec's result to be recorded (pass/fail/inconclusive), not merely
  attempted.
- **errors:** an inconclusive or REGRESSING result does not corrupt any
  code — it is a recorded outcome that gates whether `LGBM_GROW_DEFER_SYNC`
  is flipped default-ON in a separate follow-up commit (citing the measured
  P100 ratio + spike number, per the `LGBM_DESC_HOIST`/`LGBM_PARTITION_*`
  precedent) or remains an opt-in/rolled-back experiment, mirroring
  `on_device_default() stays false`'s existing precedent
  (`[PROJECT: .planning/PROJECT.md Key Decisions]`).
- **side effects:** none to production code (a bench run + a recorded
  result); any resulting default-flip decision is a SEPARATE, later
  change, not part of this spec's acceptance.
- **acceptance examples:**
  - A Kaggle P100 session run produces a counts-ledger showing the
    `deferred_read_fused=` positive tripwire (SPEC-DRGL-05) firing on arm
    (b) and NOT firing on arm (a); the recorded median wall-time ratio and
    preds-comparison result are both present in the phase's completion
    record.
- **evidence:**
  - `[PROJECT: memory/kaggle-bench-workflow.md]`,
    `[PROJECT: memory/ondevice-perf-campaign.md]` (bench-protocol
    precedent).
  - `[LOCAL crates/lgbm-treelearner/examples/rocm_drain_profile.rs:1-16]`
    (the drain-ledger harness this run reuses/extends).
- **non-goals:** deciding whether to flip any default flag ON based on
  this result — that is an explicit, separate follow-up decision, not
  automatic.

# 6. Acceptance Scenarios (end-to-end)

- **AS-0 (Wave 0):** the uncommitted parprefix/rocm-drain-profile diff is
  committed as its own commit, unmodified, before any Wave 1-2 edit
  begins. → SPEC-DRGL-00.
- **AS-1 (Wave 1 / Rank 2):** on a real-ROCm resident-perm grow, the
  device-resolved smaller/larger role assignment produces a
  byte-identical tree to today's host-branch version, for a corpus that
  exercises a leaf-id re-pick scenario. → SPEC-DRGL-01, -02, -03.
- **AS-2 (Wave 2 / Rank 1):** on the SAME corpus, the batched
  `read_split(i)`+`pick(i+1)` fusion produces a byte-identical tree with
  the per-split sync count reduced per the re-derived Rank-1 closed form.
  → SPEC-DRGL-04, -05, -06.
- **~~AS-3 (Wave 3 / Rank 3)~~ [DROPPED 2026-07-14]** — the fully
  device-resident loop is no longer in scope (§1). No acceptance scenario
  covers SPEC-DRGL-07..10.
- **AS-4 (Wave 4):** a Kaggle CUDA/P100 order-alternated warm-median-of-3
  A/B run — `LGBM_GROW_DEFER_SYNC` OFF (baseline) vs ON (treatment) — with
  the `deferred_read_fused=` counts-ledger tripwire proving arm (b) ran the
  fusion, records a wall-time verdict and a preds-comparison result for the
  Wave-1→2 (Rank-1) deferral vs the two-separate-reads baseline. →
  SPEC-DRGL-11.

# 7. Impact Scope

| Spec | Classification | Impacted symbols/files |
|---|---|---|
| DRGL-00 | procedural (git only) | `crates/lgbm-compute/src/kernels/split.rs`, `crates/lgbm-compute/tests/scan_pargain_parity.rs`, `crates/lgbm-treelearner/src/phase_prof.rs`, `crates/lgbm-treelearner/examples/rocm_drain_profile.rs` (commit only, no content edit) |
| DRGL-01 | Must change (hot device path) | `crates/lgbm-compute/src/kernels/partition.rs` (`DeviceLeafSplits`, `read_leaf`/`write_leaf` → `read_split`/`record_split`); 5 existing callers per CodeGraph blast-radius |
| DRGL-02 | Must change (new device kernel) | `crates/lgbm-compute/src/kernels/partition.rs` or `best_split.rs` (new kernel, co-located with the device-resident reduce family) |
| DRGL-03 | Must change | `crates/lgbm-compute/src/kernels/grow_driver.rs:2925-2938` |
| DRGL-04 | Must change (new `Backend` trait surface) | `crates/lgbm-compute/src/lib.rs` (`Backend` trait); `RocmBackend`/`CudaBackend` implementors |
| DRGL-05 | Must change (hot device path) | `crates/lgbm-compute/src/kernels/grow_driver.rs` (`grow_tree_on_device_resident` per-split loop, `:2677-3213`; new `LGBM_GROW_DEFER_SYNC` opt-in flag, default OFF) |
| DRGL-06 | Must change (test only) | `crates/lgbm-compute/tests/on_device_sync_count.rs`, `crates/oracle-harness/tests/on_device_sync_count.rs` |
| ~~DRGL-07..10~~ | **[DROPPED 2026-07-14]** | Rank-3 device-resident bookkeeping / fixed-schedule loop / re-derivations — removed from scope (§1); not implemented |
| DRGL-11 | Verification/bench only | `crates/lgbm-treelearner/examples/rocm_drain_profile.rs` (or CUDA equivalent); no production code — A/B via `LGBM_GROW_DEFER_SYNC` OFF/ON |
| CPU f64-fold anchor / merge gate | **Explicitly out of scope, all waves** | `grow_tree_on_device_resident` never executes on `CpuBackend` (`resident_pool_supported()` false by construction) |
| `crates/lgbm-treelearner`, `crates/lgbm-boosting`, `crates/lgbm`, Python bindings | **Out of scope** | No public API surface change; purely internal to `lgbm-compute`'s device orchestration — verify via a full workspace test run after each wave |
| `vendor/cubecl-cuda` fork (CUDA-Graph work) | **Explicitly out of scope** | Different lever (enqueue amortization); already concluded not-worth-it on P100; unrelated to any wave here |

**Blast-radius note:** Waves 1-2 touch the hot `grow_driver.rs`/`partition.rs`
device path but stay within cubecl 0.10's existing capabilities (research §5
"cubecl 0.10 is sufficient, unmodified, for the Rank-1 sub-approach"). The
actual sync-timing change (DRGL-05) is gated behind the default-OFF
`LGBM_GROW_DEFER_SYNC` flag, so it cannot affect the default resident driver
until a P100-verdict-backed follow-up flips it ON. Waves 1-2 are **never
exercised by the CPU merge gate** (§9 R-1), so their `--features rocm` gates
are load-bearing.

# 8. Compatibility and Migration

- Waves 1-2 are **behavior-preserving** changes to an internal (non-public,
  `lgbm-compute`-crate-private) hot path — no `lgbm-compute` public API
  change, no model-format change. The resident driver's PUBLIC entry point
  (`grow_tree_on_device_resident`, called only from
  `grow_tree_on_device_driver_with_cfg`) keeps its existing signature.
- The Wave-2 sync-deferral (DRGL-05) is **additive and opt-in**
  (`LGBM_GROW_DEFER_SYNC`, default OFF) — with the flag unset the driver is
  byte-identical to pre-phase behavior; the flag is flipped default-ON only
  by a separate follow-up after SPEC-DRGL-11's P100 verdict, per PROJECT.md's
  `on_device_default() stays false` precedent. Waves 0-1 (DRGL-00..03) are
  byte-identical prep (buffer widening + on-device role assignment) that
  land unconditionally.
- No persisted-schema/model-format migration in any wave.
- Test discipline: real-device gates (`--features rocm`) are load-bearing
  for every wave past Wave 0 and are NOT caught by the default `cargo
  test` merge gate (research §16) — run them explicitly per §5's
  acceptance examples before considering any wave complete.

# 9. Risks and Open Questions

- **R-0 (RESOLVED 2026-07-14 by the scope trim) — the PROJECT.md conflict
  no longer applies.** The earlier version of this SPEC targeted Rank 3, the
  fully device-resident loop, which directly conflicted with
  `.planning/PROJECT.md`'s "Out of Scope" entry (*"Fully GPU-resident
  (no-host-round-trip) best-first grow loop — architecturally shelved
  (per-leaf sync floor), opt-in and known-slow"*, `:70`) and its Key
  Decision (*"`on_device_default()` stays false | Real-CUDA A/B found the
  fully-resident path 1.12-2.2× slower (per-leaf sync floor)"*, `:115`).
  **The user dropped Rank 3 this session (§1), so this plan no longer builds
  the Out-of-Scope capability.** Waves 1-2 (DRGL-00..06) are behavior-
  preserving optimizations of the *existing* resident arm and the Wave-2
  sync-deferral is opt-in/default-OFF pending a P100 verdict — fully
  consistent with `on_device_default() stays false`. **No PROJECT.md edit or
  human milestone-reconciliation is required to proceed.** (If a future
  phase revives Rank 3, the original R-0 reconciliation requirement
  re-applies — see the dropped SPEC-DRGL-07..10.)
- **R-1 (resident-arm test-coverage exposure — still applies to Waves 1-2):**
  `grow_tree_on_device_resident` is NEVER exercised by the CPU f64-fold merge
  gate (§4/§7, research §6) — any Wave-1/2 regression can land and pass the
  default `cargo test` merge gate undetected. Mitigation: treat every
  `--features rocm` gate in the DRGL-01..06 tasks as load-bearing, not
  optional (PLAN.md P-3), per research §6's prevention/verification note.
- **R-2 (design-spike gap, research §8):** the exact bit-layout of
  SPEC-DRGL-02's role-assignment record is not pre-resolved by research —
  resolve the exact field packing in T-02's Red step (extend
  `LEAF_SPLIT_STRIDE` 6→9 vs a parallel same-indexed buffer), following the
  fixed-stride device-record precedent. *(The dropped SPEC-DRGL-07's
  `DeviceLeafState` layout is no longer a live design question.)*
- **R-3 (the 23% premise is an APU compute+sync bundle — the payoff is
  unproven until measured):** research §4 (re-verified 2026-07-14) establishes
  the "~23%" is a **local 8-CU APU drain-ledger bucket** in which
  `GROW_PICK_NS`/`GROW_PARTITION_NS` **wrap the device kernel AND the
  blocking readback together** — so the portion actually recoverable by
  deferring syncs is the *readback fraction only*, strictly `< 23%`, and its
  transfer to the P100 target is unproven (the strongest P100 datum, the
  CUDA-graph campaign, found the P100 residual **device-compute-bound**).
  Because the Wave-2 deferral (DRGL-05) ships default-OFF, a wrong premise
  never silently degrades the default path. **Two-stage mitigation:** (a)
  local **free-run-vs-drain A/B** on real gfx1152 (`LGBM_GROW_DEFER_SYNC`
  OFF vs ON, both with and without `LGBM_GROW_DRAIN`) as the pre-Kaggle
  sanity check that the deferral actually moves free-run wall time and not
  just a drain bucket; (b) the **Kaggle P100 verdict** (SPEC-DRGL-11) as the
  authority that gates whether `LGBM_GROW_DEFER_SYNC` is ever flipped
  default-ON. A regressing or inconclusive result keeps the flag opt-in
  (or reverts DRGL-05 entirely) but does not affect the byte-identical
  Waves 0-1 prep — mirroring `on_device_default() stays false`.
- **R-4 (cubecl 0.10 ceiling):** no wave in this SPEC requires anything
  cubecl 0.10 cannot express (research §8 "sufficient, unmodified, for
  Rank 1"; Wave 3 stays within `#[cube]`-code + existing `Backend`-trait
  extension patterns, NOT a persistent-kernel/indirect-dispatch
  primitive, which research confirms does not exist in 0.10). If a Wave-3
  design spike (R-2) discovers a genuine need for device-computed launch
  dimensions, that is OUT OF SCOPE for this SPEC and must be re-escalated
  as its own research item (mirrors the CUDA-Graph fork precedent).
- **OQ-1 (Wave 1):** whether SPEC-DRGL-02's role-assignment record is a
  3-cell EXTENSION of the existing 6-cell `ChildRanges`/
  `LEAF_SPLIT_STRIDE` layout or a separate parallel buffer — resolve in
  SPEC-DRGL-02's Red step (research flagged this class of layout decision
  as needing "a short design spike," §19 Q3, not resolved by research
  itself).
- **OQ-2 (Wave 2):** the exact set of `Backend` trait methods needed for
  the fixed-grid variant — one method per (build, subtract, scan) or one
  fused method mirroring `build_fix_scan_resident_into_frontier`'s
  existing fused-vs-separate arm split (`grow_driver.rs:3062-3208`
  already has 3 build/subtract/scan ARM variants) — resolve in
  SPEC-DRGL-04's Red step by following whichever existing arm the fixed-
  grid change most naturally extends.
- **~~OQ-3 (Wave 3)~~ [DROPPED 2026-07-14]** — the `DeviceLeafState`
  placement question is moot; Rank 3 is out of scope.
- **OQ-4 (Wave 4):** whether the Kaggle validation targets P100
  specifically (the established `[[kaggle-bench-workflow]]` target) or a
  ROCm-class Kaggle instance, given this phase's premise (the ~23% tail)
  is ROCm-measured (research §6, unresolved) — resolve when scheduling
  Wave 4, defaulting to P100 per the established workflow unless the user
  specifies otherwise. (Because the ~23% is ROCm-measured, a ROCm-class
  Kaggle target may be the more faithful A/B; flagged for the user.)

# 10. Traceability and Sources

- Research: `.planning/plans/device-resident-grow-loop/research.md` (read
  in full this session — §1 Research Summary, §3 uncommitted working
  tree, §4 full call/data flow, §5 pick/read_leaf precise semantics, §6
  C++ prior art, §7 bit-exactness contract, §8 cubecl 0.10 capabilities,
  §9 batch-N-splits rejection, §10 ranked sub-approaches, §14 project
  impact scope, §16 pitfalls/risks, §17 testing strategy, §18 planning
  guidance, §19 open questions).
- Verified local symbols (this session, direct Read + CodeGraph):
  `grow_driver.rs:2329-3250` (full `grow_tree_on_device_resident`,
  re-read in full this session), `partition.rs:342-408` (`DeviceLeafSplits`),
  `lib.rs:490-622` (`DeviceFrontier`), `lib.rs:1120-1165`
  (`Backend` trait CpuBackend-stub pattern), `best_split.rs:2293-2446`
  (`PickExport`/`find_best_from_all_splits_on`),
  `grow_driver.rs:400-445` (env-gate + `AtomicU8` override pattern),
  `on_device_sync_count.rs:170-326` (both crates, sync closed forms
  re-confirmed this session), `resident_perm_partition.rs:1-25,88`
  (byte-identity gate idiom), `cuda_on_device.rs:261,374`,
  `learner_parity.rs:3061`.
- Project constraints: `CLAUDE.md`, `AGENTS.md`,
  `docs/cuda-kernel-design.md:155-232,907-935` (C++ CUDA reference).
- `.planning/PROJECT.md` — read this session; contains the R-0 conflict
  documented in §9 (Out-of-Scope entry + Key Decision predating this
  phase's locked user decision).
- Format/structure template: `.planning/plans/parity-gap-closure/SPEC.md`
  and `PLAN.md` (heading structure, frontmatter, per-spec fields matched
  verbatim); `.planning/plans/quantized-grad-param-plumbing/SPEC.md`
  (secondary cross-check).
- Evidence labels throughout: `[VERIFIED: ...]` = directly read/confirmed
  this session; `[LOCAL ...]` = read this session, cited without a
  research-doc intermediary; `[PROJECT: ...]` = cited from a project doc
  (research.md, memory files, docs/, PROJECT.md); design choices this
  SPEC introduces beyond research's own scope are marked `proposed` in §4
  and flagged in §9 OQ-1..3.
