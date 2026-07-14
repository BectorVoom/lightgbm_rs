# TDD Implementation Plan — Device-Resident Grow-Loop Sync Deferral (Rank 2 → Rank 1, STOP after Wave 2)

Derived from `.planning/plans/device-resident-grow-loop/SPEC.md` (draft) and
`research.md`. Every task is Red → Green → Refactor → Verify. Tasks are
ordered by dependency, not file layout, per the incremental Rank 2 → Rank 1
sequence locked by the user (SPEC.md §1). **No task is marked complete
during planning.**

> **Scope (REVISED 2026-07-14):** this plan builds **Waves 0 → 1 → 2 → 4** and
> **STOPS after Wave 2** — banking the Rank-1 sync-halving win. **Rank 3 (the
> fully device-resident loop) is DROPPED**; the former T-07..T-10 tasks are
> retained below only as **[DROPPED]**-marked reference. Dropping Rank 3
> **resolves the former SPEC §9 R-0 PROJECT.md conflict** (Waves 1-2 are
> behavior-preserving changes to the *existing* resident arm, not the shelved
> Out-of-Scope capability) — **no human PROJECT.md reconciliation is required
> to proceed.** The actual sync-timing change (T-05) ships behind a default-OFF
> `LGBM_GROW_DEFER_SYNC` flag; Wave 4 (T-11) is its P100 perf verdict, which
> gates any later default-flip.

## Global preconditions (do once, before any Wave 1 Green step)

- **P-0 Confirm dependencies (AGENTS.md rule).** No new external crate is
  needed for any wave (SPEC §3). `cubecl` stays pinned at `0.10.0`. Record
  this confirmation in every wave's commit message (AGENTS.md rule 3:
  "Include dependency information in Git commit messages").
- **P-1 ROCm env.** Every real-device task below needs the local ROCm
  environment set up per `[[local-rocm-gpu]]`:
  ```bash
  export ROCM_PATH=/home/user/rocm/opt/rocm-7.1.1 HIP_PATH=$ROCM_PATH ROCM_HOME=$ROCM_PATH
  export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH PATH=$ROCM_PATH/bin:$PATH
  ```
- **P-2 Git hygiene (locked decision #2).** T-00 (Wave 0) MUST land before any
  Wave 1 edit touches `crates/lgbm-compute/src/kernels/split.rs` or any other
  file in the uncommitted diff's footprint.
- **P-3 Test discipline.** Every `--features rocm` gate cited below is
  load-bearing and is NOT run by the default `cargo test` merge gate
  (SPEC §9 R-1) — a wave is not "done" until its rocm-gated tests have
  actually been run and observed passing on real hardware, not merely
  compiled.

Validation commands referenced below (adapted from research §17):
```bash
cargo test -p lgbm-compute                                            # cpu-runnable unit/reference tests
cargo test -p lgbm-compute --features rocm                            # real-device gates
cargo test -p oracle-harness --features rocm
cargo test -p lgbm-compute --features rocm -- --exact on_device_sync_count_is_num_features_independent
cargo test -p oracle-harness --features rocm -- --exact learner_parity_on_device_resident_fast_path_gate
cargo test -p lgbm-compute --features rocm -- --exact resident_tree_bit_exact_to_u64_integer_path
cargo test -p lgbm-compute --features rocm -- --exact resident_score_within_envelope_of_host_cuda
LGBM_CUDA_ON_DEVICE=1 LGBM_PHASE_PROF=1 LGBM_GROW_DRAIN=1 \
  cargo run --release -p lgbm-treelearner --features rocm --example rocm_drain_profile
```

---

## Wave 0 — Prior-work hygiene (do first, blocks everything else)

### T-00 — Commit the uncommitted parprefix/rocm-drain-profile diff · SPEC-DRGL-00
- **prereqs:** none. **files:** commit-only —
  `crates/lgbm-compute/src/kernels/split.rs`,
  `crates/lgbm-compute/tests/scan_pargain_parity.rs`,
  `crates/lgbm-treelearner/src/phase_prof.rs`,
  `crates/lgbm-treelearner/examples/rocm_drain_profile.rs`.
- **Red:** N/A — this is a procedural git task, not a code change; there is
  no failing test to write. The "Red" step instead is a **verification
  check**: run `git status --porcelain` and `git diff --stat` and confirm
  the diff matches SPEC-DRGL-00's documented 4-file shape (`+752/-0`
  `split.rs`, `+258/-0` `scan_pargain_parity.rs`, `+6/-3` `phase_prof.rs`,
  new `rocm_drain_profile.rs`) exactly, with no drift since research was
  written. Expected "failure" if it does NOT match: STOP, do not commit
  blind — re-diff against research §3 and report the discrepancy before
  proceeding.
- **Green:** `git add` the 4 files by exact path (never `git add -A`/`.`,
  per the harness's git safety protocol) and commit with a message
  recording: (a) what the diff contains (parallel-prefix-scan
  `LGBM_SCAN_PARPREFIX` work + ROCm-default-ON flip, from
  `[[cudagraph-campaign]]`/`[[ondevice-perf-campaign]]`), (b) that no new
  external crate dependency was introduced (AGENTS.md rule 3 — dependency
  info in the commit message), (c) that this commit is a prerequisite,
  content-unmodified landing of prior work, done first per this phase's
  Wave-0 ordering.
- **Refactor:** none (no content change).
- **Verify:** `git status --porcelain` no longer lists the 4 files;
  `git show --stat HEAD` shows exactly those 4 files with unchanged
  insertion/deletion counts vs the pre-commit `git diff --stat`.
- **Implementation Steps:**
  1. `git status --porcelain` — confirm current shape matches SPEC-DRGL-00.
  2. `git diff --stat -- crates/lgbm-compute/src/kernels/split.rs
     crates/lgbm-compute/tests/scan_pargain_parity.rs
     crates/lgbm-treelearner/src/phase_prof.rs` — confirm line counts.
  3. `git add crates/lgbm-compute/src/kernels/split.rs
     crates/lgbm-compute/tests/scan_pargain_parity.rs
     crates/lgbm-treelearner/src/phase_prof.rs
     crates/lgbm-treelearner/examples/rocm_drain_profile.rs`.
  4. Commit with the message content described in Green above.
  5. `git status --porcelain` post-check; confirm `.planning/plans/
     device-resident-grow-loop/`, `planning/`, `vendor/` remain untouched
     (unrelated, per research §3).
- **Completion Criteria:**
  - [ ] `git status --porcelain` shows the 4 files no longer modified/untracked.
  - [ ] `git show --stat HEAD` line counts match the pre-commit diff exactly.
  - [ ] The commit message names the dependency confirmation (P-0) and the
        SPEC-DRGL-00 ID.
  - [ ] No other file (`planning/`, `vendor/`, `.planning/plans/
        device-resident-grow-loop/`) was included in this commit.
- **Risks and Guardrails:** if `git status` has drifted from the documented
  4-file shape (e.g. a 5th file appeared, or line counts differ), STOP and
  re-verify against a fresh `research.md` re-read before committing —
  do not assume the research document's snapshot is still accurate.

---

## Wave 1 — Rank 2: on-device role assignment (small, low-risk stepping stone)

> Precondition for this whole wave: T-00 complete.

### T-01 — Widen `DeviceLeafSplits` to a per-split append-only buffer · SPEC-DRGL-01
- **prereqs:** T-00. **files:** Modify:
  `crates/lgbm-compute/src/kernels/partition.rs` (`DeviceLeafSplits<R>`,
  `LEAF_SPLIT_STRIDE`, `ChildRanges`, `read_leaf`/`write_leaf` call sites).
  Test: `crates/lgbm-compute/src/kernels/partition.rs` (`#[cfg(test)]`
  module, existing pattern) + `crates/oracle-harness/tests/partition_parity.rs`
  (re-audit only) + `crates/lgbm-compute/tests/resident_perm_partition.rs`
  (re-audit only, per SPEC-DRGL-01's blast-radius note).
- **1. Red:** Add a unit test `device_leaf_splits_survives_leaf_id_repick` in
  `partition.rs`'s `#[cfg(test)]` module: allocate a widened
  `DeviceLeafSplits` with `capacity = 2*(num_leaves-1)` for a small
  `num_leaves`, `record_split` two DIFFERENT splits whose `new_left`
  happens to reuse the SAME leaf id (simulating the re-pick scenario), then
  `read_split` BOTH and assert neither aliases the other's data. Expected
  failure: `record_split`/`read_split` do not exist yet (compile error) —
  the CURRENT `write_leaf`/`read_leaf` are leaf-id-keyed and WOULD alias on
  this exact scenario if run today.
  Run: `cargo test -p lgbm-compute device_leaf_splits_survives_leaf_id_repick`.
- **2. Green:** Rename/replace the struct fields per SPEC §4's typed
  contract (`capacity`, `next_split_idx`), implement `record_split`
  (append-only, monotonically incrementing index) and `read_split`
  (indexed by split index, not leaf id, panics if `s >= next_split_idx`,
  mirroring today's `assert!(leaf_id < self.num_leaves, ...)` bounds-check
  style). Update the 5 existing call sites (`grow_driver.rs`, per
  CodeGraph blast-radius) to pass a `split_idx` instead of a `leaf_id` —
  a purely mechanical rename at every current call site since the CURRENT
  code always uses `leaf_id == new_left` and there is a 1:1 split-order
  correspondence until Wave 2 actually defers a read. Do NOT change the
  driver's control flow yet (no deferred reads in this task — SPEC-DRGL-05
  owns that).
  Run: `cargo test -p lgbm-compute`.
- **3. Refactor:** extract any shared bounds-check logic between
  `record_split`/`read_split` into a small private helper if duplicated;
  keep the `ComputeError::Runtime` guard on `capacity == 0` exactly as
  the existing `num_leaves == 0` guard reads today.
  Run: `cargo test -p lgbm-compute`.
- **4. Verify:**
  Run: `cargo test -p lgbm-compute`.
  Run: `cargo test -p lgbm-compute --features rocm` (existing
  `resident_perm_partition.rs` gates must still pass — same tree, same
  layout, only the addressing scheme changed).
  Run: `cargo test -p oracle-harness --features rocm -- partition_parity`.
  Confirm: SPEC-DRGL-01's acceptance example (re-pick scenario) passes;
  every pre-existing `DeviceLeafSplits` caller still compiles and its
  existing tests are unchanged in outcome.
- **Implementation Steps:**
  1. Write the Red test.
  2. Rename `DeviceLeafSplits::new(num_leaves)` → `new(capacity)`; update
     the struct fields per §4.
  3. Implement `record_split`/`read_split`; delete `write_leaf`/`read_leaf`
     (or keep as deprecated thin wrappers ONLY if a call site cannot be
     migrated in this task — prefer full migration).
  4. Update `grow_driver.rs`'s 5 call sites to track a `split_idx` counter
     alongside the existing per-split loop index (`_split` in the `for
     _split in 0..(num_leaves-1)` loop is ALREADY a natural split index —
     reuse it directly rather than inventing a second counter).
  5. Update the allocation call from `num_leaves` to `2*(num_leaves-1)`
     capacity at the `DeviceLeafSplits::new` call site
     (`grow_driver.rs:2657-2659`).
  6. Run the full validation set above.
- **Completion Criteria:**
  - [ ] The Red test fails for the stated principal reason (aliasing / API
        absence) before this task's production changes.
  - [ ] The Green implementation passes the focused test.
  - [ ] All 5 pre-existing `DeviceLeafSplits` callers compile and pass.
  - [ ] `resident_perm_partition.rs` and `partition_parity.rs` gates
        unchanged (still pass) on real ROCm hardware.
  - [ ] No unrelated behavior added (no deferred-read timing change yet).
- **Risks and Guardrails:** the widened allocation is strictly larger
  (`2*(num_leaves-1)` vs `num_leaves` i32 cells × `LEAF_SPLIT_STRIDE`) —
  bounded and finite, but confirm no existing test hard-codes the OLD
  buffer size in a way that would now silently pass on out-of-bounds
  reads; add an explicit oversized-index panic test if none exists.

### T-02 — On-device smaller/larger role-assignment kernel · SPEC-DRGL-02 (dep: T-01)
- **prereqs:** T-01. **files:** Create/Modify: `crates/lgbm-compute/src/kernels/partition.rs`
  (new `#[cube]` kernel + host launcher, co-located near
  `resident_scatter_fused_bc_smem_kernel` per its real-device-only gating
  precedent). Test: `#[cfg(test)]` module in the same file +
  `crates/lgbm-compute/tests/resident_perm_partition.rs` (rocm-gated new
  test).
- **1. Red:** Add a CPU-runnable reference-function unit test
  `role_assignment_matches_host_branch_for_all_count_orderings` — a plain
  Rust twin (following the `scan_pargain_parity.rs` idiom: prove the
  reformulation bit-equal to serial/host logic BEFORE writing the kernel)
  that computes `(smaller_is_left, smaller_slot, larger_slot)` from
  `(left_count, right_count, next_slot, parent_slot)` and asserts it
  matches `grow_driver.rs:2925-2938`'s exact `<` semantics for
  `left_count < right_count`, `==`, and `>` cases. Expected failure: the
  reference function does not exist yet.
  Run: `cargo test -p lgbm-compute role_assignment_matches_host_branch`.
- **2. Green:** Implement the plain-Rust reference function first (proves
  the decision logic), then the `#[cube]` device kernel that performs the
  identical pure-integer comparison and writes the 3-field role record
  into (or alongside) the split's `DeviceLeafSplits` record (resolve OQ-1
  from SPEC.md §9 here: either extend `LEAF_SPLIT_STRIDE` from 6 to 9, or
  add a parallel same-indexed buffer — pick whichever keeps
  `record_split`/`read_split` simplest, document the choice in the kernel's
  doc comment). Gate the kernel real-device-only
  (`<R as Runtime>::name(client) != "cpu"`, mirroring
  `partition_bc_fused`, `partition.rs:1302-1305`) since it is only USED on
  the resident-perm arm which already requires a real device backend.
  Run: `cargo test -p lgbm-compute`.
- **3. Refactor:** ensure the kernel calls the SAME comparison expression
  as the plain-Rust reference (no independent reimplementation drift);
  add a `role_assign=` COUNTS tripwire (mirrors the `scan_parprefix=`
  precedent) proving the kernel actually ran when enabled.
  Run: `cargo test -p lgbm-compute`.
- **4. Verify:**
  Run: `cargo test -p lgbm-compute --features rocm` — a NEW real-device
  test in `resident_perm_partition.rs` comparing the device kernel's
  output to the plain-Rust reference for a real split sequence (mirrors
  `partition_bc_fusion_byte_identical_to_three_launch`'s byte-identity
  idiom).
  Confirm: SPEC-DRGL-02's 3 acceptance examples (left-smaller,
  right-smaller, exact-tie) all pass, including the tie case's `<` (not
  `<=`) semantics.
- **Implementation Steps:**
  1. Write the plain-Rust reference function + its CPU-runnable Red test.
  2. Implement the `#[cube]` kernel calling the identical comparison.
  3. Wire the kernel's output into (or alongside) `DeviceLeafSplits`'s
     per-split record (resolve OQ-1 here).
  4. Add the `role_assign=` tripwire.
  5. Add the real-device byte-identity test in `resident_perm_partition.rs`.
  6. Run the full validation set above.
- **Completion Criteria:**
  - [ ] The Red test (plain-Rust reference) fails before the reference
        function exists.
  - [ ] The device kernel's output matches the plain-Rust reference for
        all 3 count-ordering cases on real ROCm hardware.
  - [ ] The tie-case (`left_count == right_count`) reproduces today's
        `smaller_is_left=false` result exactly.
  - [ ] No FP reorder introduced (pure integer comparison, unit-tested).
- **Risks and Guardrails:** a `<` vs `<=` mismatch on the tie case
  silently swaps which child gets the fresh pool slot — this is the
  single highest-value assertion in this task's test suite; do not skip
  the tie case.

### T-03 — Host loop consumes the device-resolved role assignment · SPEC-DRGL-03 (dep: T-01, T-02)
- **prereqs:** T-01, T-02. **files:** Modify:
  `crates/lgbm-compute/src/kernels/grow_driver.rs:2925-2938`.
- **1. Red:** Add (or extend) a real-device regression test
  `resident_role_assignment_wiring_matches_pre_change_tree` in
  `resident_perm_partition.rs` (or `cuda_on_device.rs`, whichever already
  hosts the closest existing driver-level structure-parity test) that
  grows a fixed corpus BEFORE this task's wiring change (capture the tree
  as a byte fixture / inline expected structure) — following the
  `partition_bc_fusion_byte_identical_to_three_launch` idiom. Expected
  failure at Red time: N/A in the strict sense (the pre-change tree IS
  today's tree) — instead, write the test to assert the tree structure
  against a **hand-captured golden from today's driver**, so that if T-03's
  Green step introduces ANY divergence, this test catches it. Frame this
  explicitly as a regression-pinning Red (captures current behavior as the
  target, not a not-yet-existing feature).
  Run: `cargo test -p lgbm-compute --features rocm resident_role_assignment_wiring`.
- **2. Green:** Replace `grow_driver.rs:2925-2938`'s
  `let smaller_is_left = left_count < right_count; ...` block with a read
  of T-02's device-resolved role record (via the SAME synchronous
  `read_split` call already happening this wave — no new sync introduced
  in this task). Delete the now-dead `left_count < right_count`
  expression (SPEC-DRGL-03's explicit postcondition — "no double-source
  of truth").
  Run: `cargo test -p lgbm-compute --features rocm`.
- **3. Refactor:** confirm no other call site still reads
  `left_count`/`right_count` for role bookkeeping (grep guard); tidy
  variable naming so the role-assignment source is unambiguous at the
  call site.
  Run: `cargo test -p lgbm-compute --features rocm`.
- **4. Verify:**
  Run: `cargo test -p lgbm-compute --features rocm`
  (`resident_role_assignment_wiring_matches_pre_change_tree` +
  `resident_perm_partition.rs`'s existing full suite).
  Run: `cargo test -p oracle-harness --features rocm -- --exact
  learner_parity_on_device_resident_fast_path_gate` (must still pass —
  this is the load-bearing "still grows the same tree as the anchor"
  gate, unaffected by an orchestration-only change).
  Confirm: AS-1 (SPEC.md §6) is met.
- **Implementation Steps:**
  1. Capture today's driver's output for a fixed small corpus as the Red
     test's golden (byte fixture or inline assertion).
  2. Replace the host branch with T-02's device-resolved record read.
  3. Delete the dead `left_count < right_count` expression.
  4. Run the full validation set above.
- **Completion Criteria:**
  - [ ] The captured pre-change golden and the post-change tree are
        byte-identical.
  - [ ] `left_count < right_count` no longer appears in the role-assignment
        block (grep-verified).
  - [ ] `learner_parity_on_device_resident_fast_path_gate` still passes
        unchanged.
- **Risks and Guardrails:** this task is the FIRST end-to-end wiring of
  Waves 1-2's device-resident pieces into the live driver — treat any
  divergence from the captured golden as a hard stop, not a "close
  enough" result (this is a bit-exact-preserving change by construction,
  per SPEC §7 reasoning carried over from research §7).

---

## Wave 2 — Rank 1: fixed-grid builds + batched readback fusion

> Precondition for this whole wave: T-03 complete and its real-device gates
> passing (Wave 1 is a complete, independently shippable unit per research
> §10 Rank 2's own framing — re-confirm this before starting Wave 2).

### T-04 — Fixed-worst-case-grid + device early-exit build/subtract/scan variants · SPEC-DRGL-04 (dep: T-03)
- **prereqs:** T-03. **files:** Modify: `crates/lgbm-compute/src/lib.rs`
  (`Backend` trait — new method, CpuBackend-stub-pattern default body
  returning `ComputeError::Runtime`, mirroring `build_resident_leaf_rows_handle`,
  `lib.rs:1127-1145`); `RocmBackend`/`CudaBackend` implementors. Test:
  new `#[cfg(feature = "rocm")]` test in `resident_perm_partition.rs` or a
  new `fixed_grid_build_parity.rs`.
- **1. Red:** Add a real-device byte-identity test
  `fixed_grid_build_byte_identical_to_exact_grid` — for a fixed split
  sequence, run TODAY's exact-grid `build_resident_leaf`/
  `subtract_resident`/`scan_resident_leaf_into_frontier` chain and capture
  its resident histogram/frontier state, then run the new fixed-grid
  variant with the grid sized off the PARENT'S row count instead of the
  exact child count, and assert byte-identical output. Expected failure:
  the new `Backend` trait method does not exist yet (CpuBackend default
  `Err` fires, or compile error).
  Run: `cargo test -p lgbm-compute --features rocm fixed_grid_build_byte_identical`.
- **2. Green:** Add the new `Backend` trait method (§4 typed contract,
  `build_subtract_scan_fixed_grid_into_frontier` or the split-out
  per-op variant chosen per SPEC §9 OQ-2) with the CpuBackend default
  `Err(ComputeError::Runtime{...})` stub; implement the real-device-only
  `#[cube]` kernel(s) that launch with `parent_row_count_upper_bound` as
  the grid dimension and an early-exit guard reading the child's real
  count from `DeviceLeafSplits` (T-01's widened buffer). Do NOT wire this
  into the live driver loop yet — this task proves the KERNEL is correct
  in isolation; T-05 wires it in.
  Run: `cargo test -p lgbm-compute --features rocm`.
- **3. Refactor:** if the early-exit guard logic is duplicated across
  build/subtract/scan, extract a shared device-side bounds-check helper
  (mirrors the existing per-kernel bounds-check style in this crate — do
  NOT over-abstract across unrelated kernels).
  Run: `cargo test -p lgbm-compute --features rocm`.
- **4. Verify:**
  Run: `cargo test -p lgbm-compute --features rocm`.
  Confirm: the byte-identity test passes for a MULTI-split sequence (not
  just one split) so an off-by-one in the early-exit guard across
  DIFFERENT grid/child-count ratios would surface.
- **Implementation Steps:**
  1. Write the byte-identity Red test using today's exact-grid path as
     the golden.
  2. Add the `Backend` trait method + CpuBackend stub.
  3. Implement the real-device fixed-grid kernel(s) with the early-exit
     guard.
  4. Run the byte-identity test to Green.
  5. Extend the test to a multi-split sequence.
- **Completion Criteria:**
  - [ ] The Red test fails (method absent) before this task's changes.
  - [ ] The fixed-grid variant's output is byte-identical to the
        exact-grid variant across a multi-split sequence on real ROCm
        hardware.
  - [ ] CpuBackend's stub returns `ComputeError::Runtime` (never called
        from a CPU-anchor code path — verified by SPEC §4/§7's existing
        dispatch guard).
- **Risks and Guardrails:** this is SPEC.md's flagged single highest-risk
  item in Wave 2 (an early-exit off-by-one either double-processes or
  undercounts rows) — do not consider this task done on a single-split
  test; the multi-split byte-identity requirement in Verify is mandatory,
  not optional.

### T-05 — Batched `read_split(i)` + `pick(i+1)` fusion · SPEC-DRGL-05 (dep: T-04)
- **prereqs:** T-04. **files:** Modify:
  `crates/lgbm-compute/src/kernels/grow_driver.rs`
  (`grow_tree_on_device_resident`'s per-split loop, `:2677-3213`; new
  `read_deferred_split_and_pick` helper per §4 typed contract).
- **1. Red:** Extend T-03's regression test (or add a new one,
  `resident_batched_read_matches_two_separate_reads`) with the SPEC-DRGL-01
  re-pick scenario driven through the LIVE driver (not just the isolated
  buffer test from T-01): construct a corpus where a leaf id is `new_left`
  of split `i` and picked again at split `i+1`, run the driver with T-04's
  fixed-grid kernels wired in but BEFORE this task's batching change (two
  separate reads, as today), capture the tree as golden; then this task's
  Green step must reproduce it with ONE batched read per iteration.
  Expected failure at Red time: same framing as T-03 (regression-pinning);
  the interesting assertion is the SYNC COUNT, which this test should also
  capture via `on_device_sync_count_take()` BEFORE this task's change (as
  a documented "before" baseline) so Green's after-count can be compared.
  Run: `cargo test -p lgbm-compute --features rocm resident_batched_read`.
- **2. Green:** Add the `LGBM_GROW_DEFER_SYNC` opt-in flag (default OFF)
  using the `OnceLock`+`AtomicU8`-override template
  (`resident_perm_partition_enabled`, `grow_driver.rs:417-445`). When OFF,
  the driver keeps today's two-separate-reads path verbatim; when ON,
  implement `read_deferred_split_and_pick` per §4's typed contract and
  replace the two `bump_sync()` sites (`grow_driver.rs:2701`, `:2845`) with
  one combined call per iteration, handling the LAST split's deferred read by
  folding it into the existing grow-tail perm readback (`:3224-3227`) instead
  of leaving it unread. Per the locked "P100 verdict before default-ON"
  decision, this flag stays OFF until SPEC-DRGL-11's verdict justifies a
  separate default-flip commit.
  Run: `cargo test -p lgbm-compute --features rocm`.
- **3. Refactor:** clean up the now-unused separate read call sites; add
  a `deferred_read_fused=` COUNTS tripwire (mirrors `scan_parprefix=`)
  proving the fusion actually ran per split.
  Run: `cargo test -p lgbm-compute --features rocm`.
- **4. Verify:**
  Run: `cargo test -p lgbm-compute --features rocm` (flag OFF — default path
  unchanged).
  Run BOTH flag states of the parity gates and require byte-identical trees:
  `cargo test -p oracle-harness --features rocm -- --exact
  learner_parity_on_device_resident_fast_path_gate` and the same with
  `LGBM_GROW_DEFER_SYNC=1` prefixed; likewise
  `resident_tree_bit_exact_to_u64_integer_path` in both states.
  Confirm: flag-ON tree byte-identical to flag-OFF (and to the pre-batching
  golden); with the flag ON the sync count visibly DROPPED via
  `on_device_sync_count_take()` — do NOT yet assert the new closed form here
  (that is T-06's job).
- **Implementation Steps:**
  1. Capture the pre-batching golden tree + "before" sync count.
  2. Implement `read_deferred_split_and_pick`.
  3. Wire it into the per-split loop, replacing the two separate reads.
  4. Handle the last-split edge case via the grow-tail readback.
  5. Add the `deferred_read_fused=` tripwire.
  6. Run the full validation set above.
- **Completion Criteria:**
  - [ ] The Red test's golden tree matches the Green implementation's
        tree byte-for-byte.
  - [ ] The re-pick scenario (leaf id reused across `i`/`i+1`) is
        correctly handled (unaliased, per T-01's buffer guarantee).
  - [ ] The last-split edge case does not leave any split's child-range
        record unread.
  - [ ] The observed sync count dropped from the pre-batching baseline
        (exact new value pinned by T-06, not this task).
- **Risks and Guardrails:** this task depends on T-01 (non-overwriting
  buffer) AND T-04 (fixed-grid kernels) both being correct — if either
  has a latent bug, this task's regression test is the FIRST point that
  would surface it end-to-end; do not skip the re-pick-scenario corpus.

### T-06 — Rank-1 sync-count closed-form re-derivation · SPEC-DRGL-06 (dep: T-05)
- **prereqs:** T-05. **files:** Modify:
  `crates/lgbm-compute/tests/on_device_sync_count.rs`,
  `crates/oracle-harness/tests/on_device_sync_count.rs`.
- **1. Red:** Because DRGL-05's deferral is behind `LGBM_GROW_DEFER_SYNC`
  (default OFF), the EXISTING `analytic_rp = 2 * NUM_LEAVES` assertion must
  STILL pass with the flag unset (the default path is unchanged) — do NOT
  weaken it. Add a NEW rocm-gated lane assertion for the flag-ON count
  (`resident_sync_lane_defer_sync`, run with `LGBM_GROW_DEFER_SYNC=1`),
  initially asserting a placeholder that FAILS, so the Red step is a genuine
  failing assertion showing the new (lower) actual count.
  Run: `LGBM_GROW_DEFER_SYNC=1 cargo test -p lgbm-compute --features rocm --
  --exact on_device_sync_count_defer_sync`.
- **2. Green:** Re-derive the flag-ON exact closed form from a FRESH
  `bump_sync()`/`bump_launch()` grep of the post-T-05 deferral path
  (per research §4's re-derivation method — do not trust either file's old
  comment blindly, per the documented "closed-form drift" pattern). Add the
  new flag-ON constant + doc-comment to BOTH `on_device_sync_count.rs` files,
  keeping the flag-OFF `2 * NUM_LEAVES` form intact alongside it. Keep all
  assertions EXACT-equality (`assert_eq!`), never `<=`/`<`.
  Run: `cargo test -p lgbm-compute --features rocm`.
  Run: `cargo test -p oracle-harness --features rocm`.
- **3. Refactor:** if the oracle-harness copy's pre-existing stale-note
  pattern (research §4) makes it easier to just re-derive both from one
  shared closed-form constant/module, consider extracting — but do NOT
  over-engineer a shared abstraction between two independently-owned test
  crates if it's not a natural fit; a well-commented duplicate constant is
  acceptable (matches the existing precedent of two independently
  maintained closed forms).
  Run: `cargo test -p lgbm-compute --features rocm && cargo test -p oracle-harness --features rocm`.
- **4. Verify:**
  Run: `cargo test -p lgbm-compute --features rocm -- --exact
  on_device_sync_count_is_num_features_independent`.
  Run: `cargo test -p oracle-harness --features rocm -- --exact
  on_device_sync_count_collapses_to_num_leaves`.
  Confirm: AS-2 (SPEC.md §6) is met; both files' constants agree with
  each other and with the fresh source-level grep.
- **Implementation Steps:**
  1. Run the existing sync-count tests post-T-05 and record the actual
     (new, lower) count.
  2. Grep `bump_sync()` call sites in `grow_driver.rs` post-T-05 and
     derive the closed form by hand.
  3. Update both test files' constants + doc comments.
  4. Run the full validation set above.
- **Completion Criteria:**
  - [ ] Both files assert the SAME new exact closed form.
  - [ ] The closed form is derived from a fresh source grep, not copied
        from research.md's Rank-1 estimate without re-verification.
  - [ ] `num_features`-independence still holds under the new form.
- **Risks and Guardrails:** per SPEC §5 SPEC-DRGL-06, this exact class of
  drift already has ONE documented precedent in this codebase (the
  oracle-harness copy's prior stale note) — do not repeat it; update both
  files in the SAME task/commit, never one without the other.

---

## ~~Wave 3 — Rank 3 extension: device-resident bookkeeping + fixed host schedule~~ — [DROPPED 2026-07-14]

> **⚠️ DROPPED FROM SCOPE (2026-07-14).** The user chose to stop after Wave 2.
> **T-07, T-08, T-09, T-10 below are NOT part of this plan** — they are
> retained verbatim, `[DROPPED]`-marked, only as reference for a possible
> future phase that would first have to reconcile the `PROJECT.md:70`/`:115`
> Out-of-Scope + "1.12–2.2× slower" decision (the former SPEC §9 R-0). None of
> them is to be implemented, tested, or gated by this plan. **The live plan
> jumps from T-06 straight to Wave 4 (T-11).** Do not execute any task in this
> section.

### ~~T-07 — Device-resident per-leaf bookkeeping state · SPEC-DRGL-07~~ — [DROPPED 2026-07-14, reference only]
- **prereqs:** T-06 + the Wave-3 checkpoint. **files:** Modify:
  `crates/lgbm-compute/src/lib.rs` (`DeviceFrontier<R>` extension per §4
  typed contract, resolving OQ-3 — new sibling field vs new struct,
  decided in this task's Red/Green). Create (if OQ-3 resolves to a
  separate struct): `crates/lgbm-compute/src/kernels/leaf_state.rs` (or
  add `DeviceLeafState` to `grow_driver.rs`, whichever the OQ-3 decision
  favors).
- **1. Red:** Add a real-device round-trip test
  `device_leaf_state_root_seed_round_trips` — seed a `DeviceLeafState` (or
  extended `DeviceFrontier`) with the root leaf's known values
  (`row_begin=0, row_count=num_data, sum_g, sum_h, slot=0, depth=0`), read
  it back via a TEST/DEBUG-only accessor (mirrors
  `DeviceFrontier::read_best_leaf`, `lib.rs:615-621`), and assert exact
  equality. Expected failure: the type/method does not exist yet.
  Run: `cargo test -p lgbm-compute --features rocm device_leaf_state_root_seed`.
- **2. Green:** Implement the fixed-stride device SoA (following
  `LEAF_SPLIT_STRIDE`'s precedent) sized `num_leaves`, with device-side
  seed and TEST/DEBUG read-back methods. Do NOT wire this into the live
  driver loop yet (T-09 owns that) — this task proves the STORAGE is
  correct in isolation, mirroring T-04's isolation-first approach.
  Run: `cargo test -p lgbm-compute --features rocm`.
- **3. Refactor:** align field ordering/naming with `ResidentDriverLeaf`
  (`grow_driver.rs:1860-1882`) so the eventual T-09 wiring is a
  near-mechanical field-by-field port, not a redesign.
  Run: `cargo test -p lgbm-compute --features rocm`.
- **4. Verify:**
  Run: `cargo test -p lgbm-compute --features rocm`.
  Confirm: SPEC-DRGL-07's acceptance example (root-seed round-trip)
  passes; the resolved OQ-3 decision is documented in the type's doc
  comment.
- **Implementation Steps:**
  1. Resolve OQ-3 (sibling field on `DeviceFrontier` vs separate struct).
  2. Write the round-trip Red test.
  3. Implement the fixed-stride SoA + seed/read-back methods.
  4. Align field naming with `ResidentDriverLeaf`.
  5. Run the full validation set above.
- **Completion Criteria:**
  - [ ] The round-trip test passes on real ROCm hardware.
  - [ ] OQ-3's resolution is documented in-code.
  - [ ] Field set matches `ResidentDriverLeaf` exactly (no missing/extra
        field vs the host struct it mirrors).
- **Risks and Guardrails:** SPEC.md flags this as the single highest-risk
  spec in the whole plan (never exercised by the CPU merge gate) — keep
  this task's scope to STORAGE correctness only; resist the temptation to
  also wire in update logic here (T-08/T-09 own that), so a failure is
  attributable to one cause.

### ~~T-08 — Device-resident scannability-gate computation · SPEC-DRGL-08~~ — [DROPPED 2026-07-14, reference only]
- **prereqs:** T-07. **files:** Create/Modify: new `#[cube]` kernel reading
  `DeviceLeafState` (co-located with T-07's type). Test: `#[cfg(test)]`
  module + a real-device gate test.
- **1. Red:** Add a plain-Rust reference-function unit test (CPU-runnable,
  following the `scan_pargain_parity.rs` idiom again) covering all 4
  boundary conditions from `grow_driver.rs:3021-3034`
  (`row_count`-too-small, `depth`-capped, `sum_h<=0`, `row_count<=0`) —
  assert the reference function reproduces each `false` case and one
  `true` case exactly. Expected failure: the reference function does not
  exist yet.
  Run: `cargo test -p lgbm-compute scannability_gate_matches_host_predicate`.
- **2. Green:** Implement the plain-Rust reference first, then the
  `#[cube]` device kernel performing the identical comparison chain
  against T-07's resident state, writing the resident
  `(smaller_scannable, larger_scannable)` flag pair.
  Run: `cargo test -p lgbm-compute`.
- **3. Refactor:** ensure the kernel and the plain-Rust reference share
  the SAME literal comparison order (no independent drift risk).
  Run: `cargo test -p lgbm-compute`.
- **4. Verify:**
  Run: `cargo test -p lgbm-compute --features rocm` — a real-device test
  comparing the kernel's resident flag output to the plain-Rust reference
  for each of the 4 boundary conditions.
  Confirm: SPEC-DRGL-08's 4 boundary-condition acceptance examples all
  pass.
- **Implementation Steps:**
  1. Write the plain-Rust reference + CPU-runnable Red test covering all
     4 boundary conditions.
  2. Implement the device kernel calling the identical comparison.
  3. Add the real-device comparison test.
  4. Run the full validation set above.
- **Completion Criteria:**
  - [ ] All 4 boundary conditions produce bit-identical results to
        `grow_driver.rs:3021-3034`'s host predicate.
  - [ ] No FP reorder (integer/threshold comparisons only, per SPEC §5).
- **Risks and Guardrails:** the `#[allow(clippy::neg_cmp_op_on_partial_ord)]`
  annotation on the existing host predicate
  (`grow_driver.rs:3025,3030`) signals this comparison chain has a known
  subtlety (partial-order NaN handling on `s_h > 0.0`/`l_h > 0.0`) —
  reproduce the EXACT comparison operators, do not "simplify" the chain
  during the device port.

### ~~T-09 — Host loop collapses to a fixed schedule polling only the stop flag · SPEC-DRGL-09~~ — [DROPPED 2026-07-14, reference only]
- **prereqs:** T-07, T-08. **files:** Create:
  `grow_tree_on_device_resident_rank3` in
  `crates/lgbm-compute/src/kernels/grow_driver.rs` (new entry point, §4
  typed contract); new `LGBM_GROW_RANK3` env hatch (mirrors
  `resident_perm_partition_enabled`'s `OnceLock`+`AtomicU8` template,
  `grow_driver.rs:417-445`), default OFF.
- **1. Red:** Add a real-device structure-parity test
  `rank3_schedule_matches_wave2_driver_output` — grow the SAME corpus with
  (a) the existing Wave-2 `grow_tree_on_device_resident` (today's
  default) and (b) the new `grow_tree_on_device_resident_rank3` behind
  `LGBM_GROW_RANK3=1`, and assert byte-identical trees/layouts for BOTH
  an early-stopping corpus (fewer than `num_leaves-1` real splits) and a
  full-growth corpus. Expected failure: the new entry point does not
  exist yet.
  Run: `cargo test -p lgbm-compute --features rocm rank3_schedule_matches_wave2`.
- **2. Green:** Implement `grow_tree_on_device_resident_rank3`: extend the
  on-device pick (T-08 / existing `frontier_pick_best_leaf_device`) to
  also write the device `stop` flag (`frontier.stop_handle()`,
  `lib.rs:517`) when `best_leaf==-1` OR `!(gain>0.0)`; collapse the host
  per-split loop body to a fixed `for _ in 0..(num_leaves-1)` schedule
  that, each iteration, polls the stop flag (one small read — this is the
  ONLY per-iteration host-visible value in the Rank-3 arm) and, if set,
  short-circuits the remaining iterations as device no-ops (skips
  launching build/subtract/scan/partition for iterations past the real
  stop point). Preserve `check_root_seed_finite`/`check_tree_leaves_finite`
  exactly as today. Add the `LGBM_GROW_RANK3` env hatch guarding which
  entry point `grow_tree_on_device_driver_with_cfg` dispatches to,
  defaulting OFF (existing arm stays default).
  Run: `cargo test -p lgbm-compute --features rocm`.
- **3. Refactor:** ensure the new entry point shares as much of
  `grow_tree_on_device_resident`'s SETUP/root-fold/tail code as possible
  (do not duplicate the whole function body — factor the per-split loop
  body divergence into a clearly-separated block or a parameterized
  helper) — matching this codebase's existing arm-selection style (e.g.
  the `on_device_f64_fused_build()` if/else arms within the SAME function
  today).
  Run: `cargo test -p lgbm-compute --features rocm`.
- **4. Verify:**
  Run: `cargo test -p lgbm-compute --features rocm`.
  Run: `LGBM_GROW_RANK3=1 cargo test -p oracle-harness --features rocm --
  --exact learner_parity_on_device_resident_fast_path_gate`.
  Run: `LGBM_GROW_RANK3=1 cargo test -p lgbm-compute --features rocm --
  --exact resident_tree_bit_exact_to_u64_integer_path`.
  Run: `LGBM_GROW_RANK3=1 cargo test -p lgbm-compute --features rocm --
  --exact resident_score_within_envelope_of_host_cuda`.
  Confirm: with `LGBM_GROW_RANK3` UNSET, every pre-existing test's outcome
  is byte-identical to before this task (regression guard on the
  DEFAULT arm); with it SET, all 3 structure/envelope gates above pass.
- **Implementation Steps:**
  1. Extend the on-device pick to also write the stop flag.
  2. Implement the fixed-schedule loop body with the stop-flag poll +
     device no-op short-circuit for post-stop iterations.
  3. Add the `LGBM_GROW_RANK3` env hatch, default OFF.
  4. Wire the dispatch in `grow_tree_on_device_driver_with_cfg`.
  5. Run the full validation set above, BOTH with the hatch set and unset.
- **Completion Criteria:**
  - [ ] With `LGBM_GROW_RANK3` unset, the existing default driver's
        behavior is byte-identical to pre-T-09 (regression guard).
  - [ ] With `LGBM_GROW_RANK3=1`, early-stopping and full-growth corpora
        both produce byte-identical trees to the Wave-2 driver.
  - [ ] `check_root_seed_finite`/`check_tree_leaves_finite` tripwires are
        preserved, not bypassed.
- **Risks and Guardrails:** SPEC.md's single highest-risk task — an
  off-by-one in the fixed schedule vs the actual stop point, or a stale
  device-state read before its producing kernel completes, silently
  grows a WRONG tree with no host-visible signal until a structure-parity
  gate catches it. Do not consider this task done without BOTH the
  early-stopping AND full-growth corpus cases in the Red test — an
  early-stopping-only test would not catch an off-by-one in the
  "continue past the real stop point" no-op path.

### ~~T-10 — Rank-3 sync-count re-derivation + structure-parity re-validation · SPEC-DRGL-10~~ — [DROPPED 2026-07-14, reference only]
- **prereqs:** T-09. **files:** Modify:
  `crates/lgbm-compute/tests/on_device_sync_count.rs`,
  `crates/oracle-harness/tests/on_device_sync_count.rs`. Verification only
  (no production changes): `crates/lgbm-compute/tests/cuda_on_device.rs:261,374`,
  `crates/oracle-harness/tests/learner_parity.rs:3061`.
- **1. Red:** Run the existing sync-count tests under `LGBM_GROW_RANK3=1`
  — they will either fail (asserting the Rank-1 closed form against
  Rank-3's different pattern) or need a NEW `resident_sync_lane`-style
  function added for the Rank-3 hatch (follow the existing pattern of a
  separate `#[cfg(feature = "rocm")]` lane function per arm, mirroring
  today's `resident_sync_lane` alongside the legacy-arm lane in the same
  file). Confirm the failure/gap is exactly "no Rank-3 closed form
  asserted yet."
  Run: `LGBM_GROW_RANK3=1 cargo test -p lgbm-compute --features rocm`.
- **2. Green:** Grep `bump_sync()` call sites in the T-09 Rank-3 code path
  and derive the exact closed form (expected to be strictly BELOW T-06's
  Rank-1 form, dominated by the per-iteration stop-flag poll instead of
  the pick+read_split batched read — derive exactly, do not assume).
  Add the new lane function + exact-equality assertion to BOTH
  `on_device_sync_count.rs` files, run under `LGBM_GROW_RANK3=1`.
  Run: `LGBM_GROW_RANK3=1 cargo test -p lgbm-compute --features rocm`.
  Run: `LGBM_GROW_RANK3=1 cargo test -p oracle-harness --features rocm`.
- **3. Refactor:** align the Rank-3 lane function's structure/naming with
  the existing `resident_sync_lane` pattern for consistency.
  Run: `LGBM_GROW_RANK3=1 cargo test -p lgbm-compute --features rocm && LGBM_GROW_RANK3=1 cargo test -p oracle-harness --features rocm`.
- **4. Verify:**
  Run: `LGBM_GROW_RANK3=1 cargo test -p lgbm-compute --features rocm -- --exact resident_tree_bit_exact_to_u64_integer_path`.
  Run: `LGBM_GROW_RANK3=1 cargo test -p lgbm-compute --features rocm -- --exact resident_score_within_envelope_of_host_cuda`.
  Run: `LGBM_GROW_RANK3=1 cargo test -p oracle-harness --features rocm -- --exact learner_parity_on_device_resident_fast_path_gate`.
  Confirm: AS-3 (SPEC.md §6) is met — all 4 gates (3 structure/envelope +
  1 new sync closed form) pass under the Rank-3 hatch.
- **Implementation Steps:**
  1. Run existing sync-count tests under `LGBM_GROW_RANK3=1`, observe the
     gap/failure.
  2. Grep post-T-09 `bump_sync()` sites and derive the Rank-3 closed form.
  3. Add the new lane + assertion to both files.
  4. Run all 3 existing structure/envelope gates under the hatch.
  5. Run the full validation set above.
- **Completion Criteria:**
  - [ ] The Rank-3 closed form is asserted exactly (not `<=`) in both
        `on_device_sync_count.rs` files.
  - [ ] `resident_tree_bit_exact_to_u64_integer_path`,
        `resident_score_within_envelope_of_host_cuda`, and
        `learner_parity_on_device_resident_fast_path_gate` all pass under
        `LGBM_GROW_RANK3=1`.
  - [ ] The Rank-1 (T-06) closed form is UNCHANGED and still passes with
        the hatch UNSET (regression guard — Rank 3 is additive/opt-in).
- **Risks and Guardrails:** per SPEC §9 R-1, these 3 real-device gates are
  the ONLY thing standing between a silent Rank-3 divergence and a merged
  regression — do not mark this task (or Wave 3) complete on a partial
  run; all 3 must be observed passing on real hardware in the same
  session as T-09's changes.

---

## Wave 4 — Perf validation (required before phase completion)

### T-11 — Kaggle CUDA/P100 perf validation of the Wave 1→2 (Rank-1) chain · SPEC-DRGL-11 (dep: T-06)
- **prereqs:** T-00..T-06 all complete and locally ROCm-validated.
  **files:** no production files — a Kaggle notebook/session run using
  `crates/lgbm-treelearner/examples/rocm_drain_profile.rs` (or its
  CUDA-feature equivalent) plus a recorded result (this plan does not
  prescribe a specific output artifact path; follow
  `[[kaggle-bench-workflow]]`'s established log-fetch convention).
- **1. Red (local free-run-vs-drain pre-check — the SPEC §9 R-3 gate).**
  Before spending a Kaggle session, confirm locally on real gfx1152 that the
  deferral moves FREE-RUN wall time, not just a drain bucket. Run
  `rocm_drain_profile.rs` four ways and diff — `LGBM_GROW_DEFER_SYNC` OFF vs
  ON, each with and without `LGBM_GROW_DRAIN`:
  ```bash
  # drain (ranks phases) vs free-run (prices the wall), flag OFF then ON:
  for defer in 0 1; do for drain in 1 ""; do \
    LGBM_GROW_DEFER_SYNC=$defer LGBM_CUDA_ON_DEVICE=1 LGBM_PHASE_PROF=1 ${drain:+LGBM_GROW_DRAIN=1} \
    cargo run --release -p lgbm-treelearner --features rocm --example rocm_drain_profile; done; done
  ```
  The "~23%" (pick ~13% + partition ~10%) is a DRAIN bucket that **bundles
  device compute with the blocking readback** (SPEC §9 R-3) — the
  sync-deferrable share is strictly `< 23%`. Only proceed to the Kaggle run
  if the flag-ON free-run wall is measurably lower than flag-OFF (or the
  decomposition otherwise justifies the P100 spend). Also confirm no prior
  Kaggle run exists for this flag-ON arm (genuine baseline vs treatment).
- **2. Green:** Run the established Kaggle P100 protocol
  (`[[kaggle-bench-workflow]]` account/log-fetch/embed-patch mechanics,
  reused verbatim per "Do Not Hand-Roll") as a **same-session runtime
  toggle**: drive `rocm_drain_profile.rs` (or its CUDA equivalent)
  order-alternated (A/B/B/A) for a warm-median-of-3 wall-time comparison of
  arm (a) `LGBM_GROW_DEFER_SYNC` OFF (two-separate-reads baseline) vs arm (b)
  `LGBM_GROW_DEFER_SYNC=1` (the Wave-2 fusion) on the SAME session. Capture
  the COUNTS ledger for both arms (`partition_resident=`, `role_assign=`,
  `deferred_read_fused=` tripwires) as positive proof each arm ran its
  intended path — `deferred_read_fused=` MUST fire on (b) and NOT (a).
  Compare predictions: structural bit-identity check (CPU-anchor-vs-CUDA)
  plus the existing `resident_score_within_envelope_of_host_cuda`-style
  numeric envelope for the CUDA-vs-CUDA A/B.
  Command reference (adapt to the CUDA feature/session per
  `[[kaggle-bench-workflow]]`):
  ```bash
  LGBM_CUDA_ON_DEVICE=1 LGBM_PHASE_PROF=1 LGBM_GROW_DRAIN=1 \
    cargo run --release -p lgbm-treelearner --features cuda --example rocm_drain_profile
  ```
- **3. Refactor:** N/A (no production code in this task).
- **4. Verify:** confirm the recorded result includes: (a) the
  order-alternated warm-median-of-3 wall-time ratio, (b) the counts-ledger
  positive-tripwire proof for both arms, (c) the preds-comparison result.
  Confirm: AS-4 (SPEC.md §6) is met — the result is RECORDED, whatever its
  verdict (pass/fail/inconclusive per SPEC-DRGL-11's error-handling
  clause).
- **Implementation Steps:**
  1. Confirm T-00..T-06 are complete and their real-device gates pass.
  2. Run the local free-run-vs-drain pre-check (Red); proceed only on a
     flag-ON free-run win.
  3. Set up the Kaggle P100 session per `[[kaggle-bench-workflow]]`.
  4. Run the order-alternated warm-median-of-3 A/B (flag OFF vs ON, same
     session).
  5. Capture and record the counts ledger + preds comparison + wall-time
     verdict.
- **Completion Criteria:**
  - [ ] The local free-run-vs-drain decomposition is recorded (justifies or
        vetoes the Kaggle spend).
  - [ ] A wall-time verdict (with measured ratio) is recorded for
        `LGBM_GROW_DEFER_SYNC=1` vs OFF.
  - [ ] The counts ledger shows `deferred_read_fused=` on arm (b) and its
        ABSENCE on arm (a).
  - [ ] The preds comparison result (bit-identical or within envelope) is
        recorded.
  - [ ] No default-flag flip (`LGBM_GROW_DEFER_SYNC` default-ON) happens as
        part of this task — that is an explicit, separate follow-up decision
        per SPEC-DRGL-11's non-goals.
- **Risks and Guardrails:** per research §16 "Mistaking 'reduces
  drain-ledger bucket' for 'reduces wall time'" — always compare FREE-RUN
  wall time, not just the drained per-phase bucket; an inconclusive or
  regressing result is a valid, complete outcome for this task (it does
  not block marking T-11 done — it blocks a LATER default-flip decision,
  which is out of this plan's scope).

---

## Execution order & parallelism

```text
T-00 -> T-01 -> T-02 -> T-03 -> T-04 -> T-05 -> T-06 -> T-11        (live plan)
                                              (T-07..T-10 DROPPED)
```

1. **T-00** — strictly first; blocks every other task (P-2).
2. **Wave 1 (T-01 → T-02 → T-03)** — strictly sequential; T-02 depends on
   T-01's widened buffer, T-03 depends on both.
3. **Wave 2 (T-04 → T-05 → T-06)** — strictly sequential; T-04 is
   isolable/testable independent of the live driver, but T-05 depends on
   it being correct first (and adds the default-OFF `LGBM_GROW_DEFER_SYNC`
   flag). T-06 must follow T-05 (closed form is derived FROM the shipped
   flag-ON pattern).
4. **~~Wave 3 (T-07..T-10)~~ [DROPPED 2026-07-14]** — not built. Rank 3 is
   out of scope; the former Wave-3 human checkpoint (PROJECT.md R-0
   reconciliation) is moot because the conflicting capability is no longer
   being built.
5. **Wave 4 (T-11)** — depends on T-00..T-06 being locally ROCm-validated;
   begins with the local free-run-vs-drain pre-check (the SPEC §9 R-3 gate,
   relocated here from the dropped Wave-3 checkpoint) before the Kaggle P100
   A/B of `LGBM_GROW_DEFER_SYNC` OFF vs ON.

No task in this plan is parallelizable with another in a DIFFERENT wave
(each wave's capstone task depends on the whole prior wave). WITHIN Wave 1,
T-01 could theoretically start its Red step in parallel with drafting T-02's
plain-Rust reference function, but Green/Verify remain strictly sequential
(T-02's Green needs T-01's Green complete). Treat all tasks as sequential
in practice given the small task count and the depth of the dependency
chain — do not attempt tool-level parallel execution across waves.

## Definition of done (per wave)

- All Red tests turned Green; Refactor left behavior unchanged (byte-
  identity/structure-parity gates are the primary evidence, per this
  phase's "no FP reorder" contract, SPEC §7 reasoning).
- Every real-device (`--features rocm`) gate cited in that wave's tasks
  has been ACTUALLY RUN and observed passing on real hardware this
  session (not merely compiled) — P-3.
- Commit message(s) record the dependency confirmation (P-0, AGENTS.md
  rule 3) and cite the SPEC-DRGL IDs closed.
- Wave 2 additionally requires: the `LGBM_GROW_DEFER_SYNC` flag defaults OFF
  and the driver's flag-OFF behavior is byte-identical to pre-phase; the
  parity gates pass in BOTH flag states.
- Wave 4 additionally requires: the local free-run-vs-drain pre-check plus a
  recorded Kaggle result (any verdict) — the phase is not "done" without it,
  per the retained "P100 verdict before default-ON" gate.

## Rollback / compatibility notes

- T-00 is a pure git-history change — revert = `git revert` the single
  commit (or leave it; it is orthogonal to every later task per research
  §3).
- Wave 1 (T-01..T-03) changes `DeviceLeafSplits`'s internal layout and the
  role-assignment source — revert = restore the per-leaf-id
  `read_leaf`/`write_leaf` API and the host `left_count < right_count`
  branch; no public API or model-format change to roll back.
- Wave 2 (T-04..T-06) is additive/opt-in: the sync-timing change is behind
  `LGBM_GROW_DEFER_SYNC` (default OFF) — revert = delete the flag branch and
  `read_deferred_split_and_pick`, restoring the two separate `bump_sync()`
  sites; the flag-OFF path was never altered, so the default is unaffected.
- **~~Wave 3 (T-07..T-10)~~ [DROPPED]** — not built; nothing to roll back.
- Wave 4 (T-11) has no production code to roll back — it is a
  measurement/record only.
