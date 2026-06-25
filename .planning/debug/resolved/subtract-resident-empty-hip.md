---
slug: subtract-resident-empty-hip
status: resolved
trigger: "the broken subtract_resident hip tests"
created: 2026-06-26
updated: 2026-06-26
resolved: 2026-06-26
fix_commit: 8aed100
---

## Resolution

**root_cause:** Phase-12 co-pack (`4bb7da7`) deferred the smaller child's scan past
`subtract_resident`. On the FUSED directly-built path the smaller "scan" IS its histogram
build+store (`build_fix_scan_resident` is the only thing that puts the smaller Handle into
`smaller_slot`), so the larger child's `subtract_resident` (parent − smaller) ran BEFORE the
smaller histogram existed → `mirror[smaller_slot]==None` → "subtract_resident: smaller slot is
empty". Co-pack never fires on the fused path (gate requires `!smaller_fused`), so the deferral
bought it nothing and only broke it. The hip RESIDENT-test failure was a knock-on test-isolation
artifact: the fused test panicked before restoring `LGBM_FUSED_FORCE`, leaking it into the
resident test.

**classification:** latent REAL product bug in the fused path (OFF by default,
`FUSED_MAX_NUM_DATA=-1`; only engaged via `LGBM_FUSED_FORCE`) — the subtract trick was broken for
any workload routed to fused. Resident-test breakage = separate test-isolation artifact.

**fix (commit `8aed100`):**
- `crates/lgbm-treelearner/src/learner.rs`: when `smaller_fused`, run the smaller child's
  build_fix_scan EARLY (before the larger subtract) and reuse those records at the deferred scan
  site — restores the pre-Phase-12 (260608-t3t) build→subtract order for the fused case only.
  Resident scan-only deferral + co-pack path byte-unchanged.
- `crates/oracle-harness/tests/learner_parity.rs`: a shared `FORCE_ENV_LOCK` mutex serializes the
  two force-env tests; `LGBM_*_FORCE` restored before `.expect` so a failing train can't leak.

**verification (orchestrator-reconfirmed, real gfx1100):** `hip::learner_parity_{resident,fused}_equals_host_tree_on_hip`
2/2 PASS parallel AND serial (was 0/2). `lgbm-treelearner --lib` 77/0; `oracle-harness` (default
CPU merge gate) all pass; CPU f64 anchor byte-untouched.

**files_changed:** `crates/lgbm-treelearner/src/learner.rs`, `crates/oracle-harness/tests/learner_parity.rs`
---

# Debug Session: subtract-resident-empty-hip

## Symptoms

**Expected behavior:**
The two hip full-train parity tests pass — a GPU resident/fused tree trains and matches the host
tree / CPU f64 anchor structurally (per `assert_gpu_tree_matches_cpu_anchor`):
- `hip::learner_parity_resident_equals_host_tree_on_hip`
- `hip::learner_parity_fused_equals_host_tree_on_hip`
(Per memory [[def-f8u-01-flaky-resident-hip-test]], these were pinned to the CPU f64 anchor and
12/12 green after fix fw1 commit `d82611b` — so they passed historically.)

**Actual behavior:**
Both tests PANIC during training (not a parity mismatch — they never reach the assertions):
```
resident train ok: Compute(Runtime { detail: "subtract_resident: smaller slot is empty" })
  panic at crates/oracle-harness/tests/learner_parity.rs:2116
fused train ok: Compute(Runtime { detail: "subtract_resident: smaller slot is empty" })
  panic at crates/oracle-harness/tests/learner_parity.rs:2204
```
`test result: FAILED. 0 passed; 2 failed` (the `.expect("resident train ok")` / `.expect("fused
train ok")` unwrap the Err).

**Error messages:**
`Compute(Runtime { detail: "subtract_resident: smaller slot is empty" })`
Raised at `crates/lgbm-compute/src/lib.rs:2470` — inside `RocmBackend::subtract_resident`, when
`self.resident_pool.borrow().get(smaller_slot)` returns `None` (the smaller child's histogram
Handle is NOT resident at that slot). Sibling guards: `:2465` "parent slot is empty",
`:2535` "scan_resident_siblings: smaller slot is empty" (the co-pack path has the same check).
NOTE: "slot is empty" = the pool's `Option<Handle>` at that index is `None` (histogram never
stored / was cleared), which is DISTINCT from "the leaf has 0 rows".

**Timeline:**
Discovered 2026-06-26 during spike-035 (routing rocm partition to host) when these were run as a
candidate parity gate. Confirmed PRE-EXISTING on master and ENV-INDEPENDENT: both fail with
`LGBM_ROCM_HOST_PARTITION` ON and OFF (with the env off, `prefers_host_partition()` returns the
trait default false = original master behavior), so the spike-035 gate did NOT cause it. Memory
says they were 12/12 green after fw1 (`d82611b`), so a change LANDED BETWEEN fw1 and now
regressed them — strong git-bisect candidate (Phase 12 co-pack and the resident-pool slot
lifecycle are prime suspects; the co-pack path at `:2535` shares the same empty-slot guard).

**Reproduction:**
```
cargo test -p oracle-harness --features rocm --test learner_parity hip:: 2>&1 | tail -25
```
on the local ROCm GPU (spoofed 8-CU gfx1152 APU). Deterministic (0/2 pass, multiple runs).
Test shape: `spine_corpus(3000, 8, 48)` (3000 rows × 8 features × 48 bins), `num_leaves=31`,
`LGBM_RESIDENT_FORCE` set to force the resident path on small data.

## Initial suspect (hypothesis to test, not a conclusion)

A resident-pool SLOT-LIFECYCLE bug: under a deep tree (31 leaves) on small data (3000×8), the
growth loop calls `subtract_resident(parent, smaller, larger)` for a split whose `smaller_slot`
has no Handle stored. Candidate causes to investigate:
1. **Regression bisect:** `git bisect` between fw1 (`d82611b`, last-known-green) and HEAD,
   running the repro, to pinpoint the commit that introduced the empty-slot. Phase 12 co-pack
   (`scan_resident_siblings`, `4bb7da7`/`73a56e7`/`1f49650`) and any resident-pool slot
   alloc/free change are the prime suspects.
2. **Slot assignment vs build order:** does the smaller child get a pool slot assigned but its
   histogram never built/stored before the subtract — e.g. an empty/degenerate smaller leaf
   (0 or sub-`min_data_in_leaf` rows) where the build is skipped but the subtract still fires?
   The co-pack eligibility gate (`learner.rs:1788`) and the `smaller_scannable`
   (`sum_hessian>0 && num_data>0`) condition may interact: if the smaller child is skipped as
   non-scannable but the larger child still requests a subtract against it.
3. **Is it a real product bug or a test-only forcing artifact?** `LGBM_RESIDENT_FORCE` pushes
   tiny leaves onto the resident path that production routing (size gate, 260608-s2b "Lever B
   num_data gate") would never send there. Determine whether production training can hit this
   (real bug → fix the path) or only the forced test can (test bug → fix the test/guard).
   Spike-035's 5000-row example trained the resident path FINE; the bench small (2000×12) too —
   so the trigger is specific (deep tree starving a child on this exact shape).

## Constraints (project hard rules)

- The CPU f64 anchor is the bit-exact merge gate — must stay untouched. The fix is on the
  ROCm/resident path or the test.
- Requires `--features rocm` + the real GPU to reproduce/verify.
- Do NOT git-add the untracked reference trees (`LightGBM*/`, `cuml-main/`, `.serena/`).
- Run on the main tree (master), NOT a worktree — the oracle gate needs the untracked
  `LightGBM/` ref tree + `lib_lightgbm` 4.6 (worktrees break for it; CONVENTIONS hw2/j1l).

## Current Focus

reasoning_checkpoint:
  hypothesis: "Phase 12 co-pack (4bb7da7) DEFERRED the smaller child's scan to AFTER
    subtract_resident. On the FUSED path the smaller scan IS the smaller histogram BUILD
    (build_fix_scan_resident stores the Handle into smaller_slot). So subtract_resident now
    runs BEFORE the smaller histogram exists → mirror[smaller_slot]==None → 'smaller slot is
    empty'. The resident-test failure is a SECONDARY effect: the fused test panics at
    .expect('fused train ok') BEFORE its remove_var(LGBM_FUSED_FORCE), leaking the env var;
    the resident learner then computes fused_eligible=true and takes the same broken path."
  confirming_evidence:
    - "FUSED test ALONE fails deterministically 3/3 (its own broken path)."
    - "RESIDENT test ALONE passes deterministically 6/6 (the non-fused build_resident_leaf_into
      stores smaller_slot BEFORE subtract — DBG trace shows clean slots 0..28, all build OK)."
    - "Both tests --test-threads=1: BOTH fail (fused panics first, leaks LGBM_FUSED_FORCE=1
      because the panic skips remove_var, then resident reads the leaked var → fused path)."
    - "Co-pack gate (learner.rs:1785) requires !smaller_fused → co-pack NEVER fires on fused,
      so the deferral provides zero benefit on the fused path and only breaks it."
    - "scan_leaf_histogram fused_build branch (learner.rs:2486-2528) is the ONLY thing that
      stores the smaller Handle into the slot; it runs at the deferred scan site (1865), AFTER
      subtract (1652)."
  falsification_test: "Running the smaller fused scan BEFORE the larger subtract (restoring the
    pre-Phase-12 build→subtract order) must make the fused test pass; if it still errored,
    the build-store-vs-subtract ordering would not be the cause."
  fix_rationale: "When smaller_fused, run the smaller child's build_fix_scan_resident BEFORE the
    larger subtract (un-defer it for the fused case only). This restores the 260608-t3t order
    the fused path was validated under. Non-fused/resident path keeps the Phase-12 deferral +
    co-pack unchanged (co-pack excludes fused anyway). Also harden the two tests so a panicking
    train cannot leak its force env var to the sibling test (RAII restore)."
  blind_spots: "After fixing the fused path, the parallel-run env-var race (resident concurrently
    seeing FUSED_FORCE=1) could still flip resident onto the fused path — but that path is now
    correct and still pins to the anchor, so it should stay green. Verify with the full parallel
    run + the bit-exact CPU gate."
next_action: remove DBG instrumentation; un-defer the smaller fused scan (run before subtract);
  add RAII env-var restore to both hip tests; re-run hip:: parallel + serial + cpu/oracle gate.

## Evidence

- timestamp 2026-06-26: error originates `lib.rs:2470` (`subtract_resident`, smaller_slot Handle
  is None); env-independent; deterministic 0/2. Test shape spine_corpus(3000,8,48), leaves=31,
  LGBM_RESIDENT_FORCE. Spike-035 5000-row + bench 2000×12 resident trains succeed → shape-specific.
- timestamp 2026-06-26: ISOLATION RESULTS (DBG-instrumented, real gfx1100). FUSED test ALONE
  FAILS 3/3. RESIDENT test ALONE PASSES 6/6 (DBG trace: clean slots 0..28, every smaller build
  "build OK", buildable always true — min_data=5, hess=1). Both serial (--test-threads=1) BOTH
  FAIL. => The "shape-specific" framing was wrong; the trigger is the FUSED path + cross-test
  env-var leakage, NOT the deep-tree shape per se. The deep tree just guarantees ≥1 subtract.
- timestamp 2026-06-26: MECHANISM. fused path: smaller_fused=true → build deferred to scan
  (learner.rs:1561 sets smaller_resident_slot=Some(slot) with NO build). Phase-12 moved the
  smaller scan to 1865 (after subtract at 1652). The fused scan (build_fix_scan_resident,
  2514) is the ONLY store of the smaller Handle. So subtract reads an empty slot. Co-pack gate
  (1785) requires !smaller_fused, so fused never co-packs → deferral is pure loss for fused.
- timestamp 2026-06-26: ENV LEAK. fused test panics at learner_parity.rs:2204
  (.expect("fused train ok")) BEFORE remove_var(LGBM_FUSED_FORCE) at 2205 → var stays set →
  next/concurrent resident learner computes fused_eligible=true (resident_pool.rs:254) → takes
  the broken fused path → fails too. Explains the serial both-fail + parallel both-fail.

## Resolution

root_cause: Phase 12 co-pack (commit 4bb7da7) deferred the smaller child's scan past
  subtract_resident; on the FUSED path that scan is also the smaller histogram BUILD/store, so
  the larger child's subtract derives from an empty smaller slot. The resident test fails only
  as a knock-on of the fused test leaking LGBM_FUSED_FORCE on its pre-remove_var panic.
fix: (1) PRODUCT — in `SerialTreeLearner::find_best_splits` (learner.rs) un-defer the smaller
  child's fused scan: when `smaller_fused`, run `scan_leaf_histogram` (the build_fix_scan_resident
  build+store+scan) into `smaller_records_early` BEFORE the larger build/subtract block, and reuse
  those records at the Phase-12 deferred scan site. Restores the pre-Phase-12 (260608-t3t)
  build→subtract order. Co-pack still excludes fused (gate requires !smaller_fused), so the
  resident scan-only deferral + co-pack path is byte-unchanged. (2) TEST — serialize the two hip
  force-env tests with a shared `FORCE_ENV_LOCK` mutex and restore `LGBM_*_FORCE` BEFORE `.expect`
  so a failing train cannot leak its force var into a sibling test (the panic-before-remove_var
  leak that turned the resident test failure into a knock-on of the fused failure).
  CPU f64 anchor untouched (fix is on the ROCm/fused path + the test only).
verification:
  - hip:: (both tests) PASS parallel AND serial, 5/5 deterministic (was 0/2).
  - lgbm-treelearner --lib (default) 77 passed; (--features rocm) 77 passed.
  - oracle-harness (default, bit-exact CPU gate): ALL PASS (no nonzero-failure lines).
  - oracle-harness --features rocm (full hip + kernel parity suite): ALL PASS.
files_changed:
  - crates/lgbm-treelearner/src/learner.rs (un-defer smaller fused scan)
  - crates/oracle-harness/tests/learner_parity.rs (FORCE_ENV_LOCK + panic-safe env restore)
classification: latent REAL product bug in the fused path (currently OFF by default,
  FUSED_MAX_NUM_DATA=-1, engaged only via LGBM_FUSED_FORCE) — the subtract trick was broken for
  ANY workload routed to fused; the fix makes the path correct for any future enablement. The
  resident-test failure was a TEST-ISOLATION artifact (env-var leak on the fused panic).

## Eliminated

- hypothesis: caused by spike-035 `LGBM_ROCM_HOST_PARTITION` gate — ELIMINATED (fails with env
  off = master behavior).
