# Phase 21: Harden the On-Device Driver (re-cut) - Research

**Researched:** 2026-07-02
**Domain:** On-device (cubecl) tree-growth driver hardening + STRUCTURE parity gate breadth + requirement/ROADMAP bookkeeping (pure-Rust, no new deps)
**Confidence:** HIGH (all findings verified against committed source, git history, and phase artifacts in-repo)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Phase 21 becomes a **hardening/parity-slack** phase for the on-device driver. Its original ODL-18/ODL-19 driver-integration scope is accepted as **delivered by Phase 20**. No re-implementation of the driver — strengthen it.
- **D-02: Targeted risk cases, not a broad sweep.** Add a small, deliberate set of STRUCTURE-bit-exact cases each pinned to the cpu f64 anchor (tie-aware `default_left`, leaf values within ~1e-5):
  - a **deep tree with >2 simultaneously-live leaves**;
  - a **no-split / single-leaf** tree (`best_leaf == −1` break path);
  - a **`min_data_in_leaf` / `min_sum_hessian_in_leaf`-constrained** case.
- **D-03: Fix + dedicated repro test.** Replace the `(parent_slot + 1) % num_slots` heuristic in `HistArena::swap` (`histogram_arena.rs:365`) with a **free-slot scan against `leaf_to_slot`** (drop the now-internal `parent_leaf` key) per the 18-REVIEW.md WR-01 fix sketch. Add a **regression test that constructs a >2-live-leaf grow scenario proven to alias under the old heuristic** and asserts no slot corruption.
- **D-04: cubecl-cpu is the gate; ROCm is a best-effort smoke.** The merge gate stays the deterministic **cubecl-cpu f64 anchor**. A real-ROCm run — if attempted — is pinned to the cpu anchor (def-f8u-01, never GPU-vs-GPU) and is **informative, not blocking**.
- **D-05: Mark done + new hardening requirement ID.** Mark **ODL-18 and ODL-19 Complete** (delivered Phase 20) in REQUIREMENTS.md and its traceability table. Add a **new hardening requirement (e.g. `ODL-18H`)** covering the targeted corpus + WR-01 fix + repro, mapped to Phase 21. Re-cut the ROADMAP Phase 21 **body** (Goal / Success Criteria / Notes) via `/gsd-phase`. This bookkeeping reconciliation is **in-scope for the phase plan**.

### Claude's Discretion
- Exact fixture parameters for the targeted corpus (row counts, feature counts, `num_leaves`, threshold values) — pick the smallest configs that provably (a) yield >2 live leaves at once and (b) trigger each constraint/edge branch.
- Whether the WR-01 repro lives as a `lgbm-compute` unit test or an oracle-harness case — pick whichever most directly demonstrates the aliasing.

### Deferred Ideas (OUT OF SCOPE)
- **Wire on-device `EvalKernel` metric eval into GBDT** (replace host-side eval over the mirrored score) — later slack/follow-up phase.
- **Full rows×features×leaves×constraints parity sweep** — rejected under D-02 in favor of targeted cases.
- **On-device categorical splits** → Phase 22 (ODL-22).
- **Perf-validation / default-ON rollout DoD** (Kaggle A/B, `device_launches`, wall-clock ratio, default flip) → Phase 23 (ODL-20/ODL-21).
- Five perf-campaign todos (large-data fixture, GPU loop profiling, GPU/CPU crossover, low-row A/B, LDS atomic lowering) → Phase 23.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODL-18 | On-device single-GPU tree-learner driver: per-leaf grow loop end-to-end on device (root init → build/subtract → best-split → tree split → partition, up to `num_leaves−1`, break on `best_leaf == −1`), reconstituting `(Tree, DataPartition)`; STRUCTURE bit-exact to cpu f64 anchor (tie-aware `default_left`), leaf values ~1e-5. | **Already DELIVERED in Phase 20** (crit 5, `grow_tree_on_device_driver`). D-05: mark **Complete**. This research documents the delivered surface so the planner records the reconciliation, not a rebuild. |
| ODL-19 | Every new kernel keeps f32 + u64 fixed-point build, no f64 per-row hot loops; CPU/ROCm/host-CUDA byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset. | **Already DELIVERED in Phase 20** (crit 6). D-05: mark **Complete**. Driver's only per-row device work is the Phase-16 build + partition kernels; f64 confined to O(num_bin) scalar fix/compact/gain. |
| ODL-18H *(new, D-05)* | Hardening: targeted STRUCTURE parity corpus (deep >2-live-leaf, no-split, min-data/min-hessian-constrained) + WR-01 slot-aliasing fix + repro test, all anchored to cpu f64 fold. Mapped to Phase 21. | This document. Note WR-01 code fix + repro are **already landed** (see §WR-01 Status); the net-new work is the parity corpus breadth (§D-02) and the bookkeeping (§D-05). |
</phase_requirements>

## Summary

Phase 21 is a **hardening re-cut**, not a build. The on-device grow driver it targets already exists and was verified in Phase 20 (VERIFICATION.md `passed`, 6/6, `requirements_verified: [ODL-16, ODL-17, ODL-18, ODL-19]`). Three deliverables remain: (1) the WR-01 `HistArena::swap` slot-aliasing fix, (2) a small **targeted** STRUCTURE parity corpus, and (3) requirement/ROADMAP bookkeeping.

**Two upstream framings in CONTEXT.md are factually stale and must be reconciled at plan time — this is the single most important research finding:**

1. **WR-01 is already fixed and tested.** The `(parent_slot + 1) % num_slots` heuristic was replaced by the exact free-slot scan the CONTEXT describes, in commit **`c9a7fd1` ("fix(18): WR-01 pick free non-aliasing slot in HistArena::swap")** during Phase 18's review-fix cycle. The dedicated repro test (`swap_multileaf_never_aliases_live_sibling_slot`) and the pool-exhaustion test (`swap_errors_when_pool_exhausted`) are **already present and passing** in `histogram_arena.rs`. `[VERIFIED: git log + source read]`

2. **The live driver does NOT consume `HistArena::swap`.** `grow_tree_on_device_driver` keeps each leaf's histogram in its own `DriverLeaf.hist: Vec<f64>` and derives the larger child with `subtract_histograms_f64_on` into a fresh `Vec`. It never imports or calls `HistArena` (verified: `HistArena` appears in `grow_driver.rs` only in a doc-comment). Therefore the CONTEXT's claim that "the multi-leaf grow loop is the first live consumer of `HistArena::swap`" and the `<specifics>` "linchpin ... would alias under the old heuristic" claim **do not hold**. The `>2-live-leaf` parity corpus case broadens parity evidence (valuable) but does **not** exercise the WR-01 swap path — that path is exercised only by the already-present `HistArena` unit tests. `[VERIFIED: grep + source read]`

**Primary recommendation:** Plan Phase 21 as **verify-WR-01 + broaden-parity + bookkeeping**, explicitly decoupling the two WR-01 purposes: (a) the parity corpus proves the driver is bit-exact at depth; (b) the *already-landed* `HistArena` repro test locks the slot-aliasing bug closed. Treat the min-data/min-hessian corpus case as the one item needing a small design decision (the driver hardcodes its `GainConfig` — see §D-02.C). All work stays additive, gated by `LGBM_CUDA_ON_DEVICE`, anchored to the cubecl-cpu f64 fold, never GPU-vs-GPU. No new dependencies.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| On-device grow loop orchestration | `lgbm-compute` (`grow_driver.rs`) | — | Kept in lgbm-compute to avoid the `treelearner → compute → treelearner` crate cycle (D-01/Option A). Uses only lgbm-compute-reachable types (`GrowFeature`, `BinColumn`, lgbm-dataset enums). |
| Per-leaf histogram storage | `lgbm-compute` driver (`DriverLeaf.hist: Vec<f64>`) | — | Driver owns per-leaf host-side `Vec<f64>` histograms; NOT the `HistArena` slot pool. No cross-leaf slot aliasing surface in the live path. |
| Histogram slot-pool (`HistArena`) | `lgbm-compute` kernels | (currently dead w.r.t. driver) | Exists + WR-01-fixed + unit-tested, but not wired into the driver. A future perf refactor (Phase 23) could adopt it; parity hardening does not require it. |
| STRUCTURE parity gate | `oracle-harness` (`learner_parity.rs`) | — | Hosts `learner_parity_on_device_structure_gate` + the tie-aware cpu-f64-anchor comparator. New corpus cases plug in here. |
| cpu f64 anchor | `lgbm-treelearner::SerialTreeLearner` via `CpuBackend` | — | The deterministic reference the on-device tree pins to (never a second GPU f32 path). |
| Requirement/ROADMAP bookkeeping | `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md` | `/gsd-phase` | D-05 reconciliation surface. |

## Standard Stack

No external packages. All work is internal to the existing Cargo workspace.

### Core (existing, in-repo — reuse, do not add)
| Component | Location | Purpose | Why Standard |
|-----------|----------|---------|--------------|
| `grow_tree_on_device_driver` | `crates/lgbm-compute/src/kernels/grow_driver.rs:418` | The end-to-end per-leaf grow loop (the thing being hardened). | Delivered + verified Phase 20; own `DriverLeaf` bookkeeping, no crate cycle. |
| `HistArena` (+ `swap`, `rotate`) | `crates/lgbm-compute/src/kernels/histogram_arena.rs` | Histogram slot-pool + subtraction-trick handle rotation. WR-01 fix lives here. | Already fixed (c9a7fd1) + fully unit-tested; NOT wired into the driver. |
| `learner_parity_on_device_structure_gate` | `crates/oracle-harness/tests/learner_parity.rs:2316` | The env-gated STRUCTURE parity gate. Extend with D-02 cases. | Existing gate; tie-aware comparator already present. |
| `assert_on_device_tree_matches_cpu_anchor` | `learner_parity.rs:2239` | Tie-aware comparator (structure bit-exact; leaves ~1e-5; `default_left` flip only on genuine `split_gain` near-tie). | Reusable verbatim for every new corpus case. |
| `cpu_anchor_tree` | `learner_parity.rs:2290` | Builds the deterministic cpu f64 anchor via `SerialTreeLearner` + `CpuBackend`. | The bit-exact reference; parameterized by `GainConfig`. |
| `grow_features_of` | `learner_parity.rs:1969` | `FeatureColumn` → `GrowFeature` field-by-field mapper. | Reuse for new corpora. |
| `on_device_proving_corpus` | `learner_parity.rs:2357` | The existing single 4-leaf / 12-row / 2-feature corpus. | The pattern to clone for new cases. |
| `proving_slice_config` | `grow_driver.rs:214` | The driver's hardcoded L2 continuous config (min_data=1, min_hessian=0). | The config the anchor must match — and the constraint for D-02.C (see below). |

**Installation:** None. `cargo test -p lgbm-compute` and `cargo test -p oracle-harness` are the existing gates.

## Package Legitimacy Audit

**Not applicable — this phase installs no external packages.** `git diff HEAD --stat -- '**/Cargo.toml'` shows no dependency changes. All work reuses in-repo crates (`lgbm-compute`, `oracle-harness`, `lgbm-treelearner`, `lgbm-model`, `lgbm-dataset`) and the existing `cubecl` stack. `[VERIFIED: git diff]`

## Architecture Patterns

### The live grow loop (what actually runs)

```
grow_tree_on_device_driver(client, g[f32], h[f32], features[GrowFeature], num_leaves, max_depth)
  │
  ├─ proving_slice_config()            ← cfg is HARDCODED here (min_data=1, min_hessian=0)
  │
  ├─ ROOT: ordered f64 fold over all rows → (root_sum_g, root_sum_h)   [the one blessed per-row f64]
  │        build_leaf_hist (Phase-16 construct_histograms_f64_on + FixHistogram + compact)
  │        scan_leaf (Phase-4/17 find_best_split_f64_on + cross-feature split_gt argmax)
  │        leaves = vec![ DriverLeaf{ rows, sum_g, sum_h, hist:Vec<f64>, best, best_fpos, depth } ]
  │        DeviceCudaTree::new + add_bias(root_output)
  │
  └─ for _split in 0..(num_leaves-1):                        ← best-first leaf-wise loop
        best_leaf = argmax over leaves (split_gt tie rule: gain>, then lower real feature idx)
        if best_fpos < 0 || !(best.gain > 0.0) { break }     ← THE no-split / best_leaf==-1 PATH (:511)
        partition_leaf_stable  (Phase-18 route)  → left_rows / right_rows
        tree.split_on_device   (Phase-18 DeviceCudaTree mutation) → right_leaf_index
        seed children sums from SplitInfo (NOT a re-fold; kEpsilon-carrying)
        smaller child: build_leaf_hist directly ; larger child: subtract_histograms_f64_on(parent - smaller)
        BeforeFindBestSplit gates (both_too_small via min_data ; max_depth) → scan each child
  │
  └─ to_host_tree + reconstruct LeafPartitionLayout from per-leaf DriverLeaf.rows
```

**Key structural facts for the planner:**
- **All leaves stay live** in `leaves: Vec<DriverLeaf>` for the whole loop (best-first never frees). Any `num_leaves ≥ 4` therefore has ≥3 leaves live simultaneously at the last split. The existing 4-leaf corpus already reaches 3-live; a `num_leaves = 6–8` corpus makes ">2 live" unmistakable. `[VERIFIED: source read grow_driver.rs:470-655]`
- **No `HistArena`, no resident slot pool, no data→leaf ping-pong buffer in the live path.** Per-leaf histograms are `Vec<f64>`; the final `LeafPartitionLayout` is rebuilt from per-leaf `rows` (`grow_driver.rs:658-675`). The `build_leaf_map_on` / `LeafMapBufferStrategy` A/B helper (`grow_driver.rs:135`) is a **standalone oracle harness**, not called by the driver. `[VERIFIED: grep]`

### The gated STRUCTURE gate pattern (how it runs non-vacuously without ROCm)

```rust
// learner_parity.rs:2316 (pattern to replicate for each D-02 case)
let env_on = std::env::var("LGBM_CUDA_ON_DEVICE").as_deref() == Ok("1");
let grown = backend.grow_tree_on_device(&g, &h, &grow_features, num_leaves, max_depth)?;
if env_on {
    let (on_device_tree, layout) = grown.expect("Ok(Some) when env set");
    let anchor = cpu_anchor_tree(&features, &g, &h, cfg, num_leaves, max_depth);
    assert_on_device_tree_matches_cpu_anchor(&on_device_tree, &anchor, "on-device");
    // + layout row-conservation asserts
} else {
    assert!(grown.is_none(), "byte-unchanged merge gate: Ok(None) when env unset");
}
```

- The gated flip is `CpuBackend::on_device_growth_supported()` / `cuda_on_device_enabled()` → returns `LGBM_CUDA_ON_DEVICE == "1"` (`lib.rs:2288`). The **default merge-gate run** (env unset) exercises the `Ok(None)` defer path; the **`LGBM_CUDA_ON_DEVICE=1` run** exercises the real driver on the **cubecl-cpu runtime** — so the gate is non-vacuous *without any GPU*. `[VERIFIED: source read]`
- Verification proves non-vacuity via `cargo test ... -- --exact learner_parity_on_device_structure_gate` (the `--exact` confirms the named test actually ran). `[CITED: 20-VERIFICATION.md line 11]`

### Pattern: adding a D-02 corpus case
Clone `on_device_proving_corpus()` → new `fn <case>_corpus()`; add a `#[test]` mirroring `learner_parity_on_device_structure_gate`, reusing `grow_features_of`, `cpu_anchor_tree`, and `assert_on_device_tree_matches_cpu_anchor`. Each is env-gated the same way (Some/None on `LGBM_CUDA_ON_DEVICE`).

### Anti-Patterns to Avoid
- **Do NOT rebuild the driver.** ODL-18/19 are delivered; scope is hardening only (D-01).
- **Do NOT compare two GPU f32 paths** (def-f8u-01). Every candidate pins to the single cpu f64 anchor.
- **Do NOT wire `HistArena` into the driver** to "make WR-01 live." That is a perf/memory refactor (Phase 23 territory) and would risk the bit-exact contract for no parity benefit. The driver's per-leaf `Vec<f64>` is correct and simpler.
- **Do NOT weaken the tie-aware comparator** to make a case pass — a `default_left` flip on a non-tie is a real divergence (comparator already hard-fails it).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tie-aware structure comparison | A new tree-diff | `assert_on_device_tree_matches_cpu_anchor` (`:2239`) | Already handles structure-bit-exact + ~1e-5 leaves + `default_left` near-tie logic. |
| cpu f64 anchor | A fresh reference train | `cpu_anchor_tree` (`:2290`) | Drives the real `SerialTreeLearner` under an explicit `GainConfig`. |
| `FeatureColumn`→`GrowFeature` | Inline field copies | `grow_features_of` (`:1969`) | One maintained mapper. |
| WR-01 slot-scan | A new picker | Already landed `swap` free-slot scan (`histogram_arena.rs:381-392`) | Fixed + tested (c9a7fd1). |
| Slot-aliasing repro | A new aliasing test | `swap_multileaf_never_aliases_live_sibling_slot` (`histogram_arena.rs:637`) | Already provably aliases under the old heuristic (trace below). |

**Key insight:** The parity infrastructure is complete; Phase 21 is 90% *adding fixtures + reconciling docs*, not building machinery.

## WR-01 Status (D-03) — the critical reconciliation

**The fix is landed and correct.** Current `HistArena::swap` (`histogram_arena.rs:347-415`):
```rust
let occupied: HashSet<usize> = self.leaf_to_slot.values().copied().collect();
let fresh = (0..self.num_slots)
    .find(|s| *s != parent_slot && !occupied.contains(s))
    .ok_or_else(|| ComputeError::Runtime { /* pool exhausted */ })?;
// larger inherits parent_slot; smaller takes fresh; drop the now-internal parent_leaf key
```
This is exactly the D-03 sketch (free-slot scan against `leaf_to_slot` + drop parent key). `parent_leaf` remains a *parameter* (needed to look up `parent_slot`) but its *map entry* is dropped (`:406-408`) — "drop the now-internal parent_leaf key" means the occupancy entry, which is done.

**The repro test is present and provably aliases under the old heuristic** (`swap_multileaf_never_aliases_live_sibling_slot`, `:637-678`). Traced: 4-slot pool; `swap(0,1,2)`→leaf1@slot1,leaf2@slot0; `swap(2,3,4)`→leaf3@slot2,leaf4@slot0; then `swap(1,5,6)` with parent leaf1@slot1: NEW code picks `fresh=3` (first free); OLD `(1+1)%4=2` would collide with leaf3@slot2. The test asserts all four live leaves hold distinct slots. `[VERIFIED: trace + source]`

**Planner implication:** D-03's code change and repro are **already satisfied by committed work**. Options for the planner (pick one, record it):
1. **Recommended — verify-only task:** a Phase-21 task that (a) confirms `swap`'s free-slot scan + drop-parent-key are present, (b) confirms both repro tests (`swap_multileaf_never_aliases_live_sibling_slot`, `swap_errors_when_pool_exhausted`) pass, and (c) documents in the plan/SUMMARY that WR-01 was closed in c9a7fd1 (Phase 18) and that the live driver does not consume `HistArena` — so ODL-18H's WR-01 clause is a *confirmation*, not a rewrite. This keeps the honest audit trail.
2. Only if strengthening is desired: add one more repro variant (e.g. a 5-slot / deeper live set, or an assertion that the *driver's* per-leaf `Vec<f64>` model has no aliasing by construction). Low value; the existing test already covers the failure mode.

**Do not** re-apply the fix (it exists) or introduce `HistArena` into the driver to "make it live."

## Runtime State Inventory

> Rename/refactor-adjacent (hardening), but this phase edits code + tests + planning docs only — no persisted runtime state.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastores keyed on any renamed/edited symbol. | None. |
| Live service config | None — no external service holds Phase-21 state. | None. |
| OS-registered state | None. | None. |
| Secrets/env vars | `LGBM_CUDA_ON_DEVICE` gates the on-device path (read at runtime; not persisted). New tests must set/unset it per the existing pattern. `LGBM_RESIDENT_FORCE`/`LGBM_FUSED_FORCE` are ROCm-only (out of the default gate). | Tests read the env; no key renames. |
| Build artifacts | None — no package rename; `Cargo.toml` unchanged. | None. |

**Verified:** No runtime state migration is involved. This is a code+test+docs phase. `[VERIFIED: grep + git diff]`

## Common Pitfalls

### Pitfall 1: Treating the `>2-live-leaf` corpus as the WR-01 exerciser
**What goes wrong:** Planning "one case, two purposes" (parity + WR-01 aliasing) per the CONTEXT `<specifics>`.
**Why it happens:** CONTEXT assumes the driver consumes `HistArena::swap`; it does not (uses per-leaf `Vec<f64>`).
**How to avoid:** Decouple. The parity corpus proves driver depth-correctness; the *already-present* `HistArena` unit repro locks the aliasing bug. Do not assert the corpus "would have aliased."
**Warning signs:** A test comment claiming the driver corpus exercises `swap`.

### Pitfall 2: min_data / min_hessian case can't reach the driver through the current seam
**What goes wrong:** Writing a `min_data_in_leaf`/`min_sum_hessian_in_leaf`-constrained corpus and calling `backend.grow_tree_on_device(...)` — the constraint never binds.
**Why it happens:** The `Backend::grow_tree_on_device` seam has **no `GainConfig` parameter**, and the driver **hardcodes `proving_slice_config()`** (`min_data_in_leaf: 1, min_sum_hessian_in_leaf: 0.0`, `grow_driver.rs:214-225,443`). With min_data=1/min_hessian=0 no data/hessian gate ever activates. `[VERIFIED: source]`
**How to avoid:** See §D-02.C — thread a config into the driver via a `_with_cfg` variant called *directly* by the test (keeps the trait seam and merge gate byte-unchanged). The constraint plumbing already exists downstream (`scan_leaf` passes cfg to `find_best_split_f64_on`; the `both_too_small` gate at `:639` reads `min_data`; the `sum_h > 0` guard at `:366`) — only the *entry point* hardcodes the config.
**Warning signs:** A "constrained" test that grows the identical tree as the unconstrained one.

### Pitfall 3: Anchor config mismatch
**What goes wrong:** The on-device tree and `cpu_anchor_tree` grow under different `GainConfig` → spurious structural divergence.
**Why it happens:** The driver pins its own config; the anchor takes an explicit `cfg` argument.
**How to avoid:** Every gate passes the **identical** `GainConfig` to `cpu_anchor_tree` that the driver used (the existing gate uses `proving_slice_config()` for both — `:2321,2331`). For a `_with_cfg` driver variant, pass that same cfg to the anchor.
**Warning signs:** Structure asserts failing only in the env-on run.

### Pitfall 4: On-device kernel goldens are host re-transcriptions (WR-03 / MEMORY)
**What goes wrong:** Over-claiming the parity gate proves fidelity to compiled `lib_lightgbm`.
**Why it happens:** Phase-18 partition/tree/predict goldens are hand re-transcriptions (`external_libs/` unvendored at capture time). See MEMORY `on-device-kernel-goldens-are-re-transcriptions`.
**How to avoid:** State the anchor precisely: the STRUCTURE gate proves `on-device == cpu SerialTreeLearner`. That anchor *is* independently bit-exact to real `lib_lightgbm 4.6` (CLAUDE.md; the serial learner's `spine_real.txt`/`mfb_pos_real.txt` real-binary goldens), so the chain is strong — but the *kernel-level* Phase-18 goldens remain re-transcriptions. Do not conflate the two claims in the SUMMARY.
**Warning signs:** A claim that a green on-device gate = compiled-library fidelity for the partition/predict kernels.

### Pitfall 5: Vacuous gate without the env var
**What goes wrong:** Running only the default `cargo test` and believing the on-device path was exercised.
**How to avoid:** Every new case must run in **both** lanes: env-unset (asserts `Ok(None)`, byte-unchanged) and `LGBM_CUDA_ON_DEVICE=1 ... -- --exact <test_name>` (asserts the real tree). Mirror the Phase-20 verification commands.

## D-02 Corpus Design (recommendations)

All cases: continuous-feature + `MissingType::None` + L2 (the proving slice); anchor via `cpu_anchor_tree` with the identical cfg; assert via `assert_on_device_tree_matches_cpu_anchor`; env-gated Some/None.

### A. Deep tree, >2 simultaneously-live leaves
- **Params:** 2–3 numeric features, ~16–24 rows, `num_leaves = 6–8`, `max_depth = -1`, monotone-ish distinct-gain gradients (mirror the existing corpus's clean, near-tie-free shape so the assert is a genuine bit-exact check, not a tie-tolerated pass).
- **Why it works:** best-first keeps all leaves live; ≥4 leaves guarantees ≥3 live at the final splits. Smallest clearly-">2" config is `num_leaves ≥ 5`. Exercises repeated subtraction-trick derivation + `split_gt` argmax across many live leaves.
- **Serves:** parity breadth (NOT the WR-01 swap path — see Pitfall 1).

### B. No-split / single-leaf tree (`best_leaf == −1` break path)
- **Params:** either (i) `num_leaves = 1` (loop `for _ in 0..0` never runs → root-bias-only tree), or (ii) a corpus where the root has no admissible positive-gain split so the loop breaks at `grow_driver.rs:511` (`best_fpos < 0 || !(best.gain > 0.0)`). Constant gradient (all g equal, h=1) yields zero gain → immediate break. `[VERIFIED: break at :511]`
- **Why it works:** Directly covers the `break` path the driver shares with C++ `SerialTreeLearner`. The anchor (`SerialTreeLearner`) takes the same no-split path, so the trees match (a 1-leaf tree with just the root output).
- **Note:** the driver seeds the root leaf value via `calculate_splitted_leaf_output` + `add_bias` (`:483-485`), so a never-split root still matches the anchor's leaf value within ~1e-5.

### C. min_data_in_leaf / min_sum_hessian_in_leaf-constrained
- **Blocker:** the seam has no `GainConfig`; the driver hardcodes `proving_slice_config()` (see Pitfall 2). **A design decision is required.** Recommended, lowest-risk option:
  - Add an internal `grow_tree_on_device_driver_with_cfg(client, g, h, features, num_leaves, max_depth, cfg: GainConfig)` in `grow_driver.rs`; make the existing `grow_tree_on_device_driver` delegate to it with `proving_slice_config()`. **The trait seam (`Backend::grow_tree_on_device`) is untouched → merge gate stays byte-unchanged.**
  - The D-02.C test calls `grow_tree_on_device_driver_with_cfg` **directly** on the cubecl-cpu client with a constrained cfg (e.g. `min_data_in_leaf: 4` and/or `min_sum_hessian_in_leaf: <positive>`), and anchors to `cpu_anchor_tree(..., that_same_cfg, ...)`.
  - Corpus: sized so an *unconstrained* split would occur but the constraint forbids it (e.g. a would-be child with < min_data rows, or hessian sum below min_hessian), forcing a different/earlier stop. This makes the constraint *observably* bind (different tree than min_data=1).
- **Alternatives considered:**
  - Add a `GainConfig` to the trait seam — larger blast radius (touches `CpuBackend`/`GpuBackend<R>`/all callers, and the learner call-site `learner.rs:764`); risks the byte-unchanged contract. Reject unless the planner wants a general config-through-seam capability (out of hardening scope).
  - Bump `proving_slice_config()` globally — would change *every* existing gate's anchor. Reject.
- **Confidence:** `[VERIFIED: source]` for the blocker; the `_with_cfg` remedy is the natural minimal change (it is Claude's-discretion territory under CONTEXT: "smallest configs that provably trigger each constraint/edge branch").

## D-05 Bookkeeping Mechanics (exact edit surface)

### REQUIREMENTS.md (`.planning/REQUIREMENTS.md`)
- **Line 50** `- [ ] **ODL-18**: ...` → `- [x]` (mark Complete). `[VERIFIED: grep]`
- **Line 51** `- [ ] **ODL-19**: ...` → `- [x]`. `[VERIFIED: grep]`
- **Add a new ODL-18H requirement** near ODL-18/19 (checklist form), text ≈: *"Hardening of the on-device driver: targeted STRUCTURE parity corpus (deep >2-live-leaf, no-split, min-data/min-hessian-constrained), the WR-01 `HistArena::swap` free-slot fix + repro test, all cpu-f64-anchored, cubecl-cpu default lane; mapped to Phase 21."* Mark `[ ]` (to be checked at phase completion).
- **Traceability table (lines 103-109):** `ODL-18 | Phase 21 | Pending` → `ODL-18 | Phase 20 | Complete`; `ODL-19 | Phase 21 | Pending` → `ODL-19 | Phase 20 | Complete`; add row `ODL-18H | Phase 21 | Pending`. `[VERIFIED: grep lines 105-106]`
- **Per-phase count table (lines 121-124):** update the "Phase 21 — Driver integration + parity gate | ODL-18, ODL-19 | 2" row to reflect the re-cut (Phase 21 → ODL-18H; ODL-18/ODL-19 attributed to Phase 20). Keep counts consistent with the traceability table. `[VERIFIED: grep]`

### ROADMAP.md (`.planning/ROADMAP.md`)
- The **checklist line** already reads: *"Phase 21: Hardening/Slack (was End-to-End Driver — absorbed into Phase 20 per D-01) — ... Re-cut via `/gsd-phase` before planning."* `[VERIFIED: grep]`
- The **Phase 21 body** (Goal / Success Criteria / Notes) is STALE (still the driver-integration text) and must be re-cut via **`/gsd-phase`** to the hardening scope (WR-01 confirmation, targeted parity corpus, bookkeeping; explicitly out: categorical → 22, perf/default-on → 23). This is in-scope for the phase plan (D-05).

### ROADMAP Notes open question (bounded verification per the objective)
The Phase-21/20 Notes flag: *"cubecl 0.10 `Handle` in-place aliasing vs ping-pong double-buffering for the data→leaf map; batched `client.read(vec![h])` readback semantics."* For **this parity-hardening phase**:
- **Data→leaf aliasing:** resolved and MOOT. The A/B (`learner_parity_on_device_buffer_strategy_ab`, `:1997`) already locked **double-buffer** (both strategies matched the cpu anchor; double-buffer chosen as conservative default). More importantly, the **live driver does not use a running data→leaf device buffer at all** — it carries per-leaf `Vec<u32>` row lists on the host and rebuilds `LeafPartitionLayout` at the end (`:658-675`). So there is no in-place-vs-ping-pong parity risk in the grow loop. `[VERIFIED: source]`
- **Batched `client.read(vec![h])`:** a **perf** concern only (MEMORY `gpu-lazy-dispatch...` / `tlk`: no batched-read lever to wire on the confirmed 1-launch/1-read production launchers). **Purely Phase-23** (perf DoD); it does **not** affect the parity gate. `[VERIFIED: MEMORY notes]`
- **Recommendation:** the plan should record "ROADMAP open question does not affect Phase-21 parity hardening: data→leaf aliasing is moot for the live driver; batched read is Phase-23 perf."

## Code Examples

### Env-gated D-02 case skeleton (clone of the existing gate)
```rust
// Source: crates/oracle-harness/tests/learner_parity.rs:2316 (verified pattern)
#[test]
fn learner_parity_on_device_deep_multileaf_gate() {
    let backend = CpuBackend;
    let (features, g, h, num_leaves, max_depth) = deep_multileaf_corpus(); // new: num_leaves 6-8
    let grow_features = grow_features_of(&features);
    let cfg = lgbm_compute::kernels::grow_driver::proving_slice_config();
    let env_on = std::env::var("LGBM_CUDA_ON_DEVICE").as_deref() == Ok("1");
    let grown = backend
        .grow_tree_on_device(&g, &h, &grow_features, num_leaves, max_depth)
        .expect("grow_tree_on_device seam ok");
    if env_on {
        let (tree, layout) = grown.expect("Ok(Some) when env set");
        let anchor = cpu_anchor_tree(&features, &g, &h, cfg, num_leaves, max_depth);
        assert_on_device_tree_matches_cpu_anchor(&tree, &anchor, "deep-multileaf");
        assert_eq!(layout.leaf_count.iter().sum::<i32>(), g.len() as i32);
    } else {
        assert!(grown.is_none(), "byte-unchanged merge gate");
    }
}
```

### min_data case entry point (proposed `_with_cfg` delegation)
```rust
// Proposed: crates/lgbm-compute/src/kernels/grow_driver.rs
pub fn grow_tree_on_device_driver<R: cubecl::Runtime>(/* existing args */)
    -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError> {
    grow_tree_on_device_driver_with_cfg(client, gradients, hessians, features,
                                        num_leaves, max_depth, proving_slice_config())
}
pub fn grow_tree_on_device_driver_with_cfg<R: cubecl::Runtime>(
    /* existing args + */ cfg: GainConfig,
) -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError> {
    // body identical, replacing `let cfg = proving_slice_config();` with the param.
    // Trait seam Backend::grow_tree_on_device stays unchanged → merge gate byte-unchanged.
}
```

## State of the Art

| Old Framing (CONTEXT/MEMORY) | Current Reality (verified this session) | Evidence | Impact |
|------------------------------|------------------------------------------|----------|--------|
| WR-01 `HistArena::swap` is an open, unfixed latent bug to fix in Phase 21 | Fixed in commit **c9a7fd1** (Phase 18) with the exact free-slot scan; repro + exhaustion tests present + passing | `git log`, `histogram_arena.rs:381-392,637-705`, `18-REVIEW-FIX.md` | D-03 code+repro already satisfied → Phase 21 verifies, not rebuilds |
| WR-01 "will bite the Phase-21 multi-leaf grow loop" (MEMORY `phase18-wr01-histarena-swap-aliasing`) | Prediction did not materialize — the driver sidesteps `HistArena` entirely (per-leaf `Vec<f64>`) | `grow_driver.rs` (no HistArena import/call) | Update the mental model; the corpus case ≠ swap-path exerciser |
| ODL-18/ODL-19 "Pending, Phase 21" | Delivered + verified in Phase 20 (6/6) | `20-VERIFICATION.md`, REQUIREMENTS lines 50-51 stale | D-05 reconciliation |
| data→leaf Handle aliasing is an open parity question | Resolved (double-buffer locked) AND moot for the live driver (host per-leaf rows) | `learner_parity.rs:1997-2083`, `grow_driver.rs:658-675` | Not a Phase-21 risk |

**Deprecated/outdated:**
- CONTEXT `<specifics>` "linchpin ... would alias under the old heuristic" — inaccurate for the driver; the aliasing scenario lives only in the `HistArena` unit test.
- ROADMAP Phase 21 body — stale driver-integration text; re-cut via `/gsd-phase`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Adding a `grow_tree_on_device_driver_with_cfg` variant is the minimal way to exercise the min_data/min_hessian case without touching the trait seam. | D-02.C | If the team prefers a config-through-seam design, the plan takes a slightly larger (but still additive) change; parity outcome unchanged. `[ASSUMED — design recommendation]` |
| A2 | `num_leaves = 6–8` is a sufficient "deep >2-live-leaf" fixture; exact rows/features left to the planner. | D-02.A | Under-sizing (e.g. num_leaves=4) still reaches 3-live but is less clearly "deep"; low risk (Claude's discretion per CONTEXT). `[ASSUMED — fixture sizing]` |
| A3 | A constant-gradient corpus reliably triggers the no-split break (`best.gain > 0.0` false). | D-02.B | If a degenerate positive gain arises, use `num_leaves = 1` (loop never runs) as the guaranteed no-split path. `[ASSUMED — corpus behavior; break path VERIFIED]` |

**All other claims in this document are `[VERIFIED]` against committed source, git history, or phase artifacts.**

## Open Questions (RESOLVED)

1. **min_data/min_hessian entry point (D-02.C)** — **RESOLVED: narrow `_with_cfg` variant.**
   - What we knew: the seam has no `GainConfig`; the driver hardcodes `proving_slice_config()`; downstream constraint plumbing already exists.
   - Decision (adopted by the plans): `grow_tree_on_device_driver_with_cfg` delegation (trait seam untouched, merge gate byte-unchanged) — planned in 21-01 Task 2 and consumed by 21-02 Task 3.

2. **WR-01 disposition (D-03)** — **RESOLVED: verify-and-document (no added repro variant).**
   - What we knew: the fix + repro are already committed and passing (`c9a7fd1`).
   - Decision (adopted by the plans): a verify-and-document task (Option 1 in §WR-01 Status) — planned as 21-01 Task 1; no additional repro variant.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust/cargo workspace | all tests | ✓ | existing | — |
| cubecl-cpu runtime (`cpu_client`) | default STRUCTURE gate lane | ✓ | existing (`cubecl` 0.10) | — |
| `LGBM_CUDA_ON_DEVICE` env | non-vacuous env-on gate run | ✓ (env var) | — | env-unset lane still runs (asserts `Ok(None)`) |
| ROCm (`cubecl-hip`, `--features rocm`) | D-04 best-effort smoke only | spoofed 8-CU APU (MEMORY `rocm-gfx1100-available`) | — | **Not required** — cubecl-cpu is the gate; ROCm is informative, non-blocking |

**Missing dependencies with no fallback:** None — the merge gate runs entirely on cubecl-cpu.
**Missing dependencies with fallback:** Real discrete GPU (only the spoofed APU is local; full hardware validation is Phase-23 Kaggle DoD, out of scope).

## Validation Architecture

> `workflow.nyquist_validation: true` — section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (cargo test); oracle-harness integration tests + lgbm-compute unit tests |
| Config file | none (cargo default) |
| Quick run command | `cargo test -p lgbm-compute --lib histogram_arena` (WR-01 unit repro) |
| Full suite command | `cargo test --workspace` (env unset — the byte-unchanged merge gate) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ODL-18H (WR-01) | `swap` picks a free non-aliasing slot with >2 live leaves | unit | `cargo test -p lgbm-compute --lib swap_multileaf_never_aliases_live_sibling_slot` | ✅ (present, passing) |
| ODL-18H (WR-01) | pool-exhaustion returns typed error, not aliasing | unit | `cargo test -p lgbm-compute --lib swap_errors_when_pool_exhausted` | ✅ |
| ODL-18H (deep) | on-device tree bit-exact at >2 live leaves | integration | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- --exact learner_parity_on_device_deep_multileaf_gate` | ❌ Wave 0 |
| ODL-18H (no-split) | `best_leaf == −1` break path matches anchor | integration | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- --exact learner_parity_on_device_nosplit_gate` | ❌ Wave 0 |
| ODL-18H (constrained) | min_data/min_hessian binds, tree matches constrained anchor | integration | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- --exact learner_parity_on_device_mindata_gate` | ❌ Wave 0 (needs `_with_cfg`) |
| ODL-18/19 (regression) | env-unset workspace stays byte-unchanged green | full suite | `cargo test --workspace` | ✅ |

### Sampling Rate
- **Per task commit:** the relevant quick unit/integration test above.
- **Per wave merge:** `cargo test -p oracle-harness --test learner_parity` (both env lanes) + `cargo test -p lgbm-compute --lib`.
- **Phase gate:** `cargo test --workspace` (env unset) green + each new gate green under `LGBM_CUDA_ON_DEVICE=1 ... -- --exact` (non-vacuous).

### Wave 0 Gaps
- [ ] `deep_multileaf_corpus()` + `learner_parity_on_device_deep_multileaf_gate` — covers ODL-18H (deep).
- [ ] no-split corpus + `learner_parity_on_device_nosplit_gate` — covers ODL-18H (no-split break).
- [ ] `grow_tree_on_device_driver_with_cfg` + constrained corpus + `learner_parity_on_device_mindata_gate` — covers ODL-18H (constrained). (Needs the driver `_with_cfg` variant first.)
- [ ] No framework install needed (cargo test present).

## Security Domain

> `security_enforcement: true` (ASVS L1) — section included. This phase is numerical/test/docs with **no** external input, network, auth, or crypto surface.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | minimal | The driver + `HistArena` already return typed `ComputeError` on bad `num_bin`/length/slot/pool-exhaustion (no panic on adversarial internal input). New `_with_cfg` variant should preserve typed-error boundaries. |
| V6 Cryptography | no | — |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Slot aliasing corrupts another leaf's histogram (WR-01) | Tampering (data integrity) | Free-slot scan + typed error on pool exhaustion — **already landed** (c9a7fd1); repro test locks it. |
| Out-of-range bin / length mismatch into a kernel | Tampering / DoS (panic) | Existing `ComputeError` validation at host boundaries (`grow_driver.rs`, `histogram_arena.rs`); `--exact` gates confirm no vacuous pass. |
| Env-gated path changes default behavior | Tampering (silent behavior change) | `LGBM_CUDA_ON_DEVICE` unset = byte-unchanged (`Ok(None)`); merge gate enforces it. |

**No new attack surface** is introduced (no new deps, no I/O, no untrusted input). `[VERIFIED: scope]`

## Sources

### Primary (HIGH confidence)
- `crates/lgbm-compute/src/kernels/histogram_arena.rs` — WR-01 fix (`swap` :347-415), repro tests (:637-705). Read in full.
- `crates/lgbm-compute/src/kernels/grow_driver.rs` — the live driver (:418-676), `DriverLeaf`, `proving_slice_config`, hardcoded cfg, break path (:511). Read in full.
- `crates/oracle-harness/tests/learner_parity.rs` — STRUCTURE gate (:2316), comparator (:2239), anchor (:2290), `grow_features_of` (:1969), buffer A/B (:1997), proving corpus (:2357), rocm cells (:2391+).
- `git log -- histogram_arena.rs` → commit c9a7fd1 "fix(18): WR-01 ...".
- `.planning/phases/20-.../20-VERIFICATION.md` — ODL-18/19 delivered (6/6, crit 5/6).
- `.planning/phases/18-.../18-REVIEW.md` + `18-REVIEW-FIX.md` — WR-01 finding + applied fix (iteration-1 commit c9a7fd1).
- `.planning/REQUIREMENTS.md` (lines 50-51, 103-124), `.planning/ROADMAP.md` (Phase 21 checklist + Notes).
- `.planning/config.json` — `nyquist_validation: true`, `security_enforcement: true`.

### Secondary (MEDIUM confidence)
- MEMORY notes: `phase18-wr01-histarena-swap-aliasing` (prediction now falsified), `on-device-kernel-goldens-are-re-transcriptions`, `on-device-driver-crate-cycle-constraint`, `def-f8u-01-flaky-resident-hip-test`, `gpu-lazy-dispatch...`/`tlk`, `rocm-gfx1100-available`.

### Tertiary (LOW confidence)
- None — all load-bearing claims verified against source/git this session.

## Metadata

**Confidence breakdown:**
- Standard stack (in-repo reuse): HIGH — all files read, no external deps.
- Architecture (driver = per-leaf `Vec<f64>`, no HistArena): HIGH — verified by grep + full source read.
- WR-01 already fixed: HIGH — git commit c9a7fd1 + source + 18-REVIEW-FIX.md.
- D-02.C blocker (hardcoded cfg / no seam config): HIGH — source; the `_with_cfg` remedy is a `[ASSUMED]` design recommendation (A1).
- Bookkeeping line numbers: HIGH — grepped exact lines.

**Research date:** 2026-07-02
**Valid until:** stable while Phase 21 is planned/executed (~30 days); the only volatility is if someone edits `grow_driver.rs`/`histogram_arena.rs`/`learner_parity.rs` before planning — re-grep the cited line numbers if so.
