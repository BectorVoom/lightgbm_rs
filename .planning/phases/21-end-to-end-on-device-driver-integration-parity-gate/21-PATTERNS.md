# Phase 21: Harden the On-Device Driver (re-cut) - Pattern Map

**Mapped:** 2026-07-02
**Files analyzed:** 5 (2 code, 1 verify-only, 2 bookkeeping)
**Analogs found:** 5 / 5 (every new/modified surface has an in-repo analog — this is a hardening phase; nothing is greenfield)

> **CRITICAL upstream correction (from RESEARCH.md, honored here):** WR-01 is
> ALREADY fixed at commit `c9a7fd1` (Phase 18), and the live driver
> (`grow_tree_on_device_driver`) does NOT consume `HistArena::swap` — it keeps
> per-leaf `DriverLeaf.hist: Vec<f64>`. So `histogram_arena.rs` is **verify/document
> only**, and the deep-multileaf parity corpus does **not** exercise the swap path.
> Do not plan "one case, two purposes."

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/oracle-harness/tests/learner_parity.rs` (add 3 D-02 gate tests + 3 corpus fns) | test | request-response (parity assertion) | `learner_parity_on_device_structure_gate` (same file, `:2316`) + `on_device_proving_corpus` (`:2357`) | exact (clone the existing gate) |
| `crates/lgbm-compute/src/kernels/grow_driver.rs` (add `grow_tree_on_device_driver_with_cfg`) | utility (kernel driver) | transform / batch | `grow_tree_on_device_driver` (same file, `:418`) | exact (extract-param refactor of the existing fn) |
| `crates/lgbm-compute/src/kernels/histogram_arena.rs` (VERIFY/DOCUMENT only — WR-01 already landed) | utility | transform | `swap` (`:347`) + `swap_multileaf_never_aliases_live_sibling_slot` (`:637`) | exact (already present + passing) |
| `.planning/REQUIREMENTS.md` (mark ODL-18/19 done, add ODL-18H) | config/docs | — | Existing checklist + traceability rows (`:50-51`, `:105-106`, `:122`) | exact |
| `.planning/ROADMAP.md` (re-cut Phase 21 body via `/gsd-phase`) | config/docs | — | Existing Phase-body format | exact |

## Pattern Assignments

### `crates/oracle-harness/tests/learner_parity.rs` (test, request-response) — 3 new D-02 gate tests + 3 corpus fns

**Analog:** `learner_parity_on_device_structure_gate` (`:2316`) — clone verbatim per D-02 case, swapping only the corpus fn (and, for the min-data case, the driver entry point + cfg).

**Gate skeleton to clone** (`:2316-2351`):
```rust
#[test]
fn learner_parity_on_device_structure_gate() {
    let backend = CpuBackend;
    let (features, g, h, num_leaves, max_depth) = on_device_proving_corpus();
    let grow_features = grow_features_of(&features);
    // The driver pins the proving-slice config; the anchor MUST use the identical one.
    let cfg = lgbm_compute::kernels::grow_driver::proving_slice_config();

    let env_on = std::env::var("LGBM_CUDA_ON_DEVICE").as_deref() == Ok("1");
    let grown = backend
        .grow_tree_on_device(&g, &h, &grow_features, num_leaves, max_depth)
        .expect("grow_tree_on_device seam ok");

    if env_on {
        let (on_device_tree, layout) =
            grown.expect("with LGBM_CUDA_ON_DEVICE=1 the driver must grow the tree (Ok(Some))");
        let anchor = cpu_anchor_tree(&features, &g, &h, cfg, num_leaves, max_depth);
        assert_on_device_tree_matches_cpu_anchor(&on_device_tree, &anchor, "on-device");
        assert_eq!(layout.num_data as usize, g.len(), "layout covers every row");
        assert_eq!(layout.leaf_count.iter().sum::<i32>(), g.len() as i32, "...");
        assert_eq!(layout.leaf_begin.len(), on_device_tree.num_leaves as usize, "...");
    } else {
        assert!(grown.is_none(), "byte-unchanged merge gate: Ok(None) when env unset");
    }
}
```

**Reuse verbatim (do NOT re-implement):**
- **Comparator** `assert_on_device_tree_matches_cpu_anchor` (`:2239-2284`) — structure bit-exact; `decision_type` strict on every bit except `default_left` (bit1) which flips ONLY on a genuine `split_gain` near-tie (corroborated by identical `threshold[node]` + identical `child_row_counts`). Uses `DEFAULT_LEFT_MASK`, `SPLIT_GAIN_TIE_TOL`.
- **cpu f64 anchor** `cpu_anchor_tree` (`:2290-2304`) — drives `SerialTreeLearner::new(&CpuBackend, &cpu_client(), cfg, num_leaves, max_depth).with_features(...).train(g, h, true)`. Takes an explicit `cfg: GainConfig` — pass the SAME cfg the driver used (Pitfall 3).
- **`FeatureColumn`→`GrowFeature` mapper** `grow_features_of` (`:1969-1986`) — field-by-field copy; reuse for every new corpus.

**Corpus fn to clone** — `on_device_proving_corpus` (`:2357-2389`), a 12-row / 2-feature / 4-leaf `MissingType::None` L2 corpus returning `(Vec<FeatureColumn>, Vec<f32>, Vec<f32>, i32, i32)`:
```rust
fn on_device_proving_corpus() -> (Vec<FeatureColumn>, Vec<f32>, Vec<f32>, i32, i32) {
    let grad = vec![-6.0f32, -6.0, -5.0, -5.0, -1.0, -1.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0];
    let hess = vec![1.0f32; 12];
    let f0 = FeatureColumn {
        bins: BinColumn::new(vec![0u32, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 6),
        num_bin: 6, offset: lgbm_treelearner::offset_for_most_freq_bin(0),
        min_bin: 0, max_bin: 5, default_bin: 6, most_freq_bin: 0,
        missing_type: MissingType::None,
        bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        real_feature_index: 0, ..Default::default()
    };
    // f1: 4-bin column ... real_feature_index: 1 ...
    (vec![f0, f1], grad, hess, 4, -1)
}
```

**Per-case corpus deltas (D-02, Claude's discretion on exact params):**
- **A. deep >2-live-leaf** (`deep_multileaf_corpus` → `learner_parity_on_device_deep_multileaf_gate`): same shape, bump to `num_leaves = 6–8`, ~16–24 rows, distinct-gain gradients (near-tie-free so the assert is genuine bit-exact). Best-first keeps all leaves live ⇒ ≥3 live at final splits. Broadens parity ONLY — NOT the swap path (Pitfall 1).
- **B. no-split break** (`nosplit_corpus` → `learner_parity_on_device_nosplit_gate`): either `num_leaves = 1` (loop `0..0` never runs) or constant gradient (all g equal, h=1 ⇒ zero gain ⇒ break at `grow_driver.rs:511`). Anchor takes the same no-split path.
- **C. min_data/min_hessian-constrained** (`mindata_corpus` → `learner_parity_on_device_mindata_gate`): calls the NEW `grow_tree_on_device_driver_with_cfg` DIRECTLY on `cpu_client()` with a constrained cfg (e.g. `min_data_in_leaf: 4`), anchoring `cpu_anchor_tree(..., that_same_cfg, ...)`. The trait seam `backend.grow_tree_on_device` has no cfg param, so this case bypasses it (see next file). Corpus sized so an unconstrained split WOULD occur but the constraint forbids it (observably binds).

**Env-lane discipline (both lanes required — Pitfall 5):** every case runs env-unset (asserts `Ok(None)`, byte-unchanged) AND `LGBM_CUDA_ON_DEVICE=1 ... -- --exact <test>` (asserts the real tree, non-vacuous).

---

### `crates/lgbm-compute/src/kernels/grow_driver.rs` (utility/kernel driver, transform) — add `grow_tree_on_device_driver_with_cfg`

**Analog:** the existing `grow_tree_on_device_driver` (`:418-676`) — the change is an extract-parameter refactor: make the existing fn delegate, and move the body (verbatim) into a `_with_cfg` variant whose only diff is the cfg source.

**Current signature + the one line to parameterize** (`:418-443`):
```rust
pub fn grow_tree_on_device_driver<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    gradients: &[f32],
    hessians: &[f32],
    features: &[GrowFeature],
    num_leaves: i32,
    max_depth: i32,
) -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError> {
    // ... num_leaves/features/length guards (unchanged) ...
    let cfg = proving_slice_config();   // :443  ← the ONLY line that changes
    let min_data = cfg.min_data_in_leaf;
    // ... rest of the body ...
}
```

**Delegation pattern to add** (per RESEARCH §"min_data case entry point"):
```rust
pub fn grow_tree_on_device_driver<R: cubecl::Runtime>(/* existing args */)
    -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError> {
    grow_tree_on_device_driver_with_cfg(
        client, gradients, hessians, features, num_leaves, max_depth, proving_slice_config())
}

pub fn grow_tree_on_device_driver_with_cfg<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    gradients: &[f32], hessians: &[f32], features: &[GrowFeature],
    num_leaves: i32, max_depth: i32,
    cfg: GainConfig,                    // NEW param replaces `let cfg = proving_slice_config();`
) -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError> {
    // body identical to today's grow_tree_on_device_driver, minus the `let cfg = ...` line
}
```

**Why this shape (from RESEARCH, don't deviate):**
- The trait seam `Backend::grow_tree_on_device` is UNTOUCHED (no `GainConfig` param) ⇒ merge gate stays byte-unchanged. Do NOT thread cfg through the seam (larger blast radius: `CpuBackend`/`GpuBackend<R>`/`learner.rs:764`) and do NOT bump `proving_slice_config()` globally (changes every existing gate's anchor).
- The downstream constraint plumbing ALREADY exists: `scan_leaf` passes cfg to `find_best_split_f64_on`; the `both_too_small` gate reads `min_data`; the `sum_h > 0` guard exists. Only the ENTRY POINT hardcodes the config.

**`proving_slice_config` for reference** (`:214-225`) — the default the delegating fn passes:
```rust
pub fn proving_slice_config() -> GainConfig {
    GainConfig {
        min_data_in_leaf: 1, min_sum_hessian_in_leaf: 0.0, max_delta_step: 0.0,
        lambda_l1: 0.0, lambda_l2: 0.0, min_gain_to_split: 0.0, path_smooth: 0.0,
        ..Default::default()
    }
}
```

**The break path both anchor and driver share** (`:510-513`) — the no-split case (D-02.B) hits this:
```rust
// No positive-gain split anywhere ⇒ stop (best_leaf == -1 sentinel analog).
if best_fpos < 0 || !(best.gain > 0.0) { break; }
```

**Preserve typed-error boundaries** (V5): the new variant keeps the same `ComputeError::Runtime`/`LengthMismatch` guards (`:426-442`).

---

### `crates/lgbm-compute/src/kernels/histogram_arena.rs` (utility, transform) — VERIFY / DOCUMENT ONLY

**Analog / already-landed fix:** `HistArena::swap` (`:347-415`) — the D-03 free-slot scan is ALREADY present (commit `c9a7fd1`, Phase 18). Do NOT re-apply.

**The landed fix** (`:381-408`, this IS the D-03 sketch):
```rust
let occupied: std::collections::HashSet<usize> =
    self.leaf_to_slot.values().copied().collect();
let fresh = (0..self.num_slots)
    .find(|s| *s != parent_slot && !occupied.contains(s))
    .ok_or_else(|| ComputeError::Runtime { /* pool exhausted */ })?;
assert_ne!(fresh, parent_slot, "...never alias smaller and larger into one slot");
self.leaf_to_slot.insert(larger_leaf, parent_slot);
self.leaf_to_slot.insert(smaller_leaf, fresh);
if parent_leaf != larger_leaf && parent_leaf != smaller_leaf {
    self.leaf_to_slot.remove(&parent_leaf);   // drop the now-internal parent key
}
```

**The already-present repro tests to CONFIRM pass** (do not rewrite):
- `swap_multileaf_never_aliases_live_sibling_slot` (`:637-678`) — 4-slot pool; `swap(0,1,2)` → `swap(2,3,4)` → `swap(1,5,6)`; asserts all four live leaves hold distinct slots. Old `(1+1)%4==2` would have aliased live leaf 3.
- `swap_errors_when_pool_exhausted` (`:684-695`) — typed error on exhaustion, not aliasing.
- `swap_rejects_single_slot_pool` (`:699-705`) — typed error for `num_slots < 2`.

**Task shape (RESEARCH Option 1 — recommended):** a verify-and-document task: (a) confirm the free-slot scan + drop-parent-key are present, (b) confirm the three repro tests pass (`cargo test -p lgbm-compute --lib histogram_arena`), (c) document in SUMMARY that WR-01 closed in `c9a7fd1` and the live driver does not consume `HistArena` (per-leaf `Vec<f64>`). Add a new repro variant ONLY if explicitly desired (low value).

---

### `.planning/REQUIREMENTS.md` (config/docs) — D-05 bookkeeping

**Analog:** the existing checklist + traceability rows (verified line numbers):
- `:50` `- [ ] **ODL-18**: ...` → `- [x]` (mark Complete, delivered Phase 20).
- `:51` `- [ ] **ODL-19**: ...` → `- [x]`.
- Add a new `- [ ] **ODL-18H**: ...` checklist item near ODL-18/19 (hardening: targeted STRUCTURE parity corpus + WR-01 confirmation + repro, cpu-f64-anchored, cubecl-cpu default lane; mapped to Phase 21).
- Traceability table `:105` `| ODL-18 | Phase 21 | Pending |` → `| ODL-18 | Phase 20 | Complete |`; `:106` `| ODL-19 | Phase 21 | Pending |` → `| ODL-19 | Phase 20 | Complete |`; add row `| ODL-18H | Phase 21 | Pending |`.
- Per-phase count table `:122` `| Phase 21 — Driver integration + parity gate | ODL-18, ODL-19 | 2 |` → re-cut to attribute ODL-18/19 to Phase 20 and Phase 21 → ODL-18H; keep counts consistent with the traceability table.

---

### `.planning/ROADMAP.md` (config/docs) — D-05 re-cut

**Analog:** existing Phase-body format (Goal / Success Criteria / Notes). The Phase 21 checklist line already says "re-cut to hardening"; the **body** is STALE driver-integration text and must be re-cut via `/gsd-phase` to the hardening scope (WR-01 confirmation, targeted parity corpus, bookkeeping; out: categorical → Phase 22, perf/default-on → Phase 23). Record the Notes resolution: "data→leaf aliasing is moot for the live driver (host per-leaf rows); batched `client.read` is Phase-23 perf — neither affects the Phase-21 parity gate."

## Shared Patterns

### Anchor discipline (never GPU-vs-GPU)
**Source:** `cpu_anchor_tree` (`learner_parity.rs:2290`) + `assert_on_device_tree_matches_cpu_anchor` (`:2239`)
**Apply to:** every new D-02 gate test.
Every on-device tree pins to the single cubecl-cpu f64 anchor (`SerialTreeLearner` + `CpuBackend`), STRUCTURE bit-exact, leaves within `ROCM_LEAF_VALUE_TOL` (~1e-5), `default_left` tie-aware. Never compare two GPU f32 paths (def-f8u-01).

### Env-gated non-vacuous gate
**Source:** `learner_parity_on_device_structure_gate` (`:2323-2350`)
**Apply to:** every new D-02 gate test.
```rust
let env_on = std::env::var("LGBM_CUDA_ON_DEVICE").as_deref() == Ok("1");
// env_on: assert Ok(Some) real tree vs anchor
// env unset: assert grown.is_none()  ← byte-unchanged merge gate
```
Verified non-vacuous by `LGBM_CUDA_ON_DEVICE=1 cargo test ... -- --exact <test_name>`.

### Config-parity between driver and anchor (Pitfall 3)
**Source:** `learner_parity.rs:2321,2331` (gate passes `proving_slice_config()` to BOTH driver and anchor)
**Apply to:** all three D-02 cases — the anchor's `cfg` MUST equal the cfg the driver grew under. For case C, that means the SAME constrained cfg goes to both `grow_tree_on_device_driver_with_cfg` and `cpu_anchor_tree`.

### Typed-error boundaries (V5 / security)
**Source:** `grow_driver.rs:426-442` guards; `histogram_arena.rs:355,385` `ComputeError::Runtime`
**Apply to:** the new `_with_cfg` variant — preserve `ComputeError::Runtime`/`LengthMismatch` on bad input; no panics on internal adversarial input.

## No Analog Found

None. Every surface in this hardening phase has a direct in-repo analog (this is a copy-and-parameterize / verify-and-document phase, not greenfield). The only NEW symbol is `grow_tree_on_device_driver_with_cfg`, which is an extract-param refactor of `grow_tree_on_device_driver`.

## Metadata

**Analog search scope:** `crates/oracle-harness/tests/learner_parity.rs`, `crates/lgbm-compute/src/kernels/grow_driver.rs`, `crates/lgbm-compute/src/kernels/histogram_arena.rs`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`
**Files scanned:** 5 (all cited line numbers re-verified against committed source this session)
**Pattern extraction date:** 2026-07-02
</content>
</invoke>
