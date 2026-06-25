# GPU scan round-trip — regime-split attribution & sibling co-pack

Implementation blueprint from spikes **023, 024**. After the u64 build (018/019) and feature-per-lane
scan (021) shipped, this pair re-profiled the GPU per-tree round-trip and attacked the per-leaf
scan-readback SYNC floor by co-packing sibling scans. Co-pack was WIRED in phase 12. Read this
alongside `gpu-split-scan-occupancy.md` (the scan-occupancy story it continues).

## Requirements

- Spoofed 8-CU APU ⇒ judge the **SIGN** of device-time ratios, never raw cold wall-clock (a single
  cold run can show a spurious 3.8×). Use the `LGBM_PHASE_PROF` whole-train BUDGET + the
  `LGBM_SCAN_DRAIN` build-drain A/B, 2–3 process restarts.
- Bit-exact gate unchanged: co-pack is byte-identical by construction (each feature's sequential
  scan is unchanged — no spike-016 reorder); gate with `kernel_parity --features rocm`.

## How to Build It

### Re-profile FIRST — the bottleneck is REGIME-SPLIT (023, the kill-check)

Run `LGBM_PHASE_PROF=1 LGBM_SCAN_PROF=1 LGBM_SCAN_DRAIN=1` across small/medium AND wide. The
per-tree GPU cost (shape-independent @ num_leaves=31) is ~30 build + ~29 subtract + **~59
scan-readback SYNCS = ONE per leaf-node** (both siblings of every split scanned in TWO separate
launches+readbacks) ≈ ~118 device launches/tree. Attribution splits by regime:
- **launch-bound (small/medium):** the scan-readback SYNC floor is the largest reclaimable residual
  (~48%/35% of the round-trip; ~44µs/sync ≈ pure fixed latency). Host `partition` ~13%.
- **compute-bound (large/wide):** build-compute DOMINATES and GROWS with rows (68% → **96.5% @
  1M×500**, undiminished by u64 — the win made atomics cheaper, not the build smaller); scan-sync
  collapses to 3.2%. Host `partition` grows to 23% (a CPU-track residual — see
  `partition-memory-traffic.md`).

The histogram **subtraction trick already runs on-device** (`subtract_resident`, no double-rebuild)
— that lever is CLOSED. Build levers are exhausted on this APU; wide routes to CPU.

### Co-pack the two sibling scans into ONE launch+readback (024, VALIDATED → WIRED)

The two siblings of every split are scanned separately (2 launches, 2 `read_one_unchecked` syncs).
Co-pack both into one launch over `2×n_feats` feature-slots + one readback (lane `g<n` ⇒ sibling-A
feat `g`; `g≥n` ⇒ sibling-B feat `g−n`), feature-per-lane W=64 (the 021 packing):

- **Isolated:** sign-stable **~2.0×** small→n=256, ~1.5–1.7× at wide (the per-launch fixed SYNC
  latency dominates scan compute on the APU, so halving the sync COUNT ≈ halves the time).
- **Bit-exact:** B's two halves byte-identical to A's two scans, every cell.
- **Honest e2e** (cold-isolated overstates warm, the 021 rule): co-pack halves the sync COUNT not
  the compute ⇒ reclaims ~½ the genuine scan-sync = **~10–15% small/medium, ~1.5% wide** (per 023's
  scan-sync fractions) — NOT 2× e2e.
- **Wiring (done, phase 12 — VERIFIED LIVE + DEFAULT-ON by quick task 260625-obl):** a 2-slot
  production scan kernel (two histogram Handles, 2×n SplitInfos, one readback) + a growth-loop
  reorder that defers the smaller-child scan past `subtract_resident` (both Handles are co-resident
  there) + an oracle parity re-pin. **Call site:** `learner.rs:1839`
  `self.backend.scan_resident_siblings(...)` inside the `copack_feats`-gated block (gate at
  `learner.rs:1788`) → `RocmBackend::scan_resident_siblings` (`lib.rs:2507`) →
  `find_best_splits_fused_siblings_from_handles_on` (`split.rs:1584`) →
  `find_best_splits_fused_siblings_kernel` (`split.rs:1079`).
- **Default-ON, NOT "behind a flag that must be set ON".** `LGBM_SIBLING_COPACK` is parsed by
  `sibling_copack_override()` (`resident_pool.rs:282`) as a three-way override mirroring
  `LGBM_RESIDENT_FORCE`: **`None` (unset) ⇒ co-pack engages** whenever the structural correctness
  gate holds; **`=0` is the byte-identical OFF switch** (force the two-separate-scans path); `=1` is
  force-on (identical to unset today — there is no separate co-pack size threshold yet). A future
  reader must NOT misread "exists but unwired / needs the flag set ON" — the co-pack rides the
  resident size gate and is live by default; you set `=0` only to get a clean A/B off-path.
- The gate ANDs in `larger_is_resident_subtract` (the only layout the kernel was validated against —
  WR-03, NOT the `larger_resident_slot == larger_slot_id` proxy) plus `resident_eligible`,
  `copack_override != Some(false)`, `smaller_resident_only`, `smaller_scannable`, and IDENTICAL spine
  membership (one shared `feats` for both siblings); any false case falls back BYTE-UNCHANGED.
- **Verification anchors (the bit-exact gates, run in BOTH copack modes):** CPU —
  `kernel_parity_sibling_copack_equals_two_scans_on_cpu` (cubecl-cpu W=1, 2-slot == two separate
  single-slot scans, every SplitInfo field) + `raw_bin_train_matches_cpp_golden` (C++ golden RED-FLAG
  sentinel, byte-idempotent across modes) + the `learner_parity_*` suite; ROCm —
  `kernel_parity_split_within_tol_on_hip` (~1e-6 hip split) +
  `kernel_parity_sibling_copack_equals_two_scans_on_hip` (pinned to the cubecl-cpu native anchor, NOT
  GPU-vs-GPU — def-f8u-01). A committed-golden CHANGE is a HARD RED FLAG (co-pack is bit-exact by
  construction), never auto-re-pin.

## What to Avoid

- **Don't trust raw GPU wall-clock** — the phase_prof "scan=96%" was an async-readback artifact; the
  build launches un-synced so its compute materializes inside the scan's `read_one_unchecked`. Use
  `LGBM_SCAN_DRAIN` to re-attribute build vs genuine scan-sync.
- **Don't pursue further per-leaf sync cuts beyond sibling co-pack** (spike-025 SUPERSEDED): build+
  fix+scan FUSION is a known NULL (collapsing forces a sequential f64 build, 260608-t3t), and the
  leaf-wise data dependency caps further cuts (each split needs its children's splits read back
  before selecting the next leaf). Only un-tried variant = depth-wise FRONTIER batching, which
  changes the growth policy (parity risk) for uncertain APU ROI.
- **Re-profile after EVERY build change** — the bottleneck has moved repeatedly (014→015→023);
  co-pack's e2e is bounded by whatever the current scan-sync fraction is.

## Constraints

- cubecl authoring friction (from the 024 trail): scalars pass RAW (not `ScalarArg`); u32→f64 needs
  `f64::cast_from` (not `as`); the cube macro supports neither a `#[cube]` helper nor `macro_rules!`
  in the body (inline directly); the best-gain sentinel must init from a plain `0.0f64` literal (a
  `-1.0e30f64` scientific literal trips MLIR lowering) — gains are ≥0 so `0.0` is valid.
- ROI: ROCm-parity-track. Small/medium (where co-pack helps most) is exactly where the CPU crushes
  the GPU on this 8-CU APU (spike-001: GPU 0.06–0.36× at 20k–100k) — co-pack nudges the crossover,
  doesn't flip the regime. Real value on a discrete gfx110x.

## Origin

Synthesized from spikes: 023 (post-021 round-trip regime-split attribution — measurement, the
kill-check), 024 (sibling-scan co-pack — VALIDATED ~2× isolated + bit-exact, WIRED phase 12).
Source files in: `sources/023-post-021-roundtrip-attribution/`, `sources/024-batch-sibling-scans/`
(+ the `spike024_sibling_scan_ab.rs` isolated A/B). Continues `gpu-split-scan-occupancy.md`.
