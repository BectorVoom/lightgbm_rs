# Pitfalls Research

**Domain:** On-device GBDT tree learner in CubeCL (porting `CUDASingleGPUTreeLearner` into lightgbm_rs) — v1.1 milestone
**Researched:** 2026-06-28
**Confidence:** HIGH (grounded in this project's own spikes 015–054, def-f8u-01, hip-split-parity debug, and PROJECT.md non-negotiables — not generic GPU advice)

> Scope note: these are the failure modes specific to *this* milestone — replacing the host-driven per-leaf growth loop (~8,570 small serial launches/train) with an on-device whole-tree growth loop that keeps build → best-split → partition resident. Every pitfall below has already bitten this codebase at least once (cited), or is a direct consequence of a non-negotiable the project has already committed to. (Supersedes the 2026-06-05 v1.0 PITFALLS, which assumed a now-retired 1e-12 oracle.)

---

## Critical Pitfalls

### Pitfall 1: Comparing two nondeterministic GPU f32 paths to each other (the oracle anti-pattern)

**What goes wrong:**
The new on-device learner produces a tree; a test asserts it equals the existing host-orchestrated GPU tree (or the old per-leaf path) to ~1e-6 on every leaf. Both are f32-atomic GPU paths whose accumulation order is nondeterministic, so a leaf output that sits on the 1e-6 knife-edge wobbles ~0.9e-6…1.1e-6 run-to-run and the test goes flaky — red ~half the runs with **zero** code change. This already happened: `learner_parity_resident_equals_host_tree_on_hip` (def-f8u-01) failed ~4 of 6 consecutive runs on unchanged master.

**Why it happens:**
Atomic adds commit in arbitrary order, so neither GPU f32 path is a stable reference. Two non-references compared to each other have no anchor — the "expected" side drifts too. Developers reach for GPU-vs-GPU because it feels like the most direct "did I change behavior?" check.

**How to avoid:**
Pin **both** GPU trees (new on-device AND existing host path) to the **deterministic cpu f64 anchor**, never to each other. The established pattern (commit d82611b, `assert_gpu_tree_matches_cpu_anchor` + `cpu_anchor_tree` in `oracle-harness/tests/learner_parity.rs`):
- **Structural fields BIT-EXACT** vs the f64 anchor (split feature, threshold, left_count, child structure, default_left subject to tie-awareness below). GPU structure == CPU structure is rock-stable for the spine corpus.
- **Leaf VALUES within an f32 envelope** `ROCM_LEAF_VALUE_TOL=1e-5` (the `sqrt(R)·ε_f32·mean|g|` accumulation envelope; single-path-vs-anchor measures ~1.4–1.75e-6, so 1e-5 is comfortable headroom, not a knife-edge).
- Histogram-cell ~1e-6 GPU-vs-f64 contract stays unchanged.

**Warning signs:**
A parity test that re-runs red/green/red on identical bytes; an `abs_diff` that prints ~1.0e-6 ± 0.1e-6; any assert whose "expected" value is itself read back from a GPU kernel.

**Phase to address:**
Slice 0 (oracle scaffolding) — build `assert_on_device_tree_matches_cpu_anchor` BEFORE writing any on-device kernel. This test is the merge gate for every subsequent slice.

---

### Pitfall 2: f32 reduction-order differences flip split structure (default_left / equal-gain ties)

**What goes wrong:**
On-device best-split selection reduces gains across the leaf frontier in a different float order than the f64 anchor. At f32 precision two gains that the anchor distinguishes by ~1 ULP (e.g. f64 60.2 vs 60.19999999999999) both round to the *same* f32 (60.20000076), so the kernel's tie-break picks a different `default_left`, or on an equal-gain plateau a different (but equivalent) threshold. A naive bit-exact structural assert then hard-fails on what is actually a faithful f32 mirror.

**Why it happens:**
Floating-point reductions are non-associative (PROJECT.md calls this out explicitly). The reverse-vs-forward gain gap is **linear in default-bin mass**, so only an essentially-empty default bin can be flipped by ~1e-12 reorder noise — but it *does* flip, ~34% of the time on the relevant fixtures (spike-016/022). This is **by construction**, not a kernel bug (hip-split-parity-default-left, RESOLVED 2026-06-09).

**How to avoid:**
Make the oracle **tie-aware**, not loose. The established pattern (commit 1832206):
- A `default_left` flip is allowed **only** on a verified f32 tie — same threshold AND same left_count AND net gains f32-equal — and is surfaced to stderr for the 04-ROCM-GAPS ledger. A flip on any **non-tie** split still hard-fails.
- Use a deterministic tie-break in the kernel that matches the anchor's keep-first rule (reverse-first / lowest-`t`), so present-data splits are reproduced within the ~1e-6 gate. Spike-022 verified max present-data leaf-output Δ = **0.0** across 480k histograms with a populated default bin — the structure is recoverable, you just must not assert it as naive bit-equality.
- Keep the leaf-output magnitude gaps (forward 3.81e-6 / reverse 7.63e-6 / skip_default 1.91e-6) non-blocking via surface-not-fail + the relative gate, never by widening the structural tolerance.

**Warning signs:**
Structural parity fails only on `default_left`; failures concentrate on fixtures with empty/sparse default bins; the same physical split (same threshold + same left set) recorded with opposite default direction.

**Phase to address:**
Slice that implements on-device best-split selection. The tie-aware assert must land in the same slice as the selection kernel — do not defer it.

---

### Pitfall 3: f64 hot loops in new on-device kernels (the consumer-NVIDIA 1/32 trap)

**What goes wrong:**
The on-device learner is written with f64 gain math / leaf-output / histogram accumulation for "accuracy", or a fused mega-kernel is built in f64 for convenience. On consumer NVIDIA (T4/P100) f64 throughput is **1/32 of f32**, so the kernel is f64-throttled. This is measured, not hypothetical: forcing the existing f64 `build_fix_scan` fusion (`LGBM_FUSED_FORCE=1`) on real CUDA was **5.4× WORSE** (58.3s vs 10.7s; ~571ms/tree vs 95ms), systematic across 3 arms (spike-052).

**Why it happens:**
The cpu f64 anchor is bit-exact and tempting to mirror directly on-device; f64 "feels safer" for reductions. But the device contract is ~1e-6 f32, not f64 — the f64 anchor is a *host* deterministic reference, not the device numeric type. The separate (fast) path is fast precisely because it uses **u64 fixed-point** integer-atomic build (spike-018) and reserves f64 only for the thin scan spine.

**How to avoid:**
- **Keep the u64 fixed-point build path** for histogram accumulation in the on-device loop — do not regress to f32 atomicAdd (CAS-retry loop) or f64. (`Atomic<i64>` is broken in cubecl-hip 0.10 — use u64 two's-complement, see Pitfall 5.)
- Audit every new kernel for where f64 sneaks in: **(a) gain math** in best-split selection, **(b) leaf-output** computation (`sum_grad/sum_hess`), **(c) score accumulation** into the resident score buffer, **(d) any fused build+scan mega-kernel**. Keep these f32 / fixed-point on-device; only the deterministic *host* anchor stays f64.
- If a fused kernel is genuinely needed, it must be a **non-f64** kernel — that is new kernel work with its own parity gate, not a toggle flip.

**Warning signs:**
A kernel's name or signature carries `f64`/`_f64_on`; per-tree ms jumps ~6× when a fusion/accumulation path is enabled; T4/P100 walls regress while the APU looks fine (the APU does not expose the 1/32 penalty the same way — see Pitfall 4).

**Phase to address:**
Every kernel-authoring slice. Add a CI/review checklist item "no f64 in device hot loops" to the slice definition of done. The first build slice must establish the u64 fixed-point accumulator.

---

### Pitfall 4: Trusting spoofed-APU lever signs for real discrete CUDA

**What goes wrong:**
Correctness and perf are validated only on the local "GPU", which is a **spoofed 8-CU APU** (HSA_OVERRIDE-spoofed gfx1152 Radeon 860M, shared DDR5), not the discrete gfx1100 it reports. Occupancy/perf conclusions drawn there mis-predict real NVIDIA: occupancy mattered on the APU (autotune beat the heuristic ~10%, spike-040) but P=1 is optimal on real CUDA (FORCE_P flat-to-worse, spike-051/053); fusion was "flat-to-negative" on the APU but **catastrophic** (5.4×) on CUDA (spike-052). The APU mis-predicted *every* lever in the 051–054 arc.

**Why it happens:**
The APU is the only local GPU and gives fast iteration; it's natural to treat its numbers as representative. But CU-count and memory-bandwidth axes are confounded (8 CU + shared DDR5 vs 96-CU discrete + dedicated VRAM), so occupancy/perf signs do not transfer. (Carve-out: **wavefront divergence IS faithful** on the APU — spike-036 — because it's a scheduler property, not a CU/memory property. That is the one micro-arch effect that survives the spoof.)

**How to avoid:**
Split the work by what each platform can answer:
- **Correctness / parity → local, on cubecl-cpu (the f64 anchor) and the APU's ~1e-6 gate.** Bit-exact structure and the f32 envelope are valid locally; def-f8u-01 and hip-split-parity were both resolved locally. Do the overwhelming majority of dev here — it's free and fast.
- **Perf / real-CUDA ratio → Kaggle only**, using the reusable zero-code env-toggle probe harness (`sources/051-*/spike051_kaggle.py`): one wheel build, N env-arm subprocesses under `LGBM_PHASE_PROF=1`, in-session A/B deltas only (absolute Kaggle walls drift across T4/P100/T4×2 sessions). Read the **max-launches** `phase_prof` dump (the timed run, not the ~445-launch warmup) and the `before=`-prefixed absolute-ms line, not the `%:` line.
- **Never gate a perf decision on an APU number.** State explicitly in each perf claim which hardware produced it.

**Warning signs:**
A perf win that only shows on the APU; an occupancy/P sweep that contradicts spike-051's "P=1 optimal on CUDA"; any claim that "the GPU is faster" without a Kaggle in-session A/B.

**Phase to address:**
The Kaggle perf-validation phase (the verification surface for the milestone). But the *structure* — keeping correctness local — must be designed into the slice plan from Slice 0 so Kaggle is only needed at perf-milestones, not every iteration (Kaggle round-trips are slow: push to GitHub master, clone, run, poll-to-COMPLETE, download).

---

### Pitfall 5: CubeCL cube-macro / runtime gotchas (no global barrier, broken intrinsics, GAT spelling, launch_unchecked safety)

**What goes wrong:**
An on-device whole-tree loop needs cross-workgroup coordination (the leaf frontier spans many cubes), uses intrinsics that don't exist or are broken in cubecl-hip 0.10, and dispatches via `launch_unchecked`. Each has a concrete trap this project has already logged:
- **No global barrier across workgroups.** There is no device-wide sync inside one kernel launch — you cannot "build all leaves, barrier, then select" within a single cube-spanning kernel. Cooperation is only intra-cube (`sync_cube` / plane ops). The frontier loop must be structured as a small number of *launches* with host-visible boundaries, or use the resident-pool + per-cube segmentation, not a phantom global barrier.
- **`Atomic<i64>` is broken in cubecl-hip 0.10** — use **u64 two's-complement** fixed-point instead (logged in gpu-build-fixedpoint-atomics).
- **`wrapping_add` is not a cube intrinsic** (cube-macro gotcha logged spike-045).
- **GAT spelling:** an `InputGenerator::generate<'a>` return must be spelled `<Vec<Handle> as TuneInputs>::At<'a>` or you hit E0195 (spike-038) — relevant if the on-device kernels get autotuned.
- **`plane_inclusive_sum`** lowers to a Hillis-Steele `__shfl_up` loop that works only up to `PLANE_DIM` (32/64); `num_bin` reaches 256 ≫ plane width, so a real within-feature scan needs a **segmented LDS block-scan**, not a bare plane sum (spike-022).
- **`launch_unchecked` safety obligation:** it is `unsafe` — the caller guarantees handle/buffer sizes and launch dims match the kernel's indexing. Out-of-bounds = silent corruption or UB, not a panic.

**Why it happens:**
GBDT growth is conceptually "process the whole frontier together", which maps naturally onto a single big kernel — but that requires a global barrier the hardware/CubeCL does not provide. The intrinsic/GAT traps are version-specific cubecl-hip 0.10 quirks not in the docs (the `cubecl_manual` autotuning doc is wrong on 3 load-bearing points, spike-037).

**How to avoid:**
- Architect the frontier loop as **few large launches with explicit host-or-resident sync points**, mirroring how official `CUDASingleGPUTreeLearner` sequences build → split → partition — not as one monolithic barrier-dependent kernel.
- Reuse the **resident pool** pattern already shipped (p90 resident pool, resident_bins uploaded once-per-train) so on-device state persists across launches without host round-trips.
- Keep a checklist of the known-broken cubecl-hip 0.10 constructs (Atomic<i64>, wrapping_add, plane-sum >PLANE_DIM, GAT spelling) in the slice plan; prefer reading **from the cubecl source**, not the manual.
- For every `launch_unchecked`, document the size/bounds invariant it relies on next to the call.

**Warning signs:**
A kernel that assumes all cubes have finished before any cube reads a shared result; a compile error E0195 on a tune generator; silent NaN/garbage in a histogram cell (out-of-bounds atomic); `Atomic<i64>` in a diff.

**Phase to address:**
The first on-device build slice (resident pool + launch sequencing) and the selection slice (plane/scan limits). Bake the cubecl-0.10-gotcha checklist into Slice 0.

---

### Pitfall 6: Porting the monolithic learner all at once (scope-creep) instead of a thin first slice

**What goes wrong:**
`cuda_single_gpu_tree_learner.cpp` is large and tightly coupled (build → best-split-across-frontier → partition → score-update, all on-device with a histogram pool and the subtraction trick). Attempting a full one-shot port produces a giant untested kernel set where a parity failure cannot be localized — you cannot tell whether the bug is in build, selection, partition, or the subtraction trick, and the f32 reduction-order ambiguities (Pitfalls 1–2) make every failure look plausibly "expected".

**Why it happens:**
The reference is a coherent whole and feels atomic; partial ports seem to require throwaway scaffolding. The milestone is explicitly "milestone-sized, high-uncertainty" (spike-052/054) which tempts a big-bang.

**How to avoid:**
Slice vertically, each slice gated against the f64 anchor before the next:
- **Slice 1 (thinnest):** on-device growth for a *single* shape/config (e.g. fixed num_leaves, one objective), reusing existing shipped kernels (u64 build, feature-per-lane scan, sibling co-pack) wired into a resident loop — proving the *orchestration* (fewer launches) end-to-end, not new numerics.
- Add on-device partition, then on-device frontier-wide selection, then the histogram-pool + subtraction trick on-device, one slice each.
- Each slice: structural bit-exact + leaf-value-envelope vs the f64 anchor (Pitfall 1) locally, then a Kaggle launch-count check (did `device_launches` actually drop from 8,570?).
- **Re-attribute after every wire** — this project's bottleneck moved **4 times** across spikes 014→015→023→034; a shipped lever relocates the cost. Do not assume the next slice attacks the same bottleneck.

**Warning signs:**
A slice that touches build, selection, and partition simultaneously; a parity failure you cannot localize to one phase; a PR with no intermediate anchor-gate.

**Phase to address:**
Roadmap structure itself — the roadmapper should sequence the milestone as ≥4 vertical slices, each with its own anchor gate and a Kaggle launch-count checkpoint, not one "port the learner" phase.

---

### Pitfall 7: Breaking the existing host-CUDA / ROCm / CPU paths while adding the on-device path

**What goes wrong:**
The on-device learner is wired in a way that changes the default code path for ROCm or CPU, so the shipped, parity-validated host-orchestrated path (and the bit-exact CPU merge gate) regress. A latent example already bit this project: the Phase-12 co-pack scan deferral ran `subtract_resident` *before* the fused smaller histogram was built, a sequencing bug surfaced only when a new GPU lever was gated (debug 8aed100).

**Why it happens:**
On-device growth shares kernels (build, scan, subtract, partition) with the existing path; "improving" a shared kernel or reordering the resident chain silently changes the old path. The subtraction trick (larger child = parent − smaller child) is order-sensitive: if the on-device loop reorders build vs subtract, the existing path's invariant breaks.

**How to avoid:**
- **Feature-gate / fallback** the on-device learner (PROJECT.md non-negotiable: "Coexistence with the existing host-orchestrated path so ROCm + CPU routing stay untouched"). New path behind a flag/runtime selection; old path is the default until the new one is parity- AND perf-proven.
- Gate **every** change with the full bit-exact suite: `cargo test -p lgbm-compute --lib`, `-p lgbm-treelearner --lib`, `-p oracle-harness` (esp. `raw_bin_train_matches_cpp_golden`, `learner_parity`), `-p lgbm`. The CPU f64 anchor is the hard merge gate and must stay bit-exact to lib_lightgbm 4.6 on both committed corpora.
- When the on-device loop sequences build/subtract/scan, preserve the **build-smaller-child-before-subtract** invariant explicitly; add an assertion or test that the subtraction trick's parent histogram is fully built before any child subtract reads it.
- Use **default-false trait methods** for backend routing (the established pattern: `prefers_host_partition` / `data_partition_native`), never a global mutable switch.

**Warning signs:**
A shared kernel diff with no CPU-anchor test run; `raw_bin_train_matches_cpp_golden` red; a ROCm parity test that was green before the on-device wiring; subtract reading a histogram that a reorder left half-built.

**Phase to address:**
Every slice (the merge gate). The fallback/feature-gate design is a Slice 0 / Slice 1 architectural decision.

---

### Pitfall 8: Assuming on-device growth removes the launch-bound wall without measuring launches

**What goes wrong:**
The whole milestone premise is "fewer, bigger kernels instead of ~8,570 small serial launches". If the on-device loop still issues per-leaf launches (e.g. one launch per frontier node because the frontier wasn't truly batched), the architectural gap is not closed — the learner is "on-device" in name but still launch-bound. The gap is architectural and asymptotes ~2× even at wide shapes (spike-054 — lgb_rs CUDA never beats official across 50→500 feat); a half-measure leaves it there.

**Why it happens:**
It's easy to move state on-device (resident pool) while keeping the host-driven *control* loop that fires a launch per node. `device_launches` stays ~8,570. The win only materializes when the **frontier is processed in batched launches** (build all frontier leaves' histograms, select across the whole frontier, partition all at once), gated by the best-first dependency chain being expressed on-device.

**How to avoid:**
- Make **`device_launches` a first-class success metric** of every slice, measured on Kaggle. Baseline is 8,570 / 100 trees (≈86/tree). A slice that doesn't materially cut it has not delivered the architectural win, regardless of local correctness.
- Verify the cost is launch-bound the same 3 ways spike-051/052/054 did: `build=0` (async issue), occupancy-insensitive (P=1 optimal), sync-cheap (~0.14ms/sync) — if those still hold after a slice, the launches are still the wall.
- Keep sibling co-pack default-on (it is load-bearing, banks 2,790 launches/syncs for ~390ms — spike-052) and stack the ~4% `LGBM_AUTOTUNE=0` cuda win, but know these are not the architectural lever.

**Warning signs:**
`device_launches` ≈ 8,570 after the on-device slice; per-tree ms unchanged on Kaggle despite "on-device" state; the lgb_rs/official ratio still ~5–6× at 50 feat.

**Phase to address:**
The Kaggle perf-validation checkpoints attached to each slice; the milestone's definition of done must include a measured `device_launches` reduction and a closed lgb_rs/official ratio, not just green parity tests.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| f64 gain/leaf math on-device "to match the anchor" | Trivially passes bit-exact parity | 5.4× slowdown on consumer NVIDIA (1/32 f64); defeats the whole milestone | **Never** in a device hot loop — f64 stays host-anchor only |
| GPU-vs-GPU parity assert (new path == old path) | Feels like the most direct regression check | Flaky ~half the runs (def-f8u-01); blocks CI; no stable reference | **Never** — always pin to the f64 anchor |
| One monolithic on-device kernel set ("port it all") | One PR, mirrors the reference file | Unlocalizable parity failures amid f32 tie noise; high blow-up risk | **Never** — slice vertically with per-slice anchor gates |
| Validate perf only on the local APU | Fast, free, no Kaggle round-trip | Mis-predicts every CUDA lever-sign (051–054); ships the wrong tuning | Only for *correctness*; never for a perf/routing decision |
| Naive bit-exact `default_left` assert | Strict, simple | Hard-fails on legitimate f32 ties ~34% of fixtures | **Never** — use the tie-aware assert (commit 1832206) |
| Change the default path to the new on-device learner before perf-proven | Less plumbing | Regresses shipped ROCm/CPU parity; loses the safe fallback | Only after parity + Kaggle perf both pass; until then feature-gate |
| Skip `device_launches` measurement, trust "it's on-device now" | Faster slice sign-off | Ships a launch-bound learner that didn't close the gap | **Never** — launch count is the milestone's success metric |

## Integration / Platform Gotchas

Mistakes when connecting to the specific tools and hardware in this milestone.

| Surface | Common Mistake | Correct Approach |
|---------|----------------|------------------|
| cubecl-hip 0.10 atomics | Use `Atomic<i64>` for fixed-point histogram | Broken in 0.10 — use **u64 two's-complement** fixed-point |
| cubecl cube-macro | Call `wrapping_add` as if it's a cube intrinsic | Not an intrinsic — restructure (logged spike-045) |
| cubecl plane ops | `plane_inclusive_sum` for a 256-bin within-feature scan | Plane width 32/64 ≪ 256 → needs a **segmented LDS block-scan** |
| cubecl whole-tree kernel | Expect a device-wide barrier inside one launch | None exists — sequence as few launches / resident pool, intra-cube `sync_cube` only |
| `launch_unchecked` | Treat as safe; mismatch dims/handles | It's `unsafe` — document and uphold the size/bounds invariant per call |
| cubecl autotune (`cubecl::tune`) | Follow the `cubecl_manual` doc | Manual wrong on 3 points — code from source; accumulating kernels need a **fresh-output** InputGenerator (else corrupt 27×); key on `log2(rows)` |
| Kaggle phase_prof | Read the `%:` line or the warmup dump | Read the **max-launches** (timed) dump and the `before=`-prefixed absolute-ms line; A/B in-session only |
| Subtraction trick on-device | Reorder build vs subtract in the resident loop | Build smaller child fully **before** any subtract reads the parent (debug 8aed100) |

## Performance Traps

Patterns that work at small scale / on the APU but fail on real CUDA or at the milestone's target shapes.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| f64 fused mega-kernel | ~6× per-tree slowdown on T4/P100, fine on APU | Keep u64 fixed-point build + f32; no f64 hot loop | Any consumer NVIDIA GPU (1/32 f64) |
| Tuning build occupancy / P for CUDA | P sweep flat-to-worse, autotune over-tunes | P=1 is optimal on CUDA (spike-051/053); don't lift `BUILD_PSET` | Real NVIDIA narrow shapes |
| Chasing sync reduction beyond co-pack | Marginal/no gain from fewer syncs | Syncs are ~0.14ms; keep co-pack default-on, stop there | Real CUDA (no sync headroom) |
| "On-device" but still per-leaf launches | `device_launches` ≈ 8,570 unchanged | Batch the frontier; measure launch count every slice | Every shape — the architectural wall |
| Trusting APU occupancy/fusion signs | Local win that vanishes/inverts on Kaggle | Validate perf only on real CUDA | Discrete NVIDIA (confounded axes) |
| Within-feature parallel scan as a "win" | Helps only narrow (≤256 feat) where GPU is least competitive | Don't wire — feature-per-lane (W=64) is the shipped win | Wide production shapes (wash-to-regression) |

## Numerical / Parity Mistakes

Domain-specific correctness issues (this milestone's analog of "security").

| Mistake | Risk | Prevention |
|---------|------|------------|
| Assert two GPU f32 paths equal each other at 1e-6 | Flaky CI, no stable reference (def-f8u-01) | Pin both to the f64 anchor; structure bit-exact, values within 1e-5 envelope |
| Naive structural bit-equality on `default_left` | Hard-fails on f32 near-ties (~34% of fixtures) | Tie-aware assert: flip allowed only on verified f32 tie, non-tie flip hard-fails |
| Widen the magnitude tolerance to "fix" parity | Hides real regressions behind a loose gate | Keep ~1e-6 / 1e-5 envelopes; surface (not fail) sub-tolerance gaps to the ledger |
| Drop the u64 fixed-point build for f32 atomics | f32 atomicAdd = CAS-retry (slow) + nondeterministic + ~3600× worse accuracy | Keep u64 fixed-point (spike-018); deterministic + fast + accurate |
| Let the on-device path silently change the CPU anchor's numerics | Breaks the hard bit-exact merge gate vs lib_lightgbm 4.6 | Feature-gate; run the full anchor suite on every change |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **On-device growth loop:** Often missing the actual launch-count reduction — verify `device_launches` dropped materially from 8,570/100-trees on Kaggle, not just that state is resident.
- [ ] **Best-split selection:** Often missing tie-aware `default_left` handling — verify on the empty-default-bin fixtures, non-tie flips still hard-fail.
- [ ] **Parity test:** Often pinned GPU-vs-GPU — verify both trees are asserted against the **f64 anchor**, not each other.
- [ ] **New kernels:** Often have an f64 leaf-output or gain accumulation hiding — grep for `f64` in device hot loops; verify u64 fixed-point build is intact.
- [ ] **Fallback path:** Often the new path quietly became the default — verify ROCm + CPU routing is untouched and the new learner is feature-gated.
- [ ] **Subtraction trick on-device:** Often reordered — verify the smaller child histogram is built before any subtract reads the parent (8aed100 class bug).
- [ ] **Perf claim:** Often APU-only — verify the number is a Kaggle in-session A/B on real NVIDIA, with the platform stated.
- [ ] **Merge gate:** Often skipped on a "GPU-only" change — verify `raw_bin_train_matches_cpp_golden` + `learner_parity` + the lgbm/treelearner/compute suites all green.

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Flaky GPU-vs-GPU parity test | LOW | Re-point both sides to `cpu_anchor_tree`; structure bit-exact + value within 1e-5 (pattern d82611b) |
| `default_left` hard-fail on a tie | LOW | Swap to the tie-aware assert (commit 1832206); flip allowed only on verified f32 tie |
| f64 kernel tanks on CUDA | MEDIUM | Rewrite the hot loop in u64 fixed-point / f32; f64 stays host-anchor only — may require a new (non-f64) kernel, not a toggle |
| Monolithic port won't localize a failure | HIGH | Roll back; re-slice vertically; gate each slice against the anchor before stacking the next |
| Shared-kernel change broke the CPU anchor | MEDIUM | Revert the shared change; re-implement behind the feature gate; re-run the full bit-exact suite |
| "On-device" but launches unchanged | HIGH | Re-attribute on Kaggle (build=0? P-insensitive? sync-cheap?); redesign the frontier as batched launches, not per-node |
| Validated perf on APU, wrong on CUDA | MEDIUM | Re-run the zero-code env-toggle probe on Kaggle; trust only in-session A/B deltas |

## Pitfall-to-Phase Mapping

How roadmap slices should address these pitfalls. (Slice numbers are indicative — the roadmapper should preserve the ordering and the anchor-gate-per-slice discipline.)

| Pitfall | Prevention Slice | Verification |
|---------|------------------|--------------|
| 1. GPU-vs-GPU oracle | Slice 0 (oracle scaffolding, before any kernel) | `assert_on_device_tree_matches_cpu_anchor` exists; both paths pinned to f64 anchor |
| 2. f32 reduction-order / tie flips | Slice implementing on-device selection | Tie-aware `default_left` assert; non-tie flips hard-fail; spike-022 fixtures pass |
| 3. f64 hot loops | Every kernel slice (start: first build slice) | grep no `f64` in device hot loops; u64 fixed-point build present; Kaggle per-tree ms not 6× |
| 4. Spoofed-APU perf trust | Kaggle perf-validation phase (design from Slice 0) | Every perf claim is a Kaggle in-session A/B with platform stated |
| 5. CubeCL 0.10 gotchas | Slice 0 checklist + first build/selection slices | No Atomic<i64>/wrapping_add/global-barrier assumptions; documented launch_unchecked invariants |
| 6. Monolithic port | Roadmap structure (≥4 vertical slices) | Each slice has its own anchor gate + Kaggle launch-count checkpoint |
| 7. Break existing paths | Every slice (merge gate) | Full bit-exact suite green; on-device path feature-gated; subtract-order invariant held |
| 8. Still launch-bound | Kaggle checkpoint per slice + milestone DoD | `device_launches` materially < 8,570; lgb_rs/official ratio closing |

## Sources

- `.planning/PROJECT.md` — v1.1 milestone non-negotiables (CPU f64 bit-exact merge gate; CUDA/ROCm ~1e-6; NO f64 hot loops; keep u64 fixed-point; coexistence/feature-gate with the host path). HIGH confidence (canonical project doc).
- `.claude/skills/spike-findings-lightgbm_rs/SKILL.md` — cross-cutting bit-exact gates, the cube-macro gotchas, re-attribute-after-every-wire rule, the spoofed-APU caveat. HIGH.
- `.../references/cuda-architectural-launch-bound.md` (spikes 051–054, real NVIDIA Kaggle) — the f64-trap (5.4×), P=1 optimal, sync-cheap, launch-bound mechanism (8,570 launches), the on-device learner as the one lever, the zero-code probe harness. HIGH (real-hardware measurement).
- `.../references/gpu-split-scan-occupancy.md` (spikes 016/021/022/022b) — f32 reorder parity-safe within ~1e-6, default_left flips cosmetic + tie-aware, plane-sum vs 256-bin limit, feature-per-lane shipped. HIGH.
- `.planning/spikes/052-cuda-launch-fusion/README.md` — the f64 fused-kernel 5.4× regression evidence + sync ~0.14ms. HIGH (Kaggle A/B, 3 arms).
- Memory: `def-f8u-01-flaky-resident-hip-test.md` — the GPU-vs-GPU flaky-oracle failure + the anchor-pinned fix (d82611b). HIGH (resolved defect).
- Memory: `hip-split-parity-preexisting-defect.md` — the default_left f32 near-tie + tie-aware assert fix (commit 1832206). HIGH (resolved defect).

---
*Pitfalls research for: on-device GBDT tree learner in CubeCL (lightgbm_rs v1.1 milestone)*
*Researched: 2026-06-28*
