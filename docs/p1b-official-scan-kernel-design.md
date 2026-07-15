# P1b — Official-Shape Parallel Scan Kernel: Ready-to-Code Design

Status 2026-07-15: **SHIPPED — rung-4 Kaggle P100 A/B is a WIN; CUDA default flipped
ON.** `lgb-rs-p1b-official-scan` (P100, order-alt warm-median-3, 500k×50×100): base
4.604s → official **4.233s = 1.0876× (−371ms)**, preds **BIT-IDENTICAL (max_abs 0.0)**,
counts scan_official=2980/0, TREE_COUNT_OK, CUDA parity gate green; nsys shows the scan
kernel drop **131 µs → 16–20 µs/launch** (subtract-fuse twin 20.1 µs ×2385, single
16.1 µs) — the first lever to beat the scan wall where pargain/parprefix both lost.
Default flipped ON for **CUDA only** (`_ => name == "cuda"` in `scan_official_enabled`);
hip stays parprefix (gfx1151 drain: official 714 ms/scan vs parprefix 691 ms, ~3%
slower, no win). Hatch `LGBM_SCAN_OFFICIAL=0` reverts; cpu anchor untouched (never runs
it → bit-exact merge gate `resident_tree_bit_exact_to_u64_integer_path` on CpuBackend
unaffected). Earlier status (kernel coded + rungs 1-3) preserved below.

Status 2026-07-15 (superseded): **KERNEL CODED + LOCALLY VALIDATED (rungs 1-3 green on
real gfx1151); Kaggle P100 A/B (rung 4) is the remaining step.** The three kernel twins
(`find_best_splits_fused_staged_official_kernel` / `..._siblings_..._official_kernel`
/ `..._siblings_subtract_..._official_kernel`), the `official_branch_block` #[cube]
helper, the f64 block collectives (`block_inclusive_scan_f64` / `block_max_f64` /
`block_min_u32`), the `LGBM_SCAN_OFFICIAL` gate (`scan_official_enabled` +
`set_scan_official_override`, default OFF both backends), the `scan_official=` counts
tripwire, and the launcher wiring (official PRECEDES parprefix/pargain in
`launch_staged_{single,siblings}_scan` + the subtract-fuse launcher, CubeDim=256 +
`plane_dim`) are all landed in `crates/lgbm-compute/src/kernels/split.rs`. Rung-2
(`pargain_kernel_matches_legacy_kernel_on_device` official arm) + rung-3
(`resident_score_within_envelope_of_host_cuda` with `LGBM_SCAN_OFFICIAL=1`, anchor
bit-exact still green) PASS on gfx1151. Work is UNCOMMITTED on
`refactor/remove-dead-toggles`. Follows the Phase-0 nsys round + P1a (`745a59c`).
Target: staged scan 131 µs/launch → official's ~7.5 µs class (nsys §5); the biggest
remaining device bucket (313 + 87 = 400 ms/train).

Original design (below) followed as-built; the only implementation refinement worth
noting: `official_branch_block`'s argmax uses `block_max_f64(cand_gain)` +
`block_min_u32(key)` with `is_splittable` folded from a separate
`block_max_f64(valid?1:0)` OR (matching the staged kernels' independent
`is_splittable` flag), and reads the histogram from the LDS stage (same cooperative
staging as the staged family) rather than directly from global — keeping all three
twins structurally identical to their staged counterparts.

## The hypothesis (why this may win where parprefix lost)

pargain (spike094) and parprefix (spike104) both LOST on P100 — but both kept the
staged scaffold at **CUBE_DIM=64** (`SCAN_STAGED_CUBE_DIM`), parallelizing phase-1
accumulate within 64 lanes. Official launches **256 threads/block, one per bin**
(`FindBestSplitsForLeafKernel<<<num_tasks, 256>>>`), a `ShufflePrefixSum` across
all 256, then argmax. At num_bin=255 our 64-lane version leaves each lane doing 4
bins serially; a **256-wide** block is 4× the parallelism per feature and lifts
P100 occupancy from ~200 active lanes (2-lane serial staged) to ~25.6k threads
(100 blocks × 256). **This 256-wide geometry is the untested variable** — the
plan's "clean full-shape rewrite, not another hybrid." Real risk it still loses
(P100 cheap 1:2 f64 + barrier cost); the A/B settles it. ROCm sign may differ
again (parprefix already default-ON there).

## Algorithm — PROVEN bit-exact (committed test, do not re-derive)

`crates/lgbm-compute/tests/scan_pargain_parity.rs::official_branch` +
`assert_official_parity` + 3 tests (fan-out / early-done / exact-tie) are GREEN on
the cpu lane. The reformulation:

- Remove the serial `done` early-break recurrence; replace with the **stateless
  per-candidate guard** `active && !cont && !brk`.
- **Why exact:** reverse `right_count` is non-decreasing (`round_int(h·cnt_factor)
  ≥ 0`) ⇒ `cont` (near too small) monotone true→false, `brk` (far too small)
  monotone false→true. Serial `done` suppresses the first `!cont&brk` candidate
  onward; by `brk` monotonicity `{consider} == {active && !cont && !brk}`. Forward
  symmetric.
- **NEAR side = directly accumulated** (`acc_g`/`acc_h`/`acc_cnt`), **FAR side =
  complement** `total − acc` — matching each serial branch's exact arithmetic
  (reverse accumulates RIGHT, forward LEFT). Passing `total − (total − acc)` for
  the near side reorders the f64 subtract → 2-ULP drift (the bug the test caught).
- Counts/threshold/left_count stay **integer-exact** under any add order (integers
  < 2^53 sum exactly); only g/h gains reorder → **~1e-6** (the documented GPU
  envelope, same contract as pargain/parprefix; the cpu ANCHOR never runs this).

## Kernel structure

New kernels mirror the staged family's launcher signature EXACTLY (drop-in):
`find_best_splits_fused_staged_official_kernel` (single) +
`..._siblings_official_kernel` (co-pack, the live hip/cuda default arm) +
`..._siblings_subtract_official_kernel` (subtract-fuse twin). Same args as
`find_best_splits_fused_staged_kernel` (split.rs:1788). **CUBE_DIM = 256**
(one lane per bin; num_bin ≤ 255 ⇒ no striding). CubeCount unchanged (n features,
or 2·n via CUBE_POS for siblings). Replace the `scan_rev/fwd_branch_staged` +
`merge_finalize_staged` body with:

```
official_branch_block(sm, state_rev, forward=false, ...)   // all 256 lanes
sync_cube()
official_branch_block(sm, state_fwd, forward=true, ...)    // reuse LDS scratch
sync_cube()
if UNIT_POS == 0 { merge_finalize_staged(state_rev, state_fwd, out, f*12, ...) }
```

### `official_branch_block` (#[cube] helper, f64, real-GPU)

Lane `k = UNIT_POS` owns candidate k (k < count; higher lanes inert). Steps:

1. **Per-bin contribution (masked):** compute `(t, active, bi)` exactly as
   `official_branch` (reverse: `t = t_start − k`, `in_range = t ≥ 1−offset`;
   forward: `t = k`; both apply `skip_default_bin`). Load `g = active?sm[bi]:0`,
   `h = active?sm[bi+1]:0`, `qc = active? round_int(h·cnt_factor) : 0` (as f64,
   exact int).
2. **Block inclusive prefix-sum in lane order** of `g`, `h`, `qc` →
   `acc_g`, `acc_h+K_EPSILON`, `acc_cnt`. Use a NEW f64 two-level block scan
   `block_inclusive_scan_f64(v, plane_dim)` modeled verbatim on
   `best_split.rs::stage1_block_scan` (plane_inclusive_sum + LDS cross-plane
   carry, STAGE1_N_PLANES_MAX). Call it 3× (or pack). Seed the hessian's
   `K_EPSILON` by adding it AFTER the scan to lane's acc (constant offset, matches
   serial's `sum_*_hessian = kEpsilon` init).
   **Order note:** plane scan reorders adds ⇒ g/h ~1e-6 (fine); qc integer-exact.
3. **Per-lane candidate:** near=(acc_g, acc_h, acc_cnt); far=complement. Assign
   left/right by `forward` (reverse: left=far, right=near; forward: left=near,
   right=far — the committed reference's exact split). Compute `cont`/`brk`/
   `consider`, `current_gain = get_split_gains(...)`, `valid = consider &&
   current_gain > min_gain_shift`, `cand_gain = valid?current_gain:0`.
4. **Block argmax (gain desc, k asc):** `gmax = block_max_f64(cand_gain)`;
   `winner_k = block_min_u32( (cand_gain==gmax && valid) ? k : BIG )`. `is_split =
   gmax > 0` (valid ⟹ gain>min_gain_shift≥0 ⟹ cand_gain>0 ⟹ first valid also
   beats the 0 seed, so gmax>0 ⟺ any valid — matches serial). Provide
   `block_max_f64`/`block_min_u32` as f64/u32 twins of stage1's pattern (plane_max
   / plane_min + LDS carry; `primitives.rs::plane_block_{max,min}_kernel_f32` are
   the f32 launch-kernel precedents to copy as #[cube] helpers).
5. **Winner writes state:** `if k == winner_k && is_split { state[0..6] = [1.0,
   cand_gain, threshold_f, left_count_f, sum_left_g, sum_left_h] }`; lane 0 seeds
   the no-split state first (`if UNIT_POS==0 { state = [0,0,0,0,0,0] }`,
   sync_cube before the winner write). `threshold = forward? t+offset :
   t-1+offset`. `sum_left_*`, `left_count` = the reference's left side.
   `merge_finalize_staged` (split.rs:1714, UNCHANGED) consumes the 6-cell state.

### cubecl gotchas (from memory — heed)

- All state/scratch f64 in LDS; avoid mutable-i32-locals-in-select and post-loop
  array-indexing (cubecl-cpu MLIR limits) — but this kernel is **real-GPU-only**
  (gated), so cpu-lowering is moot; still, keep the field carry in f64 locals.
- `SharedMemory::<f64>::new(N)` needs a usize-typed const size.
- Do NOT `sync_cube()` inside a divergent `if UNIT_POS==…` — the block scans/
  reduces sync internally and must be called by ALL lanes uniformly.

## Wiring

- Gate `scan_official_enabled(runtime_name)` (env `LGBM_SCAN_OFFICIAL`, default
  OFF both backends until the A/B; `=1` forces on cuda|hip). **Precedence:** in
  `launch_staged_{single,siblings}_scan` place official BEFORE parprefix/pargain
  when enabled. Counts tripwire `scan_official=` (mirror `scan_pargain=` in
  split.rs + phase_prof.rs). `set_scan_official_override` for in-process A/B.
- Threads: bump CUBE_DIM to 256 only for the official launch path (its own
  `launch` with `CubeDim::new_1d(256)`); leave the staged/pargain/parprefix
  launches at `scan_cube_dim()`.
- The `_devcount` deferral twins (T-12/13) are NOT needed for the A/B (deferral is
  default-OFF); add later only if official becomes default AND deferral revives.

## Validation ladder (each gates the next)

1. ✅ **Algorithm parity** (cpu, committed `b30b014`).
2. **Kernel-vs-legacy byte/envelope** on gfx1151: add an `official` arm to
   `real_gpu_gated::pargain_kernel_matches_legacy_kernel_on_device`
   (scan_pargain_parity.rs) — `is_splittable` + counts + threshold BIT-EQUAL to
   the legacy serial kernel, gain ≤ 1e-6·(1+|g|), 7 corpora × 5 feats. Env
   `LGBM_SCAN_OFFICIAL=1`.
3. **Driver numerics** on gfx1151: `float_envelope_500k_rocm_resident` +
   `resident_score_within_envelope_of_host_cuda` with `LGBM_SCAN_OFFICIAL=1`; and
   default-OFF `resident_tree_bit_exact_to_u64_integer_path` still green (anchor
   never runs official).
4. **Kaggle P100 A/B** `lgb-rs-p1b-official-scan` (order-alternated warm-median-3,
   500k×50×100): base vs official; preds within envelope (NOT necessarily
   bit-identical — official reorders f64 → tree may differ ~1e-6, so gate on
   PRED envelope + tree-count, not max_abs=0.0); counts `scan_official` nonzero;
   drain scan bucket delta. Flip default per backend only on a measured win
   (ROCm may sign-differ — run gfx1151 drain too).

## Risk / stop criteria

If the P100 drain scan bucket does NOT drop (or wall regresses) → official joins
pargain/parprefix as a documented net-negative, kept opt-in; the scan gap is then
CONCLUSIVELY a cubecl-primitive/upstream problem (plane scan can't match CUDA
`ShufflePrefixSum`), and §7's "accept ~1.47× / pivot to cubecl-upstream" path is
the answer. Either way the A/B is decisive — this is the last in-codebase scan
lever.
