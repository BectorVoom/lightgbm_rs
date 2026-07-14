# T-05 (SPEC-DRGL-05) — Deferral Surgery Design Note

Author: analysis session 2026-07-15, after T-12/T-13 landed. Purpose: capture the
FULL deferral mechanism (and a subtlety the PLAN/SPEC did not anticipate) so the
delicate hot-loop surgery can be **executed** in a fresh, headroom-rich session
rather than re-derived. This is design only — no code shipped for T-05 yet.

## What ships (unchanged from SPEC-DRGL-05)
- Gate: `LGBM_GROW_DEFER_SYNC` env, **default OFF**, `OnceLock`+`AtomicU8`-override
  template (mirror `resident_perm_partition_enabled`, grow_driver.rs:417-445).
- Flag OFF ⇒ today's loop verbatim (two `bump_sync`: pick @2706, read_split @2852).
- Flag ON ⇒ the deferred loop below. Gate: flag-ON tree **byte-identical** to
  flag-OFF (`learner_parity_on_device_resident_fast_path_gate`,
  `resident_tree_bit_exact_to_u64_integer_path`) on gfx1151.
- `deferred_read_fused=` tripwire (mirror `scan_parprefix=`).

## The mechanism (one-iteration deferral)
Today each split iteration `i` blocks TWICE: `pick(i)` (@2706) then `read_split(i)`
(@2852, split_point). Fuse `read_split(i-1)` with `pick(i)` into ONE
`client.read(vec![leaf_splits_handle, frontier_pick_handle])`
(`Backend::read_batched`, lib.rs:1286) at the TOP of iteration `i`. So `split_point(i-1)`
arrives at the START of `i`, and split `i-1`'s host bookkeeping is applied then.

Deferred loop body (flag ON):
```
deferred = None   // (scalars-sans-counts, new_left, new_right, p_begin, p_count) of split i-1
for split_idx in 0..(num_leaves-1):
    (deferred_cr, export) = read_deferred_split_and_pick(...)   // ONE client.read
    if let Some(d) = deferred:                                  // apply split i-1 now
        sp = deferred_cr.unwrap().left_count
        leaves[d.new_left ].row_begin=d.p_begin;         row_count=sp
        leaves[d.new_right].row_begin=d.p_begin+sp;      row_count=d.p_count-sp
        split_tree_scheduled(d.scalars with left_count=sp, right_count=d.p_count-sp)
    best_leaf = export.cells[6]; if best_leaf<0 break
    ...decode best, best_fpos...; if best_fpos<0 || !(gain>0) break
    p_begin=leaves[best_leaf].row_begin; p_count=leaves[best_leaf].row_count  // set by deferred step
    partition(...) → writes ranges[split_idx] on device, NO readback (deferred to i+1)
    new_left=best_leaf; new_right=leaves.len(); left_slot=next_slot; next_slot+=1; right_slot=parent_slot
    // seed children sum_g/sum_h (from export)/slot/depth into leaves[]; ROW RANGES DEFERRED
    // BUILD + SUBTRACT + SCAN + FOLD  (device-count, no host split_point — see below)
    deferred = Some((this split's scalars-sans-counts, new_left, new_right, p_begin, p_count))
// grow tail: apply the LAST split's deferred read (fold read_split into the existing
// read_perm @3253 batched read), then to_host_tree.
```

Why this is safe on ordering:
- `best_leaf(i)` was created at some split `j<i`; its row range is set by iteration
  `j+1 ≤ i`'s deferred step, which runs BEFORE `partition(i)`. Re-pick (child of
  `i-1` picked at `i`) is covered: set at `i`'s deferred step. T-01's
  non-overwriting per-split buffer is what makes the deferred `read_split(i-1)`
  read the right slot even when leaf ids collide.
- Tree mutations stay in order `0,1,2,…`, each deferred by one; the
  `right_leaf_index == tree.num_leaves` invariant (tree.rs:750) still holds because
  at `i`'s deferred step exactly `i` mutations (splits `0..i-1`) have run, so
  `new_right(i-1)=i == tree.num_leaves=i`. Last split mutates in the grow tail.

## THE SUBTLETY THE PLAN MISSED: fold target needs LEFT/RIGHT, not SMALLER/LARGER
At iteration `i` the build/subtract/scan/fold must run BEFORE `read_split(i)` — so
without host `split_point(i)`. T-04 (fixed-grid build) + T-12/T-13 (device-num_data
scans) already remove the host-count need for the BUILD and SCAN. Remaining blockers,
all today keyed on `smaller_is_left` (@2952-2965), which is `split_point`-derived and
now DEFERRED:
- **pool slots**: smaller→fresh(next_slot), larger→parent_slot. *Host-decidable
  WITHOUT smaller_is_left*: pass next_slot (fresh) + parent_slot; the fixed-grid
  build builds the smaller into the fresh slot on device (T-04 reads roles). OK.
- **fold target**: `scan(smaller_slot) → frontier[smaller_leaf]` where
  `smaller_leaf = smaller_is_left ? new_left : new_right`. **This still needs
  smaller_is_left.** ⇐ the real blocker.

**Resolution — reframe "build smaller" as "build LEFT" for the deferred arm:**
- ALWAYS build the LEFT child (rows `[p_begin, p_begin+split_point)`) into next_slot;
  SUBTRACT `parent − left` → RIGHT into parent_slot. Fold `next_slot(left)→
  frontier[new_left]`, `parent_slot(right)→frontier[new_right]`. **No smaller_is_left
  anywhere** (new_left, new_right, next_slot, parent_slot all host-known).
- **Byte-identity holds**: the resident build is u64 fixed-point; `parent = left +
  right` exactly (integer, order-free), so subtract-derived RIGHT == built-RIGHT
  bit-for-bit, and building LEFT directly vs subtract-deriving it are identical.
  Today the LARGER child is already subtract-derived and the bit-exact gate passes,
  so both "which child is built" choices yield the unique correct u64 histogram.
- **Perf caveat**: build-left may build the LARGER histogram (slower than build-smaller).
  Acceptable for the default-OFF experimental arm; the T-11 P100 A/B prices it. If it
  dominates, revisit with a device-resolved-out_leaf reduce (fold target chosen on
  device from roles) so build-smaller can be kept — heavier kernel work, deferred.

**⇒ Implication for T-12/T-13:** their devcount scans resolve num_data via
`is_smaller`+roles. The build-LEFT arm wants a LEFT/RIGHT resolution instead:
`left = ranges[6s+2]`, `right = parent_count − ranges[6s+2]` (NO roles read — simpler).
So T-05 needs a `which_child ∈ {Left,Right}` variant of `resolve_child_num_data` (and
the devcount scan/reduce entry points) alongside the shipped `is_smaller` one. Small,
mechanical, but it means T-05 loops back to generalize the T-12/T-13 API. The
co-pack (T-13) sibling twin already scans BOTH children (A=first, B=second) — feed it
A=left(num_data=left_count), B=right(num_data=right_count); it needs the same
Left/Right resolution rather than smaller/larger. Depth cap (host-known via
`leaves[].depth`) is the only host-side scannability gate to keep; size/hessian fall
out of the scan's own reject (tree-identical — an unscannable child folds the no-split
sentinel, never picked).

## §4 typed contract (from SPEC.md, reaffirmed)
```rust
fn read_deferred_split_and_pick<B, R>(
    backend: &B, client: &ComputeClient<R>,
    leaf_splits: &DeviceLeafSplits<R>,
    deferred_split_idx: Option<usize>,        // None on the root iteration
    frontier: &DeviceFrontier<R>,
    prev_smaller: i32, prev_larger: i32, cur_num_leaves: usize,
) -> Result<(Option<ChildRanges>, PickExport), ComputeError>;   // both from ONE client.read
```

## Sync-count consequence (feeds T-06)
Per-grow blocking syncs: today `2*num_leaves` = 1 root scan + (L-1) pick + (L-1)
read_split + 1 tail perm. After fusion the (L-1) read_splits collapse into the pick
reads (batched) and the last folds into the tail perm read ⇒ ≈ `num_leaves + O(1)`
(exact form: re-derive from a fresh `bump_sync()` grep of the flag-ON path — that is
T-06's job, DO NOT guess it here).

## ❌ WITHDRAWN REVISION (device-`out_leaf` reduce) — DO NOT USE
> A "device-out_leaf reduce" idea (keep build-smaller, resolve only the fold target on
> device) was recorded here and in commit `3e83c08`. **It is WRONG.** It missed that the
> SCAN also takes each child's `sum_gradient`/`sum_hessian`, and the SMALLER child's sum is
> `smaller_is_left ? left_sum : right_sum` — still deferred. Only the LEFT/RIGHT framing
> makes the sums host-known (the §8.3 pick export carries `left_sum_g/h` + `right_sum_g/h`
> directly ⇒ LEFT child sum = left_sum, RIGHT child sum = right_sum, no smaller_is_left). So
> the ORIGINAL build-LEFT / LEFT-RIGHT design below STANDS. The withdrawn text is kept
> struck-through only so the mistake isn't re-derived.

<details><summary>~~withdrawn device-out_leaf text~~</summary>

Deeper analysis found the build-LEFT reframe would touch the VALIDATED T-04 build +
T-12/T-13 scans (needs a Left/Right num_data variant). A lower-risk alternative keeps
ALL validated kernels and localizes the only irreducible `smaller_is_left` dependency —
the FOLD TARGET — to the reduce:

- Keep **build-smaller** (T-04) + the **is_smaller** devcount scans (T-12/T-13) exactly
  as shipped. At iter i: build smaller→next_slot, subtract→parent_slot, scan smaller
  (is_smaller=1) + larger (is_smaller=0) — all device-count, no host split_point.
- Add a **device-`out_leaf` reduce** variant: the fold's target frontier slot is resolved
  ON DEVICE — `smaller_leaf = roles[3*split_slot]!=0 ? new_left : new_right` (and the
  inverse for larger) — instead of a host `out_leaf` arg. `new_left`/`new_right` are
  host-known (best_leaf, leaves.len()); only the smaller/larger→left/right MAP is on
  device. This is the ONLY new kernel; scans/build unchanged.
- Everything else is host-deferrable or order-independent, so NO other smaller_is_left
  use blocks the deferral:
  - children `.slot` (next vs parent) — deferred to i+1 (read at split time, later).
  - row ranges + tree-mutation counts — deferred to i+1 (design above).
  - §8.3 self-invalidation `prev_smaller`/`prev_larger` — pass `{new_left,new_right}`
    (the pick invalidates BOTH child slots; the set is order-independent, so no
    smaller_is_left needed). ✅ verified against `frontier_pick_best_leaf_device`'s
    self-invalidation semantics.
- Net new kernel work vs build-LEFT: ONE device-`out_leaf` reduce ... **Use this approach.**
  ← WRONG (missed the scan's per-child sum dependency); see the withdrawal note above.

</details>

## ✅ CORRECT APPROACH — build-LEFT / LEFT-RIGHT (the original design, reaffirmed)
Frame the deferred arm by physical side (LEFT/RIGHT), NOT smaller/larger, because the pick
export makes the LEFT/RIGHT sums host-known:
- **num_data**: LEFT = `ranges[6*split_slot+2]` (= split_point), RIGHT = `parent_count −
  that`. Resolved on device; NO roles read.
- **sums**: LEFT = `export.left_sum_g/h`, RIGHT = `export.right_sum_g/h` — HOST-known.
- **fold target**: LEFT → `frontier[new_left]`, RIGHT → `frontier[new_right]` — HOST-known.
- **pool slots**: build LEFT into `next_slot` (fresh), SUBTRACT → RIGHT into `parent_slot`.
- **byte-identity**: u64 resident histogram is order-free (`parent = left + right` exactly);
  build-LEFT-directly == subtract-derived-LEFT, so the tree is byte-identical to the
  build-smaller flag-OFF path.
- children `.slot`, row ranges, tree-mutation counts: deferred one iteration. §8.3
  self-invalidation: pass `{new_left,new_right}` (order-independent).

Flag infra SHIPPED (`439af57`): `grow_defer_sync_enabled()` (default OFF) +
`set_grow_defer_sync_override()` + `deferred_read_fused=` tripwire + phase_prof wiring.

### Task order (CORRECT)
1. ✅ DONE (`439af57`): flag gate + tripwire.
2. **Generalize the devcount num_data resolve to `which ∈ {Left=0,Right=1,Smaller=2,
   Larger=3}`** (from today's `is_smaller` bool): `left=ranges[6s+2]`, `right=parent_count−
   left`, `smaller=select(smaller_is_left,left,right)`, `larger=` inverse. Mechanical rename
   of the `is_smaller` param → `which: u32` across the devcount kernels / `NumDataSrc` /
   launchers / Backend methods; existing T-12/T-13 callers+tests pass `which=2/3` (identical
   behavior); the loop passes `which=0/1`. Add LEFT/RIGHT rows to the byte-identity tests.
3. **build-LEFT fixed-grid variant**: build the LEFT child (rows `[p_begin, p_begin+
   split_point)`, i.e. `begin_off=0, count=ranges[6s+2]`) into `next_slot`. Simpler than
   T-04's smaller-resolve (no roles). Isolation test: build-LEFT hist == exact-grid LEFT.
4. **Loop restructure** behind the flag: batched read (fuse read_split(i-1)+pick(i));
   build-LEFT + subtract + LEFT/RIGHT devcount scans (which=0/1) folding host-known
   new_left/new_right; deferred bookkeeping (row ranges + `.slot` + tree mutation applied at
   top of i for i-1); `{new_left,new_right}` self-invalidation; last split folds into the
   tail read_perm.
5. Real-device byte-identity (flag-ON == flag-OFF, re-pick + full-growth, gfx1151).
6. Then T-06 (sync closed form), then T-11 (P100 A/B).

### Original (build-LEFT) task order — SUPERSEDED, kept for reference
1. Add `grow_defer_sync_enabled()` gate (default OFF) + `deferred_read_fused=` tripwire.
2. Add Left/Right `resolve_child_num_data` + devcount scan/reduce entry variants
   (generalize T-12/T-13; keep is_smaller variants for their existing callers/tests).
3. Restructure the loop per above behind the flag; wire fixed-grid build (T-04) +
   Left/Right devcount scans + deferred bookkeeping + batched read + tail fold.
4. Real-device byte-identity: flag-ON tree == flag-OFF, re-pick corpus + full-growth,
   on gfx1151. Iterate (this is the headroom-hungry GPU-debug part).
5. Then T-06 (sync closed form), then T-11 (P100 A/B).
