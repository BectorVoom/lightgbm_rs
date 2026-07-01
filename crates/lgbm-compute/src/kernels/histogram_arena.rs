//! Pre-allocated-once histogram arena + `hist_t**` handle-rotation contract — **16-02**.
//!
//! Owning plan: **16-02**. Scope locked by **ODL-10**, **D-02**, **D-09**.
//!
//! ## What lives here
//! [`HistArena`] is the host-side bookkeeping for the histogram-subtraction trick
//! (the port of the AMD-fork `USED_HISTOGRAM_BUFFER_NUM` slot pool +
//! `cuda_hist_` arena, `docs/cuda-kernel-design.md` §7.0 / §17). It owns a fixed
//! pool of `num_slots` CubeCL `Handle`s, each sized for `slot_len_elems` `hist_t`
//! cells (stride-2 `[g0,h0,g1,h1,…]`), allocated **once** in [`HistArena::new`]
//! via a single counted `client.empty` closure — the SAME allocate-exactly-once
//! discipline as [`crate::kernels::split_info::DeviceSplitInfo::new`] (D-09): the
//! alloc counter is frozen after construction and `client.empty` appears nowhere
//! else in the module.
//!
//! ## The handle contract (D-02)
//! The build->fix->subtract entry (16-04) drives an explicit
//! `{parent, smaller, larger}` triple. The subtraction trick
//! (`FeatureHistogram::Subtract`, §17) requires `larger = parent - smaller`
//! derived **in the parent's buffer** — so [`HistArena::rotate`] (16-02 Task 2)
//! reassigns slot **indices**: `larger_idx <- parent_idx` (the larger child
//! inherits the parent buffer in-place) and `smaller_idx <- a fresh slot` distinct
//! from `parent_idx`. Rotation reassigns indices ONLY — it performs ZERO additional
//! `client.empty` calls and NO bulk histogram copy.
//!
//! ## Scope (NOT here)
//! This struct models ONE `{parent, smaller, larger}` triple under an explicit,
//! anchor-testable contract — it demonstrates the rotation in ISOLATION. The
//! cross-tree whole-pool SWAP (`SplitTreeStructureKernel`) is **Phase 18** (16-CONTEXT
//! §9, pool SWAP DEFERRED). There is no tree-growth driver and no whole-tree pool
//! manager here.
//!
//! ## Analog file
//! `crates/lgbm-compute/src/kernels/split_info.rs` — the `device_allocations`
//! counted-`client.empty` "allocated exactly once" pattern this module mirrors, and
//! the V5 `checked_mul` slab-sizing boundary (T-16-02-01).

use core::marker::PhantomData;
use std::collections::HashMap;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::error::ComputeError;

/// A host pre-allocated histogram slot pool + the `{parent, smaller, larger}`
/// handle-rotation contract (the `USED_HISTOGRAM_BUFFER_NUM` arena analog).
///
/// The slot buffers are allocated **once** in [`Self::new`] (D-09) and reused: the
/// rotation (16-02 Task 2) only reassigns the three role indices into the pool, so
/// the larger child is derived in-place in the parent's buffer and the smaller child
/// lands in a fresh non-aliasing slot with NO reallocation and NO bulk copy.
///
/// See the module docs for the full contract; the whole-tree pool SWAP is Phase 18.
pub struct HistArena<R: cubecl::Runtime> {
    /// The reserved device-resident slot handles (allocated once, D-09).
    slots: Vec<Handle>,
    /// Number of `hist_t` cells each slot buffer is sized for (stride-2
    /// `[g,h,g,h,…]`, i.e. `2 * num_bin`).
    slot_len_elems: usize,
    /// Number of slot buffers in the pool (`== slots.len()`).
    num_slots: usize,
    /// Count of `client.empty` device allocations performed — must equal
    /// `num_slots` after [`Self::new`] and NEVER change (proves "allocated exactly
    /// once": no per-rotation / per-derive alloc, D-09).
    device_allocations: usize,
    /// Current slot index holding the parent histogram.
    parent_idx: usize,
    /// Current slot index holding the smaller child histogram.
    smaller_idx: usize,
    /// Current slot index the larger child is derived into (becomes `parent_idx`
    /// after [`Self::rotate`]).
    larger_idx: usize,
    /// Leaf-index → slot-index table for the cross-tree whole-pool SWAP (Phase 18,
    /// D-09). Maps a live leaf id to the pool slot holding its histogram; mutated
    /// ONLY by [`Self::set_leaf_slot`] / [`Self::swap`] (index reassignment, zero
    /// `client.empty`).
    leaf_to_slot: HashMap<usize, usize>,
    _runtime: PhantomData<R>,
}

impl<R: cubecl::Runtime> HistArena<R> {
    /// Allocate the histogram slot pool for `num_slots` slots, each `slot_len_elems`
    /// `hist_t` cells — **one `client.empty` per slot, exactly once** (D-09). No
    /// allocation happens anywhere else in the module (no per-rotation / per-derive
    /// device alloc).
    ///
    /// The byte size of each slot slab (`slot_len_elems * size_of::<f64>()`) is
    /// computed in `usize` and overflow-checked at the V5 boundary (threat
    /// T-16-02-01) before any allocation.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `num_slots == 0`, if `slot_len_elems == 0`, or
    /// if the slab byte length `slot_len_elems * size_of::<f64>()` overflows `usize`.
    pub fn new(
        client: &ComputeClient<R>,
        num_slots: usize,
        slot_len_elems: usize,
    ) -> Result<Self, ComputeError> {
        if num_slots == 0 {
            return Err(ComputeError::Runtime {
                detail: "HistArena::new: num_slots must be >= 1".to_string(),
            });
        }
        if slot_len_elems == 0 {
            return Err(ComputeError::Runtime {
                detail: "HistArena::new: slot_len_elems must be >= 1".to_string(),
            });
        }

        // T-16-02-01: slab byte sizing in usize, checked for overflow before any
        // alloc (mirrors split_info.rs:276-284).
        let elem_size = core::mem::size_of::<f64>();
        let slab_bytes = slot_len_elems.checked_mul(elem_size).ok_or_else(|| {
            ComputeError::Runtime {
                detail: format!(
                    "HistArena::new: slab byte length {slot_len_elems} * {elem_size} \
                     overflows usize"
                ),
            }
        })?;

        // Count every device allocation so the "allocated exactly once" invariant is
        // structurally verifiable: this closure is the ONLY caller of `client.empty`
        // in the whole module, and it runs only here in `new` (D-09).
        let mut device_allocations = 0usize;
        let mut alloc = || -> Handle {
            device_allocations += 1;
            client.empty(slab_bytes)
        };

        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            slots.push(alloc());
        }

        // Structural proof of D-09: exactly one alloc per slot, nothing else.
        assert_eq!(
            device_allocations, num_slots,
            "HistArena::new must allocate exactly num_slots device buffers"
        );

        // Initial contract triple: parent in slot 0; the smaller/larger roles start
        // distinct from the parent where the pool permits (rotate() reassigns them).
        let smaller_idx = if num_slots >= 2 { 1 } else { 0 };

        Ok(HistArena {
            slots,
            slot_len_elems,
            num_slots,
            device_allocations,
            parent_idx: 0,
            smaller_idx,
            larger_idx: 0,
            leaf_to_slot: HashMap::new(),
            _runtime: PhantomData,
        })
    }

    /// Number of slot buffers in the pool.
    #[must_use]
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Number of `hist_t` cells each slot is sized for (`2 * num_bin`).
    #[must_use]
    pub fn slot_len_elems(&self) -> usize {
        self.slot_len_elems
    }

    /// The number of device buffers allocated — equals `num_slots` after
    /// [`Self::new`] and NEVER changes (proves "allocated exactly once": no
    /// per-rotation / per-derive alloc, D-09).
    #[must_use]
    pub fn device_allocations(&self) -> usize {
        self.device_allocations
    }

    /// Current slot index holding the parent histogram.
    #[must_use]
    pub fn parent_idx(&self) -> usize {
        self.parent_idx
    }

    /// Current slot index holding the smaller child histogram.
    #[must_use]
    pub fn smaller_idx(&self) -> usize {
        self.smaller_idx
    }

    /// Current slot index the larger child is derived into.
    #[must_use]
    pub fn larger_idx(&self) -> usize {
        self.larger_idx
    }

    /// The device `Handle` for the parent slot (clone — no allocation).
    #[must_use]
    pub fn parent_handle(&self) -> Handle {
        self.slots[self.parent_idx].clone()
    }

    /// The device `Handle` for the smaller-child slot (clone — no allocation).
    #[must_use]
    pub fn smaller_handle(&self) -> Handle {
        self.slots[self.smaller_idx].clone()
    }

    /// The device `Handle` for the larger-child slot (clone — no allocation).
    ///
    /// After [`Self::rotate`] this is the SAME slot as [`Self::parent_handle`]
    /// (`larger_idx == parent_idx`): the larger child is derived in-place in the
    /// parent's buffer (D-02 / §17).
    #[must_use]
    pub fn larger_handle(&self) -> Handle {
        self.slots[self.larger_idx].clone()
    }

    /// Rotate the `hist_t**` role indices for the subtraction trick (D-02 / §17):
    /// the larger child is derived **in-place in the parent's buffer**
    /// (`larger_idx <- parent_idx`) and the smaller child is assigned a **fresh
    /// slot** distinct from the parent (`smaller_idx <- fresh`).
    ///
    /// This reassigns INDICES ONLY — it performs ZERO `client.empty` calls and NO
    /// bulk histogram copy, so [`Self::device_allocations`] is unchanged across any
    /// number of `rotate()` calls. After rotation the round-trip contract holds:
    /// handing `{parent_handle(), smaller_handle()}` to a subtract that computes
    /// `larger = parent - smaller` lands the result in the `larger` slot — which is
    /// the parent's old buffer (`larger_idx == old parent_idx`).
    ///
    /// The no-alias invariant `smaller_idx != parent_idx` (== `larger_idx`) is
    /// enforced (T-16-02-02): a parent/smaller alias would let the in-place subtract
    /// corrupt the parent before the smaller is read.
    ///
    /// This is the ONE-triple demonstration; the cross-tree whole-pool SWAP is
    /// Phase 18 (16-CONTEXT §9).
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if the pool has fewer than 2 slots — a fresh,
    /// non-aliasing slot for the smaller child cannot be allocated from a 1-slot
    /// pool without aliasing the parent.
    pub fn rotate(&mut self) -> Result<(), ComputeError> {
        if self.num_slots < 2 {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "HistArena::rotate: a non-aliasing smaller slot requires \
                     num_slots >= 2 (have {})",
                    self.num_slots
                ),
            });
        }

        // The larger child is derived in-place in the parent's buffer.
        self.larger_idx = self.parent_idx;
        // The smaller child takes a fresh slot distinct from the parent (== larger).
        let fresh = (self.parent_idx + 1) % self.num_slots;
        self.smaller_idx = fresh;

        // T-16-02-02: never alias parent and smaller into the same slot. With
        // `num_slots >= 2`, `(parent_idx + 1) % num_slots != parent_idx` always
        // holds; assert it as a hard invariant (a violation is a logic bug, not a
        // recoverable runtime condition).
        debug_assert_ne!(
            self.smaller_idx, self.parent_idx,
            "HistArena::rotate must never alias parent and smaller into one slot"
        );
        assert_ne!(
            self.smaller_idx, self.larger_idx,
            "HistArena::rotate: smaller must not alias the larger (== parent) slot"
        );

        Ok(())
    }

    // =====================================================================
    // Phase-18 cross-tree whole-pool SWAP (D-09) — the `SplitTreeStructureKernel`
    // leaf-indexed histogram-pool pointer swap. Extends (does NOT rebuild) the
    // per-split `rotate()` above with a leaf-index → slot-handle table.
    // =====================================================================

    /// Assign leaf `leaf` to pool slot `slot` (index reassignment ONLY, no
    /// allocation). Seeds the leaf→slot table for the root / a freshly-created
    /// leaf before it is split via [`Self::swap`].
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `slot >= num_slots`.
    pub fn set_leaf_slot(&mut self, leaf: usize, slot: usize) -> Result<(), ComputeError> {
        if slot >= self.num_slots {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "HistArena::set_leaf_slot: slot {slot} >= num_slots {}",
                    self.num_slots
                ),
            });
        }
        self.leaf_to_slot.insert(leaf, slot);
        Ok(())
    }

    /// The pool slot index currently holding leaf `leaf`'s histogram, if assigned.
    #[must_use]
    pub fn leaf_slot(&self, leaf: usize) -> Option<usize> {
        self.leaf_to_slot.get(&leaf).copied()
    }

    /// The device `Handle` for leaf `leaf`'s histogram slot (clone — no allocation).
    ///
    /// # Panics
    /// If `leaf` has no slot assignment (call [`Self::set_leaf_slot`] /
    /// [`Self::swap`] first).
    #[must_use]
    pub fn leaf_handle(&self, leaf: usize) -> Handle {
        let slot = self
            .leaf_to_slot
            .get(&leaf)
            .copied()
            .expect("HistArena::leaf_handle: leaf has no slot assignment");
        self.slots[slot].clone()
    }

    /// The `SplitTreeStructureKernel` whole-pool swap (`cuda_data_partition.cu:827-906`,
    /// D-09): the leaf `parent_leaf` splits into `left_leaf` / `right_leaf`. The
    /// **larger** child inherits the parent's slot (so the subtraction trick derives
    /// `larger = parent − smaller` IN-PLACE in that buffer); the **smaller** child
    /// takes a FRESH non-aliasing slot and rebuilds directly (§17, Pitfall 5).
    /// `smaller_is_left` picks the branch (`num_data[left] < num_data[right]`).
    ///
    /// Reassigns leaf→slot INDICES only — ZERO `client.empty` (`device_allocations`
    /// frozen), NO bulk histogram copy. Mirrors [`Self::rotate`]'s no-alias
    /// discipline (T-16-02-02): the smaller child never aliases the larger's slot.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `parent_leaf` has no slot assignment or the pool
    /// has fewer than 2 slots (a non-aliasing fresh slot cannot be supplied).
    pub fn swap(
        &mut self,
        parent_leaf: usize,
        left_leaf: usize,
        right_leaf: usize,
        smaller_is_left: bool,
    ) -> Result<(), ComputeError> {
        if self.num_slots < 2 {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "HistArena::swap: a non-aliasing fresh slot requires num_slots >= 2 (have {})",
                    self.num_slots
                ),
            });
        }
        let parent_slot = self.leaf_to_slot.get(&parent_leaf).copied().ok_or_else(|| {
            ComputeError::Runtime {
                detail: format!("HistArena::swap: parent leaf {parent_leaf} has no slot assignment"),
            }
        })?;

        let (smaller_leaf, larger_leaf) = if smaller_is_left {
            (left_leaf, right_leaf)
        } else {
            (right_leaf, left_leaf)
        };

        // Larger child inherits the parent buffer in-place (`larger = parent − smaller`).
        // Smaller child takes a fresh slot distinct from the parent (== larger).
        let fresh = (parent_slot + 1) % self.num_slots;

        // T-16-02-02: never alias the smaller and larger children into one slot.
        assert_ne!(
            fresh, parent_slot,
            "HistArena::swap must never alias the smaller and larger children into one slot"
        );

        self.leaf_to_slot.insert(larger_leaf, parent_slot);
        self.leaf_to_slot.insert(smaller_leaf, fresh);
        // Track the roles so `parent_handle`/`smaller_handle`/`larger_handle` and the
        // allocation-frozen invariant stay consistent with `rotate()`.
        self.parent_idx = parent_slot;
        self.larger_idx = parent_slot;
        self.smaller_idx = fresh;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu_client;

    /// D-09: `new` allocates exactly `num_slots` device buffers — the counter equals
    /// `num_slots` and `slots.len()` matches.
    #[test]
    fn new_allocates_exactly_num_slots() {
        let client = cpu_client();
        for &n in &[1usize, 2, 3, 8] {
            let arena = HistArena::new(&client, n, 16).unwrap();
            assert_eq!(arena.device_allocations(), n, "counter must equal num_slots");
            assert_eq!(arena.num_slots(), n);
            assert_eq!(arena.slot_len_elems(), 16);
        }
    }

    /// T-16-02-01: a zero `num_slots` is rejected with a typed error, not a panic.
    #[test]
    fn new_rejects_zero_num_slots() {
        let client = cpu_client();
        let res = HistArena::new(&client, 0, 16);
        assert!(matches!(res, Err(ComputeError::Runtime { .. })));
    }

    /// T-16-02-01: a zero `slot_len_elems` is rejected with a typed error, not a panic.
    #[test]
    fn new_rejects_zero_slot_len() {
        let client = cpu_client();
        let res = HistArena::new(&client, 3, 0);
        assert!(matches!(res, Err(ComputeError::Runtime { .. })));
    }

    /// T-16-02-01: an overflowing slab byte length (`slot_len_elems * 8` > usize::MAX)
    /// is rejected via `checked_mul` with a typed error, not a wrapping alloc.
    #[test]
    fn new_rejects_overflowing_slab() {
        let client = cpu_client();
        let res = HistArena::new(&client, 1, usize::MAX);
        assert!(matches!(res, Err(ComputeError::Runtime { .. })));
    }

    /// D-09: reading the role handles/indices performs ZERO additional allocation —
    /// the counter is frozen after `new` (no `client.empty` outside `new`).
    #[test]
    fn accessors_do_not_allocate() {
        let client = cpu_client();
        let arena = HistArena::new(&client, 3, 16).unwrap();
        let before = arena.device_allocations();
        let _p = arena.parent_handle();
        let _s = arena.smaller_handle();
        let _l = arena.larger_handle();
        let _ = (arena.parent_idx(), arena.smaller_idx(), arena.larger_idx());
        assert_eq!(
            arena.device_allocations(),
            before,
            "accessors must not allocate"
        );
    }

    /// The initial contract triple seeds the parent in slot 0 with smaller distinct
    /// where the pool permits (>= 2 slots).
    #[test]
    fn initial_triple_seeds_parent_slot_zero() {
        let client = cpu_client();
        let arena = HistArena::new(&client, 3, 16).unwrap();
        assert_eq!(arena.parent_idx(), 0);
        assert_ne!(arena.smaller_idx(), arena.parent_idx());
    }

    /// D-02 / T-16-02-02: after `rotate()`, `larger_idx == previous parent_idx`,
    /// `smaller_idx != parent_idx`, and `smaller_idx != larger_idx` — the larger
    /// child inherits the parent buffer in-place and the smaller takes a fresh,
    /// non-aliasing slot. Anchor-tests the INDEX bookkeeping in isolation.
    #[test]
    fn rotate_bookkeeping_no_alias() {
        let client = cpu_client();
        let mut arena = HistArena::new(&client, 3, 16).unwrap();
        let old_parent = arena.parent_idx();

        arena.rotate().unwrap();

        assert_eq!(
            arena.larger_idx(),
            old_parent,
            "larger must inherit the parent slot in-place"
        );
        assert_ne!(
            arena.smaller_idx(),
            arena.parent_idx(),
            "smaller must not alias the parent slot"
        );
        assert_ne!(
            arena.smaller_idx(),
            arena.larger_idx(),
            "smaller must not alias the larger (== parent) slot"
        );
    }

    /// D-09: the allocation counter is identical before and after any number of
    /// `rotate()` calls — rotation reassigns indices only, no `client.empty`.
    #[test]
    fn rotate_does_not_allocate() {
        let client = cpu_client();
        let mut arena = HistArena::new(&client, 3, 16).unwrap();
        let before = arena.device_allocations();
        for _ in 0..5 {
            arena.rotate().unwrap();
        }
        assert_eq!(
            arena.device_allocations(),
            before,
            "rotate() must perform zero allocations"
        );
    }

    /// A 1-slot pool cannot supply a non-aliasing smaller slot — `rotate()` rejects
    /// it with a typed error rather than aliasing the parent (T-16-02-02).
    #[test]
    fn rotate_rejects_single_slot_pool() {
        let client = cpu_client();
        let mut arena = HistArena::new(&client, 1, 16).unwrap();
        let res = arena.rotate();
        assert!(matches!(res, Err(ComputeError::Runtime { .. })));
    }

    /// D-02 round-trip on the cpu f64 anchor (never GPU-vs-GPU): after `rotate()`,
    /// driving `{parent, smaller}` through the VERBATIM `subtract_hist_kernel`
    /// (`out = parent - smaller`) with the arena's `larger` slot as the output lands
    /// the derived histogram in the `larger` (== old parent) slot — proving the
    /// in-place derivation contract. The allocation counter stays frozen.
    #[test]
    fn rotate_subtract_lands_in_larger_parent_slot() {
        use crate::kernels::subtract::subtract_hist_kernel;

        let client = cpu_client();
        let slot_len = 8usize; // 4 bins × 2 (stride-2 [g,h,g,h,…])
        let mut arena = HistArena::new(&client, 3, slot_len).unwrap();
        let old_parent = arena.parent_idx();

        arena.rotate().unwrap();
        // The larger child is derived into the old parent slot.
        assert_eq!(arena.larger_idx(), old_parent);

        // Representative parent / smaller-child stride-2 histograms.
        let parent_data = vec![10.0f64, 5.0, 8.0, 4.0, 9.0, 3.0, 7.0, 2.0];
        let smaller_data = vec![3.0f64, 2.0, 1.0, 1.0, 4.0, 1.0, 2.0, 1.0];
        let expected: Vec<f64> = parent_data
            .iter()
            .zip(&smaller_data)
            .map(|(p, c)| p - c)
            .collect();

        let h_parent = client.create_from_slice(f64::as_bytes(&parent_data));
        let h_smaller = client.create_from_slice(f64::as_bytes(&smaller_data));
        // The OUTPUT is the arena's larger slot (== old parent slot, in-place).
        let h_larger = arena.larger_handle();

        let allocs_before_launch = arena.device_allocations();

        // SAFETY: `h_parent`/`h_smaller` are each sized `slot_len` f64 cells, and the
        // arena's larger slot was allocated for `slot_len` f64 cells in `new`; all
        // three outlive the launch and the kernel touches only indices `0..slot_len`
        // (the grid over-covers but the `while i < n` bound guards every write).
        unsafe {
            subtract_hist_kernel::launch(
                &client,
                CubeCount::Static(64, 1, 1),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(h_parent, slot_len),
                ArrayArg::from_raw_parts(h_smaller, slot_len),
                ArrayArg::from_raw_parts(h_larger.clone(), slot_len),
            );
        }

        let bytes = client.read_one_unchecked(h_larger);
        let got = f64::from_bytes(&bytes).to_vec();
        assert_eq!(got, expected, "derived larger histogram must equal parent - smaller");

        // The whole round-trip allocated nothing in the arena (D-09).
        assert_eq!(
            arena.device_allocations(),
            allocs_before_launch,
            "rotate + subtract round-trip must not grow the arena allocation count"
        );
    }

    /// D-09 whole-pool SWAP: `set_leaf_slot` + `swap` reassign leaf→slot INDICES
    /// only (zero `client.empty`), the larger child inherits the parent slot, and
    /// the smaller child takes a fresh non-aliasing slot.
    #[test]
    fn swap_bookkeeping_no_alloc_no_alias() {
        let client = cpu_client();
        let mut arena = HistArena::new(&client, 3, 16).unwrap();
        let before = arena.device_allocations();
        arena.set_leaf_slot(0, 0).unwrap(); // parent leaf 0 lives in slot 0

        // Left is smaller → larger (right=2) inherits parent slot; smaller (left=1) fresh.
        arena.swap(0, 1, 2, true).unwrap();
        let larger_slot = arena.leaf_slot(2).unwrap();
        let smaller_slot = arena.leaf_slot(1).unwrap();
        assert_eq!(larger_slot, 0, "larger child must inherit the parent slot");
        assert_ne!(smaller_slot, larger_slot, "smaller must not alias the larger slot");
        assert_eq!(arena.device_allocations(), before, "swap must not allocate");

        // Mirror branch: right smaller → larger (left) inherits, smaller (right) fresh.
        arena.set_leaf_slot(3, 1).unwrap();
        arena.swap(3, 4, 5, false).unwrap();
        assert_eq!(arena.leaf_slot(4).unwrap(), 1, "larger (left) inherits parent slot");
        assert_ne!(arena.leaf_slot(5).unwrap(), 1, "smaller (right) must be fresh");
        assert_eq!(arena.device_allocations(), before, "swap must not allocate");
    }

    /// A 1-slot pool cannot supply a non-aliasing fresh slot — `swap()` rejects it
    /// with a typed error rather than aliasing (T-16-02-02).
    #[test]
    fn swap_rejects_single_slot_pool() {
        let client = cpu_client();
        let mut arena = HistArena::new(&client, 1, 16).unwrap();
        arena.set_leaf_slot(0, 0).unwrap();
        assert!(matches!(arena.swap(0, 1, 2, true), Err(ComputeError::Runtime { .. })));
    }

    /// D-09 round-trip on the cpu f64 anchor (never GPU-vs-GPU): after the
    /// leaf-indexed `swap()`, driving `{parent, smaller}` through the VERBATIM
    /// `subtract_hist_kernel` with the LARGER child's slot as output lands the
    /// `parent − smaller` result in the larger child's slot (== the old parent
    /// slot), proving the subtraction-trick reuse is correct. Allocations frozen.
    #[test]
    fn swap_subtract_lands_in_larger_slot() {
        use crate::kernels::subtract::subtract_hist_kernel;

        let client = cpu_client();
        let slot_len = 8usize; // 4 bins × 2 (stride-2 [g,h,g,h,…])
        let mut arena = HistArena::new(&client, 3, slot_len).unwrap();
        arena.set_leaf_slot(0, 0).unwrap(); // parent leaf 0 in slot 0

        // Left smaller: larger = right (leaf 2) inherits slot 0; smaller = left (leaf 1) fresh.
        arena.swap(0, 1, 2, true).unwrap();
        assert_eq!(arena.leaf_slot(2).unwrap(), 0);

        let parent_data = vec![10.0f64, 5.0, 8.0, 4.0, 9.0, 3.0, 7.0, 2.0];
        let smaller_data = vec![3.0f64, 2.0, 1.0, 1.0, 4.0, 1.0, 2.0, 1.0];
        let expected: Vec<f64> = parent_data
            .iter()
            .zip(&smaller_data)
            .map(|(p, c)| p - c)
            .collect();

        let h_parent = client.create_from_slice(f64::as_bytes(&parent_data));
        let h_smaller = client.create_from_slice(f64::as_bytes(&smaller_data));
        // OUTPUT is the LARGER child's slot (leaf 2 == old parent slot 0, in-place).
        let h_larger = arena.leaf_handle(2);
        let allocs_before = arena.device_allocations();

        // SAFETY: `h_parent`/`h_smaller` sized `slot_len` f64 cells; the larger slot
        // was allocated for `slot_len` f64 cells in `new`; all outlive the launch and
        // the kernel guards `i < n`.
        unsafe {
            subtract_hist_kernel::launch(
                &client,
                CubeCount::Static(64, 1, 1),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(h_parent, slot_len),
                ArrayArg::from_raw_parts(h_smaller, slot_len),
                ArrayArg::from_raw_parts(h_larger.clone(), slot_len),
            );
        }

        let bytes = client.read_one_unchecked(h_larger);
        let got = f64::from_bytes(&bytes).to_vec();
        assert_eq!(got, expected, "derived larger histogram must equal parent - smaller");
        assert_eq!(
            arena.device_allocations(),
            allocs_before,
            "swap + subtract round-trip must not grow the arena allocation count"
        );
    }
}
