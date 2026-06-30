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
}
