//! SoA pre-allocated device split-record (`CUDASplitInfo` analog) — **14-04**.
//!
//! Owning plan: **14-04**. Scope locked by **ODL-02**, **D-05**, **D-06**, **D-08**.
//!
//! ## What lives here
//! A **Struct-of-Arrays, host pre-allocated** device split-record (the port of the
//! AMD-fork `include/LightGBM/cuda/cuda_split_info.hpp` `CUDASplitInfo`): one
//! CubeCL `Handle` **per field**, each sized `[num_leaf_slots]` (numeric) or
//! `[num_leaf_slots * MAX_CAT_PER_SPLIT]` (the reserved categorical slabs),
//! allocated **once** in [`DeviceSplitInfo::new`] via `client.empty(...)` and
//! indexed by leaf-slot (RESEARCH Pattern 5). There is **NO per-split / per-record
//! device allocation anywhere** (D-08) — the C++ `AllocateCatVectorsKernel`
//! per-split `cudaMalloc` is exactly the anti-pattern CubeCL pre-allocation
//! eliminates; the categorical slabs are reserved at construction so **Phase 22
//! fills them, it does not restructure them**.
//!
//! ## Host-staged this phase (D-07 / RESEARCH A6)
//! The C++ deep-copy `operator=` becomes a "copy leaf-slot `a` → leaf-slot `b`
//! across every field buffer". For Phase-14 correctness a **host-side index copy**
//! is sufficient (RESEARCH A6): the authoritative per-field values live in
//! host-pre-allocated `Vec`s ([`HostSlots`], allocated once alongside the device
//! handles), and [`DeviceSplitInfo::copy_slot`] is a pure host slot/slab copy that
//! performs **zero allocation** (no `client.empty` / `create_from_slice` / `Vec`
//! growth on the copy path). The device-kernel slot-copy (`buf[b] = buf[a]` per
//! field) and the 8/16-int packed readback packet are deferred to their §8/§17/§18
//! consumers (D-07) — the pre-allocated device handles are reserved here so that
//! consumer writes into them, never reallocating.
//!
//! ## Numeric fields (D-06)
//! `is_valid`, `leaf_index`, `gain`, `inner_feature_index`, `threshold` (u32),
//! `default_left`, and per-side (`left`/`right`) `sum_gradients`/`sum_hessians`
//! (f64), `sum_of_gradients_hessians` (i64 — stored as a plain i64 buffer; the u64
//! two's-complement ATOMIC accumulation idiom is a Phase-16 concern, only the
//! buffer is reserved here, RESEARCH Pitfall 2), `count` (i32), `gain` (f64),
//! `value` (f64). This mirrors the `CUDASplitInfo` field list exactly.
//!
//! ## Categorical fields RESERVED (D-06)
//! `num_cat_threshold` (i32 `[num_leaf_slots]`), `cat_threshold` (u32) and
//! `cat_threshold_real` (i32) pre-allocated as reserved slabs sized
//! `[num_leaf_slots * MAX_CAT_PER_SPLIT]`.
//!
//! ## Analog files
//! `crates/lgbm-compute/src/kernels/split.rs` (the `client.empty` "allocate only
//! when the kernel writes every cell" pre-alloc idiom) + `histogram.rs` (the
//! resident-pool "allocate once, reuse" convention and the `HIST_LDS_MAX`
//! comptime-const convention mirrored by [`MAX_CAT_PER_SPLIT`]).

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::error::ComputeError;

/// Reserved per-slot width of the categorical threshold slabs (`cat_threshold` /
/// `cat_threshold_real`), in elements — **RESOLVES RESEARCH Open Question 3**.
///
/// Set to the C++ `config_->max_cat_threshold` **default of 32**
/// (`LightGBM-release-4.6.0.99/include/LightGBM/config.h:486`,
/// `int max_cat_threshold = 32;`). This is a **Phase-22-tunable** constant: the
/// SoA layout is invariant to the cap — only the slab length
/// (`num_leaf_slots * MAX_CAT_PER_SPLIT`) changes if Phase 22 revises it — so
/// raising/lowering it never restructures the record (D-06). Mirrors the
/// `HIST_LDS_MAX` comptime-const convention in `histogram.rs`.
pub const MAX_CAT_PER_SPLIT: usize = 32;

/// The number of distinct device field buffers allocated by
/// [`DeviceSplitInfo::new`] — 18 numeric + `num_cat_threshold` + the two reserved
/// categorical slabs. Used by the structural "allocated exactly once" assertion
/// ([`DeviceSplitInfo::device_allocations`]).
pub const NUM_FIELD_BUFFERS: usize = 21;

/// The numeric (scalar) per-slot view of a split record — the `CUDASplitInfo`
/// numeric fields, all `Copy` so reading/writing a slot performs no allocation.
///
/// `num_cat_threshold` is the count of active categorical thresholds for the slot
/// (≤ [`MAX_CAT_PER_SPLIT`]); the threshold values themselves live in the reserved
/// slabs, accessed via [`DeviceSplitInfo::cat_threshold`] /
/// [`DeviceSplitInfo::set_cat_thresholds`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitScalars {
    /// Whether this slot holds a valid split (`CUDASplitInfo::is_valid`).
    pub is_valid: bool,
    /// The leaf this split applies to (`leaf_index`).
    pub leaf_index: i32,
    /// The split gain (`gain`, f64).
    pub gain: f64,
    /// The inner feature index chosen (`inner_feature_index`).
    pub inner_feature_index: i32,
    /// The bin threshold (`threshold`, u32).
    pub threshold: u32,
    /// Whether missing values route left (`default_left`).
    pub default_left: bool,

    /// Left child gradient sum (`left_sum_gradients`, f64).
    pub left_sum_gradients: f64,
    /// Left child hessian sum (`left_sum_hessians`, f64).
    pub left_sum_hessians: f64,
    /// Left child quantized grad/hess accumulator (`left_sum_of_gradients_hessians`,
    /// i64 — reserved buffer only this phase, Pitfall 2).
    pub left_sum_gh_quant: i64,
    /// Left child row count (`left_count`, i32).
    pub left_count: i32,
    /// Left child gain (`left_gain`, f64).
    pub left_gain: f64,
    /// Left child leaf value/output (`left_value`, f64).
    pub left_value: f64,

    /// Right child gradient sum (`right_sum_gradients`, f64).
    pub right_sum_gradients: f64,
    /// Right child hessian sum (`right_sum_hessians`, f64).
    pub right_sum_hessians: f64,
    /// Right child quantized grad/hess accumulator (`right_sum_of_gradients_hessians`,
    /// i64 — reserved buffer only this phase).
    pub right_sum_gh_quant: i64,
    /// Right child row count (`right_count`, i32).
    pub right_count: i32,
    /// Right child gain (`right_gain`, f64).
    pub right_gain: f64,
    /// Right child leaf value/output (`right_value`, f64).
    pub right_value: f64,

    /// Count of active categorical thresholds for this slot (`num_cat_threshold`),
    /// ≤ [`MAX_CAT_PER_SPLIT`].
    pub num_cat_threshold: i32,
}

impl Default for SplitScalars {
    fn default() -> Self {
        SplitScalars {
            is_valid: false,
            leaf_index: -1,
            gain: 0.0,
            inner_feature_index: -1,
            threshold: 0,
            default_left: false,
            left_sum_gradients: 0.0,
            left_sum_hessians: 0.0,
            left_sum_gh_quant: 0,
            left_count: 0,
            left_gain: 0.0,
            left_value: 0.0,
            right_sum_gradients: 0.0,
            right_sum_hessians: 0.0,
            right_sum_gh_quant: 0,
            right_count: 0,
            right_gain: 0.0,
            right_value: 0.0,
            num_cat_threshold: 0,
        }
    }
}

/// The host-pre-allocated SoA field buffers (allocated once in
/// [`DeviceSplitInfo::new`]). These are the authoritative Phase-14 storage; the
/// host slot-copy ([`DeviceSplitInfo::copy_slot`]) mutates these in place with no
/// allocation. Phase 17 writes the device handles directly via kernels (D-07).
struct HostSlots {
    is_valid: Vec<u8>,
    leaf_index: Vec<i32>,
    gain: Vec<f64>,
    inner_feature_index: Vec<i32>,
    threshold: Vec<u32>,
    default_left: Vec<u8>,

    left_sum_gradients: Vec<f64>,
    left_sum_hessians: Vec<f64>,
    left_sum_gh_quant: Vec<i64>,
    left_count: Vec<i32>,
    left_gain: Vec<f64>,
    left_value: Vec<f64>,

    right_sum_gradients: Vec<f64>,
    right_sum_hessians: Vec<f64>,
    right_sum_gh_quant: Vec<i64>,
    right_count: Vec<i32>,
    right_gain: Vec<f64>,
    right_value: Vec<f64>,

    num_cat_threshold: Vec<i32>,
    cat_threshold: Vec<u32>,
    cat_threshold_real: Vec<i32>,
}

/// The pre-allocated reserved device-resident SoA record — one CubeCL [`Handle`]
/// per field, allocated **once** in [`DeviceSplitInfo::new`] (D-05/D-08). Phase 17
/// argmax/copy kernels write into these handles; Phase 14 reserves them.
pub struct DeviceBuffers {
    /// `is_valid` (`[num_leaf_slots]` u8).
    pub is_valid: Handle,
    /// `leaf_index` (`[num_leaf_slots]` i32).
    pub leaf_index: Handle,
    /// `gain` (`[num_leaf_slots]` f64).
    pub gain: Handle,
    /// `inner_feature_index` (`[num_leaf_slots]` i32).
    pub inner_feature_index: Handle,
    /// `threshold` (`[num_leaf_slots]` u32).
    pub threshold: Handle,
    /// `default_left` (`[num_leaf_slots]` u8).
    pub default_left: Handle,

    /// `left_sum_gradients` (`[num_leaf_slots]` f64).
    pub left_sum_gradients: Handle,
    /// `left_sum_hessians` (`[num_leaf_slots]` f64).
    pub left_sum_hessians: Handle,
    /// `left_sum_of_gradients_hessians` (`[num_leaf_slots]` i64, reserved).
    pub left_sum_gh_quant: Handle,
    /// `left_count` (`[num_leaf_slots]` i32).
    pub left_count: Handle,
    /// `left_gain` (`[num_leaf_slots]` f64).
    pub left_gain: Handle,
    /// `left_value` (`[num_leaf_slots]` f64).
    pub left_value: Handle,

    /// `right_sum_gradients` (`[num_leaf_slots]` f64).
    pub right_sum_gradients: Handle,
    /// `right_sum_hessians` (`[num_leaf_slots]` f64).
    pub right_sum_hessians: Handle,
    /// `right_sum_of_gradients_hessians` (`[num_leaf_slots]` i64, reserved).
    pub right_sum_gh_quant: Handle,
    /// `right_count` (`[num_leaf_slots]` i32).
    pub right_count: Handle,
    /// `right_gain` (`[num_leaf_slots]` f64).
    pub right_gain: Handle,
    /// `right_value` (`[num_leaf_slots]` f64).
    pub right_value: Handle,

    /// `num_cat_threshold` (`[num_leaf_slots]` i32).
    pub num_cat_threshold: Handle,
    /// `cat_threshold` reserved slab (`[num_leaf_slots * MAX_CAT_PER_SPLIT]` u32).
    pub cat_threshold: Handle,
    /// `cat_threshold_real` reserved slab (`[num_leaf_slots * MAX_CAT_PER_SPLIT]` i32).
    pub cat_threshold_real: Handle,
}

/// A CubeCL-safe, host-pre-allocated SoA device split-record (the `CUDASplitInfo`
/// analog) with **no per-split device allocation** (ODL-02 / D-05 / D-06 / D-08).
///
/// See the module docs for the field list, the reserved categorical slabs, and the
/// host-staged slot-copy contract.
pub struct DeviceSplitInfo<R: cubecl::Runtime> {
    /// The reserved device-resident SoA handles (allocated once, D-08).
    pub device: DeviceBuffers,
    /// Host-pre-allocated authoritative field buffers (allocated once).
    host: HostSlots,
    /// Number of leaf slots the record is sized for.
    num_leaf_slots: usize,
    /// Count of `client.empty` device allocations performed — must equal
    /// [`NUM_FIELD_BUFFERS`] (every allocation happens in [`Self::new`]).
    device_allocations: usize,
    _runtime: PhantomData<R>,
}

impl<R: cubecl::Runtime> DeviceSplitInfo<R> {
    /// Allocate the SoA record for `num_leaf_slots` slots — **one `client.empty`
    /// per field, exactly once** (D-05/D-08). No allocation happens anywhere else
    /// (no per-split / per-record device alloc).
    ///
    /// The categorical slabs are reserved at `num_leaf_slots * MAX_CAT_PER_SPLIT`
    /// (D-06). Slab sizing is computed in `usize` and overflow-checked at the V5
    /// boundary (threat T-14-04-03).
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `num_leaf_slots == 0`, or if the reserved
    /// categorical slab length `num_leaf_slots * MAX_CAT_PER_SPLIT` overflows
    /// `usize`.
    pub fn new(
        client: &ComputeClient<R>,
        num_leaf_slots: usize,
    ) -> Result<Self, ComputeError> {
        if num_leaf_slots == 0 {
            return Err(ComputeError::Runtime {
                detail: "DeviceSplitInfo::new: num_leaf_slots must be >= 1".to_string(),
            });
        }
        // T-14-04-03: slab sizing in usize, checked for overflow before any alloc.
        let cat_slab_len = num_leaf_slots.checked_mul(MAX_CAT_PER_SPLIT).ok_or_else(|| {
            ComputeError::Runtime {
                detail: format!(
                    "DeviceSplitInfo::new: categorical slab length \
                     {num_leaf_slots} * {MAX_CAT_PER_SPLIT} overflows usize"
                ),
            }
        })?;

        // Count every device allocation so the "allocated exactly once" invariant
        // is structurally verifiable: this closure is the ONLY caller of
        // `client.empty` in the whole module, and it runs only here in `new`.
        let mut device_allocations = 0usize;
        let mut alloc = |elem_size: usize, len: usize| -> Handle {
            device_allocations += 1;
            client.empty(len * elem_size)
        };

        let n = num_leaf_slots;
        let device = DeviceBuffers {
            is_valid: alloc(core::mem::size_of::<u8>(), n),
            leaf_index: alloc(core::mem::size_of::<i32>(), n),
            gain: alloc(core::mem::size_of::<f64>(), n),
            inner_feature_index: alloc(core::mem::size_of::<i32>(), n),
            threshold: alloc(core::mem::size_of::<u32>(), n),
            default_left: alloc(core::mem::size_of::<u8>(), n),

            left_sum_gradients: alloc(core::mem::size_of::<f64>(), n),
            left_sum_hessians: alloc(core::mem::size_of::<f64>(), n),
            left_sum_gh_quant: alloc(core::mem::size_of::<i64>(), n),
            left_count: alloc(core::mem::size_of::<i32>(), n),
            left_gain: alloc(core::mem::size_of::<f64>(), n),
            left_value: alloc(core::mem::size_of::<f64>(), n),

            right_sum_gradients: alloc(core::mem::size_of::<f64>(), n),
            right_sum_hessians: alloc(core::mem::size_of::<f64>(), n),
            right_sum_gh_quant: alloc(core::mem::size_of::<i64>(), n),
            right_count: alloc(core::mem::size_of::<i32>(), n),
            right_gain: alloc(core::mem::size_of::<f64>(), n),
            right_value: alloc(core::mem::size_of::<f64>(), n),

            num_cat_threshold: alloc(core::mem::size_of::<i32>(), n),
            cat_threshold: alloc(core::mem::size_of::<u32>(), cat_slab_len),
            cat_threshold_real: alloc(core::mem::size_of::<i32>(), cat_slab_len),
        };

        let host = HostSlots {
            is_valid: vec![0u8; n],
            leaf_index: vec![-1i32; n],
            gain: vec![0.0f64; n],
            inner_feature_index: vec![-1i32; n],
            threshold: vec![0u32; n],
            default_left: vec![0u8; n],

            left_sum_gradients: vec![0.0f64; n],
            left_sum_hessians: vec![0.0f64; n],
            left_sum_gh_quant: vec![0i64; n],
            left_count: vec![0i32; n],
            left_gain: vec![0.0f64; n],
            left_value: vec![0.0f64; n],

            right_sum_gradients: vec![0.0f64; n],
            right_sum_hessians: vec![0.0f64; n],
            right_sum_gh_quant: vec![0i64; n],
            right_count: vec![0i32; n],
            right_gain: vec![0.0f64; n],
            right_value: vec![0.0f64; n],

            num_cat_threshold: vec![0i32; n],
            cat_threshold: vec![0u32; cat_slab_len],
            cat_threshold_real: vec![0i32; cat_slab_len],
        };

        Ok(DeviceSplitInfo {
            device,
            host,
            num_leaf_slots,
            device_allocations,
            _runtime: PhantomData,
        })
    }

    /// The number of leaf slots this record is sized for.
    #[must_use]
    pub fn num_leaf_slots(&self) -> usize {
        self.num_leaf_slots
    }

    /// The number of device buffers allocated — equals [`NUM_FIELD_BUFFERS`] after
    /// [`Self::new`] and never changes (proves "allocated exactly once": no
    /// per-split / per-copy alloc, D-08).
    #[must_use]
    pub fn device_allocations(&self) -> usize {
        self.device_allocations
    }

    /// Read the numeric scalar fields of `slot` (no allocation).
    ///
    /// This is an infallible-by-contract HOST helper (IN-01): unlike the
    /// device-facing write paths (`set_scalars` / `set_cat_thresholds` /
    /// `copy_slot`), which return a typed [`ComputeError::Runtime`] via
    /// `check_slot` because their inputs cross the V5 boundary (T-14-04-01), the
    /// read accessors take a host-supplied `slot` index and follow standard Rust
    /// slice-indexing convention — an out-of-range slot is a caller bug, asserted
    /// (panic), not a recoverable runtime error. Exempt from V5 by contract.
    ///
    /// # Panics
    /// If `slot >= num_leaf_slots`.
    #[must_use]
    pub fn scalars(&self, slot: usize) -> SplitScalars {
        assert!(slot < self.num_leaf_slots, "slot index out of range");
        let h = &self.host;
        SplitScalars {
            is_valid: h.is_valid[slot] != 0,
            leaf_index: h.leaf_index[slot],
            gain: h.gain[slot],
            inner_feature_index: h.inner_feature_index[slot],
            threshold: h.threshold[slot],
            default_left: h.default_left[slot] != 0,
            left_sum_gradients: h.left_sum_gradients[slot],
            left_sum_hessians: h.left_sum_hessians[slot],
            left_sum_gh_quant: h.left_sum_gh_quant[slot],
            left_count: h.left_count[slot],
            left_gain: h.left_gain[slot],
            left_value: h.left_value[slot],
            right_sum_gradients: h.right_sum_gradients[slot],
            right_sum_hessians: h.right_sum_hessians[slot],
            right_sum_gh_quant: h.right_sum_gh_quant[slot],
            right_count: h.right_count[slot],
            right_gain: h.right_gain[slot],
            right_value: h.right_value[slot],
            num_cat_threshold: h.num_cat_threshold[slot],
        }
    }

    /// Write the numeric scalar fields of `slot` (no allocation). Does NOT touch the
    /// categorical slabs (use [`Self::set_cat_thresholds`]); `num_cat_threshold` is
    /// written from `scalars.num_cat_threshold`.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `slot >= num_leaf_slots`.
    pub fn set_scalars(&mut self, slot: usize, scalars: &SplitScalars) -> Result<(), ComputeError> {
        self.check_slot(slot)?;
        let h = &mut self.host;
        h.is_valid[slot] = u8::from(scalars.is_valid);
        h.leaf_index[slot] = scalars.leaf_index;
        h.gain[slot] = scalars.gain;
        h.inner_feature_index[slot] = scalars.inner_feature_index;
        h.threshold[slot] = scalars.threshold;
        h.default_left[slot] = u8::from(scalars.default_left);
        h.left_sum_gradients[slot] = scalars.left_sum_gradients;
        h.left_sum_hessians[slot] = scalars.left_sum_hessians;
        h.left_sum_gh_quant[slot] = scalars.left_sum_gh_quant;
        h.left_count[slot] = scalars.left_count;
        h.left_gain[slot] = scalars.left_gain;
        h.left_value[slot] = scalars.left_value;
        h.right_sum_gradients[slot] = scalars.right_sum_gradients;
        h.right_sum_hessians[slot] = scalars.right_sum_hessians;
        h.right_sum_gh_quant[slot] = scalars.right_sum_gh_quant;
        h.right_count[slot] = scalars.right_count;
        h.right_gain[slot] = scalars.right_gain;
        h.right_value[slot] = scalars.right_value;
        h.num_cat_threshold[slot] = scalars.num_cat_threshold;
        Ok(())
    }

    /// Write the active categorical thresholds of `slot` into the reserved slabs and
    /// set `num_cat_threshold[slot]` to their length (no allocation). The two slices
    /// must be the same length and fit the reserved [`MAX_CAT_PER_SPLIT`] width.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `slot >= num_leaf_slots`.
    /// [`ComputeError::LengthMismatch`] if the two slices differ in length.
    /// [`ComputeError::Runtime`] if `thresholds.len() > MAX_CAT_PER_SPLIT`.
    pub fn set_cat_thresholds(
        &mut self,
        slot: usize,
        thresholds: &[u32],
        thresholds_real: &[i32],
    ) -> Result<(), ComputeError> {
        self.check_slot(slot)?;
        if thresholds.len() != thresholds_real.len() {
            return Err(ComputeError::LengthMismatch {
                expected: thresholds.len(),
                actual: thresholds_real.len(),
            });
        }
        if thresholds.len() > MAX_CAT_PER_SPLIT {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "DeviceSplitInfo::set_cat_thresholds: {} thresholds exceed the reserved \
                     MAX_CAT_PER_SPLIT = {MAX_CAT_PER_SPLIT} slab width (Phase-22-tunable)",
                    thresholds.len()
                ),
            });
        }
        let base = slot * MAX_CAT_PER_SPLIT;
        let len = thresholds.len();
        self.host.cat_threshold[base..base + len].copy_from_slice(thresholds);
        self.host.cat_threshold_real[base..base + len].copy_from_slice(thresholds_real);
        self.host.num_cat_threshold[slot] = len as i32;
        Ok(())
    }

    /// The active categorical thresholds of `slot` (length `num_cat_threshold[slot]`),
    /// borrowing the reserved slab (no allocation).
    ///
    /// Infallible-by-contract host read accessor, exempt from V5 — see
    /// [`Self::scalars`] (IN-01).
    ///
    /// # Panics
    /// If `slot >= num_leaf_slots`.
    #[must_use]
    pub fn cat_threshold(&self, slot: usize) -> &[u32] {
        assert!(slot < self.num_leaf_slots, "slot index out of range");
        let len = self.host.num_cat_threshold[slot].max(0) as usize;
        let base = slot * MAX_CAT_PER_SPLIT;
        &self.host.cat_threshold[base..base + len]
    }

    /// The active real categorical thresholds of `slot` (length
    /// `num_cat_threshold[slot]`), borrowing the reserved slab (no allocation).
    ///
    /// Infallible-by-contract host read accessor, exempt from V5 — see
    /// [`Self::scalars`] (IN-01).
    ///
    /// # Panics
    /// If `slot >= num_leaf_slots`.
    #[must_use]
    pub fn cat_threshold_real(&self, slot: usize) -> &[i32] {
        assert!(slot < self.num_leaf_slots, "slot index out of range");
        let len = self.host.num_cat_threshold[slot].max(0) as usize;
        let base = slot * MAX_CAT_PER_SPLIT;
        &self.host.cat_threshold_real[base..base + len]
    }

    /// Deep-copy leaf-slot `src` → leaf-slot `dst` across **every** field buffer
    /// (the `CUDASplitInfo::operator=` analog) — host-side this phase, with
    /// **zero allocation** on the copy path (D-07 / RESEARCH A6). Copies the numeric
    /// scalars and the FULL reserved categorical slab window
    /// (`[src*W, src*W+W) → [dst*W, dst*W+W)`), so `dst` is byte-identical to `src`
    /// and all other slots are untouched.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if either `src` or `dst` is `>= num_leaf_slots`.
    pub fn copy_slot(&mut self, src: usize, dst: usize) -> Result<(), ComputeError> {
        self.check_slot(src)?;
        self.check_slot(dst)?;
        if src == dst {
            return Ok(());
        }
        let h = &mut self.host;
        // Numeric scalars: one element copy per field buffer (no alloc).
        h.is_valid[dst] = h.is_valid[src];
        h.leaf_index[dst] = h.leaf_index[src];
        h.gain[dst] = h.gain[src];
        h.inner_feature_index[dst] = h.inner_feature_index[src];
        h.threshold[dst] = h.threshold[src];
        h.default_left[dst] = h.default_left[src];
        h.left_sum_gradients[dst] = h.left_sum_gradients[src];
        h.left_sum_hessians[dst] = h.left_sum_hessians[src];
        h.left_sum_gh_quant[dst] = h.left_sum_gh_quant[src];
        h.left_count[dst] = h.left_count[src];
        h.left_gain[dst] = h.left_gain[src];
        h.left_value[dst] = h.left_value[src];
        h.right_sum_gradients[dst] = h.right_sum_gradients[src];
        h.right_sum_hessians[dst] = h.right_sum_hessians[src];
        h.right_sum_gh_quant[dst] = h.right_sum_gh_quant[src];
        h.right_count[dst] = h.right_count[src];
        h.right_gain[dst] = h.right_gain[src];
        h.right_value[dst] = h.right_value[src];
        h.num_cat_threshold[dst] = h.num_cat_threshold[src];
        // Categorical reserved slabs: copy the whole MAX_CAT_PER_SPLIT-wide window
        // in place (`copy_within` — no allocation).
        let w = MAX_CAT_PER_SPLIT;
        h.cat_threshold.copy_within(src * w..src * w + w, dst * w);
        h.cat_threshold_real.copy_within(src * w..src * w + w, dst * w);
        Ok(())
    }

    /// V5 slot-index validation (threat T-14-04-01): reject out-of-bounds slots with
    /// a typed error before any indexed write.
    fn check_slot(&self, slot: usize) -> Result<(), ComputeError> {
        if slot >= self.num_leaf_slots {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "DeviceSplitInfo: slot index {slot} out of range \
                     (num_leaf_slots = {})",
                    self.num_leaf_slots
                ),
            });
        }
        Ok(())
    }
}
