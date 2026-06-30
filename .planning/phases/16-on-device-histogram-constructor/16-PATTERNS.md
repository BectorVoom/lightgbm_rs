# Phase 16: On-Device Histogram Constructor - Pattern Map

**Mapped:** 2026-07-01
**Files analyzed:** 7 (2 new, 5 modified/extended) + 2 test files
**Analogs found:** 9 / 9 (every primitive already exists; this is a composition phase)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/kernels/histogram.rs` (EXTEND: two-tier build on §13 geometry, dense+sparse × shared+global, de-quant placement) | kernel + launcher | streaming / scatter-accumulate (pub-sub atomics) | `construct_leaf_hist_resident_lds_kernel_u64` (same file, lines 1264-1314) | exact (same accumulation idiom, new geometry) |
| `crates/lgbm-compute/src/kernels/histogram.rs` (EXTEND: dequant + FixHistogram, DROP compact) | kernel | transform / reduce | `fix_compact_kernel` (same file, lines 2331-2428) | exact (reuse, omit compact block) |
| `crates/lgbm-compute/src/kernels/subtract.rs` (REUSE as-is) | kernel | transform (element-wise) | `subtract_hist_kernel` f64/f32/`_vec` (lines 43-106) | exact (no change) |
| `crates/lgbm-compute/src/kernels/histogram_arena.rs` (NEW: `hist_t**` pool + handle rotation) | store / arena | request-response (host bookkeeping) | `DeviceSplitInfo::new` (split_info.rs, lines 286-321) | role-match (once-alloc pattern) |
| `crates/lgbm-compute/src/kernels/primitives.rs` (REUSE: Fix reduce) | utility | reduce | `reduce_sum_body` (lines 412-424) + `plane_reduce_sum_f32_on` (1456) | exact (no change) |
| `crates/lgbm-compute/src/kernels/row_data.rs` (REUSE: §13 geometry inputs) | model / config | config | `FeaturePartitionLayout` + `divide_cuda_feature_groups` (lines 43-64, 74) | exact (Phase-15, no change) |
| `crates/lgbm-compute/src/kernels/column_data.rs` (REUSE: Fix per-feature scalars) | model | config | `ColumnFeatureMeta` / `CudaColumnData` (lines 32, 65) | exact (no change) |
| `crates/lgbm-compute/src/lib.rs` (`on_device_growth_supported` stays false; new launcher hangs off the build path) | provider / entry | request-response | `grow_tree_on_device` (lines 1239-1290) | role-match (seam stays no-op) |
| `crates/oracle-harness/tests/kernel_parity.rs` (EXTEND: build/fix/subtract goldens) | test | golden replay | `kernel_parity_histogram_bit_exact_on_cpu` (line 162) | exact (extend) |
| `crates/lgbm-compute/tests/rocm_cuda_mirror.rs` (EXTEND: two-tier + sparse + spill cases) | test | parity assert | `cpu_anchor` + `assert_close` (lines 40-109) | exact (extend scaffold) |

## Pattern Assignments

### `histogram.rs` — two-tier BUILD kernel (kernel, scatter-accumulate)

**Analog:** `construct_leaf_hist_resident_lds_kernel_u64<B: Int>` (`histogram.rs:1264-1314`)

**Signature + gate to copy** (1264-1275):
```rust
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn construct_leaf_hist_resident_lds_kernel_u64<B: Int>(
    resident_bins: &Array<B>,   // native bin width (u8/u16/u32) — qix
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,      // length num_features + 1 (sentinel = slot_len)
    num_data: usize,
    out: &mut Array<Atomic<u64>>,
) {
```

**Core three-phase LDS body to lift onto §13 geometry** (1282-1314):
```rust
let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_MAX);
// 1. zero this feature's active LDS cells (u64 zero = additive identity bits)
let mut c = UNIT_POS as usize;
while c < feat_len { sub[c].store(0u64); c += cd; }
sync_cube();
// 2. scatter rows into LDS, quantizing round(v*2^30) → i64 → u64 bits
let bin = u32::cast_from(resident_bins[col + leaf_rows[k] as usize]) as usize;
let ti = bin * 2;                                   // [2b] grad / [2b+1] hess
let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE_F32)));
let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE_F32)));
sub[ti].fetch_add(qg);
sub[ti + 1].fetch_add(qh);
sync_cube();
// 3. merge LDS → global slot (wrapping u64 add == i64 two's-complement)
let mut m = UNIT_POS as usize;
while m < feat_len { out[base + m].fetch_add(sub[m].load()); m += cd; }
```

**Phase-16 change (per inline comment 1289-1290 + D-03):** map `f = CUBE_POS_X` → **partition** (blockIdx.x); add `threadIdx.x` = column within partition; `threadIdx.y × blockIdx.y` = row stripes; the step-3 merge becomes the cross-block `atomicAdd_system`-equiv (still `out[...].fetch_add` — there is NO grid barrier in cubecl 0.10). The shipped per-feature kernel above stays **byte-unchanged and coexists**.

**`SCALE_F32` constant** (`histogram.rs:635`): `const SCALE_F32: f32 = 1_073_741_824.0; // 2^30`

**cpu anchor twin** (the D-06 single-owner side) — `construct_histograms_f64_on` (144-190), launched at `CubeCount::Static(1,1,1)` + `CubeDim::new_1d(1)` (180), accumulating into a caller-**zeroed** f64 `out` (166-167). The cpu anchor uses NO atomics; the hip path uses the LDS atomics above. ONE `#[cube]` generic, comptime/runtime-split (D-06).

**HARD CONSTRAINT (1258-1264):** cell type MUST be `Atomic<u64>` with `.store(0u64)` / `.fetch_add(qbits)`. NEVER `Atomic<i64>` — cubecl-hip 0.10 link-fails. `wrapping_add` is NOT a kernel intrinsic — the atomic wraps natively.

---

### `histogram.rs` — DEQUANT + FixHistogram (kernel, transform/reduce)

**Analog:** `fix_compact_kernel` (`histogram.rs:2331-2428`) — reuse the dequant + Fix logic, **DROP the compact block** (Pitfall 5; §7 does build→fix→subtract only).

**Dequant-once first pass** (2348-2367):
```rust
const SCALE_F64: f64 = 1_073_741_824.0; // 2^30 (matches build-side SCALE_F32; f64 here)
for w in 0..nb {
    let wbi = base + (w as usize) * 2;
    hist[wbi]     = f64::cast_from(i64::cast_from(h_raw[wbi]))     / SCALE_F64;
    hist[wbi + 1] = f64::cast_from(i64::cast_from(h_raw[wbi + 1])) / SCALE_F64;
}
```

**FixHistogram repair — cpu-anchor serial fold (load-bearing ASCENDING order)** (2373-2398):
```rust
let do_fix = mfb > 0 && mfb < nb;          // skip mfb==0 and out-of-range (Pitfall 4)
if do_fix {
    let mfbu = mfb as usize;
    let mut g = 0.0f64; let mut h = 0.0f64;
    g += sum_gradient;  h += sum_hessian;  // RAW leaf totals (host f64, never quantized)
    let count = nb;
    for i in 0..count {                     // ASCENDING — never reorder on the anchor path
        let bi = base + (i as usize) * 2;
        let take = i != mfb;                // branchless exclude of the most-freq bin
        g -= select(take, hist[bi],     0.0);
        h -= select(take, hist[bi + 1], 0.0);
    }
    let mi = base + mfbu * 2;
    hist[mi] = g; hist[mi + 1] = h;         // feat_hist[mfb·2] = leaf_total − Σ
}
```

**DROP this block (Pitfall 5)** — the compact step at `histogram.rs:2400-2427` (`if off > 0 { ... }`) is a CPU-learner artifact; the §7 on-device path does not compact.

**hip variant (D-06):** replace the ascending fold with `ShuffleReduceSum`/`plane_sum` over `num_bin_aligned`. See `primitives.rs::plane_reduce_sum_f32_on` (1456). Assumption A2: if `num_bin_aligned` > plane width, needs multi-plane reduce; cpu anchor unaffected.

**V5 launcher checks to mirror** (`fix_compact_f64_on`, 2445-2448): `num_bin == 0` → typed error; `2*num_bin` overflow → typed error; `slot_off + 2*num_bin > raw.len()` → `ComputeError::LengthMismatch`; empty feats → `Ok` with NO launch.

---

### `subtract.rs` — SubtractHistogram (kernel, element-wise transform) — REUSE VERBATIM

**Analog:** `subtract_hist_kernel` (`subtract.rs:43-52`)
```rust
#[cube(launch)]
pub fn subtract_hist_kernel(parent: &Array<f64>, child: &Array<f64>, out: &mut Array<f64>) {
    let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
    let mut i = ABSOLUTE_POS as usize;
    let n = parent.len() as usize;
    while i < n { out[i] = parent[i] - child[i]; i += stride; }
}
```
- `larger = parent − smaller`; bit-exact-by-construction per cell (no atomics/reduction/ordering).
- f32 mirror: `subtract_hist_kernel_f32` (62-70) for the no-f64 hip device.
- SIMD twin: `subtract_hist_kernel_vec<F: Float, N: Size>` (93-106), gated behind `n % N == 0 && N > 1` via `pick_vec_width` (116).
- **Host guard (D-02):** apply the subtract only when `larger.leaf_index ≥ 0`, and only **after** the parent build is fully synced (Pitfall 1 / 8aed100 ordering invariant).

---

### `histogram_arena.rs` (NEW) — `hist_t**` pool + handle rotation (store, host bookkeeping)

**Analog:** `DeviceSplitInfo::new` allocate-exactly-once pattern (`split_info.rs:286-321`)
```rust
// Count every device allocation so "allocated exactly once" is structurally verifiable:
// this closure is the ONLY caller of client.empty in the module, runs only in new().
let mut device_allocations = 0usize;
let mut alloc = |elem_size: usize, len: usize| -> Handle {
    device_allocations += 1;
    client.empty(len * elem_size)
};
// ... one alloc(...) per buffer; assert device_allocations == NUM_FIELD_BUFFERS afterwards.
```

**Phase-16 application (D-02 / D-09):** a `USED_HISTOGRAM_BUFFER_NUM`-slot `hist_t` pool, allocated once in `new` (one `client.empty` per slot, counted + asserted). Rotation reassigns **slot indices**, never reallocates: `larger ← parent's slot index` (in-place), `smaller ← a fresh slot`. Recommended representation (Open Q2): a small `HistArena { slots: Vec<Handle>, parent_idx, smaller_idx, larger_idx }`. Anchor-test the index bookkeeping in isolation; the cross-tree `SplitTreeStructureKernel` swap is **Phase 18** — do not build it here.

**V5 (Security V5 / split_info.rs:276-284):** slab sizing in `usize` with `checked_mul` before any alloc; reject `num_*_slots == 0`.

---

### Shared geometry inputs — `row_data.rs` / `column_data.rs` (REUSE, no change)

**`FeaturePartitionLayout`** (`row_data.rs:43-64`) — fields the build launcher consumes directly:
`feature_partition_column_index_offsets` (partition `i` owns columns `[off[i], off[i+1])`), `column_hist_offsets` (partition-local per-column bin offset), `partition_hist_offsets` (global per-partition bin offset), `max_num_column_per_partition` (= `block_dim_x`), `num_feature_partitions` (= `grid_dim_x`), `num_large_bin_partition` (gates the `_GlobalMemory` spill, D-04), `shared_hist_size` (SMEM budget; `max_num_bin_per_partition = shared_hist_size / 2`). Produced by `divide_cuda_feature_groups(num_bin_per_column, shared_hist_size)` (74).

**Fix per-feature scalars** — `column_data.rs` `ColumnFeatureMeta` (32) supplies `most_freq_bin`, `num_bin`, `offset`.

## Shared Patterns

### Anchor: cpu f64 fold, never GPU-vs-GPU (D-06)
**Source:** `histogram.rs:144-190` (`construct_histograms_f64_on`, `CubeDim::new_1d(1)`, line 180) + `primitives.rs:412-424` (`reduce_sum_body`, `if UNIT_POS == 0` single-owner ascending fold).
**Apply to:** every accumulating kernel (build + Fix reduce). cpu anchor = single-owner ordered fold (no atomics — cubecl-cpu atomics nondeterministic, Pitfall 7); hip = two-tier LDS atomics / plane reduce. ONE `#[cube]` generic, comptime/runtime-split.

### Allocate exactly once outside the hot loop (D-09)
**Source:** `split_info.rs:286-321`.
**Apply to:** the histogram arena + the `_GlobalMemory` spill buffer (sized `grid_dim_y · num_total_bin · {4 DP, 2 SP}`, D-04 / Pitfall 6). Counted + asserted; never `client.empty` in the per-tree loop.

### Caller must zero accumulation buffers before launch
**Source:** `histogram.rs:161-167` — `client.empty` returns UNINITIALIZED pooled memory; allocate the accumulation `out` from an explicit zero slice (`vec![0.0; out_len]` → `create_from_slice`) so a fresh launch does not fold onto stale values.
**Apply to:** the build `out` (u64 zero bits) and the Fix `hist` f64 output.

### V5 launch-arg bounds before `launch_unchecked` (Security V5)
**Source:** `histogram.rs:2445-2448` + `split_info.rs:276-284`.
**Apply to:** every new launcher — `num_bin > 0`, `2*num_bin` overflow, `slot_off + region ≤ raw.len()`, arena slot indices in range; typed `ComputeError`. `#[cfg(feature = "gpu")]`-gate all device kernels (cpu anchor builds without it).

### Ordering invariant: build-fully-synced-before-subtract (Pitfall 1, 8aed100)
**Source:** spike-findings / debug 8aed100; guard mandated by D-05.
**Apply to:** the build→fix→subtract sequence — the parent (or fused smaller) histogram MUST be fully built + synced before any child subtract reads it. With no grid barrier, the "sync" is the kernel boundary between build and subtract.

### Test scaffold: cpu_anchor + assert_close(tol)
**Source:** `rocm_cuda_mirror.rs:40-109`.
```rust
fn assert_close(anchor: &[f64], gpu: &[f64], what: &str) {
    const ABS: f64 = 5e-6;  const REL: f64 = 1e-5;   // f32-atomic envelope (04-ROCM-GAPS)
    for (i, (a, b)) in anchor.iter().zip(gpu).enumerate() {
        let tol = ABS + REL * a.abs();
        assert!((a - b).abs() <= tol, "{what}: cell {i} ...");
    }
}
```
**Apply to:** extend with §13-aware cases — sparse `row_ptr_type` {16,32,64}, large-bin/global-spill, `most_freq_bin ≠ 0` Fix repair, build-before-subtract ordering, `[2b]/[2b+1]` interleave assert. Anchor to the cpu f64 fold (`construct_histograms_cpu`, used in `cpu_anchor` at 94-109); **never GPU-vs-GPU** (def-f8u-01). Golden replay home: `kernel_parity.rs:162` (`kernel_parity_histogram_bit_exact_on_cpu`).

## No Analog Found

None. Every primitive, launcher idiom, arena pattern, geometry input, and test scaffold this phase needs already exists in `crates/lgbm-compute/`. Phase 16 is composition + restructuring, not new-primitive work. The only genuinely net-new code is the **wiring** (the §13-geometry remap of the existing u64 LDS body, the `_GlobalMemory` spill variant, and the arena handle struct) — all built from the analogs above.

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/kernels/` (histogram, subtract, primitives, split_info, row_data, column_data), `crates/lgbm-compute/src/lib.rs`, `crates/lgbm-compute/tests/`, `crates/oracle-harness/tests/`
**Files scanned:** 10
**Pattern extraction date:** 2026-07-01
