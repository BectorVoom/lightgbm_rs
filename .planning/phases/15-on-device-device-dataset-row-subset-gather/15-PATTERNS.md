# Phase 15: On-Device Device Dataset + Row-Subset Gather - Pattern Map

**Mapped:** 2026-06-30
**Files analyzed:** 6 (4 new source modules, 1 modified mod.rs, 2 new test files)
**Analogs found:** 6 / 6 (every new file has a strong in-tree analog — this is a pure port-into-established-idiom phase)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/kernels/row_data.rs` (new) | model + kernel (§13 `CUDARowData` row/partition store, layout math + width-dispatched accessors) | transform (host integer layout) + batch (device read) | `crates/lgbm-compute/src/kernels/partition.rs` (generic-over-`Int` kernel + `launch_native!`) + `lib.rs` `upload_resident_bins`/`resident_bin_width` | role-match (new layout, established dispatch idiom) |
| `crates/lgbm-compute/src/kernels/column_data.rs` (new) | model (§3 `CUDAColumnData` column store + per-feature meta) | file-I/O (host→device upload, no consumer yet) | `crates/lgbm-compute/src/lib.rs` `BinColumn` (`:52`) + `upload_resident_bins` (`:2364`) | role-match |
| `crates/lgbm-compute/src/kernels/copy_subrow.rs` (new) | kernel + launcher (`CopySubrow` gather + bagging-draw wrapper) | transform (per-row gather) + event-driven (LCG draw) | `crates/lgbm-compute/src/kernels/partition.rs` (`:46`/`:371`) + `crates/lgbm-compute/src/kernels/random.rs` (`draw_next_float_on` `:240`) | exact (same kernel shape + reused launcher) |
| `crates/lgbm-compute/src/kernels/mod.rs` (modify) | config (module registration) | — | the existing `pub mod random; pub mod split;` block (`:23-27`) | exact |
| `crates/lgbm-compute/tests/device_dataset_parity.rs` (new) | test | request-response (host-vs-device asserts) | `crates/lgbm-compute/tests/cuda_random_parity.rs` | exact |
| `crates/lgbm-compute/tests/copy_subrow_parity.rs` (new) | test | request-response | `crates/lgbm-compute/tests/cuda_random_parity.rs` | exact |

---

## Pattern Assignments

### `crates/lgbm-compute/src/kernels/row_data.rs` (§13 row + feature-partition store)

Two distinct concerns: (a) **host integer layout math** (`DivideCUDAFeatureGroups`, offset tables, the dense/sparse re-lay) — pure `usize` arithmetic, zero parity surface beyond producing the right offsets; (b) **device width-dispatched reads** — copy the generic-over-`Int` kernel + host width-match dispatch.

**Analog A (width dispatch):** `crates/lgbm-compute/src/kernels/partition.rs`

**Generic-over-`Int` kernel signature + index read** (partition.rs:46-86) — this is the CubeCL-idiomatic replacement for the C++ `void* const* in_cuda_data_by_column` + `uint8_t* column_bit_type` table. The `u32::cast_from(bins[i])` widens any of u8/u16/u32 to a u32 index losslessly:
```rust
#[cube(launch)]
pub fn data_partition_kernel<B: Int>(
    bins: &Array<B>,
    route: &mut Array<u32>,
    // ... scalars ...
) {
    let i = ABSOLUTE_POS;
    if i < bins.len() {
        // u32::cast_from(x: u32) is the identity; <u8>/<u16> widen the index losslessly.
        let bin = u32::cast_from(bins[i]) as i32;
        // ...
    }
}
```

**Host width-match dispatch via `launch_native!` macro** (partition.rs:371-393) — this is the `bit_type()∈{8,16,32}` dispatch surface; for sparse, nest a second match on `row_ptr_bit_type()∈{16,32,64}` (the 3×3 `InitSparseData<BIN,PTR>`). Anything outside the 3×3 must return `ComputeError`, NOT fall through:
```rust
macro_rules! launch_native {
    ($w:ty, $slice:expr) => {{
        let h_bins = client.create_from_slice(<$w>::as_bytes($slice));
        unsafe {
            data_partition_kernel::launch::<$w, R>(
                client,
                CubeCount::Static(cube_count, 1, 1),
                CubeDim::new_1d(cube_dim),
                ArrayArg::from_raw_parts(h_bins, n),
                /* ... */
            );
        }
    }};
}
match bins {
    BinColumn::U8(v)  => launch_native!(u8,  v),
    BinColumn::U16(v) => launch_native!(u16, v),
    BinColumn::U32(v) => launch_native!(u32, v),
}
```

**Generic-over-`Int` precedent confirmed in histogram.rs:1090** (`construct_leaf_hist_resident_kernel<B: Int>`, `resident_bins: &Array<B>` "native bin width (u8/u16/u32)") — the same `<B: Int>` resident-read idiom; row_data.rs accessors follow it.

**Analog B (once-per-train resident upload + narrowest-width concat):** `crates/lgbm-compute/src/lib.rs` `upload_resident_bins` (:2364-2395) and `resident_bin_width`.

**CRITICAL layout difference (Pitfall 3):** the existing `resident_bins` buffer is **feature-major** (`f * num_data + row`, lib.rs:2360); §13 `CUDARowData` is **row-major partition-local** (`data[idx·ncol+tx]`). Do **NOT** reuse the existing buffer — build a NEW row-major partition-local buffer in `GetDenseDataPartitioned`. The narrowest-uniform-width concat + `client.create_from_slice(u8::as_bytes(&concat))` upload shape is reusable verbatim:
```rust
// lib.rs:2382-2395 — pick narrowest uniform width, concat, upload at native width
let width = resident_bin_width(feature_bins);
let handle = match width {
    ResidentBinWidth::U8 => {
        let mut concat: Vec<u8> = Vec::with_capacity(num_features * num_data);
        for &col in feature_bins { /* extend per-variant, lossless upcast */ }
        client.create_from_slice(u8::as_bytes(&concat))
    }
    // U16 / U32 arms ...
};
```

**Sparse re-lay correctness crux (Pitfall 4):** `GetSparseDataPartitioned` must subtract `partition_hist_start` (= `partition_hist_offsets[i]`) from each bin so bins are partition-local. Pure host integer arithmetic; assert in a 2-partition synthesized test that re-lay'd bin == `global_bin - partition_hist_offsets[partition]`.

**Partition packing formula (Pitfall 1):** `max_num_bin_per_partition = shared_hist_size / 2` (grad+hess pair per entry); `DP_SHARED_HIST_SIZE = 6144` → budget 3072 bins/partition; over-budget column → own large-bin partition (`NumLargeBinPartition() += 1`).

---

### `crates/lgbm-compute/src/kernels/column_data.rs` (§3 column store)

**Analog:** `crates/lgbm-compute/src/lib.rs` `BinColumn` (:52-188) + `upload_resident_bins` (:2364).

The per-column buffers ARE `BinColumn`s already (narrowest u8/u16/u32 storage, `bin(row)` :105, `to_u32_vec` :145, `gather(rows)` :134). Column-major store = one upload per column (or a column-major concat) using the same `create_from_slice(<$w>::as_bytes(..))` native-width upload as the `launch_native!` macro above. Per-feature meta (`bit_type`, `feature_{min,max}_bin`, `offset`, `most_freq_bin`, `default_bin`, missing/mfb flags, `feature_to_column`) is a plain host struct.

**BinColumn variant set (the width axis to dispatch on):**
```rust
// lib.rs:52-59
pub enum BinColumn {
    U8(Vec<u8>),    // num_bin <= 256 (default max_bin=255 common case)
    U16(Vec<u16>),  // 256 < num_bin <= 65536
    U32(Vec<u32>),  // num_bin > 65536
}
```

No consumer until Phase 18 — build + parity-test the binned values + numeric meta; leave a documented TODO for categorical-bitset meta (Phase 22). Do NOT wire it.

---

### `crates/lgbm-compute/src/kernels/copy_subrow.rs` (`CopySubrow` gather + bagging draw)

**Analog A (the gather kernel):** `crates/lgbm-compute/src/kernels/partition.rs` (:46 kernel shape, :371 host dispatch). The kernel is the same generic-over-`Int`, one-unit-per-`ABSOLUTE_POS`, bounds-guarded form. D-07: same width in/out, no widen:
```rust
#[cube(launch)]
pub fn copy_subrow_kernel<B: Int>(
    in_col: &Array<B>, out_col: &mut Array<B>,
    used_indices: &Array<i32>, num_used: u32,
) {
    let local = ABSOLUTE_POS;
    if local < num_used {
        let src = used_indices[local] as u32; // host-validated in [0, num_data)
        out_col[local] = in_col[src];
    }
}
```
`COPY_SUBROW_BLOCK_SIZE = 1024`, `cube_count = num_used.div_ceil(1024)`. Drive once per column, dispatching the monomorph on the column width (the `launch_native!` match). The host oracle is `BinColumn::gather(used_indices)` (lib.rs:134) — assert the device subset == host gather per width.

**V5 boundary validation (copy this exactly):** partition.rs:222-241 validates every bin index `< num_bin` before the `unsafe` launch and returns `ComputeError::BinIndexOutOfRange`. For `CopySubrow`, validate every `used_indices[i] ∈ [0, num_data)` before launch:
```rust
// partition.rs:233-241 — the per-index boundary check to mirror for used_indices
for (row, &b) in bins.iter().enumerate() {
    if b >= num_bin {
        return Err(ComputeError::BinIndexOutOfRange { row, bin: b, num_bin });
    }
}
```

**Analog B (the bagging draw):** `crates/lgbm-compute/src/kernels/random.rs` `draw_next_float_on` (:240-268). Reuse VERBATIM — one task per `BAGGING_RAND_BLOCK` block, `k = 1024`, output row-major so row `i`'s draw is `out[i]`:
```rust
// random.rs:240 signature — pass n_blocks seeds, k = BAGGING_RAND_BLOCK
pub fn draw_next_float_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    seeds: &[u32],
    k: u32,
) -> Result<Vec<f32>, ComputeError> { /* ... */ }
```
`cuda_next_float` (random.rs:77) is the f32-exact draw (divisor exactly `32768.0f32`). Seeds = `(bagging_seed + block) as u32` (iteration-0 anchor). **Overflow guard:** `validate_draw_inputs` (random.rs:146) `checked_mul(n, k)` — already inside the launcher.

**Route comparison (Pitfall 6):** promote the f32 draw to f64 before `< bagging_fraction`, exactly as host (sample_strategy.rs:257 `let draw = rands[block].next_float() as f64`). For this phase, compute the route host-side from the readback draw stream and assert vs host `bag_data_indices` (Open Question 2 recommendation).

---

### `crates/lgbm-compute/src/kernels/mod.rs` (modify — register modules)

**Analog:** the existing ungated module block (mod.rs:23-27). Add the 3 new modules ungated (NOT `#[cfg(feature="gpu")]` — they must run on the cpu f64 anchor), mirroring `random`/`split`:
```rust
// mod.rs:23-27 — append row_data, column_data, copy_subrow in the same ungated block
pub mod primitives;
pub mod random;
pub mod split;
pub mod split_info;
pub mod subtract;
```

---

### `crates/lgbm-compute/tests/device_dataset_parity.rs` & `copy_subrow_parity.rs` (new tests)

**Analog:** `crates/lgbm-compute/tests/cuda_random_parity.rs` — the host-vs-device anchor shape. Key conventions to mirror:
- `use lgbm_compute::runtime::cpu_client;` — run on the cpu f64 anchor (D-08, never GPU-vs-GPU).
- Exact-bit equality: `assert_eq!` on integers; `f32::to_bits()` for floats (no tolerance — bins/draws are exact).
- Per-task row-major indexing: `device[t * K + j]` vs `host` recurrence (the bagging draw spans ≥2 blocks → >1024 rows).
- Host oracle for binned values = the v1.0 Rust binning (`BinColumn`); for bagging = `BaggingSampleStrategy::bag_data_indices()` (sample_strategy.rs:376).

```rust
// cuda_random_parity.rs:41-60 — the host-vs-device assert shape to copy
let client = cpu_client();
let device = draw_next_float_on(&client, &states, K as u32).unwrap();
for (t, &seed) in seeds.iter().enumerate() {
    let mut host = Random::new(seed);
    for j in 0..K {
        let expected = host.next_short(0, 32768);
        assert_eq!(device[t * K + j], expected, "mismatch seed={seed} draw={j}");
    }
}
```

**D-04 sparse synthesizer (test helper):** generate columns whose nnz crosses 2^16 and 2^32 to force each `row_ptr_type{16,32,64}`, plus a column over the shared-hist budget (large-bin spill). All 9 cells of the 3×3 must be reachable; anchor each to host binned values.

---

## Shared Patterns

### Width dispatch (the `void*`-table replacement) — applies to row_data, column_data, copy_subrow
**Source:** `crates/lgbm-compute/src/kernels/partition.rs:46` (kernel) + `:371` (host match) + `histogram.rs:1090` (resident `<B:Int>` precedent)
Generic-over-`cubecl::Int` `#[cube]` kernel; host monomorphizes by `match BinColumn` (and a second `match` on `row_ptr_bit_type` for sparse CSR). `u32::cast_from(arr[i])` widens the index losslessly. Native-width upload (`<$w>::as_bytes`) = 4× fewer bytes for u8. NEVER an in-kernel byte-width branch (spike-004).

### Once-per-train resident upload + alloc-once lifecycle (D-09) — applies to row_data, column_data
**Source:** `crates/lgbm-compute/src/lib.rs:2364` (`upload_resident_bins`, `RefCell<Option<…>>` cache) + `crates/lgbm-compute/src/kernels/split_info.rs:267,292` (the single `client.empty` site in `new`)
Upload the resident dataset ONCE per train (cache the `Handle` in interior mutability), allocate every device buffer once in a `new`-style constructor — zero per-tree / per-call / per-row alloc. `split_info.rs` counts its allocations so "allocated exactly once" is structurally verifiable:
```rust
// split_info.rs:289-293 — the ONLY client.empty caller, runs only in new()
let mut device_allocations = 0usize;
let mut alloc = |elem_size: usize, len: usize| -> Handle {
    device_allocations += 1;
    client.empty(len * elem_size)
};
```

### V5 boundary validation before unsafe launch — applies to copy_subrow, row_data, column_data
**Source:** `crates/lgbm-compute/src/kernels/partition.rs:222-241` + `random.rs:146` (`validate_draw_inputs` `checked_mul`)
Validate every index/length BEFORE the `unsafe { create_from_slice / launch }`, returning `ComputeError` (the C++ raw-pointer table has no bounds check; the Rust port adds one). `used_indices[i] ∈ [0, num_data)`; `checked_mul` on every `len * width` / `n_blocks * BLOCK` sizing; `row_ptr_type` wide enough for actual max nnz. Keep all `cubecl` unsafe confined to the launcher with a SAFETY comment (CMP-01).

### Host bagging anchor (D-05) — applies to copy_subrow bagging draw + tests
**Source:** `crates/lgbm-boosting/src/sample_strategy.rs` — `BAGGING_RAND_BLOCK = 1024` (:46), per-block seeding `Random::new(bagging_seed + i)` constructed once (:160-162), `bagging()` draw order (:239-282), `bag_data_indices()` (:376)
The parity surface is the per-block RNG stream + the route layout `[in-bag asc] ++ [OOB desc]` (OOB tail reversed, :278). The device draw must reproduce the BLOCK STRUCTURE bit-for-bit, not just the final set:
```rust
// sample_strategy.rs:254-257 — block index + f32→f64 promoted route compare
let block = (i / BAGGING_RAND_BLOCK) as usize;
let draw = rands[block].next_float() as f64;   // f32 draw promoted to f64
if draw < threshold { /* in-bag, fill left asc */ } else { /* OOB, fill right desc */ }
```
Pitfall 5: for this phase's single-draw anchor, seed `bagging_seed + block` (iteration-0). Multi-iteration continuity needs host block-state supply — keep the seed-supply seam explicit (Phase-21 driver scope).

### Anchor to cpu f64 fold, never GPU-vs-GPU (D-08, def-f8u-01) — applies to ALL tests
**Source:** `crates/lgbm-compute/tests/cuda_random_parity.rs` (header + `cpu_client()` + `to_bits()` asserts)
Every parity assert runs on `cpu_client()` against host truth (host Rust binning / host `Random` / host `bag_data_indices`). Exact-bit (`assert_eq!` / `f32::to_bits()`), no tolerance for index/draw values; ~1e-5 envelope only for f32 leaf/score numerics (not relevant to this dataset/gather phase).

---

## No Analog Found

None. Every new file maps to a strong in-tree analog — this phase is ~90% wiring proven primitives (generic-over-`Int` dispatch, `draw_next_float_on`, `BinColumn`, the resident-upload + alloc-once lifecycle, the `cuda_random_parity` test shape) into two new layouts + one gather. The only genuinely new code is the host-side `DivideCUDAFeatureGroups` integer layout/offset tables and the sparse partition-local CSR re-lay — both plain `usize` math with zero parity surface beyond producing correct offsets, unit-tested against hand-computed expectations.

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/kernels/` (partition, random, split_info, histogram), `crates/lgbm-compute/src/lib.rs` (BinColumn, resident upload), `crates/lgbm-boosting/src/sample_strategy.rs`, `crates/lgbm-compute/tests/cuda_random_parity.rs`
**Files scanned:** 7 (all line-targeted from RESEARCH.md references; no blind globbing needed — research pre-identified every analog)
**Pattern extraction date:** 2026-06-30
