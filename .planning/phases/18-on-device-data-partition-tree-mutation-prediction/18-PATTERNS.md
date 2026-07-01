# Phase 18: On-Device Data Partition, Tree Mutation & Prediction - Pattern Map

**Mapped:** 2026-07-01
**Files analyzed:** 11 (3 new kernels, 2 extended kernels, 1 reused kernel, 2 new tests, 1 extended test, 1 extended C++ capture, fixtures)
**Analogs found:** 11 / 11 (all exact or strong role + data-flow matches — this is a port that wires proven in-tree primitives)

All analogs are IN-TREE. Every excerpt below is line-referenced against the existing
`lgbm-compute` crate / `oracle-harness` tests / `xtask/cpp`. The C++ port-source
(`LightGBM-release-4.6.0.99/`) line refs live in `18-RESEARCH.md` (Kernel Inventory Map);
this document maps each NEW/MODIFIED Rust file to the existing Rust file it copies from.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/kernels/data_partition.rs` (NEW) | kernel/module | transform (mark → prefix-sum → scatter row permutation) | `kernels/partition.rs` (route decision + native-width dispatch + host V5) **composed with** `kernels/primitives.rs` (block scan bodies) | composite (exact role) |
| `crates/lgbm-compute/src/kernels/tree.rs` (NEW) | kernel/module | transform (scalar SoA field writes: Split / Shrinkage / AddBias) | `kernels/split.rs` (`#[cube(launch)]` + branchless `select`) **and** `kernels/split_info.rs` (`SplitScalars` field list, counted-alloc) | role-match + composite |
| `crates/lgbm-compute/src/kernels/predict.rs` (NEW) | kernel/module | transform (tree-walk gather → `double` score add) | `kernels/partition.rs` `data_partition_kernel<B: Int>` (per-row route via `u32::cast_from`) **and** `kernels/column_data.rs`/`row_data.rs` (8/16/32 dispatch) | role-match + composite |
| `crates/lgbm-compute/src/kernels/primitives.rs` (EXTEND) | kernel/module | transform (integer block prefix-sum) | itself — add `u16`/`u32` launch wrappers of the existing `N: Numeric` bodies | extend-in-place |
| `crates/lgbm-compute/src/kernels/histogram_arena.rs` (EXTEND) | utility (host bookkeeping) | event-driven (per-split handle rotation) | itself — add a leaf-indexed whole-pool `swap()` alongside `rotate()` | extend-in-place |
| `crates/lgbm-compute/src/kernels/split_info.rs` (REUSE) | model (SoA record) | CRUD (slot read/write) | itself — `SplitScalars` = `CUDASplitInfo`, consumed unchanged | reuse |
| `crates/oracle-harness/tests/partition_parity.rs` (NEW) | test (golden harness) | request-response (parse fixture → drive cpu fold → assert bit-exact) | `tests/best_split_parity.rs` | exact |
| `crates/oracle-harness/tests/tree_mutation_parity.rs` (NEW) | test (golden harness) | request-response | `tests/best_split_parity.rs` | exact |
| `crates/oracle-harness/tests/predict_parity.rs` (EXTEND) | test (golden harness) | request-response (model + matrix → predict → compare_within) | itself — add `on_device` + `cat` cells | extend-in-place |
| `xtask/cpp/kernel_capture.cpp` (EXTEND) | config/tooling (golden capture) | batch (build corpus → emit text goldens) | itself — `PCaseSpec`/`EmitPCase`/`BuildPartitionCorpus` | extend-in-place |
| `tests/fixtures/kernels/partition.txt` (+ new predict fixture) (NEW/EXTEND) | test fixture (golden data) | file-I/O (`PCASE/PBINS/PORDER/PSPLIT` text) | itself | extend-in-place |

## Shared Patterns (apply to ALL new kernels)

### SP-1 — One `#[cube]` generic body + thin per-type launch wrappers
**Source:** `crates/lgbm-compute/src/kernels/primitives.rs:82-112` (body) + `:159-181` (wrappers)
**Apply to:** `data_partition.rs`, `tree.rs`, `predict.rs`
The canonical shape: the numerical core lives in ONE `#[cube] fn foo_body<N: Numeric>(...)`;
thin `#[cube(launch_unchecked)] fn foo_kernel_f64(...)` / `_f32(...)` wrappers delegate. D-12
("one `#[cube]` generic, comptime/runtime-split reduction order") is exactly this convention.
```rust
// primitives.rs:82-91 — the shared body owns the math + the inclusive/exclusive branch
#[cube]
fn block_scan_body<N: Numeric>(
    data: &Array<N>, out: &mut Array<N>, block_totals: &mut Array<N>,
    block_size: u32, n: u32, inclusive: u32,
) { if UNIT_POS == 0 { /* ascending serial fold, single owner */ } }

// primitives.rs:159-169 — thin wrapper, byte-identical except cell type
#[cube(launch_unchecked)]
fn block_scan_kernel_f64(data: &Array<f64>, out: &mut Array<f64>, /*...*/) {
    block_scan_body::<f64>(data, out, block_totals, block_size, n, inclusive);
}
```

### SP-2 — Branchless `select(...)` for every conditional store (cubecl-cpu MLIR constraint)
**Source:** `crates/lgbm-compute/src/kernels/partition.rs:88-92`, `split.rs:246-292`
**Apply to:** the route decision in `data_partition.rs` + `predict.rs`; the SoA field writes in `tree.rs`
LOAD-BEARING for cubecl-cpu lowering: never nested-`if` mutation chains inside a `#[cube]` body.
```rust
// partition.rs:88-92 — the SplitInner route as pure select()
let is_default = bin < min_bin || bin > max_bin;
let gt = bin > th;
let go_right = select(is_default, default_to_right, gt);
route[i] = select(go_right, 1u32, 0u32);
```

### SP-3 — Pre-allocate-once, counted `client.empty`; NO per-split device alloc (D-15)
**Source:** `crates/lgbm-compute/src/kernels/split_info.rs:289-321` and `histogram_arena.rs:120-135`
**Apply to:** the scatter scratch (`cuda_out_data_indices_in_leaf_`), per-block left/right count
buffers, and the 16-int packet in `data_partition.rs`
The idiom: a single `alloc` closure is the ONLY caller of `client.empty` in the module, runs
only in `new`, and increments a `device_allocations` counter asserted equal to the field count.
```rust
// split_info.rs:289-293
let mut device_allocations = 0usize;
let mut alloc = |elem_size: usize, len: usize| -> Handle {
    device_allocations += 1;
    client.empty(len * elem_size)
};
```

### SP-4 — V5 boundary validation + confined `unsafe` launch (CMP-01)
**Source:** `crates/lgbm-compute/src/kernels/partition.rs:222-282`, `primitives.rs:216-234`
**Apply to:** every host driver in `data_partition.rs`, `tree.rs`, `predict.rs`
Validate `num_bin>0`, `threshold<num_bin`, every `bins[i]<num_bin` BEFORE the launch; wrap the
`launch_unchecked` in a single `unsafe` block with a SAFETY comment proving every index `< len`.
`validate_scan_inputs` (`primitives.rs:216`) is the reusable guard for the u16/u32 scans (rejects
`block_size==0`, `num_blocks>1024`).

### SP-5 — `double` (f64) score accumulator + scalar gain/output ONLY; NO f64 per-row hot loop (D-14)
**Source:** design constraint (spike-052); mirror `split_info.rs` f64 scalar fields (`:87,107,121`)
**Apply to:** `predict.rs` (`score[data_index] += leaf_value[~node]` in f64) + `tree.rs` scalar math
Per-row bin reads stay integer/native-width (`u32::cast_from`, `partition.rs:86`); f64 appears only
in the accumulator and the scalar leaf-value/gain math.

## Pattern Assignments

### `crates/lgbm-compute/src/kernels/data_partition.rs` (NEW — kernel/module, transform)

**Primary analog:** `crates/lgbm-compute/src/kernels/partition.rs` (route decision + host driver
shape) — **REFERENCE ONLY for the routing decision** (D-01); the scatter body is NEW and composes
`primitives.rs` scans. Do NOT extend `partition.rs` (D-01: keep the host-gather as decision reference).

**Route-decision to lift (the shared §specifics decision, reuse in BOTH mark + predict):**
`partition.rs:44-94` — `data_partition_kernel<B: Int>` is the `MissingType::None` numeric route.
Phase 18 keeps this generic-over-`Int` per-row shape and ADDS the D-02 comptime flag fan-out
(`MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, USE_MIN_BIN, BIN_TYPE`)
as comptime params. The existing `th = threshold + min_bin` (`--th if most_freq_bin==0`) and the
`default_to_right = most_freq_bin > threshold` core (`:75-92`) are the no-missing base case.

**Native-width dispatch macro to copy (partition.rs:371-393):**
```rust
// partition.rs:371-393 — the u8/u16/u32 monomorph dispatch; copy verbatim for the
// GenDataToLeftBitVector<BIN_TYPE> 8/16/32 fan-out
macro_rules! launch_native {
    ($w:ty, $slice:expr) => {{
        let h_bins = client.create_from_slice(<$w>::as_bytes($slice));
        unsafe { data_partition_kernel::launch::<$w, R>(client, /* ... */); }
    }};
}
match bins {
    BinColumn::U8(v) => launch_native!(u8, v),
    BinColumn::U16(v) => launch_native!(u16, v),
    BinColumn::U32(v) => launch_native!(u32, v),
}
```

**Scatter + block-offset from primitives (the NEW mark→prefix-sum→scatter core):**
Build on `primitives.rs:82-112` (`block_scan_body` = `PrepareOffset` block `ShufflePrefixSum`) and
the 3-launch host driver `primitives.rs:268-326` (`block scan → scan_block_totals → add_base` =
`AggregateBlockOffsetKernel{0,1}`). The `SplitInnerKernel` scatter is a NEW `#[cube]` reading the two
offset buffers (exclusive left rank `= block_to_left_offset[tid-1]`, right `= tid - rank`). See the
inclusive-vs-exclusive pitfall: `PrepareOffset` uses inclusive + `[idx-1]`; `AggregateBlockOffset`
uses the exclusive block-totals scan (`primitives.rs:117-130` `scan_block_totals_body`).

**cpu f64 anchor (D-04 CONFIRMED plain stable partition) — reuse the `gather_route` shape:**
`partition.rs:290-304` `gather_route` IS the exact stable-partition anchor (left rows route==0 in
original order, then right rows route==1). This is order-equivalent to the §9 block-tiled scatter
(RESEARCH D-04 gate). The anchor is this two-pass gather; the hip kernel runs the real 4-stage scatter.

---

### `crates/lgbm-compute/src/kernels/tree.rs` (NEW — kernel/module, transform)

**Primary analog:** `crates/lgbm-compute/src/kernels/split.rs` (the `#[cube(launch)]` scalar-write
kernel shape + `round_int`/`select` convention) **and** `crates/lgbm-compute/src/kernels/split_info.rs`
(`SplitScalars` — the `CUDASplitInfo` field list `SplitKernel` reads).

**`SplitScalars` field list `SplitKernel` consumes (split_info.rs:80-126) — read, do not redefine:**
```rust
// split_info.rs:80-126 — the CUDASplitInfo analog: default_left, per-side
// sum_gradients/sum_hessians (f64), left/right_value (f64), threshold (u32),
// inner_feature_index, num_cat_threshold + the reserved cat slabs (:230-233).
pub struct SplitScalars { pub is_valid: bool, pub leaf_index: i32, pub gain: f64,
    pub inner_feature_index: i32, pub threshold: u32, pub default_left: bool,
    pub left_sum_gradients: f64, /* ...12 more... */ pub num_cat_threshold: i32 }
```
`SplitKernel` (`<<<3,5>>>`, 14 numeric field writes, NaN→0) mutates the device flat `CUDATree`
BEFORE partition and returns `right_leaf_index` (D-07 ordering — §1/§10 hard invariant). `ShrinkageKernel`
(`leaf_value *= rate`) / `AddBiasKernel` (`leaf_value += val`) are elementwise `#[cube]` bodies —
copy the SP-1 body+wrapper shape; scalar math stays f64 (SP-5, D-14).

**Launch-wrapper shape to copy (split.rs:393):** `#[cube(launch)] pub fn find_best_split_kernel(...)`
is the exact thin-wrapper precedent; the branchless `select(...)` field-write style at `split.rs:246-292`
is the cubecl-cpu-safe way to do the NaN→0 conditional writes.

**Device `CUDATree` flat arrays (D-07):** model these as counted-`client.empty` SoA handles exactly
like `DeviceSplitInfo::new` (`split_info.rs:267-357`) — one buffer per field
(`cuda_leaf_value_`, `cuda_left_child_`, `cuda_decision_type_`, thresholds/counts/depth, `cat_boundaries`/bitsets),
allocated once. Host `lgbm_model::Tree` reconstructed per-tree for the anchor compare (D-07, discretion).

---

### `crates/lgbm-compute/src/kernels/predict.rs` (NEW — kernel/module, transform)

**Primary analog:** `crates/lgbm-compute/src/kernels/partition.rs` (per-row native-width read via
`u32::cast_from`) **and** `column_data.rs`/`row_data.rs` (the §13 8/16/32 columnar store to read).

**8/16/32 width dispatch to read (column_data.rs:31-52):** `ColumnFeatureMeta` already carries
`bit_type ∈ {8,16,32}`, `feature_min_bin`, `feature_max_bin`, `offset`, `most_freq_bin`, `default_bin`,
`missing_is_zero`, `missing_is_na`, `feature_to_column` — the EXACT fields the tree-walk remap needs
(`bin ∈ [min,max] ? bin-min+offset : most_freq_bin`). Read this struct; do not re-derive meta.

**Shared route decision (D-02/D-05 — one transcription, §specifics Pitfall 4):** the numeric
missing/default branch is IDENTICAL between predict and the partition mark. Write it ONCE as a
`#[cube]` inline fn taking the comptime flags and call from both `data_partition.rs` and `predict.rs`.
The base numeric form is `partition.rs:75-92`; the predict extension adds the `node<0 → score += leaf_value[~node]`
(f64 accumulator, SP-5) and the categorical `FindInBitsetCUDA` membership branch.

**`FindInBitsetCUDA<T>` helper:** a tiny NEW shared `#[cube]` helper (`bits[pos/32] >> (pos%32) & 1`,
guard `pos >= n → false`) used by BOTH the cat partition route and the cat predict route (Don't-Hand-Roll:
port once). No in-tree analog for the bitset op itself; the `<T>`-generic + `select` shape mirrors SP-1/SP-2.

**§9 `AddPredictionToScoreKernel<USE_BAGGING>` (D-06):** the per-row leaf-map gather-add belongs here
too — a `#[cube]` gather-add over the data-index→leaf map into the f64 score (SP-5). `USE_INDICES`/`USE_BAGGING`
are comptime bools (indices vs identity), same fan-out convention as SP-1.

---

### `crates/lgbm-compute/src/kernels/primitives.rs` (EXTEND — kernel/module)

**Analog:** itself. The bodies are ALREADY `N: Numeric` generic (`block_scan_body:82`,
`scan_block_totals_body:117`, `add_base_body:135`). The GAP (RESEARCH #4 + A1/Open-Q1): only f64/f32
launch wrappers exist (`:159-211`) and only f64/f32 host drivers (`:245-326`).

**What to add (instantiation, NOT a rewrite):** `u16` launch wrappers (`PrepareOffset` per-block
`ShufflePrefixSum<uint16_t>`, inclusive) and `u32` wrappers (`AggregateBlockOffset` block-totals,
exclusive), plus their host drivers modeled on `prefix_sum_f64_on` (`:268-326`). Copy the wrapper
pair verbatim, swapping the cell type:
```rust
// primitives.rs:183-191 — copy this pair for u16 and u32
#[cube(launch_unchecked)]
fn scan_block_totals_kernel_f64(block_totals: &mut Array<f64>, num_blocks: u32) {
    scan_block_totals_body::<f64>(block_totals, num_blocks);
}
```
Reuse `validate_scan_inputs` (`:216-234`) unchanged. **Wave-0 de-risk (A1):** a 1-block u16 + u32
scan parity test on hip before committing the scatter to the generic body; fall back to a u32-widened
scan if u16 doesn't lower on hip 0.10 (parity-neutral).

---

### `crates/lgbm-compute/src/kernels/histogram_arena.rs` (EXTEND — utility, event-driven)

**Analog:** itself. The module docs ALREADY state the whole-tree pool SWAP is explicitly Phase 18
(`:29-30`). `rotate()` (`:236-267`) is the per-split `{parent, smaller, larger}` index reassignment
(larger inherits parent in-place, smaller takes a fresh non-aliasing slot) — do NOT rebuild it (D-09).

**What to add:** a leaf-index → slot-handle table and a `swap(left_leaf, right_leaf, smaller_is_left)`
that reassigns handles exactly as `SplitTreeStructureKernel` does (RESEARCH HistArena section: left-smaller
→ `pool[right]=parent_ptr`, `pool[left]=fresh`; mirror for right-smaller). The counted-alloc discipline
(`:120-135`) and the no-alias assert style (`:257-264`) carry forward verbatim — the swap reassigns
INDICES only, zero `client.empty`, `device_allocations` frozen. Add a `histogram_arena::swap` unit test
in the SAME `#[cfg(test)] mod tests` block (the `rotate_subtract_lands_in_larger_parent_slot` test at
`:402-456` is the round-trip precedent: rotate/swap → drive `subtract_hist_kernel` → assert result lands
in the correct slot on the cpu f64 anchor, never GPU-vs-GPU).

---

### `crates/lgbm-compute/src/kernels/split_info.rs` (REUSE — no change)

`SplitScalars` (`:80-126`) + `DeviceSplitInfo` (`:241-571`, `NUM_FIELD_BUFFERS=21`, prealloc D-15) are
the `CUDASplitInfo` record consumed unchanged by `SplitKernel` (`tree.rs`) and `SplitTreeStructureKernel`
(`data_partition.rs`). The reserved cat slabs (`:230-233`, `MAX_CAT_PER_SPLIT=32`) already exist for
the cat membership path. `copy_slot` (`:522-555`) is the deep-copy `operator=` analog. Read-only reuse.

---

### `crates/oracle-harness/tests/partition_parity.rs` (NEW — test, request-response)

**Analog (EXACT):** `crates/oracle-harness/tests/best_split_parity.rs` — the canonical Phase-17
golden-replay harness (RESEARCH Fixture Strategy).

**Idioms to copy verbatim:**
```rust
// best_split_parity.rs:49-56 — CARGO_MANIFEST_DIR fixture path (NEVER untracked LightGBM/),
// raw-f64-bits parse (zero rounding)
fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kernels")
}
fn parse_f64_bits(s: &str) -> f64 { f64::from_bits(s.parse::<u64>().expect("f64-bits u64 field")) }
```
Plus the `field`/`parse_i64`/`parse_u32`/`parse_f64_field` token helpers (`:59-81`) and the
`#[ignore = "Wave-0 scaffold; un-ignore when <plan> lands"]` gate on the primary assertion (`:17-19`)
so the merge gate stays green until the numeric core lands (D-13/ODL-19). Assert BIT-EXACT via
`oracle_harness::comparator::compare_exact_f64_bits` (`:47`). Cover: row-order (`PORDER`), cat membership,
and the 16-int packet fields. Drive the cpu fold; SKIP gracefully when the fixture is absent.

---

### `crates/oracle-harness/tests/tree_mutation_parity.rs` (NEW — test, request-response)

**Analog (EXACT):** `best_split_parity.rs` (same idioms as above). Covers ODL-14: `SplitKernel` field
writes (reuse the `split.txt` cases per RESEARCH test map) + the Split-before-partition ordering assert
(the returned `right_leaf_index` must feed the partition launch — Pitfall 3). Shrinkage/AddBias get a
`#[cfg(test)]` unit in `lgbm-compute` (`tree::shrinkage`).

---

### `crates/oracle-harness/tests/predict_parity.rs` (EXTEND — test, request-response)

**Analog:** itself. Add `on_device` (numeric 8/16/32) + `cat` (cat_onehot/cat_manyvsmany) cells.

**Idioms already in the file to reuse:**
```rust
// predict_parity.rs:38-51 — graceful SKIP when a golden is absent
fn read_golden(corpus: &str, file: &str) -> Option<String> {
    match std::fs::read_to_string(&path) { Ok(s) => Some(s), Err(_) => { eprintln!("SKIP ..."); None } }
}
// predict_parity.rs:90-97 — load model + matrix; :122 compare_within(.., ORACLE_TOL)
```
`load_corpus` (`:90`) + `parse_matrix` (`:55`) + `compare_within(&rust_f32, &golden_f32, ORACLE_TOL)`
(`:122`) are the exact precedent. The categorical fixture models already exist
(`tests/fixtures/categorical/cat_onehot.*`, `cat_manyvsmany.*`). Objective inverse-link stays HOST-side
at the readback boundary this phase (Phase-19 moves it on-device).

---

### `xtask/cpp/kernel_capture.cpp` (EXTEND — config/tooling, batch)

**Analog:** itself. The Phase-4 partition emitter is the template to extend (D-11).

**Existing structures to extend (kernel_capture.cpp:957-1007):**
```cpp
// :957 — PCaseSpec: bins + min/max/threshold/most_freq_bin
struct PCaseSpec { std::string name; std::vector<uint32_t> bins;
    int num_bin, min_bin, max_bin, threshold, most_freq_bin; std::string note; };
// :968 EmitPCase → PCASE/PBINS/PORDER/PSPLIT ; :1009 BuildPartitionCorpus ; :1161-1177 file write
```
Extend `PCaseSpec`/`EmitPCase`/`BuildPartitionCorpus` for: (a) the full flag fan-out (missing/NA/
default_left/MFB cases), (b) categorical membership routing, (c) the 16-int child-stats packet, and
(d) tree-walk predict over numeric AND categorical models. The `argv` driver layout is at `:1102-1111`
(`<hist_out> <master_seed> <split_out> <partition_out> <subtract_out>`) — add new output paths there.
**A4/Open-Q2:** prefer a HOST-reconstructed 16-int packet golden (compute the same fields from the C++
`Tree`/partition on CPU) over an instrumented CUDA build; the cpu f64 fold is the authoritative anchor,
the C++ golden is a cross-check.

---

### `tests/fixtures/kernels/partition.txt` (EXTEND) + new predict fixture (NEW — file-I/O)

**Analog:** itself. Format `KERNEL_MASTER_SEED / COUNTS partition=<n> / PCASE.../PBINS/PORDER/PSPLIT`
(`partition.txt:1-21`). Extend with the flag-fan-out + cat + packet cases; regenerate via the extended
`kernel_capture.cpp`. Add a predict fixture in the `best_split.txt` bit-hex convention (raw LE f64 as
decimal u64 — zero parse rounding).

## Shared Patterns Summary Table

| Pattern | Source (file:lines) | Applies To |
|---------|---------------------|------------|
| SP-1 `#[cube]` body + thin f64/f32 wrappers | `primitives.rs:82-112, 159-181` | data_partition, tree, predict, primitives |
| SP-2 branchless `select()` stores | `partition.rs:88-92`, `split.rs:246-292` | data_partition, predict, tree |
| SP-3 counted `client.empty` prealloc-once (D-15) | `split_info.rs:289-321`, `histogram_arena.rs:120-135` | data_partition (scratch/packet), tree (CUDATree) |
| SP-4 V5 validate + confined `unsafe` (CMP-01) | `partition.rs:222-282`, `primitives.rs:216-234` | all host drivers |
| SP-5 f64 accumulator only, no f64 per-row loop (D-14) | `split_info.rs:87,107,121` (f64 scalars) | predict, tree |
| Golden-harness idioms (CARGO_MANIFEST_DIR, f64-bits, SKIP, `#[ignore]` Wave-0) | `best_split_parity.rs:49-81, 17-19`; `predict_parity.rs:38-51` | partition_parity, tree_mutation_parity, predict_parity |
| Env gate (OnceLock-cached, OFF-by-default, D-13) | `lib.rs:1311-1315` `cuda_on_device_enabled()` | the call-site gate; `on_device_growth_supported` stays `false` (`:1239`) |

## No Analog Found

No files lack an analog. Two SUB-COMPONENTS are genuinely new algorithm code (no in-tree precedent),
but each still copies a shape pattern from the table above:

| Sub-component | Home file | Reason no exact analog | Shape it copies |
|---------------|-----------|------------------------|-----------------|
| `SplitInnerKernel` scatter (in-block rank → contiguous child ranges) | `data_partition.rs` | No existing device scatter; host-gather `partition.rs` is decision-only | SP-1 body/wrapper + `primitives.rs` scan offset buffers |
| `FindInBitsetCUDA<T>` bitset membership | `predict.rs` (shared w/ partition cat) | No bitset op in the crate | SP-1 `<T>`-generic + SP-2 `select` |
| leaf-indexed whole-pool `swap()` | `histogram_arena.rs` | `rotate()` is one-triple only; whole-pool swap deferred to P18 | extends `rotate()` (`:236-267`) + counted-alloc discipline |

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/kernels/` (partition, split, split_info, primitives,
histogram_arena, column_data, row_data), `crates/lgbm-compute/src/lib.rs` (backend seam),
`crates/oracle-harness/tests/` (best_split_parity, predict_parity), `xtask/cpp/kernel_capture.cpp`,
`tests/fixtures/kernels/partition.txt`.
**Files scanned:** 11 analogs read line-by-line.
**Pattern extraction date:** 2026-07-01
**Port-source line refs (C++):** see `18-RESEARCH.md` Kernel Inventory Map (`LightGBM-release-4.6.0.99/`).
