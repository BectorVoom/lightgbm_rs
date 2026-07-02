# Phase 17: On-Device Best-Split Finder - Pattern Map

**Mapped:** 2026-07-01
**Files analyzed:** 4 (3 new, 1 extended)
**Analogs found:** 4 / 4 (all exact or strong role+data-flow matches)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/kernels/best_split.rs` (NEW) | kernel/module | transform (histogram → per-task record → argmax → 8-int export) | `crates/lgbm-compute/src/kernels/split.rs` (`#[cube]` body + f64/f32 launch wrappers) + `primitives.rs` (LDS scan/reduce) + `split_info.rs` (SoA record) | exact (role) + composite |
| `crates/oracle-harness/tests/best_split_parity.rs` (NEW) | test (golden harness) | request-response (parse fixture → drive cpu fold → assert) | `crates/oracle-harness/tests/kernel_parity.rs` (`parse_split` + `kernel_parity_split_bit_exact_on_cpu`) | exact |
| `crates/oracle-harness/tests/fixtures/kernels/best_split.txt` (NEW) | test fixture (golden data) | file-I/O (bit-hex golden records) | `crates/oracle-harness/tests/fixtures/kernels/split.txt` | exact |
| `crates/lgbm-compute/tests/rocm_backend_parity.rs` (EXTEND) | test (hip parity) | request-response (cpu anchor vs hip f32) | itself — `rocm_backend_find_best_split_bit_exact` | extend-in-place |

## Pattern Assignments

### `crates/lgbm-compute/src/kernels/best_split.rs` (kernel/module, transform)

This is the largest file and composes patterns from FOUR analogs. Extract each below.

---

#### Pattern A — One `#[cube]` generic body + thin f64/f32 launch wrappers (D-01/D-06/D-08)

**Analog:** `crates/lgbm-compute/src/kernels/split.rs` (`split_scan_body` at line 142, `find_best_split_kernel` at 393, `find_best_split_kernel_f32` at 446) and `primitives.rs` (`block_scan_body` at 82 → `block_scan_kernel_f64`/`_f32` at 159/171).

The stage-1 `split_eval_body` follows the SAME shape as `split_scan_body`: one shared `#[cube]` fn holds the numerical core, and thin `#[cube(launch)]` wrappers pass `hist_base`/`out_base` bases. **Note two divergences the analog's doc comment itself flags — D-17-01 must diverge from `split.rs` here** (per RESEARCH D-01):

- `split.rs::round_int` (line 108-111) is `(int)(x + 0.5f)` — **WRONG rounding for this phase**. The CUDA core uses `__double2int_rn` (round-ties-even). Write a NEW even-rounding helper (RESEARCH "Count Recovery" §, `f64::round_ties_even` or the branch-free identity).
- `split.rs::split_scan_body` accumulates left-sums **incrementally**; the CUDA core scans-then-complements (`parent_total − cumulative`). Same wrapper *shape*, different accumulation *body*.

**Launch-wrapper shape to copy (split.rs:393-435):**
```rust
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn find_best_split_kernel(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    num_bin: i32,
    // ... scalar args ...
) {
    split_scan_body(hist, 0u32, out, 0u32, num_bin, /* ... */);
}
```
The f32 mirror (`find_best_split_kernel_f32`, split.rs:444) is byte-identical except the cell type (`f32` in place of `f64`) — copy that convention verbatim for the hip mirror.

**cubecl-cpu MLIR lowering constraints (LOAD-BEARING — copy verbatim, split.rs:180-202):**
- Loop-carried mutables MUST init from LITERALS, never directly from a scalar kernel arg (`let mut best_gain = 0.0f64;`, not `= sum_hessian`).
- Every conditional store MUST be branchless `select(...)`, not nested `if` mutation chains.
- Prefer bounded `for`/counter loops over decrementing `while` with index mutation.

---

#### Pattern B — Gain math reuse for `USE_SMOOTHING=false`; net-new smoothing branch (D-02)

**Analog:** `crates/lgbm-compute/src/gain.rs` (`threshold_l1` at 45, `get_leaf_gain` at 70, `get_split_gains` at 87, `calculate_splitted_leaf_output` at 135, `get_leaf_gain_given_output` at 108 — currently a plain host `fn`, NOT `#[cube]`).

RESEARCH §D-02 verdict: **bit-identical to CUDA for the 3 non-smoothing flags — REUSE directly.** These `#[cube]` fns are already callable from inside a kernel body. For `USE_SMOOTHING=true` (net-new):
- Add the output-blend to `CalculateSplittedLeafOutput` (RESEARCH Code Examples, form `(B)`).
- Add the given-output gain path to `GetLeafGain` (form `(D)`), which requires **promoting `get_leaf_gain_given_output` (gain.rs:108) from a host `fn` to `#[cube]`** so it runs on device.
- Extend `get_split_gains` signature with `left_count, right_count, path_smooth, parent_output` (ignored when smoothing off).

**Branchless `Sign` idiom to preserve (gain.rs:51-53) — do NOT use `f64::signum`:**
```rust
let pos = select(s > 0.0, 1.0, 0.0);
let neg = select(s < 0.0, 1.0, 0.0);
(pos - neg) * reg_s   // Common::Sign(s) * max(0,|s|-l1)
```
The f32 mirrors (`threshold_l1_f32` at 163, etc.) already exist — reuse for the hip path. Note gain.rs:166-172 the WR-05 rule: pin EVERY f32 literal (`0.0f32`, `1.0f32`) so cubecl cannot widen to f64.

**`GainConfig` surface (gain.rs:231):** reuse as-is; it already carries `path_smooth`, `lambda_l1/l2`, `min_gain_to_split`, `min_data_in_leaf`, `min_sum_hessian_in_leaf` plus the categorical fields Phase-22 needs.

---

#### Pattern C — SoA pre-allocated split-record + 8-int export (D-11)

**Analog:** `crates/lgbm-compute/src/kernels/split_info.rs` (`SplitScalars` at 81, `DeviceBuffers` at 188, `DeviceSplitInfo::new` at 267).

`DeviceSplitInfo` already mirrors the `CUDASplitInfo` field list (`is_valid`, `threshold`, `default_left`, per-side `sum_gradients/sum_hessians/count/gain/value`, `num_cat_threshold` + reserved cat slabs) and **allocates every field buffer exactly ONCE in `new` via `client.empty(...)`** (split_info.rs:290-321). Copy the alloc-once idiom:

```rust
let mut device_allocations = 0usize;
let mut alloc = |elem_size: usize, len: usize| -> Handle {
    device_allocations += 1;
    client.empty(len * elem_size)   // the ONLY client.empty caller in the module
};
```
Stage-1 writes slot `[t]` (smaller) / `[t+num_tasks]` (larger); stage-3 exports the 8-int subset. The `device_allocations` counter (split_info.rs:369) is the structural "allocated exactly once" invariant — reuse it for the global-memory scratch buffers (D-05) too. Pre-allocate scratch (`feature_hist_{grad,hess,stat}_buffer`, `feature_hist_index_buffer`) in the same `new`-style constructor, never per-split.

**V5 boundary check idiom (split_info.rs:277, 559):** `checked_mul` slab sizing + `check_slot` typed-error before any indexed write. Mirror for `num_tasks`/`num_leaves`/`num_bin` at the launch boundary.

---

#### Pattern D — LDS `SharedMemory`/`sync_cube` block scan + `CubeDim(1)` cpu fold (D-03)

**Analog:** `crates/lgbm-compute/src/kernels/primitives.rs` (`block_scan_body` at 82; module doc lines 8-14 describe the cpu-single-owner / gpu-`SharedMemory`+`sync_cube` split; bitonic argsort `bitonic_argsort_body` at 697 for the deferred Phase-22 categorical core).

The cpu fold is a `CubeDim::new_1d(1)` single-owner serial inclusive accumulate (block_scan_body:91-111 — `if UNIT_POS == 0 { while i < lim { acc += data[i]; out[i] = acc; } }`). **Reuse this SHAPE** for the stage-1 within-feature scan cpu anchor (naturally handles >256 bins for the D-05 global-memory variant with no LDS cap — RESEARCH A4).

For the hip block-parallel scan, **borrow the `SharedMemory::new`/`sync_cube()` idiom** (primitives.rs module doc lines 12-14) but write a NEW two-level warp+carry scan — do NOT reuse the generic `block_scan` (its segment/`block_totals` recombination contract is a different shape, RESEARCH §D-03). The interleaved `[2b]=grad / [2b+1]=hess` layout is the READ pattern only; scan grad and hess as TWO separate scalars (RESEARCH §D-03).

The `ReduceBestGain` block-argmax family also builds on the primitives shuffle-reduction idiom (primitives.rs:407+ `reduce_max`/`plane_max`). Tie-break: strict `>` keeps the LOWEST thread index (RESEARCH Pitfall 5) — match `split.rs`'s `take = gain > best_gain` convention on the cpu fold.

---

#### Pattern E — Extra-trees RNG per task (D-07, USE_RAND)

**Analog:** `crates/lgbm-compute/src/kernels/random.rs` (`cuda_rand_advance` at 55, `cuda_rand_int32` at 69, `draw_rand_int32_on` launcher at 205).

Reuse the `CUDARandom` LCG directly for `rand_threshold = NextInt(0, num_bin-2)`. The plain-`u32` arithmetic (`state * 214013u32 + 2531011u32`, NOT `wrapping_*`) is the load-bearing device idiom (random.rs:52-58). The single-owner draw-kernel shape (random.rs:87-104) — `if UNIT_POS == 0 { walk tasks, thread state per draw }` — is the pattern for the per-task RNG seed. **Open Q1 (RESEARCH):** confirm the `InitCUDARandomKernel` per-task seed formula in `cuda_best_split_finder.cu` before writing the USE_RAND golden.

**Launcher round-trip idiom (random.rs:175-196):** `create_from_slice` (upload) → `empty` (output) → `launch_unchecked` → `read_one_unchecked` (single readback). This is the single-8-int-readback contract (SC#2) shape for stage-3 export.

---

### `crates/oracle-harness/tests/best_split_parity.rs` (test, request-response)

**Analog:** `crates/oracle-harness/tests/kernel_parity.rs` (`parse_split` at 285, `replicate_candidates` at 373, `kernel_parity_split_bit_exact_on_cpu` at 511).

**Fixture-path + skip-if-absent idiom (kernel_parity.rs:511-522) — copy verbatim:**
```rust
let path = kernels_dir().join("best_split.txt");
let Ok(text) = std::fs::read_to_string(&path) else {
    eprintln!("... SKIP — fixture {} not found ...", path.display());
    return;
};
let cases = parse_split(&text);
assert!(!cases.is_empty(), "fixture present but parsed zero cases");
```

**Bit-exact parse helpers (kernel_parity.rs:46-104):** `parse_f64_bits` (u64→f64 via `from_bits`), `parse_f32_bits_list`, `parse_u32`, `field(tokens, key)` keyed lookup. Golden values are stored as bit-hex (zero parse rounding), asserted with `oracle_harness::comparator::compare_exact_f64_bits` (import at line 35).

**Per-record parse loop (kernel_parity.rs:285-362):** line-by-line `split_whitespace()` header + typed keyed fields (`SCASE name=... num_bin=... offset=...`), then labeled data lines (`SHIST`, `SCAND_REV`, `SCAND_FWD`, `SWIN`). Assert the label token matches (`assert_eq!(ht[0], "SHIST", ...)`). Mirror this for the best-split record: task scalars + expected `CUDASplitInfo` fields + 8-int export.

**Coverage-tracking asserts (kernel_parity.rs:648-652):** track booleans (`saw_reverse_winner`, `saw_forward_winner`) and assert each category was exercised. Mirror for the D-07 matrix: `saw_use_l1`, `saw_smoothing`, `saw_rand`, `saw_globalmem`, `saw_empty`, `saw_default_left_tie`.

**cpu-fold drive:** `let client = cpu_client(); let backend = CpuBackend;` (kernel_parity.rs:521-522) then drive the new stage kernels and `compare_exact_f64_bits` field-by-field.

---

### `crates/oracle-harness/tests/fixtures/kernels/best_split.txt` (test fixture, file-I/O)

**Analog:** `crates/oracle-harness/tests/fixtures/kernels/split.txt`.

**Format to copy (split.txt lines 6-11):**
```
KERNEL_MASTER_SEED <n>
COUNTS best_split=<n>
SCASE name=<id> num_bin=<n> offset=<n> default_bin=<n> skip_default_bin=<0|1> \
      na_as_missing=<0|1> use_l1=<0|1> use_smoothing=<0|1> use_rand=<0|1> \
      min_data_in_leaf=<n> min_sum_hessian_in_leaf=<f64-bits> lambda_l1=<f64-bits> ...
SHIST <f64-bits;f64-bits;...>     # interleaved [g0,h0,g1,h1,...]
SWIN is_valid=1 threshold=<n> default_left=<0|1> left_count=<n> right_count=<n> \
     left_sum_gradient=<f64-bits> left_sum_hessian=<f64-bits> ... value=<f64-bits> gain=<f64-bits>
```
All float fields are bit-hex (u64 decimal for f64, u32 for f32) — see split.txt line 10 (`sum_gradient=13830554455654793216`). Add per-D-07-category records: default-template (fwd+rev, smaller+larger), USE_L1, USE_SMOOTHING (with `parent_output`), USE_RAND (with seed + drawn `rand_threshold`), empty/sparse-default-bin, global-memory (`num_bin>256`). RESEARCH §D-07 has the full 6-row matrix.

---

### `crates/lgbm-compute/tests/rocm_backend_parity.rs` (test, hip parity — EXTEND)

**Analog:** itself — `rocm_backend_find_best_split_bit_exact` (line 47).

Extend, don't restructure. Copy the existing test shape (lines 47-95): build inputs, drive `CpuBackend` (f64 anchor) + `RocmBackend` (f32 mirror), flatten `SplitInfo` to a `Vec<f64>` cell list (`to_cells` at 79), assert. **Two adaptations:**
- The existing `assert_bit_exact` (line 13) is bit-for-bit; the tie-aware `default_left` test needs a TOLERANT comparator — a flip is accepted ONLY on a verified f32 tie (same threshold + left_count + f32-equal gains), else hard-fail (RESEARCH "Tie-Aware default_left" §; def-f8u-01 / hip-split-parity precedent). Add a `default_left_tie` test using `ORACLE_TOL=1e-6` for gains and the tie-gate on `default_left`.
- Feature-gated `#![cfg(feature = "rocm")]` (line 7) — the extension stays under the same gate.

## Shared Patterns

### Anchor to cpu f64 fold, never GPU-vs-GPU (D-08, def-f8u-01)
**Source:** `crates/lgbm-compute/tests/rocm_backend_parity.rs:13-24` (`assert_bit_exact`) + `gain.rs:150-159` (f32-mirror rationale).
**Apply to:** every numeric output. Structure bit-exact to the cpu f64 fold; hip f32 within ~1e-6; the fold is THE anchor, hip is a mirror — never compare two f32 paths.

### Alloc-once, no per-split device alloc (D-11)
**Source:** `crates/lgbm-compute/src/kernels/split_info.rs:289-293` (the counted `alloc` closure) + `random.rs:177-178` (`empty` safe because the kernel writes every cell).
**Apply to:** `DeviceSplitInfo` reuse + all D-05 global-memory scratch buffers. Allocate in a `new`-style constructor; the `device_allocations` counter proves the invariant.

### `launch_unchecked` round-trip with single readback (SC#2)
**Source:** `crates/lgbm-compute/src/kernels/random.rs:184-196`.
```rust
unsafe { kernel::launch_unchecked(client, CubeCount::Static(..), CubeDim::new_1d(..),
    ArrayArg::from_raw_parts(h_in, n), ArrayArg::from_raw_parts(h_out.clone(), out_len), ...); }
let bytes = client.read_one_unchecked(h_out);
```
**Apply to:** all stage kernels. The stage-3 8-int export is the ONLY device→host transfer per iteration — no incidental readbacks.

### V5 launch-boundary validation before `launch_unchecked`
**Source:** `crates/lgbm-compute/src/kernels/primitives.rs:216` (`validate_scan_inputs`) + `random.rs:146` (`validate_draw_inputs`, `checked_mul` overflow guard) + `split_info.rs:277`.
**Apply to:** validate `num_bin`/`num_tasks`/`num_leaves`/buffer sizes and reject overflow with a typed `ComputeError::Runtime` BEFORE any unchecked launch (RESEARCH Security V5).

### Count-recovery rounding — DIVERGE from the analog (D-01 landmine)
**Source (WRONG for this phase):** `split.rs:108-111` `round_int` = `(int)(x+0.5f)`; `kernel_parity.rs:366` same.
**Apply:** write a NEW round-ties-even helper (`f64::round_ties_even` or the branch-free identity in RESEARCH "Count Recovery" §). This is the single most dangerous parity landmine — the reason D-01 mandates a separate fold.

## No Analog Found

None. Every new file maps to a strong existing analog. Three sub-behaviors are net-new *within* an existing analog's shape (flagged inline, not missing analogs):

| Behavior | Nearest pattern | Why net-new |
|----------|-----------------|-------------|
| `__double2int_rn` round-ties-even count recovery | `split.rs::round_int` (shape) | CUDA uses ties-even, analog uses round-half-up (D-01) |
| `USE_SMOOTHING` output-blend gain branch | `gain.rs` (extend) | Absent in host default template (D-02, RESEARCH §D-02 (B)/(D)) |
| `_GlobalMemory` >256-bin strided scan (hip) | `primitives.rs` LDS idiom (borrow) | Strided global-scratch variant; cpu fold is the same serial body with a larger bound (D-05, A4) |

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/kernels/`, `crates/lgbm-compute/src/gain.rs`, `crates/lgbm-compute/tests/`, `crates/oracle-harness/tests/` (+ `fixtures/kernels/`)
**Files scanned:** 7 analogs read (split.rs, split_info.rs, primitives.rs, random.rs, gain.rs, kernel_parity.rs, rocm_backend_parity.rs) + split.txt fixture
**Pattern extraction date:** 2026-07-01
