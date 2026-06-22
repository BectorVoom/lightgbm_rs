# Phase 11: gpu-fixedpoint-int-atomics - Pattern Map

**Mapped:** 2026-06-22
**Files analyzed:** 6 production sites + 2 parity-test sites
**Analogs found:** 8 / 8 (all analogs are IN-CODEBASE; the spike examples are working twins)

## Orienting finding (read before planning)

The ROCm resident histogram path has **two** build sub-paths, only ONE of which is the
fixed-point target:

1. **f32-atomic RAW build → f64 widen chain** (`resident_raw_build_into` →
   `fix_compact_kernel`), driven by `Backend::build_resident_leaf` /
   `build_fix_compact_resident_f64_on`. This is the DEFAULT large-leaf resident path and
   the **f32-atomic accumulation this phase replaces with u64 fixed-point**. The f32→f64
   widen in `fix_compact_kernel`'s first pass becomes the **u64→i64 dequant** `(bits as
   i64) / 2^30`.

2. **Sequential f64 fold** (`build_fix_scan_fused_kernel` Stage 1), driven by
   `Backend::build_fix_scan_resident`. This is ALREADY bit-exact (single-owner cube, no
   atomics) and is **gated OFF by default** (`FUSED_MAX_NUM_DATA = -1`). It is NOT a
   fixed-point target — but it is the SUBTRACT-TRICK PARENT operand source, so its f64
   output must remain dimensionally compatible with the dequantized fixed-point output (it
   already is: both are f64 fixed+compacted histograms). No change required, but the audit
   must confirm the two paths still produce the same f64 cell layout.

The non-resident `construct_histograms_lds_f32_on` seam (used by
`Backend::construct_histograms`, the per-call non-resident path) is the **secondary**
f32-atomic site. Whether to also convert it (or leave it f32) is Claude's-discretion
buffer-plumbing; the resident path is the phase's primary deliverable.

The **subtract trick** (`subtract_hist_kernel` / `subtract_histograms_f64_from_handles_on`)
operates on **f64 cells AFTER dequant** — so it is UNCHANGED. The integer accumulation is
confined to the RAW build + its merge buffer; once dequantized to f64 in the
fix/widen pass, the rest of the chain (subtract, scan) is identical. This is the key
scoping decision: **widen the cell type ONLY between RAW-build and dequant**, not through
the whole pool.

## File Classification

| New/Modified File / Symbol | Role | Data Flow | Closest Analog | Match Quality |
|----------------------------|------|-----------|----------------|---------------|
| `kernels/histogram.rs` :: new `construct_leaf_hist_resident_lds_kernel_u64` (or `<C>`-generic cell type) | kernel (cube) | event-driven / atomic-scatter | `examples/gpu_int_vs_f32_psweep.rs::build_u64_rp` (exact twin) + existing `construct_leaf_hist_resident_lds_kernel` (f32 original) | exact |
| `kernels/histogram.rs` :: `resident_raw_build_into` (u64 buffer alloc + dispatch) | utility (launcher) | request-response | itself (f32 version) | exact (in-place type swap) |
| `kernels/histogram.rs` :: `fix_compact_kernel` (widen pass → dequant pass) | kernel (cube) | transform | itself (f32→f64 widen) | exact (cast swap) |
| `kernels/histogram.rs` :: `build_fix_compact_resident_f64_on` (h_raw alloc f32→u64) | utility (launcher) | request-response | itself (f32 `h_raw`) | exact |
| `lib.rs` :: `RocmBackend::build_resident_leaf` (seam) | service (Backend impl) | request-response | itself + `construct_histograms` seam (lines 1965-1992, the f32 ~1e-6 contract doc) | exact |
| `lib.rs` :: overflow guard (new) | utility | transform | `gpu_fixedpoint_i64.rs` SCALE constant + spike-018 overflow note | role-match |
| `crates/lgbm-compute/tests/rocm_row_partition.rs` :: re-pin | test | request-response | `row_partition_batched_matches_cpu_anchor_p1_and_p_gt_1` (the `cpu_ref` anchor pin) | exact |
| `crates/lgbm-compute/tests/rocm_parallel_histogram.rs` / `rocm_backend_parity.rs` :: re-pin | test | request-response | `lds_within_tolerance_of_cpu_f64_anchor` + `assert_bit_exact` | exact |

## Pattern Assignments

### `kernels/histogram.rs` :: new u64 fixed-point resident LDS build kernel (kernel, atomic-scatter)

**Primary analog (the validated twin):** `examples/gpu_int_vs_f32_psweep.rs::build_u64_rp`
(lines 81-123) — already the spike-007 `build_rp` layout (CubeCount=(feats,P)) with u64
two's-complement atomics, the realistic resident regime.

**Production analog being replaced:** `kernels/histogram.rs::construct_leaf_hist_resident_lds_kernel`
(lines 1163-1211) — the f32 original. The new kernel keeps this signature/structure
EXACTLY (resident column gather, row-partition over `CUBE_POS_Y`, slot_off sentinel,
per-feature LDS sub-hist) and swaps ONLY the cell type + the quantize/store/merge idiom.

**The quantize/store idiom to copy** (from `build_u64_rp` lines 99-117 — the LDS-atomic
two's-complement pattern):
```rust
const SCALE: f32 = 1_073_741_824.0; // 2^30 (spike-018a: within ~1e-6, exact in cancelling regime)
let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_U64);   // 2 u64/bin, NOT Atomic<i64>
// zero:
let mut c = UNIT_POS as usize;
while c < fl { sub[c].store(0u64); c += cd; }   // u64 store is FINE; i64 store is NOT (cubecl-hip 0.10)
sync_cube();
// scatter — quantize round(value*2^30) as i64, store BITS as u64; wrapping fetch_add == i64 two's-complement add:
let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE)));
let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE)));
sub[ti].fetch_add(qg);
sub[ti + 1].fetch_add(qh);
// merge LDS → global u64 slot (same per-cell merge as the f32 path):
out[base + m].fetch_add(sub[m].load());
```

**Hard constraint (spike-018b, line 24 of CONTEXT):** MUST be `Atomic<u64>`. `Atomic<i64>`
compiles but FAILS at runtime — cubecl-hip 0.10 lowers `Atomic<i64>::store` to
`atomicExch(long long*)` which HIP lacks. The `fetch_add` path reinterpret-casts to
`uint64*` correctly; `store` does not. So: store the i64 quantized value's BITS as u64.

**Note the cell-count is unchanged:** the existing `HIST_LDS_MAX = 512` is "2 f32
cells/bin" (`lib.rs:620`/`histogram.rs:620`). For u64 it becomes "2 u64 cells/bin" — the
spike uses `HIST_LDS_U64 = 512` (same element COUNT, 2× bytes). The 256-bin cap is
unchanged (512 cells = 256 bins × 2). LDS budget doubles (2 KiB → 4 KiB/cube); confirm
this fits (gfx1100/APU LDS is 64 KiB — fine).

**`<B: Int>` resident-width generic** is preserved exactly (lines 1163, 1738-1742): the
bin INDEX read stays `u32::cast_from(resident_bins[...])`; only the OUTPUT cell type
changes. The `launch_lds!`/`launch_naive!` macro dispatch (lines 1720-1742, 1771-1794)
carries over verbatim with the new kernel name.

---

### `kernels/histogram.rs` :: `resident_raw_build_into` (utility/launcher) + `build_fix_compact_resident_f64_on`

**Analog:** itself (the f32 version, lines 1660-1796 and 2093-2213).

**The buffer-type swap** — in `build_fix_compact_resident_f64_on` (lines 2113-2127), the
RAW buffer alloc changes f32 → u64:
```rust
// BEFORE (f32 RAW):
let zeros32 = vec![0.0f32; slot_len];
let h_raw = client.create_from_slice(f32::as_bytes(&zeros32));
// AFTER (u64 fixed-point RAW):
let zeros_u64 = vec![0u64; slot_len];
let h_raw = client.create_from_slice(u64::as_bytes(&zeros_u64));
```
`resident_raw_build_into` (line 1660) signature is unchanged except `h_out: Handle` now
points at a u64 buffer; its `launch_lds!`/`launch_naive!` arms dispatch the new u64
kernel. The `slot_off_sentinel` helper (line 1643) and `row_partition_count` (line ~735)
are width-independent — untouched.

**Audit checklist for the wider cell type (the SPEC.md item-2 audit):** every
`f32::as_bytes(&zeros…)` allocating a RAW build target (lines 842, 961, 1120, 1450, 1601,
1681, **2114**) must be classified: RAW-build merge buffers → u64; grad/hess INPUT buffers
(`h_g`/`h_h`, lines 1681-1682) stay f32 (the kernel quantizes them in-kernel). The
non-resident `construct_histograms_lds_f32_on` (line 817) and the batched non-resident
launchers (lines 932, 1098) are SEPARATE seams — convert per Claude's discretion.

---

### `kernels/histogram.rs` :: `fix_compact_kernel` — widen pass becomes dequant pass (kernel, transform)

**Analog:** itself (lines 1830-1925), the f32→f64 widen at lines 1852-1861.

The folded WIDEN first-pass currently does `hist[wbi] = f64::cast_from(h_raw[wbi])` over an
`h_raw: &Array<f32>`. With fixed-point, `h_raw` becomes `&Array<u64>` (or `i64`) and the
widen becomes a **dequant**:
```rust
// BEFORE (f32 widen, lines 1857-1861):
for w in 0..nb {
    let wbi = base + (w as usize) * 2;
    hist[wbi]     = f64::cast_from(h_raw[wbi]);
    hist[wbi + 1] = f64::cast_from(h_raw[wbi + 1]);
}
// AFTER (u64-bits → i64 → f64/2^30 dequant — the host-side idiom from
// gpu_fixedpoint_i64.rs:191 `(*v as f64 / SCALE as f64)` ported in-kernel):
const SCALE_F64: f64 = 1_073_741_824.0; // 2^30
for w in 0..nb {
    let wbi = base + (w as usize) * 2;
    hist[wbi]     = f64::cast_from(i64::cast_from(h_raw[wbi]))     / SCALE_F64;
    hist[wbi + 1] = f64::cast_from(i64::cast_from(h_raw[wbi + 1])) / SCALE_F64;
}
```
**Everything AFTER the widen pass (the FixHistogram fold lines 1863-1892 + compact lines
1894-1925) is UNCHANGED** — it operates on the already-f64 `hist`. This is the seam that
confines the integer accumulation: RAW build is integer, dequant happens here, fix/compact/
subtract/scan stay f64. The `sum_gradient`/`sum_hessian` leaf scalars (line 1842-1843) are
the RAW host-side f64 totals — also unchanged (they were never quantized).

**The host-side dequant reference** (the exact arithmetic, from `gpu_fixedpoint_i64.rs`
lines 170-171, 191): `u64::from_bytes(...).iter().map(|&u| u as i64)` then `/ SCALE`. The
in-kernel version above reproduces it cell-by-cell.

---

### `lib.rs` :: `RocmBackend::build_resident_leaf` (service/Backend seam, request-response)

**Analog:** itself (lines 2239-2284) + the `construct_histograms` seam doc (lines
1965-1992) for the ~1e-6 contract language.

The seam body is UNCHANGED — it calls `build_fix_compact_resident_f64_on` and stores the
returned f64 `Handle`. The buffer-type change is entirely INSIDE that launcher (above). The
ONLY edit here is the DOC: the seam comment must note the RAW accumulation is now u64
fixed-point (MORE accurate, deterministic) rather than f32-atomic, within the same ~1e-6
gate. Copy the contract-language pattern from the `construct_histograms` seam doc:

> "Swapping this seam … CHANGES the seam's accumulation … that is the GPU's ~1e-6
> best-effort contract, not a bit-exact one. The CpuBackend f64 fold is the bit-exact hard
> merge gate and is unaffected."

`subtract_resident` (lines 2302-2333), `move_resident` (2289-2297), `scan_resident_leaf`
(2339-2359), `build_fix_scan_resident` (2367-2409) are **UNCHANGED** — they consume the
post-dequant f64 Handle. `upload_resident_bins` (2074-2135) is **UNCHANGED** — bins are
still u8/u16/u32 indices, not histogram cells.

---

### `lib.rs` :: overflow guard (new utility, transform)

**Analog:** `gpu_fixedpoint_i64.rs` SCALE constant (line 24) + spike-018 disposition note
(README lines 63, 113-116): "i64@2^30 safe to ~1e9 rows × |g|≤8; add a documented scale/
range check or clamp for pathological extreme leaves."

Place a documented bound at the `build_fix_compact_resident_f64_on` boundary (the same
place its other V5 checks live, lines 2149-2171): assert/clamp that
`leaf_rows.len() * max_abs_grad * 2^30 < i64::MAX`. No exact in-codebase analog for the
guard arithmetic — the SCALE constant and the bound (`≤~1e9 rows × |g|≤8`) come from the
spike. Document the bound inline (SPEC.md item 3).

---

### Parity test re-pins (test, request-response)

**Primary analog:** `crates/lgbm-compute/tests/rocm_row_partition.rs` ::
`row_partition_batched_matches_cpu_anchor_p1_and_p_gt_1` (lines 53-130) and
`naive_batched_fallback_matches_cpu_anchor_within_tol` (lines 144-190).

**The pin-to-anchor pattern (def-f8u-01 — pin to the CPU f64 anchor, NEVER GPU-vs-GPU):**
```rust
// f64 anchor computed host-side in the EXACT cell layout (lines 23-40):
fn cpu_ref(...) -> Vec<f64> {
    let mut out = vec![0.0f64; slot_len];
    for ... { out[base + bin*2] += f64::from(g[r as usize]);
              out[base + bin*2 + 1] += f64::from(h[r as usize]); }
    out
}
// relative-error gate (lines 45-49):
fn max_rel(a: &[f64], b: &[f64]) -> f64 { ... .fold(0.0, f64::max) }
// assert vs anchor (lines 96-98):
assert!(rel_p1 < 1e-5, "...diverged from the cpu f64 anchor beyond the f32 LDS gate: {rel_p1:e}");
```

**What changes for fixed-point:** the new u64 path is ~3600× MORE accurate (spike-018b:
5.9e-9 vs f32 2.2e-5), so the re-pinned assert should TIGHTEN from `1e-5` toward exact in
the cancelling/integer regimes (spike-018a: exact). Keep the `cpu_ref` f64 anchor and the
`max_rel`/`max_abs` helpers verbatim; tighten the tolerance constant and update the comment
(the f32-LDS-gate rationale no longer applies — it's now a fixed-point gate). Add a
DETERMINISM assert (two runs bit-equal) — copy from `gpu_fixedpoint_i64.rs` lines 181-182
(`let i64_det = i64a == i64b;`).

**Secondary analog (the seam-level f32 pin):**
`rocm_parallel_histogram.rs::lds_within_tolerance_of_cpu_f64_anchor` (lines 41-63) — the
`construct_histograms_cpu` vs `construct_histograms_lds_f32_on` + `max_rel < 1e-5` pattern.
If the non-resident seam is also converted, re-pin this the same way.

**Bit-exact pin (for the UNCHANGED subtract/f64 paths):**
`rocm_backend_parity.rs::assert_bit_exact` (lines 13-26) + the subtract test (lines 98-109)
stay bit-exact — they operate on f64 cells and are not touched by the integer build.

## Shared Patterns

### u64 two's-complement fixed-point idiom (the phase's core mechanism)
**Source:** `examples/gpu_fixedpoint_i64.rs::build_u64` (lines 68-102),
`examples/gpu_int_vs_f32_psweep.rs::build_u64_rp` (lines 81-123).
**Apply to:** the new resident build kernel; the dequant pass in `fix_compact_kernel`.
```rust
const SCALE: f32 = 1_073_741_824.0;  // 2^30
// quantize+store (in build kernel):
let q = u64::cast_from(i64::cast_from(f32::round(value * SCALE)));
sub[ti].fetch_add(q);                // Atomic<u64>, wrapping == i64 two's-complement
// dequant (in fix/widen kernel or host readback):
let v_f64 = f64::cast_from(i64::cast_from(bits)) / 2.0f64.powi(30);
```

### cubecl-hip 0.10 Atomic<u64> constraint (hard gate)
**Source:** spike-018b (README lines 36-44), CONTEXT line 24.
**Apply to:** every atomic cell in the new kernel.
- USE `Atomic<u64>` + `.store(0u64)` + `.fetch_add(qbits)`.
- NEVER `Atomic<i64>` — `::store` lowers to `atomicExch(long long*)`, absent in HIP.

### Pin-to-CPU-f64-anchor (def-f8u-01)
**Source:** `rocm_row_partition.rs` `cpu_ref` + `max_rel` (lines 23-49);
`rocm_backend_parity.rs::assert_bit_exact` (lines 13-26).
**Apply to:** every re-pinned parity test.
- Compute an f64 anchor host-side in the exact cell layout; compare GPU-vs-anchor.
- NEVER compare two nondeterministic GPU f32 paths to each other (the MEMORY def-f8u-01
  lesson — the fixed-point path being deterministic actually RELAXES this, but the test
  must still pin to the anchor, not to another GPU run).

### LAUNCH_UNCHECKED safety-comment template (NRW-01 / CMP-01)
**Source:** the SAFETY blocks at `resident_raw_build_into` (lines 1692-1719) and
`build_fix_compact_resident_f64_on` (lines 2176-2194).
**Apply to:** the new u64 kernel's launcher arms — the bounds proof is IDENTICAL (only the
cell type changed; the index arithmetic and host-side V5 checks are unchanged), so the
existing template carries over with the cell-type words swapped.

## No Analog Found

| File / Symbol | Role | Data Flow | Reason |
|---------------|------|-----------|--------|
| overflow-guard arithmetic | utility | transform | No existing range/clamp check for accumulation overflow in the codebase; the bound (`≤~1e9 rows × \|g\|≤8` at 2^30) comes from spike-018 README, not from an existing function. Pattern the *placement* on the existing V5 checks (histogram.rs:2149-2171) but the arithmetic is new. |

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/kernels/{histogram,subtract}.rs`,
`crates/lgbm-compute/src/lib.rs`, `crates/lgbm-compute/examples/{gpu_fixedpoint_i64,
gpu_int_vs_f32_psweep}.rs`, `crates/lgbm-compute/tests/{rocm_row_partition,
rocm_parallel_histogram,rocm_backend_parity}.rs`, `crates/lgbm-treelearner/src/
{resident_pool,learner}.rs`.
**Files scanned:** 10
**Pattern extraction date:** 2026-06-22
