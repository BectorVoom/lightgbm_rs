---
phase: 14-foundation-shared-device-primitives-device-structs-rng
verified: 2026-06-29T11:09:04Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 14: Foundation — Shared Device Primitives + Device Structs/RNG Verification Report

**Phase Goal:** The reusable CubeCL device primitives and device structs/RNG every later subsystem builds on are ported and validated, and the existing on-device growth seam + anchor-pinned oracle are re-established/extended — all additive and off by default.
**Verified:** 2026-06-29T11:09:04Z
**Status:** passed
**Re-verification:** No — initial verification
**Requirements:** ODL-01, ODL-02

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
| - | ----- | ------ | -------- |
| 1 | Shared device primitives (block+global prefix-sum incl/excl, shuffle reductions sum/max/min/dot, index-only bitonic argsort, weighted/unweighted percentile) exist as reusable CubeCL kernels, anchor-pinned where they carry numeric output | VERIFIED | `crates/lgbm-compute/src/kernels/primitives.rs` (1572 lines) — all 20 `pub fn` present incl. `prefix_sum_{inclusive,exclusive}_f64/f32_on`, `reduce_{sum,max,min}_f64_on`, `dot_product_f64_on`, `bitonic_argsort_on`, `bitonic_argsort_global_on`, `bitonic_argsort_items_on`, `percentile_{unweighted,weighted}_f32_on`, plane variants. 22 serial-f64-anchored self-tests green (`primitives_self.rs`); `primitive_parity.rs` cross-validates 16 prefix-sum + 12 reduction + 13 argsort + 9 unweighted-percentile records against committed C++ goldens — 4 passed CPU. Weighted-percentile C++ golden cross-check deferred to NVIDIA/Kaggle (D-02, documented), but the kernel exists and is serial-f64 bit-exact in self-test. |
| 2 | CubeCL-safe pre-allocated device split-record (CUDASplitInfo analog, no per-split in-kernel alloc) + CUDARandom LCG bit-identical to host `Random` | VERIFIED | `split_info.rs` (557 lines): `DeviceSplitInfo<R>` SoA, one `client.empty` per field allocated once in `new()`, `device_allocations()` alloc-once counter, `copy_slot` zero-alloc (`copy_within`), `MAX_CAT_PER_SPLIT=32` reserved cat slabs. `random.rs`: `cuda_rand_advance/int16/int32/next_float` plain-u32 `214013*x+2531011` (no `wrapping_*`/`Atomic<i64>`). `cuda_random_parity.rs` asserts device stream == host `lgbm_core::random::Random` via `to_bits()`/i32 equality (no tolerance) — 5 passed; `split_info.rs` tests 9 passed. |
| 3 | Additive `grow_tree_on_device` seam + default-false `on_device_growth_supported()` + tie-aware `assert_on_device_tree_matches_cpu_anchor` oracle re-established/extended; never GPU-vs-GPU | VERIFIED | `lgbm-compute/src/lib.rs:1239-1241` `on_device_growth_supported()` → `false`; `:1289`/`:2224` `grow_tree_on_device()` → `Ok(None)` (both impls). `learner.rs:498` `on_device_eligible = backend.on_device_growth_supported() && cuda_on_device_env()` (AND-gate, dead while discriminator false). `learner_parity.rs:2185` `assert_on_device_tree_matches_cpu_anchor` + Slice-0 tests `..._seam_is_provable_noop_slice0` / `..._oracle_host_fallback_slice0` assert discriminator false + `Ok(None)` for CpuBackend AND GpuBackend. `learner_parity` 29 passed CPU. |
| 4 | `LGBM_CUDA_ON_DEVICE` OFF by default; CPU/ROCm/host-CUDA paths byte-unchanged; full merge gate green | VERIFIED | `cuda_on_device_env()` requires exactly `"1"`; default unset → false → on-device fork dead, host path unchanged. **Independently re-ran** `cargo test --workspace` with `LGBM_CUDA_ON_DEVICE` unset → **816 passed, 0 failed, 3 ignored** — exactly matches the reported gate. `raw_bin_train_matches_cpp_golden`, `learner_parity`, lgbm/treelearner/compute suites all green. |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-compute/src/kernels/primitives.rs` | Full-depth + skeleton primitives | VERIFIED | 1572 lines, 20 pub fns, wired in `kernels/mod.rs` |
| `crates/lgbm-compute/src/kernels/split_info.rs` | SoA pre-allocated split-record | VERIFIED | 557 lines, alloc-once counter, zero-alloc copy_slot |
| `crates/lgbm-compute/src/kernels/random.rs` | CUDARandom LCG | VERIFIED | 268 lines, plain-u32 recurrence, no wrapping_*/Atomic<i64> |
| `crates/lgbm-compute/tests/primitives_self.rs` | Serial-f64 anchor self-tests | VERIFIED | 22 tests green |
| `crates/lgbm-compute/tests/split_info.rs` | Structural tests | VERIFIED | 9 tests green |
| `crates/lgbm-compute/tests/cuda_random_parity.rs` | Host-Random parity | VERIFIED | 5 tests green, to_bits exact |
| `crates/oracle-harness/tests/primitive_parity.rs` | C++ golden replay | VERIFIED | 512 lines, 4 tests green CPU |
| `crates/oracle-harness/fixtures/primitives/*.txt` | Committed C++ goldens | VERIFIED | prefix_sum 17 / reduce 25 / argsort 14 / percentile 19 records, committed |

### Key Link Verification

| From | To | Via | Status |
| ---- | -- | --- | ------ |
| `learner.rs` on_device fork | `Backend::grow_tree_on_device` | AND-gate `on_device_growth_supported() && cuda_on_device_env()` → dead | WIRED (no-op) |
| `cuda_random_parity.rs` | `lgbm_core::random::Random` | `next_short`/`next_float`/`to_bits()` bit-exact assert | WIRED |
| `primitive_parity.rs` | `fixtures/primitives/*.txt` | `load_or_skip` + per-record replay vs Rust primitives | WIRED |
| `kernels/mod.rs` | primitives/split_info/random | `pub mod` barrel | WIRED |

### Behavioral Spot-Checks (re-run by verifier, CPU/default features)

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Full merge gate | `cargo test --workspace` (env unset) | 816 passed, 0 failed, 3 ignored | PASS |
| Primitive C++ golden parity | `cargo test -p oracle-harness --test primitive_parity` | 4 passed | PASS |
| Seam no-op + oracle | `cargo test -p oracle-harness --test learner_parity` | 29 passed | PASS |
| CUDARandom host parity | `cargo test -p lgbm-compute --test cuda_random_parity` | 5 passed | PASS |
| Split-record alloc-once | `cargo test -p lgbm-compute --test split_info` | 9 passed | PASS |
| Primitive self-tests | `cargo test -p lgbm-compute --test primitives_self` | 22 passed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| ODL-01 | 14-03, 14-05, 14-06 | Shared device-primitive kernels (prefix-sum, reductions, argsort, percentile), anchor-pinned | SATISFIED | SC#1 — all primitives present, self-tested + C++-golden-replayed |
| ODL-02 | 14-04 | Pre-allocated CUDASplitInfo analog + CUDARandom bit-identical to host Random | SATISFIED | SC#2 — DeviceSplitInfo SoA alloc-once + RNG bit-exact |

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX` debt markers; no `todo!`/`unimplemented!`/placeholder in phase source files.

### Deferred Items (intentional, addressed in later phases — not gaps)

| Item | Addressed In | Evidence |
| ---- | ------------ | -------- |
| Weighted-percentile C++ golden cross-check (`status=deferred_kaggle_nvcc`) | Phase 19 (objectives/ranking) | D-02; kernel exists + serial-f64-anchored now; non-idempotent on spoofed APU, deferred to NVIDIA capture |
| Multi-block argsort / items-sort depth hardening | Phase 19/22 | D-02 skeletons; correct on supported input regime, depth deferred to first real consumer |
| Recursive >1024-block global prefix-sum | Phase 15 | Guarded (`MAX_GLOBAL_SCAN_BLOCKS`) typed error, not truncation |
| Device-kernel slot-copy + 8/16-int readback packet | Phase 17/18 | D-07; reserved handles pre-allocated |
| Categorical cat_threshold slab fill | Phase 22 | D-06; slabs reserved at MAX_CAT_PER_SPLIT |

### Notes (informational)

- **ROCm leg not independently re-run.** SUMMARY claims `--features rocm`: primitive_parity 5/5, learner_parity 33/33 (f32 ~1e-6 hip leg + both Slice-0 tests under rocm). This was NOT re-run by the verifier — it requires the spoofed gfx1152 APU + `--features rocm` and is the best-effort secondary gate, not the hard merge gate. The hard gate per CLAUDE.md is the cubecl-cpu f64 anchor, which was independently re-run and is green (816/0/3). ROCm re-validation is optional and does not affect the verdict.
- **Split-record host-staging.** `DeviceSplitInfo` keeps authoritative per-field storage in host `Vec`s with reserved-once device handles, because cubecl 0.10 has no in-place device-write API (documented decision, D-05). The ODL-02 literal contract — "one pre-allocated device buffer per field, allocated once, no per-split in-kernel alloc" — is satisfied: device handles are `client.empty` once in `new()`, proven by the `device_allocations()` counter. Phase 17 device kernels will write these handles directly.

### Gaps Summary

No gaps. All four ROADMAP success criteria are independently verified against the live codebase. The seam is a genuine no-op (`on_device_growth_supported()==false`, `grow_tree_on_device()==Ok(None)`, env AND-gate dead by default). The merge gate (`cargo test --workspace`, env unset) was re-run by the verifier and reproduces the claimed 816/0/3 exactly. Primitives, split-record, and RNG exist as substantive implementations, are wired into the kernels barrel, and are anchored to the cubecl-cpu f64 fold / committed C++ goldens / the host `Random` oracle — never GPU-vs-GPU. Requirements ODL-01 and ODL-02 are satisfied.

---

_Verified: 2026-06-29T11:09:04Z_
_Verifier: Claude (gsd-verifier)_
