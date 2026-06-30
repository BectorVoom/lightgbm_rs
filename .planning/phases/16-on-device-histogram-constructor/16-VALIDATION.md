---
phase: 16
slug: on-device-histogram-constructor
status: approved
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-01
---

# Phase 16 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (workspace) |
| **Config file** | none — Cargo workspace; tests live in `crates/lgbm-compute/tests/` + `#[cfg(test)]` modules |
| **Quick run command** | `cargo test -p lgbm-compute` |
| **Full suite command** | `cargo test` (the full merge gate — must stay green, ODL-19) |
| **Estimated runtime** | ~quick: tens of seconds; full: minutes |

*Note: on-device (`LGBM_CUDA_ON_DEVICE`) parity tests requiring the cubecl-hip backend run only where the ROCm GPU is present; the cpu f64-fold anchor tests run everywhere and are the hard gate. Never GPU-vs-GPU (def-f8u-01).*

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-compute`
- **After every plan wave:** Run `cargo test` (full merge gate)
- **Before `/gsd-verify-work`:** Full suite must be green; default paths byte-unchanged
- **Max feedback latency:** ~60 seconds (quick crate-scoped run)

---

## Per-Task Verification Map

> Derived from the 5 PLAN.md task lists. Anchor every numeric output to the cubecl-cpu f64 fold (bit-exact structure; ROCm/CUDA f32 within ~1e-6); never GPU-vs-GPU (def-f8u-01). Threat surface is internal memory-safety only (no network/auth/untrusted input).

| Task | Plan | Wave | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|------|------|------|-------------|-----------------|-----------|-------------------|-------------|--------|
| T1: sparse + large-bin/spill + mfb≠0 fixtures (D-05) | 01 | 0 | ODL-09/10 | bounds-checked fixture data | fixture | `cargo test -p oracle-harness kernel_parity` | ❌ W0 | ⬜ pending |
| T2: ordering-invariant harness + [2b]/[2b+1] assert (D-05) | 01 | 0 | ODL-09/10 | no aliased reads | harness | `cargo test -p lgbm-compute rocm_cuda_mirror` | ❌ W0 | ⬜ pending |
| T1: HistArena allocate-exactly-once slot pool (D-02/D-09) | 02 | 1 | ODL-10 | no double-alloc / leak | unit | `cargo test -p lgbm-compute histogram_arena` | ❌ W0 | ⬜ pending |
| T2: hist_t** rotation — no aliasing (D-02) | 02 | 1 | ODL-10 | parent≠smaller slot assert | unit | `cargo test -p lgbm-compute histogram_arena` | ❌ W0 | ⬜ pending |
| T1: two-tier §13 build kernel, dense+sparse (D-03/D-06/D-08) | 03 | 1 | ODL-09 | V5 launch-arg bounds | parity | `cargo test -p lgbm-compute construct_leaf_hist_partition` | ❌ W0 | ⬜ pending |
| T2: _GlobalMemory spill variant (D-04/D-09) | 03 | 1 | ODL-09 | spill-buffer bounds | parity | `cargo test -p lgbm-compute spill` | ❌ W0 | ⬜ pending |
| T3: de-quant-once → hist_t + launcher (D-01/D-08/D-05) | 03 | 1 | ODL-09 | no f64 hot loop | parity | `cargo test -p lgbm-compute construct_leaf_hist_partition` | ❌ W0 | ⬜ pending |
| T1: FixHistogram mfb repair, DROP compact (D-01/D-06) | 04 | 2 | ODL-10 | mfb bound 0<mfb<num_bin | parity | `cargo test -p lgbm-compute fix_histogram` | ❌ W0 | ⬜ pending |
| T2: Subtract via rotation + build-synced ordering (D-01/D-02) | 04 | 2 | ODL-10 | parent synced before read | parity | `cargo test -p lgbm-compute subtract` | ❌ W0 | ⬜ pending |
| T3: ConstructHistogramForLeaf entry, env-gated (D-07/D-08) | 04 | 2 | ODL-10 | OFF by default; growth=false | unit | `cargo test -p lgbm-compute on_device_growth_supported` | ❌ W0 | ⬜ pending |
| T1: hard merge gate — green, no f64 loop, no Atomic<i64> (D-07/D-08/ODL-19) | 05 | 3 | ODL-09/10 | default byte-unchanged | gate | `cargo test` + grep gates | ✅ | ⬜ pending |
| T2: ROCm f32 parity human-verify (D-06) | 05 | 3 | ODL-09/10 | ~1e-6 vs cpu anchor | manual | `LGBM_CUDA_ON_DEVICE=1 cargo test --features hip` (on ROCm host) | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky · "❌ W0" = file created by the Wave 0 fixtures task*

---

## Wave 0 Requirements

> From RESEARCH.md Validation Architecture — test gaps to land before/with the build kernel:

- [ ] Synthetic **sparse** fixture (forces `row_ptr_type` {16,32,64}) — reuse Phase-15 synthetic sparse column
- [ ] Synthetic **large-bin / global-spill** fixture (`NumLargeBinPartition() > 0`) — reuse Phase-15 synthetic large-bin column
- [ ] Purpose-built **`most_freq_bin ≠ 0`** column to force `FixHistogram`'s omit-and-repair path (DEF-07-02); anchor the repaired default-bin value (`leaf_total − scanned Σ`)
- [ ] **Build-smaller-before-subtract ordering-invariant** harness (the 8aed100-class guard: parent fully built/synced before any child subtract reads it)
- [ ] **Interleaved `[2b]/[2b+1]` layout** assert (grad at `2b`, hess at `2b+1`)
- [ ] Committed dense corpora build anchor (existing) — bit-exact to cpu f64 fold

*If none new: "Existing infrastructure covers all phase requirements." — NOT the case here; Wave 0 fixtures above are required.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ROCm/CUDA f32 parity within ~1e-6 | ODL-09/10 | Requires physical ROCm GPU backend | Run `LGBM_CUDA_ON_DEVICE=1 cargo test -p lgbm-compute --features hip` on the ROCm host; compare to cpu f64 anchor (never GPU-vs-GPU) |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (16-05 T2 is an exempt human-verify checkpoint for physical-GPU parity)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (16-01 fixtures/harnesses land first; build/fix kernels `depends_on` them)
- [x] No watch-mode flags
- [x] Feedback latency < 60s (quick crate-scoped run)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-07-01 (plan-checker VERIFICATION PASSED, 0 blockers)
