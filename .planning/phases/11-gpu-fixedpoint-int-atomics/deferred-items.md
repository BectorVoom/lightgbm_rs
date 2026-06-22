# Phase 11 — Deferred / Out-of-Scope Items

## DEF-11-OOS-01 — `cuda_mirror_full_corpus_leaf_matches_anchor` flaky (f32-atomic nondeterminism)

**Discovered:** Plan 11-02 Task 2 (confirmation run of the unchanged paths).

**Symptom:** `crates/lgbm-compute/tests/rocm_cuda_mirror.rs::cuda_mirror_full_corpus_leaf_matches_anchor`
intermittently fails the full-corpus large-leaf assert by a small margin, e.g. cell 50
`anchor 0.12500596 vs gpu 0.12501263` (|diff| 6.68e-6 > tol 6.25e-6, ~7% over the relative
tolerance). Across 4 consecutive runs observed FAILED, FAILED, ok, ok — **flaky**, not a hard
regression.

**Root cause (PRE-EXISTING, out of scope for Phase 11):**
- The test drives `construct_hist_cuda_mirror_kernel` (the CUDA-faithful f32 mirror), which
  accumulates via **f32 atomicAdd CAS-retry** — order-dependent, nondeterministic. On the
  full-corpus (largest) leaf the f32-accumulation envelope occasionally overshoots the test's
  `ABS 5e-6 / REL 1e-5` gate by a few percent.
- This is a DIFFERENT kernel from the resident u64 build that Phase 11 changed. Confirmed:
  `git log 434efb3..HEAD -- crates/lgbm-compute/src/kernels/histogram.rs` shows the phase-11
  commits (`6ec996e`, `cc3b040`, `c95518d`) touch ONLY the resident LDS build kernel +
  `fix_compact_kernel`; **none touch `construct_hist_cuda_mirror_kernel`**. The test file
  `rocm_cuda_mirror.rs` has **no phase-11 commits**. So this flakiness predates Phase 11 and
  is unrelated to the u64 fixed-point work.

**Why not fixed here (SCOPE BOUNDARY):** Plan 11-02 is confirmation-only for the unchanged
paths; the deviation rules forbid fixing pre-existing failures in unrelated files. The fix
direction is exactly the phase thesis — port the cuda_mirror kernel to u64 fixed-point
atomics (deterministic) OR widen its full-corpus tolerance to the documented f32-atomic
envelope. Neither is in 11-02 scope.

**Note:** the OTHER three cuda_mirror tests (`dense`, `empty_leaf`, `resident_matches_cpu_anchor_within_tol`)
pass reliably; only the full-corpus (largest, most atomic contention) case is flaky.

**Disposition:** deferred — candidate for a follow-up that either (a) ports
`construct_hist_cuda_mirror_kernel` to the same u64 fixed-point atomics this phase shipped for
the resident path, or (b) relaxes the full-corpus assert to the documented f32-atomic envelope.
