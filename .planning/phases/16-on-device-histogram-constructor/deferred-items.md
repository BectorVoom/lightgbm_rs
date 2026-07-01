# Phase 16 Deferred / Out-of-Scope Items

## DEF-16-OOS-01 — autotune LaunchKey Display format mismatch (pre-existing)

- **Test:** `lgbm-compute --lib kernels::autotune::tests::launch_key_display_and_namespace`
- **Failure:** `autotune.rs:148` asserts `Display` == `"LaunchKey(b10,f50,b256)"` but the
  impl now emits `"LaunchKey(bucket=10,feats=50,bins=256)"`.
- **Scope:** `autotune.rs` is NOT touched by any 16-04 task (`git diff --name-only` confirms).
  Pre-existing test/code drift from a prior session. Logged per SCOPE BOUNDARY; not fixed here.

## DEF-16-OOS-02 — flaky f32-atomic full-corpus mirror near-tie (pre-existing, def-f8u-01 class)

- **Test:** `lgbm-compute --features rocm --test rocm_cuda_mirror hip::cuda_mirror_full_corpus_leaf_matches_anchor`
- **Failure:** nondeterministic. Over 6 runs on the spoofed 8-CU APU it passed ~1/6 and
  otherwise diverged from the cpu f64 anchor by `|diff|` 7.15e-6 – 9.18e-6 (varying cell,
  varying magnitude each run) vs an ABS/REL-derived tol of ~5.0e-6 – 6.25e-6 — i.e. a
  marginal over-tolerance near-tie, not a deterministic miss.
- **Root cause:** the f32-atomic accumulation path (`construct_leaf_hist_resident_lds_kernel_u64`
  / the f32 resident mirror) has nondeterministic add-order on the APU; this is the documented
  def-f8u-01 "never hold two nondeterministic GPU f32 paths to 1e-6" class.
- **Scope:** PRE-EXISTING — this is one of the 4 shipped CUDA-mirror tests (per 16-03-SUMMARY),
  NOT a Phase-16 on-device build/fix/subtract case. The shipped
  `construct_leaf_hist_resident_lds_kernel_u64` is **byte-unchanged** this phase
  (git-diff verified identical pre-phase `6b1ea9d` → HEAD), so Phase 16 did not cause it.
  It runs only under `--features rocm` on the physical GPU; the **default merge gate**
  (`cargo test --workspace`, ODL-19) is GREEN 845/0. All NEW Phase-16 deterministic
  on-device cases pass. Logged per SCOPE BOUNDARY; the tolerance/flakiness fix is a
  separate concern (tighten the assert to a multi-run/seed-stable envelope, or pin the
  f32 mirror's add-order) — surfaced to the human at the 16-05 ROCm-parity checkpoint.
