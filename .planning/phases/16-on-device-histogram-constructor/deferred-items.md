# Phase 16 Deferred / Out-of-Scope Items

## DEF-16-OOS-01 — autotune LaunchKey Display format mismatch (pre-existing)

- **Test:** `lgbm-compute --lib kernels::autotune::tests::launch_key_display_and_namespace`
- **Failure:** `autotune.rs:148` asserts `Display` == `"LaunchKey(b10,f50,b256)"` but the
  impl now emits `"LaunchKey(bucket=10,feats=50,bins=256)"`.
- **Scope:** `autotune.rs` is NOT touched by any 16-04 task (`git diff --name-only` confirms).
  Pre-existing test/code drift from a prior session. Logged per SCOPE BOUNDARY; not fixed here.
