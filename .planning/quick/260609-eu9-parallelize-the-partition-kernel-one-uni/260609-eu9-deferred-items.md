---
quick_id: 260609-eu9
type: deferred-items
date: 2026-06-09
---

# Deferred Items — 260609-eu9

## DEF-EU9-01 — `rocm_backend_parity.rs` test bitrotted (pre-existing, not a regression)

**Status:** OUT OF SCOPE for this task; logged for a future fixup.

`crates/lgbm-compute/tests/rocm_backend_parity.rs` does not compile under
`cargo test -p lgbm-compute --features rocm`:

```
error[E0423]: expected value, found struct `RocmBackend`
  --> crates/lgbm-compute/tests/rocm_backend_parity.rs:29:15  (let gpu = RocmBackend;)
  ... also lines 50, 100, 115
```

**Root cause:** `RocmBackend` is no longer a unit struct — it gained `RefCell`
device-state fields (`resident_bins`, `resident_pool`) in tasks nn7 / p90. The test
still constructs it as `let gpu = RocmBackend;`, which is now invalid. There may be
further bitrot in the file beyond construction.

**Verified pre-existing:** stashing this task's `partition.rs` edit and recompiling
reproduces the identical 4 errors on clean HEAD — unrelated to the partition change.

**Coverage impact:** none lost. The file's `rocm_backend_data_partition_matches`
(CPU vs GPU partition) is already covered bit-exact by
`oracle-harness ... kernel_parity_partition_exact_on_hip` (which passes, 15/15 hip).

**Fix (future):** replace the unit-struct constructions with the current
`RocmBackend` constructor and reconcile any other API drift, OR delete the file if
fully superseded by the oracle-harness hip parity layer. Small, mechanical, but its
own task to keep this commit atomic.
