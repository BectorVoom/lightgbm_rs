# Phase 14: Scaffold + Oracle (Slice 0) - Context

**Gathered:** 2026-06-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the **on-device-growth seam** and its **anchor-pinned tie-aware oracle** with
**ZERO behavior change** — isolating wiring risk from kernel risk before any kernel is
written. Concretely:

- An additive `Backend::grow_tree_on_device` method + a default-false
  `on_device_growth_supported()` discriminator (ODL-01).
- A decide-once routing fork in `SerialTreeLearner` that falls through to the existing
  host path when on-device growth is not eligible.
- An `assert_on_device_tree_matches_cpu_anchor` oracle scaffold pinning tree STRUCTURE
  bit-exact to the cpu f64 anchor (tie-aware `default_left`) with leaf values within a
  ~1e-5 f32 envelope (ODL-02) — never comparing two nondeterministic GPU paths.

Off by default behind `LGBM_CUDA_ON_DEVICE`. CPU, ROCm, and the existing host-CUDA path
stay byte-identical to master. **No new compute kernels, no new Cargo feature** — pure
additive-discriminator wiring on the established `prefers_host_partition` /
`resident_eligible` idiom.

**In scope:** the seam method, the discriminator, the routing fork, the env gate, the
oracle scaffold + tie-aware comparator (dormant).
**Out of scope (own phases):** any actual on-device kernel/growth (Slice 1+), cross-leaf
selection (Slice 2), on-device partition (Slice 3), default-on rollout (Phase 19).

</domain>

<decisions>
## Implementation Decisions

### Oracle behavior in Slice 0 (no kernel yet)
- **D-01:** The Slice-0 oracle runs **LIVE via a host fallback**, not dormant. The oracle
  *test* obtains a real tree with
  `backend.grow_tree_on_device(..)?.unwrap_or_else(|| host_grow(..))` and feeds it to
  `assert_on_device_tree_matches_cpu_anchor`, so the comparator + seam signature + plumbing
  are exercised end-to-end and GREEN before any kernel exists.
- **D-02 (reconciliation — important for the planner):** The **production** routing fork
  is untouched/byte-identical: it uses `Ok(None) ⇒ fall through to host` (see D-03), so the
  default path issues no extra work. The host-fallback `unwrap_or_else` lives in the
  **oracle test only** as a stand-in for the not-yet-existent on-device tree. Do NOT route
  production through the fallback as live behavior — that is the test's job, not the
  learner's.

### Seam contract (`grow_tree_on_device` return type)
- **D-03:** Signature returns **`Result<Option<(Tree, DataPartition)>>`**. Default impl
  (and the off/unsupported path) returns **`Ok(None)`** meaning "I did not grow it";
  `train_inner`'s fork is `if let Some(t) = backend.grow_tree_on_device(..)? { return Ok(t) }`
  then falls through to the host path. `None` keeps the default path error-noise-free
  (preferred over a typed `Err(NotSupported)`).
- **D-03-RESOLVED (2026-06-28, post-research + user decision — Option A):** The literal
  `Result<Option<(Tree, DataPartition)>>` on the `Backend` trait is **INFEASIBLE** —
  `DataPartition` lives in `lgbm-treelearner`, which already depends on `lgbm-compute`
  (home of `Backend`), so naming it in a trait method creates a **circular crate
  dependency** (RESEARCH.md §"D-03 Feasibility", `[VERIFIED]` from 3 Cargo.toml edges).
  **Resolution (user-selected Option A):** the seam returns
  **`Result<Option<(Tree, P)>>`** where `P` is a **lower-crate partition payload** — the
  raw leaf-row index layout `DataPartition` already wraps, defined in `lgbm-core` or
  `lgbm-dataset` (or plain `Vec<i32>` + leaf bounds). `lgbm-treelearner` reconstructs
  `DataPartition` from `P` inside the `train_inner` fork. Add an `lgbm-model` dep to
  `lgbm-compute` so `Tree` is nameable (acyclic — `lgbm-model` does not depend on
  `lgbm-compute`). This honors D-01 (method on `Backend`), the `(tree, partition)` shape,
  the `Ok(None)` default, and the additive-only constraint. **Planner:** pick the concrete
  `P` type (recommend a named struct in `lgbm-core`/`lgbm-dataset` over bare `Vec<i32>`
  for clarity); the exact seam signature cannot be written until `P` is named.

### Tie-aware `default_left` comparator timing
- **D-04:** **Ship the tie-aware comparator NOW (dormant).** Slice 0 builds the full
  comparator — structure bit-exact PLUS tie-aware `default_left` acceptance — reusing the
  existing f32-vs-f64 near-tie logic at `kernel_parity.rs:1597`. It is dormant only in that
  no kernel yet produces a flip to exercise the tie branch. Phase 16 (Slice 2) merely
  ACTIVATES it against the on-device selection output — no new comparator work there. This
  satisfies Phase 14 SC#3 ("tie-aware scaffold") and Phase 16's "do NOT defer the tie-aware
  assert" simultaneously: the assert exists from Slice 0; it goes live in Slice 2.

### Env gate read placement
- **D-05:** Read `LGBM_CUDA_ON_DEVICE` **once at `SerialTreeLearner::new` construction**
  and cache `on_device_eligible = backend.on_device_growth_supported() && env_cuda_on_device()`.
  This **deliberately diverges** from `resident_eligible` (which recomputes in
  `train_inner`): on-device eligibility has no per-train, size-dependent input the way
  resident does (resident size-gates on `num_data`), so a per-train env re-read buys
  nothing and adds a syscall per tree. **Planner note:** do not "normalize" this back into
  `train_inner` to match the resident idiom — the divergence is intentional. It still ANDs
  the backend discriminator exactly as resident ANDs `resident_pool_supported()`, so
  `CpuBackend` (discriminator false) can never be eligible.

### Claude's Discretion
- Exact naming of the cached field (`on_device_eligible` suggested), the env-parse helper,
  and where `assert_on_device_tree_matches_cpu_anchor` lives (extend / sit beside
  `assert_gpu_tree_matches_cpu_anchor` in `oracle-harness/tests/learner_parity.rs`).
- Whether the comparator is a new fn or a tie-aware-extended generalization of the existing
  `assert_gpu_tree_matches_cpu_anchor` — implementation detail, left to planning.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` §"Milestone v1.1" → "Phase 14: Scaffold + Oracle (Slice 0)" —
  goal, 3 success criteria, non-negotiables, cubecl-0.10-gotcha checklist.
- `.planning/REQUIREMENTS.md` — ODL-01 (seam + discriminator) and ODL-02 (anchor-pinned
  tie-aware oracle); Out-of-Scope table (no megakernel, no CPU/ROCm routing change).

### Seam / discriminator templates (mirror these idioms exactly)
- `crates/lgbm-treelearner/src/learner.rs:680` — `resident_eligible` decide-once fork in
  `train_inner` (the routing-fork template; D-05 diverges by reading at `new` instead).
- `crates/lgbm-compute/src/lib.rs:898,1352,2134` — `prefers_host_partition` default-false
  trait-method discriminator (the additive-discriminator template). `resident_pool_supported`
  is the AND-gate precedent.
- `crates/lgbm-treelearner/src/resident_pool.rs:99,141` — `resident_eligible` + the
  `LGBM_RESIDENT_FORCE` env-read idiom.

### Oracle templates
- `crates/oracle-harness/tests/learner_parity.rs:2046` —
  `assert_gpu_tree_matches_cpu_anchor` (structure bit-exact + `ROCM_LEAF_VALUE_TOL=1e-5`
  leaf envelope) and `cpu_anchor_tree` (the deterministic cpu f64 anchor builder). The
  Slice-0 oracle extends/parallels this.
- `crates/oracle-harness/tests/kernel_parity.rs:1597-1603` — the f32-vs-f64 near-tie
  `default_left` acceptance logic to reuse for D-04's tie-aware branch.
- `crates/lgbm-compute/tests/rocm_cuda_mirror.rs` — `*_matches_cpu_anchor_within_tol`
  anchor-comparison precedent.

### Engineering memory (gotchas baked into this slice)
- def-f8u-01 (commit 1832206 / d82611b): never compare two nondeterministic GPU f32 paths
  to each other — always pin to the cpu f64 anchor. Directly motivates ODL-02 / D-04.
- cubecl-0.10 checklist (from ROADMAP Notes): no global barrier; `Atomic<i64>` broken;
  `wrapping_add` not an intrinsic; plane-sum ≤ plane width; `launch_unchecked` is unsafe.
  No kernels in this slice, but bake the checklist into the seam doc-comment for Slice 1.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `assert_gpu_tree_matches_cpu_anchor` + `cpu_anchor_tree` (`learner_parity.rs:2046,2083`):
  the oracle scaffold extends these — structure bit-exact + 1e-5 leaf envelope already done.
- `kernel_parity.rs:1597` near-tie `default_left` acceptance: lift into the tie-aware branch.
- `RocmBackend::with_resident(bool)` + `resident_pool_supported()`: the precedent for a
  backend-flag discriminator that defaults false and is force-overridable for tests.

### Established Patterns
- Decide-once eligibility (`resident_eligible` ANDs `resident_pool_supported()`): D-05
  mirrors the AND-gate but reads the env at `new` instead of `train_inner` (intentional).
- Default-false trait-method discriminators on ONE backend (`prefers_host_partition`),
  never a global switch — additive gating that leaves CPU/ROCm byte-unchanged.
- `LGBM_*` env idiom: `!matches!(env::var("LGBM_X").as_deref(), Ok("0"))` style
  (`autotune.rs:86`); `LGBM_CUDA_ON_DEVICE` is the inverse default (off unless `=1`).

### Integration Points
- `Backend` trait in `crates/lgbm-compute/src/lib.rs` — add `grow_tree_on_device`
  (default `Ok(None)`) + `on_device_growth_supported()` (default `false`).
- `SerialTreeLearner` in `crates/lgbm-treelearner/src/learner.rs` — cache
  `on_device_eligible` in `new`; add the `if let Some(t) = ..` fork at the top of
  `train_inner` ahead of the resident/host branches.
- Oracle in `crates/oracle-harness/tests/learner_parity.rs`.

</code_context>

<specifics>
## Specific Ideas

- Merge gate is the hard gate throughout: `raw_bin_train_matches_cpp_golden`,
  `learner_parity`, and the lgbm/treelearner/compute suites must be green AND byte-unchanged
  with `LGBM_CUDA_ON_DEVICE` unset.
- `GpuBackend<R>` override must still return the no-op (`Ok(None)`) / typed-error path so the
  default route is provably untouched (Phase 14 SC#2).

</specifics>

<deferred>
## Deferred Ideas

- Actual on-device continuous-feature growth, `hist_t**` subtraction-trick rotation, u64
  fixed-point / no-f64 kernel constraint → **Phase 15 (Slice 1, ODL-03/06/07)**.
- Cross-leaf on-device best-split selection; ACTIVATING the dormant tie-aware comparator
  against a real on-device flip → **Phase 16 (Slice 2, ODL-04)**.
- On-device data partition / leaf-index update → **Phase 17 (Slice 3, ODL-05)**.
- Categorical / bagging / GOSS / on-device score update → **Phase 18 (ODL-08/09/10)**.
- Kaggle `device_launches` A/B + default-on rollout → **Phase 19 (ODL-11/12)**.

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 14-scaffold-oracle-slice-0*
*Context gathered: 2026-06-28*
