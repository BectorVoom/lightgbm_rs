# Phase 23: Perf-Validation + Default-On Rollout (DoD) - Pattern Map

**Mapped:** 2026-07-02
**Files analyzed:** 7 (4 Rust source edits, 1 new Rust test file, 2 new Python/evidence artifacts)
**Analogs found:** 7 / 7 (all in-tree; this is a routing-flip + measurement phase, no novel algorithms)

> This is a **routing/config-flip + measurement** phase. Every seam already exists; the net-new
> code is a tri-state resolver, a `cfg!(feature="cuda")` helper, on-device launch instrumentation
> (L-1), and a committed Kaggle A/B harness. "Copy the pattern from X" here means "mirror the exact
> existing idiom at X" — the risk is **coherence across 3 toggle sites**, not new logic.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/lib.rs` (tri-state resolver + `on_device_default()`) | config | transform (env→bool) | `split_2lane_enabled()` @ `lib.rs:1336` + `runtime.rs` cfg aliases | exact (in-file idiom) |
| `crates/lgbm-treelearner/src/learner.rs` (L-2 reconcile `cuda_on_device_env`) | config | transform (env→bool) | `cuda_on_device_enabled()` @ `lib.rs:1324` (the source of truth to defer to) | exact |
| `crates/lgbm-compute/src/kernels/grow_driver.rs` (L-1 launch counting) | utility/driver | event-driven (per-launch bump) | `learner.rs:2281` host `phase_prof::bump(&BUILD_RESIDENT_CNT)` | role-match (cross-crate — see L-1 landmine) |
| `crates/lgbm-treelearner/src/phase_prof.rs` (surface on-device counter) | utility | batch (counter→stderr line) | `BUILD_RESIDENT_CNT` decl @ `:79` + COUNTS emit @ `:194-199` | exact (in-file idiom) |
| `crates/lgbm-compute/tests/*.rs` (NEW: tri-state + default unit tests) | test | request-response | `crates/lgbm-compute/tests/capability.rs` / `split_info.rs` | role-match |
| `.planning/phases/23-.../ab_harness.py` (NEW: committed Kaggle A/B) | script | batch (train→parse→emit) | `continue_benchmark.py` (repo root) | exact (extends the family) |
| `.planning/phases/23-.../results.{md,json}` (NEW: DoD evidence artifact) | config/data | batch (emitted output) | `kaggle_out/*` capture + `poll_kaggle.sh` | role-match |

## Pattern Assignments

### `crates/lgbm-compute/src/lib.rs` — tri-state resolver + CUDA-only default (D-01/D-02/D-03)

**Analog:** `split_2lane_enabled()` @ `lib.rs:1336-1340` (OnceLock env-parse idiom) and the current
`cuda_on_device_enabled()` @ `lib.rs:1324-1328` (the binary parse being replaced).

**OnceLock env-parse idiom to mirror** (`lib.rs:1324-1328`, current binary form — REPLACE this body,
KEEP the `pub fn cuda_on_device_enabled() -> bool` name + signature so `score_updater.rs:75`,
`on_device_growth_supported()` @ `:1358`/`:2288` need no call-site change):
```rust
#[must_use]
pub fn cuda_on_device_enabled() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LGBM_CUDA_ON_DEVICE").map(|v| v == "1").unwrap_or(false))
}
```

**Sibling OnceLock idiom** (`lib.rs:1336-1340`) — the exact read-once shape to reuse for the new
tri-state cache (note: `#[cfg(feature = "cpu")]`-gated; the tri-state resolver must NOT be cpu-gated
since cuda/rocm builds call it):
```rust
#[cfg(feature = "cpu")]
fn split_2lane_enabled() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LGBM_SPLIT_2LANE").map(|v| v == "1").unwrap_or(false))
}
```

**cfg-gated runtime-binding precedent for D-03** (`runtime.rs:137-138, 184-185`) — this is the exact
"runtime binding = cargo feature" idiom `on_device_default()` mirrors (a `-F cuda` build has the
`cuda` cfg set; `cpu`/`rocm` builds do not):
```rust
#[cfg(feature = "cpu")]
pub type ActiveRuntime = cubecl::cpu::CpuRuntime;
// ...
#[cfg(feature = "cuda")]
pub type CudaRuntime = cubecl::cuda::CudaRuntime;
```

**Target shape** (from RESEARCH §Pattern 1 — tri-state override cache + compile-time default helper,
resolved at the seam so the default is a *routing* property not an *env* property, D-02):
```rust
fn cuda_on_device_override() -> Option<bool> { /* OnceLock<Option<bool>>: "1"=>Some(true), "0"=>Some(false), _=>None */ }
#[must_use] fn on_device_default() -> bool { cfg!(feature = "cuda") } // D-03; ROCm/CPU => false => SC-4
#[must_use] pub fn cuda_on_device_enabled() -> bool { cuda_on_device_override().unwrap_or_else(on_device_default) }
```

**Pitfall P-3 (dual-feature build):** `impl<R: cubecl::Runtime> Backend for GpuBackend<R>` (`lib.rs:2270`)
is ONE impl for cuda/rocm/wgpu; `cfg!(feature="cuda")` is global, so a `-F cuda,rocm` build would
mis-default ROCm on. Mono-feature Kaggle/ROCm builds are correct; add a code comment asserting
mono-feature, or use the per-runtime sealed-const keyed on concrete `R` (RESEARCH Alternatives table)
if the dual-feature guarantee is wanted.

---

### `crates/lgbm-treelearner/src/learner.rs` — reconcile the duplicate parse (L-2)

**Analog / source of truth:** `lgbm_compute::cuda_on_device_enabled()` (already the resolved gate).

**The duplicate to delete/fold** (`learner.rs:471-473`):
```rust
fn cuda_on_device_env() -> bool {
    matches!(std::env::var("LGBM_CUDA_ON_DEVICE").as_deref(), Ok("1"))
}
```

**Double-gate call site to simplify** (`learner.rs:539`) — `on_device_growth_supported()` already
returns the resolved `cuda_on_device_enabled()`, so the `&& cuda_on_device_env()` is a redundant
SECOND parse that will disagree with the new tri-state (`"0"` off-switch would work in compute but
the learner's unset-under-cuda would still read false):
```rust
on_device_eligible: backend.on_device_growth_supported() && cuda_on_device_env(),
```
**Recommendation (RESEARCH §Pattern 2):** drop the `&& cuda_on_device_env()` so the base gate is
`backend.on_device_growth_supported()` alone (single source of truth). Also update the mirror at
`learner.rs:3844` (`refresh_on_device_eligibility`). The `on_device_eligible_gate()` categorical+quantized
negation (`:486-492`) is unchanged (D-10 host-fallback).

**Early-return fork that skips host counters** (`learner.rs:817-826`) — the on-device path returns
BEFORE the host build/subtract/scan counter bumps, which is exactly why L-1 is needed:
```rust
if let Some((tree, payload)) = self.backend.grow_tree_on_device(
    gradients, hessians, &grow_features, self.num_leaves, self.max_depth,
)? {
    let part = DataPartition::from_payload(payload);
    return Ok((tree, Vec::new(), ColSamplerTrace::default(), part));
}
```

---

### `crates/lgbm-compute/src/kernels/grow_driver.rs` — on-device launch instrumentation (L-1, SC-2)

**Analog (host counter bump):** `learner.rs:2281` (`phase_prof::bump(&BUILD_RESIDENT_CNT)` right
before the per-leaf histogram build); companions at `:1873` (SUBTRACT), `:2060`/`:2742` (SCAN),
`:2725` (FUSED). The host pattern is one `bump()` per real device launch:
```rust
crate::phase_prof::bump(&crate::phase_prof::BUILD_RESIDENT_CNT);
```

**The on-device launch sites to instrument** (each is a real `*_on` kernel dispatch):
- `build_leaf_hist()` @ `grow_driver.rs:388` → `construct_histograms_f64_on` @ `:409` (called at root `:601`, smaller child `:898`) — the BUILD analog.
- `subtract_histograms_f64_on` @ `grow_driver.rs:903` — the SUBTRACT analog (larger child).
- `scan_leaf()` @ `grow_driver.rs:432` → `find_best_split_f64_on` @ `:479` (root `:605`, children `:931`) — the SCAN analog.

**⚠️ L-1 CROSS-CRATE LANDMINE (crate-cycle constraint):** `phase_prof` lives in **lgbm-treelearner**,
but `grow_driver.rs` lives in **lgbm-compute**, which is BELOW lgbm-treelearner in the dependency
graph (memory: `on-device-driver-crate-cycle-constraint`). `grow_driver` **cannot** call
`crate::phase_prof::bump` the way `learner.rs` does. The counters must either (a) be defined in a
lgbm-compute-local module and re-surfaced, or (b) the driver returns a launch count in its payload
that the learner bumps after the early-return at `learner.rs:823`. Option (b) keeps all `AtomicU64`
counters in `phase_prof` (single COUNTS emitter) and respects the crate boundary — recommend the
planner evaluate both. RESEARCH Open Q2 recommends a dedicated `on_device=` sub-field folded into the
`device_launches=` total so the harness regex stays stable.

---

### `crates/lgbm-treelearner/src/phase_prof.rs` — surface the on-device counter in COUNTS

**Analog:** the `BUILD_RESIDENT_CNT` family — declaration, `bump()`, and the COUNTS emit block are
all in this one file; mirror all three for any new counter.

**Counter declaration idiom** (`phase_prof.rs:79-82`):
```rust
pub static BUILD_RESIDENT_CNT: AtomicU64 = AtomicU64::new(0);
pub static SUBTRACT_RESIDENT_CNT: AtomicU64 = AtomicU64::new(0);
pub static SCAN_RESIDENT_CNT: AtomicU64 = AtomicU64::new(0);
pub static FUSED_CNT: AtomicU64 = AtomicU64::new(0);
```

**Inert-unless-enabled bump** (`phase_prof.rs:86-91`) — parity-neutral, zero-overhead in the merge gate:
```rust
#[inline]
pub fn bump(counter: &AtomicU64) {
    if enabled() { counter.fetch_add(1, Ordering::Relaxed); }
}
```

**COUNTS emit block to extend** (`phase_prof.rs:190-199`) — the exact line the harness regex parses;
keep `device_launches=<total>` first-field so `COUNTS = re.compile(...)` (RESEARCH Code Examples) is
unchanged; add an `on_device=` sub-field inside the parenthesized breakdown for clarity:
```rust
let bld_cnt = BUILD_RESIDENT_CNT.swap(0, Ordering::Relaxed);
// ... sub/scn/fus ...
if bld_cnt + sub_cnt + scn_cnt + fus_cnt > 0 {
    let launches = bld_cnt + sub_cnt + scn_cnt + fus_cnt;
    eprintln!(
        "[phase_prof:{label}] COUNTS: device_launches={launches} (build_resident={bld_cnt} subtract_resident={sub_cnt} scan_resident={scn_cnt} fused={fus_cnt}) | scan_roundtrips(syncs)={scn_cnt}"
    );
}
```
**Note the `> 0` guard (`:194`)** — this is why an uninstrumented on-device train suppresses the whole
COUNTS line (P-2); a non-zero on-device counter is what makes SC-2 measurable.

---

### `crates/lgbm-compute/tests/*.rs` — NEW unit tests (tri-state + default)

**Analog:** `crates/lgbm-compute/tests/split_info.rs` / `capability.rs` (integration-test file layout
under `crates/lgbm-compute/tests/`). For the resolver, an in-`lib.rs` `#[cfg(test)] mod` is also valid.

**Coverage (RESEARCH Test Map + Wave-0):**
- `cuda_on_device_override`: unset→`None`, `"0"`→`Some(false)`, `"1"`→`Some(true)`, malformed→`None`.
- `on_device_default()`: asserts `false` under the default `cpu` build (SC-4 anchor); `false` under `rocm`.
- **Pitfall P-1:** OnceLock read-once — a test cannot flip env mid-process; set env before first read
  or use the existing driver-override seams (`Gbdt::set_boosting_on_cuda`), mirroring how
  `resident_score_ab.rs:168` skips the default-check when `=1`.

---

### `.planning/phases/23-.../ab_harness.py` — NEW committed Kaggle A/B harness (D-07)

**Analog:** `continue_benchmark.py` (repo root) — the full clone→maturin `-F cuda`→install→official-lgb
→`make_classification`→timed A/B flow, authenticated as `boomvector`. Extend it; do not rewrite.

**Setup + build flow to reuse** (`continue_benchmark.py:11-38`):
```python
run("pip install -U numpy scipy scikit-learn")
run("git clone https://github.com/BectorVoom/lightgbm_rs.git")  # or checkout/pull
run("pip install maturin"); run("rm -rf lightgbm_rs/target/wheels/")
run("cd lightgbm_rs/crates/lgbm-python && maturin build --release -F cuda -j 2")
run("pip install $(ls lightgbm_rs/target/wheels/*.whl | head -n 1) --force-reinstall")
# official lightgbm from source with USE_CUDA=ON (:37) for the context ratio
```

**Timed-arm skeleton to generalize** (`continue_benchmark.py:61-76`) — wrap in the D-05 matrix
(3 runs × 2 shapes {(500000,50),(100000,500)} × 2 paths), take medians:
```python
start = time.time(); model.fit(X, y); elapsed = time.time() - start
```

**Net-new vs the analog (from RESEARCH Code Examples):**
- Set per-arm env: host arm `LGBM_CUDA_ON_DEVICE=0` (new tri-state off-switch), on-device arm `=1`; both `LGBM_PHASE_PROF=1`.
- Parse the COUNTS line from stderr: `re.compile(r"\[phase_prof:\w+\] COUNTS: device_launches=(?P<launches>\d+) \(build_resident=(?P<build>\d+) subtract_resident=(?P<sub>\d+) scan_resident=(?P<scan>\d+) fused=(?P<fused>\d+)\)")`; divide by `n_estimators=100`.
- D-11 parity assertion (hard gate): `max(|p_on − p_host|) <= 1e-6` on `raw_score=True` predictions over the actual dataset; anchor to host-CUDA/official, NOT on-device-f32-vs-host-f32 unanchored (def-f8u-01).
- PASS iff BOTH shapes: `median(on_device) <= 1.05 * median(host_cuda)` (D-04).
- **Security:** no Kaggle tokens embedded (auth stays out-of-band in the CLI config); pin repo `BectorVoom/lightgbm_rs`; official lgb `--no-binary`.

**Poll/output analog:** `poll_kaggle.sh` (status loop on `boomvector/lgb-rs-cuda-bench` →
`kaggle kernels output ... -p kaggle_out`).

---

### `.planning/phases/23-.../results.{md,json}` — NEW DoD evidence artifact (D-07/D-08)

**Analog:** `kaggle_out/*` (raw captured stderr/stdout) — the harness emits structured JSON (raw runs
+ medians + ratios + launches + parity max_abs) and an MD verdict table. Report set (D-08): on-device
vs host-CUDA (the ≤5% gate), context ratio vs official (~4.46× pre-on-device bar), and
`device_launches/tree` vs the **8,570 / 100-trees** baseline (SC-2).

## Shared Patterns

### OnceLock read-once env gate (parity-neutral, default-safe)
**Source:** `lib.rs:1324` (`cuda_on_device_enabled`), `lib.rs:1336` (`split_2lane_enabled`), `phase_prof.rs:112` (`enabled`).
**Apply to:** the tri-state resolver + `on_device_override` cache.
**Invariant:** read ONCE per process; under the default `cpu` feature the env-unset lane must resolve
to `false` so the merge gate is byte-identical (SC-4). Tests cannot flip env mid-process (P-1) — use
the driver-override seams instead.

### cfg-gated runtime binding (no runtime device sniffing) — D-03
**Source:** `runtime.rs:137,168,178,185,208` (`#[cfg(feature = "…")]` type aliases + client ctors).
**Apply to:** `on_device_default() = cfg!(feature = "cuda")`.
**Invariant:** device class is a compile-time cargo-feature property, never a runtime `device_type`
string check. ROCm/CPU builds have `cfg!(feature="cuda")==false` ⇒ default-off (SC-4 upheld structurally).

### phase_prof counter → COUNTS line (event-driven measurement)
**Source:** `phase_prof.rs:79-91` (decl + `bump`), `phase_prof.rs:190-199` (emit + `> 0` guard).
**Apply to:** L-1 on-device launch counter (subject to the crate-cycle constraint above).
**Invariant:** inert unless `LGBM_PHASE_PROF=1`; keep `device_launches=<total>` as the first COUNTS
field so the harness regex is stable; the `> 0` guard means an uninstrumented path emits nothing.

### boosting-layer toggle inherits the resolver for free — D-02
**Source:** `score_updater.rs:75` (`boosting_on_cuda: lgbm_compute::cuda_on_device_enabled()`),
`gbdt.rs:1490` (`set_boosting_on_cuda` driver/test override).
**Apply to:** verification only — because the resolver name/signature is unchanged, the boosting layer
automatically picks up tri-state + CUDA-only default. Existing tests `resident_score_ab.rs:187` /
`score_updater_parity.rs:108` assert "unset ⇒ OFF" and STILL PASS under the cpu test build
(`on_device_default()==false`). No edit needed here — this is the SC-4 coherence check.

## No Analog Found

None. Every seam and every measurement primitive already exists in-tree; this phase is
reconciliation + instrumentation + a committed harness, not new subsystems.

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/{lib.rs,runtime.rs,kernels/grow_driver.rs}`,
`crates/lgbm-treelearner/src/{learner.rs,phase_prof.rs}`, `crates/lgbm-boosting/src/{gbdt.rs,score_updater.rs}`,
`crates/lgbm-compute/tests/`, `crates/oracle-harness/tests/`, repo-root `benchmark.py`/`continue_benchmark.py`/`poll_kaggle.sh`.
**Files scanned:** ~11 (line ranges verified against the current working tree).
**Pattern extraction date:** 2026-07-02
```
