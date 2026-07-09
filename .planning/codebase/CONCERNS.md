# Codebase Concerns

**Analysis Date:** 2026-07-09

## Tech Debt

**On-device (fully GPU-resident) CUDA tree learner — architecturally stuck, opt-in only:**
- Issue: `on_device_default()` hardcoded to `false` after 5 consecutive real-CUDA A/B failures (Phases 23-31, spikes 055-088). Root cause moved through several iterations — per-tree bin re-upload (fixed, largest single win, 67-78% of gap), host control-plane cost (fixed), and finally landed on a per-leaf device sync/readback floor (6044-9044 blocking readbacks/grow) that is architectural, not a bug.
- Files: `crates/lgbm-compute/src/lib.rs:1990-2058` (`on_device_default`, extensive inline history of every failed attempt), `crates/lgbm-compute/src/kernels/grow_driver.rs` (3500 lines, host-driven best-first grow loop), `crates/lgbm-treelearner/src/resident_pool.rs`.
- Impact: The on-device path is fully implemented, bit-exact/parity-tested, but 1.12-2.2x SLOWER than the host-orchestrated CUDA path in every measured shape. It is reachable only via `LGBM_CUDA_ON_DEVICE="1"` env opt-in; production default routes through the host-driven loop.
- Fix approach: Per spike-088/081 findings, would require a genuinely device-resident best-first grow loop that never blocks on a per-leaf readback — a larger rewrite than incremental patches; currently shelved.

**`grow_driver.rs` and `learner.rs` are oversized control-flow hubs:**
- Issue: `crates/lgbm-compute/src/kernels/grow_driver.rs` (3500 lines) and `crates/lgbm-treelearner/src/learner.rs` (5100 lines, the largest file in the workspace) mix kernel dispatch, host bookkeeping, environment-variable-gated A/B branches, and profiling instrumentation in single files.
- Files: `crates/lgbm-treelearner/src/learner.rs`, `crates/lgbm-compute/src/kernels/grow_driver.rs`, `crates/lgbm-compute/src/kernels/best_split.rs` (4185 lines), `crates/lgbm-compute/src/kernels/split.rs` (4090 lines), `crates/lgbm-compute/src/lib.rs` (4773 lines).
- Impact: High-risk surface for regressions; any change touching the grow loop must be cross-checked against many env-flag-gated paths (`LGBM_SPLIT_2LANE`, `LGBM_CUDA_ON_DEVICE`, `LGBM_SCAN_CUBEDIM`, `LGBM_BENCH_SWEEP`, etc.) that all live inline.
- Fix approach: No active plan; acceptable given the parity-test coverage, but any future refactor should preserve every A/B env-gate's semantics (each is load-bearing for a completed spike).

**`HistArena::swap` slot-aliasing fragility (WR-01):**
- Issue: A previously identified latent slot-aliasing bug in histogram-arena swap logic; test coverage for the multi-leaf on-device grow loop specifically targets this class of bug (tie-break pick using stale/aliased buffers).
- Files: `crates/oracle-harness/tests/on_device_tie_break_parity.rs`, `crates/oracle-harness/tests/learner_parity.rs:2157-2802` (multiple `WR-01` regression comments), `crates/oracle-harness/tests/primitive_parity.rs:333`, `crates/oracle-harness/tests/objective_parity_rank.rs:14` (CUDA hessian-buffer aliasing, `_GlobalMemory` variant).
- Impact: Requires careful review of any change to histogram-pool reuse (`HistogramPool`, `resident_pool.rs`) or shared/global-memory hessian buffers on the CUDA kernel side; regressions here are silent numerical corruption, not crashes.
- Fix approach: Guarded today by dedicated parity tests (`objective_parity_rank.rs`, `on_device_tie_break_parity.rs`); no structural fix beyond the current test net.

**Categorical-feature GPU kernel seam is a stub (Phase 22 deferred):**
- Issue: Metadata plumbing and `_GlobalMemory` kernel variants for categorical features are allocated/reserved but unused.
- Files: `crates/lgbm-compute/src/kernels/column_data.rs:28`, `crates/lgbm-compute/src/kernels/best_split.rs:1126,1194`.
- Impact: Categorical-split GPU support is incomplete; falls back to whatever the continuous-only kernel path does.
- Fix approach: Tracked inline as "Phase 22 / v2 QGD-02" seam, not currently scheduled.

**Widespread `#[allow(clippy::too_many_arguments)]`:**
- Issue: ~25+ kernel/dispatch functions across `lgbm-compute` (data_partition.rs, predict.rs, objective_rank.rs, metric_pointwise.rs) and `lgbm-boosting/gbdt.rs` silence the too-many-arguments lint rather than group parameters into structs.
- Files: `crates/lgbm-compute/src/kernels/data_partition.rs`, `objective_rank.rs`, `predict.rs`, `metric_pointwise.rs`; `crates/lgbm-boosting/src/gbdt.rs:2033,2094`.
- Impact: Function signatures are hard to read/extend safely; mirrors the C++ reference's `Tree::Split`-style wide-parameter-list convention (see CONVENTIONS.md), so this is partly intentional parity with the port target, but still a maintainability cost for GPU kernel signatures specifically (`#[cube]` functions cannot easily take struct params in cubecl 0.10).
- Fix approach: Not urgent; would require cubecl struct-parameter support to fix without losing kernel semantics.

## Known Bugs

**No open/tracked bugs found in current source.** All previously identified defects referenced in project memory (DEF-07-02, DEF-08-OOS-01, DEF-f8u-01 flaky resident HIP test, hip split parity near-tie) are marked RESOLVED or pre-existing/parked in commit history; none have live `FIXME`/`XXX` markers in `crates/`.

**Sanctioned but green-reporting-nothing tests exist (WR-02 stale-note pattern):**
- Symptoms: Several tests in `oracle-harness` assert nothing and exist only as stale documentation notes, explicitly `#[ignore]`d "so it does not report green while covering nothing."
- Files: `crates/oracle-harness/tests/learner_parity.rs:312,322,430,784`, `crates/oracle-harness/tests/predict_parity.rs:235`.
- Trigger: Anyone running `cargo test --workspace -- --ignored` without reading the surrounding comments could mistake these for real coverage.
- Workaround: Comments are clear; the `#[ignore]` itself is the safeguard the codebase relies on.

## Security Considerations

**168 `unsafe` blocks concentrated in `lgbm-compute` (24 files):**
- Risk: `unsafe` is used heavily in kernel code — `histogram.rs` (25), `primitives.rs` (22), `split.rs` (21), `best_split.rs` (12), `tree.rs` (10), `predict.rs` (8), `objective_binary.rs` (8) — almost certainly `cubecl`'s `#[cube]` macro-generated GPU kernel code plus manual pointer/buffer indexing for performance.
- Files: `crates/lgbm-compute/src/kernels/*.rs` (see counts above); single isolated `unsafe` occurrences also exist in `crates/lgbm-dataset/src` and `crates/lgbm-treelearner/src`.
- Current mitigation: Extensive bit-exact/parity test suite (`oracle-harness` crate, 3000+ line parity test files) catches numerical divergence that unsafe misuse would likely produce; no memory-safety audit tooling (Miri, ASan) referenced in CI config found.
- Recommendations: If not already done, run `cargo miri test` on the CPU-only feature set periodically to catch UB in the manual-unsafe (non-macro-generated) portions; the two single-`unsafe` occurrences outside `lgbm-compute` (`lgbm-dataset`, `lgbm-treelearner`) are small enough to be worth a manual safety-comment audit.

**No secrets or credential files detected in tracked source** (Kaggle token lives at `~/.kaggle/access_token`, outside the repo, per `AGENTS.md`).

## Performance Bottlenecks

**Single-threaded `DataPartition::split` remains the dominant CPU-vs-C++ gap:**
- Problem: At scale, tall-narrow shapes (e.g. 500k rows × 50 features) run ~1.7-2.2x slower than C++ LightGBM; the root cause traced to `DataPartition::split` not being parallelized the way C++'s reference implementation is, despite several rayon/cubecl-cpu parallelization spikes.
- Files: `crates/lgbm-treelearner/src/data_partition.rs`.
- Cause: Per project memory, direct parallelization attempts (spike-026, cubecl-cpu scan+scatter) hit a shared-DRAM-bandwidth wall rather than a contention wall; wins only appear in cache-resident regimes (~100k balanced), and regress on skewed/large inputs. The "fused-gather partition" win (narrower route-scratch + fused bin-gather) is shipped and reduces but does not eliminate the gap.
- Improvement path: No further-open lever per current memory (`partition-parallel-null.md`); this is treated as a closed/accepted gap, not an active TODO.

**GPU (ROCm/CUDA host-orchestrated) path loses to 16-core CPU on the local (spoofed 8-CU APU) hardware in most shapes.**
- Problem: `RocmBackend`/`CudaBackend` crossover vs CPU has been "erased" at the current hardware tier; real discrete-CUDA numbers (via Kaggle) show a narrower but still-present on-device-vs-host gap (1.12-2.2x, see Tech Debt above).
- Files: `crates/lgbm-compute/src/lib.rs` (backend cascade `rocm > cuda > wgpu > cpu`, see `crates/lgbm/Cargo.toml` feature comments), `crates/lgbm-compute/src/kernels/grow_driver.rs`.
- Cause: Local hardware is a spoofed 8-CU integrated APU (not real gfx1100), so all local GPU perf numbers are confounded; occupancy tuning calibrated for 96 CUs is ~12x off for the actual hardware. Real perf validation requires Kaggle-hosted CUDA runs (documented workflow, see `xtask/` and `AGENTS.md` Kaggle token reference).
- Improvement path: Any future GPU perf work MUST be validated against real Kaggle CUDA hardware, not the local APU, per project memory (`rocm-gfx1100-available.md`).

## Fragile Areas

**Env-var-gated A/B branches scattered across 20 files:**
- Files: `crates/lgbm-compute/src/lib.rs`, `crates/lgbm-compute/src/kernels/{histogram,split,grow_driver,autotune}.rs`, `crates/lgbm-compute/src/fusion_prof.rs`, `crates/lgbm-treelearner/src/{phase_prof,resident_pool,data_partition,learner}.rs`, `crates/lgbm/src/booster.rs`, plus numerous `examples/spike*.rs` and `oracle-harness/tests/*` files.
- Why fragile: Each env var (`LGBM_CUDA_ON_DEVICE`, `LGBM_SPLIT_2LANE`, `LGBM_SCAN_CUBEDIM`, `LGBM_BENCH_SWEEP`, `LGBM_UNIFIED_BFS_THRESHOLD`, `LGBM_UNIFIED_SUBSCAN_THRESHOLD`, `LGBM_PHASE_PROF`, etc.) toggles a production code path that must remain bit-exact regardless of setting; there is no central registry of these flags, so a new contributor can easily miss one when refactoring the function it gates.
- Safe modification: Grep for the specific env var name across the whole workspace (including `examples/` and `tests/`) before changing any function that reads one, since spike-derived thresholds (e.g. core-count-derived clamp formulas in `data_partition.rs`) encode nontrivial tuning history in the surrounding comments — do not delete these comments when touching the code.
- Test coverage: Each flag generally has a dedicated A/B test/example (`cuda_on_device.rs`, `phase30_ab.rs`, `phase31_ab.rs`, `spike079_bin_hoist_ab.rs`) but there is no single "flag inventory" test that fails if a flag is silently dropped.

**`unwrap()`/`expect()`-heavy production code (no `Result` propagation in hot paths):**
- Files: highest counts by crate — `lgbm-compute/src` (117 unwrap + 54 expect + 5 panic!), `lgbm-objective/src` (67 unwrap), `lgbm/src` (41 unwrap + 16 expect), `lgbm-model/src` (51 unwrap + 15 expect + 2 panic!), `lgbm-metric/src` (54 unwrap), `lgbm-dataset/src` (34 unwrap), `lgbm-treelearner/src` (28 unwrap + 29 expect + 4 panic!), `lgbm-boosting/src` (27 unwrap + 31 expect).
- Why fragile: A large fraction of these are almost certainly on `OnceLock`/`Mutex` locks, slice-index invariants, and cubecl buffer reads that are logically infallible given upstream validation — but the sheer density (500+ unwrap/expect sites workspace-wide) means malformed input (e.g. adversarial config combinations) is more likely to panic than return a structured `thiserror` error, despite the project's stated convention of using `thiserror` at library boundaries (per CLAUDE.md constraints).
- Safe modification: When adding new I/O-adjacent code (config parsing, dataset loading, Python FFI boundary), prefer `Result` + `thiserror` over `unwrap`/`expect`, consistent with the stated project convention; this is a project-wide gap rather than a single hotspot.
- Test coverage: Parity/A/B test suites (`oracle-harness`) validate numerical behavior on known-good inputs but do not appear to fuzz for panic-inducing malformed configs.

## Scaling Limits

**Histogram pool sized by `num_leaves × total_bins`:**
- Current capacity: Memory-bound by the fixed-size histogram pool used with the subtraction trick (mirrors C++ design).
- Limit: Very high `max_bin` combined with very high `num_leaves` grows pool memory linearly in both; not independently re-verified in this pass but inherited directly from the ported C++ design (see ARCHITECTURE.md "Histogram pool memory" constraint) — same limit class as upstream LightGBM.
- Scaling path: No Rust-specific scaling work identified beyond the flattened-arena + cross-tree-reuse optimization already shipped (spikes 010/012, ~7% win, bit-exact).

## Dependencies at Risk

**`cubecl = "0.10.0"` is a young, fast-moving compute crate:**
- Risk: The project's entire GPU/CPU-parallel-kernel abstraction sits on `cubecl` 0.10; project memory notes multiple version-specific API details (LDS `SharedMemory`/`sync_cube`/shared-atomics API, `CubeCount::Dynamic` behavior as a hidden blocking readback) that were discovered empirically and are load-bearing for correctness/perf, not documented assumptions.
- Impact: A `cubecl` version bump could silently change readback/sync semantics (as already observed with `CubeCount::Dynamic`), invalidating tuned thresholds or reintroducing the on-device sync-floor problem in a different form.
- Migration plan: None active; workspace pins `cubecl = "0.10.0"` in the root `Cargo.toml` `[workspace.dependencies]`. Any upgrade should re-run the full `oracle-harness` parity suite plus the real-CUDA Kaggle A/B harness before merging.

## Missing Critical Features

**Categorical-feature GPU kernel support** — see Tech Debt above; continuous-only kernels are production-complete, categorical `_GlobalMemory` variants are stubbed.

**Fully GPU-resident (no host round-trip) best-first grow loop** — see Tech Debt above; currently blocked on an architectural sync-floor problem, not scheduled.

## Test Coverage Gaps

**On-device CUDA path validated only against a spoofed local APU + intermittent Kaggle CUDA runs:**
- What's not tested: Continuous/automated real-discrete-GPU regression testing; all "real-CUDA" A/B verdicts in project history are one-off Kaggle notebook runs, not a repeatable CI gate.
- Files: `crates/lgbm-compute/tests/cuda_on_device.rs`, `crates/oracle-harness/tests/on_device_e2e_ab_corpus.rs` (explicitly `#[ignore]`d, run-once workflow).
- Risk: Regressions to on-device perf/correctness on real hardware could go undetected between manual Kaggle runs; the opt-in-only status (`on_device_default() = false`) partly mitigates blast radius since it's not on the default path.
- Priority: Low (feature is opt-in and known-slow; not user-facing by default), but High if the flip to default-on is ever attempted without restoring a repeatable real-GPU gate.

**On-device kernel goldens are host re-transcriptions, not captured from a compiled `lib_lightgbm`:**
- What's not tested: Phase 17/18 partition/tree/predict goldens for on-device kernels prove internal transcription self-consistency, not fidelity to the actual C++ reference binary.
- Files: `crates/oracle-harness/tests/` (on-device partition/tree/predict parity suites — see `on-device-kernel-goldens-are-retranscriptions` in project memory for full context).
- Risk: A latent divergence between the hand-transcribed golden and true C++ behavior would not be caught by these specific tests; the CPU f64-fold anchor tests (bit-exact vs real `lib_lightgbm` 4.6) are the ones with genuine external-reference fidelity.
- Priority: Medium — mitigated by the CPU anchor being the hard merge gate; on-device correctness is anchored transitively (device vs CPU-f64, not device vs real C++) per the `def-f8u-01` "never compare two nondeterministic GPU f32 paths to each other at 1e-6" lesson.

---

*Concerns audit: 2026-07-09*
