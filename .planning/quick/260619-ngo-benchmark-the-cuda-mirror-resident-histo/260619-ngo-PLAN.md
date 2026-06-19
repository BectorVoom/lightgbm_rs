---
phase: 260619-ngo
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-compute/examples/mirror_vs_lds.rs
autonomous: true
requirements: [NGO-01]
must_haves:
  truths:
    - "On gfx1100, a single rocm-gated example A/Bs the mirror resident kernel vs the wired LDS resident kernel, both fed the SAME resident bins + SAME leaf rows, both returning a comparable slot_len RAW f64 histogram."
    - "The two RAW histograms agree within the f32-atomic envelope (ABS 5e-6 / REL 1e-5) as a same-input sanity check — proving the A/B compares correct computations."
    - "The LDS timing includes its production host-side grad/hess gather AND reports a gather-excluded sub-number; the resident bins upload is OUTSIDE both timed loops (excluded per 260619-mwr)."
    - "Warm-vs-cold honored: >=2 discarded warm-ups, median of >=5 timed launches per variant."
    - "The SUMMARY contains the real gfx1100 A/B table (mirror vs LDS median ms per size/bin + speedup) and an evidence-based wiring recommendation (default / gated / primitive)."
  artifacts:
    - path: "crates/lgbm-compute/examples/mirror_vs_lds.rs"
      provides: "rocm-gated A/B micro-bench of the two resident histogram kernels"
      contains: "construct_histograms_cuda_mirror_resident_on"
  key_links:
    - from: "crates/lgbm-compute/examples/mirror_vs_lds.rs"
      to: "lgbm_compute::kernels::histogram::build_leaf_histograms_resident_f32_on"
      via: "incumbent LDS launcher, timed per-leaf"
      pattern: "build_leaf_histograms_resident_f32_on"
    - from: "crates/lgbm-compute/examples/mirror_vs_lds.rs"
      to: "lgbm_compute::kernels::histogram::construct_histograms_cuda_mirror_resident_on"
      via: "candidate mirror launcher, timed per-leaf"
      pattern: "construct_histograms_cuda_mirror_resident_on"
---

<objective>
Benchmark the CUDA-mirror resident histogram kernel against the existing WIRED LDS
resident build kernel on the local gfx1100, then recommend a wiring disposition. This is
MEASUREMENT + RECOMMENDATION ONLY — the user chose "benchmark first, then decide". Do NOT
wire either kernel into lib.rs / the learner / the wired path in this task.

Purpose: produce the real A/B numbers that decide the follow-up wiring (faster→default;
slower→gated-opt-in or leave-as-primitive), under the project's honesty mandate (report
flat/negative results plainly).

Output: a rocm-gated example `crates/lgbm-compute/examples/mirror_vs_lds.rs` and a SUMMARY
carrying the gfx1100 A/B table + an explicit evidence-based wiring recommendation.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@crates/lgbm-compute/examples/cuda_mirror_overhead.rs
@crates/lgbm-compute/tests/rocm_cuda_mirror.rs

# Both launcher signatures + the production LDS grad/hess host-gather pattern:
@crates/lgbm-compute/src/kernels/histogram.rs
@crates/lgbm-compute/src/lib.rs

# Load-bearing benchmark discipline (warm-vs-cold; GPU-hist regime meaning):
@.claude/skills/spike-findings-lightgbm_rs/SKILL.md
</context>

<interface_contract>
Both kernels are rocm-gated, take a pre-uploaded resident bin `Handle`, and return a
`Vec<f64>` of `slot_len` RAW concatenated histogram cells — directly comparable. They
differ ONLY in their native grad/hess input form; feed each its native form computing the
SAME leaf histogram from the SAME resident bins + SAME leaf rows.

INCUMBENT (LDS, the wired path) — histogram.rs ~811:
  build_leaf_histograms_resident_f32_on(
    client, resident_bins: Handle, num_features, num_data,
    slot_off: &[usize] /* num_features entries, NO sentinel */, slot_len,
    leaf_rows: &[u32] /* the leaf's rows */,
    gradients: &[f32], hessians: &[f32] /* LEAF-LENGTH, gathered HOST-SIDE: ord_g[k]=grad[leaf_rows[k]] */,
  ) -> Result<Vec<f64>>
  Production cost (per lib.rs:497) GATHERS ord_g/ord_h host-side to leaf length before the
  call — that gather is part of the real per-leaf LDS cost.

CANDIDATE (mirror) — histogram.rs ~1250:
  construct_histograms_cuda_mirror_resident_on(
    client, resident_bins: Handle, num_data, num_features,
    data_indices: &[u32] /* == the leaf's rows */,
    grad: &[f32], hess: &[f32] /* FULL-CORPUS, length num_data; gathered IN-KERNEL via grad[data_index] */,
    slot_off: &[usize] /* num_features entries; launcher adds its own sentinel internally */, slot_len, num_bin: u32,
  ) -> Result<Vec<f64>>

Note the slot_off contract differs: the LDS launcher consumes a sentinel-free slot_off and
builds the sentinel internally (`resident_raw_build_into`); the mirror launcher also takes a
sentinel-free slot_off and builds its sentinel internally (`slot_off_sentinel`). Pass the
SAME plain `slot_off: Vec<usize>` (num_features entries) to BOTH.
</interface_contract>

<tasks>

<task type="auto">
  <name>Task 1: Write the rocm-gated mirror-vs-LDS A/B micro-bench example</name>
  <files>crates/lgbm-compute/examples/mirror_vs_lds.rs</files>
  <action>
Create `crates/lgbm-compute/examples/mirror_vs_lds.rs` modeled structurally on
`cuda_mirror_overhead.rs` (same `#[cfg(not(feature="rocm"))]` stub + `#[cfg(feature="rocm")]`
main, same `median` helper, same deterministic hash-style bin generation, same
read-back-forces-sync timing pattern). Per NGO-01.

Sweep design (per the measurement design): FEATS=50; BIN_SWEEP=[16,64,256]; NUM_DATA=200_000;
and TWO leaf sizes — a LARGE leaf (≈ full corpus, e.g. ALL rows `0..NUM_DATA`) and a MID leaf
(≈ 50k rows, e.g. `(0..NUM_DATA).step_by(4)`). WARMUP=3 discarded, TIMED=7 (median of >=5),
copying the exact warm-vs-cold discipline from `cuda_mirror_overhead.rs` and the SKILL rule.

Build inputs ONCE per (bin, leaf) cell:
- Resident feature-major bins `resident[f*NUM_DATA + row]` via the same deterministic hash as
  `cuda_mirror_overhead.rs` (NO RNG), uploaded ONCE via `client.create_from_slice(u32::as_bytes(&resident))`
  to a single `Handle` reused (cloned) by BOTH timed loops — the upload-once win is settled in
  260619-mwr and is EXCLUDED from this comparison (time only the per-leaf build call).
- Full-corpus `grad: Vec<f32>` / `hess: Vec<f32>` (length NUM_DATA) — the mirror's native form.
- `leaf_rows: Vec<u32>` = the leaf's `data_indices` (same array fed to both).
- Plain `slot_off: Vec<usize>` = `(0..FEATS).map(|f| f*2*num_bin)`, `slot_len = FEATS*2*num_bin` — fed to both.
- Host-gathered LEAF-LENGTH `ord_g`/`ord_h` for the LDS path: `ord_g[k] = grad[leaf_rows[k] as usize]`.

Time THREE numbers per cell (each call ends by reading `out[0]` to force device sync, as in
the model example):
  (1) LDS gather-INCLUDED — the production cost: INSIDE the timed closure, gather ord_g/ord_h
      from full-corpus grad/hess to leaf length, THEN call
      `build_leaf_histograms_resident_f32_on(client, resident_handle.clone(), FEATS, NUM_DATA,
      &slot_off, slot_len, &leaf_rows, &ord_g, &ord_h)`. This is what production pays.
  (2) LDS gather-EXCLUDED — kernel-only attribution: gather ord_g/ord_h ONCE outside the timed
      loop, time only the launcher call. (Warm up and median this separately.)
  (3) MIRROR — `construct_histograms_cuda_mirror_resident_on(client, resident_handle.clone(),
      NUM_DATA, FEATS, &leaf_rows, &grad, &hess, &slot_off, slot_len, num_bin)`; the in-kernel
      gather is intrinsic, so there is no gather-excluded variant for the mirror.

Each variant: WARMUP discarded launches, then median of TIMED. Print a side-by-side table:
columns = leaf_size, bins, mirror_ms, lds_incl_ms, lds_excl_ms, speedup (lds_incl/mirror) — so
>1.0 means the mirror is faster than the production LDS path; also show speedup_kernel
(lds_excl/mirror) for the kernel-only view. Print explanatory footer lines (what's excluded:
resident upload; what's included: LDS host gather in lds_incl) mirroring the model example's
footer style.

SANITY ASSERT (same-input correctness, NOT the parity gate): once per (bin, leaf) cell, after
the timed loops, assert the mirror RAW histogram and the LDS RAW histogram agree within
ABS 5e-6 / REL 1e-5 (the f32-atomic envelope from `rocm_cuda_mirror.rs::assert_close`). This
pins them ONLY to each other as a same-input check so the A/B compares correct computations;
the real parity gate stays the CPU f64 anchor (already covered by existing tests) — add a
comment saying exactly this. On mismatch, panic with the diverging cell index + values.

Doc-comment the example header: purpose (260619-ngo A/B), the fairness notes (resident upload
excluded, LDS host gather included in lds_incl + reported excluded separately, same leaf/bins
for both), the warm-vs-cold discipline, "MANUAL rocm bench — NOT a test gate", "MEASUREMENT
ONLY — does NOT wire into lib.rs/the learner", and the run command:
`cargo run --release -p lgbm-compute --features rocm --example mirror_vs_lds`.

Do NOT modify any kernel, lib.rs, the learner, or the CPU f64 anchor. Reuse both launchers
as-is. No fenced code in the file beyond normal Rust source; no new deps.
  </action>
  <verify>
    <automated>cargo build --release -p lgbm-compute --features rocm --example mirror_vs_lds 2>&1 | tail -5</automated>
  </verify>
  <done>The example compiles under `--features rocm` (rocm-gated, stub under no-rocm), calls BOTH resident launchers with the shared resident Handle, times the three numbers with WARMUP>=2 / median>=5, and contains the same-input ABS 5e-6 / REL 1e-5 sanity assert. No kernel/lib.rs/learner edits.</done>
</task>

<task type="auto">
  <name>Task 2: Run on gfx1100, capture the A/B table, write the wiring recommendation</name>
  <files>crates/lgbm-compute/examples/mirror_vs_lds.rs</files>
  <action>
Run `cargo run --release -p lgbm-compute --features rocm --example mirror_vs_lds` on the local
gfx1100. To honor the SKILL's drift discipline, run the example 2–3 times (separate process
invocations) and use the run whose medians are stable / report the representative run; note any
warmup drift. Capture the full side-by-side table (leaf_size × bins → mirror_ms, lds_incl_ms,
lds_excl_ms, speedup, speedup_kernel) and confirm the same-input sanity assert passed (no panic)
across all cells — i.e. the two kernels computed the same histogram within the f32-atomic
envelope, so the timing comparison is between correct computations.

Then derive the wiring recommendation from the numbers, following the user's stated rule and
the honesty mandate:
  - If the mirror is FASTER than the production LDS path (speedup = lds_incl/mirror > 1.0)
    consistently across the regime where GPU matters (large/mid leaf, the bin sweep) →
    recommend WIRE AS DEFAULT.
  - If the mirror is SLOWER (speedup < 1.0) or mixed/within noise → recommend WIRE GATED
    (opt-in) or LEAVE AS PRIMITIVE, stating which based on whether there's ANY regime where it
    wins (gated if a clear win-regime exists; primitive if none).
  Cite the kernel-only view (speedup_kernel = lds_excl/mirror) to separate "the mirror's
  in-kernel gather helps/hurts" from "the LDS host gather is the tax" — this tells the
  follow-up whether the LDS path's real weakness is the host gather (which a future change could
  remove independently) vs the kernel compute itself.

Write the SUMMARY at
`.planning/quick/260619-ngo-benchmark-the-cuda-mirror-resident-histo/260619-ngo-SUMMARY.md`
containing: (1) the real gfx1100 A/B table verbatim, (2) the same-input sanity-check result,
(3) the explicit evidence-based wiring recommendation (default / gated / primitive) with the
specific numbers behind it, stated plainly per the honesty mandate (report a flat or negative
result honestly if that's what the data shows), and (4) a one-line note that wiring is the
deliberately-deferred follow-up — NOT done in this task.

Do NOT wire anything. NEVER git-add LightGBM*/ or cuml-main/.
  </action>
  <verify>
    <automated>test -f .planning/quick/260619-ngo-benchmark-the-cuda-mirror-resident-histo/260619-ngo-SUMMARY.md && grep -iq "recommend" .planning/quick/260619-ngo-benchmark-the-cuda-mirror-resident-histo/260619-ngo-SUMMARY.md && echo OK</automated>
  </verify>
  <done>SUMMARY exists with the real gfx1100 A/B table, the same-input sanity-check result, and an explicit default/gated/primitive wiring recommendation backed by the measured numbers. Nothing wired; no reference trees git-added.</done>
</task>

</tasks>

<verification>
- `cargo build --release -p lgbm-compute --features rocm --example mirror_vs_lds` succeeds.
- The example runs on gfx1100 and prints a complete A/B table for both leaf sizes × {16,64,256}.
- The same-input sanity assert (ABS 5e-6 / REL 1e-5) passes for every cell — confirming the
  A/B compares correct, equivalent computations.
- `git status` shows ONLY `crates/lgbm-compute/examples/mirror_vs_lds.rs` (+ the planning dir)
  changed — NO edits to histogram.rs, lib.rs, the learner, the CPU anchor; no LightGBM*/cuml-main staged.
</verification>

<success_criteria>
- A new rocm-gated example A/Bs the mirror resident kernel vs the wired LDS resident kernel
  with the resident upload excluded, the LDS host gather included (and separately excluded),
  same leaf rows + same bins, warm-vs-cold honored (>=2 warmup, median >=5).
- SUMMARY contains the real gfx1100 A/B table AND an explicit evidence-based wiring
  recommendation (default if faster; gated-or-primitive if slower) per the user's rule and the
  honesty mandate.
- NOTHING is wired into lib.rs / the learner / the wired path; kernels and the CPU f64 anchor
  are untouched.
</success_criteria>

<output>
Create `.planning/quick/260619-ngo-benchmark-the-cuda-mirror-resident-histo/260619-ngo-SUMMARY.md` when done.
</output>
