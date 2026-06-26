---
phase: quick-260627-agx
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-compute/src/kernels/subtract.rs
autonomous: true
requirements: [QUICK-260627-agx]
must_haves:
  truths:
    - "The vectorized subtract produces byte-identical output to the existing scalar kernel on every cell (f64 and f32)"
    - "The production all-256-bin shape (n divisible by max width) runs the Vector<F,N> path"
    - "Mixed-cardinality shapes (n not divisible by width) fall back to the existing scalar kernel, still bit-exact"
    - "The CPU f64 anchor subtract_histograms_cpu_native is unchanged"
    - "The rocm resident subtract (subtract_histograms_f64_from_handles_on) uses the vectorized kernel when divisible, no read-back"
  artifacts:
    - path: "crates/lgbm-compute/src/kernels/subtract.rs"
      provides: "Vectorized #[cube(launch)] subtract kernel + width-gated launchers"
      contains: "Vector<"
  key_links:
    - from: "subtract_histograms_f64_on / subtract_histograms_f32_on / subtract_histograms_f64_from_handles_on"
      to: "subtract_hist_kernel_vec"
      via: "divisibility-gated dispatch on client.io_optimized_vector_sizes max width"
      pattern: "io_optimized_vector_sizes"
---

<objective>
Wire the spike-041 `Vector<P,N>` vectorization into the production element-wise
histogram subtract kernel so the rocm RESIDENT subtract and the portable
cuda/wgpu / generic-f64 launch use vectorized loads/stores, bit-exactly.

Purpose: capture the ROCm-parity-track perf win (spike-041 VALIDATED: cubecl-hip
f32 vec4 1.06–1.29× sign-stable, cubecl-cpu vec16 up to 3.7×) on a non-dominant
phase without risking the numerical-parity merge gate. Minimal + additive only.
Output: a single new vectorized `#[cube(launch)]` kernel plus width-gated dispatch
inside the three existing launchers, all in `crates/lgbm-compute/src/kernels/subtract.rs`.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md
@.planning/spikes/041-line-feasibility-subtract/README.md
@.planning/spikes/CONVENTIONS.md
@crates/lgbm-compute/src/kernels/subtract.rs
@crates/lgbm-compute/examples/spike041_vector_subtract_ab.rs

Key facts (cubecl 0.10, verified by spike-041 on hip + cpu, bit-exact every cell):
- Vectorized type is `Vector<P: Scalar, N: Size>` (NOT `Line<T>` — later rename).
- Kernel sig `&Array<Vector<F, N>>`; `N: Size` generic. The width is a RUNTIME
  `usize` positional arg inserted right after `CubeDim` in the launch call:
  `subtract_hist_kernel_vec::launch::<F, R>(client, count, dim, vector_size, parent, child, out, n_vec)`.
- `ArrayArg::from_raw_parts(handle, n_elements / vector_size)` — length in VECTOR
  units over the SAME byte buffer. A bare `usize` kernel param is passed RAW.
- Max width: `client.io_optimized_vector_sizes(std::mem::size_of::<F>()).next()`
  (the iterator yields widest-first; hip f32 → [4,2,1], cpu f64 → [8,4,2,1]).
- `Vector` impls element-wise `Sub` → bit-exact to scalar by construction.
- The CPU production path is NATIVE (`subtract_histograms_cpu_native`, lib.rs:1343)
  and is NOT one of the three launchers below — do not touch it.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add the vectorized subtract kernel and width-gate the three launchers</name>
  <files>crates/lgbm-compute/src/kernels/subtract.rs</files>
  <behavior>
    - subtract_hist_kernel_vec is byte-identical (to_bits) to the scalar kernel for
      f64 at len 256000 (500feat×256bin×2, divisible by width 8/4/2) — every cell.
    - Same for f32 at len 256000 (divisible by width 16/8/4/2).
    - subtract_histograms_f64_on and subtract_histograms_f32_on return output
      byte-identical to subtract_histograms_cpu_native / serial `p - c` for BOTH a
      width-divisible length (256000) AND a NON-divisible length (12345, which falls
      back to the scalar kernel) — the existing subtract_parallel_equals_serial_*
      tests already assert this and MUST stay green.
    - Empty input (n == 0) and length-mismatch error paths are unchanged.
  </behavior>
  <action>
    Add ONE new generic vectorized kernel and route each of the three existing
    launchers through it when, and only when, the flat length divides the chosen
    width. Lowest-risk divisibility gate (per task brief): NO tail logic — divisible
    shapes vectorize, everything else stays on the proven scalar kernel.

    1. Add a new `#[cube(launch)]` kernel `subtract_hist_kernel_vec<F: Float, N: Size>`
       taking `parent: &Array<Vector<F, N>>`, `child: &Array<Vector<F, N>>`,
       `out: &mut Array<Vector<F, N>>`, `n_vec: usize`. Body is the SAME 1D
       grid-stride loop as the existing scalar kernels but operating on whole
       vectors: `out[i] = parent[i] - child[i]` with `while i < n_vec`. Use the
       canonical launch ABI from `spike041_vector_subtract_ab.rs` (verified on
       hip+cpu). Document it as bit-exact-by-construction (element-wise Vector::sub,
       no float reorder, no atomics/reduction) per spike-041 and CONVENTIONS 313–351.
       Keep the existing scalar `subtract_hist_kernel` and `subtract_hist_kernel_f32`
       UNCHANGED — they are the fallback for non-divisible lengths.

    2. Add a tiny private helper to pick the width, e.g.
       `fn pick_vec_width<R: cubecl::Runtime>(client: &ComputeClient<R>, elem_size: usize, n: usize) -> usize`
       returning `client.io_optimized_vector_sizes(elem_size).next()` filtered so the
       result is `> 1` AND `n % width == 0`, else `1`. Width `1` means "use scalar".

    3. In `subtract_histograms_f64_on<R>`: after the existing length/empty validation
       and the three `create_from_slice` handle allocations, compute
       `let vs = pick_vec_width(client, std::mem::size_of::<f64>(), n);`. If `vs > 1`,
       launch `subtract_hist_kernel_vec::launch::<f64, R>` with `CubeCount::Static(64,1,1)`,
       `CubeDim::new_1d(256)`, the runtime `vs` arg right after the dim, and the three
       `ArrayArg::from_raw_parts(handle, n / vs)` (vector-unit lengths) plus the raw
       `n / vs` bound. Else keep the EXISTING scalar `subtract_hist_kernel::launch`.
       Read-back (`read_one_unchecked`) and return are unchanged. Keep the SAFETY
       comment; note both buffers cover the same `n` f64 cells whether read as scalar
       or as `n/vs` vectors.

    4. In `subtract_histograms_f32_on<R>`: identical change with `f32` /
       `subtract_hist_kernel_f32` as the scalar fallback and
       `std::mem::size_of::<f32>()`.

    5. In `subtract_histograms_f64_from_handles_on<R>` (cfg rocm, the resident hot
       path — consumes input Handles, returns the `out` Handle, NO read-back): after
       allocating `h_out`, compute `let vs = pick_vec_width(client, std::mem::size_of::<f64>(), len);`
       and dispatch the vectorized kernel over `len / vs` vector units when `vs > 1`,
       else the existing scalar `subtract_hist_kernel`. Return `h_out` unchanged.

    Do NOT change the three launchers' public signatures, the empty/zero guards, the
    length-mismatch errors, or `subtract_histograms_cpu_native`. Do NOT add tail
    handling — note "mixed-cardinality tail vectorization is a possible follow-on"
    in a doc comment only. ROI is ROCm-parity-track and bounded; keep it minimal.

    6. Add two unit tests mirroring the existing `subtract_parallel_equals_serial_*`:
       `subtract_vec_equals_serial_f64` and `_f32`, each asserting `to_bits()`
       equality vs a serial `p - c` at a width-DIVISIBLE length (256000) — proving the
       vectorized branch itself is bit-exact on the cpu client. The existing
       12345-length cases already cover the non-divisible scalar fallback.
  </action>
  <verify>
    <automated>cargo test -p lgbm-compute --lib subtract 2>&1 | tail -25</automated>
  </verify>
  <done>
    `subtract_hist_kernel_vec` exists; all three launchers dispatch through it under
    the `n % vs == 0 && vs > 1` gate and fall back to the scalar kernels otherwise;
    `subtract_histograms_cpu_native` is untouched; the new + existing
    `subtract_*_equals_serial_*` tests pass byte-identically on the cpu client.
  </done>
</task>

<task type="auto">
  <name>Task 2: Run the full merge gate (parity-critical + rocm build)</name>
  <files>crates/lgbm-compute/src/kernels/subtract.rs</files>
  <action>
    Run the complete merge gate from the task brief and confirm every suite is green
    (bit-exact parity is non-negotiable). If any parity test changes a golden or
    fails, the vectorized dispatch is NOT bit-exact for that shape — revert that
    launcher to scalar-only and re-investigate; do not relax any tolerance or golden.
    The CPU f64 anchor is the hard gate; the rocm parity tests guard the resident path.
  </action>
  <verify>
    <automated>cargo test -p lgbm-treelearner --lib 2>&1 | tail -15 && cargo test -p lgbm 2>&1 | tail -15 && cargo test -p oracle-harness raw_bin_train_matches_cpp_golden 2>&1 | tail -15</automated>
  </verify>
  <done>
    `cargo test -p lgbm-treelearner --lib`, `cargo test -p lgbm`, and
    `cargo test -p oracle-harness` (incl. `raw_bin_train_matches_cpp_golden`) pass;
    `cargo test -p oracle-harness --features rocm` (kernel_parity / subtract parity)
    passes; `cargo build --release --features rocm` succeeds. No golden file changed.
  </done>
</task>

</tasks>

<verification>
- `cargo test -p lgbm-compute --lib subtract` — vectorized kernel bit-exact (cpu, f64+f32).
- `cargo test -p lgbm-treelearner --lib` — tree-learner subtract integration green.
- `cargo test -p lgbm` — top-level crate green.
- `cargo test -p oracle-harness raw_bin_train_matches_cpp_golden` — C++ golden parity.
- `cargo test -p oracle-harness --features rocm` — rocm kernel/subtract parity green.
- `cargo build --release --features rocm` — resident path compiles vectorized.
- No golden/fixture file modified anywhere in the diff.
</verification>

<success_criteria>
The three subtract launchers run `Vector<F,N>` loads/stores on width-divisible flat
lengths and fall back to the proven scalar kernels otherwise; output is byte-identical
to the prior scalar path on every cell; the CPU native f64 anchor is untouched; and the
entire merge gate (CPU goldens + rocm parity + rocm release build) is green.
</success_criteria>

<output>
Create `.planning/quick/260627-agx-vectorize-cubecl-subtract-kernel-with-ve/260627-agx-SUMMARY.md` when done.
</output>
