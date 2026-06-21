---
phase: 04-compute-backend-cpu-first-integer-histograms-rocm
reviewed: 2026-06-05T20:01:40Z
depth: standard
files_reviewed: 19
files_reviewed_list:
  - crates/lgbm-compute/src/error.rs
  - crates/lgbm-compute/src/gain.rs
  - crates/lgbm-compute/src/kernels/histogram.rs
  - crates/lgbm-compute/src/kernels/mod.rs
  - crates/lgbm-compute/src/kernels/partition.rs
  - crates/lgbm-compute/src/kernels/split.rs
  - crates/lgbm-compute/src/kernels/subtract.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-compute/src/runtime.rs
  - crates/lgbm-compute/Cargo.toml
  - crates/lgbm-compute/tests/capability.rs
  - crates/lgbm-compute/tests/cmp01_containment.rs
  - crates/lgbm-compute/tests/determinism_spike.rs
  - crates/lgbm-compute/tests/rocm_smoke.rs
  - crates/oracle-harness/tests/kernel_parity.rs
  - crates/oracle-harness/Cargo.toml
  - xtask/cpp/kernel_capture.cpp
  - xtask/cpp/CMakeLists.txt
  - xtask/src/main.rs
findings:
  critical: 2
  warning: 6
  info: 4
  total: 12
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-06-05T20:01:40Z
**Depth:** standard
**Files Reviewed:** 19
**Status:** issues_found

## Summary

This phase ships the `lgbm-compute` CubeCL kernel layer (histogram / find_best_split /
data_partition / subtract), a typed `ComputeError` boundary, a startup capability gate, the
in-kernel gain math, and the C++ `kernel_capture.cpp` oracle harness plus the
`oracle-harness` parity replay. The transcription discipline is generally careful and the
verbatim-vs-C++ comments are unusually detailed.

Because the non-negotiable contract here is *numerical fidelity to the C++ reference*, the
review focused hardest on (a) gate ORDER / epsilon placement / fold order in the gain scan,
(b) divergences between the production `#[cube]` primitives and the test/oracle replication
helpers (a mismatch there can either mask a real bug or falsely fail), and (c) the
self-consistency of the C++ golden generator against the Rust kernel it is supposed to gate.

Two BLOCKER-class divergences were found where the *oracle / test replication* path computes
the sign function differently from the production kernel and the C++ reference (`signum` vs
`Sign`), and where the partition golden generator never validates `threshold < num_bin`,
allowing a golden case to silently disagree with the kernel's validation contract. Six
WARNING-class issues cover the gain-shift L1 sign discrepancy reaching production via a host
pre-step, an unguarded `hist[bi+1]` index relying on offset arithmetic that the safety comment
asserts but does not bound, a `default_left` sentinel mismatch vs the C++ "no-split" default,
and several robustness gaps in the capture harness.

No structural-findings substrate was supplied with this review, so all findings below are
narrative (direct code review).

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Oracle/host L1 sign uses `signum()` (≠ C++ `Sign`, differs at g==0 and signed zero)

**File:** `crates/oracle-harness/tests/kernel_parity.rs:499-506` (host replication `leaf_gain`),
`crates/oracle-harness/tests/kernel_parity.rs:1017-1024` (`leaf_gain_f32`)
**Issue:**
The production gain primitive `gain::threshold_l1` implements `Common::Sign(s) = (s>0)-(s<0)`,
which is **0** at `s == 0.0` (and treats `+0.0`/`-0.0` identically). The oracle's host
replication of the whole-leaf `gain_shift` instead uses Rust `f64::signum`:

```rust
let s = g.signum() * (g.abs() - l1).max(0.0);
```

`f64::signum(0.0)` returns `+1.0` and `f64::signum(-0.0)` returns `-1.0` — it never returns 0.
For the L1 branch with `sum_gradient == 0.0` (or a `-0.0` accumulation result), the oracle's
`min_gain_shift` diverges from what the C++ reference and the production kernel compute. Because
`win_gain` is reported as `raw_gain - min_gain_shift`, this makes the bit-exact winner-field
assertion (`compare_exact_f64_bits`) compare the kernel's correct value against a
*wrongly-computed* expectation. With the current golden corpus (`l1_forward` has
`sum_gradient = -2.0`, non-zero) the bug is latent, but any future L1 case with a zero gradient
sum — a realistic boundary — will either falsely fail or, worse, mask a real kernel regression.
A parity oracle that does not faithfully reproduce `Sign` cannot certify the 1e-12 contract.
**Fix:** Replace `signum` with the project `Sign` semantics in both helpers:
```rust
fn sign(x: f64) -> f64 { ((x > 0.0) as i32 - (x < 0.0) as i32) as f64 }
let s = sign(g) * (g.abs() - l1).max(0.0);
```
(and the f32 analogue in `leaf_gain_f32`). Better: call the public
`lgbm_compute::gain::get_leaf_gain` directly instead of re-deriving the formula, eliminating the
second source of truth entirely.

### CR-02: `find_best_split` host `gain_shift` (production) shares the same `signum`-class risk only in the test — but the C++ golden's `min_gain_shift` is authoritative; verify the kernel host path uses the correct `Sign`

**File:** `crates/lgbm-compute/src/kernels/split.rs:571-578`
**Issue:**
The *production* host pre-step computes `gain_shift` via `crate::gain::get_leaf_gain` →
`threshold_l1`, which uses the correct `select`-based `Sign` (so production is correct). The
BLOCKER is that the **bit-exact parity gate that is supposed to prove this** (CR-01) computes
`min_gain_shift` with a different sign rule, so the test cannot actually detect a sign
regression in the production `gain_shift`. Concretely: the winner-field comparison in
`kernel_parity.rs` subtracts the *oracle's* `min_gain_shift` from the *kernel's* raw gain on the
got side (`si.gain` already has the host `min_gain_shift` subtracted) and compares against the
golden's `win_gain` (which the C++ harness computed with the correct `Sign`). The got side uses
production `Sign`; the golden side uses C++ `Sign`; but the *no-split / splittable* boundary and
any future zero-gradient L1 case route through the oracle's `signum` helper in
`replicate_candidates` (line 378 calls `leaf_gain(...)` for `min_gain_shift`), so the
per-candidate `compare_exact_f64_bits` at lines 530/533 will diverge from the kernel for any
zero-gradient L1 case. This is the same defect surface as CR-01 but reached through the
per-candidate assertion path, which is the primary localizing gate. It must be fixed for the
gain MATH assertion to be trustworthy.
**Fix:** Same as CR-01 — make `leaf_gain`/`leaf_gain_f32` in the parity test use `(x>0)-(x<0)`
sign semantics (or delegate to `gain::get_leaf_gain`). After the fix, add a golden case with
`sum_gradient == 0.0` and `lambda_l1 > 0` to `BuildSplitCorpus` so the boundary is actually
exercised.

## Warnings

### WR-01: Partition golden generator never validates `threshold < num_bin`; golden can encode a case the kernel rejects

**File:** `xtask/cpp/kernel_capture.cpp:911-940` (`EmitPCase`),
`crates/lgbm-compute/src/kernels/partition.rs:126-130` (kernel rejects `threshold >= num_bin`)
**Issue:**
`data_partition_on` returns `ComputeError::Runtime` when `threshold >= num_bin`. The C++
`EmitPCase`/`SplitRoute` path has no such guard — it computes `th = threshold + min_bin` and
routes regardless. If any `PCaseSpec` were authored with `threshold >= num_bin` (an easy
mistake, since `threshold` here is a raw bin offset and the corpus mixes `min_bin`/`max_bin`),
the golden would contain a `PORDER`/`PSPLIT` the kernel can never reproduce: the parity test
would `panic!("data_partition failed")` rather than compare. The current corpus happens to keep
`threshold < num_bin`, but the generator should refuse to emit an out-of-contract case so a
future edit fails loudly at capture time, not at replay.
**Fix:** Add an assertion in `EmitPCase` (or `BuildPartitionCorpus`) mirroring the kernel
contract:
```cpp
if (cs.threshold >= cs.num_bin) { std::cerr << "PCASE " << cs.name << ": threshold >= num_bin\n"; std::abort(); }
```

### WR-02: `hist[bi + 1]` read depends on `t+offset < num_bin` but the kernel only bounds `t`, not `t+offset` against the allocation

**File:** `crates/lgbm-compute/src/kernels/split.rs:186-191` (REVERSE), `251-253` (FORWARD),
SAFETY comment at `599-603`
**Issue:**
In REVERSE, `t = t_start - k` where `t_start = num_bin - 1 - offset`; `bi = (t as usize) * 2`
and the kernel reads `hist[bi]` and `hist[bi + 1]`. The histogram allocation is `2 * num_bin`
cells, so the read is in-range only if `0 <= t < num_bin`. With `offset > 0`, `t_start =
num_bin-1-offset < num_bin-1`, fine; the lower bound is `t_end = 1-offset` which can be
**negative** when `offset > 0`. `rev_count = num_bin - 1` (NOT `num_bin-1-offset+... `), so the
REVERSE loop runs `num_bin-1` iterations from `t_start` down to `t_start-(num_bin-2) = 1-offset`.
For `offset >= 2`, the final `t` values are negative, and `(t as usize)` wraps to an enormous
`usize`, so `hist[bi]` would index far out of bounds. On the f64 cpu kernel this is a
`#[cube]`-lowered array access whose bounds behavior is backend-dependent (cubecl-cpu may not
bounds-check), i.e. a potential OOB read / UB, exactly the class the V5 validation claims to
prevent. The host SAFETY comment asserts "`t+offset` in range" but never validates `offset`
against `num_bin`, and the only golden with `offset>0` is `default_bin_skip` (offset=1), which
does not reach negative `t` because `num_bin=5`, `rev_count=4`, min `t = (5-1-1) - 3 = 0`. The
defect is unexercised, not absent.
**Fix:** Either (a) validate `offset` at the host boundary so `t` cannot go negative
(`rev_count` should be `num_bin - 1` but the loop must stop at `t >= max(0, 1-offset)` — clamp
`rev_count = min(num_bin-1, t_start + offset)` i.e. `t_start - (1-offset).max(0) + 1`), or (b)
gate the body on `t >= 0` with a `select`/guard so negative `t` reads index 0 inertly. The C++
reference loop condition is `t >= 1 - offset`, and `t` never goes negative there because the
loop bound encodes it; the Rust `0..rev_count` counter loses that bound and must restore it.

### WR-03: `SplitInfo::none()` sets `default_left: true`, but the kernel/host "no-split" path returns it with `default_left` from a stale read — confirm the C++ no-split default

**File:** `crates/lgbm-compute/src/gain.rs:284-298`, consumed at
`crates/lgbm-compute/src/kernels/split.rs:665-667`
**Issue:**
`SplitInfo::none()` hard-codes `default_left: true`. The C++ `WinSplit` default also has
`default_left = true` (`kernel_capture.cpp:335`), so the *value* matches today. The risk is
semantic: the host returns `SplitInfo::none()` whenever `!is_splittable`, discarding the kernel's
`best_default_left` (out[9]). If a downstream Phase-5 consumer ever inspects `default_left` on a
no-split result, the contract is "always true" — undocumented and easy to mis-rely-on. The
parity test only asserts `default_left` when `win_is_splittable` (line 602), so a wrong no-split
`default_left` would never be caught.
**Fix:** Document on `SplitInfo::none()` that `default_left` is a don't-care sentinel for
no-split, or assert it is never read by consumers. Low risk, but the invariant should be written
down given the fidelity contract.

### WR-04: `cfg_skip_default_bin` heuristic (`default_bin < num_bin`) may not match C++ `SKIP_DEFAULT_BIN` template dispatch

**File:** `crates/lgbm-compute/src/kernels/split.rs:783-791`
**Issue:**
The comment admits this is a *conservative approximation* of the C++ `SKIP_DEFAULT_BIN`
template-bool dispatch: "we conservatively skip whenever a valid in-range `default_bin` is
present." The real C++ dispatch keys `SKIP_DEFAULT_BIN` on `meta_->offset` and whether the
default bin is actually inside the scanned range — not merely `default_bin < num_bin`. The
parity test passes `skip_default_bin` from the golden line, but the *kernel* recomputes it via
this heuristic (line 614) and ignores the golden's `skip_default_bin` field, so the test only
validates the heuristic against the C++ harness's *own* `cfg.skip_default_bin` choice, which is
hand-set per case (`default_bin_skip` uses `skip=true`). For cases where `default_bin < num_bin`
but C++ would NOT set `SKIP_DEFAULT_BIN` (e.g. `forward_winner` uses `default_bin=4==num_bin`, so
heuristic yields false — coincidentally correct), the heuristic could diverge. This is a
correctness landmine for Phase-5 when real `offset`/`default_bin` combinations appear.
**Fix:** Pass the authoritative `skip_default_bin` flag from the caller (it is already in the
golden and in the C++ `SplitCfg`) into `find_best_split` rather than re-deriving it, OR transcribe
the exact C++ dispatch predicate. At minimum, add a golden case where `default_bin < num_bin` but
`skip_default_bin == false` to prove the heuristic.

### WR-05: f32 split kernel `select(s > 0.0, 1.0, 0.0)` relies on literal-type inference for f32; verify no f64 promotion

**File:** `crates/lgbm-compute/src/gain.rs:142-144` (`threshold_l1_f32`), `351-353` etc.
**Issue:**
In `threshold_l1_f32`, `select(s > 0.0, 1.0, 0.0)` uses bare `1.0`/`0.0` literals where `s: f32`.
The f64 sibling (line 51-52) uses the same bare literals with `s: f64`. If cubecl `#[cube]`
literal inference resolves `1.0` to f64 in the f32 function (because `select`'s value type is not
pinned by `s`), the subtract `(pos - neg) * reg_s` would mix f32/f64 or silently widen — a
divergence the f32 path is supposed to avoid. The f64 anchor is unaffected. This needs a
compile/behavior check on the hip path; the comment in the f64 version notes the `if/else` form
"mis-lowers to a constant on cubecl-cpu," implying literal lowering here is fragile.
**Fix:** Pin the literal types explicitly: `select(s > 0.0f32, 1.0f32, 0.0f32)` in all `_f32`
primitives (and `0.0f32` for `reg_s`'s `f32::max(0.0, ...)`), removing any inference ambiguity.

### WR-06: Capture harness `SparseBin::Data` has an unguarded `deltas_[i_delta]` that can read past the end

**File:** `xtask/cpp/kernel_capture.cpp:173-182` (`SparseBin::Data`), `197-211`
(`ConstructHistogram`)
**Issue:**
`SparseBin::Data` loops `++i_delta; cur_pos += deltas_[i_delta];` and only checks
`i_delta >= num_vals_` *after* indexing `deltas_[i_delta]`. For an `idx` larger than any stored
row (or a malformed delta stream), `deltas_[i_delta]` is read before the bound check, an OOB
read in the C++ oracle generator. Because this is the golden source of truth, an OOB read here
silently corrupts a fixture (UB), which would then be replayed as "C++ truth." The
`all_bin0_sparse` case (empty delta stream → `deltas_` holds a single trailing `0`) is the
closest trigger. This is dev-only code, but it feeds the authoritative golden.
**Fix:** Reorder the guard to check `i_delta >= static_cast<int>(deltas_.size())` (or
`num_vals_`) *before* dereferencing `deltas_[i_delta]`, matching the upstream
`sparse_bin.hpp` bound discipline.

## Info

### IN-01: `data_partition` validates bins but the kernel re-reads `bins[i]` as i32 without using the validated `num_bin` bound in-kernel

**File:** `crates/lgbm-compute/src/kernels/partition.rs:66-74`
**Issue:** Host validates `bins[i] < num_bin` (good), but the kernel converts `bins[i] as i32`
and compares against `min_bin/max_bin/th`. For `num_bin` near `i32::MAX` the `as i32` cast could
wrap; `data_size_t = i32` in C++ bounds `num_bin` well below that, so this is informational, not a
live bug. Document the `num_bin <= i32::MAX` precondition.

### IN-02: Magic launch shape `CubeCount::Static(1,1,1)` / `CubeDim::new_1d(1)` repeated in 4 kernels without a shared constant

**File:** `histogram.rs:127-128`, `partition.rs:156-157`, `split.rs:606-608`, `subtract.rs:93-94`
**Issue:** The single-owner ordered-fold launch shape is the load-bearing determinism invariant
and is duplicated verbatim across four launchers. A shared `fn single_owner_launch_dims()` or
named const would make the invariant (and any future change) single-sourced and self-documenting.
**Fix:** Extract a helper returning `(CubeCount, CubeDim)` for the sequential anchor.

### IN-03: `rocm_smoke.rs` / hip parity asserts plane/atomic capabilities but those features are never exercised by the f32 single-owner kernels

**File:** `crates/lgbm-compute/tests/rocm_smoke.rs:31-56`,
`crates/lgbm-compute/src/runtime.rs:76-82`
**Issue:** `reduce_path()` returns `Plane` on hip, but every kernel launches single-owner
`CubeDim(1)` regardless, so `has_plane`/`plane_size`/`has_f32_atomic` are probed and asserted but
never actually drive a code path in Phase 4. This is fine as forward-looking capability plumbing,
but the asserted matrix gives a false impression that the Plane path is used. A one-line comment
that `ReducePath::Plane` is currently informational (no kernel consumes it yet) would prevent
confusion.

### IN-04: Duplicate `round_int` definitions (kernel vs oracle) risk drifting

**File:** `crates/lgbm-compute/src/kernels/split.rs:77-86` and
`crates/oracle-harness/tests/kernel_parity.rs:363-365`
**Issue:** `round_int(x) = (int)(x + 0.5f)` is transcribed twice (production `#[cube]` + oracle
host). Both currently use the f32-widened `0.5f`. They must stay bit-identical; a future edit to
one and not the other would silently break parity. Consider exposing the production `round_int`
(or a plain-Rust twin) from the crate and importing it in the test, as is already done for
`get_split_gains`.

---

_Reviewed: 2026-06-05T20:01:40Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
