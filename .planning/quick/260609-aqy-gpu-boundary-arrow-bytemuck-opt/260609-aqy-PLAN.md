---
phase: quick-260609-aqy
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md
autonomous: true
requirements: [QUICK-260609-aqy]
must_haves:
  truths:
    - "An analysis doc exists ranking every host<->device boundary optimization opportunity by (value x likelihood) / risk"
    - "Every file:line citation in the doc resolves to real code that matches the claim"
    - "Every opportunity carries a parity verdict (ADOPT / INVESTIGATE-FURTHER / REJECT-with-reason)"
    - "The doc contains an explicit 'bytemuck vs CubeElement::as_bytes' verdict and an 'arrow-rs at the boundary' verdict"
  artifacts:
    - path: ".planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md"
      provides: "Ranked, evidence-grounded GPU-boundary optimization analysis"
      contains: "Parity Verdict"
  key_links:
    - from: "260609-aqy-ANALYSIS.md"
      to: "crates/lgbm-compute/src/kernels/histogram.rs"
      via: "file:line evidence citations"
      pattern: "histogram\\.rs:[0-9]+"
---

<objective>
Survey the host<->device (CPU/GPU) boundary in `lgbm-compute` and produce a ranked,
evidence-grounded ANALYSIS DOCUMENT of potential optimizations involving (a) `bytemuck`
and (b) `arrow-rs`, plus the genuine boundary-copy costs the orchestrator scouted.

This is an INVESTIGATION task. The deliverable is the analysis doc, NOT a refactor.
Implementation is out of scope unless a single opportunity is trivially small, zero
parity-risk, and parity-neutral (Task 2, optional).

Purpose: Decide — with file:line evidence and against the hard f32 ~1e-6 / cubecl-cpu
f64-fold bit-exact merge gate — whether bytemuck or arrow-rs add real value at the GPU
boundary, and which (if any) boundary-copy reductions are worth pursuing. The L3
device-resident finding (host<->device round-trip was NOT the GPU bottleneck) must
frame every estimate so gains are not overstated.

Output: `.planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md`
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md

# Primary boundary code (read all before writing the doc)
@crates/lgbm-compute/src/kernels/histogram.rs
@crates/lgbm-compute/src/kernels/split.rs
@crates/lgbm-compute/src/kernels/subtract.rs
@crates/lgbm-compute/src/kernels/partition.rs
@crates/lgbm-compute/src/lib.rs
@crates/lgbm-compute/src/runtime.rs

# Prior arrow-rs investigation — DO NOT re-derive its conclusions; cite them
@.planning/quick/260609-9nu-gpu-boundary-arrow-bytemuck-opt/260609-9nu-FINDINGS-adopt-arrow-rs.md
</context>

<verified_facts>
The orchestrator already scouted the boundary. Anchor the analysis on these (re-verify
each against the actual code with a fresh file:line read — do NOT trust this list blindly):

1. **bytemuck is NOT a dependency** anywhere (`grep -rn bytemuck crates/*/Cargo.toml`
   returns nothing). The code ALREADY does zero-copy slice->bytes via cubecl's
   `CubeElement::as_bytes` (upload) and `T::from_bytes` (readback): e.g.
   `client.create_from_slice(f32::as_bytes(grad))` (histogram.rs:159) and
   `f64::from_bytes(&bytes).to_vec()` (histogram.rs:189). For the cast itself, bytemuck
   is REDUNDANT with cubecl. Determine where (if anywhere) bytemuck adds value BEYOND
   `as_bytes` (struct-of-arrays casts, non-CubeElement types, eliminating intermediate
   Vecs) — or conclude it does not.

2. **Output-buffer zero-alloc + upload** pattern: `vec![0; n]` then `as_bytes(&zeros)`
   uploaded — histogram.rs:166-167, split.rs:799-800, subtract.rs:99-100,
   partition.rs:230-231, plus histogram.rs:291-292, 368-369, 470-471, 575-576, 904-905,
   949-950. Evaluate whether `client.empty(size)` could skip the host zero-alloc+upload.
   PARITY-CRITICAL: the histogram construct kernels use `out[ti] += ...` (accumulate) and
   atomic `fetch_add` — they DEPEND on zero-initialized output (the histogram.rs:161-165
   comment states `empty()` may recycle a stale pooled buffer). `empty()` would BREAK
   these unless a device-side zero/memset is issued. Distinguish accumulate-from-zero
   outputs (must stay zeroed) from fully-overwritten outputs (e.g. the split out-cells
   at split.rs:797-800 are each WRITTEN by the kernel — assess if safe).

3. **Per-element widening collect on readback**:
   `f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)).collect()` at histogram.rs:392,
   495, 606. This is a type conversion (f32->f64), NOT a reinterpret — bytemuck CANNOT
   remove it. Note it as an allocation cost and whether it is necessary.

4. **Host-side concat/gather copies before upload**: the resident-bins concat at
   lib.rs:813-820 (`extend_from_slice` into one buffer, then `as_bytes`), the per-leaf
   `gathered_bins`/`ord_g`/`ord_h` gathers at histogram.rs:456-469. Assess copy cost and
   whether bytemuck/arrow change anything (they do not change a gather).

5. **arrow-rs** is NOT a workspace dep; only `polars-arrow` is present (via pyo3-polars,
   lgbm-python). Prior quick task 260609-9nu + its arrow addendum ALREADY investigated
   arrow-rs for the dataset and concluded: no win for storage/binning, breaks parity,
   tree already has polars-arrow not arrow-rs, earmark arrow-rs (parquet/arrow-csv) only
   for v2 ING-* ingestion. CITE that, do NOT re-derive. For THIS task assess arrow-rs
   NARROWLY at the GPU boundary: does an arrow `Buffer` (64-byte aligned, contiguous)
   feed `create_from_slice` any better than the existing `&[T]`? Data is already
   contiguous `&[T]` — likely marginal. Confirm or refute with evidence.

6. **Hard constraint (CLAUDE.md):** f32 end-to-end, ~1e-6 parity vs C++; the cubecl-cpu
   f64-fold path is bit-exact and is the merge gate. ANY proposed change must be
   parity-neutral or the analysis MUST flag the parity risk explicitly.

7. **Critical prior finding (memory L3, STATE.md):** the device-resident histogram pool
   work proved the host<->device ROUND-TRIP was NOT the GPU bottleneck (mixed win, small
   inputs regressed). Frame ALL boundary-copy estimates against this — "profile before
   assuming the boundary is the bottleneck." Do NOT overstate gains.
</verified_facts>

<tasks>

<task type="auto">
  <name>Task 1: Write the ranked GPU-boundary optimization analysis doc</name>
  <files>.planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md</files>
  <action>
Read every file in `<context>` and re-verify each claim in `<verified_facts>` with a
fresh file:line read (the line numbers above are scouted hints, not gospel — confirm the
actual current line for every citation you put in the doc; a stale citation fails this
task). Use Grep to enumerate ALL boundary sites, not just the cited ones: search
lgbm-compute for `create_from_slice`, `from_bytes`, `client.empty`, `vec!\[0`,
`extend_from_slice`, and `as_bytes` so the survey is exhaustive.

Then write the analysis doc with these sections:

1. **Summary** — one paragraph: the boundary's shape (cubecl already does zero-copy
   reinterpret via CubeElement; the real costs are host zero-allocs, per-element widening
   collects, and host gathers), and the headline verdict on bytemuck and arrow-rs.

2. **The L3 reality check** — quote the STATE.md / memory finding that the round-trip was
   NOT the GPU bottleneck (mixed win, small inputs regressed). This frames every estimate
   below; state explicitly that boundary-copy wins are bounded and "profile before
   assuming the boundary is the bottleneck."

3. **Ranked opportunities** — a table ordered by (value x likelihood) / risk, then one
   subsection per opportunity. EACH opportunity MUST have: file:line evidence | current
   code (named, not pasted as a big block — a short snippet is fine inline) | proposed
   change | expected impact (framed against L3) | parity risk | **Verdict**
   (ADOPT / INVESTIGATE-FURTHER / REJECT-with-reason). At minimum cover:
   - `client.empty()` for FULLY-OVERWRITTEN output buffers (e.g. split out-cells) vs the
     accumulate/atomic buffers that MUST stay zeroed (histogram construct, subtract out,
     partition route). Be precise about which is which and the parity hazard of getting
     it wrong (stale pooled buffer -> wrong histogram -> parity break).
   - The per-element f32->f64 widening collects (histogram.rs:392/495/606) — note these
     are unavoidable type conversions, not removable by bytemuck.
   - The host concat/gather copies (lib.rs resident-bins concat; per-leaf gathers).
   - Any other site the Grep sweep surfaces.

4. **Verdict: bytemuck vs CubeElement::as_bytes** — explicit. Determine whether bytemuck
   adds anything over what cubecl already provides for the cast itself (it is redundant),
   and whether there is ANY niche (struct-of-arrays, non-CubeElement types, Vec
   elimination) where it helps. State ADOPT / REJECT with the reason.

5. **Verdict: arrow-rs at the GPU boundary** — explicit and NARROW (boundary only, not
   the dataset — cite 260609-9nu for the dataset/ingestion conclusion, do not re-derive).
   Does an arrow `Buffer` feed `create_from_slice` better than the existing contiguous
   `&[T]`? State ADOPT / REJECT with the reason.

6. **Recommendation** — what (if anything) to implement now, what to earmark, what to
   reject. If a trivially-small, zero-parity-risk, parity-neutral win exists and is worth
   it, name it as the candidate for the optional Task 2; otherwise state "analysis-only,
   no implementation recommended" and explain why (most likely outcome given L3).

NEVER weaken or contradict the f32 ~1e-6 / cubecl-cpu f64-fold bit-exact contract. If an
opportunity has ANY parity risk, the verdict must flag it explicitly. Do NOT recommend
changing any accumulate-from-zero / atomic buffer to `empty()` without a device-side
zero, and say so.
  </action>
  <verify>
    <automated>test -f .planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md && grep -qi "bytemuck" .planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md && grep -qi "arrow" .planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md && grep -ciE "ADOPT|INVESTIGATE-FURTHER|REJECT" .planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md | grep -qvE '^0$' && for ln in $(grep -oE "histogram\.rs:[0-9]+" .planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-ANALYSIS.md | sed 's/.*://' | sort -u); do test "$ln" -le "$(wc -l < crates/lgbm-compute/src/kernels/histogram.rs)" || { echo "STALE CITATION histogram.rs:$ln"; exit 1; }; done; echo OK</automated>
  </verify>
  <done>
The analysis doc exists at the required path. It contains an explicit "bytemuck vs
CubeElement::as_bytes" verdict and an "arrow-rs at the boundary" verdict, a ranked
opportunities table, and every opportunity has a parity verdict
(ADOPT / INVESTIGATE-FURTHER / REJECT-with-reason). Every `histogram.rs:N` citation
points to a line that exists in the file (the verify command fails on any stale line).
The L3 "round-trip is not the bottleneck" framing is present. No recommendation
weakens the f32 ~1e-6 / cubecl-cpu bit-exact parity contract.
  </done>
</task>

</tasks>

<verification>
- `260609-aqy-ANALYSIS.md` exists and is the sole non-trivial output.
- Spot-check 5+ file:line citations across histogram.rs / split.rs / subtract.rs /
  partition.rs / lib.rs — each must match the claim made about it.
- The bytemuck and arrow-rs verdicts are both present and explicit.
- The accumulate-from-zero vs fully-overwritten buffer distinction is correctly drawn
  (a wrong `empty()` recommendation for an atomic/accumulate buffer is a parity bug and
  fails verification).
</verification>

<success_criteria>
A reviewer can read 260609-aqy-ANALYSIS.md and, without re-reading the code, know:
(1) whether to adopt bytemuck (and why), (2) whether to adopt arrow-rs at the boundary
(and why), (3) the ranked list of real boundary-copy opportunities with parity verdicts,
and (4) that none of it threatens the bit-exact / ~1e-6 parity gate. Estimates are framed
honestly against the L3 "boundary is not the bottleneck" finding.
</success_criteria>

<output>
Create `.planning/quick/260609-aqy-gpu-boundary-arrow-bytemuck-opt/260609-aqy-SUMMARY.md` when done.

Commit rules: the analysis doc (260609-aqy-ANALYSIS.md) is committed atomically by the
executor (it is the deliverable, a code-tree doc under .planning/quick). Do NOT commit
PLAN.md, SUMMARY.md, or STATE.md — the orchestrator handles the docs commit.
</output>
