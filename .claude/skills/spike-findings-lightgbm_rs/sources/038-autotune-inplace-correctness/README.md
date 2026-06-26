---
spike: 038
name: autotune-inplace-correctness
type: standard
validates: "Given an accumulating histogram build kernel (+ in-place partition), when the tuner re-benchmarks each variant N times, then the output is corrupted unless a fresh/reset InputGenerator is interposed — and parity holds after the fix"
verdict: VALIDATED
related: [037, 035, 027]
tags: [gpu, rocm, autotune, correctness, in-place, input-generator, kill-question]
---

# Spike 038: Autotune In-Place Correctness (the 2nd kill question)

## What This Validates

The autotune manual (§3 NOTE) warns that benchmarking an in-place-mutating kernel
"will accumulate values repeatedly during the cold run." Our histogram BUILD
(`fetch_add` into a resident `out`) and partition (`indices` read-modify-write) are exactly
that hazard. This spike asks: **does it actually corrupt the result, and is there a clean,
parity-preserving fix?** If autotune can only run on stateless kernels, it can't wrap our
hot path as-is.

## Research / Mechanism

Read from the real 0.10 source (not the manual):
- `tune/tuner.rs:183` — during tuning, `let test_inputs = tunables.generate_inputs(key, inputs)`;
  every benchmark rep of every variant runs on `test_inputs.clone()` (`:207`).
- `tune/local.rs:170` — AFTER tuning, `execute` runs the WINNING variant **once** on the
  **original** `inputs`.
- `CloneInputGenerator` returns the SAME device handles (a cubecl `Handle::clone` is a
  ref-count bump, **not** a buffer copy). So with it, every benchmark launch `fetch_add`s
  into the caller's REAL `out` ⇒ the buffer holds (Σ benchmark launches) × the histogram.

**Correctness metric — grad conservation (order-independent):** each feature accumulates
every row's grad exactly once, so a correct histogram has
`Σ(grad cells) == num_features × Σ(ord_g)`. f32-atomic reorder noise doesn't move this sum,
so it cleanly separates "correct" from "accumulated N×" without any tolerance games.

## How to Run

```bash
cargo run --release --features rocm --example spike038_autotune_inplace_correctness
```

## What to Expect

Two arms, same two-variant TunableSet, differing only in the InputGenerator:
- **[A] CloneInputGenerator** (the manual's pattern) → total grad ≫ 1× expected = CORRUPTED.
- **[B] FreshOutGenerator** (the fix) → total grad == 1.0000× expected = CORRECT.

## Investigation Trail

1. **Wrote the fix as a real `InputGenerator` impl** (`FreshOutGenerator`), not a closure —
   it captures the client + slot_len and, on each `generate`, returns a NEW `Vec<Handle>`
   with index 5 (`out`) replaced by a freshly-allocated zeroed device buffer. The benchmark
   reps hammer the throwaway; the real `out` stays pristine until the final clean run.
   - API gotcha (1 rebuild): `InputGenerator::generate<'a>` is a GAT method
     (`-> I::At<'a>`). The impl signature MUST spell the return through
     `<Vec<Handle> as TuneInputs>::At<'a>` so `'a` is actually used, else E0195
     "lifetimes do not match" (a bare `-> Vec<Handle>` leaves `'a` unconstrained).
2. **Ran both arms on the device** (rows=200k, feats=50, Σord_g ⇒ expected total grad = −300):

   | Arm | InputGenerator | total grad | ratio | verdict |
   |-----|----------------|-----------:|------:|---------|
   | A | `CloneInputGenerator` | −8100.0 | **27.0×** | CORRUPTED |
   | B | `FreshOutGenerator` | −300.0 | **1.0000×** (rel_err 0) | CORRECT |

   The **27×** is the total benchmark-launch count across both variants accumulated into the
   real buffer — corroborates the spike-037 "max grad-cell" inflation, and shows the
   corruption magnitude is *non-deterministic in general* (it tracks however many samples
   the tuner decides to run).
3. **Generalized the classification** (see Results) — not every kernel is at risk.

## Results

**VERDICT: VALIDATED.** The hazard is real (27× corruption with the manual's pattern) AND
has a clean, parity-exact fix: a **fresh-output `InputGenerator`**. After the fix the real
`out` is touched exactly once and matches a clean single histogram to `rel_err 0` (grad
conservation), so autotune is correctness-safe for our accumulating build kernel.

**Kernel-safety classification (the carry-forward rule):**
| Kernel class | Example | Autotune-safe with `CloneInputGenerator`? |
|--------------|---------|-------------------------------------------|
| **Overwrites** output (`store`, not `+=`) | scan/split writing fresh slots | ✅ yes — re-running recomputes the same value |
| **Accumulates** into output (`fetch_add`) | histogram BUILD (`build_rp`) | ❌ no — needs a **fresh-output** InputGenerator (this spike) |
| **In-place read-modify-write** | partition `indices` routing | ❌ no — needs an InputGenerator that **deep-copies** the mutated buffer each `generate` (a fresh *zeroed* buffer is insufficient; you must clone the original contents) |

**On partition specifically:** it's the harder sub-case (the buffer is both source and
destination), but it's also **not a live GPU-autotune target** — spike-035 routes the rocm
partition on the HOST by default (the device round-trip was pure overhead). So the only
accumulating GPU kernel we'd actually autotune is the histogram build, for which
`FreshOutGenerator` is sufficient and proven here.

**Carry-forward requirement for the real build:** any TunableSet wrapping the histogram
build MUST use a fresh-output InputGenerator, never `CloneInputGenerator`. Bake this into
the wiring (and a debug-only grad-conservation assert is a cheap guard). The CPU f64 anchor
is untouched — this is rocm-only kernel plumbing.

**Surprise:** the corruption factor isn't a fixed "+1"; it's the *entire* benchmark sample
budget (27 here). A casual reading of the manual ("accumulate during the cold run") under-
sells it — it's not a small bias, it's a 27× wrong answer that silently varies with the
tuner's sampling.
