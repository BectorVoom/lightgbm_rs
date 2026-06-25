---
spike: 032
name: partition-validation-fold
type: standard
validates: "Given the shipped spike-027 host partition (DataPartition::split_fused_host) does a REDUNDANT validation random-gather plus the route random-gather over the leaf, when the validation is folded into pass-1 (V1) or relocated once-per-train (V2), then the partition op runs faster across size×skew×bin-width — byte-identical [left|right] every cell"
verdict: VALIDATED
related: [027, 026, 028, 003]
tags: [performance, cpu, partition, memory-traffic, redundant-gather, validation, bit-exact, isolated-ab, narrow-bins]
---

# Spike 032: Eliminate the Redundant Validation Gather in `split_fused_host`

## What This Validates

Given the SHIPPED spike-027 host partition (`DataPartition::split_fused_host`,
`crates/lgbm-treelearner/src/data_partition.rs:206`), when the redundant validation
random-gather is eliminated — by folding the bin range-check into pass-1's route gather
(**V1 fold**) or relocating it once-per-train (**V2 relocate**) — then the partition op
runs faster across size×skew×bin-width, **byte-identical** `[left|right]` every cell.

This attacks the **#1 CPU-vs-C++ gap** (single-threaded `DataPartition::split`,
~29% of tall-narrow train) on the live "cut TRAFFIC, not add cores" lever class that
026→027 established (and that 026's rayon-null / 028's double-buffer-null closed the
alternatives to).

## The bug, found by reading the shipped code

Spike-027's experiment validated a **one-random-gather** fused partition
(`v1_fused_u8route`). But the PRODUCTION wiring of `split_fused_host` does **TWO** random
gathers over the leaf's scattered rows:

1. **Validation pass** (`data_partition.rs:236-246`): `for i in 0..count { feature_bins.bin(row) }`
   range-checks every leaf row's bin and surfaces the lowest-index offender.
2. **Pass-1 route+count** (`data_partition.rs:270-275`): `feature_bins.bin(row)` **again**.

On a memory-bandwidth/latency-bound op, the second random gather over a column that
exceeds cache re-misses for every row — exactly the traffic the whole 026→027 arc fought
to cut. The validation re-read was added for parity with the native op's error semantics;
it is removable two ways, both bit-exact.

## Research

No external deps — pure host logic against the real `BinColumn`. Method follows the
CONVENTIONS "CPU / host isolated-A/B harness (026–029)": self-contained
`crates/lgbm-compute/examples/spike032_*.rs`, deterministic LCG, **scattered** leaf
(shuffled row ids ⇒ random gather), sweep size (16k→4M) × skew (0.0/0.9) × bin width
(U8 production / U32), median of 25 interleaved reps + warmup discard, ≥2 process
restarts, byte-identity parity column every cell.

Variants (user choice: "try both, A/B them"):

| Variant | Validation | Gathers | Bit-exact to V0 |
|---------|-----------|---------|-----------------|
| **V0 shipped** | separate pass | **2** | baseline |
| **V1 fold** | folded into pass-1 (early-return before any `indices` mutation) | **1** + branch | yes (success AND error: same lowest-index, no mutation) |
| **V2 relocate** | none per-split (once-per-train; the spike-003b/r4o precedent, = C++) | **1** | yes (valid bins) |

## How to Run

```
cargo run -p lgbm-compute --example spike032_partition_validation_fold_ab --release
# >=2 restarts: LGBM_SPIKE_RUN=2 …, LGBM_SPIKE_RUN=3 …
```

## What to Expect

Per (rows × skew × width): V0/V1/V2 median ms + `v0/v1`, `v0/v2` ratios (>1 ⇒ faster) +
a `parity` column. Expect a regime-split: null-to-slight-loss at cache-resident small
leaves (2nd gather is free there), a growing SEP-WIN as the column exceeds cache.

## Investigation Trail

- Read `split_fused_host` end-to-end; found the validation loop (`:236-246`) is a SECOND
  full random gather, distinct from the spike-027 design that measured ONE.
- Built V0 (faithful shipped replica incl. the validation loop), V1 (fold), V2 (relocate
  = the original 027 `v1_fused_u8route`). All `#[inline(never)]` so the timed bodies are
  not merged/DCE'd; `black_box` sinks.
- 3 process restarts (LGBM_SPIKE_RUN=1/2/3). Sign-stable; parity OK every cell.

## Results

**VERDICT: VALIDATED — the shipped partition pays a redundant validation random-gather;
removing it is a sign-stable, bit-exact win that grows with leaf size and skew. Biggest
where partition cost actually concentrates: large scattered leaves (the root/shallow
splits).**

Representative ratios (median across 3 restarts; `v0/v2` = remove-validation speedup):

| rows | skew | width | v0/v1 (fold) | v0/v2 (relocate) |
|------|------|-------|--------------|-------------------|
| 16,384 | 0.0 | 8 | ~0.88× (loss) | ~0.88× (loss) |
| 100,000 | 0.9 | 8 | ~1.32× | ~1.22× |
| 1,000,000 | 0.9 | 8 | ~1.41× | ~1.38× |
| 4,000,000 | 0.0 | 8 | ~1.17× | ~1.20× |
| 4,000,000 | 0.9 | 8 | ~1.38× | ~1.38× |
| 1,000,000 | 0.9 | 32 | ~1.30× | ~1.37× |
| 4,000,000 | 0.9 | 32 | ~1.58× | **~1.82×** |

Reading:
- **Production U8 width:** ~1.14–1.21× balanced / ~1.2–1.4× skewed at ≥1M rows.
  Skewed data wins MORE — the route/scatter is cheaper there, so the gather is a larger
  fraction, so deleting one gather matters more.
- **Wider U32 columns** exceed cache sooner ⇒ bigger win (up to ~1.8× at 4M skewed).
- **Small cache-resident leaves (16k balanced):** ~0.88× — the 2nd gather hits L1/L2 (no
  re-miss to delete) and V1/V2's per-call alloc/branch costs show. Absolutely tiny
  (~0.05 ms); deep small leaves are cheap and NOT where partition time concentrates.
- **V1 fold ≈ V2 relocate at U8** (the production width); V2 only pulls ahead at U32/4M
  (no per-row branch). ⇒ **fold captures the full production win while KEEPING the
  defensive range-check** (and its bit-exact lowest-index error contract).

### Honest end-to-end (cold-isolated overstates warm — CONVENTIONS)

The isolated op is a fraction of train (partition ≈ 23–29% of tall-narrow / GPU train per
023; less at wide). Cutting it ~1.2–1.4× at the dominant large-leaf shape ⇒ a few-% e2e
on the tall-narrow CPU gap — a FREE, bit-exact reclaim on the campaign's #1 residual, not
a headline multiplier. Real leaves at the root split (~all rows, scattered) sit squarely
in the win regime; the small-leaf slight-loss is absolutely negligible.

### Disposition — WIRE candidate (human call)

**Recommend V1 fold** for production: it deletes the redundant gather, keeps the
per-split defensive validation at ~zero extra cost (one gather + one branch), is
bit-exact on BOTH the success and the error path (early-return before any `indices`
mutation ⇒ same lowest-index offender as V0), and ties V2 at the production U8 width.
V2 relocate is marginally faster at wide U32 but needs a once-per-train validation pass
to preserve the error contract (larger blast radius). Wiring is a follow-on `/gsd-quick`:
edit `split_fused_host` to fold the `:236-246` check into the `:270-275` loop; re-run the
bit-exact gate (`lgbm-treelearner --lib` incl. `split_fused_equals_serial` + oracle
`raw_bin_train_parity`).

Bit-exact gate: N/A for the spike itself (probe-only example; live kernel untouched).
The wire is the gated change.
