---
spike: 031
name: crossfeature-gradhess-reuse
type: standard
validates: "Given a memory-bound wide build, when grad/hess are staged once and reused across features packed per cube, then device read-traffic and build time fall bit-exactly"
verdict: CLOSED
related: [030, 009, 028]
tags: [performance, gpu, rocm, histogram, build, gradhess-reuse, coalesce, read-once, marginal-roi, closed-by-030, not-built]
---

# Spike 031: Cross-Feature Grad/Hess Reuse — CLOSED (answered by spike-030, not built)

## What This Was

Gated on spike-030 showing the wide build memory-bound: stage grad/hess once in LDS/registers
and reuse them across several features packed per cube (re-testing spike-009's *pre-u64* null
now that the bottleneck had moved), to cut the cross-feature redundant grad/hess read traffic.

## Why It's Closed (decision 2026-06-25)

**Spike-030's "remove-the-suspect" roofline re-attribution answered this without a separate
build, on two counts:**

1. **Original premise INVALIDATED.** grad/hess global reads are only **8–14%** of the wide
   build (030's CONST_GH variant). They are NOT the bottleneck, so reusing them across features
   cannot move the clock. The premise — inherited from the pre-u64 "build is the wide cost"
   framing — was wrong about *which* read dominates.

2. **The real bottleneck (uncoalesced bin gather, 86–95%) has only ~1.4× headroom, and it's
   unamortizable.** 030's REAL_ORDER variant showed LightGBM's STABLE partition already gives
   monotone-increasing `leaf_rows`, reaching **~70% of the coalesced ceiling**. Any coalescing
   scheme either (a) adds a reorder PASS that can't amortize — the build reads each bin ONCE
   per leaf and the stable order changes every split (the same read-once wall that killed CPU
   double-buffering, spike-028), or (b) full-scans in natural order with a membership mask,
   reading skipped rows at coalesced speed, which only breaks even below ~1/5 selectivity — the
   deep, cheap leaves — while LOSING on the shallow high-row leaves that dominate build time.
   Net ≈ null-to-slight-loss on this APU.

## Disposition

NOT BUILT. The live build is effectively tuned on the spoofed 8-CU APU (the GPU still loses to
the 16-core CPU anchor everywhere — ROCm-parity-track). The ONE place this reopens:
**discrete gfx110x**, where the uncoalesced penalty is harsher (GDDR6, no shared-DDR5 cache)
and the random→monotone gap may widen. **Re-run `examples/spike030_build_roofline_ab.rs` on
real discrete hardware first**; only if REAL_ORDER sits well below the coalesced ceiling there
is a coalesced-build (pre-ordered bins fused into partition, or membership-mask full-scan)
worth prototyping. See spike-030 for the full evidence.
