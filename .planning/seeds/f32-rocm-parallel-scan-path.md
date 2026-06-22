---
title: f32 parallel-atomic on-device build+scan for the ROCm backend
trigger_condition: when GPU wide-shape (≥250k rows × ≥100 feat) train perf becomes a priority, OR after the f32-scan spike confirms the win
planted_date: 2026-06-22
---

# Seed: f32 parallel-atomic on-device scan (ROCm)

> **⚠️ INVALIDATED by spike-015 (2026-06-22).** This seed assumed the bottleneck was a
> sequential-f64 build/scan switchable to f32. Spike-015 found the wide build ALREADY
> runs parallel f32-atomic and the real cost is that build's atomic-contention compute
> (86–92%), not precision. There is no "f32 scan path" lever to plant. Superseded by the
> spike-015 forward lever (finer per-warp LDS sub-histogram privatization) + the routing
> reality (CPU beats GPU ~4× on wide shapes). Kept for trail; do not promote.

## Idea

Replace the GPU resident scan's **sequential-f64** build+scan
(`build_fix_scan_resident` / `scan_resident_leaf`) with a **parallel-f32-atomic**
on-device build+scan on the `rocm` backend.

## Why this is planted, not built now

The profiler ([[gpu-bottleneck-moved-to-seq-f64-scan]]) shows SCAN is ~96% of GPU
train wall and 9.6× the CPU scan, because gfx1100 can't run the parallel-f64-atomic
path upstream's CUDA kernel assumes, and falls back to sequential f64. f32 is
user-sanctioned and within the ~1e-6 ROCm contract. But before committing a phase, the
spike [[spike-f32-parallel-atomic-on-device-scan]] must confirm:

1. how much of the 9.6× the f32 parallel-atomic path actually recovers, and
2. that parity vs the CPU f64 anchor stays within ~1e-6 on the wide shapes.

## Trigger

Promote to a phase when GPU wide-shape perf is prioritized, gated on a positive spike
result. If the spike shows the gain is small or parity blows past ~1e-6, this seed dies.

## Scope sketch (if promoted)

- New ROCm-only kernel: parallel f32 atomic histogram build + on-device argmax scan per leaf.
- Keep the CPU f64-fold path byte-identical (hard gate untouched).
- Parity test: GPU f32 scan vs CPU anchor ≤ ~1e-6 on 250k/500k/1M × 500.
- Watch the existing flaky-resident-hip lesson: pin GPU trees to the CPU anchor, never
  compare two nondeterministic GPU f32 paths to each other at 1e-6.
