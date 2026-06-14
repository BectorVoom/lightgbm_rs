#!/usr/bin/env python3
"""Emit the SAME deterministic identity-binned 'small' corpus the Rust bench uses
(rows=2000, feat=12, bins=32) as a headerless CSV (label first column) for the
LightGBM CLI. Recipe mirrors bench_crossover.rs::make_corpus exactly."""
import sys

ROWS, FEAT, BINS = 2000, 12, 32
MASK = (1 << 64) - 1


def val(row, f):
    if row < BINS:
        return (row + f) % BINS
    h = (row * 2_654_435_761) & MASK
    h = (h + ((f * 40_503 + 0x9E37_79B9) & MASK)) & MASK
    return h % BINS


out = sys.argv[1] if len(sys.argv) > 1 else "small.csv"
with open(out, "w") as fh:
    for row in range(ROWS):
        acc = 0.0
        vals = []
        for f in range(FEAT):
            v = val(row, f)
            vals.append(v)
            acc += v * (1.0 + (f % 5))
        label = acc * 0.01
        fh.write(",".join([f"{label:.6f}"] + [str(v) for v in vals]) + "\n")
print(f"wrote {out}: {ROWS}x{FEAT}, bins={BINS}")
