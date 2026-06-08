---
quick_id: 260609-9nu
title: Investigate dataset for arrow-rs optimisation opportunities
status: complete
date: 2026-06-08
type: investigation
---

# Summary

Investigation-only task (no code changes). Deliverable: `260609-9nu-FINDINGS.md`.

**Key result:** `lgbm-dataset` contains no Apache Arrow / arrow-rs code (the
"arrow" grep hits are the word "narrow" in comments). The real Arrow boundary is
the polars C-stream ingest in `crates/lgbm-python`, which performs a wasteful
column→row→column double transpose over a nested `Vec<Vec<f64>>` layout.

**Top opportunities:** O1 eliminate the double transpose (let the columnar Arrow
data feed a column-oriented ingest), O2 flatten `Vec<Vec<f64>>` to a contiguous
strided buffer. Both benefit all Python ingest paths and are parity-neutral.

See FINDINGS.md for the ranked list (O1–O5) with file:line references.
