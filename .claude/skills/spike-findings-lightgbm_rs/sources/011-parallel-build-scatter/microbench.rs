// Spike 011 microbench — extracted from crates/lgbm-compute/src/lib.rs (mod par_build_tests).
// Lives in-crate because it needs the private fold_one_feature + slot layout.
// Run: cargo test -p lgbm-compute --release --lib spike011_microbench -- --ignored --nocapture

    /// Spike 011 microbench — isolates the parallel build's `Vec<Vec<f64>>`+copy
    /// strategy (BEFORE) against the disjoint-slot scatter (AFTER, the live
    /// `build_histograms_into` parallel branch) on ONE representative large leaf,
    /// many launches, same process. Ignored by default; run with:
    ///   cargo test -p lgbm-compute --release --lib spike011_microbench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn spike011_microbench() {
        use super::fold_one_feature;
        use std::time::Instant;

        // Representative large leaf: a big leaf is where the parallel path fires.
        // Sweep via env (default 1M); 16384 is the live parallel threshold.
        let rows: u32 = std::env::var("SPIKE011_ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
        let nfeat = 50usize;
        let bins = 256u32;
        let cols: Vec<BinColumn> = (0..nfeat as u32)
            .map(|f| {
                let v: Vec<u32> = (0..rows)
                    .map(|r| {
                        let h = (r as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
                        (h % bins as u64) as u32
                    })
                    .collect();
                BinColumn::new(v, bins)
            })
            .collect();
        let num_bins: Vec<u32> = vec![bins; nfeat];
        let mut slot_off = Vec::new();
        let mut off = 0usize;
        for &nb in &num_bins {
            slot_off.push(off);
            off += 2 * nb as usize;
        }
        let slot_len = off;
        // Whole leaf, scattered order (worst-case gather, like a real leaf).
        let leaf_rows: Vec<u32> = (0..rows).map(|i| i.wrapping_mul(2_654_435_761) % rows).collect();
        let ord_g: Vec<f32> = (0..rows).map(|i| (i % 13) as f32 * 0.1).collect();
        let ord_h: Vec<f32> = (0..rows).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
        let refs: Vec<&BinColumn> = cols.iter().collect();

        // BEFORE (the LIVE production strategy): per-feature private `Vec<f64>`
        // accumulators + a sequential reassembly copy.
        let before = |out: &mut [f64]| {
            use rayon::prelude::*;
            let hists: Vec<Vec<f64>> = (0..refs.len())
                .into_par_iter()
                .map(|fpos| {
                    let mut h = vec![0.0f64; 2 * num_bins[fpos] as usize];
                    fold_one_feature(refs[fpos], &leaf_rows, &ord_g, &ord_h, &mut h);
                    h
                })
                .collect();
            for (fpos, h) in hists.into_iter().enumerate() {
                out[slot_off[fpos]..slot_off[fpos] + h.len()].copy_from_slice(&h);
            }
        };
        // AFTER (the REJECTED scatter): carve `out` into disjoint per-feature slots
        // and fold each feature straight into its slot (no intermediate, no copy).
        let after = |out: &mut [f64]| {
            use rayon::prelude::*;
            let mut slots: Vec<&mut [f64]> = Vec::with_capacity(refs.len());
            let mut rest = &mut out[..];
            let mut base = 0usize;
            for (fpos, &nb) in num_bins.iter().enumerate() {
                let (_gap, after_gap) = rest.split_at_mut(slot_off[fpos] - base);
                let (slot, tail) = after_gap.split_at_mut(2 * nb as usize);
                slots.push(slot);
                rest = tail;
                base = slot_off[fpos] + 2 * nb as usize;
            }
            slots.into_par_iter().enumerate().for_each(|(fpos, slot)| {
                fold_one_feature(refs[fpos], &leaf_rows, &ord_g, &ord_h, slot);
            });
        };

        let launches = 30usize;
        let warm = 5usize;
        // Interleave the two strategies per launch to cancel thermal/scheduler drift.
        let mut t_before = Vec::with_capacity(launches);
        let mut t_after = Vec::with_capacity(launches);
        let mut sink = 0.0f64;
        for i in 0..(launches + warm) {
            let mut out_b = vec![0.0f64; slot_len];
            let s0 = Instant::now();
            before(&mut out_b);
            let d_b = s0.elapsed();
            sink += out_b[0] + out_b[slot_len - 1];

            let mut out_a = vec![0.0f64; slot_len];
            let s1 = Instant::now();
            after(&mut out_a);
            let d_a = s1.elapsed();
            sink += out_a[0] + out_a[slot_len - 1];

            // Byte-equality spot check (same fold order ⇒ identical).
            assert_eq!(out_b[0].to_bits(), out_a[0].to_bits());
            assert_eq!(out_b[slot_len - 1].to_bits(), out_a[slot_len - 1].to_bits());

            if i >= warm {
                t_before.push(d_b);
                t_after.push(d_a);
            }
        }
        std::hint::black_box(sink);
        t_before.sort();
        t_after.sort();
        let med = |v: &[std::time::Duration]| v[v.len() / 2];
        let mb = med(&t_before);
        let ma = med(&t_after);
        let ratio = mb.as_secs_f64() / ma.as_secs_f64();
        eprintln!(
            "spike011 microbench ({rows} rows x {nfeat} feat x {bins} bins, {launches} launches, interleaved):\n  \
             LIVE   Vec<Vec>+copy (private accumulators): {mb:?}/build\n  \
             SCATTER into shared `out` (rejected)       : {ma:?}/build\n  \
             ratio live/scatter: {ratio:.3}x  (<1 ⇒ scatter SLOWER ⇒ keep Vec<Vec>)"
        );
    }
}
