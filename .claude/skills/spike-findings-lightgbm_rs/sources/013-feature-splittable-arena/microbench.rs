// Spike 013 microbench — extracted from crates/lgbm-treelearner/src/learner.rs (mod spike013).
// Run: cargo test -p lgbm-treelearner --release --lib spike013_feature_splittable -- --ignored --nocapture

mod spike013 {
    //! Spike 013 — is the per-tree `feature_splittable = vec![vec![true; nf]; nl]`
    //! bool matrix (`learner.rs:~891`) worth flattening / reusing? It is the same
    //! `vec![template; n]` clone-memcpy pattern as the histogram pool (spike 010),
    //! but ~1.5KB instead of multi-MB. This isolates its per-tree construction cost
    //! as a fraction of per-tree train time. Run:
    //!   cargo test -p lgbm-treelearner --release --lib spike013_feature_splittable -- --ignored --nocapture
    use std::time::Instant;

    #[test]
    #[ignore]
    fn spike013_feature_splittable() {
        // (name, num_leaves, num_features, per-tree wall ns from bench_train)
        let shapes = [
            ("small ", 31usize, 12usize, 266_000.0f64),
            ("medium", 31, 30, 1_320_000.0),
            ("large ", 31, 50, 4_400_000.0),
        ];
        let trees = 500usize;
        let warm = 50usize;
        let med = |v: &mut Vec<std::time::Duration>| {
            v.sort();
            v[v.len() / 2]
        };
        eprintln!("spike013 feature_splittable per-tree construction (median of {trees}):");
        for (name, nl, nf, per_tree_ns) in shapes {
            let mut sink = 0usize;

            // (1) CURRENT: vec![vec![true; nf]; nl] — nl allocs + clone-memcpy.
            let mut t_cur = Vec::with_capacity(trees);
            for i in 0..(trees + warm) {
                let s = Instant::now();
                let m: Vec<Vec<bool>> = vec![vec![true; nf]; nl];
                let d = s.elapsed();
                sink += m[i % nl][i % nf] as usize;
                if i >= warm {
                    t_cur.push(d);
                }
            }

            // (2) FLAT: one vec![true; nl*nf] arena.
            let mut t_flat = Vec::with_capacity(trees);
            for i in 0..(trees + warm) {
                let s = Instant::now();
                let m: Vec<bool> = vec![true; nl * nf];
                let d = s.elapsed();
                sink += m[i % (nl * nf)] as usize;
                if i >= warm {
                    t_flat.push(d);
                }
            }

            // (3) REUSE: allocate once, per tree just reset to true (fill).
            let mut arena = vec![true; nl * nf];
            let mut t_reuse = Vec::with_capacity(trees);
            for i in 0..(trees + warm) {
                let s = Instant::now();
                arena.iter_mut().for_each(|b| *b = true);
                let d = s.elapsed();
                sink += arena[i % (nl * nf)] as usize;
                if i >= warm {
                    t_reuse.push(d);
                }
            }

            std::hint::black_box(sink);
            let cur = med(&mut t_cur);
            let flat = med(&mut t_flat);
            let reuse = med(&mut t_reuse);
            let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() * 1e9 / per_tree_ns;
            eprintln!(
                "  {name} ({nl}×{nf}): cur={:>8.3?} ({:.3}% of tree)  flat={:>8.3?}  reuse={:>8.3?}  | ceiling={:.3}% of per-tree",
                cur, pct(cur), flat, reuse, pct(cur) - pct(reuse),
            );
        }
    }
}

#[cfg(test)]
