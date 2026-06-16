// Spike 010 ceiling microbench — extracted from crates/lgbm-treelearner/src/histogram_pool.rs
// Run: cargo test -p lgbm-treelearner --release --lib spike010_pool_alloc_ceiling -- --ignored --nocapture

mod spike010 {
    //! Spike 010 — what is the per-tree CEILING for replacing the pool's
    //! `Vec<Vec<f64>>` buffers? `HistogramPool::new` runs once per tree
    //! (`learner.rs:816`): `num_leaves` slot allocations, each `slot_len` f64,
    //! zeroed then overwritten. This isolates that cost three ways at the bench
    //! shapes, in-process, so we measure the lever's ceiling (not a noisy
    //! end-to-end train). Run:
    //!   cargo test -p lgbm-treelearner --release --lib spike010_pool_alloc_ceiling -- --ignored --nocapture
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn spike010_pool_alloc_ceiling() {
        // (name, num_leaves, slot_len = Σ 2*num_bin, per-tree wall from bench_train)
        // small : 12 feat × 32 bins → 12*64=768 ; ~266 µs/tree
        // medium: 30 feat × 64 bins → 30*128=3840; ~1.32 ms/tree
        // large : 50 feat ×128 bins → 50*256=12800; ~4.40 ms/tree
        let shapes = [
            ("small ", 31i32, 768usize, 266_000.0f64),
            ("medium", 31, 3_840, 1_320_000.0),
            ("large ", 31, 12_800, 4_400_000.0),
        ];
        let trees = 200usize;
        let warm = 20usize;
        let med = |v: &mut Vec<std::time::Duration>| {
            v.sort();
            v[v.len() / 2]
        };
        eprintln!("spike010 pool-alloc ceiling ({trees}'trees', median; sink defeats DCE):");
        for (name, nl, sl, per_tree_ns) in shapes {
            let cells = nl as usize * sl;
            let mut sink = 0.0f64;

            // (1) CURRENT: Vec<Vec<f64>> — nl separate allocs + zero, every tree.
            let mut t_cur = Vec::with_capacity(trees);
            for i in 0..(trees + warm) {
                let s = Instant::now();
                let pool = HistogramPool::new(nl, sl);
                let d = s.elapsed();
                sink += pool.buffer(0)[0] + pool.buffer((nl - 1) as usize)[sl - 1];
                if i >= warm {
                    t_cur.push(d);
                }
            }

            // (2) FLAT ARENA: one `vec![0.0; nl*sl]` alloc + zero, every tree.
            let mut t_flat = Vec::with_capacity(trees);
            for i in 0..(trees + warm) {
                let s = Instant::now();
                let buf = vec![0.0f64; cells];
                let d = s.elapsed();
                sink += buf[0] + buf[cells - 1];
                if i >= warm {
                    t_flat.push(d);
                }
            }

            // (3) REUSED ARENA: allocate ONCE, per tree only the index-map reset
            // (buffers are overwritten by construct before read, so no zeroing).
            // This is the hoist-across-trees ideal: per-tree buffer cost → 0.
            let mut arena = vec![0.0f64; cells];
            let mut mapper = vec![-1i32; nl as usize];
            let mut t_reuse = Vec::with_capacity(trees);
            for i in 0..(trees + warm) {
                let s = Instant::now();
                for m in mapper.iter_mut() {
                    *m = -1;
                }
                let d = s.elapsed();
                // touch the arena like a real per-tree use would (1 cell) to keep it live
                arena[i % cells] = i as f64;
                sink += arena[0];
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
                "  {name}: cur(Vec<Vec>)={:>9.3?} ({:>4.1}% of tree)  flat={:>9.3?} ({:>4.1}%)  reuse={:>9.3?} ({:>4.1}%)  | ceiling(cur−reuse)={:>4.1}% of per-tree",
                cur, pct(cur), flat, pct(flat), reuse, pct(reuse),
                pct(cur) - pct(reuse),
            );
        }
    }
}
