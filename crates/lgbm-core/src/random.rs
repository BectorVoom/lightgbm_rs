//! Bit-for-bit port of the C++ `Random` LCG from
//! `LightGBM/include/LightGBM/utils/random.h` (the entire class).
//!
//! This is the **single most parity-critical port in the foundation**: config
//! seed derivation, bagging, feature sampling, GOSS, and DART all inherit it.
//! Every constant and ordering detail is load-bearing.
//!
//! # Determinism / security
//!
//! `Random` is a **deterministic, NON-cryptographic** reproducibility PRNG (the
//! classic MSVC `rand` recurrence). It exists to reproduce the C++ reference
//! bit-for-bit and MUST NEVER be used as a source of secure randomness
//! (Security V6). Only the seeded constructor is ported; the C++ default
//! constructor draws from `std::random_device` and is not reproducible.
//!
//! # Arithmetic notes
//!
//! The state `x` is C++ `unsigned int` → Rust `u32`, advanced with
//! `wrapping_mul`/`wrapping_add` (the `214013 * x` multiply overflows u32 by
//! design). Porting over `i32`/`i64` would panic in debug builds and diverge.

use std::collections::BTreeSet;

/// A wrapper for a deterministic, non-cryptographic random generator (the
/// LightGBM `Random` LCG, ported bit-for-bit).
pub struct Random {
    /// LCG state. C++: `unsigned int x = 123456789;`
    x: u32,
}

impl Random {
    /// Construct with a specific seed.
    ///
    /// C++: `explicit Random(int seed) { x = seed; }`. The `seed as u32`
    /// reinterpretation matches C++ assigning a signed `int` into an
    /// `unsigned int`.
    pub fn new(seed: i32) -> Self {
        Random { x: seed as u32 }
    }

    /// C++ `RandInt16`: `x = (214013 * x + 2531011); return (x >> 16) & 0x7FFF;`
    ///
    /// Returns a 15-bit non-negative value in `[0, 32767]`.
    fn rand_int16(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
        ((self.x >> 16) & 0x7FFF) as i32
    }

    /// C++ `RandInt32`: `x = (214013 * x + 2531011); return x & 0x7FFFFFFF;`
    ///
    /// Returns a 31-bit non-negative value.
    fn rand_int32(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
        (self.x & 0x7FFF_FFFF) as i32
    }

    /// C++ `NextShort`: `RandInt16() % (upper - lower) + lower`.
    ///
    /// Random integer in `[lower, upper)` (int16-derived).
    pub fn next_short(&mut self, lower: i32, upper: i32) -> i32 {
        self.rand_int16() % (upper - lower) + lower
    }

    /// C++ `NextInt`: `RandInt32() % (upper - lower) + lower`.
    ///
    /// Random integer in `[lower, upper)` (int32-derived).
    pub fn next_int(&mut self, lower: i32, upper: i32) -> i32 {
        self.rand_int32() % (upper - lower) + lower
    }

    /// C++ `NextFloat`: `static_cast<float>(RandInt16()) / 32768.0f`.
    ///
    /// Random float in `[0.0, ~0.99997)`. Stays `f32` end-to-end; the divisor
    /// is exactly `32768.0` (NOT `65536`, NOT computed in `f64`).
    pub fn next_float(&mut self) -> f32 {
        (self.rand_int16() as f32) / 32768.0_f32
    }

    /// C++ `Sample(N, K)`: sample `K` ordered values from `{0, 1, ..., N-1}`.
    ///
    /// Replicates the four-way branch exactly:
    /// - `K > N || K <= 0` → empty
    /// - `K == N` → full range `0..N`
    /// - `K > 1 && K > (N / log2(K))` (computed in `f64`) → streaming branch
    /// - otherwise → ordered-set branch (C++ `std::set` → [`BTreeSet`]), with
    ///   the collision-reinsert rule.
    pub fn sample(&mut self, n: i32, k: i32) -> Vec<i32> {
        let mut ret: Vec<i32> = Vec::new();
        if k > n || k <= 0 {
            return ret;
        } else if k == n {
            ret.reserve(k as usize);
            for i in 0..n {
                ret.push(i);
            }
        } else if k > 1 && (k as f64) > (n as f64 / (k as f64).log2()) {
            ret.reserve(k as usize);
            for i in 0..n {
                // C++: double prob = (K - ret.size()) / static_cast<double>(N - i);
                let prob = (k - ret.len() as i32) as f64 / (n - i) as f64;
                if (self.next_float() as f64) < prob {
                    ret.push(i);
                }
            }
        } else {
            let mut sample_set: BTreeSet<i32> = BTreeSet::new();
            for r in (n - k)..n {
                let v = self.next_int(0, r + 1);
                // C++: if (!sample_set.insert(v).second) sample_set.insert(r);
                if !sample_set.insert(v) {
                    sample_set.insert(r);
                }
            }
            ret.reserve(sample_set.len());
            for &v in &sample_set {
                ret.push(v);
            }
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference RandInt16 computed independently from the C++ recurrence.
    fn ref_rand_int16(x: &mut u32) -> i32 {
        *x = x.wrapping_mul(214013).wrapping_add(2531011);
        ((*x >> 16) & 0x7FFF) as i32
    }

    #[test]
    fn first_rand_int16_matches_cpp_recurrence() {
        // x = 214013*123456789 + 2531011 over u32 wraparound, then (x>>16)&0x7FFF.
        let mut x: u32 = 123456789;
        let expected = ref_rand_int16(&mut x);

        let mut rng = Random::new(123456789);
        // rand_int16 is private; exercise via next_short(0, 32768) which is
        // rand_int16() % 32768 + 0 == rand_int16() (value < 32768).
        let got = rng.next_short(0, 32768);
        assert_eq!(got, expected);
        assert!((0..=32767).contains(&got));
    }

    #[test]
    fn next_float_is_f32_in_range() {
        let mut rng = Random::new(42);
        for _ in 0..10_000 {
            let f = rng.next_float();
            assert!(f >= 0.0, "next_float must be >= 0.0, got {f}");
            assert!(f <= 0.99997, "next_float must be <= 0.99997, got {f}");
        }
    }

    #[test]
    fn next_float_matches_rand_int16_over_32768() {
        // Independently recompute: float(RandInt16()) / 32768.0f.
        let mut x: u32 = 7u32; // seed 7
        let mut rng = Random::new(7);
        for _ in 0..1000 {
            let expected = (ref_rand_int16(&mut x) as f32) / 32768.0_f32;
            let got = rng.next_float();
            assert_eq!(got.to_bits(), expected.to_bits(), "exact-bit f32 mismatch");
        }
    }

    #[test]
    fn next_int_within_range() {
        let mut rng = Random::new(99);
        for _ in 0..1000 {
            let v = rng.next_int(5, 17);
            assert!((5..17).contains(&v), "next_int out of range: {v}");
        }
    }

    #[test]
    fn sample_edge_cases() {
        let mut rng = Random::new(1);
        assert_eq!(rng.sample(10, 0), Vec::<i32>::new()); // K <= 0
        assert_eq!(rng.sample(10, 11), Vec::<i32>::new()); // K > N
        assert_eq!(rng.sample(5, 5), vec![0, 1, 2, 3, 4]); // K == N
    }

    #[test]
    fn sample_streaming_branch_selected_above_threshold() {
        // For N=100, K=50: 50 > 100/log2(50)=100/5.64=17.7 → streaming branch.
        // Verify output is ordered, unique, in range, and K-ish (streaming is
        // probabilistic so size may differ; just assert structural invariants).
        let mut rng = Random::new(123);
        let out = rng.sample(100, 50);
        assert!(out.windows(2).all(|w| w[0] < w[1]), "must be strictly ascending");
        assert!(out.iter().all(|&v| (0..100).contains(&v)));
    }

    #[test]
    fn sample_set_branch_selected_below_threshold() {
        // For N=1000, K=3: 3 > 1000/log2(3)=1000/1.585=631 is FALSE → set branch.
        // Set branch yields exactly K ordered unique values.
        let mut rng = Random::new(456);
        let out = rng.sample(1000, 3);
        assert_eq!(out.len(), 3, "set branch yields exactly K values");
        assert!(out.windows(2).all(|w| w[0] < w[1]), "ascending & unique");
        assert!(out.iter().all(|&v| (0..1000).contains(&v)));
    }

    #[test]
    fn sample_branch_boundary_arithmetic() {
        // Sanity-check the boundary condition computation itself for a pair
        // straddling the threshold. N=64, K=8: log2(8)=3, 64/3=21.33, 8>21.33
        // is FALSE → set branch (exactly K). N=64, K=32: log2(32)=5, 64/5=12.8,
        // 32>12.8 TRUE → streaming.
        let mut rng = Random::new(789);
        let set_side = rng.sample(64, 8);
        assert_eq!(set_side.len(), 8);
        let mut rng2 = Random::new(789);
        let stream_side = rng2.sample(64, 32);
        assert!(stream_side.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn lcg_does_not_panic_on_overflow_seed() {
        // Large seed exercises the u32 wrapping multiply; must not panic in debug.
        let mut rng = Random::new(i32::MAX);
        let _ = rng.next_short(0, 32768);
        let _ = rng.next_int(0, 1000);
        let _ = rng.next_float();
    }
}
