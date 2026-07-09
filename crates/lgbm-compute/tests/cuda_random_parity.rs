//! Device `CUDARandom` LCG vs host `Random` bit-exact parity (Phase 14 Plan 04,
//! ODL-02 / D-04).
//!
//! ## Anchor discipline (D-04 / D-10)
//! The device draw stream is asserted **bit-for-bit** against the existing,
//! already-C++-bit-exact host `Random` (`lgbm_core::random::Random`) — NEVER
//! GPU-vs-GPU. `RandInt16`/`RandInt32` via exact `i32` equality, `NextFloat` via
//! `f32::to_bits()` equality (no float tolerance — RNG draws are exact-bit). This
//! test IS the end-to-end verification of RESEARCH Assumption A2 (plain `u32`
//! `*`/`+` wraps two's-complement on the device, giving a bit-identical stream).
//!
//! The host `Random` exposes `next_short`/`next_float` publicly; `next_short(0,
//! 32768) == RandInt16()` (value < 32768, so the modulus is the identity — the
//! exact trick the host's own unit tests use, `random.rs:142`). The private
//! `RandInt32` is mirrored by a local recurrence reference (`ref_rand_int32`,
//! mirroring `random.rs`'s own `ref_rand_int16` test helper) — the same
//! `214013·x + 2531011` recurrence the host advances.

use lgbm_compute::kernels::random::{draw_next_float_on, draw_rand_int16_on, draw_rand_int32_on};
use lgbm_compute::runtime::cpu_client;
use lgbm_core::random::Random;

/// Local independent reference for the host's private `RandInt32` — the same
/// `214013·x + 2531011` recurrence, masked to 31 bits.
fn ref_rand_int32(x: &mut u32) -> i32 {
    *x = x.wrapping_mul(214013).wrapping_add(2531011);
    (*x & 0x7FFF_FFFF) as i32
}

/// The seeds under test, including edge cases: zero, small, large (exercises the
/// u32 wrapping multiply), and a negative i32 reinterpreted as u32 (matches
/// `Random::new(seed) { x: seed as u32 }`).
fn test_seeds() -> Vec<i32> {
    vec![0, 1, 7, 42, 123_456_789, i32::MAX, -1, -12345]
}

const K: usize = 64;

#[test]
fn device_rand_int16_matches_host_random() {
    let client = cpu_client();
    let seeds = test_seeds();
    let states: Vec<u32> = seeds.iter().map(|&s| s as u32).collect();

    let device = draw_rand_int16_on(&client, &states, K as u32).unwrap();
    assert_eq!(device.len(), seeds.len() * K);

    for (t, &seed) in seeds.iter().enumerate() {
        let mut host = Random::new(seed);
        for j in 0..K {
            // next_short(0, 32768) == RandInt16() (value < 32768).
            let expected = host.next_short(0, 32768);
            assert_eq!(
                device[t * K + j],
                expected,
                "RandInt16 mismatch seed={seed} draw={j}"
            );
        }
    }
}

#[test]
fn device_rand_int32_matches_host_recurrence() {
    let client = cpu_client();
    let seeds = test_seeds();
    let states: Vec<u32> = seeds.iter().map(|&s| s as u32).collect();

    let device = draw_rand_int32_on(&client, &states, K as u32).unwrap();
    assert_eq!(device.len(), seeds.len() * K);

    for (t, &seed) in seeds.iter().enumerate() {
        let mut x = seed as u32;
        for j in 0..K {
            let expected = ref_rand_int32(&mut x);
            assert_eq!(
                device[t * K + j],
                expected,
                "RandInt32 mismatch seed={seed} draw={j}"
            );
        }
    }
}

#[test]
fn device_next_float_matches_host_random_to_bits() {
    let client = cpu_client();
    let seeds = test_seeds();
    let states: Vec<u32> = seeds.iter().map(|&s| s as u32).collect();

    let device = draw_next_float_on(&client, &states, K as u32).unwrap();
    assert_eq!(device.len(), seeds.len() * K);

    for (t, &seed) in seeds.iter().enumerate() {
        let mut host = Random::new(seed);
        for j in 0..K {
            let expected = host.next_float();
            assert_eq!(
                device[t * K + j].to_bits(),
                expected.to_bits(),
                "NextFloat to_bits mismatch seed={seed} draw={j}"
            );
        }
    }
}

#[test]
fn independent_seeds_produce_independent_streams() {
    // Two different seeds must yield different streams (sanity that the seed is
    // actually threaded into the device state, not ignored).
    let client = cpu_client();
    let a = draw_rand_int32_on(&client, &[1u32], 8).unwrap();
    let b = draw_rand_int32_on(&client, &[2u32], 8).unwrap();
    assert_ne!(a, b);
}

#[test]
fn zero_draws_and_empty_seeds_are_handled() {
    let client = cpu_client();
    assert_eq!(draw_rand_int16_on(&client, &[1u32], 0).unwrap(), Vec::<i32>::new());
    assert_eq!(draw_rand_int16_on(&client, &[], 4).unwrap(), Vec::<i32>::new());
}
