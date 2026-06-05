//! RNG parity: replay the Rust `lgbm_core::random::Random` over EVERY committed
//! C++ golden case and assert bit-for-bit equality (FND-01, ORA-02, D-14).
//!
//! - Integer draws (NextShort / NextInt) are compared EXACTLY.
//! - `NextFloat` is compared by raw f32 bit pattern (exact-bit equality) — the
//!   `~1e-6` oracle tolerance applies to *algorithm* float outputs, NOT to RNG
//!   draws, which must be identical to the last bit.
//! - `Sample(N, K)` output is compared as an exact ordered sequence.
//!
//! This test reads ONLY the committed fixture; it needs NO C++ toolchain
//! (D-06). Until the human runs `cargo run -p xtask -- regen` and commits the
//! fixture, the file is absent and the test reports a skip (so `cargo test`
//! stays green pre-capture); once committed it validates every case.

use std::path::PathBuf;

use lgbm_core::random::Random;

/// Number of draws per method, per RNG case — MUST match `rng_capture.cpp`.
const DRAWS_PER_METHOD: usize = 32;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/rng_sequence.txt")
}

fn parse_i32_list(s: &str) -> Vec<i32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<i32>().expect("i32 field"))
        .collect()
}

fn parse_u32_list(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<u32>().expect("u32 (float-bits) field"))
        .collect()
}

/// Extract a `key=value` token's value from a whitespace-split line.
fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

#[test]
fn rng_parity_replays_every_committed_case() {
    let path = fixture_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "rng_parity: SKIP — fixture {} not found. Run `cargo run -p xtask -- regen` \
             on a machine with a C++ toolchain and commit the golden set.",
            path.display()
        );
        return;
    };

    let mut rng_cases = 0usize;
    let mut sample_cases = 0usize;

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "MASTER_SEED" | "COUNTS" => { /* metadata, not replayed */ }

            "RNG" => {
                let seed: i32 = field(&tokens, "seed")
                    .expect("RNG seed")
                    .parse()
                    .expect("RNG seed i32");
                let exp_int16 = parse_i32_list(field(&tokens, "int16").expect("int16"));
                let exp_int32 = parse_i32_list(field(&tokens, "int32").expect("int32"));
                let exp_float_bits = parse_u32_list(field(&tokens, "float").expect("float"));
                let exp_int = parse_i32_list(field(&tokens, "int").expect("int"));

                assert_eq!(exp_int16.len(), DRAWS_PER_METHOD, "line {}", lineno + 1);

                // Replay in the EXACT order rng_capture.cpp used:
                //   NextShort(0,32768) x32, NextInt(0,2^30) x32,
                //   NextFloat x32, NextInt(-1000,1000) x32.
                let mut r = Random::new(seed);
                for (i, &exp) in exp_int16.iter().enumerate() {
                    let got = r.next_short(0, 32768);
                    assert_eq!(
                        got, exp,
                        "RNG seed={seed} int16 draw {i} (line {})",
                        lineno + 1
                    );
                }
                for (i, &exp) in exp_int32.iter().enumerate() {
                    let got = r.next_int(0, 1_073_741_824);
                    assert_eq!(
                        got, exp,
                        "RNG seed={seed} int32 draw {i} (line {})",
                        lineno + 1
                    );
                }
                for (i, &exp_bits) in exp_float_bits.iter().enumerate() {
                    let got = r.next_float();
                    assert_eq!(
                        got.to_bits(),
                        exp_bits,
                        "RNG seed={seed} float draw {i} exact-bit (line {})",
                        lineno + 1
                    );
                }
                for (i, &exp) in exp_int.iter().enumerate() {
                    let got = r.next_int(-1000, 1000);
                    assert_eq!(
                        got, exp,
                        "RNG seed={seed} int draw {i} (line {})",
                        lineno + 1
                    );
                }
                rng_cases += 1;
            }

            "SAMPLE" => {
                let seed: i32 = field(&tokens, "seed")
                    .expect("SAMPLE seed")
                    .parse()
                    .expect("SAMPLE seed i32");
                let n: i32 = field(&tokens, "N").expect("N").parse().expect("N i32");
                let k: i32 = field(&tokens, "K").expect("K").parse().expect("K i32");
                let expected = parse_i32_list(field(&tokens, "result").expect("result"));

                let mut r = Random::new(seed);
                let got = r.sample(n, k);
                assert_eq!(
                    got,
                    expected,
                    "SAMPLE seed={seed} N={n} K={k} ordered output (line {})",
                    lineno + 1
                );
                sample_cases += 1;
            }

            other => panic!("unknown fixture record `{other}` at line {}", lineno + 1),
        }
    }

    assert!(
        rng_cases > 0 && sample_cases > 0,
        "fixture must contain both RNG and SAMPLE cases (got rng={rng_cases}, sample={sample_cases})"
    );
    eprintln!("rng_parity: replayed {rng_cases} RNG cases + {sample_cases} Sample cases — all bit-for-bit.");
}
