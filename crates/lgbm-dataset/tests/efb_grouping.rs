//! EFB grouping golden replay (layer 3, DAT-05): for every committed EFB case,
//! rebuild the Rust `BinMapper` set + the bundled `Dataset`
//! (`construct_bundled`, `enable_bundle=true`) on the SAME D-06 #4 corpus and
//! assert the grouping is BIT-IDENTICAL to the C++ golden:
//!
//! - feature -> group membership (`feature2group_` / `feature2subfeature_`),
//! - per-group `bin_offsets_` + `num_total_bin_` + the `group_is_multi_val` flag,
//! - the per-row bundled bin index per single-value group.
//!
//! Plus a CONTROL case (`control_no_bundle`) where no features are mutually
//! exclusive, proving the `enable_bundle` dispatch boundary (one single-feature
//! group per feature).
//!
//! Grouping is RNG- and stable-sort-driven (RESEARCH Pitfall 5), so any drift in
//! the `Sample`/`NextShort` sequence or the stable sort changes the result — this
//! golden catches it. Compared bit-exact via the oracle-harness exact comparators
//! (NOT the `~1e-6` oracle tolerance).
//!
//! Idioms follow `oracle-harness/tests/rng_parity.rs`: `CARGO_MANIFEST_DIR`
//! fixture path (never the untracked `LightGBM/` tree), graceful SKIP pre-capture,
//! and localizing assert messages naming group + feature + row.
//!
//! Fixture written by `cargo run -p xtask -- bin-capture` into
//! `tests/fixtures/efb_grouping.txt`. Record format:
//! ```text
//! ECASE name=<id> num_rows=<n> num_features=<f> max_bin=<m> \
//!       is_enable_sparse=<0|1> num_groups=<g> num_used=<u>
//! FCOL f=<f> <f64bits;...>                  # full per-feature column (raw f64 bits)
//! MEMBERSHIP real=<r>,group=<g>,sub=<s>;... # packed group-major order
//! GROUP g=<g> num_total_bin=<u64> is_multi_val=<0|1> bin_offsets=<u32;...> \
//!       features=<real;...>
//! GROW g=<g> <u32;...>                      # per-row bundled bin index (single-val)
//! ```

use std::path::PathBuf;

use lgbm_core::config::Config;
use lgbm_dataset::bin_mapper::BinMapper;
use lgbm_dataset::dataset::EfbSamples;
use lgbm_dataset::Dataset;
use oracle_harness::comparator::compare_exact_u32;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/efb_grouping.txt")
}

fn field<'a>(tokens: &'a [&'a str], key: &str) -> Option<&'a str> {
    tokens
        .iter()
        .find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn parse_i32(tokens: &[&str], key: &str) -> i32 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing i32 field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad i32 field `{key}`"))
}

fn parse_u64(tokens: &[&str], key: &str) -> u64 {
    field(tokens, key)
        .unwrap_or_else(|| panic!("missing u64 field `{key}`"))
        .parse()
        .unwrap_or_else(|_| panic!("bad u64 field `{key}`"))
}

fn parse_f64_bits_list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| f64::from_bits(t.parse::<u64>().expect("f64-bits u64 field")))
        .collect()
}

fn parse_u32_list(s: &str) -> Vec<u32> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|t| t.parse::<u32>().expect("u32 field"))
        .collect()
}

/// One packed membership slot (group-major order, matching the C++ dump).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Member {
    real: i32,
    group: i32,
    sub: i32,
}

#[derive(Debug)]
struct GroupGolden {
    g: i32,
    num_total_bin: u64,
    is_multi_val: bool,
    bin_offsets: Vec<u32>,
    features: Vec<i32>,
    grow: Option<Vec<u32>>, // per-row bundled bin index (single-val groups only)
}

#[derive(Debug)]
struct EfbGolden {
    name: String,
    num_rows: i32,
    num_features: i32,
    max_bin: i32,
    is_enable_sparse: bool,
    num_groups: i32,
    columns: Vec<Vec<f64>>, // columns[f][row]
    membership: Vec<Member>,
    groups: Vec<GroupGolden>,
}

fn parse(text: &str) -> Vec<EfbGolden> {
    let mut out = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t[0] != "ECASE" {
            continue; // MASTER_SEED / COUNTS
        }
        let name = field(&t, "name").expect("ECASE name").to_string();
        let num_rows = parse_i32(&t, "num_rows");
        let num_features = parse_i32(&t, "num_features");
        let max_bin = parse_i32(&t, "max_bin");
        let is_enable_sparse = parse_i32(&t, "is_enable_sparse") != 0;
        let num_groups = parse_i32(&t, "num_groups");

        let mut columns: Vec<Vec<f64>> = vec![Vec::new(); num_features as usize];
        for _ in 0..num_features {
            let ct: Vec<&str> = lines.next().expect("FCOL").split_whitespace().collect();
            assert_eq!(ct[0], "FCOL", "expected FCOL for `{name}`");
            let f = parse_i32(&ct, "f");
            columns[f as usize] = parse_f64_bits_list(ct.get(2).copied().unwrap_or(""));
        }

        let mt: Vec<&str> = lines
            .next()
            .expect("MEMBERSHIP")
            .split_whitespace()
            .collect();
        assert_eq!(mt[0], "MEMBERSHIP", "expected MEMBERSHIP for `{name}`");
        let membership: Vec<Member> = if mt.len() < 2 {
            Vec::new()
        } else {
            mt[1]
                .split(';')
                .map(|slot| {
                    let mut real = 0;
                    let mut group = 0;
                    let mut sub = 0;
                    for kv in slot.split(',') {
                        let (k, v) = kv.split_once('=').expect("member kv");
                        let v: i32 = v.parse().expect("member i32");
                        match k {
                            "real" => real = v,
                            "group" => group = v,
                            "sub" => sub = v,
                            other => panic!("unknown member key `{other}`"),
                        }
                    }
                    Member { real, group, sub }
                })
                .collect()
        };

        let mut groups = Vec::with_capacity(num_groups as usize);
        for _ in 0..num_groups {
            let gt: Vec<&str> = lines.next().expect("GROUP").split_whitespace().collect();
            assert_eq!(gt[0], "GROUP", "expected GROUP for `{name}`");
            let g = parse_i32(&gt, "g");
            let num_total_bin = parse_u64(&gt, "num_total_bin");
            let is_multi_val = parse_i32(&gt, "is_multi_val") != 0;
            let bin_offsets = parse_u32_list(field(&gt, "bin_offsets").expect("bin_offsets"));
            let features = parse_u32_list(field(&gt, "features").expect("features"))
                .into_iter()
                .map(|v| v as i32)
                .collect();

            // an optional GROW line follows for single-value groups.
            let grow = if matches!(lines.peek(), Some(l) if l.trim_start().starts_with("GROW ")) {
                let wt: Vec<&str> = lines.next().expect("GROW").split_whitespace().collect();
                assert_eq!(parse_i32(&wt, "g"), g, "GROW g mismatch in `{name}`");
                Some(parse_u32_list(wt.get(2).copied().unwrap_or("")))
            } else {
                None
            };

            groups.push(GroupGolden {
                g,
                num_total_bin,
                is_multi_val,
                bin_offsets,
                features,
                grow,
            });
        }

        out.push(EfbGolden {
            name,
            num_rows,
            num_features,
            max_bin,
            is_enable_sparse,
            num_groups,
            columns,
            membership,
            groups,
        });
    }
    out
}

/// Rebuild the per-feature mappers + EFB sample inputs exactly as the C++ harness
/// did: a mapper over the FULL column (sample_cnt == num_rows), and the per-column
/// non-zero sample indices/values (every row whose value != 0.0).
fn build_inputs(c: &EfbGolden) -> (Vec<BinMapper>, EfbSamples) {
    let mut mappers = Vec::with_capacity(c.num_features as usize);
    let mut sample_indices = Vec::with_capacity(c.num_features as usize);
    let mut sample_values = Vec::with_capacity(c.num_features as usize);
    let mut num_per_col = Vec::with_capacity(c.num_features as usize);

    for f in 0..c.num_features as usize {
        let col = &c.columns[f];
        let mapper = BinMapper::find_bin_numeric(
            col.clone(),
            c.max_bin,
            /*min_data_in_bin=*/ 1,
            /*min_split_data=*/ 1,
            /*pre_filter=*/ false,
            /*use_missing=*/ true,
            /*zero_as_missing=*/ false,
            c.num_rows as usize,
            &[],
        );
        mappers.push(mapper);

        let mut idx = Vec::new();
        let mut vals = Vec::new();
        for (r, &v) in col.iter().enumerate() {
            if v != 0.0 {
                idx.push(r as i32);
                vals.push(v);
            }
        }
        num_per_col.push(idx.len() as i32);
        sample_indices.push(idx);
        sample_values.push(vals);
    }

    let samples = EfbSamples {
        sample_indices,
        sample_values,
        num_per_col,
        num_sample_col: c.num_features,
        total_sample_cnt: c.num_rows,
    };
    (mappers, samples)
}

#[test]
fn efb_grouping_replays_every_committed_case() {
    let path = fixture_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "efb_grouping: SKIP — fixture {} not found. Run \
             `cargo run -p xtask -- bin-capture` on a machine with a C++ toolchain \
             and commit the golden set.",
            path.display()
        );
        return;
    };

    let cases = parse(&text);
    assert!(!cases.is_empty(), "fixture present but parsed zero cases");

    let mut bundled_seen = false;
    let mut control_seen = false;

    for c in &cases {
        let (mappers, samples) = build_inputs(c);

        let cfg = Config {
            enable_bundle: true,
            is_enable_sparse: c.is_enable_sparse,
            ..Config::default()
        };

        let mut ds = Dataset::construct_bundled(mappers, c.num_rows, &cfg, &samples)
            .unwrap_or_else(|e| panic!("case `{}`: construct_bundled failed: {e:?}", c.name));

        // Push every feature value (mirrors the C++ harness PushData loop): route
        // each real feature through its owning group/sub-feature via push_value.
        for r in 0..c.num_rows {
            for f in 0..c.num_features as usize {
                if ds.feature_to_group(f) >= 0 {
                    ds.push_value(f, r, c.columns[f][r as usize]);
                }
            }
        }
        let finished = ds.finish_load();

        // 1. group COUNT parity (the headline EFB outcome).
        assert_eq!(
            finished.num_groups(),
            c.num_groups,
            "case `{}`: num_groups mismatch (EFB grouping drift)",
            c.name
        );

        if c.num_groups < c.num_features {
            bundled_seen = true;
        }
        if c.name == "control_no_bundle" {
            control_seen = true;
            assert_eq!(
                c.num_groups, c.num_features,
                "control must be one group per feature"
            );
        }

        // 2. feature -> group/subfeature membership parity. Rebuild the packed
        //    group-major order from the FinishedDataset and compare to the golden.
        //    Invert feature_to_group/subfeature over all real features.
        let mut rust_membership: Vec<Member> = Vec::with_capacity(c.membership.len());
        // For each (group, sub) slot in order, find the real feature mapping there.
        // Build the inverse: real feature -> (group, sub).
        let mut by_group: Vec<Vec<(i32, i32)>> =
            vec![Vec::new(); finished.num_groups() as usize]; // group -> [(sub, real)]
        for real in 0..c.num_features {
            let g = finished.feature_to_group(real as usize);
            let s = finished.feature_to_subfeature(real as usize);
            if g >= 0 {
                by_group[g as usize].push((s, real));
            }
        }
        for (g, subs) in by_group.iter_mut().enumerate() {
            subs.sort_by_key(|&(s, _)| s);
            for &(s, real) in subs.iter() {
                rust_membership.push(Member {
                    real,
                    group: g as i32,
                    sub: s,
                });
            }
        }
        assert_eq!(
            rust_membership, c.membership,
            "case `{}`: feature->group membership mismatch (group-major packed order)",
            c.name
        );

        // 3. per-group layout parity: bin_offsets_, num_total_bin_, multi_val flag,
        //    sub-feature membership, and the per-row bundled bin index.
        for gg in &c.groups {
            let fg = finished.feature_group(gg.g as usize);

            assert_eq!(
                fg.num_total_bin_, gg.num_total_bin,
                "case `{}` group {}: num_total_bin_ mismatch",
                c.name, gg.g
            );
            compare_exact_u32(&fg.bin_offsets_, &gg.bin_offsets).unwrap_or_else(|m| {
                panic!("case `{}` group {}: bin_offsets_ {m}", c.name, gg.g);
            });
            assert_eq!(
                fg.is_multi_val(),
                gg.is_multi_val,
                "case `{}` group {}: is_multi_val mismatch",
                c.name, gg.g
            );
            assert_eq!(
                fg.num_feature(),
                gg.features.len() as i32,
                "case `{}` group {}: sub-feature count mismatch",
                c.name, gg.g
            );

            // per-row bundled bin index for single-value groups.
            if let Some(grow) = &gg.grow {
                assert!(
                    !fg.is_multi_val(),
                    "case `{}` group {}: golden has GROW but group is multi-val",
                    c.name,
                    gg.g
                );
                let bin = fg
                    .bin_data()
                    .unwrap_or_else(|| panic!("case `{}` group {}: no bin_data", c.name, gg.g));
                let rust_grow: Vec<u32> = (0..c.num_rows).map(|r| bin.data(r)).collect();
                compare_exact_u32(&rust_grow, grow).unwrap_or_else(|m| {
                    panic!(
                        "case `{}` group {} (features {:?}): per-row bundled index {m}",
                        c.name, gg.g, gg.features
                    );
                });
            }
        }
    }

    assert!(
        bundled_seen,
        "corpus did not cover a case where EFB bundled features (num_groups < num_features)"
    );
    assert!(
        control_seen,
        "corpus did not cover the no-bundle control (one group per feature)"
    );

    eprintln!(
        "efb_grouping: replayed {} EFB cases — group membership + bin_offsets_ + \
         num_total_bin_ + per-row bundled indices all bit-for-bit.",
        cases.len()
    );
}
