//! CMP-01 containment guard.
//!
//! `lgbm-compute` is the ONLY crate allowed to name a cubecl runtime. This test
//! fails if any *other* workspace crate either (a) lists `cubecl` as a
//! dependency in its `Cargo.toml`, or (b) references the `cubecl` token in a
//! non-comment line of its `src/`. It is the automated enforcement of the
//! alpha-churn containment boundary — it would fail the moment someone added
//! `cubecl` to e.g. `lgbm-core/Cargo.toml`.

use std::path::{Path, PathBuf};

/// Workspace root = two levels up from this crate's manifest dir
/// (`crates/lgbm-compute` -> `crates` -> repo root).
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate manifest dir has a grandparent (the workspace root)")
        .to_path_buf()
}

/// Crates above `lgbm-compute` that must never name a cubecl runtime.
const UPPER_CRATES: &[&str] = &["lgbm-core", "lgbm-dataset", "lgbm-model", "oracle-harness"];

/// Strip an inline `//`-comment and a leading `#`-comment so we only inspect the
/// code portion of a line (TOML uses `#`, Rust uses `//`).
fn code_portion(line: &str) -> &str {
    let trimmed = line.trim_start();
    if trimmed.starts_with('#') {
        return "";
    }
    match line.find("//") {
        Some(idx) => &line[..idx],
        None => line,
    }
}

#[test]
fn no_upper_crate_names_cubecl_in_cargo_toml() {
    let root = workspace_root();
    for crate_name in UPPER_CRATES {
        let manifest = root.join("crates").join(crate_name).join("Cargo.toml");
        let text = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|e| panic!("could not read {}: {e}", manifest.display()));
        for (lineno, line) in text.lines().enumerate() {
            let code = code_portion(line);
            assert!(
                !code.contains("cubecl"),
                "CMP-01 VIOLATION: crate `{crate_name}` names `cubecl` in {} (line {}): {}",
                manifest.display(),
                lineno + 1,
                line.trim()
            );
        }
    }
}

#[test]
fn no_upper_crate_references_cubecl_in_src() {
    let root = workspace_root();
    for crate_name in UPPER_CRATES {
        let src = root.join("crates").join(crate_name).join("src");
        if !src.exists() {
            continue;
        }
        let mut stack = vec![src];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir)
                .unwrap_or_else(|e| panic!("could not read dir {}: {e}", dir.display()))
            {
                let entry = entry.expect("readable dir entry");
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()));
                for (lineno, line) in text.lines().enumerate() {
                    let code = code_portion(line);
                    assert!(
                        !code.contains("cubecl"),
                        "CMP-01 VIOLATION: crate `{crate_name}` references `cubecl` in {} (line {}): {}",
                        path.display(),
                        lineno + 1,
                        line.trim()
                    );
                }
            }
        }
    }
}
