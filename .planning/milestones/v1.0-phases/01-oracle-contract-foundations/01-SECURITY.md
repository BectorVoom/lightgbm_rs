---
phase: 01
slug: oracle-contract-foundations
status: verified
threats_open: 0
asvs_level: 1
created: 2026-06-06
---

# Phase 01 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Register authored at plan time across 01-01/01-02/01-03 PLAN.md `<threat_model>` blocks.
> All 7 `mitigate` threats verified present in implementation by gsd-security-auditor; 2 accepted risks documented.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| caller → `Config::from_params` | Untrusted parameter strings (CLI args / Python bindings / param files) cross into the library here — the primary attack surface | Arbitrary key/value strings (untrusted) |
| developer → `xtask regen` | Dev-only: derives the randomized golden case set from a recorded master seed; reads/writes repo-local files and shells to CMake/C++ | Repo-local paths + fixed compile-time seed (no network, no OS entropy) |
| committed fixtures → test process | `rng_sequence.txt` / `REFERENCE_MANIFEST.md` read at test time as trusted, version-controlled data | Pinned golden data (trusted) |
| drift-checker test → in-repo C++ source | Reads `config.h` / `config_auto.cpp` as trusted, version-controlled workspace files | Workspace-relative source paths (trusted) |
| crate consumer → `Random` API | Deterministic, NON-cryptographic numeric draw API | Numeric seeds / draws (non-secret) |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-1-01 | Denial of Service | `Config::from_params` parsing untrusted strings | mitigate | All parses/`check_*` return typed `ConfigError` via `Result`; `present()` makes empty==absent a no-op; no panic on caller input. Verified: `config/set.rs` + `config_validation.rs::fuzz_hostile_strings_never_panic` (3000) + 2000 randomized panic-free | closed |
| T-1-02 | Denial of Service | `Random` LCG arithmetic | mitigate | `u32` state with `wrapping_mul`/`wrapping_add` — deliberate overflow never panics. Verified: `random.rs:28,45,53` + `lcg_does_not_panic_on_overflow_seed` (`random.rs:222`) | closed |
| T-1-03 | Information Disclosure | `Random` API surface | mitigate | Rustdoc marks `Random` deterministic / NON-cryptographic; never presented as secure randomness (V6). Verified: `random.rs:9-15,24-25` | closed |
| T-1-04 | Tampering | xtask regen file I/O + case generation | mitigate | `workspace_root()` from `CARGO_MANIFEST_DIR` + `Cargo.toml` guard; all I/O is `root.join(<fixed in-repo subpath>)` — no traversal. Case set from `MASTER_SEED` const only, no OS entropy → deterministic/idempotent (V12). Verified: `xtask/src/main.rs:13-15` | closed |
| T-1-05 | Tampering | drift-checker file reads | mitigate | Test reads only `workspace_root().join("LightGBM/...")`, no user input, no traversal (V12). Verified: `config_drift.rs:13-23,33-45` | closed |
| T-1-06 | Denial of Service | seed/int parsing arithmetic | mitigate | `parse_int` → `Err(InvalidType)` never panics; `next_short` modulo over bounded 15-bit value. Verified: `config/set.rs:547-555`, `random.rs:60-62` | closed |
| T-1-08 | Tampering | non-deterministic alias-collision resolution | mitigate | `key_alias_transform` + `sort_alias (len,key)` tie-break removes HashMap-iteration-order dependence. Verified: `config/set.rs:362-421` + `colliding_alias_resolution_is_deterministic` (N=200) + 3 collision regressions | closed |
| T-1-09 | Information Disclosure | empty-string rejection of valid unset params | accept | No sensitive data; empty==absent matches C++ `Get*` guard. See Accepted Risks Log | closed |
| T-1-SC | Tampering | cargo dependency installs (thiserror/anyhow/cubecl) | accept | Only pinned first-party crates; `Cargo.lock` committed and git-tracked. See Accepted Risks Log | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| R-1-09 | T-1-09 | Empty-string-is-absent semantics involve no sensitive data and intentionally match the C++ reference `Get*` (`count(name) > 0 && !empty()`) behavior. Covered by `empty_seed_is_noop` / `empty_enum_values_are_noop`. Low risk. | appservice27 (user) | 2026-06-06 |
| R-1-SC | T-1-SC | Only CLAUDE.md-mandated, registry-verified, already-pinned first-party staples (thiserror, anyhow, cubecl); no new/obscure packages; `Cargo.lock` committed and git-tracked. Supply-chain surface unchanged. | appservice27 (user) | 2026-06-06 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-06 | 9 | 9 | 0 | gsd-security-auditor (verify-mitigations mode) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-06
