# Quick 260620-9cp — FINDINGS (honest NULL)

**Lever:** parallelize the per-leaf, cross-feature best-split scan in `CpuBackend`
(`find_best_splits_batched`) so a SINGLE rayon fork/join is amortized across ALL features
of a leaf (replacing the serial `for f in feats` loop). Order-preserving
(`par_iter().map().collect()`) ⇒ byte-identical to serial ⇒ bit-exact.

**Verdict: NULL.** The isolated SCAN win on the wide config (−13%) does NOT survive into
warm train-wall: the per-leaf scan fork/join contends with the already-rayon-parallelized
BUILD path, so warm train-wall *regresses* +11% on wide and ~1.8× on narrow. Adoption
criterion (wide train-wall gain AND no narrow regression) fails. Serial stays the effective
default; parallel kept env-reachable (`LGBM_PAR_SCAN_THRESHOLD`) and proven bit-exact
forced-on.

## Measured numbers (cubecl-cpu f64 anchor; warm; bench_split_scan.rs; LGBM_PHASE_PROF=1)

A = SERIAL (`LGBM_PAR_SCAN_THRESHOLD=1000000`), 3 runs:
```
narrow: SCAN_NS=33514816  BUILD_NS=61241479  warm_wall=168.842ms
wide:   SCAN_NS=353187074 BUILD_NS=438755232 warm_wall=1066.232ms
narrow: SCAN_NS=31442369  BUILD_NS=57299246  warm_wall=156.259ms
wide:   SCAN_NS=341779984 BUILD_NS=428415356 warm_wall=1040.916ms
narrow: SCAN_NS=31740105  BUILD_NS=57044143  warm_wall=154.304ms
wide:   SCAN_NS=357059589 BUILD_NS=453774998 warm_wall=1086.194ms
```

B = PARALLEL (`LGBM_PAR_SCAN_THRESHOLD=0`), 3 runs:
```
narrow: SCAN_NS=99330716  BUILD_NS=72888680  warm_wall=279.373ms
wide:   SCAN_NS=309604222 BUILD_NS=517620544 warm_wall=1174.816ms
narrow: SCAN_NS=110811544 BUILD_NS=77767380  warm_wall=301.332ms
wide:   SCAN_NS=306883355 BUILD_NS=526413396 warm_wall=1184.005ms
narrow: SCAN_NS=104365565 BUILD_NS=73069866  warm_wall=286.682ms
wide:   SCAN_NS=308615025 BUILD_NS=525894615 warm_wall=1184.289ms
```

DEFAULT-as-tuned-at-64 (narrow serial, wide parallel), 3 runs — confirms wide regresses even
when narrow is gated to serial:
```
narrow: warm_wall=166.403ms   wide: SCAN_NS=316734910 BUILD_NS=523105021 warm_wall=1191.231ms
narrow: warm_wall=157.420ms   wide: SCAN_NS=320019099 BUILD_NS=548659124 warm_wall=1231.063ms
narrow: warm_wall=162.339ms   wide: SCAN_NS=305275701 BUILD_NS=496755887 warm_wall=1143.608ms
```

## Medians + interpretation
| config | metric     | A=serial | B=parallel | Δ           | sign-stable |
|--------|------------|----------|------------|-------------|-------------|
| narrow | SCAN_NS    | 31.74ms  | 104.37ms   | +229% WORSE | yes          |
| narrow | train-wall | 156.3ms  | 286.7ms    | +84% WORSE  | yes          |
| wide   | SCAN_NS    | 353.19ms | 308.62ms   | −13% better | yes          |
| wide   | train-wall | 1066.2ms | 1184.3ms   | +11% WORSE  | yes          |

**Root cause of the NULL:** the per-leaf scan fork/join steals rayon worker time from the
already-parallel BUILD path — wide `BUILD_NS` rose 438→520ms when scan was parallelized. The
13% SCAN_NS reduction is outweighed by the BUILD slowdown, so the net warm train-wall is
*slower*. On narrow (10 features) one fork/join can't be amortized at all → 3.3× SCAN
regression. This is the same per-feature/per-leaf rayon-dispatch tax family that NULL'd
260620-8v4 and gated the unconditional BUILD path (Spike 005), surfacing here as
BUILD/SCAN thread contention rather than per-call fork/join.

## Disposition
- Code kept (order-preserving, gated, bit-exact-proven) but **default threshold = usize::MAX**
  ⇒ serial is the effective default; parallel reachable only via explicit env override.
- Consistent with the project's audit-before-wire value and the q2z/sgu/tlk/8v4 NULL
  discipline. No win manufactured.
