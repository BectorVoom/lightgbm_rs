#!/usr/bin/env bash
# quick 260620-c5v Task-1 sweep: threshold-vs-cores crossover for both unified fusions.
# Each cell = median train-wall (ms) of the bench harness (itself median-of-5), run 3x for a 3-run median.
# A = fusion OFF (env threshold = usize::MAX-ish large), B = fusion ON (env threshold = 0).
# The OTHER fusion is pinned to its current constant so each sweep isolates one fusion.
set -euo pipefail
BIN=./target/release/examples/bench_train
HI=100000000   # force-off (feature counts never reach this)
export BENCH_ROWS=20000 BENCH_BINS=128 LGBM_PHASE_PROF=1

# extract train_median ms (the "custom" row, train_median col) as a float.
# $1 = the threshold env assignment string for the fusion-under-test (e.g. LGBM_UNIFIED_BFS_THRESHOLD=0)
run_ms() {
  local kv=$1 k v
  k=${kv%%=*}; v=${kv#*=}
  export "$k"="$v"
  # Duration's {:.2?} prints "693.21ms" OR "1.03s" OR "812.34µs" — normalize to ms.
  $BIN 2>/dev/null | awk '/^custom/{
    m=$5;
    if (m ~ /ms$/)      { sub(/ms$/,"",m); print m+0 }
    else if (m ~ /µs$/) { sub(/µs$/,"",m); print (m+0)/1000.0 }
    else if (m ~ /s$/)  { sub(/s$/,"",m);  print (m+0)*1000.0 }
    else                { print m+0 }
  }'
}

median3() { printf '%s\n' "$1" "$2" "$3" | sort -n | sed -n 2p; }

sweep_fusion() {
  local fusion=$1   # BFS or SUBSCAN
  local cores=$2
  local feat=$3
  local bfs sub a_thresh_var b
  if [[ $fusion == BFS ]]; then
    # isolate BFS: pin SUBSCAN to its constant 130
    export LGBM_UNIFIED_SUBSCAN_THRESHOLD=130
    aoff="LGBM_UNIFIED_BFS_THRESHOLD=$HI"
    bon="LGBM_UNIFIED_BFS_THRESHOLD=0"
  else
    # isolate SUBSCAN: pin BFS to its constant 100
    export LGBM_UNIFIED_BFS_THRESHOLD=100
    aoff="LGBM_UNIFIED_SUBSCAN_THRESHOLD=$HI"
    bon="LGBM_UNIFIED_SUBSCAN_THRESHOLD=0"
  fi
  export RAYON_NUM_THREADS=$cores BENCH_FEATURES=$feat
  local a1 a2 a3 b1 b2 b3 am bm
  a1=$(run_ms "$aoff"); b1=$(run_ms "$bon")
  a2=$(run_ms "$aoff"); b2=$(run_ms "$bon")
  a3=$(run_ms "$aoff"); b3=$(run_ms "$bon")
  am=$(median3 "$a1" "$a2" "$a3"); bm=$(median3 "$b1" "$b2" "$b3")
  # min/max of A and B across the 3 runs for spread-overlap check
  local amin amax bmin bmax
  amin=$(printf '%s\n' "$a1" "$a2" "$a3" | sort -n | head -1)
  amax=$(printf '%s\n' "$a1" "$a2" "$a3" | sort -n | tail -1)
  bmin=$(printf '%s\n' "$b1" "$b2" "$b3" | sort -n | head -1)
  bmax=$(printf '%s\n' "$b1" "$b2" "$b3" | sort -n | tail -1)
  # pct = (B-A)/A *100  (negative = B faster)
  local pct stable
  pct=$(awk -v a="$am" -v b="$bm" 'BEGIN{printf "%.1f", (b-a)/a*100}')
  # sign-stable B-faster = bmax < amin (no spread overlap, B entirely below A)
  if awk -v bx="$bmax" -v an="$amin" 'BEGIN{exit !(bx<an)}'; then stable="WIN"; else stable="--"; fi
  printf '%-7s cores=%-3s feat=%-4s  A[%s..%s]m=%s  B[%s..%s]m=%s  pct=%6s  %s\n' \
    "$fusion" "$cores" "$feat" "$amin" "$amax" "$am" "$bmin" "$bmax" "$bm" "$pct" "$stable"
}

FUSION=${1:?BFS|SUBSCAN}
CORES=${2:?cores}
shift 2
for f in "$@"; do
  sweep_fusion "$FUSION" "$CORES" "$f"
done
