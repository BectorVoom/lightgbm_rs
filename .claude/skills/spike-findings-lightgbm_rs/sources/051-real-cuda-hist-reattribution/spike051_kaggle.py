#!/usr/bin/env python3
"""Spike-051 — real-CUDA histogram-phase RE-ATTRIBUTION (occupancy sweep).

The 001-040 perf campaign tuned the GPU build kernel on a SPOOFED 8-CU APU.
Spike-040 found the APU heuristic under-partitions to P=1 at the production
50-feature width; autotune (BUILD_PSET = [1,4,8,16,32], default-ON) beats it ~10%
ON THE APU. This spike re-attributes on REAL discrete NVIDIA (Kaggle) at the
500k x 50 repro: is the GPU `hist+split` (build+subtract) phase OCCUPANCY-STARVED
COMPUTE (=> lift the row-partition P / BUILD_PSET ceiling = spike-053) or
LAUNCH/ORCHESTRATION-bound (=> fuse launches = spike-052)?

DECISIVE PROBE (zero code change — all existing env toggles on current master):
  sweep `LGBM_AUTOTUNE_FORCE_P` in {1,4,8,16,32,64,128} vs DEFAULT(autotune-on)
  vs `LGBM_AUTOTUNE=0`(heuristic P=1), all under LGBM_PHASE_PROF=1, and read the
  `hist+split`/`build` phase device-time per arm.

Reads (in-session deltas — absolute Kaggle walls are NOT cross-session comparable):
  - build/hist+split keeps DROPPING past P=32  => ceiling too low on NVIDIA => 053 GREEN
  - DEFAULT(autotune) != best forced P          => autotune mis-fires on cubecl-cuda
  - build flat vs P, residual dominates         => launch/orchestration-bound => 052

Kaggle: kernel_type=script, enable_gpu=true, enable_internet=true.
"""
import os
import re
import subprocess
import sys

REPO_URL = os.environ.get("REPO_URL", "https://github.com/BectorVoom/lightgbm_rs.git")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "")  # default master

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


# Inner bench: ONE backend / ONE arm per subprocess (phase_prof atomics are
# process-global, so a fresh process per arm keeps stderr cleanly attributable).
INNER = r'''
import os, sys, time
import numpy as np
from sklearn.datasets import make_classification

which = sys.argv[1]          # "rs" | "off"
N_SAMPLES, N_FEATURES, SEED = 500_000, 50, 42
X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=SEED)
X = np.ascontiguousarray(X, dtype=np.float64)
params = dict(objective="binary", metric="binary_logloss", device_type="cuda",
              num_leaves=31, learning_rate=0.1, n_estimators=100, verbose=-1)
if which == "rs":
    import lightgbm_rs as lgb
else:
    import lightgbm as lgb
# warmup (amortize alloc; warm the autotune cache for the DEFAULT arm), then timed.
warm = lgb.LGBMClassifier(**{**params, "n_estimators": 5}); warm.fit(X, y)
t0 = time.time(); m = lgb.LGBMClassifier(**params); m.fit(X, y); dt = time.time() - t0
print(f"RESULT {which} force_p={os.environ.get('LGBM_AUTOTUNE_FORCE_P','-')} "
      f"autotune={os.environ.get('LGBM_AUTOTUNE','1')} train_time_s={dt:.3f}", flush=True)
'''


def parse_phase(stderr):
    """Pull the LAST (timed-run) phase line metrics from a phase_prof dump."""
    out = {}
    for line in stderr.splitlines():
        if "hist+split=" in line and "build=" in line:
            for key, pat in (("histsplit", r"hist\+split=([\d.]+)"),
                             ("build", r"build=([\d.]+)"),
                             ("scan", r"scan=([\d.]+)"),
                             ("partition", r"partition=([\d.]+)")):
                m = re.search(pat, line)
                if m:
                    out[key] = float(m.group(1))
        if "BUDGET:" in line and "phases=" in line:
            for key, pat in (("phases", r"phases=([\d.]+)"),
                             ("in_learner_other", r"in_learner_other=([\d.]+)"),
                             ("learner", r"learner=([\d.]+)")):
                m = re.search(pat, line)
                if m:
                    out[key] = float(m.group(1))
        if "COUNTS:" in line and "device_launches=" in line:
            m = re.search(r"device_launches=(\d+)", line)
            if m:
                out["launches"] = int(m.group(1))
    return out


def parse_wall(stdout):
    for line in stdout.splitlines():
        if line.startswith("RESULT"):
            return float(line.split("train_time_s=")[-1])
    return float("nan")


def run_arm(label, extra_env, which="rs"):
    env = dict(os.environ)
    env["LGBM_PHASE_PROF"] = "1"
    env.pop("LGBM_AUTOTUNE_FORCE_P", None)
    env.pop("LGBM_AUTOTUNE", None)
    env.update(extra_env)
    print(f"\n===== ARM {label} ({extra_env}) =====", flush=True)
    res = subprocess.run(f"python3 inner_bench.py {which}", shell=True,
                         capture_output=True, text=True, env=env)
    print("--- stdout ---\n" + res.stdout, flush=True)
    print("--- stderr (phase_prof) ---\n" + res.stderr, flush=True)
    rec = parse_phase(res.stderr)
    rec["wall"] = parse_wall(res.stdout)
    rec["label"] = label
    return rec


def main():
    run("pip install -q -U numpy scipy scikit-learn maturin")
    run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    os.environ["PATH"] += ":/root/.cargo/bin"

    if not os.path.exists("lightgbm_rs"):
        run(f"git clone {REPO_URL}")
    else:
        run("cd lightgbm_rs && git checkout -- . && git fetch --all")
    if REPO_BRANCH:
        run(f"cd lightgbm_rs && git checkout {REPO_BRANCH} && git pull")
    else:
        run("cd lightgbm_rs && git checkout master && git pull")
    run("cd lightgbm_rs && git log --oneline -1")

    print("\n########## BUILD lightgbm_rs CUDA wheel (-F cuda) ##########")
    run("rm -rf lightgbm_rs/target/wheels/")
    run("cd lightgbm_rs/crates/lgbm-python && maturin build --release -F cuda -j 2")
    run("pip install $(ls lightgbm_rs/target/wheels/*.whl | head -n 1) --force-reinstall")

    with open("inner_bench.py", "w") as f:
        f.write(INNER)

    # Wipe any stale persistent autotune cache so the DEFAULT arm cold-tunes clean.
    run("rm -rf lightgbm_rs/target/autotune ~/.cache/cubecl 2>/dev/null || true", check=False)

    arms = []
    # The occupancy sweep (FORCE_P pins P, bypassing the ROWPART_P_MAX=16 clamp):
    for p in (1, 4, 8, 16, 32, 64, 128):
        arms.append(run_arm(f"force_p={p}", {"LGBM_AUTOTUNE_FORCE_P": str(p)}))
    # Production default (autotune over BUILD_PSET=[1,4,8,16,32], default-ON):
    arms.append(run_arm("default(autotune)", {}))
    # Heuristic fallback (row_partition_count => P=1 at 50 feat):
    arms.append(run_arm("autotune=0(heuristic)", {"LGBM_AUTOTUNE": "0"}))

    # In-session official-CUDA reference for the gap (best-effort; image may lack CUDA).
    try:
        off = run_arm("official-cuda", {}, which="off")
    except Exception as e:
        off = {"label": "official-cuda", "wall": float("nan")}
        print(f"official-cuda arm failed: {e}", flush=True)

    print("\n\n==================== SPIKE-051 OCCUPANCY SWEEP (real CUDA, 500k x 50) ====================")
    hdr = f"{'arm':22} {'wall_s':>8} {'histsplit':>10} {'build':>8} {'scan':>8} {'partn':>8} {'launches':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in arms:
        print(f"{r.get('label',''):22} {r.get('wall',float('nan')):8.3f} "
              f"{r.get('histsplit',float('nan')):10.1f} {r.get('build',float('nan')):8.1f} "
              f"{r.get('scan',float('nan')):8.1f} {r.get('partition',float('nan')):8.1f} "
              f"{r.get('launches',0):9d}")
    print(f"\nofficial-cuda wall: {off.get('wall', float('nan')):.3f} s")

    # Verdict hints (in-session deltas):
    fp = {r["label"]: r for r in arms}
    def b(lbl):
        return fp.get(lbl, {}).get("histsplit", float("nan"))
    print("\n--- read-out ---")
    print(f"  hist+split  P=1 -> P=32 : {b('force_p=1'):.1f} -> {b('force_p=32'):.1f} ms")
    print(f"  hist+split  P=32 -> P=64 -> P=128 : "
          f"{b('force_p=32'):.1f} -> {b('force_p=64'):.1f} -> {b('force_p=128'):.1f} ms")
    print("  (keeps dropping past 32 => BUILD_PSET ceiling too low on NVIDIA => spike-053 GREEN)")
    print(f"  default(autotune) hist+split: {b('default(autotune)'):.1f} ms "
          f"(should match the best forced-P; if worse => autotune mis-fires on cubecl-cuda)")


if __name__ == "__main__":
    main()
