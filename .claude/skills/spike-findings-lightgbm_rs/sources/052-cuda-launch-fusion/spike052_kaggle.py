#!/usr/bin/env python3
"""Spike-052 — CUDA LAUNCH FUSION (cut the 8,570 launches / 2,890 syncs).

Spike-051 localized the real-CUDA hist+split cost as LAUNCH/SYNC-LATENCY-bound
(occupancy-insensitive; build=0 async; 8570 launches / 2890 syncs / 100 trees),
and found `fused=0` — the prototyped directly-built-child fusion (`build_fix_scan`,
gated by `fused_directly_built_eligible`, `FUSED_MAX_NUM_DATA=-1`) is OFF on the
production CUDA path. It was benched "flat-to-negative" ON THE SPOOFED APU, where
sync latency is ~free. On real NVIDIA over PCIe, cutting launches/syncs should pay.

DECISIVE PROBE (zero code change — existing toggles on current master):
  - baseline            : copack default-ON, fused OFF  (= 051 default)
  - fused=1             : LGBM_FUSED_FORCE=1   (fuse build+fix+scan for the smaller child)
  - fused=1 + AT0       : stack the 051 P=1 micro-win (LGBM_AUTOTUNE=0)
  - copack=0            : LGBM_SIBLING_COPACK=0 (confirm copack is currently helping)
  - fused=1 + copack=1  : both forced on
all under LGBM_PHASE_PROF=1. Read launches / syncs / wall / phases per arm.

Reads (in-session deltas — absolute Kaggle walls are NOT cross-session comparable):
  - fused=1 cuts launches/syncs AND drops wall  => 052 GREEN, wire fused default-on for cuda
  - fused=1 flat/worse on launches-dropped       => per-leaf fusion insufficient; the real
                                                    lever is the architectural on-device
                                                    multi-leaf learner (milestone-sized)
"""
import os
import re
import subprocess
import sys

REPO_URL = os.environ.get("REPO_URL", "https://github.com/BectorVoom/lightgbm_rs.git")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


INNER = r'''
import os, sys, time
import numpy as np
from sklearn.datasets import make_classification
which = sys.argv[1]
N_SAMPLES, N_FEATURES, SEED = 500_000, 50, 42
X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=SEED)
X = np.ascontiguousarray(X, dtype=np.float64)
params = dict(objective="binary", metric="binary_logloss", device_type="cuda",
              num_leaves=31, learning_rate=0.1, n_estimators=100, verbose=-1)
if which == "rs":
    import lightgbm_rs as lgb
else:
    import lightgbm as lgb
warm = lgb.LGBMClassifier(**{**params, "n_estimators": 5}); warm.fit(X, y)
t0 = time.time(); m = lgb.LGBMClassifier(**params); m.fit(X, y); dt = time.time() - t0
print(f"RESULT {which} train_time_s={dt:.3f}", flush=True)
'''


def parse_timed(stderr):
    """Capture full phase_prof records; return the TIMED one (max device_launches).
    Reads the ABSOLUTE-ms line (starts with 'before='), NOT the '%:' percentage line."""
    recs, cur = [], {}
    for line in stderr.splitlines():
        m = re.search(r"\[phase_prof:train\]\s+(.*)", line)
        if not m:
            continue
        s = m.group(1).replace("\\n", "")
        if s.startswith("before=") and "hist+split=" in s and "build=" in s:
            cur = {}
            cur["histsplit"] = float(re.search(r"hist\+split=([\d.]+)", s).group(1))
            cur["scan"] = float(re.search(r"scan=([\d.]+)", s).group(1))
            cur["partition"] = float(re.search(r"partition=([\d.]+)", s).group(1))
        elif s.startswith("LOOP:"):
            cur["t1iter"] = float(re.search(r"train_one_iter=([\d.]+)", s).group(1))
            cur["learner"] = float(re.search(r"learner=([\d.]+)", s).group(1))
        elif s.startswith("COUNTS:"):
            cur["launches"] = int(re.search(r"device_launches=(\d+)", s).group(1))
            cur["build_l"] = int(re.search(r"build_resident=(\d+)", s).group(1))
            cur["sub_l"] = int(re.search(r"subtract_resident=(\d+)", s).group(1))
            cur["scan_l"] = int(re.search(r"scan_resident=(\d+)", s).group(1))
            cur["fused_l"] = int(re.search(r"fused=(\d+)", s).group(1))
            cur["syncs"] = int(re.search(r"syncs\)=(\d+)", s).group(1))
        elif s.startswith("BUDGET:"):
            cur["phases"] = float(re.search(r"phases=([\d.]+)", s).group(1))
            recs.append(cur); cur = {}
    if not recs:
        return {}
    return max(recs, key=lambda r: r.get("launches", 0))


def parse_wall(stdout):
    for line in stdout.splitlines():
        if line.startswith("RESULT"):
            return float(line.split("train_time_s=")[-1])
    return float("nan")


def run_arm(label, extra_env, which="rs"):
    env = dict(os.environ)
    env["LGBM_PHASE_PROF"] = "1"
    for k in ("LGBM_FUSED_FORCE", "LGBM_SIBLING_COPACK", "LGBM_AUTOTUNE"):
        env.pop(k, None)
    env.update(extra_env)
    print(f"\n===== ARM {label} ({extra_env}) =====", flush=True)
    res = subprocess.run(f"python3 inner_bench.py {which}", shell=True,
                         capture_output=True, text=True, env=env)
    print("--- stdout ---\n" + res.stdout, flush=True)
    print("--- stderr (phase_prof) ---\n" + res.stderr, flush=True)
    rec = parse_timed(res.stderr)
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
    run(f"cd lightgbm_rs && git checkout {REPO_BRANCH or 'master'} && git pull")
    run("cd lightgbm_rs && git log --oneline -1")
    print("\n########## BUILD lightgbm_rs CUDA wheel (-F cuda) ##########")
    run("rm -rf lightgbm_rs/target/wheels/")
    run("cd lightgbm_rs/crates/lgbm-python && maturin build --release -F cuda -j 2")
    run("pip install $(ls lightgbm_rs/target/wheels/*.whl | head -n 1) --force-reinstall")
    with open("inner_bench.py", "w") as f:
        f.write(INNER)
    run("rm -rf lightgbm_rs/target/autotune ~/.cache/cubecl 2>/dev/null || true", check=False)

    arms = [
        run_arm("baseline",        {}),
        run_arm("fused=1",         {"LGBM_FUSED_FORCE": "1"}),
        run_arm("fused=1+AT0",     {"LGBM_FUSED_FORCE": "1", "LGBM_AUTOTUNE": "0"}),
        run_arm("copack=0",        {"LGBM_SIBLING_COPACK": "0"}),
        run_arm("fused=1+copack=1",{"LGBM_FUSED_FORCE": "1", "LGBM_SIBLING_COPACK": "1"}),
    ]

    print("\n\n==================== SPIKE-052 LAUNCH-FUSION (real CUDA, 500k x 50) ====================")
    hdr = (f"{'arm':18} {'wall_s':>8} {'t1iter':>8} {'learner':>8} {'phases':>8} "
           f"{'launches':>9} {'build':>7} {'subtr':>7} {'scan':>7} {'fused':>7} {'syncs':>7}")
    print(hdr); print("-" * len(hdr))
    for r in arms:
        print(f"{r.get('label',''):18} {r.get('wall',float('nan')):8.3f} "
              f"{r.get('t1iter',float('nan')):8.0f} {r.get('learner',float('nan')):8.0f} "
              f"{r.get('phases',float('nan')):8.0f} {r.get('launches',0):9d} "
              f"{r.get('build_l',0):7d} {r.get('sub_l',0):7d} {r.get('scan_l',0):7d} "
              f"{r.get('fused_l',0):7d} {r.get('syncs',0):7d}")
    base = arms[0]
    print("\n--- read-out (in-session deltas vs baseline) ---")
    for r in arms[1:]:
        dl = r.get("launches", 0) - base.get("launches", 0)
        ds = r.get("syncs", 0) - base.get("syncs", 0)
        dw = r.get("t1iter", float("nan")) - base.get("t1iter", float("nan"))
        print(f"  {r['label']:16}: dlaunches={dl:+6d} dsyncs={ds:+6d} dt1iter={dw:+8.0f} ms")
    print("  (fused=1 cuts launches/syncs AND wall => 052 GREEN; flat/worse => need the "
          "architectural on-device multi-leaf learner)")


if __name__ == "__main__":
    main()
