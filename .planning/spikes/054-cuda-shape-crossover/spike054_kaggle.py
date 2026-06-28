#!/usr/bin/env python3
"""Spike-054 — CUDA shape crossover: where (if ever) does lgb_rs CUDA approach official?

051+052 closed the cheap-win search: the narrow 50-feat lgb_rs-CUDA gap (~5-6x vs
official) is ARCHITECTURAL — 8570 small serial kernel launches gated by the best-first
build->subtract->scan dependency chain. Launch COUNT scales with leaves (~constant across
feature count), but per-launch WORK scales with feature count. So at WIDE shapes each
build kernel is well-fed and the fixed per-launch overhead amortizes => the launch-bound
fraction should SHRINK and lgb_rs CUDA should get RELATIVELY closer to official.

This sweeps feature count {50, 200, 500} at 500k rows and measures:
  - lgb_rs CUDA wall vs official CUDA wall  => the ratio (does it improve with width?)
  - lgb_rs phase_prof (phases / launches / per-launch work) => does launch-bound shrink?

Reads (in-session ratios; absolute walls drift across Kaggle sessions):
  - ratio (lgb_rs/official) DROPS toward 1 as feats rise => route wide to CUDA; narrow gap
    is the architectural long-pole (confirms 051/052)
  - ratio FLAT/rising                                     => architectural gap dominates at
    every width; GPU not competitive vs official regardless of shape
"""
import os
import re
import subprocess
import sys

REPO_URL = os.environ.get("REPO_URL", "https://github.com/BectorVoom/lightgbm_rs.git")

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
n_features = int(sys.argv[2])
N_SAMPLES, SEED = 500_000, 42
# keep a fixed informative fraction so wider = more real splits, not noise columns
n_inf = max(5, n_features // 2)
X, y = make_classification(n_samples=N_SAMPLES, n_features=n_features,
                           n_informative=n_inf, random_state=SEED)
X = np.ascontiguousarray(X, dtype=np.float64)
params = dict(objective="binary", metric="binary_logloss", device_type="cuda",
              num_leaves=31, learning_rate=0.1, n_estimators=100, verbose=-1)
if which == "rs":
    import lightgbm_rs as lgb
else:
    import lightgbm as lgb
warm = lgb.LGBMClassifier(**{**params, "n_estimators": 5}); warm.fit(X, y)
t0 = time.time(); m = lgb.LGBMClassifier(**params); m.fit(X, y); dt = time.time() - t0
print(f"RESULT {which} feats={n_features} train_time_s={dt:.3f}", flush=True)
'''


def parse_timed(stderr):
    recs, cur = [], {}
    for line in stderr.splitlines():
        m = re.search(r"\[phase_prof:train\]\s+(.*)", line)
        if not m:
            continue
        s = m.group(1).replace("\\n", "")
        if s.startswith("before=") and "hist+split=" in s:
            cur = {"histsplit": float(re.search(r"hist\+split=([\d.]+)", s).group(1))}
        elif s.startswith("LOOP:"):
            cur["t1iter"] = float(re.search(r"train_one_iter=([\d.]+)", s).group(1))
            cur["learner"] = float(re.search(r"learner=([\d.]+)", s).group(1))
        elif s.startswith("COUNTS:"):
            cur["launches"] = int(re.search(r"device_launches=(\d+)", s).group(1))
            cur["syncs"] = int(re.search(r"syncs\)=(\d+)", s).group(1))
        elif s.startswith("BUDGET:"):
            cur["phases"] = float(re.search(r"phases=([\d.]+)", s).group(1))
            recs.append(cur); cur = {}
    return max(recs, key=lambda r: r.get("launches", 0)) if recs else {}


def parse_wall(stdout):
    for line in stdout.splitlines():
        if line.startswith("RESULT"):
            return float(line.split("train_time_s=")[-1])
    return float("nan")


def run_arm(which, feats):
    env = dict(os.environ)
    env["LGBM_PHASE_PROF"] = "1"
    print(f"\n===== ARM {which} feats={feats} =====", flush=True)
    res = subprocess.run(f"python3 inner_bench.py {which} {feats}", shell=True,
                         capture_output=True, text=True, env=env)
    print("--- stdout ---\n" + res.stdout, flush=True)
    if which == "rs":
        print("--- stderr (phase_prof) ---\n" + res.stderr, flush=True)
    rec = parse_timed(res.stderr) if which == "rs" else {}
    rec["wall"] = parse_wall(res.stdout)
    return rec


def main():
    run("pip install -q -U numpy scipy scikit-learn maturin")
    run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    os.environ["PATH"] += ":/root/.cargo/bin"
    if not os.path.exists("lightgbm_rs"):
        run(f"git clone {REPO_URL}")
    else:
        run("cd lightgbm_rs && git checkout -- . && git fetch --all && git checkout master && git pull")
    print("\n########## BUILD lightgbm_rs CUDA wheel ##########")
    run("rm -rf lightgbm_rs/target/wheels/")
    run("cd lightgbm_rs/crates/lgbm-python && maturin build --release -F cuda -j 2")
    run("pip install $(ls lightgbm_rs/target/wheels/*.whl | head -n 1) --force-reinstall")

    # Official LightGBM with CUDA (source build) for the reference ratio.
    print("\n########## BUILD official LightGBM (CUDA) ##########")
    have_off = True
    try:
        run("python3 -c \"import lightgbm as lgb,numpy as np;"
            "lgb.LGBMClassifier(device_type='cuda',num_leaves=2,n_estimators=1)"
            ".fit(np.zeros((10,2)),np.zeros(10))\"")
    except subprocess.CalledProcessError:
        try:
            run("pip uninstall -y lightgbm")
            run("pip install --no-binary lightgbm lightgbm -C cmake.define.USE_CUDA=ON")
        except subprocess.CalledProcessError:
            have_off = False
            print("!!! official CUDA build failed — will report lgb_rs only", flush=True)

    with open("inner_bench.py", "w") as f:
        f.write(INNER)

    FEATS = [50, 200, 500]
    rows = []
    for nf in FEATS:
        run("rm -rf lightgbm_rs/target/autotune ~/.cache/cubecl 2>/dev/null || true", check=False)
        rs = run_arm("rs", nf)
        off = run_arm("off", nf) if have_off else {"wall": float("nan")}
        rows.append((nf, rs, off))

    print("\n\n==================== SPIKE-054 SHAPE CROSSOVER (real CUDA, 500k rows) ====================")
    hdr = (f"{'feats':>6} {'rs_wall':>8} {'off_wall':>9} {'ratio':>6} "
           f"{'t1iter':>8} {'phases':>8} {'launches':>9} {'ms/launch':>9}")
    print(hdr); print("-" * len(hdr))
    for nf, rs, off in rows:
        rw, ow = rs.get("wall", float("nan")), off.get("wall", float("nan"))
        ratio = rw / ow if ow and ow == ow else float("nan")
        ln = rs.get("launches", 0)
        mpl = rs.get("phases", float("nan")) / ln if ln else float("nan")
        print(f"{nf:6d} {rw:8.3f} {ow:9.3f} {ratio:6.2f} "
              f"{rs.get('t1iter',float('nan')):8.0f} {rs.get('phases',float('nan')):8.0f} "
              f"{ln:9d} {mpl:9.3f}")
    print("\n--- read-out ---")
    print("  ratio(rs/official) DROPS as feats rise => narrow gap is architectural; route wide to CUDA")
    print("  ms/launch RISES with feats (more work/launch) but launches ~constant =>")
    print("    launch-bound FRACTION shrinks at width (confirms 051/052 launch-bound diagnosis)")


if __name__ == "__main__":
    main()
