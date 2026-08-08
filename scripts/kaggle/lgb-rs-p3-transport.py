#!/usr/bin/env python3
"""P3 transport levers (Kaggle P100) — same-wheel env-toggled A/B of the cubecl
dispatch-layer forks, plus the first real per-launch host-cost decomposition.

Context (docs/ondevice-cuda-perf-plan.md §11-12): post-P2 state is rs 5.72s vs
official 2.79s, and the drained grow (4.05s) is DISPATCH-bound — ~18.5k launches
+ 6.2k blocking syncs/train through cubecl 0.10. Two fork levers attack the
per-launch transport tax; both are env-gated in the wheel built from main:

  CUBECL_DEVICE_INLINE=1   — vendored cubecl-common: run device tasks INLINE on
                             the caller thread (reentrant mutex) instead of
                             crossing a channel to a dedicated server thread.
  CUBECL_CUDA_INFO_ARENA=1 — vendored cubecl-cuda: per-launch info upload
                             (sm_60 has no grid_constants) via a persistent
                             pinned+device ring instead of pool reserve →
                             Bytes staging → drop-queue blocking flush.
  CUBECL_CUDA_LAUNCH_PROF=1 — fork profiler (diagnostic runs only).

v2 hardening after the v1 incident: official PINNED to lightgbm==4.6.0 (v1's
unpinned build resolved 4.7.0 whose first CUDA worker hung 8h on the P100),
900s per-arm timeout, corpus-ready marker.

Arms (order ROTATED, N_RUNS warm-median, fresh process per run, one wheel):
  official         — official LightGBM 4.6.0 CUDA (source build)
  rs_base          — defaults (fork code present, levers OFF = upstream behavior)
  rs_arena         — info arena only
  rs_inline        — inline device handle only
  rs_inline_arena  — both (candidate ship config)

Gates: NUM_TREES==100 every run; ALL rs arms' preds BYTE-IDENTICAL to rs_base.
Diagnostics after the timed arms: LAUNCH_PROF runs (base / inline / inline_arena)
and drain-mode ledgers (base / inline_arena).

SECURITY: no credentials embedded; repo is public BectorVoom/lightgbm_rs.
"""
import json
import os
import re
import statistics
import subprocess
import sys

BENCH_ROOT = os.environ.get("BENCH_ROOT", "/kaggle/working")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

N_SAMPLES, N_FEATURES = 500_000, 50
N_ESTIMATORS = 100
NUM_LEAVES = 31
N_RUNS = 3
SEED = 42
N_PRED = 50_000

_WORKER_SRC = r'''
import json, os, sys, time
import numpy as np

cfg = json.loads(sys.argv[1])
data = np.load(cfg["data_path"])
X = data["X"]; y = data["y"]

params = dict(cfg["params"])
params.setdefault("verbosity", -1)
params.setdefault("num_threads", 0)
n_pred = cfg["n_pred"]

try:
    if cfg["backend"] == "official":
        import lightgbm as lgb
        train_set = lgb.Dataset(X, y)
        start = time.time()
        booster = lgb.train(params, train_set, num_boost_round=cfg["num_boost_round"])
        wall = time.time() - start
        num_trees = booster.num_trees()
        preds = booster.predict(X[:n_pred])
    else:
        import lightgbm_rs as lgb_rs
        train_set = lgb_rs.Dataset(X, y)
        start = time.time()
        booster = lgb_rs.train(params, train_set, num_boost_round=cfg["num_boost_round"])
        wall = time.time() - start
        num_trees = booster.num_trees()
        preds = booster.predict(X[:n_pred])
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"ARM_ERROR={type(e).__name__}: {e}")
    sys.exit(3)

np.save(cfg["pred_path"], np.asarray(preds, dtype=np.float64))
print(f"NUM_TREES={num_trees}")
print(f"WALLCLOCK={wall}")
'''

NUM_TREES_RE = re.compile(r"^NUM_TREES=(-?\d+)$", re.MULTILINE)
CONSTRUCT_RE = re.compile(r"^OFFICIAL_CONSTRUCT=([0-9.]+)$", re.MULTILINE)
TRAIN_RE = re.compile(r"^OFFICIAL_TRAIN=([0-9.]+)$", re.MULTILINE)
BINNING_RE = re.compile(r"binning=([0-9.]+)ms")
ARM_ERROR_RE = re.compile(r"^ARM_ERROR=(.*)$", re.MULTILINE)
COUNTS_RE = re.compile(r"COUNTS: .*grad_passthru=(\d+) grow_pool=(\d+)")
LAUNCH_PROF_RE = re.compile(r"^cubecl-launch-prof.*$", re.MULTILINE)

# Round 3b (fast resolve): wheel defaults ship inline + funcattr-once + the
# two-level module cache (bucket by type-name ptr/mode/cube-dim, full-KernelId
# EQUALITY inside — skips the ~85us/launch full-id hash). rs_slow isolates it.
ARMS = {
    "official": {"backend": "official", "env": {}},
    "rs": {"backend": "rs", "env": {}},
}
ARM_ORDER = ["official", "rs"]


def run(cmd, check=True):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=check)


def run_worker(worker_path, data_path, arm, params, pred_path, extra_env=None, n_rounds=None):
    env = dict(os.environ)
    if arm["backend"] != "official":
        env["LGBM_PHASE_PROF"] = "1"
        env["LGBM_AUTOTUNE"] = "0"
        env.update(arm["env"])
    if extra_env:
        env.update(extra_env)
    cfg = {
        "data_path": data_path,
        "backend": arm["backend"],
        "params": params,
        "num_boost_round": n_rounds if n_rounds is not None else N_ESTIMATORS,
        "pred_path": pred_path,
        "n_pred": N_PRED,
    }
    try:
        proc = subprocess.run(
            [sys.executable, worker_path, json.dumps(cfg)],
            env=env, capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired as te:
        print("  ARM_TIMEOUT after 900s (arm hung — killed)", flush=True)
        class _P:  # minimal stand-in
            stdout = (te.stdout or b"").decode() if isinstance(te.stdout, bytes) else (te.stdout or "")
            stderr = (te.stderr or b"").decode() if isinstance(te.stderr, bytes) else (te.stderr or "")
        proc = _P()
    wall = None
    for line in proc.stdout.splitlines():
        if line.startswith("WALLCLOCK="):
            wall = float(line.split("=", 1)[1])
    m = NUM_TREES_RE.search(proc.stdout)
    m_con = CONSTRUCT_RE.search(proc.stdout)
    m_tr = TRAIN_RE.search(proc.stdout)
    binnings = BINNING_RE.findall(proc.stderr)
    e = ARM_ERROR_RE.search(proc.stdout)
    counts = COUNTS_RE.findall(proc.stderr)
    passthru = int(counts[-1][0]) if counts else None
    prof_lines = LAUNCH_PROF_RE.findall(proc.stderr)
    if wall is None:
        print("  worker stdout:", proc.stdout[-3000:])
        print("  worker stderr:", proc.stderr[-3000:])
    return {
        "wall": wall,
        "num_trees": int(m.group(1)) if m else None,
        "arm_error": e.group(1) if e else None,
        "grad_passthru": passthru,
        "launch_prof_last": prof_lines[-1] if prof_lines else None,
        "official_construct": float(m_con.group(1)) if m_con else None,
        "official_train": float(m_tr.group(1)) if m_tr else None,
        "rs_binning_ms": float(binnings[-1]) if binnings else None,
        "launch_prof_all": prof_lines[-2:] if prof_lines else None,
        "stderr_tail": proc.stderr[-6000:],
    }


def warm_median(walls):
    vals = [w for w in walls if w is not None]
    warm = vals[1:] if len(vals) > 1 else vals
    return statistics.median(warm) if warm else None


def main():
    run("nvidia-smi || true", check=False)
    run("pip install -U numpy scipy scikit-learn maturin")
    run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    os.environ["PATH"] += ":/root/.cargo/bin"

    run("git clone --depth 1 -b main https://github.com/BectorVoom/lightgbm_rs.git")
    run("cd lightgbm_rs && git rev-parse HEAD")
    run(f"cd lightgbm_rs/crates/lgbm-python && maturin build --release --features cuda -j 2 -o {BENCH_ROOT}/wheels")
    run(f"pip install $(ls {BENCH_ROOT}/wheels/*.whl | head -n 1) --force-reinstall --no-deps -q")

    # official LightGBM with CUDA — PINNED to 4.6.0 (the baseline every prior
    # session measured; v1's unpinned install resolved 4.7.0 whose first CUDA
    # worker hung the session)
    try:
        subprocess.run(
            "python3 -c 'import lightgbm as lgb; import numpy as np; "
            "assert lgb.__version__ == \"4.6.0\", lgb.__version__; "
            "lgb.LGBMRegressor(device_type=\"cuda\", num_leaves=2, n_estimators=1)"
            ".fit(np.zeros((10,2)), np.zeros(10))'", shell=True, check=True)
    except subprocess.CalledProcessError:
        run("pip uninstall -y lightgbm", check=False)
        run("pip install --no-binary lightgbm 'lightgbm==4.6.0' -C cmake.define.USE_CUDA=ON",
            check=False)
        smoke = subprocess.run(
            "python3 -c 'import lightgbm as lgb; import numpy as np; "
            "lgb.LGBMRegressor(device_type=\"cuda\", num_leaves=2, n_estimators=1)"
            ".fit(np.zeros((10,2)), np.zeros(10))'", shell=True)
        if smoke.returncode != 0:
            print("OFFICIAL_UNAVAILABLE: CUDA source build failed on this host — "
                  "skipping the official arm; rs arms still run.", flush=True)
            ARM_ORDER.remove("official")

    import numpy as np
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=N_SAMPLES, n_features=N_FEATURES, n_informative=20,
        noise=5.0, random_state=SEED,
    )
    out_dir = f"{BENCH_ROOT}/p3"
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, "corpus.npz")
    np.savez(data_path, X=X.astype(np.float64), y=y.astype(np.float64))
    print("corpus ready", flush=True)

    worker_path = os.path.join(out_dir, "_worker.py")
    with open(worker_path, "w") as f:
        f.write(_WORKER_SRC)

    params = {"objective": "regression", "num_leaves": NUM_LEAVES,
              "learning_rate": 0.1, "device_type": "cuda", "seed": SEED,
              "min_data_in_leaf": 20}

    walls = {a: [] for a in ARM_ORDER}
    trees_ok = {a: True for a in ARM_ORDER}
    pred_paths = {}

    for r in range(N_RUNS):
        order = ARM_ORDER[r % len(ARM_ORDER):] + ARM_ORDER[:r % len(ARM_ORDER)]
        for arm_name in order:
            arm = ARMS[arm_name]
            pred_path = os.path.join(out_dir, f"pred_{arm_name}_r{r}.npy")
            res = run_worker(worker_path, data_path, arm, params, pred_path)
            print(f"  r={r} {arm_name:16s} wall={res['wall']} trees={res['num_trees']} "
                  f"construct={res['official_construct']} train={res['official_train']} "
                  f"rs_binning_ms={res['rs_binning_ms']} err={res['arm_error']}", flush=True)
            walls[arm_name].append(res["wall"])
            if res["wall"] is not None and res["num_trees"] != N_ESTIMATORS:
                trees_ok[arm_name] = False
            if res["wall"] is not None:
                pred_paths[arm_name] = pred_path

    identity = {}
    if "rs" in pred_paths:
        base = np.load(pred_paths["rs"])
        for a in ():
            if a in pred_paths:
                other = np.load(pred_paths[a])
                identity[a] = float(np.max(np.abs(other - base))) if other.shape == base.shape else None
        if "official" in pred_paths:
            off = np.load(pred_paths["official"])
            identity["official_envelope"] = float(np.max(np.abs(off - base)))

    print("\n=== nsys device-kernel head-to-head (20 trees) ===")
    import shutil
    nsys = shutil.which("nsys")
    if not nsys:
        run("ls /opt/nvidia 2>/dev/null || true", check=False)
        run("apt-get update -qq 2>/dev/null; apt-get install -y -qq nsight-systems-cli 2>&1 | tail -1 || true", check=False)
        nsys = shutil.which("nsys")
    if not nsys:
        cand = subprocess.run("ls /opt/nvidia/nsight-systems*/bin/nsys 2>/dev/null | head -1",
                              shell=True, capture_output=True, text=True).stdout.strip()
        nsys = cand or None
    if nsys:
        for arm_name in ("official", "rs"):
            arm = ARMS[arm_name]
            env = dict(os.environ)
            if arm["backend"] != "official":
                env["LGBM_PHASE_PROF"] = "0"
                env["LGBM_AUTOTUNE"] = "0"
                env.update(arm["env"])
            cfg = {"data_path": data_path, "backend": arm["backend"], "params": params,
                   "num_boost_round": 20, "pred_path": os.path.join(out_dir, f"pred_{arm_name}_nsys.npy"),
                   "n_pred": 1000}
            rep = os.path.join(out_dir, f"prof_{arm_name}")
            try:
                subprocess.run([nsys, "profile", "-o", rep, "--trace=cuda", "-f", "true",
                                sys.executable, worker_path, json.dumps(cfg)],
                               env=env, capture_output=True, text=True, timeout=900)
                st = subprocess.run([nsys, "stats", "--report", "cuda_gpu_kern_sum", rep + ".nsys-rep"],
                                    capture_output=True, text=True, timeout=600)
                print(f"--- nsys cuda_gpu_kern_sum {arm_name} (top) ---")
                for i, ln in enumerate(st.stdout.splitlines()):
                    if i > 45:
                        break
                    if ln.strip():
                        print(ln)
            except Exception as e:
                print(f"  nsys {arm_name} failed: {type(e).__name__}: {e}")
    else:
        print("NSYS_UNAVAILABLE (not in image, apt install failed)")

    print("\n=== 1-tree diagnostics (construct + fixed overhead) ===")
    one_tree = {}
    for arm_name in ("official", "rs"):
        arm = ARMS[arm_name]
        pred_path = os.path.join(out_dir, f"pred_{arm_name}_1tree.npy")
        res = run_worker(worker_path, data_path, arm, params, pred_path, n_rounds=1)
        one_tree[arm_name] = res["wall"]
        print(f"  1tree {arm_name} wall={res['wall']} rs_binning_ms={res['rs_binning_ms']}")

    print("\n=== launch-prof diagnostics (not timed) ===")
    prof_summary = {}
    for arm_name in ("rs",):
        arm = ARMS[arm_name]
        pred_path = os.path.join(out_dir, f"pred_{arm_name}_prof.npy")
        res = run_worker(worker_path, data_path, arm, params, pred_path,
                          extra_env={"CUBECL_CUDA_LAUNCH_PROF": "1"})
        prof_summary[arm_name] = res["launch_prof_all"]
        print(f"  prof {arm_name} wall={res['wall']}")
        for ln in (res["launch_prof_all"] or [])[-2:]:
            print(f"    {ln}")

    print("\n=== drain-mode ledgers ===")
    for arm_name in ("rs",):
        arm = ARMS[arm_name]
        pred_path = os.path.join(out_dir, f"pred_{arm_name}_drain.npy")
        res = run_worker(worker_path, data_path, arm, params, pred_path,
                          extra_env={"LGBM_GROW_DRAIN": "1"})
        print(f"  drain {arm_name} wall={res['wall']}")
        print(res["stderr_tail"][-4500:])

    results = {
        "shape": [N_SAMPLES, N_FEATURES], "n_estimators": N_ESTIMATORS,
        "num_leaves": NUM_LEAVES, "n_runs": N_RUNS,
        "medians_warm_s": {a: warm_median(w) for a, w in walls.items()},
        "raw_walls": walls,
        "tree_count_ok": trees_ok,
        "pred_identity_max_abs_vs_rs_base": identity,
        "launch_prof": prof_summary,
        "one_tree_walls": one_tree,
    }
    print("\n=== RESULTS_JSON ===")
    print(json.dumps(results, indent=2))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    for fn in os.listdir(out_dir):
        if fn.endswith(".npy") or fn == "corpus.npz":
            os.remove(os.path.join(out_dir, fn))


if __name__ == "__main__":
    main()
