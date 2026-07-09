import os
import subprocess
import time
import sys

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

try:
    print("Setting up dependencies...")
    run("pip install -U numpy scipy scikit-learn")
    run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    os.environ['PATH'] += ":/root/.cargo/bin"
    
    if not os.path.exists("lightgbm_rs"):
        print("Cloning repository...")
        run("git clone https://github.com/BectorVoom/lightgbm_rs.git")
    else:
        print("Pulling latest changes...")
        run("cd lightgbm_rs && git checkout -- . && git pull")
        
    

    
    print("Building lightgbm_rs...")
    run("pip install maturin")
    run("rm -rf lightgbm_rs/target/wheels/")
    run("cd lightgbm_rs/crates/lgbm-python && maturin build --release -F cuda -j 2")
    run("pip install $(ls lightgbm_rs/target/wheels/*.whl | head -n 1) --force-reinstall")
    
    print("Checking official lightgbm CUDA support...")
    try:
        run("python3 -c 'import lightgbm as lgb; import numpy as np; lgb.LGBMClassifier(device_type=\"cuda\", num_leaves=2, n_estimators=1).fit(np.zeros((10,2)), np.zeros(10))'")
    except subprocess.CalledProcessError:
        print("Installing official lightgbm...")
        run("pip uninstall -y lightgbm")
        run("pip install --no-binary lightgbm lightgbm -C cmake.define.USE_CUDA=ON")
    
    inner_script = """
import time
import sys
import numpy as np
from sklearn.datasets import make_classification
import lightgbm as lgb
import lightgbm_rs as lgb_rs

print("Generating dataset...")
X, y = make_classification(n_samples=500000, n_features=50, random_state=42)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'device_type': 'cuda',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'verbose': -1
}

print("\\n--- Testing Official LightGBM (CUDA) ---")
start = time.time()
lgb_model = lgb.LGBMClassifier(**params)
lgb_model.fit(X, y)
lgb_time = time.time() - start
print(f"LightGBM (CUDA) training time: {lgb_time:.2f} seconds")

print("\\n--- Testing LightGBM-rs (CUDA) ---")
start = time.time()
lgb_rs_model = lgb_rs.LGBMClassifier(**params)
lgb_rs_model.fit(X, y)
lgb_rs_time = time.time() - start
print(f"LightGBM-rs (CUDA) training time: {lgb_rs_time:.2f} seconds")

if lgb_rs_time > 0:
    print(f"\\n>>> Speedup (LightGBM-rs is faster by): {lgb_time / lgb_rs_time:.2f}x <<<")
"""
    with open("benchmark_inner.py", "w") as f:
        f.write(inner_script)
        
    print("Running inner benchmark script...")
    res = subprocess.run("python3 benchmark_inner.py", shell=True, capture_output=True, text=True)
    print("STDOUT:")
    print(res.stdout)
    print("STDERR:")
    print(res.stderr)
    if res.returncode != 0:
        print("Inner benchmark failed.")
        sys.exit(1)

    
except Exception as e:
    print(f"Benchmark failed: {e}")
    sys.exit(1)
