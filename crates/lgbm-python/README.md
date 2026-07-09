# lightgbm-rs

Python bindings for **LightGBM-rs** — a pure-Rust reimplementation of Microsoft's
LightGBM gradient-boosting library, numerically faithful to the C++ reference
(~1e-6 absolute on the CPU anchor) and built on [PyO3](https://pyo3.rs).

The package mirrors the official `lightgbm` low-level surface for the in-scope
APIs, so you can switch `import lightgbm` → `import lightgbm_rs`:

```python
import numpy as np
import lightgbm_rs as lgb

X = np.random.standard_normal((1000, 10))
y = X @ np.random.standard_normal(10)

ds = lgb.Dataset(X, y)
model = lgb.train({"objective": "regression", "num_leaves": 31}, ds, num_boost_round=100)
pred = model.predict(X)  # owned numpy array
```

Input is marshalled into owned Rust buffers and the GIL is released
(`Python::detach`) around the CPU-bound train/predict, so background Python
threads make progress during training.

## Building from source

```bash
pip install maturin
maturin develop --release   # inside this directory, into your active venv
```

See the workspace root for the full project, license, and the numerical-parity
contract.
