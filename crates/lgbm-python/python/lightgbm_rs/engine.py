"""High-level training engine: ``train`` (callback list) + ``cv`` (D-09).

Pure-Python orchestration over the validated compiled ``_core`` engine — no new
Rust. The official package mutates a live C++ booster one tree at a time and
pulls eval history out of the engine. The ``_core`` surface here exposes a
batch ``train(params, dataset, num_boost_round) -> Booster`` plus
``Booster.predict``; it does NOT surface per-iteration eval history. So this
module drives the boosting loop in Python by re-training the booster with an
incrementing ``num_boost_round`` each round and evaluating the requested
metric(s) on the validation sets in Python (the metrics are the standard
LightGBM metrics, computed on the transformed predictions ``_core`` returns).

This makes the official ``callbacks=[...]`` protocol — ``early_stopping``,
``log_evaluation``, ``record_evaluation``, ``reset_parameter`` — work
identically, and lets ``cv`` aggregate per-iteration mean/std across folds, all
without any new numerical code.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import _core
from .callback import CallbackEnv, EarlyStopException

__all__ = ["Booster", "CVBooster", "train", "cv"]

# Re-export the compiled booster type name for isinstance checks / typing.
Booster = _core.Booster


# ---------------------------------------------------------------------------
# Metric computation (transformed-prediction metrics, mirroring LightGBM names)
# ---------------------------------------------------------------------------

# Alias table mapping the user-facing metric name to a canonical name.
_METRIC_ALIASES = {
    "l2": "l2",
    "mean_squared_error": "l2",
    "mse": "l2",
    "regression": "l2",
    "regression_l2": "l2",
    "l2_root": "rmse",
    "root_mean_squared_error": "rmse",
    "rmse": "rmse",
    "l1": "l1",
    "mean_absolute_error": "l1",
    "mae": "l1",
    "regression_l1": "l1",
    "binary_logloss": "binary_logloss",
    "binary": "binary_logloss",
    "multi_logloss": "multi_logloss",
    "multiclass": "multi_logloss",
    "softmax": "multi_logloss",
    "auc": "auc",
}

# Whether higher is better for each canonical metric.
_HIGHER_BETTER = {
    "l2": False,
    "rmse": False,
    "l1": False,
    "binary_logloss": False,
    "multi_logloss": False,
    "auc": True,
}

_EPS = 1e-15


def _default_metric_for_objective(objective: str) -> str:
    obj = str(objective)
    if obj in ("binary",):
        return "binary_logloss"
    if obj in ("multiclass", "softmax", "multiclassova"):
        return "multi_logloss"
    return "l2"


def _resolve_metrics(params: Dict[str, Any]) -> List[str]:
    """Resolve the list of canonical metric names from ``params`` (mirrors LightGBM)."""
    metric = params.get("metric", None)
    objective = params.get("objective", "regression")
    if metric is None or metric == "" or metric == "None" or metric == []:
        return [_default_metric_for_objective(objective)]
    if isinstance(metric, str):
        names = [m.strip() for m in metric.split(",") if m.strip()]
    elif isinstance(metric, (list, tuple)):
        names = [str(m).strip() for m in metric]
    else:
        names = [str(metric)]
    resolved: List[str] = []
    for n in names:
        if n in ("", "None"):
            resolved.append(_default_metric_for_objective(objective))
        else:
            resolved.append(_METRIC_ALIASES.get(n, n))
    # De-duplicate, preserving order.
    seen = set()
    out = []
    for n in resolved:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _eval_metric(metric: str, preds: np.ndarray, y_true: np.ndarray) -> float:
    """Compute a canonical metric on transformed predictions ``preds``."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if metric == "l2":
        p = preds.reshape(-1)
        return float(np.mean((p - y) ** 2))
    if metric == "rmse":
        p = preds.reshape(-1)
        return float(math.sqrt(np.mean((p - y) ** 2)))
    if metric == "l1":
        p = preds.reshape(-1)
        return float(np.mean(np.abs(p - y)))
    if metric == "binary_logloss":
        p = np.clip(preds.reshape(-1), _EPS, 1.0 - _EPS)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    if metric == "multi_logloss":
        p = np.clip(preds, _EPS, 1.0 - _EPS)
        idx = y.astype(np.int64)
        return float(-np.mean(np.log(p[np.arange(len(idx)), idx])))
    if metric == "auc":
        p = preds.reshape(-1)
        return _auc(p, y)
    raise ValueError(f"Unsupported metric for evaluation: {metric!r}")


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the rank statistic (ties averaged)."""
    pos = labels > 0.5
    n_pos = int(np.sum(pos))
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = float(np.sum(ranks[pos]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Trained-booster wrapper (the _core.Booster cannot carry Python attributes)
# ---------------------------------------------------------------------------


class _TrainedBooster:
    """Thin wrapper around a ``_core.Booster`` carrying training metadata.

    ``_core.Booster`` is a compiled ``#[pyclass]`` and cannot hold arbitrary
    Python attributes (``best_iteration``, ``params``, ``best_score``), so the
    Python ``train``/``cv`` engine returns this wrapper, delegating ``predict``,
    ``feature_importance``, ``num_iteration`` and ``refit`` to the core booster.
    """

    def __init__(
        self,
        booster: _core.Booster,
        params: Dict[str, Any],
        best_iteration: int = -1,
        best_score: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self._booster = booster
        self.params = params
        self.best_iteration = best_iteration
        self.best_score: Dict[str, Dict[str, float]] = best_score or {}

    def predict(self, data, **kwargs):
        arr = np.asarray(self._booster.predict(np.ascontiguousarray(data, dtype=np.float64)))
        # Mirror official Booster.predict: a single-column output is squeezed to 1-D.
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.reshape(-1)
        return arr

    def feature_importance(self, importance_type: str = "split"):
        return np.asarray(self._booster.feature_importance(importance_type))

    def num_iteration(self) -> int:
        return self._booster.num_iteration()

    def num_trees(self) -> int:
        return self._booster.num_iteration()

    def refit(self, data, label, decay_rate: float = 0.9):
        self._booster.refit(
            np.ascontiguousarray(data, dtype=np.float64),
            np.asarray(label, dtype=np.float64),
            decay_rate,
        )
        return self

    @property
    def core_booster(self) -> _core.Booster:
        return self._booster


class CVBooster:
    """A collection of per-fold boosters (mirrors ``lightgbm.engine.CVBooster``)."""

    def __init__(self) -> None:
        self.boosters: List[_TrainedBooster] = []
        self.best_iteration = -1


# ---------------------------------------------------------------------------
# Public train (callback list over the _core engine)
# ---------------------------------------------------------------------------


def train(
    params: Dict[str, Any],
    train_set,
    num_boost_round: int = 100,
    valid_sets: Optional[Sequence] = None,
    valid_names: Optional[Sequence[str]] = None,
    fobj: Optional[Callable] = None,
    feval: Optional[Callable] = None,
    callbacks: Optional[Sequence[Callable]] = None,
    *,
    _train_label: Optional[np.ndarray] = None,
    _valid_data: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> _TrainedBooster:
    """Train a booster, applying the callback list around each boosting round.

    ``train_set`` is a ``_core.Dataset``. ``valid_sets`` is a list of
    ``(X, y)`` tuples (numpy) used for evaluation/early-stopping; the training
    set's labels are supplied via ``_train_label`` so train-set metrics can be
    evaluated (the official package's ``valid_sets=[train_set]`` idiom).

    Returns a :class:`_TrainedBooster` carrying ``best_iteration`` (0-based, or
    ``-1`` if early stopping did not fire).
    """
    params = dict(params)
    callbacks = list(callbacks or [])
    callbacks_before = sorted((cb for cb in callbacks if getattr(cb, "before_iteration", False)), key=lambda x: getattr(x, "order", 0))
    callbacks_after = sorted((cb for cb in callbacks if not getattr(cb, "before_iteration", False)), key=lambda x: getattr(x, "order", 0))

    metrics = _resolve_metrics(params)

    # Assemble the evaluation sets: list of (name, X, y).
    eval_sets: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if _valid_data is not None:
        names = list(valid_names) if valid_names else [f"valid_{i}" for i in range(len(_valid_data))]
        for i, (Xi, yi) in enumerate(_valid_data):
            nm = names[i] if i < len(names) else f"valid_{i}"
            eval_sets.append((nm, np.ascontiguousarray(Xi, dtype=np.float64), np.asarray(yi, dtype=np.float64)))

    best_booster: Optional[_core.Booster] = None
    best_iteration = -1
    last_booster: Optional[_core.Booster] = None

    for i in range(num_boost_round):
        env_iter = i
        # before_iteration callbacks (e.g. reset_parameter) may mutate params.
        for cb in callbacks_before:
            cb(
                CallbackEnv(
                    model=None,
                    params=params,
                    iteration=env_iter,
                    begin_iteration=0,
                    end_iteration=num_boost_round,
                    evaluation_result_list=None,
                )
            )

        # Train i+1 rounds over the validated _core engine (pure-Python loop).
        booster = _core.train(params, train_set, num_boost_round=i + 1, fobj=fobj, feval=feval)
        last_booster = booster

        # Evaluate the requested metrics on each validation set.
        eval_result_list: List[Tuple[str, str, float, bool]] = []
        for nm, Xi, yi in eval_sets:
            preds = np.asarray(booster.predict(Xi))
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.reshape(-1)
            for metric in metrics:
                value = _eval_metric(metric, preds, yi)
                eval_result_list.append((nm, metric, value, _HIGHER_BETTER.get(metric, False)))

        try:
            for cb in callbacks_after:
                cb(
                    CallbackEnv(
                        model=None,
                        params=params,
                        iteration=env_iter,
                        begin_iteration=0,
                        end_iteration=num_boost_round,
                        evaluation_result_list=eval_result_list,
                    )
                )
        except EarlyStopException as e:
            best_iteration = e.best_iteration
            break

    # If early stopping fired, retrain to the best iteration so predict() uses it.
    if best_iteration >= 0:
        best_booster = _core.train(params, train_set, num_boost_round=best_iteration + 1, fobj=fobj, feval=feval)
        result = best_booster
    else:
        result = last_booster if last_booster is not None else _core.train(params, train_set, num_boost_round=num_boost_round, fobj=fobj, feval=feval)

    return _TrainedBooster(result, params, best_iteration=best_iteration)


# ---------------------------------------------------------------------------
# Cross-validation (pure-Python k-fold over train())
# ---------------------------------------------------------------------------


def _make_n_folds(
    num_data: int,
    folds: Optional[Sequence],
    nfold: int,
    stratified: bool,
    labels: Optional[np.ndarray],
    shuffle: bool,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build (train_idx, valid_idx) pairs, mirroring ``engine._make_n_folds``."""
    if folds is not None:
        return [(np.asarray(tr), np.asarray(te)) for tr, te in folds]

    indices = np.arange(num_data)
    if stratified and labels is not None:
        # Stratified split: keep class proportions per fold.
        rng = np.random.default_rng(seed)
        fold_of = np.empty(num_data, dtype=np.int64)
        for cls in np.unique(labels):
            cls_idx = indices[labels == cls]
            if shuffle:
                rng.shuffle(cls_idx)
            fold_of[cls_idx] = np.arange(len(cls_idx)) % nfold
        out = []
        for k in range(nfold):
            te = indices[fold_of == k]
            tr = indices[fold_of != k]
            out.append((tr, te))
        return out

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = indices.copy()
        rng.shuffle(indices)
    test_folds = np.array_split(indices, nfold)
    out = []
    for k in range(nfold):
        te = test_folds[k]
        tr = np.concatenate([test_folds[j] for j in range(nfold) if j != k])
        out.append((tr, te))
    return out


def cv(
    params: Dict[str, Any],
    train_set: Tuple[np.ndarray, np.ndarray],
    num_boost_round: int = 100,
    folds: Optional[Sequence] = None,
    nfold: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    metrics: Optional[Any] = None,
    fobj: Optional[Callable] = None,
    feval: Optional[Callable] = None,
    callbacks: Optional[Sequence[Callable]] = None,
    seed: int = 0,
    eval_train_metric: bool = False,
    return_cvbooster: bool = False,
) -> Dict[str, List[float]]:
    """K-fold cross-validation (pure-Python orchestration over ``train``).

    ``train_set`` is an ``(X, y)`` tuple of numpy arrays (this engine bins the
    raw rows internally per fold). Returns the official cv-results dict shape:
    ``{"valid <metric>-mean": [...], "valid <metric>-stdv": [...]}``.
    """
    params = dict(params)
    if metrics is not None:
        params["metric"] = metrics
    metric_names = _resolve_metrics(params)

    X, y = train_set
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    num_data = X.shape[0]

    fold_pairs = _make_n_folds(num_data, folds, nfold, stratified, y, shuffle, seed)

    # per metric -> list over iterations -> list over folds of metric value
    per_iter_fold: Dict[str, List[List[float]]] = {m: [[] for _ in range(num_boost_round)] for m in metric_names}
    cvbooster = CVBooster() if return_cvbooster else None

    for tr_idx, te_idx in fold_pairs:
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        ds = _core.Dataset(Xtr, ytr)
        last = None
        for i in range(num_boost_round):
            booster = _core.train(params, ds, num_boost_round=i + 1, fobj=fobj, feval=feval)
            last = booster
            preds = np.asarray(booster.predict(Xte))
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.reshape(-1)
            for m in metric_names:
                per_iter_fold[m][i].append(_eval_metric(m, preds, yte))
        if cvbooster is not None and last is not None:
            cvbooster.boosters.append(_TrainedBooster(last, params))

    results: Dict[str, List[float]] = {}
    for m in metric_names:
        means = []
        stdvs = []
        for i in range(num_boost_round):
            vals = np.asarray(per_iter_fold[m][i], dtype=np.float64)
            means.append(float(np.mean(vals)))
            stdvs.append(float(np.std(vals)))
        results[f"valid {m}-mean"] = means
        results[f"valid {m}-stdv"] = stdvs

    if cvbooster is not None:
        results["cvbooster"] = cvbooster  # type: ignore[assignment]
    return results
