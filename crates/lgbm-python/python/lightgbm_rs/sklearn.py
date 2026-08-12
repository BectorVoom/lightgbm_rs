"""scikit-learn style estimators (PYB-03).

Pure-Python mirror of the official ``lightgbm.sklearn`` hierarchy —
``LGBMModel`` (base) and the three task estimators ``LGBMRegressor`` /
``LGBMClassifier`` / ``LGBMRanker`` — built over the validated compiled
``_core`` engine and the Python :mod:`lightgbm_rs.engine` ``train``. No new
numerical code: ``fit`` marshals ``X``/``y`` into a ``_core.Dataset`` and calls
the engine; ``predict`` delegates to the trained booster.

scikit-learn base classes are used when available (optional import) so the
estimators are usable in sklearn pipelines / ``GridSearchCV``; if scikit-learn
is absent the estimators degrade to plain objects with the same API.

Custom objective / metric callables are bridged to the ``_core`` ``fobj``/
``feval`` contract via :class:`_ObjectiveFunctionWrapper` /
:class:`_EvalFunctionWrapper` (PYB-04), exactly mirroring the official proxies'
``func(y_true, y_pred[, weight[, group]])`` argument-count dispatch.
"""

from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from . import _core
from .engine import _TrainedBooster
from .engine import train as _engine_train

# Optional scikit-learn integration (degrade gracefully if absent).
try:
    from sklearn.base import BaseEstimator
    from sklearn.base import ClassifierMixin as _ClassifierMixin
    from sklearn.base import RegressorMixin as _RegressorMixin
    from sklearn.preprocessing import LabelEncoder as _LabelEncoder

    _SKLEARN_INSTALLED = True
except ImportError:  # pragma: no cover - exercised only without sklearn
    _SKLEARN_INSTALLED = False

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class _RegressorMixin:  # type: ignore[no-redef]
        pass

    class _ClassifierMixin:  # type: ignore[no-redef]
        pass

    _LabelEncoder = None  # type: ignore[assignment]

__all__ = [
    "LGBMModel",
    "LGBMRegressor",
    "LGBMClassifier",
    "LGBMRanker",
]

_MULTICLASS_OBJECTIVES = {"multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr"}


class _ObjectiveFunctionWrapper:
    """Bridge a sklearn-style objective ``func(y_true, y_pred[, w[, g]])`` to ``_core`` ``fobj``.

    ``_core.train`` passes the closure ``(preds, dataset)`` where ``dataset`` is
    the label array (booster.rs ``train_set_as_pyobject``). This proxy flips the
    argument order to the official ``func(y_true, y_pred)`` and dispatches on the
    user callable's arity (2/3/4).
    """

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, preds: np.ndarray, dataset: Any) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.asarray(dataset, dtype=np.float64).reshape(-1)
        argc = len(signature(self.func).parameters)
        if argc == 2:
            return self.func(labels, preds)
        weight = np.ones(len(labels), dtype=np.float64)
        if argc == 3:
            return self.func(labels, preds, weight)
        if argc == 4:
            group = None
            return self.func(labels, preds, weight, group)
        raise TypeError(f"Self-defined objective function should have 2, 3 or 4 arguments, got {argc}")


class _EvalFunctionWrapper:
    """Bridge a sklearn-style metric ``func(y_true, y_pred[, w[, g]])`` to ``_core`` ``feval``."""

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, preds: np.ndarray, dataset: Any) -> Tuple[str, float, bool]:
        labels = np.asarray(dataset, dtype=np.float64).reshape(-1)
        argc = len(signature(self.func).parameters)
        if argc == 2:
            return self.func(labels, preds)
        weight = np.ones(len(labels), dtype=np.float64)
        if argc == 3:
            return self.func(labels, preds, weight)
        if argc == 4:
            group = None
            return self.func(labels, preds, weight, group)
        raise TypeError(f"Self-defined eval function should have 2, 3 or 4 arguments, got {argc}")


class LGBMModel(BaseEstimator):
    """Base estimator implementing the official ``LGBMModel`` API surface (PYB-03)."""

    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        objective: Optional[Union[str, Callable]] = None,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        device: Optional[str] = None,
        gpu_device_id: Optional[int] = None,
        importance_type: str = "split",
        **kwargs: Any,
    ) -> None:
        """Construct the estimator.

        ``n_jobs``, ``device`` and ``gpu_device_id`` are the compute knobs:

        - ``n_jobs`` -> ``num_threads``. ``None`` leaves LightGBM's default (one
          worker per core). Because the underlying thread pool is process-global
          (mirroring C++ ``omp_set_num_threads``), the FIRST value used in a
          process wins; a later, different value warns rather than silently
          training on the wrong worker count.
        - ``device`` -> ``device_type``: ``"cpu"`` (default), ``"gpu"`` or
          ``"cuda"``. Selects the compute backend at fit time. A device whose
          backend was not compiled into this wheel raises ``ValueError``; call
          :func:`lightgbm_rs.get_device_capabilities` to see what is available.
        - ``gpu_device_id`` -> the CubeCL device index when training on a GPU.
          Ignored on ``device="cpu"``.
        """
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device
        self.gpu_device_id = gpu_device_id
        self.importance_type = importance_type
        self._other_params: Dict[str, Any] = dict(kwargs)
        self._objective: Optional[Union[str, Callable]] = objective
        self._booster: Optional[_TrainedBooster] = None
        self._n_features: int = -1
        self._best_iteration: int = -1

    # --- params plumbing (sklearn get_params/set_params) ------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = {
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "objective": self.objective,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "device": self.device,
            "gpu_device_id": self.gpu_device_id,
            "importance_type": self.importance_type,
        }
        params.update(self._other_params)
        return params

    def set_params(self, **params: Any) -> "LGBMModel":
        for key, value in params.items():
            if hasattr(self, key) and not key.startswith("_"):
                setattr(self, key, value)
            else:
                self._other_params[key] = value
            if key == "objective":
                self._objective = value
        return self

    # --- core params translation -----------------------------------------

    def _build_core_params(self) -> Dict[str, Any]:
        """Translate the sklearn attribute set into a ``_core`` params dict."""
        params: Dict[str, Any] = {
            "boosting": self.boosting_type,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_child_samples,
            "bagging_fraction": self.subsample,
            "feature_fraction": self.colsample_bytree,
            "lambda_l1": self.reg_alpha,
            "lambda_l2": self.reg_lambda,
        }
        if self.random_state is not None:
            params["seed"] = int(self.random_state)
        if self.n_jobs is not None:
            params["num_threads"] = int(self.n_jobs)
        # Device selection. `device_type` picks the backend at fit time and
        # `gpu_device_id` the CubeCL device index; both are left absent (not
        # defaulted here) when unset, so the core's own defaults apply and a
        # `device_type` passed through **kwargs is not overwritten.
        if self.device is not None:
            params["device_type"] = str(self.device)
        if self.gpu_device_id is not None:
            params["gpu_device_id"] = int(self.gpu_device_id)
        params.update(self._other_params)
        params.setdefault("verbosity", -1)
        return params

    def _resolve_objective(self) -> Union[str, Callable, None]:
        return self._objective

    # --- fit / predict ----------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
        eval_set: Optional[List[Tuple[Any, Any]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_metric: Optional[Union[str, Callable]] = None,
        callbacks: Optional[List[Callable]] = None,
        **kwargs: Any,
    ) -> "LGBMModel":
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self._n_features = X.shape[1]

        params = self._build_core_params()
        obj = self._resolve_objective()

        fobj = None
        if callable(obj):
            fobj = _ObjectiveFunctionWrapper(obj)
        elif obj is not None:
            params["objective"] = obj

        feval = None
        if callable(eval_metric):
            feval = _EvalFunctionWrapper(eval_metric)
        elif isinstance(eval_metric, str):
            params["metric"] = eval_metric

        # Validation sets become (X, y) numpy tuples for the Python engine.
        valid_data = None
        valid_names = None
        if eval_set:
            valid_data = [
                (np.ascontiguousarray(np.asarray(vx, dtype=np.float64)), np.asarray(vy, dtype=np.float64).reshape(-1))
                for vx, vy in eval_set
            ]
            valid_names = list(eval_names) if eval_names else None

        train_set = _core.Dataset(X, y)
        self._booster = _engine_train(
            params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_data,
            valid_names=valid_names,
            fobj=fobj,
            feval=feval,
            callbacks=callbacks,
            _train_label=y,
            _valid_data=valid_data,
        )
        self._best_iteration = self._booster.best_iteration
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Estimator not fitted, call fit before exploiting the model.")
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        return self._booster.predict(X)

    # --- introspection properties ----------------------------------------

    @property
    def n_features_in_(self) -> int:
        if self._n_features < 0:
            raise AttributeError("No n_features_in_ found. Need to call fit beforehand.")
        return self._n_features

    @property
    def booster_(self) -> _TrainedBooster:
        if self._booster is None:
            raise RuntimeError("No booster found. Need to call fit beforehand.")
        return self._booster

    @property
    def best_iteration_(self) -> int:
        return self._best_iteration

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("No feature_importances found. Need to call fit beforehand.")
        return self._booster.feature_importance(importance_type=self.importance_type)

    @property
    def objective_(self) -> Union[str, Callable, None]:
        return self._objective

    # --- pickle support (D-10) -------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle a fitted estimator over the booster's model string (D-10).

        sklearn pipelines pickle their steps; the compiled ``_core.Booster`` is
        carried through the wrapper's own ``__getstate__`` (model text), and the
        rest of the estimator is plain attributes. A non-string custom objective/
        metric callable is intentionally NOT serialized here beyond what ``pickle``
        can already handle on the attribute dict (it is restored on unpickle if it
        is itself picklable, matching the official package's behavior).
        """
        return dict(self.__dict__)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


class LGBMRegressor(_RegressorMixin, LGBMModel):
    """LightGBM-rs regressor (mirrors ``lightgbm.LGBMRegressor``)."""

    def _resolve_objective(self) -> Union[str, Callable, None]:
        if self._objective is None:
            return "regression"
        return self._objective

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "LGBMRegressor":  # type: ignore[override]
        super().fit(X, y, **kwargs)
        return self


class LGBMClassifier(_ClassifierMixin, LGBMModel):
    """LightGBM-rs classifier with ``predict_proba`` / ``classes_`` (mirrors official)."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "LGBMClassifier":  # type: ignore[override]
        y_arr = np.asarray(y)
        if _SKLEARN_INSTALLED and _LabelEncoder is not None:
            self._le = _LabelEncoder().fit(y_arr)
            self._classes = self._le.classes_
            y_encoded = self._le.transform(y_arr).astype(np.float64)
        else:
            self._classes = np.unique(y_arr)
            mapping = {c: i for i, c in enumerate(self._classes)}
            y_encoded = np.asarray([mapping[v] for v in y_arr], dtype=np.float64)
        self._n_classes = len(self._classes)

        # Choose the default objective based on class count when not set.
        if self._objective is None or self._objective in (None, "binary", "multiclass"):
            if self._n_classes > 2:
                self._objective = "multiclass"
                self._other_params["num_class"] = int(self._n_classes)
            else:
                self._objective = "binary"

        super().fit(X, y_encoded, **kwargs)
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        proba = self.predict_proba(X, **kwargs)
        class_index = np.argmax(proba, axis=1)
        return self._classes[class_index]

    def predict_proba(self, X: Any, **kwargs: Any) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Estimator not fitted, call fit before exploiting the model.")
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        raw = np.asarray(self._booster.predict(X))
        if raw.ndim == 1:
            # binary: _core returns P(class==1); expand to two columns.
            proba_pos = raw.reshape(-1)
            return np.column_stack([1.0 - proba_pos, proba_pos])
        return raw

    @property
    def classes_(self) -> np.ndarray:
        if not hasattr(self, "_classes"):
            raise AttributeError("No classes_ found. Need to call fit beforehand.")
        return self._classes

    @property
    def n_classes_(self) -> int:
        if not hasattr(self, "_n_classes"):
            raise AttributeError("No n_classes_ found. Need to call fit beforehand.")
        return self._n_classes


class LGBMRanker(LGBMModel):
    """LightGBM-rs ranker (lambdarank default; mirrors ``lightgbm.LGBMRanker``)."""

    def _resolve_objective(self) -> Union[str, Callable, None]:
        if self._objective is None:
            return "lambdarank"
        return self._objective

    def fit(  # type: ignore[override]
        self,
        X: Any,
        y: Any,
        group: Any = None,
        **kwargs: Any,
    ) -> "LGBMRanker":
        if group is not None:
            # Ranking group sizes flow through the params dict to _core.
            self._other_params["group"] = ",".join(str(int(g)) for g in np.asarray(group).reshape(-1))
        super().fit(X, y, **kwargs)
        return self
