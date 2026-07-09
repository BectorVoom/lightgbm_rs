"""Plotting helpers (D-09): ``plot_importance`` / ``plot_tree`` / ``plot_metric``.

Mirror the official ``lightgbm.plotting`` surface. matplotlib and graphviz are
OPTIONAL dependencies — they are imported lazily INSIDE each function so the
package imports cleanly without them; a missing dep raises an informative
``ImportError`` only when the corresponding plot is requested.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = ["plot_importance", "plot_tree", "plot_metric"]


def _get_booster(model: Any) -> Any:
    """Accept either an estimator (``.booster_``) or a (trained) booster directly."""
    if hasattr(model, "booster_"):
        return model.booster_
    return model


def plot_importance(
    booster: Any,
    ax: Any = None,
    height: float = 0.2,
    importance_type: str = "split",
    title: str = "Feature importance",
    xlabel: str = "Feature importance",
    ylabel: str = "Features",
    max_num_features: Optional[int] = None,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    precision: Optional[int] = 3,
    **kwargs: Any,
) -> Any:
    """Plot per-feature importances as a horizontal bar chart (matplotlib optional)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError("matplotlib is required to plot importance — install lightgbm_rs[plotting].") from e

    bst = _get_booster(booster)
    importance = np.asarray(bst.feature_importance(importance_type))
    n = len(importance)
    feature_names = [f"Column_{i}" for i in range(n)]

    order = np.argsort(importance)
    values = importance[order]
    names = [feature_names[i] for i in order]
    if max_num_features is not None and max_num_features > 0:
        values = values[-max_num_features:]
        names = names[-max_num_features:]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align="center", height=height, **kwargs)
    for x, y in zip(values, ylocs):
        label = f"{x:.{precision}f}" if precision is not None else str(x)
        ax.text(x + 1, y, label, va="center")
    ax.set_yticks(ylocs)
    ax.set_yticklabels(names)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(grid)
    return ax


def plot_metric(
    eval_result: Union[Dict[str, Dict[str, List[float]]], Any],
    metric: Optional[str] = None,
    dataset_names: Optional[List[str]] = None,
    ax: Any = None,
    xlabel: str = "Iterations",
    ylabel: str = "auto",
    title: str = "Metric during training",
    figsize: Optional[Tuple[float, float]] = None,
    grid: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot one metric's curve over boosting iterations (matplotlib optional).

    ``eval_result`` is the dict populated by :func:`lightgbm_rs.record_evaluation`
    (``{dataset_name: {metric_name: [values...]}}``) or the cv-results dict.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError("matplotlib is required to plot metric — install lightgbm_rs[plotting].") from e

    if not isinstance(eval_result, dict) or not eval_result:
        raise ValueError("eval_result should be a non-empty dictionary")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if dataset_names is None:
        dataset_names = list(eval_result.keys())

    resolved_metric = metric
    for name in dataset_names:
        metric_dict = eval_result[name]
        if resolved_metric is None:
            resolved_metric = next(iter(metric_dict))
        results = metric_dict[resolved_metric]
        xs = range(1, len(results) + 1)
        ax.plot(xs, results, label=name, **kwargs)

    ax.legend(loc="best")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(resolved_metric if ylabel == "auto" else ylabel)
    if title:
        ax.set_title(title)
    ax.grid(grid)
    return ax


def plot_tree(
    booster: Any,
    ax: Any = None,
    tree_index: int = 0,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> Any:
    """Render a single tree via graphviz (matplotlib + graphviz optional).

    .. note::

        Tree rendering requires the C++-compatible model text dump
        (``model_to_string``), which lands with model persistence (D-10) — a
        separate plan. Until then this raises an informative
        :class:`NotImplementedError`; the matplotlib/graphviz optional-dep
        contract is still honoured (those imports are attempted first so a
        missing dep surfaces the standard ImportError).
    """
    try:
        import graphviz  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without graphviz
        raise ImportError("graphviz is required to plot trees — install lightgbm_rs[plotting].") from e
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError("matplotlib is required to plot trees — install lightgbm_rs[plotting].") from e

    raise NotImplementedError(
        "plot_tree requires the model text dump (model_to_string), which lands with "
        "model persistence (D-10) — not yet available in this build."
    )
