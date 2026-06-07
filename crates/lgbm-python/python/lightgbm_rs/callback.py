"""Training-callback list protocol (D-09).

Pure-Python mirror of the official ``lightgbm.callback`` module: the
``callbacks=[...]`` API used by :func:`lightgbm_rs.train` / :func:`lightgbm_rs.cv`.
Each factory returns a callable carrying ``order`` and ``before_iteration``
attributes; the engine sorts the list by ``order`` and dispatches the
before-/after-iteration callbacks around each boosting round.

The four official factories are mirrored exactly by name and semantics:

- :func:`early_stopping` — stop when the validation score stops improving.
- :func:`log_evaluation` — log eval results every ``period`` rounds.
- :func:`record_evaluation` — record the eval history into a dict.
- :func:`reset_parameter` — change a parameter per round (``before_iteration``).

Unlike the official package (which mutates a live C++ booster in place), the
``lightgbm_rs`` engine rebuilds the booster with an incrementing
``num_boost_round`` each iteration over the validated ``_core.train`` surface
(no new Rust). The callback protocol on top is identical.
"""

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

__all__ = [
    "CallbackEnv",
    "EarlyStopException",
    "early_stopping",
    "log_evaluation",
    "record_evaluation",
    "reset_parameter",
]

# An eval-result tuple is ``(dataset_name, metric_name, value, is_higher_better)``
# from train(), or ``(dataset_name, metric_name, mean, is_higher_better, stdv)``
# from cv().
_EvalResultTuple = Any
_ListOfEvalResultTuples = List[_EvalResultTuple]
_EvalResultDict = Dict[str, Dict[str, List[Any]]]


class EarlyStopException(Exception):
    """Raised by the early-stopping callback to break the boosting loop.

    Mirrors ``lightgbm.callback.EarlyStopException``. The engine catches it and
    truncates the model to ``best_iteration`` (0-based).
    """

    def __init__(self, best_iteration: int, best_score: _ListOfEvalResultTuples) -> None:
        super().__init__()
        self.best_iteration = best_iteration
        self.best_score = best_score


@dataclass
class CallbackEnv:
    """The environment passed to every callback (mirrors the official dataclass)."""

    model: Any
    params: Dict[str, Any]
    iteration: int
    begin_iteration: int
    end_iteration: int
    evaluation_result_list: Optional[_ListOfEvalResultTuples]


def _format_eval_result(value: _EvalResultTuple, show_stdv: bool) -> str:
    dataset_name, metric_name, metric_value, *rest = value
    out = f"{dataset_name}'s {metric_name}: {metric_value:g}"
    if show_stdv and len(value) == 5:
        out += f" + {value[4]:g}"
    return out


class _LogEvaluationCallback:
    def __init__(self, period: int = 1, show_stdv: bool = True) -> None:
        self.order = 10
        self.before_iteration = False
        self.period = period
        self.show_stdv = show_stdv

    def __call__(self, env: CallbackEnv) -> None:
        if self.period > 0 and env.evaluation_result_list and (env.iteration + 1) % self.period == 0:
            result = "\t".join(_format_eval_result(x, self.show_stdv) for x in env.evaluation_result_list)
            print(f"[{env.iteration + 1}]\t{result}")


def log_evaluation(period: int = 1, show_stdv: bool = True) -> _LogEvaluationCallback:
    """Create a callback that logs the evaluation results every ``period`` rounds."""
    return _LogEvaluationCallback(period=period, show_stdv=show_stdv)


class _RecordEvaluationCallback:
    def __init__(self, eval_result: _EvalResultDict) -> None:
        self.order = 20
        self.before_iteration = False
        if not isinstance(eval_result, dict):
            raise TypeError("eval_result should be a dictionary")
        self.eval_result = eval_result

    def _init(self, env: CallbackEnv) -> None:
        if env.evaluation_result_list is None:
            raise RuntimeError("record_evaluation() callback enabled but no evaluation results found")
        self.eval_result.clear()
        for item in env.evaluation_result_list:
            dataset_name, metric_name, *_ = item
            self.eval_result.setdefault(dataset_name, OrderedDict())
            if len(item) == 4:
                self.eval_result[dataset_name].setdefault(metric_name, [])
            else:
                self.eval_result[dataset_name].setdefault(f"{metric_name}-mean", [])
                self.eval_result[dataset_name].setdefault(f"{metric_name}-stdv", [])

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if env.evaluation_result_list is None:
            raise RuntimeError("record_evaluation() callback enabled but no evaluation results found")
        for item in env.evaluation_result_list:
            dataset_name, metric_name, metric_value, *_ = item
            if len(item) == 4:
                self.eval_result[dataset_name][metric_name].append(metric_value)
            else:
                metric_std_dev = item[4]
                self.eval_result[dataset_name][f"{metric_name}-mean"].append(metric_value)
                self.eval_result[dataset_name][f"{metric_name}-stdv"].append(metric_std_dev)


def record_evaluation(eval_result: _EvalResultDict) -> Callable:
    """Create a callback that records the evaluation history into ``eval_result``."""
    return _RecordEvaluationCallback(eval_result=eval_result)


class _ResetParameterCallback:
    def __init__(self, **kwargs: Union[list, Callable]) -> None:
        self.order = 10
        self.before_iteration = True
        self.kwargs = kwargs

    def __call__(self, env: CallbackEnv) -> None:
        new_parameters = {}
        for key, value in self.kwargs.items():
            if isinstance(value, list):
                if len(value) != env.end_iteration - env.begin_iteration:
                    raise ValueError(f"Length of list {key!r} has to be equal to 'num_boost_round'.")
                new_param = value[env.iteration - env.begin_iteration]
            elif callable(value):
                new_param = value(env.iteration - env.begin_iteration)
            else:
                raise ValueError(
                    "Only list and callable values are supported "
                    "as a mapping from boosting round index to new parameter value."
                )
            if new_param != env.params.get(key, None):
                new_parameters[key] = new_param
        if new_parameters:
            env.params.update(new_parameters)


def reset_parameter(**kwargs: Union[list, Callable]) -> Callable:
    """Create a callback that resets parameters per boosting round (before_iteration)."""
    return _ResetParameterCallback(**kwargs)


def _should_enable_early_stopping(stopping_rounds: Any) -> bool:
    if not isinstance(stopping_rounds, int):
        raise TypeError(f"early_stopping_round should be an integer. Got '{type(stopping_rounds).__name__}'")
    return stopping_rounds > 0


class _EarlyStoppingCallback:
    def __init__(
        self,
        stopping_rounds: int,
        first_metric_only: bool = False,
        verbose: bool = True,
        min_delta: Union[float, List[float]] = 0.0,
    ) -> None:
        self.enabled = _should_enable_early_stopping(stopping_rounds)
        self.order = 30
        self.before_iteration = False
        self.stopping_rounds = stopping_rounds
        self.first_metric_only = first_metric_only
        self.verbose = verbose
        self.min_delta = min_delta
        self._reset_storages()

    def _reset_storages(self) -> None:
        self.best_score: List[float] = []
        self.best_iter: List[int] = []
        self.best_score_list: List[_ListOfEvalResultTuples] = []
        self.cmp_op: List[Callable[[float, float], bool]] = []
        self.first_metric = ""

    def _gt_delta(self, curr_score: float, best_score: float, delta: float) -> bool:
        return curr_score > best_score + delta

    def _lt_delta(self, curr_score: float, best_score: float, delta: float) -> bool:
        return curr_score < best_score - delta

    def _is_train_set(self, dataset_name: str, env: CallbackEnv) -> bool:
        # In lightgbm_rs the training set, when added to valid_sets, is named "train".
        return dataset_name == "train"

    def _init(self, env: CallbackEnv) -> None:
        if not env.evaluation_result_list:
            raise ValueError("For early stopping, at least one dataset and eval metric is required for evaluation")

        is_dart = str(env.params.get("boosting", "")) == "dart" or str(env.params.get("boosting_type", "")) == "dart"
        if is_dart:
            self.enabled = False
            return

        first_dataset_name, first_metric_name, *_ = env.evaluation_result_list[0]

        only_train_set = len(env.evaluation_result_list) == 1 and self._is_train_set(first_dataset_name, env)
        if only_train_set:
            self.enabled = False
            return

        if self.verbose:
            print(f"Training until validation scores don't improve for {self.stopping_rounds} rounds")

        self._reset_storages()

        n_metrics = len({m[1] for m in env.evaluation_result_list})
        n_datasets = len(env.evaluation_result_list) // n_metrics
        if isinstance(self.min_delta, list):
            if not all(t >= 0 for t in self.min_delta):
                raise ValueError("Values for early stopping min_delta must be non-negative.")
            if len(self.min_delta) == 0:
                deltas = [0.0] * n_datasets * n_metrics
            elif len(self.min_delta) == 1:
                deltas = self.min_delta * n_datasets * n_metrics
            else:
                if len(self.min_delta) != n_metrics:
                    raise ValueError("Must provide a single value for min_delta or as many as metrics.")
                deltas = self.min_delta * n_datasets
        else:
            if self.min_delta < 0:
                raise ValueError("Early stopping min_delta must be non-negative.")
            deltas = [self.min_delta] * n_datasets * n_metrics

        self.first_metric = first_metric_name
        for eval_ret, delta in zip(env.evaluation_result_list, deltas):
            self.best_iter.append(0)
            if eval_ret[3]:  # higher is better
                self.best_score.append(float("-inf"))
                self.cmp_op.append(partial(self._gt_delta, delta=delta))
            else:
                self.best_score.append(float("inf"))
                self.cmp_op.append(partial(self._lt_delta, delta=delta))

    def _final_iteration_check(self, *, env: CallbackEnv, metric_name: str, i: int) -> None:
        if env.iteration == env.end_iteration - 1:
            if self.verbose:
                best_score_str = "\t".join(_format_eval_result(x, True) for x in self.best_score_list[i])
                print(f"Did not meet early stopping. Best iteration is:\n[{self.best_iter[i] + 1}]\t{best_score_str}")
            raise EarlyStopException(self.best_iter[i], self.best_score_list[i])

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if not self.enabled:
            return
        if env.evaluation_result_list is None:
            raise RuntimeError("early_stopping() callback enabled but no evaluation results found")
        first_time_updating_best_score_list = self.best_score_list == []
        for i in range(len(env.evaluation_result_list)):
            dataset_name, metric_name, metric_value, *_ = env.evaluation_result_list[i]
            if first_time_updating_best_score_list or self.cmp_op[i](metric_value, self.best_score[i]):
                self.best_score[i] = metric_value
                self.best_iter[i] = env.iteration
                if first_time_updating_best_score_list:
                    self.best_score_list.append(env.evaluation_result_list)
                else:
                    self.best_score_list[i] = env.evaluation_result_list
            if self.first_metric_only and self.first_metric != metric_name:
                continue
            if self._is_train_set(dataset_name, env):
                continue
            elif env.iteration - self.best_iter[i] >= self.stopping_rounds:
                if self.verbose:
                    eval_result_str = "\t".join(_format_eval_result(x, True) for x in self.best_score_list[i])
                    print(f"Early stopping, best iteration is:\n[{self.best_iter[i] + 1}]\t{eval_result_str}")
                raise EarlyStopException(self.best_iter[i], self.best_score_list[i])
            self._final_iteration_check(env=env, metric_name=metric_name, i=i)


def early_stopping(
    stopping_rounds: int,
    first_metric_only: bool = False,
    verbose: bool = True,
    min_delta: Union[float, List[float]] = 0.0,
) -> _EarlyStoppingCallback:
    """Create a callback that activates early stopping (mirrors the official factory)."""
    return _EarlyStoppingCallback(
        stopping_rounds=stopping_rounds,
        first_metric_only=first_metric_only,
        verbose=verbose,
        min_delta=min_delta,
    )
