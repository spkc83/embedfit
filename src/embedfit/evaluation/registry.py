"""Name-based lookup of positive-class score metrics for ``Trainer.evaluate``."""

from collections.abc import Callable, Hashable, Iterable
from typing import Any

from embedfit.evaluation.calibration import brier_score
from embedfit.evaluation.classification import pr_auc, roc_auc
from embedfit.exceptions import ConfigurationError

MetricFunction = Callable[..., float]
"""``metric(y_true, scores, *, positive_label) -> float``; ``scores`` are positive-class
probabilities, one per sample."""

_METRICS: dict[str, MetricFunction] = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "brier_score": brier_score,
}

# Every name Trainer.evaluate reports by default, so a requested metric never shadows one.
_RESERVED = frozenset(_METRICS) | {
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "precision",
    "recall",
    "f1",
}


def register_metric(name: str, metric: MetricFunction, *, overwrite: bool = False) -> None:
    """Make a custom metric selectable with ``Trainer.evaluate(metrics=[name])``.

    Args:
        name: Non-empty name, reported as the metric's key. The names ``Trainer.evaluate``
            reports by default (``roc_auc``, ``accuracy``, ...) are reserved.
        metric: Callable invoked as ``metric(y_true, scores, *, positive_label)`` that returns
            a float, where ``scores`` holds the predicted probability of ``positive_label``.
            Every ranking and calibration function in ``embedfit.evaluation`` except the
            dollar metrics has this form; bind other options with ``functools.partial``.
        overwrite: Replace an existing registration instead of raising.

    Raises:
        ConfigurationError: If the name is empty, reserved, or already registered, or the
            metric is not callable.
    """
    if not isinstance(name, str) or not name or name != name.strip():
        raise ConfigurationError(f"Invalid metric name {name!r}.")
    if name in _RESERVED:
        raise ConfigurationError(f"Metric name '{name}' is reserved for a built-in metric.")
    if not callable(metric):
        raise ConfigurationError(f"Metric '{name}' must be callable.")
    if name in _METRICS and not overwrite:
        raise ConfigurationError(
            f"Metric '{name}' is already registered; pass overwrite=True to replace."
        )
    _METRICS[name] = metric


def available_metrics() -> list[str]:
    """Return the registered metric names, sorted."""
    return sorted(_METRICS)


def get_metric(name: str) -> MetricFunction:
    """Return the metric registered under ``name``.

    Raises:
        ConfigurationError: If the name is not registered.
    """
    metric = _METRICS.get(name)
    if metric is None:
        raise ConfigurationError(
            f"Unknown metric '{name}'. Available metrics: {available_metrics()}."
        )
    return metric


def requested_metrics(metrics: Iterable[str], positive_label: Hashable | None) -> tuple[str, ...]:
    """Validate the extra metric names passed to ``Trainer.evaluate``.

    Raises:
        ConfigurationError: If ``metrics`` is not an iterable of str or is a bare str, holds an
            unknown, duplicate, or default name, or is non-empty while ``positive_label`` is
            ``None``.
    """
    if isinstance(metrics, str) or not isinstance(metrics, Iterable):
        raise ConfigurationError(
            f"metrics must be a sequence of metric names, got {type(metrics).__name__}."
        )
    names: tuple[Any, ...] = tuple(metrics)
    for name in names:
        if not isinstance(name, str):
            raise ConfigurationError(f"metric names must be str, got {type(name).__name__}.")
        if name in _RESERVED:
            raise ConfigurationError(f"Metric '{name}' is computed by default; do not request it.")
        get_metric(name)
    if len(set(names)) != len(names):
        raise ConfigurationError("metrics must not contain duplicates.")
    if names and positive_label is None:
        raise ConfigurationError(
            "Registered metrics score the positive class; create the model with a positive_label."
        )
    return names
