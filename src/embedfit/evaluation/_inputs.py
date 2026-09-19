"""Shared input validation for metrics. Metrics receive arrays and never call a model."""

import math
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np

from embedfit.data.labels import same_label
from embedfit.exceptions import ConfigurationError, UndefinedMetricError


def positive_mask(y_true: Iterable[Hashable], positive_label: Hashable) -> np.ndarray:
    """Return a boolean array marking the samples whose label is ``positive_label``.

    Labels match only when both value and type agree, so ``1``, ``1.0``, and ``True`` are
    different labels.

    Raises:
        ConfigurationError: If ``y_true`` is a string, is not one-dimensional, or contains
            arrays, tensors, or lists instead of scalar labels. Without this check such input
            would match nothing and be scored as all-negative.
    """
    if isinstance(y_true, str | bytes):
        raise ConfigurationError("y_true must be a sequence of labels, not a single string.")
    if hasattr(y_true, "ndim") and hasattr(y_true, "tolist"):  # numpy array or torch tensor
        if y_true.ndim != 1:
            raise ConfigurationError(
                f"y_true must be one-dimensional, got {y_true.ndim} dimensions."
            )
        if isinstance(y_true, np.ndarray) and y_true.dtype != object and len(y_true):
            # Vectorised path: every element shares one type, so compare types once.
            if type(y_true[0].item()) is not type(positive_label):
                return np.zeros(len(y_true), dtype=bool)
            return np.asarray(y_true == positive_label, dtype=bool)
        labels = y_true.tolist()
    else:
        labels = list(y_true)
    for label in labels:
        if isinstance(label, list | dict | set) or getattr(label, "ndim", 0) > 0:
            raise ConfigurationError(
                "y_true must contain scalar labels; "
                f"found an element of type {type(label).__name__}."
            )
    return np.fromiter((same_label(label, positive_label) for label in labels), dtype=bool)


def validate_scores(scores: Any, *, num_samples: int, name: str = "scores") -> np.ndarray:
    """Return ``scores`` as a finite 1-D float64 array of length ``num_samples``."""
    array = np.asarray(scores)
    if array.dtype == bool or not np.issubdtype(array.dtype, np.number):
        raise ConfigurationError(f"{name} must be numeric (not boolean), got dtype {array.dtype}.")
    if array.ndim != 1:
        raise ConfigurationError(f"{name} must be one-dimensional, got {array.ndim} dimensions.")
    if len(array) != num_samples:
        raise ConfigurationError(
            f"y_true and {name} must have the same length, got {num_samples} and {len(array)}."
        )
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ConfigurationError(f"{name} contain NaN or infinite values.")
    return array


def binary_inputs(
    y_true: Iterable[Hashable],
    scores: Any,
    positive_label: Hashable,
    *,
    greater_is_riskier: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a binary scoring problem; return ``(mask, scores)`` with higher = riskier."""
    mask = positive_mask(y_true, positive_label)
    if len(mask) == 0:
        raise UndefinedMetricError("Metrics are undefined for an empty sample.")
    values = validate_scores(scores, num_samples=len(mask))
    return mask, values if greater_is_riskier else -values


def require_positives(mask: np.ndarray, metric: str) -> None:
    """Raise unless at least one sample is positive."""
    if not mask.any():
        raise UndefinedMetricError(
            f"{metric} is undefined: no sample has the positive label "
            "(labels must match positive_label in both value and type)."
        )


def require_negatives(mask: np.ndarray, metric: str) -> None:
    """Raise unless at least one sample is negative."""
    if mask.all():
        raise UndefinedMetricError(f"{metric} is undefined: every sample has the positive label.")


def validate_fraction(value: Any, name: str, *, allow_zero: bool) -> float:
    """Return ``value`` as a float in ``(0, 1]``, or ``[0, 1]`` when ``allow_zero``."""
    valid = isinstance(value, int | float) and not isinstance(value, bool)
    if not valid or not (0 <= value <= 1) or (value == 0 and not allow_zero):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ConfigurationError(f"{name} must be a number in {interval}, got {value!r}.")
    return float(value)


def review_count(num_samples: int, review_rate: float) -> int:
    """Number of items reviewed at a review rate: ``ceil(num_samples * review_rate)``.

    This is the single rounding policy for every review-rate metric. Rounding up means a
    positive rate always reviews at least one item, and the count never exceeds
    ``num_samples``. The product is rounded to 9 decimals first, so that ``100 * 0.07`` counts
    as 7 rather than 7.000000000000001; consequently a genuine fractional part below 5e-10 is
    also ignored, which only matters beyond roughly a billion samples.
    """
    rate = validate_fraction(review_rate, "review_rate", allow_zero=False)
    return max(1, min(num_samples, math.ceil(round(num_samples * rate, 9))))


def expected_top_k_sum(values: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Expected sum of ``values`` over the ``k`` highest-scoring items.

    Items with equal scores are interchangeable, so when a group of tied items straddles the
    cut-off the result is the expectation under uniformly random tie-breaking: a group of ``g``
    tied items of which ``m`` fit contributes ``m / g`` of its total. The result therefore does
    not depend on input order.
    """
    order = np.argsort(-scores, kind="stable")
    _, starts, counts = np.unique(-scores[order], return_index=True, return_counts=True)
    group_sums = np.add.reduceat(values[order].astype(np.float64), starts)
    taken = np.clip(k - starts, 0, counts)
    return float((group_sums * taken / counts).sum())


def threshold_curve(mask: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, ...]:
    """Counts of flagged items at every distinct score used as a threshold.

    An item is flagged when ``score >= threshold``, so tied items are always flagged together.

    Returns:
        ``(thresholds, true_positives, false_positives)`` with thresholds in descending order.
    """
    order = np.argsort(-scores, kind="stable")
    thresholds, starts = np.unique(-scores[order], return_index=True)
    ends = np.append(starts[1:], len(scores)) - 1
    true_positives = np.cumsum(mask[order])[ends]
    false_positives = np.cumsum(~mask[order])[ends]
    return -thresholds, true_positives, false_positives
