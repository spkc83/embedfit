"""Risk-oriented ranking metrics for rare-event problems.

Conventions shared by every function here:

- ``positive_label`` is required; the positive class is never inferred.
- Scores rank items from most to least risky (``greater_is_riskier=False`` flips them).
- A threshold flags every item with ``score >= threshold``, so tied items are flagged together.
- "Top K" metrics treat tied items as interchangeable and report the expected value under
  random tie-breaking, so results never depend on input order.
- Review-rate metrics review ``ceil(N * review_rate)`` items.
"""

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np

from embedfit.evaluation._inputs import (
    binary_inputs,
    expected_top_k_sum,
    positive_mask,
    require_negatives,
    require_positives,
    review_count,
    threshold_curve,
    validate_fraction,
    validate_scores,
)
from embedfit.exceptions import ConfigurationError, UndefinedMetricError


def _validate_k(k: Any, num_samples: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or not 1 <= k <= num_samples:
        raise ConfigurationError(f"k must be an int in [1, {num_samples}], got {k!r}.")
    return int(k)


def recall_at_fpr(
    y_true: Iterable[Hashable],
    scores: Any,
    *,
    positive_label: Hashable,
    target_fpr: float,
    greater_is_riskier: bool = True,
) -> float:
    """Recall at the most permissive threshold whose false-positive rate is within a target.

    Thresholds are taken at observed score values without interpolation: among thresholds with
    ``FPR <= target_fpr`` the lowest is used. If even the highest-scoring tied group exceeds
    the target, nothing is flagged and recall is 0.

    Args:
        y_true: True labels.
        scores: One score per sample.
        positive_label: The label scores refer to.
        target_fpr: Maximum false-positive rate, in [0, 1].
        greater_is_riskier: Whether higher scores mean "more likely positive".

    Raises:
        ConfigurationError: If inputs are malformed.
        UndefinedMetricError: If there are no positives or no negatives.
    """
    mask, values = binary_inputs(
        y_true, scores, positive_label, greater_is_riskier=greater_is_riskier
    )
    target = validate_fraction(target_fpr, "target_fpr", allow_zero=True)
    require_positives(mask, "Recall at FPR")
    require_negatives(mask, "Recall at FPR")
    _, true_positives, false_positives = threshold_curve(mask, values)
    allowed = false_positives / (~mask).sum() <= target
    return float(true_positives[allowed].max() / mask.sum()) if allowed.any() else 0.0


def precision_at_k(
    y_true: Iterable[Hashable],
    scores: Any,
    *,
    positive_label: Hashable,
    k: int,
    greater_is_riskier: bool = True,
) -> float:
    """Fraction of the ``k`` highest-scoring items that are positive.

    Defined (as 0) when there are no positives. ``k`` is an int in ``[1, N]``; see
    ``recall_at_fpr`` for the other arguments and the module notes for ties.

    Raises:
        ConfigurationError: If inputs are malformed or ``k`` is outside ``[1, N]``.
    """
    mask, values = binary_inputs(
        y_true, scores, positive_label, greater_is_riskier=greater_is_riskier
    )
    count = _validate_k(k, len(mask))
    return expected_top_k_sum(mask, values, count) / count


def recall_at_k(
    y_true: Iterable[Hashable],
    scores: Any,
    *,
    positive_label: Hashable,
    k: int,
    greater_is_riskier: bool = True,
) -> float:
    """Fraction of all positives found among the ``k`` highest-scoring items.

    ``k`` is an int in ``[1, N]``; see ``recall_at_fpr`` for the other arguments and the
    module notes for ties.

    Raises:
        ConfigurationError: If inputs are malformed or ``k`` is outside ``[1, N]``.
        UndefinedMetricError: If there are no positives.
    """
    mask, values = binary_inputs(
        y_true, scores, positive_label, greater_is_riskier=greater_is_riskier
    )
    count = _validate_k(k, len(mask))
    require_positives(mask, "Recall at K")
    return expected_top_k_sum(mask, values, count) / float(mask.sum())


def precision_at_review_rate(
    y_true: Iterable[Hashable],
    scores: Any,
    *,
    positive_label: Hashable,
    review_rate: float,
    greater_is_riskier: bool = True,
) -> float:
    """Precision among the ``ceil(N * review_rate)`` highest-scoring items.

    Args:
        review_rate: Fraction of items that can be reviewed, in (0, 1].

    Raises:
        ConfigurationError: If inputs are malformed or ``review_rate`` is outside (0, 1].
    """
    mask, _ = binary_inputs(y_true, scores, positive_label)
    return precision_at_k(
        mask,
        scores,
        positive_label=True,
        k=review_count(len(mask), review_rate),
        greater_is_riskier=greater_is_riskier,
    )


def recall_at_review_rate(
    y_true: Iterable[Hashable],
    scores: Any,
    *,
    positive_label: Hashable,
    review_rate: float,
    greater_is_riskier: bool = True,
) -> float:
    """Recall among the ``ceil(N * review_rate)`` highest-scoring items.

    Args:
        review_rate: Fraction of items that can be reviewed, in (0, 1].

    Raises:
        ConfigurationError: If inputs are malformed or ``review_rate`` is outside (0, 1].
        UndefinedMetricError: If there are no positives.
    """
    mask, _ = binary_inputs(y_true, scores, positive_label)
    return recall_at_k(
        mask,
        scores,
        positive_label=True,
        k=review_count(len(mask), review_rate),
        greater_is_riskier=greater_is_riskier,
    )


def _positive_amounts(mask: np.ndarray, amounts: Any) -> np.ndarray:
    values = validate_scores(amounts, num_samples=len(mask), name="amounts")
    if (values < 0).any():
        raise ConfigurationError("amounts must be non-negative.")
    positive_amounts = np.where(mask, values, 0.0)
    if positive_amounts.sum() == 0:
        raise UndefinedMetricError(
            "Dollar recall is undefined: the positive samples have zero total amount."
        )
    return positive_amounts


def dollar_recall(
    y_true: Iterable[Hashable],
    flagged: Any,
    amounts: Any,
    *,
    positive_label: Hashable,
) -> float:
    """Share of the positive class's total amount that was flagged.

    Only positive samples contribute to the numerator and denominator; amounts of negative
    samples are ignored.

    Args:
        y_true: True labels.
        flagged: Boolean array, true where the item was flagged for review.
        amounts: Non-negative, finite amount per sample.
        positive_label: The label that counts as a positive.

    Raises:
        ConfigurationError: If lengths differ, ``flagged`` is not boolean, or an amount is
            negative, NaN, or infinite.
        UndefinedMetricError: If the sample is empty or the positive samples' total amount
            is zero.
    """
    mask = positive_mask(y_true, positive_label)
    if len(mask) == 0:
        raise UndefinedMetricError("Metrics are undefined for an empty sample.")
    decisions = np.asarray(flagged)
    if decisions.dtype != bool or decisions.shape != mask.shape:
        raise ConfigurationError("flagged must be a boolean array with one entry per sample.")
    positive_amounts = _positive_amounts(mask, amounts)
    return float(positive_amounts[decisions].sum() / positive_amounts.sum())


def dollar_recall_at_review_rate(
    y_true: Iterable[Hashable],
    scores: Any,
    amounts: Any,
    *,
    positive_label: Hashable,
    review_rate: float,
    greater_is_riskier: bool = True,
) -> float:
    """Dollar recall when the ``ceil(N * review_rate)`` highest-scoring items are reviewed.

    ``review_rate`` is the fraction of items that can be reviewed, in (0, 1]. See
    ``dollar_recall`` for amount handling, ``recall_at_fpr`` for the other arguments, and the
    module notes for ties.

    Raises:
        ConfigurationError: If inputs are malformed.
        UndefinedMetricError: If the positive samples' total amount is zero.
    """
    mask, values = binary_inputs(
        y_true, scores, positive_label, greater_is_riskier=greater_is_riskier
    )
    positive_amounts = _positive_amounts(mask, amounts)
    captured = expected_top_k_sum(positive_amounts, values, review_count(len(mask), review_rate))
    return captured / float(positive_amounts.sum())
