"""Decision-threshold selection. Use validation data only, never test data."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from embedfit.evaluation._inputs import (
    binary_inputs,
    require_negatives,
    require_positives,
    review_count,
    threshold_curve,
    validate_fraction,
)
from embedfit.exceptions import ConfigurationError, UndefinedMetricError

OBJECTIVES = ("max_f1", "target_precision", "target_recall", "target_fpr", "review_rate")


@dataclass(frozen=True)
class ThresholdResult:
    """A selected threshold and what it achieves on the data it was selected on.

    Items with ``score >= threshold`` are flagged. ``threshold`` is ``inf`` when the objective
    can only be met by flagging nothing.

    Attributes:
        threshold: The selected score threshold.
        objective: The objective that was optimised.
        precision: Precision at the threshold (0 when nothing is flagged).
        recall: Recall at the threshold.
        fpr: False-positive rate at the threshold (NaN when there are no negatives).
        review_rate: Fraction of all items flagged.
    """

    threshold: float
    objective: str
    precision: float
    recall: float
    fpr: float
    review_rate: float


def select_threshold(
    y_true: Iterable[Hashable],
    scores: Any,
    *,
    positive_label: Hashable,
    objective: str,
    target: float | None = None,
) -> ThresholdResult:
    """Choose a decision threshold on validation data.

    Selecting a threshold on test data leaks test labels into the decision rule and inflates
    every reported metric; pass validation data only.

    Candidate thresholds are the observed score values. Objectives:

    - ``"max_f1"``: highest F1; ties go to the higher threshold.
    - ``"target_precision"``: highest recall subject to ``precision >= target``; ties go to
      the higher threshold.
    - ``"target_recall"``: highest threshold with ``recall >= target`` (fewest items flagged).
    - ``"target_fpr"``: lowest threshold with ``FPR <= target``.
    - ``"review_rate"``: lowest threshold flagging at most ``ceil(N * target)`` items. Because
      tied items are flagged together, fewer items than the capacity may be flagged.

    Args:
        y_true: True labels.
        scores: One score per sample. Higher must mean more likely positive; negate
            distance-like scores first, and read the returned threshold on the negated scale.
        positive_label: The label scores refer to. Required; never inferred.
        objective: One of the objectives above.
        target: The target value, required for every objective except ``"max_f1"``.

    Raises:
        ConfigurationError: If the objective is unknown, ``target`` is missing, superfluous, or
            out of range, or inputs are malformed.
        UndefinedMetricError: If there are no positives, ``"target_fpr"`` has no negatives, or
            no threshold reaches ``"target_precision"``.
    """
    if objective not in OBJECTIVES:
        raise ConfigurationError(f"objective must be one of {list(OBJECTIVES)}, got {objective!r}.")
    if (target is None) != (objective == "max_f1"):
        raise ConfigurationError(
            "target must be omitted for 'max_f1' and given for every other objective."
        )
    mask, values = binary_inputs(y_true, scores, positive_label)
    require_positives(mask, "Threshold selection")
    if objective == "target_fpr":
        require_negatives(mask, "Threshold selection for a target FPR")

    thresholds, true_positives, false_positives = threshold_curve(mask, values)
    flagged = true_positives + false_positives
    num_positives, num_negatives = int(mask.sum()), int((~mask).sum())
    precision = true_positives / flagged
    recall = true_positives / num_positives
    with np.errstate(invalid="ignore", divide="ignore"):
        fpr = false_positives / num_negatives if num_negatives else np.full(len(flagged), np.nan)

    # Thresholds are in descending order, so np.argmax (first maximum) prefers higher ones.
    if objective == "max_f1":
        chosen = int(np.argmax(2 * true_positives / (flagged + num_positives)))
    else:
        assert target is not None
        goal = validate_fraction(target, "target", allow_zero=objective == "target_fpr")
        if objective == "review_rate":
            eligible = flagged <= review_count(len(mask), goal)
        else:
            eligible = {
                "target_precision": precision >= goal,
                "target_recall": recall >= goal,
                "target_fpr": fpr <= goal,
            }[objective]
        if not eligible.any():
            if objective == "target_precision":
                raise UndefinedMetricError(
                    f"No threshold reaches precision >= {target}; "
                    f"the best available is {precision.max():.4f}."
                )
            return ThresholdResult(
                float("inf"), objective, 0.0, 0.0, 0.0 if num_negatives else float("nan"), 0.0
            )
        positions = np.flatnonzero(eligible)
        if objective == "target_recall":
            chosen = int(positions[0])
        elif objective == "target_precision":
            chosen = int(positions[np.argmax(recall[positions])])
        else:
            chosen = int(positions[-1])

    return ThresholdResult(
        threshold=float(thresholds[chosen]),
        objective=objective,
        precision=float(precision[chosen]),
        recall=float(recall[chosen]),
        fpr=float(fpr[chosen]),
        review_rate=float(flagged[chosen] / len(mask)),
    )
