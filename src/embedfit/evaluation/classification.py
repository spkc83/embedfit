"""Standard classification metrics over labels and scores."""

from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import numpy as np
from sklearn import metrics as sk

from embedfit.data.labels import LabelMapping
from embedfit.evaluation._inputs import binary_inputs, require_negatives, require_positives
from embedfit.exceptions import ConfigurationError, UndefinedMetricError


def _encode_pair(
    y_true: Iterable[Hashable], y_pred: Iterable[Hashable], labels: Sequence[Hashable] | np.ndarray
) -> tuple[np.ndarray, np.ndarray, LabelMapping]:
    mapping = labels if isinstance(labels, LabelMapping) else LabelMapping(labels)
    true_indices, predicted_indices = mapping.encode(y_true), mapping.encode(y_pred)
    if len(true_indices) != len(predicted_indices):
        raise ConfigurationError(
            f"y_true and y_pred must have the same length, got {len(true_indices)} and "
            f"{len(predicted_indices)}."
        )
    if len(true_indices) == 0:
        raise UndefinedMetricError("Metrics are undefined for an empty sample.")
    return true_indices, predicted_indices, mapping


def confusion_matrix(
    y_true: Iterable[Hashable], y_pred: Iterable[Hashable], *, labels: Sequence[Hashable]
) -> np.ndarray:
    """Count predictions per true class.

    Args:
        y_true: True labels.
        y_pred: Predicted labels, same length.
        labels: All class labels; they fix the row and column order.

    Returns:
        Integer array shaped ``[num_classes, num_classes]``; entry ``[i, j]`` counts samples
        with true label ``labels[i]`` predicted as ``labels[j]``.

    Raises:
        ConfigurationError: If lengths differ or a label is not in ``labels``.
        UndefinedMetricError: If the sample is empty.
    """
    true_indices, predicted_indices, mapping = _encode_pair(y_true, y_pred, labels)
    matrix: np.ndarray = sk.confusion_matrix(
        true_indices, predicted_indices, labels=np.arange(len(mapping))
    )
    return matrix


def classification_metrics(
    y_true: Iterable[Hashable],
    y_pred: Iterable[Hashable],
    *,
    labels: Sequence[Hashable],
    positive_label: Hashable | None = None,
) -> dict[str, float]:
    """Compute threshold-dependent classification metrics from hard predictions.

    Args:
        y_true: True labels.
        y_pred: Predicted labels, same length.
        labels: All class labels. Classes absent from the data still count in macro averages
            (with a score of 0), so results are comparable across samples.
        positive_label: When given, adds ``precision``, ``recall``, and ``f1`` for that class.

    Returns:
        ``accuracy``, ``balanced_accuracy`` (mean recall over classes present in ``y_true``),
        ``macro_precision``, ``macro_recall``, ``macro_f1``, ``weighted_f1``, and the
        positive-class metrics when requested. A ratio with a zero denominator is 0.

    Raises:
        ConfigurationError: If lengths differ, a label is not in ``labels``, or
            ``positive_label`` is not in ``labels``.
        UndefinedMetricError: If the sample is empty.
    """
    true_indices, predicted_indices, mapping = _encode_pair(y_true, y_pred, labels)
    every_class = np.arange(len(mapping))
    precision, recall, f1, support = sk.precision_recall_fscore_support(
        true_indices, predicted_indices, labels=every_class, zero_division=0
    )
    present = support > 0
    results = {
        "accuracy": float((true_indices == predicted_indices).mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
    }
    if positive_label is not None:
        index = mapping.index_of(positive_label)
        results.update(
            precision=float(precision[index]), recall=float(recall[index]), f1=float(f1[index])
        )
    return results


def roc_auc(
    y_true: Iterable[Hashable],
    scores: Any,
    *,
    positive_label: Hashable,
    greater_is_riskier: bool = True,
) -> float:
    """Area under the ROC curve: the probability that a positive outranks a negative.

    Tied scores count as half. See ``pr_auc`` for the arguments.

    Raises:
        UndefinedMetricError: If there are no positives or no negatives.
    """
    mask, values = binary_inputs(
        y_true, scores, positive_label, greater_is_riskier=greater_is_riskier
    )
    require_positives(mask, "ROC-AUC")
    require_negatives(mask, "ROC-AUC")
    return float(sk.roc_auc_score(mask, values))


def pr_auc(
    y_true: Iterable[Hashable],
    scores: Any,
    *,
    positive_label: Hashable,
    greater_is_riskier: bool = True,
) -> float:
    """Area under the precision-recall curve, computed as average precision.

    Average precision is the step-wise sum ``Σ (R_n − R_{n−1}) · P_n`` over distinct score
    thresholds; it does not interpolate between points, which would be optimistic. For a
    random ranking it equals the positive rate, so it is informative on imbalanced data.

    Args:
        y_true: True labels.
        scores: One score per sample.
        positive_label: The label scores refer to. Required; never inferred.
        greater_is_riskier: Whether higher scores mean "more likely positive". Pass ``False``
            for scores with the opposite direction.

    Raises:
        ConfigurationError: If inputs are malformed.
        UndefinedMetricError: If there are no positives.
    """
    mask, values = binary_inputs(
        y_true, scores, positive_label, greater_is_riskier=greater_is_riskier
    )
    require_positives(mask, "PR-AUC")
    return float(sk.average_precision_score(mask, values))
