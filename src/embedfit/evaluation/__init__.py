"""Classification, risk, and calibration metrics, threshold selection, and a metric registry.

Every function takes arrays of labels and scores; none of them calls a model.
"""

from embedfit.evaluation.calibration import (
    CalibrationBins,
    brier_score,
    calibration_bins,
    expected_calibration_error,
)
from embedfit.evaluation.classification import (
    classification_metrics,
    confusion_matrix,
    pr_auc,
    roc_auc,
)
from embedfit.evaluation.registry import available_metrics, get_metric, register_metric
from embedfit.evaluation.risk import (
    dollar_recall,
    dollar_recall_at_review_rate,
    precision_at_k,
    precision_at_review_rate,
    recall_at_fpr,
    recall_at_k,
    recall_at_review_rate,
)
from embedfit.evaluation.thresholds import ThresholdResult, select_threshold

__all__ = [
    "CalibrationBins",
    "ThresholdResult",
    "available_metrics",
    "brier_score",
    "calibration_bins",
    "classification_metrics",
    "confusion_matrix",
    "dollar_recall",
    "dollar_recall_at_review_rate",
    "expected_calibration_error",
    "get_metric",
    "pr_auc",
    "precision_at_k",
    "precision_at_review_rate",
    "recall_at_fpr",
    "recall_at_k",
    "recall_at_review_rate",
    "register_metric",
    "roc_auc",
    "select_threshold",
]
