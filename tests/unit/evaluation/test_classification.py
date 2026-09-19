import numpy as np
import pytest

from embedfit.evaluation import classification_metrics, confusion_matrix, pr_auc, roc_auc
from embedfit.exceptions import ConfigurationError, UndefinedMetricError

LABELS = ["a", "b", "c"]
TRUE = ["a", "a", "b", "b", "c", "c"]
PRED = ["a", "b", "b", "b", "c", "a"]

Y = ["risk", "ok", "risk", "ok", "ok", "risk", "ok", "ok", "ok", "ok"]
S = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]


def test_confusion_matrix_rows_are_true_classes_in_label_order():
    expected = [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert confusion_matrix(TRUE, PRED, labels=LABELS).tolist() == expected
    reordered = confusion_matrix(TRUE, PRED, labels=["c", "b", "a"])
    assert reordered.tolist() == [[1, 0, 1], [0, 2, 0], [0, 1, 1]]


def test_classification_metrics_hand_computed():
    # Per class (precision, recall, F1): a = (1/2, 1/2, 1/2), b = (2/3, 1, 4/5), c = (1, 1/2, 2/3).
    result = classification_metrics(TRUE, PRED, labels=LABELS, positive_label="b")
    assert result == pytest.approx(
        {
            "accuracy": 4 / 6,
            "balanced_accuracy": 2 / 3,
            "macro_precision": (1 / 2 + 2 / 3 + 1) / 3,
            "macro_recall": 2 / 3,
            "macro_f1": (1 / 2 + 4 / 5 + 2 / 3) / 3,
            "weighted_f1": (1 / 2 + 4 / 5 + 2 / 3) / 3,
            "precision": 2 / 3,
            "recall": 1.0,
            "f1": 4 / 5,
        }
    )
    assert "precision" not in classification_metrics(TRUE, PRED, labels=LABELS)


def test_accuracy_is_misleading_on_rare_events_but_balanced_accuracy_is_not():
    truth = ["ok"] * 99 + ["risk"]
    always_ok = ["ok"] * 100
    result = classification_metrics(truth, always_ok, labels=["ok", "risk"], positive_label="risk")
    assert result["accuracy"] == 0.99
    assert result["balanced_accuracy"] == 0.5
    assert (result["recall"], result["precision"], result["f1"]) == (0.0, 0.0, 0.0)


def test_absent_class_counts_as_zero_in_macro_but_not_in_balanced_accuracy():
    result = classification_metrics(["a", "b"], ["a", "b"], labels=LABELS)
    assert result["accuracy"] == 1.0 and result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["weighted_f1"] == 1.0


def test_non_string_and_mixed_labels():
    result = classification_metrics([0, 1, "x"], [0, "x", "x"], labels=[0, 1, "x"])
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_pr_auc_is_average_precision():
    # Positives sit at ranks 1, 3, and 6: (1/1 + 2/3 + 3/6) / 3.
    assert pr_auc(Y, S, positive_label="risk") == pytest.approx((1 + 2 / 3 + 1 / 2) / 3)


def test_roc_auc_counts_correctly_ordered_pairs():
    # The positives outrank 7, 6, and 4 of the 7 negatives: 17 of 21 pairs.
    assert roc_auc(Y, S, positive_label="risk") == pytest.approx(17 / 21)
    assert roc_auc(Y, [-s for s in S], positive_label="risk", greater_is_riskier=False) == (
        pytest.approx(17 / 21)
    )


def test_tied_scores():
    assert roc_auc([1, 0], [0.5, 0.5], positive_label=1) == 0.5  # A tie counts as half.
    assert pr_auc([1, 0, 0, 0], [0.5] * 4, positive_label=1) == 0.25  # The positive rate.


def test_random_ranking_on_imbalanced_data_scores_near_the_positive_rate():
    rng = np.random.default_rng(0)
    labels = (rng.random(50_000) < 0.01).astype(int)
    scores = rng.random(50_000)
    assert pr_auc(labels, scores, positive_label=1) == pytest.approx(0.01, abs=0.005)
    assert roc_auc(labels, scores, positive_label=1) == pytest.approx(0.5, abs=0.03)


def test_undefined_cases_raise():
    with pytest.raises(UndefinedMetricError, match="PR-AUC is undefined"):
        pr_auc(["ok"] * 3, [0.1, 0.2, 0.3], positive_label="risk")
    with pytest.raises(UndefinedMetricError, match="no sample has the positive label"):
        roc_auc(["ok"] * 3, [0.1, 0.2, 0.3], positive_label="risk")
    with pytest.raises(UndefinedMetricError, match="every sample has the positive label"):
        roc_auc(["risk"] * 3, [0.1, 0.2, 0.3], positive_label="risk")
    with pytest.raises(UndefinedMetricError, match="empty"):
        classification_metrics([], [], labels=LABELS)


def test_invalid_inputs():
    with pytest.raises(ConfigurationError, match="same length"):
        classification_metrics(TRUE, PRED[:-1], labels=LABELS)
    with pytest.raises(ConfigurationError, match="not one of the model's labels"):
        classification_metrics(TRUE, PRED, labels=["a", "b"])
    with pytest.raises(ConfigurationError, match="not one of"):
        classification_metrics(TRUE, PRED, labels=LABELS, positive_label="z")
