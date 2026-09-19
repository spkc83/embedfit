"""Threshold selection on the reference ranking used in test_risk.py.

threshold  .90  .80  .70  .60  .50  .40
TP / FP    1/0  1/1  2/1  2/2  2/3  3/3      (3 positives, 7 negatives)
F1         .50  .40  .67  .57  .50  .67
"""

import math

import pytest

from embedfit.evaluation import select_threshold
from embedfit.exceptions import ConfigurationError, UndefinedMetricError

Y = ["risk", "ok", "risk", "ok", "ok", "risk", "ok", "ok", "ok", "ok"]
S = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]
POS = {"positive_label": "risk"}


def test_max_f1_prefers_the_higher_threshold_on_ties():
    result = select_threshold(Y, S, objective="max_f1", **POS)
    assert result.threshold == 0.70  # F1 = 2/3 at both .70 and .40.
    assert (result.precision, result.recall) == pytest.approx((2 / 3, 2 / 3))
    assert result.fpr == pytest.approx(1 / 7) and result.review_rate == pytest.approx(0.3)
    assert result.objective == "max_f1"


@pytest.mark.parametrize(
    "objective, target, threshold",
    [
        ("target_precision", 0.6, 0.70),  # Precision >= .6 at .90 and .70; .70 has more recall.
        ("target_precision", 1.0, 0.90),
        ("target_recall", 0.5, 0.70),  # The highest threshold reaching 2 of 3 positives.
        ("target_recall", 1.0, 0.40),
        ("target_fpr", 0.0, 0.90),
        ("target_fpr", 0.15, 0.70),  # One false positive allowed (1/7 = .143).
        ("review_rate", 0.2, 0.80),  # ceil(10 * .2) = 2 items.
        ("review_rate", 1.0, 0.05),
    ],
)
def test_objectives(objective, target, threshold):
    assert select_threshold(Y, S, objective=objective, target=target, **POS).threshold == threshold


def test_unreachable_precision_raises_with_the_best_available():
    with pytest.raises(UndefinedMetricError, match="best available is 0.5000"):
        select_threshold(
            [0, 1], [0.9, 0.1], positive_label=1, objective="target_precision", target=0.9
        )


def test_flagging_nothing_is_reported_as_an_infinite_threshold():
    # Both items tie, so any threshold flags the negative too: FPR 0 is only met by no flags.
    result = select_threshold(
        [1, 0], [0.5, 0.5], positive_label=1, objective="target_fpr", target=0.0
    )
    assert math.isinf(result.threshold)
    assert (result.precision, result.recall, result.fpr, result.review_rate) == (0, 0, 0, 0)
    # Three tied items exceed a capacity of one.
    tied = select_threshold(
        [1, 0, 0], [0.5] * 3, positive_label=1, objective="review_rate", target=0.3
    )
    assert math.isinf(tied.threshold)


def test_ties_are_flagged_together():
    result = select_threshold(
        [1, 0, 1, 0], [0.9, 0.9, 0.2, 0.1], positive_label=1, objective="target_recall", target=0.5
    )
    assert result.threshold == 0.9 and result.review_rate == 0.5 and result.precision == 0.5


def test_no_negatives():
    result = select_threshold([1, 1], [0.9, 0.1], positive_label=1, objective="max_f1")
    assert result.threshold == 0.1 and math.isnan(result.fpr)
    with pytest.raises(UndefinedMetricError, match="every sample has the positive label"):
        select_threshold([1, 1], [0.9, 0.1], positive_label=1, objective="target_fpr", target=0.1)


def test_no_positives():
    with pytest.raises(UndefinedMetricError, match="no sample has the positive label"):
        select_threshold([0, 0], [0.9, 0.1], positive_label=1, objective="max_f1")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"objective": "max_accuracy"}, "objective must be one of"),
        ({"objective": "max_f1", "target": 0.5}, "target must be omitted"),
        ({"objective": "target_fpr"}, "target must be omitted for 'max_f1' and given"),
        ({"objective": "target_recall", "target": 0.0}, r"\(0, 1\]"),
        ({"objective": "target_fpr", "target": 1.5}, r"\[0, 1\]"),
        ({"objective": "review_rate", "target": 0}, r"target must be a number in \(0, 1\]"),
        ({"objective": "review_rate", "target": 1.5}, "target must be"),
    ],
)
def test_invalid_configuration(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        select_threshold(Y, S, **kwargs, **POS)
