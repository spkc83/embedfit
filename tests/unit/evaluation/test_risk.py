"""Risk metrics against values worked out by hand.

Reference ranking (P = positive, n = negative), 3 positives and 7 negatives:

    score   .90  .80  .70  .60  .50  .40  .30  .20  .10  .05
    label    P    n    P    n    n    P    n    n    n    n
    amount  100   9   50    9    9   850   9    9    9    9
"""

import numpy as np
import pytest

from embedfit.evaluation import (
    dollar_recall,
    dollar_recall_at_review_rate,
    precision_at_k,
    precision_at_review_rate,
    recall_at_fpr,
    recall_at_k,
    recall_at_review_rate,
)
from embedfit.evaluation._inputs import review_count
from embedfit.exceptions import ConfigurationError, UndefinedMetricError

Y = ["risk", "ok", "risk", "ok", "ok", "risk", "ok", "ok", "ok", "ok"]
S = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]
A = [100, 9, 50, 9, 9, 850, 9, 9, 9, 9]
POS = {"positive_label": "risk"}


@pytest.mark.parametrize(
    "target_fpr, expected",
    [(0.0, 1 / 3), (0.1, 1 / 3), (1 / 7, 2 / 3), (0.2, 2 / 3), (0.5, 1.0), (1.0, 1.0)],
)
def test_recall_at_fpr(target_fpr, expected):
    # FPR <= 1/7 admits one false positive (.80), so the threshold drops to .70: TP = 2 of 3.
    assert recall_at_fpr(Y, S, target_fpr=target_fpr, **POS) == pytest.approx(expected)


def test_recall_at_fpr_flags_tied_items_together():
    # The only threshold flags a positive and a negative at once: FPR = 1 > 0, so nothing
    # can be flagged within the target.
    assert recall_at_fpr([1, 0], [0.5, 0.5], positive_label=1, target_fpr=0.0) == 0.0
    assert recall_at_fpr([1, 0], [0.5, 0.5], positive_label=1, target_fpr=1.0) == 1.0


@pytest.mark.parametrize(
    "k, precision, recall", [(1, 1.0, 1 / 3), (3, 2 / 3, 2 / 3), (6, 0.5, 1.0), (10, 0.3, 1.0)]
)
def test_precision_and_recall_at_k(k, precision, recall):
    assert precision_at_k(Y, S, k=k, **POS) == pytest.approx(precision)
    assert recall_at_k(Y, S, k=k, **POS) == pytest.approx(recall)


def test_ties_at_the_cutoff_use_the_expected_value_and_ignore_input_order():
    labels, scores = np.array([1, 0, 0, 1]), np.array([0.5, 0.5, 0.5, 0.1])
    # Three items tie for the top spot and one of them is positive: any one of them is
    # positive with probability 1/3; any two contain 2/3 of a positive in expectation.
    for order in ([0, 1, 2, 3], [2, 1, 0, 3], [3, 2, 0, 1]):
        y, s = labels[order], scores[order]
        assert precision_at_k(y, s, positive_label=1, k=1) == pytest.approx(1 / 3)
        assert precision_at_k(y, s, positive_label=1, k=2) == pytest.approx(1 / 3)
        assert recall_at_k(y, s, positive_label=1, k=2) == pytest.approx(1 / 3)
        assert recall_at_k(y, s, positive_label=1, k=3) == pytest.approx(1 / 2)


def test_review_rate_metrics_review_the_ceiling_of_n_times_rate():
    # ceil(10 * 0.25) = 3 items reviewed.
    assert precision_at_review_rate(Y, S, review_rate=0.25, **POS) == pytest.approx(2 / 3)
    assert recall_at_review_rate(Y, S, review_rate=0.25, **POS) == pytest.approx(2 / 3)
    assert recall_at_review_rate(Y, S, review_rate=1.0, **POS) == 1.0
    assert precision_at_review_rate(Y, S, review_rate=1e-9, **POS) == 1.0  # At least one item.


@pytest.mark.parametrize(
    "n, rate, expected",
    [(100, 0.07, 7), (100, 0.01, 1), (1000, 0.001, 1), (10, 0.25, 3), (3, 1.0, 3), (7, 1e-12, 1)],
)
def test_review_count_rounding_policy(n, rate, expected):
    assert review_count(n, rate) == expected  # 100 * 0.07 is 7.000000000000001 in floats.


@pytest.mark.parametrize("rate", [0, -0.1, 1.01, "0.1", None, True])
def test_review_rate_boundaries(rate):
    with pytest.raises(ConfigurationError, match="review_rate"):
        recall_at_review_rate(Y, S, review_rate=rate, **POS)


def test_dollar_recall_counts_only_positive_amounts():
    flagged = np.array(S) >= 0.70  # Flags .90, .80, .70: positives worth 100 and 50 of 1000.
    assert dollar_recall(Y, flagged, A, **POS) == pytest.approx(0.15)
    inflated = [a if y == "risk" else a * 1e6 for a, y in zip(A, Y, strict=True)]
    assert dollar_recall(Y, flagged, inflated, **POS) == pytest.approx(0.15)


def test_dollar_recall_at_review_rate():
    assert dollar_recall_at_review_rate(Y, S, A, review_rate=0.3, **POS) == pytest.approx(0.15)
    assert dollar_recall_at_review_rate(Y, S, A, review_rate=0.6, **POS) == pytest.approx(1.0)
    # Count recall at the same capacity is 2/3: the missed positive holds most of the value.
    assert recall_at_review_rate(Y, S, review_rate=0.3, **POS) == pytest.approx(2 / 3)


def test_dollar_recall_splits_tied_amounts_in_expectation():
    # Two tied positives worth 10 and 30, one slot: expected capture (10 + 30) / 2 of 40.
    result = dollar_recall_at_review_rate(
        [1, 1, 0, 0], [0.9, 0.9, 0.1, 0.1], [10, 30, 5, 5], positive_label=1, review_rate=0.25
    )
    assert result == pytest.approx(0.5)


@pytest.mark.parametrize(
    "amounts, error, match",
    [
        ([1, -1, 1, 1, 1, 1, 1, 1, 1, 1], ConfigurationError, "non-negative"),
        ([1, np.nan, 1, 1, 1, 1, 1, 1, 1, 1], ConfigurationError, "NaN or infinite"),
        ([1, np.inf, 1, 1, 1, 1, 1, 1, 1, 1], ConfigurationError, "NaN or infinite"),
        ([1, 2, 3], ConfigurationError, "same length"),
        (["a"] * 10, ConfigurationError, "numeric"),
        ([0, 5, 0, 5, 5, 0, 5, 5, 5, 5], UndefinedMetricError, "zero total amount"),
    ],
)
def test_invalid_amounts(amounts, error, match):
    with pytest.raises(error, match=match):
        dollar_recall_at_review_rate(Y, S, amounts, review_rate=0.3, **POS)


def test_dollar_recall_requires_boolean_flags():
    with pytest.raises(ConfigurationError, match="boolean"):
        dollar_recall(Y, S, A, **POS)


def test_no_positive_examples():
    negatives = ["ok"] * 10
    assert precision_at_k(negatives, S, k=3, **POS) == 0.0  # Precision stays defined.
    for metric, kwargs in (
        (recall_at_k, {"k": 3}),
        (recall_at_review_rate, {"review_rate": 0.3}),
        (recall_at_fpr, {"target_fpr": 0.1}),
    ):
        with pytest.raises(UndefinedMetricError, match="no sample has the positive label"):
            metric(negatives, S, **kwargs, **POS)
    with pytest.raises(UndefinedMetricError, match="every sample has the positive label"):
        recall_at_fpr(["risk"] * 10, S, target_fpr=0.1, **POS)


def test_positive_label_must_match_in_type_so_it_is_never_guessed():
    with pytest.raises(UndefinedMetricError, match="value and type"):
        recall_at_k([1.0, 0.0], [0.9, 0.1], positive_label=1, k=1)
    assert recall_at_k(np.array([True, False]), [0.9, 0.1], positive_label=True, k=1) == 1.0


def test_highly_imbalanced_data():
    rng = np.random.default_rng(0)
    labels = np.zeros(100_000, dtype=int)
    labels[:20] = 1
    scores = rng.random(100_000)
    scores[:10] += 1.0  # Half of the positives rank above everything else.
    assert recall_at_review_rate(labels, scores, positive_label=1, review_rate=0.0001) == 0.5
    assert precision_at_review_rate(labels, scores, positive_label=1, review_rate=0.0001) == 1.0
    assert recall_at_fpr(labels, scores, positive_label=1, target_fpr=0.0) == 0.5


def test_score_direction_is_explicit():
    distances = [1 - s for s in S]  # Lower distance = riskier.
    assert precision_at_k(Y, distances, k=3, greater_is_riskier=False, **POS) == pytest.approx(
        2 / 3
    )
    assert precision_at_k(Y, distances, k=3, **POS) == 0.0


@pytest.mark.parametrize("k", [0, 11, 2.5, True])
def test_invalid_k(k):
    with pytest.raises(ConfigurationError, match="k must be an int"):
        precision_at_k(Y, S, k=k, **POS)


@pytest.mark.parametrize(
    "scores, match",
    [
        ([0.1] * 9, "same length"),
        ([np.nan] * 10, "NaN"),
        ([[0.1] * 10], "one-dimensional"),
        ([True] * 10, "not boolean"),
        (["a"] * 10, "numeric"),
    ],
)
def test_invalid_scores(scores, match):
    with pytest.raises(ConfigurationError, match=match):
        precision_at_k(Y, scores, k=1, **POS)


def test_empty_sample():
    with pytest.raises(UndefinedMetricError, match="empty"):
        precision_at_k([], [], k=1, **POS)


class TestLabelInputForms:
    """Labels in an unsupported form must never be scored as "all negative"."""

    SCORES = [0.9, 0.8, 0.7, 0.1]

    def test_torch_tensor_and_numpy_labels_give_the_same_result(self):
        import torch

        for labels in (
            [1, 0, 1, 0],
            np.array([1, 0, 1, 0]),
            torch.tensor([1, 0, 1, 0]),
            np.array([1, 0, 1, 0], dtype=object),
            (label for label in [1, 0, 1, 0]),
        ):
            assert precision_at_k(labels, self.SCORES, positive_label=1, k=2) == 0.5

    def test_string_and_bool_arrays(self):
        labels = np.array(["risk", "ok", "risk", "ok"])
        assert precision_at_k(labels, self.SCORES, positive_label="risk", k=1) == 1.0
        assert precision_at_k(labels == "risk", self.SCORES, positive_label=True, k=1) == 1.0
        # Type still matters on the vectorised path: integer 1 is not the boolean True.
        assert precision_at_k(labels == "risk", self.SCORES, positive_label=1, k=1) == 0.0

    @pytest.mark.parametrize(
        "labels, match",
        [
            (np.array([[1], [0], [1], [0]]), "one-dimensional, got 2"),
            ([[1], [0], [1], [0]], "scalar labels; found an element of type list"),
            ([np.array([1, 2])] * 4, "scalar labels; found an element of type ndarray"),
            ("1010", "not a single string"),
        ],
    )
    def test_malformed_labels_raise_instead_of_scoring_zero(self, labels, match):
        from embedfit.evaluation import brier_score

        with pytest.raises(ConfigurationError, match=match):
            precision_at_k(labels, self.SCORES, positive_label=1, k=2)
        with pytest.raises(ConfigurationError, match=match):
            brier_score(labels, self.SCORES, positive_label=1)

    def test_column_vector_tensor_is_rejected(self):
        import torch

        with pytest.raises(ConfigurationError, match="one-dimensional"):
            precision_at_k(torch.tensor([[1], [0]]), [0.9, 0.1], positive_label=1, k=1)


def test_review_rate_metrics_honour_score_direction():
    distances = [1 - s for s in S]
    assert recall_at_review_rate(
        Y, distances, review_rate=0.25, greater_is_riskier=False, **POS
    ) == pytest.approx(2 / 3)
    assert dollar_recall_at_review_rate(
        Y, distances, A, review_rate=0.3, greater_is_riskier=False, **POS
    ) == pytest.approx(0.15)


def test_dollar_recall_on_an_empty_sample():
    with pytest.raises(UndefinedMetricError, match="empty"):
        dollar_recall([], np.array([], dtype=bool), [], **POS)
