import numpy as np
import pytest

from embedfit.evaluation import brier_score, calibration_bins, expected_calibration_error
from embedfit.exceptions import ConfigurationError, UndefinedMetricError


def test_brier_score_hand_computed():
    # ((0.8 - 1)^2 + (0.3 - 0)^2) / 2
    assert brier_score(["risk", "ok"], [0.8, 0.3], positive_label="risk") == pytest.approx(0.065)
    assert brier_score([1, 0], [1.0, 0.0], positive_label=1) == 0.0
    assert brier_score([1, 0], [0.0, 1.0], positive_label=1) == 1.0


def test_calibration_bins_hand_computed():
    bins = calibration_bins([0, 1, 1, 1], [0.1, 0.2, 0.6, 1.0], positive_label=1, n_bins=2)
    assert bins.edges.tolist() == [0.0, 0.5, 1.0]
    assert bins.counts.tolist() == [2, 2]  # 1.0 belongs to the last bin.
    assert bins.mean_predicted == pytest.approx([0.15, 0.8])
    assert bins.fraction_positive == pytest.approx([0.5, 1.0])
    # ECE = (2 * |0.15 - 0.5| + 2 * |0.8 - 1.0|) / 4
    assert expected_calibration_error(
        [0, 1, 1, 1], [0.1, 0.2, 0.6, 1.0], positive_label=1, n_bins=2
    ) == pytest.approx(0.275)


def test_empty_bins_are_nan_and_excluded_from_ece():
    bins = calibration_bins([1, 0], [0.95, 0.05], positive_label=1, n_bins=10)
    assert bins.counts.sum() == 2 and np.isnan(bins.mean_predicted[5])
    assert expected_calibration_error([1, 0], [0.95, 0.05], positive_label=1) == pytest.approx(0.05)


def test_perfectly_calibrated_predictions_have_zero_ece():
    probabilities = np.repeat([0.25, 0.75], 400)
    labels = np.concatenate([[1] * 100 + [0] * 300, [1] * 300 + [0] * 100])
    assert expected_calibration_error(labels, probabilities, positive_label=1) == pytest.approx(0)


@pytest.mark.parametrize("probabilities", [[1.2, 0.1], [-0.1, 0.5]])
def test_probabilities_outside_the_unit_interval(probabilities):
    with pytest.raises(ConfigurationError, match=r"\[0, 1\]"):
        brier_score([1, 0], probabilities, positive_label=1)


def test_invalid_bins_and_empty_input():
    with pytest.raises(ConfigurationError, match="n_bins"):
        calibration_bins([1, 0], [0.5, 0.5], positive_label=1, n_bins=0)
    with pytest.raises(UndefinedMetricError, match="empty"):
        brier_score([], [], positive_label=1)


@pytest.mark.parametrize("n_bins", [3, 7, 10, 20])
def test_bin_membership_agrees_with_the_returned_edges(n_bins):
    # Probabilities on and next to bin boundaries, where float arithmetic is least forgiving.
    boundaries = np.linspace(0, 1, n_bins + 1)
    probabilities = np.clip(
        np.concatenate(
            [boundaries, np.nextafter(boundaries, 0), np.nextafter(boundaries, 1), [0.3, 0.6, 0.7]]
        ),
        0,
        1,
    )
    labels = np.arange(len(probabilities)) % 2
    bins = calibration_bins(labels, probabilities, positive_label=1, n_bins=n_bins)
    assert bins.counts.sum() == len(probabilities)
    for index in np.flatnonzero(bins.counts):
        upper_ok = bins.mean_predicted[index] < bins.edges[index + 1] or index == n_bins - 1
        assert bins.edges[index] <= bins.mean_predicted[index] and upper_ok
