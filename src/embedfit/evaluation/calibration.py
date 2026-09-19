"""Calibration of predicted probabilities for the positive class."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from embedfit.evaluation._inputs import binary_inputs
from embedfit.exceptions import ConfigurationError


def _probabilities(
    y_true: Iterable[Hashable], probabilities: Any, positive_label: Hashable
) -> tuple[np.ndarray, np.ndarray]:
    mask, values = binary_inputs(y_true, probabilities, positive_label)
    if values.min() < 0 or values.max() > 1:
        raise ConfigurationError("probabilities must lie in [0, 1].")
    return mask, values


def brier_score(
    y_true: Iterable[Hashable], probabilities: Any, *, positive_label: Hashable
) -> float:
    """Mean squared error between positive-class probabilities and outcomes; lower is better.

    Raises:
        ConfigurationError: If inputs are malformed or a probability is outside [0, 1].
        UndefinedMetricError: If the sample is empty.
    """
    mask, values = _probabilities(y_true, probabilities, positive_label)
    return float(np.mean((values - mask) ** 2))


@dataclass(frozen=True)
class CalibrationBins:
    """Reliability-diagram data over equal-width probability bins.

    Attributes:
        edges: Bin edges, shaped ``[n_bins + 1]``. Bins are ``[low, high)``; the last bin also
            includes 1.0.
        counts: Samples per bin.
        mean_predicted: Mean predicted probability per bin (NaN for an empty bin).
        fraction_positive: Observed positive rate per bin (NaN for an empty bin).
    """

    edges: np.ndarray
    counts: np.ndarray
    mean_predicted: np.ndarray
    fraction_positive: np.ndarray


def calibration_bins(
    y_true: Iterable[Hashable],
    probabilities: Any,
    *,
    positive_label: Hashable,
    n_bins: int = 10,
) -> CalibrationBins:
    """Group predictions into equal-width probability bins.

    Raises:
        ConfigurationError: If inputs are malformed or ``n_bins`` is not a positive int.
        UndefinedMetricError: If the sample is empty.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ConfigurationError(f"n_bins must be a positive int, got {n_bins!r}.")
    mask, values = _probabilities(y_true, probabilities, positive_label)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Index from the returned edges, so that counts always agree with them.
    bin_index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(bin_index, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        mean_predicted = np.bincount(bin_index, weights=values, minlength=n_bins) / counts
        fraction_positive = np.bincount(bin_index, weights=mask, minlength=n_bins) / counts
    return CalibrationBins(edges, counts, mean_predicted, fraction_positive)


def expected_calibration_error(
    y_true: Iterable[Hashable],
    probabilities: Any,
    *,
    positive_label: Hashable,
    n_bins: int = 10,
) -> float:
    """Sample-weighted mean gap between predicted and observed positive rates across bins.

    Raises:
        ConfigurationError: If inputs are malformed.
        UndefinedMetricError: If the sample is empty.
    """
    bins = calibration_bins(y_true, probabilities, positive_label=positive_label, n_bins=n_bins)
    filled = bins.counts > 0
    gaps = np.abs(bins.mean_predicted[filled] - bins.fraction_positive[filled])
    return float((gaps * bins.counts[filled]).sum() / bins.counts.sum())
