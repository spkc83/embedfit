"""The contract for lightweight classifiers that operate on embeddings."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Self

import numpy as np

from embedfit.exceptions import ConfigurationError


def validate_features(features: np.ndarray, *, expected_dim: int | None = None) -> np.ndarray:
    """Check that ``features`` is a finite ``[N, D]`` float matrix and return it as float32.

    Embeddings are float32 throughout EmbedFit, so float64 input is cast down.

    Raises:
        ConfigurationError: If the shape, width, or values are invalid.
    """
    matrix = np.asarray(features)
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.floating):
        raise ConfigurationError(
            f"features must be a 2-D float array [N, D]; got shape {matrix.shape}, "
            f"dtype {matrix.dtype}."
        )
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise ConfigurationError(
            f"features have dimension {matrix.shape[1]}, expected {expected_dim}."
        )
    if not np.isfinite(matrix).all():
        raise ConfigurationError("features contain NaN or infinite values.")
    return matrix.astype(np.float32, copy=False)


class ClassificationHead(ABC):
    """A classifier over ``[N, D]`` feature matrices and integer class indices.

    Heads know nothing about images, backbones, or external label names; the model translates
    labels to indices ``0..num_classes-1`` before calling a head.

    A head that should survive ``EmbedFitModel.save_pretrained`` sets ``registry_name`` to the
    name it is registered under and implements ``save`` and ``load``.
    """

    registry_name: ClassVar[str | None] = None

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether ``fit`` has completed."""

    @abstractmethod
    def fit(self, features: np.ndarray, targets: np.ndarray, *, num_classes: int) -> None:
        """Fit the classifier.

        Args:
            features: Float array shaped ``[N, D]``.
            targets: Integer class indices shaped ``[N]``, each in ``[0, num_classes)``.
            num_classes: Total number of classes. Every class must appear in ``targets``, so
                that ``predict_proba`` always has ``num_classes`` meaningful columns.

        Raises:
            ConfigurationError: If shapes disagree, a target is out of range, or a class has
                no training samples.
        """

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities shaped ``[N, num_classes]``; column ``i`` is class ``i``.

        Raises:
            HeadNotFittedError: If called before ``fit``.
        """

    def save(self, path: str | Path) -> None:
        """Write the head's options and fitted state to a directory.

        Only hyperparameters and learned parameters may be written: never training samples.

        Raises:
            ConfigurationError: If this head type does not implement persistence.
        """
        raise ConfigurationError(f"{type(self).__name__} does not support saving.")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Rebuild a head from a directory written by ``save``.

        Raises:
            ConfigurationError: If this head type does not implement persistence.
        """
        raise ConfigurationError(f"{cls.__name__} does not support loading.")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the most probable class index for each row, shaped ``[N]``."""
        indices: np.ndarray = self.predict_proba(features).argmax(axis=1)
        return indices


def validate_targets(targets: np.ndarray, *, num_samples: int, num_classes: int) -> np.ndarray:
    """Check class-index targets against the feature count and class count.

    Raises:
        ConfigurationError: If the length differs from ``num_samples``, a value is outside
            ``[0, num_classes)``, or any class is absent.
    """
    indices = np.asarray(targets)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ConfigurationError("targets must be a 1-D integer array of class indices.")
    if len(indices) != num_samples:
        raise ConfigurationError(
            f"features and targets must have the same length, got {num_samples} and {len(indices)}."
        )
    if len(indices) and (indices.min() < 0 or indices.max() >= num_classes):
        raise ConfigurationError(f"targets must lie in [0, {num_classes}).")
    counts = np.bincount(indices, minlength=num_classes)
    absent = np.flatnonzero(counts == 0).tolist()
    if absent:
        raise ConfigurationError(
            f"Class indices {absent} have no training samples; every class needs at least one."
        )
    return indices
