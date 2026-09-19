"""scikit-learn logistic regression as a classification head."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from embedfit.exceptions import ConfigurationError, HeadNotFittedError
from embedfit.heads.base import ClassificationHead, validate_features, validate_targets


class LogisticRegressionHead(ClassificationHead):
    """Multinomial logistic regression on embeddings.

    Args:
        C: Inverse L2 regularisation strength; smaller values regularise more.
        max_iter: Maximum solver iterations.
        class_weight: ``None`` or ``"balanced"``, as in scikit-learn. ``"balanced"`` reweights
            classes inversely to their frequency, which matters for rare-event data.
        seed: Random state passed to the solver.

    Raises:
        ConfigurationError: If an option is invalid.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | None = None,
        seed: int = 42,
    ) -> None:
        if not C > 0:
            raise ConfigurationError(f"C must be positive, got {C!r}.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ConfigurationError(f"max_iter must be a positive int, got {max_iter!r}.")
        if class_weight not in (None, "balanced"):
            raise ConfigurationError(
                f"class_weight must be None or 'balanced', got {class_weight!r}."
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ConfigurationError(f"seed must be an int in [0, 2**32 - 1], got {seed!r}.")
        self.C = float(C)
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.seed = seed
        self._estimator: Any = None
        self._feature_dim: int | None = None
        self._num_classes = 0

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has completed."""
        return self._estimator is not None

    def fit(self, features: np.ndarray, targets: np.ndarray, *, num_classes: int) -> None:
        """Fit on ``[N, D]`` features and ``[N]`` class indices. See ``ClassificationHead.fit``."""
        matrix = validate_features(features)
        indices = validate_targets(targets, num_samples=len(matrix), num_classes=num_classes)
        estimator = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.seed,
        )
        estimator.fit(matrix, indices)
        # Every class is present, so estimator.classes_ is exactly 0..num_classes-1 and the
        # probability columns already follow class-index order.
        self._estimator = estimator
        self._feature_dim = matrix.shape[1]
        self._num_classes = num_classes

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return ``[N, num_classes]`` probabilities; column ``i`` is class index ``i``."""
        if self._estimator is None:
            raise HeadNotFittedError("The head has not been fitted; call fit first.")
        matrix = validate_features(features, expected_dim=self._feature_dim)
        if len(matrix) == 0:
            return np.empty((0, self._num_classes), dtype=np.float64)
        probabilities: np.ndarray = self._estimator.predict_proba(matrix)
        return probabilities
