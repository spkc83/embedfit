"""scikit-learn logistic regression as a classification head."""

import json
from pathlib import Path
from typing import Any, Self

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

    registry_name = "logistic_regression"

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

    def save(self, path: str | Path) -> None:
        """Write options to ``config.json`` and fitted parameters to ``parameters.npz``.

        The estimator is stored as plain arrays (coefficients and intercepts), not as a pickle,
        so loading a saved head never executes code from the file.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "type": self.registry_name,
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "seed": self.seed,
            "fitted": self.is_fitted,
            "num_classes": self._num_classes,
            "feature_dim": self._feature_dim,
        }
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if self._estimator is not None:
            np.savez(
                directory / "parameters.npz",
                coef=self._estimator.coef_,
                intercept=self._estimator.intercept_,
                n_iter=self._estimator.n_iter_,
            )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Rebuild a head written by ``save``; a fitted head predicts identically.

        Raises:
            ConfigurationError: If files are missing, malformed, or inconsistent.
        """
        directory = Path(path)
        try:
            config = json.loads((directory / "config.json").read_text())
            head = cls(
                C=config["C"],
                max_iter=config["max_iter"],
                class_weight=config["class_weight"],
                seed=config["seed"],
            )
            if not config["fitted"]:
                return head
            num_classes, feature_dim = config["num_classes"], config["feature_dim"]
            with np.load(directory / "parameters.npz", allow_pickle=False) as parameters:
                coef, intercept = parameters["coef"], parameters["intercept"]
                n_iter = parameters["n_iter"]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"'{directory}' does not contain a valid saved LogisticRegressionHead."
            ) from exc
        # scikit-learn stores a single row of coefficients for a two-class problem.
        rows = 1 if num_classes == 2 else num_classes
        if coef.shape != (rows, feature_dim) or intercept.shape != (rows,):
            raise ConfigurationError(
                f"'{directory}' holds parameters shaped {coef.shape} and {intercept.shape}, "
                f"which do not match {num_classes} classes and {feature_dim} features."
            )
        estimator = LogisticRegression(
            C=head.C,
            max_iter=head.max_iter,
            class_weight=head.class_weight,
            random_state=head.seed,
        )
        estimator.coef_, estimator.intercept_, estimator.n_iter_ = coef, intercept, n_iter
        estimator.classes_ = np.arange(num_classes)
        estimator.n_features_in_ = feature_dim
        head._estimator, head._feature_dim, head._num_classes = estimator, feature_dim, num_classes
        return head
