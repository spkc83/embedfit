"""A PyTorch linear classifier: a standalone head, and the classifier of ``end_to_end`` mode."""

import json
from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch import nn

from embedfit.exceptions import ConfigurationError, HeadNotFittedError
from embedfit.heads.base import ClassificationHead, validate_features, validate_targets


class LinearHead(ClassificationHead):
    """Softmax regression on embeddings, trained with cross-entropy.

    ``fit`` runs full-batch L-BFGS from a seeded initialisation, so it is deterministic. In
    ``end_to_end`` training the layer is instead trained jointly with the backbone and handed
    to the head with ``from_module``.

    Args:
        weight_decay: L2 penalty on the weights (>= 0).
        max_iter: Maximum L-BFGS iterations (>= 1).
        seed: Seed for the weight initialisation.

    Raises:
        ConfigurationError: If an option is invalid.
    """

    registry_name = "linear"

    def __init__(self, *, weight_decay: float = 1e-4, max_iter: int = 200, seed: int = 42) -> None:
        valid = isinstance(weight_decay, int | float) and not isinstance(weight_decay, bool)
        if not valid or weight_decay < 0:
            raise ConfigurationError(f"weight_decay must be a number >= 0, got {weight_decay!r}.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ConfigurationError(f"max_iter must be a positive int, got {max_iter!r}.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ConfigurationError(f"seed must be an int in [0, 2**32 - 1], got {seed!r}.")
        self.weight_decay = float(weight_decay)
        self.max_iter = max_iter
        self.seed = seed
        self._layer: nn.Linear | None = None

    @classmethod
    def from_module(cls, layer: nn.Linear, **options: float | int) -> Self:
        """Wrap an already trained ``nn.Linear`` (``embedding_dim → num_classes``) as a head.

        The layer is copied to the CPU in float32 and detached from any optimizer.
        """
        if not isinstance(layer, nn.Linear) or layer.bias is None:
            raise ConfigurationError("layer must be an nn.Linear with a bias.")
        head = cls(**options)  # type: ignore[arg-type]
        copy = nn.Linear(layer.in_features, layer.out_features)
        copy.load_state_dict({k: v.detach().float().cpu() for k, v in layer.state_dict().items()})
        head._layer = copy.requires_grad_(False)
        return head

    @property
    def is_fitted(self) -> bool:
        """Whether the head holds trained weights."""
        return self._layer is not None

    @property
    def num_classes(self) -> int | None:
        """Number of output classes, or ``None`` before fitting."""
        return self._layer.out_features if self._layer is not None else None

    def fit(self, features: np.ndarray, targets: np.ndarray, *, num_classes: int) -> None:
        """Fit on ``[N, D]`` features and ``[N]`` class indices. See ``ClassificationHead.fit``."""
        matrix = validate_features(features)
        indices = validate_targets(targets, num_samples=len(matrix), num_classes=num_classes)
        inputs, labels = torch.from_numpy(matrix), torch.from_numpy(indices.astype(np.int64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            layer = nn.Linear(matrix.shape[1], num_classes)
        optimizer = torch.optim.LBFGS(
            layer.parameters(), max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss: torch.Tensor = nn.functional.cross_entropy(layer(inputs), labels)
            loss = loss + 0.5 * self.weight_decay * layer.weight.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        self._layer = layer.requires_grad_(False)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return ``[N, num_classes]`` probabilities; column ``i`` is class index ``i``."""
        if self._layer is None:
            raise HeadNotFittedError("The head has not been fitted; call fit first.")
        matrix = validate_features(features, expected_dim=self._layer.in_features)
        with torch.no_grad():
            probabilities: np.ndarray = (
                torch.softmax(self._layer(torch.from_numpy(matrix)), dim=1).double().numpy()
            )
        return probabilities

    def save(self, path: str | Path) -> None:
        """Write options to ``config.json`` and the weights to ``parameters.npz`` (no pickle)."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "type": self.registry_name,
            "weight_decay": self.weight_decay,
            "max_iter": self.max_iter,
            "seed": self.seed,
            "fitted": self.is_fitted,
        }
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (directory / "parameters.npz").unlink(missing_ok=True)
        if self._layer is not None:
            np.savez(
                directory / "parameters.npz",
                weight=self._layer.weight.numpy(),
                bias=self._layer.bias.numpy(),
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
                weight_decay=config["weight_decay"],
                max_iter=config["max_iter"],
                seed=config["seed"],
            )
            if not config["fitted"]:
                return head
            with np.load(directory / "parameters.npz", allow_pickle=False) as parameters:
                weight, bias = parameters["weight"], parameters["bias"]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"'{directory}' does not contain a valid saved LinearHead."
            ) from exc
        if weight.ndim != 2 or bias.shape != weight.shape[:1] or weight.shape[0] < 2:
            raise ConfigurationError(
                f"'{directory}' holds parameters shaped {weight.shape} and {bias.shape}, "
                "which are not a [num_classes, D] weight and a [num_classes] bias."
            )
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.load_state_dict(
            {"weight": torch.from_numpy(weight).float(), "bias": torch.from_numpy(bias).float()}
        )
        head._layer = layer.requires_grad_(False)
        return head
