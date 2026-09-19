"""Cosine pair loss: regress the cosine similarity of a pair onto its target."""

import torch
import torch.nn.functional as F
from torch import nn

from embedfit.exceptions import ConfigurationError


class CosinePairLoss(nn.Module):
    """Weighted mean squared error between pair cosine similarity and the pair target.

    Positive pairs (target 1) are pulled towards cosine 1. Negative pairs (target 0) are
    pushed towards cosine ``negative_target``; with the default 0 that is orthogonality, the
    objective used by SetFit.

    Args:
        negative_target: Cosine similarity that negative pairs are regressed onto, in [-1, 1).

    Raises:
        ConfigurationError: If ``negative_target`` is outside [-1, 1).
    """

    def __init__(self, negative_target: float = 0.0) -> None:
        super().__init__()
        valid = isinstance(negative_target, int | float) and not isinstance(negative_target, bool)
        if not valid or not -1 <= negative_target < 1:
            raise ConfigurationError(
                f"negative_target must be a number in [-1, 1), got {negative_target!r}."
            )
        self.negative_target = float(negative_target)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss for a batch of pairs.

        Args:
            left: Embeddings of the first images, shaped ``[P, D]``.
            right: Embeddings of the second images, shaped ``[P, D]``.
            targets: ``[P]`` tensor of 1 (similar) and 0 (dissimilar).
            weights: Optional positive ``[P]`` pair weights; the loss is their weighted mean.

        Returns:
            A scalar tensor.

        Raises:
            ConfigurationError: If shapes disagree, the batch is empty, a target is not 0 or
                1, or a weight is not positive.
        """
        if left.ndim != 2 or left.shape != right.shape or targets.shape != left.shape[:1]:
            raise ConfigurationError(
                "left and right must be [P, D] and targets [P]; got "
                f"{tuple(left.shape)}, {tuple(right.shape)}, and {tuple(targets.shape)}."
            )
        if len(targets) == 0:
            raise ConfigurationError("The pair batch is empty.")
        targets = targets.to(left.device)
        if not ((targets == 0) | (targets == 1)).all():
            raise ConfigurationError("targets must contain only 0 and 1.")
        half = left.dtype in (torch.float16, torch.bfloat16)
        first, second = (left.float(), right.float()) if half else (left, right)
        similarity = F.cosine_similarity(first, second, dim=1)
        goal = torch.where(targets == 1, 1.0, self.negative_target).to(similarity.dtype)
        errors = (similarity - goal) ** 2
        if weights is None:
            return errors.mean()
        weights = weights.to(device=similarity.device, dtype=similarity.dtype)
        if weights.shape != targets.shape or not (weights > 0).all():
            raise ConfigurationError("weights must be positive and shaped like targets.")
        return (errors * weights).sum() / weights.sum()
