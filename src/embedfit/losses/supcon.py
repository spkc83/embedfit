"""Supervised contrastive loss (Khosla et al., 2020)."""

import torch
import torch.nn.functional as F
from torch import nn

from embedfit.exceptions import ConfigurationError


class SupConLoss(nn.Module):
    """Pull same-label embeddings together and push different-label embeddings apart.

    For an anchor ``i`` with positives ``P(i)`` (other samples with the same label) and
    ``A(i)`` all other samples in the batch::

        L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i.z_p / t) / sum_{a in A(i)} exp(z_i.z_a / t) )

    The loss is the mean of ``L_i`` over anchors that have at least one positive. An anchor is
    never compared with itself. Anchors without a positive still act as negatives for others
    but contribute no term of their own.

    Args:
        temperature: Softmax temperature ``t`` (> 0). Lower values weight hard negatives more.
        normalize: L2-normalise embeddings first, making ``z_i.z_j`` a cosine similarity. Turn
            this off only for embeddings that are already unit length.

    Raises:
        ConfigurationError: If ``temperature`` is not positive.
    """

    def __init__(self, temperature: float = 0.07, *, normalize: bool = True) -> None:
        super().__init__()
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            raise ConfigurationError(f"temperature must be a number, got {temperature!r}.")
        if not temperature > 0:
            raise ConfigurationError(f"temperature must be > 0, got {temperature!r}.")
        self.temperature = float(temperature)
        self.normalize = normalize

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss for one batch.

        Args:
            embeddings: Float tensor shaped ``[B, D]``, or ``[B, V, D]`` for ``V`` views
                (augmentations) of each sample. Views of one sample share its label, so they
                are positives of each other.
            targets: Integer class indices shaped ``[B]``.

        Returns:
            A scalar tensor.

        Raises:
            ConfigurationError: If shapes are inconsistent, or no sample in the batch has a
                positive. A class-balanced sampler with at least two samples per class (or
                at least two views) prevents the latter.
        """
        given_shapes = (tuple(embeddings.shape), tuple(targets.shape))
        if embeddings.ndim == 3 and targets.ndim == 1 and len(embeddings) == len(targets):
            views = embeddings.shape[1]
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
            targets = targets.repeat_interleave(views)
        if embeddings.ndim != 2 or targets.ndim != 1 or len(embeddings) != len(targets):
            raise ConfigurationError(
                "embeddings must be [B, D] or [B, V, D] with targets [B]; got "
                f"{given_shapes[0]} and {given_shapes[1]}."
            )
        if targets.dtype.is_floating_point or targets.dtype == torch.bool:
            raise ConfigurationError(f"targets must be integer class indices, got {targets.dtype}.")

        # Half-precision inputs (mixed-precision training) are raised to float32 for stable
        # similarities; float32 and float64 inputs keep their precision.
        half = embeddings.dtype in (torch.float16, torch.bfloat16)
        features = embeddings.float() if half else embeddings
        targets = targets.to(features.device)
        if self.normalize:
            features = F.normalize(features, dim=1)
        logits = features @ features.T / self.temperature

        not_self = ~torch.eye(len(targets), dtype=torch.bool, device=targets.device)
        positives = (targets[:, None] == targets[None, :]) & not_self
        positives_per_anchor = positives.sum(dim=1)
        has_positive = positives_per_anchor > 0
        if not has_positive.any():
            raise ConfigurationError(
                "No sample in the batch shares a label with another sample, so the supervised "
                "contrastive loss is undefined. Use a class-balanced batch sampler with at "
                "least 2 samples per class."
            )

        # log_softmax over every other sample; the anchor itself is masked out with -inf.
        log_probability = F.log_softmax(logits.masked_fill(~not_self, float("-inf")), dim=1)
        positive_log_probability = log_probability.masked_fill(~positives, 0.0).sum(dim=1)
        per_anchor = -positive_log_probability[has_positive] / positives_per_anchor[has_positive]
        return per_anchor.mean()
