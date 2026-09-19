"""The contract every backbone provider adapter implements."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import torch
from torch import nn

from embedfit.data.image import ImageLike
from embedfit.exceptions import ImageInputError


class ImageBackbone(nn.Module, ABC):
    """A pretrained (or trainable) encoder producing one vector per image.

    An adapter owns everything checkpoint-specific: construction, input preprocessing
    (resolution and normalisation constants), feature extraction, pooling, and the metadata
    needed to rebuild it (ADR 0003). Code outside the adapter assumes none of these.

    Subclasses implement ``identifier``, ``embedding_dim``, ``preprocess``,
    ``forward_features``, and ``save_pretrained``.
    """

    _frozen: bool = False

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Provider-qualified name, e.g. ``"torchvision/resnet18"``."""

    @property
    def pooling(self) -> str:
        """How per-image features are pooled into one vector.

        ``"provider_default"`` means the checkpoint's canonical output. Adapters that offer a
        choice override this; the value is stored in saved models.
        """
        return "provider_default"

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the vectors returned by ``forward_features``."""

    @abstractmethod
    def preprocess(self, image: ImageLike) -> torch.Tensor:
        """Convert one image to this checkpoint's input tensor.

        Args:
            image: Any form accepted by ``embedfit.data.to_rgb_tensor``.

        Returns:
            A float32 CPU tensor shaped ``[3, H, W]``. Every call on the same backbone returns
            the same ``H`` and ``W``, so outputs can be stacked into a batch.

        Raises:
            ImageInputError: If the image is malformed.
        """

    @abstractmethod
    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed a batch of preprocessed images.

        Args:
            pixel_values: Float tensor shaped ``[batch, 3, H, W]`` on the backbone's device.

        Returns:
            Float tensor shaped ``[batch, embedding_dim]``, pooled but not L2-normalised.
        """

    @abstractmethod
    def save_pretrained(self, path: str | Path) -> None:
        """Write weights and the configuration needed to rebuild this backbone to ``path``.

        Only weights, hyperparameters, and preprocessing settings may be written.
        """

    @staticmethod
    def _check_pixel_values(pixel_values: torch.Tensor) -> None:
        """Raise ``ImageInputError`` unless ``pixel_values`` is shaped ``[batch, 3, H, W]``."""
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ImageInputError(
                f"pixel_values must have shape [batch, 3, H, W]; got {tuple(pixel_values.shape)}."
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Alias for ``forward_features``."""
        return self.forward_features(pixel_values)

    @property
    def is_frozen(self) -> bool:
        """Whether ``freeze`` is in effect."""
        return self._frozen

    def freeze(self) -> None:
        """Disable gradients for every current parameter and hold the module in eval mode.

        While frozen, ``train()`` leaves this module in eval mode, so a parent module entering
        training cannot update batch-norm statistics or re-enable dropout here.
        """
        self._frozen = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        """Re-enable gradients for every parameter. The train/eval mode is left unchanged."""
        self._frozen = False
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> Self:
        """Set training mode; a frozen backbone stays in eval mode regardless of ``mode``."""
        return super().train(mode and not self._frozen)
