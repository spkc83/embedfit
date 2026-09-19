"""``EmbedFitModel``: a backbone plus the embedding policy built around it."""

import logging
from collections.abc import Hashable, Sequence
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from embedfit.backbones.base import ImageBackbone
from embedfit.backbones.registry import create_backbone
from embedfit.data.image import ImageLike
from embedfit.exceptions import ConfigurationError, ImageInputError
from embedfit.utils.device import resolve_device

logger = logging.getLogger(__name__)


def _validate_labels(
    labels: Sequence[Hashable] | np.ndarray | None, positive_label: Hashable | None
) -> tuple[Hashable, ...] | None:
    if labels is None:
        if positive_label is not None:
            raise ConfigurationError("positive_label requires labels to be given.")
        return None
    if isinstance(labels, np.ndarray) and labels.ndim == 1:
        labels = labels.tolist()
    if isinstance(labels, str) or not isinstance(labels, Sequence):
        raise ConfigurationError(
            "labels must be a sequence of label values, e.g. a list or 1-D numpy array."
        )
    if len(labels) < 2:
        raise ConfigurationError(f"labels must contain at least 2 classes, got {len(labels)}.")
    try:
        unique_count = len(set(labels))
    except TypeError as exc:
        raise ConfigurationError("labels must be hashable values such as str or int.") from exc
    if unique_count != len(labels):
        raise ConfigurationError("labels must be unique.")
    # Compare types too: ``False == 0`` must not make False a member of [0, 1].
    if positive_label is not None and not any(
        type(label) is type(positive_label) and label == positive_label for label in labels
    ):
        raise ConfigurationError("positive_label must be one of labels, with the same type.")
    return tuple(labels)


class EmbedFitModel(nn.Module):
    """An image-embedding model: a backbone and an embedding-normalisation policy.

    Args:
        backbone: The encoder. It owns all preprocessing.
        labels: Class labels in the order that defines class indices. The order is taken as
            given and never sorted.
        positive_label: For binary tasks, the label that risk scores refer to. It is never
            inferred from label order.
        normalize_embeddings: Whether ``encode`` and ``forward`` L2-normalise embeddings by
            default.

    Raises:
        ConfigurationError: If ``backbone`` is not an ``ImageBackbone``, labels are fewer than
            two or not unique, or ``positive_label`` is not among ``labels``.
    """

    def __init__(
        self,
        backbone: ImageBackbone,
        *,
        labels: Sequence[Hashable] | np.ndarray | None = None,
        positive_label: Hashable | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, ImageBackbone):
            raise ConfigurationError(
                f"backbone must be an ImageBackbone, got {type(backbone).__name__}."
            )
        self.backbone = backbone
        self.labels = _validate_labels(labels, positive_label)
        self.positive_label = positive_label
        self.normalize_embeddings = normalize_embeddings

    @classmethod
    def from_pretrained(
        cls,
        identifier: str,
        *,
        labels: Sequence[Hashable] | np.ndarray | None = None,
        positive_label: Hashable | None = None,
        normalize_embeddings: bool = True,
        device: str | torch.device | None = None,
        **backbone_kwargs: Any,
    ) -> Self:
        """Build a model around a backbone named by a provider-qualified identifier.

        Args:
            identifier: ``"<provider>/<name>"``, e.g. ``"torchvision/resnet18"``.
            labels: See the class documentation.
            positive_label: See the class documentation.
            normalize_embeddings: See the class documentation.
            device: Where to place the model. ``None`` leaves it on CPU; ``"auto"`` selects
                CUDA → MPS → CPU; an explicit device that is unavailable raises.
            **backbone_kwargs: Forwarded to the provider adapter, e.g. ``weights=None`` or
                ``image_size=448`` for ``TorchvisionBackbone``.

        Returns:
            The model, in eval mode.

        Raises:
            BackboneProviderError: If the identifier is malformed or its provider is unknown.
            ConfigurationError: If labels or the device are invalid.
        """
        validated = _validate_labels(labels, positive_label)  # Fail before building weights.
        target = resolve_device(device) if device is not None else None
        backbone = create_backbone(identifier, **backbone_kwargs)
        model = cls(
            backbone,
            labels=validated,
            positive_label=positive_label,
            normalize_embeddings=normalize_embeddings,
        )
        if target is not None:
            model.to(target)
        model.eval()
        logger.info("Built %s: embedding dimension %d", backbone.identifier, backbone.embedding_dim)
        return model

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embeddings."""
        return self.backbone.embedding_dim

    @property
    def device(self) -> torch.device:
        """Device holding the backbone's parameters (CPU for a parameter-free backbone)."""
        parameter = next(self.backbone.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    def freeze_backbone(self) -> None:
        """Stop gradient updates to the backbone and hold it in eval mode."""
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        """Allow gradient updates to the backbone again."""
        self.backbone.unfreeze()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Differentiable embedding of preprocessed images, for use in training loops.

        Args:
            pixel_values: Float tensor shaped ``[batch, 3, H, W]`` on the model's device, from
                stacked ``backbone.preprocess`` outputs.

        Returns:
            Float tensor shaped ``[batch, embedding_dim]``, L2-normalised when
            ``normalize_embeddings`` is true.
        """
        embeddings: torch.Tensor = self.backbone.forward_features(pixel_values)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, dim=1)
        return embeddings

    def encode(
        self,
        images: ImageLike | Sequence[ImageLike],
        *,
        batch_size: int = 32,
        normalize: bool | None = None,
        convert_to_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Embed images without tracking gradients.

        Runs the backbone in eval mode and restores its previous mode afterwards.

        Args:
            images: A ``list`` or ``tuple`` of images is a batch; any other value is one image.
                Each image may be any form accepted by ``embedfit.data.to_rgb_tensor`` (PIL
                image, numpy array, torch tensor, or file path), and forms may be mixed. A 4-D
                array is rejected: pass a list of images instead.
            batch_size: Number of images per forward pass.
            normalize: L2-normalise each embedding. ``None`` uses ``normalize_embeddings``.
            convert_to_numpy: Return ``numpy.ndarray`` instead of a CPU ``torch.Tensor``.

        Returns:
            A float32 array (whatever the backbone's dtype) shaped ``[N, embedding_dim]`` in
            input order; ``N`` is 1 for a
            single image and 0 for an empty list.

        Raises:
            ConfigurationError: If ``batch_size`` is not a positive integer.
            ImageInputError: If an image is malformed. The message gives the image's position
                in the input, never its contents.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigurationError(f"batch_size must be a positive int, got {batch_size!r}.")
        items = list(images) if isinstance(images, list | tuple) else [images]
        apply_norm = self.normalize_embeddings if normalize is None else normalize
        parameter = next(self.backbone.parameters(), None)
        device = self.device
        dtype = parameter.dtype if parameter is not None else torch.float32

        chunks: list[torch.Tensor] = []
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(items), batch_size):
                    pixels = []
                    for offset, image in enumerate(items[start : start + batch_size]):
                        try:
                            pixels.append(self.backbone.preprocess(image))
                        except ImageInputError as exc:
                            raise ImageInputError(
                                f"Invalid image at index {start + offset}: {exc}"
                            ) from exc
                    batch = torch.stack(pixels).to(device=device, dtype=dtype)
                    embeddings = self.backbone.forward_features(batch)
                    if apply_norm:
                        embeddings = F.normalize(embeddings, dim=1)
                    chunks.append(embeddings.float().cpu())
        finally:
            self.backbone.train(was_training)

        result = torch.cat(chunks) if chunks else torch.empty(0, self.embedding_dim)
        logger.debug("Encoded %d images into dimension %d", len(items), self.embedding_dim)
        return result.numpy() if convert_to_numpy else result
