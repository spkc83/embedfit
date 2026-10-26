"""``EmbedFitModel``: a backbone plus the embedding policy built around it."""

import logging
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from embedfit.backbones.base import ImageBackbone
from embedfit.backbones.registry import create_backbone
from embedfit.data.image import ImageLike
from embedfit.data.labels import LabelMapping
from embedfit.exceptions import (
    BackboneProviderError,
    ConfigurationError,
    HeadNotFittedError,
    ImageInputError,
)
from embedfit.heads.base import ClassificationHead, validate_features
from embedfit.heads.registry import create_head
from embedfit.serialization import is_model_directory, load_model_as, save_model
from embedfit.utils.device import resolve_device

logger = logging.getLogger(__name__)

_UNSET: Any = object()  # Distinguishes "not passed" from a passed default value.


def _exists(path: str) -> bool:
    try:
        return Path(path).exists()
    except OSError:  # e.g. a name too long for the file system.
        return False


def _build_label_mapping(
    labels: Sequence[Hashable] | np.ndarray | LabelMapping | None, positive_label: Hashable | None
) -> LabelMapping | None:
    if labels is None:
        if positive_label is not None:
            raise ConfigurationError("positive_label requires labels to be given.")
        return None
    mapping = labels if isinstance(labels, LabelMapping) else LabelMapping(labels)
    if positive_label is not None and positive_label not in mapping:
        raise ConfigurationError("positive_label must be one of labels, with the same type.")
    return mapping


class EmbedFitModel(nn.Module):
    """An image-embedding model: a backbone, an embedding policy, and a lightweight classifier.

    Args:
        backbone: The encoder. It owns all preprocessing.
        labels: Class labels in the order that defines class indices. The order is taken as
            given and never sorted.
        positive_label: For binary tasks, the label that risk scores refer to. It is never
            inferred from label order.
        normalize_embeddings: Whether ``encode`` and ``forward`` L2-normalise embeddings by
            default. The head is fitted and applied on embeddings produced under this policy.
        head: Classifier fitted by ``fit_head``: a registered head name or a configured
            ``ClassificationHead`` instance, e.g.
            ``LogisticRegressionHead(class_weight="balanced")``. ``head_name`` records the name,
            or ``None`` for an instance.

    Raises:
        ConfigurationError: If ``backbone`` is not an ``ImageBackbone``, labels are fewer than
            two or not unique, ``positive_label`` is not among ``labels``, or ``head`` is unknown.
    """

    def __init__(
        self,
        backbone: ImageBackbone,
        *,
        labels: Sequence[Hashable] | np.ndarray | LabelMapping | None = None,
        positive_label: Hashable | None = None,
        normalize_embeddings: bool = True,
        head: str | ClassificationHead = "logistic_regression",
    ) -> None:
        super().__init__()
        if not isinstance(backbone, ImageBackbone):
            raise ConfigurationError(
                f"backbone must be an ImageBackbone, got {type(backbone).__name__}."
            )
        self.backbone = backbone
        if not isinstance(head, str | ClassificationHead):
            raise ConfigurationError(
                f"head must be a head name or a ClassificationHead, got {type(head).__name__}."
            )
        self.label_mapping = _build_label_mapping(labels, positive_label)
        self.positive_label = positive_label
        self.normalize_embeddings = normalize_embeddings
        self.head: ClassificationHead = create_head(head) if isinstance(head, str) else head
        self.head_name: str | None = head if isinstance(head, str) else None

    @classmethod
    def from_pretrained(
        cls,
        identifier: str | Path,
        *,
        labels: Sequence[Hashable] | np.ndarray | LabelMapping | None = _UNSET,
        positive_label: Hashable | None = _UNSET,
        normalize_embeddings: bool = _UNSET,
        head: str | ClassificationHead = _UNSET,
        device: str | torch.device | None = None,
        **backbone_kwargs: Any,
    ) -> Self:
        """Build a model from a provider-qualified backbone identifier, or load a saved model.

        Args:
            identifier: ``"<provider>/<name>"``, e.g. ``"torchvision/resnet18"``; or the path
                of a directory written by ``save_pretrained``. A directory containing
                ``embedfit_config.json`` is a saved model and takes precedence, even when its
                path also reads as an identifier; any other directory is never mistaken for
                one. A saved model restores its own labels, positive label, normalisation
                policy, and head, so only ``device`` may be passed with it.
            labels: See the class documentation. Default: none.
            positive_label: See the class documentation. Default: none.
            normalize_embeddings: See the class documentation. Default: ``True``.
            head: See the class documentation. Default: ``"logistic_regression"``.
            device: Where to place the model. ``None`` leaves it on CPU; ``"auto"`` selects
                CUDA → MPS → CPU; an explicit device that is unavailable raises.
            **backbone_kwargs: Forwarded to the provider adapter, e.g. ``weights=None`` or
                ``image_size=448`` for ``TorchvisionBackbone``.

        Returns:
            The model, in eval mode.

        Raises:
            BackboneProviderError: If the identifier is malformed or its provider is unknown.
            ConfigurationError: If labels or the device are invalid, any option other than
                ``device`` is passed together with a saved model (whatever its value), the
                saved model is incomplete or inconsistent, or ``identifier`` is an existing
                path that is not a saved model.
        """
        options = {
            "labels": labels,
            "positive_label": positive_label,
            "normalize_embeddings": normalize_embeddings,
            "head": head,
        }
        if isinstance(identifier, str | Path) and is_model_directory(identifier):
            passed = sorted(
                [name for name, value in options.items() if value is not _UNSET]
                + list(backbone_kwargs)
            )
            if passed:
                raise ConfigurationError(
                    f"{passed} cannot be set when loading a saved model; the saved values "
                    "are restored. Only device may be passed."
                )
            return load_model_as(cls, identifier, device=device)

        not_a_bundle = ConfigurationError(
            f"'{identifier}' is not a saved model directory: it has no embedfit_config.json."
        )
        if not isinstance(identifier, str):
            raise not_a_bundle
        if labels is _UNSET:
            labels = None
        if positive_label is _UNSET:
            positive_label = None
        if normalize_embeddings is _UNSET:
            normalize_embeddings = True
        if head is _UNSET:
            head = "logistic_regression"
        # Validate cheap arguments before building weights.
        mapping = _build_label_mapping(labels, positive_label)
        target = resolve_device(device) if device is not None else None
        try:
            backbone = create_backbone(identifier, **backbone_kwargs)
        except BackboneProviderError as exc:
            if _exists(identifier):  # A real path that is simply not a bundle.
                raise not_a_bundle from exc
            raise
        model = cls(
            backbone,
            labels=mapping,
            positive_label=positive_label,
            normalize_embeddings=normalize_embeddings,
            head=head,
        )
        if target is not None:
            model.to(target)
        model.eval()
        logger.info("Built %s: embedding dimension %d", backbone.identifier, backbone.embedding_dim)
        return model

    @property
    def labels(self) -> tuple[Hashable, ...] | None:
        """The class labels in class-index order, or ``None`` if none were given."""
        return self.label_mapping.labels if self.label_mapping is not None else None

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

    def unfreeze_backbone(
        self, last_blocks: int | None = None, trainable_parameters: str = "all"
    ) -> None:
        """Allow gradient updates to the whole backbone, or only to a selection of it.

        Args:
            last_blocks: As in ``ImageBackbone.unfreeze``.
            trainable_parameters: As in ``ImageBackbone.unfreeze``.

        Raises:
            ConfigurationError: If ``last_blocks`` is out of range for the backbone, or
                ``trainable_parameters`` is unknown or selects no parameter.
        """
        self.backbone.unfreeze(last_blocks, trainable_parameters)

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
                image, numpy array, torch tensor, file path, or encoded image file bytes), and
                forms may be mixed. A 4-D array is rejected: pass a list of images instead.
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

    def _features(
        self,
        images: ImageLike | Sequence[ImageLike] | None,
        embeddings: np.ndarray | None,
        batch_size: int,
    ) -> np.ndarray:
        if embeddings is not None and images is not None:
            raise ConfigurationError("Pass exactly one of images or embeddings, not both.")
        if embeddings is not None:
            return validate_features(embeddings, expected_dim=self.embedding_dim)
        if images is None:
            raise ConfigurationError("Pass exactly one of images or embeddings.")
        encoded = self.encode(images, batch_size=batch_size, convert_to_numpy=True)
        assert isinstance(encoded, np.ndarray)
        return encoded

    def fit_head(
        self,
        images: Sequence[ImageLike] | None = None,
        labels: Sequence[Hashable] | np.ndarray | None = None,
        *,
        embeddings: np.ndarray | None = None,
        batch_size: int = 32,
    ) -> None:
        """Fit the classification head; the backbone is not updated.

        Args:
            images: Training images, encoded with the model's normalisation policy. Give
                either ``images`` or ``embeddings``, never both.
            labels: One label per sample. Every label must be one of the model's ``labels``,
                and every model label must occur at least once.
            embeddings: Precomputed ``[N, embedding_dim]`` embeddings. They must come from this
                model's ``encode`` with its default normalisation, or predictions made from
                images will not match.
            batch_size: Images per forward pass when encoding.

        Raises:
            ConfigurationError: If the model has no ``labels``, inputs are inconsistent, a
                label is unknown, or a class has no samples.
        """
        if self.label_mapping is None:
            raise ConfigurationError(
                "fit_head needs the model's class labels; pass labels=[...] when creating the "
                "model. Class order is never inferred from the training data."
            )
        if labels is None:
            raise ConfigurationError("fit_head requires labels.")
        targets = self.label_mapping.encode(labels)
        features = self._features(images, embeddings, batch_size)
        self.head.fit(features, targets, num_classes=len(self.label_mapping))
        logger.info(
            "Fitted head on %d samples across %d classes", len(features), len(self.label_mapping)
        )

    def predict_proba(
        self,
        images: ImageLike | Sequence[ImageLike] | None = None,
        *,
        embeddings: np.ndarray | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return class probabilities.

        Args:
            images: Images to classify. Give either ``images`` or ``embeddings``.
            embeddings: Precomputed ``[N, embedding_dim]`` embeddings from this model.
            batch_size: Images per forward pass when encoding.

        Returns:
            Array shaped ``[N, num_classes]``. Column ``i`` is the probability of
            ``model.labels[i]``: the order given at construction, never sorted.

        Raises:
            HeadNotFittedError: If the head has not been fitted.
        """
        if not self.head.is_fitted:
            raise HeadNotFittedError(
                "The classification head is not fitted; call fit_head or Trainer.train first."
            )
        return self.head.predict_proba(self._features(images, embeddings, batch_size))

    def predict(
        self,
        images: ImageLike | Sequence[ImageLike] | None = None,
        *,
        embeddings: np.ndarray | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return the most probable label for each input.

        Args:
            images: Images to classify. Give either ``images`` or ``embeddings``.
            embeddings: Precomputed ``[N, embedding_dim]`` embeddings from this model.
            batch_size: Images per forward pass when encoding.

        Returns:
            Array shaped ``[N]`` of external labels (the values given as ``labels``).

        Raises:
            HeadNotFittedError: If the head has not been fitted.
        """
        probabilities = self.predict_proba(images, embeddings=embeddings, batch_size=batch_size)
        assert self.label_mapping is not None  # A fitted head implies labels were given.
        return self.label_mapping.decode(probabilities.argmax(axis=1))

    def risk_score(
        self,
        images: ImageLike | Sequence[ImageLike] | None = None,
        *,
        embeddings: np.ndarray | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return the probability of ``positive_label`` for each input; higher means riskier.

        Returns:
            Array shaped ``[N]`` with values in [0, 1].

        Raises:
            ConfigurationError: If the model was created without ``positive_label``. The
                positive class is never inferred from label order.
            HeadNotFittedError: If the head has not been fitted.
        """
        if self.positive_label is None or self.label_mapping is None:
            raise ConfigurationError(
                "risk_score needs an explicit positive class; pass positive_label=... when "
                "creating the model."
            )
        probabilities = self.predict_proba(images, embeddings=embeddings, batch_size=batch_size)
        column: np.ndarray = probabilities[:, self.label_mapping.index_of(self.positive_label)]
        return column

    def save_pretrained(
        self, path: str | Path, *, training_metadata: Mapping[str, Any] | None = None
    ) -> None:
        """Save the model to a directory that ``from_pretrained`` can load.

        The bundle holds the backbone's weights and preprocessing settings, the head's state,
        the labels in class-index order, the positive label, and the normalisation policy. It
        never holds training samples, identifiers, or metadata, and nothing is pickled.

        Args:
            path: Target directory; created if missing.
            training_metadata: Optional JSON-serialisable aggregate facts about training.

        Raises:
            ConfigurationError: If a label is not a ``str``, ``int``, ``float``, or ``bool``,
                the head does not support saving, or ``training_metadata`` is not
                JSON-serialisable.
        """
        save_model(self, path, training_metadata=training_metadata)
