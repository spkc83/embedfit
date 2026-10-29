"""The contract every backbone provider adapter implements."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import torch
from torch import nn

from embedfit.data.image import ImageLike
from embedfit.exceptions import ConfigurationError, ImageInputError

TRAINABLE_PARAMETERS = ("all", "layernorm", "bias")
"""Which backbone parameters ``ImageBackbone.unfreeze`` can make trainable (ADR 0021)."""

# Normalisation layers of torch, subclasses included: batch and instance norm (``_NormBase``),
# layer, group, and RMS norm (torch 2.4 and later).
_NORMALIZATION_TYPES: tuple[type[nn.Module], ...] = (
    nn.modules.batchnorm._NormBase,
    nn.LayerNorm,
    nn.GroupNorm,
    *((nn.RMSNorm,) if hasattr(nn, "RMSNorm") else ()),
)


def check_last_blocks(last_blocks: object, num_blocks: int, identifier: str) -> None:
    """Raise ``ConfigurationError`` unless ``last_blocks`` is an int in ``[1, num_blocks]``."""
    if num_blocks == 0:
        raise ConfigurationError(
            f"Backbone '{identifier}' defines no blocks, so partial unfreezing is not "
            "available for it; leave unfreeze_last_blocks at None."
        )
    if (
        isinstance(last_blocks, bool)
        or not isinstance(last_blocks, int)
        or not 1 <= last_blocks <= num_blocks
    ):
        raise ConfigurationError(
            f"last_blocks must be an int in [1, {num_blocks}] for backbone '{identifier}', "
            f"got {last_blocks!r}."
        )


class ImageBackbone(nn.Module, ABC):
    """A pretrained (or trainable) encoder producing one vector per image.

    An adapter owns everything checkpoint-specific: construction, input preprocessing
    (resolution and normalisation constants), feature extraction, pooling, and the metadata
    needed to rebuild it (ADR 0003). Code outside the adapter assumes none of these.

    Subclasses implement ``identifier``, ``embedding_dim``, ``preprocess``,
    ``forward_features``, and ``save_pretrained``, and may implement ``parameter_blocks`` to
    support partial unfreezing, and ``is_normalization`` to recognise their provider's own
    normalisation layers.
    """

    _frozen: bool = False
    _partial: bool = False

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

        Only weights, hyperparameters, and preprocessing settings may be written. The settings
        go in ``<path>/config.json`` as a JSON object with a ``"provider"`` key naming the
        registered provider whose loader reads the directory back; saved models fingerprint
        that file.
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

    def parameter_blocks(self) -> list[list[str]]:
        """Names of the parameters in each block of this backbone, input side first.

        A block is a stage or layer that partial unfreezing counts (``unfreeze(last_blocks=...)``).
        Names are as returned by ``named_parameters()``. A parameter in no block is placed by
        registration order: before the first block it belongs to the stem, which never trains
        under partial unfreezing; between two blocks it joins the block after it; after the
        last block it belongs to the tail (a final norm, for example), which trains with the
        last block. The default returns no blocks, so partial unfreezing is unavailable.
        """
        return []

    @property
    def num_blocks(self) -> int:
        """Number of blocks partial unfreezing can count (0 if the backbone defines none)."""
        return len(self.parameter_blocks())

    def _names_of(self, *modules: nn.Module) -> list[str]:
        """Names, under this backbone, of the parameters of ``modules``, in registration order."""
        wanted = {id(parameter) for module in modules for parameter in module.parameters()}
        return [name for name, parameter in self.named_parameters() if id(parameter) in wanted]

    def block_positions(self) -> tuple[dict[str, int], int]:
        """The block of every parameter, and the number of blocks.

        Returns:
            A map from each name of ``named_parameters()`` to its block index in
            ``[0, num_blocks)``, -1 for the stem, or ``num_blocks`` for the tail (placed as
            ``parameter_blocks`` describes), and ``num_blocks``. Partial unfreezing and
            layer-wise learning-rate decay both count blocks this way (ADRs 0015 and 0020).
        """
        blocks = self.parameter_blocks()
        owner = {name: index for index, names in enumerate(blocks) for name in names}
        positions: dict[str, int] = {}
        pending: list[str] = []
        for name, _ in self.named_parameters():
            if name not in owner:
                pending.append(name)
                continue
            # Unowned parameters before the first block form the stem; later ones join the
            # block that follows them.
            stem = not positions
            positions.update((waiting, -1 if stem else owner[name]) for waiting in pending)
            pending.clear()
            positions[name] = owner[name]
        positions.update((waiting, len(blocks)) for waiting in pending)
        return positions, len(blocks)

    def freeze(self) -> None:
        """Disable gradients for every current parameter and hold the module in eval mode.

        While frozen, ``train()`` leaves this module in eval mode, so a parent module entering
        training cannot update batch-norm statistics or re-enable dropout here.
        """
        self._frozen = True
        self._partial = False
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def is_normalization(self, module: nn.Module) -> bool:
        """Whether ``module`` is a normalisation layer, for ``trainable_parameters="layernorm"``.

        The default recognises torch's batch, instance, layer, group, and RMS norm, and their
        subclasses. Adapters whose provider has normalisation layers of its own that subclass
        none of these extend it.
        """
        return isinstance(module, _NORMALIZATION_TYPES)

    def trainable_parameter_names(
        self, last_blocks: int | None = None, trainable_parameters: str = "all"
    ) -> list[str]:
        """Names of the parameters that ``unfreeze`` with the same arguments makes trainable.

        Args:
            last_blocks: ``None`` for the whole backbone, or a count in ``[1, num_blocks]``:
                only the last that many blocks and the tail (see ``parameter_blocks``).
            trainable_parameters: Which parameters within that part (ADR 0021): ``"all"``;
                ``"layernorm"``, the parameters that normalisation layers (``is_normalization``)
                hold themselves, the scale and shift of batch norm included; or ``"bias"``, every
                parameter named ``bias`` or ending in ``_bias`` (``in_proj_bias`` of
                ``nn.MultiheadAttention``, for example).

        Returns:
            Names as returned by ``named_parameters()``, in that order.

        Raises:
            ConfigurationError: If ``trainable_parameters`` is unknown, ``last_blocks`` is out
                of range or the backbone defines no blocks, or ``trainable_parameters`` other
                than ``"all"`` selects no parameter.
        """
        if trainable_parameters not in TRAINABLE_PARAMETERS:
            raise ConfigurationError(
                f"trainable_parameters must be one of {list(TRAINABLE_PARAMETERS)}, "
                f"got {trainable_parameters!r}."
            )
        names = [name for name, _ in self.named_parameters()]
        if last_blocks is not None:
            positions, num_blocks = self.block_positions()
            check_last_blocks(last_blocks, num_blocks, self.identifier)
            names = [name for name in names if positions[name] >= num_blocks - last_blocks]
        if trainable_parameters == "all":
            return names
        if trainable_parameters == "layernorm":
            # By module type: a parameter belongs to the layer that holds it directly.
            wanted = {
                id(parameter)
                for module in self.modules()
                if self.is_normalization(module)
                for parameter in module.parameters(recurse=False)
            }
            kept = {name for name, parameter in self.named_parameters() if id(parameter) in wanted}
        else:
            kept = {
                name
                for name in names
                if (local := name.rpartition(".")[2]) == "bias" or local.endswith("_bias")
            }
        names = [name for name in names if name in kept]
        if not names:
            within = "" if last_blocks is None else f" within its last {last_blocks} blocks"
            raise ConfigurationError(
                f"trainable_parameters={trainable_parameters!r} selects no parameter of backbone "
                f"'{self.identifier}'{within}."
            )
        return names

    def unfreeze(self, last_blocks: int | None = None, trainable_parameters: str = "all") -> None:
        """Re-enable gradients for every parameter, or only for a selection of them.

        Args:
            last_blocks: ``None`` considers every parameter. A count in ``[1, num_blocks]``
                considers only the last that many blocks and the tail (see
                ``parameter_blocks``); the rest is frozen.
            trainable_parameters: ``"all"`` makes every considered parameter trainable;
                ``"layernorm"`` or ``"bias"`` only the normalisation-layer or bias parameters
                among them (see ``trainable_parameter_names``, ADR 0021).

        While only part of the backbone trains, ``train()`` keeps every batch-norm layer without
        a trainable parameter in eval mode, so the frozen part's running statistics do not
        change.

        Raises:
            ConfigurationError: As ``trainable_parameter_names``. Nothing changes in that case.
        """
        trainable = set(self.trainable_parameter_names(last_blocks, trainable_parameters))
        partial = last_blocks is not None or trainable_parameters != "all"
        self._frozen, self._partial = False, partial
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in trainable)
        if partial:
            self.train(self.training)

    def train(self, mode: bool = True) -> Self:
        """Set training mode; a frozen backbone stays in eval mode regardless of ``mode``.

        Under partial unfreezing, batch-norm layers without a trainable parameter stay in eval
        mode too, however this method is reached (a parent module, or ``encode`` restoring
        the previous mode).
        """
        super().train(mode and not self._frozen)
        if mode and self._partial:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                    parameter.requires_grad for parameter in module.parameters()
                ):
                    module.eval()
        return self
