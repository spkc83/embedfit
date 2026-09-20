"""Adapter for ``timm`` image models used as feature extractors."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from embedfit.backbones._preprocess import (
    channel_stats,
    pool_tokens,
    resize_crop_normalize,
    validate_interpolation,
    validate_pooling,
    validate_size,
)
from embedfit.backbones.base import ImageBackbone
from embedfit.data.image import ImageLike, to_rgb_tensor
from embedfit.exceptions import BackboneProviderError, ConfigurationError

_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "weights.pt"


def _import_timm() -> Any:
    try:
        import timm
    except ImportError as exc:
        raise BackboneProviderError(
            "The 'timm' backbone provider requires the timm package. "
            "Install it with: pip install 'embedfit[timm]'"
        ) from exc
    return timm


class TimmBackbone(ImageBackbone):
    """A ``timm`` model built without its classifier (``num_classes=0``).

    Preprocessing follows the model's own ``timm`` data configuration: resize the shorter side
    to ``image_size / crop_pct`` with the configured interpolation, centre-crop to
    ``image_size``, and normalise with the configured mean and standard deviation.

    Args:
        name: ``timm`` model name, optionally with a pretrained tag, e.g.
            ``"convnext_tiny.fb_in22k_ft_in1k"``.
        pretrained: Download and load the pretrained weights. ``False`` builds a randomly
            initialised model and downloads nothing.
        pooling: ``"provider_default"`` uses the model's own pooled features. ``"cls"`` and
            ``"mean"`` pool the token sequence of transformer models (class token, or mean of
            the patch tokens) and are rejected for models without one.
        image_size: Side of the square input. Defaults to the model's configured size.
            Transformer models are rebuilt for this size (``img_size``); convolutional models
            accept any size.
        interpolation: Overrides the configured resize mode.
        mean: Overrides the configured normalisation mean.
        std: Overrides the configured normalisation standard deviation (with ``mean``).

    Raises:
        BackboneProviderError: If ``timm`` is not installed or the model name is unknown.
        ConfigurationError: If an option is invalid or the pooling does not suit the model.
    """

    def __init__(
        self,
        name: str,
        *,
        pretrained: bool = True,
        pooling: str = "provider_default",
        image_size: int | None = None,
        interpolation: str | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        timm = _import_timm()
        self._pooling = validate_pooling(pooling)
        if image_size is not None:
            validate_size(image_size, "image_size")
        if not timm.is_model(name.split(".")[0]):
            close = timm.list_models(f"*{name.split('.')[0][:6]}*")[:3]
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise BackboneProviderError(
                f"Unknown timm model '{name}'.{hint} See timm.list_models() for valid names."
            )
        options: dict[str, Any] = {"pretrained": pretrained, "num_classes": 0}
        try:
            sized = {"img_size": image_size} if image_size is not None else {}
            self.model = timm.create_model(name, **options, **sized)
        except TypeError:  # Convolutional models take no img_size: they accept any input size.
            self.model = timm.create_model(name, **options)

        data = timm.data.resolve_data_config({}, model=self.model)
        self.image_size: int = image_size if image_size is not None else int(data["input_size"][-1])
        self.resize_size: int = max(self.image_size, round(self.image_size / data["crop_pct"]))
        self.interpolation = validate_interpolation(interpolation or data["interpolation"])
        if mean is None and std is None:
            mean, std = data["mean"], data["std"]
        self.mean, self.std = channel_stats(mean, std)
        self.name = name
        self.pretrained = pretrained

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():  # One probe fixes the output width and validates the pooling.
            probe = self.forward_features(torch.zeros(1, 3, self.image_size, self.image_size))
        self.model.train(was_training)
        self._embedding_dim = int(probe.shape[1])

    @property
    def identifier(self) -> str:
        """``"timm/<name>"``."""
        return f"timm/{self.name}"

    @property
    def pooling(self) -> str:
        """The pooling policy: ``"provider_default"``, ``"cls"``, or ``"mean"``."""
        return self._pooling

    @property
    def embedding_dim(self) -> int:
        """Width of the pooled features."""
        return self._embedding_dim

    def preprocess(self, image: ImageLike) -> torch.Tensor:
        """Resize, centre-crop, and normalise one image as the model's data config specifies.

        Returns:
            A float32 CPU tensor shaped ``[3, image_size, image_size]``.

        Raises:
            ImageInputError: If the image is malformed.
        """
        return resize_crop_normalize(
            to_rgb_tensor(image),
            resize_size=self.resize_size,
            crop_size=self.image_size,
            interpolation=self.interpolation,
            mean=self.mean,
            std=self.std,
        )

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed a batch shaped ``[batch, 3, H, W]`` into ``[batch, embedding_dim]``.

        Raises:
            ImageInputError: If ``pixel_values`` is not a 4-D tensor with 3 channels.
            ConfigurationError: If token pooling is requested from a model without tokens.
        """
        self._check_pixel_values(pixel_values)
        if self._pooling == "provider_default":
            pooled: torch.Tensor = self.model(pixel_values)
            return pooled.flatten(start_dim=1)
        tokens = self.model.forward_features(pixel_values)
        prefix = int(getattr(self.model, "num_prefix_tokens", 0))
        return pool_tokens(tokens, self._pooling, prefix_tokens=prefix)

    def save_pretrained(self, path: str | Path) -> None:
        """Write ``config.json`` (name, pooling, preprocessing, dtype) and ``weights.pt``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "provider": "timm",
            "name": self.name,
            "pretrained": self.pretrained,
            "pooling": self._pooling,
            "image_size": self.image_size,
            "interpolation": self.interpolation,
            "mean": self.mean,
            "std": self.std,
            "embedding_dim": self._embedding_dim,
            "dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
        }
        (directory / _CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
        torch.save(self.state_dict(), directory / _WEIGHTS_FILE)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "TimmBackbone":
        """Load a backbone written by ``save_pretrained``. Downloads nothing.

        Raises:
            ConfigurationError: If files are missing, malformed, or do not match.
        """
        directory = Path(path)
        config_path, weights_path = directory / _CONFIG_FILE, directory / _WEIGHTS_FILE
        try:
            config = json.loads(config_path.read_text())
            if config["provider"] != "timm":
                raise KeyError("provider")
            backbone = cls(
                config["name"],
                pretrained=False,
                pooling=config["pooling"],
                image_size=config["image_size"],
                interpolation=config["interpolation"],
                mean=config["mean"],
                std=config["std"],
            )
            dtype = getattr(torch, str(config["dtype"]), None)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"'{directory}' is not a TimmBackbone directory written by save_pretrained."
            ) from exc
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ConfigurationError(f"'{config_path}' names an unsupported dtype.")
        backbone.pretrained = config["pretrained"]
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state)
        except Exception as exc:  # Unpickling, key, and shape errors all mean the same thing.
            raise ConfigurationError(
                f"'{weights_path}' is not a tensor-only state dict matching this backbone's "
                "configuration, as written by save_pretrained."
            ) from exc
        backbone.to(dtype)
        backbone.eval()
        return backbone
