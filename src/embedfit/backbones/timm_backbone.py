"""Adapter for ``timm`` image models used as feature extractors."""

import contextlib
import difflib
import functools
import importlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from embedfit.backbones._preprocess import (
    channel_stats,
    pool_tokens,
    resize_crop_normalize,
    validate_interpolation,
    validate_pooling,
    validate_resize,
    validate_size,
)
from embedfit.backbones.base import ImageBackbone
from embedfit.data.image import ImageLike, to_rgb_tensor
from embedfit.exceptions import BackboneProviderError, ConfigurationError, EmbedFitError

_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "weights.pt"
# timm normalisation layers that subclass no torch norm; looked up by name because the set
# differs between timm versions.
_TIMM_NORMALIZATION_NAMES = (
    "RmsNorm",
    "RmsNorm2d",
    "RmsNormFp32",
    "RmsNorm2dFp32",
    "SimpleNorm",
    "SimpleNorm2d",
    "SimpleNormFp32",
    "SimpleNorm2dFp32",
    "EvoNorm2dB0",
    "EvoNorm2dB1",
    "EvoNorm2dB2",
    "EvoNorm2dS0",
    "EvoNorm2dS1",
    "EvoNorm2dS2",
    "FilterResponseNormAct2d",
    "FilterResponseNormTlu2d",
    "GlobalResponseNorm",
)


def _import_timm() -> Any:
    try:
        import timm
    except ImportError as exc:
        raise BackboneProviderError(
            "The 'timm' backbone provider requires the timm package. "
            "Install it with: pip install 'embedfit[timm]'"
        ) from exc
    return timm


@functools.cache
def _timm_normalization_types() -> tuple[type[nn.Module], ...]:
    layers = _import_timm().layers
    # timm.layers does not re-export every class: GlobalResponseNorm lives only in its module.
    modules = [layers]
    with contextlib.suppress(ImportError):
        modules.append(importlib.import_module("timm.layers.grn"))
    return tuple(
        {
            getattr(m, name)
            for name in _TIMM_NORMALIZATION_NAMES
            for m in modules
            if hasattr(m, name)
        }
    )


class TimmBackbone(ImageBackbone):
    """A ``timm`` model built without its classifier (``num_classes=0``).

    Preprocessing follows the model's own ``timm`` data configuration, with the same arithmetic
    as ``timm.data.create_transform``: resize to ``floor(image_size / crop_pct)`` (the shorter
    side for ``crop_mode="center"``, both sides for ``"squash"``) with the configured
    interpolation, centre-crop to ``image_size``, and normalise with the configured mean and
    standard deviation. ``crop_mode="border"`` is not supported and raises.

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
        resize_size: Overrides the resize step: an int for the shorter side, or
            ``(height, width)``.
        interpolation: Overrides the configured resize mode.
        mean: Overrides the configured normalisation mean.
        std: Overrides the configured normalisation standard deviation (with ``mean``).

    Raises:
        BackboneProviderError: If ``timm`` is not installed, the model name or pretrained tag
            is unknown, or the model's data configuration uses an unsupported ``crop_mode``.
        ConfigurationError: If an option is invalid, the pooling does not suit the model, or
            the model does not accept ``image_size``.
    """

    def __init__(
        self,
        name: str,
        *,
        pretrained: bool = True,
        pooling: str = "provider_default",
        image_size: int | None = None,
        resize_size: int | tuple[int, int] | None = None,
        interpolation: str | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        timm = _import_timm()
        self._pooling = validate_pooling(pooling)
        if image_size is not None:
            validate_size(image_size, "image_size")
        if not isinstance(name, str) or not timm.is_model(name.split(".")[0]):
            close = difflib.get_close_matches(str(name).split(".")[0], timm.list_models(), n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise BackboneProviderError(
                f"Unknown timm model '{name}'.{hint} See timm.list_models() for valid names."
            )
        options: dict[str, Any] = {"pretrained": pretrained, "num_classes": 0}
        try:
            try:
                sized = {"img_size": image_size} if image_size is not None else {}
                self.model = timm.create_model(name, **options, **sized)
            except TypeError as exc:
                if "img_size" not in str(exc):
                    raise
                # Convolutional models take no img_size: they accept any input size.
                self.model = timm.create_model(name, **options)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise BackboneProviderError(
                f"timm could not build '{name}'; check the model name and its pretrained tag."
            ) from exc

        data = timm.data.resolve_data_config({}, model=self.model)
        self.image_size: int = image_size if image_size is not None else int(data["input_size"][-1])
        if resize_size is None:
            # timm floors here; rounding would shift the centre crop by a pixel for crop_pct 0.9.
            scaled = max(self.image_size, math.floor(self.image_size / data["crop_pct"]))
            crop_mode = data.get("crop_mode", "center")
            if crop_mode == "squash":
                resize_size = (scaled, scaled)
            elif crop_mode == "center":
                resize_size = scaled
            else:
                raise BackboneProviderError(
                    f"timm model '{name}' uses crop_mode '{crop_mode}', which is not supported; "
                    "pass resize_size explicitly to choose the resize yourself."
                )
        self.resize_size: int | tuple[int, int] = validate_resize(resize_size)
        self.interpolation = validate_interpolation(interpolation or data["interpolation"])
        if mean is None and std is None:
            mean, std = data["mean"], data["std"]
        self.mean, self.std = channel_stats(mean, std)
        self.name = name
        self.pretrained = pretrained

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():  # One probe fixes the output width and validates the pooling.
                probe = self.forward_features(torch.zeros(1, 3, self.image_size, self.image_size))
        except EmbedFitError:
            raise  # Already actionable, e.g. a pooling policy the model cannot serve.
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise ConfigurationError(
                f"timm model '{name}' does not accept {self.image_size}x{self.image_size} input."
            ) from exc
        finally:
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
        prefix = getattr(self.model, "num_prefix_tokens", None)
        if tokens.ndim == 3 and not isinstance(prefix, int):
            # Guessing 0 would average a class token into "mean" and refuse a valid "cls".
            raise ConfigurationError(
                f"timm model '{self.name}' does not declare num_prefix_tokens, so its class and "
                "patch tokens cannot be told apart; use pooling='provider_default'."
            )
        return pool_tokens(tokens, self._pooling, prefix_tokens=int(prefix or 0))

    def parameter_blocks(self) -> list[list[str]]:
        """The blocks of timm's own coarse parameter grouping (``group_matcher(coarse=True)``).

        Each parameter is matched as timm matches it (``re.match`` on its name inside the timm
        model, the ``stem`` pattern first, then the ``blocks`` patterns in order), and the
        first number a matching pattern captures is its block. Parameters in the stem, or
        matched by a pattern that captures no number (timm attaches a final norm to the
        previous group this way), are in no block. Models without such a grouping define no
        blocks.
        """
        matcher = getattr(self.model, "group_matcher", None)
        groups = matcher(coarse=True) if callable(matcher) else None
        if not isinstance(groups, dict):
            return []
        rules = groups.get("blocks", [])
        patterns = [rules] if isinstance(rules, str) else [rule[0] for rule in rules]
        stem = groups.get("stem")
        blocks: dict[int, list[str]] = {}
        for name, _ in self.model.named_parameters():
            if isinstance(stem, str) and re.match(stem, name):
                continue
            match = next((m for p in patterns if (m := re.match(p, name))), None)
            if match is not None and match.groups() and match.group(1) is not None:
                blocks.setdefault(int(match.group(1)), []).append(f"model.{name}")
        return [blocks[number] for number in sorted(blocks)]

    def is_normalization(self, module: nn.Module) -> bool:
        """Torch's normalisation layers, and timm's own RMS, simple, EvoNorm, and filter-response
        norms, which subclass none of them (timm's layer, group, and batch norms do)."""
        return super().is_normalization(module) or isinstance(module, _timm_normalization_types())

    def save_pretrained(self, path: str | Path) -> None:
        """Write ``config.json`` (name, pooling, preprocessing, dtype) and ``weights.pt``.

        Every preprocessing value is written, including the resize size, so a later ``timm``
        release that changes a model's defaults cannot change a saved backbone.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "provider": "timm",
            "name": self.name,
            "pretrained": self.pretrained,
            "pooling": self._pooling,
            "image_size": self.image_size,
            "resize_size": self.resize_size,
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
            if config["provider"] != "timm" or not isinstance(config["pretrained"], bool):
                raise KeyError("provider")
            resize = config["resize_size"]
            backbone = cls(
                config["name"],
                pretrained=False,
                pooling=config["pooling"],
                image_size=config["image_size"],
                resize_size=tuple(resize) if isinstance(resize, list) else resize,
                interpolation=config["interpolation"],
                mean=config["mean"],
                std=config["std"],
            )
            dtype = getattr(torch, str(config["dtype"]), None)
            if backbone.embedding_dim != config["embedding_dim"]:
                raise ValueError("embedding_dim")
            backbone.pretrained = config["pretrained"]
        except (OSError, KeyError, TypeError, ValueError, AttributeError, EmbedFitError) as exc:
            raise ConfigurationError(
                f"'{directory}' is not a TimmBackbone directory written by save_pretrained."
            ) from exc
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ConfigurationError(f"'{config_path}' names an unsupported dtype.")
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
