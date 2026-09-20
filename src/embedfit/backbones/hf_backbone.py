"""Adapter for Hugging Face Transformers vision encoders."""

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
    validate_resize,
    validate_size,
)
from embedfit.backbones.base import ImageBackbone
from embedfit.data.image import ImageLike, to_rgb_tensor
from embedfit.exceptions import BackboneProviderError, ConfigurationError, EmbedFitError

_CONFIG_FILE = "config.json"
_MODEL_DIRECTORY = "model"
# PIL resampling codes used by image processors: 0 nearest, 2 bilinear, 3 bicubic.
_RESAMPLE = {0: "nearest", 2: "bilinear", 3: "bicubic"}


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise BackboneProviderError(
            "The 'hf' backbone provider requires the transformers package. "
            "Install it with: pip install 'embedfit[hf]'"
        ) from exc
    return transformers


def _unsupported(setting: str) -> BackboneProviderError:
    return BackboneProviderError(
        f"The checkpoint's image processor uses {setting}, which this adapter does not "
        "reproduce. Pass image_size, resize_size, interpolation, mean, and std explicitly to "
        "define the preprocessing yourself."
    )


def _square(size: dict[str, Any], what: str) -> int:
    if int(size["height"]) != int(size["width"]):
        raise _unsupported(f"a non-square {what}")
    return int(size["height"])


def _processor_settings(processor: Any) -> dict[str, Any]:
    """Translate an image processor's settings into this adapter's preprocessing options.

    Understood: ``size`` (``shortest_edge`` or a square ``height``/``width``), a square
    ``crop_size`` with ``do_center_crop``, ConvNeXt-style ``crop_pct``, ``resample``
    (nearest, bilinear, bicubic), ``do_normalize`` with ``image_mean``/``image_std``, and
    rescaling by 1/255. Anything else raises instead of being silently ignored.
    """
    if getattr(processor, "do_resize", True) is False:
        raise _unsupported("do_resize=False")
    factor = getattr(processor, "rescale_factor", 1 / 255)
    if getattr(processor, "do_rescale", True) is False or abs(float(factor) - 1 / 255) > 1e-9:
        raise _unsupported("a pixel rescaling other than 1/255")
    resample = getattr(processor, "resample", 2)
    code = int(getattr(resample, "value", resample))
    if code not in _RESAMPLE:
        raise _unsupported(f"resampling filter {code}")

    size = {k: v for k, v in dict(getattr(processor, "size", None) or {}).items() if v is not None}
    crop = {
        k: v for k, v in dict(getattr(processor, "crop_size", None) or {}).items() if v is not None
    }
    do_crop = bool(getattr(processor, "do_center_crop", False)) and bool(crop)
    crop_pct = getattr(processor, "crop_pct", None)
    resize: int | tuple[int, int]
    if crop_pct is not None and "shortest_edge" in size:
        # ConvNeXt: below 384, resize the shorter side to edge / crop_pct and centre-crop;
        # from 384 up, squash to a square without cropping.
        edge = int(size["shortest_edge"])
        resize = int(edge / float(crop_pct)) if edge < 384 else (edge, edge)
        side = edge
    elif "shortest_edge" in size and set(size) == {"shortest_edge"}:
        resize = int(size["shortest_edge"])
        side = _square(crop, "crop_size") if do_crop else resize
    elif set(size) == {"height", "width"}:
        edge = _square(size, "size")
        resize = (edge, edge)
        side = _square(crop, "crop_size") if do_crop else edge
    else:
        raise _unsupported(f"a size given as {sorted(size)}")
    normalise = bool(getattr(processor, "do_normalize", True))
    return {
        "resize_size": resize,
        "image_size": side,
        "interpolation": _RESAMPLE[code],
        "mean": getattr(processor, "image_mean", None) if normalise else None,
        "std": getattr(processor, "image_std", None) if normalise else None,
    }


class HFBackbone(ImageBackbone):
    """A Transformers vision encoder loaded with ``AutoModel``.

    Args:
        repo_id: Hub repository, e.g. ``"facebook/dinov2-base"``, or a local directory in
            Transformers format. With ``config`` it is only the name recorded in
            ``identifier``.
        pooling: ``"provider_default"`` returns the model's ``pooler_output`` when it has one
            and its class token otherwise. ``"cls"`` always returns the class token, and
            ``"mean"`` the mean of the patch tokens, excluding the class token and any register
            tokens (``config.num_register_tokens``). Token pooling is rejected for models that
            produce feature maps instead of tokens.
        config: A ``PretrainedConfig``. When given, a randomly initialised model is built from
            it and nothing is downloaded; ``image_size`` then defaults to
            ``config.image_size`` and no normalisation is applied unless ``mean`` and ``std``
            are given.
        image_size: Side of the square input. Defaults to the checkpoint's image processor.
            Most transformer encoders only accept the size they were built for; a size the
            model rejects raises when the backbone is created.
        resize_size: Resize before cropping: an int for the shorter side, or
            ``(height, width)``. Defaults to the image processor's setting. When only
            ``image_size`` is given, the processor's resize-to-crop ratio is kept, so passing
            the checkpoint's own size changes nothing.
        interpolation: Resize mode; defaults to the image processor's, or ``"bilinear"``.
        mean: Normalisation mean; defaults to the image processor's.
        std: Normalisation standard deviation (with ``mean``).

    Raises:
        BackboneProviderError: If ``transformers`` is not installed, the checkpoint cannot be
            loaded, or its image processor uses a setting this adapter does not reproduce.
        ConfigurationError: If an option is invalid or the pooling does not suit the model.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        pooling: str = "provider_default",
        config: Any = None,
        image_size: int | None = None,
        resize_size: int | tuple[int, int] | None = None,
        interpolation: str | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        _model: Any = None,
    ) -> None:
        super().__init__()
        transformers = _import_transformers()
        self._pooling = validate_pooling(pooling)
        if image_size is not None:
            validate_size(image_size, "image_size")

        if not isinstance(repo_id, str) or not repo_id:
            raise ConfigurationError("repo_id must be a non-empty str.")
        settings: dict[str, Any] = {}
        if _model is not None:  # from_pretrained: weights are already loaded.
            self.model = _model
            configured = getattr(_model.config, "image_size", None)
        elif config is not None:
            self.model = transformers.AutoModel.from_config(config)
            configured = getattr(config, "image_size", None)
        else:
            try:
                self.model = transformers.AutoModel.from_pretrained(repo_id)
                processor = transformers.AutoImageProcessor.from_pretrained(repo_id)
            except (OSError, ValueError) as exc:
                raise BackboneProviderError(
                    f"Could not load Hugging Face checkpoint '{repo_id}' and its image "
                    "processor. Check the repository id, network access, and that it is a "
                    "vision encoder."
                ) from exc
            settings = _processor_settings(processor)
            configured = settings["image_size"]

        explicit_size = image_size is not None
        if image_size is None:
            if not isinstance(configured, int) or isinstance(configured, bool):
                raise ConfigurationError(
                    "The model configuration has no integer image_size; pass image_size."
                )
            image_size = configured
        self.image_size: int = image_size
        default_resize = settings.get("resize_size", self.image_size)
        if explicit_size and resize_size is None and settings:
            # Keep the processor's resize-to-crop ratio at the requested size.
            ratio = self.image_size / settings["image_size"]
            if isinstance(default_resize, tuple):
                default_resize = (int(default_resize[0] * ratio), int(default_resize[1] * ratio))
            else:
                default_resize = int(default_resize * ratio)
        self.resize_size: int | tuple[int, int] = validate_resize(
            resize_size if resize_size is not None else default_resize
        )
        self.interpolation = validate_interpolation(
            interpolation or settings.get("interpolation", "bilinear")
        )
        if mean is None and std is None:
            mean, std = settings.get("mean"), settings.get("std")
        self.mean, self.std = channel_stats(mean, std)
        self.repo_id = repo_id

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():  # One probe fixes the output width and validates the pooling.
                probe = self.forward_features(torch.zeros(1, 3, self.image_size, self.image_size))
        except EmbedFitError:
            raise  # Already actionable, e.g. a pooling policy the model cannot serve.
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise ConfigurationError(
                f"Hugging Face model '{repo_id}' does not accept "
                f"{self.image_size}x{self.image_size} input."
            ) from exc
        finally:
            self.model.train(was_training)
        self._embedding_dim = int(probe.shape[1])

    @property
    def identifier(self) -> str:
        """``"hf/<repo_id>"``."""
        return f"hf/{self.repo_id}"

    @property
    def pooling(self) -> str:
        """The pooling policy: ``"provider_default"``, ``"cls"``, or ``"mean"``."""
        return self._pooling

    @property
    def embedding_dim(self) -> int:
        """Width of the pooled features."""
        return self._embedding_dim

    def preprocess(self, image: ImageLike) -> torch.Tensor:
        """Resize, centre-crop, and normalise one image as the checkpoint's processor would.

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
        outputs = self.model(pixel_values=pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if self._pooling == "provider_default" and pooled is not None:
            features: torch.Tensor = pooled
            return features.flatten(start_dim=1)
        tokens = outputs.last_hidden_state
        if self._pooling == "provider_default" and tokens.ndim != 3:
            raise ConfigurationError(
                "This checkpoint returns neither a pooler_output nor a token sequence, so it has "
                "no default pooled embedding this adapter can use."
            )
        has_class_token = hasattr(getattr(self.model, "embeddings", None), "cls_token")
        registers = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        policy = "cls" if self._pooling == "provider_default" else self._pooling
        return pool_tokens(tokens, policy, prefix_tokens=int(has_class_token) + registers)

    def save_pretrained(self, path: str | Path) -> None:
        """Write ``config.json`` (pooling and preprocessing) and the model in ``model/``.

        The model is stored in Transformers' own format as safetensors, so nothing is pickled.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "provider": "hf",
            "repo_id": self.repo_id,
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
        self.model.save_pretrained(directory / _MODEL_DIRECTORY, safe_serialization=True)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "HFBackbone":
        """Load a backbone written by ``save_pretrained``. Downloads nothing.

        Raises:
            ConfigurationError: If files are missing, malformed, or do not match.
        """
        transformers = _import_transformers()
        directory = Path(path)
        try:
            config = json.loads((directory / _CONFIG_FILE).read_text())
            if config["provider"] != "hf":
                raise KeyError("provider")
            dtype = getattr(torch, str(config["dtype"]), None)
            if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
                raise KeyError("dtype")
            model = transformers.AutoModel.from_pretrained(
                directory / _MODEL_DIRECTORY, local_files_only=True, use_safetensors=True
            )
            resize = config["resize_size"]
            backbone = cls(
                config["repo_id"],
                pooling=config["pooling"],
                image_size=config["image_size"],
                resize_size=tuple(resize) if isinstance(resize, list) else resize,
                interpolation=config["interpolation"],
                mean=config["mean"],
                std=config["std"],
                _model=model,
            )
            if backbone.embedding_dim != config["embedding_dim"]:
                raise ValueError("embedding_dim")
        except Exception as exc:  # JSON, key, type, safetensors, and Transformers errors alike.
            raise ConfigurationError(
                f"'{directory}' is not an HFBackbone directory written by save_pretrained."
            ) from exc
        backbone.to(dtype)
        backbone.eval()
        return backbone
