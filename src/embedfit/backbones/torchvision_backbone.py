"""Adapter for torchvision classification models used as feature extractors."""

import difflib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from embedfit.backbones.base import ImageBackbone
from embedfit.data.image import ImageLike, to_rgb_tensor
from embedfit.exceptions import BackboneProviderError, ConfigurationError

_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "weights.pt"
# Attribute holding the classification layer, by torchvision architecture family:
# fc (ResNet, RegNet, Inception, ShuffleNet), classifier (ConvNeXt, EfficientNet, MobileNet,
# DenseNet, VGG), heads (VisionTransformer), head (Swin).
_HEAD_ATTRIBUTES = ("fc", "classifier", "heads", "head")
_DEFAULT_IMAGE_SIZE = 224
_DEFAULT_INTERPOLATION = "bilinear"


def _import_torchvision() -> Any:
    try:
        import torchvision
    except ImportError as exc:
        raise BackboneProviderError(
            "The 'torchvision' backbone provider requires the torchvision package. "
            "Install it with: pip install 'embedfit[torchvision]'"
        ) from exc
    return torchvision


def _remove_classifier(model: nn.Module, name: str) -> int:
    """Replace the final linear classifier with identity; return the feature dimension."""
    for attribute in _HEAD_ATTRIBUTES:
        head = getattr(model, attribute, None)
        if isinstance(head, nn.Linear):
            setattr(model, attribute, nn.Identity())
            return head.in_features
        if isinstance(head, nn.Sequential):
            linear_positions = [i for i, layer in enumerate(head) if isinstance(layer, nn.Linear)]
            if linear_positions:
                last_linear = head[linear_positions[-1]]
                assert isinstance(last_linear, nn.Linear)
                head[linear_positions[-1]] = nn.Identity()
                return last_linear.in_features
    raise BackboneProviderError(
        f"torchvision model '{name}' has no linear classification layer under any of "
        f"{list(_HEAD_ATTRIBUTES)}, so its embedding output cannot be located. "
        "Wrap it in a custom ImageBackbone instead."
    )


def _as_triple(values: Sequence[float] | None, option: str) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ConfigurationError(f"{option} must have 3 values (one per RGB channel).")
    return (float(values[0]), float(values[1]), float(values[2]))


class TorchvisionBackbone(ImageBackbone):
    """A torchvision classification model with its classifier removed.

    The embedding is the input to the model's final linear layer, e.g. the 512-d pooled
    features of ``resnet18`` or the class-token output of ``vit_b_16``.

    Preprocessing follows the selected weights' own torchvision preset: resize the shorter
    side to ``resize_size`` with the preset's interpolation mode (antialiased), centre-crop to
    ``image_size``, then normalise with the preset's mean and standard deviation. No
    normalisation constants are assumed when ``weights`` is ``None``. Resizing operates on the
    float tensor rather than on 8-bit pixels, so values differ slightly from
    ``weights.transforms()``: about 0.001 of the intensity range on average, and up to a few
    percent at hard edges under bicubic interpolation.

    Auxiliary classifiers (GoogLeNet, Inception v3) are removed, so the output is a single
    tensor in both train and eval mode. Architectures whose classifier is not a linear layer
    (e.g. SqueezeNet) are rejected.

    Args:
        name: torchvision model name, e.g. ``"resnet18"`` (see
            ``torchvision.models.list_models()``).
        weights: Name of a torchvision weights enum member such as ``"IMAGENET1K_V1"``, or
            ``"DEFAULT"`` for the model's recommended weights. Pretrained weights are downloaded
            on first use. ``None`` builds a randomly initialised model and downloads nothing.
        image_size: Side length of the square centre crop fed to the model. Defaults to the
            weights preset's crop size, or 224 without weights. When given together with
            weights, ``resize_size`` keeps the preset's resize-to-crop ratio. Not every
            architecture accepts every resolution: fixed-resolution models such as ViT reject
            any size other than their own.
        resize_size: Shorter-side length before cropping; overrides the rule above.
        interpolation: Resize mode, ``"nearest"``, ``"bilinear"``, or ``"bicubic"``; overrides
            the preset. Defaults to the preset's mode, or ``"bilinear"`` without weights.
        mean: Per-channel normalisation mean; overrides the preset. With ``weights=None`` and
            no ``mean``/``std``, pixel values stay in [0, 1].
        std: Per-channel normalisation standard deviation; must be given together with ``mean``.

    Raises:
        BackboneProviderError: If torchvision is not installed, the model or weights name is
            unknown, or the architecture has no linear classifier to remove.
        ConfigurationError: If a preprocessing option is invalid.
    """

    def __init__(
        self,
        name: str,
        *,
        weights: str | None = "DEFAULT",
        image_size: int | None = None,
        resize_size: int | None = None,
        interpolation: str | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        torchvision = _import_torchvision()
        models = torchvision.models

        available = models.list_models(module=models)
        if name not in available:
            suggestions = difflib.get_close_matches(name, available, n=3)
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise BackboneProviderError(
                f"Unknown torchvision model '{name}'.{hint} "
                "See torchvision.models.list_models() for valid names."
            )

        preset = None
        if weights is not None:
            weights_enum = models.get_model_weights(name)
            try:
                preset = weights_enum[weights].transforms()
            except KeyError:
                raise BackboneProviderError(
                    f"Unknown weights '{weights}' for torchvision model '{name}'. "
                    f"Available: {[member.name for member in weights_enum]} or 'DEFAULT'."
                ) from None
            self.model = models.get_model(name, weights=weights_enum[weights])
        else:
            self.model = models.get_model(name, weights=None)

        for option, value in (("image_size", image_size), ("resize_size", resize_size)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ConfigurationError(f"{option} must be an int, got {type(value).__name__}.")
            if value is not None and value < 1:
                raise ConfigurationError(f"{option} must be positive, got {value}.")
        if (mean is None) != (std is None):
            raise ConfigurationError("mean and std must be given together.")

        preset_crop = preset.crop_size[0] if preset is not None else _DEFAULT_IMAGE_SIZE
        preset_resize = preset.resize_size[0] if preset is not None else _DEFAULT_IMAGE_SIZE
        self.image_size: int = image_size if image_size is not None else preset_crop
        if resize_size is not None:
            self.resize_size: int = resize_size
        else:
            self.resize_size = round(self.image_size * preset_resize / preset_crop)
        if self.resize_size < self.image_size:
            raise ConfigurationError(
                f"resize_size ({self.resize_size}) must be >= image_size ({self.image_size})."
            )
        if interpolation is None:
            interpolation = (
                preset.interpolation.value if preset is not None else _DEFAULT_INTERPOLATION
            )
        if interpolation not in ("nearest", "bilinear", "bicubic"):
            raise ConfigurationError(
                f"interpolation must be 'nearest', 'bilinear', or 'bicubic', got {interpolation!r}."
            )
        self.interpolation: str = interpolation
        if mean is None and preset is not None:
            mean, std = preset.mean, preset.std
        self.mean = _as_triple(mean, "mean")
        self.std = _as_triple(std, "std")
        if self.std is not None and min(self.std) <= 0:
            raise ConfigurationError("std values must be positive.")

        # Auxiliary classifiers make train-mode output a tuple and are absent from some
        # pretrained state dicts; removing them gives one output type and one key set.
        if hasattr(self.model, "aux_logits"):
            self.model.aux_logits = False
            for auxiliary in ("aux1", "aux2", "AuxLogits"):
                if hasattr(self.model, auxiliary):
                    setattr(self.model, auxiliary, None)

        native_size = getattr(self.model, "image_size", None)
        if isinstance(native_size, int) and native_size != self.image_size:
            raise ConfigurationError(
                f"torchvision model '{name}' only accepts {native_size}x{native_size} input; "
                f"got image_size={self.image_size}."
            )

        self.name = name
        self.weights = weights
        self._embedding_dim = _remove_classifier(self.model, name)

    @property
    def identifier(self) -> str:
        """``"torchvision/<name>"``."""
        return f"torchvision/{self.name}"

    @property
    def embedding_dim(self) -> int:
        """Input width of the removed classifier layer."""
        return self._embedding_dim

    def preprocess(self, image: ImageLike) -> torch.Tensor:
        """Resize, centre-crop, and normalise one image for this checkpoint.

        Args:
            image: Any form accepted by ``embedfit.data.to_rgb_tensor``.

        Returns:
            A float32 CPU tensor shaped ``[3, image_size, image_size]``.

        Raises:
            ImageInputError: If the image is malformed.
        """
        from torchvision.transforms.v2 import functional as F

        tensor = to_rgb_tensor(image)
        tensor = F.resize(
            tensor,
            [self.resize_size],
            interpolation=F.InterpolationMode(self.interpolation),
            antialias=True,
        )
        if self.interpolation == "bicubic":
            tensor = tensor.clamp(0.0, 1.0)  # Bicubic overshoots; 8-bit pipelines clip too.
        tensor = F.center_crop(tensor, [self.image_size, self.image_size])
        if self.mean is not None and self.std is not None:
            tensor = F.normalize(tensor, mean=list(self.mean), std=list(self.std))
        result: torch.Tensor = tensor.contiguous()
        return result

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed a batch of preprocessed images.

        Args:
            pixel_values: Float tensor shaped ``[batch, 3, H, W]`` from stacked ``preprocess``
                outputs, on the backbone's device.

        Returns:
            Float tensor shaped ``[batch, embedding_dim]``. Not L2-normalised.

        Raises:
            ImageInputError: If ``pixel_values`` is not a 4-D tensor with 3 channels.
        """
        self._check_pixel_values(pixel_values)
        features: torch.Tensor = self.model(pixel_values)
        return features.flatten(start_dim=1)

    def parameter_blocks(self) -> list[list[str]]:
        """The stages or layers of the supported architecture families.

        ResNet, ResNeXt, and Wide ResNet: ``layer1`` to ``layer4``. Vision Transformer: the
        encoder layers. RegNet: the trunk stages. ConvNeXt and Swin: the stages of
        ``features``, each with the downsampling layer before it, so the counts match timm's
        for the same architectures. EfficientNet: the stages between the stem and the final
        convolution. Other architectures define no blocks.
        """
        models = _import_torchvision().models
        model = self.model
        stages: list[list[nn.Module]]
        if isinstance(model, models.ResNet):
            stages = [[model.layer1], [model.layer2], [model.layer3], [model.layer4]]
        elif isinstance(model, models.VisionTransformer):
            stages = [[layer] for layer in model.encoder.layers]
        elif isinstance(model, models.RegNet):
            stages = [[stage] for stage in model.trunk_output]
        elif isinstance(model, models.ConvNeXt | models.SwinTransformer):
            # features: stem, stage, downsampling, stage, downsampling, stage, ...
            layers = list(model.features)[1:]
            stages = [layers[:1]] + [layers[i : i + 2] for i in range(1, len(layers), 2)]
        elif isinstance(model, models.EfficientNet):
            stages = [[stage] for stage in list(model.features)[1:-1]]
        else:
            return []
        return [self._names_of(*modules) for modules in stages]

    def save_pretrained(self, path: str | Path) -> None:
        """Write the configuration and current weights to a directory.

        Args:
            path: Target directory; created if missing. Receives ``config.json`` (model name,
                source weights name, preprocessing settings, embedding dimension) and
                ``weights.pt`` (the state dict, including any fine-tuning).
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "provider": "torchvision",
            "name": self.name,
            "weights": self.weights,
            "image_size": self.image_size,
            "resize_size": self.resize_size,
            "interpolation": self.interpolation,
            "transform_input": getattr(self.model, "transform_input", None),
            "mean": self.mean,
            "std": self.std,
            "embedding_dim": self._embedding_dim,
            "dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
        }
        (directory / _CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
        torch.save(self.state_dict(), directory / _WEIGHTS_FILE)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "TorchvisionBackbone":
        """Load a backbone previously written by ``save_pretrained``. Downloads nothing.

        Args:
            path: Directory containing ``config.json`` and ``weights.pt``.

        Returns:
            A backbone with identical preprocessing and weights, in eval mode.

        Raises:
            ConfigurationError: If the directory is missing either file, or the configuration
                is malformed, incomplete, or was not written by ``TorchvisionBackbone``.
        """
        directory = Path(path)
        config_path, weights_path = directory / _CONFIG_FILE, directory / _WEIGHTS_FILE
        if not config_path.is_file() or not weights_path.is_file():
            raise ConfigurationError(
                f"'{directory}' is not a TorchvisionBackbone directory: "
                f"expected {_CONFIG_FILE} and {_WEIGHTS_FILE}."
            )
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"'{config_path}' is not valid JSON.") from exc
        if not isinstance(config, dict) or config.get("provider") != "torchvision":
            raise ConfigurationError(f"'{config_path}' was not written by TorchvisionBackbone.")
        required = {
            "name",
            "weights",
            "image_size",
            "resize_size",
            "interpolation",
            "transform_input",
            "mean",
            "std",
            "embedding_dim",
        }
        missing = sorted(required - config.keys())
        if missing:
            raise ConfigurationError(f"'{config_path}' is missing required keys: {missing}.")
        backbone = cls(
            config["name"],
            weights=None,
            image_size=config["image_size"],
            resize_size=config["resize_size"],
            interpolation=config["interpolation"],
            mean=config["mean"],
            std=config["std"],
        )
        backbone.weights = config["weights"]
        if config["transform_input"] is not None:
            # torchvision enables this input rescaling only when built from pretrained weights.
            backbone.model.transform_input = config["transform_input"]
        if backbone.embedding_dim != config["embedding_dim"]:
            raise ConfigurationError(
                f"'{config_path}' records embedding_dim={config['embedding_dim']} but "
                f"'{config['name']}' produces {backbone.embedding_dim}."
            )
        dtype = getattr(torch, str(config.get("dtype", "float32")), None)
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
