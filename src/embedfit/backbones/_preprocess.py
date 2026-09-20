"""Resize, centre-crop, and normalise float image tensors without torchvision.

Adapters whose provider does not ship tensor transforms share these helpers; every constant
(sizes, interpolation, mean, std) still comes from the adapter's checkpoint (ADR 0003).
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from embedfit.exceptions import ConfigurationError

INTERPOLATIONS = ("nearest", "bilinear", "bicubic")
POOLINGS = ("provider_default", "cls", "mean")


def validate_interpolation(value: str) -> str:
    if value not in INTERPOLATIONS:
        raise ConfigurationError(
            f"interpolation must be one of {list(INTERPOLATIONS)}, got {value!r}."
        )
    return value


def validate_pooling(value: str) -> str:
    if value not in POOLINGS:
        raise ConfigurationError(f"pooling must be one of {list(POOLINGS)}, got {value!r}.")
    return value


def validate_size(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be a positive int, got {value!r}.")
    return value


def validate_resize(value: object) -> int | tuple[int, int]:
    """Accept an int (shorter side) or a ``(height, width)`` pair of positive ints."""
    if isinstance(value, list | tuple) and len(value) == 2:
        return (validate_size(value[0], "resize_size"), validate_size(value[1], "resize_size"))
    return validate_size(value, "resize_size")


def channel_stats(
    mean: Sequence[float] | None, std: Sequence[float] | None
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    """Validate per-channel normalisation constants; both or neither must be given."""
    if (mean is None) != (std is None):
        raise ConfigurationError("mean and std must be given together.")
    if mean is None or std is None:
        return None, None
    if len(mean) != 3 or len(std) != 3:
        raise ConfigurationError("mean and std must have 3 values (one per RGB channel).")
    if min(std) <= 0:
        raise ConfigurationError("std values must be positive.")
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def resize_crop_normalize(
    image: torch.Tensor,
    *,
    resize_size: int | tuple[int, int],
    crop_size: int,
    interpolation: str,
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
) -> torch.Tensor:
    """Preprocess one float ``[3, H, W]`` tensor in [0, 1].

    Args:
        image: The image, as produced by ``embedfit.data.to_rgb_tensor``.
        resize_size: An int resizes the shorter side to that length, keeping the aspect ratio;
            a ``(height, width)`` pair resizes to exactly that shape.
        crop_size: Side of the square centre crop taken afterwards. It must not exceed the
            resized image.
        interpolation: ``"nearest"``, ``"bilinear"``, or ``"bicubic"`` (the latter two are
            antialiased).
        mean: Per-channel mean, or ``None`` for no normalisation.
        std: Per-channel standard deviation, given together with ``mean``.

    Returns:
        A contiguous float32 tensor shaped ``[3, crop_size, crop_size]``.
    """
    _, height, width = image.shape
    if isinstance(resize_size, int):
        # Same arithmetic as torchvision, timm, and Transformers: the long side is truncated,
        # not rounded, so the result lines up with their pipelines to the pixel.
        if height <= width:
            target = (resize_size, int(resize_size * width / height))
        else:
            target = (int(resize_size * height / width), resize_size)
    else:
        target = resize_size
    if target != (height, width):
        smooth = interpolation != "nearest"
        image = F.interpolate(
            image.unsqueeze(0),
            size=target,
            mode=interpolation,
            antialias=smooth,
            **({"align_corners": False} if smooth else {}),
        ).squeeze(0)
        if interpolation == "bicubic":
            image = image.clamp(0.0, 1.0)  # Bicubic overshoots; 8-bit pipelines clip too.
    # torchvision's centre crop rounds half offsets to even, e.g. 62.5 -> 62.
    top = int(round((target[0] - crop_size) / 2.0))
    left = int(round((target[1] - crop_size) / 2.0))
    if top < 0 or left < 0:
        raise ConfigurationError(
            f"crop_size {crop_size} exceeds the resized image {target[0]}x{target[1]}."
        )
    image = image[:, top : top + crop_size, left : left + crop_size]
    if mean is not None and std is not None:
        image = (image - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    return image.contiguous()


def pool_tokens(tokens: torch.Tensor, pooling: str, *, prefix_tokens: int) -> torch.Tensor:
    """Pool a ``[B, T, D]`` token sequence to ``[B, D]``.

    ``"cls"`` takes the first token and needs at least one prefix (class) token; ``"mean"``
    averages the patch tokens, i.e. everything after the ``prefix_tokens`` leading tokens.
    """
    if tokens.ndim != 3:
        raise ConfigurationError(
            f"pooling='{pooling}' needs a token sequence [batch, tokens, dim], but this model "
            f"produces features shaped {tuple(tokens.shape)}; use pooling='provider_default'."
        )
    if pooling == "cls":
        if prefix_tokens < 1:
            raise ConfigurationError("pooling='cls' needs a model with a class token.")
        return tokens[:, 0]
    return tokens[:, prefix_tokens:].mean(dim=1)
