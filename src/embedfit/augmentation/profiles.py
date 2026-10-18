"""Built-in augmentation profiles, in plain torch so the core needs no torchvision.

Both act on the canonical image tensor (float32 ``[3, H, W]`` in [0, 1] at native resolution)
before the backbone's own resizing and normalisation, and draw every random number from torch's
global generator (see ``embedfit.augmentation.pipeline``).
"""

import math
import numbers

import numpy as np
import torch
import torch.nn.functional as F

from embedfit.exceptions import ConfigurationError

_LUMA = (0.299, 0.587, 0.114)  # ITU-R 601 weights, as torchvision's colour transforms use.
_CROP_ATTEMPTS = 10


def _number(value: object, name: str, low: float, high: float) -> float:
    """``value`` as a float, if it is a real number (NumPy scalars included, bool not)."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        number = math.nan
    else:
        number = float(value)
    if not (math.isfinite(number) and low <= number <= high):
        raise ConfigurationError(f"{name} must be a number in [{low}, {high}], got {value!r}.")
    return number


def _interval(value: object, name: str, high: float) -> tuple[float, float]:
    """A ``(low, upper)`` pair with ``0 < low <= upper <= high``."""
    if not isinstance(value, tuple | list) or len(value) != 2:
        raise ConfigurationError(f"{name} must be a (low, high) pair, got {value!r}.")
    low, upper = (_number(v, name, 0.0, high) for v in value)
    if low <= 0 or low > upper:
        raise ConfigurationError(f"{name} must satisfy 0 < low <= high <= {high}, got {value!r}.")
    return low, upper


def _uniform(low: float, high: float) -> float:
    return low + (high - low) * torch.rand(()).item()


def _grayscale(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor(_LUMA, dtype=image.dtype)
    return torch.einsum("c,chw->hw", weights, image).unsqueeze(0)


def _blend(image: torch.Tensor, other: torch.Tensor, factor: float) -> torch.Tensor:
    """``factor * image + (1 - factor) * other``, clipped to [0, 1]."""
    return torch.lerp(other.expand_as(image), image, factor).clamp_(0.0, 1.0)


def _jitter(
    image: torch.Tensor, brightness: float, contrast: float, saturation: float
) -> torch.Tensor:
    """Brightness, contrast, then saturation, each by a factor drawn from ``[1 - s, 1 + s]``."""
    if brightness:
        image = image.mul(_uniform(max(0.0, 1 - brightness), 1 + brightness)).clamp_(0.0, 1.0)
    if contrast:
        factor = _uniform(max(0.0, 1 - contrast), 1 + contrast)
        image = _blend(image, _grayscale(image).mean(), factor)
    if saturation:
        image = _blend(image, _grayscale(image), _uniform(max(0.0, 1 - saturation), 1 + saturation))
    return image


def _crop(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """A ``height`` x ``width`` window at a random position."""
    top = int(torch.randint(image.shape[1] - height + 1, ()).item())
    left = int(torch.randint(image.shape[2] - width + 1, ()).item())
    return image[:, top : top + height, left : left + width]


def _rotate(image: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate about the centre, keeping the size; uncovered corners repeat the border pixels."""
    _, height, width = image.shape
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    # affine_grid works in [-1, 1] coordinates on both axes; the aspect terms keep the rotation
    # rigid in pixel space on non-square images.
    theta = torch.tensor(
        [[cos, -sin * height / width, 0.0], [sin * width / height, cos, 0.0]], dtype=image.dtype
    )
    grid = F.affine_grid(theta[None], [1, 3, height, width], align_corners=False)
    rotated = F.grid_sample(
        image[None], grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return rotated[0].clamp_(0.0, 1.0)


class NaturalAugmentation:
    """Augmentation for photographs and other natural images.

    1. A random crop covering ``crop_scale`` of the image area with a log-uniform aspect ratio
       in ``crop_ratio`` (up to 10 attempts, then the whole image), kept at native resolution.
    2. A horizontal flip with probability ``flip_probability``.
    3. With probability ``jitter_probability``, brightness, contrast, and saturation jitter:
       each scaled by a factor drawn uniformly from ``[1 - s, 1 + s]``.

    The defaults are milder than those used to train from scratch, because the backbone is
    pretrained.

    Raises:
        ConfigurationError: If a parameter is out of range.
    """

    def __init__(
        self,
        *,
        crop_scale: tuple[float, float] = (0.5, 1.0),
        crop_ratio: tuple[float, float] = (3 / 4, 4 / 3),
        flip_probability: float = 0.5,
        jitter_probability: float = 0.8,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
    ) -> None:
        self.crop_scale = _interval(crop_scale, "crop_scale", 1.0)
        self.crop_ratio = _interval(crop_ratio, "crop_ratio", math.inf)
        self.flip_probability = _number(flip_probability, "flip_probability", 0.0, 1.0)
        self.jitter_probability = _number(jitter_probability, "jitter_probability", 0.0, 1.0)
        self.brightness = _number(brightness, "brightness", 0.0, 1.0)
        self.contrast = _number(contrast, "contrast", 0.0, 1.0)
        self.saturation = _number(saturation, "saturation", 0.0, 1.0)

    def _crop_size(self, height: int, width: int) -> tuple[int, int]:
        log_low, log_high = (math.log(r) for r in self.crop_ratio)
        for _ in range(_CROP_ATTEMPTS):
            area = height * width * _uniform(*self.crop_scale)
            ratio = math.exp(_uniform(log_low, log_high))
            crop_width = round(math.sqrt(area * ratio))
            crop_height = round(math.sqrt(area / ratio))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                return crop_height, crop_width
        return height, width

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Augment one float ``[3, H, W]`` tensor in [0, 1]; the input is not modified."""
        image = _crop(image, *self._crop_size(image.shape[1], image.shape[2]))
        if torch.rand(()).item() < self.flip_probability:
            image = image.flip(-1)
        if torch.rand(()).item() < self.jitter_probability:
            image = _jitter(image, self.brightness, self.contrast, self.saturation)
        return image.contiguous()


class DocumentAugmentation:
    """Augmentation for scanned or photographed pages, forms, and other documents.

    Only changes that leave the document's meaning intact:

    1. A mild crop covering ``crop_scale`` of the area with the page's own aspect ratio, so at
       least about 92% of each side remains with the default (0.85, 1.0).
    2. A rotation by up to ``max_rotation`` degrees either way (scan skew), with the uncovered
       corners filled by repeating the border, so no paper colour is assumed.
    3. Brightness and contrast scaled by factors drawn from ``[1 - s, 1 + s]`` (exposure).

    There are no flips (they mirror text and layout), no saturation or hue changes (ink and
    stamp colours can carry meaning), and no large crops or erasing (they can remove the
    deciding field).

    Raises:
        ConfigurationError: If a parameter is out of range.
    """

    def __init__(
        self,
        *,
        crop_scale: tuple[float, float] = (0.85, 1.0),
        max_rotation: float = 2.0,
        brightness: float = 0.1,
        contrast: float = 0.1,
    ) -> None:
        self.crop_scale = _interval(crop_scale, "crop_scale", 1.0)
        self.max_rotation = _number(max_rotation, "max_rotation", 0.0, 180.0)
        self.brightness = _number(brightness, "brightness", 0.0, 1.0)
        self.contrast = _number(contrast, "contrast", 0.0, 1.0)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Augment one float ``[3, H, W]`` tensor in [0, 1]; the input is not modified."""
        _, height, width = image.shape
        side = math.sqrt(_uniform(*self.crop_scale))
        image = _crop(image, max(1, round(height * side)), max(1, round(width * side)))
        if self.max_rotation:
            image = _rotate(image, _uniform(-self.max_rotation, self.max_rotation))
        image = _jitter(image, self.brightness, self.contrast, 0.0)
        return image.contiguous()
