"""Conversion of user-supplied images to one canonical tensor form.

Every backbone adapter receives images as float32 RGB tensors shaped ``[3, H, W]`` with values
in [0, 1] at the image's native resolution. Resizing and normalisation are checkpoint-specific
and stay inside the adapter (ADR 0003); this module only removes input-format differences.
"""

from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from embedfit.exceptions import ImageInputError

ImageLike: TypeAlias = Image.Image | np.ndarray | torch.Tensor | str | Path
"""A PIL image, numpy array, torch tensor, or path to an image file."""


def _pil_to_array(image: Image.Image) -> np.ndarray:
    """Decode a PIL image to a channel-last array: uint8 RGB, or float grayscale in [0, 1]."""
    n_frames = getattr(image, "n_frames", 1)
    if n_frames > 1:
        raise ImageInputError(
            f"Multi-frame image with {n_frames} frames is ambiguous; "
            "select a single frame before passing it in."
        )
    image = ImageOps.exif_transpose(image)
    # Pillow's convert("RGB") clips high-bit-depth modes to 0-255 instead of rescaling them.
    if image.mode.startswith("I;16"):
        return np.asarray(image).astype(np.float32) / 65535.0
    if image.mode == "F":
        return np.array(image)  # Validated against [0, 1] like any float array.
    if image.mode == "I":
        raise ImageInputError(
            "PIL mode 'I' (32-bit integer) has no defined intensity range; convert it to a "
            "float array in [0, 1] before passing it in."
        )
    # Grayscale, bilevel, palette, alpha, and CMYK modes are converted to 3-channel RGB.
    return np.array(image.convert("RGB"))


def _read_image_file(path: Path) -> np.ndarray:
    # Messages omit the path: file names can carry sensitive identifiers. The chained
    # exception keeps it available to a developer reading the traceback.
    try:
        with Image.open(path) as image:
            return _pil_to_array(image)
    except FileNotFoundError as exc:
        raise ImageInputError("Image file does not exist.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageInputError("Image file could not be decoded; it may be corrupt.") from exc


def to_rgb_tensor(image: ImageLike) -> torch.Tensor:
    """Convert one image to a float32 RGB tensor shaped ``[3, H, W]`` with values in [0, 1].

    Args:
        image: One of

            - ``PIL.Image.Image``. 8-bit modes are converted to RGB, 16-bit grayscale
              (``I;16``) is scaled by 65535, mode ``F`` must already lie in [0, 1], and 32-bit
              integer mode ``I`` and multi-frame images are rejected. The EXIF orientation tag
              is applied, so the tensor matches the image as displayed;
            - ``numpy.ndarray`` shaped ``[H, W]`` or ``[H, W, C]`` (channel-last);
            - ``torch.Tensor`` shaped ``[H, W]`` or ``[C, H, W]`` (channel-first);
            - ``str`` or ``pathlib.Path`` naming an image file, decoded with Pillow.

            Arrays and tensors must have 1 or 3 channels and be ``uint8`` in [0, 255] or finite
            floating point in [0, 1]. Single-channel input is replicated to three channels.

    Returns:
        A new CPU tensor at the image's native resolution. It never shares memory with
        ``image`` and is detached from any autograd graph.

    Raises:
        ImageInputError: If the type, shape, channel count, dtype, or value range is
            unsupported, or a file is missing or undecodable. Messages describe structure only,
            never pixel values or file paths.
    """
    if isinstance(image, Image.Image):
        image = _pil_to_array(image)
    elif isinstance(image, str | Path):
        image = _read_image_file(Path(image))

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            tensor = torch.from_numpy(np.array(image)).unsqueeze(0)
        elif image.ndim == 3:
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        else:
            raise ImageInputError(
                f"numpy image must have shape [H, W] or [H, W, C]; got {image.ndim} dimensions. "
                "Pass a batch as a list of images."
            )
    elif isinstance(image, torch.Tensor):
        if image.ndim == 2:
            tensor = image.unsqueeze(0)
        elif image.ndim == 3:
            tensor = image
        else:
            raise ImageInputError(
                f"tensor image must have shape [H, W] or [C, H, W]; got {image.ndim} dimensions. "
                "Pass a batch as a list of images."
            )
    else:
        raise ImageInputError(
            f"Unsupported image type {type(image).__name__}; expected PIL.Image.Image, "
            "numpy.ndarray, torch.Tensor, str, or pathlib.Path."
        )

    channels, height, width = tensor.shape
    if height == 0 or width == 0:
        raise ImageInputError(f"Image has an empty spatial dimension: {height}x{width}.")
    if channels not in (1, 3):
        raise ImageInputError(
            f"Image must have 1 or 3 channels; got {channels}. "
            "numpy arrays are [H, W, C]; tensors are [C, H, W]."
        )
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(device="cpu", dtype=torch.float32) / 255.0
    elif tensor.is_floating_point():
        if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
            raise ImageInputError(
                "Floating-point image values must be finite and lie in [0, 1]; "
                "pass uint8 for [0, 255] data or rescale first."
            )
        # Always copy so the result never shares memory with the caller's array or tensor.
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32, copy=True)
    else:
        raise ImageInputError(
            f"Unsupported image dtype {tensor.dtype}; expected uint8 in [0, 255] "
            "or floating point in [0, 1]."
        )

    if channels == 1:
        tensor = tensor.expand(3, -1, -1)  # Grayscale → RGB by channel replication.
    return tensor.contiguous()
