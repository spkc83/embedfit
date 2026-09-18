"""A tiny randomly initialised CNN backbone that needs no pretrained checkpoint.

``DummyBackbone`` lets the full framework (encoding, representation training, serialization)
run offline on CPU in seconds. It is trainable, so tests exercise real gradient flow rather
than mocks, but it carries no pretrained knowledge and is not meant for real tasks.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from embedfit.exceptions import ConfigurationError, ImageInputError

_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "weights.pt"


def _to_chw_float(image: Any) -> torch.Tensor:
    """Convert a PIL image, numpy array, or tensor to a float ``[C, H, W]`` tensor in [0, 1]."""
    if isinstance(image, Image.Image):
        n_frames = getattr(image, "n_frames", 1)
        if n_frames > 1:
            raise ImageInputError(
                f"Multi-frame image with {n_frames} frames is ambiguous; "
                "select a single frame before passing it in."
            )
        # Grayscale, palette, and alpha modes are converted to 3-channel RGB.
        array = np.asarray(image.convert("RGB"))
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0

    if isinstance(image, np.ndarray):
        # numpy images are channel-last: [H, W] or [H, W, C].
        if image.ndim == 2:
            tensor = torch.from_numpy(np.array(image)).unsqueeze(0)
        elif image.ndim == 3:
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        else:
            raise ImageInputError(
                f"numpy image must have shape [H, W] or [H, W, C]; got {image.ndim} dimensions."
            )
    elif isinstance(image, torch.Tensor):
        # Tensor images are channel-first: [H, W] or [C, H, W].
        if image.ndim == 2:
            tensor = image.unsqueeze(0)
        elif image.ndim == 3:
            tensor = image
        else:
            raise ImageInputError(
                f"tensor image must have shape [H, W] or [C, H, W]; got {image.ndim} dimensions."
            )
    else:
        raise ImageInputError(
            f"Unsupported image type {type(image).__name__}; "
            "expected PIL.Image.Image, numpy.ndarray, or torch.Tensor."
        )

    channels, height, width = tensor.shape
    if height == 0 or width == 0:
        raise ImageInputError(f"Image has an empty spatial dimension: {height}x{width}.")
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
    elif channels != 3:
        raise ImageInputError(
            f"Image must have 1 or 3 channels; got {channels}. "
            "numpy arrays are [H, W, C]; tensors are [C, H, W]."
        )
    return tensor


class DummyBackbone(nn.Module):
    """Small trainable CNN producing one embedding vector per image.

    Follows the backbone contract from ``SPEC.md`` §7: ``embedding_dim``, ``preprocess``,
    ``forward_features``, ``freeze``, ``unfreeze``, and ``save_pretrained``.

    Args:
        embedding_dim: Size of the output embedding.
        image_size: Side length, in pixels, that ``preprocess`` resizes images to.
        seed: Seed for weight initialisation. The same seed always yields the same weights and
            leaves the global torch RNG state untouched.

    Raises:
        ConfigurationError: If ``embedding_dim`` or ``image_size`` is not a positive integer,
            or ``seed`` is not an integer.
    """

    def __init__(self, embedding_dim: int = 32, image_size: int = 32, seed: int = 0) -> None:
        super().__init__()
        for name, value in (("embedding_dim", embedding_dim), ("image_size", image_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"{name} must be a positive int, got {value!r}.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError(f"seed must be an int, got {type(seed).__name__}.")
        self._embedding_dim = embedding_dim
        self.image_size = image_size
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.projection = nn.Linear(16, embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the vectors returned by ``forward_features``."""
        return self._embedding_dim

    def preprocess(self, image: Any) -> torch.Tensor:
        """Convert one image to the model's input tensor.

        Args:
            image: A ``PIL.Image.Image``; a numpy array shaped ``[H, W]`` or ``[H, W, C]``; or a
                torch tensor shaped ``[H, W]`` or ``[C, H, W]``. Arrays and tensors must be
                ``uint8`` in [0, 255] or finite floating point in [0, 1]. Grayscale input is
                replicated to three channels.

        Returns:
            A new float32 CPU tensor shaped ``[3, image_size, image_size]`` with values in
            [0, 1]. It never shares memory with ``image`` and is detached from any autograd
            graph. No mean/std normalisation is applied, because this backbone has no pretrained
            statistics.

        Raises:
            ImageInputError: If the type, shape, channel count, dtype, or floating-point value
                range is unsupported. The message describes structure only, never pixel values.
        """
        tensor = _to_chw_float(image)
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor.contiguous()

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed a batch of preprocessed images.

        Args:
            pixel_values: Float tensor shaped ``[batch, 3, H, W]``, as produced by stacking
                ``preprocess`` outputs.

        Returns:
            Float tensor shaped ``[batch, embedding_dim]``. Not L2-normalised.

        Raises:
            ImageInputError: If ``pixel_values`` is not a 4-D tensor with 3 channels.
        """
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ImageInputError(
                f"pixel_values must have shape [batch, 3, H, W]; got {tuple(pixel_values.shape)}."
            )
        embeddings: torch.Tensor = self.projection(self.features(pixel_values))
        return embeddings

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Alias for ``forward_features``."""
        return self.forward_features(pixel_values)

    def freeze(self) -> None:
        """Disable gradients for every parameter and switch to eval mode."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        """Re-enable gradients for every parameter. The train/eval mode is left unchanged."""
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def save_pretrained(self, path: str | Path) -> None:
        """Write the configuration and weights to a directory.

        Args:
            path: Target directory; created if missing. Receives ``config.json`` and
                ``weights.pt``. Only hyperparameters and weights are written.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "provider": "dummy",
            "embedding_dim": self._embedding_dim,
            "image_size": self.image_size,
            "seed": self.seed,
        }
        (directory / _CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
        torch.save(self.state_dict(), directory / _WEIGHTS_FILE)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DummyBackbone":
        """Load a backbone previously written by ``save_pretrained``.

        Args:
            path: Directory containing ``config.json`` and ``weights.pt``.

        Returns:
            A backbone with identical configuration and weights, in eval mode.

        Raises:
            ConfigurationError: If the directory is missing either file, or the configuration
                is malformed, incomplete, or was not written by ``DummyBackbone``.
        """
        directory = Path(path)
        config_path, weights_path = directory / _CONFIG_FILE, directory / _WEIGHTS_FILE
        if not config_path.is_file() or not weights_path.is_file():
            raise ConfigurationError(
                f"'{directory}' is not a DummyBackbone directory: "
                f"expected {_CONFIG_FILE} and {_WEIGHTS_FILE}."
            )
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"'{config_path}' is not valid JSON.") from exc
        if not isinstance(config, dict) or config.get("provider") != "dummy":
            raise ConfigurationError(f"'{config_path}' was not written by DummyBackbone.")
        missing = sorted({"embedding_dim", "image_size", "seed"} - config.keys())
        if missing:
            raise ConfigurationError(f"'{config_path}' is missing required keys: {missing}.")
        backbone = cls(
            embedding_dim=config["embedding_dim"],
            image_size=config["image_size"],
            seed=config["seed"],
        )
        backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        backbone.eval()
        return backbone
