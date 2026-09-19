"""A tiny randomly initialised CNN backbone that needs no pretrained checkpoint.

``DummyBackbone`` lets the full framework (encoding, representation training, serialization)
run offline on CPU in seconds. It is trainable, so tests exercise real gradient flow rather
than mocks, but it carries no pretrained knowledge and is not meant for real tasks.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from embedfit.backbones.base import ImageBackbone
from embedfit.data.image import ImageLike, to_rgb_tensor
from embedfit.exceptions import ConfigurationError

_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "weights.pt"


class DummyBackbone(ImageBackbone):
    """Small trainable CNN producing one embedding vector per image.

    Registered with the provider registry as ``"dummy/tiny"``.

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
    def identifier(self) -> str:
        """Always ``"dummy/tiny"``."""
        return "dummy/tiny"

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the vectors returned by ``forward_features``."""
        return self._embedding_dim

    def preprocess(self, image: ImageLike) -> torch.Tensor:
        """Convert one image to the model's input tensor.

        Args:
            image: Any form accepted by ``embedfit.data.to_rgb_tensor``.

        Returns:
            A new float32 CPU tensor shaped ``[3, image_size, image_size]`` with values in
            [0, 1]. It never shares memory with ``image`` and is detached from any autograd
            graph. No mean/std normalisation is applied, because this backbone has no pretrained
            statistics.

        Raises:
            ImageInputError: If the type, shape, channel count, dtype, or floating-point value
                range is unsupported. The message describes structure only, never pixel values.
        """
        tensor = to_rgb_tensor(image)
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
        self._check_pixel_values(pixel_values)
        embeddings: torch.Tensor = self.projection(self.features(pixel_values))
        return embeddings

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
            "dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
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
