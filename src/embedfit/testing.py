"""Synthetic image generators for tests, examples, and offline experiments.

The images contain learnable visual structure (oriented bars on a noisy background) so that a
tiny encoder can genuinely separate the classes. Nothing here resembles real documents.
"""

import numpy as np
from PIL import Image

from embedfit.exceptions import ConfigurationError

BAR_CLASSES: tuple[str, ...] = ("vertical", "horizontal", "diagonal", "anti_diagonal")
"""Class names, in label-index order, that ``make_bar_images`` can generate."""


def _draw_bar(canvas: np.ndarray, kind: str, offset: int, thickness: int) -> None:
    size = canvas.shape[0]
    center = size // 2 + offset
    low, high = max(center - thickness // 2, 0), min(center - thickness // 2 + thickness, size)
    if kind == "vertical":
        canvas[:, low:high] = 1.0
    elif kind == "horizontal":
        canvas[low:high, :] = 1.0
    else:
        rows, cols = np.indices(canvas.shape)
        if kind == "anti_diagonal":
            cols = size - 1 - cols
        distance = rows - cols - offset + thickness // 2
        canvas[(distance >= 0) & (distance < thickness)] = 1.0


def make_bar_images(
    num_per_class: int,
    *,
    num_classes: int = 3,
    image_size: int = 32,
    noise: float = 0.1,
    seed: int = 0,
    grayscale: bool = False,
) -> tuple[list[Image.Image], list[str]]:
    """Generate a balanced synthetic dataset of oriented-bar images.

    Each image is a bright bar (vertical, horizontal, diagonal, or anti-diagonal) at a jittered
    position on a dark background, with additive Gaussian noise.

    Args:
        num_per_class: Number of images generated for each class (>= 1).
        num_classes: Number of classes, between 2 and ``len(BAR_CLASSES)``. Classes are taken
            from ``BAR_CLASSES`` in order.
        image_size: Side length in pixels (>= 8).
        noise: Standard deviation of the Gaussian pixel noise, on a [0, 1] intensity scale.
        seed: Seed for a local generator; the global NumPy RNG is not touched. The same
            arguments always produce identical images.
        grayscale: Return single-channel ``"L"`` images instead of ``"RGB"``.

    Returns:
        ``(images, labels)``: PIL images and their class names, grouped by class in
        ``BAR_CLASSES`` order (all ``"vertical"`` first, and so on). Shuffle before use if order
        matters.

    Raises:
        ConfigurationError: If any argument is out of range.
    """
    if num_per_class < 1:
        raise ConfigurationError(f"num_per_class must be >= 1, got {num_per_class}.")
    if not 2 <= num_classes <= len(BAR_CLASSES):
        raise ConfigurationError(
            f"num_classes must be between 2 and {len(BAR_CLASSES)}, got {num_classes}."
        )
    if image_size < 8:
        raise ConfigurationError(f"image_size must be >= 8, got {image_size}.")
    if noise < 0:
        raise ConfigurationError(f"noise must be >= 0, got {noise}.")

    rng = np.random.default_rng(seed)
    thickness = max(image_size // 8, 2)
    max_offset = image_size // 4
    images: list[Image.Image] = []
    labels: list[str] = []
    for kind in BAR_CLASSES[:num_classes]:
        for _ in range(num_per_class):
            canvas = np.zeros((image_size, image_size), dtype=np.float64)
            _draw_bar(canvas, kind, int(rng.integers(-max_offset, max_offset + 1)), thickness)
            canvas += rng.normal(0.0, noise, size=canvas.shape)
            pixels = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)
            image = Image.fromarray(pixels)  # 2-D uint8 → mode "L".
            images.append(image if grayscale else image.convert("RGB"))
            labels.append(kind)
    return images, labels
