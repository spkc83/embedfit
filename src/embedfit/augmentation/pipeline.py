"""The augmentation-pipeline contract and its seeded, validated call."""

from collections.abc import Callable

import numpy as np
import torch

from embedfit.exceptions import ConfigurationError

Augmentation = Callable[[torch.Tensor], torch.Tensor]
"""A task augmentation: one image tensor in, one image tensor out.

It receives a float32 CPU tensor shaped ``[3, H, W]`` with values in [0, 1] (a private copy it
may modify) and returns a floating-point tensor shaped ``[3, H', W']`` with values in [0, 1].
The size may change; the backbone's own preprocessing resizes afterwards. Randomness must come
from torch's global generator (``torch.rand``, ``torch.randint``, ...), which embedfit seeds for
every call, as ``torchvision.transforms.v2`` transforms do.
"""

_AUGMENT_STREAM = 0x41554740  # Keeps augmentation seeds apart from every other seed use.
_PROBE_SIZE = 32


def _draw_seed(seed: int, epoch: int, draw: int) -> int:
    state = np.random.SeedSequence([seed, _AUGMENT_STREAM, epoch, draw]).generate_state(
        1, np.uint64
    )
    return int(state[0])


def _check_output(pipeline: Augmentation, output: object) -> torch.Tensor:
    """Return ``output`` as float32, or raise describing its structure, never its values."""
    name = type(pipeline).__name__
    if not isinstance(output, torch.Tensor):
        raise ConfigurationError(
            f"Augmentation pipeline {name} must return a torch.Tensor, got {type(output).__name__}."
        )
    if output.ndim != 3 or output.shape[0] != 3 or min(output.shape[1:]) < 1:
        raise ConfigurationError(
            f"Augmentation pipeline {name} must return a tensor shaped [3, H, W] with H, W >= 1, "
            f"got shape {tuple(output.shape)}."
        )
    if not output.is_floating_point():
        raise ConfigurationError(
            f"Augmentation pipeline {name} must return a floating-point tensor, got {output.dtype}."
        )
    output = output.detach().to(device="cpu", dtype=torch.float32)
    low, high = torch.aminmax(output)
    if not (0 <= low.item() and high.item() <= 1):  # Also false for NaN; infinities are out.
        raise ConfigurationError(
            f"Augmentation pipeline {name} returned values that are not finite or lie outside "
            "[0, 1]; clamp its output."
        )
    return output


def augment_image(
    pipeline: Augmentation, image: torch.Tensor, *, seed: int, epoch: int, draw: int
) -> torch.Tensor:
    """Apply ``pipeline`` to one canonical image tensor with a seed of its own.

    The global CPU generator is seeded from ``(seed, epoch, draw)`` inside
    ``torch.random.fork_rng``, so the result depends only on those three numbers (not on the
    data-loader worker that runs it) and the global random state is unchanged afterwards.
    Accelerator generators are neither seeded nor touched.

    Args:
        pipeline: The augmentation.
        image: Float32 ``[3, H, W]`` tensor in [0, 1], from ``embedfit.data.to_rgb_tensor``.
        seed: The run's seed (``TrainingArguments.seed``).
        epoch: 0-based epoch.
        draw: Position of this image among the epoch's draws, so that a sample drawn twice in
            one epoch gets two different views.

    Returns:
        The augmented float32 ``[3, H', W']`` tensor in [0, 1].

    Raises:
        ConfigurationError: If the pipeline's output breaks the ``Augmentation`` contract.
    """
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(_draw_seed(seed, epoch, draw))
        output = pipeline(image)
    return _check_output(pipeline, output)


def probe_augmentation(pipeline: Augmentation, *, seed: int) -> None:
    """Run ``pipeline`` once on a synthetic image, so a malformed one fails before training.

    Raises:
        ConfigurationError: If the output breaks the ``Augmentation`` contract.
    """
    ramp = torch.linspace(0.0, 1.0, _PROBE_SIZE)
    image = torch.stack([ramp.expand(_PROBE_SIZE, -1), ramp[:, None].expand(-1, _PROBE_SIZE)])
    image = torch.cat([image, image.mean(dim=0, keepdim=True)]).contiguous()
    augment_image(pipeline, image, seed=seed, epoch=0, draw=0)
