"""N-per-class sampling for few-shot experiments."""

from collections import defaultdict
from typing import Any

import numpy as np

from embedfit.data.dataset import ImageDataset
from embedfit.data.labels import label_key
from embedfit.exceptions import ConfigurationError


def sample_few_shot(dataset: ImageDataset, *, shots: int, seed: int) -> ImageDataset:
    """Draw ``shots`` samples of every class, without replacement.

    Sample only from training data: drawing few-shot examples from a pool that also feeds
    validation or test sets leaks them.

    Args:
        dataset: The pool to sample from.
        shots: Samples per class (>= 1).
        seed: Seed for a local random generator; the global NumPy state is untouched. The
            same seed, shots, and dataset order always give the same samples.

    Returns:
        A dataset of ``shots * num_classes`` samples in their original relative order. Draws
        for different ``shots`` at one seed are independent: the 4-shot sample is not
        guaranteed to be a subset of the 8-shot sample.

    Raises:
        ConfigurationError: If arguments are invalid or a class has fewer than ``shots``
            samples. Classes are never topped up by sampling with replacement.
    """
    if not isinstance(dataset, ImageDataset):
        raise ConfigurationError(f"dataset must be an ImageDataset, got {type(dataset).__name__}.")
    if isinstance(shots, bool) or not isinstance(shots, int | np.integer) or shots < 1:
        raise ConfigurationError(f"shots must be a positive int, got {shots!r}.")
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ConfigurationError(f"seed must be a non-negative int, got {seed!r}.")
    if len(dataset) == 0:
        raise ConfigurationError("dataset is empty.")

    by_class: dict[Any, list[int]] = defaultdict(list)
    try:
        for position, label in enumerate(dataset.labels):
            by_class[label_key(label)].append(position)
    except TypeError as exc:
        raise ConfigurationError("Labels must be hashable values such as str or int.") from exc
    short = sum(len(positions) < shots for positions in by_class.values())
    if short:
        smallest = min(len(positions) for positions in by_class.values())
        raise ConfigurationError(
            f"{short} of {len(by_class)} classes have fewer than {shots} samples "
            f"(the smallest has {smallest})."
        )
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    for positions in by_class.values():
        chosen.extend(rng.choice(positions, size=int(shots), replace=False).tolist())
    return dataset.subset(sorted(chosen))
