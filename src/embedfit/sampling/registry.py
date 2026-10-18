"""Name-based construction of the batch samplers used by ``supcon`` and ``end_to_end`` training."""

from collections.abc import Callable, Iterator
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import BatchSampler as _TorchBatchSampler
from torch.utils.data import RandomSampler

from embedfit.exceptions import ConfigurationError
from embedfit.sampling.balanced import BalancedClassBatchSampler, batch_shape


@runtime_checkable
class BatchSampler(Protocol):
    """What a registered sampler returns: batches of dataset positions, epoch by epoch.

    Training calls ``set_epoch`` before iterating each epoch, and ``len`` must equal the number
    of batches an iteration yields, which must be at least one and the same in every epoch.
    Batches must depend only on the factory's ``seed`` and the epoch, because the setup check
    iterates a separate instance for every epoch before training.
    """

    def __iter__(self) -> Iterator[list[int]]: ...

    def __len__(self) -> int: ...

    def set_epoch(self, epoch: int) -> None: ...


SamplerFactory = Callable[..., BatchSampler]
"""``factory(targets, *, batch_size, seed) -> BatchSampler``; ``targets`` are class indices."""


class _RandomBatchSampler:
    """Uniformly shuffled batches, reseeded every epoch: the ``end_to_end`` default.

    The batches are those of ``DataLoader(batch_size=..., sampler=RandomSampler(generator))``
    with the generator seeded ``seed * 100_003 + epoch``.
    """

    def __init__(self, num_samples: int, batch_size: int, seed: int) -> None:
        self._generator = torch.Generator()
        self._seed = seed
        self._batches = _TorchBatchSampler(
            RandomSampler(range(num_samples), generator=self._generator),
            batch_size=batch_size,
            drop_last=False,
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[list[int]]:
        self._generator.manual_seed(self._seed * 100_003 + self.epoch)
        return iter(self._batches)


def _balanced(targets: np.ndarray, *, batch_size: int, seed: int) -> BatchSampler:
    classes_per_batch, samples_per_class = batch_shape(batch_size, len(np.unique(targets)))
    return BalancedClassBatchSampler(
        targets, classes_per_batch=classes_per_batch, samples_per_class=samples_per_class, seed=seed
    )


def _random(targets: np.ndarray, *, batch_size: int, seed: int) -> BatchSampler:
    return _RandomBatchSampler(len(targets), batch_size, seed)


_SAMPLERS: dict[str, SamplerFactory] = {"balanced": _balanced, "random": _random}


def needs_batch_check(name: str) -> bool:
    """Whether the setup check must iterate the sampler's batches. The built-in ``balanced``
    sampler is valid by construction: every class has 2+ samples and each batch draws 2+ per
    class."""
    return _SAMPLERS.get(name) is not _balanced


def register_sampler(name: str, factory: SamplerFactory, *, overwrite: bool = False) -> None:
    """Make a custom batch sampler selectable with ``TrainingArguments.sampler``.

    Args:
        name: Non-empty name.
        factory: Callable invoked as ``factory(targets, *, batch_size, seed)`` that returns a
            ``BatchSampler``. ``targets`` is the integer class index of every sample, in
            dataset order. Options of your own go in a closure or ``functools.partial``.
        overwrite: Replace an existing registration instead of raising.

    Raises:
        ConfigurationError: If the name is empty or already registered, or the factory is not
            callable.
    """
    if not isinstance(name, str) or not name or name != name.strip():
        raise ConfigurationError(f"Invalid sampler name {name!r}.")
    if not callable(factory):
        raise ConfigurationError(f"Sampler '{name}' factory must be callable.")
    if name in _SAMPLERS and not overwrite:
        raise ConfigurationError(
            f"Sampler '{name}' is already registered; pass overwrite=True to replace."
        )
    _SAMPLERS[name] = factory


def available_samplers() -> list[str]:
    """Return the registered sampler names, sorted."""
    return sorted(_SAMPLERS)


def create_sampler(name: str, targets: Any, *, batch_size: int, seed: int) -> BatchSampler:
    """Build a batch sampler by name.

    Raises:
        ConfigurationError: If the name is not registered, the factory rejects the call, or
            the result lacks ``__iter__``, ``__len__``, or ``set_epoch``.
    """
    factory = _SAMPLERS.get(name)
    if factory is None:
        raise ConfigurationError(
            f"Unknown sampler '{name}'. Available samplers: {available_samplers()}."
        )
    try:
        sampler = factory(np.asarray(targets), batch_size=batch_size, seed=seed)
    except TypeError as exc:
        raise ConfigurationError(
            f"Sampler '{name}' does not accept (targets, *, batch_size, seed), or raised "
            f"TypeError: {exc}"
        ) from exc
    if not isinstance(sampler, BatchSampler):
        raise ConfigurationError(
            f"Sampler '{name}' returned a {type(sampler).__name__}; a sampler needs __iter__, "
            "__len__, and set_epoch."
        )
    return sampler
