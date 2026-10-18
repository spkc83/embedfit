"""Class-balanced batches for contrastive training."""

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np

from embedfit.exceptions import ConfigurationError


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ConfigurationError(f"{name} must be a positive int, got {value!r}.")
    return int(value)


def batch_shape(batch_size: int, num_classes: int) -> tuple[int, int]:
    """Split ``batch_size`` into ``(classes_per_batch, samples_per_class)``.

    As many classes as fit with at least two samples each are used, and the batch is divided
    evenly between them, so the realised batch size can be slightly below ``batch_size``.

    Raises:
        ConfigurationError: If there are fewer than 2 classes, or ``batch_size`` is below 4,
            the minimum for two classes with a positive pair each.
    """
    if num_classes < 2:
        raise ConfigurationError(
            f"Contrastive training needs at least 2 classes in the data, got {num_classes}."
        )
    if batch_size < 4:
        raise ConfigurationError(
            f"Contrastive training needs batch_size >= 4 (2 classes x 2 samples), got {batch_size}."
        )
    classes_per_batch = min(num_classes, batch_size // 2)
    return classes_per_batch, batch_size // classes_per_batch


class BalancedClassBatchSampler:
    """Yield batches holding ``samples_per_class`` items from each of ``classes_per_batch`` classes.

    Contrastive losses need several samples of a class in the same batch; on imbalanced data
    uniform batches rarely contain two samples of a rare class.

    Each batch draws its classes without replacement, cycling through a shuffled class order
    so that every class is visited equally often. Within a class, samples are drawn without
    replacement from a shuffled queue that refills when exhausted, so every sample is used
    before any is repeated. A class with fewer than ``samples_per_class`` samples contributes
    all of its samples instead of duplicates, because a duplicated image would be a trivial
    positive pair.

    Use as the ``batch_sampler`` of a ``torch.utils.data.DataLoader``.

    Args:
        targets: Class index of every sample, in dataset order.
        classes_per_batch: Classes per batch; at most the number of distinct classes.
        samples_per_class: Samples drawn from each chosen class (>= 2).
        seed: Base seed. Batches depend only on ``seed`` and the epoch set by ``set_epoch``.
        batches_per_epoch: Batches per iteration. Defaults to
            ``ceil(N / (classes_per_batch * samples_per_class))``: about ``N`` draws per epoch.
            Because draws are balanced by class, that is not one pass over the data: samples
            of rare classes recur within an epoch and samples of frequent classes are skipped.

    Raises:
        ConfigurationError: If an argument is invalid, or any class has fewer than 2 samples.
    """

    def __init__(
        self,
        targets: Sequence[int] | np.ndarray,
        *,
        classes_per_batch: int,
        samples_per_class: int,
        seed: int,
        batches_per_epoch: int | None = None,
    ) -> None:
        indices = np.asarray(targets)
        if indices.ndim != 1 or len(indices) == 0 or not np.issubdtype(indices.dtype, np.integer):
            raise ConfigurationError("targets must be a non-empty 1-D array of class indices.")
        self.classes_per_batch = _positive_int(classes_per_batch, "classes_per_batch")
        self.samples_per_class = _positive_int(samples_per_class, "samples_per_class")
        if self.samples_per_class < 2:
            raise ConfigurationError(
                "samples_per_class must be >= 2 so that every sample has a positive in its batch."
            )
        if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
            raise ConfigurationError(f"seed must be a non-negative int, got {seed!r}.")

        by_class: dict[int, list[int]] = defaultdict(list)
        for position, target in enumerate(indices.tolist()):
            by_class[target].append(position)
        smallest = min(len(positions) for positions in by_class.values())
        if smallest < 2:
            raise ConfigurationError(
                "Every class needs at least 2 samples for contrastive training; "
                f"the smallest class has {smallest}."
            )
        if self.classes_per_batch > len(by_class):
            raise ConfigurationError(
                f"classes_per_batch ({self.classes_per_batch}) exceeds the number of classes "
                f"({len(by_class)})."
            )
        self._by_class = dict(sorted(by_class.items()))
        self.seed = int(seed)
        self.epoch = 0
        batch_size = self.classes_per_batch * self.samples_per_class
        default_batches = -(-len(indices) // batch_size)
        self.batches_per_epoch = (
            default_batches
            if batches_per_epoch is None
            else _positive_int(batches_per_epoch, "batches_per_epoch")
        )

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose batches the next iteration yields."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    @staticmethod
    def _draw(
        queue: list[int], pool: Sequence[int], count: int, rng: np.random.Generator
    ) -> list[int]:
        """Pop ``count`` distinct items, refilling ``queue`` with a shuffle of ``pool`` as needed.

        An item that is already part of the draw when it comes up again (possible right after a
        refill) keeps its place at the head of the queue, so no item ever loses a turn.
        """
        if count > len(pool):
            raise ConfigurationError(f"Cannot draw {count} distinct items from {len(pool)}.")
        drawn: list[int] = []
        deferred: list[int] = []
        while len(drawn) < count:
            if not queue:
                queue.extend(rng.permutation(pool).tolist())
            candidate = queue.pop()
            (deferred if candidate in drawn else drawn).append(candidate)
        queue.extend(reversed(deferred))
        return drawn

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        classes = list(self._by_class)
        class_queue: list[int] = []
        sample_queues: dict[int, list[int]] = {target: [] for target in classes}
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for target in self._draw(class_queue, classes, self.classes_per_batch, rng):
                members = self._by_class[target]
                take = min(self.samples_per_class, len(members))
                batch.extend(self._draw(sample_queues[target], members, take, rng))
            yield batch
