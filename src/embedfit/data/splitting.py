"""Train/test splitting that respects classes, groups, and time, plus overlap checks.

Every function is deterministic given its ``seed`` and the dataset order, uses a local random
generator (the global NumPy state is untouched), and returns new ``ImageDataset`` objects that
preserve the original sample order within each side.
"""

import logging
import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from types import MappingProxyType
from typing import Any

import numpy as np

from embedfit.data.dataset import ImageDataset
from embedfit.data.labels import label_key
from embedfit.exceptions import ConfigurationError, DataLeakageError

logger = logging.getLogger(__name__)

OVERLAP_FIELDS = ("sample_id", "group_id")


def _validate_test_size(test_size: Any) -> float:
    valid = isinstance(test_size, int | float) and not isinstance(test_size, bool)
    if not valid or not 0 < test_size < 1:
        raise ConfigurationError(f"test_size must be a number in (0, 1), got {test_size!r}.")
    return float(test_size)


def _rng(seed: Any) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
        raise ConfigurationError(f"seed must be a non-negative int, got {seed!r}.")
    return np.random.default_rng(int(seed))


def _positions_by_key(values: list[Any], keys: list[Any], name: str) -> dict[Any, list[int]]:
    grouped: dict[Any, list[int]] = defaultdict(list)
    try:
        for position, key in enumerate(keys):
            grouped[key].append(position)
    except TypeError as exc:
        raise ConfigurationError(f"{name} values must be hashable, such as str or int.") from exc
    return grouped


def _positions_by_class(dataset: ImageDataset) -> dict[Any, list[int]]:
    labels = dataset.labels
    try:
        keys = [label_key(label) for label in labels]
    except TypeError as exc:
        raise ConfigurationError("Labels must be hashable values such as str or int.") from exc
    return _positions_by_key(labels, keys, "label")


def _warn_about_missing_classes(train: ImageDataset, test: ImageDataset, split: str) -> None:
    """Log (counts only) when a class occurs on one side of a split but not the other."""
    train_classes = set(_positions_by_class(train))
    test_classes = set(_positions_by_class(test))
    for side, absent in (
        ("train", test_classes - train_classes),
        ("test", train_classes - test_classes),
    ):
        if absent:
            logger.warning(
                "%s: %d of %d classes have no samples on the %s side",
                split,
                len(absent),
                len(train_classes | test_classes),
                side,
            )


def _require_dataset(dataset: Any, minimum: int = 2) -> None:
    if not isinstance(dataset, ImageDataset):
        raise ConfigurationError(f"dataset must be an ImageDataset, got {type(dataset).__name__}.")
    if len(dataset) < minimum:
        raise ConfigurationError(
            f"dataset must contain at least {minimum} samples, got {len(dataset)}."
        )


def _field_values(dataset: ImageDataset, name: str) -> list[Any]:
    values = [getattr(sample, name) for sample in dataset]
    missing = sum(value is None for value in values)
    if missing:
        raise ConfigurationError(
            f"{missing} of {len(values)} samples have no {name}; every sample needs one."
        )
    return values


def _split(dataset: ImageDataset, test_positions: set[int]) -> tuple[ImageDataset, ImageDataset]:
    train = [position for position in range(len(dataset)) if position not in test_positions]
    return dataset.subset(train), dataset.subset(sorted(test_positions))


def stratified_split(
    dataset: ImageDataset, *, test_size: float, seed: int
) -> tuple[ImageDataset, ImageDataset]:
    """Split at random while keeping each class's share equal on both sides.

    Each class contributes ``round(n_class * test_size)`` samples to the test side (Python
    rounding: halves go to the even count), clamped so that both sides receive at least one
    sample of every class.

    This split ignores groups and time. Near-duplicates of one item or later observations can
    land on both sides, so treat it as a debugging baseline (see ``grouped_split`` and
    ``temporal_split``).

    Args:
        dataset: The samples to split.
        test_size: Fraction of each class assigned to the test side, in (0, 1).
        seed: Seed for the shuffle.

    Returns:
        ``(train, test)``.

    Raises:
        ConfigurationError: If arguments are invalid or a class has fewer than 2 samples.
    """
    _require_dataset(dataset)
    fraction, rng = _validate_test_size(test_size), _rng(seed)
    by_class = _positions_by_class(dataset)
    smallest = min(len(positions) for positions in by_class.values())
    if smallest < 2:
        raise ConfigurationError(
            "Every class needs at least 2 samples for a stratified split; "
            f"the smallest class has {smallest}."
        )
    test_positions: set[int] = set()
    for positions in by_class.values():
        count = min(max(round(len(positions) * fraction), 1), len(positions) - 1)
        test_positions.update(rng.permutation(positions)[:count].tolist())
    return _split(dataset, test_positions)


def grouped_split(
    dataset: ImageDataset, *, test_size: float, seed: int
) -> tuple[ImageDataset, ImageDataset]:
    """Split at random so that no ``group_id`` appears on both sides.

    Groups are visited in shuffled order, and a group joins the test side when that brings
    the test size closer to ``test_size`` of the samples. At least one group always lands on
    each side. Because groups move whole, the realised fraction can differ from the request
    in either direction, markedly so when one group holds most of the data; a warning is
    logged when it misses the request by more than half.

    Classes are not balanced, and a class whose samples all share one group ends up entirely
    on one side. That is logged as a warning (counts only); check the result when it matters,
    especially for rare classes.

    Args:
        dataset: Samples that all carry a ``group_id``.
        test_size: Target fraction of samples on the test side, in (0, 1).
        seed: Seed for the group shuffle.

    Returns:
        ``(train, test)``.

    Raises:
        ConfigurationError: If arguments are invalid, a sample has no ``group_id``, or there
            are fewer than 2 groups.
    """
    _require_dataset(dataset)
    fraction, rng = _validate_test_size(test_size), _rng(seed)
    group_ids = _field_values(dataset, "group_id")
    by_group = _positions_by_key(group_ids, group_ids, "group_id")
    if len(by_group) < 2:
        raise ConfigurationError(f"A grouped split needs at least 2 groups, got {len(by_group)}.")
    groups = list(by_group)
    order = rng.permutation(len(groups)).tolist()
    target = fraction * len(dataset)
    test_positions: set[int] = set()
    # The last group in the shuffled order always stays in train, so train is never empty.
    for index in order[:-1]:
        members = by_group[groups[index]]
        if abs(len(test_positions) + len(members) - target) < abs(len(test_positions) - target):
            test_positions.update(members)
    if not test_positions:  # Every group overshoots: take the first rather than none.
        test_positions.update(by_group[groups[order[0]]])
    realised = len(test_positions) / len(dataset)
    if abs(realised - fraction) > 0.5 * fraction:
        logger.warning(
            "grouped_split: test side holds %d of %d samples (%.1f%%), requested %.1f%%",
            len(test_positions),
            len(dataset),
            100 * realised,
            100 * fraction,
        )
    train, test = _split(dataset, test_positions)
    _warn_about_missing_classes(train, test, "grouped_split")
    return train, test


def temporal_split(
    dataset: ImageDataset,
    *,
    cutoff: datetime | None = None,
    test_size: float | None = None,
) -> tuple[ImageDataset, ImageDataset]:
    """Split by time: train strictly before the cut-off, test at or after it.

    An out-of-time test set is the closest proxy for deployment, where the model only ever
    sees data from after it was trained. Samples sharing the cut-off timestamp all go to the
    test side, so one instant is never split.

    Classes are not balanced: a class seen only before (or only after) the cut-off is absent
    from one side, which is logged as a warning (counts only).

    Args:
        dataset: Samples that all carry a ``timestamp``.
        cutoff: The first instant belonging to the test side. Give either this or ``test_size``.
        test_size: Fraction in (0, 1); the cut-off becomes the timestamp of the sample that
            starts the most recent ``ceil(N * test_size)`` samples, so ties at that instant can
            make the test side larger.

    Returns:
        ``(train, test)``, each in original sample order.

    Raises:
        ConfigurationError: If not exactly one of ``cutoff`` and ``test_size`` is given, a
            sample has no ``timestamp`` or one that is not a ``datetime`` (strings, numbers,
            and ``date`` objects are rejected rather than compared), timestamps mix
            timezone-aware and naive values, or either side would be empty.
    """
    _require_dataset(dataset)
    if (cutoff is None) == (test_size is None):
        raise ConfigurationError("Pass exactly one of cutoff or test_size.")
    timestamps = _field_values(dataset, "timestamp")
    wrong_type = [type(stamp).__name__ for stamp in timestamps if not isinstance(stamp, datetime)]
    if wrong_type:
        raise ConfigurationError(
            f"{len(wrong_type)} of {len(timestamps)} timestamps are not datetime objects "
            f"(found {sorted(set(wrong_type))}); convert them before splitting."
        )
    try:
        if cutoff is None:
            count = math.ceil(round(len(dataset) * _validate_test_size(test_size), 9))
            cutoff = sorted(timestamps)[len(timestamps) - count]
        elif not isinstance(cutoff, datetime):
            raise ConfigurationError(f"cutoff must be a datetime, got {type(cutoff).__name__}.")
        test_positions = {position for position, stamp in enumerate(timestamps) if stamp >= cutoff}
    except TypeError as exc:
        raise ConfigurationError(
            "Timestamps (and cutoff) must be datetimes that are all timezone-aware or all naive."
        ) from exc
    if not test_positions or len(test_positions) == len(dataset):
        raise ConfigurationError(
            "The cut-off leaves one side empty: "
            f"{len(dataset) - len(test_positions)} train and {len(test_positions)} test samples."
        )
    train, test = _split(dataset, test_positions)
    _warn_about_missing_classes(train, test, "temporal_split")
    return train, test


@dataclass(frozen=True, eq=False)
class SplitOverlapReport:
    """Identifier values shared between data splits.

    Attributes:
        overlaps: Maps ``(split_a, split_b, field)`` to the values present in both splits.
            Only non-empty overlaps are listed.
        missing: Maps ``(split, field)`` to the number of samples without a value, which
            therefore could not be checked.
    """

    overlaps: Mapping[tuple[str, str, str], frozenset[Hashable]] = field(default_factory=dict)
    missing: Mapping[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Read-only copies: a report cannot be edited into saying something else.
        object.__setattr__(self, "overlaps", MappingProxyType(dict(self.overlaps)))
        object.__setattr__(self, "missing", MappingProxyType(dict(self.missing)))

    @property
    def has_overlap(self) -> bool:
        """Whether any identifier value appears in more than one split."""
        return bool(self.overlaps)

    def summary(self) -> str:
        """Describe the overlaps by count. Identifier values are never included."""
        if not self.overlaps:
            return "No overlap between splits."
        parts = [
            f"{len(values)} shared {name} value(s) between '{first}' and '{second}'"
            for (first, second, name), values in sorted(self.overlaps.items())
        ]
        return "; ".join(parts) + "."

    def __repr__(self) -> str:
        return f"SplitOverlapReport({self.summary()!r})"

    def raise_if_any(self) -> None:
        """Raise ``DataLeakageError`` when any overlap exists.

        Raises:
            DataLeakageError: With a count-only summary of the overlaps.
        """
        if self.has_overlap:
            raise DataLeakageError(f"Data leakage between splits: {self.summary()}")


def find_split_overlap(
    splits: Mapping[str, ImageDataset],
    *,
    fields: Sequence[str] = OVERLAP_FIELDS,
) -> SplitOverlapReport:
    """Find ``sample_id`` and ``group_id`` values that occur in more than one split.

    A shared ``sample_id`` means the same sample is in two splits; a shared ``group_id`` means
    related samples (scans of one item, one entity's items) straddle them. Either lets a model
    score well by memorising rather than generalising.

    Args:
        splits: Named datasets, e.g. ``{"train": train, "validation": val, "test": test}``.
        fields: Identifier fields to compare, from ``"sample_id"`` and ``"group_id"``.

    Returns:
        A report of every pairwise overlap. Samples without a value for a field are counted
        in ``missing`` and otherwise ignored. Identifiers match on value and type, so ``1``
        and ``True`` are different identifiers.

    Raises:
        ConfigurationError: If fewer than two splits are given, a field name is invalid, or a
            field is absent from *every* sample of a split, which would make the check
            silently vacuous, or an identifier is not hashable.
    """
    if not isinstance(splits, Mapping) or len(splits) < 2:
        raise ConfigurationError("splits must be a mapping of at least two named datasets.")
    invalid = [name for name in fields if name not in OVERLAP_FIELDS]
    if invalid or not fields:
        raise ConfigurationError(
            f"fields must be chosen from {list(OVERLAP_FIELDS)}, got {list(fields)}."
        )
    values: dict[tuple[str, str], dict[Any, Hashable]] = {}
    missing: dict[tuple[str, str], int] = {}
    for split_name, dataset in splits.items():
        if not isinstance(dataset, ImageDataset):
            raise ConfigurationError(f"Split '{split_name}' must be an ImageDataset.")
        for name in fields:
            present = [getattr(s, name) for s in dataset if getattr(s, name) is not None]
            if len(dataset) and not present:
                raise ConfigurationError(
                    f"No sample in split '{split_name}' has a {name}, so overlap on {name} "
                    "cannot be checked. Provide the field or remove it from fields."
                )
            try:
                values[split_name, name] = {label_key(value): value for value in present}
            except TypeError as exc:
                raise ConfigurationError(
                    f"{name} values must be hashable, such as str or int."
                ) from exc
            if len(present) < len(dataset):
                missing[split_name, name] = len(dataset) - len(present)
    overlaps = {}
    for first, second in combinations(splits, 2):
        for name in fields:
            shared = values[first, name].keys() & values[second, name].keys()
            if shared:
                overlaps[first, second, name] = frozenset(
                    values[first, name][key] for key in shared
                )
    return SplitOverlapReport(overlaps=overlaps, missing=missing)
