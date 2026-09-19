"""Stable mapping between external class labels and integer class indices."""

from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import numpy as np

from embedfit.exceptions import ConfigurationError


def _plain(label: Any) -> Any:
    """Unwrap numpy scalars so ``np.str_("a")`` and ``"a"`` are the same label."""
    return label.item() if isinstance(label, np.generic) else label


def _key(label: Any) -> tuple[type, Any]:
    # The type is part of the key: ``False == 0`` must not make False an alias of label 0.
    label = _plain(label)
    return (type(label), label)


class LabelMapping:
    """An ordered, immutable set of class labels.

    The position of a label is its class index. The order is taken exactly as given and is
    never sorted, so probability columns and the positive class never depend on sort order.

    Args:
        labels: At least two unique hashable labels (``str``, ``int``, ...), as a sequence or
            1-D numpy array. numpy scalars are unwrapped to Python values. NaN is rejected.

    Raises:
        ConfigurationError: If fewer than two labels are given, or labels are not unique,
            not hashable, or NaN.
    """

    def __init__(self, labels: Sequence[Hashable] | np.ndarray) -> None:
        if isinstance(labels, np.ndarray) and labels.ndim == 1:
            labels = labels.tolist()
        if isinstance(labels, str) or not isinstance(labels, Sequence):
            raise ConfigurationError(
                "labels must be a sequence of label values, e.g. a list or 1-D numpy array."
            )
        plain = tuple(_plain(label) for label in labels)
        if any(isinstance(label, float) and label != label for label in plain):
            raise ConfigurationError("NaN cannot be used as a label: it never equals itself.")
        if len(plain) < 2:
            raise ConfigurationError(f"labels must contain at least 2 classes, got {len(plain)}.")
        try:
            index = {_key(label): position for position, label in enumerate(plain)}
        except TypeError as exc:
            raise ConfigurationError("labels must be hashable values such as str or int.") from exc
        if len(index) != len(plain):
            raise ConfigurationError("labels must be unique.")
        self._labels = plain
        self._index = index

    @property
    def labels(self) -> tuple[Hashable, ...]:
        """The labels in class-index order."""
        return self._labels

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, label: object) -> bool:
        try:
            return _key(label) in self._index
        except TypeError:
            return False

    def index_of(self, label: Hashable) -> int:
        """Return the class index of one label.

        Raises:
            ConfigurationError: If the label is not in the mapping (same value *and* type).
        """
        if label not in self:
            raise ConfigurationError(
                f"Label {label!r} is not one of the model's labels {list(self._labels)}; "
                "labels must match in both value and type."
            )
        return self._index[_key(label)]

    def encode(self, labels: Iterable[Hashable]) -> np.ndarray:
        """Convert external labels to an ``int64`` array of class indices.

        Raises:
            ConfigurationError: If any label is not in the mapping.
        """
        return np.fromiter((self.index_of(label) for label in labels), dtype=np.int64)

    def decode(self, indices: Iterable[int]) -> np.ndarray:
        """Convert class indices to an array of external labels.

        Returns:
            A 1-D numpy array. Its dtype is natural (e.g. ``<U`` for ``str``, ``int64`` for
            ``int``) when all labels share one scalar type, and ``object`` otherwise.
        """
        kinds = {type(label) for label in self._labels}
        if len(kinds) == 1 and kinds <= {str, int, float, bool}:
            lookup = np.array(self._labels)
        else:  # Mixed or non-scalar (e.g. tuple) labels must stay single elements.
            lookup = np.empty(len(self._labels), dtype=object)
            lookup[:] = self._labels
        return lookup[np.asarray(list(indices), dtype=np.int64)]
