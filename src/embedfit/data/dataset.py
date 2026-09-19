"""Record-level dataset: normalises sample records, never pixels."""

from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, overload

import numpy as np

from embedfit.data.image import ImageLike
from embedfit.exceptions import ConfigurationError

_FIELDS = ("image", "label", "sample_id", "group_id", "timestamp", "amount", "metadata")


@dataclass(frozen=True, eq=False, repr=False)
class ImageSample:
    """One labelled image with optional evaluation context.

    Samples compare and hash by identity: images may be arrays, which have no well-defined
    equality, so two samples are equal only if they are the same object. Use ``sample_id`` or
    ``group_id`` to relate samples.

    Attributes:
        image: PIL image, numpy array, torch tensor, or file path. Paths stay paths and are
            decoded only when the image is encoded.
        label: The class label.
        sample_id: Opaque identifier.
        group_id: Identifier shared by samples that must not be split across train and test.
        timestamp: When the sample was observed, for temporal splitting.
        amount: Monetary or other weight for value-weighted metrics.
        metadata: Free-form context, stored as a read-only copy. Never used as a model
            feature and never logged.
    """

    image: ImageLike
    label: Hashable
    sample_id: str | None = None
    group_id: str | None = None
    timestamp: datetime | None = None
    amount: float | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is not None:
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __repr__(self) -> str:
        # Images, amounts, identifiers, and metadata may be sensitive: show field presence only.
        present = [name for name in _FIELDS[2:] if getattr(self, name) is not None]
        return f"ImageSample(image=<{type(self.image).__name__}>, optional_fields={present})"


class ImageDataset:
    """An in-memory sequence of ``ImageSample`` records.

    Args:
        records: A sequence of ``ImageSample`` objects or mappings with the keys ``image`` and
            ``label`` and, optionally, ``sample_id``, ``group_id``, ``timestamp``, ``amount``,
            and ``metadata``. Any ``torch.utils.data.Dataset`` with ``__len__`` that yields such
            mappings is accepted; every item is read once, at construction, so a source that
            decodes images in ``__getitem__`` is held in memory. Store file paths as ``image``
            to keep decoding lazy.

    Raises:
        ConfigurationError: If a record has the wrong type, lacks ``image`` or ``label``, or
            has unknown keys. Messages name the record's position and key names only.
    """

    def __init__(self, records: Sequence[ImageSample | Mapping[str, Any]]) -> None:
        if (
            isinstance(records, Mapping | str)
            or not hasattr(records, "__len__")
            or not hasattr(records, "__getitem__")
        ):
            raise ConfigurationError(
                "records must be a sequence (or map-style torch Dataset) of samples."
            )
        self._samples = [
            self._normalise(records[position], position) for position in range(len(records))
        ]

    @staticmethod
    def _normalise(record: Any, position: int) -> ImageSample:
        if isinstance(record, ImageSample):
            return record
        if not isinstance(record, Mapping):
            raise ConfigurationError(
                f"Record {position} must be an ImageSample or a mapping, "
                f"got {type(record).__name__}."
            )
        missing = [key for key in ("image", "label") if key not in record]
        if missing:
            raise ConfigurationError(f"Record {position} is missing required keys: {missing}.")
        unknown = sorted(str(key) for key in record.keys() - set(_FIELDS))
        if unknown:
            raise ConfigurationError(
                f"Record {position} has unknown keys {unknown}; allowed keys are {list(_FIELDS)}. "
                "Put extra fields under 'metadata'."
            )
        return ImageSample(**record)

    @classmethod
    def from_lists(cls, images: Sequence[ImageLike], labels: Sequence[Hashable]) -> "ImageDataset":
        """Build a dataset from parallel lists of images and labels.

        Raises:
            ConfigurationError: If either argument is not a sized sequence, or the two differ
                in length.
        """
        if not hasattr(images, "__len__") or not hasattr(labels, "__len__"):
            raise ConfigurationError("images and labels must be sequences, e.g. lists.")
        if len(images) != len(labels):
            raise ConfigurationError(
                f"images and labels must have the same length, got {len(images)} and {len(labels)}."
            )
        return cls([ImageSample(image, label) for image, label in zip(images, labels, strict=True)])

    def __len__(self) -> int:
        return len(self._samples)

    @overload
    def __getitem__(self, position: int) -> ImageSample: ...

    @overload
    def __getitem__(self, position: slice) -> "ImageDataset": ...

    def __getitem__(self, position: int | slice) -> "ImageSample | ImageDataset":
        """Return one sample for an ``int``, or a new ``ImageDataset`` for a ``slice``."""
        if isinstance(position, slice):
            return ImageDataset(self._samples[position])
        return self._samples[position]

    def subset(self, positions: Iterable[int]) -> "ImageDataset":
        """Return a new dataset holding the samples at ``positions``, in the order given.

        Raises:
            ConfigurationError: If a position is not an int (Python or numpy) or is out of range.
        """
        chosen = list(positions)
        for position in chosen:
            is_int = isinstance(position, int | np.integer) and not isinstance(position, bool)
            if not is_int or not -len(self) <= position < len(self):
                raise ConfigurationError(
                    f"subset positions must be ints in [-{len(self)}, {len(self)}), "
                    f"got {position!r}."
                )
        return ImageDataset([self._samples[int(position)] for position in chosen])

    def __iter__(self) -> Iterator[ImageSample]:
        return iter(self._samples)

    def __repr__(self) -> str:
        return f"ImageDataset(num_samples={len(self)})"

    @property
    def images(self) -> list[ImageLike]:
        """The images, in record order (file paths are returned undecoded)."""
        return [sample.image for sample in self._samples]

    @property
    def labels(self) -> list[Hashable]:
        """The labels, in record order."""
        return [sample.label for sample in self._samples]
