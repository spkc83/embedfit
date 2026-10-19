"""Record-level dataset: normalises sample records, never pixels."""

import logging
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from embedfit.data.image import ImageLike, is_readable
from embedfit.exceptions import ConfigurationError

if TYPE_CHECKING:
    import datasets
    import pandas

logger = logging.getLogger(__name__)

_FIELDS = ("image", "label", "sample_id", "group_id", "timestamp", "amount", "metadata")
CorruptImagePolicy = Literal["raise", "skip"]


@dataclass(frozen=True, eq=False, repr=False)
class ImageSample:
    """One labelled image with optional evaluation context.

    Samples compare and hash by identity: images may be arrays, which have no well-defined
    equality, so two samples are equal only if they are the same object. Use ``sample_id`` or
    ``group_id`` to relate samples.

    Attributes:
        image: PIL image, numpy array, torch tensor, file path, or encoded image bytes. Paths
            and bytes stay as given and are decoded only when the image is encoded.
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
        on_corrupt_image: ``"raise"`` (default) reads no image here; a missing or undecodable
            file raises ``ImageInputError`` when the image is encoded. ``"skip"`` fully decodes
            every file-path and bytes image once, now, and drops the records whose file is
            missing, cannot be decoded, is truncated, or has more than twice Pillow's
            decompression-bomb limit; ``num_skipped`` reports how many. In-memory images are
            never dropped.

    Raises:
        ConfigurationError: If a record has the wrong type, lacks ``image`` or ``label``, or
            has unknown keys, or ``on_corrupt_image`` is invalid. Messages name the record's
            position and key names only.
        ImageInputError: With ``"skip"``, if an image path names a directory or an unreadable
            file; that is a setup error, not a corrupt image.
    """

    def __init__(
        self,
        records: Sequence[ImageSample | Mapping[str, Any]],
        *,
        on_corrupt_image: CorruptImagePolicy = "raise",
    ) -> None:
        self._build(records, on_corrupt_image, skipped=0)

    def _build(self, records: Any, on_corrupt_image: Any, skipped: int) -> None:
        _check_policy(on_corrupt_image)
        if (
            isinstance(records, Mapping | str)
            or not hasattr(records, "__len__")
            or not hasattr(records, "__getitem__")
        ):
            raise ConfigurationError(
                "records must be a sequence (or map-style torch Dataset) of samples."
            )
        samples = [self._normalise(records[position], position) for position in range(len(records))]
        count = skipped  # Records dropped as unreadable, by an adapter or below.
        if on_corrupt_image == "skip":
            kept = [sample for sample in samples if _readable(sample.image)]
            count += len(samples) - len(kept)
            samples = kept
        if count:
            logger.warning(
                "Skipped %d of %d records whose image could not be read",
                count,
                count + len(samples),
            )
        self._samples = samples
        self._num_skipped = count

    @classmethod
    def _with_skipped(
        cls,
        records: Sequence[Mapping[str, Any]],
        on_corrupt_image: CorruptImagePolicy,
        skipped: int,
    ) -> "ImageDataset":
        """Build a dataset that also reports ``skipped`` records an adapter dropped already.

        A subclass's ``__init__`` is not called; only ``ImageDataset``'s state is set.
        """
        dataset = cls.__new__(cls)
        dataset._build(records, on_corrupt_image, skipped)
        return dataset

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
    def from_lists(
        cls,
        images: Sequence[ImageLike],
        labels: Sequence[Hashable],
        *,
        on_corrupt_image: CorruptImagePolicy = "raise",
    ) -> "ImageDataset":
        """Build a dataset from parallel lists of images and labels.

        ``on_corrupt_image`` is as for ``ImageDataset``.

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
        return cls(
            [ImageSample(image, label) for image, label in zip(images, labels, strict=True)],
            on_corrupt_image=on_corrupt_image,
        )

    @classmethod
    def from_pandas(
        cls,
        frame: "pandas.DataFrame",
        *,
        columns: Mapping[str, str] | None = None,
        metadata_columns: Sequence[str] = (),
        on_corrupt_image: CorruptImagePolicy = "raise",
    ) -> "ImageDataset":
        """Build a dataset from a pandas DataFrame (needs the ``pandas`` extra).

        Args:
            frame: One row per sample.
            columns: Maps sample fields (``image``, ``label``, ``sample_id``, ``group_id``,
                ``timestamp``, ``amount``) to column names. ``image`` and ``label`` default to
                the columns of the same name; the optional fields are read only when mapped.
                Columns that are neither mapped nor listed in ``metadata_columns`` are ignored.
            metadata_columns: Columns copied, by name, into each sample's ``metadata``.
            on_corrupt_image: As for ``ImageDataset``.

        The image column may hold file paths, PIL images, numpy arrays, torch tensors, encoded
        image bytes, or Hugging Face image dicts ``{"bytes": ..., "path": ...}`` (bytes are used
        when present, otherwise the path, which must be absolute). Missing values (``None``,
        NaN, NaT) in the optional fields and in metadata become ``None``; numpy scalars and 0-d
        arrays or tensors become Python values. Error messages give the row position, never the
        index or a value.

        Raises:
            ConfigurationError: If pandas is not installed, ``frame`` is not a DataFrame, the
                mapping names an unknown field or a missing or duplicated column, or a label is
                missing or unhashable.
            ImageInputError: If an image dict has no bytes and a relative path, under
                ``"raise"``; under ``"skip"`` such a record is dropped and counted.
        """
        _check_policy(on_corrupt_image)
        from embedfit.data.adapters import pandas_records

        records, skipped = pandas_records(frame, columns, metadata_columns, on_corrupt_image)
        return cls._with_skipped(records, on_corrupt_image, skipped)

    @classmethod
    def from_hf_dataset(
        cls,
        dataset: "datasets.Dataset",
        *,
        columns: Mapping[str, str] | None = None,
        metadata_columns: Sequence[str] = (),
        on_corrupt_image: CorruptImagePolicy = "raise",
    ) -> "ImageDataset":
        """Build a dataset from a map-style Hugging Face ``datasets.Dataset`` (needs the
        ``datasets`` extra).

        ``columns`` and ``metadata_columns`` are as for ``from_pandas``. An image column with
        the ``datasets.Image`` feature is read without decoding, and each image is kept as its
        encoded file bytes, so every encoded image of a Hub dataset is held in memory: 10,000
        JPEG pages of about 100 KB each take about 1 GB. Keep a column of file paths instead
        for data larger than memory. An array column (``Array2D``, ``Array3D``, or nested lists of
        numbers) becomes numpy arrays of its declared dtype; other image columns pass through
        as stored. A column holding a list of images per row is rejected. Values are read as
        Python values whatever the dataset's format (``with_format``).
        ``ClassLabel`` columns keep their integer values; ``dataset.features[column].int2str``
        gives the class names. The dataset's row selection (``select``, ``shuffle``,
        ``filter``) is respected.

        Raises:
            ConfigurationError: If ``datasets`` is not installed, ``dataset`` is a
                ``DatasetDict`` (select a split), an ``IterableDataset``, or not a dataset, the
                mapping names an unknown field or a missing column, the image column holds a
                list of images per row, or a label is missing or unhashable.
            ImageInputError: As for ``from_pandas``.
        """
        _check_policy(on_corrupt_image)
        from embedfit.data.adapters import hf_records

        records, skipped = hf_records(dataset, columns, metadata_columns, on_corrupt_image)
        return cls._with_skipped(records, on_corrupt_image, skipped)

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
        skipped = f", num_skipped={self._num_skipped}" if self._num_skipped else ""
        return f"ImageDataset(num_samples={len(self)}{skipped})"

    @property
    def num_skipped(self) -> int:
        """Records dropped as unreadable when this dataset was built (0 for derived datasets)."""
        return self._num_skipped

    @property
    def images(self) -> list[ImageLike]:
        """The images, in record order (file paths are returned undecoded)."""
        return [sample.image for sample in self._samples]

    @property
    def labels(self) -> list[Hashable]:
        """The labels, in record order."""
        return [sample.label for sample in self._samples]


def _check_policy(on_corrupt_image: Any) -> None:
    if on_corrupt_image not in ("raise", "skip"):
        raise ConfigurationError(
            f"on_corrupt_image must be 'raise' or 'skip', got {on_corrupt_image!r}."
        )


def _readable(image: ImageLike) -> bool:
    """Whether an image survives the ``skip`` policy: in-memory images always do."""
    if isinstance(image, str | Path | bytes):
        return is_readable(image)
    return True
