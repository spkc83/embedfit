"""Record builders for the optional pandas and Hugging Face ``datasets`` adapters.

``ImageDataset.from_pandas`` and ``ImageDataset.from_hf_dataset`` call these; the optional
packages are imported only here, when an adapter is used.
"""

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embedfit.exceptions import ConfigurationError, ImageInputError

_MAPPABLE = ("image", "label", "sample_id", "group_id", "timestamp", "amount")
_DEFAULT_COLUMNS = {"image": "image", "label": "label"}


def _import(package: str) -> Any:
    try:
        return __import__(package)
    except ImportError as exc:
        raise ConfigurationError(
            f"This adapter requires the {package} package. "
            f"Install it with: pip install 'embedfit[{package}]'"
        ) from exc


def _resolve_columns(
    columns: Mapping[str, str] | None,
    metadata_columns: Sequence[str],
    available: Sequence[str],
    container: str,
) -> tuple[dict[str, str], list[str]]:
    """Validate the field → column mapping. Column names are configuration, not data."""
    mapping = {**_DEFAULT_COLUMNS, **(columns or {})}
    for field in mapping:
        if field == "metadata":
            raise ConfigurationError("List metadata columns in metadata_columns, not columns.")
        if field not in _MAPPABLE:
            raise ConfigurationError(
                f"Unknown field {field!r} in columns; allowed fields are {list(_MAPPABLE)}."
            )
    if isinstance(metadata_columns, str) or not isinstance(metadata_columns, Sequence):
        raise ConfigurationError("metadata_columns must be a sequence of column names.")
    seen: set[str] = set()
    for column in mapping.values():
        if column in seen:
            raise ConfigurationError(f"The column {column!r} is mapped to more than one field.")
        seen.add(column)
    for column in [*mapping.values(), *metadata_columns]:
        if column not in available:
            raise ConfigurationError(f"The column {column!r} is not in the {container}.")
    return mapping, list(metadata_columns)


def _plain(value: Any) -> Any:
    """Unwrap numpy scalars and 0-d arrays or tensors to Python values."""
    if isinstance(value, np.generic) or (
        isinstance(value, np.ndarray | torch.Tensor) and value.ndim == 0
    ):
        return value.item()
    return value


def _is_missing(value: Any) -> bool:
    """Whether a plain value is missing: None, NaN, or pandas' NaT or NA."""
    if value is None or (isinstance(value, float) and value != value):
        return True
    pandas = sys.modules.get("pandas")
    return pandas is not None and (value is pandas.NaT or value is pandas.NA)


def _image(value: Any) -> Any:
    """Map a Hugging Face image dict to its bytes, or to its path when it has no bytes.

    Returns ``None`` for a dict with neither bytes nor an absolute path: a relative path is
    relative to an archive or a download location, not to the working directory.
    """
    if isinstance(value, Mapping) and set(value) == {"bytes", "path"}:
        if value["bytes"] is not None:
            return bytes(value["bytes"])
        if value["path"] is not None and Path(value["path"]).is_absolute():
            return Path(value["path"])
        return None
    return value


def _records(
    values: Mapping[str, list[Any]],
    mapping: Mapping[str, str],
    metadata_columns: Sequence[str],
    on_corrupt_image: str,
) -> tuple[list[dict[str, Any]], int]:
    """Assemble record mappings from per-column value lists, dropping unresolvable images."""
    records: list[dict[str, Any]] = []
    skipped = 0
    for position in range(len(values[mapping["image"]])):
        image = _image(values[mapping["image"]][position])
        if image is None:
            if on_corrupt_image == "skip":
                skipped += 1
                continue
            raise ImageInputError(
                f"Record {position}: the image has no bytes and a relative path, which cannot "
                "be resolved; give an absolute path or the encoded bytes."
            )
        label = _plain(values[mapping["label"]][position])
        if _is_missing(label):
            raise ConfigurationError(f"Record {position} has no label.")
        try:
            hash(label)
        except TypeError:
            raise ConfigurationError(
                f"Record {position} has an unhashable label of type {type(label).__name__}."
            ) from None
        record: dict[str, Any] = {"image": image, "label": label}
        for field, column in mapping.items():
            if field not in record:
                value = _plain(values[column][position])
                record[field] = None if _is_missing(value) else value
        if metadata_columns:
            record["metadata"] = {
                column: None if _is_missing(value) else value
                for column in metadata_columns
                for value in (_plain(values[column][position]),)
            }
        records.append(record)
    return records, skipped


def pandas_records(
    frame: Any,
    columns: Mapping[str, str] | None,
    metadata_columns: Sequence[str],
    on_corrupt_image: str,
) -> tuple[list[dict[str, Any]], int]:
    """Records from a DataFrame, and the number dropped for an unresolvable image dict."""
    pandas = _import("pandas")
    if not isinstance(frame, pandas.DataFrame):
        raise ConfigurationError(
            f"from_pandas expects a pandas DataFrame, got {type(frame).__name__}."
        )
    mapping, metadata = _resolve_columns(
        columns, metadata_columns, list(frame.columns), "DataFrame"
    )
    duplicated = set(frame.columns[frame.columns.duplicated()])
    for column in [*mapping.values(), *metadata]:
        if column in duplicated:
            raise ConfigurationError(
                f"The column {column!r} appears more than once in the DataFrame."
            )
    values = {column: frame[column].tolist() for column in {*mapping.values(), *metadata}}
    return _records(values, mapping, metadata, on_corrupt_image)


def _array_dtype(datasets: Any, feature: Any) -> str | None:
    """The declared numpy dtype of an array-valued feature; ``None`` for any other feature.

    Raises:
        ConfigurationError: If the feature holds a list of images per row.
    """
    arrays = tuple(getattr(datasets, f"Array{n}D") for n in range(2, 6))
    if isinstance(feature, arrays):
        return str(feature.dtype)
    lists: tuple[Any, ...] = tuple(
        kind
        for kind in (getattr(datasets, name, None) for name in ("List", "LargeList", "Sequence"))
        if isinstance(kind, type)
    )
    depth = 0
    while isinstance(feature, lists):
        feature, depth = feature.feature, depth + 1
    if depth and isinstance(feature, datasets.Image):
        raise ConfigurationError(
            "The image column holds a list of images per row; build a column with one image "
            "per row, e.g. with dataset.map."
        )
    if depth and isinstance(feature, datasets.Value):
        try:
            if np.dtype(feature.dtype).kind in "biuf":
                return str(feature.dtype)
        except TypeError:
            pass  # A non-numeric Value such as "string".
    return None


def hf_records(
    dataset: Any,
    columns: Mapping[str, str] | None,
    metadata_columns: Sequence[str],
    on_corrupt_image: str,
) -> tuple[list[dict[str, Any]], int]:
    """Records from a map-style ``datasets.Dataset``, and the number dropped for an
    unresolvable image dict."""
    datasets = _import("datasets")
    if isinstance(dataset, datasets.DatasetDict):
        raise ConfigurationError(
            "from_hf_dataset expects one split; select a split, e.g. dataset['train']."
        )
    if isinstance(dataset, datasets.IterableDataset):
        raise ConfigurationError(
            "from_hf_dataset needs a map-style datasets.Dataset, not an IterableDataset."
        )
    if not isinstance(dataset, datasets.Dataset):
        raise ConfigurationError(
            "from_hf_dataset expects a Hugging Face datasets.Dataset, "
            f"got {type(dataset).__name__}."
        )
    mapping, metadata = _resolve_columns(columns, metadata_columns, dataset.column_names, "dataset")
    # Read Python values whatever the dataset's format: a torch or numpy format would give
    # 0-d tensor labels, which hash by identity, and its formatter can fail on its own.
    dataset = dataset.with_format(None)
    image_column = mapping["image"]
    feature = dataset.features[image_column]
    dtype = None
    if isinstance(feature, datasets.Image):
        # Read the encoded bytes instead of decoding every image now. The cast writes no
        # cache file and keeps the dataset's row selection.
        dataset = dataset.cast_column(image_column, datasets.Image(decode=False))
    else:
        dtype = _array_dtype(datasets, feature)
    values = {column: list(dataset[column]) for column in {*mapping.values(), *metadata}}
    if dtype is not None:
        values[image_column] = [
            None if value is None else np.asarray(value, dtype=dtype)
            for value in values[image_column]
        ]
    return _records(values, mapping, metadata, on_corrupt_image)
