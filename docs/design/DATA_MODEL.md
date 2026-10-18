# Dataset and Data Model

## Canonical sample

```python
@dataclass(frozen=True, eq=False, repr=False)
class ImageSample:
    image: ImageLike
    label: Hashable
    sample_id: str | None = None
    group_id: str | None = None
    timestamp: datetime | None = None
    amount: float | None = None
    metadata: Mapping[str, Any] | None = None
```

`ImageDataset` holds these records. It accepts `ImageSample` objects or mappings with the same keys (unknown keys are rejected; extra fields belong under `metadata`), and any map-style `torch.utils.data.Dataset` that yields such mappings. `repr` of a sample or dataset shows field presence and counts only. Samples compare by identity, since array images have no well-defined equality, and `metadata` is kept as a read-only copy.

Records are read once when the dataset is created. A source that decodes images inside `__getitem__` is therefore held in memory; storing file paths as `image` keeps decoding lazy, because paths are opened only when a batch is encoded. Encoded image file bytes (`bytes`) are also kept as given and decoded only when encoded.

## Adapters

`ImageDataset.from_pandas(frame, ...)` (needs the `pandas` extra) and `ImageDataset.from_hf_dataset(dataset, ...)` (needs the `datasets` extra) build a dataset from a table. Without the extra they raise `ConfigurationError` naming the package to install; the core install imports neither package.

- `columns` maps fields to column names. `image` and `label` default to the columns of the same name; `sample_id`, `group_id`, `timestamp`, and `amount` are read only when mapped, so a column is never picked up by accident. `metadata_columns` lists the columns copied, by name, into `metadata`. Other columns are ignored. An unknown field, a missing column, a column mapped to two fields, a used column name that appears twice in a DataFrame, or `metadata` in `columns` is a `ConfigurationError` that names the field or column.
- Missing values (`None`, NaN, NaT, `pd.NA`) in the optional fields and in metadata become `None` in both adapters, so a missing `group_id` does not become a NaN group that never equals itself. A missing label is an error, and so is an unhashable one (such as a list). numpy scalars and 0-d arrays or tensors become Python values. Errors name the row position, never the DataFrame index or a value.
- The image column may hold any accepted image form, encoded bytes, or Hugging Face image dicts `{"bytes": ..., "path": ...}`. A dict's bytes are used when present; otherwise its path, which must be absolute. A relative path is relative to an archive or download location, not to the working directory, so a dict with no bytes and a relative path is an `ImageInputError` at construction under `"raise"` and is dropped and counted under `"skip"`.
- A Hugging Face column with the `datasets.Image` feature is read without decoding (cast to `Image(decode=False)`, which writes no cache file and keeps the dataset's `select`/`shuffle`/`filter` row selection), and each image is kept as its encoded bytes. **Every encoded image of such a dataset is therefore held in memory**: 10,000 JPEG pages of about 100 KB each take about 1 GB. For data larger than memory, keep a column of file paths instead. `ClassLabel` columns keep their integer values; `dataset.features[column].int2str` gives the class names. An array image column (`Array2D`, `Array3D`, or nested lists of numbers) becomes numpy arrays of the dtype the feature declares, so a `uint8` column stays `uint8`. A column holding a list of images per row is rejected. Values are always read as Python values, whatever the dataset's `with_format` setting: a torch or numpy format would turn labels into 0-d tensors, and with `datasets` 4.5 the numpy, torch, and jax formatters fail once `torchvision` 0.27 or later is imported. `DatasetDict` (select a split) and `IterableDataset` are rejected.

## Splitting

`stratified_split`, `grouped_split`, and `temporal_split` return `(train, test)` datasets that partition the input and keep its sample order. `ImageDataset.subset(positions)` builds a dataset from explicit positions. `find_split_overlap` reports `sample_id` and `group_id` values shared between named splits. See the [leakage guide](../guides/leakage.md) for when to use which.

## Label handling

`embedfit.data.LabelMapping` maps external labels to class indices:

```text
external label ↔ class index
```

The labels keep the order in which they are given and are never sorted, so the position of a label is its class index. A saved model stores them in `labels.json`, in class-index order, together with the positive label. The positive class is always named explicitly with `positive_label`; it is never derived from label order.

## Image handling

Convert only at data/preprocessing boundary: `embedfit.data.to_rgb_tensor` maps every accepted input form to a float32 RGB tensor `[3, H, W]` in [0, 1] at native resolution, and backbone adapters take it from there ([ADR 0007](../adrs/0007-canonical-image-tensor.md)).

Avoid loading every file into memory when dataset stores paths.

## Missing and corrupt files

A path that names a missing or undecodable file raises `ImageInputError` when the image is encoded, not when the dataset is created. The message names neither the file nor its path, and the error has no chained cause, because file names can carry identifiers.

`on_corrupt_image="skip"` (on `ImageDataset`, `from_lists`, `from_pandas`, and `from_hf_dataset`) drops such records instead. It fully decodes every file-path and bytes image once, when the dataset is built, so the dataset's length and positions are fixed before any sampler indexes them. A record is dropped when its file is missing, its data cannot be identified by Pillow, is truncated, a Pillow plugin fails to parse it, or it has more than twice Pillow's decompression-bomb pixel limit (`PIL.Image.MAX_IMAGE_PIXELS`; raise the limit to keep large trusted images). Between one and two times the limit Pillow only emits its `DecompressionBombWarning`, and the image is kept. A path that names a directory or a file without read permission still raises `ImageInputError`: that is a setup error, not a corrupt image. A readable file that `to_rgb_tensor` rejects by policy, such as a multi-frame file, is kept and raises when encoded. In-memory images are never dropped. To trace kept records back to their source rows, map a `sample_id` column.

`ImageDataset.num_skipped` is the number of records dropped when the dataset was built (datasets derived by `subset`, slicing, or splitting report 0), and one warning gives the count and the total: `Skipped 3 of 7 records whose image could not be read`. Neither names a file, path, or label. Under the default `"raise"`, no image is read at construction and an image with more than twice the decompression-bomb limit raises `ImageInputError` when encoded.

## Grayscale

`to_rgb_tensor` replicates single-channel input (grayscale PIL images, `[H, W]` and one-channel arrays and tensors) to three RGB channels, so every backbone receives RGB.

## Multi-frame files

A multi-frame image, such as a multi-page TIFF or an animated GIF, raises `ImageInputError` rather than having one frame selected silently. Select a single frame before passing it in.
