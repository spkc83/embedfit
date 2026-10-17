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

Records are read once when the dataset is created. A source that decodes images inside `__getitem__` is therefore held in memory; storing file paths as `image` keeps decoding lazy, because paths are opened only when a batch is encoded.

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

A `skip` policy that drops such files and reports how many were dropped is specified for v0.1 but not built ([release plan](../../plans/V0_1_RELEASE.md)).

## Grayscale

`to_rgb_tensor` replicates single-channel input (grayscale PIL images, `[H, W]` and one-channel arrays and tensors) to three RGB channels, so every backbone receives RGB.

## Multi-frame files

A multi-frame image, such as a multi-page TIFF or an animated GIF, raises `ImageInputError` rather than having one frame selected silently. Select a single frame before passing it in.
