# Dataset and Data Model

## Canonical sample

```python
@dataclass
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

Internally maintain a stable mapping:

```text
external label ↔ class index
```

Serialize the mapping.

Never derive positive-class semantics from sort order.

## Image handling

Convert only at data/preprocessing boundary: `embedfit.data.to_rgb_tensor` maps every accepted input form to a float32 RGB tensor `[3, H, W]` in [0, 1] at native resolution, and backbone adapters take it from there ([ADR 0007](../adrs/0007-canonical-image-tensor.md)).

Avoid loading every file into memory when dataset stores paths.

## Corrupt image policy

Configurable:

- `raise` default
- `skip`

Skipping must increment/report a count.

## Grayscale

RGB-trained backbones may convert grayscale to RGB by channel replication.

This behavior must be explicit in the provider adapter/docs.

## Multi-frame files

v0.1 should reject or clearly warn on ambiguous multi-frame inputs rather than silently selecting arbitrary pages.
