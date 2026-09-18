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

The exact implementation may differ, but semantics should remain.

## Label handling

Internally maintain a stable mapping:

```text
external label ↔ class index
```

Serialize the mapping.

Never derive positive-class semantics from sort order.

## Image handling

Convert only at data/preprocessing boundary.

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
