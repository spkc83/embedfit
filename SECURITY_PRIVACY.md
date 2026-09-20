# Security and Privacy Requirements

## Threat model

Reference deployments may process financial document images containing:

- account numbers
- routing numbers
- signatures
- names
- addresses
- transaction amounts
- institution identifiers

Treat image and metadata content as potentially sensitive.

## Library defaults

The library must not:

- log image contents;
- log raw arrays/tensors;
- write training examples to checkpoints;
- serialize arbitrary metadata into model bundles;
- include raw metadata in exception strings;
- send telemetry.

## Logging

Safe examples:

```text
Loaded 12,410 training samples
Embedding dimension: 768
Skipped 2 corrupt images
Validation PR-AUC: 0.42
```

Unsafe:

```text
Failed row: {"account_number": "...", ...}
```

## Metadata

Metadata is never a model feature automatically.

Any structured-feature fusion must be an explicit future feature.

## Model artifacts

Model bundle may include:

- weights
- configs
- label names
- class mappings
- preprocessing information
- aggregate training metadata

It must not include examples or identifiers unless a user explicitly implements a custom extension.

## Example data

Repository examples must use generated/synthetic images only.

Do not commit real checks or realistic account/routing numbers.

## How the library meets these requirements

| Requirement | Mechanism |
| --- | --- |
| No image content or raw arrays in logs | Log messages are constant templates whose arguments are counts, mode and backbone names, seeds, and aggregate metrics. `tests/unit/test_privacy_audit.py` fails if library code prints, logs a computed message, or passes a new kind of value to a logger. |
| No metadata or identifiers in logs or exceptions | `ImageSample` and `ImageDataset` reprs show field presence and counts only. Validation errors name record positions, key names, types, shapes, and counts. `SplitOverlapReport` and `DataLeakageError` give counts; the overlapping identifiers are available only programmatically. |
| No values taken from data in exception strings | Image errors describe type, shape, and dtype, and never file paths, because file names can embed identifiers. An unknown label is reported by its type together with the model's configured labels. `RelationshipRule` errors name the field, never its values. |
| No training examples in checkpoints | A model bundle holds weights, configuration, label names, and aggregate training metadata (arguments and loss curves, without `output_dir`). Nothing is pickled, so loading a bundle executes no code from it (ADR 0009). |
| Metadata is never a model feature | `metadata` is stored as a read-only copy and read only by an explicitly configured `RelationshipRule`, to decide which pairs are similar; it never reaches the backbone or the head. |
| No telemetry | The library makes no network calls except a backbone provider downloading a checkpoint that was explicitly requested. |
| Synthetic examples only | `examples/` generates every image; `tests/integration/test_examples.py` checks that no example opens files or URLs. |

Two things remain the caller's responsibility. Class label names and the opaque `id` of a retrieval result do appear in messages, reprs, and saved bundles, because results are unusable without them; do not encode sensitive values in label names or identifiers. Embeddings and scores computed from sensitive images are themselves sensitive.
