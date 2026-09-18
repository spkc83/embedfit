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
