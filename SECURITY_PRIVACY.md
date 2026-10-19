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
- serialize arbitrary metadata into model bundles. `save_pretrained(training_metadata=...)` writes whatever the caller passes; `Trainer.save_model` passes only the training arguments (without `output_dir`) and the training summary: mode, sample and step counts, loss curves, and best epoch;
- include raw metadata in exception strings;
- send telemetry.

## Logging

Safe examples:

```text
Loaded 12,410 training samples
Embedding dimension: 768
Refreshed 400 training embeddings for hard-negative mining
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

Benchmark scripts in `benchmarks/` may use public research datasets. They download the data at run time into the repository's git-ignored `.cache/` folder, and keep intermediate run output in the git-ignored `.runs/` folder; only aggregate metrics are committed.

## How the library meets these requirements

| Requirement | Mechanism |
| --- | --- |
| No image content or raw arrays in logs | Log messages are constant templates whose arguments are counts, mode and backbone names, seeds, and aggregate metrics. `tests/unit/test_privacy_audit.py` parses every module and fails if library code prints, writes to the console, warns, logs a computed message, or passes a logger anything other than a count, a mode or backbone name, an aggregate, a type name, or a library exception whose message the audit has already checked; its own tests confirm that it rejects leaks such as `logger.info("%s", self.metadata)`. |
| No metadata or identifiers in logs or exceptions | `ImageSample` and `ImageDataset` reprs show field presence and counts only. Validation errors name record positions, key names, types, shapes, and counts. `SplitOverlapReport` and `DataLeakageError` give counts; the overlapping identifiers are available only programmatically. The `skip` corrupt-image policy reports only how many records it dropped (`num_skipped` and one count-only warning), never which. Dataset-adapter errors name fields, column names, and row positions, never a DataFrame index or a value. |
| No values taken from data in exception strings | Image errors describe type, shape, and dtype, and never file paths or the content of encoded bytes, because file names can embed identifiers; they are raised without a chained cause, so the path does not reach a traceback or `logger.exception` either. The audit also fails if an exception message interpolates a sample field other than by type, shape, or count. An unknown label is reported by its type together with the model's configured labels. `RelationshipRule` errors name the field, never its values. |
| No training examples in checkpoints | A model bundle holds weights, configuration, label names, and aggregate training metadata (arguments and loss curves, without `output_dir` and `checkpoint_dir`). Nothing is pickled, so loading a bundle executes no code from it (ADR 0009). Representation checkpoints are the same bundles, with an index of epochs and losses (ADR 0016); hard-negative mining embeddings are never written. A bundle's top directory is created owner-only (mode `0700`, as the staging directory it is renamed from); the files in it and `checkpoints.json` get ordinary file permissions (`0666` minus the umask). |
| Metadata is never a model feature | `metadata` is stored as a read-only copy and read only by an explicitly configured `RelationshipRule`, to decide which pairs are similar; it never reaches the backbone or the head. |
| No telemetry | The library makes no network calls except a backbone provider downloading a checkpoint that was explicitly requested. The default test suite blocks internet sockets, so any other network use fails a test. |
| Synthetic examples only | `examples/` generates every image. `tests/integration/test_examples.py` parses each example (and their shared module) and fails on calls that read files, datasets, or URLs, and on an example that does not call an image generator. It is a guard against accidents, not proof against a determined bypass. |

Two things remain the caller's responsibility. Class label names and the opaque `id` of a retrieval result do appear in messages, reprs, and saved bundles, because results are unusable without them; do not encode sensitive values in label names or identifiers. Embeddings and scores computed from sensitive images are themselves sensitive.
