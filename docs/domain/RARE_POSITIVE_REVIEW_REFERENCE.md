# Rare-Positive Document and Image Risk Review: Reference Use Case

This file provides domain context for examples and tests. It must not dictate generic library semantics.

In this use case, a reviewer must find a small number of known positive items among many negatives. Positive labels are scarce, they can arrive late, and only a limited number of items can be reviewed.

## Potential labels

Illustrative only:

- negative
- duplicate submission
- template reuse
- tampered region
- edited field
- synthetic or generated image
- other positive

## Potential relationship fields

- physical item ID
- template ID
- source ID
- submitting entity ID
- suspected campaign ID
- case or event ID
- intake channel

These can be used for grouping or sampling, but the library must treat them as arbitrary relationship fields.

## Why high resolution matters

Relevant visual evidence can be small:

- handwriting strokes
- small printed codes
- edits inside a field
- signatures and stamps
- scanner/compression artifacts
- repeated backgrounds or templates

Therefore do not assume a full document should always be compressed to 224×224.

## Suggested future representation

```text
full image embedding
+ region crop embeddings (one per field or area of interest)
→ learned or simple fusion
→ final item embedding
```

Automatic field detection is outside v0.1.

## Operational terminology

Organisation-specific definitions and operational labels vary. Do not encode event or outcome semantics into generic library code.

## Decisioning warning

A visual score is one feature in a broader review process. Similarity to prior positives is a signal, not proof.
