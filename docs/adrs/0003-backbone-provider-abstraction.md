# ADR 0003 — Backbone Provider Adapters Own Preprocessing

Status: Accepted

## Decision

Use provider adapters for torchvision, timm, and Hugging Face.

Each adapter owns:

- checkpoint construction
- image preprocessing
- feature extraction
- pooling
- provider metadata

## Rationale

Pretrained checkpoints have model-specific preprocessing. A global ImageNet transform is incorrect for a generic framework.

## Consequences

Core `EmbedFitModel` must not assume resolution or normalization constants.
