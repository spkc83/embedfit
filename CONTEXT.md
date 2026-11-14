# Project Context

## Problem

Classical image fine-tuning is often data-hungry. In rare-positive document and image risk review, high-quality positive labels can be scarce, delayed, expensive, and highly imbalanced.

The project should test and operationalize a vision analogue of the SetFit pattern:

1. start from a strong pretrained encoder;
2. use a small labeled set to improve the embedding geometry;
3. freeze the adapted encoder;
4. train a lightweight classifier on embeddings.

The learned representation should also be reusable for nearest-neighbor retrieval, clustering, duplicate-pattern detection, and novelty diagnostics.

## Reference use case: rare-positive document and image risk review

A reviewer must find a small number of positive items among many negatives, learning from only a few labelled positive examples. Illustrative categories may include:

- negative
- duplicate submission
- template reuse
- tampered region
- edited field
- synthetic or generated image
- other positive

These are **examples only**. The core framework must never require this taxonomy.

## Why embeddings matter

A binary classifier can miss operational context.

An embedding model can support:

- supervised classification
- similarity to prior positive items
- retrieval of related items
- recurring template detection
- positive-pattern clustering
- prototype comparison
- drift diagnostics

## Why ImageNet backbones are allowed

ImageNet-pretrained models are valid baseline backbones because they supply general visual representations.

However, the architecture must not assume ImageNet supervision is optimal. It should support self-supervised and foundation-model encoders such as DINO-family models and future domain-adapted checkpoints.

## Important domain constraints

### Rare events

Accuracy can be misleading. PR-AUC, fixed-FPR recall, review-capacity metrics, and value-weighted capture (`dollar_recall`) are more useful.

### High resolution

A complete document downscaled to 224×224 can destroy fine evidence. The framework must not hard-code one image resolution.

### Leakage risk

Potential leakage can arise from:

- scans of the same physical item in multiple splits;
- same campaign/template in train and test;
- post-event labels or outcome codes;
- temporal information unavailable at decision time.

### Sensitive data

Document images can contain identifiers, names, addresses, signatures, and other sensitive information. The framework must default to not logging images or metadata.

## Product boundary

EmbedFit is a machine-learning framework, not a decision system.

It produces model scores, embeddings, predictions, and evaluation artifacts. Business rules and final review decisions are out of scope.
