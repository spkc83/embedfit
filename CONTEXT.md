# Project Context

## Problem

Classical image fine-tuning is often data-hungry. In financial document-risk problems, high-quality fraud labels can be scarce, delayed, expensive, and highly imbalanced.

The project should test and operationalize a vision analogue of the SetFit pattern:

1. start from a strong pretrained encoder;
2. use a small labeled set to improve the embedding geometry;
3. freeze the adapted encoder;
4. train a lightweight classifier on embeddings.

The learned representation should also be reusable for nearest-neighbor retrieval, clustering, duplicate-pattern detection, and novelty diagnostics.

## Reference use case: check deposit risk

Illustrative risk categories may include:

- legitimate
- duplicate presentment
- counterfeit check
- altered amount
- altered payee
- washed check
- forged signature
- synthetic check
- other fraud / suspicious item

These are **examples only**. The core framework must never require this taxonomy.

## Why embeddings matter

A binary classifier can miss operational context.

An embedding model can support:

- supervised classification
- similarity to prior fraudulent items
- retrieval of related items
- recurring template detection
- fraud-pattern clustering
- prototype comparison
- drift diagnostics

## Why ImageNet backbones are allowed

ImageNet-pretrained models are valid baseline backbones because they supply general visual representations.

However, the architecture must not assume ImageNet supervision is optimal. It should support self-supervised and foundation-model encoders such as DINO-family models and future domain-adapted checkpoints.

## Important domain constraints

### Rare events

Accuracy can be misleading. PR-AUC, fixed-FPR recall, investigation-capacity metrics, and dollar-weighted capture are more useful.

### High resolution

A complete check downscaled to 224×224 can destroy fine evidence. The framework must not hard-code one image resolution.

### Leakage risk

Potential leakage can arise from:

- scans of the same physical check in multiple splits;
- same fraud ring/template in train and test;
- post-event labels or return codes;
- temporal information unavailable at decision time.

### Sensitive data

Check images can contain account/routing numbers, names, addresses, signatures, and other sensitive information. The framework must default to not logging images or metadata.

## Product boundary

EmbedFit is a machine-learning framework, not a fraud decision system.

It produces model scores, embeddings, predictions, and evaluation artifacts. Business rules and final deposit decisions are out of scope.
