# ADR 0006 — Minimal Core Dependencies; Offline Test Utilities Ship in the Package

Status: Accepted

## Decision

1. The core install depends only on `torch`, `numpy`, `scikit-learn`, and `pillow`. Backbone providers (`torchvision`, `timm`, `transformers`) and dataset adapters (`datasets`, `pandas`) are optional extras. Using a provider whose package is missing must raise an actionable error naming the extra to install; it must never fall back to another provider.
2. `DummyBackbone` lives in `embedfit.backbones` and the synthetic image generator lives in `embedfit.testing`. Both are part of the installed package rather than the `tests/` tree.

## Rationale

Every supported workflow needs tensors, arrays, the logistic-regression head, and image decoding, so those four packages are unavoidable. Each provider is needed only by users of that provider, and the three together are heavy and version-sensitive.

`DummyBackbone` and the synthetic generator are needed outside the test suite: repository examples must run on synthetic data (`SECURITY_PRIVACY.md`), the few-shot benchmark needs an offline smoke path, and downstream users extending EmbedFit with custom heads, losses, or samplers need the same fast fixtures.

## Consequences

`DummyBackbone` is a real trainable network that follows the same backbone contract as provider adapters, so tests using it exercise genuine gradient flow (`AGENTS.md`, prohibited shortcuts). It carries no pretrained knowledge, and its documentation says it is unsuitable for real tasks.

`embedfit.testing` must stay domain-neutral and free of provider dependencies.

The default test suite runs with only the core dependencies installed.
