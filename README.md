# EmbedFit

> **Status:** pre-alpha — see [What the package provides](#what-the-package-provides)  
> **Primary modality for v0.1:** images  
> **Reference use case:** few-shot check-fraud / deposit-risk classification and similarity retrieval  
> **Package name:** `embedfit` (provisional but recommended)

EmbedFit is a SetFit-inspired framework for adapting pretrained embedding models to supervised tasks with very small labeled datasets.

The v0.1 implementation focuses on vision:

```text
pretrained vision encoder
        ↓
few-shot representation adaptation
        ↓
task-specific image embeddings
        ↓
lightweight classifier / retrieval / anomaly signals
```

The initial financial-services use case is check deposit risk, including fraud and RDI/returned-item patterns, but **the core library must remain domain-neutral**.

## Start here

A coding agent should read these files in order:

1. [`AGENTS.md`](AGENTS.md)
2. [`CONTEXT.md`](CONTEXT.md)
3. [`SPEC.md`](SPEC.md)
4. [`ARCHITECTURE.md`](ARCHITECTURE.md)
5. [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
6. [`ACCEPTANCE_CRITERIA.md`](ACCEPTANCE_CRITERIA.md)
7. [`TEST_STRATEGY.md`](TEST_STRATEGY.md)
8. [`SECURITY_PRIVACY.md`](SECURITY_PRIVACY.md)

Then execute work from [`TASKS.md`](TASKS.md).

## What the package provides

EmbedFit is built milestone by milestone against [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md); [`TASKS.md`](TASKS.md) tracks each item. The capabilities under [Core v0.1 capabilities](#core-v01-capabilities) are the target, not the current feature set.

Importable today:

| Component | Location | Purpose |
| --- | --- | --- |
| `set_seed` | `embedfit.utils` | Seeds Python, NumPy, and torch (CPU + CUDA); optional deterministic mode |
| `resolve_device` | `embedfit.utils` | `"auto"` resolves CUDA → MPS → CPU; an explicit device never falls back silently |
| `EmbedFitError`, `ConfigurationError`, `ImageInputError` | `embedfit.exceptions` | Error hierarchy; messages describe structure, never image or metadata content |
| `DummyBackbone` | `embedfit.backbones` | Tiny trainable CNN with no checkpoint, for offline tests and examples |
| `make_bar_images` | `embedfit.testing` | Deterministic synthetic image classes with learnable structure |

`EmbedFitModel`, `Trainer`, and the other public objects in [`SPEC.md`](SPEC.md) §5 are not implemented yet.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest -q            # offline, CPU-only; excludes tests marked `network` or `slow`
ruff check .
ruff format --check .
mypy
python -m build
```

Provider tests that download checkpoints are opt-in: `pytest -m network`.

Optional extras install backbone providers and dataset adapters: `torchvision`, `timm`, `hf`, `datasets`, `pandas`, or `all`. The core install depends only on `torch`, `numpy`, `scikit-learn`, and `pillow` ([ADR 0006](docs/adrs/0006-core-dependencies-and-testing-utilities.md)).

Branching, commit, and documentation conventions are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Core v0.1 capabilities

- torchvision, timm, and Hugging Face vision backbones
- frozen-feature + logistic-regression baseline
- supervised contrastive adaptation
- pairwise contrastive adaptation
- optional triplet learning
- few-shot benchmark harness
- image embedding extraction
- nearest-neighbor retrieval
- fraud/risk-oriented metrics
- leakage-safe grouped and temporal evaluation helpers
- save/load model bundles
- CPU/offline test suite

## Non-goals for v0.1

- OCR
- document-layout transformers
- VLM/LLM reasoning
- object detection / segmentation
- automatic check-field detection
- production vector databases
- fraud decision-engine orchestration
- large-scale distributed training

## Naming

See [`PROJECT_NAMING.md`](PROJECT_NAMING.md). `EmbedFit` is recommended over `ImageSetFit` because it describes the abstraction rather than the first modality and avoids looking like an official SetFit package.
