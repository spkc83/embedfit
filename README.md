# EmbedFit

> **Status:** implementation bootstrap / specification repository  
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

Then execute work from [`TASKS.md`](TASKS.md).

## Intended developer loop

```bash
cd embedfit-init
git init
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,vision]"
pytest
```

The repository intentionally starts with architecture/specification files and a minimal source skeleton. The implementation agent should build incrementally and keep the specs synchronized with code.

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
