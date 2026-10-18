# Test Strategy

## Principles

1. Core tests must run offline.
2. Core tests must run on CPU.
3. Tests should exercise real framework behavior, not only mocks.
4. Large pretrained checkpoints belong in optional integration tests.
5. Numerical behavior should be tested on tiny deterministic problems.

## Test categories

### Unit

Targets:

- config validation
- seed/device helpers
- SupCon math
- pair sampling
- balanced sampling
- augmentation profiles, per-draw seeding, and the augmentation registry
- heads
- metrics
- retrieval
- serialization metadata

### Integration

Use `DummyBackbone` plus small synthetic image datasets.

Test:

```text
dataset
→ trainer
→ representation training
→ head fit
→ predict
→ save
→ reload
→ predict
```

### Provider adapters

Offline tests build torchvision, timm, and Transformers models with random weights and compare EmbedFit's preprocessing with the providers' own pipelines. Each provider's test module skips when its optional extra is not installed (`pytest.importorskip`).

Only the tests that download a pretrained checkpoint are marked:

```python
@pytest.mark.network
@pytest.mark.slow
```

The `addopts` in `pyproject.toml` deselect both markers and pass `--strict-markers`. Run the checkpoint tests with `pytest -m network`. There is no CI configuration in the repository.

## Suite-wide enforcement

`tests/conftest.py` applies to every test:

- an autouse network guard: opening an IPv4 or IPv6 connection fails unless the test is marked `network`; Unix-domain sockets stay usable, because data-loader worker processes rely on them. `test_the_default_suite_cannot_reach_the_network` checks the guard itself;
- an autouse fixture that restores the Python, NumPy, and torch global RNG state after each test, so tests do not depend on their order;
- `torch.set_num_threads(1)`, for speed on tiny tensors and bit-for-bit reproducible tiny training runs.

`tests/unit/test_privacy_audit.py` parses every module under `src/embedfit` and checks what library code may print, log, and raise (`SECURITY_PRIVACY.md`). `tests/integration/test_examples.py` runs every example offline and checks its results.

## Synthetic dataset

Build simple classes with learnable visual structure.

Example:

- class 0: vertical bar
- class 1: horizontal bar
- class 2: diagonal bar

Add controlled noise.

This allows testing whether a tiny encoder/loss can actually learn.

## SupCon tests

Verify:

- finite scalar loss
- finite gradients
- zero/invalid positive-pair conditions handled
- loss decreases on a toy separable dataset
- same-label embeddings become more similar on a short deterministic experiment

## Pair sampler tests

Verify:

- no self-pairs
- bounded pair count
- class semantics
- deterministic seed behavior
- no O(N²) explosion in default mode

## Metrics tests

Use hand-computed toy arrays.

Every risk metric should include at least one manually verified expected value.

## Serialization

Round trip:

1. train tiny model/head;
2. save;
3. reload;
4. compare embeddings;
5. compare predictions;
6. compare class ordering;
7. compare probabilities.

## Determinism

When deterministic mode is enabled, same seed and data should yield equivalent tiny-run outputs within tolerance.

## Coverage

Target >=85% coverage for:

- model
- trainer
- losses
- heads
- sampling
- evaluation
- retrieval
- serialization
