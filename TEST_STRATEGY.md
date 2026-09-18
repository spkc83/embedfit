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

### Provider integration

Marked tests:

```python
@pytest.mark.network
@pytest.mark.slow
```

Examples:

- torchvision checkpoint if packaged weights are not required
- timm model construction
- Hugging Face DINO-family loading

Default CI excludes these.

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

- no self-pairs unless configured
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
