import math

import pytest
import torch
import torch.nn.functional as F

from embedfit.exceptions import ConfigurationError
from embedfit.losses import SupConLoss, available_losses, create_loss, register_loss, registry


def reference_loss(embeddings, targets, temperature):
    """Direct transcription of the SupCon formula with explicit loops."""
    z = F.normalize(embeddings.double(), dim=1)
    terms = []
    for i in range(len(z)):
        positives = [p for p in range(len(z)) if p != i and targets[p] == targets[i]]
        if not positives:
            continue
        denominator = sum(
            math.exp(float(z[i] @ z[a]) / temperature) for a in range(len(z)) if a != i
        )
        terms.append(
            -sum(
                math.log(math.exp(float(z[i] @ z[p]) / temperature) / denominator)
                for p in positives
            )
            / len(positives)
        )
    return sum(terms) / len(terms)


def test_hand_computed_two_pairs():
    # Two orthogonal pairs: each anchor sees one positive (similarity 1) and two negatives
    # (similarity 0), so every term is -log(e^{1/t} / (e^{1/t} + 2)).
    embeddings = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    targets = torch.tensor([0, 0, 1, 1])
    expected = -math.log(math.exp(1 / 0.5) / (math.exp(1 / 0.5) + 2))
    assert SupConLoss(temperature=0.5)(embeddings, targets).item() == pytest.approx(expected)


@pytest.mark.parametrize("temperature", [0.07, 0.5, 1.0])
def test_matches_loop_reference_on_random_batches(temperature):
    generator = torch.Generator().manual_seed(0)
    for _ in range(5):
        embeddings = torch.randn(12, 6, generator=generator)
        targets = torch.randint(0, 4, (12,), generator=generator)
        loss = SupConLoss(temperature)(embeddings, targets).item()
        assert loss == pytest.approx(reference_loss(embeddings, targets, temperature), rel=1e-5)


def test_anchor_without_positive_is_a_negative_but_adds_no_term():
    embeddings = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
    targets = torch.tensor([0, 0, 1, 1, 2])  # Class 2 is alone.
    loss = SupConLoss(0.3)(embeddings, targets).item()
    assert loss == pytest.approx(reference_loss(embeddings, targets, 0.3), rel=1e-5)
    without_singleton = SupConLoss(0.3)(embeddings[:4], targets[:4]).item()
    assert loss != pytest.approx(without_singleton)  # It still acts as a negative.


def test_no_positive_pair_anywhere_is_an_error():
    with pytest.raises(ConfigurationError, match="class-balanced batch sampler"):
        SupConLoss()(torch.randn(3, 4), torch.tensor([0, 1, 2]))


def test_well_separated_classes_score_lower_than_mixed_ones():
    targets = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    mixed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.1, 1.0]])
    assert SupConLoss(0.1)(separated, targets) < SupConLoss(0.1)(mixed, targets)


def test_gradients_are_finite_and_stable_at_extreme_similarity():
    # Near-duplicate vectors with a tiny temperature push logits to +-1e4.
    embeddings = (torch.ones(8, 16) + 1e-4 * torch.randn(8, 16)).requires_grad_()
    loss = SupConLoss(temperature=1e-4)(embeddings, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(embeddings.grad).all()


def test_scale_invariance_comes_from_normalisation():
    embeddings = torch.randn(6, 5, generator=torch.Generator().manual_seed(2))
    targets = torch.tensor([0, 0, 0, 1, 1, 1])
    assert SupConLoss()(embeddings, targets).item() == pytest.approx(
        SupConLoss()(embeddings * 37.0, targets).item(), rel=1e-5
    )
    raw = SupConLoss(normalize=False)
    assert raw(embeddings, targets).item() != pytest.approx(raw(embeddings * 2, targets).item())


def test_multi_view_input_treats_views_as_positives():
    views = torch.randn(3, 2, 4, generator=torch.Generator().manual_seed(3))
    targets = torch.tensor([0, 1, 2])  # No two samples share a class: views are the positives.
    flat, repeated = views.reshape(6, 4), torch.tensor([0, 0, 1, 1, 2, 2])
    assert SupConLoss(0.2)(views, targets).item() == pytest.approx(
        SupConLoss(0.2)(flat, repeated).item()
    )


def test_double_precision_is_preserved_and_passes_gradcheck():
    embeddings = torch.randn(
        6, 4, dtype=torch.float64, requires_grad=True, generator=torch.Generator().manual_seed(5)
    )
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    assert SupConLoss(0.5)(embeddings, targets).dtype == torch.float64
    assert torch.autograd.gradcheck(lambda e: SupConLoss(0.5)(e, targets), (embeddings,))


def test_error_reports_the_shapes_the_caller_passed():
    with pytest.raises(ConfigurationError, match=r"got \(3, 2, 4\) and \(2,\)"):
        SupConLoss()(torch.randn(3, 2, 4), torch.tensor([0, 1]))


def test_half_precision_input_is_computed_in_float32():
    embeddings = torch.randn(6, 8, generator=torch.Generator().manual_seed(4))
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = SupConLoss()(embeddings.to(torch.bfloat16), targets)
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(SupConLoss()(embeddings, targets).item(), rel=0.05)


@pytest.mark.parametrize(
    "embeddings, targets, match",
    [
        (torch.randn(4, 3), torch.tensor([0, 0, 1]), r"\[B, D\]"),
        (torch.randn(4), torch.tensor([0, 0, 1, 1]), r"\[B, D\]"),
        (torch.randn(4, 3), torch.tensor([0.0, 0.0, 1.0, 1.0]), "integer class indices"),
        (torch.randn(4, 3), torch.tensor([True, True, False, False]), "integer class indices"),
    ],
)
def test_invalid_inputs(embeddings, targets, match):
    with pytest.raises(ConfigurationError, match=match):
        SupConLoss()(embeddings, targets)


@pytest.mark.parametrize("temperature", [0, -0.1, "0.1", True, None])
def test_invalid_temperature(temperature):
    with pytest.raises(ConfigurationError, match="temperature"):
        SupConLoss(temperature)


def test_registry(monkeypatch):
    monkeypatch.setattr(registry, "_LOSSES", dict(registry._LOSSES))
    assert "supcon" in available_losses()
    assert create_loss("supcon", temperature=0.2).temperature == 0.2
    with pytest.raises(ConfigurationError, match="Unknown loss 'magic'.*supcon"):
        create_loss("magic")
    with pytest.raises(ConfigurationError, match=r"does not accept the options \['margin'\]"):
        create_loss("supcon", margin=1.0)
    with pytest.raises(ConfigurationError, match="already registered"):
        register_loss("supcon", SupConLoss)
    register_loss("warm", lambda: SupConLoss(temperature=1.0))
    assert create_loss("warm").temperature == 1.0
