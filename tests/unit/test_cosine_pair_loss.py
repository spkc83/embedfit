import math

import pytest
import torch

from embedfit.exceptions import ConfigurationError
from embedfit.losses import CosinePairLoss, create_loss


def test_hand_computed():
    left = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    right = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])  # cosines 1, 0, 1/sqrt(2)
    targets = torch.tensor([1, 0, 1])
    expected = ((1 - 1) ** 2 + (0 - 0) ** 2 + (1 / math.sqrt(2) - 1) ** 2) / 3
    assert CosinePairLoss()(left, right, targets).item() == pytest.approx(expected)
    # Treating the third pair as a negative instead: (1/sqrt(2) - 0)^2 = 1/2.
    assert CosinePairLoss()(left, right, torch.tensor([1, 0, 0])).item() == pytest.approx(0.5 / 3)


def test_weights_give_a_weighted_mean():
    left = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    right = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # cosines 0 and 1, both labelled positive.
    targets = torch.tensor([1, 1])
    loss = CosinePairLoss()(left, right, targets, torch.tensor([3.0, 1.0]))
    assert loss.item() == pytest.approx((3 * 1.0 + 1 * 0.0) / 4)


def test_negative_target_moves_the_goal_for_negatives_only():
    left, right = torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]])  # cosine -1
    assert CosinePairLoss(negative_target=-1.0)(left, right, torch.tensor([0])).item() == 0.0
    assert CosinePairLoss()(left, right, torch.tensor([0])).item() == pytest.approx(1.0)


def test_gradients_pull_positives_together_and_push_negatives_apart():
    left = torch.tensor([[1.0, 0.2], [1.0, 0.2]], requires_grad=True)
    right = torch.tensor([[0.6, 0.8], [0.6, 0.8]])
    before = torch.nn.functional.cosine_similarity(left, right).detach()
    CosinePairLoss()(left, right, torch.tensor([1, 0])).backward()
    stepped = (left - 0.1 * left.grad).detach()
    after = torch.nn.functional.cosine_similarity(stepped, right)
    assert after[0] > before[0] and after[1] < before[1]
    assert torch.isfinite(left.grad).all()


def test_precision_handling():
    left = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    right = torch.randn(4, 6, dtype=torch.float64)
    targets = torch.tensor([1, 0, 1, 0])
    assert torch.autograd.gradcheck(lambda x: CosinePairLoss()(x, right, targets), (left,))
    assert (
        CosinePairLoss()(left.to(torch.bfloat16), right.to(torch.bfloat16), targets).dtype
        == torch.float32
    )


@pytest.mark.parametrize(
    "left, right, targets, weights, match",
    [
        (torch.ones(2, 3), torch.ones(3, 3), torch.tensor([1, 0]), None, r"\[P, D\]"),
        (torch.ones(2, 3), torch.ones(2, 3), torch.tensor([1]), None, r"\[P, D\]"),
        (torch.ones(0, 3), torch.ones(0, 3), torch.tensor([]), None, "empty"),
        (torch.ones(2, 3), torch.ones(2, 3), torch.tensor([1, 2]), None, "only 0 and 1"),
        (
            torch.ones(2, 3),
            torch.ones(2, 3),
            torch.tensor([1, 0]),
            torch.tensor([1.0, 0.0]),
            "weights must be positive",
        ),
    ],
)
def test_invalid_inputs(left, right, targets, weights, match):
    with pytest.raises(ConfigurationError, match=match):
        CosinePairLoss()(left, right, targets, weights)


@pytest.mark.parametrize("value", [1, 1.5, -1.1, "0", True])
def test_invalid_negative_target(value):
    with pytest.raises(ConfigurationError, match="negative_target"):
        CosinePairLoss(negative_target=value)


def test_registered():
    assert isinstance(create_loss("cosine_pair", negative_target=-0.5), CosinePairLoss)
