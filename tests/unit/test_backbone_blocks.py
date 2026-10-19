"""Backbone blocks and partial unfreezing (ADR 0015), on the dummy backbone."""

import pytest
import torch
from torch import nn

from embedfit import EmbedFitModel
from embedfit.backbones import DummyBackbone, ImageBackbone
from embedfit.exceptions import ConfigurationError


def trainable(backbone):
    return {name for name, p in backbone.named_parameters() if p.requires_grad}


class _Stacked(ImageBackbone):
    """A stem, two blocks with a parameter between them, and a tail norm."""

    def __init__(self, with_blocks=True):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.first = nn.Conv2d(4, 4, 1)
        self.between = nn.Conv2d(4, 4, 1)
        self.second = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.norm = nn.BatchNorm2d(4)
        self.with_blocks = with_blocks

    identifier = "test/stacked"
    embedding_dim = 4

    def preprocess(self, image):
        raise NotImplementedError

    def forward_features(self, pixel_values):
        x = self.norm(self.second(self.between(self.first(self.stem(pixel_values)))))
        return x.mean(dim=(2, 3))

    def save_pretrained(self, path):
        raise NotImplementedError

    def parameter_blocks(self):
        if not self.with_blocks:
            return super().parameter_blocks()
        return [self._names_of(self.first), self._names_of(self.second)]


def test_dummy_blocks_partition_its_parameters_in_order(dummy_backbone):
    blocks = dummy_backbone.parameter_blocks()
    assert dummy_backbone.num_blocks == len(blocks) == 3
    flat = [name for block in blocks for name in block]
    assert flat == [name for name, _ in dummy_backbone.named_parameters()]
    assert blocks[-1] == ["projection.weight", "projection.bias"]


@pytest.mark.parametrize("last_blocks", [1, 2, 3])
def test_unfreeze_last_blocks_trains_only_those_blocks(dummy_backbone, last_blocks):
    dummy_backbone.freeze()
    dummy_backbone.unfreeze(last_blocks=last_blocks)
    expected = {n for block in dummy_backbone.parameter_blocks()[-last_blocks:] for n in block}
    assert trainable(dummy_backbone) == expected
    assert not dummy_backbone.is_frozen
    dummy_backbone.train()
    assert dummy_backbone.training


def test_stem_and_parameters_between_blocks_follow_registration_order():
    backbone = _Stacked()
    backbone.unfreeze(last_blocks=1)
    # `between` precedes the second block and joins it; the final norm is the tail.
    assert trainable(backbone) == {
        "between.weight",
        "between.bias",
        "second.0.weight",
        "second.0.bias",
        "second.1.weight",
        "second.1.bias",
        "norm.weight",
        "norm.bias",
    }
    backbone.unfreeze(last_blocks=2)
    assert trainable(backbone) == {
        name for name, _ in backbone.named_parameters() if not name.startswith("stem.")
    }


def test_whole_unfreeze_and_freeze_after_partial(dummy_backbone):
    dummy_backbone.unfreeze(last_blocks=1)
    dummy_backbone.unfreeze()
    assert trainable(dummy_backbone) == {n for n, _ in dummy_backbone.named_parameters()}
    dummy_backbone.unfreeze(last_blocks=1)
    dummy_backbone.freeze()
    assert trainable(dummy_backbone) == set() and dummy_backbone.is_frozen
    dummy_backbone.train()
    assert not dummy_backbone.training


@pytest.mark.parametrize("last_blocks", [0, 4, -1, True, 1.0])
def test_out_of_range_block_counts_are_rejected(dummy_backbone, last_blocks):
    before = trainable(dummy_backbone)
    with pytest.raises(ConfigurationError, match=r"\[1, 3\]"):
        dummy_backbone.unfreeze(last_blocks=last_blocks)
    assert trainable(dummy_backbone) == before


def test_a_backbone_without_blocks_cannot_be_partially_unfrozen():
    backbone = _Stacked(with_blocks=False)
    assert backbone.num_blocks == 0
    with pytest.raises(ConfigurationError, match="defines no blocks"):
        backbone.unfreeze(last_blocks=1)
    backbone.freeze()
    backbone.unfreeze()
    assert trainable(backbone) == {n for n, _ in backbone.named_parameters()}


def test_batch_norm_in_frozen_blocks_stays_in_eval_mode():
    backbone = _Stacked()
    frozen_norm = nn.BatchNorm2d(4)
    backbone.first.add_module("norm", frozen_norm)  # Part of the first block.
    backbone.unfreeze(last_blocks=1)
    backbone.train()
    assert backbone.training
    assert not frozen_norm.training  # In a frozen block.
    assert backbone.second[1].training and backbone.norm.training  # Trainable.
    # A parent module entering train mode, or encode() restoring it, applies the same rule.
    backbone.eval()
    nn.Sequential(backbone).train()
    assert not frozen_norm.training and backbone.norm.training


def test_whole_unfreeze_leaves_batch_norm_to_train_mode():
    backbone = _Stacked()
    backbone.unfreeze(last_blocks=1)
    backbone.unfreeze()
    backbone.train()
    assert all(m.training for m in backbone.modules())


def test_model_passes_last_blocks_through(dummy_backbone):
    model = EmbedFitModel(dummy_backbone, labels=["a", "b"])
    model.freeze_backbone()
    model.unfreeze_backbone(last_blocks=1)
    assert trainable(dummy_backbone) == {"projection.weight", "projection.bias"}
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in dummy_backbone.parameters())


def test_frozen_parameters_receive_no_gradient(dummy_backbone, bar_dataset):
    images, _ = bar_dataset
    dummy_backbone.unfreeze(last_blocks=1)
    pixels = torch.stack([dummy_backbone.preprocess(image) for image in images[:4]])
    dummy_backbone(pixels).sum().backward()
    for name, parameter in dummy_backbone.named_parameters():
        assert (parameter.grad is not None) == name.startswith("projection."), name


def test_dummy_counts_an_inserted_layer_with_parameters_as_a_block():
    backbone = DummyBackbone(seed=0)
    backbone.features.insert(1, nn.BatchNorm2d(8))
    assert backbone.num_blocks == 4
