"""Per-part learning rates: the optimizer's parameter groups (ADR 0020)."""

import pytest
import torch
from torch import nn

from embedfit import EmbedFitModel, TrainingArguments
from embedfit.exceptions import ConfigurationError
from embedfit.representation import _parameter_groups

RATE = 1e-3


def make_backbone():
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=["a", "b"], embedding_dim=8).backbone


def stem_and_tail_backbone():
    """Four parameterised layers: a stem, two blocks, and a tail (ADR 0015)."""
    backbone = make_backbone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        backbone.features.insert(1, nn.BatchNorm2d(8))
    blocks = backbone.parameter_blocks()  # conv, batch norm, conv, projection
    backbone.parameter_blocks = lambda: blocks[1:3]
    return backbone


def head():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return nn.Linear(8, 2)


def rates_by_name(backbone, extra, groups):
    names = {id(p): n for n, p in backbone.named_parameters()}
    names.update({id(p): f"extra.{n}" for n, p in nn.ModuleList(extra).named_parameters()})
    return {names[id(p)]: group["lr"] for group in groups for p in group["params"]}


def args(**kwargs):
    return TrainingArguments(training_mode="end_to_end", learning_rate=RATE, **kwargs)


def test_the_defaults_give_one_group_of_every_trainable_parameter_in_order():
    backbone, classifier = make_backbone(), head()
    backbone.unfreeze(last_blocks=2)
    groups = _parameter_groups(backbone, [classifier], args())
    trainable = [p for p in nn.ModuleList([backbone, classifier]).parameters() if p.requires_grad]
    assert len(groups) == 1 and groups[0]["lr"] == RATE
    assert [id(p) for p in groups[0]["params"]] == [id(p) for p in trainable]
    assert len(trainable) < len(list(backbone.parameters())) + 2  # The first block is frozen.


def test_a_decay_of_one_and_a_head_rate_equal_to_the_rate_change_nothing():
    backbone, classifier = make_backbone(), head()
    default = _parameter_groups(backbone, [classifier], args())
    same = _parameter_groups(
        backbone, [classifier], args(head_learning_rate=RATE, layer_lr_decay=1.0)
    )
    assert [g["lr"] for g in same] == [RATE]
    assert [id(p) for p in same[0]["params"]] == [id(p) for p in default[0]["params"]]


def test_the_head_rate_applies_to_the_extra_modules_only():
    backbone, classifier = make_backbone(), head()
    groups = _parameter_groups(backbone, [classifier], args(head_learning_rate=0.1))
    assert [g["lr"] for g in groups] == [RATE, 0.1]
    assert [id(p) for p in groups[1]["params"]] == [id(p) for p in classifier.parameters()]


def test_layer_decay_scales_blocks_stem_and_tail():
    backbone, classifier = stem_and_tail_backbone(), head()
    groups = _parameter_groups(backbone, [classifier], args(layer_lr_decay=0.5))
    rates = rates_by_name(backbone, [classifier], groups)
    expected = {
        "features.0": RATE * 0.25,  # Stem: decay ** n with n = 2 blocks.
        "features.1": RATE * 0.5,  # Block 0.
        "features.3": RATE,  # Block 1, the last.
        "projection": RATE,  # Tail.
        "extra.0": RATE,  # No head rate: the extra modules train at learning_rate.
    }
    for name, rate in rates.items():
        assert rate == pytest.approx(expected[name.rsplit(".", 1)[0]]), name
    assert len(rates) == len(list(backbone.parameters())) + 2
    assert sorted(g["lr"] for g in groups) == pytest.approx([RATE * 0.25, RATE * 0.5, RATE])


def test_layer_decay_counts_every_block_but_leaves_frozen_ones_out():
    backbone = make_backbone()  # Blocks: conv, conv, projection.
    backbone.unfreeze(last_blocks=2)
    groups = _parameter_groups(backbone, [], args(layer_lr_decay=0.5, head_learning_rate=0.1))
    rates = rates_by_name(backbone, [], groups)
    assert {name.rsplit(".", 1)[0]: rate for name, rate in rates.items()} == {
        "features.2": RATE * 0.5,
        "projection": RATE,
    }


def test_a_parameter_shared_with_the_backbone_is_grouped_once_at_the_backbone_rate():
    backbone = make_backbone()
    classifier = nn.Linear(16, 8)
    classifier.weight = backbone.projection.weight  # Tied: one tensor in both modules.
    options = args(head_learning_rate=0.1, layer_lr_decay=0.5)
    groups = _parameter_groups(backbone, [classifier], options)
    grouped = [id(p) for group in groups for p in group["params"]]
    assert len(grouped) == len(set(grouped)) == len(list(backbone.parameters())) + 1
    rates = {id(p): group["lr"] for group in groups for p in group["params"]}
    assert rates[id(backbone.projection.weight)] == RATE  # The tail block, not the head.
    assert rates[id(classifier.bias)] == 0.1


def test_block_positions_place_stem_blocks_and_tail():
    backbone = stem_and_tail_backbone()
    positions, num_blocks = backbone.block_positions()
    assert num_blocks == 2
    assert {name: positions[name] for name, _ in backbone.named_parameters()} == {
        "features.0.weight": -1,
        "features.0.bias": -1,
        "features.1.weight": 0,
        "features.1.bias": 0,
        "features.3.weight": 1,
        "features.3.bias": 1,
        "projection.weight": 2,
        "projection.bias": 2,
    }


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"training_mode": "end_to_end", "head_learning_rate": 0}, "head_learning_rate must be"),
        ({"training_mode": "end_to_end", "head_learning_rate": -1e-3}, "head_learning_rate"),
        ({"training_mode": "end_to_end", "head_learning_rate": True}, "head_learning_rate"),
        ({"training_mode": "end_to_end", "head_learning_rate": "1e-3"}, "head_learning_rate"),
        ({"training_mode": "end_to_end", "head_learning_rate": float("inf")}, "must be finite"),
        ({"training_mode": "end_to_end", "head_learning_rate": float("nan")}, "must be finite"),
        ({"learning_rate": float("inf")}, "learning_rate must be finite"),
        ({"learning_rate": float("nan")}, "learning_rate must be > 0"),
        ({"training_mode": "supcon", "head_learning_rate": 1e-3}, "trains neither"),
        ({"training_mode": "pair", "head_learning_rate": 1e-3}, "trains neither"),
        ({"training_mode": "frozen", "head_learning_rate": 1e-3}, "trains neither"),
        ({"layer_lr_decay": 0}, "layer_lr_decay must be in"),
        ({"layer_lr_decay": 1.5}, "layer_lr_decay must be in"),
        ({"layer_lr_decay": -0.5}, "layer_lr_decay must be in"),
        ({"layer_lr_decay": float("nan")}, "layer_lr_decay must be in"),
        ({"layer_lr_decay": True}, "layer_lr_decay must be in"),
        ({"training_mode": "frozen", "layer_lr_decay": 0.5}, "layer_lr_decay only applies"),
    ],
)
def test_invalid_values_fail_at_construction(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        TrainingArguments(**kwargs)


def test_valid_values_are_kept():
    supcon = TrainingArguments(projection_dim=4, head_learning_rate=1e-3, layer_lr_decay=1)
    assert (supcon.head_learning_rate, supcon.layer_lr_decay) == (1e-3, 1)
    pair = TrainingArguments(training_mode="pair", layer_lr_decay=0.7)
    assert (pair.head_learning_rate, pair.layer_lr_decay) == (None, 0.7)
