"""Batch-norm statistics during representation training (ADR 0013)."""

import pytest
import torch
from torch import nn

from embedfit import EmbedFitModel, ImageDataset, TrainingArguments
from embedfit.exceptions import ConfigurationError
from embedfit.representation import train_representation
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def batch_norm_model():
    """The demo backbone with a batch-norm layer after its first convolution."""
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model.backbone.features.insert(1, nn.BatchNorm2d(8))
    model.unfreeze_backbone()
    return model


def batch_norm(model):
    return next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))


def args(mode, **kwargs):
    options = {
        "training_mode": mode,
        "num_epochs": 2,
        "batch_size": 6,
        "learning_rate": 1e-2,
        "num_workers": 0,
        **kwargs,
    }
    return TrainingArguments(**options)


@pytest.mark.parametrize("mode", ["end_to_end", "supcon", "pair"])
def test_running_statistics_stay_fixed_while_affine_parameters_train(mode):
    model = batch_norm_model()
    layer = batch_norm(model)
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    train_representation(model, dataset(6, 0), None, args(mode))
    for name in ("running_mean", "running_var", "num_batches_tracked"):
        assert torch.equal(layer.state_dict()[name], before[name]), name
    assert not torch.equal(layer.weight, before["weight"])
    assert not torch.equal(layer.bias, before["bias"])


def test_running_statistics_update_when_not_frozen():
    model = batch_norm_model()
    layer = batch_norm(model)
    before = layer.running_mean.clone()
    train_representation(model, dataset(6, 0), None, args("supcon", freeze_batch_norm=False))
    assert not torch.equal(layer.running_mean, before)
    assert layer.num_batches_tracked.item() > 0


@pytest.mark.parametrize(
    ("mode", "options"),
    [
        ("supcon", {}),
        # Mining re-encodes the training set at the start of every epoch, which restores the
        # backbone's train mode; batch norm must be put back into eval mode after it.
        ("pair", {"hard_negative_fraction": 0.5}),
    ],
)
def test_batch_norm_is_in_eval_mode_for_every_training_batch(mode, options):
    model = batch_norm_model()
    modes = []
    layer = batch_norm(model)
    layer.register_forward_hook(
        lambda module, *_: modes.append((module.training, model.backbone.training))
    )
    train_representation(model, dataset(6, 0), None, args(mode, **options))
    during_training = [bn for bn, backbone in modes if backbone]
    assert during_training and not any(during_training)


def test_freeze_batch_norm_must_be_a_bool():
    with pytest.raises(ConfigurationError, match="freeze_batch_norm must be a bool"):
        TrainingArguments(freeze_batch_norm=1)
