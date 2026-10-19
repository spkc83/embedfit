"""Training only the last blocks of the backbone (ADR 0015)."""

import json
import logging

import pytest
import torch
from torch import nn

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.exceptions import ConfigurationError
from embedfit.representation import train_representation
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def make_model():
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32)


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


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.backbone.state_dict().items()}


@pytest.mark.parametrize("mode", ["supcon", "pair", "end_to_end"])
def test_only_the_last_blocks_change(mode, tmp_path):
    model = make_model()
    last = set(model.backbone.parameter_blocks()[-1])
    before = snapshot(model)
    trainer = Trainer(model, args(mode, unfreeze_last_blocks=1), dataset(6, 0))
    output = trainer.train()
    assert output.representation_steps > 0
    after = snapshot(model)
    for name, value in after.items():
        assert torch.equal(value, before[name]) != (name in last), name
    trainer.save_model(tmp_path)
    recorded = json.loads((tmp_path / "training_metadata.json").read_text())
    assert recorded["training_arguments"]["unfreeze_last_blocks"] == 1


def test_direct_training_honours_the_argument():
    model = make_model()
    model.unfreeze_backbone()  # Whole backbone trainable before the call.
    before = snapshot(model)
    train_representation(model, dataset(6, 0), None, args("supcon", unfreeze_last_blocks=1))
    changed = {n for n, v in snapshot(model).items() if not torch.equal(v, before[n])}
    assert changed == set(model.backbone.parameter_blocks()[-1])


def test_end_to_end_still_starts_from_a_linear_probe(caplog):
    model = make_model()
    with caplog.at_level(logging.INFO, logger="embedfit"):
        Trainer(model, args("end_to_end", unfreeze_last_blocks=2), dataset(6, 0)).train()
    assert "Starting from a linear probe" in caplog.text
    assert model.head_name == "linear" and model.head.is_fitted


def batch_norm_model(position):
    """The demo backbone with a batch-norm layer inserted into ``features``."""
    model = make_model()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model.backbone.features.insert(position, nn.BatchNorm2d(8 if position == 1 else 16))
    return model


def test_batch_norm_in_a_frozen_block_keeps_its_statistics_even_in_train_mode():
    model = batch_norm_model(1)  # Its own block, the second of four.
    layer = model.backbone.features[1]
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    train_representation(
        model, dataset(6, 0), None, args("supcon", freeze_batch_norm=False, unfreeze_last_blocks=2)
    )
    for name, value in layer.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_batch_norm_in_a_trained_block_follows_freeze_batch_norm():
    model = batch_norm_model(4)  # After the second convolution: the third of four blocks.
    layer = model.backbone.features[4]
    before = layer.running_mean.clone()
    train_representation(
        model, dataset(6, 0), None, args("supcon", freeze_batch_norm=False, unfreeze_last_blocks=2)
    )
    assert not torch.equal(layer.running_mean, before)


def test_too_many_blocks_fail_when_the_trainer_is_created():
    with pytest.raises(ConfigurationError, match=r"\[1, 3\]") as error:
        Trainer(make_model(), args("supcon", unfreeze_last_blocks=4), dataset(6, 0))
    assert "dummy/tiny" in str(error.value)


def test_encode_restoring_train_mode_keeps_frozen_batch_norm_in_eval():
    model = batch_norm_model(1)
    layer = model.backbone.features[1]
    model.unfreeze_backbone(last_blocks=2)
    model.train()
    model.encode(dataset(2, 0).images)
    assert model.backbone.training and not layer.training


def test_hard_negative_re_encoding_keeps_frozen_batch_norm_statistics():
    model = batch_norm_model(1)
    layer = model.backbone.features[1]
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    options = {"freeze_batch_norm": False, "unfreeze_last_blocks": 2}
    train_representation(
        model, dataset(6, 0), None, args("pair", hard_negative_fraction=0.5, **options)
    )
    for name, value in layer.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_augmented_training_changes_only_the_last_block():
    model = make_model()
    before = snapshot(model)
    options = {"augmentation": "document", "unfreeze_last_blocks": 1}
    Trainer(model, args("supcon", **options), dataset(6, 0)).train()
    changed = {n for n, v in snapshot(model).items() if not torch.equal(v, before[n])}
    assert changed == set(model.backbone.parameter_blocks()[-1])
