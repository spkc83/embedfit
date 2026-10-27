"""Training only the normalisation-layer or bias parameters of the backbone (ADR 0021)."""

import json

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


def make_model(norm=False):
    """The demo backbone, with a group norm after its first convolution if ``norm``."""
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32)
    if norm:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model.backbone.features.insert(1, nn.GroupNorm(2, 8))
            with torch.no_grad():  # Away from the identity, so that its gradients are not zero.
                model.backbone.features[1].weight.uniform_(0.5, 1.5)
    return model


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


def changed(model, before):
    return {name for name, value in snapshot(model).items() if not torch.equal(value, before[name])}


BIASES = {"features.0.bias", "features.2.bias", "projection.bias"}
# With the group norm inserted, the second convolution moves to features.3.
NORMED_BIASES = {"features.0.bias", "features.1.bias", "features.3.bias", "projection.bias"}
NORM = {"features.1.weight", "features.1.bias"}


@pytest.mark.parametrize("mode", ["supcon", "pair", "end_to_end"])
@pytest.mark.parametrize(("kind", "expected"), [("bias", NORMED_BIASES), ("layernorm", NORM)])
def test_only_the_selected_parameters_change(mode, kind, expected, tmp_path):
    model = make_model(norm=True)
    before = snapshot(model)
    trainer = Trainer(model, args(mode, trainable_parameters=kind), dataset(6, 0))
    assert trainer.train().representation_steps > 0
    assert changed(model, before) == expected
    trainer.save_model(tmp_path)
    recorded = json.loads((tmp_path / "training_metadata.json").read_text())
    assert recorded["training_arguments"]["trainable_parameters"] == kind


def test_the_selection_composes_with_the_last_blocks():
    model = make_model(norm=True)
    before = snapshot(model)
    options = {"trainable_parameters": "bias", "unfreeze_last_blocks": 2}
    Trainer(model, args("supcon", **options), dataset(6, 0)).train()
    assert changed(model, before) == {"features.3.bias", "projection.bias"}


def test_direct_training_honours_the_argument():
    model = make_model()
    model.unfreeze_backbone()  # Whole backbone trainable before the call.
    before = snapshot(model)
    train_representation(model, dataset(6, 0), None, args("pair", trainable_parameters="bias"))
    assert changed(model, before) == BIASES


@pytest.mark.parametrize(
    ("mode", "options"),
    [("end_to_end", {}), ("supcon", {"projection_dim": 8})],
)
def test_the_parts_outside_the_backbone_always_train(mode, options, monkeypatch):
    optimised = []
    adamw = torch.optim.AdamW

    def spy(params, *rest, **kwargs):
        params = list(params)
        for entry in params:
            optimised.extend(entry["params"] if isinstance(entry, dict) else [entry])
        return adamw(params, *rest, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    model = make_model()
    train_representation(
        model, dataset(6, 0), None, args(mode, trainable_parameters="bias", **options)
    )
    backbone = {id(p) for p in model.backbone.parameters()}
    names = {id(p): n for n, p in model.backbone.named_parameters()}
    assert {names[id(p)] for p in optimised if id(p) in backbone} == BIASES
    outside = [p for p in optimised if id(p) not in backbone]
    # The classifier's weight and bias, or the projection head's two layers.
    assert len(outside) == (2 if mode == "end_to_end" else 4)


def test_a_selection_without_backbone_parameters_fails_when_the_trainer_is_created():
    model = make_model()  # No normalisation layer.
    before = snapshot(model)
    with pytest.raises(ConfigurationError, match="'layernorm' selects no parameter"):
        Trainer(model, args("supcon", trainable_parameters="layernorm"), dataset(6, 0))
    with pytest.raises(ConfigurationError, match="'layernorm' selects no parameter"):
        train_representation(
            model, dataset(6, 0), None, args("supcon", trainable_parameters="layernorm")
        )
    assert changed(model, before) == set()


def test_the_default_trains_the_whole_backbone_as_all_does():
    results = []
    for options in ({}, {"trainable_parameters": "all"}):
        model = make_model(norm=True)
        before = snapshot(model)
        Trainer(model, args("supcon", **options), dataset(6, 0)).train()
        assert changed(model, before) == set(before)
        results.append(snapshot(model))
    for name, value in results[0].items():
        assert torch.equal(value, results[1][name]), name


def test_batch_norm_statistics_stay_fixed_by_default_while_its_affine_parameters_train():
    model = make_model()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model.backbone.features.insert(1, nn.BatchNorm2d(8))
    layer = model.backbone.features[1]
    statistics = {
        "running_mean": layer.running_mean.clone(),
        "running_var": layer.running_var.clone(),
    }
    weight = layer.weight.detach().clone()
    train_representation(
        model, dataset(6, 0), None, args("supcon", trainable_parameters="layernorm")
    )
    assert not torch.equal(layer.weight, weight)
    for name, value in statistics.items():
        assert torch.equal(getattr(layer, name), value), name


def test_a_saved_model_holds_the_full_weights(tmp_path):
    model = make_model()
    trainer = Trainer(model, args("supcon", trainable_parameters="bias"), dataset(6, 0))
    trainer.train()
    trainer.save_model(tmp_path)
    reloaded = EmbedFitModel.from_pretrained(tmp_path)
    state = model.backbone.state_dict()
    assert reloaded.backbone.state_dict().keys() == state.keys()
    for name, value in reloaded.backbone.state_dict().items():
        assert torch.equal(value, state[name]), name


def test_bias_only_training_keeps_batch_norm_statistics_fixed_by_default():
    model = make_model()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model.backbone.features.insert(1, nn.BatchNorm2d(8))
    layer = model.backbone.features[1]
    statistics = {name: value.clone() for name, value in layer.named_buffers()}
    bias, weight = layer.bias.detach().clone(), layer.weight.detach().clone()
    train_representation(model, dataset(6, 0), None, args("supcon", trainable_parameters="bias"))
    assert not torch.equal(layer.bias, bias)  # Its shift trains,
    assert torch.equal(layer.weight, weight)  # its scale does not,
    for name, value in layer.named_buffers():  # and neither do its statistics.
        assert torch.equal(value, statistics[name]), name


def test_the_selection_combines_with_per_part_learning_rates(monkeypatch):
    groups = []
    adamw = torch.optim.AdamW

    def spy(params, *rest, **kwargs):
        params = list(params)
        groups.extend(params)
        return adamw(params, *rest, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    model = make_model(norm=True)
    before = snapshot(model)
    options = {"trainable_parameters": "layernorm", "layer_lr_decay": 0.5}
    result = train_representation(
        model,
        dataset(6, 0),
        None,
        args("end_to_end", head_learning_rate=1e-3, **options),
    )
    assert changed(model, before) == NORM
    names = {id(p): n for n, p in model.backbone.named_parameters()}
    # initial_lr: by the end of training the schedule has decayed lr to zero.
    rates = {names.get(id(p), "classifier"): g["initial_lr"] for g in groups for p in g["params"]}
    assert all(p.requires_grad for g in groups for p in g["params"])
    # The norm is the second of four blocks: decayed twice. The classifier keeps its own rate.
    assert rates == {
        "features.1.weight": 1e-2 * 0.5**2,
        "features.1.bias": 1e-2 * 0.5**2,
        "classifier": 1e-3,
    }
    assert sum(len(g["params"]) for g in groups) == 4
    assert result.classifier is not None
