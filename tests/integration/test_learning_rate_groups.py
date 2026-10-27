"""Training with per-part learning rates (ADR 0020)."""

import json

import pytest
import torch

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.exceptions import ConfigurationError
from embedfit.representation import _schedule, train_representation
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])
RATE = 1e-2


def dataset():
    return ImageDataset.from_lists(*make_bar_images(6, num_classes=3, seed=0))


def make_model():
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32)


def args(mode, **kwargs):
    options = {
        "training_mode": mode,
        "max_steps": 4,
        "batch_size": 6,
        "learning_rate": RATE,
        "num_workers": 0,
        **kwargs,
    }
    if mode == "supcon":
        options.setdefault("projection_dim", 8)
    return TrainingArguments(**options)


@pytest.fixture
def optimizers(monkeypatch):
    """Every AdamW optimizer created, each with the learning rates of its groups at each step."""
    created = []

    class Recording(torch.optim.AdamW):
        def __init__(self, *arguments, **options):
            super().__init__(*arguments, **options)
            self.history = []
            created.append(self)

        def step(self, closure=None):
            self.history.append([group["lr"] for group in self.param_groups])
            return super().step(closure)

    monkeypatch.setattr(torch.optim, "AdamW", Recording)
    return created


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.backbone.state_dict().items()}


@pytest.mark.parametrize("mode", ["supcon", "end_to_end"])
def test_by_default_one_group_trains_everything_at_the_learning_rate(mode, optimizers):
    train_representation(make_model(), dataset(), None, args(mode))
    (optimizer,) = optimizers
    assert [group["initial_lr"] for group in optimizer.param_groups] == [RATE]
    assert optimizer.param_groups[0]["weight_decay"] == 0.01


@pytest.mark.parametrize("mode", ["supcon", "end_to_end"])
def test_neutral_values_train_exactly_like_the_defaults(mode):
    trained = []
    for options in ({}, {"head_learning_rate": RATE, "layer_lr_decay": 1.0}):
        model = make_model()
        train_representation(model, dataset(), None, args(mode, **options))
        trained.append(snapshot(model))
    for name, value in trained[0].items():
        assert torch.equal(value, trained[1][name]), name


@pytest.mark.parametrize("mode", ["supcon", "end_to_end"])
def test_every_group_follows_the_schedule_from_its_own_rate(mode, optimizers):
    options = {"head_learning_rate": 0.1, "layer_lr_decay": 0.5, "warmup_ratio": 0.5}
    result = train_representation(make_model(), dataset(), None, args(mode, **options))
    (optimizer,) = optimizers
    initial = [group["initial_lr"] for group in optimizer.param_groups]
    # Blocks conv, conv, projection: decay ** 2, decay ** 1, decay ** 0; then the head.
    assert initial == pytest.approx([RATE * 0.25, RATE * 0.5, RATE, 0.1])
    factor = _schedule(4, 0.5)
    assert result.steps == len(optimizer.history) == 4
    for step, rates in enumerate(optimizer.history):
        assert rates == pytest.approx([rate * factor(step) for rate in initial])
    assert all(group["weight_decay"] == 0.01 for group in optimizer.param_groups)


def test_partial_unfreezing_leaves_frozen_blocks_out_of_every_group(optimizers):
    model = make_model()
    before = snapshot(model)
    options = {"unfreeze_last_blocks": 2, "layer_lr_decay": 0.5}
    Trainer(model, args("pair", **options), dataset()).train()
    (optimizer,) = optimizers
    assert [group["initial_lr"] for group in optimizer.param_groups] == [RATE * 0.5, RATE]
    grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
    first_block = set(model.backbone.parameter_blocks()[0])
    for name, parameter in model.backbone.named_parameters():
        assert (id(parameter) in grouped) != (name in first_block), name
    after = snapshot(model)
    assert all(torch.equal(after[name], before[name]) for name in first_block)


def test_the_head_rate_moves_the_head_while_the_backbone_barely_trains(optimizers):
    model = make_model()
    before = snapshot(model)
    options = {"learning_rate": 1e-9, "head_learning_rate": 1e-1}
    result = train_representation(model, dataset(), None, args("end_to_end", **options))
    probe = optimizers[0].param_groups[1]["params"]
    assert [id(p) for p in probe] == [id(p) for p in result.classifier.parameters()]
    after = snapshot(model)
    backbone_change = max((after[n] - before[n]).abs().max().item() for n in after)
    assert backbone_change < 1e-6
    assert optimizers[0].state[probe[0]]["step"] == 4


def test_a_backbone_without_blocks_cannot_decay(monkeypatch):
    model = make_model()
    monkeypatch.setattr(model.backbone, "parameter_blocks", lambda: [])
    with pytest.raises(ConfigurationError, match="layer_lr_decay") as error:
        Trainer(model, args("supcon", layer_lr_decay=0.5), dataset())
    assert "dummy/tiny" in str(error.value)


def test_the_options_are_recorded_with_the_model_and_the_checkpoints(tmp_path):
    options = {"head_learning_rate": 0.1, "layer_lr_decay": 0.8}
    checkpoints = tmp_path / "checkpoints"
    trainer = Trainer(
        make_model(),
        args("end_to_end", checkpoint_dir=str(checkpoints), **options),
        dataset(),
    )
    trainer.train()
    trainer.save_model(tmp_path / "model")
    metadata = json.loads((tmp_path / "model" / "training_metadata.json").read_text())
    assert {k: metadata["training_arguments"][k] for k in options} == options
    saved = [
        path for path in checkpoints.rglob("*.json") if "training_arguments" in path.read_text()
    ]
    assert saved
    for path in saved:
        recorded = json.loads(path.read_text())["training_arguments"]
        assert {k: recorded[k] for k in options} == options
