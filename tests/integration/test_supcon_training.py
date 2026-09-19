"""SupCon end to end on synthetic bars with DummyBackbone (acceptance criterion D)."""

import logging

import numpy as np
import pytest
import torch

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.exceptions import ConfigurationError
from embedfit.representation import batch_shape
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def make_model(**kwargs):
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32, **kwargs)


def supcon_args(**kwargs):
    options = {
        "training_mode": "supcon",
        "num_epochs": 10,
        "batch_size": 12,
        "learning_rate": 1e-2,
        "temperature": 0.1,
        "num_workers": 0,
        **kwargs,
    }
    return TrainingArguments(**options)


def class_separation(model, data):
    """Mean same-class cosine similarity minus mean different-class cosine similarity."""
    embeddings = model.encode(data.images)
    similarity = embeddings @ embeddings.T
    same = np.array(data.labels)[:, None] == np.array(data.labels)[None, :]
    off_diagonal = ~np.eye(len(same), dtype=bool)
    return similarity[same & off_diagonal].mean() - similarity[~same].mean()


@pytest.fixture(scope="module")
def trained():
    train, validation, test = dataset(12, 0), dataset(8, 2), dataset(10, 1)
    model = make_model()
    before_weights = [p.detach().clone() for p in model.backbone.parameters()]
    before_separation = class_separation(model, test)
    trainer = Trainer(model, supcon_args(), train, eval_dataset=validation)
    output = trainer.train()
    return {
        "model": model,
        "trainer": trainer,
        "output": output,
        "test": test,
        "before_weights": before_weights,
        "before_separation": before_separation,
    }


def test_encoder_weights_change_and_loss_is_finite_and_decreasing(trained):
    output, model = trained["output"], trained["model"]
    changed = [
        not torch.equal(p, b)
        for p, b in zip(model.backbone.parameters(), trained["before_weights"], strict=True)
    ]
    assert all(changed)
    assert output.representation_steps == 30  # 10 epochs x ceil(36 / 12) batches.
    assert len(output.train_losses) == len(output.eval_losses) == 10
    assert np.isfinite(output.train_losses).all() and np.isfinite(output.eval_losses).all()
    assert output.train_losses[-1] < 0.7 * output.train_losses[0]
    assert output.eval_losses[-1] < 0.7 * output.eval_losses[0]
    assert output.best_epoch == int(np.argmin(output.eval_losses)) + 1


def test_same_class_embeddings_move_together_on_unseen_images(trained):
    after = class_separation(trained["model"], trained["test"])
    assert after > trained["before_separation"] + 0.3


def test_backbone_is_frozen_afterwards_and_head_is_fitted(trained):
    model = trained["model"]
    assert model.backbone.is_frozen and not model.training and model.head.is_fitted
    assert not any(p.requires_grad for p in model.backbone.parameters())


def test_adapted_embeddings_beat_the_frozen_baseline(trained):
    test = trained["test"]
    adapted = trained["trainer"].evaluate(test)["accuracy"]
    baseline = Trainer(make_model(), TrainingArguments(training_mode="frozen"), dataset(12, 0))
    baseline.train()
    assert adapted >= 0.95 and adapted > baseline.evaluate(test)["accuracy"] + 0.2


def test_inference_needs_no_sampler_or_trainer(trained):
    model, test = trained["model"], trained["test"]
    assert model.predict(test.images[0]).shape == (1,)  # One image: no batch structure needed.
    assert model.predict_proba(test.images[:5]).shape == (5, 3)


def test_same_seed_reproduces_the_run():
    runs = []
    for _ in range(2):
        model = make_model()
        output = Trainer(model, supcon_args(num_epochs=2), dataset(6, 0)).train()
        runs.append((output.train_losses, model.encode(dataset(2, 1).images)))
    assert runs[0][0] == runs[1][0]
    assert np.array_equal(runs[0][1], runs[1][1])
    other = make_model()
    different = Trainer(other, supcon_args(num_epochs=2, seed=7), dataset(6, 0)).train()
    assert different.train_losses != runs[0][0]


def test_best_validation_epoch_is_restored():
    from embedfit import representation
    from embedfit.losses import SupConLoss

    train = dataset(6, 0)
    images, labels = make_bar_images(4, num_classes=3, seed=2)
    # Rotate the validation labels so that they contradict the images: the better the
    # backbone separates the real classes, the worse this validation loss becomes, which
    # makes the first epoch the best one.
    rotated = [
        LABELS[(LABELS.index(label) + 1) % 3] if i % 2 else label for i, label in enumerate(labels)
    ]
    validation = ImageDataset.from_lists(images, rotated)
    model = make_model()
    args = supcon_args(num_epochs=6)
    output = Trainer(model, args, train, eval_dataset=validation).train()
    assert output.best_epoch == int(np.argmin(output.eval_losses)) + 1
    assert output.best_epoch < 6, "expected an earlier epoch to be best in this setup"

    # Validation batches are identical every epoch, so re-measuring the loss of the final
    # model tells which epoch's weights it holds.
    loader, _ = representation._loader(
        model, validation, args, seed=args.seed + representation._EVAL_SEED_OFFSET
    )
    with torch.no_grad():
        losses = [SupConLoss(args.temperature)(model(x), y).item() for x, y in loader]
    assert np.mean(losses) == pytest.approx(output.eval_losses[output.best_epoch - 1], rel=1e-5)
    assert np.mean(losses) != pytest.approx(output.eval_losses[-1], rel=1e-3)


def test_without_validation_data_the_last_epoch_is_kept():
    output = Trainer(make_model(), supcon_args(num_epochs=3), dataset(6, 0)).train()
    assert output.eval_losses == () and output.best_epoch == 3


def test_validation_data_never_trains_the_backbone():
    train = dataset(6, 0)
    with_validation, without_validation = make_model(), make_model()
    Trainer(with_validation, supcon_args(num_epochs=1), train, eval_dataset=dataset(4, 2)).train()
    Trainer(without_validation, supcon_args(num_epochs=1), train).train()
    # One epoch: the single checkpoint is the last epoch either way, so weights must match.
    for first, second in zip(
        with_validation.backbone.parameters(), without_validation.backbone.parameters(), strict=True
    ):
        assert torch.equal(first, second)


def test_gradient_accumulation_reduces_optimizer_steps():
    output = Trainer(
        make_model(), supcon_args(num_epochs=2, gradient_accumulation_steps=2), dataset(8, 0)
    ).train()
    assert output.representation_steps == 2  # ceil(2 batches / 2) per epoch.


def test_unfrozen_backbone_can_be_kept_trainable():
    model = make_model()
    Trainer(
        model, supcon_args(num_epochs=1, freeze_backbone_before_head=False), dataset(6, 0)
    ).train()
    assert not model.backbone.is_frozen and model.head.is_fitted and not model.training


def test_bf16_mixed_precision_runs_on_cpu():
    output = Trainer(
        make_model(), supcon_args(num_epochs=2, mixed_precision="bf16"), dataset(6, 0)
    ).train()
    assert np.isfinite(output.train_losses).all()


@pytest.mark.parametrize(
    "kwargs, data, match",
    [
        ({"batch_size": 3}, dataset(6, 0), "batch_size >= 4"),
        ({"mixed_precision": "fp16"}, dataset(6, 0), "requires a CUDA device"),
        (
            {},
            ImageDataset.from_lists(*[x[:13] for x in make_bar_images(6, num_classes=3)]),
            "smallest class has 1",
        ),
    ],
)
def test_invalid_setups_fail_when_the_trainer_is_created(kwargs, data, match):
    with pytest.raises(ConfigurationError, match=match):
        Trainer(make_model(), supcon_args(**kwargs), data)


def test_validation_set_is_checked_up_front_too():
    lonely = ImageDataset.from_lists(*[x[:13] for x in make_bar_images(6, num_classes=3)])
    with pytest.raises(ConfigurationError, match="smallest class has 1"):
        Trainer(make_model(), supcon_args(), dataset(6, 0), eval_dataset=lonely)


@pytest.mark.parametrize(
    "batch_size, classes, expected",
    [(32, 2, (2, 16)), (32, 10, (10, 3)), (12, 3, (3, 4)), (5, 9, (2, 2))],
)
def test_batch_shape(batch_size, classes, expected):
    assert batch_shape(batch_size, classes) == expected


def test_logs_show_both_phases_and_losses_without_sample_content(caplog):
    caplog.set_level(logging.INFO, logger="embedfit")
    Trainer(make_model(), supcon_args(num_epochs=1), dataset(6, 0)).train()
    assert "Phase 1/2 (representation): 'supcon' for 1 epochs" in caplog.text
    assert "Epoch 1/1: train loss" in caplog.text and "Phase 2/2 (head)" in caplog.text
    assert "vertical" not in caplog.text


class RecordingAdamW(torch.optim.AdamW):
    """AdamW that records how the training loop drives it."""

    record: dict = {}

    def __init__(self, params, **kwargs):
        type(self).record = {"kwargs": kwargs, "lrs": [], "grad_norms": [], "zero_grads": 0}
        super().__init__(params, **kwargs)

    def step(self, *args, **kwargs):
        grads = [p.grad for group in self.param_groups for p in group["params"]]
        self.record["lrs"].append(self.param_groups[0]["lr"])
        self.record["grad_norms"].append(torch.cat([g.flatten() for g in grads]).norm().item())
        return super().step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.record["zero_grads"] += 1
        return super().zero_grad(*args, **kwargs)


@pytest.fixture
def recorded(monkeypatch):
    from embedfit import representation

    monkeypatch.setattr(representation.torch.optim, "AdamW", RecordingAdamW)
    return RecordingAdamW


def test_optimizer_receives_the_configured_hyperparameters_and_schedule(recorded):
    args = supcon_args(num_epochs=5, learning_rate=0.02, weight_decay=0.3, warmup_ratio=0.4)
    Trainer(make_model(), args, dataset(8, 0)).train()  # 2 batches/epoch: 10 steps, 4 warm-up.
    record = recorded.record
    assert record["kwargs"] == {"lr": 0.02, "weight_decay": 0.3}
    expected = [0.02 * f for f in (0.25, 0.5, 0.75, 1.0, 1.0, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6)]
    assert record["lrs"] == pytest.approx(expected)
    assert record["zero_grads"] >= 10  # Gradients are cleared for every step.


def test_gradients_are_clipped_only_when_requested(recorded):
    clipped = supcon_args(num_epochs=1, max_grad_norm=1e-3)
    Trainer(make_model(), clipped, dataset(8, 0)).train()
    assert max(recorded.record["grad_norms"]) <= 1e-3 * 1.001
    Trainer(make_model(), supcon_args(num_epochs=1, max_grad_norm=None), dataset(8, 0)).train()
    assert max(recorded.record["grad_norms"]) > 1e-2


def test_a_short_final_accumulation_group_is_averaged_over_its_own_size(recorded):
    # 30 samples -> 3 batches per epoch. With accumulation 2 the groups are (2, 1); with
    # accumulation 1 every batch is its own group. The lone third batch must produce the same
    # gradient either way, not half of it.
    data = dataset(10, 0)
    Trainer(
        make_model(),
        supcon_args(
            num_epochs=1, learning_rate=1e-9, max_grad_norm=None, gradient_accumulation_steps=2
        ),
        data,
    ).train()
    grouped = recorded.record["grad_norms"]
    Trainer(
        make_model(), supcon_args(num_epochs=1, learning_rate=1e-9, max_grad_norm=None), data
    ).train()
    single = recorded.record["grad_norms"]
    assert len(grouped) == 2 and len(single) == 3
    assert grouped[1] == pytest.approx(single[2], rel=1e-4)


def test_accumulation_larger_than_the_epoch_does_not_shrink_the_step(recorded):
    data = dataset(8, 0)  # 2 batches per epoch.
    Trainer(
        make_model(),
        supcon_args(
            num_epochs=1, learning_rate=1e-9, max_grad_norm=None, gradient_accumulation_steps=10
        ),
        data,
    ).train()
    whole_epoch = recorded.record["grad_norms"]
    Trainer(
        make_model(),
        supcon_args(
            num_epochs=1, learning_rate=1e-9, max_grad_norm=None, gradient_accumulation_steps=2
        ),
        data,
    ).train()
    assert whole_epoch == pytest.approx(recorded.record["grad_norms"], rel=1e-5)


def test_each_epoch_draws_different_batches(monkeypatch):
    from embedfit.sampling import BalancedClassBatchSampler

    epochs = []
    original = BalancedClassBatchSampler.__iter__

    def recording_iter(self):
        batches = list(original(self))
        epochs.append(batches)
        return iter(batches)

    monkeypatch.setattr(BalancedClassBatchSampler, "__iter__", recording_iter)
    Trainer(make_model(), supcon_args(num_epochs=3), dataset(8, 0)).train()
    assert len(epochs) == 3 and epochs[0] != epochs[1] != epochs[2]


@pytest.mark.parametrize(
    "total, ratio, expected",
    [
        (4, 0.0, [1.0, 0.75, 0.5, 0.25, 0.0]),
        (10, 0.3, [1 / 3, 2 / 3, 1.0, 1.0, 6 / 7, 5 / 7, 4 / 7, 3 / 7, 2 / 7, 1 / 7, 0.0]),
        (1, 0.0, [1.0, 0.0]),
        (2, 0.99, [0.5, 1.0, 0.0]),
    ],
)
def test_schedule_warms_up_linearly_then_decays_to_zero(total, ratio, expected):
    from embedfit.representation import _schedule

    factor = _schedule(total, ratio)
    assert [factor(step) for step in range(total + 1)] == pytest.approx(expected)


def test_missing_class_is_rejected_before_any_training(caplog):
    images, labels = make_bar_images(6, num_classes=2, seed=0)  # The model also knows "diagonal".
    with caplog.at_level(logging.INFO, logger="embedfit"):
        with pytest.raises(ConfigurationError, match="1 of the model's 3 labels have no samples"):
            Trainer(make_model(), supcon_args(), ImageDataset.from_lists(images, labels))
    assert "Epoch" not in caplog.text


def test_empty_validation_set_and_fully_frozen_backbone_are_configuration_errors():
    from embedfit.representation import train_representation

    with pytest.raises(ConfigurationError, match="eval_dataset is empty"):
        Trainer(make_model(), supcon_args(), dataset(6, 0), eval_dataset=ImageDataset([]))
    model = make_model()
    model.freeze_backbone()
    with pytest.raises(ConfigurationError, match="Every backbone parameter is frozen"):
        train_representation(model, dataset(6, 0), None, supcon_args())


def test_training_discards_partial_freezing_as_documented():
    model = make_model()
    model.backbone.features.requires_grad_(False)
    Trainer(
        model, supcon_args(num_epochs=1, freeze_backbone_before_head=False), dataset(6, 0)
    ).train()
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_divergence_raises_an_embedfit_error():
    from embedfit.exceptions import EmbedFitError, TrainingDivergedError

    assert issubclass(TrainingDivergedError, EmbedFitError)
    assert issubclass(TrainingDivergedError, FloatingPointError)
    model = make_model()
    with torch.no_grad():
        model.backbone.projection.weight.fill_(float("nan"))
    with pytest.raises(TrainingDivergedError, match="non-finite at epoch 1"):
        Trainer(model, supcon_args(num_epochs=1), dataset(6, 0)).train()
