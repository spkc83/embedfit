"""Hard-negative mining inside pair training: train-only, no gradients, refresh schedule."""

import logging

import numpy as np
import pytest
import torch

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def make_model():
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32)


def mining_args(**kwargs):
    options = {
        "training_mode": "pair",
        "num_epochs": 4,
        "batch_size": 32,
        "learning_rate": 1e-2,
        "num_workers": 0,
        "hard_negative_fraction": 0.5,
        **kwargs,
    }
    return TrainingArguments(**options)


def test_training_with_mining_adapts_the_backbone_and_fits_the_head():
    model = make_model()
    trainer = Trainer(model, mining_args(num_epochs=6), dataset(10, 0), eval_dataset=dataset(6, 2))
    output = trainer.train()
    assert np.isfinite(output.train_losses).all() and np.isfinite(output.eval_losses).all()
    assert output.train_losses[-1] < output.train_losses[0]
    assert model.head.is_fitted and model.backbone.is_frozen
    # Chance is 1/3. Mining adds variance between seeds (0.73 to 1.00 over five backbone
    # seeds), so the bound is loose on purpose.
    assert trainer.evaluate(dataset(10, 1))["accuracy"] > 0.65


def test_mining_changes_training_and_is_reproducible():
    train = dataset(8, 0)
    mined = [Trainer(make_model(), mining_args(), train).train().train_losses for _ in range(2)]
    plain = Trainer(make_model(), mining_args(hard_negative_fraction=0.0), train).train()
    assert mined[0] == mined[1]
    assert mined[0] != plain.train_losses


def test_only_training_images_are_encoded_without_gradients_on_the_refresh_schedule(monkeypatch):
    train, validation = dataset(6, 0), dataset(4, 2)
    model = make_model()
    calls = []
    encode = model.encode

    def spy(images, **kwargs):
        calls.append(
            (len(images), {id(i) for i in images}, torch.is_grad_enabled(), model.backbone.training)
        )
        return encode(images, **kwargs)

    monkeypatch.setattr(model, "encode", spy)
    original_fit_head = model.fit_head
    monkeypatch.setattr(
        model, "fit_head", lambda *a, **k: calls.append("head") or original_fit_head(*a, **k)
    )
    args = mining_args(num_epochs=5, hard_negative_refresh_epochs=2)
    Trainer(model, args, train, eval_dataset=validation).train()

    mining_calls = calls[: calls.index("head")]
    train_ids = {id(image) for image in train.images}
    validation_ids = {id(image) for image in validation.images}
    assert len(mining_calls) == 3  # Epochs 0, 2, and 4.
    for count, image_ids, _, _ in mining_calls:
        assert count == len(train) and image_ids == train_ids
        assert not image_ids & validation_ids  # Validation data is never mined.


def test_mining_encodes_in_eval_mode_without_gradients_and_leaves_batch_statistics_alone():
    """Observed inside the backbone, only during mining refreshes."""
    from torch import nn

    from embedfit.backbones import DummyBackbone

    class BatchNormBackbone(DummyBackbone):
        def __init__(self):
            super().__init__(embedding_dim=32)
            self.norm = nn.BatchNorm2d(3)
            self.mining = False
            self.observed = []

        def forward_features(self, pixel_values):
            if self.mining:
                self.observed.append(
                    (self.training, torch.is_grad_enabled(), int(self.norm.num_batches_tracked))
                )
            return super().forward_features(self.norm(pixel_values))

    backbone = BatchNormBackbone()
    model = EmbedFitModel(backbone, labels=LABELS)
    encode = model.encode

    def mining_encode(images, **kwargs):
        backbone.mining = True
        before = (
            backbone.training,
            int(backbone.norm.num_batches_tracked),
            backbone.norm.running_mean.clone(),
        )
        try:
            return encode(images, **kwargs)
        finally:
            backbone.mining = False
            # Restored to training mode, and no batch statistic moved during the refresh.
            assert backbone.training == before[0]
            assert int(backbone.norm.num_batches_tracked) == before[1]
            assert torch.equal(backbone.norm.running_mean, before[2])

    model.encode = mining_encode
    from embedfit import representation

    representation.train_representation(model, dataset(6, 0), None, mining_args(num_epochs=2))
    assert backbone.observed, "mining never encoded anything"
    assert all(training is False and grad is False for training, grad, _ in backbone.observed)
    assert int(backbone.norm.num_batches_tracked) > 0  # Training batches did update them.


def test_only_the_training_objective_mines(monkeypatch):
    from embedfit import representation

    seen = []
    original = representation.PairSampler.sample

    def spy(self, epoch=0, *, embeddings=None, hard_negative_fraction=0.0):
        seen.append((len(self._classes), embeddings is not None, hard_negative_fraction))
        return original(
            self, epoch, embeddings=embeddings, hard_negative_fraction=hard_negative_fraction
        )

    monkeypatch.setattr(representation.PairSampler, "sample", spy)
    Trainer(
        make_model(), mining_args(num_epochs=2), dataset(6, 0), eval_dataset=dataset(4, 2)
    ).train()
    mined = [call for call in seen if call[1]]
    assert mined == [(18, True, 0.5), (18, True, 0.5)]  # 18 training samples, never the 12.
    assert all(fraction == 0.0 for size, has, fraction in seen if size == 12)


def test_refresh_is_logged_with_counts_only(caplog, logged_messages):
    caplog.set_level(logging.INFO, logger="embedfit")
    Trainer(make_model(), mining_args(num_epochs=2), dataset(6, 0)).train()
    assert logged_messages().count("Refreshed 18 training embeddings") == 2


def test_learning_rate_schedule_is_planned_on_the_maximum_pair_count(monkeypatch):
    from embedfit import representation

    rates = []
    original = torch.optim.AdamW.step
    monkeypatch.setattr(
        representation.torch.optim.AdamW,
        "step",
        lambda self, *a, **k: rates.append(self.param_groups[0]["lr"]) or original(self, *a, **k),
    )
    args = mining_args(num_epochs=3, batch_size=16)
    output = Trainer(make_model(), args, dataset(6, 0)).train()
    planned = 3 * -(-(18 * 8) // 16)  # 18 samples x (4 + 4) partners, in batches of 16.
    assert output.representation_steps <= planned
    assert rates[0] == pytest.approx(args.learning_rate)
    assert 0 < rates[-1] <= args.learning_rate * (planned - len(rates) + 1) / planned + 1e-12
