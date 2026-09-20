"""The end-to-end cross-entropy baseline (SPEC §3, docs/research/BASELINES.md)."""

import logging

import numpy as np
import pytest
import torch

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.exceptions import ConfigurationError
from embedfit.heads import LinearHead, LogisticRegressionHead
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def make_model(**kwargs):
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32, **kwargs)


def e2e_args(**kwargs):
    options = {
        "training_mode": "end_to_end",
        "num_epochs": 30,
        "batch_size": 12,
        "learning_rate": 2e-2,
        "num_workers": 0,
        **kwargs,
    }
    return TrainingArguments(**options)


@pytest.fixture(scope="module")
def trained():
    model = make_model()
    before = [p.detach().clone() for p in model.backbone.parameters()]
    trainer = Trainer(model, e2e_args(), dataset(12, 0), eval_dataset=dataset(8, 2))
    return model, trainer, trainer.train(), before


def test_backbone_and_classifier_are_trained_jointly(trained):
    model, _, output, before = trained
    assert all(
        not torch.equal(p, b) for p, b in zip(model.backbone.parameters(), before, strict=True)
    )
    assert output.training_mode == "end_to_end" and output.representation_steps == 90
    assert output.train_losses[-1] < 0.6 * output.train_losses[0]
    assert output.eval_losses[-1] < 0.8 * output.eval_losses[0]
    assert output.best_epoch == int(np.argmin(output.eval_losses)) + 1


def test_the_jointly_trained_classifier_becomes_the_head(trained):
    model, trainer, _, _ = trained
    assert isinstance(model.head, LinearHead) and model.head.is_fitted
    assert model.head_name == "linear" and model.head.num_classes == 3
    assert model.backbone.is_frozen and not model.training
    test = dataset(10, 1)
    assert trainer.evaluate(test)["accuracy"] > 0.8
    assert model.predict_proba(test.images).shape == (30, 3)


def test_the_jointly_trained_classifier_is_refitted_to_convergence(caplog, logged_messages):
    caplog.set_level(logging.INFO, logger="embedfit")
    Trainer(make_model(), e2e_args(num_epochs=1), dataset(6, 0)).train()
    messages = logged_messages()
    assert "Phase 1/2 (representation): 'end_to_end'" in messages
    assert "Phase 2/2 (head): fitting LinearHead" in messages and "Fitted head on 18" in messages


def test_runs_are_reproducible_per_seed():
    runs = [
        Trainer(make_model(), e2e_args(num_epochs=3), dataset(6, 0)).train().train_losses
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    other = Trainer(make_model(), e2e_args(num_epochs=3, seed=11), dataset(6, 0)).train()
    assert other.train_losses != runs[0]


def test_best_epoch_restores_backbone_and_classifier_together():
    from embedfit import representation

    images, labels = make_bar_images(4, num_classes=3, seed=2)
    rotated = [LABELS[(LABELS.index(label) + 1) % 3] for label in labels]
    validation = ImageDataset.from_lists(images, rotated)  # Contradicts the images on purpose.
    model = make_model()
    # Early epochs lower even this loss while the initial logits settle; once the classifier
    # becomes confident about the real classes the contradictory labels make it rise again.
    result = representation.train_representation(
        model, dataset(6, 0), validation, e2e_args(num_epochs=20)
    )
    assert result.best_epoch < 20 and result.eval_losses[-1] > min(result.eval_losses)
    # The restored backbone and classifier together reproduce the best epoch's recorded loss.
    model.eval()
    pixels = torch.stack([model.backbone.preprocess(image) for image in images])
    targets = torch.tensor([LABELS.index(label) for label in rotated])
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(result.classifier(model(pixels)), targets).item()
    assert loss == pytest.approx(result.eval_losses[result.best_epoch - 1], rel=1e-4)
    assert loss != pytest.approx(result.eval_losses[-1], rel=1e-2)


def test_validation_loss_is_identical_for_identical_weights(monkeypatch):
    """Uneven validation batches must not be reshuffled, or checkpoint selection follows noise."""
    from embedfit import representation

    monkeypatch.setattr(representation.torch.optim.AdamW, "step", lambda self, *a, **k: None)
    validation = dataset(5, 2)  # 15 samples in batches of 6: 6, 6, and 3.
    result = representation.train_representation(
        make_model(), dataset(6, 0), validation, e2e_args(num_epochs=5, batch_size=6)
    )
    assert len(set(result.eval_losses)) == 1  # The weights never changed, nor may the loss.
    assert result.best_epoch == 1  # Ties go to the earliest epoch.


def test_validation_loss_is_the_mean_over_samples(monkeypatch):
    from embedfit import representation

    monkeypatch.setattr(representation.torch.optim.AdamW, "step", lambda self, *a, **k: None)
    validation = dataset(5, 2)
    model = make_model()
    result = representation.train_representation(
        model, dataset(6, 0), validation, e2e_args(num_epochs=1, batch_size=6)
    )
    model.eval()
    pixels = torch.stack([model.backbone.preprocess(image) for image in validation.images])
    targets = torch.tensor([LABELS.index(label) for label in validation.labels])
    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(result.classifier(model(pixels)), targets)
    assert result.eval_losses[0] == pytest.approx(expected.item(), rel=1e-5)


def test_training_batches_are_reshuffled_every_epoch():
    from embedfit import representation

    model = make_model()
    objective = representation._objective(
        model,
        dataset(6, 0),
        e2e_args(batch_size=6),
        (),
        seed=3,
        classifier=torch.nn.Linear(32, 3),
    )
    orders = []
    for epoch in (0, 1, 0):
        objective.set_epoch(epoch)
        orders.append(list(iter(objective.loader.sampler)))
    assert orders[0] != orders[1] and orders[0] == orders[2]
    assert sorted(orders[0]) == list(range(18))


def test_an_untrained_baseline_falls_back_to_frozen_features_not_to_chance():
    """With a learning rate too small to train anything, the head must still be usable."""
    train, test = dataset(8, 0), dataset(10, 1)
    trainer = Trainer(make_model(), e2e_args(num_epochs=2, learning_rate=1e-9), train)
    trainer.train()
    frozen = Trainer(make_model(head="linear"), TrainingArguments(training_mode="frozen"), train)
    frozen.train()
    untrained = trainer.evaluate(test)["accuracy"]
    # Both heads are L-BFGS fits of the same convex problem on (almost) the same features;
    # they stop within tolerance of each other, a test sample or two apart.
    assert abs(untrained - frozen.evaluate(test)["accuracy"]) <= 0.1
    assert untrained > 0.4  # Chance is 1/3: a classifier that never trained would sit there.


def test_trained_model_survives_save_and_reload(trained, tmp_path):
    model, trainer, _, _ = trained
    images = dataset(4, 3).images
    loaded = EmbedFitModel.from_pretrained(trainer.save_model(tmp_path / "m"))
    assert isinstance(loaded.head, LinearHead) and loaded.head_name == "linear"
    assert np.allclose(loaded.predict_proba(images), model.predict_proba(images), atol=1e-6)


def test_head_configuration_conflicts_fail_when_the_trainer_is_created():
    train = dataset(6, 0)
    with pytest.raises(ConfigurationError, match="trains its own linear classifier"):
        Trainer(make_model(head=LogisticRegressionHead()), e2e_args(), train)
    accepted = Trainer(make_model(), e2e_args(head="linear", num_epochs=1), train)
    accepted.train()
    assert accepted.model.head_name == "linear"


def test_the_head_reproduces_the_training_time_probabilities():
    """The classifier used in training is exactly the head used at inference."""
    from embedfit import representation

    train = dataset(6, 0)
    model = make_model()
    result = representation.train_representation(model, train, None, e2e_args(num_epochs=2))
    model.eval()
    pixels = torch.stack([model.backbone.preprocess(image) for image in train.images[:5]])
    with torch.no_grad():
        expected = torch.softmax(result.classifier(model(pixels)), dim=1).numpy()
    head = LinearHead.from_module(result.classifier)
    assert np.allclose(head.predict_proba(model.encode(train.images[:5])), expected, atol=1e-6)


def test_temperature_does_not_affect_the_baseline():
    losses = [
        Trainer(make_model(), e2e_args(num_epochs=2, temperature=t), dataset(6, 0))
        .train()
        .train_losses
        for t in (0.07, 1.0)
    ]
    assert losses[0] == losses[1]
