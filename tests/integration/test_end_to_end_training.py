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


def test_logs_say_that_no_separate_head_was_fitted(caplog, logged_messages):
    caplog.set_level(logging.INFO, logger="embedfit")
    Trainer(make_model(), e2e_args(num_epochs=1), dataset(6, 0)).train()
    messages = logged_messages()
    assert "Phase 1/2 (representation): 'end_to_end'" in messages
    assert "Phase 2/2 (head): skipped" in messages and "Fitted head" not in messages


def test_batches_are_reshuffled_each_epoch_from_the_seed():
    runs = [
        Trainer(make_model(), e2e_args(num_epochs=3), dataset(6, 0)).train().train_losses
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    other = Trainer(make_model(), e2e_args(num_epochs=3, seed=11), dataset(6, 0)).train()
    assert other.train_losses != runs[0]


def test_best_epoch_restores_backbone_and_classifier_together():
    images, labels = make_bar_images(4, num_classes=3, seed=2)
    rotated = [LABELS[(LABELS.index(label) + 1) % 3] for label in labels]
    validation = ImageDataset.from_lists(images, rotated)  # Contradicts the images on purpose.
    model = make_model()
    # Early epochs lower even this loss while the initial logits settle; once the classifier
    # becomes confident about the real classes the contradictory labels make it rise again.
    output = Trainer(model, e2e_args(num_epochs=20), dataset(6, 0), eval_dataset=validation).train()
    assert output.best_epoch < 20 and output.eval_losses[-1] > min(output.eval_losses)
    # The restored pair reproduces the recorded validation loss of the best epoch.
    targets = torch.tensor([LABELS.index(label) for label in rotated])
    probabilities = torch.from_numpy(model.predict_proba(images))
    loss = torch.nn.functional.nll_loss(probabilities.log(), targets).item()
    assert loss == pytest.approx(output.eval_losses[output.best_epoch - 1], rel=1e-3)
    assert loss != pytest.approx(output.eval_losses[-1], rel=1e-2)


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
    assert Trainer(make_model(), e2e_args(head="linear"), train) is not None


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
