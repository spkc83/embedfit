"""The end-to-end cross-entropy baseline (SPEC §3, docs/research/BASELINES.md)."""

import logging

import numpy as np
import pytest
import torch
from torch import nn

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
    # Fine-tuning takes the loss far below the linear probe it starts from (ADR 0012). The
    # first epoch is no reference: at this learning rate it disturbs the probe's classifier.
    probe, features, targets = probe_for(make_model(), dataset(12, 0), seed=42)
    with torch.no_grad():
        logits = probe.to_module()(torch.from_numpy(features))
        start = torch.nn.functional.cross_entropy(
            logits, torch.from_numpy(targets.astype(np.int64))
        )
    assert output.train_losses[-1] < 0.1 * start.item()
    assert output.eval_losses[-1] < 0.1 * start.item()
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
    # At this learning rate the first epoch disturbs the starting probe; this loss falls as
    # training recovers and rises again once the classifier becomes confident about the real
    # classes, which the contradictory labels penalise.
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
        orders.append([i for batch in objective.loader.batch_sampler for i in batch])
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


def probe_for(model, train, seed):
    features = model.encode(train.images)
    targets = model.label_mapping.encode(train.labels)
    probe = LinearHead(seed=seed)
    probe.fit(features, targets, num_classes=len(LABELS))
    return probe, features, targets


def test_fine_tuning_starts_from_a_converged_linear_probe():
    """ADR 0012: the classifier starts from a probe fitted on the untouched backbone."""
    from embedfit import representation

    train = dataset(6, 0)
    model = make_model()
    # One batch holds all 18 samples, so the first epoch's loss is measured before any step,
    # and the learning rate is too small for that one step to move anything measurably.
    args = e2e_args(num_epochs=1, batch_size=32, learning_rate=1e-12)
    probe, features, targets = probe_for(model, train, args.seed)
    result = representation.train_representation(model, train, None, args)

    trained = LinearHead.from_module(result.classifier).predict_proba(features)
    assert np.allclose(trained, probe.predict_proba(features), atol=1e-5)
    inputs, labels = torch.from_numpy(features), torch.from_numpy(targets.astype(np.int64))
    with torch.no_grad():
        probe_loss = torch.nn.functional.cross_entropy(probe.to_module()(inputs), labels).item()
        torch.manual_seed(args.seed)
        random_loss = torch.nn.functional.cross_entropy(nn.Linear(32, 3)(inputs), labels).item()
    assert result.train_losses[0] == pytest.approx(probe_loss, rel=1e-4)
    assert result.train_losses[0] < random_loss  # What the baseline started from before.


def test_the_probe_sees_the_embeddings_the_model_produces():
    from embedfit import representation

    train = dataset(6, 0)
    model = make_model(normalize_embeddings=False)
    model.unfreeze_backbone()
    model.train()
    args = e2e_args()
    probe, features, _ = probe_for(model, train, args.seed)
    assert not np.allclose(np.linalg.norm(features, axis=1), 1.0)  # Really unnormalised.
    layer = representation._linear_probe(model, train, args)
    assert model.backbone.training  # encode() restored the training mode.
    with torch.no_grad():
        probabilities = torch.softmax(layer(torch.from_numpy(features)), dim=1).double().numpy()
    assert np.allclose(probabilities, probe.predict_proba(features), atol=1e-6)


def test_a_class_without_training_samples_fails_before_training():
    from embedfit import representation

    two_classes = ImageDataset.from_lists(*make_bar_images(6, num_classes=2, seed=0))
    with pytest.raises(ConfigurationError, match="1 of the model's 3 labels have no samples"):
        representation.train_representation(make_model(), two_classes, None, e2e_args())


def test_the_classifier_is_fine_tuned_with_the_backbone(caplog, logged_messages):
    """LP-FT trains the probe further; a classifier left out of the optimizer stays at it."""
    from embedfit import representation

    caplog.set_level(logging.INFO, logger="embedfit")
    train = dataset(6, 0)
    model = make_model()
    args = e2e_args(num_epochs=3)
    probe, _, _ = probe_for(model, train, args.seed)
    result = representation.train_representation(model, train, None, args)
    start = probe.to_module()
    assert not torch.allclose(result.classifier.weight, start.weight, atol=1e-3)
    assert not torch.allclose(result.classifier.bias, start.bias, atol=1e-3)
    assert "Starting from a linear probe: training cross-entropy" in logged_messages()
