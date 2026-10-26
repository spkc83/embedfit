"""Joint cross-entropy and supervised contrastive training: ``end_to_end`` with a
``contrastive_weight`` (ADR 0022)."""

import numpy as np
import pytest
import torch
from torch import nn

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments, representation
from embedfit.exceptions import ConfigurationError
from embedfit.heads import LinearHead
from embedfit.losses import SupConLoss
from embedfit.sampling import BalancedClassBatchSampler
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def make_model(**kwargs):
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32, **kwargs)


def joint_args(**kwargs):
    options = {
        "training_mode": "end_to_end",
        "num_epochs": 3,
        "batch_size": 6,
        "learning_rate": 1e-2,
        "num_workers": 0,
        "contrastive_weight": 0.5,
        **kwargs,
    }
    return TrainingArguments(**options)


def trained_state(args, train, validation=None):
    model = make_model()
    result = representation.train_representation(model, train, validation, args)
    return model, result


def test_weight_zero_is_exactly_the_cross_entropy_baseline():
    train, validation = dataset(6, 0), dataset(4, 2)
    baseline, base = trained_state(joint_args(contrastive_weight=0.0), train, validation)
    default, plain = trained_state(
        TrainingArguments(
            training_mode="end_to_end",
            num_epochs=3,
            batch_size=6,
            learning_rate=1e-2,
            num_workers=0,
        ),
        train,
        validation,
    )
    assert (base.train_losses, base.eval_losses) == (plain.train_losses, plain.eval_losses)
    for first, second in zip(
        baseline.backbone.state_dict().values(), default.backbone.state_dict().values(), strict=True
    ):
        assert torch.equal(first, second)
    assert torch.equal(base.classifier.weight, plain.classifier.weight)
    assert torch.equal(base.classifier.bias, plain.classifier.bias)


def batch(per_class=2, seed=5):
    images, labels = make_bar_images(per_class, num_classes=3, seed=seed)
    model = make_model()
    pixels = torch.stack([model.backbone.preprocess(image) for image in images])
    return pixels, torch.tensor([LABELS.index(label) for label in labels])


def losses(args, pixels, labels, projection=None):
    """The loss of one fixed batch under ``args``, and the model's embeddings and logits."""
    model = make_model()
    model.eval()
    torch.manual_seed(0)
    classifier = nn.Linear(32, 3)
    objective = representation._objective(
        model, dataset(4, 0), args, (), seed=0, classifier=classifier, projection=projection
    )
    with torch.no_grad():
        loss = objective.loss((pixels, labels))
        embeddings = model(pixels.flatten(0, -4))
    return loss, embeddings, classifier


def test_weight_zero_is_the_plain_cross_entropy_of_a_batch():
    pixels, labels = batch()
    loss, embeddings, classifier = losses(joint_args(contrastive_weight=0.0), pixels, labels)
    with torch.no_grad():
        expected = nn.functional.cross_entropy(classifier(embeddings), labels)
    assert torch.equal(loss, expected)


@pytest.mark.parametrize("weight", [0.5, 2.0])
def test_a_weight_adds_exactly_that_multiple_of_supcon(weight):
    pixels, labels = batch()
    plain, embeddings, _ = losses(joint_args(contrastive_weight=0.0), pixels, labels)
    joint, _, _ = losses(joint_args(contrastive_weight=weight, temperature=0.2), pixels, labels)
    supcon = SupConLoss(temperature=0.2)(embeddings, labels)
    assert supcon > 0
    assert joint.item() == pytest.approx(plain.item() + weight * supcon.item(), rel=1e-6)


def test_the_contrastive_term_uses_the_projection_head():
    pixels, labels = batch()
    args = joint_args(projection_dim=8)
    projection = representation._projection_head(make_model(), args)
    joint, embeddings, classifier = losses(args, pixels, labels, projection=projection)
    with torch.no_grad():
        expected = nn.functional.cross_entropy(classifier(embeddings), labels)
        expected += 0.5 * SupConLoss(temperature=0.07)(projection(embeddings), labels)
    assert joint.item() == pytest.approx(expected.item(), rel=1e-6)


def test_cross_entropy_applies_to_every_view():
    pixels, labels = batch()
    views = torch.stack([pixels, pixels.flip(-1)], dim=1)  # [B, 2, C, H, W]
    args = joint_args(contrastive_views=2, augmentation="natural")
    joint, embeddings, classifier = losses(args, views, labels)
    with torch.no_grad():
        repeated = labels.repeat_interleave(2)
        expected = nn.functional.cross_entropy(classifier(embeddings), repeated)
        grouped = embeddings.unflatten(0, views.shape[:2])
        expected += 0.5 * SupConLoss(temperature=0.07)(grouped, labels)
    assert joint.item() == pytest.approx(expected.item(), rel=1e-6)


def test_the_projection_head_is_trained_and_discarded():
    train = dataset(6, 0)
    model = make_model()
    trainer = Trainer(model, joint_args(projection_dim=8), train)
    output = trainer.train()
    assert output.training_mode == "end_to_end" and np.isfinite(output.train_losses).all()
    assert isinstance(model.head, LinearHead) and model.head_name == "linear"
    assert all(parameter.shape != (8, 32) for parameter in model.parameters())  # No projection.
    assert model.encode(train.images[:2]).shape == (2, 32)


def test_a_contrastive_weight_defaults_to_class_balanced_batches():
    model = make_model()
    sampler = representation._batch_sampler(joint_args(), np.array([0, 0, 1, 1, 2, 2]), seed=0)
    assert isinstance(sampler, BalancedClassBatchSampler)
    plain = representation._batch_sampler(
        joint_args(contrastive_weight=0.0), np.array([0, 0, 1, 1, 2, 2]), seed=0
    )
    assert not isinstance(plain, BalancedClassBatchSampler)
    explicit = representation._objective(
        model, dataset(4, 0), joint_args(sampler="random"), (), seed=0
    )
    assert not isinstance(explicit.loader.batch_sampler, BalancedClassBatchSampler)


def test_a_batch_without_positives_fails_before_training():
    with pytest.raises(ConfigurationError, match="no two samples of the same class"):
        Trainer(make_model(), joint_args(sampler="random", batch_size=1), dataset(4, 0))
    # Without the contrastive term, the same batches are fine.
    Trainer(
        make_model(),
        joint_args(sampler="random", batch_size=1, contrastive_weight=0.0),
        dataset(4, 0),
    )


def test_the_balanced_default_needs_two_samples_per_class():
    with pytest.raises(ConfigurationError, match="at least 2 samples"):
        Trainer(make_model(), joint_args(), dataset(1, 0))


def test_validation_loss_is_the_combined_objective(monkeypatch):
    monkeypatch.setattr(representation.torch.optim.AdamW, "step", lambda self, *a, **k: None)
    train, validation = dataset(6, 0), dataset(4, 2)
    args = joint_args(num_epochs=1, temperature=0.2)
    model, result = trained_state(args, train, validation)
    model.eval()
    targets = model.label_mapping.encode(validation.labels)
    sampler = representation._batch_sampler(
        args, targets, args.seed + representation._EVAL_SEED_OFFSET
    )
    sampler.set_epoch(0)
    total, count = 0.0, 0
    with torch.no_grad():
        for positions in sampler:
            pixels = torch.stack(
                [model.backbone.preprocess(validation.images[i]) for i in positions]
            )
            labels = torch.from_numpy(targets[positions].astype(np.int64))
            embeddings = model(pixels)
            loss = nn.functional.cross_entropy(result.classifier(embeddings), labels)
            loss += 0.5 * SupConLoss(temperature=0.2)(embeddings, labels)
            total += loss.item() * len(positions)
            count += len(positions)
    assert result.eval_losses[0] == pytest.approx(total / count, rel=1e-5)


def test_joint_training_runs_with_accumulation_and_mixed_precision():
    args = joint_args(gradient_accumulation_steps=2, mixed_precision="bf16", max_steps=3)
    output = Trainer(make_model(), args, dataset(6, 0)).train()
    assert output.representation_steps == 3 and np.isfinite(output.train_losses).all()


def test_joint_runs_are_reproducible_and_differ_from_the_baseline():
    train = dataset(6, 0)
    runs = [trained_state(joint_args(), train)[1].train_losses for _ in range(2)]
    assert runs[0] == runs[1]
    baseline = trained_state(joint_args(contrastive_weight=0.0), train)[1].train_losses
    assert baseline != runs[0]


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("nan"), True, "0.5"])
def test_the_weight_must_be_a_finite_non_negative_number(weight):
    with pytest.raises(ConfigurationError, match="contrastive_weight must be a finite number"):
        joint_args(contrastive_weight=weight)


@pytest.mark.parametrize("mode", ["supcon", "pair", "frozen"])
def test_the_weight_only_applies_to_end_to_end(mode):
    with pytest.raises(ConfigurationError, match="contrastive_weight only applies"):
        TrainingArguments(training_mode=mode, contrastive_weight=0.5)


@pytest.mark.parametrize(
    "options", [{"projection_dim": 8}, {"contrastive_views": 2, "augmentation": "natural"}]
)
def test_contrastive_options_need_a_weight_in_end_to_end(options):
    with pytest.raises(ConfigurationError, match="only applies to training_mode='supcon', or"):
        joint_args(contrastive_weight=0.0, **options)
    assert joint_args(**options).training_mode == "end_to_end"


def test_the_head_rate_covers_the_classifier_and_the_projection_head(monkeypatch):
    """ADR 0020: the modules training creates besides the backbone share head_learning_rate."""
    heads, optimizers = [], []
    build = representation._projection_head
    monkeypatch.setattr(
        representation,
        "_projection_head",
        lambda model, args: heads.append(build(model, args)) or heads[-1],
    )

    class Recording(torch.optim.AdamW):
        def __init__(self, *arguments, **options):
            super().__init__(*arguments, **options)
            optimizers.append(self)

    monkeypatch.setattr(representation.torch.optim, "AdamW", Recording)
    model = make_model()
    args = joint_args(projection_dim=8, head_learning_rate=0.5, max_steps=1)
    result = representation.train_representation(model, dataset(6, 0), None, args)
    [head], [optimizer] = heads, optimizers
    by_rate = {
        group["initial_lr"]: {id(p) for p in group["params"]} for group in optimizer.param_groups
    }
    expected = {id(p) for p in (*result.classifier.parameters(), *head.parameters())}
    assert by_rate[0.5] == expected
    assert by_rate[args.learning_rate] == {id(p) for p in model.backbone.parameters()}
