"""Pairwise contrastive training end to end (acceptance criterion E and the pair path)."""

import numpy as np
import pytest
import torch

from embedfit import EmbedFitModel, ImageDataset, Trainer, TrainingArguments
from embedfit.data import ImageSample
from embedfit.exceptions import ConfigurationError
from embedfit.sampling import RelationshipRule
from embedfit.testing import BAR_CLASSES, make_bar_images

LABELS = list(BAR_CLASSES[:3])


def dataset(per_class, seed):
    return ImageDataset.from_lists(*make_bar_images(per_class, num_classes=3, seed=seed))


def make_model():
    return EmbedFitModel.from_pretrained("dummy/tiny", labels=LABELS, embedding_dim=32)


def pair_args(**kwargs):
    options = {
        "training_mode": "pair",
        "num_epochs": 5,
        "batch_size": 32,
        "learning_rate": 1e-2,
        "num_workers": 0,
        **kwargs,
    }
    return TrainingArguments(**options)


@pytest.fixture(scope="module")
def trained():
    model = make_model()
    before = [p.detach().clone() for p in model.backbone.parameters()]
    trainer = Trainer(model, pair_args(), dataset(12, 0), eval_dataset=dataset(8, 2))
    return model, trainer, trainer.train(), before


def test_pair_training_adapts_the_backbone_and_then_fits_the_head(trained):
    model, trainer, output, before = trained
    assert all(
        not torch.equal(p, b) for p, b in zip(model.backbone.parameters(), before, strict=True)
    )
    assert output.training_mode == "pair" and output.representation_steps > 0
    assert np.isfinite(output.train_losses).all() and np.isfinite(output.eval_losses).all()
    assert output.train_losses[-1] < 0.5 * output.train_losses[0]
    assert output.eval_losses[-1] < 0.5 * output.eval_losses[0]
    assert model.backbone.is_frozen and model.head.is_fitted and not model.training


def test_pair_training_beats_the_frozen_baseline(trained):
    _, trainer, _, _ = trained
    test = dataset(10, 1)
    baseline = Trainer(make_model(), TrainingArguments(training_mode="frozen"), dataset(12, 0))
    baseline.train()
    adapted = trainer.evaluate(test)["accuracy"]
    assert adapted >= 0.95 and adapted > baseline.evaluate(test)["accuracy"] + 0.2


def test_same_seed_reproduces_the_run():
    losses = [
        Trainer(make_model(), pair_args(num_epochs=2), dataset(6, 0)).train().train_losses
        for _ in range(2)
    ]
    assert losses[0] == losses[1]
    assert (
        Trainer(make_model(), pair_args(num_epochs=2, seed=9), dataset(6, 0)).train().train_losses
        != losses[0]
    )


def test_step_count_and_schedule_follow_the_sampled_pairs(monkeypatch):
    from embedfit import representation
    from embedfit.sampling import PairSampler

    train = dataset(6, 0)
    args = pair_args(num_epochs=3, batch_size=16, positives_per_example=2, negatives_per_example=2)
    sampler = PairSampler(train, positives_per_example=2, negatives_per_example=2, seed=args.seed)
    expected_steps = sum(-(-len(sampler.sample(epoch)) // 16) for epoch in range(3))

    learning_rates = []
    original = torch.optim.AdamW.step

    def recording_step(self, *a, **k):
        learning_rates.append(self.param_groups[0]["lr"])
        return original(self, *a, **k)

    monkeypatch.setattr(representation.torch.optim.AdamW, "step", recording_step)
    output = Trainer(make_model(), args, train).train()
    assert output.representation_steps == expected_steps == len(learning_rates)
    assert learning_rates[0] == pytest.approx(args.learning_rate)
    assert learning_rates[-1] == pytest.approx(args.learning_rate / expected_steps)


def test_relationship_rules_reach_the_sampler_and_change_training():
    images, labels = make_bar_images(6, num_classes=3, seed=0)
    train = ImageDataset(
        [
            ImageSample(image, label, group_id=f"ring{i % 4}")
            for i, (image, label) in enumerate(zip(images, labels, strict=True))
        ]
    )
    rule = RelationshipRule("group_id", relation="same", pair_type="positive", weight=2.0)
    plain = Trainer(make_model(), pair_args(num_epochs=2), train).train()
    ruled = Trainer(make_model(), pair_args(num_epochs=2), train, relationship_rules=[rule]).train()
    assert ruled.train_losses != plain.train_losses


@pytest.mark.parametrize("mode", ["frozen", "supcon"])
def test_relationship_rules_are_rejected_outside_pair_mode(mode):
    rule = RelationshipRule("group_id")
    args = TrainingArguments(training_mode=mode, batch_size=12, num_workers=0)
    with pytest.raises(ConfigurationError, match="only apply to training_mode='pair'"):
        Trainer(make_model(), args, dataset(6, 0), relationship_rules=[rule])


def test_unpairable_data_fails_when_the_trainer_is_created():
    images, _ = make_bar_images(1, num_classes=3, seed=0)
    singletons = ImageDataset.from_lists(images, LABELS)
    with pytest.raises(ConfigurationError, match="No positive pair can be formed"):
        Trainer(make_model(), pair_args(), singletons)
    with pytest.raises(ConfigurationError, match="No positive pair can be formed"):
        Trainer(make_model(), pair_args(), dataset(6, 0), eval_dataset=singletons)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"positives_per_example": -1},
        {"negatives_per_example": 1.5},
        {"positives_per_example": 0, "negatives_per_example": 0},
    ],
)
def test_pair_arguments_are_validated_at_construction(kwargs):
    with pytest.raises(ConfigurationError, match="per_example"):
        TrainingArguments(**kwargs)
