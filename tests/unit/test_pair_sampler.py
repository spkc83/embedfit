from collections import Counter

import numpy as np
import pytest

from embedfit import ImageDataset
from embedfit.data import ImageSample
from embedfit.exceptions import ConfigurationError
from embedfit.sampling import ImagePair, PairSampler, RelationshipRule


def make_dataset(per_class=(10, 6, 4)) -> ImageDataset:
    samples = []
    for class_index, count in enumerate(per_class):
        for _ in range(count):
            position = len(samples)
            samples.append(
                ImageSample(
                    f"{position}.png",
                    f"c{class_index}",
                    sample_id=f"s{position}",
                    group_id=f"ring{position % 5}",
                    metadata={"template": position % 2},
                )
            )
    return ImageDataset(samples)


LABELS = make_dataset().labels


def test_class_semantics_without_rules():
    pairs = PairSampler(make_dataset(), seed=0).sample()
    assert pairs and all(isinstance(pair, ImagePair) for pair in pairs)
    for pair in pairs:
        same_class = LABELS[pair.left] == LABELS[pair.right]
        assert pair.target == int(same_class)  # positive → same class, negative → different.
        assert pair.weight == 1.0 and pair.relationship is None


def test_no_self_pairs_and_no_unordered_duplicates():
    pairs = PairSampler(make_dataset(), seed=0).sample()
    assert all(pair.left != pair.right for pair in pairs)
    unordered = [frozenset((pair.left, pair.right)) for pair in pairs]
    assert len(set(unordered)) == len(unordered)


def test_pair_count_is_bounded_and_far_below_all_pairs():
    dataset = ImageDataset.from_lists(["x"] * 300, [f"c{i % 3}" for i in range(300)])
    sampler = PairSampler(dataset, positives_per_example=2, negatives_per_example=3, seed=0)
    pairs = sampler.sample()
    assert len(pairs) <= len(sampler) == 300 * 5
    assert len(pairs) < 0.04 * (300 * 299 / 2)  # Nowhere near the 44 850 possible pairs.
    assert len(pairs) > 0.8 * len(sampler)  # The quota is nearly filled despite de-duplication.


def test_positives_and_negatives_are_balanced_per_anchor():
    pairs = PairSampler(
        make_dataset(), positives_per_example=3, negatives_per_example=3, seed=1
    ).sample()
    targets = Counter(pair.target for pair in pairs)
    assert abs(targets[1] - targets[0]) <= 0.2 * len(pairs)
    per_anchor = Counter((pair.left, pair.target) for pair in pairs)
    assert max(per_anchor.values()) <= 3


def test_small_class_caps_its_positive_pairs():
    dataset = ImageDataset.from_lists(["x"] * 8, ["a", "a", "b", "b", "b", "b", "b", "b"])
    pairs = PairSampler(dataset, positives_per_example=4, negatives_per_example=0, seed=0).sample()
    a_pairs = [pair for pair in pairs if dataset.labels[pair.left] == "a"]
    assert len(a_pairs) == 1  # Class "a" has 2 samples: exactly one positive pair exists.


def test_pairs_depend_only_on_seed_and_epoch():
    np.random.seed(3)
    expected_next = np.random.rand()
    np.random.seed(3)
    sampler = PairSampler(make_dataset(), seed=5)
    first = sampler.sample(epoch=0)
    assert np.random.rand() == expected_next  # Global RNG untouched.
    assert first == PairSampler(make_dataset(), seed=5).sample(epoch=0)
    assert first != sampler.sample(epoch=1)
    assert first != PairSampler(make_dataset(), seed=6).sample(epoch=0)
    assert sampler.sample(epoch=0) == first


def test_rule_overrides_class_labels_only_where_it_matches():
    rule = RelationshipRule(field="group_id", relation="same", pair_type="positive", weight=2.0)
    dataset = make_dataset()
    pairs = PairSampler(
        dataset, positives_per_example=6, negatives_per_example=6, seed=0, rules=[rule]
    ).sample()
    groups = [sample.group_id for sample in dataset]
    by_rule = [pair for pair in pairs if pair.relationship == "same:group_id"]
    assert by_rule and any(LABELS[p.left] != LABELS[p.right] for p in by_rule)
    for pair in pairs:
        same_group = groups[pair.left] == groups[pair.right]
        if same_group:
            assert (pair.target, pair.weight, pair.relationship) == (1, 2.0, "same:group_id")
        else:
            assert pair.relationship is None and pair.weight == 1.0
            assert pair.target == int(LABELS[pair.left] == LABELS[pair.right])


def test_metadata_field_negative_rule_and_rule_order():
    different_template = RelationshipRule("template", relation="different", pair_type="negative")
    same_group = RelationshipRule("group_id", pair_type="positive", weight=3.0)
    dataset = make_dataset()
    templates = [sample.metadata["template"] for sample in dataset]
    groups = [sample.group_id for sample in dataset]
    for rules, winner in (
        ([different_template, same_group], "different:template"),
        ([same_group, different_template], "same:group_id"),
    ):
        pairs = PairSampler(
            dataset, positives_per_example=8, negatives_per_example=8, seed=0, rules=rules
        ).sample()
        both = [
            p
            for p in pairs
            if templates[p.left] != templates[p.right] and groups[p.left] == groups[p.right]
        ]
        assert both and all(pair.relationship == winner for pair in both)  # First rule wins.


def test_samples_without_a_value_never_match_a_rule():
    dataset = ImageDataset(
        [
            ImageSample(f"{i}.png", "a" if i < 3 else "b", group_id=None if i % 2 else "g")
            for i in range(6)
        ]
    )
    rule = RelationshipRule("group_id", pair_type="positive")
    for pair in PairSampler(dataset, seed=0, rules=[rule]).sample():
        if pair.relationship:
            assert pair.left % 2 == 0 and pair.right % 2 == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"positives_per_example": -1}, "positives_per_example"),
        ({"negatives_per_example": 1.5}, "negatives_per_example"),
        ({"positives_per_example": 0, "negatives_per_example": 0}, "cannot both be 0"),
        ({"seed": -1}, "seed"),
        ({"rules": ["group_id"]}, "RelationshipRule objects"),
    ],
)
def test_invalid_options(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        PairSampler(make_dataset(), **{"seed": 0, **kwargs})


def test_impossible_datasets():
    with pytest.raises(ConfigurationError, match="at least 2 samples"):
        PairSampler(ImageDataset.from_lists(["x"], ["a"]), seed=0)
    with pytest.raises(ConfigurationError, match="No negative pair can be formed"):
        PairSampler(ImageDataset.from_lists(["x"] * 3, ["a"] * 3), seed=0)
    with pytest.raises(ConfigurationError, match="No positive pair can be formed"):
        PairSampler(ImageDataset.from_lists(["x"] * 3, ["a", "b", "c"]), seed=0)
    with pytest.raises(ConfigurationError, match="must be an ImageDataset"):
        PairSampler(["a", "b"], seed=0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"field": ""}, "field"),
        ({"field": "g", "relation": "similar"}, "relation"),
        ({"field": "g", "pair_type": "neutral"}, "pair_type"),
        ({"field": "g", "weight": 0}, "weight"),
        ({"field": "g", "weight": True}, "weight"),
    ],
)
def test_invalid_rules(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        RelationshipRule(**kwargs)
