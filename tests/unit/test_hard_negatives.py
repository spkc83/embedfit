import numpy as np
import pytest

from embedfit import ImageDataset, TrainingArguments
from embedfit.exceptions import ConfigurationError
from embedfit.sampling import PairSampler

# Three classes on a line: "a" at 0 degrees, "b" close to it at 20 degrees, "c" far at 120.
ANGLES = {"a": 0.0, "b": 20.0, "c": 120.0}
LABELS = [label for label in "abc" for _ in range(10)]


def geometry(seed=0):
    rng = np.random.default_rng(seed)
    angles = np.radians([ANGLES[label] for label in LABELS]) + rng.normal(0, 0.02, len(LABELS))
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def make_sampler(**kwargs):
    dataset = ImageDataset.from_lists(["x"] * len(LABELS), LABELS)
    return PairSampler(dataset, positives_per_example=2, negatives_per_example=4, seed=0, **kwargs)


def negative_classes_of(pairs, anchor_class):
    return [LABELS[p.right] for p in pairs if p.target == 0 and LABELS[p.left] == anchor_class]


def test_mined_negatives_are_the_nearest_other_class():
    sampler = make_sampler()
    random_pairs = sampler.sample(0)
    mined_pairs = sampler.sample(0, embeddings=geometry(), hard_negative_fraction=1.0)
    # For anchors of class "a" the hard negatives are all "b"; random ones are about half "c".
    assert set(negative_classes_of(mined_pairs, "a")) == {"b"}
    assert 0.2 < np.mean([c == "c" for c in negative_classes_of(random_pairs, "a")]) < 0.8
    # Class "c" is far from both others, so its hardest negatives are "b" (100 degrees away,
    # against 120 for "a").
    assert set(negative_classes_of(mined_pairs, "c")) == {"b"}


def test_mined_negatives_are_exactly_the_top_of_the_similarity_ranking():
    embeddings = geometry()
    pairs = make_sampler().sample(0, embeddings=embeddings, hard_negative_fraction=1.0)
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    anchor = 0  # The first anchor has no earlier pairs to exclude.
    chosen = sorted(p.right for p in pairs if p.left == anchor and p.target == 0)
    others = [i for i, label in enumerate(LABELS) if label != LABELS[anchor]]
    ranked = sorted(others, key=lambda i: (-float(unit[i] @ unit[anchor]), i))
    assert chosen == sorted(ranked[:4])


def test_fraction_splits_the_quota_between_hard_and_random():
    embeddings = geometry()
    half = make_sampler().sample(0, embeddings=embeddings, hard_negative_fraction=0.5)
    classes = negative_classes_of(half, "a")
    assert "c" in classes and classes.count("b") > classes.count("c")  # 2 hard + 2 random.
    none = make_sampler().sample(0, embeddings=embeddings, hard_negative_fraction=0.0)
    assert none == make_sampler().sample(0)  # Fraction 0 is exactly the random sampler.


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 1.0])
def test_mining_keeps_every_pair_guarantee(fraction):
    # With a partial fraction the random pass runs after the mined one and must not pick a
    # partner the mined pass already took.
    pairs = make_sampler().sample(3, embeddings=geometry(), hard_negative_fraction=fraction)
    assert all(p.left != p.right for p in pairs)
    unordered = [frozenset((p.left, p.right)) for p in pairs]
    assert len(set(unordered)) == len(unordered)
    assert all(p.target == int(LABELS[p.left] == LABELS[p.right]) for p in pairs)
    per_anchor = {}
    for p in pairs:
        per_anchor[p.left, p.target] = per_anchor.get((p.left, p.target), 0) + 1
    assert max(per_anchor.values()) <= 4


def test_mining_is_deterministic_and_scale_free():
    embeddings = geometry()
    first = make_sampler().sample(1, embeddings=embeddings, hard_negative_fraction=0.5)
    assert first == make_sampler().sample(
        1, embeddings=embeddings * 37.0, hard_negative_fraction=0.5
    )


def test_positives_stay_random_under_mining():
    embeddings = geometry()
    mined = make_sampler().sample(0, embeddings=embeddings, hard_negative_fraction=1.0)
    plain = make_sampler().sample(0)
    assert sorted((p.left, p.right) for p in mined if p.target == 1) == sorted(
        (p.left, p.right) for p in plain if p.target == 1
    )


@pytest.mark.parametrize(
    "embeddings, fraction, match",
    [
        (None, 0.5, "needs finite float embeddings"),
        (np.zeros((5, 2), dtype=np.float32), 0.5, r"shaped \[30, D\]"),
        (np.zeros(30, dtype=np.float32), 0.5, r"shaped \[30, D\]"),
        (np.full((30, 2), np.nan, dtype=np.float32), 0.5, "finite"),
        (np.zeros((30, 2), dtype=np.int64), 0.5, "float"),
        (geometry(), 1.5, r"\[0, 1\]"),
        (geometry(), -0.1, r"\[0, 1\]"),
        (geometry(), True, r"\[0, 1\]"),
    ],
)
def test_invalid_mining_arguments(embeddings, fraction, match):
    with pytest.raises(ConfigurationError, match=match):
        make_sampler().sample(0, embeddings=embeddings, hard_negative_fraction=fraction)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"training_mode": "pair", "hard_negative_fraction": 1.5}, r"\[0, 1\]"),
        ({"training_mode": "pair", "hard_negative_refresh_epochs": 0}, "refresh_epochs"),
        (
            {"training_mode": "supcon", "hard_negative_fraction": 0.5},
            "requires training_mode='pair'",
        ),
        (
            {"training_mode": "pair", "hard_negative_fraction": 0.5, "negatives_per_example": 0},
            "negatives_per_example > 0",
        ),
    ],
)
def test_mining_arguments_are_validated_with_the_configuration(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        TrainingArguments(**kwargs)


def test_mining_matches_a_brute_force_ranking_on_random_problems():
    rng = np.random.default_rng(0)
    for trial in range(40):
        count, classes = int(rng.integers(6, 40)), int(rng.integers(2, 5))
        labels = [f"c{i % classes}" for i in range(count)]
        embeddings = rng.normal(size=(count, 4)).astype(np.float32)
        if trial % 4 == 0:
            embeddings[count // 2 :] = embeddings[: count - count // 2]  # Exact duplicates.
        dataset = ImageDataset.from_lists(["x"] * count, labels)
        sampler = PairSampler(dataset, positives_per_example=1, negatives_per_example=3, seed=trial)
        pairs = sampler.sample(0, embeddings=embeddings, hard_negative_fraction=1.0)
        unit = embeddings.astype(np.float64)
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        taken: set[frozenset[int]] = set()
        by_anchor: dict[int, list] = {}
        for pair in pairs:  # Shuffled output: rebuild the anchor order first.
            by_anchor.setdefault(pair.left, []).append(pair)
        for anchor in range(count):
            mined = sorted(p.right for p in by_anchor.get(anchor, []) if p.target == 0)
            pool = [
                i
                for i in range(count)
                if labels[i] != labels[anchor] and frozenset((anchor, i)) not in taken
            ]
            positives = [p.right for p in by_anchor.get(anchor, []) if p.target == 1]
            scores = {i: float(unit[i] @ unit[anchor]) for i in pool}
            ranked = sorted(pool, key=lambda i: (-round(scores[i], 9), i))
            assert mined == sorted(ranked[: len(mined)]), (trial, anchor)
            assert len(mined) == min(3, len(pool))
            taken.update(frozenset((anchor, i)) for i in mined + positives)


def test_a_fraction_that_rounds_to_no_mined_negative_is_rejected():
    assert TrainingArguments(
        training_mode="pair", hard_negative_fraction=0.125
    ).hard_negative_fraction
    with pytest.raises(ConfigurationError, match="rounds to no mined negative"):
        TrainingArguments(training_mode="pair", hard_negative_fraction=0.1)  # 4 * 0.1 -> 0.
    with pytest.raises(ConfigurationError, match="rounds to no mined negative"):
        make_sampler().sample(0, embeddings=geometry(), hard_negative_fraction=0.05)


def test_mining_a_few_thousand_samples_takes_seconds():
    import time

    count = 3000
    dataset = ImageDataset.from_lists(["x"] * count, [f"c{i % 5}" for i in range(count)])
    embeddings = np.random.default_rng(0).normal(size=(count, 64)).astype(np.float32)
    start = time.perf_counter()
    pairs = PairSampler(dataset, seed=0).sample(
        0, embeddings=embeddings, hard_negative_fraction=0.5
    )
    assert len(pairs) > 20_000 and time.perf_counter() - start < 30  # Was minutes per epoch.
