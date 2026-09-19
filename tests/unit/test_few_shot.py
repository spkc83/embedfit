from collections import Counter

import numpy as np
import pytest

from embedfit import ImageDataset
from embedfit.data import ImageSample
from embedfit.exceptions import ConfigurationError
from embedfit.sampling import sample_few_shot


def make_pool() -> ImageDataset:
    labels = ["a"] * 10 + ["b"] * 6 + ["c"] * 4
    return ImageDataset(
        [ImageSample(f"{i}.png", label, sample_id=f"s{i}") for i, label in enumerate(labels)]
    )


def ids(dataset):
    return [sample.sample_id for sample in dataset]


@pytest.mark.parametrize("shots", [1, 2, 4])
def test_draws_exactly_n_per_class_without_replacement(shots):
    sample = sample_few_shot(make_pool(), shots=shots, seed=0)
    assert Counter(sample.labels) == {"a": shots, "b": shots, "c": shots}
    assert len(set(ids(sample))) == 3 * shots
    positions = [int(i[1:]) for i in ids(sample)]
    assert positions == sorted(positions)  # Original relative order.


def test_seed_determinism_without_touching_global_rng():
    np.random.seed(3)
    expected_next = np.random.rand()
    np.random.seed(3)
    first = sample_few_shot(make_pool(), shots=2, seed=11)
    assert np.random.rand() == expected_next
    assert ids(first) == ids(sample_few_shot(make_pool(), shots=2, seed=11))
    draws = {tuple(ids(sample_few_shot(make_pool(), shots=2, seed=s))) for s in range(10)}
    assert len(draws) > 1


def test_every_sample_of_a_class_is_reachable():
    seen = set()
    for seed in range(60):
        seen.update(ids(sample_few_shot(make_pool(), shots=1, seed=seed)))
    assert seen == set(ids(make_pool()))


def test_short_class_is_an_error_not_sampling_with_replacement():
    with pytest.raises(
        ConfigurationError,
        match=r"1 of 3 classes have fewer than 5 samples "
        r"\(the smallest has 4\)",
    ):
        sample_few_shot(make_pool(), shots=5, seed=0)


def test_labels_are_separated_by_type():
    pool = ImageDataset.from_lists(["x"] * 6, [1, 1, 1, True, True, True])
    assert sorted(map(repr, sample_few_shot(pool, shots=2, seed=0).labels)) == [
        "1",
        "1",
        "True",
        "True",
    ]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"shots": 0}, "shots"),
        ({"shots": 1.5}, "shots"),
        ({"shots": True}, "shots"),
        ({"seed": -1}, "seed"),
        ({"seed": None}, "seed"),
    ],
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        sample_few_shot(make_pool(), **{"shots": 2, "seed": 0, **kwargs})


def test_invalid_dataset():
    with pytest.raises(ConfigurationError, match="must be an ImageDataset"):
        sample_few_shot(["a"], shots=1, seed=0)
    with pytest.raises(ConfigurationError, match="empty"):
        sample_few_shot(ImageDataset([]), shots=1, seed=0)


def test_numpy_integers_are_accepted():
    plain = sample_few_shot(make_pool(), shots=2, seed=5)
    assert ids(sample_few_shot(make_pool(), shots=np.int64(2), seed=np.int32(5))) == ids(plain)


def test_unhashable_labels():
    pool = ImageDataset.from_lists(["a", "b"], [["x"], ["y"]])
    with pytest.raises(ConfigurationError, match="Labels must be hashable"):
        sample_few_shot(pool, shots=1, seed=0)
