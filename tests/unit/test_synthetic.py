import numpy as np
import pytest

from embedfit.exceptions import ConfigurationError
from embedfit.testing import BAR_CLASSES, make_bar_images


def test_counts_labels_and_format():
    images, labels = make_bar_images(5, num_classes=3, image_size=24)
    assert len(images) == len(labels) == 15
    assert labels == [name for name in BAR_CLASSES[:3] for _ in range(5)]
    assert all(image.size == (24, 24) and image.mode == "RGB" for image in images)


def test_grayscale_option():
    images, _ = make_bar_images(1, grayscale=True)
    assert all(image.mode == "L" for image in images)


def test_deterministic_and_independent_of_global_rng():
    np.random.seed(0)
    first, _ = make_bar_images(3, seed=7)
    np.random.seed(12345)
    second, _ = make_bar_images(3, seed=7)
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(first, second, strict=True)
    )
    different, _ = make_bar_images(3, seed=8)
    assert not np.array_equal(np.asarray(first[0]), np.asarray(different[0]))


def test_bars_have_the_documented_orientation():
    images, _ = make_bar_images(1, num_classes=4, image_size=32, noise=0.0)
    vertical, horizontal, diagonal, anti = (np.asarray(i.convert("L")) > 127 for i in images)
    assert vertical.all(axis=0).any() and not vertical.all(axis=1).any()
    assert horizontal.all(axis=1).any() and not horizontal.all(axis=0).any()
    # Along a diagonal bar the row and column indices rise together; along an anti-diagonal
    # bar they move in opposite directions. Neither fills a whole row or column.
    thickness = 32 // 8
    for mask in (vertical, horizontal):
        assert mask.sum() == thickness * 32
    for mask, sign in ((diagonal, 1), (anti, -1)):
        assert (mask.sum(axis=1).max(), mask.sum(axis=0).max()) == (thickness, thickness)
        assert not mask.all(axis=0).any() and not mask.all(axis=1).any()
        rows, cols = np.nonzero(mask)
        assert sign * np.corrcoef(rows, cols)[0, 1] > 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_per_class": 0},
        {"num_per_class": 2, "num_classes": 1},
        {"num_per_class": 2, "num_classes": 5},
        {"num_per_class": 2, "image_size": 4},
        {"num_per_class": 2, "noise": -0.1},
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ConfigurationError):
        make_bar_images(**kwargs)
