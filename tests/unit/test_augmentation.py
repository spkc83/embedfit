"""Augmentation profiles and the seeded, validated pipeline call (SPEC §13)."""

import math

import numpy as np
import pytest
import torch

from embedfit.augmentation import DocumentAugmentation, NaturalAugmentation
from embedfit.augmentation.pipeline import augment_image, probe_augmentation
from embedfit.data import to_rgb_tensor
from embedfit.exceptions import ConfigurationError
from embedfit.testing import make_bar_images

PROFILES = [NaturalAugmentation, DocumentAugmentation]


def bar_image():
    images, _ = make_bar_images(1, num_classes=2, image_size=40, seed=0)
    return to_rgb_tensor(images[1])


def split_image(height=30, width=40):
    """Dark left half, bright right half: a horizontal mirror swaps them."""
    image = torch.zeros(3, height, width)
    image[:, :, width // 2 :] = 1.0
    return image


def run(pipeline, image, seed):
    torch.manual_seed(seed)
    return pipeline(image.clone())


@pytest.mark.parametrize("profile", PROFILES)
def test_output_is_a_float_rgb_tensor_in_the_unit_range(profile):
    pipeline, image = profile(), bar_image()
    for seed in range(30):
        output = run(pipeline, image, seed)
        assert output.dtype == torch.float32 and output.ndim == 3 and output.shape[0] == 3
        assert min(output.shape[1:]) >= 1
        assert torch.isfinite(output).all() and output.min() >= 0 and output.max() <= 1


@pytest.mark.parametrize("profile", PROFILES)
def test_the_torch_seed_decides_the_output(profile):
    pipeline, image = profile(), bar_image()
    assert torch.equal(run(pipeline, image, 3), run(pipeline, image, 3))
    assert any(not torch.equal(run(pipeline, image, 3), run(pipeline, image, s)) for s in (4, 5))


@pytest.mark.parametrize("profile", PROFILES)
def test_the_input_tensor_is_not_modified(profile):
    image = bar_image()
    original = image.clone()
    torch.manual_seed(0)
    profile()(image)
    assert torch.equal(image, original)


@pytest.mark.parametrize("profile", PROFILES)
def test_grayscale_input_stays_grayscale(profile):
    image = bar_image()[:1].expand(3, -1, -1).contiguous()
    for seed in range(20):
        output = run(profile(), image, seed)
        assert torch.allclose(output[0], output[1]) and torch.allclose(output[1], output[2])


def test_natural_flips_horizontally_about_half_the_time():
    pipeline = NaturalAugmentation(
        crop_scale=(1.0, 1.0), crop_ratio=(4 / 3, 4 / 3), jitter_probability=0.0
    )
    image = split_image()
    outcomes = []
    for seed in range(50):
        output = run(pipeline, image, seed)
        if torch.equal(output, image):
            outcomes.append("kept")
        else:
            assert torch.equal(output, image.flip(-1))
            outcomes.append("flipped")
    assert 10 < outcomes.count("flipped") < 40


def test_natural_crops_stay_within_the_scale_and_ratio_bounds():
    pipeline = NaturalAugmentation(crop_scale=(0.5, 1.0), jitter_probability=0.0)
    image = torch.rand(3, 120, 160, generator=torch.Generator().manual_seed(0))
    for seed in range(50):
        output = run(pipeline, image, seed)
        height, width = output.shape[1:]
        if (height, width) == (120, 160):
            continue  # The whole image: also the fallback when no crop fits.
        area = height * width / (120 * 160)
        assert 0.5 - 0.03 <= area <= 1.0 + 0.03
        assert 3 / 4 - 0.05 <= width / height <= 4 / 3 + 0.05


def test_natural_colour_jitter_changes_colour_and_can_be_disabled():
    image = bar_image()
    plain = NaturalAugmentation(crop_scale=(1.0, 1.0), crop_ratio=(1.0, 1.0), flip_probability=0)
    assert any(not torch.equal(run(plain, image, seed), image) for seed in range(5))
    off = NaturalAugmentation(
        crop_scale=(1.0, 1.0), crop_ratio=(1.0, 1.0), flip_probability=0, jitter_probability=0
    )
    assert all(torch.equal(run(off, image, seed), image) for seed in range(5))


def test_document_never_mirrors_the_page():
    pipeline = DocumentAugmentation(max_rotation=0.0, brightness=0.0, contrast=0.0)
    image = split_image()
    for seed in range(200):
        output = run(pipeline, image, seed)
        width = output.shape[2]
        assert output[:, :, : width // 4].mean() < output[:, :, -(width // 4) :].mean()


def test_document_crops_are_mild_and_keep_the_aspect_ratio():
    pipeline = DocumentAugmentation(max_rotation=0.0, brightness=0.0, contrast=0.0)
    image = torch.rand(3, 200, 150, generator=torch.Generator().manual_seed(0))
    sizes = set()
    for seed in range(100):
        height, width = run(pipeline, image, seed).shape[1:]
        sizes.add((height, width))
        assert height >= 0.92 * 200 - 1 and width >= 0.92 * 150 - 1
        assert abs(width / height - 150 / 200) < 0.01
    assert len(sizes) > 1


def test_document_rotation_stays_within_its_bound():
    image = torch.zeros(3, 101, 101)
    image[:, :, 50] = 1.0  # A vertical line through the centre.
    pipeline = DocumentAugmentation(
        crop_scale=(1.0, 1.0), max_rotation=2.0, brightness=0.0, contrast=0.0
    )
    # A line tilted by 2 degrees moves its ends by tan(2°) * 50 ≈ 1.75 pixels.
    tilts = []
    for seed in range(30):
        output = run(pipeline, image, seed)[0]
        top = (output[5] * torch.arange(101.0)).sum() / output[5].sum()
        bottom = (output[95] * torch.arange(101.0)).sum() / output[95].sum()
        tilts.append(abs(float(top - bottom)))
    assert max(tilts) <= math.tan(math.radians(2.0)) * 90 + 0.5
    assert max(tilts) > 0.5  # The page does rotate.


def test_document_keeps_the_hue():
    image = torch.zeros(3, 20, 20)
    image[0] = 0.8  # A red page stays red: contrast moves every channel towards the same grey.
    pipeline = DocumentAugmentation(crop_scale=(1.0, 1.0), max_rotation=0.0)
    for seed in range(20):
        output = run(pipeline, image, seed)
        assert torch.equal(output[1], output[2]) and (output[0] > output[1]).all()


@pytest.mark.parametrize(
    "profile, kwargs, match",
    [
        (NaturalAugmentation, {"crop_scale": (0.0, 1.0)}, "crop_scale"),
        (NaturalAugmentation, {"crop_scale": (0.9, 0.5)}, "crop_scale"),
        (NaturalAugmentation, {"crop_scale": (0.5, 1.5)}, "crop_scale"),
        (NaturalAugmentation, {"crop_ratio": (0.0, 1.0)}, "crop_ratio"),
        (NaturalAugmentation, {"flip_probability": 1.5}, "flip_probability"),
        (NaturalAugmentation, {"jitter_probability": -0.1}, "jitter_probability"),
        (NaturalAugmentation, {"brightness": -0.1}, "brightness"),
        (NaturalAugmentation, {"saturation": float("nan")}, "saturation"),
        (DocumentAugmentation, {"crop_scale": (0.0, 1.0)}, "crop_scale"),
        (DocumentAugmentation, {"max_rotation": -1.0}, "max_rotation"),
        (DocumentAugmentation, {"max_rotation": 200.0}, "max_rotation"),
        (DocumentAugmentation, {"contrast": True}, "contrast"),
    ],
)
def test_invalid_parameters_fail_at_construction(profile, kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        profile(**kwargs)


def noisy(image):
    return (image + torch.rand(image.shape)).clamp(0, 1)


def test_augment_image_is_reproducible_per_seed_epoch_and_draw():
    image = bar_image()
    reference = augment_image(noisy, image, seed=1, epoch=2, draw=3)
    assert torch.equal(reference, augment_image(noisy, image, seed=1, epoch=2, draw=3))
    for changed in ({"seed": 2}, {"epoch": 3}, {"draw": 4}):
        keys = {"seed": 1, "epoch": 2, "draw": 3, **changed}
        assert not torch.equal(reference, augment_image(noisy, image, **keys))


def test_augment_image_leaves_the_global_generators_alone(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    before = torch.get_rng_state()
    augment_image(noisy, bar_image(), seed=0, epoch=0, draw=0)
    assert torch.equal(before, torch.get_rng_state())
    assert calls == []  # Only the CPU generator is seeded, never the accelerators'.


def test_augment_image_returns_float32():
    output = augment_image(lambda image: image.double(), bar_image(), seed=0, epoch=0, draw=0)
    assert output.dtype == torch.float32


class Broken:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image)


@pytest.mark.parametrize(
    "transform, match",
    [
        (lambda image: image.numpy(), "torch.Tensor"),
        (lambda image: image[:1], r"\[3, H, W\]"),
        (lambda image: image[0], r"\[3, H, W\]"),
        (lambda image: image[:, :0], r"\[3, H, W\]"),
        (lambda image: (image * 255).to(torch.uint8), "floating-point"),
        (lambda image: image * 3 + 1, r"\[0, 1\]"),
        (lambda image: image * float("nan"), r"\[0, 1\]"),
    ],
)
def test_a_malformed_pipeline_output_is_a_configuration_error(transform, match):
    image = bar_image()
    with pytest.raises(ConfigurationError, match=match) as raised:
        augment_image(Broken(transform), image, seed=0, epoch=0, draw=0)
    assert "Broken" in str(raised.value)
    assert raised.value.__cause__ is None
    with pytest.raises(ConfigurationError, match=match):
        probe_augmentation(Broken(transform), seed=0)


def test_the_probe_accepts_the_built_in_profiles():
    for profile in PROFILES:
        probe_augmentation(profile(), seed=0)


def test_numpy_numbers_are_accepted_as_parameters():
    profile = DocumentAugmentation(max_rotation=np.int64(2), brightness=np.float32(0.1))
    assert profile.max_rotation == 2.0 and isinstance(profile.max_rotation, float)
    assert NaturalAugmentation(crop_scale=(np.float64(0.5), np.int64(1))).crop_scale == (0.5, 1.0)
