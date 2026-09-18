import numpy as np
import pytest
import torch
from PIL import Image

from embedfit.backbones import DummyBackbone
from embedfit.exceptions import ConfigurationError, ImageInputError


def _batch(backbone, images):
    return torch.stack([backbone.preprocess(image) for image in images])


def test_embedding_shape_and_dtype(dummy_backbone, bar_dataset):
    images, _ = bar_dataset
    embeddings = dummy_backbone.forward_features(_batch(dummy_backbone, images))
    assert embeddings.shape == (len(images), dummy_backbone.embedding_dim)
    assert embeddings.is_floating_point()
    assert torch.isfinite(embeddings).all()


def test_same_seed_gives_identical_weights_and_leaves_global_rng_alone():
    torch.manual_seed(99)
    expected_next = torch.rand(1)
    torch.manual_seed(99)
    first, second = DummyBackbone(seed=5), DummyBackbone(seed=5)
    assert torch.equal(torch.rand(1), expected_next)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)
    other = DummyBackbone(seed=6)
    assert not torch.equal(first.projection.weight, other.projection.weight)


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (48, 20)),
        Image.new("L", (48, 20)),
        Image.new("RGBA", (48, 20)),
        np.zeros((20, 48), dtype=np.uint8),
        np.zeros((20, 48, 1), dtype=np.uint8),
        np.zeros((20, 48, 3), dtype=np.float32),
        torch.zeros(20, 48),
        torch.zeros(1, 20, 48, dtype=torch.uint8),
        torch.zeros(3, 20, 48),
    ],
    ids=[
        "pil-rgb",
        "pil-gray",
        "pil-rgba",
        "np-hw",
        "np-hw1",
        "np-hw3-float",
        "t-hw",
        "t-1hw-uint8",
        "t-3hw",
    ],
)
def test_preprocess_accepts_supported_inputs(dummy_backbone, image):
    pixels = dummy_backbone.preprocess(image)
    assert pixels.shape == (3, 32, 32)
    assert pixels.dtype == torch.float32


def test_preprocess_preserves_axis_and_channel_layout(dummy_backbone):
    # Every channel and both spatial axes are distinguishable, so a transposed or
    # channel-reversed conversion cannot pass.
    rows = torch.arange(32, dtype=torch.float32).view(32, 1).expand(32, 32)
    cols = torch.arange(32, dtype=torch.float32).view(1, 32).expand(32, 32)
    expected = torch.stack([rows / 31, cols / 31, torch.full((32, 32), 0.25)])
    hwc_uint8 = (expected.permute(1, 2, 0) * 255).round().to(torch.uint8)
    expected_uint8 = hwc_uint8.permute(2, 0, 1).float() / 255

    assert torch.equal(dummy_backbone.preprocess(expected), expected)
    assert torch.equal(dummy_backbone.preprocess(expected.permute(1, 2, 0).numpy()), expected)
    assert torch.equal(dummy_backbone.preprocess(hwc_uint8.numpy()), expected_uint8)
    pil = Image.fromarray(hwc_uint8.numpy())
    assert torch.equal(dummy_backbone.preprocess(pil), expected_uint8)


def test_preprocess_never_aliases_or_tracks_the_input(dummy_backbone):
    source = torch.full((3, 32, 32), 0.5, requires_grad=True)
    pixels = dummy_backbone.preprocess(source)
    assert pixels.data_ptr() != source.data_ptr()
    assert not pixels.requires_grad

    read_only = np.full((32, 32, 3), 0.5, dtype=np.float32)
    read_only.setflags(write=False)
    dummy_backbone.preprocess(read_only).zero_()
    assert read_only.min() == 0.5


def test_preprocess_scales_uint8_and_replicates_grayscale(dummy_backbone):
    pixels = dummy_backbone.preprocess(np.full((32, 32), 255, dtype=np.uint8))
    assert torch.allclose(pixels, torch.ones(3, 32, 32))


@pytest.mark.parametrize(
    "bad, match",
    [
        ("not-an-image", "Unsupported image type"),
        (np.zeros((4, 4, 3, 1), dtype=np.uint8), "dimensions"),
        (np.zeros((8, 8, 2), dtype=np.uint8), "1 or 3 channels"),
        (torch.zeros(4, 8, 8), "1 or 3 channels"),
        (np.zeros((8, 8), dtype=np.int64), "dtype"),
        (np.zeros((0, 8), dtype=np.uint8), "empty"),
        (np.full((8, 8, 3), 200.0, dtype=np.float32), r"lie in \[0, 1\]"),
        (np.full((8, 8), -0.1), r"lie in \[0, 1\]"),
        (torch.full((3, 8, 8), float("nan")), "finite"),
    ],
)
def test_preprocess_rejects_malformed_inputs(dummy_backbone, bad, match):
    with pytest.raises(ImageInputError, match=match):
        dummy_backbone.preprocess(bad)


def test_error_messages_do_not_leak_pixel_values(dummy_backbone):
    secret = np.full((8, 8, 2), 173, dtype=np.uint8)
    with pytest.raises(ImageInputError) as excinfo:
        dummy_backbone.preprocess(secret)
    assert "173" not in str(excinfo.value)


def test_multiframe_image_rejected(dummy_backbone, tmp_path):
    frames = [Image.new("L", (16, 16), color=value) for value in (0, 255)]
    path = tmp_path / "frames.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    with Image.open(path) as image, pytest.raises(ImageInputError, match="Multi-frame"):
        dummy_backbone.preprocess(image)


def test_forward_features_rejects_wrong_shape(dummy_backbone):
    with pytest.raises(ImageInputError, match="pixel_values"):
        dummy_backbone.forward_features(torch.zeros(3, 32, 32))


def test_freeze_and_unfreeze(dummy_backbone):
    dummy_backbone.train()
    dummy_backbone.freeze()
    assert not any(p.requires_grad for p in dummy_backbone.parameters())
    assert not dummy_backbone.training
    dummy_backbone.unfreeze()
    assert all(p.requires_grad for p in dummy_backbone.parameters())


def test_frozen_backbone_receives_no_gradients(dummy_backbone, bar_dataset):
    dummy_backbone.freeze()
    output = dummy_backbone.forward_features(_batch(dummy_backbone, bar_dataset[0][:4]))
    assert not output.requires_grad


def test_gradients_flow_when_unfrozen(dummy_backbone, bar_dataset):
    dummy_backbone.freeze()
    dummy_backbone.unfreeze()
    dummy_backbone.forward_features(_batch(dummy_backbone, bar_dataset[0][:4])).sum().backward()
    grads = [p.grad for p in dummy_backbone.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_save_and_reload_round_trip(dummy_backbone, bar_dataset, tmp_path):
    with torch.no_grad():  # Perturb so the round trip cannot pass by re-seeding alone.
        dummy_backbone.projection.weight.add_(0.5)
    dummy_backbone.eval()
    batch = _batch(dummy_backbone, bar_dataset[0])
    dummy_backbone.save_pretrained(tmp_path / "backbone")
    loaded = DummyBackbone.from_pretrained(tmp_path / "backbone")
    assert loaded.embedding_dim == dummy_backbone.embedding_dim
    assert loaded.image_size == dummy_backbone.image_size
    with torch.no_grad():
        assert torch.allclose(loaded.forward_features(batch), dummy_backbone(batch))
    assert sorted(p.name for p in (tmp_path / "backbone").iterdir()) == [
        "config.json",
        "weights.pt",
    ]


def test_from_pretrained_rejects_foreign_directory(tmp_path):
    with pytest.raises(ConfigurationError, match="not a DummyBackbone directory"):
        DummyBackbone.from_pretrained(tmp_path)


@pytest.mark.parametrize(
    "config_text, match",
    [
        ("{not json", "not valid JSON"),
        ('{"provider": "other"}', "not written by DummyBackbone"),
        ('{"provider": "dummy", "seed": 0}', "missing required keys"),
    ],
)
def test_from_pretrained_rejects_bad_config(dummy_backbone, tmp_path, config_text, match):
    dummy_backbone.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(config_text)
    with pytest.raises(ConfigurationError, match=match):
        DummyBackbone.from_pretrained(tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_dim": 0},
        {"image_size": -1},
        {"embedding_dim": 2.5},
        {"image_size": True},
        {"seed": 2.5},
    ],
)
def test_invalid_constructor_arguments(kwargs):
    with pytest.raises(ConfigurationError):
        DummyBackbone(**kwargs)
