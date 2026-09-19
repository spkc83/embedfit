"""Offline tests: every model here is built with ``weights=None``, so nothing is downloaded."""

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("torchvision")

from embedfit import EmbedFitModel  # noqa: E402
from embedfit.backbones import ImageBackbone, TorchvisionBackbone  # noqa: E402
from embedfit.exceptions import BackboneProviderError, ConfigurationError  # noqa: E402


@pytest.fixture
def resnet() -> TorchvisionBackbone:
    torch.manual_seed(0)
    return TorchvisionBackbone("resnet18", weights=None, image_size=64)


@pytest.mark.filterwarnings("ignore:The default weight initialization:FutureWarning")
@pytest.mark.parametrize(
    "name, expected_dim",
    [
        ("resnet18", 512),
        ("mobilenet_v3_small", 1024),
        ("googlenet", 1024),
        ("squeezenet1_0", None),
    ],
)
def test_classifier_removal_by_architecture_family(name, expected_dim):
    if expected_dim is None:  # Convolutional classifier: no linear layer to remove.
        with pytest.raises(BackboneProviderError, match="no linear classification layer"):
            TorchvisionBackbone(name, weights=None)
        return
    backbone = TorchvisionBackbone(name, weights=None, image_size=64)
    assert isinstance(backbone, ImageBackbone)
    assert backbone.identifier == f"torchvision/{name}"
    assert backbone.embedding_dim == expected_dim
    for training in (False, True):  # Auxiliary heads must not turn train output into a tuple.
        backbone.train(training)
        with torch.no_grad():
            assert backbone(torch.rand(2, 3, 64, 64)).shape == (2, expected_dim)


def test_without_weights_no_normalization_constants_are_assumed(resnet):
    assert resnet.mean is None and resnet.std is None
    white = resnet.preprocess(Image.new("RGB", (100, 80), "white"))
    assert white.shape == (3, 64, 64)
    assert torch.allclose(white, torch.ones_like(white))


def test_explicit_normalization_is_applied():
    backbone = TorchvisionBackbone(
        "resnet18", weights=None, image_size=32, mean=[0.5, 0.5, 0.5], std=[0.25, 0.5, 1.0]
    )
    pixels = backbone.preprocess(Image.new("RGB", (32, 32), "white"))
    assert torch.allclose(pixels[:, 0, 0], torch.tensor([2.0, 1.0, 0.5]))


def test_preprocess_resizes_shorter_side_then_center_crops():
    backbone = TorchvisionBackbone("resnet18", weights=None, image_size=32, resize_size=32)
    # White only in the middle half of a 2:1 image: a centre crop is all white, whereas
    # squashing the whole image to a square would keep black bands at the sides.
    wide = np.zeros((32, 64, 3), dtype=np.uint8)
    wide[:, 16:48] = 255
    pixels = backbone.preprocess(wide)
    assert pixels.shape == (3, 32, 32)
    assert pixels.min() > 0.95


@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic"])
def test_preprocess_matches_torchvision_preset(interpolation):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms._presets import ImageClassification

    mean, std = [0.4, 0.5, 0.6], [0.2, 0.25, 0.3]
    backbone = TorchvisionBackbone(
        "resnet18",
        weights=None,
        image_size=48,
        resize_size=56,
        interpolation=interpolation,
        mean=mean,
        std=std,
    )
    reference = ImageClassification(
        crop_size=48,
        resize_size=56,
        mean=mean,
        std=std,
        interpolation=InterpolationMode(interpolation),
    )
    rng = np.random.default_rng(0)
    smooth = rng.random((12, 16, 3)).repeat(10, axis=0).repeat(10, axis=1)
    image = Image.fromarray((smooth * 255).astype(np.uint8))
    difference = (backbone.preprocess(image) - reference(image)).abs() * torch.tensor(std).view(
        3, 1, 1
    )
    # Measured in [0, 1] intensity units. Resizing floats rather than 8-bit pixels shifts
    # values slightly (most at hard edges under bicubic); the wrong interpolation mode is
    # several times further off (mean ~0.013, max ~0.095 on this image).
    assert difference.mean().item() < 0.003
    assert difference.max().item() < 0.03


def test_weights_preset_supplies_interpolation_and_normalization(monkeypatch):
    import torchvision

    build = torchvision.models.get_model
    monkeypatch.setattr(  # Keep the preset lookup but skip the checkpoint download.
        torchvision.models, "get_model", lambda name, weights=None: build(name, weights=None)
    )
    backbone = TorchvisionBackbone("efficientnet_b0", weights="DEFAULT")
    preset = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()
    assert backbone.interpolation == "bicubic" == preset.interpolation.value
    assert (backbone.image_size, backbone.resize_size) == (
        preset.crop_size[0],
        preset.resize_size[0],
    )
    assert backbone.mean == tuple(preset.mean) and backbone.std == tuple(preset.std)


def test_fixed_resolution_model_rejects_other_sizes_at_construction():
    with pytest.raises(ConfigurationError, match="only accepts 224x224"):
        TorchvisionBackbone("vit_b_32", weights=None, image_size=64)


def test_image_size_is_configurable_and_resize_tracks_it():
    default = TorchvisionBackbone("resnet18", weights=None)
    assert (default.image_size, default.resize_size) == (224, 224)
    large = TorchvisionBackbone("resnet18", weights=None, image_size=448)
    assert large.preprocess(Image.new("L", (600, 500))).shape == (3, 448, 448)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"image_size": 0}, "image_size must be positive"),
        ({"image_size": 64.0}, "image_size must be an int"),
        ({"image_size": 64, "resize_size": 32}, "must be >= image_size"),
        ({"interpolation": "lanczos"}, "interpolation must be"),
        ({"mean": [0.5, 0.5, 0.5]}, "given together"),
        ({"mean": [0.5], "std": [0.5]}, "3 values"),
        ({"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.0, 0.5]}, "positive"),
    ],
)
def test_invalid_preprocessing_options(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        TorchvisionBackbone("resnet18", weights=None, **kwargs)


def test_unknown_model_name_suggests_alternatives():
    with pytest.raises(BackboneProviderError, match="Did you mean.*resnet18"):
        TorchvisionBackbone("resnet18x", weights=None)


def test_unknown_weights_name_lists_valid_ones():
    with pytest.raises(BackboneProviderError, match="IMAGENET1K_V1"):
        TorchvisionBackbone("resnet18", weights="NOT_A_WEIGHT")


def test_encode_through_model_and_freeze(resnet):
    model = EmbedFitModel(resnet)
    images = [Image.new("RGB", (80, 70), color) for color in ("red", "green", "blue")]
    embeddings = model.encode(images)
    assert embeddings.shape == (3, 512)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)

    model.freeze_backbone()
    model.train()
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(module.training for module in resnet.modules())
    model.unfreeze_backbone()


def test_from_pretrained_via_registry_passes_options():
    model = EmbedFitModel.from_pretrained(
        "torchvision/resnet18", labels=["legitimate", "risk"], weights=None, image_size=64
    )
    assert model.embedding_dim == 512
    assert model.backbone.image_size == 64


def test_save_and_reload_round_trip(tmp_path):
    torch.manual_seed(0)
    original = TorchvisionBackbone(
        "resnet18",
        weights=None,
        image_size=48,
        resize_size=56,
        interpolation="bicubic",
        mean=[0.1, 0.2, 0.3],
        std=[0.4, 0.5, 0.6],
    )
    original.eval()
    original.save_pretrained(tmp_path)
    loaded = TorchvisionBackbone.from_pretrained(tmp_path)

    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (70, 90, 3), dtype=np.uint8))
    pixels = original.preprocess(image)
    assert torch.equal(loaded.preprocess(image), pixels)
    assert (loaded.identifier, loaded.weights) == (original.identifier, original.weights)
    with torch.no_grad():
        assert torch.allclose(loaded(pixels[None]), original(pixels[None]), atol=1e-6)


@pytest.mark.filterwarnings("ignore:The default weight initialization:FutureWarning")
def test_round_trip_preserves_builder_flags_set_by_pretrained_weights(tmp_path):
    # torchvision turns on ``transform_input`` only when GoogLeNet is built from weights.
    original = TorchvisionBackbone("googlenet", weights=None, image_size=64)
    original.model.transform_input = True
    original.eval()
    original.save_pretrained(tmp_path)
    loaded = TorchvisionBackbone.from_pretrained(tmp_path)
    assert loaded.model.transform_input is True
    batch = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(loaded(batch), original(batch), atol=1e-6)


def test_from_pretrained_rejects_mismatched_embedding_dim(resnet, tmp_path):
    import json

    resnet.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config | {"embedding_dim": 99}))
    with pytest.raises(ConfigurationError, match="embedding_dim=99"):
        TorchvisionBackbone.from_pretrained(tmp_path)


def test_from_pretrained_rejects_foreign_directory(tmp_path):
    with pytest.raises(ConfigurationError, match="not a TorchvisionBackbone directory"):
        TorchvisionBackbone.from_pretrained(tmp_path)


def test_missing_torchvision_names_the_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torchvision":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BackboneProviderError, match=r"embedfit\[torchvision\]"):
        TorchvisionBackbone("resnet18", weights=None)


@pytest.mark.network
@pytest.mark.slow
def test_pretrained_resnet18_uses_checkpoint_preprocessing():
    model = EmbedFitModel.from_pretrained("torchvision/resnet18", labels=["legitimate", "risk"])
    assert model.backbone.weights == "DEFAULT"
    assert (model.backbone.image_size, model.backbone.resize_size) == (224, 256)
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (311, 457, 3), dtype=np.uint8)
    )
    import torchvision

    reference = torchvision.models.ResNet18_Weights.DEFAULT.transforms()(image)
    assert (model.backbone.preprocess(image) - reference).abs().max() < 0.05
    embeddings = model.encode([Image.new("RGB", (300, 200), "gray")] * 2)
    assert embeddings.ndim == 2 and embeddings.shape == (2, 512)
