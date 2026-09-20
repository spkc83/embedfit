"""Offline tests: every model is built with ``pretrained=False``, so nothing is downloaded."""

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("timm")

from embedfit import EmbedFitModel  # noqa: E402
from embedfit.backbones import ImageBackbone, TimmBackbone, load_backbone  # noqa: E402
from embedfit.exceptions import BackboneProviderError, ConfigurationError  # noqa: E402

VIT = "vit_tiny_patch16_224"


def test_convolutional_model_uses_its_data_config():
    import timm

    backbone = TimmBackbone("resnet10t", pretrained=False)
    data = timm.data.resolve_data_config({}, model=backbone.model)
    assert isinstance(backbone, ImageBackbone) and backbone.identifier == "timm/resnet10t"
    assert backbone.pooling == "provider_default" and backbone.embedding_dim == 512
    assert backbone.image_size == data["input_size"][-1] == 176
    assert backbone.resize_size == round(176 / data["crop_pct"])
    assert (backbone.interpolation, backbone.mean, backbone.std) == (
        data["interpolation"],
        tuple(data["mean"]),
        tuple(data["std"]),
    )
    assert backbone.preprocess(Image.new("RGB", (300, 200))).shape == (3, 176, 176)


def test_preprocess_matches_timm_own_transform():
    import timm

    backbone = TimmBackbone("resnet10t", pretrained=False)
    data = timm.data.resolve_data_config({}, model=backbone.model)
    reference = timm.data.create_transform(**data, is_training=False)
    smooth = np.random.default_rng(0).random((20, 30, 3)).repeat(12, axis=0).repeat(12, axis=1)
    image = Image.fromarray((smooth * 255).astype(np.uint8))
    difference = (backbone.preprocess(image) - reference(image)).abs()
    difference = difference * torch.tensor(backbone.std).view(3, 1, 1)  # Back to [0, 1] units.
    # Float resizing differs slightly from timm's 8-bit PIL resizing, most at hard edges.
    # A one-pixel misalignment would push the maximum above 0.5 on this blocky image.
    assert difference.mean().item() < 0.01 and difference.max().item() < 0.15


def test_image_size_is_configurable_for_cnn_and_vit():
    cnn = TimmBackbone("resnet10t", pretrained=False, image_size=64)
    assert cnn(cnn.preprocess(Image.new("L", (90, 70)))[None]).shape == (1, 512)
    vit = TimmBackbone(VIT, pretrained=False, image_size=32)
    assert vit.image_size == 32 and vit(torch.rand(2, 3, 32, 32)).shape == (2, 192)


def test_pooling_policies_on_a_transformer():
    torch.manual_seed(0)
    batch = torch.rand(2, 3, 32, 32)
    outputs = {}
    for pooling in ("provider_default", "cls", "mean"):
        torch.manual_seed(1)  # Same weights for every policy.
        backbone = TimmBackbone(VIT, pretrained=False, image_size=32, pooling=pooling).eval()
        assert backbone.pooling == pooling and backbone.embedding_dim == 192
        with torch.no_grad():
            outputs[pooling] = backbone(batch)
            tokens = backbone.model.forward_features(batch)
    assert torch.allclose(outputs["cls"], tokens[:, 0])
    assert torch.allclose(outputs["mean"], tokens[:, 1:].mean(dim=1), atol=1e-6)
    assert not torch.allclose(outputs["cls"], outputs["mean"])


def test_token_pooling_is_rejected_for_models_without_tokens():
    for pooling in ("cls", "mean"):
        with pytest.raises(ConfigurationError, match="needs a token sequence"):
            TimmBackbone("resnet10t", pretrained=False, pooling=pooling)


def test_round_trip_preserves_weights_pooling_and_preprocessing(tmp_path):
    original = TimmBackbone(
        VIT,
        pretrained=False,
        image_size=32,
        pooling="mean",
        interpolation="bilinear",
        mean=[0.1, 0.2, 0.3],
        std=[0.4, 0.5, 0.6],
    )
    original.eval()
    original.save_pretrained(tmp_path)
    loaded = load_backbone(tmp_path)
    assert isinstance(loaded, TimmBackbone) and not loaded.training
    assert (loaded.identifier, loaded.pooling, loaded.image_size, loaded.interpolation) == (
        "timm/" + VIT,
        "mean",
        32,
        "bilinear",
    )
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (50, 40, 3), dtype=np.uint8))
    pixels = original.preprocess(image)
    assert torch.equal(loaded.preprocess(image), pixels)
    with torch.no_grad():
        assert torch.allclose(loaded(pixels[None]), original(pixels[None]), atol=1e-6)


def test_model_round_trip_records_the_pooling(tmp_path):
    model = EmbedFitModel.from_pretrained(
        "timm/" + VIT, pretrained=False, image_size=32, pooling="cls", labels=["a", "b"]
    )
    images = [Image.new("RGB", (40, 40), color) for color in ("red", "blue")]
    model.save_pretrained(tmp_path)
    loaded = EmbedFitModel.from_pretrained(tmp_path)
    assert loaded.backbone.pooling == "cls"
    assert np.allclose(loaded.encode(images), model.encode(images), atol=1e-6)


def test_frozen_training_flow():
    from embedfit import ImageDataset, Trainer, TrainingArguments
    from embedfit.testing import BAR_CLASSES, make_bar_images

    model = EmbedFitModel.from_pretrained(
        "timm/resnet10t", pretrained=False, image_size=32, labels=list(BAR_CLASSES[:2])
    )
    train = ImageDataset.from_lists(*make_bar_images(4, num_classes=2, seed=0))
    Trainer(model, TrainingArguments(training_mode="frozen"), train).train()
    assert model.predict_proba(train.images).shape == (8, 2)
    assert not any(p.requires_grad for p in model.backbone.parameters())


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"name": "resnet10tt"}, BackboneProviderError, "Unknown timm model"),
        ({"pooling": "max"}, ConfigurationError, "pooling must be one of"),
        ({"image_size": 0}, ConfigurationError, "image_size"),
        ({"interpolation": "lanczos"}, ConfigurationError, "interpolation"),
        ({"mean": [0.5, 0.5, 0.5]}, ConfigurationError, "given together"),
    ],
)
def test_invalid_options(kwargs, error, match):
    options = {"name": "resnet10t", "pretrained": False, **kwargs}
    with pytest.raises(error, match=match):
        TimmBackbone(options.pop("name"), **options)


def test_corrupt_directories_are_configuration_errors(tmp_path):
    with pytest.raises(ConfigurationError, match="not a TimmBackbone directory"):
        TimmBackbone.from_pretrained(tmp_path)
    backbone = TimmBackbone("resnet10t", pretrained=False, image_size=32)
    backbone.save_pretrained(tmp_path)
    (tmp_path / "weights.pt").write_bytes(b"garbage")
    with pytest.raises(ConfigurationError, match="not a tensor-only state dict"):
        TimmBackbone.from_pretrained(tmp_path)


def test_missing_package_names_the_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        lambda name, *a, **k: (_ for _ in ()).throw(ImportError("simulated"))
        if name == "timm"
        else real_import(name, *a, **k),
    )
    with pytest.raises(BackboneProviderError, match=r"embedfit\[timm\]"):
        TimmBackbone("resnet10t", pretrained=False)


@pytest.mark.network
@pytest.mark.slow
def test_pretrained_checkpoint_loads_and_encodes():
    model = EmbedFitModel.from_pretrained("timm/resnet10t.c3_in1k")
    assert model.encode([Image.new("RGB", (300, 200), "gray")]).shape == (1, 512)
