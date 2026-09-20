"""Offline tests: models are built from configs, so no checkpoint is downloaded."""

import numpy as np
import pytest
import torch
from PIL import Image

transformers = pytest.importorskip("transformers")

from embedfit import EmbedFitModel  # noqa: E402
from embedfit.backbones import HFBackbone, ImageBackbone, load_backbone  # noqa: E402
from embedfit.backbones.hf_backbone import _processor_settings  # noqa: E402
from embedfit.exceptions import BackboneProviderError, ConfigurationError  # noqa: E402


def vit_config():
    return transformers.ViTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        image_size=32,
        patch_size=8,
    )


def resnet_config():
    return transformers.ResNetConfig(
        embedding_size=8, hidden_sizes=[8, 16], depths=[1, 1], layer_type="basic"
    )


def test_model_from_config_downloads_nothing_and_assumes_no_normalization():
    backbone = HFBackbone("local/tiny-vit", config=vit_config())
    assert isinstance(backbone, ImageBackbone) and backbone.identifier == "hf/local/tiny-vit"
    assert (backbone.image_size, backbone.resize_size, backbone.interpolation) == (
        32,
        32,
        "bilinear",
    )
    assert backbone.mean is None and backbone.std is None and backbone.embedding_dim == 32
    white = backbone.preprocess(Image.new("RGB", (50, 40), "white"))
    assert white.shape == (3, 32, 32) and torch.allclose(white, torch.ones_like(white))


def test_pooling_policies():
    batch = torch.rand(2, 3, 32, 32)
    outputs = {}
    for pooling in ("provider_default", "cls", "mean"):
        torch.manual_seed(1)
        backbone = HFBackbone("local/tiny-vit", config=vit_config(), pooling=pooling).eval()
        with torch.no_grad():
            outputs[pooling] = backbone(batch)
            raw = backbone.model(pixel_values=batch)
        assert backbone.pooling == pooling and outputs[pooling].shape == (2, 32)
    assert torch.allclose(outputs["provider_default"], raw.pooler_output)
    assert torch.allclose(outputs["cls"], raw.last_hidden_state[:, 0])
    assert torch.allclose(outputs["mean"], raw.last_hidden_state[:, 1:].mean(dim=1), atol=1e-6)


def test_feature_map_models_support_only_the_default_pooling():
    backbone = HFBackbone("local/tiny-resnet", config=resnet_config(), image_size=32)
    assert backbone.embedding_dim == 16 and backbone(torch.rand(2, 3, 32, 32)).shape == (2, 16)
    with pytest.raises(ConfigurationError, match="needs a token sequence"):
        HFBackbone("local/tiny-resnet", config=resnet_config(), image_size=32, pooling="mean")


def test_processor_settings_are_translated():
    processor = transformers.ViTImageProcessor(
        size={"height": 40, "width": 40},
        resample=3,
        image_mean=[0.1, 0.2, 0.3],
        image_std=[0.4, 0.5, 0.6],
    )
    settings = _processor_settings(processor)
    assert (settings["resize_size"], settings["image_size"]) == ((40, 40), 40)
    assert settings["interpolation"] == "bicubic"
    assert (list(settings["mean"]), list(settings["std"])) == ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    cropping = transformers.BitImageProcessor(
        size={"shortest_edge": 48},
        crop_size={"height": 32, "width": 32},
        do_center_crop=True,
        resample=2,
        do_normalize=False,
    )
    settings = _processor_settings(cropping)
    assert (settings["resize_size"], settings["image_size"]) == (48, 32)
    assert settings["mean"] is None and settings["interpolation"] == "bilinear"


def test_preprocess_matches_the_image_processor():
    processor = transformers.ViTImageProcessor(
        size={"height": 32, "width": 32},
        resample=2,
        image_mean=[0.4, 0.5, 0.6],
        image_std=[0.2, 0.25, 0.3],
    )
    backbone = HFBackbone("local/tiny-vit", config=vit_config(), **_processor_settings(processor))
    smooth = np.random.default_rng(0).random((8, 10, 3)).repeat(8, axis=0).repeat(8, axis=1)
    image = Image.fromarray((smooth * 255).astype(np.uint8))
    reference = torch.from_numpy(processor(image, return_tensors="np")["pixel_values"][0])
    difference = (backbone.preprocess(image) - reference).abs() * torch.tensor(backbone.std).view(
        3, 1, 1
    )
    assert difference.mean().item() < 0.01 and difference.max().item() < 0.15


def test_round_trip_is_pickle_free_and_exact(tmp_path):
    original = HFBackbone(
        "local/tiny-vit", config=vit_config(), pooling="mean", mean=[0.5] * 3, std=[0.5] * 3
    ).eval()
    original.save_pretrained(tmp_path)
    files = sorted(p.name for p in (tmp_path / "model").iterdir())
    assert "model.safetensors" in files and not [f for f in files if f.endswith((".bin", ".pt"))]
    loaded = load_backbone(tmp_path)
    assert isinstance(loaded, HFBackbone) and not loaded.training
    assert (loaded.identifier, loaded.pooling, loaded.mean) == (
        "hf/local/tiny-vit",
        "mean",
        (0.5, 0.5, 0.5),
    )
    batch = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(loaded(batch), original(batch), atol=1e-6)


def test_model_round_trip_and_training_flow(tmp_path):
    from embedfit import ImageDataset, Trainer, TrainingArguments
    from embedfit.backbones import registry
    from embedfit.testing import BAR_CLASSES, make_bar_images

    labels = list(BAR_CLASSES[:2])
    model = EmbedFitModel(
        HFBackbone("local/tiny-vit", config=vit_config(), pooling="cls"), labels=labels
    )
    train = ImageDataset.from_lists(*make_bar_images(4, num_classes=2, seed=0))
    args = TrainingArguments(
        training_mode="supcon", num_epochs=1, batch_size=8, learning_rate=1e-3, num_workers=0
    )
    Trainer(model, args, train).train()
    model.save_pretrained(tmp_path)
    loaded = EmbedFitModel.from_pretrained(tmp_path)
    assert loaded.backbone.pooling == "cls" and "hf" in registry.available_providers()
    assert np.allclose(
        loaded.predict_proba(train.images), model.predict_proba(train.images), atol=1e-5
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pooling": "max"}, "pooling must be one of"),
        ({"image_size": -1}, "image_size"),
        ({"interpolation": "lanczos"}, "interpolation"),
        ({"std": [0.5] * 3}, "given together"),
    ],
)
def test_invalid_options(kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        HFBackbone("local/tiny-vit", config=vit_config(), **kwargs)


def test_unloadable_checkpoint_is_a_provider_error(tmp_path):
    with pytest.raises(BackboneProviderError, match="Could not load Hugging Face checkpoint"):
        HFBackbone(str(tmp_path / "missing"))
    with pytest.raises(ConfigurationError, match="not an HFBackbone directory"):
        HFBackbone.from_pretrained(tmp_path)


@pytest.mark.network
@pytest.mark.slow
def test_pretrained_dino_checkpoint_loads_and_encodes():
    model = EmbedFitModel.from_pretrained("hf/facebook/dinov2-small")
    embeddings = model.encode([Image.new("RGB", (300, 200), "gray")])
    assert embeddings.shape == (1, 384) and model.backbone.mean is not None


def test_processor_resize_is_kept_unless_a_size_is_passed(monkeypatch):
    """A checkpoint that resizes to 48 and crops to 32 keeps both; image_size overrides both."""
    processor = transformers.BitImageProcessor(
        size={"shortest_edge": 48},
        crop_size={"height": 32, "width": 32},
        do_center_crop=True,
        resample=3,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *a, **k: transformers.AutoModel.from_config(vit_config()),
    )
    monkeypatch.setattr(
        transformers.AutoImageProcessor, "from_pretrained", lambda *a, **k: processor
    )
    backbone = HFBackbone("some/checkpoint")
    assert (backbone.resize_size, backbone.image_size, backbone.interpolation) == (
        48,
        32,
        "bicubic",
    )
    assert backbone.mean == (0.5, 0.5, 0.5)

    smooth = np.random.default_rng(0).random((8, 10, 3)).repeat(12, axis=0).repeat(12, axis=1)
    image = Image.fromarray((smooth * 255).astype(np.uint8))
    reference = torch.from_numpy(processor(image, return_tensors="np")["pixel_values"][0])
    difference = (backbone.preprocess(image) - reference).abs() * 0.5
    assert difference.mean().item() < 0.01 and difference.max().item() < 0.15

    with pytest.raises(ConfigurationError, match="no integer image_size"):
        HFBackbone("local/x", config=resnet_config())  # ResNet configs carry no image size.
