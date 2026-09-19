import logging

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from embedfit import EmbedFitModel
from embedfit.backbones import DummyBackbone
from embedfit.exceptions import BackboneProviderError, ConfigurationError, ImageInputError


@pytest.fixture
def model(dummy_backbone) -> EmbedFitModel:
    return EmbedFitModel(dummy_backbone)


def test_encode_returns_one_row_per_image(model, bar_dataset):
    images, _ = bar_dataset
    embeddings = model.encode(images)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(images), model.embedding_dim)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()


def test_single_image_gives_one_row(model, bar_dataset):
    image = bar_dataset[0][0]
    single = model.encode(image)
    assert single.shape == (1, model.embedding_dim)
    assert np.allclose(single, model.encode([image]))


def test_empty_list_gives_zero_rows(model):
    assert model.encode([]).shape == (0, model.embedding_dim)


def test_embeddings_are_unit_norm_by_default(model, bar_dataset):
    norms = np.linalg.norm(model.encode(bar_dataset[0]), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_normalize_argument_overrides_model_policy(dummy_backbone, bar_dataset):
    images = bar_dataset[0][:4]
    raw_model = EmbedFitModel(dummy_backbone, normalize_embeddings=False)
    raw = raw_model.encode(images)
    assert not np.allclose(np.linalg.norm(raw, axis=1), 1.0)
    expected = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    assert np.allclose(raw_model.encode(images, normalize=True), expected, atol=1e-6)
    assert np.allclose(EmbedFitModel(dummy_backbone).encode(images, normalize=False), raw)


def test_batch_size_does_not_change_results_or_order(model, bar_dataset):
    images, _ = bar_dataset
    full = model.encode(images, batch_size=64)
    assert np.allclose(model.encode(images, batch_size=1), full, atol=1e-6)
    assert np.allclose(model.encode(images, batch_size=5), full, atol=1e-6)
    assert np.allclose(model.encode(images[::-1]), full[::-1], atol=1e-6)


def test_rows_follow_input_order(model, bar_dataset):
    images = bar_dataset[0][::4]  # Visually distinct images from every class.
    one_at_a_time = np.stack([model.encode(image)[0] for image in images])
    assert np.allclose(model.encode(images, batch_size=4), one_at_a_time, atol=1e-6)
    assert not np.allclose(one_at_a_time[0], one_at_a_time[-1], atol=1e-3)


def test_encode_matches_backbone_dtype_and_returns_float32(bar_dataset):
    model = EmbedFitModel(DummyBackbone(embedding_dim=8).to(torch.bfloat16))
    embeddings = model.encode(bar_dataset[0][:2])
    assert embeddings.dtype == np.float32 and np.isfinite(embeddings).all()


def test_tensor_output(model, bar_dataset):
    embeddings = model.encode(bar_dataset[0][:3], convert_to_numpy=False)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.device.type == "cpu" and not embeddings.requires_grad


def test_mixed_input_forms_agree(model, tmp_path):
    pixels = (np.arange(32 * 32 * 3).reshape(32, 32, 3) % 256).astype(np.uint8)
    pil = Image.fromarray(pixels)
    pil.save(tmp_path / "image.png")
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    rows = model.encode([pil, pixels, tensor, tmp_path / "image.png"])
    assert np.allclose(rows, rows[0], atol=1e-6)


def test_grayscale_equals_channel_replicated_rgb(model):
    gray = (np.arange(32 * 32).reshape(32, 32) % 256).astype(np.uint8)
    rows = model.encode([gray, np.stack([gray] * 3, axis=-1), Image.fromarray(gray)])
    assert np.allclose(rows, rows[0], atol=1e-6)


def test_encode_runs_in_eval_mode_and_restores_training_mode(bar_dataset):
    class ModeProbe(DummyBackbone):
        def forward_features(self, pixel_values):
            self.seen_training = self.training
            return super().forward_features(pixel_values)

    backbone = ModeProbe(embedding_dim=8)
    model = EmbedFitModel(backbone)
    model.train()
    model.encode(bar_dataset[0][:2])
    assert backbone.seen_training is False
    assert backbone.training is True


def test_training_mode_restored_after_error(model):
    model.train()
    with pytest.raises(ImageInputError):
        model.encode([object()])
    assert model.backbone.training


def test_malformed_image_error_names_index_not_content(model, bar_dataset):
    bad = np.full((8, 8, 2), 173, dtype=np.uint8)
    with pytest.raises(ImageInputError, match="index 2") as excinfo:
        model.encode([*bar_dataset[0][:2], bad], batch_size=2)
    assert "173" not in str(excinfo.value)


def test_batched_array_is_rejected_with_guidance(model):
    with pytest.raises(ImageInputError, match="list of images"):
        model.encode(np.zeros((4, 32, 32, 3), dtype=np.uint8))


@pytest.mark.parametrize("batch_size", [0, -1, 2.5, True])
def test_invalid_batch_size(model, batch_size):
    with pytest.raises(ConfigurationError, match="batch_size"):
        model.encode([], batch_size=batch_size)


def test_freeze_and_unfreeze_backbone(model, bar_dataset):
    model.freeze_backbone()
    assert not any(p.requires_grad for p in model.parameters())
    model.train()
    assert not model.backbone.training  # Frozen backbones stay in eval mode.
    pixels = torch.stack([model.backbone.preprocess(i) for i in bar_dataset[0][:2]])
    assert not model(pixels).requires_grad

    model.unfreeze_backbone()
    model.train()
    assert model.backbone.training
    model(pixels).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_frozen_batchnorm_statistics_do_not_move():
    class BatchNormBackbone(DummyBackbone):
        def __init__(self):
            super().__init__(embedding_dim=8)
            self.norm = nn.BatchNorm2d(3)

        def forward_features(self, pixel_values):
            return super().forward_features(self.norm(pixel_values))

    model = EmbedFitModel(BatchNormBackbone())
    model.freeze_backbone()
    model.train()
    model(torch.rand(4, 3, 32, 32) * 5)
    assert torch.equal(model.backbone.norm.running_mean, torch.zeros(3))


def test_forward_is_differentiable_and_follows_normalization_policy(dummy_backbone):
    pixels = torch.rand(3, 3, 32, 32)
    normalized = EmbedFitModel(dummy_backbone)(pixels)
    assert normalized.requires_grad
    assert torch.allclose(normalized.norm(dim=1), torch.ones(3), atol=1e-6)
    raw = EmbedFitModel(dummy_backbone, normalize_embeddings=False)(pixels)
    assert torch.allclose(raw, dummy_backbone.forward_features(pixels))


def test_from_pretrained_builds_eval_model_on_cpu():
    model = EmbedFitModel.from_pretrained(
        "dummy/tiny", labels=["legitimate", "risk"], positive_label="risk", embedding_dim=8
    )
    assert model.embedding_dim == 8
    assert model.labels == ("legitimate", "risk")
    assert model.positive_label == "risk"
    assert model.device == torch.device("cpu")
    assert not model.training


def test_from_pretrained_invalid_prefix_is_actionable():
    with pytest.raises(BackboneProviderError, match="Unknown backbone provider 'foo'"):
        EmbedFitModel.from_pretrained("foo/resnet50")


def test_from_pretrained_rejects_unavailable_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigurationError, match="not available"):
        EmbedFitModel.from_pretrained("dummy/tiny", device="cuda")


def test_label_order_is_preserved_not_sorted(dummy_backbone):
    assert EmbedFitModel(dummy_backbone, labels=["b", "a", "c"]).labels == ("b", "a", "c")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"labels": ["only"]}, "at least 2"),
        ({"labels": ["a", "a"]}, "unique"),
        ({"labels": "ab"}, "sequence"),
        ({"labels": ["a", "b"], "positive_label": "c"}, "one of labels"),
        ({"labels": [0, 1], "positive_label": False}, "same type"),
        ({"labels": [["a"], ["b"]]}, "hashable"),
        ({"labels": np.array([[0, 1]])}, "sequence"),
        ({"positive_label": "a"}, "requires labels"),
    ],
)
def test_invalid_labels(dummy_backbone, kwargs, match):
    with pytest.raises(ConfigurationError, match=match):
        EmbedFitModel(dummy_backbone, **kwargs)


def test_numpy_label_array_is_accepted(dummy_backbone):
    model = EmbedFitModel(dummy_backbone, labels=np.array(["b", "a"]), positive_label="a")
    assert model.labels == ("b", "a")


def test_backbone_must_be_an_image_backbone():
    with pytest.raises(ConfigurationError, match="ImageBackbone"):
        EmbedFitModel(nn.Linear(2, 2))


def test_logs_contain_no_pixel_or_label_content(dummy_backbone, caplog, logged_messages):
    caplog.set_level(logging.DEBUG, logger="embedfit")
    model = EmbedFitModel.from_pretrained("dummy/tiny", labels=["secret-a", "secret-b"])
    model.encode([np.full((32, 32, 3), 173, dtype=np.uint8)])
    assert caplog.records
    assert "173" not in logged_messages() and "secret" not in logged_messages()
