import pytest
import torch
from PIL import Image

from embedfit.backbones import DummyBackbone, ImageBackbone
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


def test_preprocess_resizes_to_image_size(dummy_backbone):
    pixels = dummy_backbone.preprocess(Image.new("RGB", (48, 20)))
    assert pixels.shape == (3, 32, 32)
    assert pixels.dtype == torch.float32


def test_is_an_image_backbone_with_identifier(dummy_backbone):
    assert isinstance(dummy_backbone, ImageBackbone)
    assert dummy_backbone.identifier == "dummy/tiny"


def test_forward_features_rejects_wrong_shape(dummy_backbone):
    with pytest.raises(ImageInputError, match="pixel_values"):
        dummy_backbone.forward_features(torch.zeros(3, 32, 32))


def test_freeze_and_unfreeze(dummy_backbone):
    dummy_backbone.train()
    dummy_backbone.freeze()
    assert dummy_backbone.is_frozen
    assert not any(p.requires_grad for p in dummy_backbone.parameters())
    assert not dummy_backbone.training
    dummy_backbone.train()  # A frozen backbone ignores requests to enter training mode.
    assert not dummy_backbone.training
    assert not any(m.training for m in dummy_backbone.modules())
    dummy_backbone.unfreeze()
    assert not dummy_backbone.is_frozen
    assert all(p.requires_grad for p in dummy_backbone.parameters())
    dummy_backbone.train()
    assert dummy_backbone.training


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
