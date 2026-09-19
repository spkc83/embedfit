import pytest

from embedfit.backbones import (
    DummyBackbone,
    available_providers,
    create_backbone,
    register_backbone_provider,
    registry,
)
from embedfit.exceptions import BackboneProviderError


@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_PROVIDERS", dict(registry._PROVIDERS))


def test_builtin_providers_are_registered():
    assert {"dummy", "torchvision"} <= set(available_providers())


def test_create_dummy_forwards_kwargs():
    backbone = create_backbone("dummy/tiny", embedding_dim=8)
    assert isinstance(backbone, DummyBackbone)
    assert backbone.embedding_dim == 8


def test_unknown_provider_error_is_actionable():
    with pytest.raises(BackboneProviderError) as excinfo:
        create_backbone("foo/resnet50")
    message = str(excinfo.value)
    assert "Unknown backbone provider 'foo'" in message
    assert "'torchvision/<name>'" in message
    assert "register_backbone_provider" in message


@pytest.mark.parametrize("identifier", ["resnet50", "/resnet50", "torchvision/", ""])
def test_provider_is_never_guessed(identifier):
    with pytest.raises(BackboneProviderError, match="never guessed"):
        create_backbone(identifier)


def test_non_string_identifier():
    with pytest.raises(BackboneProviderError, match="must be a str"):
        create_backbone(None)


def test_unknown_dummy_name():
    with pytest.raises(BackboneProviderError, match="dummy/tiny"):
        create_backbone("dummy/huge")


def test_custom_provider_receives_full_name_after_first_slash(isolated_registry):
    received = {}

    def factory(name, **kwargs):
        received.update(name=name, kwargs=kwargs)
        return DummyBackbone()

    register_backbone_provider("custom", factory)
    create_backbone("custom/org/model-v2", flag=True)
    assert received == {"name": "org/model-v2", "kwargs": {"flag": True}}
    assert "custom" in available_providers()


def test_duplicate_registration_requires_overwrite(isolated_registry):
    with pytest.raises(BackboneProviderError, match="already registered"):
        register_backbone_provider("dummy", lambda name: DummyBackbone())
    register_backbone_provider("dummy", lambda name: DummyBackbone(embedding_dim=4), overwrite=True)
    assert create_backbone("dummy/anything").embedding_dim == 4


@pytest.mark.parametrize("prefix", ["", "a/b", " padded "])
def test_invalid_prefix(isolated_registry, prefix):
    with pytest.raises(BackboneProviderError, match="Invalid provider prefix"):
        register_backbone_provider(prefix, lambda name: DummyBackbone())
