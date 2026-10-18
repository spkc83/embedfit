"""The augmentation-pipeline registry (SPEC §26)."""

import pytest

from embedfit.augmentation import (
    DocumentAugmentation,
    NaturalAugmentation,
    available_augmentations,
    create_augmentation,
    register_augmentation,
    registry,
)
from embedfit.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_AUGMENTATIONS", dict(registry._AUGMENTATIONS))


def identity(image):
    return image


def test_builtin_profiles_are_registered():
    assert available_augmentations() == ["document", "natural", "none"]
    assert isinstance(create_augmentation("natural"), NaturalAugmentation)
    assert isinstance(create_augmentation("document"), DocumentAugmentation)


def test_none_means_no_pipeline():
    assert create_augmentation("none") is None
    with pytest.raises(ConfigurationError, match="no options"):
        create_augmentation("none", strength=1)


def test_options_are_forwarded_and_unknown_options_rejected():
    assert create_augmentation("document", max_rotation=1.0).max_rotation == 1.0
    with pytest.raises(ConfigurationError, match="does not accept the options"):
        create_augmentation("natural", hue=0.1)


def test_a_custom_pipeline_can_be_registered_and_replaced():
    register_augmentation("scans", lambda: identity)
    assert "scans" in available_augmentations()
    assert create_augmentation("scans") is identity
    with pytest.raises(ConfigurationError, match="already registered"):
        register_augmentation("scans", lambda: identity)
    register_augmentation("scans", lambda: DocumentAugmentation(max_rotation=1.0), overwrite=True)
    assert isinstance(create_augmentation("scans"), DocumentAugmentation)


def test_built_in_profiles_can_be_overwritten_deliberately():
    register_augmentation("natural", lambda: identity, overwrite=True)
    assert create_augmentation("natural") is identity


@pytest.mark.parametrize("overwrite", [False, True])
def test_none_is_reserved(overwrite):
    with pytest.raises(ConfigurationError, match="reserved"):
        register_augmentation("none", lambda: identity, overwrite=overwrite)


@pytest.mark.parametrize("name", ["", " scans", "scans ", 5, None])
def test_invalid_names_are_rejected(name):
    with pytest.raises(ConfigurationError, match="Invalid augmentation name"):
        register_augmentation(name, lambda: identity)


def test_unknown_names_list_the_available_ones():
    with pytest.raises(ConfigurationError, match=r"Available augmentations: \['document'"):
        create_augmentation("strong")


def test_a_factory_must_return_a_callable():
    register_augmentation("broken", lambda: 3)
    with pytest.raises(ConfigurationError, match="callable"):
        create_augmentation("broken")


def test_a_factory_must_be_callable():
    with pytest.raises(ConfigurationError, match="factory must be callable"):
        register_augmentation("scans", "not a factory")
    assert "scans" not in available_augmentations()


def test_a_type_error_inside_a_factory_is_not_reported_as_an_unknown_option():
    def factory():
        raise TypeError("bad internal call")

    register_augmentation("faulty", factory)
    with pytest.raises(ConfigurationError, match="or raised TypeError: bad internal call"):
        create_augmentation("faulty")
