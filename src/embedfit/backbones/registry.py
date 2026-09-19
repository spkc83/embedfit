"""Resolution of ``<provider>/<name>`` identifiers to backbone instances.

The provider is always explicit. A bare model name is rejected rather than searched for across
providers, because the same name can mean different checkpoints with different preprocessing.
"""

from collections.abc import Callable
from typing import Any

from embedfit.backbones.base import ImageBackbone
from embedfit.exceptions import BackboneProviderError

BackboneFactory = Callable[..., ImageBackbone]
"""``factory(name, **kwargs) -> ImageBackbone`` where ``name`` is the identifier without prefix."""

_PROVIDERS: dict[str, BackboneFactory] = {}


def register_backbone_provider(
    prefix: str, factory: BackboneFactory, *, overwrite: bool = False
) -> None:
    """Make ``"<prefix>/<name>"`` identifiers resolvable.

    Args:
        prefix: Provider prefix: non-empty, without ``/``.
        factory: Callable invoked as ``factory(name, **kwargs)`` that returns an
            ``ImageBackbone``. Import heavy optional dependencies inside it, not at module level.
        overwrite: Replace an existing registration instead of raising.

    Raises:
        BackboneProviderError: If the prefix is invalid or already registered.
    """
    if not prefix or "/" in prefix or prefix != prefix.strip():
        raise BackboneProviderError(
            f"Invalid provider prefix {prefix!r}: must be non-empty with no '/' or padding."
        )
    if prefix in _PROVIDERS and not overwrite:
        raise BackboneProviderError(
            f"Backbone provider '{prefix}' is already registered; pass overwrite=True to replace."
        )
    _PROVIDERS[prefix] = factory


def available_providers() -> list[str]:
    """Return the registered provider prefixes, sorted."""
    return sorted(_PROVIDERS)


def create_backbone(identifier: str, **kwargs: Any) -> ImageBackbone:
    """Build a backbone from a provider-qualified identifier.

    Args:
        identifier: ``"<provider>/<name>"``, e.g. ``"torchvision/resnet18"``. Only the first
            ``/`` separates the provider, so names may themselves contain slashes.
        **kwargs: Forwarded to the provider's factory (see the adapter class for options).

    Returns:
        The constructed backbone.

    Raises:
        BackboneProviderError: If the identifier has no provider prefix or name, or the
            provider is not registered. The provider is never guessed.
    """
    if not isinstance(identifier, str):
        raise BackboneProviderError(
            f"Backbone identifier must be a str, got {type(identifier).__name__}."
        )
    prefix, separator, name = identifier.partition("/")
    providers = ", ".join(f"'{p}/<name>'" for p in available_providers())
    if not separator or not prefix or not name:
        raise BackboneProviderError(
            f"Backbone identifier '{identifier}' must have the form '<provider>/<name>'. "
            f"The provider is never guessed. Use one of: {providers}."
        )
    factory = _PROVIDERS.get(prefix)
    if factory is None:
        raise BackboneProviderError(
            f"Unknown backbone provider '{prefix}' in '{identifier}'. Use one of: {providers}, "
            "or add one with register_backbone_provider()."
        )
    return factory(name, **kwargs)


def _create_torchvision(name: str, **kwargs: Any) -> ImageBackbone:
    from embedfit.backbones.torchvision_backbone import TorchvisionBackbone

    return TorchvisionBackbone(name, **kwargs)


def _create_dummy(name: str, **kwargs: Any) -> ImageBackbone:
    from embedfit.backbones.dummy import DummyBackbone

    if name != "tiny":
        raise BackboneProviderError(
            f"Unknown dummy backbone '{name}'; the only one is 'dummy/tiny'."
        )
    return DummyBackbone(**kwargs)


register_backbone_provider("torchvision", _create_torchvision)
register_backbone_provider("dummy", _create_dummy)
