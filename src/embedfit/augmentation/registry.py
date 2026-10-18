"""Name-based construction of augmentation pipelines."""

from collections.abc import Callable
from typing import Any

from embedfit.augmentation.pipeline import Augmentation
from embedfit.augmentation.profiles import DocumentAugmentation, NaturalAugmentation
from embedfit.exceptions import ConfigurationError

AugmentationFactory = Callable[..., Augmentation]

NO_AUGMENTATION = "none"
"""Reserved name: training without task augmentation."""

_AUGMENTATIONS: dict[str, AugmentationFactory] = {
    "natural": NaturalAugmentation,
    "document": DocumentAugmentation,
}


def register_augmentation(
    name: str, factory: AugmentationFactory, *, overwrite: bool = False
) -> None:
    """Make a custom augmentation pipeline usable by name, e.g. in ``TrainingArguments``.

    Args:
        name: Non-empty name. ``"none"`` is reserved and always means no augmentation.
        factory: Callable returning an ``Augmentation``; receives keyword options. A tuned
            built-in is ``lambda: DocumentAugmentation(max_rotation=1.0)``.
        overwrite: Replace an existing registration instead of raising.

    Raises:
        ConfigurationError: If the name is empty, padded, reserved, or already registered, or
            the factory is not callable.
    """
    if not isinstance(name, str) or not name or name != name.strip():
        raise ConfigurationError(f"Invalid augmentation name {name!r}.")
    if not callable(factory):
        raise ConfigurationError(f"Augmentation '{name}' factory must be callable.")
    if name == NO_AUGMENTATION:
        raise ConfigurationError(
            f"'{NO_AUGMENTATION}' is reserved for training without augmentation."
        )
    if name in _AUGMENTATIONS and not overwrite:
        raise ConfigurationError(
            f"Augmentation '{name}' is already registered; pass overwrite=True to replace."
        )
    _AUGMENTATIONS[name] = factory


def available_augmentations() -> list[str]:
    """Return the usable augmentation names, ``"none"`` included, sorted."""
    return sorted([*_AUGMENTATIONS, NO_AUGMENTATION])


def create_augmentation(name: str, **kwargs: Any) -> Augmentation | None:
    """Build an augmentation pipeline by name.

    Returns:
        The pipeline, or ``None`` for ``"none"``.

    Raises:
        ConfigurationError: If the name is not registered, the factory rejects an option, raises
            ``TypeError``, or returns something that is not callable, or options are passed with
            ``"none"``.
    """
    if name == NO_AUGMENTATION:
        if kwargs:
            raise ConfigurationError(f"Augmentation '{NO_AUGMENTATION}' takes no options.")
        return None
    factory = _AUGMENTATIONS.get(name)
    if factory is None:
        raise ConfigurationError(
            f"Unknown augmentation '{name}'. Available augmentations: {available_augmentations()}."
        )
    try:
        pipeline = factory(**kwargs)
    except TypeError as exc:
        raise ConfigurationError(
            f"Augmentation '{name}' does not accept the options {sorted(kwargs)}, or raised "
            f"TypeError: {exc}"
        ) from exc
    if not callable(pipeline):
        raise ConfigurationError(
            f"Augmentation '{name}' must build a callable, got {type(pipeline).__name__}."
        )
    return pipeline
