"""Name-based construction of classification heads."""

from collections.abc import Callable
from typing import Any

from embedfit.exceptions import ConfigurationError
from embedfit.heads.base import ClassificationHead
from embedfit.heads.logistic_regression import LogisticRegressionHead

HeadFactory = Callable[..., ClassificationHead]

_HEADS: dict[str, HeadFactory] = {"logistic_regression": LogisticRegressionHead}


def register_head(name: str, factory: HeadFactory, *, overwrite: bool = False) -> None:
    """Make a custom head constructible by name.

    Args:
        name: Non-empty name used in ``TrainingArguments.head`` and ``create_head``.
        factory: Callable returning a ``ClassificationHead``; receives keyword options.
        overwrite: Replace an existing registration instead of raising.

    Raises:
        ConfigurationError: If the name is empty or already registered.
    """
    if not name or name != name.strip():
        raise ConfigurationError(f"Invalid head name {name!r}.")
    if name in _HEADS and not overwrite:
        raise ConfigurationError(
            f"Head '{name}' is already registered; pass overwrite=True to replace."
        )
    _HEADS[name] = factory


def available_heads() -> list[str]:
    """Return the registered head names, sorted."""
    return sorted(_HEADS)


def create_head(name: str, **kwargs: Any) -> ClassificationHead:
    """Build a head by name.

    Raises:
        ConfigurationError: If the name is not registered or the head rejects an option.
    """
    factory = _HEADS.get(name)
    if factory is None:
        raise ConfigurationError(f"Unknown head '{name}'. Available heads: {available_heads()}.")
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise ConfigurationError(
            f"Head '{name}' does not accept the options {sorted(kwargs)}."
        ) from exc
