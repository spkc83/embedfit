"""Name-based construction of classification heads."""

import json
from collections.abc import Callable
from pathlib import Path
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


def load_head(path: str | Path) -> ClassificationHead:
    """Rebuild a head from a directory written by ``ClassificationHead.save``.

    The head type is read from the ``"type"`` key of the directory's ``config.json`` and must
    be a registered name whose factory is a ``ClassificationHead`` subclass.

    Raises:
        ConfigurationError: If ``config.json`` is missing or malformed, or the type is not a
            registered head class.
    """
    config_path = Path(path) / "config.json"
    try:
        name = json.loads(config_path.read_text()).get("type")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise ConfigurationError(
            f"'{config_path}' is missing or is not a JSON object with a 'type' key."
        ) from exc
    factory = _HEADS.get(name) if isinstance(name, str) else None
    if not (isinstance(factory, type) and issubclass(factory, ClassificationHead)):
        raise ConfigurationError(
            f"'{config_path}' names head type {name!r}, which is not a registered head class. "
            f"Available heads: {available_heads()}."
        )
    return factory.load(path)


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
