"""Name-based construction of representation-learning losses."""

from collections.abc import Callable
from typing import Any

from torch import nn

from embedfit.exceptions import ConfigurationError
from embedfit.losses.cosine_pair import CosinePairLoss
from embedfit.losses.supcon import SupConLoss

LossFactory = Callable[..., nn.Module]

_LOSSES: dict[str, LossFactory] = {"supcon": SupConLoss, "cosine_pair": CosinePairLoss}


def register_loss(name: str, factory: LossFactory, *, overwrite: bool = False) -> None:
    """Make a custom loss constructible by name.

    Args:
        name: Non-empty name.
        factory: Callable returning an ``nn.Module`` that computes a scalar loss; receives
            keyword options.
        overwrite: Replace an existing registration instead of raising.

    Raises:
        ConfigurationError: If the name is empty or already registered.
    """
    if not name or name != name.strip():
        raise ConfigurationError(f"Invalid loss name {name!r}.")
    if name in _LOSSES and not overwrite:
        raise ConfigurationError(
            f"Loss '{name}' is already registered; pass overwrite=True to replace."
        )
    _LOSSES[name] = factory


def available_losses() -> list[str]:
    """Return the registered loss names, sorted."""
    return sorted(_LOSSES)


def create_loss(name: str, **kwargs: Any) -> nn.Module:
    """Build a loss by name.

    Raises:
        ConfigurationError: If the name is not registered or the loss rejects an option.
    """
    factory = _LOSSES.get(name)
    if factory is None:
        raise ConfigurationError(f"Unknown loss '{name}'. Available losses: {available_losses()}.")
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise ConfigurationError(
            f"Loss '{name}' does not accept the options {sorted(kwargs)}."
        ) from exc
