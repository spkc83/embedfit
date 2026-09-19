"""Representation-learning losses."""

from embedfit.losses.registry import available_losses, create_loss, register_loss
from embedfit.losses.supcon import SupConLoss

__all__ = ["SupConLoss", "available_losses", "create_loss", "register_loss"]
