"""Representation-learning losses."""

from embedfit.losses.cosine_pair import CosinePairLoss
from embedfit.losses.registry import available_losses, create_loss, register_loss
from embedfit.losses.supcon import SupConLoss

__all__ = ["CosinePairLoss", "SupConLoss", "available_losses", "create_loss", "register_loss"]
