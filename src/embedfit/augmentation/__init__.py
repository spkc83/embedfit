"""Task augmentation for representation training, kept apart from provider preprocessing."""

from embedfit.augmentation.pipeline import Augmentation
from embedfit.augmentation.profiles import DocumentAugmentation, NaturalAugmentation
from embedfit.augmentation.registry import (
    available_augmentations,
    create_augmentation,
    register_augmentation,
)

__all__ = [
    "Augmentation",
    "DocumentAugmentation",
    "NaturalAugmentation",
    "available_augmentations",
    "create_augmentation",
    "register_augmentation",
]
