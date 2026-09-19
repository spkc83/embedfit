"""Backbone provider adapters."""

from embedfit.backbones.base import ImageBackbone
from embedfit.backbones.dummy import DummyBackbone
from embedfit.backbones.registry import (
    available_providers,
    create_backbone,
    load_backbone,
    register_backbone_provider,
)
from embedfit.backbones.torchvision_backbone import TorchvisionBackbone

__all__ = [
    "DummyBackbone",
    "ImageBackbone",
    "TorchvisionBackbone",
    "available_providers",
    "create_backbone",
    "load_backbone",
    "register_backbone_provider",
]
