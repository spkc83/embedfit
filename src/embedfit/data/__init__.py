"""Dataset adapters and image data utilities."""

from embedfit.data.dataset import ImageDataset, ImageSample
from embedfit.data.image import ImageLike, to_rgb_tensor
from embedfit.data.labels import LabelMapping

__all__ = ["ImageDataset", "ImageLike", "ImageSample", "LabelMapping", "to_rgb_tensor"]
