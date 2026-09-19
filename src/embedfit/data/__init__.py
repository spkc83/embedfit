"""Dataset adapters and image data utilities."""

from embedfit.data.dataset import ImageDataset, ImageSample
from embedfit.data.image import ImageLike, to_rgb_tensor
from embedfit.data.labels import LabelMapping
from embedfit.data.splitting import (
    SplitOverlapReport,
    find_split_overlap,
    grouped_split,
    stratified_split,
    temporal_split,
)

__all__ = [
    "ImageDataset",
    "ImageLike",
    "ImageSample",
    "LabelMapping",
    "SplitOverlapReport",
    "find_split_overlap",
    "grouped_split",
    "stratified_split",
    "temporal_split",
    "to_rgb_tensor",
]
