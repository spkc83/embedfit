"""Lightweight classifiers operating on embeddings."""

from embedfit.exceptions import HeadNotFittedError
from embedfit.heads.base import ClassificationHead
from embedfit.heads.linear import LinearHead
from embedfit.heads.logistic_regression import LogisticRegressionHead
from embedfit.heads.registry import available_heads, create_head, load_head, register_head

__all__ = [
    "ClassificationHead",
    "HeadNotFittedError",
    "LinearHead",
    "LogisticRegressionHead",
    "available_heads",
    "create_head",
    "load_head",
    "register_head",
]
