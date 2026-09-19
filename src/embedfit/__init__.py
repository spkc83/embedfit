"""EmbedFit: few-shot adaptation of pretrained embedding models."""

import logging

from embedfit._version import __version__
from embedfit.benchmark import FewShotEvaluator
from embedfit.data.dataset import ImageDataset
from embedfit.exceptions import EmbedFitError
from embedfit.model import EmbedFitModel
from embedfit.retrieval.index import RetrievalIndex
from embedfit.trainer import Trainer
from embedfit.training_args import TrainingArguments

# Library code logs through the standard ``logging`` module and never configures handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "EmbedFitError",
    "EmbedFitModel",
    "FewShotEvaluator",
    "ImageDataset",
    "RetrievalIndex",
    "Trainer",
    "TrainingArguments",
    "__version__",
]
