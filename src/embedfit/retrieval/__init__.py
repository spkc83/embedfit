"""Embedding retrieval interfaces and backends."""

from embedfit.exceptions import IndexNotFittedError
from embedfit.retrieval.base import RetrievalBackend, RetrievalResult
from embedfit.retrieval.brute_force import BruteForceCosineBackend
from embedfit.retrieval.index import RetrievalIndex
from embedfit.retrieval.metrics import (
    mean_average_precision,
    mean_reciprocal_rank,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)

__all__ = [
    "BruteForceCosineBackend",
    "IndexNotFittedError",
    "RetrievalBackend",
    "RetrievalIndex",
    "RetrievalResult",
    "mean_average_precision",
    "mean_reciprocal_rank",
    "retrieval_precision_at_k",
    "retrieval_recall_at_k",
]
