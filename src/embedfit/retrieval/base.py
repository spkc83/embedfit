"""The retrieval result record and the backend contract nearest-neighbour search is built on."""

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from embedfit.exceptions import ConfigurationError


def validate_embeddings(
    embeddings: np.ndarray, *, name: str, expected_dim: int | None = None
) -> np.ndarray:
    """Check that ``embeddings`` is a finite ``[N, D]`` float matrix and return it as float32.

    Embeddings are float32 throughout EmbedFit, so float64 input is cast down. Integer input is
    rejected rather than cast, because an integer embedding is almost always a mistake.

    Args:
        embeddings: The array to validate.
        name: How the array is called in error messages, e.g. ``"embeddings"`` or ``"query"``.
        expected_dim: Required width ``D``, or ``None`` to accept any width.

    Returns:
        The same values as a float32 array, without copying when the input is already float32.

    Raises:
        ConfigurationError: If the rank, dtype, width, or values are invalid. The message
            reports shapes and dtypes only, never embedding values.
    """
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.floating):
        raise ConfigurationError(
            f"{name} must be a 2-D float array [N, D]; got shape {matrix.shape}, "
            f"dtype {matrix.dtype}."
        )
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise ConfigurationError(
            f"{name} has embedding dimension {matrix.shape[1]}, expected {expected_dim}."
        )
    if not np.isfinite(matrix).all():
        raise ConfigurationError(f"{name} contains NaN or infinite values.")
    return matrix.astype(np.float32, copy=False)


@dataclass(frozen=True, repr=False)
class RetrievalResult:
    """One neighbour returned by a search.

    Attributes:
        id: The identifier the item was indexed under; its position in the fitted embedding
            matrix when no ids were supplied.
        score: Similarity to the query, higher being more similar. Cosine similarity lies in
            [-1, 1] up to floating-point error.
        rank: 1-based position within the result list the item belongs to, counted after
            exclusions, so rank 1 is the best returned match rather than the best match overall.
        metadata: The mapping supplied for this item at fit time, as a read-only view, or
            ``None`` when no metadata was supplied for it.

    The repr states only whether metadata is present, because metadata values may be sensitive
    and results reach logs and tracebacks (``SECURITY_PRIVACY.md``). Read ``metadata`` directly
    to see its content.
    """

    id: Hashable
    score: float
    rank: int
    metadata: Mapping[str, Any] | None = None

    def __repr__(self) -> str:
        presence = "<present>" if self.metadata is not None else "None"
        return (
            f"RetrievalResult(id={self.id!r}, score={self.score!r}, rank={self.rank}, "
            f"metadata={presence})"
        )


class RetrievalBackend(ABC):
    """The vector-search primitive a ``RetrievalIndex`` is built on.

    A backend stores vectors and answers top-k queries over stored *positions* ``0..size-1``.
    Identifiers, metadata, groups, and every exclusion rule belong to ``RetrievalIndex``, so a
    future FAISS, pgvector, or Qdrant backend implements only the members below and no caller
    code changes (``docs/design/RETRIEVAL.md``).
    """

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of stored vectors; ``0`` before ``fit``."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int | None:
        """Width of the stored vectors, or ``None`` before ``fit``."""

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> None:
        """Replace the stored vectors with ``embeddings``.

        Args:
            embeddings: Finite float matrix ``[N, D]`` with ``N >= 1``.

        Raises:
            ConfigurationError: If the matrix is invalid for this backend's metric.
        """

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``k`` best stored positions for each query.

        Args:
            queries: Finite float matrix ``[Q, D]``; ``Q`` may be 0.
            k: Number of neighbours per query, at least 1.

        Returns:
            ``(positions, scores)``, both shaped ``[Q, min(k, size)]``: stored positions ordered
            by descending score, and their scores. Equal scores are ordered by ascending stored
            position, so the ordering is fully deterministic and the first ``m`` positions are
            always a prefix of the first ``m + 1``.

        Raises:
            IndexNotFittedError: If called before ``fit``.
            ConfigurationError: If ``queries`` or ``k`` is invalid.
        """
