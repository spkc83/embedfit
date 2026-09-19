"""Exact cosine search by matrix product over L2-normalised vectors."""

import numpy as np

from embedfit.exceptions import ConfigurationError, IndexNotFittedError
from embedfit.retrieval.base import RetrievalBackend, validate_embeddings

_MAX_REPORTED_POSITIONS = 10


def _l2_normalise(matrix: np.ndarray, *, name: str) -> np.ndarray:
    """Scale each row of ``matrix`` to unit length.

    Raises:
        ConfigurationError: If any row has zero norm, because the angle of the zero vector, and
            therefore its cosine similarity, is undefined. The message reports row positions and
            counts only, never values.
    """
    norms = np.linalg.norm(matrix, axis=1)
    degenerate = np.flatnonzero(norms == 0.0)
    if degenerate.size:
        shown = degenerate[:_MAX_REPORTED_POSITIONS].tolist()
        suffix = "" if degenerate.size <= _MAX_REPORTED_POSITIONS else ", first 10 shown"
        raise ConfigurationError(
            f"{name} contains {degenerate.size} zero-norm row(s) at positions {shown}{suffix}; "
            "cosine similarity is undefined for the zero vector."
        )
    normalised: np.ndarray = matrix / norms[:, None]
    return normalised


def _top_positions(scores: np.ndarray, count: int) -> np.ndarray:
    """Return the ``count`` best positions in ``scores``, best first, ties by lowest position.

    ``numpy.argpartition`` selects an arbitrary member of a tie group at the cut, so the tie
    group straddling the cut is rebuilt explicitly before sorting; the result therefore depends
    only on the scores and on insertion order.
    """
    candidates: np.ndarray
    if count >= len(scores):
        candidates = np.arange(len(scores))
    else:
        threshold = scores[np.argpartition(-scores, count - 1)[:count]].min()
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        candidates = np.concatenate([above, tied[: count - len(above)]])
    ordered: np.ndarray = candidates[np.lexsort((candidates, -scores[candidates]))]
    return ordered


class BruteForceCosineBackend(RetrievalBackend):
    """Exhaustive cosine similarity over every stored vector.

    Stored vectors and queries are L2-normalised once, so a similarity is a dot product and a
    search is a single ``[Q, D] @ [D, N]`` matrix product. Cost is ``O(Q * N * D)`` with no index
    build step and no approximation, which keeps results exactly reproducible; the approximate
    backends in ``docs/design/RETRIEVAL.md`` are what large collections will need instead.

    Because similarity is computed on normalised vectors, scores are invariant to the scale of
    the input embeddings: doubling a vector does not change its ranking or its score.
    """

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    @property
    def size(self) -> int:
        """Number of stored vectors; ``0`` before ``fit``."""
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    @property
    def embedding_dim(self) -> int | None:
        """Width of the stored vectors, or ``None`` before ``fit``."""
        return None if self._vectors is None else int(self._vectors.shape[1])

    def fit(self, embeddings: np.ndarray) -> None:
        """Store ``[N, D]`` embeddings, replacing any previous ones. See ``RetrievalBackend.fit``.

        Raises:
            ConfigurationError: If the matrix is not a finite 2-D float array, has no rows, or
                contains a zero-norm row.
        """
        matrix = validate_embeddings(embeddings, name="embeddings")
        if matrix.shape[0] == 0:
            raise ConfigurationError("embeddings must contain at least one row, got 0.")
        self._vectors = _l2_normalise(matrix, name="embeddings")

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(positions, scores)`` for each query. See ``RetrievalBackend.search``.

        Raises:
            IndexNotFittedError: If called before ``fit``.
            ConfigurationError: If ``queries`` is not a finite ``[Q, D]`` float matrix with ``D``
                matching the stored vectors, contains a zero-norm row, or ``k`` is not a
                positive int.
        """
        if self._vectors is None:
            raise IndexNotFittedError("The backend has not been fitted; call fit first.")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ConfigurationError(f"k must be a positive int, got {k!r}.")
        matrix = validate_embeddings(queries, name="query", expected_dim=self.embedding_dim)
        scores = _l2_normalise(matrix, name="query") @ self._vectors.T
        width = min(k, self.size)
        positions = np.empty((len(scores), width), dtype=np.int64)
        top_scores = np.empty((len(scores), width), dtype=np.float32)
        for row, row_scores in enumerate(scores):
            selected = _top_positions(row_scores, width)
            positions[row] = selected
            top_scores[row] = row_scores[selected]
        return positions, top_scores
