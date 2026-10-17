"""Ranking-quality metrics over retrieval results.

Each function scores an already-retrieved ranked list against the identifiers known to be
relevant for that query. Nothing here calls a model or an index, so the metrics are reusable and
testable on hand-written lists (``ARCHITECTURE.md``, "Evaluation separation").

The names carry the ``retrieval_`` prefix where a plain ``precision_at_k`` would collide with the
classification metric of the same name in ``embedfit.evaluation``, which ranks *predictions* by
score rather than neighbours by similarity.

A query with no relevant items is rejected rather than scored: its precision would be 0 by
construction and its recall has no denominator, so averaging it in would quietly depress every
number. Decide what such a query means — usually dropping it, and reporting how many were dropped
— before calling these functions.

Results are read positionally: the first element of ``results`` is treated as rank 1, whatever
``RetrievalResult.rank`` says, so slicing a result list stays meaningful. A list whose scores are
not in descending order is rejected, because ranking metrics computed on an unordered list are
silently meaningless.
"""

from collections.abc import Collection, Hashable, Sequence

from embedfit.exceptions import ConfigurationError, UndefinedMetricError
from embedfit.retrieval.base import RetrievalResult


def _scored(results: Sequence[RetrievalResult], k: int | None) -> tuple[int, list[Hashable]]:
    """Return the number of slots to score and the identifiers filling them, best first."""
    if isinstance(results, RetrievalResult | str) or not isinstance(results, Sequence):
        raise ConfigurationError(
            "results must be a sequence of RetrievalResult for one query, best match first."
        )
    previous: float | None = None
    for position, result in enumerate(results):
        if not isinstance(result, RetrievalResult):
            raise ConfigurationError(
                f"results[{position}] must be a RetrievalResult, got {type(result).__name__}."
            )
        if previous is not None and result.score > previous:
            raise ConfigurationError(
                "results must be ordered best match first; the score rises again at "
                f"position {position}."
            )
        previous = result.score
    if k is None:
        if not results:
            raise ConfigurationError(
                "k must be given when results is empty; there is no cutoff to infer."
            )
        cutoff = len(results)
    elif isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ConfigurationError(f"k must be a positive int, got {k!r}.")
    else:
        cutoff = k
    ids = [result.id for result in results]
    try:
        distinct = len(set(ids))
    except TypeError as exc:
        raise ConfigurationError("result ids must be hashable.") from exc
    if distinct != len(ids):
        raise ConfigurationError(
            f"results lists {len(ids) - distinct} id(s) more than once; a ranking holds each "
            "item at most once."
        )
    return cutoff, ids[:cutoff]


def _relevant(relevant_ids: Collection[Hashable], *, name: str) -> set[Hashable]:
    """Return the relevant identifiers as a set."""
    if isinstance(relevant_ids, str) or not isinstance(relevant_ids, Collection):
        raise ConfigurationError(f"{name} must be a collection of ids, e.g. a list or set.")
    try:
        known = set(relevant_ids)
    except TypeError as exc:
        raise ConfigurationError(f"{name} must contain hashable ids.") from exc
    if not known:
        raise UndefinedMetricError(
            f"{name} is empty; a query with no relevant items has no defined retrieval quality. "
            "Drop such queries explicitly instead."
        )
    return known


def _batch(
    batch_results: Sequence[Sequence[RetrievalResult]],
    batch_relevant_ids: Sequence[Collection[Hashable]],
) -> int:
    """Check that the two batch arguments line up and return the number of queries."""
    for values, name in (
        (batch_results, "batch_results"),
        (batch_relevant_ids, "batch_relevant_ids"),
    ):
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ConfigurationError(f"{name} must be a sequence with one entry per query.")
    if len(batch_results) != len(batch_relevant_ids):
        raise ConfigurationError(
            f"batch_results and batch_relevant_ids must have one entry per query: got "
            f"{len(batch_results)} and {len(batch_relevant_ids)}."
        )
    if not batch_results:
        raise ConfigurationError("batch_results is empty; at least one query is required.")
    return len(batch_results)


def retrieval_precision_at_k(
    results: Sequence[RetrievalResult],
    relevant_ids: Collection[Hashable],
    k: int | None = None,
) -> float:
    """Fraction of the top ``k`` results that are relevant.

    ``|{ids in the top k} ∩ relevant_ids| / k``. The denominator is always ``k``, so a query that
    returned fewer than ``k`` results is penalised for the empty slots.

    Args:
        results: One query's ranked results, best match first, as returned by
            ``RetrievalIndex.search``.
        relevant_ids: The identifiers that count as relevant for this query; must be non-empty.
        k: How many top results to score. ``None`` uses ``len(results)``, which requires at least
            one result.

    Returns:
        A value in [0, 1].

    Raises:
        ConfigurationError: If ``results`` is not an ordered sequence of ``RetrievalResult``
            or ``k`` is not a positive int.
        UndefinedMetricError: If ``relevant_ids`` is empty.
    """
    known = _relevant(relevant_ids, name="relevant_ids")
    cutoff, retrieved = _scored(results, k)
    return sum(identifier in known for identifier in retrieved) / cutoff


def retrieval_recall_at_k(
    results: Sequence[RetrievalResult],
    relevant_ids: Collection[Hashable],
    k: int | None = None,
) -> float:
    """Fraction of the relevant items that appear in the top ``k`` results.

    ``|{ids in the top k} ∩ relevant_ids| / |relevant_ids|``. The score is therefore capped below
    1 whenever there are more relevant items than result slots.

    Args:
        results: One query's ranked results, best match first.
        relevant_ids: The identifiers that count as relevant for this query; must be non-empty.
        k: How many top results to score. ``None`` uses ``len(results)``, which requires at least
            one result.

    Returns:
        A value in [0, 1].

    Raises:
        ConfigurationError: If ``results`` is not an ordered sequence of ``RetrievalResult``
            or ``k`` is not a positive int.
        UndefinedMetricError: If ``relevant_ids`` is empty.
    """
    known = _relevant(relevant_ids, name="relevant_ids")
    retrieved = set(_scored(results, k)[1])
    return len(retrieved & known) / len(known)


def mean_reciprocal_rank(
    batch_results: Sequence[Sequence[RetrievalResult]],
    batch_relevant_ids: Sequence[Collection[Hashable]],
    k: int | None = None,
) -> float:
    """Mean over queries of ``1 / rank`` of the first relevant result.

    A query whose top ``k`` results contain nothing relevant contributes 0, which is the standard
    reading: the user found nothing in the slots they were shown.

    Args:
        batch_results: One ranked result list per query, each best match first.
        batch_relevant_ids: The relevant identifiers of each query, in the same order. Each entry
            must be non-empty.
        k: How many top results to score per query. ``None`` uses each query's own result count.

    Returns:
        A value in [0, 1].

    Raises:
        ConfigurationError: If the two batches differ in length, either is empty, a result list
            is not ordered, or ``k`` is not a positive int.
        UndefinedMetricError: If a query has no relevant ids.
    """
    count = _batch(batch_results, batch_relevant_ids)
    total = 0.0
    for results, relevant_ids in zip(batch_results, batch_relevant_ids, strict=True):
        known = _relevant(relevant_ids, name="batch_relevant_ids entries")
        for rank, identifier in enumerate(_scored(results, k)[1], start=1):
            if identifier in known:
                total += 1.0 / rank
                break
    return total / count


def mean_average_precision(
    batch_results: Sequence[Sequence[RetrievalResult]],
    batch_relevant_ids: Sequence[Collection[Hashable]],
    k: int | None = None,
) -> float:
    """Mean over queries of average precision at the cutoff.

    For one query with ``n`` scored slots (``k`` if given, else that query's result count)::

        AP = sum(precision_at_i for each rank i <= n holding a relevant item) / min(R, n)

    where ``precision_at_i`` counts the relevant items among the first ``i`` results and ``R`` is
    the number of relevant identifiers. Dividing by ``min(R, n)`` rather than ``R`` makes 1.0
    reachable: a ranking whose first ``min(R, n)`` results are all relevant scores 1.0, while
    relevant items pushed below rank ``n`` still cost score.

    Args:
        batch_results: One ranked result list per query, each best match first.
        batch_relevant_ids: The relevant identifiers of each query, in the same order. Each entry
            must be non-empty.
        k: How many top results to score per query. ``None`` uses each query's own result count.

    Returns:
        A value in [0, 1].

    Raises:
        ConfigurationError: If the two batches differ in length, either is empty, a result list
            is not ordered, or ``k`` is not a positive int.
        UndefinedMetricError: If a query has no relevant ids.
    """
    count = _batch(batch_results, batch_relevant_ids)
    total = 0.0
    for results, relevant_ids in zip(batch_results, batch_relevant_ids, strict=True):
        known = _relevant(relevant_ids, name="batch_relevant_ids entries")
        cutoff, retrieved = _scored(results, k)
        found = 0
        precision_sum = 0.0
        for rank, identifier in enumerate(retrieved, start=1):
            if identifier in known:
                found += 1
                precision_sum += found / rank
        total += precision_sum / min(len(known), cutoff)
    return total / count
