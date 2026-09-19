import pytest

from embedfit.exceptions import ConfigurationError, UndefinedMetricError
from embedfit.retrieval import (
    RetrievalResult,
    mean_average_precision,
    mean_reciprocal_rank,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)


def ranked(*ids: str) -> list[RetrievalResult]:
    """A ranked result list whose scores fall by 0.1 per position."""
    return [
        RetrievalResult(id=identifier, score=1.0 - 0.1 * position, rank=position + 1)
        for position, identifier in enumerate(ids)
    ]


def test_precision_counts_relevant_hits_among_k_slots():
    results = ranked("a", "b", "c", "d")
    assert retrieval_precision_at_k(results, {"a", "c"}) == 0.5  # 2 of 4.
    assert retrieval_precision_at_k(results, {"a", "c"}, 1) == 1.0  # 1 of 1.
    assert retrieval_precision_at_k(results, {"a", "c"}, 2) == 0.5  # "a" only.
    assert retrieval_precision_at_k(results, {"a", "c"}, 3) == pytest.approx(2 / 3)


def test_precision_charges_for_slots_that_returned_nothing():
    # Only four results were retrieved, so two of the six slots are empty misses.
    assert retrieval_precision_at_k(ranked("a", "b", "c", "d"), {"a", "c"}, 6) == pytest.approx(
        2 / 6
    )


def test_precision_is_zero_when_nothing_relevant_was_retrieved():
    assert retrieval_precision_at_k(ranked("a", "b"), {"z"}) == 0.0


def test_recall_measures_coverage_of_the_relevant_set():
    results = ranked("a", "b", "c", "d")
    assert retrieval_recall_at_k(results, {"a", "c", "z"}) == pytest.approx(2 / 3)
    assert retrieval_recall_at_k(results, {"a", "c", "z"}, 1) == pytest.approx(1 / 3)
    assert retrieval_recall_at_k(results, {"a", "b", "c", "d"}) == 1.0


def test_recall_cannot_reach_one_with_more_relevant_items_than_slots():
    assert retrieval_recall_at_k(ranked("a", "b", "c"), {"a", "b", "c", "d"}, 3) == 0.75


def test_reciprocal_rank_uses_the_first_relevant_position():
    # 1/2 for the first query, 0 for the second: (0.5 + 0) / 2.
    batch = [ranked("a", "b", "c"), ranked("x", "y")]
    assert mean_reciprocal_rank(batch, [{"b"}, {"q"}]) == 0.25


def test_reciprocal_rank_ignores_relevant_items_below_the_cutoff():
    batch = [ranked("a", "b", "c")]
    assert mean_reciprocal_rank(batch, [{"c"}]) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank(batch, [{"c"}], 2) == 0.0


def test_average_precision_rewards_relevant_items_ranked_early():
    # Query 1: hits at ranks 1 and 3 give (1/1 + 2/3) / min(2, 4) = 5/6.
    # Query 2: a hit at rank 2 gives (1/2) / min(1, 2) = 1/2.
    batch = [ranked("a", "b", "c", "d"), ranked("x", "y")]
    assert mean_average_precision(batch, [{"a", "c"}, {"y"}]) == pytest.approx((5 / 6 + 1 / 2) / 2)


def test_average_precision_is_one_for_a_perfect_ranking():
    assert mean_average_precision([ranked("a", "b", "c")], [{"a", "b"}]) == 1.0
    assert mean_average_precision([ranked("a", "b", "c")], [{"a", "b"}], 1) == 1.0


def test_average_precision_prefers_the_better_ordering():
    good = mean_average_precision([ranked("a", "z", "b")], [{"a", "b"}])
    poor = mean_average_precision([ranked("z", "a", "b")], [{"a", "b"}])
    assert good > poor > 0.0


def test_metrics_read_positions_not_the_rank_field():
    # Slicing a result list leaves stale rank values behind; the metrics must not use them.
    results = ranked("a", "b", "c")[1:]
    assert [result.rank for result in results] == [2, 3]
    assert mean_reciprocal_rank([results], [{"b"}]) == 1.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: retrieval_precision_at_k(ranked("a"), set()),
        lambda: retrieval_recall_at_k(ranked("a"), []),
        lambda: mean_reciprocal_rank([ranked("a")], [set()]),
        lambda: mean_average_precision([ranked("a")], [[]]),
    ],
)
def test_a_query_without_relevant_items_is_rejected(call):
    with pytest.raises(UndefinedMetricError, match="has no defined retrieval quality"):
        call()


@pytest.mark.parametrize("k", [0, -1, 1.0, True, "3"])
def test_invalid_k(k):
    with pytest.raises(ConfigurationError, match="k must be a positive int"):
        retrieval_precision_at_k(ranked("a", "b"), {"a"}, k)


def test_k_is_required_when_no_results_were_returned():
    with pytest.raises(ConfigurationError, match="k must be given when results is empty"):
        retrieval_precision_at_k([], {"a"})
    assert retrieval_precision_at_k([], {"a"}, 5) == 0.0
    assert retrieval_recall_at_k([], {"a"}, 5) == 0.0


def test_results_out_of_score_order_are_rejected():
    rising = [
        RetrievalResult(id="a", score=0.1, rank=1),
        RetrievalResult(id="b", score=0.9, rank=2),
    ]
    with pytest.raises(ConfigurationError, match="score rises again at position 1"):
        retrieval_precision_at_k(rising, {"a"})


def test_equal_scores_are_accepted_as_ordered():
    tied = [RetrievalResult(id="a", score=0.5, rank=1), RetrievalResult(id="b", score=0.5, rank=2)]
    assert retrieval_precision_at_k(tied, {"b"}) == 0.5


@pytest.mark.parametrize(
    "results, match",
    [
        ("ab", "results must be a sequence of RetrievalResult"),
        (RetrievalResult(id="a", score=1.0, rank=1), "results must be a sequence"),
        (["a"], r"results\[0\] must be a RetrievalResult, got str"),
    ],
)
def test_invalid_results(results, match):
    with pytest.raises(ConfigurationError, match=match):
        retrieval_precision_at_k(results, {"a"}, 1)


def test_relevant_ids_must_be_a_collection():
    with pytest.raises(ConfigurationError, match="must be a collection of ids"):
        retrieval_precision_at_k(ranked("a"), "a")


@pytest.mark.parametrize("metric", [mean_reciprocal_rank, mean_average_precision])
def test_batch_arguments_must_line_up(metric):
    with pytest.raises(ConfigurationError, match="must have one entry per query: got 2 and 1"):
        metric([ranked("a"), ranked("b")], [{"a"}])
    with pytest.raises(ConfigurationError, match="at least one query is required"):
        metric([], [])
    with pytest.raises(ConfigurationError, match="batch_results must be a sequence"):
        metric("ab", [{"a"}, {"b"}])


def test_reciprocal_rank_stops_at_the_first_relevant_hit():
    # Both results are relevant: only the first counts (1/1), not 1/1 + 1/2.
    assert mean_reciprocal_rank([ranked("a", "b")], [{"a", "b"}]) == 1.0
    assert mean_reciprocal_rank([ranked("x", "a", "b")], [{"a", "b"}]) == 0.5


@pytest.mark.parametrize("metric", [retrieval_precision_at_k, retrieval_recall_at_k])
def test_a_ranking_with_a_repeated_id_is_rejected(metric):
    with pytest.raises(ConfigurationError, match="lists 1 id.s. more than once"):
        metric(ranked("a", "b", "a"), {"a"})
    with pytest.raises(ConfigurationError, match="more than once"):
        mean_average_precision([ranked("a", "a")], [{"a"}], 2)  # Would otherwise score 2.0.
