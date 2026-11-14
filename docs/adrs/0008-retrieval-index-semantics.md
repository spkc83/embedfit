# ADR 0008 — Retrieval Exclusions Are Identifier-Based and Applied Before the Top-K Cut

Status: Accepted

## Decision

1. `exclude_self` removes the indexed item whose identifier equals the query's identifier, and nothing else. It therefore requires `query_ids` — one value for a single query, one per row for a batch — and raises `ConfigurationError` without it.
2. Leakage-safe group exclusion is per query: `fit(..., groups=...)` records one group per item, and `search(..., exclude_same_group=True, query_groups=...)` removes every item sharing the query's group. A `None` group belongs to no group and matches nothing. `exclude_groups` names groups to drop for every query in the call, for forbidden groups that are not the query's own.
3. Every exclusion is applied before the top `k` is cut. The index asks the backend for `k` plus the largest number of items excluded for any query in the call, then filters.
4. Results are totally ordered: descending score, and equal scores by ascending position in the fitted matrix.
5. A zero-norm vector is rejected at `fit` and at `search`, with a message naming the row positions.
6. The rank of `query` alone decides the call: `[D]` returns `list[RetrievalResult]`, `[Q, D]` returns `list[list[RetrievalResult]]`. Fewer than `k` results are returned when fewer candidates remain; the list is never padded.
7. `metric` selects the backend. Callers use `RetrievalIndex` and `RetrievalResult`; `RetrievalBackend` and `BruteForceCosineBackend` are exported as the extension contract for exact backends.
8. Retrieval metrics are named `retrieval_precision_at_k`, `retrieval_recall_at_k`, `mean_reciprocal_rank`, and `mean_average_precision`. They read a result list positionally, reject one that is not ordered best first, and reject a query whose relevant-identifier set is empty.

## Rationale

An implementation that dropped "the top hit" or everything scoring about 1.0 would silently discard genuine near-duplicates, which is exactly the signal a risk-oriented retrieval evaluation is looking for. Matching on the identifier is unambiguous, and requiring `query_ids` makes the leakage-relevant choice visible at the call site instead of inferring it (`AGENTS.md` rule 6).

Groups express "the same underlying item photographed twice", which `AGENTS.md` rule 5 forbids counting as evidence of generalisation. They belong to the query rather than to the index, because the same index serves queries from different groups.

Filtering after the cut would silently return fewer than `k` results whenever an excluded item ranked high, which changes every metric computed at `k`. Over-fetching by the number of exclusions is exact rather than heuristic because the backend ordering is total: its top `m` is always a prefix of its top `m + 1`, so removing at most `E` items from the top `k + E` leaves the same items as filtering the full ranking.

Ties are common with normalised embeddings of duplicated inputs, and `numpy.argpartition` is free to return any member of the tie group straddling the cut. Insertion position is the only tie-break available without a second signal, and it makes runs reproducible (`SPEC.md` §23).

The cosine of the zero vector is undefined; computing it yields NaN, which propagates into rankings and metrics as an arbitrary ordering rather than an error.

Rejecting an empty relevant set with `UndefinedMetricError` matches `embedfit.evaluation`, where recall-type metrics without positives raise the same error: such a query has precision 0 by construction and no recall denominator, so averaging it in quietly depresses every reported number.

## Consequences

`search` returns a union type, so a caller that passes a batch and indexes one result list needs a cast under strict typing. The alternative — a separate `search_batch` — was rejected because `SPEC.md` §18 fixes the single-method shape.

Over-fetching costs `E` extra slots per query, where `E` is the largest per-query exclusion count in the call. Excluding a large fraction of a large index through `exclude_ids` is therefore proportionally more expensive; a future backend with native identifier filtering can override that by implementing the filter itself.

Metadata is stored as a read-only shallow copy, so later mutation of the caller's dictionary does not alter indexed results, but nested mutable values are still shared. Metadata is never inspected, logged, or used for ranking, and `RetrievalResult.__repr__` reports only whether it is present, because results reach logs and tracebacks (`SECURITY_PRIVACY.md`).

Identifiers and group ids match on type and value, as labels do throughout the library (`embedfit.data.labels.label_key`), so `0`, `False`, and `"0"` are three different identifiers.

The backend contract requires a total order whose top `m` is a prefix of its top `m + 1`; exact over-fetching depends on it. Exact backends (brute force, FAISS flat indexes, exact SQL scans) satisfy it. An approximate backend (HNSW, IVF) does not, and would need its own over-fetch margin and a filter hook in `RetrievalBackend.search`, which the v0.1 protocol does not have. Backends are selected through a private table keyed by metric; there is no public registration hook yet.

Ranked lists passed to the metrics must not repeat an id, because a repeated relevant id would be counted twice.
