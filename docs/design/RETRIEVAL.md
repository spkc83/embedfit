# Retrieval Design

## Goal

Use the same learned embedding space for classification and similarity search.

## v0.1 backend

Brute-force cosine similarity using NumPy or PyTorch.

This is intentionally simple and testable.

## Interface

```python
index = RetrievalIndex(metric="cosine")
index.fit(embeddings, ids, metadata=None)
hits = index.search(query, k=10, exclude_ids=None)
```

## Result

```python
@dataclass(frozen=True)
class RetrievalResult:
    id: str
    score: float
    rank: int
    metadata: Mapping[str, Any] | None = None
```

## Safety against evaluation leakage

Evaluation utilities should optionally exclude:

- same sample ID
- same physical-item group
- custom forbidden groups

A near-duplicate of the same original item should not count as evidence that the representation generalizes.

## Future backends

Design an internal backend protocol compatible later with:

- FAISS
- pgvector
- Qdrant
- Milvus
- OpenSearch

Do not add these dependencies in v0.1.

## Implemented semantics

`RetrievalIndex` owns identifiers, metadata, groups, and exclusions; `RetrievalBackend` owns storage and top-k search over positions `0..N-1`. `metric` selects the backend, and `"cosine"` — `BruteForceCosineBackend`, an exhaustive matrix product over L2-normalised vectors — is the only metric in v0.1. Anything else raises `ConfigurationError` listing the valid metrics. The decisions below are recorded in [ADR 0008](../adrs/0008-retrieval-index-semantics.md).

### Indexing

```python
index.fit(embeddings, ids=None, metadata=None, groups=None)
```

`embeddings` is a finite `[N, D]` float matrix with `N >= 1`. `ids` holds one unique hashable identifier per row and defaults to `0..N-1`; identifiers (and group ids) match on type and value, like labels elsewhere in the library, so `0`, `False`, and `"0"` are different identifiers. Values are stored as float32; a float64 value too large for float32 is rejected with the other non-finite values. `metadata` holds one mapping, or `None`, per row, kept as a read-only shallow copy and returned unchanged with its item; it is never inspected, logged, or used for ranking. `groups` names the underlying thing a row came from — the same physical document, customer, or session — and `None` means the row belongs to no group. Fitting again replaces everything.

### Ordering

Results are sorted by descending score. Equal scores are broken by ascending position in the fitted matrix, so the ordering is total and two runs over the same data agree exactly. Cosine similarity is scale-free: rescaling an embedding changes neither its score nor its rank. Norms are computed in float64, so this holds across the whole float32 range.

Because ties go to the earlier row, an indexed item queried with its own embedding is rank 1 unless an exact duplicate was inserted before it; the duplicate then takes rank 1 and the item follows. `exclude_self` removes the query's own item by id and keeps such duplicates.

### Cost

Brute-force search is `O(Q·N·D)` time and materialises the `[Q, N]` score matrix, so memory grows with the batch: about 195 MiB of scores for 512 queries against 100 000 items. Exclusions shared by every query use one `[N]` mask; per-query exclusions (`exclude_self`, `exclude_same_group`) add a `[Q, N]` boolean mask. Split very large query batches into chunks.

### Zero vectors

The cosine of the zero vector is undefined, so a zero-norm row raises `ConfigurationError` at `fit` and a zero-norm query raises it at `search`, naming the row positions. Nothing returns NaN.

### Searching

```python
index.search(query, k=10, *, query_ids=None, exclude_self=False, exclude_ids=None,
             query_groups=None, exclude_same_group=False, exclude_groups=None)
```

The rank of `query` decides the return type: a `[D]` vector returns `list[RetrievalResult]`, and a `[Q, D]` batch returns one such list per row. Ranks restart at 1 in each list and are assigned after exclusions. Fewer than `k` results are returned when the index minus the exclusions holds fewer than `k` items; the list is short rather than padded. Searching before `fit` raises `IndexNotFittedError`.

A batch is one matrix product rather than `Q` of them, and BLAS accumulates a blocked product in a different order from a single row, so a query's scores can differ in the last float32 digits between a batched and a single call. Items whose scores differ by less than that error can therefore swap between the two.

### Exclusions

Every exclusion is applied before the top `k` is cut, so `k` results still come back whenever `k` candidates remain.

| Option | Removes | Needs |
| --- | --- | --- |
| `exclude_self=True` | The one indexed item whose id equals the query's id | `query_ids`: one id, or one per batch row |
| `exclude_ids` | The named ids, for every query in the call | — |
| `exclude_same_group=True` | Every item whose group equals the query's group | `groups` at fit; `query_groups` per query |
| `exclude_groups` | Every item in the named groups, for every query in the call | `groups` at fit |

`exclude_self` matches on the identifier alone. A duplicate of the query stored under a different id is still returned, which is what makes near-duplicate leakage visible instead of hiding it. An identifier that is not in the index removes nothing. Asking for an exclusion without the data it needs, or passing `query_ids` or `query_groups` without the flag that uses them, raises `ConfigurationError` rather than quietly doing nothing.

### Metrics

`embedfit.retrieval.metrics` scores an already-retrieved ranked list against the identifiers known to be relevant for that query; nothing there calls a model or an index. A list must not contain the same id twice, and is read positionally — its first element is rank 1 — and a list whose scores are not descending is rejected. `k` defaults to the number of results supplied.

| Function | Definition |
| --- | --- |
| `retrieval_precision_at_k(results, relevant_ids, k=None)` | `hits in the top k / k`. Slots that returned nothing count as misses |
| `retrieval_recall_at_k(results, relevant_ids, k=None)` | `hits in the top k / len(relevant_ids)` |
| `mean_reciprocal_rank(batch_results, batch_relevant_ids, k=None)` | Mean over queries of `1 / rank` of the first relevant result, or 0 when the top `k` holds none |
| `mean_average_precision(batch_results, batch_relevant_ids, k=None)` | Mean over queries of `sum(precision at each relevant rank) / min(len(relevant_ids), k)`, so a ranking whose first `min(len(relevant_ids), k)` results are all relevant scores 1.0 |

A query with no relevant items raises `UndefinedMetricError`: its precision is 0 by construction and its recall has no denominator, so averaging it in would quietly depress every number. Drop such queries deliberately, and report how many were dropped.
