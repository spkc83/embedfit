# ADR 0002 — Two-Stage Training Is the Default

Status: Accepted. Amended by ADR 0022: `end_to_end` with a `contrastive_weight` is a representation-learning method under test; with the weight at 0 it is the baseline described here.

## Decision

The canonical training flow is:

1. representation adaptation;
2. freeze representation;
3. fit lightweight classifier.

## Rationale

This is the core property being tested from the SetFit-inspired approach.

It also cleanly separates:

- representation quality
- classifier complexity

and makes embeddings reusable for retrieval.

## Consequences

Trainer logs and checkpoints must make stages explicit.

End-to-end classification remains a baseline, not the default.
