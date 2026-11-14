# ADR 0005 — Domain-Specific Risk Logic Does Not Enter Core Modeling

Status: Accepted

## Decision

The framework is domain-neutral.

Use-case-specific behavior belongs in examples, docs, and generic metric utilities.

## Rationale

The same representation-learning architecture applies to many image-classification tasks.

## Consequences

No hard-coded:

- use-case labels
- document fields
- operational event semantics
- organisation-specific metadata
