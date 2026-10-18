# ADR 0001 — Project Name

Status: Accepted provisionally

## Decision

Use **EmbedFit** / package `embedfit`.

## Context

The initial concept was called `ImageSetFit` because it extends the SetFit mental model to images.

## Rationale

`ImageSetFit` is clear but:

- appears affiliated with the SetFit project;
- constrains the project to images;
- describes an implementation lineage more than a reusable abstraction.

`EmbedFit` better represents supervised adaptation of pretrained embedding spaces.

## Consequence

Vision is the only required v0.1 modality, but naming does not prevent future modality expansion.

The first publication is planned on Hugging Face, with a blog post, and does not depend on a package-index name. The name has not been checked against PyPI; it remains provisional until that check, which comes before any PyPI upload. The fallbacks are in `PROJECT_NAMING.md`.

## Amendment (v0.1 scope, 2026-10-18)

The last paragraph of Consequence replaces "The name remains provisional until publication/package registration checks are completed." The PyPI name check is no longer a v0.1 release item: the first publication is planned on Hugging Face, and the check comes before any PyPI upload. The status stays "Accepted provisionally".
