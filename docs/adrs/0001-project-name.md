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

The name remains provisional until publication/package registration checks are completed.
