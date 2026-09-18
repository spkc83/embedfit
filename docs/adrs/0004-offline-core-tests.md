# ADR 0004 — Default Tests Are Offline and CPU-Only

Status: Accepted

## Decision

The default test suite may not require network access or GPU hardware.

## Rationale

Fast deterministic tests are critical for coding-agent iteration and CI reliability.

## Consequences

Implement `DummyBackbone` and synthetic fixtures early.

Provider/download tests are explicitly marked and excluded by default.
