# Coding Agent Prompt — Architecture / Quality Review

Review the repository against:

- `SPEC.md`
- `ARCHITECTURE.md`
- `ACCEPTANCE_CRITERIA.md`
- `SECURITY_PRIVACY.md`
- ADRs

Focus on:

1. data leakage;
2. unintended network dependency;
3. sensitive-data logging;
4. label/probability ordering;
5. backbone preprocessing correctness;
6. serialization round-trip;
7. training/evaluation split contamination;
8. deterministic sampling;
9. public API inconsistencies;
10. missing tests.

Do not make broad refactors before producing a concrete issue list prioritized by severity.
