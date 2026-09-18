# Coding Agent Prompt — Continue

Read `AGENTS.md`, `TASKS.md`, and the current implementation state.

Identify the earliest incomplete milestone in `IMPLEMENTATION_PLAN.md`.

Implement only the coherent next slice of that milestone.

Before coding:

- inspect existing tests;
- inspect related ADRs;
- confirm no spec conflict.

After coding:

- add/update tests;
- run targeted tests;
- run full offline test suite if practical;
- run lint;
- update `TASKS.md`;
- update docs if public behavior changed.

Do not expand scope merely because adjacent improvements are attractive.
