# Contributing

[`AGENTS.md`](AGENTS.md) is the operating contract for all contributors, human or agent. This file covers repository mechanics only.

## Branches

`main` always passes every command listed under *Development* in [`README.md`](README.md).

Work happens on short-lived branches cut from `main`, one per milestone or coherent slice of a milestone:

| Prefix | Use | Example |
| --- | --- | --- |
| `feat/` | A milestone or feature slice from `IMPLEMENTATION_PLAN.md` | `feat/m1-backbone-model` |
| `fix/` | Bug fix to merged behaviour | `fix/device-index-validation` |
| `docs/` | Documentation-only change | `docs/leakage-guide` |
| `chore/` | Tooling, CI, packaging | `chore/ci-workflow` |

Milestone branches are named `feat/m<number>-<slug>`.

Once a branch meets the definition of done in `AGENTS.md`, it is merged into `main` with `git merge --no-ff`, and both `main` and the branch are pushed. Pull requests are not used. Merged branches are kept for reference and pruned, oldest first, once the repository holds more than 25.

## Commits

Write the subject in the imperative mood and describe the behaviour added or changed, not the steps taken. Code, its tests, and the documentation it affects belong in the same branch.

## Documentation

Documentation describes what the software does and why it is designed that way. Progress tracking lives in [`TASKS.md`](TASKS.md) and [`CHANGELOG.md`](CHANGELOG.md), and release planning in [`plans/`](plans/); design decisions that change architecture or public API get an ADR in `docs/adrs/`.
