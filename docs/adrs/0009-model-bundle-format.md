# ADR 0009 — Model Bundles Are Pickle-Free and Self-Describing

Status: Accepted

## Decision

1. A saved model is a directory of JSON files, NumPy `.npz` arrays, and a tensor-only PyTorch state dict. Nothing is pickled: the logistic-regression head stores its coefficients and intercepts as arrays and rebuilds the scikit-learn estimator from them.
2. Components are reloaded by registered name, never by import path: `backbone/config.json` names a backbone provider with a registered loader, and `head/config.json` names a registered head class.
3. `embedfit_config.json` carries a `format_version` and a SHA-256 fingerprint of the backbone's configuration file. Loading refuses a newer format version, a changed backbone configuration, or a mismatched embedding dimension.
4. `EmbedFitModel.from_pretrained` recognises a saved model by the presence of `embedfit_config.json`, and refuses options that would override saved values.

## Rationale

Model files travel between teams and environments. Unpickling executes arbitrary code, and pickled scikit-learn estimators also break across library versions; a logistic regression is fully described by two arrays, so there is no reason to accept either risk. Reloading by registered name keeps the same property for custom components: a bundle can only instantiate classes the running process has deliberately registered.

Preprocessing belongs to the checkpoint (ADR 0003). If a backbone's saved resolution or normalisation constants are edited while its weights stay the same, the model silently degrades; the fingerprint turns that into an error.

Identifiers look like relative paths (`torchvision/resnet18`). Deciding by the presence of the bundle's own marker file means a stray directory cannot hijack an identifier, and a genuine bundle is never mistaken for one.

Letting callers override labels or the normalisation policy at load time would produce a model whose head no longer matches its inputs or outputs.

## Consequences

Custom heads and backbone providers need a small amount of work to be saveable: a `registry_name` plus `save`/`load` for a head, a `loader` for a provider. Without it, saving fails with an explanatory error rather than writing a bundle that cannot be read back.

Labels must be JSON scalars (`str`, `int`, `float`, `bool`) to be saved.

Bundles contain weights, configuration, label names, and aggregate training metadata only (`SECURITY_PRIVACY.md`).
