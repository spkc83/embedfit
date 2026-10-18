# Project Naming Decision

## Recommendation: `EmbedFit`

Use:

- repository: `embedfit`
- Python package: `embedfit`
- product/docs name: **EmbedFit**

## Why not `ImageSetFit`

`ImageSetFit` is descriptive and immediately communicates the inspiration, but it is not the strongest long-term project name.

Concerns:

1. It sounds like an official SetFit extension or Hugging Face subproject.
2. It binds the framework to images even though the abstraction is embedding adaptation.
3. The architecture may later support video, document regions, or other modalities.
4. A distinct name makes independent governance and branding clearer.

## Why `EmbedFit`

It captures the central algorithmic abstraction:

```text
pretrained embedding model
→ fit/adapt representation with few labels
→ fit lightweight downstream head
```

It is short enough for a Python package and does not encode the initial domain.

## Naming fallback options

The first publication is planned on Hugging Face, with a blog post. Before any PyPI upload, check `embedfit` on PyPI; if it is unavailable, evaluate:

- `EmbeddingFit`
- `FewFit`
- `MetricFit`
- `RepFit`

Avoid names that imply affiliation with SetFit unless the project is intentionally released as an official extension.

## Rename strategy

All project-name assumptions should be concentrated in:

- `pyproject.toml`
- package directory name
- top-level docs
- install hints (`pip install 'embedfit[...]'`) in provider errors
- serialization: the `framework` key in the bundle config and the marker file `embedfit_config.json`; renaming either needs a new `format_version`

Do not encode `embedfit` into model architecture classes or file formats unnecessarily.
