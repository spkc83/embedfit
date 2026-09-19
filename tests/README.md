# Tests

- `tests/unit/` — pure components.
- `tests/integration/` — complete tiny workflows on `DummyBackbone` and synthetic images.
- `tests/conftest.py` — shared fixtures (`bar_dataset`, `dummy_backbone`).

The default run (`pytest -q`) requires no internet access, GPU, or pretrained checkpoint. Tests that need any of these carry the `network` or `slow` marker, which the default configuration in `pyproject.toml` deselects; run them with `pytest -m network` or `pytest -m slow`. Unregistered markers are an error. Because the marker filter lives in `addopts`, selecting a marked test by node id also needs the matching `-m` option.
