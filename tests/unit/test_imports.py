import importlib
import logging

import pytest

import embedfit

SUBPACKAGES = [
    "backbones",
    "data",
    "evaluation",
    "exceptions",
    "heads",
    "losses",
    "retrieval",
    "sampling",
    "testing",
    "utils",
]


def test_version_is_a_nonempty_string():
    assert isinstance(embedfit.__version__, str) and embedfit.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(f"embedfit.{name}")


def test_root_exports_resolve():
    for name in embedfit.__all__:
        assert hasattr(embedfit, name)


def test_library_installs_only_a_null_handler():
    handlers = logging.getLogger("embedfit").handlers
    assert handlers and all(isinstance(h, logging.NullHandler) for h in handlers)


def test_exception_hierarchy():
    from embedfit.exceptions import ConfigurationError, EmbedFitError, ImageInputError

    assert issubclass(ConfigurationError, EmbedFitError)
    assert issubclass(ImageInputError, EmbedFitError)
    assert issubclass(ConfigurationError, ValueError)
