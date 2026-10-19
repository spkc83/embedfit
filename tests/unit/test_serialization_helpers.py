"""Helpers shared by model bundles and representation checkpoints."""

import json
import os

import pytest

from embedfit import EmbedFitModel
from embedfit.exceptions import ConfigurationError
from embedfit.heads import ClassificationHead
from embedfit.serialization import check_saveable, write_json_atomic


def test_write_json_atomic_replaces_the_file(tmp_path):
    path = tmp_path / "index.json"
    write_json_atomic(path, {"epoch": 1})
    write_json_atomic(path, {"epoch": 2})
    assert json.loads(path.read_text()) == {"epoch": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["index.json"]


def test_a_failed_replace_keeps_the_previous_file_and_no_temporary(tmp_path, monkeypatch):
    path = tmp_path / "index.json"
    write_json_atomic(path, {"epoch": 1})

    def failing(source, target):
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(OSError, match="interrupted"):
        write_json_atomic(path, {"epoch": 2})
    assert json.loads(path.read_text()) == {"epoch": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["index.json"]


def test_check_saveable_rejects_what_save_model_cannot_write():
    check_saveable(EmbedFitModel.from_pretrained("dummy/tiny", labels=["a", "b"]))
    with pytest.raises(ConfigurationError, match="cannot be saved"):
        check_saveable(EmbedFitModel.from_pretrained("dummy/tiny", labels=[("a",), ("b",)]))

    class Unregistered(ClassificationHead):
        is_fitted = False

        def fit(self, features, targets, *, num_classes):
            raise NotImplementedError

        def predict_proba(self, features):
            raise NotImplementedError

    with pytest.raises(ConfigurationError, match="registry_name"):
        check_saveable(EmbedFitModel.from_pretrained("dummy/tiny", head=Unregistered()))


def test_write_json_atomic_gives_the_permissions_of_an_ordinary_file(tmp_path):
    umask = os.umask(0o022)
    try:
        write_json_atomic(tmp_path / "index.json", {"epoch": 1})
    finally:
        os.umask(umask)
    assert (tmp_path / "index.json").stat().st_mode & 0o777 == 0o644
