import pytest
import torch

from embedfit.exceptions import ConfigurationError
from embedfit.utils import device as device_module
from embedfit.utils import resolve_device


@pytest.fixture
def no_accelerators(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_module, "_mps_available", lambda: False)


def test_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


@pytest.mark.parametrize("request_value", ["auto", None])
def test_auto_falls_back_to_cpu(no_accelerators, request_value):
    assert resolve_device(request_value) == torch.device("cpu")


def test_auto_prefers_cuda_over_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device_module, "_mps_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_auto_uses_mps_when_cuda_missing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_module, "_mps_available", lambda: True)
    assert resolve_device("auto").type == "mps"


@pytest.mark.parametrize("requested", ["cuda", "cuda:0", "mps"])
def test_explicit_unavailable_device_never_falls_back(no_accelerators, requested):
    with pytest.raises(ConfigurationError, match="not available"):
        resolve_device(requested)


def test_cuda_index_out_of_range(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ConfigurationError, match="1 CUDA device"):
        resolve_device("cuda:3")


def test_invalid_device_string():
    with pytest.raises(ConfigurationError, match="Invalid device"):
        resolve_device("gpu-please")


def test_unsupported_device_type():
    with pytest.raises(ConfigurationError, match="Unsupported device type"):
        resolve_device("meta")
