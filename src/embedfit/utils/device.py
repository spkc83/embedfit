"""Device resolution with an explicit ``auto`` policy and no silent fallback."""

import logging

import torch

from embedfit.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device request to a usable ``torch.device``.

    Args:
        device: ``"auto"`` or ``None`` selects the best available device in the order
            CUDA → MPS → CPU. Any other value (``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``, or
            a ``torch.device``) is treated as an explicit request.

    Returns:
        The resolved ``torch.device``.

    Raises:
        ConfigurationError: If the value is not a valid device, or an explicitly requested
            device is unavailable. An explicit request never falls back to another device.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
        elif _mps_available():
            resolved = torch.device("mps")
        else:
            resolved = torch.device("cpu")
        logger.debug("Resolved device 'auto' to '%s'", resolved)
        return resolved

    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ConfigurationError(
            f"Invalid device {device!r}. Use 'auto', 'cpu', 'cuda', 'cuda:<index>', or 'mps'."
        ) from exc

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError(
                "Device 'cuda' was requested but CUDA is not available. "
                "Use device='auto' or device='cpu'."
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ConfigurationError(
                f"Device '{resolved}' was requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
    elif resolved.type == "mps":
        if not _mps_available():
            raise ConfigurationError(
                "Device 'mps' was requested but MPS is not available. "
                "Use device='auto' or device='cpu'."
            )
    elif resolved.type != "cpu":
        raise ConfigurationError(
            f"Unsupported device type '{resolved.type}'. Supported: cpu, cuda, mps."
        )
    return resolved
