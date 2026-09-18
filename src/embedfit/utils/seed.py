"""Seeding for every random number generator EmbedFit relies on."""

import logging
import random

import numpy as np
import torch

from embedfit.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_MAX_SEED = 2**32 - 1  # NumPy's legacy global seed must fit in an unsigned 32-bit integer.


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed the Python, NumPy, and torch (CPU and CUDA) random number generators.

    Args:
        seed: Integer in ``[0, 2**32 - 1]``. Booleans are rejected.
        deterministic: When ``True``, additionally request deterministic torch algorithms and
            disable cuDNN autotuning. Operations without a deterministic implementation emit a
            warning instead of raising, so this trades speed for reproducibility without making
            models unusable. On CUDA, full determinism of cuBLAS also requires the
            ``CUBLAS_WORKSPACE_CONFIG`` environment variable to be set before the process starts.
            When ``False`` the global determinism settings are left untouched.

    Raises:
        ConfigurationError: If ``seed`` is not an integer in the supported range.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError(f"seed must be an int, got {type(seed).__name__}.")
    if not 0 <= seed <= _MAX_SEED:
        raise ConfigurationError(f"seed must be in [0, {_MAX_SEED}], got {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Seeds CPU and all CUDA devices.

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    logger.debug("Seeded RNGs with seed=%d (deterministic=%s)", seed, deterministic)
