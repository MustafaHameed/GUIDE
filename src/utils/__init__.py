"""Utilities package for reproducibility and helper functions."""

from pathlib import Path
from .repro import set_seed, setup_reproducibility, ensure_reproducible_environment


def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists.

    Parameters
    ----------
    path:
        The directory to create. If the directory already exists, the call
        has no effect.
    """
    path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "set_seed", 
    "setup_reproducibility", 
    "ensure_reproducible_environment",
    "ensure_dir"
]