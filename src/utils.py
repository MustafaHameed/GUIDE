from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists.

    Parameters
    ----------
    path:
        The directory to create. If the directory already exists, the call
        has no effect.
    """
    path.mkdir(parents=True, exist_ok=True)
