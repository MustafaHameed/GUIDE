"""Student performance package with convenience imports."""

from .data import load_data
from .preprocessing import build_pipeline
from .model import create_model

__all__ = ["load_data", "build_pipeline", "create_model"]
