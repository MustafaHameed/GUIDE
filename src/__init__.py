"""Student performance package with convenience imports."""

from .data import load_data
from .preprocessing import build_pipeline
from .model import create_model
from .concepts import (
    group_concepts,
    estimate_concept_effects,
    export_concept_importance,
)
# Dashboard utils commented out to avoid streamlit dependency during development
# from .dashboard_utils import (
#     _safe_read_csv,
#     _list_images,
#     _show_images_grid,
#     _show_table,
#     clear_caches,
# )

__all__ = [
    "load_data",
    "build_pipeline", 
    "create_model",
    "group_concepts",
    "estimate_concept_effects",
    "export_concept_importance",
    # Dashboard functions commented out
    # "_safe_read_csv",
    # "_list_images", 
    # "_show_images_grid",
    # "_show_table",
    # "clear_caches",
]
