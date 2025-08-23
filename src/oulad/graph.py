"""Graph utilities for OULAD student-VLE interactions.

This module constructs a bipartite graph between students and VLE
objects based on the ``studentVle`` and ``vle`` tables from the OULAD
dataset.  It also defines helper functions for converting the raw data
into :class:`torch_geometric.data.HeteroData` objects that can be fed to
graph neural network models.

The graph contains two node types:

``student``
    One node per unique ``id_student``.
``vle``
    One node per unique ``id_site`` (VLE resource).

Edges go from students to VLE resources with attributes:
``weight``
    Sum of clicks for that student-resource pair.
``time``
    The day of interaction relative to the course start.  When multiple
    interactions exist, we keep the earliest timestamp.

These features are intentionally lightweight so the graph can be built
quickly for experimentation.  More complex featurisation (e.g. activity
metadata embeddings or temporal sequences) can be layered on top of this
basic representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import torch
from torch_geometric.data import HeteroData


@dataclass
class GraphBuildResult:
    """Container for the graph object and mapping dictionaries."""

    graph: HeteroData
    student_id_map: dict
    site_id_map: dict


def build_student_vle_graph(
    student_vle: pd.DataFrame,
    vle: pd.DataFrame,
) -> GraphBuildResult:
    """Create a heterogeneous graph of student‒VLE interactions.

    Parameters
    ----------
    student_vle:
        DataFrame with columns ``['id_student', 'id_site', 'sum_click', 'date']``.
    vle:
        VLE metadata table. Currently only ``id_site`` is used but other
        columns may be joined for additional node features.

    Returns
    -------
    GraphBuildResult
        Wrapper containing the :class:`HeteroData` graph and the mapping
        dictionaries from raw IDs to node indices.
    """

    # Ensure required columns exist
    required_cols = {"id_student", "id_site", "sum_click"}
    missing = required_cols - set(student_vle.columns)
    if missing:
        raise ValueError(f"student_vle missing required columns: {missing}")

    # Merge with VLE metadata (allowing extension for activity type etc.)
    merged = student_vle.merge(vle, on="id_site", how="left")

    # Create index mappings for nodes
    student_ids = merged["id_student"].unique()
    site_ids = merged["id_site"].unique()
    student_id_map = {sid: i for i, sid in enumerate(student_ids)}
    site_id_map = {sid: i for i, sid in enumerate(site_ids)}

    # Node features: simple total click counts
    student_clicks = (
        merged.groupby("id_student")["sum_click"].sum().reindex(student_ids).fillna(0)
    )
    vle_clicks = (
        merged.groupby("id_site")["sum_click"].sum().reindex(site_ids).fillna(0)
    )

    data = HeteroData()
    data["student"].x = torch.tensor(student_clicks.values, dtype=torch.float).unsqueeze(1)
    data["vle"].x = torch.tensor(vle_clicks.values, dtype=torch.float).unsqueeze(1)

    # Build edges with weights and earliest interaction time
    grouped = merged.groupby(["id_student", "id_site"])
    edge_index = []
    edge_weight = []
    edge_time = []

    for (sid, vid), group in grouped:
        edge_index.append([student_id_map[sid], site_id_map[vid]])
        edge_weight.append(group["sum_click"].sum())
        edge_time.append(group["date"].min() if "date" in group else 0)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data["student", "interacts", "vle"].edge_index = edge_index
    data["student", "interacts", "vle"].weight = torch.tensor(edge_weight, dtype=torch.float)
    data["student", "interacts", "vle"].time = torch.tensor(edge_time, dtype=torch.float)

    return GraphBuildResult(data, student_id_map, site_id_map)
