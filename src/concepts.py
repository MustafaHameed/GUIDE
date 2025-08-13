"""Utilities for concept-based explanations using causal inference.

This module groups raw features into high-level pedagogical concepts and
estimates their causal effect on the final grade using ``dowhy``.

The current grouping focuses on three concepts:
``attendance``, ``socio_economic_status`` and ``engagement``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Concept definitions
# ---------------------------------------------------------------------------

CONCEPT_GROUPS: Dict[str, List[str]] = {
    "attendance": ["absences", "failures"],
    "socio_economic_status": ["Medu", "Fedu", "traveltime"],
    "engagement": ["studytime", "goout", "Dalc", "Walc", "freetime"],
}


def group_concepts(X: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw features into concept scores.

    Parameters
    ----------
    X:
        DataFrame containing the raw features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one column per concept. Each value is the mean of the
        constituent features for that row. Missing columns are ignored so the
        function is robust to datasets with fewer features.
    """

    concept_data: Dict[str, pd.Series] = {}
    for concept, cols in CONCEPT_GROUPS.items():
        existing = [c for c in cols if c in X.columns]
        if not existing:
            continue
        concept_data[concept] = X[existing].mean(axis=1)
    return pd.DataFrame(concept_data)


def estimate_concept_effects(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Estimate the causal effect of each concept on the outcome ``y``.

    A separate ``dowhy`` ``CausalModel`` is fit for each concept where the
    remaining concepts act as confounders. Effects are computed using a
    linear-regression backdoor estimator.

    Parameters
    ----------
    X:
        Raw feature matrix.
    y:
        Outcome variable (final grade).

    Returns
    -------
    pandas.DataFrame
        Table with columns ``concept`` and ``effect`` sorted by absolute
        effect size.
    """

    from dowhy import CausalModel  # Imported lazily to avoid hard dependency

    concepts = group_concepts(X)
    effects = []
    for concept in concepts.columns:
        treatment = concept
        confounders = [c for c in concepts.columns if c != concept]
        df = pd.concat([concepts, y.rename("y")], axis=1)
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome="y",
            common_causes=confounders,
        )
        identified = model.identify_effect()
        estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
        effects.append({"concept": concept, "effect": float(estimate.value)})

    effects_df = pd.DataFrame(effects).sort_values(
        by="effect", key=lambda s: s.abs(), ascending=False
    )
    return effects_df


def export_concept_importance(
    X: pd.DataFrame,
    y: pd.Series,
    table_path: Path,
    figure_path: Path,
) -> pd.DataFrame:
    """Compute and export concept importance table and plot.

    The resulting CSV and PNG/SVG are saved to ``table_path`` and
    ``figure_path`` respectively. Parent directories are created if needed.

    Parameters
    ----------
    X, y:
        Raw feature matrix and outcome variable.
    table_path:
        Destination for the CSV table.
    figure_path:
        Destination for the bar plot.

    Returns
    -------
    pandas.DataFrame
        The computed concept effects table for further inspection.
    """

    effects = estimate_concept_effects(X, y)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(table_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(effects["concept"], effects["effect"], color="C0")
    ax.set_ylabel("Estimated effect on final grade")
    ax.set_title("Concept Importance")
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return effects


# ---------------------------------------------------------------------------
# Command line utility
# ---------------------------------------------------------------------------


def _default_paths() -> tuple[Path, Path, Path]:
    project_dir = Path(__file__).resolve().parents[1]
    data_path = project_dir / "student-mat.csv"
    tables_path = project_dir / "tables" / "concept_importance.csv"
    figures_path = project_dir / "figures" / "concept_importance.png"
    return data_path, tables_path, figures_path


if __name__ == "__main__":
    data_path, table_path, fig_path = _default_paths()
    df = pd.read_csv(data_path)
    y = df["G3"]
    X = df.drop(columns=["G3"])
    export_concept_importance(X, y, table_path, fig_path)
