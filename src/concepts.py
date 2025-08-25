"""Concept-level grouping and causal effect estimation."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import pandas as pd
import numpy as np
import warnings

try:
    from .utils import ensure_dir
except ImportError:  # pragma: no cover - fallback for direct execution
    from utils import ensure_dir

# Filter external library warnings when possible
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="dowhy.causal_estimators.regression_estimator",
)

try:
    from dowhy import CausalModel

    HAS_DOWHY = True
except ImportError:
    HAS_DOWHY = False

logger = logging.getLogger(__name__)

CONCEPT_GROUPS: Dict[str, List[str]] = {
    "attendance": ["absences", "failures"],
    "socio_economic_status": ["Medu", "Fedu", "traveltime"],
    "engagement": ["studytime", "goout", "Dalc", "Walc", "freetime"],
}


def group_concepts(X: pd.DataFrame) -> pd.DataFrame:
    """Aggregate features into broader concepts."""
    data = {}
    for concept, cols in CONCEPT_GROUPS.items():
        present = [c for c in cols if c in X.columns]
        if not present:
            continue
        data[concept] = X[present].mean(axis=1)
    return pd.DataFrame(data, index=X.index)


def estimate_concept_effects(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Estimate causal effect of concepts on outcome using DoWhy or fallback."""
    C = group_concepts(X)

    if not HAS_DOWHY:
        # Fallback to simple regression coefficients if DoWhy is not installed
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(C, y)
        effects = pd.DataFrame(
            {"concept": C.columns, "effect": np.abs(model.coef_[0])}
        ).sort_values("effect", ascending=False)
        return effects

    # With DoWhy, build a simple causal graph and estimate effects
    effects = []
    for concept in C.columns:
        data = pd.concat([C[[concept]], y.rename("outcome")], axis=1)
        # Use DOT format for the graph string
        graph = f'digraph {{ "{concept}" -> "outcome"; }}'
        try:
            model = CausalModel(
                data=data, treatment=concept, outcome="outcome", graph=graph
            )
            identified = model.identify_effect()
            estimate = model.estimate_effect(
                identified, method_name="backdoor.linear_regression"
            )
            effect_value = abs(estimate.value)
            effects.append({"concept": concept, "effect": effect_value})
        except Exception as e:
            logger.warning(
                "DoWhy failed for '%s': %s. Falling back to correlation.",
                concept,
                e,
            )
            effects.append({"concept": concept, "effect": C[concept].corr(y)})

    return pd.DataFrame(effects).sort_values("effect", ascending=False)


def export_concept_importance(effects: pd.DataFrame, out_dir: Path) -> Path:
    """Save concept importance to a CSV file."""
    ensure_dir(out_dir)
    path = out_dir / "concept_importance.csv"
    effects.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    # Use a relative import to correctly locate the 'data' module
    from .data import load_data
    from .logging_config import setup_logging

    setup_logging()
    X, y = load_data("student-mat.csv")
    concept_effects = estimate_concept_effects(X, y)
    logger.info("Estimated Concept Effects:")
    logger.info("%s", concept_effects)
    export_concept_importance(concept_effects, Path("reports"))
