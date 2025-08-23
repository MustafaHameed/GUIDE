"""Nested cross-validation for regression models.

References
----------
- Cawley and Talbot, 2010: https://jmlr.csail.mit.edu/papers/v11/cawley10a.html
"""

import logging

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import (
    RandomForestRegressor,
    BaggingRegressor,
    StackingRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from .utils import ensure_dir
except ImportError:  # pragma: no cover - fallback for direct execution
    from utils import ensure_dir
try:  # statistical tests
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multicomp import MultiComparison
    HAS_STATSMODELS = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    AnovaRM = MultiComparison = None  # type: ignore
    HAS_STATSMODELS = False

try:  # xgboost is optional
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore
    HAS_XGB = False

try:  # SHAP is optional and used only for interpretation plots
    import shap  # type: ignore
    HAS_SHAP = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    shap = None  # type: ignore
    HAS_SHAP = False


from logging_config import setup_logging

logger = logging.getLogger(__name__)

def load_regression_data(csv_path: str = "student-mat.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["G3"])
    y = df["G3"]
    return X, y, df


def build_pipeline(X, model=None):
    numeric_features = X.select_dtypes(include="number").columns
    categorical_features = X.select_dtypes(exclude="number").columns
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = RandomForestRegressor(random_state=0) if model is None else model
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("select", SelectKBest(score_func=f_regression, k="all")),
            ("model", model),
        ]
    )
    return pipe


def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_inner_cv(X_train, y_train, model, param_grid):
    pipe = build_pipeline(X_train, model=model)
    if param_grid:
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
        search = GridSearchCV(
            pipe,
            param_grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_
    pipe.fit(X_train, y_train)
    return pipe
        
def nested_cv(X, y, models, repeats=1):
    """Run nested cross-validation.

    References
    ----------
    - Cawley and Talbot, 2010: https://jmlr.csail.mit.edu/papers/v11/cawley10a.html
    """
    outer_cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=0)  # Nested CV reference above
    results: list[dict] = []
    preds: list[pd.DataFrame] = []
    for model_name, estimator, param_grid in models:
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            best_model = run_inner_cv(X_train, y_train, estimator, param_grid)
            y_pred = best_model.predict(X_test)
            metrics = evaluate_metrics(y_test, y_pred)
            results.append({"model": model_name, "fold": fold, **metrics})
            preds.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "fold": fold,
                        "index": test_idx,
                        "y_true": y_test,
                        "y_pred": y_pred,
                        "sex": X_test["sex"],
                        "school": X_test["school"],
                    }
                )
            )
    results_df = pd.DataFrame(results)
    preds_df = pd.concat(preds, ignore_index=True)
    return results_df, preds_df


def statistical_tests(results_df: pd.DataFrame, base_model: str):
    models = results_df["model"].unique()
    rows = []
    for m in models:
        if m == base_model:
            continue
        diffs = (
            results_df[results_df["model"] == base_model]["rmse"].values
            - results_df[results_df["model"] == m]["rmse"].values
        )
        if len(diffs) == 0:
            continue
        stat, p = stats.shapiro(diffs)
        if p > 0.05:
            t_stat, t_p = stats.ttest_rel(
                results_df[results_df["model"] == base_model]["rmse"],
                results_df[results_df["model"] == m]["rmse"],
            )
            effect = diffs.mean() / diffs.std(ddof=1)
            rows.append(
                {
                    "model": m,
                    "test": "paired_t",
                    "p_value": t_p,
                    "effect_size": effect,
                }
            )
        else:
            w_stat, w_p = stats.wilcoxon(diffs)
            n = len(diffs)
            rbc = 1 - 2 * w_stat / (n * (n + 1) / 2)
            rows.append(
                {
                    "model": m,
                    "test": "wilcoxon",
                    "p_value": w_p,
                    "effect_size": rbc,
                }
            )
    return pd.DataFrame(rows)


def anova_and_tukey(results_df: pd.DataFrame):
    """Aggregate metrics by model and run repeated-measures ANOVA and Tukey HSD.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`nested_cv` with columns ``model``, ``fold`` and ``rmse``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ANOVA table, Tukey HSD summary, and bootstrap confidence intervals for the
        mean RMSE of each model.
    """

    if not HAS_STATSMODELS:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("statsmodels is required for ANOVA and Tukey tests")

    # One-way repeated measures ANOVA with folds as the subject
    anova_res = AnovaRM(results_df, depvar="rmse", subject="fold", within=["model"]).fit()
    anova_table = anova_res.anova_table.reset_index().rename(columns={"index": "source"})

    # Tukey HSD post-hoc comparisons across models
    mc = MultiComparison(results_df["rmse"], results_df["model"])
    tukey_res = mc.tukeyhsd()
    tukey_table = tukey_res.summary()
    tukey_df = pd.DataFrame(tukey_table.data[1:], columns=tukey_table.data[0])

    # Bootstrap confidence intervals for each model's mean RMSE
    rng = np.random.default_rng(0)
    ci_rows = []
    for model, grp in results_df.groupby("model"):
        vals = grp["rmse"].values
        boot_means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(1000)]
        lower, upper = np.percentile(boot_means, [2.5, 97.5])
        ci_rows.append({"model": model, "mean": vals.mean(), "ci_lower": lower, "ci_upper": upper})
    ci_df = pd.DataFrame(ci_rows)

    return anova_table, tukey_df, ci_df


def shap_analysis(best_model, X):
    if not HAS_SHAP:
        logger.warning("shap is not installed; skipping SHAP analysis.")
        return

    preprocess = best_model.named_steps["preprocess"]
    select = best_model.named_steps["select"]
    X_pre = preprocess.transform(X)
    X_sel = select.transform(X_pre)
    feature_names = preprocess.get_feature_names_out()
    selected_features = feature_names[select.get_support()]
    explainer = shap.TreeExplainer(best_model.named_steps["model"])
    shap_values = explainer(X_sel)

    fig_dir = Path("figures")
    ensure_dir(fig_dir)

    shap.summary_plot(shap_values, features=X_sel, feature_names=selected_features, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png")
    plt.close()

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-5:][::-1]
    for idx in top_idx:
        shap.dependence_plot(
            idx,
            shap_values.values,
            X_sel,
            feature_names=selected_features,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(fig_dir / f"shap_dependence_{selected_features[idx]}.png")
        plt.close()



def residual_plots(preds_df, base_model):
    fig_dir = Path("figures")
    ensure_dir(fig_dir)
    df = preds_df[preds_df["model"] == base_model].copy()
    df["residual"] = df["y_true"] - df["y_pred"]
    sns.boxplot(data=df, x="sex", y="residual")
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals_by_sex.png")
    plt.close()

    sns.boxplot(data=df, x="school", y="residual")
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals_by_school.png")
    plt.close()

    try:
        sns.regplot(
            data=df,
            x="y_true",
            y="y_pred",
            lowess=True,
            line_kws={"color": "red"},
        )
    except RuntimeError:
        logger.warning("statsmodels not installed; falling back to simple linear fit.")
        sns.regplot(
            data=df,
            x="y_true",
            y="y_pred",
            line_kws={"color": "red"},
        )
    plt.tight_layout()
    plt.savefig(fig_dir / "pred_vs_actual.png")
    plt.close()


def learning_curve_plot(best_model, X, y):
    fig_dir = Path("figures")
    ensure_dir(fig_dir)
    train_sizes, train_scores, test_scores = learning_curve(
        best_model,
        X,
        y,
        cv=5,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )
    plt.plot(train_sizes, -train_scores.mean(axis=1), label="train")
    plt.plot(train_sizes, -test_scores.mean(axis=1), label="validation")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "learning_curve.png")
    plt.close()


def main(csv_path: str = "student-mat.csv", repeats: int = 1, models=None):
    X, y, _ = load_regression_data(csv_path)
    if models is None:
        models = [
            (
                "random_forest",
                RandomForestRegressor(random_state=0),
                {
                    "select__k": ["all", 20],
                    "model__n_estimators": [50],
                    "model__max_depth": [None, 5],
                },
            ),
            (
                "linear_regression",
                LinearRegression(),
                {"select__k": ["all", 20]},
            ),
            (
                "ridge",
                Ridge(),
                {
                    "select__k": ["all", 20],
                    "model__alpha": [0.1, 1.0, 10.0],
                },
            ),
            (
                "lasso",
                Lasso(random_state=0),
                {
                    "select__k": ["all", 20],
                    "model__alpha": [0.1, 1.0, 10.0],
                },
            ),
            (
                "svr",
                SVR(),
                {
                    "select__k": ["all", 20],
                    "model__C": [1.0, 10.0],
                    "model__kernel": ["rbf", "linear"],
                },
            ),
            (
                "knn",
                KNeighborsRegressor(),
                {
                    "select__k": ["all", 20],
                    "model__n_neighbors": [5, 10],
                    "model__weights": ["uniform", "distance"],
                },
            ),
            (
                "bagging",
                BaggingRegressor(
                    estimator=DecisionTreeRegressor(random_state=0),
                    random_state=0,
                ),
                {
                    "select__k": ["all", 20],
                    "model__n_estimators": [10, 20],
                    "model__estimator__max_depth": [None, 5],
                },
            ),
            (
                "gradient_boosting",
                GradientBoostingRegressor(random_state=0),
                {
                    "select__k": ["all", 20],
                    "model__n_estimators": [100, 200],
                    "model__learning_rate": [0.1, 0.01],
                    "model__max_depth": [3, 5],
                },
            ),
            (
                "mlp",
                MLPRegressor(random_state=0),
                {
                    "select__k": ["all", 20],
                    "model__hidden_layer_sizes": [(50,), (100,)],
                    "model__alpha": [0.0001, 0.001],
                },
            ),
            (
                "stacking",
                StackingRegressor(
                    estimators=[
                        ("rf", RandomForestRegressor(random_state=0)),
                        ("knn", KNeighborsRegressor()),
                    ],
                    final_estimator=LinearRegression(),
                    n_jobs=-1,
                ),
                {"select__k": ["all", 20]},
            ),            
        ]
        if HAS_XGB:
            models.append(
                (
                    "xgb",
                    XGBRegressor(random_state=0, verbosity=0),
                    {
                        "select__k": ["all", 20],
                        "model__n_estimators": [50],
                        "model__max_depth": [3, 5],
                        "model__learning_rate": [0.1, 0.01],
                    },
                )
            )        
    results_df, preds_df = nested_cv(X, y, models=models, repeats=repeats)

    table_dir = Path("tables")
    ensure_dir(table_dir)
    results_df.to_csv(table_dir / "nested_cv_regression_metrics.csv", index=False)

    base_model = models[0][0]
    stats_df = statistical_tests(results_df, base_model)
    stats_df.to_csv(table_dir / "statistical_tests.csv", index=False)
    # Repeated-measures ANOVA, Tukey HSD, and bootstrap confidence intervals
    try:
        anova_df, tukey_df, ci_df = anova_and_tukey(results_df)
        anova_df.to_csv(table_dir / "rmse_anova.csv", index=False)
        tukey_df.to_csv(table_dir / "rmse_tukey_hsd.csv", index=False)
        ci_df.to_csv(table_dir / "rmse_bootstrap_ci.csv", index=False)
    except ModuleNotFoundError:
        anova_df = tukey_df = ci_df = None


    # Comparison visualization
    fig_dir = Path("figures")
    ensure_dir(fig_dir)
    order = results_df["model"].unique()
    sns.boxplot(data=results_df, x="model", y="rmse", order=order)
    if ci_df is not None:
        for i, model in enumerate(order):
            row = ci_df[ci_df["model"] == model].iloc[0]
            err = [[row["mean"] - row["ci_lower"]], [row["ci_upper"] - row["mean"]]]
            plt.errorbar(i, row["mean"], yerr=err, fmt="o", color="black", capsize=5)
    if anova_df is not None:
        p_val = anova_df.loc[anova_df["source"] == "model", "Pr > F"].iloc[0]
        plt.figtext(0.99, 0.01, f"ANOVA p={p_val:.3f}", ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(fig_dir / "model_rmse_boxplot.png")
    plt.close()

    best_model = run_inner_cv(X, y, models[0][1], models[0][2])
    shap_analysis(best_model, X)
    residual_plots(preds_df, base_model)
    learning_curve_plot(best_model, X, y)


if __name__ == "__main__":
    setup_logging()
    main()
