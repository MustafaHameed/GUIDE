import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from scipy import stats
import shap
import matplotlib.pyplot as plt
import seaborn as sns


def load_regression_data(csv_path: str = "student-mat.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["G3"])
    y = df["G3"]
    return X, y, df


def feature_groups():
    demographic = [
        "school",
        "sex",
        "age",
        "address",
        "famsize",
        "Pstatus",
        "Medu",
        "Fedu",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
    ]
    behavioral = [
        "traveltime",
        "studytime",
        "failures",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
        "famrel",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "absences",
    ]
    return demographic, behavioral


def build_pipeline(X):
    numeric_features = X.select_dtypes(include="number").columns
    categorical_features = X.select_dtypes(exclude="number").columns
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = RandomForestRegressor(random_state=0)
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


def run_inner_cv(X_train, y_train, param_grid):
    pipe = build_pipeline(X_train)
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


def nested_cv(X, y, repeats=1):
    param_grid = {
        "select__k": ["all", 20],
        "model__n_estimators": [50],
        "model__max_depth": [None, 5],
    }
    outer_cv = RepeatedKFold(
        n_splits=10, n_repeats=repeats, random_state=0
    )
    results = []
    preds = []
    demographic, behavioral = feature_groups()
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # full model
        best_model = run_inner_cv(X_train, y_train, param_grid)
        y_pred = best_model.predict(X_test)
        metrics = evaluate_metrics(y_test, y_pred)
        results.append({"model": "full", "fold": fold, **metrics})
        preds.append(
            pd.DataFrame(
                {
                    "index": test_idx,
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "sex": X_test["sex"],
                    "school": X_test["school"],
                }
            )
        )

        # baselines
        mean_model = DummyRegressor(strategy="mean")
        mean_model.fit(X_train, y_train)
        y_mean = mean_model.predict(X_test)
        results.append({"model": "mean", "fold": fold, **evaluate_metrics(y_test, y_mean)})

        ols = LinearRegression()
        ols.fit(X_train[["G1", "G2"]], y_train)
        y_ols = ols.predict(X_test[["G1", "G2"]])
        results.append({"model": "ols_g1_g2", "fold": fold, **evaluate_metrics(y_test, y_ols)})

        y_last = X_test["G2"].values
        results.append({"model": "last_grade", "fold": fold, **evaluate_metrics(y_test, y_last)})

        # ablations
        X_train_no_g = X_train.drop(columns=["G1", "G2"])
        X_test_no_g = X_test.drop(columns=["G1", "G2"])
        best_no_g = run_inner_cv(X_train_no_g, y_train, param_grid)
        y_no_g = best_no_g.predict(X_test_no_g)
        results.append({"model": "no_g1_g2", "fold": fold, **evaluate_metrics(y_test, y_no_g)})

        behav_cols = X_train.columns.difference(behavioral)
        X_train_no_behav = X_train[behav_cols]
        X_test_no_behav = X_test[behav_cols]
        best_no_behav = run_inner_cv(X_train_no_behav, y_train, param_grid)
        y_no_behav = best_no_behav.predict(X_test_no_behav)
        results.append({"model": "no_behavior", "fold": fold, **evaluate_metrics(y_test, y_no_behav)})

        X_train_demo = X_train[demographic]
        X_test_demo = X_test[demographic]
        best_demo = run_inner_cv(X_train_demo, y_train, param_grid)
        y_demo = best_demo.predict(X_test_demo)
        results.append({"model": "demographic_only", "fold": fold, **evaluate_metrics(y_test, y_demo)})

    results_df = pd.DataFrame(results)
    preds_df = pd.concat(preds, ignore_index=True)
    return results_df, preds_df


def statistical_tests(results_df):
    models = results_df["model"].unique()
    base_model = "full"
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


def shap_analysis(best_model, X):
    preprocess = best_model.named_steps["preprocess"]
    select = best_model.named_steps["select"]
    X_pre = preprocess.transform(X)
    X_sel = select.transform(X_pre)
    feature_names = preprocess.get_feature_names_out()
    selected_features = feature_names[select.get_support()]
    explainer = shap.TreeExplainer(best_model.named_steps["model"])
    shap_values = explainer(X_sel)

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

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


def residual_plots(preds_df):
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    preds_df["residual"] = preds_df["y_true"] - preds_df["y_pred"]
    sns.boxplot(data=preds_df, x="sex", y="residual")
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals_by_sex.png")
    plt.close()

    sns.boxplot(data=preds_df, x="school", y="residual")
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals_by_school.png")
    plt.close()

    sns.regplot(data=preds_df, x="y_true", y="y_pred", lowess=True, line_kws={"color": "red"})
    plt.tight_layout()
    plt.savefig(fig_dir / "pred_vs_actual.png")
    plt.close()


def learning_curve_plot(best_model, X, y):
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
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


def main(csv_path: str = "student-mat.csv", repeats: int = 1):
    X, y, df = load_regression_data(csv_path)
    results_df, preds_df = nested_cv(X, y, repeats=repeats)

    table_dir = Path("tables")
    table_dir.mkdir(exist_ok=True)
    results_df.to_csv(table_dir / "nested_cv_metrics.csv", index=False)

    stats_df = statistical_tests(results_df)
    stats_df.to_csv(table_dir / "statistical_tests.csv", index=False)

    best_model = run_inner_cv(X, y, {
        "select__k": ["all", 20],
        "model__n_estimators": [50],
        "model__max_depth": [None, 5],
    })
    shap_analysis(best_model, X)
    residual_plots(preds_df)
    learning_curve_plot(best_model, X, y)


if __name__ == "__main__":
    main()
