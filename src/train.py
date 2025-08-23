"""Training utilities for building, evaluating, and explaining models."""

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    precision_score,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
import itertools
import logging


def _configure_warnings() -> None:
    """Suppress common warnings for cleaner output."""
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    warnings.filterwarnings("ignore", category=Warning, module="lightgbm")


_configure_warnings()

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

try:
    import dice_ml
except ImportError:
    dice_ml = None


logger = logging.getLogger(__name__)

def positive_predictive_value(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def _compute_fairness_tables(
    y_true: pd.Series,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    group_cols: list[str],
    report_dir: Path,
    suffix: str,
    baseline: dict[str, pd.DataFrame] | None = None,
    fig_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute DP and EO tables for each sensitive column and their intersections."""
    results: dict[str, pd.DataFrame] = {}
    overall_tpr = true_positive_rate(y_true, y_pred)
    overall_fpr = false_positive_rate(y_true, y_pred)
    
    # Per-column fairness
    for col in group_cols:
        if col not in X.columns:
            continue
        mf = MetricFrame(
            metrics={
                "demographic_parity": selection_rate,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=X[col],
        )
        records: list[dict[str, float | str]] = []
        for group_value, row in mf.by_group.iterrows():
            eo = max(
                abs(row["true_positive_rate"] - overall_tpr),
                abs(row["false_positive_rate"] - overall_fpr),
            )
            records.append(
                {
                    col: group_value,
                    "demographic_parity": row["demographic_parity"],
                    "equalized_odds": eo,
                }
            )
        df = pd.DataFrame(records)
        if baseline and col in baseline:
            df = df.merge(baseline[col], on=col, suffixes=("_post", "_pre"))
            df["dp_delta"] = df["demographic_parity_post"] - df["demographic_parity_pre"]
            df["eo_delta"] = df["equalized_odds_post"] - df["equalized_odds_pre"]
            df = df[
                [
                    col,
                    "demographic_parity_pre",
                    "demographic_parity_post",
                    "dp_delta",
                    "equalized_odds_pre",
                    "equalized_odds_post",
                    "eo_delta",
                ]
            ]
        df.to_csv(report_dir / f"fairness_{col}_{suffix}.csv", index=False)
        if fig_dir is not None:
            # Determine which fairness metric columns are present
            fairness_metrics = []
            for metric in ["demographic_parity", "equalized_odds"]:
                if metric in df.columns:
                    fairness_metrics.append(metric)
                elif f"{metric}_post" in df.columns:
                    fairness_metrics.append(f"{metric}_post")
            
            if fairness_metrics:
                plot_df = df.melt(id_vars=[col], value_vars=fairness_metrics,
                                  var_name="metric", value_name="value")
                sns.barplot(data=plot_df, x=col, y="value", hue="metric")
                plt.tight_layout()
                plt.savefig(fig_dir / f"fairness_{col}_{suffix}.png")
                plt.close()
            else:
                logger.warning(
                    "Skipping fairness plot for %s due to missing metric columns", col
                )
        results[col] = df

    # Intersectional fairness for combinations of columns
    if len(group_cols) > 1:
        for col_a, col_b in itertools.product(group_cols, group_cols):
            if group_cols.index(col_a) >= group_cols.index(col_b):
                continue
            pair_name = f"{col_a}_{col_b}"
            sens_df = X[[col_a, col_b]]
            mf = MetricFrame(
                metrics={
                    "demographic_parity": selection_rate,
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sens_df,
            )
            records = []
            for (val_a, val_b), row in mf.by_group.iterrows():
                eo = max(
                    abs(row["true_positive_rate"] - overall_tpr),
                    abs(row["false_positive_rate"] - overall_fpr),
                )
                records.append(
                    {
                        col_a: val_a,
                        col_b: val_b,
                        "demographic_parity": row["demographic_parity"],
                        "equalized_odds": eo,
                    }
                )
            df = pd.DataFrame(records)
            if baseline and pair_name in baseline:
                df = df.merge(baseline[pair_name], on=[col_a, col_b], suffixes=("_post", "_pre"))
                df["dp_delta"] = df["demographic_parity_post"] - df["demographic_parity_pre"]
                df["eo_delta"] = df["equalized_odds_post"] - df["equalized_odds_pre"]
                df = df[
                    [
                        col_a,
                        col_b,
                        "demographic_parity_pre",
                        "demographic_parity_post",
                        "dp_delta",
                        "equalized_odds_pre",
                        "equalized_odds_post",
                        "eo_delta",
                    ]
                ]
            df.to_csv(report_dir / f"fairness_{pair_name}_{suffix}.csv", index=False)
            if fig_dir is not None:
                id_col = df[col_a].astype(str) + "_" + df[col_b].astype(str)
                plot_df = df.assign(**{pair_name: id_col})
                plot_df = plot_df.melt(
                    id_vars=[pair_name],
                    value_vars=["demographic_parity", "equalized_odds"],
                    var_name="metric",
                    value_name="value",
                )
                sns.barplot(data=plot_df, x=pair_name, y="value", hue="metric")
                plt.tight_layout()
                plt.savefig(fig_dir / f"fairness_{pair_name}_{suffix}.png")
                plt.close()
            results[pair_name] = df

    return results

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing

from .data import load_data
from .preprocessing import build_pipeline
from .sequence_models import evaluate_sequence_model
from .uncertainty.conformal import run_conformal_prediction


PARAM_GRIDS: dict[str, dict[str, dict]] = {
    "logistic": {
        "default": {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        }
    },
    "random_forest": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
        }
    },
    "gradient_boosting": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
        }
    },
    "svm": {
        "default": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        }
    },
    "knn": {
        "default": {
            "model__n_neighbors": [5, 10, 15],
            "model__weights": ["uniform", "distance"],
        }
    },
    "naive_bayes": {
        "default": {
            "model__var_smoothing": [1e-09, 1e-08, 1e-07],
        }
    },
    "extra_trees": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
        }
    },
        "mlp": {
        "default": {
            "model__hidden_layer_sizes": [(50,), (100,)],
            "model__alpha": [0.0001, 0.001],
            "model__learning_rate_init": [0.001, 0.01],
        }
    },
    "xgboost": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
        }
    },
    "lightgbm": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [-1, 5, 10],
            "model__learning_rate": [0.05, 0.1],
        }
    },
    "bagging": {
        "default": {
            "model__n_estimators": [10, 50],
            "model__max_samples": [0.5, 1.0],
        }
    },
    "stacking": {
        "default": {
            "model__final_estimator": [
                LogisticRegression(max_iter=1000),
                RandomForestClassifier(),
            ],
            "model__passthrough": [False, True],
        }
    },
}

REGRESSION_PARAM_GRIDS: dict[str, dict[str, dict]] = {
    "linear": {"default": {}},
    "random_forest": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
        }
    },
    "gradient_boosting": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
        }
    },
    "svm": {
        "default": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        }
    },
    "knn": {
        "default": {
            "model__n_neighbors": [5, 10, 15],
            "model__weights": ["uniform", "distance"],
        }
    },
    "mlp": {
        "default": {
            "model__hidden_layer_sizes": [(50,), (100,)],
            "model__alpha": [0.0001, 0.001],
            "model__learning_rate_init": [0.001, 0.01],
        }
    },
    "bagging": {
        "default": {
            "model__n_estimators": [10, 50],
            "model__max_samples": [0.5, 1.0],
        }
    },
    "stacking": {
        "default": {
            "model__final_estimator": [
                LinearRegression(),
                RandomForestRegressor(),
            ],
            "model__passthrough": [False, True],
        }
    },
}
def main(
    csv_path: str = "student-mat.csv",
    pass_threshold: int = 10,
    group_cols: list[str] | None = None,
    model_type: str = "logistic",
    param_grid: str = "none",
    estimators: list[str] | None = None,
    final_estimator: str = "logistic",
    base_estimator: str = "decision_tree",
    sequence_model: str | None = None,
    hidden_size: int = 8,
    epochs: int = 50,
    learning_rate: float = 0.01,
    mitigation: str = "none",
    task: str = "classification",
):
    """Train model and generate evaluation artifacts.

    Parameters
    ----------
    csv_path : str, default 'student-mat.csv'
        Path to the input CSV file.
    pass_threshold : int, default 10
        Minimum ``G3`` grade considered a passing score.
    group_cols : list[str] | None, optional
        Demographic columns to compute group-level metrics for. If ``None``,
        only overall metrics are produced.
    model_type : str, default "logistic"
        Type of model to train.
    param_grid : str, default "none"
        Preset name for the hyperparameter grid. Use "none" to skip
        hyperparameter tuning.
    estimators : list[str] | None, optional
        Base estimators for stacking models.
    final_estimator : str, default "logistic"
        Meta learner for stacking models.
    base_estimator : str, default "decision_tree"
        Base estimator used by bagging models.
    sequence_model : str | None, optional
        If provided, trains a sequence model (``'rnn'`` or ``'hmm'``) on
        grade sequences instead of a standard tabular model.
    hidden_size : int, default 8
        Hidden dimension for the RNN when ``sequence_model='rnn'``.
    epochs : int, default 50
        Number of training epochs for the RNN.
    learning_rate : float, default 0.01
        Learning rate for the RNN optimizer.
    mitigation : str, default "none"
        Fairness mitigation strategy to apply (``'demographic_parity'``,
        ``'equalized_odds'``, ``'reweighing'`` or ``'adversarial'``).
        Requires ``group_cols``.
    task : str, default "classification"
        ``"classification"`` for pass/fail prediction or ``"regression"`` to
        predict the raw ``G3`` grade.
    """
    if sequence_model:
        evaluate_sequence_model(
            csv_path,
            model_type=sequence_model,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        return
    
    if task == "regression":
        X, y = load_data(csv_path, task="regression")
        model_params: dict | None = None
        if model_type == "stacking":
            model_params = {
                "estimators": estimators or ["linear", "random_forest"],
                "final_estimator": final_estimator,
            }
        elif model_type == "bagging":
            model_params = {"base_estimator": base_estimator}
        pipeline = build_pipeline(
            X, model_type=model_type, model_params=model_params, task="regression"
        )
    else:
        X, y = load_data(
            csv_path, pass_threshold=pass_threshold, task="classification"
        )
        model_params: dict | None = None
        if model_type == "stacking":
            model_params = {
                "estimators": estimators or ["logistic", "random_forest"],
                "final_estimator": final_estimator,
            }
        elif model_type == "bagging":
            model_params = {"base_estimator": base_estimator}
        pipeline = build_pipeline(
            X, model_type=model_type, model_params=model_params, task="classification"
        )

    # Prepare output directories
    fig_dir = Path('figures')
    fig_dir.mkdir(exist_ok=True)
    report_dir = Path('reports')
    report_dir.mkdir(exist_ok=True)
    table_dir = Path('tables')
    table_dir.mkdir(exist_ok=True)

    # Hold-out evaluation with validation split for calibration
    if task == "regression":
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )
        grid_dict = REGRESSION_PARAM_GRIDS
        scoring = "neg_mean_squared_error"
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )
        grid_dict = PARAM_GRIDS
        scoring = "f1"

    grid = grid_dict.get(model_type, {}).get(param_grid)

    train_bld = test_bld = None
    if group_cols:
        train_df_bld = X_train[group_cols].copy()
        test_df_bld = X_test[group_cols].copy()
        train_df_bld["label"] = y_train.values
        test_df_bld["label"] = y_test.values
        for col in group_cols:
            train_df_bld[col] = pd.Categorical(train_df_bld[col]).codes
            test_df_bld[col] = pd.Categorical(test_df_bld[col]).codes
        train_bld = BinaryLabelDataset(
            df=train_df_bld,
            label_names=["label"],
            protected_attribute_names=group_cols,
        )
        test_bld = BinaryLabelDataset(
            df=test_df_bld,
            label_names=["label"],
            protected_attribute_names=group_cols,
        )

    best_params: dict | None = None
    best_score: float | None = None

    if grid:
        search = GridSearchCV(pipeline, grid, cv=5, scoring=scoring)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = model.named_steps["model"].get_params()
        best_score = search.best_score_
        logger.info(
            "Best params from search: %s (score=%.3f)", best_params, best_score
        )
    else:
        model = pipeline
        model.fit(X_train, y_train)
        best_params = model.named_steps["model"].get_params()

    if task == "regression":
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info("RMSE: %.3f\nMAE: %.3f\nR^2: %.3f", rmse, mae, r2)
        pd.DataFrame({"rmse": [rmse], "mae": [mae], "r2": [r2]}).to_csv(
            report_dir / "regression_metrics.csv", index=False
        )
        best_params_df = pd.DataFrame([best_params or {}])
        best_params_df.insert(0, "best_score", best_score)
        best_params_df.to_csv(report_dir / "best_params.csv", index=False)
        return
    # Keep original fitted pipeline for explanations even if mitigation wraps it
    fitted_pipeline = model
    # Predictions before mitigation
    pre_y_pred = model.predict(X_test)
    pre_y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )
    pre_fairness: dict[str, pd.DataFrame] = {}
    if group_cols:
        pre_fairness = _compute_fairness_tables(
            y_test, pre_y_pred, X_test, group_cols, report_dir, "pre", fig_dir=fig_dir
        )

    y_pred = pre_y_pred
    y_prob = pre_y_prob

    if mitigation != "none" and group_cols:
        if mitigation in ["demographic_parity", "equalized_odds"]:

            sens_train = X_train[group_cols[0]]
            sens_test = X_test[group_cols[0]]
            mitigator = ThresholdOptimizer(
                estimator=model, constraints=mitigation, prefit=True
            )
            mitigator.fit(X_train, y_train, sensitive_features=sens_train)
            y_pred = mitigator.predict(X_test, sensitive_features=sens_test)

            y_prob = (
                mitigator.predict_proba(X_test, sensitive_features=sens_test)[:, 1]
                if hasattr(mitigator, "predict_proba")
                else None
            )
            model = mitigator
        elif mitigation == "reweighing" and train_bld is not None:
            from sklearn.base import clone

            rw = Reweighing()
            rw_train = rw.fit_transform(train_bld)
            model = clone(model)
            model.fit(
                X_train,
                y_train,
                model__sample_weight=rw_train.instance_weights,
            )
            y_pred = model.predict(X_test)
            y_prob = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )
        elif mitigation == "adversarial" and train_bld is not None and test_bld is not None:
            import tensorflow as tf

            sess = tf.compat.v1.Session()
            priv = [{group_cols[0]: 1}]
            unpriv = [{group_cols[0]: 0}]
            adv = AdversarialDebiasing(
                privileged_groups=priv,
                unprivileged_groups=unpriv,
                scope_name="adv_debias",
                debias=True,
                sess=sess,
            )
            adv.fit(train_bld)
            pred_ds = adv.predict(test_bld)
            y_pred = pred_ds.labels.ravel()
            y_prob = getattr(pred_ds, "scores", None)
            model = adv
        else:
            logger.warning(
                "Mitigation requested but no group column provided. Proceeding without mitigation."
            )

    if group_cols:
        _compute_fairness_tables(
            y_test,
            y_pred,
            X_test,
            group_cols,
            report_dir,
            "post",
            baseline=pre_fairness,
            fig_dir=fig_dir,
        )
    logger.info(
        "Hold-out classification report:\n%s", classification_report(y_test, y_pred)
    )

    # Export classification report as a table
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        report_dir / 'classification_report.csv', index=True
    )

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(fig_dir / 'confusion_matrix.png')
    plt.close()

    # ROC curve visualization (requires probability estimates)
    if y_prob is not None:
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.tight_layout()
        plt.savefig(fig_dir / 'roc_curve.png')
        plt.close()

        # Precision-Recall analysis
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        precision_curve = precision[:-1]
        recall_curve = recall[:-1]
        f1_scores = 2 * precision_curve * recall_curve / (
            precision_curve + recall_curve + 1e-8
        )
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        y_pred_pr = (y_prob >= best_threshold).astype(int)

        # Bootstrap confidence bands
        n_boot = 100
        rng = np.random.default_rng(42)
        boot_prec_curve = []
        boot_rec_curve = []
        boot_prec_best = []
        boot_rec_best = []
        boot_f1_best = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_prob), len(y_prob))
            y_true_b = y_test.values[idx]
            y_prob_b = y_prob[idx]
            p_b, r_b, t_b = precision_recall_curve(y_true_b, y_prob_b)
            p_interp = np.interp(thresholds, t_b, p_b[:-1], left=p_b[:-1][0], right=p_b[:-1][-1])
            r_interp = np.interp(thresholds, t_b, r_b[:-1], left=r_b[:-1][0], right=r_b[:-1][-1])
            boot_prec_curve.append(p_interp)
            boot_rec_curve.append(r_interp)
            boot_prec_best.append(np.interp(best_threshold, t_b, p_b[:-1], left=p_b[:-1][0], right=p_b[:-1][-1]))
            boot_rec_best.append(np.interp(best_threshold, t_b, r_b[:-1], left=r_b[:-1][0], right=r_b[:-1][-1]))
            pb = boot_prec_best[-1]
            rb = boot_rec_best[-1]
            boot_f1_best.append(2 * pb * rb / (pb + rb + 1e-8))

        boot_prec_curve = np.array(boot_prec_curve)
        boot_rec_curve = np.array(boot_rec_curve)
        prec_mean = boot_prec_curve.mean(axis=0)
        prec_low = np.percentile(boot_prec_curve, 2.5, axis=0)
        prec_high = np.percentile(boot_prec_curve, 97.5, axis=0)
        rec_mean = boot_rec_curve.mean(axis=0)

        # Plot PR curve with confidence band and optimal threshold
        plt.figure()
        plt.plot(recall_curve, precision_curve, label="PR curve")
        plt.fill_between(rec_mean, prec_low, prec_high, color="lightblue", alpha=0.4, label="95% CI")
        plt.scatter(
            recall_curve[best_idx],
            precision_curve[best_idx],
            color="red",
            label=f"Best thr={best_threshold:.2f}",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "pr_curve.png")
        plt.close()

        # Save threshold tuning metrics
        prec_best_arr = np.array(boot_prec_best)
        rec_best_arr = np.array(boot_rec_best)
        f1_best_arr = np.array(boot_f1_best)
        threshold_df = pd.DataFrame(
            {
                "threshold": [best_threshold],
                "precision_mean": [prec_best_arr.mean()],
                "precision_ci_lower": [np.percentile(prec_best_arr, 2.5)],
                "precision_ci_upper": [np.percentile(prec_best_arr, 97.5)],
                "recall_mean": [rec_best_arr.mean()],
                "recall_ci_lower": [np.percentile(rec_best_arr, 2.5)],
                "recall_ci_upper": [np.percentile(rec_best_arr, 97.5)],
                "f1_mean": [f1_best_arr.mean()],
                "f1_ci_lower": [np.percentile(f1_best_arr, 2.5)],
                "f1_ci_upper": [np.percentile(f1_best_arr, 97.5)],
            }
        )
        threshold_df.to_csv(table_dir / "threshold_tuning.csv", index=False)

        # Conformal prediction to generate prediction sets
        alpha = 0.1
        if mitigation in ["demographic_parity", "equalized_odds"] and group_cols:
            y_prob_val = model.predict_proba(
                X_val, sensitive_features=X_val[group_cols[0]]
            )[:, 1]
        else:
            y_prob_val = model.predict_proba(X_val)[:, 1]
        groups_test = X_test[group_cols[0]].values if group_cols else None
        conformal = run_conformal_prediction(
            y_val.values,
            y_prob_val,
            y_test.values,
            y_prob,
            groups_test=groups_test,
            alphas=[alpha],
        )[alpha]

        pd.DataFrame([conformal["overall"]]).to_csv(
            table_dir / f"conformal_overall_alpha_{alpha}.csv", index=False
        )
        if conformal["by_group"] is not None:
            conformal["by_group"].to_csv(
                table_dir
                / f"conformal_by_{group_cols[0]}_alpha_{alpha}.csv",
                index=False,
            )
            sns.barplot(
                data=conformal["by_group"], x="group", y="coverage"
            )
            plt.axhline(
                conformal["overall"]["target_coverage"],
                color="red",
                linestyle="--",
                label="target",
            )
            plt.ylabel("coverage")
            plt.tight_layout()
            plt.savefig(
                fig_dir
                / f"conformal_coverage_by_{group_cols[0]}_alpha_{alpha}.png"
            )
            plt.close()

        size_counts = (
            pd.Series(conformal["set_sizes"]).value_counts().sort_index().reset_index()
        )
        size_counts.columns = ["set_size", "count"]
        size_counts.to_csv(
            table_dir / f"conformal_set_size_dist_alpha_{alpha}.csv", index=False
        )
        sns.barplot(data=size_counts, x="set_size", y="count")
        plt.tight_layout()
        plt.savefig(
            fig_dir / f"conformal_set_size_dist_alpha_{alpha}.png"
        )
        plt.close()


    # Export best parameters and search metrics
    best_params_df = pd.DataFrame([best_params or {}])
    best_params_df.insert(0, "best_score", best_score)
    best_params_df.to_csv(report_dir / "best_params.csv", index=False)

    # Per-group evaluations
    if group_cols:
        overall_tpr = true_positive_rate(y_test, y_pred)
        overall_fpr = false_positive_rate(y_test, y_pred)
        for col in group_cols:
            if col not in X_test.columns:
                logger.warning("Column '%s' not in dataset. Skipping.", col)
                continue
            # Per-group classification artifacts
            for group_value in X_test[col].unique():
                mask = X_test[col] == group_value
                y_true_g = y_test[mask]
                y_pred_g = y_pred[mask]
                y_prob_g = y_prob[mask] if y_prob is not None else None
                if y_true_g.nunique() < 2:
                    logger.warning(
                        "Skipping group %s=%s due to single class in y_true.",
                        col,
                        group_value,
                    )
                    continue

                grp_report = classification_report(
                    y_true_g, y_pred_g, output_dict=True
                )
                pd.DataFrame(grp_report).transpose().to_csv(
                    report_dir
                    / f"classification_report_{col}_{group_value}.csv",
                    index=True,
                )

                cm_g = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1])
                plt.figure(figsize=(4, 4))
                sns.heatmap(cm_g, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig(
                    fig_dir / f"confusion_matrix_{col}_{group_value}.png"
                )
                plt.close()

                if y_prob_g is not None:
                    RocCurveDisplay.from_predictions(y_true_g, y_prob_g)
                    plt.tight_layout()
                    plt.savefig(fig_dir / f"roc_curve_{col}_{group_value}.png")
                    plt.close()


    # LIME explanations for selected samples
    try:
        preprocessor = fitted_pipeline.named_steps["preprocess"]
        train_trans = preprocessor.transform(X_train)
        if hasattr(train_trans, "toarray"):
            train_trans = train_trans.toarray()
        explainer = LimeTabularExplainer(
            train_trans,
            feature_names=preprocessor.get_feature_names_out().tolist(),
            class_names=["negative", "positive"],
            mode="classification",
        )
        feature_names = preprocessor.get_feature_names_out().tolist()
        train_df = pd.DataFrame(train_trans, columns=feature_names)
        train_df["target"] = y_train.values
        dice_data = dice_ml.Data(
            dataframe=train_df,
            continuous_features=feature_names,
            outcome_name="target",
        )
        if dice_ml is not None:
            dice_model = dice_ml.Model(
                model=fitted_pipeline.named_steps["model"], backend="sklearn"
            )
            dice = dice_ml.Dice(dice_data, dice_model, method="random")
        else:
            dice = None
        test_trans = preprocessor.transform(X_test)
        if hasattr(test_trans, "toarray"):
            test_trans = test_trans.toarray()
        mis_idx = np.where(y_test != y_pred)[0]
        if mis_idx.size == 0:
            sample_idx = np.random.choice(
                len(test_trans), size=min(3, len(test_trans)), replace=False
            )
        else:
            sample_idx = mis_idx[: min(3, mis_idx.size)]
        predict_fn = fitted_pipeline.named_steps["model"].predict_proba
        for idx in sample_idx:
            exp = explainer.explain_instance(
                test_trans[idx],
                predict_fn,
                num_features=5,
            )
            exp.save_to_file(fig_dir / f"lime_{idx}.html")
            fig = exp.as_pyplot_figure()
            fig.savefig(fig_dir / f"lime_{idx}.png")
            plt.close(fig)
            try:
                query_df = pd.DataFrame(
                    [test_trans[idx]], columns=feature_names
                )
                if dice is not None:
                    cf = dice.generate_counterfactuals(
                        query_df, total_CFs=1, desired_class="opposite"
                )
                    cf.cf_examples_list[0].final_cfs_df.to_csv(
                        report_dir / f"counterfactual_{idx}.csv", index=False
                    )
            except Exception as e:
                logger.warning(
                    "Counterfactual generation failed for index %s: %s", idx, e
                )
    except Exception as e:
        logger.warning("Skipping LIME explanations due to error: %s", e)

    # Feature importance
    fi_csv = report_dir / "feature_importance.csv"
    fi_fig = fig_dir / "feature_importance.png"
    try:
        import shap

        explainer = shap.Explainer(fitted_pipeline, X_train)
        shap_values = explainer(X_train)
        importance = np.abs(shap_values.values).mean(axis=0)
        shap.summary_plot(shap_values, X_train, show=False)
        plt.tight_layout()
        plt.savefig(fi_fig)
        plt.close()
        importance_df = (
            pd.DataFrame(
                {"feature": X_train.columns, "importance": importance}
            ).sort_values("importance", ascending=False)
        )
        importance_df.to_csv(fi_csv, index=False)
    except Exception:
        result = permutation_importance(
            fitted_pipeline, X_test, y_test, n_repeats=10, random_state=42
        )
        importance_df = (
            pd.DataFrame({
                "feature": X_test.columns,
                "importance": result.importances_mean,
            })
            .sort_values("importance", ascending=False)
        )
        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.tight_layout()
        plt.savefig(fi_fig)
        plt.close()
        importance_df.to_csv(fi_csv, index=False)

    # Partial dependence plots for top features
    top_n = min(3, len(importance_df))
    for feat in importance_df["feature"].head(top_n):
        safe_name = str(feat).replace(" ", "_")
        try:
            PartialDependenceDisplay.from_estimator(
                fitted_pipeline, X_train, [feat], kind="average"
            )
            plt.tight_layout()
            plt.savefig(fig_dir / f"pdp_{safe_name}.png")
            plt.close()

            PartialDependenceDisplay.from_estimator(
                fitted_pipeline, X_train, [feat], kind="individual"
            )
            plt.tight_layout()
            plt.savefig(fig_dir / f"ice_{safe_name}.png")
            plt.close()
        except Exception as e:
            logger.warning("Skipping PDP for %s due to error: %s", feat, e)
    # Cross-validation across all models
    table_dir = Path("tables")
    table_dir.mkdir(exist_ok=True)
    performance: list[dict[str, float | str]] = []
    for m in PARAM_GRIDS.keys():
        try:
            cv_model = build_pipeline(X, model_type=m)
            scores = cross_validate(cv_model, X, y, cv=5, scoring=["accuracy", "f1"])
            performance.append(
                {
                    "model_type": m,
                    "accuracy_mean": scores["test_accuracy"].mean(),
                    "accuracy_std": scores["test_accuracy"].std(),
                    "f1_mean": scores["test_f1"].mean(),
                    "f1_std": scores["test_f1"].std(),
                }
            )
        except Exception as e:
            logger.warning("Skipping %s due to error: %s", m, e)
    perf_df = pd.DataFrame(performance)
    perf_df.to_csv(table_dir / "model_performance.csv", index=False)

    # Visualization of model comparison
    if not perf_df.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=perf_df, x="model_type", y="f1_mean")
        plt.ylabel("Mean F1")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "model_performance.png")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model with fairness evaluation')
    parser.add_argument('--csv-path', default='student-mat.csv')
    parser.add_argument('--pass-threshold', type=int, default=10,
                        help='Minimum G3 grade considered a passing score')    
    parser.add_argument(
        '--group-cols',
        nargs='*',
        default=None,
        help='Demographic columns to evaluate',
    )
    parser.add_argument(
        '--model-type',
        choices=list(set(PARAM_GRIDS.keys()) | set(REGRESSION_PARAM_GRIDS.keys())),
        default='logistic',
        help='Type of model to train',
    )
    parser.add_argument(
        '--param-grid',
        choices=['none', 'default'],
        default='none',
        help='Preset hyperparameter grid to use',
    )
    parser.add_argument(
        '--estimators',
        nargs='*',
        default=None,
        help='Base estimators for stacking models',
    )
    parser.add_argument(
        '--final-estimator',
        default='logistic',
        help='Final estimator for stacking models',
    )
    parser.add_argument(
        '--base-estimator',
        default='decision_tree',
        help='Base estimator for bagging models',
    )
    parser.add_argument(
        '--sequence-model',
        choices=['rnn', 'hmm'],
        default=None,
        help='Train a sequence model on grades G1 and G2',
    )
    parser.add_argument(
        '--hidden-size', type=int, default=8, help='RNN hidden layer size'
    )
    parser.add_argument(
        '--epochs', type=int, default=50, help='Number of RNN training epochs'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.01, help='RNN learning rate'
    )   
    parser.add_argument(
        '--mitigation',
        choices=['none', 'demographic_parity', 'equalized_odds', 'reweighing', 'adversarial'],
        default='none',
        help='Fairness mitigation strategy to apply',
    )
    parser.add_argument(
        '--task',
        choices=['classification', 'regression'],
        default='classification',
        help='Prediction task to run',
    )
    args = parser.parse_args()
    if args.task == 'regression' and args.model_type == 'logistic':
        args.model_type = 'linear'
    main(
        csv_path=args.csv_path,
        group_cols=args.group_cols,
        model_type=args.model_type,
        param_grid=args.param_grid,
        estimators=args.estimators,
        final_estimator=args.final_estimator,
        base_estimator=args.base_estimator,
        sequence_model=args.sequence_model,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mitigation=args.mitigation,
        pass_threshold=args.pass_threshold,
        task=args.task,
    )