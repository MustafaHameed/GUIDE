from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    precision_score,
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer
try:
    import dice_ml
except ImportError:
    dice_ml = None

def positive_predictive_value(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer

from .data import load_data
from .preprocessing import build_pipeline
from .sequence_models import evaluate_sequence_model


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
def main(
    csv_path: str = "student-mat.csv",
    group_cols: list[str] | None = None,
    model_type: str = "logistic",
    param_grid: str = "none",
    estimators: list[str] | None = None,
    final_estimator: str = "logistic",
    base_estimator: str = "decision_tree",
    sequence_model: str | None = None,
    mitigation: str = "none",

):
    """Train model and generate evaluation artifacts.

    Parameters
    ----------
    csv_path : str, default 'student-mat.csv'
        Path to the input CSV file.
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
    mitigation : str, default "none"
        Fairness mitigation strategy to apply (``'demographic_parity'`` or
        ``'equalized_odds'``). Requires ``group_cols``.
    """
    if sequence_model:
        evaluate_sequence_model(csv_path, model_type=sequence_model)
        return
    
    X, y = load_data(csv_path)
    model_params: dict | None = None
    if model_type == "stacking":
        model_params = {
            "estimators": estimators or ["logistic", "random_forest"],
            "final_estimator": final_estimator,
        }
    elif model_type == "bagging":
        model_params = {"base_estimator": base_estimator}
    pipeline = build_pipeline(X, model_type=model_type, model_params=model_params)

    # Prepare output directories
    fig_dir = Path('figures')
    fig_dir.mkdir(exist_ok=True)
    report_dir = Path('reports')
    report_dir.mkdir(exist_ok=True)

    # Hold-out evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    grid = PARAM_GRIDS.get(model_type, {}).get(param_grid)

    best_params: dict | None = None
    best_score: float | None = None

    if grid:
        search = GridSearchCV(pipeline, grid, cv=5, scoring="f1")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = model.named_steps["model"].get_params()
        best_score = search.best_score_
        print(f"Best params from search: {best_params} (score={best_score:.3f})")
    else:
        model = pipeline
        model.fit(X_train, y_train)
        best_params = model.named_steps["model"].get_params()
    # Keep original fitted pipeline for explanations even if mitigation wraps it
    fitted_pipeline = model
    # Optionally apply fairness mitigation using post-processing
    if mitigation != "none":
        if not group_cols:
            print(
                "Mitigation requested but no group column provided. Proceeding without mitigation."
            )
            y_pred = model.predict(X_test)
            y_prob = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )
        else:
            sens_train = X_train[group_cols[0]]
            sens_test = X_test[group_cols[0]]
            mitigator = ThresholdOptimizer(
                estimator=model, constraints=mitigation, prefit=True
            )
            mitigator.fit(X_train, y_train, sensitive_features=sens_train)
            y_pred = mitigator.predict(X_test, sensitive_features=sens_test)
            # ThresholdOptimizer may not expose predict_proba
            y_prob = (
                mitigator.predict_proba(X_test, sensitive_features=sens_test)[:, 1]
                if hasattr(mitigator, "predict_proba")
                else None
            )
            model = mitigator  # wrapped model (use fitted_pipeline for LIME/SHAP)
    else:
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
    print("Hold-out classification report:")
    print(classification_report(y_test, y_pred))

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
                print(f"Column '{col}' not in dataset. Skipping.")
                continue
            # Per-group classification artifacts
            for group_value in X_test[col].unique():
                mask = X_test[col] == group_value
                y_true_g = y_test[mask]
                y_pred_g = y_pred[mask]
                y_prob_g = y_prob[mask] if y_prob is not None else None
                if y_true_g.nunique() < 2:
                    print(
                        f"Skipping group {col}={group_value} due to single class in y_true."
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

            # Fairness metrics via fairlearn
            mf = MetricFrame(
                metrics={
                    "demographic_parity": selection_rate,
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                    "predictive_parity": positive_predictive_value,
                },
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=X_test[col],
            )
            fairness_records: list[dict[str, float | str]] = []
            for group_value, row in mf.by_group.iterrows():
                eo = max(
                    abs(row["true_positive_rate"] - overall_tpr),
                    abs(row["false_positive_rate"] - overall_fpr),
                )
                fairness_records.append(
                    {
                        col: group_value,
                        "demographic_parity": row["demographic_parity"],
                        "predictive_parity": row["predictive_parity"],
                        "equalized_odds": eo,
                    }
                )
            if fairness_records:
                fairness_df = pd.DataFrame(fairness_records)
                fairness_path = report_dir / f"fairness_{col}.csv"
                fairness_df.to_csv(fairness_path, index=False)
                print(f"Fairness metrics for '{col}':")
                print(
                    fairness_df.to_string(
                        index=False, float_format=lambda x: f"{x:.3f}"
                    )
                )

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
                print(f"Counterfactual generation failed for index {idx}: {e}")    
    except Exception as e:
        print(f"Skipping LIME explanations due to error: {e}")

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
            print(f"Skipping PDP for {feat} due to error: {e}")
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
            print(f"Skipping {m} due to error: {e}")
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
    parser.add_argument(
        '--group-cols',
        nargs='*',
        default=None,
        help='Demographic columns to evaluate',
    )
    parser.add_argument(
        '--model-type',
        choices=list(PARAM_GRIDS.keys()),
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
        '--mitigation',
        choices=['none', 'demographic_parity', 'equalized_odds'],
        default='none',
        help='Fairness mitigation strategy to apply',
    )
    args = parser.parse_args()
    main(
        csv_path=args.csv_path,
        group_cols=args.group_cols,
        model_type=args.model_type,
        param_grid=args.param_grid,
        estimators=args.estimators,
        final_estimator=args.final_estimator,
        base_estimator=args.base_estimator,
        sequence_model=args.sequence_model,
        mitigation=args.mitigation,
    )