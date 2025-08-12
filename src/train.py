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

    # ROC curve visualization
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
        overall_positive_rate = (y_pred == 1).mean()
        for col in group_cols:
            if col not in X_test.columns:
                print(f"Column '{col}' not in dataset. Skipping.")
                continue
            fairness_records: list[dict[str, float | str]] = []
            for group_value in X_test[col].unique():
                mask = X_test[col] == group_value
                y_true_g = y_test[mask]
                y_pred_g = y_pred[mask]
                y_prob_g = y_prob[mask]
                if y_true_g.nunique() < 2:
                    # Skip groups with a single class; metrics not meaningful
                    print(
                        f"Skipping group {col}={group_value} due to single class in y_true."
                    )
                    continue

                # Classification report
                grp_report = classification_report(
                    y_true_g, y_pred_g, output_dict=True
                )
                pd.DataFrame(grp_report).transpose().to_csv(
                    report_dir
                    / f"classification_report_{col}_{group_value}.csv",
                    index=True,
                )

                # Confusion matrix
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

                # ROC curve
                RocCurveDisplay.from_predictions(y_true_g, y_prob_g)
                plt.tight_layout()
                plt.savefig(fig_dir / f"roc_curve_{col}_{group_value}.png")
                plt.close()

                # Fairness metrics
                pos_rate = (y_pred_g == 1).mean()
                disparity = abs(pos_rate - overall_positive_rate)
                fairness_records.append(
                    {
                        col: group_value,
                        "positive_rate": pos_rate,
                        "disparity": disparity,
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

    # Cross-validation for robustness
    cv_model = build_pipeline(X, model_type=model_type, model_params=best_params)
    cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='f1')
    print(f'5-fold CV F1-score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

    # Export cross-validation scores
    cv_df = pd.DataFrame({'fold': range(1, len(cv_scores) + 1), 'f1_score': cv_scores})
    cv_df.to_csv(report_dir / 'cv_scores.csv', index=False)


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
