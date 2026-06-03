from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.metrics import average_precision_score, confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WORK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WORK_DIR / "sgRNA_synthesizability"
sys.path.append(str(REPO_DIR))

from feature_engineering import build_feature_table
from train_baseline_model import N_SPLITS, OUTPUT_DIR, RANDOM_STATE


DATA_PATH = WORK_DIR / "data" / "cleaned_data.mainpeak.csv"
ALL_CLEANED_DATA_PATH = WORK_DIR / "data" / "all.cleaned_data.csv"
PASS_FAIL_METRICS_PATH = OUTPUT_DIR / "cv_metrics.csv"
CORRELATION_REDUCED_METRICS_PATH = OUTPUT_DIR / "correlation_reduced_model_comparison.csv"
PREDICTION_PLOT_PATH = OUTPUT_DIR / "mainpeak_regression_observed_vs_predicted.png"


def load_mainpeak_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series]:
    data = pd.read_csv(DATA_PATH, dtype="string")
    labels = pd.read_csv(ALL_CLEANED_DATA_PATH, dtype="string", usecols=["Ref ID", "Pass/Fail", "Length"])
    data = data.merge(labels, on="Ref ID", how="left", validate="one_to_one")
    data["MainPeak"] = pd.to_numeric(data["MainPeak"], errors="coerce")
    data["Length"] = pd.to_numeric(data["Length"], errors="coerce")
    modeling_data = data.dropna(subset=["MainPeak", "Sequence"]).reset_index(drop=True)
    features = build_feature_table(modeling_data, sequence_column="Sequence")
    target = modeling_data["MainPeak"].astype(float).to_numpy()
    groups = modeling_data["Sequence"].fillna("").astype(str)
    return modeling_data, features, target, groups


def model_definitions() -> dict[str, Pipeline]:
    return {
        "mean_baseline": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=10000, random_state=RANDOM_STATE)),
            ]
        ),
        "lasso": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.01, max_iter=10000, random_state=RANDOM_STATE)),
            ]
        ),
        "bayesian_ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", BayesianRidge()),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=220,
                        learning_rate=0.035,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=240,
                        learning_rate=0.035,
                        max_depth=3,
                        min_samples_leaf=12,
                        subsample=0.85,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=8,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_regression(model_name: str, fold: int | str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | str]:
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    pearson = stats.pearsonr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else float("nan")
    spearman = stats.spearmanr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else float("nan")
    return {
        "model": model_name,
        "fold": fold,
        "rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearson),
        "spearman_r": float(spearman),
        "observed_mean": float(np.mean(y_true)),
        "predicted_mean": float(np.mean(y_pred)),
    }


def cross_validate_models(features: pd.DataFrame, target: np.ndarray, groups: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupKFold(n_splits=N_SPLITS)
    metrics: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(splitter.split(features, target, groups=groups), start=1):
        for model_name, pipeline in model_definitions().items():
            model = clone(pipeline)
            model.fit(features.iloc[train_index], target[train_index])
            predictions = model.predict(features.iloc[valid_index])
            metrics.append(evaluate_regression(model_name, fold, target[valid_index], predictions))
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "model": model_name,
                        "row_index": valid_index,
                        "observed_mainpeak": target[valid_index],
                        "predicted_mainpeak": predictions,
                    }
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)
    for model_name, model_predictions in predictions.groupby("model"):
        metrics_df = pd.concat(
            [
                metrics_df,
                pd.DataFrame(
                    [
                        evaluate_regression(
                            model_name,
                            "overall_oof",
                            model_predictions["observed_mainpeak"].to_numpy(),
                            model_predictions["predicted_mainpeak"].to_numpy(),
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )
    return metrics_df, predictions


def pass_threshold(length: pd.Series) -> pd.Series:
    thresholds = pd.Series(np.nan, index=length.index, dtype=float)
    thresholds[length < 56] = 80.0
    thresholds[(length >= 56) & (length < 90)] = 75.0
    thresholds[(length >= 90) & (length < 120)] = 70.0
    thresholds[length >= 120] = 50.0
    return thresholds


def precision_at_recall(y_true: np.ndarray, y_score: np.ndarray, min_recall: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = precision[recall >= min_recall]
    return float(valid.max()) if len(valid) else 0.0


def precision_at_top_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    top_n = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(y_score)[::-1][:top_n]
    return float(np.mean(y_true[order]))


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def class_metric_rows(model_name: str, tn: int, fp: int, fn: int, tp: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    class_counts = {
        "Pass": {
            "true_positive": tn,
            "false_positive": fn,
            "false_negative": fp,
            "true_negative": tp,
        },
        "Fail": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
    }

    for class_label, counts in class_counts.items():
        class_tp = counts["true_positive"]
        class_fp = counts["false_positive"]
        class_fn = counts["false_negative"]
        class_tn = counts["true_negative"]
        precision = safe_divide(class_tp, class_tp + class_fp)
        recall = safe_divide(class_tp, class_tp + class_fn)
        f1 = safe_divide(2 * precision * recall, precision + recall) if not np.isnan(precision) and not np.isnan(recall) else float("nan")
        rows.append(
            {
                "model": model_name,
                "class_label": class_label,
                "support": int(class_tp + class_fn),
                "predicted_count": int(class_tp + class_fp),
                "precision": precision,
                "recall": recall,
                "specificity": safe_divide(class_tn, class_tn + class_fp),
                "f1": f1,
                "true_positive": int(class_tp),
                "false_positive": int(class_fp),
                "false_negative": int(class_fn),
                "true_negative": int(class_tn),
            }
        )
    return rows


def derived_classification_metrics(modeling_data: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = modeling_data[["Ref ID", "Length", "Pass/Fail"]].reset_index().rename(columns={"index": "row_index"})
    classified = predictions.merge(metadata, on="row_index", how="left", validate="many_to_one")
    classified["Length"] = pd.to_numeric(classified["Length"], errors="coerce")
    classified["mainpeak_pass_threshold"] = pass_threshold(classified["Length"])
    classified["predicted_rule_label"] = np.where(
        classified["predicted_mainpeak"] >= classified["mainpeak_pass_threshold"],
        "Pass",
        "Fail",
    )
    classified.loc[classified["mainpeak_pass_threshold"].isna(), "predicted_rule_label"] = pd.NA
    classified["predicted_fail_score"] = (classified["mainpeak_pass_threshold"] - classified["predicted_mainpeak"]) / classified["mainpeak_pass_threshold"]

    binary = classified[classified["Pass/Fail"].isin(["Pass", "Fail"]) & classified["predicted_fail_score"].notna()].copy()
    binary["y_true"] = binary["Pass/Fail"].eq("Fail").astype(int)
    binary["y_pred"] = binary["predicted_rule_label"].eq("Fail").astype(int)

    rows: list[dict[str, float | int | str]] = []
    class_rows: list[dict[str, float | int | str]] = []
    for model_name, model_predictions in binary.groupby("model"):
        y_true = model_predictions["y_true"].to_numpy()
        y_score = model_predictions["predicted_fail_score"].to_numpy()
        y_pred = model_predictions["y_pred"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        class_rows.extend(class_metric_rows(model_name, int(tn), int(fp), int(fn), int(tp)))
        rows.append(
            {
                "model": model_name,
                "rows": int(len(model_predictions)),
                "failures": int(y_true.sum()),
                "average_precision": float(average_precision_score(y_true, y_score)),
                "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
                "precision_at_recall_25": precision_at_recall(y_true, y_score, 0.25),
                "precision_at_recall_50": precision_at_recall(y_true, y_score, 0.50),
                "precision_at_recall_75": precision_at_recall(y_true, y_score, 0.75),
                "precision_top_05": precision_at_top_fraction(y_true, y_score, 0.05),
                "precision_top_10": precision_at_top_fraction(y_true, y_score, 0.10),
                "rule_threshold_tn": int(tn),
                "rule_threshold_fp": int(fp),
                "rule_threshold_fn": int(fn),
                "rule_threshold_tp": int(tp),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["average_precision", "roc_auc"], ascending=False)
    class_metrics = pd.DataFrame(class_rows).sort_values(["model", "class_label"])
    return metrics, class_metrics, classified


def write_prediction_plot(predictions: pd.DataFrame, best_model_name: str) -> None:
    best_predictions = predictions[predictions["model"].eq(best_model_name)]
    observed = best_predictions["observed_mainpeak"].to_numpy()
    predicted = best_predictions["predicted_mainpeak"].to_numpy()
    limits = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(observed, predicted, s=14, alpha=0.42, color="#2364aa", edgecolors="none")
    ax.plot(limits, limits, color="#b84a4a", linewidth=1.4, label="Ideal")
    ax.set_xlabel("Observed MainPeak")
    ax.set_ylabel("Predicted MainPeak")
    ax.set_title(f"Observed vs predicted MainPeak: {best_model_name}")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PREDICTION_PLOT_PATH, dpi=180)
    plt.close(fig)


def pass_fail_summary_lines() -> list[str]:
    lines: list[str] = []
    if PASS_FAIL_METRICS_PATH.exists():
        pass_fail = pd.read_csv(PASS_FAIL_METRICS_PATH)
        selected = pass_fail[(pass_fail["fold"].eq("overall_oof")) & (pass_fail["model"].eq("hist_gradient_boosting"))]
        if not selected.empty:
            row = selected.iloc[0]
            lines.append(
                f"- Previous best Pass/Fail classifier (`hist_gradient_boosting`): AP {row.average_precision:.4f}, ROC AUC {row.roc_auc:.4f}, Precision@5% {row.precision_top_05:.4f}."
            )
    if CORRELATION_REDUCED_METRICS_PATH.exists():
        reduced = pd.read_csv(CORRELATION_REDUCED_METRICS_PATH)
        selected = reduced[reduced["base_model"].eq("hist_gradient_boosting")]
        if not selected.empty:
            row = selected.iloc[0]
            lines.append(
                f"- Best correlation-pruned Pass/Fail classifier (`hist_gradient_boosting`): AP {row.average_precision_reduced:.4f}, ROC AUC {row.roc_auc_reduced:.4f}, Precision@5% {row.precision_top_05_reduced:.4f}."
            )
    return lines


def write_report(
    modeling_data: pd.DataFrame,
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    derived_metrics: pd.DataFrame,
    derived_class_metrics: pd.DataFrame,
    best_model_name: str,
) -> None:
    overall = metrics[metrics["fold"].eq("overall_oof")].sort_values("rmse")
    target = modeling_data["MainPeak"].astype(float)
    metric_table = overall[["model", "mae", "rmse", "r2", "pearson_r", "spearman_r"]].to_markdown(index=False, floatfmt=".4f")
    derived_metric_table = derived_metrics[
        [
            "model",
            "average_precision",
            "roc_auc",
            "precision_at_recall_25",
            "precision_at_recall_50",
            "precision_at_recall_75",
            "precision_top_05",
            "precision_top_10",
            "rule_threshold_tn",
            "rule_threshold_fp",
            "rule_threshold_fn",
            "rule_threshold_tp",
        ]
    ].to_markdown(index=False, floatfmt=".4f")
    best_row = overall.iloc[0]
    best_derived_row = derived_metrics.iloc[0]
    best_derived_class_table = derived_class_metrics[
        derived_class_metrics["model"].eq(best_derived_row.model)
    ][["class_label", "support", "predicted_count", "precision", "recall", "specificity", "f1"]].to_markdown(index=False, floatfmt=".4f")
    baseline_row = overall[overall["model"].eq("mean_baseline")].iloc[0]
    rmse_improvement = baseline_row.rmse - best_row.rmse
    relative_rmse_improvement = rmse_improvement / baseline_row.rmse if baseline_row.rmse else float("nan")
    comparison_lines = pass_fail_summary_lines()

    lines = [
        "# MainPeak Regression Modeling Report",
        "",
        f"Input file: `{DATA_PATH.relative_to(WORK_DIR)}`",
        f"Rows with nonmissing MainPeak and Sequence: {len(modeling_data)}",
        f"Sequence feature count: {features.shape[1]}",
        f"Cross-validation: {N_SPLITS}-fold grouped CV using decorated `Sequence` groups.",
        "",
        "## Target Summary",
        "",
        f"- MainPeak mean: {target.mean():.3f}",
        f"- MainPeak median: {target.median():.3f}",
        f"- MainPeak standard deviation: {target.std(ddof=0):.3f}",
        f"- MainPeak range: {target.min():.3f} to {target.max():.3f}",
        "",
        "## Overall Out-of-Fold Metrics",
        "",
        metric_table,
        "",
        "## Best Model",
        "",
        f"The best MainPeak model by RMSE is `{best_model_name}`, with RMSE {best_row.rmse:.4f}, MAE {best_row.mae:.4f}, R2 {best_row.r2:.4f}, Pearson r {best_row.pearson_r:.4f}, and Spearman r {best_row.spearman_r:.4f}.",
        f"Compared with the mean baseline RMSE of {baseline_row.rmse:.4f}, this improves RMSE by {rmse_improvement:.4f} ({relative_rmse_improvement:.1%}).",
        f"Observed-vs-predicted plot: `{PREDICTION_PLOT_PATH.relative_to(WORK_DIR)}`.",
        "",
        "## Derived Pass/Fail Classification From Predicted MainPeak",
        "",
        "Predicted MainPeak values were converted to derived binary labels using the length-specific thresholds documented in `sgRNA_synthesizability/.github/pass_fail_labeling_rules.md`. A predicted value below the pass threshold was treated as derived `Fail` for comparison with the original binary `Pass/Fail` labels. The continuous failure score is the threshold-normalized deficit below the pass threshold, so larger values indicate higher predicted failure risk.",
        "",
        derived_metric_table,
        "",
        f"The best derived classifier by average precision is `{best_derived_row.model}`, with AP {best_derived_row.average_precision:.4f}, ROC AUC {best_derived_row.roc_auc:.4f}, Precision@Recall25 {best_derived_row.precision_at_recall_25:.4f}, Precision@Recall50 {best_derived_row.precision_at_recall_50:.4f}, and Precision@5% {best_derived_row.precision_top_05:.4f}.",
        "",
        "### Per-Class Performance for Best Derived Classifier",
        "",
        best_derived_class_table,
        "",
        "Per-class metrics are calculated from the hard labels produced by the MainPeak threshold rule. For `Fail`, recall is failure sensitivity. For `Pass`, recall is pass specificity against false failure calls.",
        "",
        "## Comparison With Pass/Fail Models",
        "",
        *comparison_lines,
        f"- Best derived classifier from MainPeak regression (`{best_derived_row.model}`): AP {best_derived_row.average_precision:.4f}, ROC AUC {best_derived_row.roc_auc:.4f}, Precision@5% {best_derived_row.precision_top_05:.4f}.",
        "- MainPeak regression is a continuous QC modeling task, so its metrics are not directly comparable to AP/ROC AUC. The useful comparison is qualitative: MainPeak regression tests whether sequence features explain continuous chromatographic purity, while Pass/Fail classification tests whether they enrich rare failures.",
        f"- The best MainPeak model explains {best_row.r2:.1%} of out-of-fold MainPeak variance. This indicates substantial continuous purity signal in sequence features, whereas the rare-event Pass/Fail models remain more limited because they compress that continuous QC behavior into an imbalanced binary endpoint.",
        "",
        "## Outputs",
        "",
        "- `mainpeak_regression_metrics.csv`",
        "- `mainpeak_regression_oof_predictions.csv`",
        "- `mainpeak_regression_derived_classifier_metrics.csv`",
        "- `mainpeak_regression_derived_classifier_class_metrics.csv`",
        "- `mainpeak_regression_derived_classifier_predictions.csv`",
        "- `mainpeak_regression_observed_vs_predicted.png`",
        "- `mainpeak_regression_report.md`",
    ]
    (OUTPUT_DIR / "mainpeak_regression_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modeling_data, features, target, groups = load_mainpeak_data()
    metrics, predictions = cross_validate_models(features, target, groups)
    derived_metrics, derived_class_metrics, derived_predictions = derived_classification_metrics(modeling_data, predictions)
    overall = metrics[metrics["fold"].eq("overall_oof")].sort_values("rmse")
    best_model_name = str(overall.iloc[0]["model"])
    metrics.to_csv(OUTPUT_DIR / "mainpeak_regression_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "mainpeak_regression_oof_predictions.csv", index=False)
    derived_metrics.to_csv(OUTPUT_DIR / "mainpeak_regression_derived_classifier_metrics.csv", index=False)
    derived_class_metrics.to_csv(OUTPUT_DIR / "mainpeak_regression_derived_classifier_class_metrics.csv", index=False)
    derived_predictions.to_csv(OUTPUT_DIR / "mainpeak_regression_derived_classifier_predictions.csv", index=False)
    write_prediction_plot(predictions, best_model_name)
    write_report(modeling_data, features, metrics, derived_metrics, derived_class_metrics, best_model_name)
    print(f"rows={len(modeling_data)} features={features.shape[1]} best_model={best_model_name}")
    print(overall[["model", "mae", "rmse", "r2", "pearson_r", "spearman_r"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nDerived Pass/Fail metrics from predicted MainPeak:")
    print(derived_metrics[["model", "average_precision", "roc_auc", "precision_at_recall_50", "precision_top_05"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPer-class metrics for best derived classifier:")
    best_derived_model = str(derived_metrics.iloc[0]["model"])
    print(derived_class_metrics[derived_class_metrics["model"].eq(best_derived_model)][["class_label", "support", "predicted_count", "precision", "recall", "specificity", "f1"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved_outputs={OUTPUT_DIR.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()