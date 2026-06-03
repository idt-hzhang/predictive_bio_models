from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold


WORK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WORK_DIR / "sgRNA_synthesizability"
sys.path.append(str(REPO_DIR))

from feature_engineering import build_feature_table
from train_baseline_model import (
    N_SPLITS,
    OUTPUT_DIR,
    RANDOM_STATE,
    evaluate_predictions,
    model_definitions,
    model_input,
    rule_scores,
)


DATA_PATH = WORK_DIR / "data" / "cleaned_data.need_review.csv"
PASS_FAIL_METRICS_PATH = OUTPUT_DIR / "cv_metrics.csv"
PASS_FAIL_CORRELATION_REDUCED_PATH = OUTPUT_DIR / "correlation_reduced_model_comparison.csv"
CORRELATION_THRESHOLD = 0.90
POSITIVE_LABEL = "Needs Review"
NEGATIVE_LABEL = "Pass"


def load_need_review_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray, pd.Series]:
    data = pd.read_csv(DATA_PATH, dtype="string")
    modeling_data = data[data["Need_Review"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])].copy().reset_index(drop=True)
    features = build_feature_table(modeling_data, sequence_column="Sequence")
    sequences = modeling_data["Sequence"].fillna("").astype(str)
    target = modeling_data["Need_Review"].eq(POSITIVE_LABEL).astype(int).to_numpy()
    groups = sequences.copy()
    return modeling_data, features, sequences, target, groups


def target_association_priority(features: pd.DataFrame, target: np.ndarray) -> pd.Series:
    target_series = pd.Series(target, index=features.index)
    scores: dict[str, float] = {}
    for column in features.columns:
        if features[column].std(ddof=0) == 0:
            scores[column] = 0.0
        else:
            scores[column] = abs(float(features[column].corr(target_series)))
    return pd.Series(scores).fillna(0.0).sort_values(ascending=False)


def select_uncorrelated_features(features: pd.DataFrame, target: np.ndarray) -> tuple[list[str], pd.DataFrame]:
    priority = target_association_priority(features, target)
    correlations = features.corr(method="pearson").abs().fillna(0.0)
    selected: list[str] = []
    dropped: list[dict[str, float | str]] = []

    for feature in priority.index:
        if not selected:
            selected.append(feature)
            continue

        selected_correlations = correlations.loc[feature, selected]
        max_correlation = float(selected_correlations.max())
        if max_correlation > CORRELATION_THRESHOLD:
            retained_feature = str(selected_correlations.idxmax())
            dropped.append(
                {
                    "dropped_feature": feature,
                    "retained_feature": retained_feature,
                    "absolute_correlation": max_correlation,
                    "dropped_target_association": float(priority.loc[feature]),
                    "retained_target_association": float(priority.loc[retained_feature]),
                }
            )
        else:
            selected.append(feature)

    return selected, pd.DataFrame(dropped)


def cross_validate_models(
    features: pd.DataFrame,
    sequences: pd.Series,
    target: np.ndarray,
    groups: pd.Series,
    feature_set: str,
    include_rule_baseline: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    metrics: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    models = model_definitions()

    for fold, (train_index, valid_index) in enumerate(splitter.split(features, target, groups=groups), start=1):
        y_train = target[train_index]
        y_valid = target[valid_index]

        prevalence_score = np.repeat(y_train.mean(), len(valid_index))
        prevalence_name = f"{feature_set}_prevalence_baseline"
        metrics.append(evaluate_predictions(prevalence_name, fold, y_valid, prevalence_score))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "feature_set": feature_set,
                    "fold": fold,
                    "model": prevalence_name,
                    "row_index": valid_index,
                    "y_true": y_valid,
                    "risk_score": prevalence_score,
                }
            )
        )

        if include_rule_baseline:
            valid_rule_scores = rule_scores(features, train_index)[valid_index]
            rule_name = f"{feature_set}_rule_baseline"
            metrics.append(evaluate_predictions(rule_name, fold, y_valid, valid_rule_scores))
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "feature_set": feature_set,
                        "fold": fold,
                        "model": rule_name,
                        "row_index": valid_index,
                        "y_true": y_valid,
                        "risk_score": valid_rule_scores,
                    }
                )
            )

        for model_name, spec in models.items():
            model = clone(spec.pipeline)
            current_input = model_input(features, sequences, spec)
            model.fit(current_input.iloc[train_index], y_train)
            valid_scores = model.predict_proba(current_input.iloc[valid_index])[:, 1]
            full_model_name = f"{feature_set}_{model_name}"
            metrics.append(evaluate_predictions(full_model_name, fold, y_valid, valid_scores))
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "feature_set": feature_set,
                        "fold": fold,
                        "model": full_model_name,
                        "row_index": valid_index,
                        "y_true": y_valid,
                        "risk_score": valid_scores,
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
                        evaluate_predictions(
                            model_name,
                            "overall_oof",
                            model_predictions["y_true"].to_numpy(),
                            model_predictions["risk_score"].to_numpy(),
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )
    metrics_df.insert(0, "feature_set", feature_set)
    return metrics_df, predictions


def pass_fail_summary_lines() -> list[str]:
    lines: list[str] = []
    if PASS_FAIL_METRICS_PATH.exists():
        pass_fail = pd.read_csv(PASS_FAIL_METRICS_PATH)
        selected = pass_fail[(pass_fail["fold"].eq("overall_oof")) & (pass_fail["model"].eq("hist_gradient_boosting"))]
        if not selected.empty:
            row = selected.iloc[0]
            lines.append(
                f"- Main Pass/Fail full-feature HGB: PR-AUC {row.pr_auc:.4f}, ROC AUC {row.roc_auc:.4f}, F1 {row.f1_score:.4f}, Precision@5% {row.precision_top_05:.4f}."
            )
    if PASS_FAIL_CORRELATION_REDUCED_PATH.exists():
        reduced = pd.read_csv(PASS_FAIL_CORRELATION_REDUCED_PATH)
        selected = reduced[reduced["base_model"].eq("hist_gradient_boosting")]
        if not selected.empty:
            row = selected.iloc[0]
            lines.append(
                f"- Main Pass/Fail correlation-pruned HGB: PR-AUC {row.pr_auc_reduced:.4f}, ROC AUC {row.roc_auc_reduced:.4f}, F1 {row.f1_score_reduced:.4f}, Precision@5% {row.precision_top_05_reduced:.4f}."
            )
    return lines


def write_report(
    modeling_data: pd.DataFrame,
    metrics: pd.DataFrame,
    original_feature_count: int,
    reduced_feature_count: int,
    dropped_features: pd.DataFrame,
) -> None:
    overall = metrics[metrics["fold"].eq("overall_oof")].copy()
    overall = overall.sort_values("average_precision", ascending=False)
    learned_models = [name for name in model_definitions().keys()]
    learned_overall = overall[overall["model"].str.replace("full_", "", regex=False).str.replace("correlation_reduced_", "", regex=False).isin(learned_models)]
    best = learned_overall.iloc[0]
    display_columns = [
        "feature_set",
        "model",
        "average_precision",
        "pr_auc",
        "roc_auc",
        "brier_score",
        "f1_score",
        "recall_at_precision_25",
        "precision_top_05",
        "precision_top_10",
        "threshold_0_5_tn",
        "threshold_0_5_fp",
        "threshold_0_5_fn",
        "threshold_0_5_tp",
    ]
    metric_table = overall[display_columns].to_markdown(index=False, floatfmt=".4f")
    comparison_lines = pass_fail_summary_lines()
    label_counts = modeling_data["Need_Review"].value_counts().to_dict()

    lines = [
        "# Need_Review Modeling Report",
        "",
        f"Input file: `{DATA_PATH.relative_to(WORK_DIR)}`",
        f"Rows used: {len(modeling_data)}",
        f"Label counts: {label_counts}",
        f"Positive class: `{POSITIVE_LABEL}`",
        f"Original feature count: {original_feature_count}",
        f"Correlation-pruned feature count: {reduced_feature_count}",
        f"Dropped correlated feature count: {len(dropped_features)}",
        f"Cross-validation: {N_SPLITS}-fold stratified grouped CV using decorated `Sequence` groups.",
        "",
        "## Overall Out-of-Fold Metrics",
        "",
        metric_table,
        "",
        "## Best Learned Model",
        "",
        f"The best Need_Review model by PR-AUC is `{best.model}`, with PR-AUC {best.pr_auc:.4f}, ROC AUC {best.roc_auc:.4f}, Brier score {best.brier_score:.4f}, F1 {best.f1_score:.4f}, Precision@5% {best.precision_top_05:.4f}, and Precision@10% {best.precision_top_10:.4f}.",
        "",
        "## Comparison With Main Pass/Fail Models",
        "",
        *comparison_lines,
        f"- Best Need_Review model (`{best.model}`): PR-AUC {best.pr_auc:.4f}, ROC AUC {best.roc_auc:.4f}, F1 {best.f1_score:.4f}, Precision@5% {best.precision_top_05:.4f}.",
        "- The Need_Review target is less rare than Fail and is derived directly from Length and MainPeak thresholds, so it is not biologically identical to the source Pass/Fail label. Compare the metrics as target-specific ranking performance rather than as interchangeable endpoints.",
        "",
        "## Outputs",
        "",
        "- `need_review_cv_metrics.csv`",
        "- `need_review_oof_predictions.csv`",
        "- `need_review_correlation_reduced_cv_metrics.csv`",
        "- `need_review_correlation_reduced_oof_predictions.csv`",
        "- `need_review_model_comparison.csv`",
        "- `need_review_dropped_correlated_features.csv`",
        "- `need_review_modeling_report.md`",
    ]
    (OUTPUT_DIR / "need_review_modeling_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modeling_data, features, sequences, target, groups = load_need_review_data()
    selected_features, dropped_features = select_uncorrelated_features(features, target)
    reduced_features = features[selected_features]

    full_metrics, full_predictions = cross_validate_models(features, sequences, target, groups, "full", include_rule_baseline=True)
    reduced_metrics, reduced_predictions = cross_validate_models(
        reduced_features,
        sequences,
        target,
        groups,
        "correlation_reduced",
        include_rule_baseline=False,
    )
    combined_metrics = pd.concat([full_metrics, reduced_metrics], ignore_index=True)
    full_metrics.to_csv(OUTPUT_DIR / "need_review_cv_metrics.csv", index=False)
    full_predictions.to_csv(OUTPUT_DIR / "need_review_oof_predictions.csv", index=False)
    reduced_metrics.to_csv(OUTPUT_DIR / "need_review_correlation_reduced_cv_metrics.csv", index=False)
    reduced_predictions.to_csv(OUTPUT_DIR / "need_review_correlation_reduced_oof_predictions.csv", index=False)
    combined_metrics[combined_metrics["fold"].eq("overall_oof")].sort_values("average_precision", ascending=False).to_csv(
        OUTPUT_DIR / "need_review_model_comparison.csv",
        index=False,
    )
    dropped_features.to_csv(OUTPUT_DIR / "need_review_dropped_correlated_features.csv", index=False)
    write_report(modeling_data, combined_metrics, features.shape[1], len(selected_features), dropped_features)

    overall = combined_metrics[combined_metrics["fold"].eq("overall_oof")].sort_values("average_precision", ascending=False)
    print(f"rows={len(modeling_data)} positives={int(target.sum())} full_features={features.shape[1]} reduced_features={len(selected_features)}")
    print(overall[["feature_set", "model", "average_precision", "pr_auc", "roc_auc", "f1_score", "precision_top_05", "precision_top_10"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved_outputs={OUTPUT_DIR.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()