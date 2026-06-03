from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold


WORK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WORK_DIR / "sgRNA_synthesizability"
sys.path.append(str(REPO_DIR))

from feature_engineering import ID_COLUMN, TARGET_COLUMN, prepare_modeling_data
from train_baseline_model import (
    DATA_PATH,
    N_SPLITS,
    OUTPUT_DIR,
    RANDOM_STATE,
    evaluate_predictions,
    load_data,
    model_definitions,
    model_input,
)


CORRELATION_THRESHOLD = 0.90
REDUCED_FEATURE_DIR = WORK_DIR / "analysis" / "features_correlation_reduced"


def load_feature_priority(features: pd.DataFrame) -> pd.Series:
    ranking_path = OUTPUT_DIR / "feature_rankings.csv"
    if ranking_path.exists():
        rankings = pd.read_csv(ranking_path)
        if "feature" in rankings.columns:
            if "mannwhitney_p_rank" in rankings.columns:
                priority = rankings.set_index("feature")["mannwhitney_p_rank"]
            elif "random_forest_importance_rank" in rankings.columns:
                priority = rankings.set_index("feature")["random_forest_importance_rank"]
            else:
                priority = pd.Series(np.arange(1, len(rankings) + 1), index=rankings["feature"])
            return priority.reindex(features.columns).fillna(len(features.columns) + 1)

    variance = features.var(axis=0, ddof=0).rank(ascending=False, method="first")
    return variance.reindex(features.columns).fillna(len(features.columns) + 1)


def select_uncorrelated_features(
    features: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    priority = load_feature_priority(features)
    ordered_features = priority.sort_values(kind="mergesort").index.tolist()
    correlation = features.corr(method="pearson").abs().fillna(0.0)
    selected: list[str] = []
    dropped: list[dict[str, float | str]] = []

    for feature in ordered_features:
        if not selected:
            selected.append(feature)
            continue

        selected_correlations = correlation.loc[feature, selected]
        max_correlation = float(selected_correlations.max())
        if max_correlation > threshold:
            retained_feature = str(selected_correlations.idxmax())
            dropped.append(
                {
                    "dropped_feature": feature,
                    "retained_feature": retained_feature,
                    "absolute_correlation": max_correlation,
                    "dropped_priority_rank": float(priority.loc[feature]),
                    "retained_priority_rank": float(priority.loc[retained_feature]),
                }
            )
        else:
            selected.append(feature)

    selected_correlations = correlation.loc[selected, selected]
    return selected, pd.DataFrame(dropped), selected_correlations


def write_reduced_feature_outputs(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    modeling_data: pd.DataFrame,
    selected_features: list[str],
    dropped_features: pd.DataFrame,
    selected_correlations: pd.DataFrame,
) -> None:
    REDUCED_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    reduced_features = features[selected_features].copy()
    feature_output = reduced_features.copy()
    feature_output.insert(0, "id", modeling_data[ID_COLUMN] if ID_COLUMN in modeling_data.columns else pd.NA)
    feature_output.to_csv(REDUCED_FEATURE_DIR / "feature_matrix.csv", index=False)
    pd.DataFrame(
        {
            "source_row_index": modeling_data["source_row_index"],
            ID_COLUMN: modeling_data[ID_COLUMN] if ID_COLUMN in modeling_data.columns else pd.NA,
            "target": target,
            "label": modeling_data[TARGET_COLUMN],
        }
    ).to_csv(REDUCED_FEATURE_DIR / "target_vector.csv", index=False)
    pd.DataFrame(
        {
            "source_row_index": modeling_data["source_row_index"],
            "group_sequence": groups,
        }
    ).to_csv(REDUCED_FEATURE_DIR / "groups.csv", index=False)
    dropped_features.to_csv(REDUCED_FEATURE_DIR / "dropped_correlated_features.csv", index=False)
    selected_correlations.to_csv(REDUCED_FEATURE_DIR / "selected_feature_correlations.csv")
    metadata = {
        "input_path": str(DATA_PATH),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "original_feature_count": int(features.shape[1]),
        "reduced_feature_count": int(len(selected_features)),
        "dropped_feature_count": int(len(dropped_features)),
        "feature_matrix_id_column": "id",
        "model_feature_columns_exclude_id": True,
        "selection_priority": "feature_rankings.csv mannwhitney_p_rank when available, otherwise variance rank",
        "selected_features": selected_features,
    }
    (REDUCED_FEATURE_DIR / "feature_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def cross_validate_reduced_models(
    features: pd.DataFrame,
    sequences: pd.Series,
    y: np.ndarray,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    metrics: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(splitter.split(features, y, groups=groups), start=1):
        y_train = y[train_index]
        y_valid = y[valid_index]
        prevalence_score = np.repeat(y_train.mean(), len(valid_index))
        metrics.append(evaluate_predictions("correlation_reduced_prevalence_baseline", fold, y_valid, prevalence_score))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "model": "correlation_reduced_prevalence_baseline",
                    "row_index": valid_index,
                    "y_true": y_valid,
                    "risk_score": prevalence_score,
                }
            )
        )

        for model_name, spec in model_definitions().items():
            model = clone(spec.pipeline)
            current_input = model_input(features, sequences, spec)
            model.fit(current_input.iloc[train_index], y_train)
            valid_scores = model.predict_proba(current_input.iloc[valid_index])[:, 1]
            reduced_model_name = f"correlation_reduced_{model_name}"
            metrics.append(evaluate_predictions(reduced_model_name, fold, y_valid, valid_scores))
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "model": reduced_model_name,
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
    return metrics_df, predictions


def write_comparison_report(reduced_metrics: pd.DataFrame, original_feature_count: int, reduced_feature_count: int) -> pd.DataFrame:
    full_metrics_path = OUTPUT_DIR / "cv_metrics.csv"
    if not full_metrics_path.exists():
        reduced_metrics.to_csv(OUTPUT_DIR / "correlation_reduced_model_comparison.csv", index=False)
        return reduced_metrics

    full_metrics = pd.read_csv(full_metrics_path)
    full_overall = full_metrics[full_metrics["fold"].eq("overall_oof")].copy()
    reduced_overall = reduced_metrics[reduced_metrics["fold"].eq("overall_oof")].copy()
    reduced_overall["base_model"] = reduced_overall["model"].str.replace("correlation_reduced_", "", regex=False)
    comparison = reduced_overall.merge(
        full_overall,
        left_on="base_model",
        right_on="model",
        suffixes=("_reduced", "_full"),
        how="left",
    )
    comparison["original_feature_count"] = original_feature_count
    comparison["reduced_feature_count"] = reduced_feature_count
    comparison["removed_feature_count"] = original_feature_count - reduced_feature_count
    comparison["average_precision_delta"] = comparison["average_precision_reduced"] - comparison["average_precision_full"]
    comparison["pr_auc_delta"] = comparison["pr_auc_reduced"] - comparison["pr_auc_full"]
    comparison["roc_auc_delta"] = comparison["roc_auc_reduced"] - comparison["roc_auc_full"]
    comparison["f1_score_delta"] = comparison["f1_score_reduced"] - comparison["f1_score_full"]
    comparison["precision_top_05_delta"] = comparison["precision_top_05_reduced"] - comparison["precision_top_05_full"]
    comparison = comparison.sort_values("average_precision_reduced", ascending=False)
    comparison.to_csv(OUTPUT_DIR / "correlation_reduced_model_comparison.csv", index=False)

    display_columns = [
        "base_model",
        "average_precision_full",
        "average_precision_reduced",
        "average_precision_delta",
        "pr_auc_full",
        "pr_auc_reduced",
        "pr_auc_delta",
        "roc_auc_full",
        "roc_auc_reduced",
        "roc_auc_delta",
        "f1_score_full",
        "f1_score_reduced",
        "f1_score_delta",
        "precision_top_05_full",
        "precision_top_05_reduced",
        "precision_top_05_delta",
    ]
    lines = [
        "# Correlation-Reduced Model Comparison",
        "",
        f"Features were pruned by absolute Pearson correlation > {CORRELATION_THRESHOLD:.2f}.",
        f"Original feature count: {original_feature_count}",
        f"Reduced feature count: {reduced_feature_count}",
        f"Removed feature count: {original_feature_count - reduced_feature_count}",
        "",
        comparison[display_columns].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Positive deltas indicate that the correlation-reduced feature set outperformed the original full feature set for that metric.",
    ]
    (OUTPUT_DIR / "correlation_reduced_model_comparison.md").write_text("\n".join(lines) + "\n")
    return comparison


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    features, target, groups, binary_data = prepare_modeling_data(data)
    sequences = binary_data["Sequence"]
    selected_features, dropped_features, selected_correlations = select_uncorrelated_features(features)
    reduced_features = features[selected_features]
    write_reduced_feature_outputs(
        features,
        target,
        groups,
        binary_data,
        selected_features,
        dropped_features,
        selected_correlations,
    )
    metrics, predictions = cross_validate_reduced_models(reduced_features, sequences, target.to_numpy(), groups)
    metrics.to_csv(OUTPUT_DIR / "correlation_reduced_cv_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "correlation_reduced_oof_predictions.csv", index=False)
    comparison = write_comparison_report(metrics, features.shape[1], reduced_features.shape[1])
    best = comparison.iloc[0]
    print(f"original_features={features.shape[1]} reduced_features={reduced_features.shape[1]} dropped={len(dropped_features)}")
    print(
        comparison[
            [
                "base_model",
                "average_precision_full",
                "average_precision_reduced",
                "average_precision_delta",
                "roc_auc_full",
                "roc_auc_reduced",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"best_reduced_model={best['base_model']} average_precision={best['average_precision_reduced']:.4f}")
    print(f"saved_feature_outputs={REDUCED_FEATURE_DIR.relative_to(WORK_DIR)}")
    print(f"saved_model_outputs={OUTPUT_DIR.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()