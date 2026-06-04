from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


WORK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WORK_DIR / "sgRNA_synthesizability"
sys.path.append(str(REPO_DIR))

from feature_engineering import build_feature_table, prepare_modeling_data
from train_baseline_model import (
    DATA_PATH as PASS_FAIL_DATA_PATH,
    N_SPLITS,
    OUTPUT_DIR,
    RANDOM_STATE,
    evaluate_predictions,
    load_data,
    model_definitions as classification_model_definitions,
    model_input,
)

from train_mainpeak_regression import (
    DATA_PATH as MAINPEAK_DATA_PATH,
    evaluate_regression,
    load_mainpeak_data,
    model_definitions as regression_model_definitions,
)


NEED_REVIEW_DATA_PATH = WORK_DIR / "data" / "cleaned_data.need_review.csv"
DEFAULT_MAX_FEATURES = 9
DEFAULT_OUTPUT_PREFIX = "narrow_feature_panel"
MAX_ABS_CORRELATION = 0.75
COUNT_FRACTION_PREFERENCE_RATIO = 0.90
CLASSIFICATION_MODELS = [
    "logistic_l2_balanced",
    "logistic_l1_balanced",
    "elastic_net_logistic",
    "linear_svm_rbf_calibrated_probability",
    "hist_gradient_boosting",
    "balanced_random_forest",
    "lightgbm_balanced",
    "xgboost_weighted",
]
REGRESSION_MODELS = [
    "mean_baseline",
    "ridge",
    "elastic_net",
    "lasso",
    "bayesian_ridge",
    "hist_gradient_boosting",
    "gradient_boosting",
    "random_forest",
    "extra_trees",
]


def absolute_target_correlations(features: pd.DataFrame, target: np.ndarray) -> pd.Series:
    target_series = pd.Series(target, index=features.index)
    scores: dict[str, float] = {}
    for column in features.columns:
        if features[column].std(ddof=0) == 0:
            scores[column] = 0.0
        else:
            scores[column] = abs(float(features[column].corr(target_series)))
    return pd.Series(scores).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def percentile_scores(scores: pd.Series) -> pd.Series:
    if scores.max() == scores.min():
        return pd.Series(0.0, index=scores.index)
    return scores.rank(pct=True, method="average")


def feature_family(feature: str) -> str:
    suffixes = [
        "_density_middle",
        "_density_5p",
        "_density_3p",
        "_count_middle",
        "_count_5p",
        "_count_3p",
        "_position_std",
        "_span_fraction",
        "_mean_fraction",
        "_first_fraction",
        "_last_fraction",
        "_mean_position",
        "_first_position",
        "_last_position",
        "_fraction",
        "_density",
        "_count",
    ]
    for suffix in suffixes:
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return feature


def is_count_or_fraction_feature(feature: str) -> bool:
    count_fraction_markers = (
        "_count",
        "_fraction",
        "_density",
    )
    return any(marker in feature for marker in count_fraction_markers)


def load_need_review_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series]:
    data = pd.read_csv(NEED_REVIEW_DATA_PATH, dtype="string")
    modeling_data = data[data["Need_Review"].isin(["Pass", "Needs Review"])].copy().reset_index(drop=True)
    features = build_feature_table(modeling_data, sequence_column="Sequence")
    target = modeling_data["Need_Review"].eq("Needs Review").astype(int).to_numpy()
    groups = modeling_data["Sequence"].fillna("").astype(str)
    return modeling_data, features, target, groups


def select_narrow_feature_panel(max_features: int) -> tuple[list[str], pd.DataFrame]:
    pass_fail_data = load_data()
    pass_fail_features, pass_fail_target, _, _ = prepare_modeling_data(pass_fail_data)
    _, need_review_features, need_review_target, _ = load_need_review_data()
    _, mainpeak_features, mainpeak_target, _ = load_mainpeak_data()

    common_features = sorted(set(pass_fail_features.columns) & set(need_review_features.columns) & set(mainpeak_features.columns))
    pass_fail_scores = absolute_target_correlations(pass_fail_features[common_features], pass_fail_target.to_numpy())
    need_review_scores = absolute_target_correlations(need_review_features[common_features], need_review_target)
    mainpeak_scores = absolute_target_correlations(mainpeak_features[common_features], mainpeak_target)

    score_table = pd.DataFrame(
        {
            "feature": common_features,
            "pass_fail_abs_corr": pass_fail_scores.reindex(common_features).to_numpy(),
            "need_review_abs_corr": need_review_scores.reindex(common_features).to_numpy(),
            "mainpeak_abs_corr": mainpeak_scores.reindex(common_features).to_numpy(),
        }
    )
    score_table["pass_fail_percentile"] = percentile_scores(score_table.set_index("feature")["pass_fail_abs_corr"]).reindex(score_table["feature"]).to_numpy()
    score_table["need_review_percentile"] = percentile_scores(score_table.set_index("feature")["need_review_abs_corr"]).reindex(score_table["feature"]).to_numpy()
    score_table["mainpeak_percentile"] = percentile_scores(score_table.set_index("feature")["mainpeak_abs_corr"]).reindex(score_table["feature"]).to_numpy()
    score_table["composite_score"] = score_table[["pass_fail_percentile", "need_review_percentile", "mainpeak_percentile"]].mean(axis=1)
    score_table["feature_family"] = score_table["feature"].map(feature_family)
    score_table["is_count_or_fraction"] = score_table["feature"].map(is_count_or_fraction_feature)

    family_representatives: list[pd.Series] = []
    for _, family_rows in score_table.groupby("feature_family", sort=False):
        ordered = family_rows.sort_values("composite_score", ascending=False)
        best = ordered.iloc[0].copy()
        count_fraction_rows = ordered[ordered["is_count_or_fraction"]]
        if not count_fraction_rows.empty:
            best_count_fraction = count_fraction_rows.iloc[0]
            if best_count_fraction["composite_score"] >= best["composite_score"] * COUNT_FRACTION_PREFERENCE_RATIO:
                best = best_count_fraction.copy()
                best["family_selection_reason"] = "preferred_count_or_fraction_within_family"
            else:
                best["family_selection_reason"] = "non_count_fraction_score_advantage"
        else:
            best["family_selection_reason"] = "no_count_or_fraction_candidate"
        family_representatives.append(best)

    representative_table = pd.DataFrame(family_representatives)

    feature_correlations = mainpeak_features[common_features].corr(method="pearson").abs().fillna(0.0)
    selected: list[str] = []
    selected_rows: list[dict[str, float | int | str]] = []

    for row in representative_table.sort_values("composite_score", ascending=False).itertuples(index=False):
        feature = str(row.feature)
        max_correlation = float(feature_correlations.loc[feature, selected].max()) if selected else 0.0
        if max_correlation > MAX_ABS_CORRELATION:
            continue
        selected.append(feature)
        selected_rows.append(
            {
                "rank": len(selected),
                "feature": feature,
                "feature_family": str(row.feature_family),
                "is_count_or_fraction": bool(row.is_count_or_fraction),
                "family_selection_reason": str(row.family_selection_reason),
                "composite_score": float(row.composite_score),
                "pass_fail_abs_corr": float(row.pass_fail_abs_corr),
                "need_review_abs_corr": float(row.need_review_abs_corr),
                "mainpeak_abs_corr": float(row.mainpeak_abs_corr),
                "max_abs_correlation_to_previous_selected": max_correlation,
            }
        )
        if len(selected) >= max_features:
            break

    return selected, pd.DataFrame(selected_rows)


def cross_validate_classification_endpoint(
    endpoint: str,
    features: pd.DataFrame,
    sequences: pd.Series,
    target: np.ndarray,
    groups: pd.Series,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    metrics: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    models = classification_model_definitions()
    selected_models = {name: spec for name, spec in models.items() if name in CLASSIFICATION_MODELS and not spec.uses_sequence_text}
    panel_features = features[selected_features]

    for fold, (train_index, valid_index) in enumerate(splitter.split(panel_features, target, groups=groups), start=1):
        y_train = target[train_index]
        y_valid = target[valid_index]
        prevalence_score = np.repeat(y_train.mean(), len(valid_index))
        metrics.append(evaluate_predictions("prevalence_baseline", fold, y_valid, prevalence_score))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "endpoint": endpoint,
                    "fold": fold,
                    "model": "prevalence_baseline",
                    "row_index": valid_index,
                    "y_true": y_valid,
                    "risk_score": prevalence_score,
                }
            )
        )

        for model_name, spec in selected_models.items():
            model = clone(spec.pipeline)
            current_input = model_input(panel_features, sequences, spec)
            model.fit(current_input.iloc[train_index], y_train)
            valid_scores = model.predict_proba(current_input.iloc[valid_index])[:, 1]
            metrics.append(evaluate_predictions(model_name, fold, y_valid, valid_scores))
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "endpoint": endpoint,
                        "fold": fold,
                        "model": model_name,
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
    metrics_df.insert(0, "endpoint", endpoint)
    metrics_df.insert(1, "feature_count", len(selected_features))
    return metrics_df, predictions


def cross_validate_regression_endpoint(
    features: pd.DataFrame,
    target: np.ndarray,
    groups: pd.Series,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupKFold(n_splits=N_SPLITS)
    panel_features = features[selected_features]
    models = regression_model_definitions()
    metrics: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(splitter.split(panel_features, target, groups=groups), start=1):
        for model_name in REGRESSION_MODELS:
            model = clone(models[model_name])
            model.fit(panel_features.iloc[train_index], target[train_index])
            predictions = model.predict(panel_features.iloc[valid_index])
            metrics.append(evaluate_regression(model_name, fold, target[valid_index], predictions))
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "endpoint": "mainpeak_regression",
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
    metrics_df.insert(0, "endpoint", "mainpeak_regression")
    metrics_df.insert(1, "feature_count", len(selected_features))
    return metrics_df, predictions


def reference_metrics() -> dict[str, float]:
    references: dict[str, float] = {}
    cv_path = OUTPUT_DIR / "cv_metrics.csv"
    if cv_path.exists():
        cv_metrics = pd.read_csv(cv_path)
        row = cv_metrics[cv_metrics["fold"].eq("overall_oof") & cv_metrics["model"].eq("hist_gradient_boosting")]
        if not row.empty:
            references["pass_fail_full_hgb_pr_auc"] = float(row.iloc[0]["pr_auc"])
            references["pass_fail_full_hgb_f1"] = float(row.iloc[0]["f1_score"])
    correlation_path = OUTPUT_DIR / "correlation_reduced_model_comparison.csv"
    if correlation_path.exists():
        comparison = pd.read_csv(correlation_path)
        row = comparison[comparison["base_model"].eq("hist_gradient_boosting")]
        if not row.empty:
            references["pass_fail_correlation_hgb_pr_auc"] = float(row.iloc[0]["pr_auc_reduced"])
            references["pass_fail_correlation_hgb_f1"] = float(row.iloc[0]["f1_score_reduced"])
    need_review_path = OUTPUT_DIR / "need_review_model_comparison.csv"
    if need_review_path.exists():
        need_review = pd.read_csv(need_review_path)
        row = need_review.sort_values("pr_auc", ascending=False).head(1)
        if not row.empty:
            references["need_review_best_pr_auc"] = float(row.iloc[0]["pr_auc"])
            references["need_review_best_f1"] = float(row.iloc[0]["f1_score"])
    mainpeak_path = OUTPUT_DIR / "mainpeak_regression_metrics.csv"
    if mainpeak_path.exists():
        mainpeak = pd.read_csv(mainpeak_path)
        row = mainpeak[mainpeak["fold"].eq("overall_oof") & mainpeak["model"].eq("extra_trees")]
        if not row.empty:
            references["mainpeak_full_extra_trees_rmse"] = float(row.iloc[0]["rmse"])
            references["mainpeak_full_extra_trees_r2"] = float(row.iloc[0]["r2"])
    return references


def write_report(selected_features: pd.DataFrame, classification_metrics: pd.DataFrame, regression_metrics: pd.DataFrame, output_prefix: str) -> None:
    classification_overall = classification_metrics[classification_metrics["fold"].eq("overall_oof")].copy()
    learned_classification = classification_overall[~classification_overall["model"].eq("prevalence_baseline")]
    best_classification = learned_classification.sort_values(["endpoint", "pr_auc"], ascending=[True, False]).groupby("endpoint", as_index=False).head(1)
    regression_overall = regression_metrics[regression_metrics["fold"].eq("overall_oof")].copy().sort_values("rmse")
    best_regression = regression_overall.iloc[0]
    references = reference_metrics()

    classification_table = classification_overall.sort_values(["endpoint", "pr_auc"], ascending=[True, False])[
        ["endpoint", "model", "pr_auc", "roc_auc", "brier_score", "f1_score", "precision_top_05", "precision_top_10"]
    ].to_markdown(index=False, floatfmt=".4f")
    regression_table = regression_overall[["model", "mae", "rmse", "r2", "pearson_r", "spearman_r"]].to_markdown(index=False, floatfmt=".4f")
    selected_table = selected_features.to_markdown(index=False, floatfmt=".4f")
    best_classification_table = best_classification[
        ["endpoint", "model", "pr_auc", "roc_auc", "f1_score", "precision_top_05"]
    ].to_markdown(index=False, floatfmt=".4f")

    pass_fail_best = best_classification[best_classification["endpoint"].eq("pass_fail")].iloc[0]
    need_review_best = best_classification[best_classification["endpoint"].eq("need_review")].iloc[0]
    pass_fail_retained = pass_fail_best.pr_auc / references.get("pass_fail_correlation_hgb_pr_auc", np.nan)
    need_review_retained = need_review_best.pr_auc / references.get("need_review_best_pr_auc", np.nan)
    mainpeak_rmse_increase = (best_regression.rmse / references.get("mainpeak_full_extra_trees_rmse", np.nan)) - 1.0

    lines = [
        f"# {len(selected_features)}-Feature Narrow Panel Modeling Report",
        "",
        f"Selected a shared panel of {len(selected_features)} sequence-derived features using composite target association across Pass/Fail, Pass/Need_Review, and MainPeak, with pairwise absolute Pearson correlation capped at {MAX_ABS_CORRELATION:.2f} during greedy selection.",
        "",
        "## Selected Features",
        "",
        selected_table,
        "",
        "## Best Models by Endpoint",
        "",
        best_classification_table,
        "",
        f"Best MainPeak regression model: `{best_regression.model}` with RMSE {best_regression.rmse:.4f}, R2 {best_regression.r2:.4f}, Pearson r {best_regression.pearson_r:.4f}, and Spearman r {best_regression.spearman_r:.4f}.",
        "",
        "## Classification Metrics",
        "",
        classification_table,
        "",
        "## MainPeak Regression Metrics",
        "",
        regression_table,
        "",
        "## Retained Performance Versus Larger Feature Sets",
        "",
        f"- Pass/Fail: best narrow-panel model `{pass_fail_best.model}` PR-AUC {pass_fail_best.pr_auc:.4f}, retaining {pass_fail_retained:.1%} of the correlation-pruned HGB PR-AUC reference ({references.get('pass_fail_correlation_hgb_pr_auc', np.nan):.4f}).",
        f"- Pass/Need_Review: best narrow-panel model `{need_review_best.model}` PR-AUC {need_review_best.pr_auc:.4f}, retaining {need_review_retained:.1%} of the previous best Need_Review PR-AUC reference ({references.get('need_review_best_pr_auc', np.nan):.4f}).",
        f"- MainPeak regression: best narrow-panel model `{best_regression.model}` RMSE {best_regression.rmse:.4f}, a {mainpeak_rmse_increase:.1%} increase versus full-feature ExtraTrees RMSE ({references.get('mainpeak_full_extra_trees_rmse', np.nan):.4f}).",
        "",
        "The panel is intentionally small and low-redundancy, so it is most useful for interpretation and lightweight triage. It does not fully preserve the best larger-model performance, especially for MainPeak regression.",
        "",
        "## Outputs",
        "",
        f"- `{output_prefix}_selected_features.csv`",
        f"- `{output_prefix}_classification_metrics.csv`",
        f"- `{output_prefix}_classification_oof_predictions.csv`",
        f"- `{output_prefix}_regression_metrics.csv`",
        f"- `{output_prefix}_regression_oof_predictions.csv`",
        f"- `{output_prefix}_modeling_report.md`",
    ]
    (OUTPUT_DIR / f"{output_prefix}_modeling_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a low-redundancy narrow sequence-feature panel.")
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES, help="Maximum number of low-redundancy features to select.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Prefix for output files written under analysis/modeling_outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_features < 1:
        raise ValueError("--max-features must be at least 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_feature_names, selected_features = select_narrow_feature_panel(args.max_features)

    pass_fail_data = load_data()
    pass_fail_features, pass_fail_target, pass_fail_groups, pass_fail_binary_data = prepare_modeling_data(pass_fail_data)
    pass_fail_metrics, pass_fail_predictions = cross_validate_classification_endpoint(
        "pass_fail",
        pass_fail_features,
        pass_fail_binary_data["Sequence"],
        pass_fail_target.to_numpy(),
        pass_fail_groups,
        selected_feature_names,
    )

    need_review_data, need_review_features, need_review_target, need_review_groups = load_need_review_data()
    need_review_metrics, need_review_predictions = cross_validate_classification_endpoint(
        "need_review",
        need_review_features,
        need_review_data["Sequence"],
        need_review_target,
        need_review_groups,
        selected_feature_names,
    )

    _, mainpeak_features, mainpeak_target, mainpeak_groups = load_mainpeak_data()
    regression_metrics, regression_predictions = cross_validate_regression_endpoint(
        mainpeak_features,
        mainpeak_target,
        mainpeak_groups,
        selected_feature_names,
    )

    classification_metrics = pd.concat([pass_fail_metrics, need_review_metrics], ignore_index=True)
    classification_predictions = pd.concat([pass_fail_predictions, need_review_predictions], ignore_index=True)
    selected_features.to_csv(OUTPUT_DIR / f"{args.output_prefix}_selected_features.csv", index=False)
    classification_metrics.to_csv(OUTPUT_DIR / f"{args.output_prefix}_classification_metrics.csv", index=False)
    classification_predictions.to_csv(OUTPUT_DIR / f"{args.output_prefix}_classification_oof_predictions.csv", index=False)
    regression_metrics.to_csv(OUTPUT_DIR / f"{args.output_prefix}_regression_metrics.csv", index=False)
    regression_predictions.to_csv(OUTPUT_DIR / f"{args.output_prefix}_regression_oof_predictions.csv", index=False)
    write_report(selected_features, classification_metrics, regression_metrics, args.output_prefix)

    classification_overall = classification_metrics[classification_metrics["fold"].eq("overall_oof")]
    learned_classification = classification_overall[~classification_overall["model"].eq("prevalence_baseline")]
    best_classification = learned_classification.sort_values(["endpoint", "pr_auc"], ascending=[True, False]).groupby("endpoint", as_index=False).head(1)
    best_regression = regression_metrics[regression_metrics["fold"].eq("overall_oof")].sort_values("rmse").iloc[0]
    print(f"selected_features={len(selected_feature_names)} features={','.join(selected_feature_names)}")
    print(best_classification[["endpoint", "model", "pr_auc", "roc_auc", "f1_score", "precision_top_05"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"best_regression_model={best_regression.model} rmse={best_regression.rmse:.4f} r2={best_regression.r2:.4f}")
    print(f"saved_outputs={OUTPUT_DIR.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()