from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WORK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WORK_DIR / "sgRNA_synthesizability"
sys.path.append(str(REPO_DIR))

from feature_engineering import build_feature_table, prepare_modeling_data
from train_baseline_model import (
    DATA_PATH,
    HELD_OUT_LABEL,
    N_SPLITS,
    RANDOM_STATE,
    evaluate_predictions,
    load_data,
    model_definitions,
    model_input,
)


OUTPUT_DIR = WORK_DIR / "analysis" / "modeling_outputs"
MODEL_NAME = "hist_gradient_boosting"
BOOTSTRAP_ITERATIONS = 100
TOP_SELECTION_COUNT = 20
PERMUTATION_FEATURE_COUNT = 25


def load_modeling_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    data = load_data()
    features, target, groups, binary_data = prepare_modeling_data(data)
    return data, features, target, groups, binary_data, binary_data["Sequence"]


def stratified_bootstrap(train_index: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positives = train_index[y[train_index] == 1]
    negatives = train_index[y[train_index] == 0]
    sampled_positive = rng.choice(positives, size=len(positives), replace=True)
    sampled_negative = rng.choice(negatives, size=len(negatives), replace=True)
    sampled = np.concatenate([sampled_positive, sampled_negative])
    rng.shuffle(sampled)
    return sampled


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    return pd.Series(-scores).rank(method="average").to_numpy()


def feature_stability(features: pd.DataFrame, target: pd.Series, groups: pd.Series) -> pd.DataFrame:
    y = target.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_index, valid_index = next(splitter.split(features, y, groups=groups))
    records: list[pd.DataFrame] = []
    rng = np.random.default_rng(RANDOM_STATE)

    for iteration in range(1, BOOTSTRAP_ITERATIONS + 1):
        sampled_index = stratified_bootstrap(train_index, y, rng)
        model = RandomForestClassifier(
            n_estimators=90,
            max_depth=7,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE + iteration,
        )
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
        pipeline.fit(features.iloc[sampled_index], y[sampled_index])
        rf_scores = pipeline.named_steps["model"].feature_importances_
        rf_rank = ranks_from_scores(rf_scores)
        permutation_scores = np.full(len(features.columns), np.nan)
        valid_frame = features.iloc[valid_index].copy()
        baseline_score = average_precision_score(y[valid_index], pipeline.predict_proba(valid_frame)[:, 1])
        top_feature_indices = np.argsort(rf_scores)[::-1][:PERMUTATION_FEATURE_COUNT]
        for feature_index in top_feature_indices:
            permuted = valid_frame.copy()
            feature_name = features.columns[feature_index]
            permuted[feature_name] = rng.permutation(permuted[feature_name].to_numpy())
            permuted_score = average_precision_score(y[valid_index], pipeline.predict_proba(permuted)[:, 1])
            permutation_scores[feature_index] = baseline_score - permuted_score
        permutation_rank = np.full(len(features.columns), len(features.columns), dtype=float)
        permutation_rank[top_feature_indices] = ranks_from_scores(permutation_scores[top_feature_indices])
        combined_rank = (rf_rank + permutation_rank) / 2
        records.append(
            pd.DataFrame(
                {
                    "feature": features.columns,
                    "iteration": iteration,
                    "rf_rank": rf_rank,
                    "permutation_rank": permutation_rank,
                    "combined_rank": combined_rank,
                    "selected": combined_rank <= TOP_SELECTION_COUNT,
                }
            )
        )
        if iteration % 10 == 0:
            print(f"feature_stability_iteration={iteration}/{BOOTSTRAP_ITERATIONS}", flush=True)

    all_records = pd.concat(records, ignore_index=True)
    stability = (
        all_records.groupby("feature", as_index=False)
        .agg(
            mean_rank=("combined_rank", "mean"),
            rank_sd=("combined_rank", "std"),
            selection_frequency=("selected", "mean"),
            rf_mean_rank=("rf_rank", "mean"),
            permutation_mean_rank=("permutation_rank", "mean"),
        )
        .sort_values(["selection_frequency", "mean_rank"], ascending=[False, True])
    )
    stability.to_csv(OUTPUT_DIR / "feature_stability.csv", index=False)
    all_records.to_csv(OUTPUT_DIR / "feature_stability_bootstrap_details.csv", index=False)
    return stability


def selected_feature_names(stability: pd.DataFrame, count: int) -> list[str]:
    return stability.sort_values(["selection_frequency", "mean_rank"], ascending=[False, True]).head(count)["feature"].tolist()


def evaluate_reduced_model(
    model_name: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
) -> dict[str, float | int | str]:
    y = target.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    prediction_frames: list[pd.DataFrame] = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(features, y, groups=groups), start=1):
        model = clone(pipeline)
        model.fit(features.iloc[train_index], y[train_index])
        scores = model.predict_proba(features.iloc[valid_index])[:, 1]
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "row_index": valid_index,
                    "y_true": y[valid_index],
                    "risk_score": scores,
                }
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return evaluate_predictions(model_name, "overall_oof", predictions["y_true"].to_numpy(), predictions["risk_score"].to_numpy())


def simplified_model_comparison(features: pd.DataFrame, target: pd.Series, groups: pd.Series, stability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    model_specs = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=160,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    for feature_count in (5, 10, 20):
        feature_names = selected_feature_names(stability, feature_count)
        subset = features[feature_names]
        for model_label, pipeline in model_specs.items():
            row = evaluate_reduced_model(f"{model_label}_top_{feature_count}", pipeline, subset, target, groups)
            row["feature_count"] = feature_count
            row["model_family"] = model_label
            row["features"] = ";".join(feature_names)
            rows.append(row)
    full_metrics = pd.read_csv(OUTPUT_DIR / "cv_metrics.csv")
    full_row = full_metrics[(full_metrics["model"].eq(MODEL_NAME)) & (full_metrics["fold"].eq("overall_oof"))].iloc[0].to_dict()
    full_row["feature_count"] = int(features.shape[1])
    full_row["model_family"] = "hist_gradient_boosting_full"
    full_row["features"] = "all_engineered_features"
    rows.append(full_row)
    comparison = pd.DataFrame(rows)
    full_ap = float(full_row["average_precision"])
    comparison["ap_relative_to_full"] = comparison["average_precision"] / full_ap if full_ap else np.nan
    comparison = comparison.sort_values("average_precision", ascending=False)
    comparison.to_csv(OUTPUT_DIR / "simplified_model_comparison.csv", index=False)
    return comparison


def selected_model_oof(binary_data: pd.DataFrame) -> pd.DataFrame:
    predictions_path = OUTPUT_DIR / "oof_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    selected = predictions[predictions["model"].eq(MODEL_NAME)].copy()
    selected["row_index"] = selected["row_index"].astype(int)
    metadata = binary_data[["Ref ID", "Sequence", "Pass/Fail"]].reset_index(drop=True)
    selected = selected.merge(metadata, left_on="row_index", right_index=True, how="left")
    return selected


def feature_summary_table(frame: pd.DataFrame, feature_names: list[str], label: str) -> str:
    rows = []
    for feature in feature_names:
        rows.append(
            {
                "feature": feature,
                label: frame[feature].mean() if feature in frame else np.nan,
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")


def write_error_reports(binary_data: pd.DataFrame, features: pd.DataFrame, stability: pd.DataFrame) -> None:
    predictions = selected_model_oof(binary_data)
    annotated = pd.concat([predictions.reset_index(drop=True), features.iloc[predictions["row_index"].to_numpy()].reset_index(drop=True)], axis=1)
    annotated["predicted_fail_threshold_0_5"] = annotated["risk_score"] >= 0.5
    false_negatives = annotated[annotated["y_true"].eq(1) & ~annotated["predicted_fail_threshold_0_5"]].sort_values("risk_score")
    true_positives = annotated[annotated["y_true"].eq(1) & annotated["predicted_fail_threshold_0_5"]].sort_values("risk_score", ascending=False)
    false_positives = annotated[annotated["y_true"].eq(0) & annotated["predicted_fail_threshold_0_5"]].sort_values("risk_score", ascending=False)
    high_risk_passes = annotated[annotated["y_true"].eq(0)].sort_values("risk_score", ascending=False).head(20)
    top_features = selected_feature_names(stability, 10)

    fn_rows = false_negatives[["Ref ID", "risk_score", "Sequence"]].head(20).to_markdown(index=False, floatfmt=".4f")
    fp_rows = false_positives[["Ref ID", "risk_score", "Sequence"]].head(20).to_markdown(index=False, floatfmt=".4f")
    high_risk_pass_rows = high_risk_passes[["Ref ID", "risk_score", "Sequence"]].to_markdown(index=False, floatfmt=".4f")
    fn_feature_means = feature_summary_table(false_negatives, top_features, "false_negative_mean")
    tp_feature_means = feature_summary_table(true_positives, top_features, "true_positive_mean")
    fp_feature_means = feature_summary_table(false_positives, top_features, "false_positive_mean")

    fn_lines = [
        "# False Negative Report",
        "",
        f"Model: `{MODEL_NAME}` out-of-fold predictions at threshold 0.5.",
        f"False negatives: {len(false_negatives)} of {int(annotated['y_true'].sum())} failures.",
        "",
        "## Lowest-Risk Missed Failures",
        "",
        fn_rows,
        "",
        "## Top Stable Feature Means in False Negatives",
        "",
        fn_feature_means,
        "",
        "## Top Stable Feature Means in True Positives",
        "",
        tp_feature_means,
    ]
    fp_lines = [
        "# False Positive Report",
        "",
        f"Model: `{MODEL_NAME}` out-of-fold predictions at threshold 0.5.",
        f"False positives: {len(false_positives)} of {int((annotated['y_true'] == 0).sum())} passes.",
        "",
        "## False Positives at Threshold 0.5",
        "",
        fp_rows if len(false_positives) else "No pass sequences crossed the 0.5 threshold.",
        "",
        "## Highest-Risk Passes",
        "",
        high_risk_pass_rows,
        "",
        "## Top Stable Feature Means in False Positives",
        "",
        fp_feature_means if len(false_positives) else "No threshold-defined false positives were available for feature summaries.",
    ]
    (OUTPUT_DIR / "false_negative_report.md").write_text("\n".join(fn_lines) + "\n")
    (OUTPUT_DIR / "false_positive_report.md").write_text("\n".join(fp_lines) + "\n")


def write_needs_review_predictions(data: pd.DataFrame, features: pd.DataFrame, target: pd.Series, binary_data: pd.DataFrame, sequences: pd.Series) -> None:
    model_spec = model_definitions()[MODEL_NAME]
    model = clone(model_spec.pipeline)
    full_input = model_input(features, sequences, model_spec)
    y = target.to_numpy()
    model.fit(full_input, y)
    binary_scores = model.predict_proba(full_input)[:, 1]
    held_out = data[data["Pass/Fail"].eq(HELD_OUT_LABEL)].copy()
    if held_out.empty:
        pd.DataFrame(columns=["Ref ID", "Sequence", "predicted_risk", "percentile_rank"]).to_csv(
            OUTPUT_DIR / "needs_review_predictions.csv",
            index=False,
        )
        return
    held_out_features = build_feature_table(held_out)
    held_out_input = model_input(held_out_features, held_out["Sequence"], model_spec)
    held_out_scores = model.predict_proba(held_out_input)[:, 1]
    predictions = held_out[["Ref ID", "Sequence"]].copy()
    predictions["predicted_risk"] = held_out_scores
    predictions["percentile_rank"] = [float(np.mean(binary_scores <= score) * 100) for score in held_out_scores]
    predictions = predictions.sort_values("predicted_risk", ascending=False)
    predictions.to_csv(OUTPUT_DIR / "needs_review_predictions.csv", index=False)


def write_risk_tiering(binary_data: pd.DataFrame) -> None:
    predictions = selected_model_oof(binary_data).sort_values("risk_score", ascending=False).reset_index(drop=True)
    baseline = predictions["y_true"].mean()
    rank_fraction = (np.arange(len(predictions)) + 1) / len(predictions)
    predictions["risk_tier"] = np.select(
        [rank_fraction <= 0.05, rank_fraction <= 0.20],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )
    summary = (
        predictions.groupby("risk_tier", sort=False)
        .agg(
            rows=("y_true", "size"),
            failures=("y_true", "sum"),
            observed_failure_rate=("y_true", "mean"),
            mean_predicted_risk=("risk_score", "mean"),
            min_predicted_risk=("risk_score", "min"),
            max_predicted_risk=("risk_score", "max"),
        )
        .reset_index()
    )
    summary["enrichment_over_baseline"] = summary["observed_failure_rate"] / baseline
    summary.to_csv(OUTPUT_DIR / "risk_tiering.csv", index=False)
    lines = [
        "# Risk Tiering Framework",
        "",
        f"Baseline observed failure rate: {baseline:.4f}.",
        "",
        "| Tier | Definition | Rows | Failures | Observed failure rate | Mean predicted risk | Enrichment |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    definitions = {
        "High Risk": "Top 5% highest predicted risk",
        "Medium Risk": "Next 15% highest predicted risk",
        "Low Risk": "Remaining 80%",
    }
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.risk_tier} | {definitions[row.risk_tier]} | {row.rows} | {int(row.failures)} | "
            f"{row.observed_failure_rate:.4f} | {row.mean_predicted_risk:.4f} | {row.enrichment_over_baseline:.2f}x |"
        )
    (OUTPUT_DIR / "risk_tiering.md").write_text("\n".join(lines) + "\n")


def stable_feature_lookup(stability: pd.DataFrame) -> pd.DataFrame:
    return stability.set_index("feature", drop=False)


def write_design_rules(stability: pd.DataFrame) -> None:
    rankings_path = OUTPUT_DIR / "feature_rankings.csv"
    rankings = pd.read_csv(rankings_path) if rankings_path.exists() else pd.DataFrame(columns=["feature"])
    merged = rankings.merge(stability, on="feature", how="outer")
    candidates = [
        ("Elevated LNA-A burden", ["token_lA_count", "token_lA_fraction", "longest_lA_run"]),
        ("Elevated LNA-T burden", ["token_lT_count", "token_lT_fraction", "longest_lT_run"]),
        ("Longer LNA runs", ["longest_lA_run", "longest_lT_run", "longest_lC_run"]),
        ("Repeated dinucleotide structure", ["repeated_dinucleotide_count"]),
        ("Base-composition shifts", ["base_U_fraction", "base_G_fraction", "base_C_fraction", "pyrimidine_fraction"]),
        ("Phosphorothioate positional spread", ["star_position_std"]),
    ]
    lines = [
        "# Candidate Design Rules",
        "",
        "Rules are derived from univariate feature rankings, random-forest/permutation stability, and current model behavior. They should be used for review prioritization, not hard rejection.",
        "",
    ]
    for rule_name, feature_names in candidates:
        evidence = merged[merged["feature"].isin(feature_names)].copy()
        max_selection = evidence["selection_frequency"].max(skipna=True) if "selection_frequency" in evidence else np.nan
        min_p = evidence["mannwhitney_p"].min(skipna=True) if "mannwhitney_p" in evidence else np.nan
        if pd.notna(max_selection) and max_selection >= 0.70 and pd.notna(min_p) and min_p < 0.001:
            confidence = "High"
        elif (pd.notna(max_selection) and max_selection >= 0.40) or (pd.notna(min_p) and min_p < 0.001):
            confidence = "Moderate"
        else:
            confidence = "Exploratory"
        lines.extend(
            [
                f"## {rule_name}",
                "",
                f"Confidence: {confidence}",
                "",
                "Supporting features:",
                "",
                evidence[
                    [
                        column
                        for column in [
                            "feature",
                            "mannwhitney_p",
                            "mean_difference_fail_minus_pass",
                            "cohen_d",
                            "selection_frequency",
                            "mean_rank",
                        ]
                        if column in evidence.columns
                    ]
                ].to_markdown(index=False, floatfmt=".4g"),
                "",
            ]
        )
    (OUTPUT_DIR / "design_rules.md").write_text("\n".join(lines) + "\n")


def write_shap_unavailable_note() -> None:
    lines = [
        "# SHAP Feature Summary",
        "",
        "SHAP-based interpretation was not generated because the `shap` package is not installed in the configured conda environment.",
        "",
        "No new packages were installed, per project instructions. Use `feature_stability.csv`, `design_rules.md`, and the selected model's existing feature rankings as the current interpretation evidence until SHAP is available.",
    ]
    (OUTPUT_DIR / "shap_feature_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, features, target, groups, binary_data, sequences = load_modeling_inputs()
    stability = feature_stability(features, target, groups)
    simplified_model_comparison(features, target, groups, stability)
    write_error_reports(binary_data, features, stability)
    write_needs_review_predictions(data, features, target, binary_data, sequences)
    write_risk_tiering(binary_data)
    write_design_rules(stability)
    write_shap_unavailable_note()
    print(f"feature_stability_rows={len(stability)}")
    print(f"saved_outputs={OUTPUT_DIR.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()