from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from train_mainpeak_regression import N_SPLITS, OUTPUT_DIR, RANDOM_STATE, WORK_DIR, load_mainpeak_data, model_definitions


MAX_FEATURES = 40
CORRELATION_THRESHOLD = 0.90
SELECTORS = ["correlation_pruned_top_40", "mutual_info_top_40", "extra_trees_top_40"]
MODELS = ["mean_baseline", "elastic_net", "hist_gradient_boosting", "extra_trees"]


def impute_train_valid(train_features: pd.DataFrame, valid_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    train_imputed = pd.DataFrame(imputer.fit_transform(train_features), columns=train_features.columns, index=train_features.index)
    valid_imputed = pd.DataFrame(imputer.transform(valid_features), columns=valid_features.columns, index=valid_features.index)
    return train_imputed, valid_imputed


def absolute_target_correlations(features: pd.DataFrame, target: np.ndarray) -> pd.Series:
    scores: dict[str, float] = {}
    target_series = pd.Series(target, index=features.index)
    for column in features.columns:
        if features[column].std(ddof=0) == 0:
            scores[column] = 0.0
        else:
            scores[column] = abs(float(features[column].corr(target_series)))
    return pd.Series(scores).fillna(0.0)


def correlation_pruned_features(train_features: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    target_scores = absolute_target_correlations(train_features, target).sort_values(ascending=False)
    feature_correlations = train_features.corr().abs()
    selected: list[str] = []
    rows: list[dict[str, float | int | str]] = []

    for feature, score in target_scores.items():
        if selected and feature_correlations.loc[feature, selected].max() > CORRELATION_THRESHOLD:
            continue
        selected.append(feature)
        rows.append({"feature": feature, "score": float(score), "rank": len(selected)})
        if len(selected) >= MAX_FEATURES:
            break

    return pd.DataFrame(rows)


def mutual_info_features(train_features: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    scores = mutual_info_regression(train_features, target, random_state=RANDOM_STATE)
    rows = pd.DataFrame({"feature": train_features.columns, "score": scores})
    return rows.sort_values("score", ascending=False).head(MAX_FEATURES).assign(rank=lambda frame: np.arange(1, len(frame) + 1))


def extra_trees_features(train_features: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    model = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(train_features, target)
    rows = pd.DataFrame({"feature": train_features.columns, "score": model.feature_importances_})
    return rows.sort_values("score", ascending=False).head(MAX_FEATURES).assign(rank=lambda frame: np.arange(1, len(frame) + 1))


def select_features(selector_name: str, train_features: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
    if selector_name == "correlation_pruned_top_40":
        return correlation_pruned_features(train_features, target)
    if selector_name == "mutual_info_top_40":
        return mutual_info_features(train_features, target)
    if selector_name == "extra_trees_top_40":
        return extra_trees_features(train_features, target)
    raise ValueError(f"Unknown selector: {selector_name}")


def evaluate_regression(model_name: str, selector_name: str, fold: int | str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | str]:
    pearson = stats.pearsonr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else float("nan")
    spearman = stats.spearmanr(y_true, y_pred).statistic if len(np.unique(y_pred)) > 1 else float("nan")
    return {
        "selector": selector_name,
        "model": model_name,
        "fold": fold,
        "rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearson),
        "spearman_r": float(spearman),
    }


def class_regression_metrics(predictions: pd.DataFrame, modeling_data: pd.DataFrame) -> pd.DataFrame:
    metadata = modeling_data[["Pass/Fail"]].reset_index().rename(columns={"index": "row_index"})
    predictions = predictions.merge(metadata, on="row_index", how="left", validate="many_to_one")
    binary = predictions[predictions["Pass/Fail"].isin(["Pass", "Fail"])].copy()
    rows: list[dict[str, float | int | str]] = []

    for (selector_name, model_name, class_label), group in binary.groupby(["selector", "model", "Pass/Fail"]):
        observed = group["observed_mainpeak"].to_numpy()
        predicted = group["predicted_mainpeak"].to_numpy()
        metric_row = evaluate_regression(model_name, selector_name, "overall_oof", observed, predicted)
        metric_row["class_label"] = class_label
        rows.append(metric_row)

    return pd.DataFrame(rows).sort_values(["selector", "model", "class_label"])


def cross_validate_reduced_models(
    modeling_data: pd.DataFrame,
    features: pd.DataFrame,
    target: np.ndarray,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = GroupKFold(n_splits=N_SPLITS)
    models = model_definitions()
    metric_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    selected_feature_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(splitter.split(features, target, groups=groups), start=1):
        train_features, valid_features = impute_train_valid(features.iloc[train_index], features.iloc[valid_index])
        train_target = target[train_index]
        valid_target = target[valid_index]

        for selector_name in SELECTORS:
            selected = select_features(selector_name, train_features, train_target)
            selected_columns = selected["feature"].tolist()
            selected_feature_frames.append(selected.assign(selector=selector_name, fold=fold, selected_feature_count=len(selected_columns)))

            for model_name in MODELS:
                model = clone(models[model_name])
                model.fit(train_features[selected_columns], train_target)
                predictions = model.predict(valid_features[selected_columns])
                metric_rows.append(evaluate_regression(model_name, selector_name, fold, valid_target, predictions))
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "selector": selector_name,
                            "model": model_name,
                            "fold": fold,
                            "row_index": valid_index,
                            "observed_mainpeak": valid_target,
                            "predicted_mainpeak": predictions,
                        }
                    )
                )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    for (selector_name, model_name), group in predictions.groupby(["selector", "model"]):
        metrics = pd.concat(
            [
                metrics,
                pd.DataFrame(
                    [
                        evaluate_regression(
                            model_name,
                            selector_name,
                            "overall_oof",
                            group["observed_mainpeak"].to_numpy(),
                            group["predicted_mainpeak"].to_numpy(),
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    selected_features = pd.concat(selected_feature_frames, ignore_index=True)
    class_metrics = class_regression_metrics(predictions, modeling_data)
    return metrics, class_metrics, predictions, selected_features


def summarize_selected_features(selected_features: pd.DataFrame) -> pd.DataFrame:
    return (
        selected_features.groupby(["selector", "feature"], as_index=False)
        .agg(selection_count=("fold", "nunique"), mean_rank=("rank", "mean"), mean_score=("score", "mean"))
        .sort_values(["selector", "selection_count", "mean_rank"], ascending=[True, False, True])
    )


def write_report(metrics: pd.DataFrame, class_metrics: pd.DataFrame, selected_summary: pd.DataFrame) -> None:
    overall = metrics[metrics["fold"].eq("overall_oof")].sort_values("rmse")
    best = overall.iloc[0]
    overall_table = overall[["selector", "model", "mae", "rmse", "r2", "pearson_r", "spearman_r"]].to_markdown(index=False, floatfmt=".4f")
    best_class_table = class_metrics[
        class_metrics["selector"].eq(best.selector) & class_metrics["model"].eq(best.model)
    ][["class_label", "rows", "mae", "rmse", "r2", "pearson_r", "spearman_r"]].to_markdown(index=False, floatfmt=".4f")
    top_features_table = selected_summary[selected_summary["selector"].eq(best.selector)].head(20).to_markdown(index=False, floatfmt=".4f")

    lines = [
        "# Reduced-Feature MainPeak Modeling Report",
        "",
        f"Feature selectors were fit inside each {N_SPLITS}-fold grouped CV split to reduce leakage. Each selector retained up to {MAX_FEATURES} features from the 161 sequence-derived features.",
        "",
        "## Overall Out-of-Fold Regression Metrics",
        "",
        overall_table,
        "",
        "## Best Reduced-Feature Model",
        "",
        f"The best reduced-feature model is `{best.model}` with selector `{best.selector}`: RMSE {best.rmse:.4f}, MAE {best.mae:.4f}, R2 {best.r2:.4f}, Pearson r {best.pearson_r:.4f}, and Spearman r {best.spearman_r:.4f}.",
        "",
        "## Pass/Fail Class-Specific Regression Metrics",
        "",
        best_class_table,
        "",
        "Class-specific metrics evaluate MainPeak prediction error separately for rows whose original binary label is `Pass` or `Fail`. These are regression metrics on MainPeak, not derived-classification metrics.",
        "",
        "## Most Frequently Selected Features for Best Selector",
        "",
        top_features_table,
        "",
        "## Outputs",
        "",
        "- `mainpeak_reduced_feature_regression_metrics.csv`",
        "- `mainpeak_reduced_feature_class_metrics.csv`",
        "- `mainpeak_reduced_feature_oof_predictions.csv`",
        "- `mainpeak_reduced_feature_selected_features.csv`",
        "- `mainpeak_reduced_feature_selected_feature_summary.csv`",
        "- `mainpeak_reduced_feature_report.md`",
    ]
    (OUTPUT_DIR / "mainpeak_reduced_feature_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modeling_data, features, target, groups = load_mainpeak_data()
    metrics, class_metrics, predictions, selected_features = cross_validate_reduced_models(modeling_data, features, target, groups)
    selected_summary = summarize_selected_features(selected_features)

    metrics.to_csv(OUTPUT_DIR / "mainpeak_reduced_feature_regression_metrics.csv", index=False)
    class_metrics.to_csv(OUTPUT_DIR / "mainpeak_reduced_feature_class_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "mainpeak_reduced_feature_oof_predictions.csv", index=False)
    selected_features.to_csv(OUTPUT_DIR / "mainpeak_reduced_feature_selected_features.csv", index=False)
    selected_summary.to_csv(OUTPUT_DIR / "mainpeak_reduced_feature_selected_feature_summary.csv", index=False)
    write_report(metrics, class_metrics, selected_summary)

    overall = metrics[metrics["fold"].eq("overall_oof")].sort_values("rmse")
    best = overall.iloc[0]
    print(f"best_selector={best.selector} best_model={best.model} rmse={best.rmse:.4f} r2={best.r2:.4f}")
    print(overall[["selector", "model", "mae", "rmse", "r2"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved_outputs={OUTPUT_DIR.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()