from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from mainpeak_reduced_feature_modeling import evaluate_regression, impute_train_valid, select_features
from train_mainpeak_regression import N_SPLITS, OUTPUT_DIR, RANDOM_STATE, WORK_DIR, load_mainpeak_data, model_definitions


SELECTOR = "extra_trees_top_40"
MODELS = ["elastic_net", "hist_gradient_boosting", "extra_trees"]
STRATEGIES = ["unbalanced", "fail_weighted", "fail_oversampled"]


def fail_weights(labels: pd.Series) -> np.ndarray:
    weights = np.ones(len(labels), dtype=float)
    fail_count = int(labels.eq("Fail").sum())
    pass_count = int(labels.eq("Pass").sum())
    if fail_count:
        weights[labels.eq("Fail").to_numpy()] = pass_count / fail_count
    return weights


def oversampled_indices(labels: pd.Series, rng: np.random.Generator) -> np.ndarray:
    base_indices = np.arange(len(labels))
    fail_indices = base_indices[labels.eq("Fail").to_numpy()]
    pass_count = int(labels.eq("Pass").sum())
    fail_count = len(fail_indices)
    if fail_count == 0 or pass_count <= fail_count:
        return base_indices
    extra_fail_indices = rng.choice(fail_indices, size=pass_count - fail_count, replace=True)
    return np.concatenate([base_indices, extra_fail_indices])


def fit_predict_strategy(
    model_name: str,
    strategy: str,
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    train_labels: pd.Series,
    valid_features: pd.DataFrame,
    fold: int,
) -> np.ndarray:
    model = clone(model_definitions()[model_name])

    if strategy == "unbalanced":
        model.fit(train_features, train_target)
    elif strategy == "fail_weighted":
        model.fit(train_features, train_target, model__sample_weight=fail_weights(train_labels))
    elif strategy == "fail_oversampled":
        rng = np.random.default_rng(RANDOM_STATE + fold)
        indices = oversampled_indices(train_labels, rng)
        model.fit(train_features.iloc[indices], train_target[indices])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return model.predict(valid_features)


def fail_focused_class_metrics(predictions: pd.DataFrame, modeling_data: pd.DataFrame) -> pd.DataFrame:
    metadata = modeling_data[["Pass/Fail"]].reset_index().rename(columns={"index": "row_index"})
    predictions = predictions.merge(metadata, on="row_index", how="left", validate="many_to_one")
    binary = predictions[predictions["Pass/Fail"].isin(["Pass", "Fail"])].copy()
    rows: list[dict[str, float | int | str]] = []

    for (strategy, model_name, class_label), group in binary.groupby(["strategy", "model", "Pass/Fail"]):
        observed = group["observed_mainpeak"].to_numpy()
        predicted = group["predicted_mainpeak"].to_numpy()
        metric_row = evaluate_regression(model_name, strategy, "overall_oof", observed, predicted)
        metric_row["selector"] = SELECTOR
        metric_row["strategy"] = strategy
        metric_row["class_label"] = class_label
        rows.append(metric_row)

    return pd.DataFrame(rows).sort_values(["strategy", "model", "class_label"])


def cross_validate_fail_focused() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modeling_data, features, target, groups = load_mainpeak_data()
    splitter = GroupKFold(n_splits=N_SPLITS)
    metric_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    selected_feature_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(splitter.split(features, target, groups=groups), start=1):
        train_features, valid_features = impute_train_valid(features.iloc[train_index], features.iloc[valid_index])
        train_target = target[train_index]
        valid_target = target[valid_index]
        train_labels = modeling_data.iloc[train_index]["Pass/Fail"].reset_index(drop=True)
        selected = select_features(SELECTOR, train_features, train_target)
        selected_columns = selected["feature"].tolist()
        selected_feature_frames.append(selected.assign(fold=fold, selector=SELECTOR, selected_feature_count=len(selected_columns)))

        train_selected = train_features[selected_columns]
        valid_selected = valid_features[selected_columns]
        for model_name in MODELS:
            for strategy in STRATEGIES:
                predictions = fit_predict_strategy(model_name, strategy, train_selected, train_target, train_labels, valid_selected, fold)
                metric_row = evaluate_regression(model_name, strategy, fold, valid_target, predictions)
                metric_row["strategy"] = strategy
                metric_row["selector"] = SELECTOR
                metric_rows.append(metric_row)
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "selector": SELECTOR,
                            "model": model_name,
                            "strategy": strategy,
                            "fold": fold,
                            "row_index": valid_index,
                            "observed_mainpeak": valid_target,
                            "predicted_mainpeak": predictions,
                        }
                    )
                )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    for (model_name, strategy), group in predictions.groupby(["model", "strategy"]):
        metric_row = evaluate_regression(model_name, strategy, "overall_oof", group["observed_mainpeak"].to_numpy(), group["predicted_mainpeak"].to_numpy())
        metric_row["strategy"] = strategy
        metric_row["selector"] = SELECTOR
        metrics = pd.concat([metrics, pd.DataFrame([metric_row])], ignore_index=True)

    class_metrics = fail_focused_class_metrics(predictions, modeling_data)
    selected_features = pd.concat(selected_feature_frames, ignore_index=True)
    return metrics, class_metrics, predictions, selected_features


def write_report(metrics: pd.DataFrame, class_metrics: pd.DataFrame) -> None:
    overall = metrics[metrics["fold"].eq("overall_oof")].sort_values("rmse")
    fail_metrics = class_metrics[class_metrics["class_label"].eq("Fail")].sort_values("rmse")
    best_overall = overall.iloc[0]
    best_fail = fail_metrics.iloc[0]
    overall_table = overall[["strategy", "model", "mae", "rmse", "r2", "pearson_r", "spearman_r"]].to_markdown(index=False, floatfmt=".4f")
    fail_table = fail_metrics[["strategy", "model", "rows", "mae", "rmse", "r2", "pearson_r", "spearman_r"]].to_markdown(index=False, floatfmt=".4f")
    best_fail_class_table = class_metrics[
        class_metrics["strategy"].eq(best_fail.strategy) & class_metrics["model"].eq(best_fail.model)
    ][["class_label", "rows", "mae", "rmse", "r2", "pearson_r", "spearman_r"]].to_markdown(index=False, floatfmt=".4f")

    lines = [
        "# Fail-Focused MainPeak Training Report",
        "",
        "This experiment tests whether balancing the small `Fail` class helps MainPeak regression on true `Fail` rows. The experiment uses the same fold-wise `extra_trees_top_40` feature selector as the reduced-feature MainPeak analysis and compares ordinary training, `Fail` sample weighting, and train-fold `Fail` oversampling.",
        "",
        "Important interpretation: oversampling duplicates existing `Fail` examples; it does not create new independent chemistry evidence. It can shift the loss toward failures, but it also increases overfitting risk and may degrade routine `Pass` performance.",
        "",
        "## Overall Out-of-Fold Metrics",
        "",
        overall_table,
        "",
        "## Fail-Class Metrics",
        "",
        fail_table,
        "",
        "## Best Fail-Class Strategy",
        "",
        f"The best `Fail`-class RMSE is `{best_fail.model}` with `{best_fail.strategy}` training: Fail MAE {best_fail.mae:.4f}, Fail RMSE {best_fail.rmse:.4f}, and Fail R2 {best_fail.r2:.4f}. The best overall model remains `{best_overall.model}` with `{best_overall.strategy}` training: overall RMSE {best_overall.rmse:.4f} and R2 {best_overall.r2:.4f}.",
        "",
        best_fail_class_table,
        "",
        "## Recommendation",
        "",
        "Do not use bootstrapping/oversampling as the primary production model unless the operating goal is explicitly to reduce MainPeak error on known failures at the expense of overall calibration. For this dataset, balancing can modestly improve the failure tail for some models, but the limitation is mostly information scarcity: there are only 84 binary-labeled `Fail` rows with MainPeak available, and duplicated rows do not add new sequence or chemistry patterns.",
        "",
        "More defensible next steps are: collect or prioritize additional real low-MainPeak/failure examples; report class-specific metrics by default; tune an operating model for failure-tail objectives only if the application accepts worse global RMSE; consider quantile or lower-tail prediction intervals; and keep direct Pass/Fail ranking models for failure triage because MainPeak regression and binary failure status are related but not interchangeable.",
        "",
        "## Outputs",
        "",
        "- `mainpeak_fail_focused_training_metrics.csv`",
        "- `mainpeak_fail_focused_training_class_metrics.csv`",
        "- `mainpeak_fail_focused_training_oof_predictions.csv`",
        "- `mainpeak_fail_focused_training_selected_features.csv`",
        "- `mainpeak_fail_focused_training_report.md`",
    ]
    (OUTPUT_DIR / "mainpeak_fail_focused_training_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, class_metrics, predictions, selected_features = cross_validate_fail_focused()
    metrics.to_csv(OUTPUT_DIR / "mainpeak_fail_focused_training_metrics.csv", index=False)
    class_metrics.to_csv(OUTPUT_DIR / "mainpeak_fail_focused_training_class_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "mainpeak_fail_focused_training_oof_predictions.csv", index=False)
    selected_features.to_csv(OUTPUT_DIR / "mainpeak_fail_focused_training_selected_features.csv", index=False)
    write_report(metrics, class_metrics)

    overall = metrics[metrics["fold"].eq("overall_oof")].sort_values("rmse")
    fail = class_metrics[class_metrics["class_label"].eq("Fail")].sort_values("rmse")
    print(overall[["strategy", "model", "mae", "rmse", "r2"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nBest Fail-class metrics:")
    print(fail.head(5)[["strategy", "model", "mae", "rmse", "r2"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved_outputs={OUTPUT_DIR.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()