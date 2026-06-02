from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from sgRNA_synthesizability.feature_engineering import build_feature_table, prepare_modeling_data
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from feature_engineering import build_feature_table, prepare_modeling_data


WORK_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = WORK_DIR / "data" / "cleaned_data.csv"
OUTPUT_DIR = WORK_DIR / "analysis" / "modeling_outputs"
RANDOM_STATE = 42
N_SPLITS = 5
POSITIVE_LABEL = "Fail"
NEGATIVE_LABEL = "Pass"
HELD_OUT_LABEL = "Needs Review"


@dataclass(frozen=True)
class ModelSpec:
    pipeline: Pipeline
    uses_sequence_text: bool = False


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, dtype="string")


def numeric_preprocessor(scale: bool = True) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return steps


def sequence_text(values: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        return values.iloc[:, 0].fillna("").astype(str)
    return values.fillna("").astype(str)


def model_definitions() -> dict[str, ModelSpec]:
    models: dict[str, ModelSpec] = {
        "logistic_l2_balanced": ModelSpec(Pipeline(
            steps=[
                *numeric_preprocessor(scale=True),
                (
                    "model",
                    LogisticRegression(
                        l1_ratio=0.0,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )),
        "logistic_l1_balanced": ModelSpec(Pipeline(
            steps=[
                *numeric_preprocessor(scale=True),
                (
                    "model",
                    LogisticRegression(
                        l1_ratio=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )),
        "elastic_net_logistic": ModelSpec(Pipeline(
            steps=[
                *numeric_preprocessor(scale=True),
                (
                    "model",
                    LogisticRegression(
                        l1_ratio=0.35,
                        class_weight="balanced",
                        max_iter=10000,
                        solver="saga",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )),
        "linear_svm_rbf_calibrated_probability": ModelSpec(Pipeline(
            steps=[
                *numeric_preprocessor(scale=True),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=SVC(
                            kernel="rbf",
                            C=0.8,
                            gamma="scale",
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                        cv=3,
                        ensemble=False,
                    ),
                ),
            ]
        )),
        "hist_gradient_boosting": ModelSpec(Pipeline(
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
        )),
        "balanced_random_forest": ModelSpec(Pipeline(
            steps=[
                *numeric_preprocessor(scale=False),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )),
        "elastic_net_logistic_with_char_ngrams": ModelSpec(Pipeline(
            steps=[
                (
                    "features",
                    ColumnTransformer(
                        transformers=[
                            ("numeric", Pipeline(numeric_preprocessor(scale=True)), lambda frame: [column for column in frame.columns if column != "Sequence"]),
                            (
                                "sequence_ngrams",
                                Pipeline(
                                    steps=[
                                        ("to_text", FunctionTransformer(sequence_text, validate=False)),
                                        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=750)),
                                    ]
                                ),
                                ["Sequence"],
                            ),
                        ]
                    ),
                ),
                ("variance", VarianceThreshold()),
                ("select", SelectKBest(score_func=f_classif, k=250)),
                (
                    "model",
                    LogisticRegression(
                        l1_ratio=0.45,
                        class_weight="balanced",
                        max_iter=10000,
                        solver="saga",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ), uses_sequence_text=True),
    }

    if LGBMClassifier is not None:
        models["lightgbm_balanced"] = ModelSpec(Pipeline(
            steps=[
                *numeric_preprocessor(scale=False),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=180,
                        learning_rate=0.035,
                        num_leaves=12,
                        min_child_samples=25,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        verbosity=-1,
                    ),
                ),
            ]
        ))

    if XGBClassifier is not None:
        models["xgboost_weighted"] = ModelSpec(Pipeline(
            steps=[
                *numeric_preprocessor(scale=False),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=180,
                        max_depth=3,
                        learning_rate=0.035,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=8,
                        reg_lambda=2.0,
                        eval_metric="logloss",
                        scale_pos_weight=2560 / 86,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ))

    return models


def standardize(values: pd.Series, train_index: np.ndarray) -> pd.Series:
    train_values = values.iloc[train_index].astype(float)
    scale = train_values.std(ddof=0)
    if scale == 0 or np.isnan(scale):
        scale = 1.0
    return (values.astype(float) - train_values.mean()) / scale


def rule_scores(features: pd.DataFrame, train_index: np.ndarray) -> np.ndarray:
    score = (
        standardize(features["decorated_length"], train_index)
        + standardize(features["star_density"], train_index)
        + standardize(features["slash_mod_count"], train_index)
        + standardize(features["unknown_char_count"], train_index)
        + standardize(features["max_gc_window_10"], train_index)
    )
    min_score = score.min()
    max_score = score.max()
    if max_score == min_score:
        return np.repeat(0.5, len(score))
    return ((score - min_score) / (max_score - min_score)).to_numpy()


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = recall[precision >= min_precision]
    return float(valid.max()) if len(valid) else 0.0


def precision_at_top_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    top_n = max(1, math.ceil(len(y_true) * fraction))
    order = np.argsort(y_score)[::-1][:top_n]
    return float(np.mean(y_true[order]))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def evaluate_predictions(model_name: str, fold: int | str, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int | str]:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "fold": fold,
        "rows": int(len(y_true)),
        "failures": int(y_true.sum()),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "recall_at_precision_25": recall_at_precision(y_true, y_score, 0.25),
        "recall_at_precision_50": recall_at_precision(y_true, y_score, 0.50),
        "recall_at_precision_75": recall_at_precision(y_true, y_score, 0.75),
        "precision_top_01": precision_at_top_fraction(y_true, y_score, 0.01),
        "precision_top_05": precision_at_top_fraction(y_true, y_score, 0.05),
        "precision_top_10": precision_at_top_fraction(y_true, y_score, 0.10),
        "threshold_0_5_tn": int(tn),
        "threshold_0_5_fp": int(fp),
        "threshold_0_5_fn": int(fn),
        "threshold_0_5_tp": int(tp),
    }


def model_input(features: pd.DataFrame, sequences: pd.Series, spec: ModelSpec) -> pd.DataFrame:
    if not spec.uses_sequence_text:
        return features
    model_frame = features.copy()
    model_frame["Sequence"] = sequences.fillna("").astype(str).to_numpy()
    return model_frame


def cross_validate_models(
    features: pd.DataFrame,
    sequences: pd.Series,
    y: np.ndarray,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    metrics: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    models = model_definitions()

    for fold, (train_index, valid_index) in enumerate(splitter.split(features, y, groups=groups), start=1):
        y_train = y[train_index]
        y_valid = y[valid_index]

        prevalence_score = np.repeat(y_train.mean(), len(valid_index))
        metrics.append(evaluate_predictions("prevalence_baseline", fold, y_valid, prevalence_score))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "model": "prevalence_baseline",
                    "row_index": valid_index,
                    "y_true": y_valid,
                    "risk_score": prevalence_score,
                }
            )
        )

        all_rule_scores = rule_scores(features, train_index)
        valid_rule_scores = all_rule_scores[valid_index]
        metrics.append(evaluate_predictions("rule_baseline", fold, y_valid, valid_rule_scores))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "model": "rule_baseline",
                    "row_index": valid_index,
                    "y_true": y_valid,
                    "risk_score": valid_rule_scores,
                }
            )
        )

        for model_name, spec in models.items():
            model = spec.pipeline
            current_input = model_input(features, sequences, spec)
            model.fit(current_input.iloc[train_index], y_train)
            valid_scores = model.predict_proba(current_input.iloc[valid_index])[:, 1]
            metrics.append(evaluate_predictions(model_name, fold, y_valid, valid_scores))
            prediction_frames.append(
                pd.DataFrame(
                    {
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
                    [evaluate_predictions(model_name, "overall_oof", model_predictions["y_true"].to_numpy(), model_predictions["risk_score"].to_numpy())]
                ),
            ],
            ignore_index=True,
        )

    return metrics_df, predictions


def select_best_model(metrics: pd.DataFrame) -> str:
    candidate_metrics = metrics[
        (metrics["fold"] == "overall_oof")
        & metrics["model"].isin(model_definitions().keys())
    ].sort_values(["average_precision", "precision_top_05"], ascending=False)
    return str(candidate_metrics.iloc[0]["model"])


def fit_model_feature_importance(features: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    pipeline.fit(features, y)
    importances = pipeline.named_steps["model"].feature_importances_
    return pd.DataFrame(
        {
            "feature": features.columns,
            "random_forest_importance": importances,
        }
    ).sort_values("random_forest_importance", ascending=False)


def write_combined_feature_rankings(feature_importance: pd.DataFrame) -> None:
    stats_path = WORK_DIR / "analysis" / "feature_target_plots" / "feature_target_plot_summary.csv"
    if not stats_path.exists():
        feature_importance.to_csv(OUTPUT_DIR / "feature_rankings.csv", index=False)
        return

    statistical_summary = pd.read_csv(stats_path)
    ranking_columns = [
        "feature",
        "mannwhitney_p",
        "welch_p",
        "ks_p",
        "rank_biserial",
        "cohen_d",
        "mean_difference_fail_minus_pass",
        "median_difference_fail_minus_pass",
    ]
    available_columns = [column for column in ranking_columns if column in statistical_summary.columns]
    rankings = statistical_summary[available_columns].merge(feature_importance, on="feature", how="outer")
    rankings["mannwhitney_p_rank"] = rankings["mannwhitney_p"].rank(method="min", na_option="bottom")
    rankings["random_forest_importance_rank"] = rankings["random_forest_importance"].rank(
        method="min",
        ascending=False,
        na_option="bottom",
    )
    rankings = rankings.sort_values(["mannwhitney_p_rank", "random_forest_importance_rank"])
    rankings.to_csv(OUTPUT_DIR / "feature_rankings.csv", index=False)


def summarize_features(features: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    summary = features.describe().transpose().reset_index().rename(columns={"index": "feature"})
    correlations = []
    for column in features.columns:
        if features[column].std(ddof=0) == 0:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(features[column], y)[0, 1]))
    summary["correlation_with_fail"] = correlations
    return summary.sort_values("correlation_with_fail", key=lambda values: values.abs(), ascending=False)


def write_report(
    data: pd.DataFrame,
    binary_data: pd.DataFrame,
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    best_model_name: str,
) -> None:
    overall = metrics[metrics["fold"] == "overall_oof"].copy()
    overall = overall.sort_values("average_precision", ascending=False)
    metric_columns = [
        "model",
        "average_precision",
        "roc_auc",
        "brier_score",
        "recall_at_precision_25",
        "precision_top_05",
        "precision_top_10",
    ]
    metric_table = overall[metric_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    lines = [
        "# Baseline Modeling Report",
        "",
        "## Data",
        "",
        f"- Input file: `{DATA_PATH.relative_to(WORK_DIR)}`",
        f"- Total cleaned rows: {len(data)}",
        f"- Binary labeled rows: {len(binary_data)}",
        f"- Fail rows: {int((binary_data['Pass/Fail'] == POSITIVE_LABEL).sum())}",
        f"- Pass rows: {int((binary_data['Pass/Fail'] == NEGATIVE_LABEL).sum())}",
        f"- Held-out Needs Review rows: {int((data['Pass/Fail'] == HELD_OUT_LABEL).sum())}",
        f"- Feature count: {features.shape[1]}",
        "",
        "## Overall Out-of-Fold Metrics",
        "",
        "```text",
        metric_table,
        "```",
        "",
        "## Selected Model",
        "",
        f"Selected model: `{best_model_name}` based on highest out-of-fold average precision among trained model candidates.",
        "",
        "## Notes",
        "",
        "- `Needs Review` rows were excluded from training and model-selection metrics.",
        "- Cross-validation used stratified group folds with decorated `Sequence` as the group to reduce sequence-level leakage.",
        "- Accuracy is intentionally omitted from the headline metrics because the failure class is rare.",
    ]
    (OUTPUT_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    features, target, groups, binary_data = prepare_modeling_data(data)
    sequences = binary_data["Sequence"]
    y = target.to_numpy()

    metrics, predictions = cross_validate_models(features, sequences, y, groups)
    best_model_name = select_best_model(metrics)
    best_model_spec = model_definitions()[best_model_name]
    best_model = best_model_spec.pipeline
    best_model_input = model_input(features, sequences, best_model_spec)
    best_model.fit(best_model_input, y)

    held_out = data[data["Pass/Fail"].eq(HELD_OUT_LABEL)].copy()
    if held_out.empty:
        held_out_predictions = pd.DataFrame(columns=["Ref ID", "Pass/Fail", "Sequence", "failure_risk"])
    else:
        held_out_features = build_feature_table(held_out)
        held_out_input = model_input(held_out_features, held_out["Sequence"], best_model_spec)
        held_out_predictions = held_out[["Ref ID", "Pass/Fail", "Sequence"]].copy()
        held_out_predictions["failure_risk"] = best_model.predict_proba(held_out_input)[:, 1]
        held_out_predictions = held_out_predictions.sort_values("failure_risk", ascending=False)

    metrics.to_csv(OUTPUT_DIR / "cv_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    summarize_features(features, y).to_csv(OUTPUT_DIR / "feature_summary.csv", index=False)
    feature_importance = fit_model_feature_importance(features, y)
    feature_importance.to_csv(OUTPUT_DIR / "model_feature_importance.csv", index=False)
    write_combined_feature_rankings(feature_importance)
    held_out_predictions.to_csv(OUTPUT_DIR / "needs_review_predictions.csv", index=False)
    joblib.dump(best_model, OUTPUT_DIR / "baseline_model.joblib")
    (OUTPUT_DIR / "feature_schema.json").write_text(json.dumps(list(features.columns), indent=2) + "\n")
    write_report(data, binary_data, features, metrics, best_model_name)

    overall = metrics[metrics["fold"].eq("overall_oof")].sort_values("average_precision", ascending=False)
    print(f"binary_rows={len(binary_data)} failures={int(y.sum())} features={features.shape[1]}")
    print(f"best_model={best_model_name}")
    print(overall[["model", "average_precision", "roc_auc", "precision_top_05"]].to_string(index=False))
    print(f"saved_outputs={OUTPUT_DIR.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()
