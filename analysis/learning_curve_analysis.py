from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


WORK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WORK_DIR / "sgRNA_synthesizability"
sys.path.append(str(REPO_DIR))

from feature_engineering import prepare_modeling_data
from train_baseline_model import (
    DATA_PATH,
    N_SPLITS,
    RANDOM_STATE,
    evaluate_predictions,
    load_data,
    model_definitions,
    model_input,
)


OUTPUT_DIR = WORK_DIR / "analysis" / "modeling_outputs"
MODEL_NAME = "hist_gradient_boosting"
TRAINING_FRACTIONS = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)


def sample_training_indices(train_index: np.ndarray, y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return train_index

    sample_size = max(2, math.ceil(len(train_index) * fraction))
    y_train = y[train_index]
    class_counts = np.bincount(y_train, minlength=2)
    if sample_size < len(class_counts) or np.any(class_counts < 2):
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(train_index, size=sample_size, replace=False))

    sampled_index, _ = train_test_split(
        train_index,
        train_size=sample_size,
        stratify=y_train,
        random_state=seed,
    )
    return np.sort(sampled_index)


def run_learning_curve() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_data()
    features, target, groups, binary_data = prepare_modeling_data(data)
    sequences = binary_data["Sequence"]
    y = target.to_numpy()
    model_spec = model_definitions()[MODEL_NAME]
    current_input = model_input(features, sequences, model_spec)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(splitter.split(features, y, groups=groups), start=1):
        for fraction in TRAINING_FRACTIONS:
            sampled_index = sample_training_indices(
                train_index,
                y,
                fraction,
                seed=RANDOM_STATE + fold * 1000 + int(fraction * 100),
            )
            model = clone(model_spec.pipeline)
            model.fit(current_input.iloc[sampled_index], y[sampled_index])
            valid_scores = model.predict_proba(current_input.iloc[valid_index])[:, 1]
            row = evaluate_predictions(MODEL_NAME, fold, y[valid_index], valid_scores)
            row["training_fraction"] = fraction
            row["training_rows"] = int(len(sampled_index))
            row["training_failures"] = int(y[sampled_index].sum())
            rows.append(row)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "training_fraction": fraction,
                        "fold": fold,
                        "row_index": valid_index,
                        "y_true": y[valid_index],
                        "risk_score": valid_scores,
                    }
                )
            )

    fold_metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summaries: list[dict[str, float | int | str]] = []
    for fraction, fraction_predictions in predictions.groupby("training_fraction", sort=True):
        summary = evaluate_predictions(
            MODEL_NAME,
            "overall_oof",
            fraction_predictions["y_true"].to_numpy(),
            fraction_predictions["risk_score"].to_numpy(),
        )
        summary["training_fraction"] = float(fraction)
        summary["training_rows_mean"] = float(
            fold_metrics.loc[fold_metrics["training_fraction"].eq(fraction), "training_rows"].mean()
        )
        summary["training_failures_mean"] = float(
            fold_metrics.loc[fold_metrics["training_fraction"].eq(fraction), "training_failures"].mean()
        )
        summaries.append(summary)

    learning_curve = pd.DataFrame(summaries).sort_values("training_fraction")
    columns = [
        "training_fraction",
        "training_rows_mean",
        "training_failures_mean",
        "rows",
        "failures",
        "average_precision",
        "roc_auc",
        "precision_top_05",
        "precision_top_10",
    ]
    return learning_curve[columns], fold_metrics


def write_plot(learning_curve: pd.DataFrame, output_path: Path) -> None:
    x = learning_curve["training_fraction"] * 100
    fig, left_ax = plt.subplots(figsize=(8.4, 5.2))
    right_ax = left_ax.twinx()

    left_ax.plot(x, learning_curve["average_precision"], marker="o", color="#2364aa", label="Average precision")
    left_ax.plot(x, learning_curve["roc_auc"], marker="o", color="#3da35d", label="ROC AUC")
    right_ax.plot(x, learning_curve["precision_top_05"], marker="s", color="#d95f02", label="Precision@5%")
    right_ax.plot(x, learning_curve["precision_top_10"], marker="s", color="#7b3294", label="Precision@10%")

    left_ax.set_xlabel("Training data used per fold (%)")
    left_ax.set_ylabel("AP / ROC AUC")
    right_ax.set_ylabel("Top-risk-bin precision")
    left_ax.set_title("Learning curve for hist_gradient_boosting")
    left_ax.set_xticks(x)
    left_ax.grid(axis="y", alpha=0.25)
    left_lines, left_labels = left_ax.get_legend_handles_labels()
    right_lines, right_labels = right_ax.get_legend_handles_labels()
    left_ax.legend(left_lines + right_lines, left_labels + right_labels, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_interpretation(learning_curve: pd.DataFrame, output_path: Path) -> None:
    final_row = learning_curve.iloc[-1]
    previous_row = learning_curve.iloc[-2]
    ap_delta = final_row["average_precision"] - previous_row["average_precision"]
    auc_delta = final_row["roc_auc"] - previous_row["roc_auc"]
    top5_delta = final_row["precision_top_05"] - previous_row["precision_top_05"]
    plateau = abs(ap_delta) < 0.01 and abs(auc_delta) < 0.02 and abs(top5_delta) < 0.02
    conclusion = (
        "Performance appears to be near a plateau between 80% and 100% of the available training data, "
        "so future gains are more likely to require better features or clearer labels than simply reusing more of the same data."
        if plateau
        else "Performance still changes materially between 80% and 100% of the available training data, which suggests additional labeled data would theoretically help. Since more labeled failures are not expected, the practical next step is to focus on feature stability, interpretation, and risk-tiering analyses using the existing data."
    )

    table = learning_curve[
        ["training_fraction", "average_precision", "roc_auc", "precision_top_05", "precision_top_10"]
    ].to_markdown(index=False, floatfmt=".4f")
    lines = [
        "# Learning Curve Interpretation",
        "",
        f"Input data: `{DATA_PATH.relative_to(WORK_DIR)}`",
        f"Model: `{MODEL_NAME}`",
        f"Cross-validation: {N_SPLITS}-fold stratified grouped CV using decorated sequence groups.",
        "",
        "## Metrics",
        "",
        table,
        "",
        "## Plateau Assessment",
        "",
        f"From 80% to 100% training data, average precision changed by {ap_delta:+.4f}, ROC AUC by {auc_delta:+.4f}, and Precision@5% by {top5_delta:+.4f}.",
        "",
        conclusion,
        "",
        "Because the failure class is rare, small non-monotonic changes are expected across fractions. The most important signal is whether the full-data point is clearly above the 80% point across ranking metrics.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    learning_curve, fold_metrics = run_learning_curve()
    learning_curve.to_csv(OUTPUT_DIR / "learning_curve.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "learning_curve_fold_metrics.csv", index=False)
    write_plot(learning_curve, OUTPUT_DIR / "learning_curve.png")
    write_interpretation(learning_curve, OUTPUT_DIR / "learning_curve_interpretation.md")
    print(learning_curve.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved_outputs={OUTPUT_DIR.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()