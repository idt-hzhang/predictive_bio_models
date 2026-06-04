from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


WORK_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = WORK_DIR / "data" / "all.cleaned_data.csv"
OUTPUT_DIR = WORK_DIR / "analysis" / "all_data_qc_feature_boxplots"
LABEL_COLUMN = "Pass/Fail"
NUMERIC_FEATURES = [
    "Sample OD",
    "Sample Volume (uL)",
    "P=S Bonds",
    "OMe Bases",
    "Length",
    "UHPLC % Pre-MainPeak",
    "UHPLC % MainPeak",
    "UHPLC % Post-MainPeak",
    "ESI %",
    "Ship Ods",
    "Ship Nmoles",
    "Yield Guarantee (ODs)",
]
CATEGORICAL_FEATURES = ["ESI Status"]


def safe_filename(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name)
    return f"{safe}.png"


def cohen_d(pass_values: np.ndarray, fail_values: np.ndarray) -> float:
    if len(pass_values) < 2 or len(fail_values) < 2:
        return float("nan")
    pooled_variance = (
        ((len(pass_values) - 1) * np.var(pass_values, ddof=1))
        + ((len(fail_values) - 1) * np.var(fail_values, ddof=1))
    ) / (len(pass_values) + len(fail_values) - 2)
    if pooled_variance <= 0:
        return 0.0
    return float((np.mean(fail_values) - np.mean(pass_values)) / np.sqrt(pooled_variance))


def numeric_summary(feature: str, pass_values: np.ndarray, fail_values: np.ndarray, plot_file: str) -> dict[str, float | int | str]:
    try:
        mannwhitney = stats.mannwhitneyu(fail_values, pass_values, alternative="two-sided")
        mannwhitney_p = float(mannwhitney.pvalue)
    except ValueError:
        mannwhitney_p = float("nan")

    return {
        "feature": feature,
        "feature_type": "numeric",
        "plot_file": plot_file,
        "pass_count": int(len(pass_values)),
        "fail_count": int(len(fail_values)),
        "pass_mean": float(np.mean(pass_values)) if len(pass_values) else float("nan"),
        "fail_mean": float(np.mean(fail_values)) if len(fail_values) else float("nan"),
        "pass_median": float(np.median(pass_values)) if len(pass_values) else float("nan"),
        "fail_median": float(np.median(fail_values)) if len(fail_values) else float("nan"),
        "mean_difference_fail_minus_pass": float(np.mean(fail_values) - np.mean(pass_values)) if len(pass_values) and len(fail_values) else float("nan"),
        "mannwhitney_p": mannwhitney_p,
        "cohen_d": cohen_d(pass_values, fail_values),
    }


def plot_numeric_feature(data: pd.DataFrame, feature: str, output_dir: Path) -> dict[str, float | int | str]:
    plot_data = pd.DataFrame(
        {
            "value": pd.to_numeric(data[feature], errors="coerce"),
            "label": data[LABEL_COLUMN],
        }
    ).dropna(subset=["value", "label"])
    pass_values = plot_data.loc[plot_data["label"].eq("Pass"), "value"].to_numpy()
    fail_values = plot_data.loc[plot_data["label"].eq("Fail"), "value"].to_numpy()
    grouped = [pass_values, fail_values]

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.boxplot(grouped, tick_labels=["Pass", "Fail"], showfliers=True, widths=0.5)
    ax.scatter(
        np.random.default_rng(42).normal(1, 0.035, len(pass_values)),
        pass_values,
        color="#2364aa",
        alpha=0.22,
        s=10,
        edgecolors="none",
    )
    ax.scatter(
        np.random.default_rng(43).normal(2, 0.035, len(fail_values)),
        fail_values,
        color="#b84a4a",
        alpha=0.48,
        s=18,
        edgecolors="none",
    )
    ax.set_title(f"{feature} by HPLC QC outcome")
    ax.set_xlabel("Pass/Fail")
    ax.set_ylabel(feature)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / safe_filename(feature)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return numeric_summary(feature, pass_values, fail_values, str(output_path.relative_to(WORK_DIR)))


def plot_categorical_feature(data: pd.DataFrame, feature: str, output_dir: Path) -> dict[str, float | int | str]:
    plot_data = data[[LABEL_COLUMN, feature]].dropna().copy()
    counts = pd.crosstab(plot_data[feature], plot_data[LABEL_COLUMN]).reindex(columns=["Pass", "Fail"], fill_value=0)
    proportions = counts.div(counts.sum(axis=0), axis=1).fillna(0.0)

    fig, (count_ax, proportion_ax) = plt.subplots(1, 2, figsize=(9.2, 4.8))
    counts.plot(kind="bar", ax=count_ax, color=["#2364aa", "#b84a4a"], width=0.72)
    count_ax.set_title("Counts")
    count_ax.set_xlabel(feature)
    count_ax.set_ylabel("Rows")
    count_ax.tick_params(axis="x", rotation=0)
    count_ax.grid(axis="y", alpha=0.25)

    proportions.plot(kind="bar", ax=proportion_ax, color=["#2364aa", "#b84a4a"], width=0.72)
    proportion_ax.set_title("Within-class proportions")
    proportion_ax.set_xlabel(feature)
    proportion_ax.set_ylabel("Proportion")
    proportion_ax.tick_params(axis="x", rotation=0)
    proportion_ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{feature} by HPLC QC outcome", y=1.02)
    fig.tight_layout()

    output_path = output_dir / safe_filename(feature)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "feature": feature,
        "feature_type": "categorical",
        "plot_file": str(output_path.relative_to(WORK_DIR)),
        "pass_count": int(counts["Pass"].sum()),
        "fail_count": int(counts["Fail"].sum()),
        "pass_mean": float("nan"),
        "fail_mean": float("nan"),
        "pass_median": float("nan"),
        "fail_median": float("nan"),
        "mean_difference_fail_minus_pass": float("nan"),
        "mannwhitney_p": float("nan"),
        "cohen_d": float("nan"),
    }


def write_combined_numeric_plot(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 11.5))
    axes = axes.ravel()
    rng = np.random.default_rng(42)
    for ax, feature in zip(axes, NUMERIC_FEATURES):
        values = pd.to_numeric(data[feature], errors="coerce")
        pass_values = values[data[LABEL_COLUMN].eq("Pass")].dropna().to_numpy()
        fail_values = values[data[LABEL_COLUMN].eq("Fail")].dropna().to_numpy()
        ax.boxplot([pass_values, fail_values], tick_labels=["Pass", "Fail"], showfliers=True, widths=0.5)
        ax.scatter(rng.normal(1, 0.035, len(pass_values)), pass_values, color="#2364aa", alpha=0.10, s=6, edgecolors="none")
        ax.scatter(rng.normal(2, 0.035, len(fail_values)), fail_values, color="#b84a4a", alpha=0.42, s=12, edgecolors="none")
        ax.set_title(feature, fontsize=10)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Raw QC numeric features by HPLC QC outcome", y=1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "all_numeric_qc_boxplots.png", dpi=180)
    plt.close(fig)


def write_index(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# All Cleaned Data QC Feature Plots",
        "",
        "Plots compare `Pass` and `Fail` rows from `data/all.cleaned_data.csv`. Numeric features are shown as boxplots with jittered points. `ESI Status` is categorical, so it is shown as count and within-class proportion bar plots.",
        "",
        "Combined numeric panel: [all_numeric_qc_boxplots.png](all_numeric_qc_boxplots.png)",
        "",
        "| Feature | Type | Plot | Pass n | Fail n | Pass median | Fail median | MWU p | Cohen's d |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        plot_name = Path(row.plot_file).name
        lines.append(
            f"| `{row.feature}` | {row.feature_type} | [{plot_name}]({plot_name}) | "
            f"{row.pass_count} | {row.fail_count} | {row.pass_median:.4g} | {row.fail_median:.4g} | "
            f"{row.mannwhitney_p:.3g} | {row.cohen_d:.4g} |"
        )
    (output_dir / "index.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, dtype="string")
    binary_data = data[data[LABEL_COLUMN].isin(["Pass", "Fail"])].copy()
    missing_columns = [feature for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES if feature not in binary_data.columns]
    if missing_columns:
        raise ValueError(f"Missing requested columns: {missing_columns}")

    summaries = [plot_numeric_feature(binary_data, feature, OUTPUT_DIR) for feature in NUMERIC_FEATURES]
    summaries.extend(plot_categorical_feature(binary_data, feature, OUTPUT_DIR) for feature in CATEGORICAL_FEATURES)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "qc_feature_plot_summary.csv", index=False)
    write_combined_numeric_plot(binary_data, OUTPUT_DIR)
    write_index(summary, OUTPUT_DIR)
    print(f"binary_rows={len(binary_data)} pass_rows={(binary_data[LABEL_COLUMN] == 'Pass').sum()} fail_rows={(binary_data[LABEL_COLUMN] == 'Fail').sum()}")
    print(f"numeric_plots={len(NUMERIC_FEATURES)} categorical_plots={len(CATEGORICAL_FEATURES)}")
    print(f"saved_outputs={OUTPUT_DIR.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()