from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


WORK_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = WORK_DIR / "sgRNA_synthesizability"
sys.path.append(str(REPO_DIR))

from feature_engineering import DEFAULT_INPUT, load_raw_data, prepare_modeling_data, write_feature_outputs


DEFAULT_FEATURE_DIR = WORK_DIR / "analysis" / "features"
DEFAULT_OUTPUT_DIR = WORK_DIR / "analysis" / "feature_target_plots"


def ensure_feature_outputs(feature_dir: Path) -> None:
    required = ["feature_matrix.csv", "target_vector.csv"]
    if all((feature_dir / name).exists() for name in required):
        return

    data = load_raw_data(DEFAULT_INPUT)
    features, target, groups, modeling_data = prepare_modeling_data(data)
    write_feature_outputs(features, target, groups, modeling_data, feature_dir, DEFAULT_INPUT, False)


def load_features_and_target(feature_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_feature_outputs(feature_dir)
    features = pd.read_csv(feature_dir / "feature_matrix.csv")
    target = pd.read_csv(feature_dir / "target_vector.csv")
    return features, target


def safe_filename(feature_name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in feature_name)
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


def rank_biserial_from_u(u_statistic: float, pass_count: int, fail_count: int) -> float:
    if pass_count == 0 or fail_count == 0:
        return float("nan")
    return float((2 * u_statistic / (pass_count * fail_count)) - 1)


def statistical_tests(pass_values: np.ndarray, fail_values: np.ndarray) -> dict[str, float]:
    if len(pass_values) == 0 or len(fail_values) == 0:
        return {
            "mannwhitney_u": float("nan"),
            "mannwhitney_p": float("nan"),
            "rank_biserial": float("nan"),
            "welch_t": float("nan"),
            "welch_p": float("nan"),
            "ks_statistic": float("nan"),
            "ks_p": float("nan"),
            "cohen_d": float("nan"),
        }

    try:
        mannwhitney = stats.mannwhitneyu(fail_values, pass_values, alternative="two-sided")
        mannwhitney_u = float(mannwhitney.statistic)
        mannwhitney_p = float(mannwhitney.pvalue)
    except ValueError:
        mannwhitney_u = float("nan")
        mannwhitney_p = float("nan")

    if len(np.unique(pass_values)) > 1 or len(np.unique(fail_values)) > 1:
        welch = stats.ttest_ind(fail_values, pass_values, equal_var=False, nan_policy="omit")
        ks = stats.ks_2samp(fail_values, pass_values)
        welch_t = float(welch.statistic)
        welch_p = float(welch.pvalue)
        ks_statistic = float(ks.statistic)
        ks_p = float(ks.pvalue)
    else:
        welch_t = 0.0
        welch_p = 1.0
        ks_statistic = 0.0
        ks_p = 1.0

    return {
        "mannwhitney_u": mannwhitney_u,
        "mannwhitney_p": mannwhitney_p,
        "rank_biserial": rank_biserial_from_u(mannwhitney_u, len(pass_values), len(fail_values)),
        "welch_t": welch_t,
        "welch_p": welch_p,
        "ks_statistic": ks_statistic,
        "ks_p": ks_p,
        "cohen_d": cohen_d(pass_values, fail_values),
    }


def plot_feature(feature_name: str, values: pd.Series, target: pd.DataFrame, output_dir: Path) -> dict[str, float | int | str]:
    plot_data = pd.DataFrame(
        {
            "feature": pd.to_numeric(values, errors="coerce"),
            "target": target["target"],
            "label": target["label"],
        }
    ).dropna(subset=["feature", "target"])

    grouped = [
        plot_data.loc[plot_data["target"].eq(0), "feature"].to_numpy(),
        plot_data.loc[plot_data["target"].eq(1), "feature"].to_numpy(),
    ]

    pass_values = grouped[0]
    fail_values = grouped[1]
    test_results = statistical_tests(pass_values, fail_values)
    pass_median = float(np.median(pass_values)) if len(pass_values) else float("nan")
    fail_median = float(np.median(fail_values)) if len(fail_values) else float("nan")
    pass_mean = float(np.mean(pass_values)) if len(pass_values) else float("nan")
    fail_mean = float(np.mean(fail_values)) if len(fail_values) else float("nan")
    pass_std = float(np.std(pass_values, ddof=1)) if len(pass_values) > 1 else 0.0
    fail_std = float(np.std(fail_values, ddof=1)) if len(fail_values) > 1 else 0.0

    fig, (distribution_ax, box_ax) = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.3, 1.0]})
    unique_values = plot_data["feature"].nunique(dropna=True)
    if unique_values <= 12:
        bins = np.arange(plot_data["feature"].min() - 0.5, plot_data["feature"].max() + 1.5, 1)
    else:
        bins = 30

    distribution_ax.hist(pass_values, bins=bins, alpha=0.55, density=True, label="Pass", color="#3973ac")
    distribution_ax.hist(fail_values, bins=bins, alpha=0.55, density=True, label="Fail", color="#b84a4a")
    distribution_ax.set_title("Distribution")
    distribution_ax.set_xlabel(feature_name)
    distribution_ax.set_ylabel("Density")
    distribution_ax.legend(frameon=False)
    distribution_ax.grid(axis="y", alpha=0.22)

    if len(pass_values) > 1 and len(fail_values) > 1 and unique_values > 2:
        violin = box_ax.violinplot(grouped, showmeans=False, showmedians=True, showextrema=False)
        for body, color in zip(violin["bodies"], ["#3973ac", "#b84a4a"]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.28)
        if "cmedians" in violin:
            violin["cmedians"].set_color("#1f2328")

    box_ax.boxplot(grouped, tick_labels=["Pass", "Fail"], showfliers=False, widths=0.38)
    box_ax.set_title("Box/violin by target")
    box_ax.set_xlabel("Target label")
    box_ax.set_ylabel(feature_name)
    box_ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{feature_name} by HPLC QC outcome", y=1.02)
    fig.text(
        0.01,
        0.01,
        f"Pass n={len(pass_values)}, mean={pass_mean:.3g}, median={pass_median:.3g}, sd={pass_std:.3g}; "
        f"Fail n={len(fail_values)}, mean={fail_mean:.3g}, median={fail_median:.3g}, sd={fail_std:.3g}; "
        f"MWU p={test_results['mannwhitney_p']:.3g}, Welch p={test_results['welch_p']:.3g}, KS p={test_results['ks_p']:.3g}",
        ha="left",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    output_path = output_dir / safe_filename(feature_name)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "feature": feature_name,
        "plot_file": str(output_path.relative_to(WORK_DIR)),
        "pass_count": int(len(pass_values)),
        "fail_count": int(len(fail_values)),
        "pass_mean": pass_mean,
        "fail_mean": fail_mean,
        "pass_median": pass_median,
        "fail_median": fail_median,
        "pass_std": pass_std,
        "fail_std": fail_std,
        "mean_difference_fail_minus_pass": fail_mean - pass_mean,
        "median_difference_fail_minus_pass": fail_median - pass_median,
        **test_results,
    }


def write_index(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Feature vs Target Plots",
        "",
        "Each plot shows target-specific histograms plus box/violin summaries for one feature. The summary table includes Mann-Whitney U, Welch t-test, Kolmogorov-Smirnov, and effect-size statistics.",
        "",
        "| Feature | Plot | Pass mean | Fail mean | Mean diff | MWU p | KS p | Rank-biserial |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        plot_name = Path(row.plot_file).name
        lines.append(
            f"| `{row.feature}` | [{plot_name}]({plot_name}) | "
            f"{row.pass_mean:.4g} | {row.fail_mean:.4g} | {row.mean_difference_fail_minus_pass:.4g} | "
            f"{row.mannwhitney_p:.3g} | {row.ks_p:.3g} | {row.rank_biserial:.4g} |"
        )
    (output_dir / "index.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot each feature against the binary HPLC QC target.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, target = load_features_and_target(args.feature_dir)

    feature_columns = [column for column in features.columns if column != "id"]
    summaries = [plot_feature(column, features[column], target, args.output_dir) for column in feature_columns]
    summary = pd.DataFrame(summaries).sort_values(
        ["mannwhitney_p", "rank_biserial"],
        key=lambda values: values.abs() if values.name == "rank_biserial" else values,
        ascending=[True, False],
    )
    summary.to_csv(args.output_dir / "feature_target_plot_summary.csv", index=False)
    write_index(summary, args.output_dir)

    print(f"features_plotted={len(feature_columns)}")
    print(f"saved_outputs={args.output_dir.relative_to(WORK_DIR)}")


if __name__ == "__main__":
    main()
