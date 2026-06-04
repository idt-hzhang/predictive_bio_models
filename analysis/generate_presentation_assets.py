from __future__ import annotations

from pathlib import Path

import pandas as pd


WORK_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORK_DIR / "analysis" / "modeling_outputs"
PRESENTATION_SUMMARY = OUTPUT_DIR / "presentation_summary.md"
ASSET_MANIFEST = OUTPUT_DIR / "presentation_asset_manifest.csv"


def rel(path: Path) -> str:
    return str(path.relative_to(WORK_DIR))


def metric_row(path: Path, filters: dict[str, str]) -> pd.Series:
    frame = pd.read_csv(path)
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column].astype(str).eq(value)
    matches = frame[mask]
    if matches.empty:
        raise ValueError(f"No row in {path} matching {filters}")
    return matches.iloc[0]


def top_feature_rows(path: Path, count: int = 10) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.head(count)[
        [
            "feature",
            "mean_difference_fail_minus_pass",
            "median_difference_fail_minus_pass",
            "mannwhitney_p",
            "rank_biserial",
            "cohen_d",
        ]
    ]


def build_manifest() -> pd.DataFrame:
    assets = [
        ("figure_index", "analysis/feature_target_plots/index.md", "Engineered feature vs Pass/Fail figures; Pass/Fail is x-axis and feature value is y-axis."),
        ("figure_index", "analysis/all_data_qc_feature_boxplots/index.md", "Raw QC/manufacturing feature plots by Pass/Fail."),
        ("figure", "analysis/modeling_outputs/learning_curve.png", "Learning-curve figure for training-set-size discussion."),
        ("figure", "analysis/modeling_outputs/mainpeak_regression_observed_vs_predicted.png", "Observed versus predicted MainPeak regression figure."),
        ("table", "analysis/modeling_outputs/cv_metrics.csv", "Full-feature Pass/Fail classifier metrics."),
        ("table", "analysis/modeling_outputs/correlation_reduced_model_comparison.csv", "Full versus 71-feature correlation-pruned Pass/Fail comparison."),
        ("table", "analysis/modeling_outputs/need_review_model_comparison.csv", "Full versus correlation-pruned Need_Review comparison."),
        ("table", "analysis/modeling_outputs/mainpeak_regression_metrics.csv", "Full-feature MainPeak regression metrics."),
        ("table", "analysis/modeling_outputs/mainpeak_reduced_feature_regression_metrics.csv", "Top-40 reduced-feature MainPeak metrics."),
        ("table", "analysis/modeling_outputs/narrow_feature_panel_selected_features.csv", "Revised 9-feature compact cross-endpoint panel."),
        ("table", "analysis/modeling_outputs/feature_rankings.csv", "Statistical and model-derived feature ranking table."),
        ("table", "analysis/modeling_outputs/risk_tiering.csv", "Risk tier enrichment table."),
        ("report", "analysis/modeling_outputs/evaluation_report.md", "Main Pass/Fail evaluation report."),
        ("report", "analysis/modeling_outputs/need_review_modeling_report.md", "Need_Review model report."),
        ("report", "analysis/modeling_outputs/mainpeak_regression_report.md", "MainPeak regression and derived classifier report."),
        ("report", "analysis/modeling_outputs/mainpeak_reduced_feature_report.md", "Reduced-feature MainPeak report."),
        ("report", "analysis/modeling_outputs/narrow_feature_panel_modeling_report.md", "9-feature narrow-panel report."),
        ("report", "sgRNA_synthesizability/.github/feature_selection_recommendation.md", "Feature-selection recommendations for next-step decisions."),
        ("report", "sgRNA_synthesizability/README.md", "Project-level narrative summary."),
    ]
    rows = []
    for asset_type, relative_path, presentation_use in assets:
        path = WORK_DIR / relative_path
        rows.append(
            {
                "asset_type": asset_type,
                "path": relative_path,
                "exists": path.exists(),
                "presentation_use": presentation_use,
            }
        )
    return pd.DataFrame(rows)


def write_summary() -> None:
    pass_fail = metric_row(OUTPUT_DIR / "cv_metrics.csv", {"model": "hist_gradient_boosting", "fold": "overall_oof"})
    pass_fail_reduced = metric_row(
        OUTPUT_DIR / "correlation_reduced_model_comparison.csv",
        {"base_model": "hist_gradient_boosting"},
    )
    need_review = metric_row(
        OUTPUT_DIR / "need_review_model_comparison.csv",
        {"feature_set": "correlation_reduced", "model": "correlation_reduced_hist_gradient_boosting", "fold": "overall_oof"},
    )
    mainpeak = metric_row(OUTPUT_DIR / "mainpeak_regression_metrics.csv", {"model": "extra_trees", "fold": "overall_oof"})
    mainpeak_reduced = metric_row(
        OUTPUT_DIR / "mainpeak_reduced_feature_regression_metrics.csv",
        {"selector": "extra_trees_top_40", "model": "extra_trees", "fold": "overall_oof"},
    )
    narrow_pass_fail = metric_row(
        OUTPUT_DIR / "narrow_feature_panel_classification_metrics.csv",
        {"endpoint": "pass_fail", "model": "logistic_l1_balanced", "fold": "overall_oof"},
    )
    narrow_need_review = metric_row(
        OUTPUT_DIR / "narrow_feature_panel_classification_metrics.csv",
        {"endpoint": "need_review", "model": "hist_gradient_boosting", "fold": "overall_oof"},
    )
    narrow_mainpeak = metric_row(
        OUTPUT_DIR / "narrow_feature_panel_regression_metrics.csv",
        {"model": "hist_gradient_boosting", "fold": "overall_oof"},
    )
    top_features = top_feature_rows(OUTPUT_DIR / "feature_rankings.csv")
    manifest = build_manifest()
    manifest.to_csv(ASSET_MANIFEST, index=False)

    missing = manifest.loc[~manifest["exists"], "path"].tolist()
    missing_text = "None found among required presentation assets." if not missing else "\n".join(f"- `{path}`" for path in missing)
    lines = [
        "# Presentation Summary",
        "",
        "This file is a presentation-ready map of the current hackathon deliverables. It points to the figures, tables, and reports that summarize the sgRNA synthesizability modeling workflow.",
        "",
        "## Core Results",
        "",
        "| Endpoint | Recommended artifact/model | Key performance | Presentation point |",
        "| --- | --- | --- | --- |",
        f"| Source Pass/Fail | 71-feature correlation-pruned HGB | PR-AUC {pass_fail_reduced.pr_auc_reduced:.4f}, ROC AUC {pass_fail_reduced.roc_auc_reduced:.4f}, Precision@5% {pass_fail_reduced.precision_top_05_reduced:.4f} | Best current rare-fail ranking model; use for risk prioritization rather than hard calls. |",
        f"| Source Pass/Fail full-feature baseline | Full 161-feature HGB | PR-AUC {pass_fail.pr_auc:.4f}, ROC AUC {pass_fail.roc_auc:.4f}, Precision@5% {pass_fail.precision_top_05:.4f} | Baseline model before redundancy pruning. |",
        f"| Formula-derived Need_Review | 71-feature correlation-pruned HGB | PR-AUC {need_review.pr_auc:.4f}, ROC AUC {need_review.roc_auc:.4f}, F1 {need_review.f1_score:.4f}, Precision@5% {need_review.precision_top_05:.4f} | Need_Review is much more learnable than source Fail. |",
        f"| MainPeak regression | Full-feature ExtraTrees | RMSE {mainpeak.rmse:.4f}, R2 {mainpeak.r2:.4f}, Pearson r {mainpeak.pearson_r:.4f} | Continuous purity signal is strong from sequence-derived features. |",
        f"| Reduced-feature MainPeak | ExtraTrees top-40 + ExtraTrees | RMSE {mainpeak_reduced.rmse:.4f}, R2 {mainpeak_reduced.r2:.4f} | Nearly preserves full-feature MainPeak performance with about one quarter of the features. |",
        f"| 9-feature compact panel | Shared narrow panel | Pass/Fail PR-AUC {narrow_pass_fail.pr_auc:.4f}; Need_Review PR-AUC {narrow_need_review.pr_auc:.4f}; MainPeak RMSE {narrow_mainpeak.rmse:.4f} | Best lightweight cross-endpoint interpretation set. |",
        "",
        "## Feature Figures",
        "",
        "- Engineered feature plots: `analysis/feature_target_plots/index.md`. Each generated figure uses `Pass/Fail` on the x-axis and feature value on the y-axis.",
        "- Raw QC/manufacturing plots: `analysis/all_data_qc_feature_boxplots/index.md`. Numeric figures also use `Pass/Fail` on the x-axis and raw feature value on the y-axis.",
        "- Suggested first feature slides: LNA burden/run features (`token_lT_count`, `token_lA_count`, `longest_lT_run`), RNA/composition features (`rna_count_middle`, `base_U_count`), and positional chemistry features (`modified_token_position_std`, `star_position_std`).",
        "",
        "## Top Statistical Feature Signals",
        "",
        top_features.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Required Presentation Asset Manifest",
        "",
        f"Manifest CSV: `{rel(ASSET_MANIFEST)}`",
        "",
        manifest.to_markdown(index=False),
        "",
        "## Missing Assets Check",
        "",
        missing_text,
        "",
        "## Recommended Slide Order",
        "",
        "1. Problem and data imbalance: 86 Fail rows among 2,646 binary-labeled rows.",
        "2. Feature engineering overview: 161 decorated-sequence features plus parser handling of `+` LNA notation.",
        "3. Feature evidence: Pass/Fail box/violin plots and top feature ranking table.",
        "4. Pass/Fail model comparison: full versus 71-feature correlation-pruned HGB.",
        "5. Need_Review modeling: stronger formula-derived endpoint and its relationship to MainPeak.",
        "6. MainPeak regression: continuous purity is predictable; top-40 reduced model is nearly as strong as full feature set.",
        "7. Narrow feature panels: 3/5/9-feature interpretation tradeoffs.",
        "8. Operational recommendation: risk-ranking workflow, threshold calibration, and future data needs.",
    ]
    PRESENTATION_SUMMARY.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_summary()
    print(f"wrote={rel(PRESENTATION_SUMMARY)}")
    print(f"wrote={rel(ASSET_MANIFEST)}")


if __name__ == "__main__":
    main()