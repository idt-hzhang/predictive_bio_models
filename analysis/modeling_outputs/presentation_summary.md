# Presentation Summary

This file is a presentation-ready map of the current hackathon deliverables. It points to the figures, tables, and reports that summarize the sgRNA synthesizability modeling workflow.

## Core Results

| Endpoint | Recommended artifact/model | Key performance | Presentation point |
| --- | --- | --- | --- |
| Source Pass/Fail | 71-feature correlation-pruned HGB | PR-AUC 0.1139, ROC AUC 0.6529, Precision@5% 0.1353 | Best current rare-fail ranking model; use for risk prioritization rather than hard calls. |
| Source Pass/Fail full-feature baseline | Full 161-feature HGB | PR-AUC 0.1087, ROC AUC 0.6304, Precision@5% 0.1278 | Baseline model before redundancy pruning. |
| Formula-derived Need_Review | 71-feature correlation-pruned HGB | PR-AUC 0.4082, ROC AUC 0.7719, F1 0.3467, Precision@5% 0.4737 | Need_Review is much more learnable than source Fail. |
| MainPeak regression | Full-feature ExtraTrees | RMSE 6.0574, R2 0.7005, Pearson r 0.8370 | Continuous purity signal is strong from sequence-derived features. |
| Reduced-feature MainPeak | ExtraTrees top-40 + ExtraTrees | RMSE 6.0998, R2 0.6963 | Nearly preserves full-feature MainPeak performance with about one quarter of the features. |
| 9-feature compact panel | Shared narrow panel | Pass/Fail PR-AUC 0.1078; Need_Review PR-AUC 0.3749; MainPeak RMSE 6.2635 | Best lightweight cross-endpoint interpretation set. |

## Feature Figures

- Engineered feature plots: `analysis/feature_target_plots/index.md`. Each generated figure uses `Pass/Fail` on the x-axis and feature value on the y-axis.
- Raw QC/manufacturing plots: `analysis/all_data_qc_feature_boxplots/index.md`. Numeric figures also use `Pass/Fail` on the x-axis and raw feature value on the y-axis.
- Suggested first feature slides: LNA burden/run features (`token_lT_count`, `token_lA_count`, `longest_lT_run`), RNA/composition features (`rna_count_middle`, `base_U_count`), and positional chemistry features (`modified_token_position_std`, `star_position_std`).

## Top Statistical Feature Signals

| feature           |   mean_difference_fail_minus_pass |   median_difference_fail_minus_pass |   mannwhitney_p |   rank_biserial |   cohen_d |
|:------------------|----------------------------------:|------------------------------------:|----------------:|----------------:|----------:|
| token_lT_count    |                         0.1833    |                                   0 |       7.553e-14 |         0.04502 |    1.096  |
| token_lA_count    |                         0.1717    |                                   0 |       7.553e-14 |         0.04502 |    1.076  |
| token_lA_fraction |                         0.001409  |                                   0 |       7.554e-14 |         0.04502 |    1.077  |
| token_lT_fraction |                         0.001489  |                                   0 |       7.554e-14 |         0.04502 |    1.101  |
| longest_lT_run    |                         0.07983   |                                   0 |       7.729e-14 |         0.045   |    1.013  |
| longest_lA_run    |                         0.04495   |                                   0 |       8.279e-14 |         0.04495 |    0.8272 |
| longest_lC_run    |                         0.03371   |                                   0 |       8.21e-13  |         0.03409 |    0.6171 |
| token_lC_count    |                         0.03293   |                                   0 |       8.384e-13 |         0.03408 |    0.4248 |
| token_lC_fraction |                         0.0002821 |                                   0 |       8.384e-13 |         0.03408 |    0.4579 |
| token_lG_fraction |                         0.0008854 |                                   0 |       2.77e-12  |         0.04462 |    0.8487 |

## Required Presentation Asset Manifest

Manifest CSV: `analysis/modeling_outputs/presentation_asset_manifest.csv`

| asset_type   | path                                                                      | exists   | presentation_use                                                                          |
|:-------------|:--------------------------------------------------------------------------|:---------|:------------------------------------------------------------------------------------------|
| figure_index | analysis/feature_target_plots/index.md                                    | True     | Engineered feature vs Pass/Fail figures; Pass/Fail is x-axis and feature value is y-axis. |
| figure_index | analysis/all_data_qc_feature_boxplots/index.md                            | True     | Raw QC/manufacturing feature plots by Pass/Fail.                                          |
| figure       | analysis/modeling_outputs/learning_curve.png                              | True     | Learning-curve figure for training-set-size discussion.                                   |
| figure       | analysis/modeling_outputs/mainpeak_regression_observed_vs_predicted.png   | True     | Observed versus predicted MainPeak regression figure.                                     |
| table        | analysis/modeling_outputs/cv_metrics.csv                                  | True     | Full-feature Pass/Fail classifier metrics.                                                |
| table        | analysis/modeling_outputs/correlation_reduced_model_comparison.csv        | True     | Full versus 71-feature correlation-pruned Pass/Fail comparison.                           |
| table        | analysis/modeling_outputs/need_review_model_comparison.csv                | True     | Full versus correlation-pruned Need_Review comparison.                                    |
| table        | analysis/modeling_outputs/mainpeak_regression_metrics.csv                 | True     | Full-feature MainPeak regression metrics.                                                 |
| table        | analysis/modeling_outputs/mainpeak_reduced_feature_regression_metrics.csv | True     | Top-40 reduced-feature MainPeak metrics.                                                  |
| table        | analysis/modeling_outputs/narrow_feature_panel_selected_features.csv      | True     | Revised 9-feature compact cross-endpoint panel.                                           |
| table        | analysis/modeling_outputs/feature_rankings.csv                            | True     | Statistical and model-derived feature ranking table.                                      |
| table        | analysis/modeling_outputs/risk_tiering.csv                                | True     | Risk tier enrichment table.                                                               |
| report       | analysis/modeling_outputs/evaluation_report.md                            | True     | Main Pass/Fail evaluation report.                                                         |
| report       | analysis/modeling_outputs/need_review_modeling_report.md                  | True     | Need_Review model report.                                                                 |
| report       | analysis/modeling_outputs/mainpeak_regression_report.md                   | True     | MainPeak regression and derived classifier report.                                        |
| report       | analysis/modeling_outputs/mainpeak_reduced_feature_report.md              | True     | Reduced-feature MainPeak report.                                                          |
| report       | analysis/modeling_outputs/narrow_feature_panel_modeling_report.md         | True     | 9-feature narrow-panel report.                                                            |
| report       | sgRNA_synthesizability/.github/feature_selection_recommendation.md        | True     | Feature-selection recommendations for next-step decisions.                                |
| report       | sgRNA_synthesizability/README.md                                          | True     | Project-level narrative summary.                                                          |

## Missing Assets Check

None found among required presentation assets.

## Recommended Slide Order

1. Problem and data imbalance: 86 Fail rows among 2,646 binary-labeled rows.
2. Feature engineering overview: 161 decorated-sequence features plus parser handling of `+` LNA notation.
3. Feature evidence: Pass/Fail box/violin plots and top feature ranking table.
4. Pass/Fail model comparison: full versus 71-feature correlation-pruned HGB.
5. Need_Review modeling: stronger formula-derived endpoint and its relationship to MainPeak.
6. MainPeak regression: continuous purity is predictable; top-40 reduced model is nearly as strong as full feature set.
7. Narrow feature panels: 3/5/9-feature interpretation tradeoffs.
8. Operational recommendation: risk-ranking workflow, threshold calibration, and future data needs.
