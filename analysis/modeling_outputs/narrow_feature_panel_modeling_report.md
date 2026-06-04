# 9-Feature Narrow Panel Modeling Report

Selected a shared panel of 9 sequence-derived features using composite target association across Pass/Fail, Pass/Need_Review, and MainPeak, with pairwise absolute Pearson correlation capped at 0.75 during greedy selection.

## Selected Features

|   rank | feature                     | feature_family   | is_count_or_fraction   | family_selection_reason                   |   composite_score |   pass_fail_abs_corr |   need_review_abs_corr |   mainpeak_abs_corr |   max_abs_correlation_to_previous_selected |
|-------:|:----------------------------|:-----------------|:-----------------------|:------------------------------------------|------------------:|---------------------:|-----------------------:|--------------------:|-------------------------------------------:|
|      1 | rna_count_middle            | rna              | True                   | preferred_count_or_fraction_within_family |            0.9027 |               0.0406 |                 0.1918 |              0.6314 |                                     0.0000 |
|      2 | modified_token_position_std | modified_token   | False                  | non_count_fraction_score_advantage        |            0.8385 |               0.0395 |                 0.1219 |              0.4941 |                                     0.7192 |
|      3 | token_rC_count              | token_rC         | True                   | preferred_count_or_fraction_within_family |            0.8385 |               0.0212 |                 0.1970 |              0.6462 |                                     0.7444 |
|      4 | lna_span_fraction           | lna              | True                   | preferred_count_or_fraction_within_family |            0.8364 |               0.1947 |                 0.1387 |              0.1521 |                                     0.0643 |
|      5 | base_U_count                | base_U           | True                   | preferred_count_or_fraction_within_family |            0.8085 |               0.0309 |                 0.1104 |              0.5455 |                                     0.7042 |
|      6 | star_count_3p               | star             | True                   | preferred_count_or_fraction_within_family |            0.7516 |               0.0703 |                 0.0736 |              0.2558 |                                     0.4313 |
|      7 | token_lG_fraction           | token_lG         | True                   | preferred_count_or_fraction_within_family |            0.7308 |               0.1489 |                 0.1035 |              0.1300 |                                     0.7136 |
|      8 | longest_rU_run              | longest_rU_run   | False                  | no_count_or_fraction_candidate            |            0.6418 |               0.0578 |                 0.0854 |              0.1083 |                                     0.4024 |
|      9 | longest_lC_run              | longest_lC_run   | False                  | no_count_or_fraction_candidate            |            0.6128 |               0.1088 |                 0.0756 |              0.0857 |                                     0.6473 |

## Best Models by Endpoint

| endpoint    | model                  |   pr_auc |   roc_auc |   f1_score |   precision_top_05 |
|:------------|:-----------------------|---------:|----------:|-----------:|-------------------:|
| need_review | hist_gradient_boosting |   0.3749 |    0.7848 |     0.3470 |             0.4586 |
| pass_fail   | logistic_l1_balanced   |   0.1078 |    0.5770 |     0.0915 |             0.0602 |

Best MainPeak regression model: `hist_gradient_boosting` with RMSE 6.2635, R2 0.6797, Pearson r 0.8245, and Spearman r 0.7557.

## Classification Metrics

| endpoint    | model                                 |   pr_auc |   roc_auc |   brier_score |   f1_score |   precision_top_05 |   precision_top_10 |
|:------------|:--------------------------------------|---------:|----------:|--------------:|-----------:|-------------------:|-------------------:|
| need_review | hist_gradient_boosting                |   0.3749 |    0.7848 |        0.0469 |     0.3470 |             0.4586 |             0.3008 |
| need_review | xgboost_weighted                      |   0.3692 |    0.7723 |        0.2205 |     0.2204 |             0.4812 |             0.2895 |
| need_review | balanced_random_forest                |   0.3555 |    0.7814 |        0.1266 |     0.3114 |             0.4436 |             0.2782 |
| need_review | lightgbm_balanced                     |   0.3552 |    0.7772 |        0.1189 |     0.2975 |             0.4737 |             0.2820 |
| need_review | logistic_l2_balanced                  |   0.2439 |    0.7367 |        0.1975 |     0.2309 |             0.3008 |             0.2143 |
| need_review | logistic_l1_balanced                  |   0.2437 |    0.7367 |        0.1978 |     0.2307 |             0.3008 |             0.2143 |
| need_review | elastic_net_logistic                  |   0.2434 |    0.7367 |        0.1974 |     0.2312 |             0.3008 |             0.2143 |
| need_review | linear_svm_rbf_calibrated_probability |   0.2086 |    0.7594 |        0.0540 |     0.0000 |             0.2481 |             0.2180 |
| need_review | prevalence_baseline                   |   0.0606 |    0.4961 |        0.0573 |     0.0000 |             0.0677 |             0.0414 |
| pass_fail   | logistic_l1_balanced                  |   0.1078 |    0.5770 |        0.2254 |     0.0915 |             0.0602 |             0.0679 |
| pass_fail   | elastic_net_logistic                  |   0.1077 |    0.5765 |        0.2250 |     0.0921 |             0.0602 |             0.0642 |
| pass_fail   | logistic_l2_balanced                  |   0.1077 |    0.5761 |        0.2250 |     0.0922 |             0.0602 |             0.0642 |
| pass_fail   | hist_gradient_boosting                |   0.0961 |    0.6281 |        0.0315 |     0.0000 |             0.1429 |             0.1019 |
| pass_fail   | xgboost_weighted                      |   0.0936 |    0.6199 |        0.1587 |     0.1151 |             0.1654 |             0.1057 |
| pass_fail   | lightgbm_balanced                     |   0.0903 |    0.6387 |        0.1304 |     0.1216 |             0.1579 |             0.1094 |
| pass_fail   | balanced_random_forest                |   0.0851 |    0.6131 |        0.1395 |     0.1643 |             0.1654 |             0.0943 |
| pass_fail   | prevalence_baseline                   |   0.0322 |    0.4953 |        0.0314 |     0.0000 |             0.0602 |             0.0377 |
| pass_fail   | linear_svm_rbf_calibrated_probability |   0.0300 |    0.4384 |        0.0316 |     0.0000 |             0.0226 |             0.0377 |

## MainPeak Regression Metrics

| model                  |    mae |    rmse |      r2 |   pearson_r |   spearman_r |
|:-----------------------|-------:|--------:|--------:|------------:|-------------:|
| hist_gradient_boosting | 4.3362 |  6.2635 |  0.6797 |      0.8245 |       0.7557 |
| extra_trees            | 4.3593 |  6.3263 |  0.6733 |      0.8208 |       0.7554 |
| random_forest          | 4.3905 |  6.3427 |  0.6716 |      0.8195 |       0.7520 |
| gradient_boosting      | 4.4071 |  6.3926 |  0.6664 |      0.8163 |       0.7480 |
| elastic_net            | 5.4387 |  7.3640 |  0.5573 |      0.7466 |       0.6625 |
| lasso                  | 5.4372 |  7.3647 |  0.5572 |      0.7466 |       0.6615 |
| bayesian_ridge         | 5.4374 |  7.3650 |  0.5572 |      0.7465 |       0.6614 |
| ridge                  | 5.4374 |  7.3652 |  0.5572 |      0.7465 |       0.6614 |
| mean_baseline          | 8.3342 | 11.0699 | -0.0003 |     -0.0249 |      -0.0241 |

## Retained Performance Versus Larger Feature Sets

- Pass/Fail: best narrow-panel model `logistic_l1_balanced` PR-AUC 0.1078, retaining 94.7% of the correlation-pruned HGB PR-AUC reference (0.1139).
- Pass/Need_Review: best narrow-panel model `hist_gradient_boosting` PR-AUC 0.3749, retaining 91.8% of the previous best Need_Review PR-AUC reference (0.4082).
- MainPeak regression: best narrow-panel model `hist_gradient_boosting` RMSE 6.2635, a 3.4% increase versus full-feature ExtraTrees RMSE (6.0574).

The panel is intentionally small and low-redundancy, so it is most useful for interpretation and lightweight triage. It does not fully preserve the best larger-model performance, especially for MainPeak regression.

## Outputs

- `narrow_feature_panel_selected_features.csv`
- `narrow_feature_panel_classification_metrics.csv`
- `narrow_feature_panel_classification_oof_predictions.csv`
- `narrow_feature_panel_regression_metrics.csv`
- `narrow_feature_panel_regression_oof_predictions.csv`
- `narrow_feature_panel_modeling_report.md`
