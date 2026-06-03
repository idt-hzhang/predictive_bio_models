# 5-Feature Narrow Panel Modeling Report

Selected a shared panel of 5 sequence-derived features using composite target association across Pass/Fail, Pass/Need_Review, and MainPeak, with pairwise absolute Pearson correlation capped at 0.75 during greedy selection.

## Selected Features

|   rank | feature                     | feature_family   | is_count_or_fraction   | family_selection_reason                   |   composite_score |   pass_fail_abs_corr |   need_review_abs_corr |   mainpeak_abs_corr |   max_abs_correlation_to_previous_selected |
|-------:|:----------------------------|:-----------------|:-----------------------|:------------------------------------------|------------------:|---------------------:|-----------------------:|--------------------:|-------------------------------------------:|
|      1 | rna_count_middle            | rna              | True                   | preferred_count_or_fraction_within_family |            0.9027 |               0.0406 |                 0.1918 |              0.6314 |                                     0.0000 |
|      2 | modified_token_position_std | modified_token   | False                  | non_count_fraction_score_advantage        |            0.8385 |               0.0395 |                 0.1219 |              0.4941 |                                     0.7192 |
|      3 | token_rC_count              | token_rC         | True                   | preferred_count_or_fraction_within_family |            0.8385 |               0.0212 |                 0.1970 |              0.6462 |                                     0.7444 |
|      4 | lna_span_fraction           | lna              | True                   | preferred_count_or_fraction_within_family |            0.8364 |               0.1947 |                 0.1387 |              0.1521 |                                     0.0643 |
|      5 | base_U_count                | base_U           | True                   | preferred_count_or_fraction_within_family |            0.8085 |               0.0309 |                 0.1104 |              0.5455 |                                     0.7042 |

## Best Models by Endpoint

| endpoint    | model                  |   pr_auc |   roc_auc |   f1_score |   precision_top_05 |
|:------------|:-----------------------|---------:|----------:|-----------:|-------------------:|
| need_review | hist_gradient_boosting |   0.3635 |    0.7832 |     0.3070 |             0.4662 |
| pass_fail   | hist_gradient_boosting |   0.0895 |    0.6071 |     0.0000 |             0.1353 |

Best MainPeak regression model: `hist_gradient_boosting` with RMSE 6.4344, R2 0.6620, Pearson r 0.8137, and Spearman r 0.7531.

## Classification Metrics

| endpoint    | model                                 |   pr_auc |   roc_auc |   brier_score |   f1_score |   precision_top_05 |   precision_top_10 |
|:------------|:--------------------------------------|---------:|----------:|--------------:|-----------:|-------------------:|-------------------:|
| need_review | hist_gradient_boosting                |   0.3635 |    0.7832 |        0.0470 |     0.3070 |             0.4662 |             0.3083 |
| need_review | balanced_random_forest                |   0.3566 |    0.7728 |        0.1228 |     0.3156 |             0.4060 |             0.2744 |
| need_review | lightgbm_balanced                     |   0.3484 |    0.7787 |        0.1218 |     0.3040 |             0.4511 |             0.2970 |
| need_review | xgboost_weighted                      |   0.3366 |    0.7632 |        0.2258 |     0.2187 |             0.4361 |             0.2895 |
| need_review | logistic_l1_balanced                  |   0.2438 |    0.7303 |        0.2024 |     0.2172 |             0.2932 |             0.1767 |
| need_review | elastic_net_logistic                  |   0.2433 |    0.7302 |        0.2020 |     0.2176 |             0.2932 |             0.1767 |
| need_review | logistic_l2_balanced                  |   0.2430 |    0.7301 |        0.2021 |     0.2174 |             0.2932 |             0.1767 |
| need_review | linear_svm_rbf_calibrated_probability |   0.1750 |    0.7386 |        0.0542 |     0.0000 |             0.2331 |             0.1842 |
| need_review | prevalence_baseline                   |   0.0606 |    0.4961 |        0.0573 |     0.0000 |             0.0677 |             0.0414 |
| pass_fail   | hist_gradient_boosting                |   0.0895 |    0.6071 |        0.0315 |     0.0000 |             0.1353 |             0.0943 |
| pass_fail   | lightgbm_balanced                     |   0.0859 |    0.6103 |        0.1352 |     0.1105 |             0.1353 |             0.1057 |
| pass_fail   | elastic_net_logistic                  |   0.0837 |    0.5396 |        0.2359 |     0.0670 |             0.0602 |             0.0491 |
| pass_fail   | logistic_l1_balanced                  |   0.0837 |    0.5396 |        0.2364 |     0.0665 |             0.0602 |             0.0491 |
| pass_fail   | logistic_l2_balanced                  |   0.0837 |    0.5395 |        0.2360 |     0.0669 |             0.0602 |             0.0491 |
| pass_fail   | xgboost_weighted                      |   0.0832 |    0.5857 |        0.1641 |     0.0985 |             0.1429 |             0.0943 |
| pass_fail   | balanced_random_forest                |   0.0746 |    0.5931 |        0.1364 |     0.1415 |             0.1504 |             0.0868 |
| pass_fail   | prevalence_baseline                   |   0.0322 |    0.4953 |        0.0314 |     0.0000 |             0.0602 |             0.0377 |
| pass_fail   | linear_svm_rbf_calibrated_probability |   0.0284 |    0.4521 |        0.0319 |     0.0000 |             0.0150 |             0.0113 |

## MainPeak Regression Metrics

| model                  |    mae |    rmse |      r2 |   pearson_r |   spearman_r |
|:-----------------------|-------:|--------:|--------:|------------:|-------------:|
| hist_gradient_boosting | 4.4246 |  6.4344 |  0.6620 |      0.8137 |       0.7531 |
| extra_trees            | 4.4261 |  6.4525 |  0.6601 |      0.8127 |       0.7532 |
| random_forest          | 4.4368 |  6.5114 |  0.6539 |      0.8086 |       0.7503 |
| gradient_boosting      | 4.4785 |  6.5313 |  0.6518 |      0.8073 |       0.7485 |
| ridge                  | 5.6608 |  7.7996 |  0.5034 |      0.7097 |       0.6461 |
| bayesian_ridge         | 5.6608 |  7.7998 |  0.5034 |      0.7097 |       0.6460 |
| elastic_net            | 5.6654 |  7.8001 |  0.5033 |      0.7095 |       0.6469 |
| lasso                  | 5.6620 |  7.8007 |  0.5033 |      0.7096 |       0.6461 |
| mean_baseline          | 8.3342 | 11.0699 | -0.0003 |     -0.0249 |      -0.0241 |

## Retained Performance Versus Larger Feature Sets

- Pass/Fail: best narrow-panel model `hist_gradient_boosting` PR-AUC 0.0895, retaining 78.6% of the correlation-pruned HGB PR-AUC reference (0.1139).
- Pass/Need_Review: best narrow-panel model `hist_gradient_boosting` PR-AUC 0.3635, retaining 89.1% of the previous best Need_Review PR-AUC reference (0.4082).
- MainPeak regression: best narrow-panel model `hist_gradient_boosting` RMSE 6.4344, a 6.2% increase versus full-feature ExtraTrees RMSE (6.0574).

The panel is intentionally small and low-redundancy, so it is most useful for interpretation and lightweight triage. It does not fully preserve the best larger-model performance, especially for MainPeak regression.

## Outputs

- `narrow_5_feature_panel_selected_features.csv`
- `narrow_5_feature_panel_classification_metrics.csv`
- `narrow_5_feature_panel_classification_oof_predictions.csv`
- `narrow_5_feature_panel_regression_metrics.csv`
- `narrow_5_feature_panel_regression_oof_predictions.csv`
- `narrow_5_feature_panel_modeling_report.md`
