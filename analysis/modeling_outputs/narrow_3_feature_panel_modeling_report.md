# 3-Feature Narrow Panel Modeling Report

Selected a shared panel of 3 sequence-derived features using composite target association across Pass/Fail, Pass/Need_Review, and MainPeak, with pairwise absolute Pearson correlation capped at 0.75 during greedy selection.

## Selected Features

|   rank | feature                     | feature_family   | is_count_or_fraction   | family_selection_reason                   |   composite_score |   pass_fail_abs_corr |   need_review_abs_corr |   mainpeak_abs_corr |   max_abs_correlation_to_previous_selected |
|-------:|:----------------------------|:-----------------|:-----------------------|:------------------------------------------|------------------:|---------------------:|-----------------------:|--------------------:|-------------------------------------------:|
|      1 | rna_count_middle            | rna              | True                   | preferred_count_or_fraction_within_family |            0.9027 |               0.0406 |                 0.1918 |              0.6314 |                                     0.0000 |
|      2 | modified_token_position_std | modified_token   | False                  | non_count_fraction_score_advantage        |            0.8385 |               0.0395 |                 0.1219 |              0.4941 |                                     0.7192 |
|      3 | token_rC_count              | token_rC         | True                   | preferred_count_or_fraction_within_family |            0.8385 |               0.0212 |                 0.1970 |              0.6462 |                                     0.7444 |

## Best Models by Endpoint

| endpoint    | model                  |   pr_auc |   roc_auc |   f1_score |   precision_top_05 |
|:------------|:-----------------------|---------:|----------:|-----------:|-------------------:|
| need_review | balanced_random_forest |   0.3609 |    0.7699 |     0.2931 |             0.4060 |
| pass_fail   | hist_gradient_boosting |   0.0815 |    0.5886 |     0.0000 |             0.1429 |

Best MainPeak regression model: `extra_trees` with RMSE 6.5345, R2 0.6514, Pearson r 0.8073, and Spearman r 0.7481.

## Classification Metrics

| endpoint    | model                                 |   pr_auc |   roc_auc |   brier_score |   f1_score |   precision_top_05 |   precision_top_10 |
|:------------|:--------------------------------------|---------:|----------:|--------------:|-----------:|-------------------:|-------------------:|
| need_review | balanced_random_forest                |   0.3609 |    0.7699 |        0.1300 |     0.2931 |             0.4060 |             0.2669 |
| need_review | hist_gradient_boosting                |   0.3602 |    0.7734 |        0.0470 |     0.3470 |             0.4662 |             0.3008 |
| need_review | xgboost_weighted                      |   0.3478 |    0.7678 |        0.2327 |     0.2248 |             0.4511 |             0.2895 |
| need_review | lightgbm_balanced                     |   0.3439 |    0.7607 |        0.1294 |     0.2978 |             0.4511 |             0.2782 |
| need_review | logistic_l1_balanced                  |   0.1980 |    0.7048 |        0.2114 |     0.2032 |             0.2556 |             0.1805 |
| need_review | elastic_net_logistic                  |   0.1971 |    0.7048 |        0.2110 |     0.2035 |             0.2556 |             0.1805 |
| need_review | logistic_l2_balanced                  |   0.1969 |    0.7050 |        0.2111 |     0.2035 |             0.2556 |             0.1805 |
| need_review | linear_svm_rbf_calibrated_probability |   0.1383 |    0.7338 |        0.0550 |     0.0000 |             0.1579 |             0.1692 |
| need_review | prevalence_baseline                   |   0.0606 |    0.4961 |        0.0573 |     0.0000 |             0.0677 |             0.0414 |
| pass_fail   | hist_gradient_boosting                |   0.0815 |    0.5886 |        0.0316 |     0.0000 |             0.1429 |             0.0906 |
| pass_fail   | lightgbm_balanced                     |   0.0728 |    0.6051 |        0.1517 |     0.0967 |             0.1429 |             0.0906 |
| pass_fail   | xgboost_weighted                      |   0.0687 |    0.5840 |        0.1722 |     0.0984 |             0.1429 |             0.0943 |
| pass_fail   | balanced_random_forest                |   0.0646 |    0.5852 |        0.1502 |     0.1053 |             0.1429 |             0.0868 |
| pass_fail   | logistic_l1_balanced                  |   0.0386 |    0.5412 |        0.2467 |     0.0770 |             0.0451 |             0.0415 |
| pass_fail   | logistic_l2_balanced                  |   0.0385 |    0.5415 |        0.2462 |     0.0752 |             0.0451 |             0.0415 |
| pass_fail   | elastic_net_logistic                  |   0.0357 |    0.5103 |        0.2728 |     0.0662 |             0.0451 |             0.0302 |
| pass_fail   | linear_svm_rbf_calibrated_probability |   0.0336 |    0.4917 |        0.0316 |     0.0000 |             0.0526 |             0.0340 |
| pass_fail   | prevalence_baseline                   |   0.0322 |    0.4953 |        0.0314 |     0.0000 |             0.0602 |             0.0377 |

## MainPeak Regression Metrics

| model                  |    mae |    rmse |      r2 |   pearson_r |   spearman_r |
|:-----------------------|-------:|--------:|--------:|------------:|-------------:|
| extra_trees            | 4.4588 |  6.5345 |  0.6514 |      0.8073 |       0.7481 |
| hist_gradient_boosting | 4.4701 |  6.5418 |  0.6506 |      0.8068 |       0.7432 |
| random_forest          | 4.4595 |  6.5771 |  0.6469 |      0.8044 |       0.7455 |
| gradient_boosting      | 4.4957 |  6.6107 |  0.6433 |      0.8020 |       0.7436 |
| lasso                  | 5.8601 |  8.1008 |  0.4643 |      0.6814 |       0.6354 |
| ridge                  | 5.8602 |  8.1017 |  0.4642 |      0.6813 |       0.6353 |
| bayesian_ridge         | 5.8604 |  8.1018 |  0.4642 |      0.6813 |       0.6350 |
| elastic_net            | 5.8578 |  8.1034 |  0.4640 |      0.6812 |       0.6374 |
| mean_baseline          | 8.3342 | 11.0699 | -0.0003 |     -0.0249 |      -0.0241 |

## Retained Performance Versus Larger Feature Sets

- Pass/Fail: best narrow-panel model `hist_gradient_boosting` PR-AUC 0.0815, retaining 71.6% of the correlation-pruned HGB PR-AUC reference (0.1139).
- Pass/Need_Review: best narrow-panel model `balanced_random_forest` PR-AUC 0.3609, retaining 88.4% of the previous best Need_Review PR-AUC reference (0.4082).
- MainPeak regression: best narrow-panel model `extra_trees` RMSE 6.5345, a 7.9% increase versus full-feature ExtraTrees RMSE (6.0574).

The panel is intentionally small and low-redundancy, so it is most useful for interpretation and lightweight triage. It does not fully preserve the best larger-model performance, especially for MainPeak regression.

## Outputs

- `narrow_3_feature_panel_selected_features.csv`
- `narrow_3_feature_panel_classification_metrics.csv`
- `narrow_3_feature_panel_classification_oof_predictions.csv`
- `narrow_3_feature_panel_regression_metrics.csv`
- `narrow_3_feature_panel_regression_oof_predictions.csv`
- `narrow_3_feature_panel_modeling_report.md`
