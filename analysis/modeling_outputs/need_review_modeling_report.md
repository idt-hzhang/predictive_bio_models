# Need_Review Modeling Report

Input file: `data/cleaned_data.need_review.csv`
Rows used: 2653
Label counts: {'Pass': 2491, 'Needs Review': 162}
Positive class: `Needs Review`
Original feature count: 161
Correlation-pruned feature count: 71
Dropped correlated feature count: 90
Cross-validation: 5-fold stratified grouped CV using decorated `Sequence` groups.

## Overall Out-of-Fold Metrics

| feature_set         | model                                                     |   average_precision |   pr_auc |   roc_auc |   brier_score |   f1_score |   recall_at_precision_25 |   precision_top_05 |   precision_top_10 |   threshold_0_5_tn |   threshold_0_5_fp |   threshold_0_5_fn |   threshold_0_5_tp |
|:--------------------|:----------------------------------------------------------|--------------------:|---------:|----------:|--------------:|-----------:|-------------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| correlation_reduced | correlation_reduced_hist_gradient_boosting                |              0.4082 |   0.4082 |    0.7719 |        0.0456 |     0.3467 |                   0.5123 |             0.4737 |             0.3008 |               2467 |                 24 |                123 |                 39 |
| full                | full_hist_gradient_boosting                               |              0.4054 |   0.4054 |    0.7741 |        0.0454 |     0.3545 |                   0.5617 |             0.4586 |             0.3083 |               2472 |                 19 |                123 |                 39 |
| correlation_reduced | correlation_reduced_lightgbm_balanced                     |              0.3739 |   0.3739 |    0.7641 |        0.0961 |     0.3592 |                   0.5556 |             0.4436 |             0.3120 |               2251 |                240 |                 74 |                 88 |
| full                | full_balanced_random_forest                               |              0.3734 |   0.3734 |    0.7878 |        0.1036 |     0.3767 |                   0.6235 |             0.4436 |             0.2932 |               2235 |                256 |                 65 |                 97 |
| correlation_reduced | correlation_reduced_xgboost_weighted                      |              0.3733 |   0.3733 |    0.7725 |        0.1831 |     0.2500 |                   0.6358 |             0.4737 |             0.2857 |               1855 |                636 |                 48 |                114 |
| full                | full_lightgbm_balanced                                    |              0.3726 |   0.3726 |    0.7681 |        0.0942 |     0.3432 |                   0.5494 |             0.4887 |             0.2895 |               2233 |                258 |                 75 |                 87 |
| full                | full_xgboost_weighted                                     |              0.3699 |   0.3699 |    0.7759 |        0.1767 |     0.2534 |                   0.6049 |             0.4812 |             0.3008 |               1874 |                617 |                 49 |                113 |
| correlation_reduced | correlation_reduced_balanced_random_forest                |              0.3630 |   0.3630 |    0.7847 |        0.1092 |     0.3506 |                   0.5864 |             0.4436 |             0.2857 |               2206 |                285 |                 67 |                 95 |
| full                | full_elastic_net_logistic                                 |              0.2704 |   0.2704 |    0.7706 |        0.1592 |     0.2741 |                   0.5062 |             0.3835 |             0.2707 |               1954 |                537 |                 51 |                111 |
| full                | full_linear_svm_rbf_calibrated_probability                |              0.2702 |   0.2702 |    0.7885 |        0.0535 |     0.0000 |                   0.4938 |             0.3308 |             0.2669 |               2491 |                  0 |                162 |                  0 |
| full                | full_logistic_l2_balanced                                 |              0.2694 |   0.2694 |    0.7703 |        0.1579 |     0.2796 |                   0.5062 |             0.3759 |             0.2707 |               1970 |                521 |                 51 |                111 |
| full                | full_logistic_l1_balanced                                 |              0.2686 |   0.2686 |    0.7790 |        0.1563 |     0.2783 |                   0.5494 |             0.3459 |             0.2707 |               1954 |                537 |                 49 |                113 |
| correlation_reduced | correlation_reduced_logistic_l2_balanced                  |              0.2603 |   0.2603 |    0.7614 |        0.1651 |     0.2667 |                   0.4691 |             0.3609 |             0.2669 |               1938 |                553 |                 52 |                110 |
| correlation_reduced | correlation_reduced_elastic_net_logistic                  |              0.2599 |   0.2599 |    0.7645 |        0.1649 |     0.2655 |                   0.4691 |             0.3609 |             0.2669 |               1941 |                550 |                 53 |                109 |
| correlation_reduced | correlation_reduced_linear_svm_rbf_calibrated_probability |              0.2555 |   0.2555 |    0.7821 |        0.0530 |     0.0000 |                   0.5185 |             0.2857 |             0.2594 |               2491 |                  0 |                162 |                  0 |
| correlation_reduced | correlation_reduced_logistic_l1_balanced                  |              0.2550 |   0.2550 |    0.7653 |        0.1649 |     0.2647 |                   0.4877 |             0.3609 |             0.2669 |               1932 |                559 |                 52 |                110 |
| correlation_reduced | correlation_reduced_elastic_net_logistic_with_char_ngrams |              0.2546 |   0.2546 |    0.7519 |        0.1750 |     0.2439 |                   0.4506 |             0.3684 |             0.2594 |               1825 |                666 |                 47 |                115 |
| full                | full_elastic_net_logistic_with_char_ngrams                |              0.2534 |   0.2534 |    0.7492 |        0.1751 |     0.2415 |                   0.4012 |             0.3684 |             0.2481 |               1830 |                661 |                 49 |                113 |
| full                | full_rule_baseline                                        |              0.1243 |   0.1243 |    0.6742 |        0.1623 |     0.1895 |                   0.0062 |             0.2331 |             0.1429 |               2395 |                 96 |                135 |                 27 |
| full                | full_prevalence_baseline                                  |              0.0606 |   0.0606 |    0.4961 |        0.0573 |     0.0000 |                   0.0000 |             0.0677 |             0.0414 |               2491 |                  0 |                162 |                  0 |
| correlation_reduced | correlation_reduced_prevalence_baseline                   |              0.0606 |   0.0606 |    0.4961 |        0.0573 |     0.0000 |                   0.0000 |             0.0677 |             0.0414 |               2491 |                  0 |                162 |                  0 |

## Best Learned Model

The best Need_Review model by PR-AUC is `correlation_reduced_hist_gradient_boosting`, with PR-AUC 0.4082, ROC AUC 0.7719, Brier score 0.0456, F1 0.3467, Precision@5% 0.4737, and Precision@10% 0.3008.

## Comparison With Main Pass/Fail Models

- Main Pass/Fail full-feature HGB: PR-AUC 0.1087, ROC AUC 0.6304, F1 0.1165, Precision@5% 0.1278.
- Main Pass/Fail correlation-pruned HGB: PR-AUC 0.1139, ROC AUC 0.6529, F1 0.0800, Precision@5% 0.1353.
- Best Need_Review model (`correlation_reduced_hist_gradient_boosting`): PR-AUC 0.4082, ROC AUC 0.7719, F1 0.3467, Precision@5% 0.4737.
- The Need_Review target is less rare than Fail and is derived directly from Length and MainPeak thresholds, so it is not biologically identical to the source Pass/Fail label. Compare the metrics as target-specific ranking performance rather than as interchangeable endpoints.

## Outputs

- `need_review_cv_metrics.csv`
- `need_review_oof_predictions.csv`
- `need_review_correlation_reduced_cv_metrics.csv`
- `need_review_correlation_reduced_oof_predictions.csv`
- `need_review_model_comparison.csv`
- `need_review_dropped_correlated_features.csv`
- `need_review_modeling_report.md`
