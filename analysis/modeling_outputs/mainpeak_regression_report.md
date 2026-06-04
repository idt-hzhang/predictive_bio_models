# MainPeak Regression Modeling Report

Input file: `data/cleaned_data.mainpeak.csv`
Rows with nonmissing MainPeak and Sequence: 2653
Sequence feature count: 161
Cross-validation: 5-fold grouped CV using decorated `Sequence` groups.

## Target Summary

- MainPeak mean: 76.188
- MainPeak median: 79.000
- MainPeak standard deviation: 11.068
- MainPeak range: 15.000 to 97.000

## Overall Out-of-Fold Metrics

| model                  |    mae |    rmse |      r2 |   pearson_r |   spearman_r |
|:-----------------------|-------:|--------:|--------:|------------:|-------------:|
| extra_trees            | 4.1672 |  6.0574 |  0.7005 |      0.8370 |       0.7750 |
| gradient_boosting      | 4.3078 |  6.1917 |  0.6870 |      0.8289 |       0.7602 |
| hist_gradient_boosting | 4.2547 |  6.2093 |  0.6853 |      0.8281 |       0.7593 |
| random_forest          | 4.2684 |  6.2526 |  0.6809 |      0.8252 |       0.7638 |
| elastic_net            | 4.5117 |  6.5236 |  0.6526 |      0.8081 |       0.7409 |
| bayesian_ridge         | 4.5768 |  7.3527 |  0.5587 |      0.7599 |       0.7404 |
| mean_baseline          | 8.3342 | 11.0699 | -0.0003 |     -0.0249 |      -0.0241 |
| ridge                  | 4.7152 | 11.8459 | -0.1455 |      0.5621 |       0.7410 |
| lasso                  | 4.7739 | 15.6933 | -1.0104 |      0.4533 |       0.7437 |

## Best Model

The best MainPeak model by RMSE is `extra_trees`, with RMSE 6.0574, MAE 4.1672, R2 0.7005, Pearson r 0.8370, and Spearman r 0.7750.
Compared with the mean baseline RMSE of 11.0699, this improves RMSE by 5.0125 (45.3%).
Observed-vs-predicted plot: `analysis/modeling_outputs/mainpeak_regression_observed_vs_predicted.png`.

## Derived Pass/Fail Classification From Predicted MainPeak

Predicted MainPeak values were converted to derived binary labels using the length-specific thresholds documented in `sgRNA_synthesizability/.github/pass_fail_labeling_rules.md`. A predicted value below the pass threshold was treated as derived `Fail` for comparison with the original binary `Pass/Fail` labels. The continuous failure score is the threshold-normalized deficit below the pass threshold, so larger values indicate higher predicted failure risk.

| model                  |   average_precision |   roc_auc |   precision_at_recall_25 |   precision_at_recall_50 |   precision_at_recall_75 |   precision_top_05 |   precision_top_10 |   rule_threshold_tn |   rule_threshold_fp |   rule_threshold_fn |   rule_threshold_tp |
|:-----------------------|--------------------:|----------:|-------------------------:|-------------------------:|-------------------------:|-------------------:|-------------------:|--------------------:|--------------------:|--------------------:|--------------------:|
| elastic_net            |              0.1069 |    0.6217 |                   0.1228 |                   0.0423 |                   0.0413 |             0.1353 |             0.0943 |                2520 |                  40 |                  76 |                   8 |
| ridge                  |              0.0983 |    0.6168 |                   0.1438 |                   0.0450 |                   0.0390 |             0.1429 |             0.0943 |                2518 |                  42 |                  76 |                   8 |
| lasso                  |              0.0975 |    0.6104 |                   0.1409 |                   0.0420 |                   0.0383 |             0.1353 |             0.0981 |                2515 |                  45 |                  75 |                   9 |
| bayesian_ridge         |              0.0969 |    0.6197 |                   0.1419 |                   0.0443 |                   0.0393 |             0.1353 |             0.0943 |                2518 |                  42 |                  76 |                   8 |
| gradient_boosting      |              0.0724 |    0.6001 |                   0.1088 |                   0.0409 |                   0.0377 |             0.1429 |             0.0906 |                2508 |                  52 |                  77 |                   7 |
| extra_trees            |              0.0720 |    0.6123 |                   0.1392 |                   0.0447 |                   0.0376 |             0.1504 |             0.0868 |                2499 |                  61 |                  75 |                   9 |
| hist_gradient_boosting |              0.0659 |    0.6026 |                   0.1373 |                   0.0418 |                   0.0360 |             0.1353 |             0.0830 |                2490 |                  70 |                  74 |                  10 |
| random_forest          |              0.0621 |    0.5969 |                   0.1214 |                   0.0413 |                   0.0382 |             0.1504 |             0.0868 |                2501 |                  59 |                  77 |                   7 |
| mean_baseline          |              0.0343 |    0.5160 |                   0.0356 |                   0.0356 |                   0.0356 |             0.0301 |             0.0264 |                2504 |                  56 |                  82 |                   2 |

The best derived classifier by average precision is `elastic_net`, with AP 0.1069, ROC AUC 0.6217, Precision@Recall25 0.1228, Precision@Recall50 0.0423, and Precision@5% 0.1353.

### Per-Class Performance for Best Derived Classifier

| class_label   |   support |   predicted_count |   precision |   recall |   specificity |     f1 |
|:--------------|----------:|------------------:|------------:|---------:|--------------:|-------:|
| Fail          |        84 |                48 |      0.1667 |   0.0952 |        0.9844 | 0.1212 |
| Pass          |      2560 |              2596 |      0.9707 |   0.9844 |        0.0952 | 0.9775 |

Per-class metrics are calculated from the hard labels produced by the MainPeak threshold rule. For `Fail`, recall is failure sensitivity. For `Pass`, recall is pass specificity against false failure calls.

## Comparison With Pass/Fail Models

- Previous best Pass/Fail classifier (`hist_gradient_boosting`): AP 0.1087, ROC AUC 0.6304, Precision@5% 0.1278.
- Best correlation-pruned Pass/Fail classifier (`hist_gradient_boosting`): AP 0.1139, ROC AUC 0.6529, Precision@5% 0.1353.
- Best derived classifier from MainPeak regression (`elastic_net`): AP 0.1069, ROC AUC 0.6217, Precision@5% 0.1353.
- MainPeak regression is a continuous QC modeling task, so its metrics are not directly comparable to AP/ROC AUC. The useful comparison is qualitative: MainPeak regression tests whether sequence features explain continuous chromatographic purity, while Pass/Fail classification tests whether they enrich rare failures.
- The best MainPeak model explains 70.0% of out-of-fold MainPeak variance. This indicates substantial continuous purity signal in sequence features, whereas the rare-event Pass/Fail models remain more limited because they compress that continuous QC behavior into an imbalanced binary endpoint.

## Outputs

- `mainpeak_regression_metrics.csv`
- `mainpeak_regression_oof_predictions.csv`
- `mainpeak_regression_derived_classifier_metrics.csv`
- `mainpeak_regression_derived_classifier_class_metrics.csv`
- `mainpeak_regression_derived_classifier_predictions.csv`
- `mainpeak_regression_observed_vs_predicted.png`
- `mainpeak_regression_report.md`
