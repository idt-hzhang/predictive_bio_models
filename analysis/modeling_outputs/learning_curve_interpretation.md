# Learning Curve Interpretation

Input data: `data/cleaned_data.csv`
Model: `hist_gradient_boosting`
Cross-validation: 5-fold stratified grouped CV using decorated sequence groups.

## Metrics

|   training_fraction |   average_precision |   roc_auc |   precision_top_05 |   precision_top_10 |
|--------------------:|--------------------:|----------:|-------------------:|-------------------:|
|              0.1000 |              0.0836 |    0.5378 |             0.0602 |             0.0566 |
|              0.2000 |              0.0388 |    0.5646 |             0.0226 |             0.0302 |
|              0.4000 |              0.0966 |    0.5813 |             0.1203 |             0.0792 |
|              0.6000 |              0.0902 |    0.6080 |             0.0977 |             0.0792 |
|              0.8000 |              0.1026 |    0.6168 |             0.1053 |             0.0943 |
|              1.0000 |              0.1087 |    0.6304 |             0.1278 |             0.0906 |

## Plateau Assessment

From 80% to 100% training data, average precision changed by +0.0061, ROC AUC by +0.0136, and Precision@5% by +0.0226.

Performance still changes materially between 80% and 100% of the available training data, which suggests additional labeled data would theoretically help. Since more labeled failures are not expected, the practical next step is to focus on feature stability, interpretation, and risk-tiering analyses using the existing data.

Because the failure class is rare, small non-monotonic changes are expected across fractions. The most important signal is whether the full-data point is clearly above the 80% point across ranking metrics.
