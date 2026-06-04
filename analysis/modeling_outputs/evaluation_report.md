# Baseline Modeling Report

## Data

- Input file: `data/cleaned_data.csv`
- Total cleaned rows: 2657
- Binary labeled rows: 2646
- Fail rows: 86
- Pass rows: 2560
- Held-out Needs Review rows: 11
- Feature count: 161

## Overall Out-of-Fold Metrics

```text
                                model  average_precision  pr_auc  roc_auc  brier_score  f1_score  recall_at_precision_25  precision_top_05  precision_top_10
               hist_gradient_boosting             0.1087  0.1087   0.6304       0.0319    0.1165                  0.0698            0.1278            0.0906
                    lightgbm_balanced             0.1030  0.1030   0.6172       0.0871    0.1318                  0.1628            0.1429            0.0868
                     xgboost_weighted             0.0907  0.0907   0.6238       0.1232    0.1120                  0.1744            0.1504            0.0868
               balanced_random_forest             0.0848  0.0848   0.6230       0.1095    0.2069                  0.0233            0.1654            0.0943
                 logistic_l2_balanced             0.0767  0.0767   0.5930       0.1874    0.0867                  0.1047            0.1128            0.0717
elastic_net_logistic_with_char_ngrams             0.0762  0.0762   0.5833       0.1979    0.1001                  0.0814            0.1203            0.0906
                 elastic_net_logistic             0.0740  0.0740   0.5907       0.1877    0.0895                  0.1047            0.1128            0.0792
                 logistic_l1_balanced             0.0702  0.0702   0.5925       0.1868    0.0840                  0.0930            0.1128            0.0755
                        rule_baseline             0.0402  0.0402   0.4989       0.1601    0.0660                  0.0116            0.0602            0.0377
                  prevalence_baseline             0.0322  0.0322   0.4953       0.0314    0.0000                  0.0000            0.0602            0.0377
linear_svm_rbf_calibrated_probability             0.0282  0.0282   0.4110       0.0316    0.0000                  0.0000            0.0301            0.0377
```

## Selected Model

Selected model: `hist_gradient_boosting` based on highest out-of-fold average precision among trained model candidates.

## Notes

- `Needs Review` rows were excluded from training and model-selection metrics.
- Cross-validation used stratified group folds with decorated `Sequence` as the group to reduce sequence-level leakage.
- Accuracy is intentionally omitted from the headline metrics because the failure class is rare.
