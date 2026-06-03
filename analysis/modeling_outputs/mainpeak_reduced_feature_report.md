# Reduced-Feature MainPeak Modeling Report

Feature selectors were fit inside each 5-fold grouped CV split to reduce leakage. Each selector retained up to 40 features from the 161 sequence-derived features.

## Overall Out-of-Fold Regression Metrics

| selector                  | model                  |    mae |    rmse |      r2 |   pearson_r |   spearman_r |
|:--------------------------|:-----------------------|-------:|--------:|--------:|------------:|-------------:|
| extra_trees_top_40        | extra_trees            | 4.1872 |  6.0998 |  0.6963 |      0.8345 |       0.7710 |
| correlation_pruned_top_40 | extra_trees            | 4.2069 |  6.1339 |  0.6929 |      0.8325 |       0.7674 |
| mutual_info_top_40        | extra_trees            | 4.2089 |  6.1554 |  0.6907 |      0.8311 |       0.7692 |
| mutual_info_top_40        | hist_gradient_boosting | 4.3009 |  6.2538 |  0.6807 |      0.8252 |       0.7540 |
| correlation_pruned_top_40 | hist_gradient_boosting | 4.2844 |  6.2596 |  0.6801 |      0.8250 |       0.7601 |
| extra_trees_top_40        | hist_gradient_boosting | 4.3067 |  6.2999 |  0.6760 |      0.8226 |       0.7549 |
| extra_trees_top_40        | elastic_net            | 4.6670 |  6.7055 |  0.6329 |      0.7956 |       0.7551 |
| correlation_pruned_top_40 | elastic_net            | 4.6850 |  6.7129 |  0.6321 |      0.7951 |       0.7401 |
| mutual_info_top_40        | elastic_net            | 4.6623 |  6.7633 |  0.6266 |      0.7917 |       0.7494 |
| correlation_pruned_top_40 | mean_baseline          | 8.3342 | 11.0699 | -0.0003 |     -0.0249 |      -0.0241 |
| extra_trees_top_40        | mean_baseline          | 8.3342 | 11.0699 | -0.0003 |     -0.0249 |      -0.0241 |
| mutual_info_top_40        | mean_baseline          | 8.3342 | 11.0699 | -0.0003 |     -0.0249 |      -0.0241 |

## Best Reduced-Feature Model

The best reduced-feature model is `extra_trees` with selector `extra_trees_top_40`: RMSE 6.0998, MAE 4.1872, R2 0.6963, Pearson r 0.8345, and Spearman r 0.7710.

## Pass/Fail Class-Specific Regression Metrics

| class_label   |   rows |     mae |    rmse |     r2 |   pearson_r |   spearman_r |
|:--------------|-------:|--------:|--------:|-------:|------------:|-------------:|
| Fail          |     84 | 12.6806 | 16.3866 | 0.0002 |      0.7229 |       0.6465 |
| Pass          |   2560 |  3.8869 |  5.3894 | 0.7391 |      0.8608 |       0.7801 |

Class-specific metrics evaluate MainPeak prediction error separately for rows whose original binary label is `Pass` or `Fail`. These are regression metrics on MainPeak, not derived-classification metrics.

## Most Frequently Selected Features for Best Selector

| selector           | feature                      |   selection_count |   mean_rank |   mean_score |
|:-------------------|:-----------------------------|------------------:|------------:|-------------:|
| extra_trees_top_40 | token_length                 |                 5 |      1.0000 |       0.0999 |
| extra_trees_top_40 | methyl_last_position         |                 5 |      2.8000 |       0.0821 |
| extra_trees_top_40 | decorated_length             |                 5 |      3.0000 |       0.0760 |
| extra_trees_top_40 | star_last_position           |                 5 |      3.6000 |       0.0706 |
| extra_trees_top_40 | base_length                  |                 5 |      4.6000 |       0.0665 |
| extra_trees_top_40 | star_mean_position           |                 5 |      7.2000 |       0.0452 |
| extra_trees_top_40 | rna_mean_position            |                 5 |      7.6000 |       0.0442 |
| extra_trees_top_40 | modified_token_last_position |                 5 |      8.2000 |       0.0424 |
| extra_trees_top_40 | star_position_std            |                 5 |      8.4000 |       0.0430 |
| extra_trees_top_40 | rna_last_position            |                 5 |      9.4000 |       0.0390 |
| extra_trees_top_40 | base_G_count                 |                 5 |     11.0000 |       0.0345 |
| extra_trees_top_40 | base_C_count                 |                 5 |     12.2000 |       0.0307 |
| extra_trees_top_40 | rna_position_std             |                 5 |     12.6000 |       0.0292 |
| extra_trees_top_40 | methyl_position_std          |                 5 |     14.8000 |       0.0215 |
| extra_trees_top_40 | rna_count_middle             |                 5 |     15.0000 |       0.0198 |
| extra_trees_top_40 | token_rC_count               |                 5 |     15.0000 |       0.0221 |
| extra_trees_top_40 | token_rG_count               |                 5 |     17.2000 |       0.0162 |
| extra_trees_top_40 | rna_count_5p                 |                 5 |     18.4000 |       0.0127 |
| extra_trees_top_40 | rna_count_3p                 |                 5 |     18.6000 |       0.0124 |
| extra_trees_top_40 | modified_token_position_std  |                 5 |     20.0000 |       0.0104 |

## Outputs

- `mainpeak_reduced_feature_regression_metrics.csv`
- `mainpeak_reduced_feature_class_metrics.csv`
- `mainpeak_reduced_feature_oof_predictions.csv`
- `mainpeak_reduced_feature_selected_features.csv`
- `mainpeak_reduced_feature_selected_feature_summary.csv`
- `mainpeak_reduced_feature_report.md`
