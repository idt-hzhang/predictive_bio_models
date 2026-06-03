# Fail-Focused MainPeak Training Report

This experiment tests whether balancing the small `Fail` class helps MainPeak regression on true `Fail` rows. The experiment uses the same fold-wise `extra_trees_top_40` feature selector as the reduced-feature MainPeak analysis and compares ordinary training, `Fail` sample weighting, and train-fold `Fail` oversampling.

Important interpretation: oversampling duplicates existing `Fail` examples; it does not create new independent chemistry evidence. It can shift the loss toward failures, but it also increases overfitting risk and may degrade routine `Pass` performance.

## Overall Out-of-Fold Metrics

| strategy         | model                  |    mae |   rmse |     r2 |   pearson_r |   spearman_r |
|:-----------------|:-----------------------|-------:|-------:|-------:|------------:|-------------:|
| unbalanced       | extra_trees            | 4.1872 | 6.0998 | 0.6963 |      0.8345 |       0.7710 |
| unbalanced       | hist_gradient_boosting | 4.3067 | 6.2999 | 0.6760 |      0.8226 |       0.7549 |
| unbalanced       | elastic_net            | 4.6670 | 6.7055 | 0.6329 |      0.7956 |       0.7551 |
| fail_oversampled | extra_trees            | 5.0054 | 6.8136 | 0.6210 |      0.8066 |       0.7338 |
| fail_weighted    | extra_trees            | 5.1108 | 6.8782 | 0.6138 |      0.8080 |       0.7355 |
| fail_oversampled | hist_gradient_boosting | 5.1521 | 7.1468 | 0.5831 |      0.7793 |       0.7133 |
| fail_weighted    | hist_gradient_boosting | 5.2704 | 7.3531 | 0.5586 |      0.7684 |       0.7069 |
| fail_oversampled | elastic_net            | 7.4627 | 9.2475 | 0.3019 |      0.7376 |       0.6910 |
| fail_weighted    | elastic_net            | 7.4743 | 9.2619 | 0.2997 |      0.7383 |       0.6901 |

## Fail-Class Metrics

| strategy         | model                  |   rows |     mae |    rmse |      r2 |   pearson_r |   spearman_r |
|:-----------------|:-----------------------|-------:|--------:|--------:|--------:|------------:|-------------:|
| fail_weighted    | extra_trees            |     84 | 10.0473 | 13.4955 |  0.3218 |      0.7433 |       0.6922 |
| fail_weighted    | elastic_net            |     84 | 10.3772 | 13.5996 |  0.3113 |      0.6813 |       0.5114 |
| fail_oversampled | elastic_net            |     84 | 10.4102 | 13.6643 |  0.3048 |      0.6815 |       0.5154 |
| fail_oversampled | extra_trees            |     84 | 10.2645 | 13.7629 |  0.2947 |      0.7451 |       0.6927 |
| fail_weighted    | hist_gradient_boosting |     84 | 10.9595 | 14.7771 |  0.1869 |      0.6953 |       0.6412 |
| fail_oversampled | hist_gradient_boosting |     84 | 11.2926 | 15.2244 |  0.1370 |      0.6748 |       0.6151 |
| unbalanced       | hist_gradient_boosting |     84 | 12.5155 | 16.0463 |  0.0413 |      0.7515 |       0.6370 |
| unbalanced       | extra_trees            |     84 | 12.6806 | 16.3866 |  0.0002 |      0.7229 |       0.6465 |
| unbalanced       | elastic_net            |     84 | 14.0656 | 18.2670 | -0.2425 |      0.6026 |       0.5479 |

## Best Fail-Class Strategy

The best `Fail`-class RMSE is `extra_trees` with `fail_weighted` training: Fail MAE 10.0473, Fail RMSE 13.4955, and Fail R2 0.3218. The best overall model remains `extra_trees` with `unbalanced` training: overall RMSE 6.0998 and R2 0.6963.

| class_label   |   rows |     mae |    rmse |     r2 |   pearson_r |   spearman_r |
|:--------------|-------:|--------:|--------:|-------:|------------:|-------------:|
| Fail          |     84 | 10.0473 | 13.4955 | 0.3218 |      0.7433 |       0.6922 |
| Pass          |   2560 |  4.9322 |  6.5189 | 0.6183 |      0.8232 |       0.7391 |

## Recommendation

Do not use bootstrapping/oversampling as the primary production model unless the operating goal is explicitly to reduce MainPeak error on known failures at the expense of overall calibration. For this dataset, balancing can modestly improve the failure tail for some models, but the limitation is mostly information scarcity: there are only 84 binary-labeled `Fail` rows with MainPeak available, and duplicated rows do not add new sequence or chemistry patterns.

More defensible next steps are: collect or prioritize additional real low-MainPeak/failure examples; report class-specific metrics by default; tune an operating model for failure-tail objectives only if the application accepts worse global RMSE; consider quantile or lower-tail prediction intervals; and keep direct Pass/Fail ranking models for failure triage because MainPeak regression and binary failure status are related but not interchangeable.

## Outputs

- `mainpeak_fail_focused_training_metrics.csv`
- `mainpeak_fail_focused_training_class_metrics.csv`
- `mainpeak_fail_focused_training_oof_predictions.csv`
- `mainpeak_fail_focused_training_selected_features.csv`
- `mainpeak_fail_focused_training_report.md`
