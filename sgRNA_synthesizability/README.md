# sgRNA Synthesizability

This project builds a reproducible workflow for predicting whether a decorated sgRNA sequence is likely to fail HPLC QC. The current training set is derived from `data/cleaned_data.csv` and contains 2,657 cleaned rows: 2,646 binary labeled rows used for model selection, 86 `Fail` rows, 2,560 `Pass` rows, and 11 `Needs Review` rows held out from model-selection metrics.

The modeling workflow treats `Fail` as the positive class. Because failures are rare, the main model-selection metrics are average precision, ROC AUC, Brier score, recall at fixed precision, and enrichment in the highest-risk bins rather than accuracy.

## Reproducible Outputs

- Feature parser: `feature_engineering.py`
- Model training and evaluation: `train_baseline_model.py`
- Feature matrix and target outputs: `../analysis/features/`
- Model metrics and selected model artifacts: `../analysis/modeling_outputs/`
- Per-feature plots and statistical tests: `../analysis/feature_target_plots/`
- Full feature definitions: `.github/feature_list.md`

## Feature Summary

The current model matrix has 131 numeric features generated from the decorated `Sequence` field. The persisted feature matrix also includes an `id` column from `Ref ID` for linkage back to the raw data, but `id` is not used as a model feature.

| Feature family | How it is calculated | Relationship with failure target |
| --- | --- | --- |
| Parsing completeness | Counts and fractions of decorated-sequence characters not consumed by the parser. | The strongest univariate signal is parser-unrecognized content: `unknown_char_count` and `unknown_char_fraction` have Mann-Whitney p-values near `4.90e-10`, positive mean differences in failures, and Cohen's d near 1.05. |
| Sequence size and base composition | Decorated string length, parsed token length, undecorated base length, A/C/G/U/T-equivalent counts and fractions, GC/AU, purine/pyrimidine fractions, terminal GC count, and maximum GC fraction in 10-base windows. | Composition features are important in the random-forest ranking. `base_U_fraction`, `base_T_fraction`, `base_G_fraction`, `base_C_fraction`, and `pyrimidine_fraction` are among the higher-importance model features. |
| Motif and run structure | Longest homopolymer run, longest GC run, repeated dinucleotide count, and longest consecutive runs for chemistry token families and individual `rA`, `rC`, `rG`, `rU`, `mA`, `mC`, `mG`, and `mU` tokens. | Run and repeat features show meaningful signal. `longest_rU_run`, `longest_homopolymer_run`, `longest_rA_run`, and `repeated_dinucleotide_count` are among the top univariate features; repeated dinucleotides are also the top random-forest importance feature. |
| Modification burden | Counts, densities, and terminal counts for slash-delimited modifications, phosphorothioate `*` markers, methyl tokens, RNA tokens, and all modified tokens. | Global modification burden is weaker than positional information, but `star_density`, `slash_mod_count`, and unknown-character burden are useful enough to support a simple rule baseline. |
| Positional modification features | For `star`, `slash_mod`, `methyl`, `rna`, and `modified_token` families, the parser records first, last, mean, standard deviation, relative fractions, span, counts in 5-prime/middle/3-prime thirds, and window densities. | Position-aware features are repeatedly selected by both tests and model importance. `rna_mean_fraction`, `rna_mean_position`, `rna_count_3p`, `methyl_mean_fraction`, `modified_token_mean_fraction`, `star_position_std`, and `modified_token_position_std` are notable signals. |

Top statistically ranked features from `analysis/modeling_outputs/feature_rankings.csv`:

| Feature | Mann-Whitney p | Rank-biserial | Fail - Pass mean difference | Cohen's d | RF importance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `unknown_char_count` | 4.896e-10 | 0.0439 | 0.5003 | 1.0480 | 0.0004 |
| `unknown_char_fraction` | 4.896e-10 | 0.0439 | 0.0019 | 1.0500 | 0.0006 |
| `longest_rU_run` | 1.277e-04 | -0.2028 | -0.2716 | -0.3265 | 0.0110 |
| `rna_mean_fraction` | 7.229e-04 | -0.2088 | -0.0093 | -0.1825 | 0.0180 |
| `rna_count_3p` | 3.032e-03 | -0.1825 | -2.2960 | -0.1860 | 0.0134 |
| `repeated_dinucleotide_count` | 3.514e-03 | -0.1822 | -0.6506 | -0.0574 | 0.0274 |
| `longest_homopolymer_run` | 3.758e-03 | -0.1533 | -0.2087 | -0.0995 | 0.0031 |
| `longest_rA_run` | 4.637e-03 | -0.1640 | -0.2568 | -0.1142 | 0.0038 |
| `methyl_mean_fraction` | 7.274e-03 | 0.1659 | 0.0386 | 0.0766 | 0.0078 |
| `rna_mean_position` | 1.096e-02 | -0.1582 | -1.9480 | -0.1620 | 0.0161 |

The random-forest importance ranking emphasizes repeated dinucleotides, positional spread of `*` and modified tokens, base composition, and RNA-token position/fraction features. The top five random-forest features are `repeated_dinucleotide_count`, `star_position_std`, `base_U_fraction`, `token_rC_fraction`, and `base_T_fraction`.

## Model Summary

Models were evaluated with 5-fold stratified grouped cross-validation, using the decorated `Sequence` as the grouping variable to reduce sequence-level leakage. All learned models used the 131 engineered numeric features except `elastic_net_logistic_with_char_ngrams`, which combined the engineered features with character n-gram TF-IDF features from the decorated sequence.

| Model | Type and inputs | Notes |
| --- | --- | --- |
| `prevalence_baseline` | Constant train-fold failure rate. | Calibration reference for the rare positive class. |
| `rule_baseline` | Hand-built score from standardized length, star density, slash modification count, unknown character count, and max GC window. | Simple interpretable baseline. |
| `logistic_l2_balanced` | Class-weighted L2 logistic regression on numeric features. | Linear baseline with balanced class weights. |
| `logistic_l1_balanced` | Class-weighted L1 logistic regression on numeric features. | Sparse linear baseline. |
| `elastic_net_logistic` | Class-weighted elastic-net logistic regression on numeric features. | Linear model with mixed L1/L2 regularization. |
| `linear_svm_rbf_calibrated_probability` | Balanced RBF SVM with calibrated probabilities. | Nonlinear margin model; probabilities are calibrated by cross-validation. |
| `hist_gradient_boosting` | Histogram gradient boosting on numeric features. | Current best overall model by out-of-fold average precision. |
| `balanced_random_forest` | Random forest with balanced subsample class weighting. | Strong top-risk-bin enrichment and feature-importance source. |
| `elastic_net_logistic_with_char_ngrams` | Elastic-net logistic regression on numeric features plus 2- to 5-character sequence n-grams. | Tests whether raw decorated sequence fragments add signal. |
| `lightgbm_balanced` | LightGBM gradient boosting with balanced class weights. | Best recall at 25% precision and competitive top-risk-bin enrichment. |
| `xgboost_weighted` | XGBoost with positive-class weighting. | Weighted boosted-tree comparison model. |

Overall out-of-fold metrics from `analysis/modeling_outputs/cv_metrics.csv`:

| Model | Average precision | ROC AUC | Brier score | Recall at precision 25% | Recall at precision 50% | Precision top 5% | Precision top 10% | Confusion matrix at 0.5 threshold (TN/FP/FN/TP) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hist_gradient_boosting` | 0.1119 | 0.6345 | 0.0317 | 0.1279 | 0.0349 | 0.1353 | 0.0868 | 2550 / 10 / 81 / 5 |
| `lightgbm_balanced` | 0.1076 | 0.6232 | 0.0862 | 0.2209 | 0.0000 | 0.1579 | 0.0943 | 2315 / 245 / 61 / 25 |
| `balanced_random_forest` | 0.0924 | 0.6245 | 0.1082 | 0.0116 | 0.0116 | 0.1654 | 0.0943 | 2465 / 95 / 65 / 21 |
| `elastic_net_logistic_with_char_ngrams` | 0.0875 | 0.5952 | 0.1926 | 0.1163 | 0.0000 | 0.1278 | 0.0943 | 2027 / 533 / 51 / 35 |
| `xgboost_weighted` | 0.0870 | 0.6159 | 0.1244 | 0.0116 | 0.0116 | 0.1504 | 0.0906 | 2176 / 384 / 57 / 29 |
| `rule_baseline` | 0.0870 | 0.4891 | 0.0837 | 0.0581 | 0.0581 | 0.0526 | 0.0453 | 2553 / 7 / 81 / 5 |
| `logistic_l2_balanced` | 0.0699 | 0.5883 | 0.1884 | 0.1047 | 0.0000 | 0.1128 | 0.0792 | 1908 / 652 / 53 / 33 |
| `logistic_l1_balanced` | 0.0696 | 0.5924 | 0.1885 | 0.0930 | 0.0000 | 0.1203 | 0.0792 | 1865 / 695 / 52 / 34 |
| `elastic_net_logistic` | 0.0679 | 0.5815 | 0.1993 | 0.0000 | 0.0000 | 0.1128 | 0.0755 | 1829 / 731 / 52 / 34 |
| `prevalence_baseline` | 0.0322 | 0.4953 | 0.0314 | 0.0000 | 0.0000 | 0.0602 | 0.0377 | 2560 / 0 / 86 / 0 |
| `linear_svm_rbf_calibrated_probability` | 0.0283 | 0.4061 | 0.0316 | 0.0000 | 0.0000 | 0.0301 | 0.0415 | 2560 / 0 / 86 / 0 |

The current selected model is `hist_gradient_boosting`, chosen by highest out-of-fold average precision among trained model candidates. Its average precision is about 3.5 times the empirical failure prevalence baseline, and its top 5% risk bin has 13.5% failures versus a 3.2% overall failure rate. `lightgbm_balanced` and `balanced_random_forest` are also useful candidate models for triage-oriented workflows because they provide higher top-bin failure enrichment and, for LightGBM, higher recall at 25% precision.

## Key Findings

- The dataset is highly imbalanced: only 86 of 2,646 binary labeled samples are failures. Metrics that focus on ranking and positive-class discovery are more informative than accuracy.
- The best current model is `hist_gradient_boosting`, with out-of-fold average precision 0.1119, ROC AUC 0.6345, Brier score 0.0317, and 13.5% failure precision in the top 5% highest-risk samples.
- The strongest single-feature statistical signal is parser-unrecognized sequence content. Failures have higher `unknown_char_count` and `unknown_char_fraction`, suggesting that unusual decoration syntax or chemistry tokens may mark synthesis risk or data irregularity worth reviewing.
- Positional chemistry features add useful signal. RNA-token position, 3-prime RNA counts, methyl-token mean position, modified-token mean position, and positional spread of `*` and modified tokens appear in the statistical or model-importance rankings.
- Run and motif features matter. `longest_rU_run`, `longest_rA_run`, `longest_homopolymer_run`, and `repeated_dinucleotide_count` are associated with the target, and repeated dinucleotide count is the highest random-forest importance feature.
- Base-composition features contribute to model decisions, especially U/T-equivalent, C, G, pyrimidine, and RNA-token fractions.
- Absolute model performance remains modest, which is expected for a small rare-event dataset. The current models are more suitable for prioritizing high-risk samples for review than for making hard pass/fail calls at a default 0.5 threshold.