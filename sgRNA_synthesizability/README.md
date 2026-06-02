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

The current model matrix has 161 numeric features generated from the decorated `Sequence` field. The parser treats `+A`, `+C`, `+G`, `+U`, and `+T` as LNA nucleotides and normalizes them internally to `lA`, `lC`, `lG`, `lU`, and `lT`. The persisted feature matrix also includes an `id` column from `Ref ID` for linkage back to the raw data, but `id` is not used as a model feature.

| Feature family | How it is calculated | Relationship with failure target |
| --- | --- | --- |
| Parsing completeness | Counts and fractions of decorated-sequence characters not consumed by the parser. | After interpreting `+` as LNA notation, the known `+` examples are no longer counted as unknown characters. Unknown-character features remain as data-quality guards for genuinely unsupported syntax. |
| Sequence size and base composition | Decorated string length, parsed token length, undecorated base length, A/C/G/U/T-equivalent counts and fractions, GC/AU, purine/pyrimidine fractions, terminal GC count, and maximum GC fraction in 10-base windows. | Composition features are important in the random-forest ranking. `base_U_fraction`, `base_T_fraction`, `base_G_fraction`, `base_C_fraction`, and `pyrimidine_fraction` are among the higher-importance model features. |
| Motif and run structure | Longest homopolymer run, longest GC run, repeated dinucleotide count, and longest consecutive runs for chemistry token families and individual `rA`, `rC`, `rG`, `rU`, `mA`, `mC`, `mG`, `mU`, `lA`, `lC`, `lG`, `lU`, and `lT` tokens. | Run and repeat features show meaningful signal. LNA run features dominate the top univariate rankings, while repeated dinucleotides remain among the strongest random-forest importance features. |
| Modification burden | Counts, densities, and terminal counts for slash-delimited modifications, phosphorothioate `*` markers, methyl tokens, LNA tokens, RNA tokens, and all modified tokens. | LNA count and fraction features are the strongest univariate signals after correcting `+` parsing, especially `token_lT_count`, `token_lA_count`, `token_lA_fraction`, and `token_lT_fraction`. |
| Positional modification features | For `star`, `slash_mod`, `methyl`, `lna`, `rna`, and `modified_token` families, the parser records first, last, mean, standard deviation, relative fractions, span, counts in 5-prime/middle/3-prime thirds, and window densities. | Position-aware features are repeatedly selected by both tests and model importance. LNA middle-window count/density and LNA positional spread are strong univariate signals, while `star_position_std`, `rna_position_std`, and `modified_token_position_std` are important to tree models. |

Top statistically ranked features from `analysis/modeling_outputs/feature_rankings.csv`:

| Feature | Mann-Whitney p | Rank-biserial | Fail - Pass mean difference | Cohen's d | RF importance |
| --- | ---: | ---: | ---: | ---: | ---: |
| `token_lT_count` | 7.553e-14 | 0.0450 | 0.1833 | 1.0957 | 0.0000 |
| `token_lA_count` | 7.553e-14 | 0.0450 | 0.1717 | 1.0756 | 0.0000 |
| `token_lA_fraction` | 7.554e-14 | 0.0450 | 0.0014 | 1.0773 | 0.0000 |
| `token_lT_fraction` | 7.554e-14 | 0.0450 | 0.0015 | 1.1012 | 0.0000 |
| `longest_lT_run` | 7.729e-14 | 0.0450 | 0.0798 | 1.0129 | 0.0000 |
| `longest_lA_run` | 8.279e-14 | 0.0449 | 0.0449 | 0.8272 | 0.0000 |
| `longest_lC_run` | 8.210e-13 | 0.0341 | 0.0337 | 0.6171 | 0.0000 |
| `token_lC_count` | 8.384e-13 | 0.0341 | 0.0329 | 0.4248 | 0.0000 |
| `token_lC_fraction` | 8.384e-13 | 0.0341 | 0.0003 | 0.4579 | 0.0000 |
| `token_lG_fraction` | 2.770e-12 | 0.0446 | 0.0009 | 0.8487 | 0.0000 |

The random-forest importance ranking emphasizes positional spread of `*`, repeated dinucleotides, base composition, and RNA-token position/fraction features. The top five random-forest features are `star_position_std`, `repeated_dinucleotide_count`, `base_U_fraction`, `base_G_fraction`, and `base_T_fraction`.

## Model Summary

Models were evaluated with 5-fold stratified grouped cross-validation, using the decorated `Sequence` as the grouping variable to reduce sequence-level leakage. All learned models used the 161 engineered numeric features except `elastic_net_logistic_with_char_ngrams`, which combined the engineered features with character n-gram TF-IDF features from the decorated sequence.

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
| `hist_gradient_boosting` | 0.1087 | 0.6304 | 0.0319 | 0.0698 | 0.0581 | 0.1278 | 0.0906 | 2549 / 11 / 80 / 6 |
| `lightgbm_balanced` | 0.1030 | 0.6172 | 0.0871 | 0.1628 | 0.0349 | 0.1429 | 0.0868 | 2320 / 240 / 63 / 23 |
| `xgboost_weighted` | 0.0907 | 0.6238 | 0.1232 | 0.1744 | 0.0116 | 0.1504 | 0.0868 | 2174 / 386 / 58 / 28 |
| `balanced_random_forest` | 0.0848 | 0.6230 | 0.1095 | 0.0233 | 0.0000 | 0.1654 | 0.0943 | 2464 / 96 / 65 / 21 |
| `logistic_l2_balanced` | 0.0767 | 0.5930 | 0.1874 | 0.1047 | 0.0000 | 0.1128 | 0.0717 | 1896 / 664 / 52 / 34 |
| `elastic_net_logistic_with_char_ngrams` | 0.0762 | 0.5833 | 0.1979 | 0.0814 | 0.0000 | 0.1203 | 0.0906 | 1982 / 578 / 51 / 35 |
| `elastic_net_logistic` | 0.0740 | 0.5907 | 0.1877 | 0.1047 | 0.0000 | 0.1128 | 0.0792 | 1899 / 661 / 51 / 35 |
| `logistic_l1_balanced` | 0.0702 | 0.5925 | 0.1868 | 0.0930 | 0.0000 | 0.1128 | 0.0755 | 1870 / 690 / 52 / 34 |
| `rule_baseline` | 0.0402 | 0.4989 | 0.1601 | 0.0116 | 0.0116 | 0.0602 | 0.0377 | 2441 / 119 / 79 / 7 |
| `prevalence_baseline` | 0.0322 | 0.4953 | 0.0314 | 0.0000 | 0.0000 | 0.0602 | 0.0377 | 2560 / 0 / 86 / 0 |
| `linear_svm_rbf_calibrated_probability` | 0.0282 | 0.4110 | 0.0316 | 0.0000 | 0.0000 | 0.0301 | 0.0377 | 2560 / 0 / 86 / 0 |

The current selected model is `hist_gradient_boosting`, chosen by highest out-of-fold average precision among trained model candidates. Its average precision is about 3.4 times the empirical failure prevalence baseline, and its top 5% risk bin has 12.8% failures versus a 3.2% overall failure rate. `lightgbm_balanced`, `xgboost_weighted`, and `balanced_random_forest` are also useful candidate models for triage-oriented workflows because they provide competitive top-bin enrichment and, for LightGBM/XGBoost, higher recall at 25% precision.

## Key Findings

- The dataset is highly imbalanced: only 86 of 2,646 binary labeled samples are failures. Metrics that focus on ranking and positive-class discovery are more informative than accuracy.
- The best current model is `hist_gradient_boosting`, with out-of-fold average precision 0.1087, ROC AUC 0.6304, Brier score 0.0319, and 12.8% failure precision in the top 5% highest-risk samples.
- The strongest single-feature statistical signals are LNA features derived from `+` notation. Failures have higher LNA A/T counts and fractions, and LNA middle-window features are prominent in the univariate rankings.
- Positional chemistry features add useful signal. LNA position/spread, RNA-token position, modified-token mean position, and positional spread of `*` and modified tokens appear in the statistical or model-importance rankings.
- Run and motif features matter. LNA run features dominate the univariate ranking, while repeated dinucleotide count remains one of the strongest random-forest importance features.
- Base-composition features contribute to model decisions, especially U/T-equivalent, C, G, pyrimidine, and RNA-token fractions.
- Absolute model performance remains modest, which is expected for a small rare-event dataset. The current models are more suitable for prioritizing high-risk samples for review than for making hard pass/fail calls at a default 0.5 threshold.