# sgRNA Synthesizability

This project builds a reproducible workflow for predicting whether a decorated sgRNA sequence is likely to fail HPLC QC. The current training set is derived from `data/cleaned_data.csv` and contains 2,657 cleaned rows: 2,646 binary labeled rows used for model selection, 86 `Fail` rows, 2,560 `Pass` rows, and 11 `Needs Review` rows held out from model-selection metrics.

The modeling workflow treats `Fail` as the positive class. Because failures are rare, the main model-selection metrics are average precision, ROC AUC, Brier score, recall at fixed precision, and enrichment in the highest-risk bins rather than accuracy.

## Reproducible Outputs

- Feature parser: `feature_engineering.py`
- Model training and evaluation: `train_baseline_model.py`
- Learning-curve analysis: `../analysis/learning_curve_analysis.py`
- Post-selection diagnostics: `../analysis/next_step_analyses.py`
- Correlation-pruned feature-set modeling: `../analysis/correlation_reduced_modeling.py`
- Feature matrix and target outputs: `../analysis/features/`
- Correlation-pruned feature matrix and target outputs: `../analysis/features_correlation_reduced/`
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

Bootstrap stability analysis from `analysis/modeling_outputs/feature_stability.csv` shows the most stable model-derived features are dominated by motif/composition features rather than the LNA features that dominate univariate tests. The highest-selection-frequency stable features are `repeated_dinucleotide_count` (selection frequency 0.93), `token_rC_fraction` (0.88), `base_C_fraction` (0.87), `token_rG_fraction` (0.86), `purine_fraction` (0.85), `base_G_fraction` (0.84), `base_T_fraction` (0.84), `pyrimidine_fraction` (0.83), and `base_U_fraction` (0.79). This split suggests that LNA burden is a strong marginal signal, while the selected tree model relies more consistently on composition and repeat structure under resampling.

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

## Post-Selection Diagnostics

The selected `hist_gradient_boosting` model was examined with learning curves, bootstrap feature stability, reduced-feature models, error analysis, Needs Review scoring, and risk tiering.

### Learning Curve

Learning-curve analysis trained the selected model on 10%, 20%, 40%, 60%, 80%, and 100% of each training fold while preserving the same 5-fold stratified grouped CV strategy. Average precision increased from 0.1026 at 80% of the training data to 0.1087 at 100%; ROC AUC increased from 0.6168 to 0.6304; and Precision@5% increased from 0.1053 to 0.1278. Performance therefore still changes between 80% and 100% of the available training data, suggesting additional labeled data would theoretically help. Because more labeled failures are not expected, the practical path is to focus on better interpretation, stability, and operational review workflows using the current data.

### Simplified Models

Reduced-feature models using the top 5, 10, and 20 stable features did not match the full selected model. The best reduced model was `hist_gradient_boosting_top_20`, with average precision 0.0738 and ROC AUC 0.6113, about 68% of the full model's average precision. This does not meet the planned 10-15% performance-loss criterion for preferring a simpler deployment model. The full 161-feature `hist_gradient_boosting` model remains the recommended model for risk ranking.

### Correlation-Pruned Feature Set

A separate redundancy-reduction analysis removed features with absolute Pearson correlation greater than 0.90, retaining the higher-priority feature from each correlated pair using the existing feature-ranking table. This reduced the engineered feature set from 161 to 71 features and dropped 90 highly correlated features. The reduced feature matrix is written to `analysis/features_correlation_reduced/`, and model results are summarized in `analysis/modeling_outputs/correlation_reduced_model_comparison.md`.

The correlation-pruned feature set slightly improved the selected model: `hist_gradient_boosting` average precision increased from 0.1087 to 0.1139, ROC AUC increased from 0.6304 to 0.6529, and Precision@5% increased from 0.1278 to 0.1353. `xgboost_weighted` also improved in average precision, from 0.0907 to 0.1026. These results suggest that removing highly redundant features can modestly improve tree-based ranking performance and reduce feature-set complexity without losing useful signal.

### Risk Tiers

Risk tiering converts continuous out-of-fold predictions into operational review groups:

| Tier | Definition | Rows | Failures | Observed failure rate | Mean predicted risk | Enrichment |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| High Risk | Top 5% highest predicted risk | 132 | 16 | 0.1212 | 0.2506 | 3.73x |
| Medium Risk | Next 15% highest predicted risk | 397 | 16 | 0.0403 | 0.0536 | 1.24x |
| Low Risk | Remaining 80% | 2117 | 54 | 0.0255 | 0.0086 | 0.78x |

This supports using the model as a review-prioritization tool: the high-risk tier is enriched for failures, while the low-risk tier is below baseline failure prevalence.

### Error Analysis and Needs Review Samples

At the default 0.5 threshold, the selected model produces 80 false negatives and 11 false positives, reinforcing that the model should not be used as a hard pass/fail classifier. The false-positive report is still useful operationally because high-risk passing sequences often resemble difficult chemistry patterns and define the sequences most likely to be escalated for review.

The 11 `Needs Review` rows were rescored with the selected model and assigned percentile ranks relative to binary-labeled sequences. The highest-risk `Needs Review` examples have predicted risks of 0.0815 and 0.0346, corresponding to the 94.1st and 84.6th risk percentiles, while the remaining examples mostly fall much lower. This provides a small external sanity check: at least some ambiguous rows occupy a high-risk but not extreme-risk region.

### Design Rules and SHAP Status

Candidate design rules in `analysis/modeling_outputs/design_rules.md` remain review guidance rather than hard filters. Moderate-confidence signals include elevated LNA-A/LNA-T burden and longer LNA runs from univariate evidence, repeated dinucleotide structure and base-composition shifts from stability/model evidence, and phosphorothioate positional spread from model-derived stability evidence.

SHAP-specific plots were not generated because the configured conda environment does not include the `shap` package. No package was installed automatically, following the project environment instructions. The current interpretation evidence therefore comes from bootstrap stability, univariate statistics, feature rankings, error analysis, and risk-tier enrichment.

## Key Findings

- The dataset is highly imbalanced: only 86 of 2,646 binary labeled samples are failures. Metrics that focus on ranking and positive-class discovery are more informative than accuracy.
- The best current model is `hist_gradient_boosting`, with out-of-fold average precision 0.1087, ROC AUC 0.6304, Brier score 0.0319, and 12.8% failure precision in the top 5% highest-risk samples.
- Learning-curve results do not show a clear full plateau: AP, ROC AUC, and Precision@5% all improve from 80% to 100% of training data. Since additional labeled failures are not expected, future work should emphasize feature interpretation, stable review rules, and better use of existing signal.
- The high-risk tier is operationally useful: the top 5% highest-risk sequences show a 12.1% observed failure rate, or 3.73x enrichment over the 3.25% baseline.
- Reduced-feature models are not yet competitive with the full model. The best top-20-feature HGB model retains about 68% of full-model AP, so the full engineered feature set remains preferred.
- Correlation pruning is more promising than selecting only a small top-feature subset. Removing features with absolute correlation >0.90 reduced the feature set from 161 to 71 features and improved `hist_gradient_boosting` AP from 0.1087 to 0.1139.
- The strongest single-feature statistical signals are LNA features derived from `+` notation. Failures have higher LNA A/T counts and fractions, and LNA middle-window features are prominent in the univariate rankings.
- Bootstrap stability highlights a complementary set of robust model-derived features, especially repeated dinucleotide count, RNA-token fractions, base-composition fractions, purine/pyrimidine fractions, and positional spread features.
- Positional chemistry features add useful signal. LNA position/spread, RNA-token position, modified-token mean position, and positional spread of `*` and modified tokens appear in the statistical or model-importance rankings.
- Run and motif features matter. LNA run features dominate the univariate ranking, while repeated dinucleotide count remains one of the strongest random-forest importance features.
- Base-composition features contribute to model decisions, especially U/T-equivalent, C, G, pyrimidine, and RNA-token fractions.
- Absolute model performance remains modest, which is expected for a small rare-event dataset. The current models are more suitable for prioritizing high-risk samples for review than for making hard pass/fail calls at a default 0.5 threshold.