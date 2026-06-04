# Candidate Design Rules

Rules are derived from univariate feature rankings, random-forest/permutation stability, and current model behavior. They should be used for review prioritization, not hard rejection.

## Elevated LNA-A burden

Confidence: Moderate

Supporting features:

| feature           |   mannwhitney_p |   mean_difference_fail_minus_pass |   cohen_d |   selection_frequency |   mean_rank |
|:------------------|----------------:|----------------------------------:|----------:|----------------------:|------------:|
| longest_lA_run    |       8.279e-14 |                          0.04495  |    0.8272 |                     0 |       150.7 |
| token_lA_count    |       7.553e-14 |                          0.1717   |    1.076  |                     0 |       150.1 |
| token_lA_fraction |       7.554e-14 |                          0.001409 |    1.077  |                     0 |       150.7 |

## Elevated LNA-T burden

Confidence: Moderate

Supporting features:

| feature           |   mannwhitney_p |   mean_difference_fail_minus_pass |   cohen_d |   selection_frequency |   mean_rank |
|:------------------|----------------:|----------------------------------:|----------:|----------------------:|------------:|
| longest_lT_run    |       7.729e-14 |                          0.07983  |     1.013 |                     0 |       149.9 |
| token_lT_count    |       7.553e-14 |                          0.1833   |     1.096 |                     0 |       150.4 |
| token_lT_fraction |       7.554e-14 |                          0.001489 |     1.101 |                     0 |       148.9 |

## Longer LNA runs

Confidence: Moderate

Supporting features:

| feature        |   mannwhitney_p |   mean_difference_fail_minus_pass |   cohen_d |   selection_frequency |   mean_rank |
|:---------------|----------------:|----------------------------------:|----------:|----------------------:|------------:|
| longest_lA_run |       8.279e-14 |                           0.04495 |    0.8272 |                     0 |       150.7 |
| longest_lC_run |       8.21e-13  |                           0.03371 |    0.6171 |                     0 |       152.8 |
| longest_lT_run |       7.729e-14 |                           0.07983 |    1.013  |                     0 |       149.9 |

## Repeated dinucleotide structure

Confidence: Moderate

Supporting features:

| feature                     |   mannwhitney_p |   mean_difference_fail_minus_pass |   cohen_d |   selection_frequency |   mean_rank |
|:----------------------------|----------------:|----------------------------------:|----------:|----------------------:|------------:|
| repeated_dinucleotide_count |        0.003514 |                           -0.6506 |  -0.05737 |                  0.93 |       11.71 |

## Base-composition shifts

Confidence: Moderate

Supporting features:

| feature             |   mannwhitney_p |   mean_difference_fail_minus_pass |   cohen_d |   selection_frequency |   mean_rank |
|:--------------------|----------------:|----------------------------------:|----------:|----------------------:|------------:|
| base_C_fraction     |          0.457  |                          0.002704 |   0.05408 |                  0.87 |       17.91 |
| base_G_fraction     |          0.8989 |                         -0.001664 |  -0.03766 |                  0.84 |       22.39 |
| base_U_fraction     |          0.5436 |                         -0.004644 |  -0.09255 |                  0.79 |       26.84 |
| pyrimidine_fraction |          0.5992 |                         -0.00194  |  -0.02521 |                  0.83 |       26.63 |

## Phosphorothioate positional spread

Confidence: Moderate

Supporting features:

| feature           |   mannwhitney_p |   mean_difference_fail_minus_pass |   cohen_d |   selection_frequency |   mean_rank |
|:------------------|----------------:|----------------------------------:|----------:|----------------------:|------------:|
| star_position_std |          0.7161 |                            -1.602 |  -0.08057 |                  0.51 |       54.02 |

