# False Negative Report

Model: `hist_gradient_boosting` out-of-fold predictions at threshold 0.5.
False negatives: 80 of 86 failures.

## Lowest-Risk Missed Failures

|    Ref ID |   risk_score | Sequence                                                                                                                                                                                                                                                                                                                                         |
|----------:|-------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 628592042 |       0.0004 | rGrGrGrUrCrArGrGrCrCrGrGrCrGrArArArGrUrCrGrCrArCrArGrUrUrUrGrGrGrGrArArArGrCrUrGrUrGrCrArGrCrCrUrGrUrArArCrCrCrCrCrCrCrArCrGrArArArGrUrGrGrGrArArUrUrUrCrUrArCrUrGrUrUrGrUrArGrArUrCrArArArUrCrGrArUrCrCrGrUrGrArGrArArGrGrArArArUrArArUrUrUrCrUrArCrUrGrUrUrGrUrArGrArUrUrUrGrGrGrUrUrArGrGrGrUrUrArGrGrGrUrUrArGrGrG                           |
| 622668407 |       0.0011 | mG*mA*mU*rGrUrCrUrGrCrArGrGrCrCrArGrArUrGrArGrUrUrUrUrArGrArGrCrUrArGrArArArUrArGrCrArArGrUrUrArArArArUrArArGrGrCrUrArGrUrCrCrGrUrUrArUrCrArArCrUrUrGrArArArArArGrUrGrGrCrArCrCrGrArGrUrCrGrGrUrGrCrArArUrGrUrGrCrCrArUrCrUrGrGrArGrCrArCrUrCrArUrCrUrGrGrCrCrUrGrCrArGrArUrU*mU*mU*mUrU                                                         |
| 628061480 |       0.0011 | mU*mC*mU*rGrCrCrArUrArGrCrCrUrGrUrGrCrCrCrArGrUrUrUrUrArGrArGrCrUrGrUrGrCrUrGrArArArArGrCrArCrArGrCrArCrGrUrUrArArArArUrArArGrGrCrArGrUrGrArUrUrGrArArArArArUrCrCrArGrUrCrCrGrUrArUrUrCrArGrCrUrUrGrArArArArArGrUrGrArGrCrArCrCrGrArArUrCrGrGrUrGrCrGrGrUrCrCrCrGrGrGrArArArGrGrArGrArGrUrCrCrUrUrArUrCrCrArArArGrGrCrArCrArGrG*mC*mU*mA         |
| 617408190 |       0.0012 | /AltR1/rUrArArUrUrUrCrUrArCrUrArArGrUrGrUrArGrArUrUrCrGrCrUrArUrCrUrCrGrArUrGrCrCrCrCrGrC/AltR2/                                                                                                                                                                                                                                                 |
| 612550846 |       0.0012 | mG*mG*C*CArArGrGrUrArUrUrGrUrGrGrCAGCrGrUrUrUrUrArGrAmGmCmUmAmGmAmAmAmUmAmGmCrArArGrUrUrArArArArUrArArGrGrCrUrArGrUrCrCrGrUrUrArUrCrAmAmCmUmUmGmAmAmAmAmAmGmUmGmGmCmAmCmCmGmAmGmUmCmGmGmUmGmCmU*mU*mU*mU                                                                                                                                         |
| 616954920 |       0.0014 | mG*mU*mC*rUrUrGrArGrArG*mG*mA*mA*mArCrUrCrUrCrArArGrArC*mU*rArGrArUrUrGrCrUrCrCrUrUrArCrGrArGrGrArGrArCrUrUrArCrCrUrGrCrArCrGrArUrGrUmA*mG*mU                                                                                                                                                                                                    |
| 623559076 |       0.0017 | /AltR1/rCrCrUrCrUrArArGrGrUrUrUrGrCrUrUrArCrGrArGrUrUrUrUrArGrArGrCrUrArUrGrCrU/AltR2/                                                                                                                                                                                                                                                           |
| 599359171 |       0.0020 | rGrArUrUrUrArGrArCrCrArCrCrCrCrArArArArArUrGrArArGrGrGrGrArCrUrArArArArCrUrCrUrGrUrUrCrArCrCrArArUrArUrUrCrCrArGrGrCrArC                                                                                                                                                                                                                         |
| 597224368 |       0.0021 | /56-FAM/rAmG*mC*mA*rGrArCrUrUrCrUrCrCrArCrArGrGrArGrUrCrArGrGrUrUrArUrUrGrUrArCrUrCrUrCrArArUrArArArArArGrUrUrArUrUrGrArGrArArUrCrUrArCrArArUrArArUrArArGrGrCrArUrCrUrUrGrCrCrGrArArUrUrUrArCrCrGrCrCrCrUrArCrArUrArUrGrUrArGrGrGrCrGrGrU*mU*mU*mU                                                                                               |
| 599475183 |       0.0023 | mG*mC*mU*rArCrArGrGrGrUrGrGrGrCrUrUrCrUrUrCrCrUrGrUrGrUrCrArUrArGrUrUrCrCrArUrCrGrArArArGrUrGrArUrGrUrUrUrCrUrArUrGrArUrArArGrGrGrUrCrGrCrGrUrUrCrGrCrGrCrGrArCrCrCrGrUrGrGrCrGrUrUrGrGrGrGrArUrCrGrCrCrUrGrCrCrC*rUrUrCrGrGrGrGrCrGrUrCrUrCrCrCrCrA*mU*mU*mU                                                                                    |
| 616184963 |       0.0027 | mG*mC*mU*rCrGrCrGrArArCrArGrUrUrGrGrCrCrCrUrGrUrUrUrUrArGrArGrCrUrArGrArArArUrArGrCrArArGrUrUrArArArArUrArArGrGrCrUrArGrUrCrCrGrUrUrArUrCrArArCrUrUrGrArArArArArGrUrGrGrCrArCrCrGrArGrUrCrGrGrUrGrCrArCrUrCrArUrGrCrUrCrArArGrGrGrCrCrArArCrUrGrUrUrCrGrCrUrUrUrU*mU*mU*mU                                                                       |
| 599475056 |       0.0028 | mG*mC*mU*rArCrArGrGrGrUrGrGrGrCrUrUrCrUrUrCrCrUrGrUrGrUrCrArUrArGrUrUrCrCrArUrCrGrArArArGrUrGrArUrGrUrUrUrCrUrArUrGrArUrArArGrGrGrUrCrGrCrGrUrUrCrG/i2FC/rGrCrGrArCrCrCrGrUrGrGrCrGrUrUrGrGrGrGrArUrCrGrCrCrUrGrCrCrCrUrUrCrGrGrGrGrCrGrUrCrUrCrCrCrCrA*mU*mU*mU                                                                                 |
| 613340538 |       0.0036 | mA*mC*mU*rGrGrCrGrCrUrUrCrUrArUrCrUrGrArUrUrArCrUrCrUrGrArGrCrGrCrCrArUrCrArCrCrArGrCrGrArCrUrArUrGrUrCrGrUrArGrUrGrGrGrUrArArArGrCrUrCrCrCrUrCrUrUrCrGrGrArGrGrGrArGrCrArUrCrArGrArGrUrGrGrArGrCrCrUrGrUrGrArUrArArArArG*mC*mA*mA                                                                                                               |
| 594634842 |       0.0042 | mU*mA*mC*rGrUrArCrCrArGrGrUrGrGrArGrCrArCrCrCrGrUrCrArUrArGrUrCrCrCrArUrCrArUrArUmC*mC*mA*rArArArG+T+G+G+A+T+A+T+G+A+T+GrUrUrUrCrUrArUrGrArUrArArGrGrGrCrUrCrUrCrUrArArGrArGrArGrArCrCrUrGrUrGrGrCrGrUrUrGrGrGrGrArUrCrGrCrCrUrGrCrCrCrGrUrUrUrCrGrArCrGrGrGrUrGrUrCrUrCrCrCrCrA*mU*mU*mU                                                        |
| 624384315 |       0.0043 | mC*mA*mC*rArGrCrGrGrUrGrCrGrGrArGrArGrGrGrArGrUrUrUrCrArGrArGrCrCrArGrArArArUrGrGrCrArArGrUrUrGrArArArUrArArGrGrCrUrArGrUrCrCrGrUrUrArUrCrArArCrUrUrGrCrCrUrArCrGrGrGrCrArArGrUrGrGrCrArCrCrGrArGrUrCmGmG*mU*mG*mC                                                                                                                               |
| 630436740 |       0.0044 | rUrArArArGrArGrCrUrCrArGrArGrUrCrGrUrCrUrCrGrUrCrUrCrGrGrCrArCrUrArUrCrArGrArGrUrGrGrArGrGrArCrArArUrGrUrCrGrUrArArUrUrGrArCrGrCrGrArGrCrGrGrCrArGrArGrGrUrArArGrCrCrGrGrGrUrGrArCrArGrUrGrGrArArGrGrGrCrCrArUrUrArCrCrGrArCrArArCrUrCrArGrGrArArArGrGrUrUrGrUrUrGrGrCrArUrUrGrUrCrCrUrCrGrGrArGrGrGrUrArUrGrArCrCrGrCrGrUrArUrUrArCrCrGrUrArArC |
| 612091031 |       0.0046 | mA*mA*mU*rUrArUrGrCrGrGrArUrUrArCrUrArGrGrArGrUrUrUrUrArGrArGrCrUrArGrArArArUrArGrCrArArGrUrUrArArArArUrArArGrGrCrUrArGrUrCrCrGrUrUrArC*mA*mA*mC                                                                                                                                                                                                 |
| 612752922 |       0.0047 | mC*mU*T*ATrArUrCCAArCrArCrUrUrCrGrUrGrGrUrUrUrUrArGrAmGmCmUmAmGmAmAmAmUmAmGmCrArArGrUrUrArArArArUrArArGrGrCrUrArGrUrCrCrGrUrUrArUrCrAmAmCmUmUmGmAmAmAmAmAmGmUmGmGmCmAmCmCmGmAmGmUmCmGmGmUmGmCmU*mU*mU*mU                                                                                                                                         |
| 619936976 |       0.0047 | mC*mC*mC*rArUrArCrCrUrUrGrGrArGrCrArArCrGrGrGrUrUrUrUrArGrAmGmCmUmAmGmAmAmAmUmAmGmCrArArGrUrUrArArArArUrArArGrGrCrUrArGrUrCrCrGrUrUrArUrCrAmAmCmUmUmGmAmAmAmAmAmGmUmGmGmCmAmCmCmGmAmGmUmCmGmGmUmGmCrArCrCrGrCrCrArCrCrUrUrCrCrGrCrCrGrGrUrArA*mU*mU*mG*mC*mU*mC*mC*mA*mA*mG*mG*mU*mA*mU                                                          |
| 622707877 |       0.0052 | mC*mU*mC*rCrUrGrGrArArGrArUrGrUrCrCrArCrCrArGrUrUrUrUrArGrArGrCrUrArGrArArArUrArGrCrArArGrUrUrArArArArUrArArGrGrCrUrArGrUrCrCrGrUrUrArUrCrArArCrUrUrGrArArArArArGrUrGrGrCrArCrCrGrArGrUrCrGrGrUrGrCmU*mU*mU*rU                                                                                                                                   |

## Top Stable Feature Means in False Negatives

| feature                     |   false_negative_mean |
|:----------------------------|----------------------:|
| repeated_dinucleotide_count |                8.5000 |
| token_rC_fraction           |                0.1608 |
| base_C_fraction             |                0.1973 |
| token_rG_fraction           |                0.2045 |
| purine_fraction             |                0.5438 |
| base_G_fraction             |                0.2489 |
| base_T_fraction             |                0.2589 |
| pyrimidine_fraction         |                0.4562 |
| base_U_fraction             |                0.2589 |
| token_rC_count              |               17.4375 |

## Top Stable Feature Means in True Positives

| feature                     |   true_positive_mean |
|:----------------------------|---------------------:|
| repeated_dinucleotide_count |               7.5000 |
| token_rC_fraction           |               0.1685 |
| base_C_fraction             |               0.2253 |
| token_rG_fraction           |               0.1610 |
| purine_fraction             |               0.4744 |
| base_G_fraction             |               0.2310 |
| base_T_fraction             |               0.3003 |
| pyrimidine_fraction         |               0.5256 |
| base_U_fraction             |               0.3003 |
| token_rC_count              |              20.3333 |
