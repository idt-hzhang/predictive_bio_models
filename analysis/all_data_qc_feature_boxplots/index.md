# All Cleaned Data QC Feature Plots

Plots compare `Pass` and `Fail` rows from `data/all.cleaned_data.csv`. Numeric features are shown as boxplots with jittered points. `ESI Status` is categorical, so it is shown as count and within-class proportion bar plots.

Combined numeric panel: [all_numeric_qc_boxplots.png](all_numeric_qc_boxplots.png)

| Feature | Type | Plot | Pass n | Fail n | Pass median | Fail median | MWU p | Cohen's d |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Sample OD` | numeric | [Sample_OD.png](Sample_OD.png) | 2560 | 86 | 0.51 | 0.51 | 0.793 | 0.1727 |
| `Sample Volume (uL)` | numeric | [Sample_Volume__uL_.png](Sample_Volume__uL_.png) | 2560 | 86 | 50 | 50 | 0.108 | 0.1818 |
| `P=S Bonds` | numeric | [P_S_Bonds.png](P_S_Bonds.png) | 2560 | 86 | 6 | 6 | 0.794 | 0.2022 |
| `OMe Bases` | numeric | [OMe_Bases.png](OMe_Bases.png) | 2560 | 86 | 6 | 6 | 0.148 | 0.1553 |
| `Length` | numeric | [Length.png](Length.png) | 2560 | 86 | 100 | 100 | 0.799 | -0.101 |
| `UHPLC % Pre-MainPeak` | numeric | [UHPLC___Pre-MainPeak.png](UHPLC___Pre-MainPeak.png) | 2560 | 84 | 8 | 13 | 2.07e-15 | 1.653 |
| `UHPLC % MainPeak` | numeric | [UHPLC___MainPeak.png](UHPLC___MainPeak.png) | 2560 | 84 | 79 | 67 | 8.71e-16 | -1.238 |
| `UHPLC % Post-MainPeak` | numeric | [UHPLC___Post-MainPeak.png](UHPLC___Post-MainPeak.png) | 2558 | 84 | 13 | 15.5 | 0.0031 | 0.49 |
| `ESI %` | numeric | [ESI__.png](ESI__.png) | 2555 | 86 | 97 | 92.5 | 0.000249 | -0.4145 |
| `Ship Ods` | numeric | [Ship_Ods.png](Ship_Ods.png) | 2560 | 86 | 11 | 10 | 0.757 | 0.1858 |
| `Ship Nmoles` | numeric | [Ship_Nmoles.png](Ship_Nmoles.png) | 2560 | 86 | 11 | 10 | 0.794 | 0.1607 |
| `Yield Guarantee (ODs)` | numeric | [Yield_Guarantee__ODs_.png](Yield_Guarantee__ODs_.png) | 2560 | 86 | 10 | 8.5 | 0.883 | 0.2074 |
| `ESI Status` | categorical | [ESI_Status.png](ESI_Status.png) | 2547 | 86 | nan | nan | nan | nan |
