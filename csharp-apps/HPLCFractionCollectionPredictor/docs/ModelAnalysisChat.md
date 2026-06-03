# FractionCollectionModel — Analysis Chat Export

Source: `HPLCFractionCollectionPredictor/Program.cs`
Workspace: `C:\Users\jvolker\source\repos\predictive_bio_models\csharp-apps\`
Repo: https://github.com/idt-hzhang/predictive_bio_models (branch `main`)
Environment: Visual Studio Professional 2026 (18.6.2), .NET 10

---

## 1. Summary — `FractionCollectionModel`

Binary classification model defined at `Program.cs:14`
(`const string modelPath = "FractionCollectionModel.zip"`) that predicts
whether an HPLC fraction should be collected.

### Type
- **Task:** Binary classification (`Label: bool`)
- **Algorithm:** LightGBM (`Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer`)

### Trainer Options
- `NumberOfIterations = 300`
- `NumberOfLeaves = 64`
- `MinimumExampleCountPerLeaf = 20`
- `LearningRate = 0.1`
- `UnbalancedSets = true` (handles positive-class imbalance)

### Features (concatenated into `Features` vector, MinMax normalized)
- `MaxAbsorbance` (float)
- `MethodId` → `OneHotHashEncoding` (16 bits)
- `ProductId` → `OneHotHashEncoding` (16 bits)
- `IsToBeTagged` (bool → float)
- `NanomoleGuarantee` (float)
- `ShipODGuarantee` (float)

### Training Data
- CSV: `fraction-collection-training-models-2026-06-01.csv`
- 80/20 train/test split (`seed = 1`), `MLContext` seed = 1 for reproducibility.

### Evaluation
- Standard ML.NET binary metrics: Accuracy, ROC-AUC, PR-AUC, F1,
  positive/negative precision & recall, confusion matrix.
- Threshold sweep at 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80.

### Decision Threshold
- Default business threshold = **0.60** (best balanced F1).
- 0.40 alternative favors recall (~94%) for "don't miss a fraction" scenarios.
- Applied via `FractionCollectionPrediction.ApplyBusinessThreshold(threshold)`.

### Output
- Serialized to `FractionCollectionModel.zip` via `mlContext.Model.Save(...)`.
- Consumed via a `PredictionEngine<FractionCollectionRow, FractionCollectionPrediction>`,
  including an interactive console loop for ad-hoc predictions.

---

## 2. Detailed Model Statistics

LightGBM binary classifier (`UnbalancedSets=true`, 300 iters, 64 leaves, LR=0.1),
evaluated on the 20% holdout split (seed=1).

### Test Set Composition

| Metric              | Value                       |
| ------------------- | --------------------------- |
| Total test rows     | 46,590                      |
| Positives           | 10,701                      |
| Negatives           | 35,889                      |
| Positive class ratio| **22.97 %** (imbalanced)    |

### Headline Metrics (default 0.5 threshold)

| Metric             | Value      |
| ------------------ | ---------- |
| Accuracy           | **0.7870** |
| AUC (ROC)          | **0.9001** |
| AUC (PR)           | **0.6947** |
| F1                 | **0.6635** |
| Positive Precision | 0.5207     |
| Positive Recall    | **0.9141** |
| Negative Precision | 0.9669     |
| Negative Recall    | 0.7491     |

### Confusion Matrix @ 0.5

|             | Predicted + | Predicted − | Recall |
| ----------- | ----------: | ----------: | -----: |
| **Truth +** |   9,782 (TP)|     919 (FN)| 0.9141 |
| **Truth −** |   9,004 (FP)|  26,885 (TN)| 0.7491 |
| **Precision**|     0.5207 |      0.9669 |        |

Model strongly favors catching positives — high recall, moderate precision
(expected given `UnbalancedSets=true`).

### Threshold Sweep

| Threshold |     TP |     FP |    FN |     TN | Precision | Recall |     F1 | Accuracy |
| --------: | -----: | -----: | ----: | -----: | --------: | -----: | -----: | -------: |
|      0.30 | 10,318 | 11,270 |   383 | 24,619 |    0.4780 | **0.9642** | 0.6391 |   0.7499 |
|      0.40 | 10,098 | 10,085 |   603 | 25,804 |    0.5003 | 0.9437 | 0.6539 |   0.7706 |
|      0.50 |  9,782 |  9,004 |   919 | 26,885 |    0.5207 | 0.9141 | 0.6635 |   0.7870 |
|  **0.60** |  9,222 |  7,611 | 1,479 | 28,278 |    0.5479 | 0.8618 | **0.6699** |   0.8049 |
|      0.65 |  8,707 |  6,685 | 1,994 | 29,204 |    0.5657 | 0.8137 | 0.6674 |   0.8137 |
|      0.70 |  7,779 |  5,253 | 2,922 | 30,636 |    0.5969 | 0.7269 | 0.6555 |   0.8245 |
|      0.80 |  5,002 |  2,195 | 5,699 | 33,694 |    **0.6950** | 0.4674 | 0.5589 | **0.8306** |

### Interpretation

- **Best F1:** threshold **0.60** (0.6699) — matches the configured
  `businessThreshold` in `Program.cs:20`.
- **Max recall:** threshold 0.30 (96.4 %) — choose if missing a fraction is
  very costly (false-negative averse).
- **Max precision/accuracy:** threshold 0.80 — choose if false positives are
  costly, accepting ~47 % recall.
- **ROC-AUC 0.90** indicates strong overall ranking ability; the lower
  **PR-AUC (0.69)** reflects class imbalance — typical for a 23%-positive dataset.

### Sample Prediction

A test row with actual `Label = False` scored `Probability = 0.0664` →
`BusinessDecision = False` ✅ (correct, well below the 0.60 threshold).

### Artifact

- Saved to:
  `C:\Users\jvolker\source\repos\predictive_bio_models\csharp-apps\HPLCFractionCollectionPredictor\FractionCollectionModel.zip`

---

## 3. Feature Importance

Measured by **permutation importance** on the 20% test set (46,590 rows):
each raw input column was randomly shuffled across rows, then the trained
LightGBM model was re-scored. The larger the metric drop vs. the baseline,
the more the model depends on that feature.

Baseline: **AUC = 0.9001**, **F1 = 0.6635**.

### Ranked Importance (by ΔAUC)

| Rank | Feature             | AUC after shuffle |    ΔAUC | F1 after shuffle |     ΔF1 |
| ---: | ------------------- | ----------------: | ------: | ---------------: | ------: |
|    1 | **MaxAbsorbance**   |            0.5602 | **+0.3399** |       0.3286 | +0.3349 |
|    2 | **ProductId**       |            0.8386 |  +0.0615 |           0.6045 | +0.0590 |
|    3 | **MethodId**        |            0.8454 |  +0.0547 |           0.5963 | +0.0672 |
|    4 | ShipODGuarantee     |            0.8712 |  +0.0289 |           0.6429 | +0.0205 |
|    5 | NanomoleGuarantee   |            0.8843 |  +0.0158 |           0.6379 | +0.0255 |
|    6 | IsToBeTagged        |            0.8960 |  +0.0041 |           0.6595 | +0.0040 |

### Interpretation

- **MaxAbsorbance is dominant by a wide margin.** Shuffling it collapses AUC
  from 0.90 → 0.56 (barely better than random, 0.50). This is the primary
  signal driving the "collect this fraction" decision — consistent with HPLC
  domain intuition that absorbance peaks indicate the analyte of interest.
- **ProductId and MethodId form a secondary tier** (ΔAUC ≈ 0.05–0.06). These
  categorical context features (encoded via `OneHotHashEncoding`, 16 bits) let
  the model learn product/method-specific collection patterns on top of the
  absorbance signal.
- **ShipODGuarantee and NanomoleGuarantee contribute modestly**
  (ΔAUC 0.02–0.03). They capture order/quantity context that nudges the
  decision boundary but doesn't define it.
- **IsToBeTagged is nearly irrelevant** (ΔAUC 0.004). It can likely be dropped
  without meaningful accuracy loss — a candidate for simplification.

### Practical Takeaways

1. **Data quality on `MaxAbsorbance` is critical** — any miscalibration of the
   detector or unit-of-measure inconsistency will hit accuracy hard. Validate
   this column upstream.
2. **Consider removing `IsToBeTagged`** in a future iteration; the feature
   concatenation/normalization pipeline would be slightly leaner and the model
   marginally easier to interpret.
3. **Categorical encoding looks healthy** — both `MethodId` and `ProductId`
   register clear importance, so the 16-bit hash isn't collapsing them.
4. The strong reliance on a single feature explains the **0.90 ROC-AUC** vs.
   the more modest **0.69 PR-AUC**: the model ranks well overall but still
   produces many false positives on borderline absorbance values (visible in
   the threshold sweep).
