using System.Globalization;
using HPLCFractionCollectionPredictor;
using Microsoft.ML;
using Microsoft.ML.Trainers.LightGbm;

var dataPath = Path.Combine(AppContext.BaseDirectory, "fraction-collection-training-models-2026-06-01.csv");
if (!File.Exists(dataPath))
{
    dataPath = Path.Combine(Directory.GetCurrentDirectory(),
        "HPLCFractionCollectionPredictor",
        "fraction-collection-training-models-2026-06-01.csv");
}

const string modelPath = "FractionCollectionModel.zip";

// Business decision threshold. Tuned via threshold sweep:
// - 0.40 favors recall (~94% positive recall) - recommended for "don't miss a fraction"
// - 0.60 best balanced F1
const float businessThreshold = 0.60f;

var mlContext = new MLContext(1);

var data = mlContext.Data.LoadFromTextFile<FractionCollectionRow>(dataPath,
    hasHeader: true,
    separatorChar: ',',
    allowQuoting: true,
    trimWhitespace: true);

var split = mlContext.Data.TrainTestSplit(data, 0.2, seed: 1);

var lightGbmOptions = new LightGbmBinaryTrainer.Options
                      {
                          LabelColumnName = "Label",
                          FeatureColumnName = "Features",
                          NumberOfIterations = 300,
                          NumberOfLeaves = 64,
                          MinimumExampleCountPerLeaf = 20,
                          LearningRate = 0.1,
                          UnbalancedSets = true // class balancing for imbalanced positive class
                      };

var pipeline = mlContext.Transforms.Categorical.OneHotHashEncoding([
                                new("MethodIdEncoded",
                                    nameof(FractionCollectionRow.MethodId)),
                                new("ProductIdEncoded",
                                    nameof(FractionCollectionRow.ProductId))
                            ],
                            numberOfBits: 16)
                        .Append(mlContext.Transforms.Conversion.ConvertType("IsToBeTaggedF",
                            nameof(FractionCollectionRow.IsToBeTagged)))
                        .Append(mlContext.Transforms.Concatenate("Features",
                            nameof(FractionCollectionRow.MaxAbsorbance),
                            "MethodIdEncoded",
                            "ProductIdEncoded",
                            "IsToBeTaggedF",
                            nameof(FractionCollectionRow.NanomoleGuarantee),
                            nameof(FractionCollectionRow.ShipODGuarantee)))
                        .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                        .AppendCacheCheckpoint(mlContext)
                        .Append(mlContext.BinaryClassification.Trainers.LightGbm(lightGbmOptions));

Console.WriteLine("Training binary classifier (LightGBM, UnbalancedSets=true)...");
var model = pipeline.Fit(split.TrainSet);

Console.WriteLine("Evaluating...");
var predictions = model.Transform(split.TestSet);
var metrics = mlContext.BinaryClassification.Evaluate(predictions);

Console.WriteLine($"  Accuracy:          {metrics.Accuracy:0.0000}");
Console.WriteLine($"  AUC (ROC):         {metrics.AreaUnderRocCurve:0.0000}");
Console.WriteLine($"  AUC (PR):          {metrics.AreaUnderPrecisionRecallCurve:0.0000}");
Console.WriteLine($"  F1:                {metrics.F1Score:0.0000}");
Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision:0.0000}");
Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall:0.0000}");
Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision:0.0000}");
Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall:0.0000}");
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

// Threshold sweep: report precision/recall/F1 at multiple decision thresholds.
Console.WriteLine("\nThreshold sweep (using raw Probability):");
Console.WriteLine($"{"Thresh",-8}{"TP",-8}{"FP",-8}{"FN",-8}{"TN",-8}{"Prec",-8}{"Recall",-8}{"F1",-8}{"Acc",-8}");

var scored = mlContext.Data.CreateEnumerable<ScoredRow>(predictions, false).ToList();
foreach (var threshold in new[]
                          {
                              0.30f,
                              0.40f,
                              0.50f,
                              0.60f,
                              0.65f,
                              0.70f,
                              0.80f
                          })
{
    int tp = 0, fp = 0, fn = 0, tn = 0;
    foreach (var r in scored)
    {
        var predicted = r.Probability >= threshold;
        switch (r.Label)
        {
            case true when predicted:
                tp++;
                break;
            case false when predicted:
                fp++;
                break;
            case true when !predicted:
                fn++;
                break;
            default:
                tn++;
                break;
        }
    }

    var precision = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
    var recall = tp + fn == 0 ? 0 : (double)tp / (tp + fn);
    var f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
    var accuracy = (double)(tp + tn) / scored.Count;
    Console.WriteLine(
        $"{threshold,-8:0.00}{tp,-8}{fp,-8}{fn,-8}{tn,-8}{precision,-8:0.0000}{recall,-8:0.0000}{f1,-8:0.0000}{accuracy,-8:0.0000}");
}

mlContext.Model.Save(model, data.Schema, modelPath);
Console.WriteLine($"Model saved to: {Path.GetFullPath(modelPath)}");

var engine = mlContext.Model.CreatePredictionEngine<FractionCollectionRow, FractionCollectionPrediction>(model);
var sample = mlContext.Data.CreateEnumerable<FractionCollectionRow>(split.TestSet, false).First();
var pred = engine.Predict(sample);
pred.ApplyBusinessThreshold(businessThreshold);
Console.WriteLine($"\nSample prediction (threshold={businessThreshold:0.00}):");
Console.WriteLine($"  Probability:        {pred.Probability:0.0000}");
Console.WriteLine($"  PredictedLabel@0.5: {pred.PredictedLabel}");
Console.WriteLine($"  BusinessDecision:   {pred.BusinessDecision}  (Actual: {sample.Label})");

// Interactive prediction loop: let the user enter test data and see what the model predicts.
RunInteractivePredictionLoop(engine, businessThreshold);
return;

static void RunInteractivePredictionLoop(PredictionEngine<FractionCollectionRow, FractionCollectionPrediction> engine,
    float businessThreshold)
{
    Console.WriteLine("\n=== Interactive prediction ===");
    Console.WriteLine("Enter feature values when prompted. Press ENTER on the first prompt to quit.");
    Console.WriteLine("Tip: type 'sample' at the first prompt to load a built-in example row.\n");

    while (true)
    {
        Console.Write("MaxAbsorbance (float, blank to quit): ");
        var first = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(first))
        {
            break;
        }

        FractionCollectionRow row;
        if (first.Trim().Equals("sample", StringComparison.OrdinalIgnoreCase))
        {
            row = new()
                  {
                      MaxAbsorbance = 865.535f,
                      MethodId = 1232,
                      ProductId = 9619,
                      IsToBeTagged = false,
                      NanomoleGuarantee = 4,
                      ShipODGuarantee = 1.026176f
                  };
            Console.WriteLine($"  Using sample row: {Describe(row)}");
        }
        else
        {
            if (!TryParseFloat(first, out var maxAbs))
            {
                PrintParseError("MaxAbsorbance");
                continue;
            }

            if (!TryReadFloat("MethodId (int)", out var methodId))
            {
                continue;
            }

            if (!TryReadFloat("ProductId (int)", out var productId))
            {
                continue;
            }

            if (!TryReadBool("IsToBeTagged (0/1 or true/false)", out var isToBeTagged))
            {
                continue;
            }

            if (!TryReadFloat("NanomoleGuarantee (float)", out var nmole))
            {
                continue;
            }

            if (!TryReadFloat("ShipODGuarantee (float)", out var shipOD))
            {
                continue;
            }

            row = new()
                  {
                      MaxAbsorbance = maxAbs,
                      MethodId = methodId,
                      ProductId = productId,
                      IsToBeTagged = isToBeTagged,
                      NanomoleGuarantee = nmole,
                      ShipODGuarantee = shipOD
                  };
        }

        var p = engine.Predict(row);
        p.ApplyBusinessThreshold(businessThreshold);
        Console.WriteLine("  -> Prediction:");
        Console.WriteLine($"       Probability:        {p.Probability:0.0000}");
        Console.WriteLine($"       PredictedLabel@0.5: {p.PredictedLabel}");
        Console.WriteLine($"       BusinessDecision@{businessThreshold:0.00}: {p.BusinessDecision}");
        Console.WriteLine();
    }

    Console.WriteLine("Exiting interactive mode.");
}

static bool TryParseFloat(string? input, out float value) => float.TryParse(input,
    NumberStyles.Float,
    CultureInfo.InvariantCulture,
    out value);

static bool TryReadFloat(string prompt, out float value)
{
    Console.Write($"{prompt}: ");
    if (TryParseFloat(Console.ReadLine(), out value))
    {
        return true;
    }

    PrintParseError(prompt);
    return false;
}

static bool TryReadBool(string prompt, out bool value)
{
    Console.Write($"{prompt}: ");
    var raw = (Console.ReadLine() ?? string.Empty).Trim();
    switch (raw)
    {
        case "1" or "true" or "True" or "yes" or "y":
            value = true;
            return true;
        case "0" or "false" or "False" or "no" or "n":
            value = false;
            return true;
    }

    value = false;
    PrintParseError(prompt);
    return false;
}

static void PrintParseError(string field) => Console.WriteLine($"  ! Could not parse '{field}'. Skipping this row.\n");

static string Describe(FractionCollectionRow r) =>
    $"MaxAbs={r.MaxAbsorbance}, MethodId={r.MethodId}, ProductId={r.ProductId}, "
    + $"Tag={r.IsToBeTagged}, Nmole={r.NanomoleGuarantee}, ShipOD={r.ShipODGuarantee}";