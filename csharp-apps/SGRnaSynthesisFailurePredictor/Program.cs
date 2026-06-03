using Microsoft.ML;
using Microsoft.ML.Data;

var DataPath = Path.Combine(AppContext.BaseDirectory, "data.csv");
if (!File.Exists(DataPath))
    DataPath = Path.Combine(Directory.GetCurrentDirectory(), "data.csv");
const string ModelPath = "sgRNAClassificationModel.zip";

var mlContext = new MLContext(seed: 1);

// Load the full CSV, then keep only rows with a valid Pass/Fail label and a non-empty Sequence.
var allRows = mlContext.Data.LoadFromTextFile<SgRnaRawRow>(
    path: DataPath,
    hasHeader: true,
    separatorChar: ',',
    allowQuoting: true,
    trimWhitespace: true);

var filtered = mlContext.Data.CreateEnumerable<SgRnaRawRow>(allRows, reuseRowObject: false)
    .Where(r => !string.IsNullOrWhiteSpace(r.Sequence)
                && !string.IsNullOrWhiteSpace(r.PassFail)
                && (r.PassFail.Equals("Pass", StringComparison.OrdinalIgnoreCase)
                    || r.PassFail.Equals("Fail", StringComparison.OrdinalIgnoreCase)
                    || r.PassFail.Equals("Needs Review", StringComparison.OrdinalIgnoreCase)))
    .Select(r => new SgRnaSample { Sequence = r.Sequence!, Label = NormalizeLabel(r.PassFail!) })
    .ToList();

Console.WriteLine($"Loaded {filtered.Count} labeled rows.");
foreach (var g in filtered.GroupBy(x => x.Label).OrderByDescending(g => g.Count()))
    Console.WriteLine($"  {g.Key}: {g.Count()}");

var data = mlContext.Data.LoadFromEnumerable(filtered);
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);

// Featurize the Sequence string using character n-grams (sequence-only model).
var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
    .Append(mlContext.Transforms.Text.TokenizeIntoCharactersAsKeys(
        outputColumnName: "Chars", inputColumnName: nameof(SgRnaSample.Sequence)))
    .Append(mlContext.Transforms.Text.ProduceNgrams(
        outputColumnName: "Features", inputColumnName: "Chars",
        ngramLength: 4, useAllLengths: true,
        weighting: Microsoft.ML.Transforms.Text.NgramExtractingEstimator.WeightingCriteria.TfIdf))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .AppendCacheCheckpoint(mlContext)
    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
        labelColumnName: "Label", featureColumnName: "Features"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

Console.WriteLine("Training...");
var model = pipeline.Fit(split.TrainSet);

Console.WriteLine("Evaluating...");
var predictions = model.Transform(split.TestSet);
var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");
Console.WriteLine($"  MicroAccuracy:    {metrics.MicroAccuracy:0.0000}");
Console.WriteLine($"  MacroAccuracy:    {metrics.MacroAccuracy:0.0000}");
Console.WriteLine($"  LogLoss:          {metrics.LogLoss:0.0000}");
Console.WriteLine($"  LogLossReduction: {metrics.LogLossReduction:0.0000}");
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

mlContext.Model.Save(model, data.Schema, ModelPath);
Console.WriteLine($"Model saved to: {Path.GetFullPath(ModelPath)}");

// Sample prediction.
var engine = mlContext.Model.CreatePredictionEngine<SgRnaSample, SgRnaPrediction>(model);
var sampleSeq = filtered[0].Sequence;
var pred = engine.Predict(new SgRnaSample { Sequence = sampleSeq });
Console.WriteLine($"\nSample prediction for first sequence -> {pred.PredictedLabel}");

static string NormalizeLabel(string raw)
{
    var t = raw.Trim();
    if (t.Equals("Pass", StringComparison.OrdinalIgnoreCase)) return "Pass";
    if (t.Equals("Fail", StringComparison.OrdinalIgnoreCase)) return "Fail";
    return "NeedsReview";
}

public class SgRnaRawRow
{
    [LoadColumn(8)] public string? PassFail { get; set; }
    [LoadColumn(31)] public string? Sequence { get; set; }
}

public class SgRnaSample
{
    public string Sequence { get; set; } = string.Empty;
    public string Label { get; set; } = string.Empty;
}

public class SgRnaPrediction
{
    [ColumnName("PredictedLabel")] public string PredictedLabel { get; set; } = string.Empty;
    public float[] Score { get; set; } = Array.Empty<float>();
}
