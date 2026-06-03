using Microsoft.ML.Data;

namespace HPLCFractionCollectionPredictor;

public class FractionCollectionPrediction
{
    /// <summary>
    ///     Final business decision after applying the tuned probability threshold.
    ///     Call <see cref="ApplyBusinessThreshold" /> after running the prediction engine.
    /// </summary>
    public bool BusinessDecision { get; private set; }

    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    public float Probability { get; set; }

    public float Score { get; set; }

    public void ApplyBusinessThreshold(float threshold) => BusinessDecision = Probability >= threshold;
}