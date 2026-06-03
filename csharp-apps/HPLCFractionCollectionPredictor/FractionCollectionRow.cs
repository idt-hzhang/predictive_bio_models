using Microsoft.ML.Data;

namespace HPLCFractionCollectionPredictor;

public class FractionCollectionRow
{
    // Columns 5 (MaxAbsorbanceUnitOfMeasureId), 6 (VialId), 7 (FractionCollectionAnalysisId) intentionally ignored.
    [LoadColumn(8)]
    public bool IsToBeTagged { get; set; }

    [LoadColumn(2)]
    public bool Label { get; set; }

    [LoadColumn(1)]
    public float MaxAbsorbance { get; set; }

    [LoadColumn(3)]
    public float MethodId { get; set; }

    [LoadColumn(9)]
    public float NanomoleGuarantee { get; set; }

    [LoadColumn(4)]
    public float ProductId { get; set; }

    [LoadColumn(10)]
    public float ShipODGuarantee { get; set; }
}