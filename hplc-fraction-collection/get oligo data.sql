USE Purification;

DECLARE @mfgItemId INT = 835753488;

SELECT TOP (100)
       spawnOligo.MfgItemId,
       f.VialPosition,
       hrd.MethodId,
       specOligo.PROD_ID AS [ProductId],
       specOligo.GUAR_MIN_SHIP_VAL * 1E6 / ssi.EXT_COEFF AS [NanomoleGuarantee],
       specOligo.GUAR_MIN_SHIP_VAL AS [ShipODGuarantee],
       ssi.SPEC_SEQ_INFO_ID,
       ssi.EXT_COEFF AS [ExtinctionCoefficient],
       ssp.SPEC_SEQ_PRODUCT_ID
FROM dbo.Fraction AS f
    JOIN dbo.Oligo AS spawnOligo
        ON spawnOligo.OligoId = f.SpawnOligoId
    JOIN dbo.HPLCRunDetail AS hrd
        ON hrd.HPLCRunDetailId = f.HPLCRunDetailId
    JOIN Production.dbo.SPEC_OLIGO AS specOligo
        ON spawnOligo.SpecOligoId = specOligo.SPEC_OLIGO_ID
    LEFT JOIN Production.dbo.SPEC_SEQ_INFO AS ssi
        ON ssi.SPEC_OLIGO_ID = specOligo.SPEC_OLIGO_ID
    LEFT JOIN Production.dbo.SPEC_SEQ_PRODUCT AS ssp
        ON ssp.SPEC_OLIGO_ID = specOligo.SPEC_OLIGO_ID
WHERE spawnOligo.MfgItemId = @mfgItemId
ORDER BY f.FractionId DESC;